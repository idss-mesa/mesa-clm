"""The DuckDB sidecar store (DESIGN D10, D11, D21, D28): round trip of every table, CHECKs
enforced by the database (bypassing pydantic) and mirrored by the row models, the one-transaction
run buffer, the per-table write API, reads, the migration bootstrap and ``open_store``. Ported from
mesa-anyjev ``tests/test_provenance_duckdb.py``; hermetic (DuckDB files under ``tmp_path``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import duckdb
import pytest
from pydantic import ValidationError

from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.provenance.migrate import apply_migrations, current_version, migration_files
from mesa_clm.provenance.models import (
    AuditRow,
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    RunRow,
)
from mesa_clm.provenance.store import (
    SCHEMA,
    TABLES,
    DuckDBStore,
    ProvenanceStore,
    RunBuffer,
    duckdb_file,
    open_store,
)
from mesa_clm.tasks import TASKS

TERM_KEY = TASKS["term.fits"].key


def _run(**over: Any) -> RunRow:
    base: dict[str, Any] = {
        "owner": "alice",
        "card_name": "DP1.10003.001.brd_countdata",
        "card_sha256": "s" * 64,
        "planner": "static",
        "provider": "fake",
        "clm_model": "clm-latest",
        "encoder_model": "qwen3-8b",
        "schema_sha256": "h" * 64,
        "encoder_fp": "e" * 12,
        "clm_model_fp": "m" * 12,
        "framings_lock_sha": "l" * 64,
        "policy_profile": "dev",
        "config_sha256": "c" * 64,
        "vm_id": "testhost",
    }
    base.update(over)
    return RunRow(**base)


def _decision(run_id: UUID, seq: int, **over: Any) -> DecisionRow:
    base: dict[str, Any] = {
        "run_id": run_id,
        "seq": seq,
        "task_id": "term.fits",
        "task_key": TERM_KEY,
        "question_key": "q" * 16,
        "framing_id": "F7",
        "shape": "rank_fit",
        "k": 3,
        "scope": "column",
        "column_name": "observerDistance",
        "state_sha256": "a" * 64,
        "state_json": {"card": {"dataset": "DP1.x"}, "column": {"name": "observerDistance"}},
        "target_sha256": "t" * 64,
        "context_sha256": "x" * 64,
        "context_tokens": 120,
        "method": "fake",
        "model": "clm-latest",
        "level": "zero_shot",
        "calibration": "uncalibrated",
        "probs": [0.6, 0.3, 0.1],
        "raw_probs": [0.6, 0.3, 0.1],
        "confidence": 0.6,
        "clm_confidence": 0.3,
        "s_c": 1.79,
        "p_fit": 0.857,
        "anchor_index": 2,
        "answer_index": 0,
        "answer": "PATO:0000040",
        "options": ["PATO:0000040", "PATO:0000001", "__none__"],
        "rank": 0,
        "encoder_fp": "e" * 12,
        "clm_model_fp": "m" * 12,
        "schema_sha256": "h" * 64,
        "outcome": "proposed",
    }
    base.update(over)
    return DecisionRow(**base)


def _group(run_id: UUID, **over: Any) -> DecisionGroupRow:
    base: dict[str, Any] = {
        "run_id": run_id,
        "task_id": "term.fits",
        "task_key": TERM_KEY,
        "question_key": "q" * 16,
        "scope": "column",
        "column_name": "observerDistance",
        "aspect": "measurement",
        "ontology_id": "pato",
        "search_json": {"queries": ["distance"]},
        "n_candidates": 2,
        "outcome": "proposed",
    }
    base.update(over)
    return DecisionGroupRow(**base)


def _link(run_id: UUID, **over: Any) -> AvuLinkRow:
    base: dict[str, Any] = {
        "run_id": run_id,
        "attribute": "pato.distance",
        "value": "distance",
        "unit": "PATO:0000040",
        "term_curie": "PATO:0000040",
        "ontology_id": "pato",
        "aspect": "measurement",
        "column_name": "observerDistance",
        "source": "mesa-clm:annotate:proposed",
    }
    base.update(over)
    return AvuLinkRow(**base)


def _label(**over: Any) -> LabelRow:
    base: dict[str, Any] = {
        "task_id": "term.fits",
        "task_key": TERM_KEY,
        "target_sha256": "d" * 64,
        "option_key": "PATO:0000040",
        "label_source": "curator",
        "label": "Yes",
        "label_index": 0,
        "weight": 1.0,
        "state_sha256": "a" * 64,
        "state_json": {"card": {"dataset": "DP1.x"}},
        "card": "DP1.10003.001.brd_countdata",
        "origin": "override:test",
        "actor": "alice",
    }
    base.update(over)
    return LabelRow(**base)


@pytest.fixture
def store(tmp_path: Path) -> DuckDBStore:
    s = DuckDBStore(tmp_path / "prov.duckdb")
    s.ensure_schema()
    return s


# -- schema -------------------------------------------------------------------------------------------------


def test_schema_tables_exist_and_version_is_recorded(store: DuckDBStore) -> None:
    for table in TABLES:
        assert store.columns(table), table
    assert store.ensure_schema() == 1  # idempotent
    with store.connect(read_only=True) as con:
        assert con.execute(f"SELECT version FROM {SCHEMA}.schema_versions").fetchall() == [(1,)]  # noqa: S608


def test_reads_on_a_missing_file_return_nothing_and_create_nothing(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "nope" / "prov.duckdb")
    assert store.run(uuid4()) is None and store.runs() == [] and store.decisions(uuid4()) == []
    assert store.labels_for("term.fits") == [] and store.audits() == []
    assert store.decisions_for_path("/x") == [] and store.group(uuid4()) is None
    assert not (tmp_path / "nope" / "prov.duckdb").exists()


def test_reads_on_a_labels_only_file_return_nothing(tmp_path: Path) -> None:
    """A file the M0 LabelStore created has schema mesa_clm and the labels table only; the
    sidecar reads see no runs (CatalogException -> empty) and the labels."""
    path = tmp_path / "labels.duckdb"
    LabelStore(path).insert_labels([_label()])
    store = DuckDBStore(path)
    assert store.run(uuid4()) is None and store.runs() == []
    assert len(store.labels_for("term.fits")) == 1
    store.begin_run(_run())  # the first write bootstraps the rest of the schema
    assert len(store.runs()) == 1 and len(store.labels_for("term.fits")) == 1


# -- round trip ---------------------------------------------------------------------------------------------


def test_round_trip(store: DuckDBStore) -> None:
    run = _run()
    assert store.begin_run(run) == run.run_id
    group = _group(run.run_id)
    store.insert_group(group)
    d1 = _decision(run.run_id, 1, group_id=group.group_id)
    d2 = _decision(
        run.run_id,
        2,
        task_id="column.annotate",
        task_key=TASKS["column.annotate"].key,
        shape="choice",
        k=2,
        column_name="uid",
        method="rule",
        model="",
        level="none",
        calibration="none",
        probs=None,
        raw_probs=None,
        confidence=None,
        clm_confidence=None,
        s_c=None,
        p_fit=None,
        anchor_index=None,
        answer_index=1,
        answer="No",
        options=["Yes", "No"],
        rank=None,
        outcome="rule",
        reason="is_identifier",
    )
    opts = [
        DecisionOptionRow(
            decision_id=d1.decision_id,
            option_index=i,
            option_key=key,
            option_text=text,
            rank=i,
            s_c=s,
            p_fit=p,
            prob=pr,
            raw_prob=pr,
            action_sha256="f" * 64,
        )
        for i, (key, text, s, p, pr) in enumerate(
            [
                ("PATO:0000040", "distance: ...", 1.79, 0.857, 0.6),
                ("PATO:0000001", "quality: ...", 1.10, 0.75, 0.3),
                ("__none__", "None of these terms ...", 0.0, 0.5, 0.1),
            ]
        )
    ]
    assert store.insert_decisions([d1, d2], opts) == 2
    store.update_group(
        group.group_id,
        winner_decision_id=d1.decision_id,
        top_p_fit=0.857,
        group_margin=0.107,
        level="zero_shot",
        method="fake",
        outcome="proposed",
    )
    link = _link(run.run_id, group_id=group.group_id, decision_id=d1.decision_id)
    other = _link(
        run.run_id, attribute="envo.biome", value="desert", unit="ENVO:1", site_code="SRER"
    )
    assert store.insert_links([link, other]) == 2
    assert store.set_link_status([link.link_id], "accepted", accepted_by="human") == 1
    now = datetime.now(tz=UTC)
    assert (
        store.set_link_status(
            [link.link_id], "written", written_at=now, irods_path="/iplant/home/x/f.csv"
        )
        == 1
    )
    assert store.set_link_status([], "written") == 0
    assert store.link_snapshot(run.run_id, "/iplant/home/x/f.csv", "proj-1", 7) == 1
    assert store.link_snapshot(run.run_id, "/iplant/home/x/f.csv", "proj-1", 8) == 0  # once
    override = HumanOverrideRow(
        run_id=run.run_id,
        group_id=group.group_id,
        decision_id=d1.decision_id,
        link_id=link.link_id,
        actor="alice",
        via="cli",
        action="pick",
        chosen_decision_id=d1.decision_id,
        chosen_option_key="PATO:0000040",
        label_source="curator",
        labels_written=2,
        offered=[{"decision_id": str(d1.decision_id), "option_key": "PATO:0000040", "rank": 0}],
    )
    assert store.insert_override(override) == override.override_id
    dup = _label(label_id=uuid4(), label="No", label_index=1)  # same identity: first wins
    implicit = _label(
        option_key="PATO:0000001",
        label_source="curator_implicit",
        weight=0.7,
        label="No",
        label_index=1,
    )
    assert store.insert_labels([_label(), dup, implicit]) == 2
    audit = AuditRow(
        task_key=TERM_KEY,
        artifact_version="v3",
        n=60,
        n_cards=4,
        cards=["a", "b", "c", "d"],
        reviewer="alice",
        n_errors=2,
        cp95_upper=0.09,
        risk=0.05,
        passed=True,
    )
    assert store.insert_audit(audit) == audit.audit_id
    calls = [
        ClmCallRow(
            run_id=run.run_id,
            endpoint="/v1/systemone",
            model="clm-latest",
            n_questions=1,
            n_candidates=3,
            input_tokens=140,
            latency_ms=12.5,
            cache_hit=False,
        ),
        ClmCallRow(
            run_id=run.run_id,
            endpoint="/v1/embeddings",
            model="qwen3-8b",
            n_questions=0,
            n_candidates=1,
            latency_ms=3.0,
            status="ok",
        ),
    ]
    assert store.insert_clm_calls(calls) == 2
    assert (
        store.insert_clm_call(ClmCallRow(endpoint="/health", model="clm-latest", latency_ms=1.0))
        is not None
    )
    store.finish_run(
        run.run_id,
        "applied",
        finished_at=now,
        n_decisions=2,
        n_clm_calls=2,
        n_encoder_tokens=140,
        seconds=1.5,
        irods_path="/iplant/home/x/f.csv",
        project_id="proj-1",
        history_backend="direct",
    )

    got = store.run(run.run_id)
    assert got is not None
    assert got["status"] == "applied" and got["n_decisions"] == 2 and got["seconds"] == 1.5
    assert got["history_backend"] == "direct" and got["plan_json"] == {}
    assert got["finished_at"].tzinfo is not None and got["finished_at"] == now
    assert got["mesa_clm_version"] and got["owner"] == "alice"
    decisions = store.decisions(run.run_id)
    assert [d["seq"] for d in decisions] == [1, 2]
    assert decisions[0]["probs"] == [0.6, 0.3, 0.1] and decisions[1]["probs"] is None
    assert decisions[0]["options"] == ["PATO:0000040", "PATO:0000001", "__none__"]
    assert decisions[0]["state_json"]["column"]["name"] == "observerDistance"
    assert decisions[0]["anchor_index"] == 2 and decisions[1]["anchor_index"] is None
    options = store.options(run.run_id)
    assert [o["option_key"] for o in options] == ["PATO:0000040", "PATO:0000001", "__none__"]
    assert options[0]["p_fit"] == 0.857 and options[2]["masked"] is False
    groups = store.groups(run.run_id)
    assert len(groups) == 1 and groups[0]["winner_decision_id"] == str(d1.decision_id)
    assert groups[0]["top_p_fit"] == 0.857 and groups[0]["search_json"] == {"queries": ["distance"]}
    assert store.group(group.group_id) == groups[0]
    links = store.links(run.run_id)  # ordered by attribute
    assert [link_["attribute"] for link_ in links] == ["envo.biome", "pato.distance"]
    written = links[1]
    assert written["write_status"] == "written" and written["accepted_by"] == "human"
    assert written["snapshot_id"] == 7 and written["project_id"] == "proj-1"
    assert written["written_at"] == now and written["irods_path"] == "/iplant/home/x/f.csv"
    assert links[0]["site_code"] == "SRER" and links[0]["column_name"] == "observerDistance"
    overrides = store.overrides(run.run_id)
    assert (
        len(overrides) == 1 and overrides[0]["via"] == "cli" and overrides[0]["labels_written"] == 2
    )
    assert overrides[0]["offered"][0]["option_key"] == "PATO:0000040"
    assert [c["endpoint"] for c in store.clm_calls(run.run_id)] == [
        "/v1/systemone",
        "/v1/embeddings",
    ]
    assert store.clm_calls(uuid4()) == []
    audits = store.audits(TERM_KEY)
    assert (
        len(audits) == 1
        and audits[0]["cards"] == ["a", "b", "c", "d"]
        and audits[0]["passed"] is True
    )
    assert store.audits("nope") == [] and len(store.audits()) == 1
    joined = store.decisions_for_path("/iplant/home/x/f.csv")
    assert len(joined) == 1
    assert joined[0]["level"] == "zero_shot" and joined[0]["snapshot_id"] == 7
    assert joined[0]["p_fit"] == 0.857 and joined[0]["task_id"] == "term.fits"
    assert store.decisions_for_path("/iplant/home/x/f.csv", limit=0) == []
    labels = store.labels_for("term.fits")
    assert [lab["label_source"] for lab in labels] == ["curator_implicit", "curator"]
    assert labels[1]["label"] == "Yes"  # the first row of the identity won
    assert store.labels_for("term.fits", min_weight=0.8) == [labels[1]]
    assert store.labels_for("term.fits", exclude_cards=["DP1.10003.001.brd_countdata"]) == []
    assert store.labels_for("term.fits", sources=["curator_implicit"]) == [labels[0]]
    with pytest.raises(KeyError):
        store.labels_for("no.such.task")
    store.close()  # nothing to release


def test_runs_filters_and_order(store: DuckDBStore) -> None:
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    a = _run(owner="alice", started_at=t0)
    b = _run(owner="bob", started_at=t0 + timedelta(hours=1), card_name="other")
    c = _run(owner="alice", started_at=t0 + timedelta(hours=2), status="failed")
    for r in (a, b, c):
        store.begin_run(r)
    assert [r["run_id"] for r in store.runs()] == [str(c.run_id), str(b.run_id), str(a.run_id)]
    assert [r["run_id"] for r in store.runs(owner="alice")] == [str(c.run_id), str(a.run_id)]
    assert [r["run_id"] for r in store.runs(status="failed")] == [str(c.run_id)]
    assert [r["run_id"] for r in store.runs(card_name="other")] == [str(b.run_id)]
    assert [r["run_id"] for r in store.runs(since=t0 + timedelta(minutes=30))] == [
        str(c.run_id),
        str(b.run_id),
    ]
    assert len(store.runs(limit=1)) == 1 and len(store.runs(limit=None)) == 3
    store.finish_run(a.run_id, "applied", irods_path="/p")
    assert [r["run_id"] for r in store.runs(irods_path="/p")] == [str(a.run_id)]


def test_finish_run_and_update_group_reject_unknown_columns_and_values(store: DuckDBStore) -> None:
    run = _run()
    store.begin_run(run)
    with pytest.raises(ValueError, match="not an updatable run column"):
        store.finish_run(run.run_id, "failed", status_reason="x")
    with pytest.raises(ValueError, match="not a run status"):
        store.finish_run(run.run_id, "dry_run")
    with pytest.raises(ValueError, match="history_backend"):
        store.finish_run(run.run_id, "applied", history_backend="auto")
    group = _group(run.run_id)
    store.insert_group(group)
    with pytest.raises(ValueError, match="not a summary column"):
        store.update_group(group.group_id, search_json={})
    store.update_group(group.group_id)  # nothing to set is a no-op
    with pytest.raises(ValueError, match="not a write status"):
        store.set_link_status([uuid4()], "rejected")
    with pytest.raises(ValueError, match="accepted_by"):
        store.set_link_status([uuid4()], "accepted", accepted_by="robot")


def test_delete_run_removes_the_run_tables_and_keeps_labels_and_audits(store: DuckDBStore) -> None:
    run = _run()
    store.begin_run(run)
    group = _group(run.run_id)
    store.insert_group(group)
    d = _decision(run.run_id, 1, group_id=group.group_id)
    store.insert_decisions(
        [d],
        [
            DecisionOptionRow(
                decision_id=d.decision_id,
                option_index=0,
                option_key="PATO:0000040",
                option_text="t",
            )
        ],
    )
    store.insert_links([_link(run.run_id)])
    store.insert_override(
        HumanOverrideRow(
            run_id=run.run_id, group_id=group.group_id, actor="a", via="tool", action="decline"
        )
    )
    store.insert_clm_call(
        ClmCallRow(run_id=run.run_id, endpoint="/v1/systemone", model="m", latency_ms=1)
    )
    store.insert_labels([_label()])
    store.insert_audit(
        AuditRow(
            task_key=TERM_KEY,
            artifact_version="v1",
            n=1,
            n_cards=1,
            cards=["a"],
            reviewer="r",
            n_errors=0,
            cp95_upper=0.9,
            risk=0.05,
            passed=False,
        )
    )
    other = _run()
    store.begin_run(other)
    deleted = store.delete_run(run.run_id)
    assert deleted == {
        "decision_options": 1,
        "clm_calls": 1,
        "human_overrides": 1,
        "avu_links": 1,
        "decisions": 1,
        "decision_groups": 1,
        "runs": 1,
    }
    assert store.run(run.run_id) is None and store.options(run.run_id) == []
    assert store.run(other.run_id) is not None
    assert len(store.labels_for("term.fits")) == 1 and len(store.audits()) == 1
    assert sum(store.delete_run(run.run_id).values()) == 0


# -- CHECKs enforced by the database ---------------------------------------------------------------------------


def _raw_decision_insert(store: DuckDBStore, **over: Any) -> None:
    """Insert a decision row with SQL, bypassing the pydantic mirror, so the CHECK is what fails."""
    row = _decision(uuid4(), 1)
    data = row.model_dump()
    data.update(over)
    from mesa_clm.provenance.store import _cell, insert_sql

    with store.connect() as con:
        con.execute(insert_sql("decisions", list(data)), [_cell(v) for v in data.values()])


@pytest.mark.parametrize(
    "bad",
    [
        {"probs": [0.5, 0.5], "calibration": "none"},  # probs without calibration
        {
            "probs": None,
            "raw_probs": None,
            "calibration": "uncalibrated",
        },  # calibration without probs
        {"method": "rule", "level": "zero_shot"},  # a rule with a level
        {
            "method": "planner",
            "probs": None,
            "raw_probs": None,
            "calibration": "none",
            "level": "calibrated",
        },
        {"level": "zero_shot", "calibration": "platt"},  # zero_shot must be uncalibrated
        {
            "level": "calibrated",
            "calibration": "uncalibrated",
        },  # calibrated needs platt/temperature
        {"level": "probe", "calibration": "uncalibrated"},
        {
            "method": "ols_rank",
            "probs": None,
            "raw_probs": None,
            "calibration": "none",
            "level": "none",
            "outcome": "auto",
            "rank": 1,
        },
        {
            "method": "ols_rank",
            "probs": None,
            "raw_probs": None,
            "calibration": "none",
            "level": "none",
            "outcome": "proposed",
            "rank": None,
        },
        {"shape": "choice", "anchor_index": 1},  # a choice has no anchor
        {"level": "L2"},  # outside the vocabulary
        {"method": "anyjev"},
        {"outcome": "declined"},
        {"scope": "table"},
        {"shape": "noul"},
    ],
)
def test_decision_checks_are_enforced_by_duckdb(store: DuckDBStore, bad: dict[str, Any]) -> None:
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        _raw_decision_insert(store, **bad)


def test_ols_rank_proposed_with_rank_is_accepted(store: DuckDBStore) -> None:
    run = _run(degraded=True)
    store.begin_run(run)
    d = _decision(
        run.run_id,
        1,
        method="ols_rank",
        model="",
        level="none",
        calibration="none",
        probs=None,
        raw_probs=None,
        confidence=None,
        clm_confidence=None,
        s_c=None,
        p_fit=None,
        outcome="proposed",
        rank=0,
    )
    assert store.insert_decisions([d]) == 1
    assert store.decisions(run.run_id)[0]["method"] == "ols_rank"


def test_link_and_override_checks_are_enforced_by_duckdb(store: DuckDBStore) -> None:
    run = _run()
    store.begin_run(run)
    link = _link(run.run_id)
    store.insert_links([link])
    with store.connect() as con:  # accepted without accepted_by (D10)
        with pytest.raises(duckdb.ConstraintException, match="CHECK"):
            con.execute(
                f"UPDATE {SCHEMA}.avu_links SET write_status = 'accepted' WHERE link_id = ?",  # noqa: S608
                [str(link.link_id)],
            )
        with pytest.raises(duckdb.ConstraintException, match="CHECK"):
            con.execute(
                f"UPDATE {SCHEMA}.avu_links SET write_status = 'rejected' WHERE link_id = ?",  # noqa: S608
                [str(link.link_id)],
            )
        # the sentinel UNIQUE: same triple, same (empty) column and site -> duplicate
        with pytest.raises(duckdb.ConstraintException, match=r"[Dd]uplicate|UNIQUE|unique"):
            dup = _link(run.run_id, link_id=uuid4())
            from mesa_clm.provenance.store import _cell, insert_sql

            data = dup.model_dump()
            con.execute(insert_sql("avu_links", list(data)), [_cell(v) for v in data.values()])
        # a tool pick minting curator labels (D21, defect g)
        with pytest.raises(duckdb.ConstraintException, match="CHECK"):
            con.execute(
                f"INSERT INTO {SCHEMA}.human_overrides (override_id, run_id, actor, via, action, "  # noqa: S608
                "label_source, offered, ts) VALUES ('o', ?, 'agent', 'tool', 'pick', 'curator', '[]', now())",
                [str(run.run_id)],
            )
    # a distinct column makes the same triple another link (sentinels, not NULLs)
    assert store.insert_links([_link(run.run_id, column_name="other")]) == 1
    # the pydantic mirror says the same thing earlier and by name
    with pytest.raises(ValidationError, match="accepted_by"):
        _link(run.run_id, write_status="accepted")
    with pytest.raises(ValidationError, match="agent_pick"):
        HumanOverrideRow(
            run_id=run.run_id, actor="a", via="tool", action="pick", label_source="curator"
        )
    HumanOverrideRow(
        run_id=run.run_id, actor="a", via="tool", action="pick", label_source="agent_pick"
    )


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"probs": None, "raw_probs": None}, "probs must be NULL exactly when"),
        ({"calibration": "none"}, "probs must be NULL exactly when"),
        ({"method": "rule"}, "carries no distribution"),
        ({"level": "zero_shot", "calibration": "platt"}, "zero_shot record is 'uncalibrated'"),
        ({"level": "head", "calibration": "uncalibrated"}, "needs a platt or temperature"),
        (
            {
                "method": "ols_rank",
                "probs": None,
                "raw_probs": None,
                "calibration": "none",
                "level": "none",
                "outcome": "auto",
            },
            "ols_rank proposes or abstains",
        ),
        ({"shape": "choice"}, "a choice has no anchor"),
        ({"k": -1}, "greater than or equal to 0"),
        ({"task_key": ""}, "non-empty"),
        ({"unknown": 1}, "extra"),
    ],
)
def test_row_models_mirror_the_checks(over: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _decision(uuid4(), 1, **over)


def test_audit_row_mirror() -> None:
    with pytest.raises(ValidationError, match="n_errors cannot exceed n"):
        AuditRow(
            task_key="k",
            artifact_version="v",
            n=1,
            n_cards=1,
            cards=[],
            reviewer="r",
            n_errors=2,
            cp95_upper=0.5,
            risk=0.05,
            passed=False,
        )
    with pytest.raises(ValidationError):
        AuditRow(
            task_key="k",
            artifact_version="v",
            n=1,
            n_cards=1,
            cards=[],
            reviewer="r",
            n_errors=0,
            cp95_upper=1.5,
            risk=0.05,
            passed=False,
        )


# -- the run buffer ---------------------------------------------------------------------------------------------


def test_run_buffer_commits_everything_in_one_transaction(store: DuckDBStore) -> None:
    run = _run()
    buf = RunBuffer(run)
    group = _group(run.run_id)
    assert buf.insert_group(group) == group.group_id
    d = _decision(run.run_id, 1, group_id=group.group_id)
    opts = [
        DecisionOptionRow(
            decision_id=d.decision_id, option_index=0, option_key="PATO:0000040", option_text="t"
        )
    ]
    assert buf.insert_decisions([d], opts) == 1
    buf.update_group(
        group.group_id, winner_decision_id=d.decision_id, outcome="proposed", top_p_fit=0.857
    )
    with pytest.raises(ValueError, match="not summary columns"):
        buf.update_group(group.group_id, search_json={})
    with pytest.raises(KeyError):
        buf.update_group(uuid4(), outcome="abstain")
    assert buf.insert_links([_link(run.run_id, decision_id=d.decision_id)]) == 1
    buf.insert_override(
        HumanOverrideRow(
            run_id=run.run_id, group_id=group.group_id, actor="a", via="tool", action="decline"
        )
    )
    buf.insert_clm_call(
        ClmCallRow(run_id=run.run_id, endpoint="/v1/systemone", model="m", latency_ms=2)
    )
    assert (
        buf.insert_clm_calls(
            [ClmCallRow(run_id=run.run_id, endpoint="/v1/systemone", model="m", latency_ms=3)]
        )
        == 1
    )
    with pytest.raises(ValueError, match="belongs to run"):
        buf.insert_links([_link(uuid4())])
    with pytest.raises(ValueError, match="not updatable run columns"):
        buf.finish_run("decided", n_prompts=1)
    buf.finish_run("decided", seconds=0.5)
    assert buf.run.status == "decided" and buf.run.n_decisions == 1 and buf.run.n_clm_calls == 2
    assert store.run(run.run_id) is None  # nothing touched the store yet
    assert store.commit_run(buf) == run.run_id and buf.committed
    with pytest.raises(ValueError, match="already committed"):
        store.commit_run(buf)
    got = store.run(run.run_id)
    assert got is not None and got["status"] == "decided" and got["n_clm_calls"] == 2
    assert store.groups(run.run_id)[0]["winner_decision_id"] == str(d.decision_id)
    assert len(store.decisions(run.run_id)) == 1 and len(store.options(run.run_id)) == 1
    assert len(store.links(run.run_id)) == 1 and len(store.overrides(run.run_id)) == 1
    assert len(store.clm_calls(run.run_id)) == 2


def test_run_buffer_rolls_back_as_a_whole(store: DuckDBStore) -> None:
    run = _run()
    buf = RunBuffer(run)
    buf.insert_group(_group(run.run_id))
    good = _decision(run.run_id, 1)
    bad = _decision(run.run_id, 2, decision_id=uuid4())
    bad.__dict__["calibration"] = "none"  # past pydantic: only the CHECK sees it
    buf.insert_decisions([good, bad])
    buf.insert_links([_link(run.run_id)])
    with pytest.raises(duckdb.ConstraintException):
        store.commit_run(buf)
    assert not buf.committed
    assert store.run(run.run_id) is None and store.groups(run.run_id) == []
    assert store.decisions(run.run_id) == [] and store.links(run.run_id) == []


def test_per_table_write_is_one_transaction_too(store: DuckDBStore) -> None:
    run = _run()
    store.begin_run(run)
    good = _decision(run.run_id, 1)
    bad = _decision(run.run_id, 2)
    bad.__dict__["level"] = "L2"
    with pytest.raises(duckdb.ConstraintException):
        store.insert_decisions([good, bad])
    assert store.decisions(run.run_id) == []


# -- dispatch and migrations ---------------------------------------------------------------------------------


def test_open_store_and_migrate_duckdb(tmp_path: Path) -> None:
    dsn = f"duckdb:///{tmp_path / 'p.duckdb'}"
    assert current_version(dsn) == 0 and not (tmp_path / "p.duckdb").exists()
    assert apply_migrations(dsn) == 1
    assert current_version(dsn) == 1
    store: ProvenanceStore = open_store(dsn)
    assert isinstance(store, DuckDBStore) and store.run(uuid4()) is None
    store.close()
    bare = open_store(str(tmp_path / "bare.duckdb"))
    assert isinstance(bare, DuckDBStore) and (tmp_path / "bare.duckdb").exists()
    assert duckdb_file("duckdb:///~/x.duckdb") == Path("~/x.duckdb").expanduser()
    assert duckdb_file("postgresql://u:p@h/db") is None
    for bad in ("sqlite:///x", "", "   ", "mysql://h/db"):
        with pytest.raises(ValueError):
            open_store(bad)
    with pytest.raises(ValueError):
        apply_migrations("sqlite:///x")
    with pytest.raises(ValueError):
        current_version("sqlite:///x")


def test_open_store_error_never_echoes_the_dsn() -> None:
    with pytest.raises(ValueError) as exc:
        open_store("mysql://user:secret@host/db")
    assert "secret" not in str(exc.value)


def test_packaged_migrations_are_numbered() -> None:
    files = migration_files()
    assert [v for v, _, _ in files] == [1]
    assert "CREATE SCHEMA IF NOT EXISTS mesa_clm" in files[0][2]
