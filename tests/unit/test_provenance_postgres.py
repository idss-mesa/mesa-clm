"""The Postgres dialect (``pg`` extra): the same round trip, CHECKs and column set as the
DuckDB store against a real server. Gated: ``MESA_CLM_TEST_PG_DSN=postgresql://user:pw@host/db``
names a throwaway database (the fixture drops and recreates schema ``mesa_clm``); skipped
otherwise (``requires_postgres`` marker, excluded by ``addopts``)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from mesa_clm.provenance.labels import LabelRow
from mesa_clm.provenance.migrate import apply_migrations, current_version
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
from mesa_clm.provenance.store import TABLES, DuckDBStore, RunBuffer, open_store
from mesa_clm.tasks import TASKS

pytestmark = pytest.mark.requires_postgres
DSN = os.environ.get("MESA_CLM_TEST_PG_DSN")
if not DSN:
    pytest.skip("MESA_CLM_TEST_PG_DSN not set", allow_module_level=True)
psycopg = pytest.importorskip("psycopg")

TERM_KEY = TASKS["term.fits"].key


@pytest.fixture
def pg() -> Iterator[Any]:
    assert DSN
    with psycopg.connect(DSN, autocommit=True) as con:
        con.execute("DROP SCHEMA IF EXISTS mesa_clm CASCADE")
    store = open_store(DSN)
    yield store
    store.close()
    with psycopg.connect(DSN, autocommit=True) as con:
        con.execute("DROP SCHEMA IF EXISTS mesa_clm CASCADE")


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


def _decision(run_id: Any, seq: int, **over: Any) -> DecisionRow:
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
        "column_name": "c",
        "state_sha256": "a" * 64,
        "state_json": {"card": {}, "candidate": {"curie": "PATO:1"}},
        "target_sha256": "t" * 64,
        "context_sha256": "x" * 64,
        "context_tokens": 100,
        "method": "fake",
        "model": "clm-latest",
        "level": "zero_shot",
        "calibration": "uncalibrated",
        "probs": [0.7, 0.2, 0.1],
        "raw_probs": [0.7, 0.2, 0.1],
        "confidence": 0.7,
        "s_c": 1.9,
        "p_fit": 0.87,
        "anchor_index": 2,
        "answer_index": 0,
        "answer": "PATO:1",
        "options": ["PATO:1", "PATO:2", "__none__"],
        "rank": 0,
        "encoder_fp": "e" * 12,
        "clm_model_fp": "m" * 12,
        "schema_sha256": "h" * 64,
        "outcome": "proposed",
    }
    base.update(over)
    return DecisionRow(**base)


def test_postgres_schema_matches_duckdb(pg: Any, tmp_path: Path) -> None:
    assert DSN and pg.ensure_schema() >= 1 and current_version(DSN) == 1
    assert apply_migrations(DSN) == 1  # idempotent
    duck = DuckDBStore(tmp_path / "p.duckdb")
    duck.ensure_schema()
    for table in TABLES:
        assert pg.columns(table) == duck.columns(table), table


def test_postgres_round_trip(pg: Any) -> None:
    run = _run()
    pg.begin_run(run)
    group = DecisionGroupRow(
        run_id=run.run_id,
        task_id="term.fits",
        task_key=TERM_KEY,
        scope="column",
        column_name="c",
        search_json={"queries": ["distance"]},
        n_candidates=2,
        outcome="proposed",
    )
    pg.insert_group(group)
    d1 = _decision(run.run_id, 1, group_id=group.group_id)
    opts = [
        DecisionOptionRow(
            decision_id=d1.decision_id, option_index=i, option_key=k, option_text=k, rank=i, prob=p
        )
        for i, (k, p) in enumerate([("PATO:1", 0.7), ("PATO:2", 0.2), ("__none__", 0.1)])
    ]
    assert pg.insert_decisions([d1], opts) == 1
    link = AvuLinkRow(
        run_id=run.run_id,
        group_id=group.group_id,
        decision_id=d1.decision_id,
        attribute="pato.distance",
        value="distance",
        unit="PATO:1",
        column_name="c",
    )
    assert pg.insert_links([link]) == 1
    assert pg.set_link_status([link.link_id], "accepted", accepted_by="policy") == 1
    now = datetime.now(tz=UTC)
    assert (
        pg.set_link_status(
            [link.link_id], "written", written_at=now, irods_path="/iplant/home/x/f.csv"
        )
        == 1
    )
    assert pg.link_snapshot(run.run_id, "/iplant/home/x/f.csv", "proj", 7) == 1
    assert pg.link_snapshot(run.run_id, "/iplant/home/x/f.csv", "proj", 8) == 0
    pg.update_group(
        group.group_id, outcome="human", winner_decision_id=d1.decision_id, top_p_fit=0.87
    )
    pg.insert_override(
        HumanOverrideRow(
            run_id=run.run_id,
            group_id=group.group_id,
            actor="alice",
            via="elicitation",
            action="pick",
            chosen_decision_id=d1.decision_id,
            label_source="curator",
            labels_written=1,
            offered=[{"option_key": "PATO:1"}],
        )
    )
    label = LabelRow(
        task_id="term.fits",
        task_key=TERM_KEY,
        target_sha256="d" * 64,
        option_key="PATO:1",
        label_source="curator",
        label="Yes",
        label_index=0,
        weight=1.0,
        state_sha256="a" * 64,
        state_json={},
        card="DP1.x",
    )
    assert pg.insert_labels([label, label.model_copy(update={"label_id": uuid4()})]) == 1
    pg.insert_audit(
        AuditRow(
            task_key=TERM_KEY,
            artifact_version="v1",
            n=50,
            n_cards=3,
            cards=["a", "b", "c"],
            reviewer="r",
            n_errors=1,
            cp95_upper=0.08,
            risk=0.05,
            passed=True,
        )
    )
    assert (
        pg.insert_clm_calls(
            [
                ClmCallRow(
                    run_id=run.run_id,
                    endpoint="/v1/systemone",
                    model="clm-latest",
                    n_questions=1,
                    n_candidates=3,
                    latency_ms=4.0,
                )
            ]
        )
        == 1
    )
    pg.finish_run(
        run.run_id, "applied", finished_at=now, n_decisions=1, seconds=1.5, history_backend="spool"
    )
    got = pg.run(run.run_id)
    assert got and got["status"] == "applied" and got["n_decisions"] == 1 and got["plan_json"] == {}
    assert got["history_backend"] == "spool" and got["run_id"] == str(run.run_id)
    assert [r["run_id"] for r in pg.runs(owner="alice")] == [str(run.run_id)]
    assert pg.group(group.group_id)["outcome"] == "human"
    decisions = pg.decisions(run.run_id)
    assert (
        decisions[0]["probs"] == [0.7, 0.2, 0.1]
        and decisions[0]["state_json"]["candidate"]["curie"] == "PATO:1"
    )
    assert decisions[0]["options"] == ["PATO:1", "PATO:2", "__none__"]
    assert [o["option_key"] for o in pg.options(run.run_id)] == ["PATO:1", "PATO:2", "__none__"]
    links = pg.links(run.run_id)
    assert (
        links[0]["snapshot_id"] == 7
        and links[0]["write_status"] == "written"
        and links[0]["accepted_by"] == "policy"
    )
    assert pg.overrides(run.run_id)[0]["offered"] == [{"option_key": "PATO:1"}]
    assert len(pg.clm_calls(run.run_id)) == 1 and len(pg.audits(TERM_KEY)) == 1
    assert pg.labels_for("term.fits", min_weight=0.6) and not pg.labels_for(
        "term.fits", exclude_cards=["DP1.x"]
    )
    assert pg.labels_for("term.fits", sources=[]) == []
    joined = pg.decisions_for_path("/iplant/home/x/f.csv")
    assert joined[0]["level"] == "zero_shot" and joined[0]["snapshot_id"] == 7
    # the CHECKs hold in Postgres too
    bad = _decision(run.run_id, 2, decision_id=uuid4())
    bad.__dict__["calibration"] = "none"
    with pytest.raises(Exception, match=r"check|constraint"):
        pg.insert_decisions([bad])
    assert pg.decisions(run.run_id) == decisions  # rolled back, connection usable
    fresh = AvuLinkRow(run_id=run.run_id, attribute="envo.biome", value="desert", unit="ENVO:1")
    assert pg.insert_links([fresh]) == 1
    with pytest.raises(Exception, match=r"check|constraint"):
        pg.set_link_status([fresh.link_id], "accepted")  # accepted_by missing (D10)
    assert [lk["write_status"] for lk in pg.links(run.run_id)] == ["proposed", "written"]
    with pytest.raises(Exception, match=r"check|constraint"):
        pg.insert_override(
            HumanOverrideRow.model_construct(
                override_id=uuid4(),
                run_id=run.run_id,
                actor="a",
                via="tool",
                action="pick",
                label_source="curator",
                offered=[],
                ts=now,
                labels_written=0,
                group_id=None,
                decision_id=None,
                link_id=None,
                chosen_decision_id=None,
                chosen_option_key=None,
                elicitation_key=None,
            )
        )
    with pytest.raises(Exception, match=r"duplicate|unique"):
        pg.insert_links(
            [
                AvuLinkRow(
                    run_id=run.run_id,
                    attribute="pato.distance",
                    value="distance",
                    unit="PATO:1",
                    column_name="c",
                )
            ]
        )
    deleted = pg.delete_run(run.run_id)
    assert deleted["runs"] == 1 and deleted["decisions"] == 1 and deleted["decision_options"] == 3
    assert deleted["avu_links"] == 2
    assert pg.run(run.run_id) is None and len(pg.labels_for("term.fits")) == 1


def test_postgres_commit_run_is_one_transaction(pg: Any) -> None:
    run = _run()
    buf = RunBuffer(run)
    good = _decision(run.run_id, 1)
    bad = _decision(run.run_id, 2)
    bad.__dict__["level"] = "L2"
    buf.insert_decisions([good, bad])
    with pytest.raises(Exception, match=r"check|constraint"):
        pg.commit_run(buf)
    assert pg.run(run.run_id) is None and not buf.committed
    ok = RunBuffer(_run())
    ok.insert_decisions([_decision(ok.run_id, 1)])
    ok.finish_run("decided")
    assert pg.commit_run(ok) == ok.run_id
    assert pg.run(ok.run_id)["n_decisions"] == 1


def test_export_from_postgres_imports_into_duckdb(pg: Any, tmp_path: Path) -> None:
    """The Parquet copy is dialect-neutral: a run exported from Postgres imports into a DuckDB
    sidecar with identical rows."""
    from mesa_clm.provenance.export import export_run, import_run, run_rows

    buf = RunBuffer(_run())
    group = DecisionGroupRow(
        run_id=buf.run_id,
        task_id="term.fits",
        task_key=TERM_KEY,
        scope="column",
        column_name="c",
        search_json={"queries": ["q"]},
        n_candidates=1,
        outcome="proposed",
    )
    buf.insert_group(group)
    d = _decision(buf.run_id, 1, group_id=group.group_id)
    buf.insert_decisions(
        [d],
        [
            DecisionOptionRow(
                decision_id=d.decision_id,
                option_index=0,
                option_key="PATO:1",
                option_text="t",
                prob=0.7,
            )
        ],
    )
    buf.insert_links(
        [AvuLinkRow(run_id=buf.run_id, decision_id=d.decision_id, attribute="a", value="v")]
    )
    buf.finish_run("decided", seconds=0.25)
    pg.commit_run(buf)
    written = export_run(pg, buf.run_id, tmp_path / "export")
    duck = DuckDBStore(tmp_path / "p.duckdb")
    assert import_run(duck, Path(written["manifest"]).parent) == buf.run_id
    assert run_rows(duck, buf.run_id) == run_rows(pg, buf.run_id)
