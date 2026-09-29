"""Parquet export of a run (deterministic, typed, manifest-verified), its import into another
sidecar, and terminal-only pruning (DESIGN D29; plan §7.3 step 9)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import duckdb
import pytest

from mesa_clm.provenance.export import (
    MANIFEST_FORMAT,
    PENDING_LINK_STATUSES,
    export_run,
    import_run,
    is_terminal,
    mark_terminal,
    prune,
    read_export,
    run_rows,
)
from mesa_clm.provenance.labels import LabelRow
from mesa_clm.provenance.models import (
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    RunRow,
)
from mesa_clm.provenance.store import RUN_TABLES, DuckDBStore, RunBuffer
from mesa_clm.tasks import TASKS

TERM_KEY = TASKS["term.fits"].key
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


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
        "started_at": T0,
    }
    base.update(over)
    return RunRow(**base)


def _populate(store: DuckDBStore, *, status: str = "decided", **run_over: Any) -> UUID:
    """A run with one group, two decisions (three options), two links, an override and a call."""
    run = _run(**run_over)
    buf = RunBuffer(run)
    group = DecisionGroupRow(
        run_id=run.run_id,
        task_id="term.fits",
        task_key=TERM_KEY,
        scope="column",
        column_name="observerDistance",
        search_json={"queries": ["distance"]},
        n_candidates=2,
        outcome="proposed",
        ts=T0,
    )
    buf.insert_group(group)
    d1 = DecisionRow(
        run_id=run.run_id,
        seq=1,
        group_id=group.group_id,
        task_id="term.fits",
        task_key=TERM_KEY,
        question_key="q" * 16,
        framing_id="F7",
        shape="rank_fit",
        k=3,
        scope="column",
        column_name="observerDistance",
        state_sha256="a" * 64,
        state_json={"card": {"dataset": "DP1.x"}},
        target_sha256="t" * 64,
        context_sha256="x" * 64,
        context_tokens=100,
        method="fake",
        model="clm-latest",
        level="zero_shot",
        calibration="uncalibrated",
        probs=[0.6, 0.3, 0.1],
        raw_probs=[0.6, 0.3, 0.1],
        confidence=0.6,
        clm_confidence=0.3,
        s_c=1.79,
        p_fit=0.857,
        anchor_index=2,
        answer_index=0,
        answer="PATO:0000040",
        options=["PATO:0000040", "PATO:0000001", "__none__"],
        rank=0,
        encoder_fp="e" * 12,
        clm_model_fp="m" * 12,
        schema_sha256="h" * 64,
        outcome="proposed",
        ts=T0,
    )
    d2 = DecisionRow(
        run_id=run.run_id,
        seq=2,
        task_id="column.annotate",
        task_key=TASKS["column.annotate"].key,
        question_key="r" * 16,
        framing_id="F7",
        shape="choice",
        k=2,
        scope="column",
        column_name="uid",
        state_sha256="b" * 64,
        state_json={},
        target_sha256="u" * 64,
        context_sha256="y" * 64,
        context_tokens=40,
        method="rule",
        level="none",
        calibration="none",
        answer_index=1,
        answer="No",
        options=["Yes", "No"],
        encoder_fp="e" * 12,
        clm_model_fp="m" * 12,
        schema_sha256="h" * 64,
        outcome="rule",
        reason="is_identifier",
        ts=T0,
    )
    opts = [
        DecisionOptionRow(
            decision_id=d1.decision_id, option_index=i, option_key=k, option_text=k, rank=i, prob=p
        )
        for i, (k, p) in enumerate(
            [("PATO:0000040", 0.6), ("PATO:0000001", 0.3), ("__none__", 0.1)]
        )
    ]
    buf.insert_decisions([d1, d2], opts)
    buf.update_group(group.group_id, winner_decision_id=d1.decision_id, top_p_fit=0.857)
    buf.insert_links(
        [
            AvuLinkRow(
                run_id=run.run_id,
                group_id=group.group_id,
                decision_id=d1.decision_id,
                attribute="pato.distance",
                value="distance",
                unit="PATO:0000040",
                column_name="observerDistance",
            ),
            AvuLinkRow(
                run_id=run.run_id,
                attribute="envo.biome",
                value="desert",
                unit="ENVO:1",
                site_code="SRER",
            ),
        ]
    )
    buf.insert_override(
        HumanOverrideRow(
            run_id=run.run_id,
            group_id=group.group_id,
            actor="alice",
            via="cli",
            action="pick",
            chosen_decision_id=d1.decision_id,
            label_source="curator",
            labels_written=2,
            offered=[{"option_key": "PATO:0000040"}],
            ts=T0,
        )
    )
    buf.insert_clm_call(
        ClmCallRow(
            run_id=run.run_id,
            endpoint="/v1/systemone",
            model="clm-latest",
            n_questions=1,
            n_candidates=3,
            latency_ms=5.0,
            ts=T0,
        )
    )
    buf.finish_run(status, finished_at=T0 + timedelta(seconds=3), seconds=3.0)
    store.commit_run(buf)
    return run.run_id


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def store(tmp_path: Path) -> DuckDBStore:
    s = DuckDBStore(tmp_path / "prov.duckdb")
    s.ensure_schema()
    return s


# -- export -----------------------------------------------------------------------------------------------------


def test_export_writes_every_run_table_typed_and_verified(
    store: DuckDBStore, tmp_path: Path
) -> None:
    run_id = _populate(store)
    stamp = datetime(2026, 9, 2, tzinfo=UTC)
    written = export_run(store, run_id, tmp_path / "export", now=stamp)
    target = tmp_path / "export" / str(run_id)
    assert set(written) == {*RUN_TABLES, "manifest"}
    assert all(Path(p).parent == target for p in written.values())
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == MANIFEST_FORMAT and manifest["run_id"] == str(run_id)
    assert manifest["exported_at"] == stamp.isoformat() and manifest["schema_version"] == 1
    con = duckdb.connect()
    for table in RUN_TABLES:
        entry = manifest["tables"][table]
        path = target / entry["file"]
        assert path.name == f"{table}.parquet"
        assert entry["sha256"] == _sha(path) and entry["bytes"] == path.stat().st_size
        n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()
        assert n is not None and n[0] == entry["rows"]
    assert (
        manifest["tables"]["decisions"]["rows"] == 2
        and manifest["tables"]["decision_options"]["rows"] == 3
    )
    assert (
        manifest["tables"]["avu_links"]["rows"] == 2
        and manifest["tables"]["human_overrides"]["rows"] == 1
    )
    # typed columns straight from the schema, JSON stays JSON
    described = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [written["decisions"]]
    ).fetchall()
    types = {str(r[0]): str(r[1]) for r in described}
    assert types["probs"] == "JSON" and types["ts"] == "TIMESTAMP WITH TIME ZONE"
    assert types["p_fit"] == "DOUBLE" and types["decision_id"] == "VARCHAR"
    order = con.execute("SELECT seq FROM read_parquet(?)", [written["decisions"]]).fetchall()
    assert order == [(1,), (2,)]
    con.close()
    # the run row was stamped before the copy was taken
    run = store.run(run_id)
    assert run is not None and run["exported_at"] == stamp
    exported = read_export(target)["runs"][0]
    assert exported["exported_at"] == stamp
    assert not list(target.glob(".*.tmp"))


def test_export_is_deterministic(store: DuckDBStore, tmp_path: Path) -> None:
    run_id = _populate(store)
    stamp = datetime(2026, 9, 2, tzinfo=UTC)
    a = export_run(store, run_id, tmp_path / "a", now=stamp)
    b = export_run(store, run_id, tmp_path / "b", now=stamp)
    for table in RUN_TABLES:
        assert _sha(a[table]) == _sha(b[table]), table
    assert Path(a["manifest"]).read_text() == Path(b["manifest"]).read_text()
    with pytest.raises(KeyError):
        export_run(store, uuid4(), tmp_path / "c")


def test_read_export_verifies_the_manifest(store: DuckDBStore, tmp_path: Path) -> None:
    run_id = _populate(store)
    written = export_run(store, run_id, tmp_path / "export")
    target = Path(written["manifest"]).parent
    assert read_export(written["manifest"]) == read_export(target)
    with pytest.raises(FileNotFoundError):
        read_export(tmp_path / "nowhere")
    tampered = Path(written["avu_links"])
    tampered.write_bytes(tampered.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        read_export(target)
    tampered.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        read_export(target)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["format"] = "other/9"
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format"):
        read_export(target)


# -- import ---------------------------------------------------------------------------------------------------------


def test_import_round_trips_into_another_store(store: DuckDBStore, tmp_path: Path) -> None:
    run_id = _populate(store, status="applied")
    written = export_run(store, run_id, tmp_path / "export")
    other = DuckDBStore(tmp_path / "other.duckdb")
    assert import_run(other, Path(written["manifest"]).parent) == run_id
    assert run_rows(other, run_id) == run_rows(store, run_id)
    assert other.decisions(run_id)[0]["probs"] == [0.6, 0.3, 0.1]
    assert other.run(run_id)["exported_at"] == store.run(run_id)["exported_at"]  # type: ignore[index]
    with pytest.raises(ValueError, match="already exists"):
        import_run(other, Path(written["manifest"]).parent)
    with pytest.raises(KeyError):
        run_rows(other, uuid4())


# -- terminal runs and prune ----------------------------------------------------------------------------------------


def test_is_terminal_rules() -> None:
    written = {"write_status": "written", "snapshot_id": 3}
    assert is_terminal({"status": "applied"}, [written], [])
    assert is_terminal(
        {"status": "abandoned"},
        [{"write_status": "writing", "snapshot_id": None}],
        [{"outcome": "escalated"}],
    )
    assert not is_terminal({"status": "decided"}, [], [])
    assert not is_terminal({"status": "running"}, [], [])
    assert not is_terminal(
        {"status": "applied"}, [{"write_status": "written", "snapshot_id": None}], []
    )
    assert not is_terminal(
        {"status": "applied"}, [{"write_status": "spooled", "snapshot_id": None}], []
    )
    for pending in PENDING_LINK_STATUSES:
        assert not is_terminal(
            {"status": "applied"}, [{"write_status": pending, "snapshot_id": None}], []
        )
    # proposals nobody accepted, reverted or local-only links do not keep a run open
    for free in ("proposed", "reverted", "local_only", "dry_run"):
        assert is_terminal(
            {"status": "applied"}, [written, {"write_status": free, "snapshot_id": None}], []
        )
    assert not is_terminal({"status": "applied"}, [written], [{"outcome": "escalated"}])
    assert is_terminal(
        {"status": "applied"}, [written], [{"outcome": "human"}, {"outcome": "proposed"}]
    )


def test_mark_terminal_stamps_once(store: DuckDBStore) -> None:
    run_id = _populate(store, status="applied")
    links = store.links(run_id)
    # a written link without its snapshot keeps the run open (plan §7.3 step 9)
    store.set_link_status(
        [UUID(links[0]["link_id"])], "written", written_at=datetime.now(tz=UTC), irods_path="/p"
    )
    assert not mark_terminal(store, run_id)
    run = store.run(run_id)
    assert run is not None and run["terminal_at"] is None
    store.link_snapshot(run_id, "/p", "proj", 11)
    stamp = datetime(2026, 9, 3, tzinfo=UTC)
    assert mark_terminal(store, run_id, now=stamp)
    assert store.run(run_id)["terminal_at"] == stamp  # type: ignore[index]
    assert mark_terminal(store, run_id, now=stamp + timedelta(days=1))
    assert store.run(run_id)["terminal_at"] == stamp  # type: ignore[index]
    with pytest.raises(KeyError):
        mark_terminal(store, uuid4())


def test_prune_deletes_only_terminal_or_expired_runs(store: DuckDBStore, tmp_path: Path) -> None:
    now = T0 + timedelta(days=45)
    running = _populate(store, status="running")
    decided = _populate(store, status="decided")
    applied_unexported = _populate(store, status="applied")
    applied_pending = _populate(store, status="applied")
    applied_open_group = _populate(store, status="applied")
    applied_terminal = _populate(store, status="applied")
    abandoned_young = _populate(store, status="abandoned", started_at=now - timedelta(days=2))
    abandoned_old = _populate(store, status="abandoned")
    failed_old = _populate(store, status="failed")
    store.insert_labels(
        [
            LabelRow(
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
            )
        ]
    )
    for run_id in (applied_pending, applied_open_group, applied_terminal):
        link_id = UUID(store.links(run_id)[0]["link_id"])
        store.set_link_status([link_id], "written", written_at=now, irods_path="/p")
        if run_id != applied_pending:
            store.link_snapshot(run_id, "/p", "proj", 5)
        export_run(store, run_id, tmp_path / "export", now=now)
    store.update_group(UUID(store.groups(applied_open_group)[0]["group_id"]), outcome="escalated")
    # abandoned_young: finished_at from _populate is T0 + 3 s; make it recent
    store.finish_run(abandoned_young, "abandoned", finished_at=now - timedelta(days=2))

    report = prune(store, ttl_days=30, now=now)
    assert set(report.deleted) == {applied_terminal, abandoned_old}
    assert (
        report.rows["runs"] == 2 and report.rows["decisions"] == 4 and report.rows["avu_links"] == 4
    )
    assert report.skipped[running] == "running"
    assert report.skipped[decided] == "decided: not terminal"
    assert report.skipped[applied_unexported] == "applied but not exported"
    assert "not terminal" in report.skipped[applied_pending]
    assert "not terminal" in report.skipped[applied_open_group]
    assert report.skipped[abandoned_young].startswith("abandoned 2.0 d ago")
    assert report.skipped[failed_old] == "failed: not terminal"
    assert store.run(applied_terminal) is None and store.run(abandoned_old) is None
    assert store.run(applied_pending) is not None and len(store.labels_for("term.fits")) == 1
    summary = report.summary()
    assert set(summary) == {"deleted", "rows", "skipped"} and len(summary["deleted"]) == 2

    # the escalated group gets its answer: now terminal
    store.update_group(UUID(store.groups(applied_open_group)[0]["group_id"]), outcome="human")
    again = prune(store, ttl_days=30, now=now)
    assert again.deleted == [applied_open_group]

    # the escape hatch: non-terminal runs older than the TTL go too, running never
    loose = prune(store, terminal_only=False, ttl_days=30, now=now)
    assert set(loose.deleted) == {decided, failed_old}
    assert loose.skipped[running] == "running"
    assert loose.skipped[applied_unexported] == "applied but not exported"
    assert store.run(applied_pending) is not None and store.run(abandoned_young) is not None
    with pytest.raises(ValueError):
        prune(store, ttl_days=-1)
