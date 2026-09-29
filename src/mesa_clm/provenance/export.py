"""Parquet copies of a run, their import into another sidecar, and terminal-only pruning
(DESIGN D29; plan §7.3 step 9).

``export_run`` writes one Parquet file per run table under ``<out_dir>/<run_id>/`` (the project's
``.mesa/clm/runs/<run_id>/`` in production, never inside ``.mesa/ducklake/``, which belongs to
mesa-ducklake) plus a ``manifest.json`` with the sha256 and row count of every file. The rows go
through an in-memory DuckDB that carries the sidecar DDL itself, so the Parquet columns are the
schema's types (JSON stays JSON, timestamps stay ``TIMESTAMPTZ``) even for an empty table and the
CHECKs re-validate every row on the way out; ``ORDER BY`` per table
(:data:`mesa_clm.provenance.store.RUN_TABLE_ORDER`) makes the bytes a function of the rows.
The run row is stamped ``exported_at`` first, so the copy says when it was taken.

``import_run`` reads such a directory back (sha256 and row counts verified against the manifest)
and commits it into a store in one transaction. ``prune`` deletes the local rows of runs that are
*terminal*: ``applied`` and exported, every link that reached the write path ``written`` or
``spooled`` with a ``snapshot_id`` and no group still waiting for a human (``escalated``); or
``abandoned`` for longer than ``ttl_days``. Labels and audits are never pruned (D30).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from uuid import UUID

import duckdb
from pydantic import BaseModel

from mesa_clm import __version__
from mesa_clm.provenance.models import (
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    RunRow,
)
from mesa_clm.provenance.store import (
    DUCKDB_DDL,
    RUN_TABLE_ORDER,
    RUN_TABLES,
    SCHEMA,
    SCHEMA_VERSION,
    ProvenanceStore,
    RunBuffer,
    insert_sql,
    row_values,
)

MANIFEST_FORMAT: Final = "mesa-clm-run/1"
MANIFEST_NAME: Final = "manifest.json"

# The row model of each exported table (``import_run`` re-validates through it).
RUN_TABLE_MODELS: Final[dict[str, type[BaseModel]]] = {
    "runs": RunRow,
    "decision_groups": DecisionGroupRow,
    "decisions": DecisionRow,
    "decision_options": DecisionOptionRow,
    "avu_links": AvuLinkRow,
    "human_overrides": HumanOverrideRow,
    "clm_calls": ClmCallRow,
}

# Link states that mean "accepted for writing but not yet in the history": any of these keeps
# a run open. ``written``/``spooled`` links need a ``snapshot_id``; ``proposed`` (never
# accepted), ``reverted`` and ``local_only`` links do not block (plan §7.3 step 9).
PENDING_LINK_STATUSES: Final[frozenset[str]] = frozenset({"accepted", "writing", "mirror_failed"})
RECORDED_LINK_STATUSES: Final[frozenset[str]] = frozenset({"written", "spooled"})
# A group escalated to a human whose answer has not arrived (the answer rewrites the outcome).
OPEN_GROUP_OUTCOMES: Final[frozenset[str]] = frozenset({"escalated"})
TERMINAL_RUN_STATUSES: Final[frozenset[str]] = frozenset({"applied", "abandoned"})


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _lit(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_rows(store: ProvenanceStore, run_id: UUID) -> dict[str, list[dict[str, Any]]]:
    """Every run table's rows for one run, in :data:`RUN_TABLES` order (``KeyError`` for an
    unknown run)."""
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} not found")
    return {
        "runs": [run],
        "decision_groups": store.groups(run_id),
        "decisions": store.decisions(run_id),
        "decision_options": store.options(run_id),
        "avu_links": store.links(run_id),
        "human_overrides": store.overrides(run_id),
        "clm_calls": store.clm_calls(run_id),
    }


def _scratch() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB carrying the sidecar schema (typed, CHECK-validated staging)."""
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    for stmt in DUCKDB_DDL:
        con.execute(stmt)
    return con


def _stage(con: duckdb.DuckDBPyConnection, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Insert fetched rows (dicts) into the staging table through the row model, so a foreign
    or stale row fails validation instead of landing in a file."""
    model = RUN_TABLE_MODELS[table]
    validated = [model.model_validate(dict(r)) for r in rows]
    if not validated:
        return
    cols, _ = row_values(validated[0])
    con.executemany(insert_sql(table, cols), [row_values(r)[1] for r in validated])


def export_run(
    store: ProvenanceStore, run_id: UUID, out_dir: str | Path, *, now: datetime | None = None
) -> dict[str, str]:
    """Write ``<out_dir>/<run_id>/<table>.parquet`` for every run table (empty tables included,
    typed) plus ``manifest.json``; returns ``{table: path, "manifest": path}``. The run is
    stamped ``exported_at`` before the copy is taken and every file is read back (row count) and
    hashed into the manifest."""
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} not found")
    stamp = now or _now()
    store.finish_run(run_id, str(run["status"]), exported_at=stamp)
    frames = run_rows(store, run_id)
    target = Path(out_dir).expanduser() / str(run_id)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    tables: dict[str, dict[str, Any]] = {}
    con = _scratch()
    try:
        for table in RUN_TABLES:
            _stage(con, table, frames[table])
            path = target / f"{table}.parquet"
            tmp = target / f".{table}.parquet.tmp"
            con.execute(  # COPY targets cannot be bound parameters; the path is a quoted literal
                f"COPY (SELECT * FROM {SCHEMA}.{table} ORDER BY {RUN_TABLE_ORDER[table]}) "  # noqa: S608
                f"TO {_lit(tmp)} (FORMAT PARQUET)"
            )
            got = con.execute(f"SELECT count(*) FROM read_parquet({_lit(tmp)})").fetchone()  # noqa: S608
            n = int(got[0]) if got else -1
            if n != len(frames[table]):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{table}: wrote {len(frames[table])} rows, read back {n}")
            os.replace(tmp, path)
            written[table] = str(path)
            tables[table] = {
                "file": path.name,
                "rows": n,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    finally:
        con.close()
    manifest = {
        "format": MANIFEST_FORMAT,
        "run_id": str(run_id),
        "exported_at": stamp.isoformat(),
        "mesa_clm_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "tables": tables,
    }
    manifest_path = target / MANIFEST_NAME
    tmp_manifest = target / f".{MANIFEST_NAME}.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)  # manifest last: its presence means "complete"
    written["manifest"] = str(manifest_path)
    return written


def read_export(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """The rows of an exported run directory (or its manifest path), table by table, after
    verifying every file's sha256 and row count against the manifest."""
    where = Path(path).expanduser()
    manifest_path = where if where.name == MANIFEST_NAME else where / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: no run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"{manifest_path}: format {manifest.get('format')!r} is not {MANIFEST_FORMAT}"
        )
    tables = manifest.get("tables") or {}
    missing = [t for t in RUN_TABLES if t not in tables]
    if missing:
        raise ValueError(f"{manifest_path}: manifest lacks tables {missing}")
    out: dict[str, list[dict[str, Any]]] = {}
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        for table in RUN_TABLES:
            entry = tables[table]
            file = manifest_path.parent / str(entry["file"])
            if not file.is_file():
                raise FileNotFoundError(f"{file}: listed in the manifest but missing")
            digest = _sha256(file)
            if digest != entry["sha256"]:
                raise ValueError(f"{file}: sha256 {digest[:12]} differs from the manifest")
            cur = con.execute(f"SELECT * FROM read_parquet({_lit(file)})")  # noqa: S608
            names = [d[0] for d in cur.description or []]
            rows = [dict(zip(names, v, strict=True)) for v in cur.fetchall()]
            if len(rows) != int(entry["rows"]):
                raise ValueError(f"{file}: {len(rows)} rows, the manifest says {entry['rows']}")
            out[table] = rows
    finally:
        con.close()
    return out


def _rows_as(model: type[BaseModel], rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Fetched Parquet rows as validated row models (JSON text parsed back)."""
    out = []
    for r in rows:
        data = dict(r)
        for key, value in data.items():
            if isinstance(value, str) and key in _JSON_FIELDS.get(model, ()):
                data[key] = json.loads(value)
        out.append(model.model_validate(data))
    return out


def _json_fields(model: type[BaseModel]) -> frozenset[str]:
    return frozenset(
        name
        for name, info in model.model_fields.items()
        if "dict[" in str(info.annotation) or "list[" in str(info.annotation)
    )


_JSON_FIELDS: Final[dict[type[BaseModel], frozenset[str]]] = {
    m: _json_fields(m) for m in RUN_TABLE_MODELS.values()
}


def import_run(store: ProvenanceStore, path: str | Path) -> UUID:
    """Commit an exported run directory into ``store`` in one transaction; returns the run id.
    ``ValueError`` when the store already holds that run (imports never overwrite)."""
    frames = read_export(path)
    runs = _rows_as(RunRow, frames["runs"])
    if len(runs) != 1:
        raise ValueError(f"{path}: expected one run row, found {len(runs)}")
    run = runs[0]
    if store.run(run.run_id) is not None:
        raise ValueError(f"run {run.run_id} already exists in the store")
    buffer = RunBuffer(run)
    for row in _rows_as(DecisionGroupRow, frames["decision_groups"]):
        buffer.insert_group(row)
    buffer.insert_decisions(
        _rows_as(DecisionRow, frames["decisions"]),
        _rows_as(DecisionOptionRow, frames["decision_options"]),
    )
    buffer.insert_links(_rows_as(AvuLinkRow, frames["avu_links"]))
    for row in _rows_as(HumanOverrideRow, frames["human_overrides"]):
        buffer.insert_override(row)
    buffer.insert_clm_calls(_rows_as(ClmCallRow, frames["clm_calls"]))
    return store.commit_run(buffer)


# -- terminal runs and prune -------------------------------------------------------------------------


def is_terminal(
    run: Mapping[str, Any],
    links: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> bool:
    """Plan §7.3 step 9: an ``abandoned`` run is terminal; an ``applied`` run is terminal when no
    link is still on the write path (:data:`PENDING_LINK_STATUSES`), every ``written`` or
    ``spooled`` link carries a ``snapshot_id`` and no group waits for a human
    (:data:`OPEN_GROUP_OUTCOMES`). Every other status is open."""
    status = str(run.get("status"))
    if status not in TERMINAL_RUN_STATUSES:
        return False
    if status == "abandoned":
        return True
    for link in links:
        ws = str(link.get("write_status"))
        if ws in PENDING_LINK_STATUSES:
            return False
        if ws in RECORDED_LINK_STATUSES and link.get("snapshot_id") is None:
            return False
    return all(str(g.get("outcome")) not in OPEN_GROUP_OUTCOMES for g in groups)


def mark_terminal(store: ProvenanceStore, run_id: UUID, *, now: datetime | None = None) -> bool:
    """Stamp ``terminal_at`` on a run that has become terminal (once); ``True`` when it is."""
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"run {run_id} not found")
    if not is_terminal(run, store.links(run_id), store.groups(run_id)):
        return False
    if run.get("terminal_at") is None:
        store.finish_run(run_id, str(run["status"]), terminal_at=now or _now())
    return True


@dataclass
class PruneReport:
    """What ``prune`` did: the runs deleted, rows removed per table, and why the rest stayed."""

    deleted: list[UUID] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    skipped: dict[UUID, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "deleted": [str(r) for r in self.deleted],
            "rows": dict(sorted(self.rows.items())),
            "skipped": {str(k): v for k, v in self.skipped.items()},
        }


def _age_days(run: Mapping[str, Any], now: datetime) -> float:
    ended = run.get("finished_at") or run.get("started_at")
    if not isinstance(ended, datetime):
        return 0.0
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=UTC)
    return (now - ended) / timedelta(days=1)


def prune(
    store: ProvenanceStore,
    *,
    terminal_only: bool = True,
    ttl_days: int,
    now: datetime | None = None,
) -> PruneReport:
    """Delete local rows of finished runs (plan §7.3 step 9).

    Always eligible: an ``applied`` run that is terminal (:func:`is_terminal`) *and* exported
    (``exported_at`` set; D29 keeps the Parquet copy in the project); an ``abandoned`` run
    older than ``ttl_days``. With ``terminal_only=False``, any other non-running run
    (``decided``, ``partial``, ``failed``) older than ``ttl_days`` goes too, a maintenance
    escape hatch that forfeits its provenance. A ``running`` run is never touched; labels and
    audits never are (D30). Returns a :class:`PruneReport`.
    """
    if ttl_days < 0:
        raise ValueError("ttl_days must be >= 0")
    moment = now or _now()
    report = PruneReport()
    for run in store.runs(limit=None):
        run_id = UUID(str(run["run_id"]))
        status = str(run["status"])
        if status == "running":
            report.skipped[run_id] = "running"
            continue
        age = _age_days(run, moment)
        if status == "abandoned":
            if age < ttl_days:
                report.skipped[run_id] = f"abandoned {age:.1f} d ago (ttl {ttl_days} d)"
                continue
        elif status == "applied":
            if run.get("exported_at") is None:
                report.skipped[run_id] = "applied but not exported"
                continue
            if not is_terminal(run, store.links(run_id), store.groups(run_id)):
                report.skipped[run_id] = "applied but not terminal (pending links or open groups)"
                continue
        elif terminal_only:
            report.skipped[run_id] = f"{status}: not terminal"
            continue
        elif age < ttl_days:
            report.skipped[run_id] = f"{status} {age:.1f} d ago (ttl {ttl_days} d)"
            continue
        for table, n in store.delete_run(run_id).items():
            report.rows[table] = report.rows.get(table, 0) + n
        report.deleted.append(run_id)
    return report
