"""The ``mesa_clm`` sidecar: DuckDB dialect DDL, the per-operation flock store, the run buffer
and the store protocol (DESIGN D10, D11, D12, D21, D28; plan §4.6).

**DDL.** ``DUCKDB_DDL`` is the one source of the schema: one ``CREATE ... IF NOT EXISTS``
statement per object in DuckDB dialect (JSON, TEXT ids, no foreign keys), the ``labels``
statements spliced in *verbatim* from :data:`mesa_clm.provenance.labels.LABELS_DDL` (imported,
never copied), every vocabulary CHECK rendered from :mod:`mesa_clm.vocab` through
``sql_in_list``. The Postgres migration (``migrations/0001_mesa_clm.sql``) is the mechanical
translation :func:`mesa_clm.provenance.migrate.render_postgres_migration` makes of these
statements (``JSON`` -> ``JSONB``, ``DOUBLE`` -> ``DOUBLE PRECISION``, plus indexes), so the two
dialects cannot drift; ``tests/unit/test_provenance_ddl.py`` compares them table by table.

DuckDB 1.5.x rejects ``UNIQUE NULLS NOT DISTINCT`` (verified, RESEARCH.md), so every column
inside a UNIQUE key is ``NOT NULL`` (``''`` sentinels for optional text: ``avu_links.unit``,
``column_name``, ``site_code``; the labels identity columns). No uniqueness rule of this schema
needed a NULL-able key, so :data:`CODE_CHECKED_UNIQUENESS` is empty; the parity test asserts
that it lists exactly the constraints missing from the DDL.

**Locking (D11).** :class:`DuckDBStore` never holds a connection between calls. Every public
method is one operation: take ``fcntl.flock`` on :func:`mesa_clm.provenance.labels.lock_path_for`
(``<dir>/locks/provenance.lock``, the lock the M0 :class:`LabelStore` uses over the same file,
so the CLI, the bench and the MCP plugin serialise on one file), open the DuckDB file, run one
transaction, close, release. Writers take the exclusive lock; reads open the file ``read_only``
under a shared lock, which DuckDB allows several processes to hold at once. A read on a file that
does not exist returns nothing and creates nothing (D29).

**Buffering (D11).** A run's rows are collected in a :class:`RunBuffer` while the pipeline runs
(no lock held, no half-written run visible) and written by :meth:`DuckDBStore.commit_run` in one
transaction. The per-table methods (``begin_run``, ``insert_decisions``, ...) exist for the write
phase, the tools and tests, each one its own transaction.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast
from uuid import UUID

import duckdb
from pydantic import BaseModel

from mesa_clm.provenance.labels import LABELS_DDL, LabelRow, LabelStore, lock_path_for
from mesa_clm.provenance.models import (
    CALIBRATED_CALIBRATIONS,
    CALIBRATED_LEVELS,
    CALL_STATUSES,
    GROUP_SUMMARY_COLUMNS,
    HISTORY_BACKENDS,
    JSON_COLUMNS,
    LINK_OPS,
    OVERRIDE_ACTIONS,
    RUN_FINISH_COLUMNS,
    SCOPES,
    AuditRow,
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    RunRow,
)
from mesa_clm.vocab import (
    ACCEPTED_BY,
    CALIBRATIONS,
    LABEL_SOURCES,
    LEVELS,
    METHODS,
    METHODS_WITHOUT_PROBS,
    OUTCOMES,
    RUN_STATUSES,
    SHAPES,
    VIAS,
    WRITE_STATUSES,
    sql_in_list,
)

SCHEMA: Final = "mesa_clm"
SCHEMA_VERSION: Final = 1

# Sorted so the rendered CHECK is stable across processes (a frozenset has no order).
_METHODS_WITHOUT_PROBS_SQL: Final = sql_in_list(tuple(sorted(METHODS_WITHOUT_PROBS)))
_CALIBRATED_LEVELS_SQL: Final = sql_in_list(CALIBRATED_LEVELS)
_CALIBRATED_CALIBRATIONS_SQL: Final = sql_in_list(CALIBRATED_CALIBRATIONS)

# One statement per object, DuckDB dialect, one column per line (the parity test parses this
# layout). Column order = the field order of the matching row model in ``models.py``.
DUCKDB_DDL: Final[tuple[str, ...]] = (
    f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_versions (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.runs (
        run_id TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({sql_in_list(RUN_STATUSES)})),
        started_at TIMESTAMPTZ NOT NULL,
        finished_at TIMESTAMPTZ,
        terminal_at TIMESTAMPTZ,
        exported_at TIMESTAMPTZ,
        card_name TEXT NOT NULL,
        card_sha256 TEXT NOT NULL,
        irods_path TEXT,
        project_id TEXT,
        planner TEXT NOT NULL,
        planner_model TEXT,
        planner_prompt_sha256 TEXT,
        plan_json JSON NOT NULL,
        planner_fallback BOOLEAN NOT NULL DEFAULT FALSE,
        provider TEXT NOT NULL,
        tier TEXT NOT NULL,
        clm_model TEXT NOT NULL,
        encoder_model TEXT NOT NULL,
        clm_commit TEXT NOT NULL DEFAULT '',
        schema_sha256 TEXT NOT NULL,
        encoder_fp TEXT NOT NULL,
        clm_model_fp TEXT NOT NULL,
        serving_lock_sha TEXT,
        framings_lock_sha TEXT NOT NULL,
        artifacts_version TEXT,
        labels_sha256 TEXT,
        mesa_clm_version TEXT NOT NULL,
        policy_profile TEXT NOT NULL,
        config_sha256 TEXT NOT NULL,
        history_backend TEXT CHECK (history_backend IS NULL OR history_backend IN ({sql_in_list(HISTORY_BACKENDS)})),
        history_waiver_actor TEXT,
        vm_id TEXT NOT NULL,
        degraded BOOLEAN NOT NULL DEFAULT FALSE,
        n_decisions INTEGER,
        n_clm_calls INTEGER,
        n_encoder_tokens BIGINT,
        seconds DOUBLE)""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.decisions (
        decision_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        group_id TEXT,
        parent_decision_id TEXT,
        task_id TEXT NOT NULL,
        task_key TEXT NOT NULL,
        question_key TEXT NOT NULL,
        framing_id TEXT NOT NULL,
        shape TEXT NOT NULL CHECK (shape IN ({sql_in_list(SHAPES)})),
        k INTEGER NOT NULL,
        scope TEXT NOT NULL CHECK (scope IN ({sql_in_list(SCOPES)})),
        column_name TEXT,
        site_code TEXT,
        state_sha256 TEXT NOT NULL,
        state_json JSON NOT NULL,
        target_sha256 TEXT NOT NULL,
        context_sha256 TEXT NOT NULL,
        context_tokens INTEGER NOT NULL,
        truncated BOOLEAN NOT NULL DEFAULT FALSE,
        method TEXT NOT NULL CHECK (method IN ({sql_in_list(METHODS)})),
        model TEXT NOT NULL DEFAULT '',
        level TEXT NOT NULL CHECK (level IN ({sql_in_list(LEVELS)})),
        calibration TEXT NOT NULL CHECK (calibration IN ({sql_in_list(CALIBRATIONS)})),
        probs JSON,
        raw_probs JSON,
        confidence DOUBLE,
        clm_confidence DOUBLE,
        s_c DOUBLE,
        p_fit DOUBLE,
        margin DOUBLE,
        anchor_index INTEGER,
        answer_index INTEGER NOT NULL,
        answer TEXT NOT NULL,
        options JSON NOT NULL,
        rank INTEGER,
        artifact_version TEXT,
        feature_spec TEXT,
        encoder_fp TEXT NOT NULL,
        clm_model_fp TEXT NOT NULL,
        schema_sha256 TEXT NOT NULL,
        latency_ms DOUBLE,
        threshold_auto DOUBLE,
        threshold_propose DOUBLE,
        outcome TEXT NOT NULL CHECK (outcome IN ({sql_in_list(OUTCOMES)})),
        reason TEXT,
        ts TIMESTAMPTZ NOT NULL,
        CHECK ((probs IS NULL) = (calibration = 'none')),
        CHECK (method NOT IN ({_METHODS_WITHOUT_PROBS_SQL}) OR (probs IS NULL AND level = 'none')),
        CHECK (level <> 'zero_shot' OR calibration = 'uncalibrated'),
        CHECK (level NOT IN ({_CALIBRATED_LEVELS_SQL}) OR calibration IN ({_CALIBRATED_CALIBRATIONS_SQL})),
        CHECK (method <> 'ols_rank' OR (outcome IN ('proposed', 'abstain') AND rank IS NOT NULL)),
        CHECK (shape <> 'choice' OR anchor_index IS NULL))""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.decision_options (
        decision_id TEXT NOT NULL,
        option_index INTEGER NOT NULL,
        option_key TEXT NOT NULL,
        option_text TEXT NOT NULL,
        rank INTEGER,
        s_c DOUBLE,
        p_fit DOUBLE,
        prob DOUBLE,
        raw_prob DOUBLE,
        masked BOOLEAN NOT NULL DEFAULT FALSE,
        action_sha256 TEXT,
        PRIMARY KEY (decision_id, option_index),
        UNIQUE (decision_id, option_key))""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.decision_groups (
        group_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        task_key TEXT NOT NULL,
        question_key TEXT,
        scope TEXT NOT NULL CHECK (scope IN ({sql_in_list(SCOPES)})),
        column_name TEXT,
        site_code TEXT,
        aspect TEXT,
        ontology_id TEXT,
        search_json JSON NOT NULL,
        n_candidates INTEGER NOT NULL,
        winner_decision_id TEXT,
        top_p_fit DOUBLE,
        group_margin DOUBLE,
        level TEXT CHECK (level IS NULL OR level IN ({sql_in_list(LEVELS)})),
        method TEXT CHECK (method IS NULL OR method IN ({sql_in_list(METHODS)})),
        outcome TEXT NOT NULL CHECK (outcome IN ({sql_in_list(OUTCOMES)})),
        anchor_won BOOLEAN NOT NULL DEFAULT FALSE,
        escalated_from TEXT,
        ts TIMESTAMPTZ NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.avu_links (
        link_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        group_id TEXT,
        decision_id TEXT,
        project_id TEXT,
        snapshot_id BIGINT,
        spool_batch_id TEXT,
        irods_path TEXT,
        target_type TEXT NOT NULL DEFAULT 'data_object',
        attribute TEXT NOT NULL,
        value TEXT NOT NULL,
        unit TEXT NOT NULL DEFAULT '',
        op TEXT NOT NULL DEFAULT 'add' CHECK (op IN ({sql_in_list(LINK_OPS)})),
        term_curie TEXT,
        term_iri TEXT,
        term_label TEXT,
        ontology_id TEXT,
        aspect TEXT,
        column_name TEXT NOT NULL DEFAULT '',
        site_code TEXT NOT NULL DEFAULT '',
        value_kind TEXT,
        source TEXT,
        write_status TEXT NOT NULL CHECK (write_status IN ({sql_in_list(WRITE_STATUSES)})),
        accepted_by TEXT CHECK (accepted_by IS NULL OR accepted_by IN ({sql_in_list(ACCEPTED_BY)})),
        duplicate_of TEXT,
        written_at TIMESTAMPTZ,
        UNIQUE (run_id, attribute, value, unit, column_name, site_code),
        CHECK (write_status <> 'accepted' OR accepted_by IS NOT NULL))""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.human_overrides (
        override_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        group_id TEXT,
        decision_id TEXT,
        link_id TEXT,
        actor TEXT NOT NULL,
        via TEXT NOT NULL CHECK (via IN ({sql_in_list(VIAS)})),
        action TEXT NOT NULL CHECK (action IN ({sql_in_list(OVERRIDE_ACTIONS)})),
        chosen_decision_id TEXT,
        chosen_option_key TEXT,
        elicitation_key TEXT,
        label_source TEXT CHECK (label_source IS NULL OR label_source IN ({sql_in_list(LABEL_SOURCES)})),
        labels_written INTEGER NOT NULL DEFAULT 0,
        offered JSON NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        CHECK (via <> 'tool' OR label_source IS NULL OR label_source = 'agent_pick'))""",
    # The labels table exactly as M0 defined it (D1, D30): the same objects, not a copy.
    *LABELS_DDL[1:],
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.audits (
        audit_id TEXT PRIMARY KEY,
        task_key TEXT NOT NULL,
        artifact_version TEXT NOT NULL,
        n INTEGER NOT NULL CHECK (n >= 0),
        n_cards INTEGER NOT NULL CHECK (n_cards >= 0),
        cards JSON NOT NULL,
        reviewer TEXT NOT NULL,
        n_errors INTEGER NOT NULL CHECK (n_errors >= 0),
        cp95_upper DOUBLE NOT NULL CHECK (cp95_upper >= 0 AND cp95_upper <= 1),
        risk DOUBLE NOT NULL CHECK (risk >= 0 AND risk <= 1),
        passed BOOLEAN NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        CHECK (n_errors <= n))""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.clm_calls (
        call_id TEXT PRIMARY KEY,
        run_id TEXT,
        endpoint TEXT NOT NULL,
        model TEXT NOT NULL,
        n_questions INTEGER NOT NULL,
        n_candidates INTEGER NOT NULL,
        input_tokens INTEGER,
        latency_ms DOUBLE NOT NULL,
        cache_hit BOOLEAN,
        status TEXT NOT NULL CHECK (status IN ({sql_in_list(CALL_STATUSES)})),
        ts TIMESTAMPTZ NOT NULL)""",
)

# Every table of the schema, in DDL order.
TABLES: Final[tuple[str, ...]] = (
    "schema_versions",
    "runs",
    "decisions",
    "decision_options",
    "decision_groups",
    "avu_links",
    "human_overrides",
    "labels",
    "audits",
    "clm_calls",
)

# The tables that belong to one run (deleted together by ``delete_run``; exported together by
# ``export_run``), in insert order: parents before children.
RUN_TABLES: Final[tuple[str, ...]] = (
    "runs",
    "decision_groups",
    "decisions",
    "decision_options",
    "avu_links",
    "human_overrides",
    "clm_calls",
)

# Deterministic read order per run table (the export's ORDER BY; ids break every tie).
RUN_TABLE_ORDER: Final[dict[str, str]] = {
    "runs": "run_id",
    "decision_groups": "ts, group_id",
    "decisions": "seq, decision_id",
    "decision_options": "decision_id, option_index",
    "avu_links": "attribute, value, unit, column_name, site_code, link_id",
    "human_overrides": "ts, override_id",
    "clm_calls": "ts, call_id",
}

# Uniqueness rules a ``NOT NULL DEFAULT ''`` sentinel could not express, checked in code instead
# (D11). Empty: every UNIQUE key of this schema is over NOT NULL columns. The DDL-parity test
# asserts this tuple names exactly the constraints the DDL lacks.
CODE_CHECKED_UNIQUENESS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = ()


# -- row <-> SQL ----------------------------------------------------------------------------------------


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _cell(value: Any) -> Any:
    """A Python value as DuckDB takes it: ids as text, dicts and lists as JSON text."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict | list):
        return _json(value)
    return value


def row_values(row: BaseModel) -> tuple[list[str], list[Any]]:
    """``(column names, DuckDB-ready values)`` of a row model, in field order."""
    data = row.model_dump()
    return list(data), [_cell(v) for v in data.values()]


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    """A fetched row with its JSON columns parsed (DuckDB returns JSON as text)."""
    for key in JSON_COLUMNS:
        if key in row and isinstance(row[key], str):
            row[key] = json.loads(row[key])
    return row


def insert_sql(table: str, columns: Sequence[str]) -> str:
    """``INSERT INTO mesa_clm.<table> (...) VALUES (?, ...)``; ``?`` is the shared placeholder,
    the Postgres store rewrites it."""
    return (
        f"INSERT INTO {SCHEMA}.{table} ({', '.join(columns)}) "  # noqa: S608
        f"VALUES ({', '.join('?' for _ in columns)})"
    )


def finish_run_sql(status: str, stats: dict[str, Any]) -> tuple[str, list[Any]]:
    """The ``UPDATE runs`` for ``finish_run``; ``ValueError`` for a column it may not set or a
    status outside the vocabulary. The run id is the last parameter."""
    if status not in RUN_STATUSES:
        raise ValueError(f"finish_run: {status!r} is not a run status {RUN_STATUSES}")
    sets = ["status = ?"]
    values: list[Any] = [status]
    for key, value in stats.items():
        if key not in RUN_FINISH_COLUMNS:
            raise ValueError(f"finish_run: {key} is not an updatable run column")
        if key == "history_backend" and value is not None and value not in HISTORY_BACKENDS:
            raise ValueError(f"finish_run: history_backend {value!r} not in {HISTORY_BACKENDS}")
        sets.append(f"{key} = ?")
        values.append(_cell(value))
    return f"UPDATE {SCHEMA}.runs SET {', '.join(sets)} WHERE run_id = ?", values  # noqa: S608


def update_group_sql(summary: dict[str, Any]) -> tuple[str, list[Any]] | None:
    """The ``UPDATE decision_groups`` for ``update_group`` (``None`` when nothing to set);
    ``ValueError`` for a column outside :data:`GROUP_SUMMARY_COLUMNS`. Group id last."""
    sets, values = [], []
    for key, value in summary.items():
        if key not in GROUP_SUMMARY_COLUMNS:
            raise ValueError(f"update_group: {key} is not a summary column")
        sets.append(f"{key} = ?")
        values.append(_cell(value))
    if not sets:
        return None
    return f"UPDATE {SCHEMA}.decision_groups SET {', '.join(sets)} WHERE group_id = ?", values  # noqa: S608


def set_link_status_sql(
    status: str,
    *,
    written_at: datetime | None,
    irods_path: str | None,
    snapshot_id: int | None,
    spool_batch_id: str | None,
    accepted_by: str | None,
) -> tuple[str, list[Any]]:
    """The ``UPDATE avu_links`` for ``set_link_status``: only the given columns change (a
    ``reverted`` link keeps its ``written_at``). Link id last."""
    if status not in WRITE_STATUSES:
        raise ValueError(f"set_link_status: {status!r} is not a write status {WRITE_STATUSES}")
    if accepted_by is not None and accepted_by not in ACCEPTED_BY:
        raise ValueError(f"set_link_status: accepted_by {accepted_by!r} not in {ACCEPTED_BY}")
    sets = ["write_status = ?"]
    values: list[Any] = [status]
    for key, value in (
        ("written_at", written_at),
        ("irods_path", irods_path),
        ("snapshot_id", snapshot_id),
        ("spool_batch_id", spool_batch_id),
        ("accepted_by", accepted_by),
    ):
        if value is not None:
            sets.append(f"{key} = ?")
            values.append(value)
    return f"UPDATE {SCHEMA}.avu_links SET {', '.join(sets)} WHERE link_id = ?", values  # noqa: S608


LINK_SNAPSHOT_SQL: Final = (
    f"UPDATE {SCHEMA}.avu_links SET snapshot_id = ?, project_id = ? "  # noqa: S608
    "WHERE run_id = ? AND irods_path = ? AND write_status IN ('written', 'spooled') "
    "AND snapshot_id IS NULL"
)


def runs_sql(
    *,
    owner: str | None,
    status: str | None,
    card_name: str | None,
    irods_path: str | None,
    since: datetime | None,
    limit: int | None,
) -> tuple[str, list[Any]]:
    """``SELECT * FROM runs`` with the optional filters, newest first."""
    where: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("owner", owner),
        ("status", status),
        ("card_name", card_name),
        ("irods_path", irods_path),
    ):
        if value is not None:
            where.append(f"{column} = ?")
            params.append(value)
    if since is not None:
        where.append("started_at >= ?")
        params.append(since)
    sql = f"SELECT * FROM {SCHEMA}.runs"  # noqa: S608
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_at DESC, run_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return sql, params


DECISIONS_FOR_PATH_SQL: Final = (
    "SELECT l.link_id, l.run_id, l.attribute, l.value, l.unit, l.column_name, l.site_code, "  # noqa: S608
    "l.write_status, l.accepted_by, l.snapshot_id, l.spool_batch_id, l.source, l.term_curie, "
    "d.decision_id, d.task_id, d.task_key, d.question_key, d.method, d.level, d.calibration, "
    "d.confidence, d.p_fit, d.s_c, d.model, d.artifact_version, d.outcome "
    f"FROM {SCHEMA}.avu_links l LEFT JOIN {SCHEMA}.decisions d USING (decision_id) "
    "WHERE l.irods_path = ? ORDER BY l.attribute, l.value, l.unit, l.column_name, l.site_code, "
    "l.link_id LIMIT ?"
)

OPTIONS_FOR_RUN_SQL: Final = (
    f"SELECT o.* FROM {SCHEMA}.decision_options o JOIN {SCHEMA}.decisions d USING (decision_id) "  # noqa: S608
    "WHERE d.run_id = ? ORDER BY o.decision_id, o.option_index"
)


def run_table_sql(table: str) -> str:
    """``SELECT * FROM <run table> WHERE run_id = ? ORDER BY <deterministic order>``."""
    if table not in RUN_TABLE_ORDER:
        raise KeyError(table)
    if table == "decision_options":
        return OPTIONS_FOR_RUN_SQL
    return f"SELECT * FROM {SCHEMA}.{table} WHERE run_id = ? ORDER BY {RUN_TABLE_ORDER[table]}"  # noqa: S608


def delete_run_sql() -> list[tuple[str, str]]:
    """``(table, DELETE statement)`` per run table, children first; the run id is the parameter.
    ``labels`` and ``audits`` are never deleted with a run: labels are learning data (D30) and
    audits span runs."""
    out = [
        (
            "decision_options",
            f"DELETE FROM {SCHEMA}.decision_options WHERE decision_id IN "  # noqa: S608
            f"(SELECT decision_id FROM {SCHEMA}.decisions WHERE run_id = ?)",
        )
    ]
    for table in reversed(RUN_TABLES):
        if table != "decision_options":
            out.append((table, f"DELETE FROM {SCHEMA}.{table} WHERE run_id = ?"))  # noqa: S608
    return out


# -- the run buffer -------------------------------------------------------------------------------------


class RunBuffer:
    """One run's rows collected in memory and committed in one store transaction (D11).

    The pipeline records into the buffer with the store's write API (``insert_decisions``,
    ``insert_group``, ``update_group``, ``insert_links``, ``insert_override``,
    ``insert_clm_call(s)``, ``finish_run``) while no lock is held; ``store.commit_run(buffer)``
    then writes everything under one flock in one transaction, so no other process ever sees a
    half-written run. ``update_group`` and ``finish_run`` rewrite the buffered rows (re-validated),
    so the commit is pure inserts. Every row must carry the buffer's ``run_id``.
    """

    def __init__(self, run: RunRow) -> None:
        self.run = run
        self.decisions: list[DecisionRow] = []
        self.options: list[DecisionOptionRow] = []
        self.groups: dict[UUID, DecisionGroupRow] = {}
        self.links: list[AvuLinkRow] = []
        self.overrides: list[HumanOverrideRow] = []
        self.clm_calls: list[ClmCallRow] = []
        self.committed = False

    @property
    def run_id(self) -> UUID:
        return self.run.run_id

    def _own(self, rows: Iterable[BaseModel]) -> None:
        for row in rows:
            if getattr(row, "run_id", self.run_id) != self.run_id:
                raise ValueError(
                    f"row belongs to run {getattr(row, 'run_id', None)}, buffer is {self.run_id}"
                )

    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int:
        self._own(rows)
        self.decisions.extend(rows)
        self.options.extend(options)
        return len(rows)

    def insert_group(self, row: DecisionGroupRow) -> UUID:
        self._own([row])
        self.groups[row.group_id] = row
        return row.group_id

    def update_group(self, group_id: UUID, **summary: Any) -> None:
        unknown = set(summary) - GROUP_SUMMARY_COLUMNS
        if unknown:
            raise ValueError(f"update_group: {sorted(unknown)} are not summary columns")
        current = self.groups[group_id]
        self.groups[group_id] = DecisionGroupRow.model_validate({**current.model_dump(), **summary})

    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int:
        self._own(rows)
        self.links.extend(rows)
        return len(rows)

    def insert_override(self, row: HumanOverrideRow) -> UUID:
        self._own([row])
        self.overrides.append(row)
        return row.override_id

    def insert_clm_call(self, row: ClmCallRow) -> UUID:
        self._own([row])
        self.clm_calls.append(row)
        return row.call_id

    def insert_clm_calls(self, rows: Sequence[ClmCallRow]) -> int:
        self._own(rows)
        self.clm_calls.extend(rows)
        return len(rows)

    def finish_run(self, status: str, **stats: Any) -> None:
        """Set the final status and stats on the buffered run row. ``n_decisions`` and
        ``n_clm_calls`` default to what the buffer holds."""
        unknown = set(stats) - RUN_FINISH_COLUMNS
        if unknown:
            raise ValueError(f"finish_run: {sorted(unknown)} are not updatable run columns")
        stats.setdefault("n_decisions", len(self.decisions))
        stats.setdefault("n_clm_calls", len(self.clm_calls))
        self.run = RunRow.model_validate({**self.run.model_dump(), "status": status, **stats})

    def rows(self) -> dict[str, list[BaseModel]]:
        """Table -> rows, in insert order (parents first)."""
        return {
            "runs": [self.run],
            "decision_groups": list(self.groups.values()),
            "decisions": list(self.decisions),
            "decision_options": list(self.options),
            "avu_links": list(self.links),
            "human_overrides": list(self.overrides),
            "clm_calls": list(self.clm_calls),
        }


# -- the protocol ---------------------------------------------------------------------------------------


class ProvenanceStore(Protocol):
    """The store contract the pipeline, the tools, the learners and the export share."""

    def ensure_schema(self) -> int: ...
    def columns(self, table: str) -> list[tuple[str, bool]]: ...
    def commit_run(self, buffer: RunBuffer) -> UUID: ...
    def begin_run(self, run: RunRow) -> UUID: ...
    def finish_run(self, run_id: UUID, status: str, **stats: Any) -> None: ...
    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int: ...
    def insert_group(self, row: DecisionGroupRow) -> UUID: ...
    def update_group(self, group_id: UUID, **summary: Any) -> None: ...
    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int: ...
    def set_link_status(
        self,
        link_ids: Iterable[UUID],
        status: str,
        *,
        written_at: datetime | None = None,
        irods_path: str | None = None,
        snapshot_id: int | None = None,
        spool_batch_id: str | None = None,
        accepted_by: str | None = None,
    ) -> int: ...
    def link_snapshot(
        self, run_id: UUID, irods_path: str, project_id: str | None, snapshot_id: int
    ) -> int: ...
    def insert_override(self, row: HumanOverrideRow) -> UUID: ...
    def insert_labels(self, rows: Sequence[LabelRow]) -> int: ...
    def insert_audit(self, row: AuditRow) -> UUID: ...
    def insert_clm_call(self, row: ClmCallRow) -> UUID: ...
    def insert_clm_calls(self, rows: Sequence[ClmCallRow]) -> int: ...
    def delete_run(self, run_id: UUID) -> dict[str, int]: ...
    def run(self, run_id: UUID) -> dict[str, Any] | None: ...
    def runs(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
        card_name: str | None = None,
        irods_path: str | None = None,
        since: datetime | None = None,
        limit: int | None = 100,
    ) -> list[dict[str, Any]]: ...
    def decisions(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def options(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def groups(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def group(self, group_id: UUID) -> dict[str, Any] | None: ...
    def links(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def overrides(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def clm_calls(self, run_id: UUID) -> list[dict[str, Any]]: ...
    def audits(self, task_key: str | None = None) -> list[dict[str, Any]]: ...
    def decisions_for_path(self, irods_path: str, limit: int = 100) -> list[dict[str, Any]]: ...
    def labels_for(
        self,
        task_id: str,
        *,
        min_weight: float = 0.0,
        exclude_cards: Sequence[str] = (),
        sources: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


# -- DuckDB -----------------------------------------------------------------------------------------------


class DuckDBStore:
    """The per-host sidecar file, opened per operation under the D11 flock.

    ``path`` is the DuckDB file (``~/.mesa/clm/provenance.duckdb`` by default); ``lock_path``
    defaults to :func:`lock_path_for` (``path``), the lock :class:`LabelStore` takes over the
    same file. The constructor touches nothing; the first write creates the directory, the lock
    file, the database and the schema. Labels go through :class:`LabelStore` on the same file and
    lock (it owns that table's DDL and its idempotent insert).
    """

    def __init__(self, path: str | Path, *, lock_path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = (
            Path(lock_path).expanduser() if lock_path is not None else lock_path_for(self.path)
        )
        self._labels = LabelStore(self.path, lock_path=self.lock_path)

    # -- connection and lock -------------------------------------------------------------------------
    @contextmanager
    def connect(self, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        """One operation: take the flock (shared for a read-only open, exclusive otherwise),
        open DuckDB, yield, close, release. Session time zone UTC so ``TIMESTAMPTZ`` values come
        back as UTC datetimes whatever the host's zone. ``read_only`` refuses a missing file
        (callers check ``path.exists()`` first and return nothing)."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH if read_only else fcntl.LOCK_EX)
            try:
                con = duckdb.connect(str(self.path), read_only=read_only)
                try:
                    con.execute("SET TimeZone = 'UTC'")
                    yield con
                finally:
                    con.close()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _bootstrap(con: duckdb.DuckDBPyConnection) -> None:
        for stmt in DUCKDB_DDL:
            con.execute(stmt)
        con.execute(
            f"INSERT INTO {SCHEMA}.schema_versions (version) SELECT ? WHERE NOT EXISTS "  # noqa: S608
            f"(SELECT 1 FROM {SCHEMA}.schema_versions WHERE version = ?)",
            [SCHEMA_VERSION, SCHEMA_VERSION],
        )

    @contextmanager
    def _write(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """A write operation: bootstrap the schema (idempotent), then one transaction."""
        with self.connect() as con:
            self._bootstrap(con)
            con.execute("BEGIN")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")

    def ensure_schema(self) -> int:
        """Create the schema and every table if missing (also creates the file); the version."""
        with self.connect() as con:
            self._bootstrap(con)
        return SCHEMA_VERSION

    def columns(self, table: str) -> list[tuple[str, bool]]:
        """``(column_name, nullable)`` in ordinal order (empty for a missing file or table)."""
        rows = self._select(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [SCHEMA, table],
        )
        return [(str(r["column_name"]), str(r["is_nullable"]).upper() == "YES") for r in rows]

    def constraints(self, table: str) -> list[tuple[str, str]]:
        """``(constraint_type, constraint_text)`` as DuckDB reports them, for the parity test."""
        rows = self._select(
            "SELECT constraint_type, constraint_text FROM duckdb_constraints() "
            "WHERE schema_name = ? AND table_name = ? ORDER BY constraint_index",
            [SCHEMA, table],
        )
        return [(str(r["constraint_type"]), str(r["constraint_text"])) for r in rows]

    # -- writes -----------------------------------------------------------------------------------------
    @staticmethod
    def _insert(con: duckdb.DuckDBPyConnection, table: str, rows: Sequence[BaseModel]) -> int:
        if not rows:
            return 0
        cols, _ = row_values(rows[0])
        con.executemany(insert_sql(table, cols), [row_values(r)[1] for r in rows])
        return len(rows)

    def commit_run(self, buffer: RunBuffer) -> UUID:
        """Write every buffered row in one transaction (D11); the buffer is marked committed."""
        if buffer.committed:
            raise ValueError(f"run {buffer.run_id} was already committed")
        with self._write() as con:
            for table, rows in buffer.rows().items():
                self._insert(con, table, rows)
        buffer.committed = True
        return buffer.run_id

    def begin_run(self, run: RunRow) -> UUID:
        with self._write() as con:
            self._insert(con, "runs", [run])
        return run.run_id

    def finish_run(self, run_id: UUID, status: str, **stats: Any) -> None:
        sql, values = finish_run_sql(status, stats)
        with self._write() as con:
            con.execute(sql, [*values, str(run_id)])

    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int:
        with self._write() as con:
            n = self._insert(con, "decisions", rows)
            self._insert(con, "decision_options", options)
        return n

    def insert_group(self, row: DecisionGroupRow) -> UUID:
        with self._write() as con:
            self._insert(con, "decision_groups", [row])
        return row.group_id

    def update_group(self, group_id: UUID, **summary: Any) -> None:
        got = update_group_sql(summary)
        if got is None:
            return
        sql, values = got
        with self._write() as con:
            con.execute(sql, [*values, str(group_id)])

    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int:
        with self._write() as con:
            return self._insert(con, "avu_links", rows)

    def set_link_status(
        self,
        link_ids: Iterable[UUID],
        status: str,
        *,
        written_at: datetime | None = None,
        irods_path: str | None = None,
        snapshot_id: int | None = None,
        spool_batch_id: str | None = None,
        accepted_by: str | None = None,
    ) -> int:
        ids = [str(i) for i in link_ids]
        if not ids:
            return 0
        sql, values = set_link_status_sql(
            status,
            written_at=written_at,
            irods_path=irods_path,
            snapshot_id=snapshot_id,
            spool_batch_id=spool_batch_id,
            accepted_by=accepted_by,
        )
        with self._write() as con:
            con.executemany(sql, [[*values, i] for i in ids])
        return len(ids)

    def link_snapshot(
        self, run_id: UUID, irods_path: str, project_id: str | None, snapshot_id: int
    ) -> int:
        """Stamp ``snapshot_id`` (and ``project_id``) on the run's written or spooled links at
        ``irods_path`` that have none yet; returns how many."""
        with self._write() as con:
            got = con.execute(
                LINK_SNAPSHOT_SQL, [snapshot_id, project_id, str(run_id), irods_path]
            ).fetchone()
        return int(got[0]) if got else 0

    def insert_override(self, row: HumanOverrideRow) -> UUID:
        with self._write() as con:
            self._insert(con, "human_overrides", [row])
        return row.override_id

    def insert_labels(self, rows: Sequence[LabelRow]) -> int:
        """Delegates to :class:`LabelStore` (same file, same lock): ``INSERT OR IGNORE`` on the
        D1 identity, one transaction; returns how many were new."""
        return self._labels.insert_labels(rows)

    def insert_audit(self, row: AuditRow) -> UUID:
        with self._write() as con:
            self._insert(con, "audits", [row])
        return row.audit_id

    def insert_clm_call(self, row: ClmCallRow) -> UUID:
        with self._write() as con:
            self._insert(con, "clm_calls", [row])
        return row.call_id

    def insert_clm_calls(self, rows: Sequence[ClmCallRow]) -> int:
        with self._write() as con:
            return self._insert(con, "clm_calls", rows)

    def delete_run(self, run_id: UUID) -> dict[str, int]:
        """Delete a run and its decisions, options, groups, links, overrides and calls in one
        transaction (labels and audits stay); returns rows deleted per table."""
        deleted: dict[str, int] = {}
        with self._write() as con:
            for table, sql in delete_run_sql():
                got = con.execute(sql, [str(run_id)]).fetchone()
                deleted[table] = int(got[0]) if got else 0
        return deleted

    # -- reads ------------------------------------------------------------------------------------------
    def _select(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Rows as dicts from a read-only connection; nothing for a missing file or table."""
        if not self.path.exists():
            return []
        with self.connect(read_only=True) as con:
            try:
                cur = con.execute(sql, list(params))
            except duckdb.CatalogException:
                return []
            names = [d[0] for d in cur.description or []]
            values = cur.fetchall()
        return [parse_row(dict(zip(names, v, strict=True))) for v in values]

    def run(self, run_id: UUID) -> dict[str, Any] | None:
        rows = self._select(f"SELECT * FROM {SCHEMA}.runs WHERE run_id = ?", [str(run_id)])  # noqa: S608
        return rows[0] if rows else None

    def runs(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
        card_name: str | None = None,
        irods_path: str | None = None,
        since: datetime | None = None,
        limit: int | None = 100,
    ) -> list[dict[str, Any]]:
        """Runs newest first, filtered by the given columns; ``limit=None`` returns all."""
        sql, params = runs_sql(
            owner=owner,
            status=status,
            card_name=card_name,
            irods_path=irods_path,
            since=since,
            limit=limit,
        )
        return self._select(sql, params)

    def _run_table(self, table: str, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(run_table_sql(table), [str(run_id)])

    def decisions(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("decisions", run_id)

    def options(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("decision_options", run_id)

    def groups(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("decision_groups", run_id)

    def group(self, group_id: UUID) -> dict[str, Any] | None:
        rows = self._select(
            f"SELECT * FROM {SCHEMA}.decision_groups WHERE group_id = ?",  # noqa: S608
            [str(group_id)],
        )
        return rows[0] if rows else None

    def links(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("avu_links", run_id)

    def overrides(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("human_overrides", run_id)

    def clm_calls(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("clm_calls", run_id)

    def audits(self, task_key: str | None = None) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {SCHEMA}.audits"  # noqa: S608
        params: list[Any] = []
        if task_key is not None:
            sql += " WHERE task_key = ?"
            params.append(task_key)
        return self._select(sql + " ORDER BY created_at, audit_id", params)

    def decisions_for_path(self, irods_path: str, limit: int = 100) -> list[dict[str, Any]]:
        """Links at an iRODS path joined to the decision behind each (``explain`` by path)."""
        return self._select(DECISIONS_FOR_PATH_SQL, [irods_path, int(limit)])

    def labels_for(
        self,
        task_id: str,
        *,
        min_weight: float = 0.0,
        exclude_cards: Sequence[str] = (),
        sources: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """:meth:`LabelStore.labels_for` on the same file."""
        return self._labels.labels_for(
            task_id, min_weight=min_weight, exclude_cards=exclude_cards, sources=sources
        )

    def close(self) -> None:
        """Nothing to release: no connection outlives an operation."""


# -- dispatch ---------------------------------------------------------------------------------------------

_DUCKDB_RE = re.compile(r"^duckdb:///(.+)$")


def duckdb_file(dsn: str) -> Path | None:
    """The file behind ``duckdb:///<path>`` (or a bare ``*.duckdb`` path), home expanded;
    ``None`` for any other dialect."""
    if m := _DUCKDB_RE.match(dsn):
        return Path(m.group(1)).expanduser()
    if dsn.endswith(".duckdb"):
        return Path(dsn).expanduser()
    return None


def is_postgres_dsn(dsn: str) -> bool:
    return dsn.startswith(("postgresql://", "postgres://"))


def open_store(dsn: str) -> ProvenanceStore:
    """Dispatch on the DSN: ``duckdb:///<path>`` (or ``*.duckdb``) -> :class:`DuckDBStore`;
    ``postgresql://`` -> :class:`~mesa_clm.provenance.store_postgres.PostgresStore` (``pg``
    extra). Both come back with the schema in force (``ensure_schema``); the DuckDB store then
    holds no connection."""
    if not dsn or not dsn.strip():
        raise ValueError("provenance DSN is empty")
    path = duckdb_file(dsn)
    if path is not None:
        store = DuckDBStore(path)
        store.ensure_schema()
        return store
    if is_postgres_dsn(dsn):
        from mesa_clm.provenance.store_postgres import PostgresStore  # lazy: psycopg optional

        pg = PostgresStore(dsn)
        pg.ensure_schema()
        return cast(ProvenanceStore, pg)
    raise ValueError("unsupported provenance DSN (duckdb:///<path> or postgresql://...)")
