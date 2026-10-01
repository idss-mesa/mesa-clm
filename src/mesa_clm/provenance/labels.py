"""The ``mesa_clm.labels`` table: rows, DDL and a per-operation DuckDB store (DESIGN D1, D11,
D19, D21, D30).

A label says which option of a task is right for a target, with a weight that says how much to
trust it. Identity is ``(task_key, target_sha256, option_key, label_source)`` (D1;
:mod:`mesa_clm.identity`), so the same finding recorded on states that differ only in
``n_candidates`` or in a card header edit is one row, and an anchor pick needs no candidate
state. ``state_sha256`` and ``state_json`` stay on the row for parity with mesa-anyjev and so a
fit can render the exact state the label was recorded on.

Every optional text key is a ``NOT NULL DEFAULT ''`` sentinel: DuckDB 1.5.5 rejects ``UNIQUE
NULLS NOT DISTINCT`` (verified), and a NULL in a UNIQUE key would let duplicates through (D11).
``LABELS_DDL`` is one module constant so the M1 sidecar migration
(``migrations/0001_mesa_clm.sql``) can include it verbatim and the DDL-parity test can compare
the two dialects.

:class:`LabelStore` opens the DuckDB file *per operation* under ``fcntl.flock`` of the D11 lock
file, :func:`lock_path_for` = ``<dir of the DuckDB file>/locks/provenance.lock``, so the default
store at ``~/.mesa/clm/provenance.duckdb`` locks ``~/.mesa/clm/locks/provenance.lock``: no
process holds the single-writer file between calls, so the CLI, the bench and the MCP plugin can
share one host store. The M1 sidecar store (``provenance/store.py``) must derive its lock with
the same function; two different lock files over one DuckDB file would not serialise the two
stores (DuckDB is single-writer, the loser gets "Could not set lock"). The ``locks/`` directory,
the lock file, the database and its ``.wal`` are owner-only (``0700``/``0600``) whatever the
process umask (:func:`sidecar_lock`, :func:`tighten_duckdb`, :mod:`mesa_clm.perms`).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID, uuid4

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mesa_clm.perms import open_private, private_dir, tighten_file
from mesa_clm.tasks import TASKS

SCHEMA: Final = "mesa_clm"
TABLE: Final = f"{SCHEMA}.labels"

LabelSource = Literal[
    "curator",
    "curator_implicit",
    "agent_pick",
    "consensus_all",
    "consensus_majority",
    "consensus_negative",
    "teacher",
    "teacher_implicit",
    "gold",
]
LABEL_SOURCES: Final[tuple[str, ...]] = (
    "curator",
    "curator_implicit",
    "agent_pick",
    "consensus_all",
    "consensus_majority",
    "consensus_negative",
    "teacher",
    "teacher_implicit",
    "gold",
)

# One statement per object, DuckDB dialect (JSON not JSONB, TEXT ids, no foreign keys). The
# Postgres migration (M1) carries the same columns in the same order; the parity test compares.
LABELS_DDL: Final[tuple[str, ...]] = (
    f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
    f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        label_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        task_key TEXT NOT NULL,
        target_sha256 TEXT NOT NULL,
        option_key TEXT NOT NULL DEFAULT '',
        label_source TEXT NOT NULL CHECK (label_source IN ({", ".join(f"'{s}'" for s in LABEL_SOURCES)})),
        label TEXT NOT NULL,
        label_index INTEGER NOT NULL CHECK (label_index >= 0),
        weight DOUBLE NOT NULL CHECK (weight >= 0 AND weight <= 1),
        state_sha256 TEXT NOT NULL,
        state_json JSON NOT NULL,
        card TEXT NOT NULL DEFAULT '',
        product_code TEXT NOT NULL DEFAULT '',
        leak_group TEXT NOT NULL DEFAULT '',
        fold_eligible BOOLEAN NOT NULL DEFAULT TRUE,
        bench_card BOOLEAN NOT NULL DEFAULT FALSE,
        origin TEXT NOT NULL DEFAULT '',
        actor TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL,
        UNIQUE (task_key, target_sha256, option_key, label_source))""",
)

# The identity key, in DDL order; ``snapshot`` sorts by it so the Parquet bytes are stable.
IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_key",
    "target_sha256",
    "option_key",
    "label_source",
)

_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _now() -> datetime:
    return datetime.now(tz=UTC)


class LabelRow(BaseModel):
    """One label, as stored (column order = DDL order).

    ``label`` is the option's label string (``"Yes"``/``"No"`` for the rank_fit and annotate
    tasks, the option text for aspect and value_kind) and ``label_index`` its position in the
    task's options. ``product_code`` and ``leak_group`` drive leakage-aware folds (D19);
    ``fold_eligible`` is false for teacher and agent labels; ``bench_card`` marks curator labels
    recorded on a bench card (D30). ``origin`` says where the row came from (a source reference
    such as ``neon-avu-eval/results/validated.json@<sha12>``, ``override:<group_id>`` or
    ``smoke``) and ``actor`` who produced it.
    """

    model_config = ConfigDict(extra="forbid")

    label_id: UUID = Field(default_factory=uuid4)
    task_id: str
    task_key: str
    target_sha256: str
    option_key: str = ""
    label_source: LabelSource
    label: str
    label_index: int = Field(ge=0)
    weight: float = Field(ge=0.0, le=1.0)
    state_sha256: str
    state_json: dict[str, Any]
    card: str = ""
    product_code: str = ""
    leak_group: str = ""
    fold_eligible: bool = True
    bench_card: bool = False
    origin: str = ""
    actor: str = ""
    created_at: datetime = Field(default_factory=_now)

    @field_validator("task_id", "label")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("task_key")
    @classmethod
    def _key16(cls, value: str) -> str:
        if not _HEX16.match(value):
            raise ValueError("task_key must be 16 lower-case hex characters")
        return value

    @field_validator("target_sha256", "state_sha256")
    @classmethod
    def _sha64(cls, value: str) -> str:
        if not _HEX64.match(value):
            raise ValueError("must be a 64-character lower-case hex sha256")
        return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _cell(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict | list):
        return _json(value)
    return value


def _row_values(row: LabelRow) -> tuple[list[str], list[Any]]:
    data = row.model_dump()
    return list(data), [_cell(v) for v in data.values()]


def lock_path_for(duckdb_file: str | Path) -> Path:
    """The D11 sidecar lock for a DuckDB file: ``<its directory>/locks/provenance.lock``
    (``~/.mesa/clm/locks/provenance.lock`` for the default store). Every store over the same
    file, this one and the M1 sidecar store, must lock this path."""
    return Path(duckdb_file).expanduser().parent / "locks" / "provenance.lock"


@contextmanager
def sidecar_lock(lock_path: Path, *, shared: bool = False, tighten: bool = True) -> Iterator[int]:
    """Hold the D11 flock on ``lock_path`` (shared or exclusive) for one operation.

    The ``locks/`` directory and the lock file are mesa-clm's own and owner-only whatever the
    umask (:mod:`mesa_clm.perms`): missing components are created ``0700`` and a missing lock
    file ``0600``; with ``tighten`` (every write) an existing directory and file this user owns
    are set to ``0700`` / ``0600``, which repairs a lock a looser umask created before. A read
    passes ``tighten=False`` and changes no existing mode (the doctor reports, never repairs)."""
    private_dir(lock_path.parent, tighten=tighten)
    fd = open_private(lock_path, tighten=tighten)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def tighten_duckdb(path: Path) -> None:
    """The database and its write-ahead log owner-only (``0600``) after a write: DuckDB creates
    both with the umask's mode (``mesa_clm.perms``)."""
    tighten_file(path)
    tighten_file(path.with_name(path.name + ".wal"))


class LabelStore:
    """The labels table in one DuckDB file, opened per operation under a file lock (D11).

    ``path`` is the DuckDB file; ``lock_path`` defaults to :func:`lock_path_for` (``path``),
    the D11 lock shared with the M1 sidecar store. Reads on a file that does not exist yet
    return nothing instead of creating it.
    """

    def __init__(self, path: str | Path, *, lock_path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = (
            Path(lock_path).expanduser() if lock_path is not None else lock_path_for(self.path)
        )

    # -- connection and lock ---------------------------------------------------------------------
    @contextmanager
    def connect(self, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        """One operation: take the exclusive flock, open DuckDB, yield, close, release.

        The lock is held for the whole operation (readers included: a DuckDB file has one writer
        and readers may not share it while a writer holds it). ``read_only`` opens the file
        read-only, which DuckDB refuses for a file that does not exist. A write creates missing
        directories ``0700`` and leaves the database and its ``.wal`` ``0600``
        (:func:`sidecar_lock`, :func:`tighten_duckdb`).
        """
        if not read_only:
            private_dir(self.path.parent, tighten=False)
        with sidecar_lock(self.lock_path, tighten=not read_only):
            con = duckdb.connect(str(self.path), read_only=read_only)
            try:
                if not read_only:
                    tighten_duckdb(self.path)
                yield con
            finally:
                con.close()
                if not read_only:
                    tighten_duckdb(self.path)

    def ensure_schema(self) -> None:
        """Create the schema and table if missing (idempotent; also creates the file)."""
        with self.connect() as con:
            for stmt in LABELS_DDL:
                con.execute(stmt)

    def columns(self) -> list[tuple[str, bool]]:
        """``(column_name, nullable)`` in ordinal order, for the DDL-parity test."""
        if not self.path.exists():
            return []
        with self.connect(read_only=True) as con:
            rows = con.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = 'labels' ORDER BY ordinal_position",
                [SCHEMA],
            ).fetchall()
        return [(str(name), str(nullable).upper() == "YES") for name, nullable in rows]

    # -- writes -----------------------------------------------------------------------------------
    def insert_labels(self, rows: Sequence[LabelRow]) -> int:
        """Insert rows, ignoring any whose identity key already exists (idempotent ingestion);
        returns how many were inserted. The first row wins on a within-batch collision."""
        if not rows:
            return 0
        cols, _ = _row_values(rows[0])
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT OR IGNORE INTO {TABLE} ({', '.join(cols)}) VALUES ({placeholders})"  # noqa: S608
        inserted = 0
        with self.connect() as con:
            for stmt in LABELS_DDL:
                con.execute(stmt)
            con.execute("BEGIN")
            try:
                for row in rows:
                    got = con.execute(sql, _row_values(row)[1]).fetchone()
                    inserted += int(got[0]) if got else 0
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        return inserted

    # -- reads ------------------------------------------------------------------------------------
    def _select(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.connect(read_only=True) as con:
            cur = con.execute(sql, list(params))
            names = [d[0] for d in cur.description or []]
            values = cur.fetchall()
        out = []
        for row_values in values:
            row = dict(zip(names, row_values, strict=True))
            if isinstance(row.get("state_json"), str):
                row["state_json"] = json.loads(row["state_json"])
            out.append(row)
        return out

    def labels_for(
        self,
        task_id: str,
        *,
        min_weight: float = 0.0,
        exclude_cards: Sequence[str] = (),
        sources: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Every row of the task's *current* key with ``weight >= min_weight``, minus the cards
        in ``exclude_cards`` and, when ``sources`` is given, minus other sources. Rows recorded
        under a rotated key never enter a fit (they need migration; mesa-anyjev D7).
        ``KeyError`` for an unknown task. Ordered by identity key, then ``created_at``."""
        sql = f"SELECT * FROM {TABLE} WHERE task_key = ? AND weight >= ?"  # noqa: S608
        params: list[Any] = [TASKS[task_id].key, float(min_weight)]
        if exclude_cards:
            sql += " AND card NOT IN (" + ", ".join("?" for _ in exclude_cards) + ")"
            params.extend(exclude_cards)
        if sources is not None:
            if not sources:
                return []
            sql += " AND label_source IN (" + ", ".join("?" for _ in sources) + ")"
            params.extend(sources)
        sql += " ORDER BY " + ", ".join(IDENTITY_COLUMNS) + ", created_at, label_id"
        return self._select(sql, params)

    def count(self) -> int:
        """Rows in the table (0 for a store that does not exist yet)."""
        rows = self._select(f"SELECT count(*) AS n FROM {TABLE}")  # noqa: S608
        return int(rows[0]["n"]) if rows else 0

    def counts(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """``{task_id, label_source, label_index, label, weight, n}`` per group, sorted; one task
        when ``task_id`` is given."""
        sql = (
            f"SELECT task_id, label_source, label_index, label, weight, count(*) AS n FROM {TABLE}"  # noqa: S608
        )
        params: list[Any] = []
        if task_id is not None:
            sql += " WHERE task_id = ?"
            params.append(task_id)
        sql += " GROUP BY ALL ORDER BY task_id, label_source, label_index, label, weight"
        return self._select(sql, params)

    def task_ids(self) -> list[str]:
        """The distinct ``task_id`` values present, sorted."""
        rows = self._select(f"SELECT DISTINCT task_id FROM {TABLE} ORDER BY task_id")  # noqa: S608
        return [str(r["task_id"]) for r in rows]
