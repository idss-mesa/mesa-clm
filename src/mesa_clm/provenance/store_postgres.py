"""The Postgres dialect of the sidecar (optional, ``pg`` extra; DESIGN D11, D12: never the
default). Same protocol and the same SQL text as :class:`~mesa_clm.provenance.store.DuckDBStore`
(the shared statements use ``?`` placeholders and are rewritten to ``%s`` here), against the
packaged migration ``migrations/0001_mesa_clm.sql`` (JSONB, TEXT ids, indexes, the same CHECKs).
One connection, autocommit off, one transaction per public call (:meth:`PostgresStore._tx`:
commit on success, roll back on any error so the connection never stays in an aborted
transaction); a run buffer commits in one transaction. The store never touches mesa-ducklake's
schema ``mesa``; both may share a database.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from mesa_clm.provenance.labels import IDENTITY_COLUMNS, LabelRow
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
    DECISIONS_FOR_PATH_SQL,
    LINK_SNAPSHOT_SQL,
    SCHEMA,
    SCHEMA_VERSION,
    RunBuffer,
    delete_run_sql,
    finish_run_sql,
    insert_sql,
    parse_row,
    run_table_sql,
    runs_sql,
    set_link_status_sql,
    update_group_sql,
)
from mesa_clm.tasks import TASKS


def _pg(sql: str) -> str:
    """The shared ``?`` statement in psycopg's ``%s`` form (no statement holds a literal ``?``)."""
    return sql.replace("?", "%s")


def _pg_cell(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict | list):
        return Jsonb(value)
    return value


def _row(row: BaseModel) -> tuple[list[str], list[Any]]:
    data = row.model_dump()
    return list(data), [_pg_cell(v) for v in data.values()]


class PostgresStore:
    """One connection, autocommit off, one commit per public call."""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self.dsn = dsn
        self._con = psycopg.connect(dsn)

    # -- schema ------------------------------------------------------------------------------------
    def ensure_schema(self) -> int:
        from mesa_clm.provenance.migrate import apply_migrations

        return int(apply_migrations(self.dsn) or SCHEMA_VERSION)

    def columns(self, table: str) -> list[tuple[str, bool]]:
        rows = self._select(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            [SCHEMA, table],
        )
        return [(str(r["column_name"]), str(r["is_nullable"]).upper() == "YES") for r in rows]

    # -- writes --------------------------------------------------------------------------------------
    @contextmanager
    def _tx(self) -> Iterator[Any]:
        """One transaction: yield a cursor, commit on success, roll back on any error."""
        try:
            with self._con.cursor() as cur:
                yield cur
        except BaseException:
            self._con.rollback()
            raise
        self._con.commit()

    @staticmethod
    def _insert(cur: Any, table: str, rows: Sequence[BaseModel], *, ignore: bool = False) -> int:
        """Insert through ``cur`` (the caller's transaction); ``ignore`` skips conflicts."""
        if not rows:
            return 0
        cols, _ = _row(rows[0])
        sql = _pg(insert_sql(table, cols))
        if ignore:
            sql += " ON CONFLICT DO NOTHING"
        inserted = 0
        for r in rows:
            cur.execute(sql, _row(r)[1])
            inserted += max(cur.rowcount, 0)
        return inserted

    def _execute(self, sql: str, params: Sequence[Any]) -> int:
        with self._tx() as cur:
            cur.execute(_pg(sql), list(params))
            return max(int(cur.rowcount), 0)

    def commit_run(self, buffer: RunBuffer) -> UUID:
        if buffer.committed:
            raise ValueError(f"run {buffer.run_id} was already committed")
        with self._tx() as cur:
            for table, rows in buffer.rows().items():
                self._insert(cur, table, rows)
        buffer.committed = True
        return buffer.run_id

    def _insert_commit(self, table: str, rows: Sequence[BaseModel], *, ignore: bool = False) -> int:
        with self._tx() as cur:
            return self._insert(cur, table, rows, ignore=ignore)

    def begin_run(self, run: RunRow) -> UUID:
        self._insert_commit("runs", [run])
        return run.run_id

    def finish_run(self, run_id: UUID, status: str, **stats: Any) -> None:
        sql, values = finish_run_sql(status, stats)
        self._execute(sql, [*values, str(run_id)])

    def insert_decisions(
        self, rows: Sequence[DecisionRow], options: Sequence[DecisionOptionRow] = ()
    ) -> int:
        with self._tx() as cur:
            n = self._insert(cur, "decisions", rows)
            self._insert(cur, "decision_options", options)
        return n

    def insert_group(self, row: DecisionGroupRow) -> UUID:
        self._insert_commit("decision_groups", [row])
        return row.group_id

    def update_group(self, group_id: UUID, **summary: Any) -> None:
        got = update_group_sql(summary)
        if got is None:
            return
        sql, values = got
        self._execute(sql, [*values, str(group_id)])

    def insert_links(self, rows: Sequence[AvuLinkRow]) -> int:
        return self._insert_commit("avu_links", rows)

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
        with self._tx() as cur:
            cur.executemany(_pg(sql), [[*values, i] for i in ids])
        return len(ids)

    def link_snapshot(
        self, run_id: UUID, irods_path: str, project_id: str | None, snapshot_id: int
    ) -> int:
        return self._execute(LINK_SNAPSHOT_SQL, [snapshot_id, project_id, str(run_id), irods_path])

    def insert_override(self, row: HumanOverrideRow) -> UUID:
        self._insert_commit("human_overrides", [row])
        return row.override_id

    def insert_labels(self, rows: Sequence[LabelRow]) -> int:
        """``ON CONFLICT DO NOTHING`` on the D1 identity; returns how many were new."""
        return self._insert_commit("labels", rows, ignore=True)

    def insert_audit(self, row: AuditRow) -> UUID:
        self._insert_commit("audits", [row])
        return row.audit_id

    def insert_clm_call(self, row: ClmCallRow) -> UUID:
        self._insert_commit("clm_calls", [row])
        return row.call_id

    def insert_clm_calls(self, rows: Sequence[ClmCallRow]) -> int:
        return self._insert_commit("clm_calls", rows)

    def delete_run(self, run_id: UUID) -> dict[str, int]:
        deleted: dict[str, int] = {}
        with self._tx() as cur:
            for table, sql in delete_run_sql():
                cur.execute(_pg(sql), [str(run_id)])
                deleted[table] = max(cur.rowcount, 0)
        return deleted

    # -- reads ---------------------------------------------------------------------------------------
    def _select(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        try:
            with self._con.cursor() as cur:
                cur.execute(sql, list(params))
                names = [d.name for d in cur.description or []]
                rows = cur.fetchall()
        finally:
            self._con.rollback()  # end the read transaction, failed or not
        return [parse_row(dict(zip(names, v, strict=True))) for v in rows]

    def run(self, run_id: UUID) -> dict[str, Any] | None:
        rows = self._select(f"SELECT * FROM {SCHEMA}.runs WHERE run_id = %s", [str(run_id)])  # noqa: S608
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
        sql, params = runs_sql(
            owner=owner,
            status=status,
            card_name=card_name,
            irods_path=irods_path,
            since=since,
            limit=limit,
        )
        return self._select(_pg(sql), params)

    def _run_table(self, table: str, run_id: UUID) -> list[dict[str, Any]]:
        return self._select(_pg(run_table_sql(table)), [str(run_id)])

    def decisions(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("decisions", run_id)

    def options(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("decision_options", run_id)

    def groups(self, run_id: UUID) -> list[dict[str, Any]]:
        return self._run_table("decision_groups", run_id)

    def group(self, group_id: UUID) -> dict[str, Any] | None:
        rows = self._select(
            f"SELECT * FROM {SCHEMA}.decision_groups WHERE group_id = %s",  # noqa: S608
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
            sql += " WHERE task_key = %s"
            params.append(task_key)
        return self._select(sql + " ORDER BY created_at, audit_id", params)

    def decisions_for_path(self, irods_path: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._select(_pg(DECISIONS_FOR_PATH_SQL), [irods_path, int(limit)])

    def labels_for(
        self,
        task_id: str,
        *,
        min_weight: float = 0.0,
        exclude_cards: Sequence[str] = (),
        sources: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """The same rows and order as :meth:`LabelStore.labels_for` (the task's *current* key,
        ``weight >= min_weight``, minus ``exclude_cards``, only ``sources`` when given)."""
        sql = f"SELECT * FROM {SCHEMA}.labels WHERE task_key = %s AND weight >= %s"  # noqa: S608
        params: list[Any] = [TASKS[task_id].key, float(min_weight)]
        if exclude_cards:
            sql += " AND card NOT IN (" + ", ".join("%s" for _ in exclude_cards) + ")"
            params.extend(exclude_cards)
        if sources is not None:
            if not sources:
                return []
            sql += " AND label_source IN (" + ", ".join("%s" for _ in sources) + ")"
            params.extend(sources)
        sql += " ORDER BY " + ", ".join(IDENTITY_COLUMNS) + ", created_at, label_id"
        rows = self._select(sql, params)
        for row in rows:
            if isinstance(row.get("state_json"), str):
                row["state_json"] = json.loads(row["state_json"])
        return rows

    def close(self) -> None:
        self._con.close()
