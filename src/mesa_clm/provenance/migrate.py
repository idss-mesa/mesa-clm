"""Migrations: the Postgres runner (``mesa_clm.schema_versions``), the DuckDB bootstrap and the
renderer that derives the Postgres migration from the DuckDB DDL (DESIGN D11).

Modelled on ``mesa_ducklake.schema.apply_migrations`` but self-contained: migrations are
packaged (``importlib.resources``, ``migrations/NNNN_name.sql``), applied in numeric order, each
file carrying its own ``BEGIN``/``COMMIT``, and recorded in ``mesa_clm.schema_versions``. A
``duckdb:///`` DSN runs :meth:`DuckDBStore.ensure_schema` instead (DuckDB gets the DDL directly,
``CREATE ... IF NOT EXISTS``).

``0001_mesa_clm.sql`` is not written by hand: :func:`render_postgres_migration` translates
:data:`mesa_clm.provenance.store.DUCKDB_DDL` statement by statement (:func:`to_postgres`:
``JSON`` -> ``JSONB``, ``DOUBLE`` -> ``DOUBLE PRECISION``, ids stay ``TEXT``) and appends the
Postgres-only indexes, and ``tests/unit/test_provenance_ddl.py`` asserts the packaged file equals
the rendering byte for byte. A vocabulary change therefore fails CI until the file is
regenerated (``python -c 'from mesa_clm.provenance.migrate import render_postgres_migration as
r; print(r(), end="")' > src/mesa_clm/provenance/migrations/0001_mesa_clm.sql``); once 0001 has
shipped, such a change belongs in a ``0002`` migration instead and the equality test moves to
the rendering of the migration set (open item, plan §4.6).
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Final

from mesa_clm.provenance.store import (
    DUCKDB_DDL,
    SCHEMA,
    SCHEMA_VERSION,
    DuckDBStore,
    duckdb_file,
    is_postgres_dsn,
)

_NAME: Final = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
_PACKAGE: Final = "mesa_clm.provenance.migrations"

# DuckDB type -> Postgres type, applied to whole words of the DDL text. Everything else
# (TEXT, INTEGER, BIGINT, BOOLEAN, TIMESTAMPTZ, now(), CHECK/UNIQUE/PRIMARY KEY) is common.
TYPE_MAP: Final[tuple[tuple[str, str], ...]] = (
    ("JSON", "JSONB"),
    ("DOUBLE", "DOUBLE PRECISION"),
)

# Postgres-only indexes (DuckDB's ART indexes bring update caveats and buy nothing on a per-host
# file): the joins ``explain``, ``reconcile``, ``prune`` and the label loaders make.
POSTGRES_INDEXES: Final[tuple[str, ...]] = (
    f"CREATE INDEX IF NOT EXISTS runs_owner_started ON {SCHEMA}.runs (owner, started_at)",
    f"CREATE INDEX IF NOT EXISTS decisions_run_seq ON {SCHEMA}.decisions (run_id, seq)",
    f"CREATE INDEX IF NOT EXISTS decisions_question_key_ts ON {SCHEMA}.decisions (question_key, ts)",
    f"CREATE INDEX IF NOT EXISTS decisions_group ON {SCHEMA}.decisions (group_id)",
    f"CREATE INDEX IF NOT EXISTS decision_groups_run ON {SCHEMA}.decision_groups (run_id)",
    f"CREATE INDEX IF NOT EXISTS avu_links_run ON {SCHEMA}.avu_links (run_id)",
    f"CREATE INDEX IF NOT EXISTS avu_links_path ON {SCHEMA}.avu_links (irods_path, attribute, value, unit)",
    f"CREATE INDEX IF NOT EXISTS avu_links_snapshot ON {SCHEMA}.avu_links (project_id, snapshot_id)",
    f"CREATE INDEX IF NOT EXISTS human_overrides_run ON {SCHEMA}.human_overrides (run_id)",
    f"CREATE INDEX IF NOT EXISTS labels_task_source ON {SCHEMA}.labels (task_key, label_source)",
    f"CREATE INDEX IF NOT EXISTS clm_calls_run ON {SCHEMA}.clm_calls (run_id)",
)

_HEADER: Final = """-- 0001_mesa_clm: the mesa-clm decision provenance sidecar (DESIGN D11), Postgres dialect.
--
-- GENERATED from mesa_clm.provenance.store.DUCKDB_DDL by
-- mesa_clm.provenance.migrate.render_postgres_migration(); do not edit by hand. The DuckDB
-- dialect is the source: this file is its word-for-word translation (JSON -> JSONB,
-- DOUBLE -> DOUBLE PRECISION; ids are TEXT in both) plus the Postgres-only indexes, and
-- tests/unit/test_provenance_ddl.py asserts the packaged file equals the rendering and that
-- the two dialects expose the same tables, columns, nullability and CHECK constraints.
--
-- The vocabulary CHECKs (level, calibration, method, shape, outcome, write_status,
-- accepted_by, via, label_source, run status) are rendered from mesa_clm.vocab; the
-- ``labels`` table is mesa_clm.provenance.labels.LABELS_DDL verbatim (D1, D30). Every UNIQUE
-- key is over NOT NULL columns ('' sentinels) because DuckDB 1.5.x has no
-- UNIQUE NULLS NOT DISTINCT; the same shape is kept here so the dialects stay identical.
-- No foreign keys: rows are written in one transaction per run and pruned per run.
-- This schema never touches schema `mesa` (mesa-ducklake); the join into the AVU history is
-- (project_id, snapshot_id, irods_path, attribute, value, unit) on avu_links.
"""


def to_postgres(statement: str) -> str:
    """One DuckDB DDL statement in Postgres dialect (whole-word type mapping only; idempotent, so
    a statement already in Postgres dialect passes through unchanged)."""
    out = statement
    for src, dst in TYPE_MAP:
        tail = dst[len(src) :]  # "" for JSON -> JSONB, " PRECISION" for DOUBLE
        out = re.sub(rf"\b{src}\b(?!{re.escape(tail)})" if tail else rf"\b{src}\b", dst, out)
    return out


def render_postgres_migration() -> str:
    """The text of ``migrations/0001_mesa_clm.sql``: header, ``BEGIN``, every DDL statement
    translated, the indexes, the version row, ``COMMIT``."""
    lines = [_HEADER, "BEGIN;", ""]
    for stmt in DUCKDB_DDL:
        lines.append(to_postgres(stmt) + ";")
        lines.append("")
    lines.extend(f"{stmt};" for stmt in POSTGRES_INDEXES)
    lines.append("")
    lines.append(
        f"INSERT INTO {SCHEMA}.schema_versions (version) VALUES ({SCHEMA_VERSION}) "  # noqa: S608
        "ON CONFLICT DO NOTHING;"
    )
    lines.append("")
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def migration_files() -> list[tuple[int, str, str]]:
    """``(version, filename, sql)`` for every packaged migration, ascending."""
    out: list[tuple[int, str, str]] = []
    for entry in resources.files(_PACKAGE).iterdir():
        m = _NAME.match(entry.name)
        if m:
            out.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    return sorted(out)


def current_version(dsn: str) -> int:
    """The highest applied version (0 when the schema does not exist yet)."""
    path = duckdb_file(dsn)
    if path is not None:
        rows = DuckDBStore(path)._select(f"SELECT max(version) AS v FROM {SCHEMA}.schema_versions")  # noqa: S608
        return int(rows[0]["v"] or 0) if rows else 0
    if not is_postgres_dsn(dsn):
        raise ValueError("unsupported DSN (duckdb:///<path> or postgresql://...)")
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", [f"{SCHEMA}.schema_versions"])
        exists = cur.fetchone()
        if not exists or exists[0] is None:
            return 0
        cur.execute(f"SELECT COALESCE(MAX(version), 0) FROM {SCHEMA}.schema_versions")  # noqa: S608
        row = cur.fetchone()
        return int(row[0]) if row else 0


def apply_migrations(dsn: str, target: int | None = None) -> int:
    """Apply pending migrations up to ``target`` (all by default); return the version now in
    force. DuckDB: the store's bootstrap (one version). Postgres: each packaged file in its own
    transaction (the file carries ``BEGIN``/``COMMIT``)."""
    path = duckdb_file(dsn)
    if path is not None:
        return DuckDBStore(path).ensure_schema()
    if not is_postgres_dsn(dsn):
        raise ValueError("unsupported DSN (duckdb:///<path> or postgresql://...)")
    import psycopg

    applied = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_versions "
                "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            cur.execute(f"SELECT COALESCE(MAX(version), 0) FROM {SCHEMA}.schema_versions")  # noqa: S608
            row = cur.fetchone()
            current = int(row[0]) if row else 0
        conn.commit()
        for version, _name, sql in migration_files():
            if version <= current or (target is not None and version > target):
                continue
            with conn.cursor() as cur:
                cur.execute(sql)  # the file carries its own BEGIN/COMMIT
            conn.commit()
            applied = version
    return applied or current
