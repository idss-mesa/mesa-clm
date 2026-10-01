"""DDL parity (DESIGN D11): the Postgres migration file and the DuckDB statements describe the
same tables, columns, nullability, defaults and CHECK constraints; the vocabulary CHECKs come from
``mesa_clm.vocab``; the ``labels`` table is ``LABELS_DDL`` verbatim; no ``NULLS NOT DISTINCT``
anywhere and every UNIQUE key is over NOT NULL columns (the sentinel rule), with
``CODE_CHECKED_UNIQUENESS`` naming exactly what the DDL cannot express (nothing)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from mesa_clm.provenance.labels import LABELS_DDL
from mesa_clm.provenance.migrate import (
    POSTGRES_INDEXES,
    TYPE_MAP,
    migration_files,
    render_postgres_migration,
    to_postgres,
)
from mesa_clm.provenance.models import (
    CALL_STATUSES,
    HISTORY_BACKENDS,
    LINK_OPS,
    OVERRIDE_ACTIONS,
    SCOPES,
    AuditRow,
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    LabelRow,
    RunRow,
)
from mesa_clm.provenance.store import (
    CODE_CHECKED_UNIQUENESS,
    DUCKDB_DDL,
    RUN_TABLES,
    TABLES,
    DuckDBStore,
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

_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS mesa_clm\.(\w+)\s*\((.*)\)\s*;?\s*$", re.S)
_TYPE_MAP = dict(TYPE_MAP)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    not_null: bool
    default: str | None
    check: str | None


def _split_top_level(body: str) -> list[str]:
    """Split a CREATE TABLE body on the commas outside parentheses; whitespace normalised."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [" ".join(p.split()) for p in parts if p.strip()]


def parse_table(statement: str, *, map_types: bool) -> tuple[str, list[Column], list[str]]:
    """``(table, columns, table constraints)`` of one CREATE TABLE statement; ``map_types``
    applies the DuckDB -> Postgres type mapping so both dialects compare in Postgres terms."""
    m = _TABLE_RE.search(statement)
    assert m, statement[:80]
    columns: list[Column] = []
    constraints: list[str] = []
    for part in _split_top_level(m.group(2)):
        if part.startswith(("CHECK", "UNIQUE", "PRIMARY KEY")):
            constraints.append(part)
            continue
        name, typ, *rest_parts = part.split(" ", 2)
        rest = rest_parts[0] if rest_parts else ""
        if rest.startswith("PRECISION"):
            typ += " PRECISION"
            rest = rest[len("PRECISION") :].strip()
        if map_types:
            typ = _TYPE_MAP.get(typ, typ)
        default = re.search(r"DEFAULT (\S+)", rest)
        check = re.search(r"CHECK \((.*)\)$", rest)
        columns.append(
            Column(
                name,
                typ,
                "NOT NULL" in rest or "PRIMARY KEY" in rest,
                default.group(1) if default else None,
                check.group(1) if check else None,
            )
        )
    return m.group(1), columns, constraints


def _duckdb_tables() -> dict[str, tuple[list[Column], list[str]]]:
    out = {}
    for stmt in DUCKDB_DDL:
        if stmt.startswith("CREATE TABLE"):
            name, cols, cons = parse_table(stmt, map_types=True)
            out[name] = (cols, cons)
    return out


def _postgres_tables(sql: str) -> dict[str, tuple[list[Column], list[str]]]:
    out = {}
    for stmt in sql.split(";\n"):
        stmt = stmt.strip()
        if stmt.startswith("CREATE TABLE"):
            name, cols, cons = parse_table(stmt, map_types=False)
            out[name] = (cols, cons)
    return out


@pytest.fixture(scope="module")
def migration_sql() -> str:
    files = migration_files()
    assert [v for v, _, _ in files] == [1]
    assert files[0][1] == "0001_mesa_clm.sql"
    return files[0][2]


# -- the file is the rendering ------------------------------------------------------------------------


def test_packaged_migration_is_the_rendering(migration_sql: str) -> None:
    """The .sql file is generated from DUCKDB_DDL; a vocabulary edit must regenerate it."""
    assert migration_sql == render_postgres_migration()
    assert migration_sql.startswith("-- 0001_mesa_clm:")
    assert "\nBEGIN;\n" in migration_sql and migration_sql.rstrip().endswith("COMMIT;")
    assert "INSERT INTO mesa_clm.schema_versions (version) VALUES (1) ON CONFLICT DO NOTHING;" in (
        migration_sql
    )
    for index in POSTGRES_INDEXES:
        assert f"{index};" in migration_sql
    packaged = Path(__file__).resolve().parents[2] / "src/mesa_clm/provenance/migrations"
    assert (packaged / "__init__.py").read_text(encoding="utf-8") == ""


def test_to_postgres_maps_whole_words_only() -> None:
    assert to_postgres("a JSON NOT NULL, b DOUBLE, c JSONB, d TEXT") == (
        "a JSONB NOT NULL, b DOUBLE PRECISION, c JSONB, d TEXT"
    )
    assert to_postgres(to_postgres("x JSON, y DOUBLE")) == to_postgres("x JSON, y DOUBLE")


# -- table by table ----------------------------------------------------------------------------------


def test_dialects_agree_table_by_table(migration_sql: str) -> None:
    duck = _duckdb_tables()
    pg = _postgres_tables(migration_sql)
    assert set(duck) == set(pg) == set(TABLES)
    for table in TABLES:
        d_cols, d_cons = duck[table]
        p_cols, p_cons = pg[table]
        assert [c.name for c in d_cols] == [c.name for c in p_cols], table
        assert [c.type for c in d_cols] == [c.type for c in p_cols], f"{table}: types"
        assert [c.not_null for c in d_cols] == [c.not_null for c in p_cols], f"{table}: nullability"
        assert [c.default for c in d_cols] == [c.default for c in p_cols], f"{table}: defaults"
        assert [c.check for c in d_cols] == [c.check for c in p_cols], f"{table}: column CHECKs"
        assert d_cons == p_cons, f"{table}: table constraints"


def test_live_duckdb_catalog_matches_the_statements(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    assert store.columns("runs") == []  # nothing exists yet, nothing is created by a read
    assert not (tmp_path / "prov.duckdb").exists()
    assert store.ensure_schema() == 1
    duck = _duckdb_tables()
    for table in TABLES:
        live = store.columns(table)
        cols, cons = duck[table]
        assert [name for name, _ in live] == [c.name for c in cols], table
        assert [nullable for _, nullable in live] == [not c.not_null for c in cols], table
        n_checks = sum(1 for c in cols if c.check) + sum(1 for c in cons if c.startswith("CHECK"))
        types = [t for t, _ in store.constraints(table)]
        assert types.count("CHECK") == n_checks, table
        assert types.count("UNIQUE") == sum(1 for c in cons if c.startswith("UNIQUE")), table
        assert types.count("PRIMARY KEY") == 1, table


def test_row_models_follow_the_ddl_column_order() -> None:
    duck = _duckdb_tables()
    models: dict[str, type[BaseModel]] = {
        "runs": RunRow,
        "decisions": DecisionRow,
        "decision_options": DecisionOptionRow,
        "decision_groups": DecisionGroupRow,
        "avu_links": AvuLinkRow,
        "human_overrides": HumanOverrideRow,
        "labels": LabelRow,
        "audits": AuditRow,
        "clm_calls": ClmCallRow,
    }
    assert set(models) == set(TABLES) - {"schema_versions"}
    for table, model in models.items():
        assert list(model.model_fields) == [c.name for c in duck[table][0]], table
    assert set(RUN_TABLES) <= set(TABLES) and "labels" not in RUN_TABLES


# -- labels verbatim ------------------------------------------------------------------------------------


def test_labels_ddl_is_labels_ddl_verbatim(migration_sql: str) -> None:
    """The DuckDB statements *are* the M0 objects, and the Postgres table is their translation."""
    assert LABELS_DDL[0] == DUCKDB_DDL[0] == "CREATE SCHEMA IF NOT EXISTS mesa_clm"
    for stmt in LABELS_DDL[1:]:
        assert any(s is stmt for s in DUCKDB_DDL), "labels DDL must be spliced in, not copied"
        assert to_postgres(stmt) + ";" in migration_sql
    _, cols, cons = parse_table(LABELS_DDL[1], map_types=True)
    assert [c.name for c in cols] == list(LabelRow.model_fields)
    assert "UNIQUE (task_key, target_sha256, option_key, label_source)" in cons


# -- vocabularies -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "column", "vocab"),
    [
        ("runs", "status", RUN_STATUSES),
        ("runs", "history_backend", HISTORY_BACKENDS),
        ("decisions", "shape", SHAPES),
        ("decisions", "scope", SCOPES),
        ("decisions", "method", METHODS),
        ("decisions", "level", LEVELS),
        ("decisions", "calibration", CALIBRATIONS),
        ("decisions", "outcome", OUTCOMES),
        ("decision_groups", "scope", SCOPES),
        ("decision_groups", "level", LEVELS),
        ("decision_groups", "method", METHODS),
        ("decision_groups", "outcome", OUTCOMES),
        ("avu_links", "op", LINK_OPS),
        ("avu_links", "write_status", WRITE_STATUSES),
        ("avu_links", "accepted_by", ACCEPTED_BY),
        ("human_overrides", "via", VIAS),
        ("human_overrides", "action", OVERRIDE_ACTIONS),
        ("human_overrides", "label_source", LABEL_SOURCES),
        ("labels", "label_source", LABEL_SOURCES),
        ("clm_calls", "status", CALL_STATUSES),
    ],
)
def test_vocabulary_checks_are_rendered_from_vocab(
    migration_sql: str, table: str, column: str, vocab: tuple[str, ...]
) -> None:
    expected = f"{column} IN ({sql_in_list(vocab)})"
    for dialect in (_duckdb_tables(), _postgres_tables(migration_sql)):
        cols, _ = dialect[table]
        col = next(c for c in cols if c.name == column)
        assert col.check is not None and expected in col.check, (table, column)


def test_decision_invariants_are_checks_in_both_dialects(migration_sql: str) -> None:
    """Plan §4.6 invariants 1, 2, 3 and 7 as table CHECKs (D6, D28)."""
    without = sql_in_list(tuple(sorted(METHODS_WITHOUT_PROBS)))
    expected = [
        "CHECK ((probs IS NULL) = (calibration = 'none'))",
        f"CHECK (method NOT IN ({without}) OR (probs IS NULL AND level = 'none'))",
        "CHECK (level <> 'zero_shot' OR calibration = 'uncalibrated')",
        "CHECK (level NOT IN ('calibrated', 'probe', 'head') OR calibration IN ('platt', 'temperature'))",
        "CHECK (method <> 'ols_rank' OR (outcome IN ('proposed', 'abstain') AND rank IS NOT NULL))",
        "CHECK (shape <> 'choice' OR anchor_index IS NULL)",
    ]
    for dialect in (_duckdb_tables(), _postgres_tables(migration_sql)):
        assert dialect["decisions"][1] == expected
        assert (
            "CHECK (write_status <> 'accepted' OR accepted_by IS NOT NULL)"
            in dialect["avu_links"][1]
        )
        assert (
            "CHECK (via <> 'tool' OR label_source IS NULL OR label_source = 'agent_pick')"
            in dialect["human_overrides"][1]
        )
        assert "CHECK (n_errors <= n)" in dialect["audits"][1]


# -- sentinels, not NULLs ---------------------------------------------------------------------------------


def test_unique_keys_are_over_not_null_columns_and_exemptions_are_exact(
    migration_sql: str,
) -> None:
    """DuckDB 1.5.x has no UNIQUE NULLS NOT DISTINCT (D11): every UNIQUE/PRIMARY KEY column is
    NOT NULL (sentinels), and ``CODE_CHECKED_UNIQUENESS`` lists exactly the keys the DDL lacks."""
    assert "NULLS NOT DISTINCT" not in "\n".join(DUCKDB_DDL)
    sql_body = "\n".join(line for line in migration_sql.splitlines() if not line.startswith("--"))
    assert "NULLS NOT DISTINCT" not in sql_body  # the header comment may name what is avoided
    ddl_keys: set[tuple[str, tuple[str, ...]]] = set()
    for table, (cols, cons) in _duckdb_tables().items():
        by_name = {c.name: c for c in cols}
        for con in cons:
            if con.startswith(("UNIQUE", "PRIMARY KEY")):
                names = tuple(n.strip() for n in con[con.index("(") + 1 : -1].split(","))
                ddl_keys.add((table, names))
                for name in names:
                    assert by_name[name].not_null, f"{table}.{name} is nullable inside {con}"
    assert (
        "avu_links",
        ("run_id", "attribute", "value", "unit", "column_name", "site_code"),
    ) in ddl_keys
    assert ("labels", ("task_key", "target_sha256", "option_key", "label_source")) in ddl_keys
    assert ("decision_options", ("decision_id", "option_key")) in ddl_keys
    # Nothing needed a NULL-able key, so nothing is checked in code instead.
    assert CODE_CHECKED_UNIQUENESS == ()
    assert not (set(CODE_CHECKED_UNIQUENESS) & ddl_keys)


def test_sentinel_columns_are_not_null_default_empty() -> None:
    duck = _duckdb_tables()
    links = {c.name: c for c in duck["avu_links"][0]}
    for name in ("unit", "column_name", "site_code"):
        assert links[name].not_null and links[name].default == "''", name
    labels = {c.name: c for c in duck["labels"][0]}
    for name in ("option_key", "card", "product_code", "leak_group", "origin", "actor"):
        assert labels[name].not_null and labels[name].default == "''", name
    assert all(c.not_null for c in duck["labels"][0])
