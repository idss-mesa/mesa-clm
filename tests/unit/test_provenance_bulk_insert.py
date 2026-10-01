"""Sidecar bulk inserts (``provenance.store.bulk_insert``): one bound JSON value per chunk
instead of one bound row per ``executemany`` step (DuckDB 1.5.6 pays a failed pandas import per
bound value when pandas is absent).

The rows must be exactly what the per-row path wrote (compared table by table on real runs),
the data must never reach the SQL text (hostile strings round-trip and the schema survives),
chunks JSON cannot carry fall back to ``executemany``, and committing a fake run of every
fixture card stays fast (a loose bound that the per-row path, about 1.2 s per card, misses).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import duckdb
import pytest
from pydantic import BaseModel

from mesa_clm.provenance import store as st
from mesa_clm.provenance.store import (
    BULK_INSERT_ROWS,
    DUCKDB_DDL,
    RUN_TABLES,
    SCHEMA,
    DuckDBStore,
    RunBuffer,
    bulk_insert,
    bulk_insert_sql,
    insert_sql,
    row_values,
    run_table_sql,
)
from tests.fakes.pipeline import CARD_PATHS, annotator, card

HOSTILE = [
    "plain",
    "O'Brien",
    "'); DROP TABLE mesa_clm.runs; --",
    '"; DELETE FROM mesa_clm.decisions; --',
    "back\\slash \\' \\\\",
    "question ? marks ?? and $1 $2",
    'json-ish {"a": [1, 2]} and ["x"]',
    "unicode: μ° – ✓ 𝛼 中文",
    "newline\nand\ttab\r\n",
    "",
]


@pytest.fixture(scope="module")
def buffers() -> list[RunBuffer]:
    """One fake run (not committed) per fixture card."""
    out = []
    for path in CARD_PATHS:
        run = annotator(None).annotate(card(path.stem))
        assert run.buffer is not None
        out.append(run.buffer)
    return out


def _dump(path: Path, run_id: UUID) -> dict[str, list[tuple[Any, ...]]]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        con.execute("SET TimeZone = 'UTC'")
        return {t: con.execute(run_table_sql(t), [str(run_id)]).fetchall() for t in RUN_TABLES}
    finally:
        con.close()


def _commit_per_row(path: Path, buffer: RunBuffer) -> None:
    """The pre-bulk commit: one ``executemany`` per table, one bound row per step."""
    con = duckdb.connect(str(path))
    try:
        con.execute("SET TimeZone = 'UTC'")
        for stmt in DUCKDB_DDL:
            con.execute(stmt)
        con.execute("BEGIN")
        for table, rows in buffer.rows().items():
            if rows:
                cols, _ = row_values(rows[0])
                con.executemany(insert_sql(table, cols), [row_values(r)[1] for r in rows])
        con.execute("COMMIT")
    finally:
        con.close()


def test_bulk_commit_writes_exactly_what_the_per_row_path_wrote(
    buffers: list[RunBuffer], tmp_path: Path
) -> None:
    for buffer in buffers[:3]:
        bulk, per_row = tmp_path / f"bulk-{buffer.run_id}.duckdb", tmp_path / "per_row.duckdb"
        DuckDBStore(bulk).commit_run(buffer)
        buffer.committed = False  # the same rows once more, through the old path
        _commit_per_row(per_row, buffer)
        a, b = _dump(bulk, buffer.run_id), _dump(per_row, buffer.run_id)
        assert a == b
        assert a["decisions"] and a["decision_options"] and a["decision_groups"] and a["runs"]


def test_seven_card_fake_run_commits_fast(buffers: list[RunBuffer], tmp_path: Path) -> None:
    assert len(buffers) == 7
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    started = time.perf_counter()
    for buffer in buffers:
        buffer.committed = False
        store.commit_run(buffer)
    elapsed = time.perf_counter() - started
    # The per-row path took about 1.2 s per card on the development host (8+ s here); the bulk
    # path about 0.1 s. The bound is loose on purpose: slow CI runners must not flake.
    assert elapsed < 4.0, f"committing 7 fake runs took {elapsed:.2f} s"
    for buffer in buffers:
        run = store.run(buffer.run_id)
        assert run is not None and run["n_decisions"] == len(buffer.decisions)
        assert len(store.options(buffer.run_id)) == len(buffer.options)


# -- the helper on its own table ---------------------------------------------------------------


class _Row(BaseModel):
    id: str
    note: str | None = None
    score: float | None = None
    n: int | None = None
    flag: bool = False
    meta: dict[str, Any] | None = None


def _scratch() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE SCHEMA {SCHEMA}")
    con.execute(
        f"CREATE TABLE {SCHEMA}.probe (id TEXT PRIMARY KEY, note TEXT, score DOUBLE, "
        "n INTEGER, flag BOOLEAN NOT NULL, meta JSON)"
    )
    return con


def test_hostile_text_is_data_never_sql() -> None:
    con = _scratch()
    rows = [
        _Row(id=f"r{i}", note=text, score=i / 3, n=i, flag=i % 2 == 0, meta={"text": text})
        for i, text in enumerate(HOSTILE)
    ]
    assert bulk_insert(con, "probe", rows) == len(rows)
    got = con.execute(
        "SELECT id, note, score, n, flag, meta FROM mesa_clm.probe ORDER BY n"
    ).fetchall()
    assert [g[1] for g in got] == HOSTILE
    assert [g[2] for g in got] == [i / 3 for i in range(len(HOSTILE))]
    assert all(isinstance(g[5], str) for g in got)
    # Every table the hostile strings name is still there.
    tables = {r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "probe" in tables
    # The statement carries one placeholder and none of the data.
    sql = bulk_insert_sql("probe", ["id", "note"], {"id": "VARCHAR", "note": "VARCHAR"})
    assert sql.count("?") == 1 and "DROP" not in sql


def test_identifiers_are_checked() -> None:
    types = {"id": "VARCHAR"}
    with pytest.raises(ValueError, match="bad column"):
        bulk_insert_sql("probe", ["id; DROP TABLE x"], types)
    with pytest.raises(ValueError, match="bad table"):
        bulk_insert_sql("probe x", ["id"], types)
    with pytest.raises(KeyError):
        bulk_insert_sql("probe", ["missing"], types)


def test_a_failed_cast_is_an_error_never_a_null() -> None:
    con = _scratch()
    # A text where the column is an INTEGER: from_json would silently give NULL.
    with pytest.raises(duckdb.Error):
        con.execute(
            bulk_insert_sql("probe", ["id", "n", "flag"], st.column_types(con, "probe")),
            ['[{"id": "x", "n": "not a number", "flag": true}]'],
        )
    # A constraint violation surfaces as it did through executemany.
    bulk_insert(con, "probe", [_Row(id="dup")])
    with pytest.raises(duckdb.ConstraintException):
        bulk_insert(con, "probe", [_Row(id="dup")])


def test_chunks_json_cannot_carry_fall_back_to_executemany(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con = _scratch()
    seen: list[int] = []
    real = st.insert_sql

    def spy(table: str, columns: Any) -> str:
        seen.append(len(columns))
        return real(table, columns)

    monkeypatch.setattr(st, "insert_sql", spy)
    rows = [_Row(id="a", score=math.inf), _Row(id="b", score=float("nan")), _Row(id="c", score=1)]
    assert bulk_insert(con, "probe", rows) == 3
    got = dict(con.execute("SELECT id, score FROM mesa_clm.probe").fetchall())
    assert got["a"] == math.inf and math.isnan(got["b"]) and got["c"] == 1.0
    assert seen  # the non-finite chunk went through the per-row statement


def test_chunking_and_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    con = _scratch()
    assert bulk_insert(con, "probe", []) == 0
    monkeypatch.setattr(st, "BULK_INSERT_ROWS", 7)
    rows = [_Row(id=str(uuid4()), n=i) for i in range(23)]
    assert bulk_insert(con, "probe", rows) == 23
    got = [r[0] for r in con.execute("SELECT n FROM mesa_clm.probe ORDER BY n").fetchall()]
    assert got == list(range(23))
    assert BULK_INSERT_ROWS == 1000


def test_set_link_status_updates_every_chunk(
    buffers: list[RunBuffer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = next(b for b in buffers if len(b.links) >= 3)
    store = DuckDBStore(tmp_path / "links.duckdb")
    buffer.committed = False
    store.commit_run(buffer)
    monkeypatch.setattr(st, "BULK_INSERT_ROWS", 2)
    ids = [link.link_id for link in buffer.links]
    assert store.set_link_status(ids, "accepted", accepted_by="human") == len(ids)
    links = store.links(buffer.run_id)
    assert {x["write_status"] for x in links} == {"accepted"}
    assert {x["accepted_by"] for x in links} == {"human"}
