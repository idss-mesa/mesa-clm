"""``provenance.store.bulk_insert`` against ``executemany`` on edge values (DESIGN
implementation notes M1, "Bulk sidecar inserts"): random and extreme doubles (bit for bit,
subnormals and signed zeros included), the INT32 and INT64 bounds, hostile and very long text,
nested JSON, and aware and naive timestamps read in UTC and in America/Denver (a DST gap, an
ambiguous hour, an offset with seconds). The JSON path must store exactly what the per-row path
stores; ported from the M1 integration review's scratch parity test."""

from __future__ import annotations

import math
import random
import struct
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import duckdb
import pytest
from pydantic import BaseModel

from mesa_clm.provenance.store import SCHEMA, bulk_insert, insert_sql, row_values

DDL = (
    "(id INTEGER PRIMARY KEY, t TEXT, d DOUBLE, i INTEGER, b BIGINT, f BOOLEAN, j JSON, "
    "ts TIMESTAMPTZ)"
)


class Row(BaseModel):
    id: int
    t: str | None = None
    d: float | None = None
    i: int | None = None
    b: int | None = None
    f: bool | None = None
    j: Any = None
    ts: datetime | None = None


def _con(tz: str = "UTC") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET TimeZone = '{tz}'")
    con.execute(f"CREATE SCHEMA {SCHEMA}")
    for name in ("bulk", "rows"):
        con.execute(f"CREATE TABLE {SCHEMA}.{name} {DDL}")
    return con


def _key(v: Any) -> Any:
    return ("f", struct.pack("<d", v)) if isinstance(v, float) else v


def _diffs(con: duckdb.DuckDBPyConnection, rows: list[Row]) -> list[str]:
    bulk_insert(con, "bulk", rows)
    cols, _ = row_values(rows[0])
    con.executemany(insert_sql("rows", cols), [row_values(r)[1] for r in rows])
    a = con.execute(f"SELECT * FROM {SCHEMA}.bulk ORDER BY id").fetchall()  # noqa: S608
    b = con.execute(f"SELECT * FROM {SCHEMA}.rows ORDER BY id").fetchall()  # noqa: S608
    assert len(a) == len(b) == len(rows)
    return [
        f"bulk={ra!r} executemany={rb!r}"[:300]
        for ra, rb in zip(a, b, strict=True)
        if tuple(map(_key, ra)) != tuple(map(_key, rb))
    ]


def test_random_and_extreme_doubles_bit_for_bit() -> None:
    rng = random.Random(0)
    values = [rng.uniform(-1, 1) for _ in range(3000)]
    values += [
        struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0] for _ in range(3000)
    ]
    values = [v for v in values if math.isfinite(v)]
    values += [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308]
    values += [0.1, 1 / 3, 2.0**53 + 1.0, 1e-320]
    assert _diffs(_con(), [Row(id=k, d=v) for k, v in enumerate(values)]) == []


def test_integer_bounds_text_booleans_and_json() -> None:
    texts = ["", "nul\x00byte", "  ", "é", "\U0001f600", "a" * 100_000, "\\u0000", '{"a":1}']
    texts += ["x'y", "line\r\n", " ", "é"]
    rows = [
        Row(
            id=k,
            t=t,
            i=(2**31 - 1) if k % 2 else -(2**31),
            b=(2**63 - 1) if k % 2 else -(2**63),
            f=bool(k % 2),
            j={"k": t, "n": [1, 2.5, None], "u": "μ", "deep": {"x": [{"y": t}]}},
        )
        for k, t in enumerate(texts)
    ]
    assert _diffs(_con(), rows) == []


@pytest.mark.parametrize("tz", ["UTC", "America/Denver"])
def test_aware_and_naive_timestamps(tz: str) -> None:
    stamps = [
        datetime(2026, 10, 1, 8, 7, 25, 123456, tzinfo=UTC),
        datetime(2026, 10, 1, 8, 7, 25, 123456, tzinfo=timezone(timedelta(hours=-6))),
        datetime(2026, 3, 8, 2, 30),  # naive, inside Denver's DST gap
        datetime(2026, 11, 1, 1, 30),  # naive, Denver's ambiguous hour
        datetime(1900, 1, 2, tzinfo=UTC),
        datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    ]
    assert _diffs(_con(tz), [Row(id=k, ts=s) for k, s in enumerate(stamps)]) == []


def test_an_offset_with_seconds() -> None:
    lmt = timezone(timedelta(hours=-6, minutes=-59, seconds=-56))  # a local-mean-time offset
    assert _diffs(_con(), [Row(id=0, ts=datetime(1880, 1, 1, 12, tzinfo=lmt))]) == []
