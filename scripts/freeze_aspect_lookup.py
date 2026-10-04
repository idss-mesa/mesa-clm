#!/usr/bin/env python3
"""Freeze the DESIGN A6 aspect lookup table from the registered labels snapshot.

Under ``decider.closed_choice: rules`` Q2 takes the top two aspects of the M0 lookup
(``mesa_clm.bench.baselines.Lookup``). The lookup's items are frozen once, offline, from the
registered snapshot (``bench/snapshots/2026-09-29.parquet``, checked by sha256, label content and
M0's published counts): the items of M0's ``neon_aspect`` bench task, card, column name and silver
aspect, in the task's order. The table ships in the package (``src/mesa_clm/aspect_lookup.json``)
and its sha256 is pinned in ``mesa_clm.closed_choice.TABLE_SHA256``, so every host reads the same
items and nothing is fitted at serving time (DESIGN D15)::

    uv run python scripts/freeze_aspect_lookup.py --check   # exit 1 if the packaged table drifted
    uv run python scripts/freeze_aspect_lookup.py           # rewrite it; prints the new sha256

After a rewrite, put the printed sha256 in ``TABLE_SHA256``; the tests rebuild the table from the
snapshot and compare it with the packaged file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from mesa_clm.closed_choice import TABLE_PATH, TABLE_SHA256, freeze_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 on drift")
    args = ap.parse_args(argv)
    text = freeze_table()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    current = TABLE_PATH.read_text(encoding="utf-8") if TABLE_PATH.is_file() else None
    if args.check:
        if current != text:
            print(
                f"{TABLE_PATH.name}: differs from the table frozen from the snapshot",
                file=sys.stderr,
            )
            return 1
        if sha != TABLE_SHA256:
            print(f"{TABLE_PATH.name}: sha256 {sha} is not TABLE_SHA256", file=sys.stderr)
            return 1
        print(f"{TABLE_PATH.name}: in sync; sha256 {sha}")
        return 0
    TABLE_PATH.write_text(text, encoding="utf-8")
    note = "" if sha == TABLE_SHA256 else " (set mesa_clm.closed_choice.TABLE_SHA256 to it)"
    print(f"wrote {TABLE_PATH}; sha256 {sha}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
