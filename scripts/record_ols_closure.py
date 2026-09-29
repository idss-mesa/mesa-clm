#!/usr/bin/env python3
"""Record the OLS fixture closure once against the live EMBL-EBI OLS4 (plan M0).

The hermetic pipeline test replays ``tests/fixtures/ols`` for every fixture card; with a fake
CLM the aspect and ontology choices are not predictable, so the fixture set must hold every
call any decision path could make. ``mesa_clm.ols_closure.enumerate_calls`` enumerates that
closure (every non-identifier column × every aspect-allowed ontology × the static planner's
queries, the unit fallback, the site biome queries, the dataset taxon queries, and the
children of every candidate with children); this script records what is missing::

    uv run python scripts/record_ols_closure.py --dry-run      # count only, no network
    uv run python scripts/record_ols_closure.py                # record (live EBI OLS4)

Polite and resumable: requests go out sequentially at most ``--rate`` per second (default 4)
through ``ThrottledOLS`` (exponential backoff on connection errors, timeouts, 429 and 5xx;
other 4xx are final), and a call whose fixture file already exists is never sent again, so an
interrupted run is simply restarted. Recording goes through ``RecordingOLS(mode="auto")``,
which writes exactly the fixture layout the tests replay.

Two levels: the search level (``search_terms`` / ``search_term_descendants``) is enumerated
from the cards alone; the ``get_term_children`` level is derived from the recorded search
responses (``OLSLayer`` merge, sort and cap, ``max_candidates`` 12), so it is enumerated after
the search level is complete. ``RecordingOLS`` does not persist errors: a call that still fails
after the retries is listed in the report (``--report``, default
``.local/ols_closure_report.json``) and left missing, so ``test_ols_fixture_closure`` names it.
The report also carries the counts, runtime and fixture volume this run produced.

Live network is used only here, never in the default tests (``CLAUDE.md`` testing rules).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from mesa_clm.cards import DatasetCard, load_card
from mesa_clm.ols import OLS_ERRORS, RecordingOLS
from mesa_clm.ols_closure import (
    CHILDREN_SIZE,
    DEFAULT_MAX_CANDIDATES,
    SEARCH_SIZE,
    Call,
    ThrottledOLS,
    children_calls,
    count_by_method,
    describe_calls,
    enumerate_groups,
    missing_fixtures,
    search_calls,
)

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "tests" / "fixtures" / "cards"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ols"
REPORT = ROOT / ".local" / "ols_closure_report.json"


def _cards(cards_dir: Path) -> list[DatasetCard]:
    paths = sorted(cards_dir.glob("*.md"))
    if not paths:
        sys.exit(f"no cards under {cards_dir}")
    return [load_card(p) for p in paths]


def _volume(fixture_dir: Path) -> tuple[int, int]:
    """``(files, bytes)`` of the fixture directory."""
    files = list(fixture_dir.glob("*.json"))
    return len(files), sum(p.stat().st_size for p in files)


def _record(
    recorder: RecordingOLS, calls: list[Call], *, label: str, every: int = 50
) -> tuple[int, int]:
    """Send every call of ``calls`` through ``recorder`` (each one records its own fixture on
    success); returns ``(recorded, failed)``."""
    recorded = failed = 0
    started = time.monotonic()
    for i, (method, args) in enumerate(calls, 1):
        try:
            getattr(recorder, method)(**args)
            recorded += 1
        except OLS_ERRORS:
            failed += 1  # detailed in ThrottledOLS.failures
        if i % every == 0 or i == len(calls):
            elapsed = time.monotonic() - started
            print(
                f"  {label}: {i}/{len(calls)} ({recorded} recorded, {failed} failed, "
                f"{elapsed:.0f}s)",
                flush=True,
            )
    return recorded, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cards-dir", type=Path, default=CARDS_DIR)
    ap.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--rate", type=float, default=4.0, help="max requests per second (default 4)")
    ap.add_argument("--attempts", type=int, default=5, help="tries per call incl. the first")
    ap.add_argument("--backoff", type=float, default=1.0, help="first retry delay in seconds")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout per request")
    ap.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    ap.add_argument(
        "--children-for-all",
        action="store_true",
        help="children of every candidate, not only those with hasChildren",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="record at most this many missing calls (0 = all)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="enumerate and count only; nothing is sent"
    )
    args = ap.parse_args(argv)

    cards = _cards(args.cards_dir)
    groups = enumerate_groups(cards)
    level1 = search_calls(groups, search_size=SEARCH_SIZE)
    missing1 = missing_fixtures(level1, args.fixture_dir)
    files0, bytes0 = _volume(args.fixture_dir)
    print(
        f"{len(cards)} cards, {len(groups)} candidate groups -> {len(level1)} unique search "
        f"calls {count_by_method(level1)}; missing {len(missing1)} "
        f"{count_by_method(missing1)}; fixtures now {files0} files, {bytes0 / 1e6:.2f} MB"
    )

    replay = RecordingOLS(None, args.fixture_dir, "replay")
    if args.dry_run:
        if missing1:
            print("children level: not enumerable until the search level is recorded")
            return 0
        level2 = children_calls(
            groups,
            replay,
            max_candidates=args.max_candidates,
            children_for_all=args.children_for_all,
        )
        missing2 = missing_fixtures(level2, args.fixture_dir)
        print(
            f"children level: {len(level2)} unique get_term_children calls, missing "
            f"{len(missing2)}; closure total {len(level1) + len(level2)} unique calls"
        )
        return 0

    # Imported here so --dry-run needs no live client and no network.
    from mesa_mcp.ols.client import OLSClient

    throttled = ThrottledOLS(
        OLSClient(request_timeout=args.timeout),
        rate=args.rate,
        attempts=args.attempts,
        backoff=args.backoff,
    )
    recorder = RecordingOLS(throttled, args.fixture_dir, "auto")
    started = time.monotonic()

    todo1 = missing1[: args.limit] if args.limit else missing1
    print(f"search level: recording {len(todo1)} calls at <= {args.rate}/s", flush=True)
    rec1, fail1 = _record(recorder, todo1, label="search")

    # The children level is derived from the search responses; ``auto`` replays what is on
    # disk and records (and, for a still-failing search, retries then skips) the rest. A
    # ``--limit`` that truncated the search level stops here, or the enumeration itself would
    # fetch every search the limit left out.
    level2: list[Call] = []
    rec2 = fail2 = 0
    if len(todo1) < len(missing1):
        print("children level: skipped (--limit truncated the search level; rerun without it)")
    else:
        level2 = children_calls(
            groups,
            recorder,
            max_candidates=args.max_candidates,
            children_for_all=args.children_for_all,
        )
        missing2 = missing_fixtures(level2, args.fixture_dir)
        todo2 = missing2[: max(0, args.limit - len(todo1))] if args.limit else missing2
        print(
            f"children level: {len(level2)} unique get_term_children calls, missing "
            f"{len(missing2)}, recording {len(todo2)}",
            flush=True,
        )
        rec2, fail2 = _record(recorder, todo2, label="children")

    seconds = time.monotonic() - started
    files1, bytes1 = _volume(args.fixture_dir)
    still_missing = missing_fixtures(level1 + level2, args.fixture_dir)
    report: dict[str, Any] = {
        "cards": [c.name for c in cards],
        "groups": len(groups),
        "unique_calls": {
            "search": len(level1),
            "children": len(level2),
            "total": len(level1) + len(level2),
            "by_method": count_by_method(level1 + level2),
        },
        "recorded": {"search": rec1, "children": rec2},
        "failed": {"search": fail1, "children": fail2},
        "requests": throttled.requests,
        "retries": throttled.retries,
        "failures": throttled.failures,
        "still_missing": describe_calls(still_missing),
        "seconds": round(seconds, 1),
        "fixtures_before": {"files": files0, "bytes": bytes0},
        "fixtures_after": {"files": files1, "bytes": bytes1},
        "rate": args.rate,
        "max_candidates": args.max_candidates,
        "children_for_all": args.children_for_all,
        "children_size": CHILDREN_SIZE,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"done in {seconds:.0f}s: {rec1 + rec2} recorded, {fail1 + fail2} failed "
        f"({throttled.requests} requests, {throttled.retries} retries); fixtures "
        f"{files1} files, {bytes1 / 1e6:.2f} MB (+{(bytes1 - bytes0) / 1e6:.2f} MB); "
        f"still missing {len(still_missing)}; report {args.report}"
    )
    for name in report["still_missing"][:20]:
        print("  missing:", name)
    return 1 if still_missing else 0


if __name__ == "__main__":
    sys.exit(main())
