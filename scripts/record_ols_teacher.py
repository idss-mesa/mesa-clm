#!/usr/bin/env python3
"""Record the OLS ``get_term`` fixtures the teacher ingest needs, once, from live EBI OLS4 (M4,
DESIGN D19; ``learn/teacher.py``).

``labels ingest-teacher`` resolves every in-registry, aspect-mapped CURIE of the neon-ducklake
curation corpus through the OLS layer (``TermResolver.get_term``), and the hermetic suite and
the registered ingest replay ``tests/fixtures/ols-teacher`` instead of the network. This script
enumerates those CURIEs from the corpus (label-free: it reads the corpus and no silver label or
model output) and records the ``get_term`` response of each that has no fixture yet::

    uv run python scripts/record_ols_teacher.py --dry-run     # count only, no network
    uv run python scripts/record_ols_teacher.py               # record (live EBI OLS4)

Polite and resumable, like ``record_ols_closure.py``: sequential requests at most ``--rate`` per
second through ``ThrottledOLS`` (backoff on connection errors, timeouts, 429 and 5xx), a CURIE
whose fixture exists is never sent again, and the report (``--report``, default
``.local/ols_teacher_report.json``) records the counts, the time window and the fixture volume
for DESIGN's M4 disclosure. Live network is used only here, never in the default tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mesa_clm.learn.teacher import CORPUS_DIR, corpus_sha256, load_corpus, teacher_curies
from mesa_clm.ols import OLS_ERRORS, RecordingOLS, iri_for
from mesa_clm.ols_closure import ThrottledOLS
from mesa_clm.registry import prefix_of

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ols-teacher"
REPORT = ROOT / ".local" / "ols_teacher_report.json"


def _volume(fixture_dir: Path) -> tuple[int, int]:
    files = list(fixture_dir.glob("*.json"))
    return len(files), sum(p.stat().st_size for p in files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--neon-root", type=Path, default=Path("~/neon-ducklake").expanduser())
    ap.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--rate", type=float, default=4.0, help="max requests per second (default 4)")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true", help="enumerate and count only")
    args = ap.parse_args(argv)

    files = load_corpus(args.neon_root)
    curies = teacher_curies(files)
    calls = [(prefix_of(c).lower(), iri_for(c)) for c in curies]
    replay = RecordingOLS(None, args.fixture_dir, "replay")
    missing = [
        (ont, iri)
        for ont, iri in calls
        if not replay.fixture_path("get_term", {"ontology_id": ont, "iri": iri}).exists()
    ]
    files0, bytes0 = _volume(args.fixture_dir) if args.fixture_dir.is_dir() else (0, 0)
    print(
        f"{len(files)} corpus files, {sum(len(f.items) for f in files)} items, {len(curies)} "
        f"in-registry mapped CURIEs -> {len(calls)} get_term calls; missing {len(missing)}; "
        f"fixtures now {files0} files, {bytes0 / 1e6:.3f} MB"
    )
    if args.dry_run:
        return 0

    from mesa_mcp.ols.client import OLSClient  # live network only here

    throttled = ThrottledOLS(
        OLSClient(request_timeout=args.timeout),
        rate=args.rate,
        attempts=args.attempts,
        backoff=args.backoff,
    )
    recorder = RecordingOLS(throttled, args.fixture_dir, "auto")
    started_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    recorded = failed = 0
    for i, (ont, iri) in enumerate(missing, 1):
        try:
            recorder.get_term(ont, iri)
            recorded += 1
        except OLS_ERRORS:
            failed += 1
        if i % 25 == 0 or i == len(missing):
            print(f"  {i}/{len(missing)} ({recorded} recorded, {failed} failed)", flush=True)
    seconds = time.monotonic() - started
    files1, bytes1 = _volume(args.fixture_dir)
    report: dict[str, Any] = {
        "corpus_dir": str(args.neon_root.expanduser() / CORPUS_DIR),
        "corpus_sha256": corpus_sha256(args.neon_root.expanduser() / CORPUS_DIR),
        "corpus_files": len(files),
        "curies": len(curies),
        "missing_before": len(missing),
        "recorded": recorded,
        "failed": failed,
        "requests": throttled.requests,
        "retries": throttled.retries,
        "failures": throttled.failures,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "fixtures_before": {"files": files0, "bytes": bytes0},
        "fixtures_after": {"files": files1, "bytes": bytes1},
        "rate": args.rate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"done in {seconds:.0f}s: {recorded} recorded, {failed} failed ({throttled.requests} "
        f"requests, {throttled.retries} retries); fixtures {files1} files, {bytes1 / 1e6:.3f} MB; "
        f"report {args.report}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
