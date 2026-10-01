#!/usr/bin/env python3
"""Per-card annotate latency through the live stack (plan §8 M1-A "p50/p95 per card"; the M3
latency budget is fixed by amendment from these numbers).

Runs ``Annotator.annotate`` at tier ``zero_shot`` through the real provider (the serving lock the
host runs, the keyed pre-flight, the container check) on **non-bench** SRER cards only (DESIGN,
"G1 freeze": no live run looks at a bench card before the M2 cells exist): plan §9's smoke card
``DP1.00004.001.BP_30min`` plus the first others of ``~/neon-ducklake/sites/SRER/cards/anyjev`` in
``sha256(name)`` order, skipping the bench products DP1.10003.001 and DP1.10022.001. Per card:

* **cold**: the first run after the units' restart, OLS live from EMBL-EBI (recorded into a
  scratch fixture directory); clm-serve's caches start empty, but the texts every card shares
  (the closed options, the anchors) are cached by the first card's run;
* **warm**: ``--repeats`` more runs with OLS replayed from that recording (no network), so they
  time the decide phase against warm clm-serve caches.

Each run records its own ``seconds`` (the pipeline's clock), the wall clock around the call, the
CLM calls, their summed latency and the encoder tokens on clm-serve's cache misses; nothing about
the outcomes. Runs go to a scratch sidecar (``--sidecar``), never the default one.

    uv run python scripts/annotate_latency.py --out bench/results/<date>/annotate_latency.json

Keys are read through the configuration only and never printed or written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mesa_clm.cards import load_card
from mesa_clm.config import load_config
from mesa_clm.ols import OLSLayer, RecordingOLS
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers import live
from mesa_clm.service import DecisionService

ROOT = Path(__file__).resolve().parents[1]
SRER = Path("~/neon-ducklake/sites/SRER/cards/anyjev").expanduser()
SMOKE_CARD = "DP1.00004.001.BP_30min"
BENCH_PRODUCTS = ("DP1.10003.001", "DP1.10022.001")


def pick_cards(n: int) -> list[Path]:
    """Plan §9's smoke card, then the others in sha256(name) order, bench products skipped."""
    names = sorted(p.stem for p in SRER.glob("*.md") if not p.stem.startswith(BENCH_PRODUCTS))
    rest = sorted(
        (c for c in names if c != SMOKE_CARD),
        key=lambda c: hashlib.sha256(c.encode()).hexdigest(),
    )
    chosen = ([SMOKE_CARD] if SMOKE_CARD in names else []) + rest
    return [SRER / f"{c}.md" for c in chosen[:n]]


def pct(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(arr.max()), 4),
    }


def one_run(svc: DecisionService, card: Any, store: DuckDBStore) -> dict[str, Any]:
    t0 = time.perf_counter()
    run = svc.annotate(card, "latency-probe", owner="latency-probe", tier="zero_shot")
    wall = time.perf_counter() - t0
    calls = store.clm_calls(run.run_id)
    clm_ms = [float(c["latency_ms"]) for c in calls if c.get("latency_ms") is not None]
    return {
        "seconds": round(float(run.seconds), 4),
        "wall_seconds": round(wall, 4),
        "clm_calls": int(run.n_calls),
        "failed_calls": int(run.n_failed_calls),
        "clm_latency_ms_sum": round(sum(clm_ms), 2),
        "encoder_tokens": int(run.input_tokens),
        "decisions": int(run.n_decisions),
        "degraded": bool(run.degraded),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cards", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, default=ROOT / ".local/latency/prov.duckdb")
    ap.add_argument("--ols-dir", type=Path, default=ROOT / ".local/latency/ols")
    args = ap.parse_args(argv)

    cfg = load_config(None)
    stack = live.clm_provider(cfg)
    secrets = [cfg.clm.resolved_api_key() or "", cfg.encoder.resolved_api_key() or ""]
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    cards = pick_cards(args.cards)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    store = DuckDBStore(args.sidecar)
    try:
        answering, detail = live.preflight(stack, cfg)
        if not answering:
            raise SystemExit(f"the serving pair does not answer: {detail}")
        container = live.container_check(stack.lock)
        if not container.ok:
            raise SystemExit(container.note())
        from mesa_mcp.ols.client import OLSClient

        record = OLSLayer(
            RecordingOLS(OLSClient(cfg.ols.base_url), args.ols_dir, "record"),
            max_candidates=cfg.policy.max_candidates,
        )
        replay = OLSLayer(
            RecordingOLS(None, args.ols_dir, "replay"), max_candidates=cfg.policy.max_candidates
        )
        cold_svc = DecisionService.from_config(cfg, stack.provider, store=store, ols=record)
        warm_svc = DecisionService.from_config(cfg, stack.provider, store=store, ols=replay)
        per_card: dict[str, Any] = {}
        for path in cards:
            card = load_card(path)
            print(f"== {path.stem}", file=sys.stderr)
            cold = one_run(cold_svc, card, store)
            warm = [one_run(warm_svc, card, store) for _ in range(args.repeats)]
            per_card[path.stem] = {
                "cold": cold,
                "warm": {
                    "seconds": pct([w["seconds"] for w in warm]),
                    "wall_seconds": pct([w["wall_seconds"] for w in warm]),
                    "clm_latency_ms_sum": pct([w["clm_latency_ms_sum"] for w in warm]),
                    "clm_calls": sorted({w["clm_calls"] for w in warm}),
                    "encoder_tokens": sorted({w["encoder_tokens"] for w in warm}),
                    "failed_calls": sum(w["failed_calls"] for w in warm),
                    "degraded": any(w["degraded"] for w in warm),
                },
            }
    finally:
        stack.close()
    cold_all = [c["cold"]["seconds"] for c in per_card.values()]
    warm_p50 = [c["warm"]["seconds"]["p50"] for c in per_card.values()]
    warm_p95 = [c["warm"]["seconds"]["p95"] for c in per_card.values()]
    payload = {
        "format": "mesa-clm/annotate-latency/1",
        "started_at": started,
        "host": "sparky-1",
        "command": "uv run python scripts/annotate_latency.py --out " + str(args.out),
        "note": "Diagnostics: zero_shot (uncalibrated, never auto), non-bench SRER cards only; "
        "outcomes are not recorded. cold = the first run of a card after the units' restart, "
        "OLS live from EMBL-EBI; warm = repeats with OLS replayed from that recording.",
        "serving": {
            "lock_sha": stack.lock.lock_sha,
            "encoder_fp": stack.lock.encoder_fp,
            "preflight": detail,
            "container": container.note(),
        },
        "tier": "zero_shot",
        "cards": [p.stem for p in cards],
        "repeats": args.repeats,
        "per_card": per_card,
        "across_cards": {
            "cold_seconds": pct(cold_all),
            "warm_p50_seconds": pct(warm_p50),
            "warm_p95_seconds": pct(warm_p95),
        },
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    if any(s and s in text for s in secrets):
        raise SystemExit("refusing to write: a key value appears in the output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(payload["across_cards"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
