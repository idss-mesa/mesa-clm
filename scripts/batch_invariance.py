#!/usr/bin/env python3
"""Pre-registered batch-invariance experiment for the vLLM encoder route (DESIGN A3).

The M1-A probes found that a text's vector depends on what it is batched with (batched vs one
text per request: min cosine 0.9998659; clm-serve ``/v1/systemone`` vs the local route: max abs
probability difference 0.0434 at CLM's scale 100; ``bench/results/2026-09-29/serving_m1.json``).
This script measures one serving configuration (an *arm*) at a time; the arms are switched by
restarting ``mesa-clm-encoder.service`` (and with it clm-serve) between runs:

* ``B0`` - the current recipe (batched; ``VLLM_BATCH_INVARIANT`` unset, ``--max-num-seqs 8``);
* ``B1`` - kernels: the same plus ``-e VLLM_BATCH_INVARIANT=1`` (vLLM's batch-invariant matmul,
  RMSNorm, softmax and attention split settings);
* ``B2`` - serial: ``VLLM_BATCH_INVARIANT`` unset, ``--max-num-seqs 1``, the local route's
  ``EncoderClient`` at batch 1 and clm-serve's embedder at batch 1.

Measurements of an arm (``measure --arm B?``), with the 20 golden texts
(``.local/serving/encoder_golden.npz``) and the first 100 state-only contexts
(``.local/serving/collapse_contexts.json``) as the 120 texts and the 50 systemone pairs of
``.local/serving/systemone_pairs.json``:

(a) one text per request, sequential; (b) batches of 32; (c) 8 concurrent single-text requests;
(d) a repeat of (a) - max abs element difference and min cosine over every pair of (a)-(d);
(e) clm-serve ``/v1/systemone`` vs the local route (``EncoderClient`` vectors ->
``HeadProjector`` -> ``scale * cos`` -> ``render.answer_from_logits``) on the 50 pairs, max abs
probability difference, for ``clm-latest`` and ``clm-raw``; (f) latency of a 32-text
``/v1/embeddings`` batch and of a cold single rank-fit ``/v1/systemone`` request; (g) the
quickstart and tides drift probes.

``decide`` applies the rule fixed below (:data:`PREREG`, written before any arm ran) to the arm
files and writes ``bench/results/2026-10-01/batch_invariance.json``::

    uv run python scripts/batch_invariance.py measure --arm B0
    uv run python scripts/batch_invariance.py decide

Keys are read from the 0600 files under ``~/.mesa/clm/secrets`` inside this process and never
printed or written; the output is checked for both key values before it is written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import itertools
import json
import re
import shlex
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serving_probes as sp

from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.headproj import HeadProjector
from mesa_clm.clm.http import ClmHttpClient

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-01"
LOCAL = ROOT / ".local/serving"
WORK = LOCAL / "m1b"
GOLDEN = LOCAL / "encoder_golden.npz"
CONTEXTS = LOCAL / "collapse_contexts.json"
PAIRS = LOCAL / "systemone_pairs.json"
OUT = ROOT / f"bench/results/{DATE}/batch_invariance.json"
N_CONTEXTS = 100
BATCH = 32
CONCURRENCY = 8
MODELS = ("clm-latest", "clm-raw")
ARMS = ("B0", "B1", "B2")

# The pre-registration: fixed before the first arm ran (DESIGN A3 cites this block).
PREREG: dict[str, Any] = {
    "texts": "the 20 goldens of .local/serving/encoder_golden.npz, then the first 100 "
    "state-only contexts of .local/serving/collapse_contexts.json (file order)",
    "pairs": "the 50 (state, questions) pairs of .local/serving/systemone_pairs.json",
    "patterns": {
        "a": "one text per request, sequential",
        "b": f"batches of {BATCH} (EncoderClient batch={BATCH})",
        "c": f"{CONCURRENCY} concurrent single-text requests",
        "d": "repeat of (a)",
    },
    "gates": {
        "min_cos_across_patterns": 1 - 1e-6,
        "systemone_max_abs_prob_diff": 1e-4,
        "max_slowdown": 4.0,
    },
    "latency_reference_ms": {
        "source": "bench/results/2026-09-29/serving_m1.json latency (M1-A, p50)",
        "embeddings_batch_32_p50": 2820.0,
        "systemone_cold_p50": 110.1,
    },
    "rule": (
        "ADOPT B1 iff min cosine across (a)-(d) >= 1 - 1e-6 AND (e) <= 1e-4 for both models AND "
        "(f) slowdown <= 4x (both p50s against the M1-A reference). Otherwise ADOPT B2 iff the "
        "same cosine and (e) criteria hold (no slowdown criterion). If neither holds, keep the "
        "current recipe, record the noise floor, and amend the gate scope instead."
    ),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_texts() -> tuple[list[str], list[str]]:
    with np.load(GOLDEN, allow_pickle=False) as npz:
        goldens = [str(t) for t in npz["texts"]]
    contexts = json.loads(CONTEXTS.read_text(encoding="utf-8"))["contexts"][:N_CONTEXTS]
    texts = goldens + [c["text"] for c in contexts]
    kinds = ["golden"] * len(goldens) + ["context"] * len(contexts)
    return texts, kinds


def pair_stats(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Max abs element difference, min/mean cosine and bitwise-equal rows between two
    ``[n, 4096]`` float32 arrays (cosines in float64)."""
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    cos = np.sum(x64 * y64, axis=1) / (np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1))
    return {
        "max_abs_diff": float(np.abs(x64 - y64).max()),
        "min_cos": float(cos.min()),
        "mean_cos": float(cos.mean()),
        "one_minus_min_cos": float(1.0 - cos.min()),
        "bitwise_equal_rows": int(np.sum(np.all(x == y, axis=1))),
        "worst_index": int(np.argmin(cos)),
    }


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def embed_patterns(
    enc_key: str, texts: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """(a)-(d): the same texts through four request patterns."""
    url = sp.ENC_URL
    single = EncoderClient(url, enc_key, batch=BATCH, timeout=300.0, retries=1)
    batched = EncoderClient(url, enc_key, batch=BATCH, timeout=300.0, retries=1)

    def one_by_one() -> np.ndarray:
        return np.stack([single.embed([t])[0][0] for t in texts])

    local = threading.local()

    def worker(text: str) -> np.ndarray:
        if not hasattr(local, "client"):
            local.client = EncoderClient(url, enc_key, batch=1, timeout=300.0, retries=1)
        return local.client.embed([text])[0][0]

    def in_parallel() -> np.ndarray:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return np.stack(list(pool.map(worker, texts)))

    vectors: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    seconds["a"], vectors["a"] = _timed(one_by_one)
    seconds["b"], (vectors["b"], _) = _timed(lambda: batched.embed(list(texts)))
    seconds["c"], vectors["c"] = _timed(in_parallel)
    seconds["d"], vectors["d"] = _timed(one_by_one)
    single.close()
    batched.close()
    return vectors, {k: round(v, 2) for k, v in seconds.items()}


def systemone_parity(
    enc_key: str, clm: ClmHttpClient, head: HeadProjector, local_batch: int
) -> dict[str, Any]:
    """(e): clm-serve against the local route on the 50 pairs, plus the local route against
    itself (its own run-to-run floor)."""
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
    ids = [p["id"] for p in pairs]
    enc = EncoderClient(sp.ENC_URL, enc_key, batch=local_batch, timeout=300.0, retries=1)
    out: dict[str, Any] = {"n_pairs": len(pairs), "local_encoder_batch": local_batch, "gate": 1e-4}
    for model in MODELS:
        served, local_a, local_b = [], [], []
        for p in pairs:
            r = clm.system_one(p["state"], p["questions"], model=model)
            served.append({q: sp._probs(a) for q, a in r.answers.items()})
            a = sp.local_answers(enc, head, p["state"], p["questions"], model)
            b = sp.local_answers(enc, head, p["state"], p["questions"], model)
            local_a.append({q: sp._probs(v) for q, v in a.items()})
            local_b.append({q: sp._probs(v) for q, v in b.items()})
        e2e = sp._diff_stats(served, local_a, ids)
        out[model] = {
            "served_vs_local": {**e2e, "pass": e2e["max_abs_prob_diff"] <= 1e-4},
            "local_vs_local_repeat": sp._diff_stats(local_a, local_b, ids),
        }
    enc.close()
    return out


def container_facts() -> dict[str, Any]:
    """The running encoder's arguments and the non-secret environment that defines the arm
    (values only for the named variables; the key is never read), plus the log lines that show
    whether vLLM's batch-invariant mode is active."""
    names = ("VLLM_BATCH_INVARIANT", "PYTHONPATH", "VLLM_NO_USAGE_STATS")
    fmt = (
        '{{json .Config.Cmd}}\t{{range .Config.Env}}{{$k := index (split . "=") 0}}'
        "{{if eq $k " + " ".join(json.dumps(n) for n in names) + "}}{{.}};{{end}}{{end}}"
    )
    raw = sp.run(
        [
            "sg",
            "docker",
            "-c",
            shlex.join(["docker", "inspect", "mesa-clm-encoder", "--format", fmt]),
        ]
    )
    cmd_json, _, env_text = raw.partition("\t")
    try:
        cmd = json.loads(cmd_json)
    except json.JSONDecodeError:
        cmd = None
    env = dict(e.split("=", 1) for e in env_text.split(";") if "=" in e)
    since = sp._unit_prop("mesa-clm-encoder.service", "ActiveEnterTimestamp")
    log = sp.run(
        [
            "journalctl",
            "--user",
            "-u",
            "mesa-clm-encoder.service",
            "--no-pager",
            "-o",
            "cat",
            "--since",
            " ".join(since.split()[1:3]) if since else "today",
        ]
    )
    pattern = re.compile(
        r"(BATCH_INVARIANT|batch.invariant|cascade attention|matmul_kernel_persistent|"
        r"_rms_norm_kernel|attention backend|FlashAttention version|max_num_seqs|"
        r"max_num_batched_tokens)",
        re.I,
    )
    lines = [ln.split("] ", 1)[-1][:300] for ln in log.splitlines() if pattern.search(ln)]
    return {
        "cmd": cmd,
        "env": env,
        "active_since": since,
        "log_lines": sorted(set(lines))[:40],
    }


def measure(arm: str, repeats: int, out: Path) -> int:
    enc_key, clm_key = sp.read_key("encoder"), sp.read_key("clm")
    texts, kinds = load_texts()
    clm = ClmHttpClient(sp.CLM_URL, clm_key, timeout=300.0, retries=1)
    head = HeadProjector.from_npz(sp.HEAD_NPZ)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    print(f"== {arm}: container", file=sys.stderr)
    facts = container_facts()
    print(f"== {arm}: (a)-(d)", file=sys.stderr)
    vectors, pattern_seconds = embed_patterns(enc_key, texts)
    stats = {
        f"{x}_vs_{y}": pair_stats(vectors[x], vectors[y])
        for x, y in itertools.combinations("abcd", 2)
    }
    min_cos = min(s["min_cos"] for s in stats.values())
    max_diff = max(s["max_abs_diff"] for s in stats.values())
    WORK.mkdir(parents=True, exist_ok=True)
    vec_file = WORK / f"batch_invariance_{arm}_vectors.npz"
    with vec_file.open("wb") as fh:
        np.savez(fh, texts=np.array(texts), kinds=np.array(kinds), **vectors)
    print(f"== {arm}: (e) systemone parity", file=sys.stderr)
    local_batch = 1 if arm == "B2" else BATCH
    parity = systemone_parity(enc_key, clm, head, local_batch)
    print(f"== {arm}: (f) latency", file=sys.stderr)
    enc = EncoderClient(sp.ENC_URL, enc_key, batch=BATCH, timeout=300.0, retries=1)
    latency = sp.probe_latency(enc, clm, repeats)
    enc.close()
    print(f"== {arm}: (g) drift", file=sys.stderr)
    drift = sp.probe_drift(clm)
    ref = PREREG["latency_reference_ms"]
    slowdown = {
        "embeddings_batch_32": round(
            latency["embeddings_batch_32"]["p50_ms"] / ref["embeddings_batch_32_p50"], 3
        ),
        "systemone_cold": round(
            latency["systemone_cold_new_state"]["p50_ms"] / ref["systemone_cold_p50"], 3
        ),
    }
    payload = {
        "format": "mesa-clm/batch-invariance-arm/1",
        "arm": arm,
        "started_at": started,
        "host": "sparky-1",
        "command": f"uv run python scripts/batch_invariance.py measure --arm {arm}",
        "container": facts,
        "texts": {
            "n": len(texts),
            "goldens": kinds.count("golden"),
            "contexts": kinds.count("context"),
            "texts_sha256": sha256_text("\n".join(sha256_text(t) for t in texts)),
        },
        "patterns": {**PREREG["patterns"], "seconds": pattern_seconds},
        "vectors_file": str(vec_file.relative_to(ROOT)),
        "pairwise": stats,
        "across_patterns": {
            "min_cos": min_cos,
            "one_minus_min_cos": 1.0 - min_cos,
            "max_abs_diff": max_diff,
        },
        "systemone": parity,
        "latency": latency,
        "slowdown_vs_m1a": {**slowdown, "max": max(slowdown.values())},
        "drift": drift,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sp.dump_checked(payload, [enc_key, clm_key]), encoding="utf-8")
    clm.close()
    print(out)
    print(
        json.dumps(
            {
                "arm": arm,
                "min_cos": min_cos,
                "max_abs_diff": max_diff,
                "systemone": {m: parity[m]["served_vs_local"]["max_abs_prob_diff"] for m in MODELS},
                "slowdown": payload["slowdown_vs_m1a"],
                "drift": {
                    "quickstart": drift["quickstart"]["pass"],
                    "tides": drift["tides_rank"]["pass"],
                },
            },
            indent=1,
        ),
        file=sys.stderr,
    )
    return 0


def verdict(arm: dict[str, Any], *, slowdown: bool) -> dict[str, Any]:
    gates = PREREG["gates"]
    cos_ok = arm["across_patterns"]["min_cos"] >= gates["min_cos_across_patterns"]
    so = {m: arm["systemone"][m]["served_vs_local"]["max_abs_prob_diff"] for m in MODELS}
    so_ok = all(v <= gates["systemone_max_abs_prob_diff"] for v in so.values())
    out: dict[str, Any] = {
        "min_cos": arm["across_patterns"]["min_cos"],
        "min_cos_pass": cos_ok,
        "systemone_max_abs_prob_diff": so,
        "systemone_pass": so_ok,
    }
    ok = cos_ok and so_ok
    if slowdown:
        out["slowdown"] = arm["slowdown_vs_m1a"]["max"]
        out["slowdown_pass"] = out["slowdown"] <= gates["max_slowdown"]
        ok = ok and out["slowdown_pass"]
    out["adopt"] = ok
    return out


def decide(arm_files: dict[str, Path], out: Path) -> int:
    arms = {
        a: json.loads(p.read_text(encoding="utf-8")) for a, p in arm_files.items() if p.is_file()
    }
    if "B1" not in arms:
        raise SystemExit("decide needs at least the B1 arm file")
    verdicts: dict[str, Any] = {"B1": verdict(arms["B1"], slowdown=True)}
    if verdicts["B1"]["adopt"]:
        adopted = "B1"
    else:
        if "B2" not in arms:
            raise SystemExit("B1 fails the rule: measure B2 before deciding")
        verdicts["B2"] = verdict(arms["B2"], slowdown=False)
        adopted = "B2" if verdicts["B2"]["adopt"] else "none"
    if "B0" in arms:
        verdicts["B0_reference"] = verdict(arms["B0"], slowdown=True)
    # Reported, not gated by the rule above: the drift probes (g), and how far the adopted arm's
    # one-text-per-request vectors moved from the current recipe's (what rotates encoder_fp).
    secondary: dict[str, Any] = {
        "drift_g_not_gated": {
            a: {
                "quickstart_pass": arm["drift"]["quickstart"]["pass"],
                "quickstart_observed": arm["drift"]["quickstart"]["observed"],
                "quickstart_within": arm["drift"]["quickstart"]["within_tolerance_of_issue15"],
                "tides_pass": arm["drift"]["tides_rank"]["pass"],
                "tides_top_prob": arm["drift"]["tides_rank"]["top_prob"],
            }
            for a, arm in arms.items()
        }
    }
    for a in ("B1", "B2"):
        files = [WORK / f"batch_invariance_{x}_vectors.npz" for x in ("B0", a)]
        if a in arms and all(f.is_file() for f in files):
            with np.load(files[0]) as z0, np.load(files[1]) as z1:
                secondary[f"{a}_vs_B0_one_text_per_request"] = pair_stats(z0["a"], z1["a"])
    payload = {
        "format": "mesa-clm/batch-invariance/1",
        "date": DATE,
        "host": "sparky-1",
        "preregistration": PREREG,
        "arms": arms,
        "verdicts": verdicts,
        "adopted": adopted,
        "batch_invariance": {"B1": "kernels", "B2": "serial", "none": "none"}[adopted],
        "secondary": secondary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [sp.read_key("encoder"), sp.read_key("clm")]
    out.write_text(sp.dump_checked(payload, keys), encoding="utf-8")
    print(out)
    print(json.dumps({"adopted": adopted, "verdicts": verdicts}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("measure", help="measure the running configuration as one arm")
    m.add_argument("--arm", choices=ARMS, required=True)
    m.add_argument("--repeats", type=int, default=30)
    m.add_argument("--out", type=Path)
    d = sub.add_parser("decide", help="apply the pre-registered rule to the arm files")
    d.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.mode == "measure":
        out = args.out or WORK / f"batch_invariance_{args.arm}.json"
        return measure(args.arm, args.repeats, out)
    return decide({a: WORK / f"batch_invariance_{a}.json" for a in ARMS}, args.out)


if __name__ == "__main__":
    sys.exit(main())
