#!/usr/bin/env python3
"""Sequential fallback-encoder parity (plan §6.6, K0 step 2; DESIGN D5, D16).

The in-process fallback (``serving/cuda_encoder.py`` ``CudaEncoder``, the recipe
``serving/fallback_serve.py`` serves) may share nothing with the vLLM route until a recorded
parity run shows the two encoders agree. The run is sequential, never concurrent, because both
need the GPU:

1. ``dump`` - with the vLLM encoder and clm-serve **running**, in mesa-clm's environment::

       uv run python scripts/fallback_parity.py dump

   embeds the 20 golden texts (``.local/serving/encoder_golden.npz``) and the first 200 distinct
   state-only contexts of the collapse spike (``.local/serving/collapse_contexts.json``) through
   ``EncoderClient`` and records clm-serve's ``/v1/systemone`` answers (``clm-latest`` and
   ``clm-raw``) for the first 20 pairs of the serving probes
   (``.local/serving/systemone_pairs.json``) -> ``.local/serving/vllm_ref.npz`` and
   ``.local/serving/vllm_ref_systemone.json``;
2. stop both units (``systemctl --user stop mesa-clm-serve.service mesa-clm-encoder.service``);
3. ``compare`` - in the **serve venv** (torch, transformers, the patched CLM; no mesa_clm)::

       HF_HUB_OFFLINE=1 ~/.mesa/clm/serve/.venv/bin/python scripts/fallback_parity.py compare

   loads ``CudaEncoder`` (bf16, left truncation, max_len 4096, offline), embeds the same texts,
   and answers the same 20 pairs through CLM's ``Engine(embedder=CudaEncoder, device="cpu",
   action_cache="512MiB")``; writes ``bench/results/2026-09-29/fallback_parity.json`` with the
   gates (min cosine >= 0.999, mean >= 0.9999) evaluated as pre-registered, never tuned.

The process exits when done, which frees the GPU memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-29"
LOCAL = ROOT / ".local/serving"
GOLDEN = LOCAL / "encoder_golden.npz"
CONTEXTS = LOCAL / "collapse_contexts.json"
PAIRS = LOCAL / "systemone_pairs.json"
REF = LOCAL / "vllm_ref.npz"
REF_SO = LOCAL / "vllm_ref_systemone.json"
OUT = ROOT / f"bench/results/{DATE}/fallback_parity.json"
LOCK = ROOT / "serving/serving.lock.json"
HEAD_PT = Path("~/.mesa/clm/heads/CLM_v0.1-8B.pt").expanduser()
N_CONTEXTS = 200
N_PAIRS = 20
MODELS = ("clm-latest", "clm-raw")
GATE_MIN, GATE_MEAN = 0.999, 0.9999


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encoder_fp(spec: dict[str, Any]) -> str:
    """``mesa_clm.clm.fingerprint.encoder_fp`` without importing mesa_clm: sha256 of the
    canonical JSON (sorted keys, compact separators, non-ASCII kept), first 12 hex."""
    canon = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def meminfo_gib(field: str = "MemAvailable") -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024**2
    raise KeyError(field)


def probabilities(answer: dict[str, Any]) -> dict[str, float]:
    """The vendored ``probabilities_of``: a noul answer becomes ``{false, true}``."""
    if answer.get("type") == "noul":
        p = float(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def reference_texts() -> tuple[list[str], list[str]]:
    """The 20 goldens then the first 200 state-only contexts (collapse-spike order)."""
    with np.load(GOLDEN, allow_pickle=False) as npz:
        goldens = [str(t) for t in npz["texts"]]
    contexts = json.loads(CONTEXTS.read_text(encoding="utf-8"))["contexts"][:N_CONTEXTS]
    texts = goldens + [c["text"] for c in contexts]
    kinds = ["golden"] * len(goldens) + ["context"] * len(contexts)
    return texts, kinds


# -- 1. dump (mesa-clm env, vLLM up) -------------------------------------------------------------


def dump() -> int:
    from mesa_clm.clm.encoder import EncoderClient
    from mesa_clm.clm.http import ClmHttpClient

    secrets = Path("~/.mesa/clm/secrets").expanduser()
    enc = EncoderClient(
        "http://127.0.0.1:8090",
        (secrets / "encoder.key").read_text(encoding="utf-8").strip(),
        timeout=300.0,
    )
    clm = ClmHttpClient(
        "http://127.0.0.1:8700",
        (secrets / "clm.key").read_text(encoding="utf-8").strip(),
        timeout=300.0,
    )
    texts, kinds = reference_texts()
    t0 = time.perf_counter()
    vectors, tokens = enc.embed(texts)
    seconds = time.perf_counter() - t0
    # One text per request as well: batch-1 vLLM vectors are bitwise reproducible (serving
    # probes), batched ones vary at ~1e-4 cosine with the batch composition.
    single = np.stack([enc.embed([t])[0][0] for t in texts])
    s64 = single.astype(np.float64)
    b64 = vectors.astype(np.float64)
    batch_vs_single = np.sum(s64 * b64, axis=1) / (
        np.linalg.norm(s64, axis=1) * np.linalg.norm(b64, axis=1)
    )
    with np.load(GOLDEN, allow_pickle=False) as npz:
        golden_vectors = np.asarray(npz["vectors"], dtype=np.float64)
    n_gold = kinds.count("golden")
    v64 = vectors[:n_gold].astype(np.float64)
    golden_cos = np.sum(v64 * golden_vectors, axis=1) / (
        np.linalg.norm(v64, axis=1) * np.linalg.norm(golden_vectors, axis=1)
    )
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    with REF.open("wb") as fh:
        np.savez(
            fh,
            texts=np.array(texts),
            text_sha256=np.array([sha256_text(t) for t in texts]),
            kinds=np.array(kinds),
            vectors=vectors.astype(np.float32),
            vectors_single=single.astype(np.float32),
            encoder_fp=np.array(lock["encoder_fp"]),
            route=np.array("vllm"),
        )
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"][:N_PAIRS]
    served: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
    for model in MODELS:
        for p in pairs:
            r = clm.system_one(p["state"], p["questions"], model=model)
            served[model].append(
                {
                    qid: {str(k): float(v) for k, v in a.probabilities.items()}
                    for qid, a in r.answers.items()
                }
            )
    REF_SO.write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "encoder_fp": lock["encoder_fp"],
                "route": "vllm",
                "vectors_file": REF.name,
                "vectors_sha256": hashlib.sha256(REF.read_bytes()).hexdigest(),
                "n_texts": len(texts),
                "prompt_tokens": tokens,
                "embed_seconds": round(seconds, 2),
                "vllm_batched_vs_single_request": {
                    "min_cos": float(batch_vs_single.min()),
                    "mean_cos": float(batch_vs_single.mean()),
                },
                "golden_repeat_vs_encoder_golden_npz": {
                    "min_cos": float(golden_cos.min()),
                    "mean_cos": float(golden_cos.mean()),
                },
                "pairs": pairs,
                "served": served,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{REF} ({len(texts)} texts, {tokens} tokens, {seconds:.1f}s); {REF_SO}")
    print(f"golden repeat vs encoder_golden.npz: min cos {golden_cos.min():.7f}")
    return 0


# -- 3. compare (serve venv, vLLM stopped) -------------------------------------------------------


class MemSampler:
    """Samples ``MemAvailable`` every ``interval`` seconds; ``min`` is the low-water mark."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.min = meminfo_gib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.min = min(self.min, meminfo_gib())
            self._stop.wait(self.interval)

    def __enter__(self) -> MemSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def compare(max_len: int, batch: int, diagnostic_batch: int) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(ROOT / "serving"))
    import torch
    import transformers
    from clm.engine import Engine
    from cuda_encoder import MODEL, REVISION, CudaEncoder

    with np.load(REF, allow_pickle=False) as npz:
        texts = [str(t) for t in npz["texts"]]
        kinds = [str(k) for k in npz["kinds"]]
        ref = np.asarray(npz["vectors"], dtype=np.float64)
        ref_single = np.asarray(npz["vectors_single"], dtype=np.float64)
        ref_fp = str(npz["encoder_fp"])
    ref_so = json.loads(REF_SO.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    vllm_spec = dict(lock["encoder"])
    fallback_spec = {**vllm_spec, "route": "transformers", "max_len": max_len}
    if encoder_fp(vllm_spec) != lock["encoder_fp"]:
        raise SystemExit("inline encoder_fp disagrees with serving.lock.json; refusing to record")

    mem_before = meminfo_gib()
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    with MemSampler() as sampler:
        t0 = time.perf_counter()
        enc = CudaEncoder(MODEL, REVISION, max_len=max_len, device="cuda", batch=batch)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        vecs, tokens = enc.embed(texts)
        torch.cuda.synchronize()
        embed_s = time.perf_counter() - t1
        v64 = vecs.astype(np.float64)
        cos = np.sum(v64 * ref, axis=1) / (
            np.linalg.norm(v64, axis=1) * np.linalg.norm(ref, axis=1)
        )
        cos_single = np.sum(v64 * ref_single, axis=1) / (
            np.linalg.norm(v64, axis=1) * np.linalg.norm(ref_single, axis=1)
        )
        t2 = time.perf_counter()
        engine = Engine(embedder=enc, checkpoint=str(HEAD_PT), device="cpu", action_cache="512MiB")
        so: dict[str, Any] = {}
        for model in MODELS:
            diffs: list[float] = []
            agree = 0
            worst = ("", -1.0)
            for p, served in zip(ref_so["pairs"], ref_so["served"][model], strict=True):
                out = engine.answer(p["state"], p["questions"], model=model)["answers"]
                for qid, probs in served.items():
                    local = probabilities(out[qid])
                    d = max(abs(local[k] - probs[k]) for k in probs)
                    diffs.append(d)
                    agree += max(local, key=local.__getitem__) == max(probs, key=probs.__getitem__)
                    if d > worst[1]:
                        worst = (p["id"], d)
            so[model] = {
                "n_questions": len(diffs),
                "max_abs_prob_diff": float(max(diffs)),
                "mean_abs_prob_diff": float(np.mean(diffs)),
                "top_choice_agreement": f"{agree}/{len(diffs)}",
                "top_choice_agreement_rate": agree / len(diffs),
                "worst_pair": worst[0],
            }
        systemone_s = time.perf_counter() - t2
        diagnostic: dict[str, Any] = {}
        if diagnostic_batch:
            # Diagnostic only (the gate stays on the primary run): the same texts with another
            # batch size, to separate right-padding effects from the recipe itself.
            enc.batch = diagnostic_batch
            t3 = time.perf_counter()
            dvecs, _ = enc.embed(texts)
            d64 = dvecs.astype(np.float64)
            dn = np.linalg.norm(d64, axis=1)
            dcos = np.sum(d64 * ref, axis=1) / (dn * np.linalg.norm(ref, axis=1))
            dcos_single = np.sum(d64 * ref_single, axis=1) / (
                dn * np.linalg.norm(ref_single, axis=1)
            )
            dself = np.sum(d64 * v64, axis=1) / (dn * np.linalg.norm(v64, axis=1))
            diagnostic = {
                "batch": diagnostic_batch,
                "seconds": round(time.perf_counter() - t3, 2),
                "vs_batched_reference": {
                    "min_cos": float(dcos.min()),
                    "mean_cos": float(dcos.mean()),
                },
                "vs_single_request_reference": {
                    "min_cos": float(dcos_single.min()),
                    "mean_cos": float(dcos_single.mean()),
                },
                "vs_primary_fallback_run": {
                    "min_cos": float(dself.min()),
                    "mean_cos": float(dself.mean()),
                },
                "worst_text_sha256": sha256_text(texts[int(np.argmin(dcos))]),
            }
            enc.batch = batch
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    per_kind = {}
    for kind in ("golden", "context"):
        sel = np.asarray([k == kind for k in kinds])
        per_kind[kind] = {
            "n": int(sel.sum()),
            "min_cos": float(cos[sel].min()),
            "mean_cos": float(cos[sel].mean()),
        }
    worst_i = int(np.argmin(cos))
    gates = {
        "min_cos": {
            "value": float(cos.min()),
            "gate": GATE_MIN,
            "pass": bool(cos.min() >= GATE_MIN),
        },
        "mean_cos": {
            "value": float(cos.mean()),
            "gate": GATE_MEAN,
            "pass": bool(cos.mean() >= GATE_MEAN),
        },
    }
    payload = {
        "format": "mesa-clm/fallback-parity/1",
        "started_at": started,
        "host": "sparky-1",
        "commands": [
            "uv run python scripts/fallback_parity.py dump",
            "systemctl --user stop mesa-clm-serve.service mesa-clm-encoder.service",
            "HF_HUB_OFFLINE=1 ~/.mesa/clm/serve/.venv/bin/python scripts/fallback_parity.py compare"
            f" --batch {batch} --diagnostic-batch {diagnostic_batch}",
        ],
        "reference_route": {
            "route": "vllm",
            "encoder_fp": ref_fp,
            "spec": vllm_spec,
            "vectors_sha256": ref_so["vectors_sha256"],
            "golden_repeat_vs_encoder_golden_npz": ref_so["golden_repeat_vs_encoder_golden_npz"],
        },
        "fallback_route": {
            "route": "transformers",
            "encoder_fp": encoder_fp(fallback_spec),
            "spec": fallback_spec,
            "class": "serving/cuda_encoder.py CudaEncoder",
            "device": "cuda",
            "batch": batch,
            "triton_deregistered": enc.triton_deregistered,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "texts": {
            "n": len(texts),
            "goldens": kinds.count("golden"),
            "contexts": kinds.count("context"),
            "tokens": tokens,
        },
        "encoder_parity": {
            **gates,
            "pass": gates["min_cos"]["pass"] and gates["mean_cos"]["pass"],
            "per_kind": per_kind,
            "worst_text_sha256": sha256_text(texts[worst_i]),
            "worst_kind": kinds[worst_i],
            "n_below_0_9999": int((cos < 0.9999).sum()),
            "reference": "vLLM vectors from EncoderClient.embed in batches of 32 (the dump)",
            "vs_single_request_reference": {
                "min_cos": float(cos_single.min()),
                "mean_cos": float(cos_single.mean()),
                "min_pass": bool(cos_single.min() >= GATE_MIN),
                "mean_pass": bool(cos_single.mean() >= GATE_MEAN),
            },
            "vllm_batched_vs_single_request": ref_so["vllm_batched_vs_single_request"],
        },
        "diagnostic_other_batch": diagnostic or None,
        "systemone_parity": {
            "n_pairs": len(ref_so["pairs"]),
            "engine": "clm.engine.Engine(embedder=CudaEncoder, device='cpu', action_cache='512MiB')",
            "head": str(HEAD_PT.name),
            **so,
        },
        "timing_s": {
            "model_load": round(load_s, 1),
            "embed_all": round(embed_s, 2),
            "embed_ms_per_text": round(1000 * embed_s / len(texts), 1),
            "systemone_all_pairs_both_models": round(systemone_s, 2),
        },
        "memory_gib": {
            "memavailable_before": round(mem_before, 2),
            "memavailable_low_water": round(sampler.min, 2),
            "peak_footprint_memavailable_delta": round(mem_before - sampler.min, 2),
            "torch_cuda_max_allocated": round(peak_alloc, 2),
            "torch_cuda_max_reserved": round(peak_reserved, 2),
            "process_max_rss": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)
    print(json.dumps({"encoder_parity": payload["encoder_parity"], "systemone": so}, indent=1))
    return 0


# -- head-only check (serve venv, CPU; called by scripts/serving_probes.py) ----------------------


def head_check(vectors: Path) -> int:
    """Encoder vectors recorded by the serving probes through CLM's own ``HeadPair`` on CPU
    (``torch.load(..., weights_only=True)`` after patch 0003) and the engine's scoring
    (``za @ zs`` in float32 torch, ``scale * cos``, the vendored softmax); prints JSON with the
    largest differences from mesa-clm's numpy ``HeadProjector`` route."""
    import torch
    from clm.heads import HeadPair
    from clm.schema import softmax

    pair = HeadPair("clm-latest", str(HEAD_PT), "cpu")
    pair.ensure()
    with np.load(vectors, allow_pickle=False) as z:
        states = np.ascontiguousarray(z["states"], dtype=np.float32)
        actions = np.ascontiguousarray(z["actions"], dtype=np.float32)
        offsets = [int(o) for o in z["offsets"]]
        zs_np, za_np, probs_np = z["zs"], z["za"], z["probs"]
    zs = pair.project_states(states)
    za = pair.project_actions(actions)
    probs: list[float] = []
    for i in range(len(offsets) - 1):
        cos = za[offsets[i] : offsets[i + 1]] @ zs[i]
        probs.extend(softmax((pair.scale * cos / 1.0).tolist()))
    proj_diff = max(
        float(np.abs(zs.cpu().numpy().astype(np.float64) - zs_np).max()),
        float(np.abs(za.cpu().numpy().astype(np.float64) - za_np).max()),
    )
    prob_diff = float(np.abs(np.asarray(probs) - probs_np).max())
    print(
        json.dumps(
            {
                "n_questions": len(offsets) - 1,
                "n_candidates": offsets[-1],
                "torch": torch.__version__,
                "weights_only_load_ok": True,
                "headpair_scale": pair.scale,
                "max_abs_projection_diff": proj_diff,
                "max_abs_prob_diff": prob_diff,
                "projection_gate": 1e-5,
                "projection_pass": proj_diff <= 1e-5,
                "prob_pass_1e-4": prob_diff <= 1e-4,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["dump", "compare", "head"])
    ap.add_argument("--vectors", type=Path, default=LOCAL / "parity_vectors.npz")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--diagnostic-batch", type=int, default=1, help="0 disables the diagnostic re-embed"
    )
    args = ap.parse_args(argv)
    if args.mode == "dump":
        return dump()
    if args.mode == "head":
        return head_check(args.vectors)
    return compare(args.max_len, args.batch, args.diagnostic_batch)


if __name__ == "__main__":
    sys.exit(main())
