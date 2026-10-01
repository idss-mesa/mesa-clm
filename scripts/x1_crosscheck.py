#!/usr/bin/env python3
"""X1's pre-registered cross-check: offline scores from the feature store against clm-serve.

Plan §5.6 (DESIGN, "X1 framing A/B") scores every framing offline from the cache, ``s_c =
100·(zs·zc − zs·za)``, and asks for a "200-pair cross-check vs clm-serve ≤1e-4 in probs". This
script draws 200 (context, candidates + anchor) groups of the F4/F7/F9 arms of a frozen labels
snapshot deterministically and, for ``clm-latest`` and ``clm-raw``, compares per group:

* **store**: :meth:`mesa_clm.learn.offline.OfflineScorer.rank_fit_texts` over the feature store
  of the live lock (float32 vectors, format 2; the store is opened with the lock's vector recipe
  and refuses another), the route X1 uses;
* **served**: clm-serve's ``/v1/systemone`` answer to the same request, built with
  ``framings.build_request`` exactly as ``features.manifest`` builds it (the script checks that
  the request renders the group's manifest texts);
* **rank**: clm-serve's ``/v1/rank`` over the same context, instructions and candidate texts
  (plan §9's "s_c vs /v1/rank": ``s_c = ln p_c − ln p_anchor`` from the ranked probabilities);
* **fp16**: the store's vectors cast to float16 and back before scoring, which is what a
  format-1 store held (``bench/results/2026-10-01/features_build.json#/rerun/crosscheck``), kept
  as the record of why the store keeps float32.

It also re-embeds every drawn text once, one text per request, and reports how many fresh
float32 vectors equal the stored ones bit for bit. A 50-pair F1 noul supplement uses the same draw
rule. Draw: per (task, framing) stratum a share of the groups proportional to the stratum's size
(largest remainder), groups ordered by ``sha256("task|framing|target_sha256")``, first n taken.

    uv run python scripts/x1_crosscheck.py --out bench/results/<date>/x1_crosscheck.json

Keys are read through the configuration (the default key files) inside this process only and
are never printed, logged or written; the output is checked for both before it is written.
Diagnostics, not a bench cell: no label is read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mesa_clm import framings, render
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.fingerprint import clm_model_fp, load_serving_lock
from mesa_clm.clm.headproj import HeadProjector, l2
from mesa_clm.clm.http import ChoiceAnswer, ClmHttpClient, NoulAnswer, question_to_dict
from mesa_clm.config import load_config
from mesa_clm.learn import features as feat
from mesa_clm.learn.offline import OfflineScorer, RankFitScores
from mesa_clm.providers.live import live_lock_path
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.serving import HEADS_DIR, serving_home
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
ARMS = ("F4", "F7", "F9")
MODELS = ("clm-latest", "clm-raw")
GATE = 1e-4  # plan §5.6: in probability


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allocate(sizes: Mapping[Any, int], total: int) -> dict[Any, int]:
    """Largest-remainder proportional allocation of ``total`` over the strata (ties: stratum
    order)."""
    n = sum(sizes.values())
    quotas = {k: total * v / n for k, v in sizes.items()}
    out = {k: math.floor(q) for k, q in quotas.items()}
    left = total - sum(out.values())
    order = sorted(sizes, key=lambda k: (-(quotas[k] - out[k]), list(sizes).index(k)))
    for k in order[:left]:
        out[k] += 1
    return out


def s_c_from_probs(probs: Mapping[str, float]) -> dict[str, float]:
    la = math.log(probs[ANCHOR_KEY])
    return {k: math.log(p) - la for k, p in probs.items() if k != ANCHOR_KEY}


def diff_stats(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Over the groups: max / mean / p95 of each group's max |p_a − p_b|, top-choice agreement,
    the worst group and the max |Δ s_c| over every candidate."""
    diffs: list[float] = []
    ds: list[float] = []
    agree = 0
    worst = ("", -1.0)
    for r in rows:
        pa, pb = r["probs"][a], r["probs"][b]
        d = max(abs(pa[k] - pb[k]) for k in pa)
        diffs.append(d)
        agree += max(pa, key=pa.__getitem__) == max(pb, key=pb.__getitem__)
        if d > worst[1]:
            worst = (r["id"], d)
        if "s_c" in r:
            ds.extend(abs(r["s_c"][a][k] - r["s_c"][b][k]) for k in r["s_c"][a])
    out: dict[str, Any] = {
        "max_abs_prob_diff": float(max(diffs)),
        "mean_abs_prob_diff": float(np.mean(diffs)),
        "p95_abs_prob_diff": float(np.percentile(diffs, 95)),
        "top_choice_agreement": f"{agree}/{len(diffs)}",
        "worst_group": worst[0],
        "pass": bool(max(diffs) <= GATE),
    }
    if ds:
        out["max_abs_s_c_diff"] = float(max(ds))
    return out


def fp16_route(scorer: OfflineScorer, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(state side, action side)`` projections of float32 vectors rounded through float16, the
    vectors a format-1 store held (projected afresh, not from the store's cache)."""
    x16 = np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)
    if scorer.projector is None:
        return l2(x16), l2(x16)
    return scorer.projector.project_states(x16), scorer.projector.project_actions(x16)


def groups_of(snapshot: Path) -> tuple[dict[tuple[str, str, str], Any], dict[Any, Any], Any]:
    """The F4/F7/F9 groups of the manifest (context, candidates, anchor) with the request each
    was built from, checked to render exactly the manifest texts; and the snapshot pairs."""
    man = feat.manifest(snapshot)
    by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in man.rows:
        if r.framing_id not in ARMS:
            continue
        g = by_group.setdefault(
            (r.task_id, r.framing_id, r.target_sha256), {"candidates": {}, "context": None}
        )
        if r.role == "context":
            g["context"] = r.text
        elif r.role == "anchor":
            g["anchor"] = r.text
        else:
            g["candidates"][r.option_key] = r.text
    pairs = feat._snapshot_pairs(snapshot, feat.X1_TASKS)
    requests: dict[tuple[str, str, str], Any] = {}
    for task_id in feat.X1_TASKS:
        targets: dict[str, list[Any]] = {}
        for p in pairs[task_id]:
            targets.setdefault(p.target_sha256, []).append(p)
        scope = TASKS[task_id].scope
        for fid in ARMS:
            f = framings.framing(task_id, fid)
            for target, group in targets.items():
                states = [{**p.state, "scope": p.state.get("scope") or scope} for p in group]
                cands = [feat._candidate(task_id, p) for p in group]
                state, questions = framings.build_request(f, states[0], cands)
                stext, keys, texts = render.build_pairs(state, questions)[task_id]
                g = by_group[(task_id, fid, target)]
                if (
                    stext != g["context"]
                    or keys[-1] != ANCHOR_KEY
                    or texts[-1] != g["anchor"]
                    or dict(zip(keys[:-1], texts[:-1], strict=True)) != g["candidates"]
                ):
                    raise SystemExit(f"{task_id}/{fid}/{target[:12]}: request != manifest texts")
                requests[(task_id, fid, target)] = (state, questions)
    return {k: {**v, "request": requests[k]} for k, v in by_group.items()}, pairs, man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--groups", type=int, default=200)
    ap.add_argument("--noul", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = load_config(None)
    lock_path = live_lock_path()
    lock = load_serving_lock(lock_path)
    store = feat.FeatureStore.for_lock(cfg, lock)  # refuses another vector recipe or format
    meta = store.meta()
    fp_latest = clm_model_fp(lock.model_spec("clm-latest"))
    head = HeadProjector.from_npz(
        serving_home() / HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    )
    if head.source_sha256 != lock.head.sha256:
        raise SystemExit("the head export is not the lock's head")
    scorers = {
        "clm-latest": OfflineScorer("clm-latest", head, clm_model_fp=fp_latest),
        "clm-raw": OfflineScorer("clm-raw"),
    }
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    t0 = time.monotonic()

    by_group, pairs, man = groups_of(args.snapshot)
    strata: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in by_group:
        strata[(key[0], key[1])].append(key)
    sizes = {s: len(v) for s, v in strata.items()}
    alloc = allocate(sizes, args.groups)
    drawn: list[tuple[str, str, str]] = []
    for s, keys_ in strata.items():
        ordered = sorted(keys_, key=lambda k: sha256_text(f"{k[0]}|{k[1]}|{k[2]}"))
        drawn.extend(ordered[: alloc[s]])

    enc = EncoderClient.from_config(cfg.encoder, allow_remote=cfg.clm.allow_remote)
    enc.batch = 1
    clm = ClmHttpClient.from_config(cfg.clm)
    secrets = [cfg.clm.resolved_api_key() or "", cfg.encoder.resolved_api_key() or ""]
    try:
        texts: list[str] = []
        for k in drawn:
            g = by_group[k]
            texts.extend([g["context"], *g["candidates"].values(), g["anchor"]])
        distinct = list(dict.fromkeys(texts))
        stored = store.get(distinct)
        t_fresh = time.monotonic()
        fresh = np.stack([enc.embed([t])[0][0] for t in distinct])
        t_fresh = time.monotonic() - t_fresh
        fresh_unit = l2(fresh)
        bitwise = int(np.sum(np.all(fresh_unit == stored, axis=1)))
        vec_of = dict(zip(distinct, stored, strict=True))

        rows: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
        served_ms: dict[str, list[float]] = {m: [] for m in MODELS}
        served_tokens: dict[str, int] = dict.fromkeys(MODELS, 0)
        for model in MODELS:
            sc = scorers[model]
            for k in drawn:
                g = by_group[k]
                cand: dict[str, str] = g["candidates"]
                gid = f"{k[0]}/{k[1]}/{k[2][:12]}"
                off: RankFitScores = sc.rank_fit_texts(store, g["context"], cand, g["anchor"])
                xs = np.stack([vec_of[g["context"]]])
                xa = np.stack([vec_of[t] for t in [*cand.values(), g["anchor"]]])
                zs16, _ = fp16_route(sc, xs)
                _, za16 = fp16_route(sc, xa)
                f16 = sc.rank_fit(list(cand), zs16[0], za16[:-1], za16[-1])
                state, questions = g["request"]
                resp = clm.system_one(state, questions, model=model)
                answer = resp.answers[k[0]]
                if not isinstance(answer, ChoiceAnswer):
                    raise SystemExit(f"{gid}: /v1/systemone did not answer a choice")
                served = {str(a): float(b) for a, b in answer.probabilities.items()}
                if set(served) != set(off.probabilities):
                    raise SystemExit(f"{gid}: served keys differ")
                if resp.latency_ms is not None:
                    served_ms[model].append(resp.latency_ms)
                served_tokens[model] += int(resp.usage.input_tokens or 0)
                q = question_to_dict(questions[k[0]])
                criteria: dict[str, str] = dict(q["criteria"])
                answers = list(criteria.values())
                if len(set(answers)) != len(answers):
                    raise SystemExit(f"{gid}: duplicate candidate texts; /v1/rank cannot key them")
                ranked = clm.rank(state, q.get("instructions"), answers, model=model)
                by_text = {r.candidate: float(r.prob) for r in ranked}
                rank_probs = {key: by_text[text] for key, text in criteria.items()}
                rows[model].append(
                    {
                        "id": gid,
                        "n_candidates": len(cand),
                        "probs": {
                            "store": off.probabilities,
                            "served": served,
                            "rank": rank_probs,
                            "fp16": f16.probabilities,
                        },
                        "s_c": {
                            "store": off.s_c,
                            "served": s_c_from_probs(served),
                            "rank": s_c_from_probs(rank_probs),
                            "fp16": f16.s_c,
                        },
                        "winner": {"store": off.winner, "served": str(answer.choice)},
                    }
                )

        noul_alloc = allocate({t: len(pairs[t]) for t in feat.X1_TASKS}, args.noul)
        noul_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
        noul_texts: list[str] = []
        noul_items: list[tuple[str, Any, Any, dict[str, str]]] = []
        for task_id in feat.X1_TASKS:
            f = framings.framing(task_id, "F1")
            ordered = sorted(
                pairs[task_id],
                key=lambda p, t=task_id: sha256_text(f"{t}|F1|{p.target_sha256}|{p.option_key}"),
            )
            for p in ordered[: noul_alloc[task_id]]:
                state = framings.build_context(f, p.state)
                questions = {p.option_key: framings.noul_question(f)}
                stext, keys, ntexts = render.build_pairs(state, questions)[p.option_key]
                noul = dict(zip(keys, ntexts, strict=True))
                noul_items.append(
                    (f"{task_id}/F1/{p.target_sha256[:12]}", p, (state, questions), noul)
                )
                noul_texts.extend([stext, noul["false"], noul["true"]])
        missing = store.missing(noul_texts)
        if missing:
            raise SystemExit(f"{len(missing)} F1 texts are not in the store; run features build")
        for gid, p, (state, questions), noul in noul_items:
            stext = render.build_pairs(state, questions)[p.option_key][0]
            for model in MODELS:
                off_n = scorers[model].noul_texts(store, stext, noul["false"], noul["true"])
                resp = clm.system_one(state, questions, model=model)
                answer = resp.answers[p.option_key]
                if not isinstance(answer, NoulAnswer):
                    raise SystemExit(f"{gid}: /v1/systemone did not answer a noul")
                served_p = float(answer.noul)
                noul_rows[model].append(
                    {
                        "id": f"{gid}/{p.option_key}",
                        "probs": {
                            "store": {"true": off_n.p_true, "false": 1.0 - off_n.p_true},
                            "served": {"true": served_p, "false": 1.0 - served_p},
                        },
                    }
                )
    finally:
        enc.close()
        clm.close()

    composition: dict[str, int] = defaultdict(int)
    for k in drawn:
        composition[f"{k[0]}/{k[1]}"] += 1
    results = {
        m: {
            "store_vs_served": diff_stats(rows[m], "store", "served"),
            "store_vs_rank": diff_stats(rows[m], "store", "rank"),
            "rank_vs_served": diff_stats(rows[m], "rank", "served"),
            "fp16_vs_served": diff_stats(rows[m], "fp16", "served"),
        }
        for m in MODELS
    }
    payload: dict[str, Any] = {
        "format": "mesa-clm/x1-crosscheck/1",
        "started_at": started,
        "seconds": round(time.monotonic() - t0, 1),
        "host": "sparky-1",
        "command": "uv run python scripts/x1_crosscheck.py --out " + str(args.out),
        "what": "plan §5.6 X1: offline rank_fit scores from the float32 feature store vs "
        "clm-serve /v1/systemone (and /v1/rank) <= 1e-4 in probability; diagnostics, no label "
        "read",
        "snapshot": str(args.snapshot.relative_to(ROOT))
        if args.snapshot.is_relative_to(ROOT)
        else str(args.snapshot),
        "labels_sha256": man.labels_sha256,
        "serving": {
            "lock": str(lock_path).replace(str(Path.home()), "~"),
            "lock_sha": lock.lock_sha,
            "encoder_fp": lock.encoder_fp,
            "vector_recipe_sha256": lock.vector_recipe_sha256(),
        },
        "store": {
            "path": str(store.path).replace(str(Path.home()), "~"),
            "format": meta.get("format"),
            "vector_recipe_sha256": meta.get("vector_recipe_sha256"),
            "serving_lock_sha": meta.get("serving_lock_sha"),
        },
        "clm_model_fp": {m: clm_model_fp(lock.model_spec(m)) for m in MODELS},
        "draw": {
            "rule": "per (task, framing) stratum of the F4/F7/F9 groups, a share proportional to "
            "the stratum's group count (largest remainder), groups ordered by "
            "sha256('task|framing|target_sha256'), first n taken",
            "strata_sizes": {f"{a}/{b}": n for (a, b), n in sizes.items()},
            "drawn": dict(composition),
            "groups": len(drawn),
            "options": int(sum(len(by_group[k]["candidates"]) + 1 for k in drawn)),
            "distinct_texts": len(distinct),
        },
        "gate": GATE,
        "fresh_vectors": {
            "texts": len(distinct),
            "seconds": round(t_fresh, 2),
            "bitwise_equal_to_store": f"{bitwise}/{len(distinct)}",
        },
        "served": {
            m: {
                "calls_systemone": len(rows[m]),
                "calls_rank": len(rows[m]),
                "input_tokens_systemone": served_tokens[m],
                "latency_ms_p50": float(np.percentile(served_ms[m], 50)) if served_ms[m] else None,
                "latency_ms_p95": float(np.percentile(served_ms[m], 95)) if served_ms[m] else None,
            }
            for m in MODELS
        },
        "results": results,
        "noul_supplement": {
            "rule": "F1 pairs per task, proportional to the pair counts, ordered by "
            "sha256('task|F1|target_sha256|option_key'), first n",
            "pairs": {t: noul_alloc[t] for t in feat.X1_TASKS},
            **{m: diff_stats(noul_rows[m], "store", "served") for m in MODELS},
        },
        "verdict": {
            "gate": GATE,
            "store_vs_served_pass": all(results[m]["store_vs_served"]["pass"] for m in MODELS),
            "store_vs_rank_pass": all(results[m]["store_vs_rank"]["pass"] for m in MODELS),
            "noul_store_vs_served_pass": all(
                diff_stats(noul_rows[m], "store", "served")["pass"] for m in MODELS
            ),
            "fp16_vs_served_pass": all(results[m]["fp16_vs_served"]["pass"] for m in MODELS),
        },
        "groups": {
            m: [
                {
                    "id": r["id"],
                    "n_candidates": r["n_candidates"],
                    "max_abs_prob_diff": {
                        pair: max(
                            abs(r["probs"][pair.split("_vs_")[0]][key] - r["probs"]["served"][key])
                            for key in r["probs"]["served"]
                        )
                        for pair in ("store_vs_served", "rank_vs_served", "fp16_vs_served")
                    },
                    "winner": r["winner"],
                }
                for r in rows[m]
            ]
            for m in MODELS
        },
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=float)
    if any(s and s in text for s in secrets):
        raise SystemExit("refusing to write: a key value appears in the output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("fresh_vectors", "results", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
