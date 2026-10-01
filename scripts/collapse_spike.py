#!/usr/bin/env python3
"""Label-free collapse spike (plan §8 M1-A, the first task once the encoder is up; DESIGN D23).

Issue #15 reported that CLM's last-token state vectors collapse when the question is appended:
mean cosine between *different* states 0.958 with the question suffix, 0.452 for the state alone
(raw 4096-d), 0.947 after the head. This script measures the same thing on mesa-clm's own
contexts, without looking at a single label::

    uv run python scripts/collapse_spike.py      # -> bench/results/2026-10-01/collapse_spike.json

For every ``term.fits`` (285) and ``column.ontology_fits`` (190) row of the committed label
snapshot the stored ``state_json`` is put back in the builders' key order
(``learn.features.builder_ordered``: the snapshot stores canonical JSON with sorted keys at every
depth, which is identity only, plan §4.3), projected onto its ``target_state`` view
(``states.target_state_from``; an ``ontology_state`` has no ``scope`` and gets the task's
``column``) and rendered four ways:

* ``state_only`` - ``render.to_text(view)`` (the production F7 context);
* ``suffixed`` - ``render.state_text(view, TASKS[task].text)``: the anyjev question appended as
  CLM appends ``instructions``;
* ``f9_query`` - the F9 short query template (``framings``), a string context;
* ``control_f1`` - the F1 control context: the stored anyjev per-candidate state
  (``candidate_state`` / ``ontology_state``) with the question appended, one per row.

Rows sharing a target share its context, so every statistic is over the *distinct* texts of a
(task, variant): mean pairwise cosine (i < j) in the raw 4096-d encoder space and after the
released state head (512-d), the within-card and across-card means, and the fraction of the
(centred) variance on the top principal component. Only ``labels_sha256`` of the snapshot file
and row counts are recorded from the label side. The distinct state-only contexts are written to
``.local/serving/collapse_contexts_<date>.json``.

The first run (``bench/results/2026-09-29/collapse_spike.json``, ``encoder_fp`` 852efc921a8a)
rendered the sorted-key ``state_json`` directly, so its nested objects (the card header first of
all, which then began with ``columns:``) were in an order the pipeline never sends; its state-only
contexts (``.local/serving/collapse_contexts.json``) stay the fixed inputs of the fallback parity
and batch-invariance runs, which compare encoder routes on the same texts and do not depend on
the rendering. ``--sorted-keys`` reproduces that first rendering.
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

import duckdb
import numpy as np

from mesa_clm import framings, render
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.headproj import HeadProjector
from mesa_clm.learn.features import builder_ordered
from mesa_clm.states import target_state_from
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-01"
SNAPSHOT = ROOT / "bench/snapshots/2026-09-29.parquet"
LOCK = ROOT / "serving/serving.lock.json"
ENC_URL = "http://127.0.0.1:8090"
KEY_FILE = Path("~/.mesa/clm/secrets/encoder.key").expanduser()
HEAD_NPZ = Path("~/.mesa/clm/heads/npz/b2b4a8c9.npz").expanduser()
TASK_IDS = ("term.fits", "column.ontology_fits")
VARIANTS = ("state_only", "suffixed", "f9_query", "control_f1")
ISSUE15 = {"state_only_raw": 0.452, "suffixed_raw": 0.958, "suffixed_projected": 0.947}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contexts(task_id: str, state: dict[str, Any], *, sorted_keys: bool = False) -> dict[str, str]:
    """The four context texts of one snapshot row (module docstring), builder-ordered unless
    ``sorted_keys`` (the 2026-09-29 rendering)."""
    if not sorted_keys:
        state = builder_ordered(state)
    scoped = {**state, "scope": state.get("scope") or TASKS[task_id].scope}
    view = target_state_from(scoped)
    question = TASKS[task_id].text
    f9 = framings.framing(task_id, "F9")
    f1 = framings.framing(task_id, "F1")
    return {
        "state_only": render.to_text(view),
        "suffixed": render.state_text(view, question),
        "f9_query": framings.context_text(f9, scoped),
        "control_f1": framings.context_text(f1, state),
    }


def pair_stats(vectors: np.ndarray, cards: list[str]) -> dict[str, Any]:
    """Mean pairwise cosine (i < j), within/across card means, top-PC variance fraction."""
    x = np.asarray(vectors, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sims = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    card = np.asarray(cards)
    same = card[iu[0]] == card[iu[1]]
    vals = sims[iu]
    centred = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    var = s**2
    return {
        "n": len(x),
        "n_pairs": len(vals),
        "mean_pairwise_cos": round(float(vals.mean()), 4),
        "min_pairwise_cos": round(float(vals.min()), 4),
        "within_card_mean": round(float(vals[same].mean()), 4) if same.any() else None,
        "across_card_mean": round(float(vals[~same].mean()), 4) if (~same).any() else None,
        "n_within_pairs": int(same.sum()),
        "top_pc_variance_fraction": round(float(var[0] / var.sum()), 4),
        "top3_pc_variance_fraction": round(float(var[:3].sum() / var.sum()), 4),
        "norm_of_mean_vector": round(float(np.linalg.norm(x.mean(axis=0))), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / f"bench/results/{DATE}/collapse_spike.json")
    ap.add_argument(
        "--contexts-out", type=Path, default=ROOT / f".local/serving/collapse_contexts_{DATE}.json"
    )
    ap.add_argument(
        "--sorted-keys",
        action="store_true",
        help="render the snapshot's sorted-key state_json as the 2026-09-29 run did",
    )
    args = ap.parse_args(argv)
    args.out, args.contexts_out = args.out.resolve(), args.contexts_out.resolve()

    t0 = time.perf_counter()
    labels_sha256 = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    con = duckdb.connect()
    rows = con.execute(
        "select task_id, label_id, card, state_json from read_parquet(?) "
        "where task_id in (?, ?) order by task_id, label_id",
        [str(SNAPSHOT), *TASK_IDS],
    ).fetchall()
    con.close()

    # {(task, variant): {text: card}} in first-seen (label_id) order.
    distinct: dict[tuple[str, str], dict[str, str]] = {
        (t, v): {} for t in TASK_IDS for v in VARIANTS
    }
    counts: dict[str, int] = dict.fromkeys(TASK_IDS, 0)
    scope_added: dict[str, int] = dict.fromkeys(TASK_IDS, 0)
    for task_id, _label_id, card, state_json in rows:
        state = json.loads(state_json)
        counts[task_id] += 1
        scope_added[task_id] += "scope" not in state
        for variant, text in contexts(task_id, state, sorted_keys=args.sorted_keys).items():
            distinct[(task_id, variant)].setdefault(text, card)

    texts = sorted({t for d in distinct.values() for t in d})
    enc = EncoderClient(ENC_URL, KEY_FILE.read_text(encoding="utf-8").strip(), timeout=300.0)
    print(f"embedding {len(texts)} distinct texts", file=sys.stderr)
    t_embed = time.perf_counter()
    raw, tokens = enc.embed(texts)
    embed_s = time.perf_counter() - t_embed
    head = HeadProjector.from_npz(HEAD_NPZ)
    projected = head.project_states(raw)
    index = {t: i for i, t in enumerate(texts)}

    results: dict[str, Any] = {}
    for task_id in TASK_IDS:
        results[task_id] = {}
        for variant in VARIANTS:
            d = distinct[(task_id, variant)]
            idx = [index[t] for t in d]
            cards = list(d.values())
            tok = np.asarray([enc.count_tokens(t)[0] for t in d], dtype=np.float64)
            results[task_id][variant] = {
                "n_distinct_texts": len(d),
                "n_cards": len(set(cards)),
                "tokens": {
                    "p50": float(np.percentile(tok, 50)),
                    "p95": float(np.percentile(tok, 95)),
                    "max": int(tok.max()),
                },
                "raw_4096": pair_stats(raw[idx], cards),
                "projected_512": pair_stats(projected[idx], cards),
            }
    headline = {
        task_id: {
            space: {
                "state_only": results[task_id]["state_only"][space]["mean_pairwise_cos"],
                "suffixed": results[task_id]["suffixed"][space]["mean_pairwise_cos"],
                "f9_query": results[task_id]["f9_query"][space]["mean_pairwise_cos"],
                "control_f1": results[task_id]["control_f1"][space]["mean_pairwise_cos"],
            }
            for space in ("raw_4096", "projected_512")
        }
        for task_id in TASK_IDS
    }

    state_only = [
        {"task": task_id, "card": card, "text_sha256": sha256_text(text), "text": text}
        for task_id in TASK_IDS
        for text, card in sorted(
            distinct[(task_id, "state_only")].items(), key=lambda kv: sha256_text(kv[0])
        )
    ]
    args.contexts_out.parent.mkdir(parents=True, exist_ok=True)
    args.contexts_out.write_text(
        json.dumps({"labels_sha256": labels_sha256, "contexts": state_only}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    payload = {
        "format": "mesa-clm/collapse-spike/1",
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "command": "uv run python scripts/collapse_spike.py"
        + (" --sorted-keys" if args.sorted_keys else ""),
        "rendering": "sorted-key state_json"
        if args.sorted_keys
        else "builder-ordered (learn.features.builder_ordered), as the pipeline renders",
        "label_free": True,
        "snapshot": str(SNAPSHOT.relative_to(ROOT)),
        "labels_sha256": labels_sha256,
        "rows": counts,
        "ontology_state_rows_given_task_scope": scope_added,
        "encoder": {
            "url": ENC_URL,
            "model": enc.model,
            "max_len": enc.max_len,
            "encoder_fp": lock["encoder_fp"],
            "route": lock["encoder"]["route"],
            "batch_invariance": lock["encoder"].get("batch_invariance", "none"),
        },
        "head": {
            "npz": str(HEAD_NPZ.name),
            "source_sha256": head.source_sha256,
            "side": "state",
        },
        "n_distinct_texts_embedded": len(texts),
        "prompt_tokens": tokens,
        "embed_seconds": round(embed_s, 1),
        "variants": {
            "state_only": "render.to_text(target_state_from(state))",
            "suffixed": "render.state_text(view, TASKS[task].text)",
            "f9_query": "framings.context_text(framing(task, 'F9'), state)",
            "control_f1": "framings.context_text(framing(task, 'F1'), state): the stored anyjev "
            "per-candidate state + the question (one context per row)",
        },
        "headline_mean_pairwise_cos": headline,
        "issue15_reference": ISSUE15,
        "results": results,
        "contexts_file": str(args.contexts_out.relative_to(ROOT)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out)
    for task_id in TASK_IDS:
        print(task_id, json.dumps(headline[task_id]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
