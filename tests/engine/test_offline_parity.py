"""Offline scoring from cached float32 vectors against the live clm-serve (marker ``engine``;
``MESA_CLM_ENGINE=1``): plan §5.6 X1's pre-registered "cross-check vs clm-serve <= 1e-4 in
probs" and plan §9's "s_c vs /v1/rank", on a deterministic draw of F4/F7/F9 groups of the
committed snapshot (label-free). The texts are embedded one per request through the real
encoder :8090 into a feature store under ``tmp_path`` opened with the live lock's vector recipe,
scored by :class:`~mesa_clm.learn.offline.OfflineScorer` exactly as X1 scores them, and compared
with clm-serve's ``/v1/systemone`` and ``/v1/rank`` answers to the same request. The full
200-group record is ``scripts/x1_crosscheck.py`` (``bench/results/2026-10-01/x1_crosscheck.json``).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings, render
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.fingerprint import clm_model_fp, load_serving_lock
from mesa_clm.clm.headproj import HeadProjector
from mesa_clm.clm.http import ChoiceAnswer, ClmHttpClient, question_to_dict
from mesa_clm.config import Config, load_config
from mesa_clm.learn import features as feat
from mesa_clm.learn.offline import OfflineScorer
from mesa_clm.providers.live import live_lock_path
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.serving import HEADS_DIR, serving_home
from mesa_clm.tasks import TASKS

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        os.environ.get("MESA_CLM_ENGINE") != "1", reason="set MESA_CLM_ENGINE=1 (live serving)"
    ),
]

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
GATE = 1e-4  # plan §5.6, in probability
PER_STRATUM = 2  # 12 groups: 2 per (task, framing) of F4/F7/F9


def _keyed_config() -> Config:
    cfg = load_config()
    if not (cfg.clm.resolved_api_key() and cfg.encoder.resolved_api_key()):
        pytest.skip("no keys: run `mesa-clm serve keys --init` or set the *_API_KEY_FILE settings")
    return cfg


def _groups() -> list[tuple[str, str, Any, Any, str, dict[str, str], str]]:
    """``(task, framing, state, questions, context, {key: candidate}, anchor)`` for the first
    groups of each stratum in ``sha256("task|framing|target")`` order (x1_crosscheck's rule)."""
    pairs = feat._snapshot_pairs(SNAPSHOT, feat.X1_TASKS)
    out = []
    for task_id in feat.X1_TASKS:
        targets: dict[str, list[Any]] = {}
        for p in pairs[task_id]:
            targets.setdefault(p.target_sha256, []).append(p)
        scope = TASKS[task_id].scope
        for fid in ("F4", "F7", "F9"):
            f = framings.framing(task_id, fid)
            order = sorted(
                targets,
                key=lambda t, task=task_id, fr=fid: hashlib.sha256(
                    f"{task}|{fr}|{t}".encode()
                ).hexdigest(),
            )
            for target in order[:PER_STRATUM]:
                group = targets[target]
                states = [{**p.state, "scope": p.state.get("scope") or scope} for p in group]
                cands = [feat._candidate(task_id, p) for p in group]
                state, questions = framings.build_request(f, states[0], cands)
                context, keys, texts = render.build_pairs(state, questions)[task_id]
                assert keys[-1] == ANCHOR_KEY
                candidates = dict(zip(keys[:-1], texts[:-1], strict=True))
                out.append((task_id, fid, state, questions, context, candidates, texts[-1]))
    return out


def _s_c(probs: dict[str, float], key: str) -> float:
    return math.log(probs[key]) - math.log(probs[ANCHOR_KEY])


def test_offline_float32_scores_match_clm_serve(tmp_path: Path) -> None:
    cfg = _keyed_config()
    lock = load_serving_lock(live_lock_path())
    head = HeadProjector.from_npz(
        serving_home() / HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    )
    assert head.source_sha256 == lock.head.sha256
    store = feat.FeatureStore.for_lock(tmp_path / "features", lock)
    groups = _groups()
    texts = list(dict.fromkeys(t for g in groups for t in (g[4], *g[5].values(), g[6])))
    with EncoderClient.from_config(cfg.encoder, allow_remote=cfg.clm.allow_remote) as enc:
        enc.batch = 1
        vectors = np.stack([enc.embed([t])[0][0] for t in texts])
        counts = enc.token_guard(texts)
    store.add(texts, vectors, [c.tokens for c in counts])
    scorers = {
        "clm-latest": OfflineScorer(
            "clm-latest", head, clm_model_fp=clm_model_fp(lock.model_spec("clm-latest"))
        ),
        "clm-raw": OfflineScorer("clm-raw"),
    }
    worst: dict[str, float] = {}
    with ClmHttpClient.from_config(cfg.clm) as clm:
        for model, scorer in scorers.items():
            for task_id, fid, state, questions, context, candidates, anchor in groups:
                off = scorer.rank_fit_texts(store, context, candidates, anchor)
                answer = clm.system_one(state, questions, model=model).answers[task_id]
                assert isinstance(answer, ChoiceAnswer)
                served = {str(k): float(v) for k, v in answer.probabilities.items()}
                question = question_to_dict(questions[task_id])
                criteria: dict[str, str] = dict(question["criteria"])
                ranked = clm.rank(
                    state, question.get("instructions"), list(criteria.values()), model=model
                )
                by_text = {r.candidate: float(r.prob) for r in ranked}
                rank = {key: by_text[text] for key, text in criteria.items()}
                gap = max(abs(off.probabilities[k] - served[k]) for k in served)
                rank_gap = max(abs(off.probabilities[k] - rank[k]) for k in rank)
                worst[model] = max(worst.get(model, 0.0), gap, rank_gap)
                where = f"{model} {task_id}/{fid}"
                assert gap <= GATE, (where, gap)
                assert rank_gap <= GATE, (where, rank_gap)
                assert off.winner == answer.choice, where
                for key in candidates:
                    # s_c = ln p_c - ln p_anchor, where both are large enough to carry it.
                    if min(rank[key], rank[ANCHOR_KEY]) >= 1e-3:
                        assert abs(off.s_c[key] - _s_c(rank, key)) <= 1e-2, (where, key)
    assert set(worst) == set(scorers)
