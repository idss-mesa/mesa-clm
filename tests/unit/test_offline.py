"""Offline scoring (``mesa_clm.learn.offline``) against the fake engine (plan §5.6 X1; DESIGN
D2, D7): on the same vectors, the offline rank_fit group and noul scores equal
:class:`~mesa_clm.clm.fake.FakeClm`'s ``/v1/systemone`` answers to ``<= 1e-9`` for ``clm-latest``
(a numpy head) and ``clm-raw``, at two temperatures, directly and through a
:class:`~mesa_clm.learn.features.FeatureStore` (whose float16 vectors the fake is then given);
``s_c = ln p_c - ln p_anchor`` and does not depend on the rest of the set; ``p_fit = σ(s_c)``,
``confidence = max(probs)``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import render
from mesa_clm.clm.fake import FakeClm, FakeEncoder
from mesa_clm.clm.fingerprint import ClmModelSpec, clm_model_fp
from mesa_clm.clm.headproj import RAW_SCALE, HeadProjector, random_head
from mesa_clm.learn.features import FeatureStore
from mesa_clm.learn.offline import (
    LATEST_MODEL,
    RAW_MODEL,
    OfflineScorer,
    RankFitScores,
    pairwise_s_c,
    sigmoid,
)
from mesa_clm.providers.tiered import fake_fingerprint
from mesa_clm.registry import ANCHOR_KEY, ANCHORS
from mesa_clm.serving import CLM_COMMIT

TOL = 1e-9
STATE = {
    "card": {"dataset": "DP1.10003.001.brd_countdata", "product_title": "Breeding landbird"},
    "scope": "column",
    "aspect": "measurement",
    "column": {
        "name": "observerDistance",
        "description": "Radial distance between the observer and the individual(s) observed",
        "unit": "meter",
    },
}
CANDIDATES = {
    "PATO:0000040": "distance: A 1-D extent quality equal to the distance between two points.",
    "PATO:0000014": "color: A composite chromatic quality.",
    "UO:0000008": "meter: A length unit equal to the length of the path traveled by light.",
}
ANCHOR = ANCHORS["term"]
QUESTION = "Is this ontology term the right concept for this target?"
FP = fake_fingerprint().encoder_fp
MODEL_FP = clm_model_fp(ClmModelSpec(head_name=LATEST_MODEL, head_sha256="", clm_commit=CLM_COMMIT))


def _choice(candidates: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": None,
        "criteria": {**candidates, ANCHOR_KEY: ANCHOR},
    }


def _engine(collapse: float = 0.3) -> tuple[FakeEncoder, HeadProjector, FakeClm]:
    enc = FakeEncoder(collapse=collapse)
    head = random_head(7, hidden_size=enc.dim)
    return enc, head, FakeClm(enc, heads={LATEST_MODEL: head})


def _scorer(model: str, head: HeadProjector, **kw: Any) -> OfflineScorer:
    return OfflineScorer(model, None if model == RAW_MODEL else head, **kw)


def _offline_group(
    enc: FakeEncoder, scorer: OfflineScorer, question: dict[str, Any], temperature: float
) -> RankFitScores:
    """Embed exactly what the engine embeds (``render.build_pairs``): the state text alone and
    the question's candidate texts in wire order, then score offline."""
    state_text, keys, texts = render.build_pairs(STATE, {"q": question})["q"]
    zs = scorer.states(enc.embed([state_text])[0])
    za = scorer.actions(enc.embed(texts)[0])
    return scorer.rank_fit(keys[:-1], zs[0], za[:-1], za[-1], temperature=temperature)


def _assert_same(got: RankFitScores, answer: dict[str, Any]) -> None:
    want = answer["probabilities"]
    assert list(got.probabilities) == list(want)
    for key, p in want.items():
        assert abs(got.probabilities[key] - p) <= TOL
    assert got.winner == answer["choice"]
    assert abs(got.clm_confidence - answer["confidence"]) <= TOL
    for key in got.s_c:  # D2: s_c = ln p_c - ln p_anchor
        assert abs(got.s_c[key] - (math.log(want[key]) - math.log(want[ANCHOR_KEY]))) <= TOL


@pytest.mark.parametrize("model", [LATEST_MODEL, RAW_MODEL])
@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_rank_fit_equals_the_fake_engine(model: str, temperature: float) -> None:
    enc, head, clm = _engine()
    question = _choice(CANDIDATES)
    answer = clm.answer(STATE, {"q": question}, model=model, temperature=temperature)
    got = _offline_group(enc, _scorer(model, head), question, temperature)
    _assert_same(got, answer["answers"]["q"])
    assert got.keys == (*CANDIDATES, ANCHOR_KEY)
    assert got.confidence == max(got.probabilities.values())
    for key in CANDIDATES:
        assert got.p_fit(key) == sigmoid(got.s_c[key])
    scale = RAW_SCALE if model == RAW_MODEL else head.scale
    assert scale == 100.0  # the released head's clamp, and clm-raw's constant


@pytest.mark.parametrize("model", [LATEST_MODEL, RAW_MODEL])
def test_s_c_is_set_independent(model: str) -> None:
    """One candidate's s_c is the same whatever else is offered (D2), offline and in the fake."""
    enc, head, clm = _engine()
    scorer = _scorer(model, head)
    seen: list[float] = []
    for subset in (["PATO:0000040"], ["PATO:0000040", "UO:0000008"], list(CANDIDATES)):
        question = _choice({k: CANDIDATES[k] for k in subset})
        probs = clm.answer(STATE, {"q": question}, model=model)["answers"]["q"]["probabilities"]
        got = _offline_group(enc, scorer, question, 1.0)
        if seen:
            assert abs(got.s_c["PATO:0000040"] - seen[0]) <= TOL
        engine_s = math.log(probs["PATO:0000040"]) - math.log(probs[ANCHOR_KEY])
        assert abs(got.s_c["PATO:0000040"] - engine_s) <= TOL
        seen.append(got.s_c["PATO:0000040"])


@pytest.mark.parametrize("model", [LATEST_MODEL, RAW_MODEL])
def test_noul_equals_the_fake_engine(model: str) -> None:
    enc, head, clm = _engine()
    question = {"type": "noul", "instructions": QUESTION}
    want = clm.answer(STATE, {"q": question}, model=model, temperature=0.7)["answers"]["q"]
    state_text, keys, texts = render.build_pairs(STATE, {"q": question})["q"]
    assert keys == ["false", "true"]
    scorer = _scorer(model, head)
    zs = scorer.states(enc.embed([state_text])[0])
    za = scorer.actions(enc.embed(texts)[0])
    got = scorer.noul(zs[0], za[0], za[1], temperature=0.7)
    assert abs(got.p_true - want["noul"]) <= TOL
    assert abs(got.p_true - sigmoid(got.s)) <= TOL


class _StoreEncoder(FakeEncoder):
    """The fake encoder serving exactly the store's float16-widened vectors."""

    def __init__(self, store: FeatureStore) -> None:
        super().__init__()
        self.store = store

    def embed(self, texts: Sequence[str]) -> tuple[Any, int]:
        return self.store.get(list(texts)), sum(self.count_tokens(t) for t in texts)


@pytest.mark.parametrize("model", [LATEST_MODEL, RAW_MODEL])
def test_scoring_through_the_feature_store(tmp_path: Path, model: str) -> None:
    enc, head, _ = _engine()
    store = FeatureStore(tmp_path / "features", FP)
    rank_q = _choice(CANDIDATES)
    noul_q = {"type": "noul", "instructions": QUESTION}
    pairs = render.build_pairs(STATE, {"r": rank_q, "n": noul_q})
    texts = list(dict.fromkeys(t for s, _, ts in pairs.values() for t in (s, *ts)))
    vectors, _ = enc.embed(texts)
    store.add(texts, vectors, [enc.count_tokens(t) for t in texts])
    # The fake engine on the store's vectors: what clm-serve would answer from these vectors.
    clm = FakeClm(_StoreEncoder(store), heads={LATEST_MODEL: head})
    scorer = _scorer(model, head, clm_model_fp=MODEL_FP)
    state_text, keys, cand_texts = pairs["r"]
    got = scorer.rank_fit_texts(
        store, state_text, dict(zip(keys[:-1], cand_texts[:-1], strict=True)), cand_texts[-1]
    )
    _assert_same(got, clm.answer(STATE, {"r": rank_q}, model=model)["answers"]["r"])
    state_text, _, (false_text, true_text) = pairs["n"]
    noul = scorer.noul_texts(store, state_text, false_text, true_text)
    want = clm.answer(STATE, {"n": noul_q}, model=model)["answers"]["n"]["noul"]
    assert abs(noul.p_true - want) <= TOL
    projections = store.stats()["projections"]
    if model == RAW_MODEL:
        assert projections == {}  # clm-raw never caches a projection
    else:
        assert projections == {MODEL_FP: {"state": 2, "action": 6}}


def test_pairwise_s_c_equals_the_groups() -> None:
    enc, head, _ = _engine(collapse=0.0)
    scorer = _scorer(LATEST_MODEL, head)
    question = _choice(CANDIDATES)
    group = _offline_group(enc, scorer, question, 1.0)
    state_text, keys, texts = render.build_pairs(STATE, {"q": question})["q"]
    zs = scorer.states(enc.embed([state_text])[0])
    za = scorer.actions(enc.embed(texts)[0])
    n = len(CANDIDATES)
    got = pairwise_s_c(scorer.scale, np.repeat(zs, n, axis=0), za[:-1], za[-1])
    np.testing.assert_allclose(got, [group.s_c[k] for k in keys[:-1]], rtol=0, atol=TOL)
    half = pairwise_s_c(
        scorer.scale, np.repeat(zs, n, axis=0), za[:-1], za[[-1] * n], temperature=0.5
    )
    np.testing.assert_allclose(half, 2 * got, rtol=0, atol=TOL)
    with pytest.raises(ValueError, match="matching"):
        pairwise_s_c(100.0, zs, za, za[-1])


def test_refusals_and_helpers() -> None:
    enc, head, _ = _engine()
    with pytest.raises(ValueError, match="no head"):
        OfflineScorer(RAW_MODEL, head)
    with pytest.raises(ValueError, match="needs the HeadProjector"):
        OfflineScorer(LATEST_MODEL)
    scorer = _scorer(LATEST_MODEL, head)
    assert not scorer.raw and OfflineScorer(RAW_MODEL).raw
    z = scorer.states(enc.embed(["a state"])[0])
    a = scorer.actions(enc.embed(["one", "two"])[0])
    for t in (0.0, -1.0, 100.5):
        with pytest.raises(ValueError, match="temperature"):
            scorer.rank_fit(["x"], z[0], a[:1], a[1], temperature=t)
    with pytest.raises(ValueError, match="unique"):
        scorer.rank_fit(["x", "x"], z[0], a, a[1])
    with pytest.raises(ValueError, match="unique"):
        scorer.rank_fit([ANCHOR_KEY], z[0], a[:1], a[1])
    with pytest.raises(ValueError, match="2 keys but 1"):
        scorer.rank_fit(["x", "y"], z[0], a[:1], a[1])
    with pytest.raises(ValueError, match="one vector"):
        scorer.rank_fit(["x"], a, a[:1], a[1])
    with pytest.raises(ValueError, match=r"\[k, 512\]"):
        scorer.logits(z[0], np.ones((2, 7)))
    want = scorer.scale * (a.astype(np.float64) @ z[0].astype(np.float64))
    np.testing.assert_allclose(scorer.logits(z, a), want, rtol=0, atol=TOL)
    with pytest.raises(ValueError, match="clm_model_fp"):
        scorer.rank_fit_texts(FeatureStore(Path("/nonexistent"), FP), "s", {"x": "c"}, "a")
    assert sigmoid(0.0) == 0.5 and sigmoid(800.0) == 1.0 and sigmoid(-800.0) == 0.0
    assert abs(sigmoid(2.0) + sigmoid(-2.0) - 1.0) <= 1e-15
