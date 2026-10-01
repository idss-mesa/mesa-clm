"""Offline scoring from cached vectors: CLM's engine maths without clm-serve (plan §4.1, §5.6
X1 "scoring offline from the cache"; DESIGN D2, D7).

CLM scores a question by the scaled cosine between the projected state and each projected
candidate, then a softmax over that question's candidates (``engine.py:132-136``,
``schema.py``). This module repeats exactly that over vectors from
:class:`~mesa_clm.learn.features.FeatureStore` (or any other source), so X1 can score every
framing, model and anchor variant from one embedding pass:

* ``clm-latest`` (or another head): ``zs``, ``zc``, ``za`` are the L2-normalised state,
  candidate and anchor projections of :class:`~mesa_clm.clm.headproj.HeadProjector`, and
  ``scale = min(exp(logit_scale), 100)`` (:attr:`HeadProjector.scale`, ``heads.py:95``);
* ``clm-raw``: the raw 4096-d vectors, L2-normalised (CLM's ``RAW_SCALE = 100``).

A rank_fit group is one Choice over its candidates plus the anchor (D2); each candidate's
``s_c = scale * (zs·zc - zs·za) / T`` is ``ln p_c - ln p_anchor`` and does not depend on the rest
of the set, and the group's probabilities are ``render.answer_from_logits`` over the logits
``scale * cos / T`` in option order, which is how clm-serve answers. ``p_fit`` at zero shot is
``σ(s_c)``; ``confidence = max(probs)`` is computed here, CLM's own field stays
``clm_confidence`` (D7). Cosines are float64 dot products of the float32 projections, the same
expression :class:`mesa_clm.clm.fake.FakeClm` evaluates, so offline scores equal the fake's
``/v1/systemone`` answers on the same vectors (``tests/unit/test_offline.py``, ``<= 1e-9``).

What carries over depends on the encoder recipe (DESIGN A3). Under the M1-A recipe
(``encoder_fp`` 852efc921a8a) a text's vector moved by up to about 1.3e-4 in cosine with its
batch, so clm-serve and an offline replay differed by a few hundredths in probability at scale
100 (``bench/results/2026-09-29/serving_m1.json``). Under the adopted batch-invariant recipe
(``batch_invariance: kernels``, ``encoder_fp`` c3b3d5e1a283) batched and single requests agree to
1 - 3.3e-15 in cosine and clm-serve matches the local route to 3.9e-6 (``clm-latest``) and
6.6e-5 (``clm-raw``) in probability (``bench/results/2026-10-01/batch_invariance.json``,
``serving_m1b.json``). Offline scores are exact for the vectors they are given and reproduce
clm-serve only as closely as the vectors do: from float16 copies (round-trip cosine 0.9999999)
X1's 200-group cross-check missed clm-serve by up to 1.31e-3 / 3.59e-3, because float16 rounding
spread over the vector moves scores at the scale of 100 (restoring the three largest dimensions
from float32 recovers most of ``clm-raw``'s gap and none of ``clm-latest``'s;
``bench/results/2026-10-01/features_build.json#/rerun/crosscheck/cause``), so the feature store
keeps float32 vectors, from which the same groups match ``/v1/systemone`` and ``/v1/rank`` to 5.2e-6
(``clm-latest``) and 7.47e-5 (``clm-raw``; clm-serve sums its 4096-d cosines in float32, this
module in float64) within plan §5.6's 1e-4 (``bench/results/2026-10-01/x1_crosscheck.json``).
Read a store through :meth:`~mesa_clm.learn.features.FeatureStore.for_lock`, which refuses
vectors of another serving recipe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from mesa_clm import render
from mesa_clm.clm.headproj import RAW_SCALE, HeadProjector, Side, l2
from mesa_clm.learn.features import FeatureStore
from mesa_clm.registry import ANCHOR_KEY

__all__ = [
    "LATEST_MODEL",
    "RAW_MODEL",
    "NoulScores",
    "OfflineScorer",
    "RankFitScores",
    "pairwise_s_c",
    "sigmoid",
]

RAW_MODEL: Final[str] = "clm-raw"
LATEST_MODEL: Final[str] = "clm-latest"
_NOUL_KEYS: Final[tuple[str, str]] = render.NOUL_KEYS  # ("false", "true"), CLM's answer order

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
AnyFloat = npt.NDArray[np.floating[Any]]


def sigmoid(x: float) -> float:
    """``1 / (1 + e^-x)`` without overflow for large ``|x|`` (``p_fit`` at zero shot)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _check_temperature(temperature: float) -> float:
    t = float(temperature)
    if not 0.0 < t <= 100.0:
        raise ValueError("temperature must be in (0, 100] (CLM's bounds)")
    return t


def _row(z: AnyFloat, what: str) -> F32:
    arr = np.asarray(z, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"{what} must be one vector, got shape {list(np.shape(z))}")
    return arr


def _rows(z: AnyFloat, what: str, dim: int) -> F32:
    arr = np.asarray(z, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"{what} must be [k, {dim}], got shape {list(np.shape(z))}")
    return arr


def _cosines(z_state: F32, z_options: F32) -> F64:
    """``z_options @ z_state`` in float64, the engine's (and the fake's) expression."""
    return np.asarray(z_options.astype(np.float64) @ z_state.astype(np.float64), dtype=np.float64)


@dataclass(frozen=True)
class RankFitScores:
    """One rank_fit group scored offline: ``keys`` are the candidates then the anchor (the wire
    order), ``logits`` ``scale * cos / T`` per key, ``answer`` what ``render.answer_from_logits``
    makes of them (the ``/v1/systemone`` Choice answer), ``s_c`` per candidate."""

    keys: tuple[str, ...]
    logits: tuple[float, ...]
    answer: dict[str, Any]
    s_c: dict[str, float]
    anchor_key: str = ANCHOR_KEY

    @property
    def probabilities(self) -> dict[str, float]:
        """The group's softmax over candidates and anchor, keyed like the wire answer."""
        return render.probabilities_of(self.answer)

    @property
    def confidence(self) -> float:
        """``max(probs)``, computed locally (D7)."""
        return max(self.probabilities.values())

    @property
    def clm_confidence(self) -> float:
        """CLM's own ``confidence`` field (top minus mean of the rest), stored only as such (D7)."""
        return float(self.answer["confidence"])

    @property
    def winner(self) -> str:
        """The answered key; the anchor winning is an abstain (D2)."""
        return str(self.answer["choice"])

    def p_fit(self, key: str) -> float:
        """``σ(s_c)``: the zero-shot fit probability of one candidate (plan §5.3)."""
        return sigmoid(self.s_c[key])


@dataclass(frozen=True)
class NoulScores:
    """One noul (the F1 control) scored offline: ``logits`` for ``false`` and ``true``,
    ``answer`` the ``/v1/systemone`` noul answer, ``s = logit_true - logit_false``."""

    logits: tuple[float, float]
    answer: dict[str, Any]

    @property
    def p_true(self) -> float:
        """CLM's noul answer: ``σ(s)``."""
        return float(self.answer["noul"])

    @property
    def s(self) -> float:
        """``logit_true - logit_false = scale * (zs·z_true - zs·z_false) / T``."""
        return self.logits[1] - self.logits[0]


class OfflineScorer:
    """CLM's scoring for one served model over given vectors (module docstring).

    ``model`` ``clm-raw`` takes no projector; any other model needs the
    :class:`HeadProjector` it serves and, to read projections cached in a
    :class:`FeatureStore`, that head's ``clm_model_fp``.
    """

    def __init__(
        self,
        model: str = LATEST_MODEL,
        projector: HeadProjector | None = None,
        *,
        clm_model_fp: str | None = None,
    ) -> None:
        if model == RAW_MODEL and projector is not None:
            raise ValueError("clm-raw scores raw cosines; it takes no head")
        if model != RAW_MODEL and projector is None:
            raise ValueError(f"{model} needs the HeadProjector it serves")
        self.model = model
        self.projector = projector
        self.clm_model_fp = clm_model_fp

    @property
    def raw(self) -> bool:
        return self.projector is None

    @property
    def scale(self) -> float:
        """``RAW_SCALE`` for clm-raw, else the head's clamped ``exp(logit_scale)``."""
        return RAW_SCALE if self.projector is None else self.projector.scale

    # -- projections ------------------------------------------------------------------------------

    def states(self, x: AnyFloat) -> F32:
        """``[n, d]`` state-side vectors: L2-normalised raw vectors, or head projections."""
        if self.projector is None:
            return l2(np.asarray(x))
        return self.projector.project_states(np.asarray(x))

    def actions(self, x: AnyFloat) -> F32:
        """``[n, d]`` action-side vectors (candidates, anchors, noul texts)."""
        if self.projector is None:
            return l2(np.asarray(x))
        return self.projector.project_actions(np.asarray(x))

    def _store_side(self, store: FeatureStore, side: Side, texts: Sequence[str]) -> F32:
        """``side`` vectors of ``texts`` from ``store``: normalised raw vectors for clm-raw, the
        head's cached projections otherwise (computed and cached on first use)."""
        if self.projector is None:
            return l2(store.get(texts))
        if self.clm_model_fp is None:
            raise ValueError(f"{self.model}: reading cached projections needs clm_model_fp")
        return store.project(self.projector, self.clm_model_fp, side, texts)

    # -- scoring over projected vectors -----------------------------------------------------------

    def logits(self, z_state: AnyFloat, z_options: AnyFloat, temperature: float = 1.0) -> F64:
        """``scale * cos / T`` of each option against the state, float64."""
        t = _check_temperature(temperature)
        zs = _row(z_state, "z_state")
        zo = _rows(z_options, "z_options", zs.shape[0])
        return np.asarray(self.scale * _cosines(zs, zo) / t, dtype=np.float64)

    def rank_fit(
        self,
        keys: Sequence[str],
        z_state: AnyFloat,
        z_candidates: AnyFloat,
        z_anchor: AnyFloat,
        *,
        temperature: float = 1.0,
        anchor_key: str = ANCHOR_KEY,
    ) -> RankFitScores:
        """One rank_fit group: ``keys`` name the rows of ``z_candidates``; the anchor is the
        last option, as the framings put it on the wire."""
        t = _check_temperature(temperature)
        names = [str(k) for k in keys]
        if len(set(names)) != len(names) or anchor_key in names or not names:
            raise ValueError("candidate keys must be unique, non-empty and not the anchor key")
        zs = _row(z_state, "z_state")
        zc = _rows(z_candidates, "z_candidates", zs.shape[0])
        za = _row(z_anchor, "z_anchor")
        if zc.shape[0] != len(names):
            raise ValueError(f"{len(names)} keys but {zc.shape[0]} candidate vectors")
        options = np.vstack([zc, za[None, :]])
        cos = _cosines(zs, options)
        logits = (self.scale * cos / t).tolist()
        all_keys = [*names, anchor_key]
        answer = render.answer_from_logits({"type": "choice"}, all_keys, logits)
        s_c = {k: float(self.scale * (cos[i] - cos[-1]) / t) for i, k in enumerate(names)}
        return RankFitScores(
            tuple(all_keys), tuple(float(v) for v in logits), answer, s_c, anchor_key
        )

    def noul(
        self,
        z_state: AnyFloat,
        z_false: AnyFloat,
        z_true: AnyFloat,
        *,
        temperature: float = 1.0,
    ) -> NoulScores:
        """One noul: the ``false``/``true`` candidate texts CLM derives from the instructions
        (``render.candidates``), in CLM's key order."""
        t = _check_temperature(temperature)
        zs = _row(z_state, "z_state")
        options = np.vstack([_row(z_false, "z_false")[None, :], _row(z_true, "z_true")[None, :]])
        logits = (self.scale * _cosines(zs, options) / t).tolist()
        answer = render.answer_from_logits({"type": "noul"}, list(_NOUL_KEYS), logits)
        return NoulScores((float(logits[0]), float(logits[1])), answer)

    # -- scoring texts through a feature store -----------------------------------------------------

    def rank_fit_texts(
        self,
        store: FeatureStore,
        state_text: str,
        candidates: Mapping[str, str],
        anchor_text: str,
        *,
        temperature: float = 1.0,
        anchor_key: str = ANCHOR_KEY,
    ) -> RankFitScores:
        """:meth:`rank_fit` over texts whose vectors are in ``store`` (projections cached there
        for a head): the state text, ``{key: candidate text}`` in wire order, the anchor text."""
        zs = self._store_side(store, "state", [state_text])
        za = self._store_side(store, "action", [*candidates.values(), anchor_text])
        return self.rank_fit(
            list(candidates),
            zs[0],
            za[:-1],
            za[-1],
            temperature=temperature,
            anchor_key=anchor_key,
        )

    def noul_texts(
        self,
        store: FeatureStore,
        state_text: str,
        false_text: str,
        true_text: str,
        *,
        temperature: float = 1.0,
    ) -> NoulScores:
        """:meth:`noul` over texts whose vectors are in ``store``."""
        zs = self._store_side(store, "state", [state_text])
        za = self._store_side(store, "action", [false_text, true_text])
        return self.noul(zs[0], za[0], za[1], temperature=temperature)


def pairwise_s_c(
    scale: float,
    z_states: AnyFloat,
    z_candidates: AnyFloat,
    z_anchors: AnyFloat,
    *,
    temperature: float = 1.0,
) -> F64:
    """``s_c`` for ``n`` labelled pairs at once: row ``i`` scores candidate ``i`` against state
    ``i`` and anchor ``i`` (one anchor row broadcasts), ``scale * (zs·zc - zs·za) / T``. The
    vectorised form X1 needs over every (target, candidate) of a snapshot; equal to
    :meth:`OfflineScorer.rank_fit` row by row."""
    t = _check_temperature(temperature)
    zs = np.asarray(z_states, dtype=np.float32)
    zc = np.asarray(z_candidates, dtype=np.float32)
    za = np.asarray(z_anchors, dtype=np.float32)
    if za.ndim == 1:
        za = np.broadcast_to(za, zs.shape)
    if zs.ndim != 2 or zs.shape != zc.shape or zs.shape != za.shape:
        raise ValueError(
            f"states, candidates and anchors must be matching [n, d] arrays, got "
            f"{list(zs.shape)}, {list(zc.shape)}, {list(za.shape)}"
        )
    s64 = zs.astype(np.float64)
    cos_c = np.einsum("ij,ij->i", s64, zc.astype(np.float64))
    cos_a = np.einsum("ij,ij->i", s64, za.astype(np.float64))
    return np.asarray(float(scale) * (cos_c - cos_a) / t, dtype=np.float64)
