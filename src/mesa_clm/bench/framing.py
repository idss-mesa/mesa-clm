"""X1 framing A/B: scores, controls, LOCO-Platt, the decision rule and its nesting (DESIGN
"Pre-registration (G1)": X1, rule R, the LOCO folds, K1; D2, D3, D7, D20, D27, D30, D33; plan
§5.4, §5.6).

``design/m2-analysis-plan.md`` reads every frozen X1 rule as one deterministic algorithm and was
committed before the first run on real labels; the section numbers below (§n.m) are that plan's.
This module implements its X1 part and nothing else.

* **Items and scores** (:class:`TaskItems`, :func:`score_items`; §2, §3, §5). A task's items are
  the bench task's labelled (target, option) pairs (``neon_task``: the policy ``min_weight`` and
  the D30 fold filter), the same for every framing and model. Each item is scored offline from
  the feature store, ``s = scale·(zs·zc − zs·za)`` for F4/F7/F9 and the noul logit difference
  for F1 (:func:`~mesa_clm.learn.offline.pairwise_s_c` over
  :func:`~mesa_clm.learn.offline.store_vectors`), together with its score under every other
  context of its card (:func:`context_scores`, what the within-card shuffle draws from) and its
  score at the task's mean context (:func:`mean_context_scores`, report-only). Everything after
  this layer reads only these per-item numbers, the labels, weights, cards and token counts: what
  ``x1_items.parquet`` stores.
* **Statistics** (:class:`ItemEvidence`; §4–§6): the AUROC of ``s`` with its card-cluster CI,
  rule R on ΔAUROC(real − shuffle) against the mean AUROC of 200 seeded within-card derangements
  (:func:`shuffle_draws`, :func:`mesa_clm.bench.stats.rule_r_auroc_mean`), weighted Platt
  (:func:`platt_fit`, the code base's one Platt, :func:`mesa_clm.learn.calibrate.fit_platt`)
  leave-one-card-out with the 30/5 guards and the floor of 100 training items per fit
  (:func:`fold_plan`, :func:`loco_platt`), rule R on NLL.
* **The decision** (:func:`decide`; §7) is a pure function of an :class:`Evidence`, computed from
  items (:class:`ItemEvidence`) or read back from a stored trace (:class:`StoredEvidence`), so
  ``bench framing --decide --from x1.json`` (:func:`decide_from_json`) replays it bit for bit and
  recomputes it from the items file without the feature store or the label store.
* **Nesting** (:func:`nested_selection`; §8): the whole decision re-made inside each outer fold
  on its 6 training cards, and ``fold_choices``. X1 makes the choice; the citable
  ``<task>.<tier>.<A1>`` cells are the tier run's (``bench/cells.py``, ``tiers.json``), which
  receives X1's outcome as data (:func:`selection`, :func:`selections_from_json`).
* **Output** (:func:`run_x1`, :func:`write_x1`; §12): ``x1.json`` (the 16 full-run arm cells,
  built by the M2 cell builder :func:`mesa_clm.bench.cells.assemble_cell`, plus the ``x1`` block
  with every trace), ``x1_items.parquet`` and ``x1.md``.
* **Registration** (§1, §13): the run records its configuration; only the pre-registered one on
  the registered snapshot, under the registered framings lock and model fingerprints
  (:mod:`mesa_clm.bench.registered`, :func:`x1_deviations`), over exactly the configured grid
  (:func:`configured_arms`), is ``registered``. Any other run is written with every cell
  ``pre_registered: false`` and ``exploratory: true``, and :func:`decide_from_json` refuses it
  unless asked not to; it re-derives the status from what the file records and holds every number
  the file reports to its traces and its items file.

Nothing here embeds or calls a server. Label values enter only through the bench tasks, and the
vectors come from a :class:`~mesa_clm.learn.features.FeatureStore` opened with the live lock's
vector recipe.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import duckdb
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from mesa_clm import framings as fr
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.baselines import LookupEval, novel_key_block, pooled_metrics
from mesa_clm.bench.cells import assemble_cell, item_identities, mean_pairwise_cosine
from mesa_clm.bench.results import (
    DEFAULT_OUT_DIR,
    BenchCell,
    BenchResults,
    CellCounts,
    ClusterCI,
    cell_key,
    environment,
    finite,
    results_path,
    snapshot_content_sha256,
)
from mesa_clm.bench.results import FORMAT as RESULTS_FORMAT
from mesa_clm.bench.tasks.base import (
    MIN_HELDOUT_PER_CLASS,
    MIN_TRAIN_PER_CLASS,
    Fold,
    Task,
    fold_guard,
)
from mesa_clm.bench.tasks.neon import NEON_TASKS, neon_task
from mesa_clm.clm.fingerprint import ServingLock, clm_model_fp
from mesa_clm.clm.headproj import HeadProjector, l2
from mesa_clm.learn import calibrate
from mesa_clm.learn.features import (
    X1_FRAMINGS,
    X1_TASKS,
    FeatureMissing,
    FeatureStore,
    Manifest,
)
from mesa_clm.learn.offline import (
    LATEST_MODEL,
    RAW_MODEL,
    OfflineScorer,
    pairwise_s_c,
    store_vectors,
)
from mesa_clm.policy_defaults import min_weight_for
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import TASKS

__all__ = [
    "ARM_FRAMINGS",
    "ARM_ORDER",
    "AUROC_FLOOR",
    "AUROC_LOWER_GATE",
    "CALIBRATION_FLOOR",
    "CONTROL",
    "FORMAT",
    "FRAMING_IDS",
    "LATENCY_FORMAT",
    "MODELS",
    "MORE_CACHEABLE",
    "PROBE_FLOOR",
    "SHUFFLE_K",
    "SHUFFLE_SEED",
    "ArmEvidence",
    "ArmRecord",
    "ArmScores",
    "ArmVerdict",
    "CandidateProbe",
    "Choice",
    "Comparison",
    "Decision",
    "Evidence",
    "FoldPlan",
    "FoldRecord",
    "ItemEvidence",
    "LocoPlatt",
    "NestedRecord",
    "RuleRSummary",
    "StoredEvidence",
    "TaskItems",
    "TaskRecord",
    "TokenCounts",
    "X1Config",
    "X1DataError",
    "X1Exists",
    "X1ReproError",
    "X1Results",
    "X1Run",
    "arm_id",
    "candidate_probe",
    "config_deviations",
    "configured_arms",
    "context_scores",
    "decide",
    "decide_from_json",
    "derangements",
    "evaluate_task",
    "fingerprints_for_lock",
    "floor_reason",
    "fold_plan",
    "latency_summary",
    "latency_targets",
    "load_latency",
    "load_x1",
    "loco_platt",
    "markdown",
    "mean_context_scores",
    "nested_selection",
    "platt_fit",
    "probe_folds",
    "read_items",
    "registered_config",
    "run_x1",
    "run_x1_items",
    "score_items",
    "scorers_for_lock",
    "selection",
    "selections_from_json",
    "shuffle_draws",
    "split_arm",
    "write_items",
    "write_x1",
    "x1_cells",
    "x1_deviations",
    "x1_tasks",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
Clock = Callable[[], float]

FORMAT: Final[str] = "mesa-clm/bench-x1/1"
LATENCY_FORMAT: Final[str] = "mesa-clm/x1-latency/1"
NAME: Final[str] = "x1"
PLAN: Final[str] = "design/m2-analysis-plan.md"
ITEMS_SUFFIX: Final[str] = "_items.parquet"
MODELS: Final[tuple[str, ...]] = (LATEST_MODEL, RAW_MODEL)
FRAMING_IDS: Final[tuple[str, ...]] = X1_FRAMINGS
TASK_IDS: Final[tuple[str, ...]] = X1_TASKS
CONTROL: Final[str] = "F1"
# The arms rule (1) can qualify (§7.2) and, among them, the more cacheable ones (§7.5: "F7, F9 >
# F4": their contexts are the target alone, the same for every question about it).
ARM_FRAMINGS: Final[tuple[str, ...]] = ("F4", "F7", "F9")
MORE_CACHEABLE: Final[tuple[str, ...]] = ("F7", "F9")
ARM_ORDER: Final[tuple[str, ...]] = tuple(f"{f}@{m}" for f in FRAMING_IDS for m in MODELS)
# Rule (1) (§4.3, §7.2).
AUROC_FLOOR: Final[float] = 0.60
AUROC_LOWER_GATE: Final[float] = 0.5
# "floors 100 (calibrated) / 40 (probe/head)" (§6.3): the fewest labelled items any calibrator
# (or the candidate-only probe) is fitted on.
CALIBRATION_FLOOR: Final[int] = 100
PROBE_FLOOR: Final[int] = 40
# The within-card shuffle (§5): this many seeded derangements of each card's targets.
SHUFFLE_K: Final[int] = 200
SHUFFLE_SEED: Final[int] = 0
TEMPERATURE: Final[float] = 1.0
# decide_from_json's recompute tolerance on every number (§12.6).
REPRO_TOLERANCE: Final[float] = 1e-9
LATENCY_TARGETS: Final[int] = 20
TIMING_SOURCE: Final[str] = "offline_replay"
# Bench task name of each X1 task, in X1 order.
TASK_NAMES: Final[dict[str, str]] = {
    task_id: name for task_id in X1_TASKS for name, tid in NEON_TASKS.items() if tid == task_id
}
K1_NOTE: Final[str] = (
    "K1: no F4/F7/F9 arm qualifies on either model; the task's zero_shot/calibrated tiers "
    "become audit-only, proposals use ols_rank (proposed, D28) until a probe is promoted, and "
    "M4 goes probe-first"
)
PROBE_RECIPE: Final[str] = (
    "StandardScaler + LogisticRegression(C=1, max_iter=2000), unweighted (the PR #13 recipe of "
    "X2), on the raw 4096-d vectors of the items' candidate texts (shared by F4/F7/F9)"
)

Outcome = Literal["choice", "K1", "undecidable"]


class X1DataError(ValueError):
    """Inputs X1 cannot run on: an item without a manifest text, a vector or a token count, a
    snapshot mismatch, an inconsistent item table (§2.2). Raised before any statistic."""


class X1ReproError(RuntimeError):
    """``decide_from_json`` found a stored decision that its statistics or its items file do
    not reproduce, a configuration other than the code's, or (when one is required) a run that
    is not the pre-registered one (§12.6)."""


def arm_id(framing: str, model: str) -> str:
    """``F7@clm-latest``: the (framing, model) pair an X1 arm is (§1.5)."""
    return f"{framing}@{model}"


def split_arm(arm: str) -> tuple[str, str]:
    """``(framing, model)`` of an arm id."""
    framing, sep, model = arm.partition("@")
    if not sep or not framing or not model:
        raise ValueError(f"not an arm id: {arm!r}")
    return framing, model


# -- the within-card shuffle draws (§5) -----------------------------------------------------------


def derangements(m: int, k: int, rng: np.random.Generator) -> IntArray:
    """``k`` uniform derangements of ``range(m)`` (``m >= 2``) by rejection: draw
    ``rng.permutation(m)`` until no element stays in place (§5.2)."""
    if m < 2:
        raise ValueError("a derangement needs at least two elements")
    ident = np.arange(m)
    out = np.empty((k, m), dtype=np.int64)
    for row in range(k):
        while True:
            p = rng.permutation(m)
            if not np.any(p == ident):
                break
        out[row] = p
    return out


def shuffle_draws(
    card_sizes: Mapping[str, int], *, k: int = SHUFFLE_K, seed: int = SHUFFLE_SEED
) -> dict[str, IntArray | None]:
    """The within-card shuffle's draws (§5.2): one ``numpy.random.default_rng(seed)`` stream,
    cards in sorted order, ``k`` derangements of each card's targets (in sorted target order);
    ``None`` for a card with a single target (its items keep their own context, §5.6)."""
    rng = np.random.default_rng(seed)
    return {
        card: (derangements(m, k, rng) if m >= 2 else None)
        for card, m in sorted(card_sizes.items())
    }


# -- per-item data (§2, §3) ---------------------------------------------------------------------


@dataclass
class ArmScores:
    """One arm's per-item scores: ``s`` (§3); for F4/F7/F9 ``contexts``, the score of the
    item's candidate and anchor under each target context of its card (``[n, m]``, columns in the
    card's sorted target order, NaN past the card's target count; the item's own column is
    ``s``), from which the shuffle draws (§5), and ``mean_context``, the score at the task's mean
    context (§7.11, report-only)."""

    s: FloatArray
    contexts: FloatArray | None = None
    mean_context: FloatArray | None = None


@dataclass
class TokenCounts:
    """Per-item encoder token counts of one framing's texts (§7.7): the item's ``context``, its
    ``candidate`` and the ``anchor`` (for F1: ``noul_true`` and ``noul_false``)."""

    context: IntArray
    candidate: IntArray
    anchor: IntArray


@dataclass(frozen=True)
class CandidateProbe:
    """The learned candidate-only probe of one task (§7.11): held-out p(Yes) per item (NaN on a
    skipped fold), per fold its sizes and solver state, and the skipped folds."""

    probs: FloatArray
    folds: dict[str, dict[str, Any]]
    skipped: dict[str, str]


_DERIVED: Final[tuple[str, ...]] = (
    "card_arr",
    "positive",
    "guard_task",
    "card_targets",
    "slot",
    "card_size",
    "_shuffle_index",
)


@dataclass
class TaskItems:
    """The per-item data of one task (§2): identities, cards, labels (0 = Yes), D20 weights,
    and per arm the scores, per framing the token counts, per model the scale. The rest is filled
    by :func:`score_items` (label-free diagnostics: ``collapse``, ``input_tokens``,
    ``ms_per_item``; ``candidate_vectors`` for the probe) and :func:`evaluate_task`
    (``p_loco``, each arm's full-run LOCO-Platt held-out probability, NaN where skipped;
    ``p_probe``, the candidate-only probe's)."""

    task: str
    task_id: str
    target: list[str]
    option: list[str]
    card: list[str]
    label: IntArray
    weight: FloatArray
    scores: dict[str, ArmScores] = field(default_factory=dict)
    tokens: dict[str, TokenCounts] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    collapse: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    input_tokens: dict[str, int] = field(default_factory=dict)
    ms_per_item: dict[str, float] = field(default_factory=dict)
    candidate_vectors: FloatArray | None = None
    p_loco: dict[str, FloatArray] = field(default_factory=dict)
    p_probe: FloatArray | None = None

    def __post_init__(self) -> None:
        for cached in _DERIVED:  # re-validation after an edit recomputes what depends on it
            self.__dict__.pop(cached, None)
        self.label = np.asarray(self.label, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        n = len(self.target)
        if not (len(self.option) == len(self.card) == len(self.label) == len(self.weight) == n):
            raise X1DataError(f"{self.task}: the item columns differ in length")
        if n == 0:
            raise X1DataError(f"{self.task}: no items")
        if self.task_id not in X1_TASKS:
            raise X1DataError(f"{self.task}: {self.task_id} is not an X1 task ({X1_TASKS})")
        if not np.all((self.label == 0) | (self.label == 1)):
            raise X1DataError(f"{self.task}: labels must be 0 (Yes) or 1 (No)")
        if not np.all(np.isfinite(self.weight)) or np.any(self.weight <= 0):
            raise X1DataError(f"{self.task}: weights must be positive")
        if len(set(zip(self.target, self.option, strict=True))) != n:
            raise X1DataError(f"{self.task}: (target, option) identities must be unique")
        for arm, sc in self.scores.items():
            split_arm(arm)
            sc.s = self._vector(sc.s, arm, "s")
            if sc.mean_context is not None:
                sc.mean_context = self._vector(sc.mean_context, arm, "mean_context")
            if sc.contexts is not None:
                sc.contexts = self._contexts(sc.contexts, sc.s, arm)
        for framing, tc in self.tokens.items():
            for what in ("context", "candidate", "anchor"):
                arr = np.asarray(getattr(tc, what), dtype=np.int64)
                if arr.shape != (n,):
                    raise X1DataError(f"{self.task}/{framing}: {what} tokens need one per item")
                setattr(tc, what, arr)

    def _vector(self, value: npt.ArrayLike, arm: str, what: str) -> FloatArray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.n,) or not np.all(np.isfinite(arr)):
            raise X1DataError(f"{self.task}/{arm}: {what} needs one finite value per item")
        return arr

    def _contexts(self, value: npt.ArrayLike, s: FloatArray, arm: str) -> FloatArray:
        arr = np.asarray(value, dtype=np.float64)
        sizes = self.card_size
        if arr.ndim != 2 or arr.shape[0] != self.n or arr.shape[1] < int(sizes.max()):
            raise X1DataError(f"{self.task}/{arm}: contexts must be [n, targets per card]")
        valid = np.arange(arr.shape[1])[None, :] < sizes[:, None]
        if not np.all(np.isfinite(arr[valid])):
            raise X1DataError(f"{self.task}/{arm}: a context score is not finite")
        own = arr[np.arange(self.n), self.slot]
        if not np.all(np.abs(own - s) <= 1e-9 * np.maximum(1.0, np.abs(s))):
            raise X1DataError(f"{self.task}/{arm}: an item's own context column is not its s")
        out: FloatArray = np.where(valid, arr, np.nan)
        return out

    @property
    def n(self) -> int:
        return len(self.target)

    @cached_property
    def card_arr(self) -> npt.NDArray[np.str_]:
        return np.asarray(self.card, dtype=str)

    @cached_property
    def positive(self) -> BoolArray:
        """``label == Yes`` (index 0)."""
        out: BoolArray = self.label == 0
        return out

    @cached_property
    def guard_task(self) -> Task:
        """A :class:`~mesa_clm.bench.tasks.base.Task` over the labels and cards only, so the
        30/5 guards are :func:`~mesa_clm.bench.tasks.base.fold_guard` itself (§6.2)."""
        items: list[tuple[dict[str, Any], int]] = [({}, int(v)) for v in self.label.tolist()]
        return Task(self.task, TASKS[self.task_id], items, "", "", cards=list(self.card))

    @cached_property
    def card_targets(self) -> dict[str, list[str]]:
        """Each card's distinct targets, sorted (the column order of ``contexts``)."""
        out: dict[str, set[str]] = {}
        for t, c in zip(self.target, self.card, strict=True):
            out.setdefault(c, set()).add(t)
        return {c: sorted(ts) for c, ts in sorted(out.items())}

    @cached_property
    def slot(self) -> IntArray:
        """The column of each item's own target among its card's targets."""
        index = {c: {t: j for j, t in enumerate(ts)} for c, ts in self.card_targets.items()}
        return np.asarray(
            [index[c][t] for t, c in zip(self.target, self.card, strict=True)], dtype=np.int64
        )

    @cached_property
    def card_size(self) -> IntArray:
        """The number of targets on each item's card."""
        sizes = {c: len(ts) for c, ts in self.card_targets.items()}
        return np.asarray([sizes[c] for c in self.card], dtype=np.int64)

    @property
    def n_unshuffled(self) -> int:
        """Items on a card with a single target, which keep their own context (§5.6)."""
        return int(np.sum(self.card_size < 2))

    def shuffle_index(self, k: int = SHUFFLE_K, seed: int = SHUFFLE_SEED) -> IntArray:
        """``[k, n]``: the ``contexts`` column each item takes in each of the ``k`` draws of
        :func:`shuffle_draws` (its own column on a single-target card)."""
        key = (k, seed)
        cache = self.__dict__.setdefault("_shuffle_index", {})
        if key not in cache:
            draws = shuffle_draws(
                {c: len(ts) for c, ts in self.card_targets.items()}, k=k, seed=seed
            )
            idx = np.tile(self.slot, (k, 1))
            for card, perm in draws.items():
                if perm is None:
                    continue
                on = self.card_arr == card
                idx[:, on] = perm[:, self.slot[on]]
            cache[key] = idx
        out: IntArray = cache[key]
        return out

    def shuffled(self, arm: str, k: int = SHUFFLE_K, seed: int = SHUFFLE_SEED) -> FloatArray:
        """``[k, n]``: the arm's score of every item under each of the ``k`` shuffles (§5.1)."""
        ctx = self.scores[arm].contexts
        if ctx is None:
            raise X1DataError(f"{self.task}/{arm}: no context scores to shuffle")
        idx = self.shuffle_index(k, seed)
        out: FloatArray = ctx[np.arange(self.n)[None, :], idx]
        return out

    def arms(self) -> list[str]:
        """The arms scored, in :data:`ARM_ORDER`."""
        return [a for a in ARM_ORDER if a in self.scores]

    def card_items(self, card: str) -> IntArray:
        out: IntArray = np.flatnonzero(self.card_arr == card).astype(np.int64)
        return out


# -- scoring from the feature store (§3, §5, §7.11, §11) -------------------------------------------


def context_scores(
    z_targets: npt.ArrayLike,
    target_of_item: npt.ArrayLike,
    card_of_item: Sequence[str],
    z_candidates: npt.ArrayLike,
    z_anchors: npt.ArrayLike,
    scale: float,
) -> FloatArray:
    """``[n, m]``: each item's candidate and anchor scored under every target context of its
    own card (§5.1), columns in the card's target order (``target_of_item`` indexes the rows of
    ``z_targets``, which are in sorted target order), NaN past the card's target count. The
    same arithmetic as the item's own ``s`` (:func:`~mesa_clm.learn.offline.pairwise_s_c`), so
    the own column equals it."""
    zt = np.asarray(z_targets)
    owner = np.asarray(target_of_item, dtype=np.int64)
    zc = np.asarray(z_candidates)
    za = np.asarray(z_anchors)
    cards = np.asarray(card_of_item, dtype=str)
    per_card = {c: sorted(set(owner[cards == c].tolist())) for c in sorted(set(cards.tolist()))}
    width = max(len(rows) for rows in per_card.values())
    out = np.full((len(owner), width), np.nan)
    for card, rows in per_card.items():
        items = np.flatnonzero(cards == card)
        for j, row in enumerate(rows):
            zs = np.broadcast_to(zt[row], (len(items), zt.shape[1]))
            out[items, j] = pairwise_s_c(scale, zs, zc[items], za[items], temperature=TEMPERATURE)
    return out


def mean_context_scores(
    z_targets: npt.ArrayLike, z_candidates: npt.ArrayLike, z_anchors: npt.ArrayLike, scale: float
) -> FloatArray:
    """The score at the mean context (§7.11, report-only): ``scale·(z̄·zc − z̄·za)`` with z̄ the
    L2-normalised mean of the distinct targets' state-side vectors, one fixed context for every
    item."""
    zt = np.asarray(z_targets, dtype=np.float64)
    mean = zt.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not norm > 0.0:
        raise X1DataError("the mean context has no direction")
    zbar = mean / norm
    zc = np.asarray(z_candidates, dtype=np.float64)
    za = np.asarray(z_anchors, dtype=np.float64)
    out: FloatArray = float(scale) * (zc @ zbar - za @ zbar) / TEMPERATURE
    return out


@dataclass(frozen=True)
class _FramingTexts:
    """Per item: its context, candidate and anchor texts under one framing (F1: context,
    noul_true, noul_false)."""

    context: list[str]
    candidate: list[str]
    anchor: list[str]


def _framing_texts(
    index: Mapping[tuple[str, str, str, str], str], framing: str, items: TaskItems
) -> tuple[_FramingTexts, list[str]]:
    """The manifest texts of every item under ``framing`` and the keys that are missing."""
    missing: list[str] = []
    ctx: list[str] = []
    cand: list[str] = []
    anc: list[str] = []

    def get(key: tuple[str, str, str, str]) -> str:
        text = index.get(key)
        if text is None:
            missing.append("/".join(k[:12] for k in key))
            return ""
        return text

    for target, option in zip(items.target, items.option, strict=True):
        if framing == CONTROL:
            ctx.append(get((framing, target, option, "context")))
            cand.append(get((framing, target, option, "noul_true")))
            anc.append(get((framing, target, option, "noul_false")))
        else:
            ctx.append(get((framing, target, "", "context")))
            cand.append(get((framing, target, option, "candidate")))
            anc.append(get((framing, target, ANCHOR_KEY, "anchor")))
    return _FramingTexts(ctx, cand, anc), missing


def score_items(
    task: Task,
    manifest: Manifest,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    *,
    framings: Sequence[str] = FRAMING_IDS,
    models: Sequence[str] = MODELS,
    clock: Clock = time.perf_counter,
) -> TaskItems:
    """Every arm's per-item scores for ``task`` (§2.2, §3, §5) and the label-free diagnostics
    (§11), from the manifest's texts and the store's vectors. An item without a manifest text, a
    vector or a token count raises :class:`X1DataError` before any score."""
    task_id = task.task_id
    if task_id not in X1_TASKS or task_id not in manifest.tasks:
        raise X1DataError(f"{task.name}: {task_id} is not in the X1 manifest ({manifest.tasks})")
    absent = [f for f in framings if f not in manifest.framings]
    if absent:
        raise X1DataError(f"the manifest has no rows for framing(s) {absent}")
    unknown = [m for m in models if m not in scorers or scorers[m].model != m]
    if unknown:
        raise X1DataError(f"no scorer for model(s) {unknown}")
    options = list(task.option_keys)
    if len(options) != len(task.items):
        raise X1DataError(f"{task.name}: every item needs its option_key")
    items = TaskItems(
        task=task.name,
        task_id=task_id,
        target=[target for target, _ in item_identities(task)],
        option=options,
        card=list(task.cards),
        label=np.asarray(task.labels, dtype=np.int64),
        weight=np.asarray(task.weights, dtype=np.float64),
    )
    index: dict[tuple[str, str, str, str], str] = {
        (r.framing_id, r.target_sha256, r.option_key, str(r.role)): r.text
        for r in manifest.rows
        if r.task_id == task_id
    }
    texts: dict[str, _FramingTexts] = {}
    for f in framings:
        texts[f], missing = _framing_texts(index, f, items)
        if missing:
            raise X1DataError(
                f"{task.name}/{f}: {len(missing)} manifest text(s) missing for the item set "
                f"(first: {missing[0]}); every framing must render every item (§2.2)"
            )
    try:
        every = list(
            dict.fromkeys(
                t for ft in texts.values() for t in (*ft.context, *ft.candidate, *ft.anchor)
            )
        )
        counts = dict(zip(every, store.token_counts(every), strict=True))
        for f, ft in texts.items():
            items.tokens[f] = TokenCounts(
                context=np.asarray([counts[t] for t in ft.context], dtype=np.int64),
                candidate=np.asarray([counts[t] for t in ft.candidate], dtype=np.int64),
                anchor=np.asarray([counts[t] for t in ft.anchor], dtype=np.int64),
            )
            needed = set(ft.context) | set(ft.candidate) | set(ft.anchor)
            items.input_tokens[f] = int(sum(counts[t] for t in needed))
        distinct_targets = sorted(set(items.target))
        row_of = {t: k for k, t in enumerate(distinct_targets)}
        owner = np.asarray([row_of[t] for t in items.target], dtype=np.int64)
        first_item = {t: items.target.index(t) for t in distinct_targets}
        for m in models:
            sc = scorers[m]
            scale = float(sc.scale)
            items.scale[m] = scale
            for f in framings:
                ft = texts[f]
                started = clock()
                zc = store_vectors(sc, store, "action", ft.candidate)
                za = store_vectors(sc, store, "action", ft.anchor)
                if f == CONTROL:
                    zs = store_vectors(sc, store, "state", ft.context)
                    s = pairwise_s_c(scale, zs, zc, za, temperature=TEMPERATURE)
                    items.scores[arm_id(f, m)] = ArmScores(s=s)
                else:
                    contexts = [ft.context[first_item[t]] for t in distinct_targets]
                    zt = store_vectors(sc, store, "state", contexts)
                    s = pairwise_s_c(scale, zt[owner], zc, za, temperature=TEMPERATURE)
                    items.scores[arm_id(f, m)] = ArmScores(
                        s=s,
                        contexts=context_scores(zt, owner, items.card, zc, za, scale),
                        mean_context=mean_context_scores(zt, zc, za, scale),
                    )
                items.ms_per_item[arm_id(f, m)] = 1000.0 * (clock() - started) / items.n
        latest = scorers.get(LATEST_MODEL)
        for f in framings:
            ft = texts[f]
            distinct = list(dict.fromkeys(ft.context))
            raw = l2(store.get(distinct))
            projected = (
                store_vectors(latest, store, "state", distinct) if latest is not None else None
            )
            items.collapse[f] = {
                "n": len(distinct),
                "raw": mean_pairwise_cosine(raw),
                "projected": None if projected is None else mean_pairwise_cosine(projected),
            }
        arm_framings = [f for f in framings if f != CONTROL]
        if arm_framings:
            cands = texts[arm_framings[0]].candidate
            items.candidate_vectors = np.asarray(store.get(cands), dtype=np.float64)
    except FeatureMissing as exc:
        raise X1DataError(f"{task.name}: {exc}") from exc
    items.__post_init__()
    return items


# -- the candidate-only probe (§7.11) -----------------------------------------------------------


def probe_folds(items: TaskItems) -> tuple[list[Fold], dict[str, str]]:
    """The candidate-only probe's leave-one-card-out folds (§7.11): the folds it fits and the
    skipped ones with their reasons (the 30/5 guards, then the probe floor of 40 training
    items). Labels and cards only, so the replay recomputes them from the items file."""
    task = items.guard_task
    run: list[Fold] = []
    skipped: dict[str, str] = {}
    for fold in task.leave_one_card_out():
        reason = fold_guard(task, fold)
        if reason is None:
            reason = floor_reason(len(fold.train), PROBE_FLOOR, "probe")
        if reason is not None:
            skipped[fold.held_out] = reason
        else:
            run.append(fold)
    return run, skipped


def candidate_probe(items: TaskItems, x: npt.ArrayLike) -> CandidateProbe:
    """The learned candidate-only probe (§7.11): the PR #13 recipe (``bench.x2``'s replica, no
    weights) fitted leave-one-card-out on the candidate vectors ``x`` (one row per item), the
    30/5 guards and the probe floor of 40 training items per fold; report-only. A fit that
    does not converge is used as it is and flagged in its fold record (``converged``,
    ``warnings``), never refitted or skipped (§7.11, §10.2)."""
    from mesa_clm.bench.x2 import replica_pipeline  # sklearn: the bench extra, imported late

    feats = np.asarray(x, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[0] != items.n:
        raise X1DataError(f"{items.task}: the probe needs one candidate vector per item")
    probs = np.full(items.n, np.nan)
    folds: dict[str, dict[str, Any]] = {}
    run, skipped = probe_folds(items)
    for fold in run:
        train = np.asarray(fold.train, dtype=np.int64)
        test = np.asarray(fold.test, dtype=np.int64)
        model = replica_pipeline()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(feats[train], items.label[train])
        proba = np.asarray(model.predict_proba(feats[test]), dtype=np.float64)
        probs[test] = proba[:, list(model.classes_).index(0)]
        n_iter = int(np.max(model[-1].n_iter_))
        folds[fold.held_out] = {
            "n": len(test),
            "n_train": len(train),
            "n_iter": n_iter,
            "converged": n_iter < model[-1].max_iter,
            "warnings": sorted({type(w.message).__name__ for w in caught}),
        }
    return CandidateProbe(probs, folds, skipped)


# -- Platt and the LOCO folds (§6) -------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def platt_fit(
    scores: npt.ArrayLike, positive: npt.ArrayLike, weights: npt.ArrayLike | None
) -> calibrate.PlattCalibrator:
    """X1's LOCO-Platt calibrator (§6.4): the code base's one weighted Platt,
    :func:`mesa_clm.learn.calibrate.fit_platt` on ``s`` with
    :data:`~mesa_clm.learn.calibrate.PLATT_TARGETS` (``"hard"``) and the label weights as sample
    weights (D20; ``None`` for the report-only unweighted sensitivity, §6.6), so X1's LOCO-Platt
    NLL is the calibrated tier's. ``a < 0`` is flagged ``inverted``, never fixed."""
    return calibrate.fit_platt(
        scores, positive, weights, feature="s_c", targets=calibrate.PLATT_TARGETS
    )


@dataclass(frozen=True)
class FoldPlan:
    """The leave-one-card-out folds over the items ``idx`` (§6.1–§6.3): the folds that run as
    ``(card, test, train)`` item indices, the skipped folds with their reasons (the 30/5 guards,
    then the floor of 100 training items), and ``pool``, the sorted items that get a held-out
    prediction. The plan depends on labels and cards only, so every arm of a task pools the same
    items (§6.2)."""

    idx: IntArray
    folds: tuple[tuple[str, IntArray, IntArray], ...]
    skipped: dict[str, str]

    @cached_property
    def pool(self) -> IntArray:
        if not self.folds:
            return np.zeros(0, dtype=np.int64)
        out: IntArray = np.sort(np.concatenate([test for _, test, _ in self.folds]))
        return out


def floor_reason(
    n_train: int, floor: int = CALIBRATION_FLOOR, what: str = "calibrated"
) -> str | None:
    """Why a fit on ``n_train`` items is below the floor (§6.3), or ``None``."""
    if n_train < floor:
        return f"below_floor {n_train} < {floor} training items ({what})"
    return None


def fold_plan(items: TaskItems, idx: npt.ArrayLike | None = None) -> FoldPlan:
    """The LOCO folds of ``items`` restricted to ``idx`` (all items by default): one per card,
    in sorted card order; a fold runs when it passes the 30/5 guards (:func:`fold_guard`) and
    its training items reach the floor of 100 (§6.2, §6.3)."""
    sel = _index(items, idx)
    cards = sorted(set(items.card_arr[sel].tolist()))
    folds: list[tuple[str, IntArray, IntArray]] = []
    skipped: dict[str, str] = {}
    for card in cards:
        on = items.card_arr[sel] == card
        test, train = sel[on], sel[~on]
        reason = fold_guard(items.guard_task, Fold(card, test.tolist(), train.tolist()))
        if reason is None:
            reason = floor_reason(len(train))
        if reason is None:
            folds.append((card, test, train))
        else:
            skipped[card] = reason
    return FoldPlan(sel, tuple(folds), skipped)


def _index(items: TaskItems, idx: npt.ArrayLike | None) -> IntArray:
    if idx is None:
        return np.arange(items.n, dtype=np.int64)
    sel = np.unique(np.asarray(idx, dtype=np.int64))
    if len(sel) == 0 or sel[0] < 0 or sel[-1] >= items.n:
        raise X1DataError(f"{items.task}: the item subset is empty or out of range")
    return sel


@dataclass(frozen=True)
class LocoPlatt:
    """One score's pooled leave-one-card-out Platt predictions over a :class:`FoldPlan` (§6.1):
    ``probs`` is p(Yes) for each item of ``plan.pool``, ``fits`` the calibrator per fold."""

    plan: FoldPlan
    probs: FloatArray
    fits: dict[str, calibrate.PlattCalibrator]

    def matrix(self) -> FloatArray:
        """``[p, 1 − p]`` per pooled item (what the metrics and rule R read)."""
        out: FloatArray = np.column_stack([self.probs, 1.0 - self.probs])
        return out


def loco_platt(
    items: TaskItems, scores: npt.ArrayLike, plan: FoldPlan, *, weighted: bool = True
) -> LocoPlatt:
    """Fit a Platt calibrator (§6.4; weighted by the label weights unless ``weighted`` is false,
    the report-only sensitivity of §6.6) on each fold's training items and predict its held-out
    card; predictions pooled, never fold-averaged (§6.1)."""
    x = np.asarray(scores, dtype=np.float64)
    if x.shape != (items.n,):
        raise X1DataError(f"{items.task}: one score per item expected")
    probs = np.full(items.n, np.nan)
    fits: dict[str, calibrate.PlattCalibrator] = {}
    for card, test, train in plan.folds:
        w = items.weight[train] if weighted else None
        fit = platt_fit(x[train], items.positive[train], w)
        probs[test] = fit.p_yes(x[test])
        fits[card] = fit
    pooled: FloatArray = probs[plan.pool]
    return LocoPlatt(plan, pooled, fits)


# -- evidence and the decision (§4–§7) -----------------------------------------------------------


class RuleRSummary(_Model):
    """A rule R verdict as stored: :class:`mesa_clm.bench.stats.RuleR` with the bootstrap's
    settings (``B``, ``seed``, ``alpha``; ``None`` on an unavailable comparison), which the
    replay holds to the run's configuration (§12.6 (a))."""

    metric: str
    passed: bool
    reason: str
    delta: float | None
    lower_bound: float | None
    upper_bound: float | None
    n_valid: int
    n_clusters: int
    m_c: int
    wins: int
    needed: int
    per_card: dict[str, float]
    B: int | None = None
    seed: int | None = None
    alpha: float | None = None

    @classmethod
    def of(cls, r: stats.RuleR) -> RuleRSummary:
        return cls(
            metric=r.metric,
            passed=r.passed,
            reason=r.reason,
            delta=finite(r.delta),
            lower_bound=finite(r.lower_bound),
            upper_bound=finite(r.bootstrap.upper),
            n_valid=r.bootstrap.n_valid,
            n_clusters=r.bootstrap.n_clusters,
            m_c=r.sign.m_c,
            wins=r.sign.wins,
            needed=r.sign.needed,
            per_card={k: float(v) for k, v in sorted(r.sign.per_card.items())},
            B=r.bootstrap.B,
            seed=r.bootstrap.seed,
            alpha=r.bootstrap.alpha,
        )

    @classmethod
    def unavailable(cls, metric: str, reason: str) -> RuleRSummary:
        """A comparison that cannot be made (no pooled item): never ``passed``."""
        return cls(
            metric=metric,
            passed=False,
            reason=reason,
            delta=None,
            lower_bound=None,
            upper_bound=None,
            n_valid=0,
            n_clusters=0,
            m_c=0,
            wins=0,
            needed=0,
            per_card={},
        )


UNAVAILABLE: Final[frozenset[str]] = frozenset({"no_pooled_items"})


def check_rule(r: RuleRSummary, where: str, settings: tuple[int, int, float] | None = None) -> None:
    """A stored rule R verdict must follow from its own numbers (§12.6): ``needed`` is
    ⌈0.8·m_c⌉, ``m_c`` and ``wins`` count its cards, and ``reason`` and ``passed`` are what
    rule R makes of ``m_c``, the lower bound and the wins; with ``settings`` (the run's ``(B,
    seed, alpha)``) its bootstrap must have used them. :class:`X1ReproError` otherwise."""
    if r.reason in UNAVAILABLE:
        if r.passed or r.per_card or r.m_c or r.wins:
            raise X1ReproError(f"{where}: an unavailable comparison carries a verdict")
        return
    if settings is not None and (r.B, r.seed, r.alpha) != settings:
        raise X1ReproError(
            f"{where}: its bootstrap used B/seed/alpha {r.B}/{r.seed}/{r.alpha}, the run's are "
            f"{settings[0]}/{settings[1]}/{settings[2]}"
        )
    sign = stats.sign_verdict(dict(r.per_card))
    if (r.m_c, r.wins, r.needed) != (sign.m_c, sign.wins, sign.needed):
        raise X1ReproError(
            f"{where}: m_c/wins/needed {r.m_c}/{r.wins}/{r.needed} do not follow from its "
            f"cards ({sign.m_c}/{sign.wins}/{sign.needed})"
        )
    lower = math.nan if r.lower_bound is None else r.lower_bound
    reason = stats.rule_r_reason(sign, lower)
    if r.reason != reason or r.passed != (reason == "passed"):
        raise X1ReproError(
            f"{where}: rule R says {r.reason!r}/{r.passed}, its numbers say {reason!r}"
        )


class ArmEvidence(_Model):
    """What the decision reads about one arm (§7): the AUROC of ``s`` and its cluster lower
    bound (§4), the shuffle's rule R (§5.4; ``None`` for F1), the pooled LOCO-Platt NLL (§6.7;
    ``None`` without a pooled item) and how many items it pools."""

    arm: str
    framing: str
    model: str
    auroc: float | None
    auroc_lower: float | None
    shuffle: RuleRSummary | None
    nll: float | None
    n_loco: int


class ArmVerdict(_Model):
    """Rule (1) for one arm: ``qualified`` (``None`` for the F1 control) and the conditions it
    failed (``auroc_lower_bound``, ``auroc_floor``, ``shuffle_rule_r``)."""

    evidence: ArmEvidence
    qualified: bool | None
    failed: list[str]


class Comparison(_Model):
    """One rule R comparison the decision consulted: "``a`` ≻ ``b`` on NLL" (§7.13)."""

    a: str
    b: str
    rule_r: RuleRSummary


class Choice(_Model):
    framing: str
    model: str
    arm: str


class Step2(_Model):
    """Rule (2) (§7.4–§7.7): the model whose arms were ranked, the ranking, the NLL winner, the
    candidates after the cacheability step and the framing chosen."""

    model: str
    ranked: list[str]
    winner: str
    candidates: list[str]
    framing: str
    reason: str


class Step3(_Model):
    """Rule (3) (§7.8)."""

    framing: str
    model: str
    compared: bool
    reason: str


class Step4(_Model):
    """Rule (4) (§7.9): per F1 arm, whether it is ≻ every eligible arm on NLL."""

    eligible: list[str]
    beats_all: dict[str, bool]
    d3_amendment_recommended: bool


class Decision(_Model):
    """The decision trace of one task (full run) or one nested fold (§7): every arm's verdict,
    the token counts the tie-break may read, the outcome and choice, the steps, and every rule R
    comparison consulted, in order. Self-contained: :class:`StoredEvidence` replays it."""

    rules: str = "X1 decision rules (1)-(5), DESIGN 'Pre-registration (G1)'; rule (6) withdrawn"
    plan: str = f"{PLAN} §7"
    arms: dict[str, ArmVerdict]
    tokens_per_target: dict[str, float | None]
    outcome: Outcome
    choice: Choice | None
    step2: Step2 | None = None
    step3: Step3 | None = None
    step4: Step4 | None = None
    comparisons: list[Comparison] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Evidence(Protocol):
    """What :func:`decide` reads: the arms available, each arm's evidence, the encoder tokens per
    target of a framing, and rule R on NLL between two arms (§7.13)."""

    def arms(self) -> Sequence[str]: ...

    def arm(self, arm: str) -> ArmEvidence: ...

    def tokens_per_target(self, framing: str) -> float | None: ...

    def compare_nll(self, a: str, b: str) -> RuleRSummary: ...


def _cost(framing: str, tokens: Mapping[str, float | None]) -> tuple[int, float, int]:
    """The cost order (§7.7): F7/F9 before F4, fewer tokens per target, then F7."""
    value = tokens.get(framing)
    return (
        0 if framing in MORE_CACHEABLE else 1,
        math.inf if value is None else float(value),
        0 if framing == "F7" else 1,
    )


def decide(evidence: Evidence) -> Decision:
    """X1's decision rules (1)-(5) for one task (§7), a pure function of ``evidence``."""
    present = set(evidence.arms())
    available = [a for a in ARM_ORDER if a in present]
    verdicts: dict[str, ArmVerdict] = {}
    for arm in available:
        ev = evidence.arm(arm)
        if ev.framing == CONTROL:
            verdicts[arm] = ArmVerdict(evidence=ev, qualified=None, failed=[])
            continue
        failed: list[str] = []
        if not (ev.auroc_lower is not None and ev.auroc_lower > AUROC_LOWER_GATE):
            failed.append("auroc_lower_bound")
        if not (ev.auroc is not None and ev.auroc >= AUROC_FLOOR):
            failed.append("auroc_floor")
        if not (ev.shuffle is not None and ev.shuffle.passed):
            failed.append("shuffle_rule_r")
        verdicts[arm] = ArmVerdict(evidence=ev, qualified=not failed, failed=failed)
    framings_present = [f for f in FRAMING_IDS if any(a.startswith(f"{f}@") for a in available)]
    tokens = {f: evidence.tokens_per_target(f) for f in framings_present}
    log: list[Comparison] = []
    memo: dict[tuple[str, str], bool] = {}

    def better(a: str, b: str) -> bool:
        if (a, b) not in memo:
            r = evidence.compare_nll(a, b)
            log.append(Comparison(a=a, b=b, rule_r=r))
            memo[(a, b)] = r.passed
        return memo[(a, b)]

    def nll(arm: str) -> float:
        value = verdicts[arm].evidence.nll
        if value is None:
            raise X1DataError(f"{arm}: no LOCO-Platt NLL")
        return value

    qualified = {
        m: [f for f in ARM_FRAMINGS if (v := verdicts.get(arm_id(f, m))) and v.qualified]
        for m in MODELS
    }
    eligible = [arm_id(f, m) for m in MODELS for f in qualified[m]]
    if not eligible:
        return Decision(
            arms=verdicts, tokens_per_target=tokens, outcome="K1", choice=None, notes=[K1_NOTE]
        )
    if any(verdicts[a].evidence.nll is None for a in eligible):
        return Decision(
            arms=verdicts,
            tokens_per_target=tokens,
            outcome="undecidable",
            choice=None,
            notes=[
                "eligible arms exist but LOCO-Platt pooled no item (guards or floor); rules "
                "(2)-(4) cannot rank, no choice is made (§7.12)"
            ],
        )
    m0 = next(m for m in MODELS if qualified[m])
    ranked = sorted(qualified[m0], key=lambda f: (nll(arm_id(f, m0)), _cost(f, tokens)))
    winner = ranked[0]
    if winner in MORE_CACHEABLE:
        # §7.5, the literal reading: no arm is more cacheable than F7 or F9, so the NLL winner
        # stands; the F7/F9 tie-break belongs to the cacheability step below.
        framing, candidates = winner, [winner]
        reason = f"{winner} has the lowest LOCO-Platt NLL; no arm is more cacheable"
    else:
        more = [f for f in MORE_CACHEABLE if f in qualified[m0]]
        candidates = [f for f in more if not better(arm_id(winner, m0), arm_id(f, m0))]
        if not candidates:
            framing = winner
            reason = (
                f"{winner} ≻ every more cacheable qualified arm on NLL"
                if more
                else f"{winner} is the only qualified arm"
            )
            candidates = [winner]
        else:
            framing, why = _resolve_f7_f9(candidates, m0, nll, tokens, better)
            reason = f"{winner} not ≻ {', '.join(candidates)} on NLL; {why}"
    step2 = Step2(
        model=m0,
        ranked=[arm_id(f, m0) for f in ranked],
        winner=arm_id(winner, m0),
        candidates=[arm_id(f, m0) for f in candidates],
        framing=framing,
        reason=reason,
    )
    step3 = _model_rule(framing, m0, qualified, better)
    choice = Choice(framing=framing, model=step3.model, arm=arm_id(framing, step3.model))
    beats: dict[str, bool] = {}
    for m in MODELS:
        control = arm_id(CONTROL, m)
        if control not in verdicts or verdicts[control].evidence.nll is None:
            continue
        results = [better(control, e) for e in eligible]
        beats[control] = all(results)
    step4 = Step4(eligible=eligible, beats_all=beats, d3_amendment_recommended=any(beats.values()))
    notes = []
    if step4.d3_amendment_recommended:
        notes.append(
            "rule (4): an F1 arm is ≻ every eligible arm on NLL; an amendment reopening D3 is "
            "recommended, nothing is adopted (D3: never automatically)"
        )
    return Decision(
        arms=verdicts,
        tokens_per_target=tokens,
        outcome="choice",
        choice=choice,
        step2=step2,
        step3=step3,
        step4=step4,
        comparisons=log,
        notes=notes,
    )


def _resolve_f7_f9(
    candidates: Sequence[str],
    model: str,
    nll: Callable[[str], float],
    tokens: Mapping[str, float | None],
    better: Callable[[str, str], bool],
) -> tuple[str, str]:
    """The cacheability step's choice among the more cacheable candidates (§7.6): one of them is
    taken; between F7 and F9 the lower-NLL one if it is ≻ the other on NLL, else ("F7/F9 tie")
    the one with fewer encoder tokens per target, then F7."""
    if len(candidates) == 1:
        return candidates[0], f"take the more cacheable {candidates[0]}"
    lead = min(candidates, key=lambda f: (nll(arm_id(f, model)), _cost(f, tokens)))
    other = next(f for f in candidates if f != lead)
    if better(arm_id(lead, model), arm_id(other, model)):
        return lead, f"{lead} ≻ {other} on NLL"
    t_lead, t_other = tokens.get(lead), tokens.get(other)
    if t_lead is not None and t_other is not None and t_lead != t_other:
        fewer = lead if t_lead < t_other else other
        return fewer, f"F7/F9 tie (neither ≻ on NLL): {fewer} has fewer encoder tokens/target"
    return "F7", "F7/F9 tie with equal (or unknown) tokens/target: F7"


def _model_rule(
    framing: str,
    m0: str,
    qualified: Mapping[str, Sequence[str]],
    better: Callable[[str, str], bool],
) -> Step3:
    """Rule (3) (§7.8): ``clm-latest`` unless ``clm-raw`` ≻ it on NLL at the chosen framing; an
    arm that did not qualify is never chosen."""
    if m0 != LATEST_MODEL:
        return Step3(
            framing=framing,
            model=m0,
            compared=False,
            reason=f"{LATEST_MODEL} has no qualified arm; {m0} is the only model with one",
        )
    if framing not in qualified.get(RAW_MODEL, ()):
        return Step3(
            framing=framing,
            model=LATEST_MODEL,
            compared=False,
            reason=f"{arm_id(framing, RAW_MODEL)} did not qualify (or was not run)",
        )
    if better(arm_id(framing, RAW_MODEL), arm_id(framing, LATEST_MODEL)):
        return Step3(
            framing=framing,
            model=RAW_MODEL,
            compared=True,
            reason=f"{RAW_MODEL} ≻ {LATEST_MODEL} on NLL at {framing}",
        )
    return Step3(
        framing=framing,
        model=LATEST_MODEL,
        compared=True,
        reason=f"{RAW_MODEL} not ≻ {LATEST_MODEL} on NLL at {framing}",
    )


class ItemEvidence:
    """:class:`Evidence` computed from per-item data over the items ``idx`` (all by default; a
    nested fold passes its 6 training cards' items): §4 AUROC CIs, §5.4 the shuffle's rule R,
    §6 LOCO-Platt over ``idx``'s own folds, §7.13 rule R on NLL. One cluster resample per item
    set is shared by every bootstrap on it (identical to drawing it per call: seed and sorted
    clusters fix it). Results are cached."""

    def __init__(
        self,
        items: TaskItems,
        idx: npt.ArrayLike | None = None,
        *,
        B: int = stats.DEFAULT_B,
        seed: int = stats.DEFAULT_SEED,
        alpha: float = stats.DEFAULT_ALPHA,
        shuffle_k: int = SHUFFLE_K,
        shuffle_seed: int = SHUFFLE_SEED,
    ) -> None:
        self.items = items
        self.plan = fold_plan(items, idx)
        self.idx = self.plan.idx
        self.B = B
        self.seed = seed
        self.alpha = alpha
        self.shuffle_k = shuffle_k
        self.shuffle_seed = shuffle_seed
        self._auroc: dict[tuple[str, str], stats.BootstrapCI] = {}
        self._shuffle: dict[str, tuple[stats.BootstrapCI, stats.RuleR]] = {}
        self._loco: dict[tuple[str, str, bool], LocoPlatt] = {}
        self._compare: dict[tuple[str, str], RuleRSummary] = {}

    @cached_property
    def _labels(self) -> IntArray:
        out: IntArray = self.items.label[self.idx]
        return out

    @cached_property
    def _clusters(self) -> list[str]:
        return [str(c) for c in self.items.card_arr[self.idx].tolist()]

    @cached_property
    def _weights(self) -> FloatArray:
        return stats.bootstrap_weights(self._clusters, B=self.B, seed=self.seed)

    @cached_property
    def _pool_weights(self) -> FloatArray:
        cards = [str(c) for c in self.items.card_arr[self.plan.pool].tolist()]
        return stats.bootstrap_weights(cards, B=self.B, seed=self.seed)

    def arms(self) -> list[str]:
        return self.items.arms()

    def _feature(self, arm: str, feature: str) -> FloatArray:
        sc = self.items.scores[arm]
        value = getattr(sc, feature)
        if value is None:
            raise X1DataError(f"{self.items.task}/{arm}: no {feature}")
        out: FloatArray = value
        return out

    def auroc(self, arm: str, feature: str = "s") -> stats.BootstrapCI:
        """The AUROC of ``feature`` (``s`` or ``mean_context``) over ``idx`` with its
        card-cluster CI (§4)."""
        key = (arm, feature)
        if key not in self._auroc:
            x = self._feature(arm, feature)[self.idx]
            self._auroc[key] = stats.auroc_ci(
                x,
                self._labels,
                self._clusters,
                B=self.B,
                seed=self.seed,
                alpha=self.alpha,
                weights=self._weights,
            )
        return self._auroc[key]

    def shuffle(self, arm: str) -> tuple[stats.BootstrapCI, stats.RuleR] | None:
        """The mean AUROC of the ``K`` shuffles with its CI, and rule R "s ≻ shuffle on AUROC"
        (§5.4); ``None`` for an arm without context scores (F1)."""
        if self.items.scores[arm].contexts is None:
            return None
        if arm not in self._shuffle:
            real = self._feature(arm, "s")[self.idx]
            shuffled = self.items.shuffled(arm, self.shuffle_k, self.shuffle_seed)[:, self.idx]
            ci = stats.mean_auroc_ci(
                shuffled,
                self._labels,
                self._clusters,
                B=self.B,
                seed=self.seed,
                alpha=self.alpha,
                weights=self._weights,
            )
            rule = stats.rule_r_auroc_mean(
                real,
                shuffled,
                self._labels,
                self._clusters,
                B=self.B,
                seed=self.seed,
                alpha=self.alpha,
                weights=self._weights,
            )
            self._shuffle[arm] = (ci, rule)
        return self._shuffle[arm]

    def loco(self, arm: str, feature: str = "s", *, weighted: bool = True) -> LocoPlatt:
        """LOCO-Platt of ``feature`` over ``idx``'s folds (§6)."""
        key = (arm, feature, weighted)
        if key not in self._loco:
            self._loco[key] = loco_platt(
                self.items, self._feature(arm, feature), self.plan, weighted=weighted
            )
        return self._loco[key]

    def loco_nll(self, arm: str, feature: str = "s") -> float | None:
        pool = self.plan.pool
        if len(pool) == 0:
            return None
        lp = self.loco(arm, feature)
        return finite(stats.metric_value("nll", lp.matrix(), self.items.label[pool]))

    def arm(self, arm: str) -> ArmEvidence:
        framing, model = split_arm(arm)
        ci = self.auroc(arm)
        sh = self.shuffle(arm)
        return ArmEvidence(
            arm=arm,
            framing=framing,
            model=model,
            auroc=finite(ci.point),
            auroc_lower=finite(ci.lower),
            shuffle=None if sh is None else RuleRSummary.of(sh[1]),
            nll=self.loco_nll(arm),
            n_loco=len(self.plan.pool),
        )

    def tokens_per_target(self, framing: str) -> float | None:
        """§7.7 over the targets of ``idx``."""
        return _tokens_per_target(self.items, framing, self.idx)

    def compare_nll(self, a: str, b: str) -> RuleRSummary:
        """Rule R "``a`` ≻ ``b`` on NLL" over the pooled LOCO-Platt predictions (§7.13)."""
        key = (a, b)
        if key not in self._compare:
            pool = self.plan.pool
            if len(pool) == 0:
                self._compare[key] = RuleRSummary.unavailable("nll", "no_pooled_items")
            else:
                cards = [str(c) for c in self.items.card_arr[pool].tolist()]
                rule = stats.rule_r(
                    "nll",
                    self.loco(a).matrix(),
                    self.loco(b).matrix(),
                    self.items.label[pool],
                    cards,
                    B=self.B,
                    seed=self.seed,
                    alpha=self.alpha,
                    weights=self._pool_weights,
                )
                self._compare[key] = RuleRSummary.of(rule)
        return self._compare[key]


def _target_tokens(items: TaskItems, framing: str, idx: IntArray) -> dict[str, int]:
    """Encoder tokens per target of ``framing`` over the items ``idx`` (§7.7, §11.2): F4/F7/F9
    the target's context and anchor once plus each item's candidate; F1 each item's context and
    its two noul texts (one request per candidate)."""
    tc = items.tokens[framing]
    out: dict[str, int] = {}
    seen: set[str] = set()
    for i in idx.tolist():
        target = items.target[i]
        if framing == CONTROL:
            total = int(tc.context[i] + tc.candidate[i] + tc.anchor[i])
        else:
            total = int(tc.candidate[i])
            if target not in seen:
                total += int(tc.context[i] + tc.anchor[i])
        seen.add(target)
        out[target] = out.get(target, 0) + total
    return out


def _tokens_per_target(items: TaskItems, framing: str, idx: IntArray) -> float | None:
    if framing not in items.tokens or len(idx) == 0:
        return None
    per = _target_tokens(items, framing, idx)
    return float(np.mean(np.asarray(list(per.values()), dtype=np.float64)))


class StoredEvidence:
    """:class:`Evidence` read back from a stored :class:`Decision` (§12.6 (a)): replaying
    :func:`decide` over it must give the stored decision again, bit for bit; a comparison the
    stored trace never consulted is an :class:`X1ReproError`."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def arms(self) -> list[str]:
        return list(self.decision.arms)

    def arm(self, arm: str) -> ArmEvidence:
        return self.decision.arms[arm].evidence

    def tokens_per_target(self, framing: str) -> float | None:
        if framing not in self.decision.tokens_per_target:
            raise X1ReproError(f"the stored decision has no token count for {framing}")
        return self.decision.tokens_per_target[framing]

    def compare_nll(self, a: str, b: str) -> RuleRSummary:
        for c in self.decision.comparisons:
            if (c.a, c.b) == (a, b):
                return c.rule_r
        raise X1ReproError(f"the stored decision never compared {a} with {b}")


# -- nesting (§8) ---------------------------------------------------------------------------------


class FoldChoice(_Model):
    """One outer fold's choice (§8.3): the inner outcome and, for a choice, the arm."""

    outcome: Outcome
    framing: str | None
    model: str | None
    arm: str | None


class FoldRecord(_Model):
    """One outer fold (§8.2): the inner decision on the 6 training cards."""

    card: str
    n_train: int
    n_test: int
    decision: Decision


class NestedRecord(_Model):
    """The nested selection (§8.2–§8.5): every outer fold's inner decision, ``fold_choices`` and
    the folds whose inner outcome chose no arm (``inner_k1``, ``inner_undecidable``)."""

    folds: dict[str, FoldRecord]
    fold_choices: dict[str, FoldChoice]
    skipped_folds: dict[str, str]


def _inner_skip(outcome: str) -> str | None:
    return {"K1": "inner_k1", "undecidable": "inner_undecidable"}.get(outcome)


def nested_selection(
    items: TaskItems,
    *,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
    shuffle_k: int = SHUFFLE_K,
    shuffle_seed: int = SHUFFLE_SEED,
) -> NestedRecord:
    """The whole decision re-made inside each outer fold on its 6 training cards (§8.2) and the
    fold's choice (§8.3); a fold whose inner outcome is K1 or undecidable chooses no arm (§8.5).
    Nothing here reads the held-out card."""
    folds: dict[str, FoldRecord] = {}
    choices: dict[str, FoldChoice] = {}
    skipped: dict[str, str] = {}
    for card in sorted(set(items.card)):
        test = items.card_items(card)
        train = np.flatnonzero(items.card_arr != card).astype(np.int64)
        decision = decide(
            ItemEvidence(
                items,
                train,
                B=B,
                seed=seed,
                alpha=alpha,
                shuffle_k=shuffle_k,
                shuffle_seed=shuffle_seed,
            )
        )
        folds[card] = FoldRecord(card=card, n_train=len(train), n_test=len(test), decision=decision)
        c = decision.choice
        choices[card] = FoldChoice(
            outcome=decision.outcome,
            framing=None if c is None else c.framing,
            model=None if c is None else c.model,
            arm=None if c is None else c.arm,
        )
        why = _inner_skip(decision.outcome)
        if why is not None:
            skipped[card] = why
    return NestedRecord(folds=folds, fold_choices=choices, skipped_folds=skipped)


# -- records, cells and the results file (§12) ------------------------------------------------


class ShuffleRecord(_Model):
    """§5: the ``k`` shuffles' mean AUROC with its CI and rule R "s ≻ shuffle on AUROC"."""

    k: int
    seed: int
    auroc: ClusterCI
    rule_r: RuleRSummary
    n_unshuffled: int


class MeanContextRecord(_Model):
    """§7.11 (report-only): the score at the mean context, its AUROC and LOCO-Platt metrics."""

    auroc: ClusterCI
    n_loco: int
    nll: float | None
    acc: float | None
    ece: float | None


class CalibrationSummary(_Model):
    """Pooled LOCO-Platt metrics of one fit (§6.6): NLL, ECE, accuracy, Brier and the
    calibration in the large, mean p(Yes) minus the Yes rate."""

    nll: float | None
    ece: float | None
    acc: float | None
    brier: float | None
    mean_p_minus_rate: float | None


class SensitivityRecord(_Model):
    """§6.6 (report-only): the same folds with the D20-weighted Platt (the arm's) and with an
    unweighted one, so the intercept's lean towards Yes can be read off."""

    n: int
    weighted: CalibrationSummary
    unweighted: CalibrationSummary


class LocoRecord(_Model):
    """§6: the arm's LOCO-Platt folds, skips and pooled NLL (the full metrics are its cell)."""

    n: int
    nll: float | None
    folds: dict[str, calibrate.PlattCalibrator]
    skipped: dict[str, str]
    inverted: list[str]


class ArmRecord(_Model):
    """One arm of the full run (§12.1): its statistics, its cell and its diagnostics."""

    arm: str
    framing: str
    model: str
    question_key: str
    scale: float | None
    cell: str
    auroc: ClusterCI
    shuffle: ShuffleRecord | None
    mean_context: MeanContextRecord | None
    loco: LocoRecord
    sensitivity: SensitivityRecord | None
    qualified: bool | None
    failed: list[str]
    diagnostics: dict[str, Any]


class CandidateProbeRecord(_Model):
    """§7.11 (report-only): the learned candidate-only probe of one task."""

    recipe: str = PROBE_RECIPE
    n: int
    n_folds: int
    skipped: dict[str, str]
    auroc: ClusterCI | None
    nll: float | None
    acc: float | None
    ece: float | None
    brier: float | None
    folds: dict[str, dict[str, Any]]


class ShuffleDraws(_Model):
    """§5.2: the draws every arm of the task shares."""

    k: int
    seed: int
    cards_with_one_target: list[str]
    n_unshuffled: int
    index_sha256: str


class TaskRecord(_Model):
    """One task of the X1 run: the item summary, every arm, the full decision (A1's choice for
    this task) and the nested folds (either is ``None`` when the run left it out)."""

    task: str
    task_id: str
    task_key: str
    n: int
    n_targets: int
    class_counts: dict[int, int]
    cards: dict[str, int]
    min_weight: float | None
    label_sources: dict[str, int]
    excluded: dict[str, int]
    scales: dict[str, float]
    loco_skipped: dict[str, str]
    collapse: dict[str, dict[str, float | int | None]]
    shuffle: ShuffleDraws
    candidate_probe: CandidateProbeRecord | None
    arms: dict[str, ArmRecord]
    decision: Decision | None
    a1: Choice | None
    nested: NestedRecord | None


class X1Config(_Model):
    """The run's configuration (§13): what was run (tasks, framings, models, the full run and the
    nesting), the bootstraps and the shuffle draws, and every constant :func:`decide` and the
    fitters apply. :func:`registered_config` is the pre-registered one."""

    B: int
    seed: int
    alpha: float
    tasks: list[str]
    framings: list[str]
    models: list[str]
    full: bool
    nested: bool
    shuffle_k: int
    shuffle_seed: int
    auroc_floor: float = AUROC_FLOOR
    auroc_lower_gate: float = AUROC_LOWER_GATE
    sign_min_items: int = stats.SIGN_MIN_ITEMS
    sign_fraction: float = stats.SIGN_FRACTION
    min_clusters: int = stats.MIN_CLUSTERS
    min_train_per_class: int = MIN_TRAIN_PER_CLASS
    min_heldout_per_class: int = MIN_HELDOUT_PER_CLASS
    calibration_floor: int = CALIBRATION_FLOOR
    probe_floor: int = PROBE_FLOOR
    temperature: float = TEMPERATURE
    fitters: dict[str, Any] = Field(default_factory=lambda: _jsonable(calibrate.FITTER_CONSTANTS))
    shuffle: str = (
        "k seeded uniform within-card derangements of the targets (one default_rng(seed) stream, "
        "cards sorted, rejection sampling); ΔAUROC = AUROC(s) − mean_k AUROC(s_k) per replicate"
    )
    mean_context: str = "context replaced by the L2-normalised mean context of the task"
    candidate_probe: str = PROBE_RECIPE


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


_RUN_PARAMETERS: Final[tuple[str, ...]] = (
    "B",
    "seed",
    "alpha",
    "tasks",
    "framings",
    "models",
    "full",
    "nested",
    "shuffle_k",
    "shuffle_seed",
)


def registered_config() -> X1Config:
    """The pre-registered configuration (§13.1): both tasks, the four framings, both models, the
    full run and the nesting, B=2000, seed 0, alpha 0.05, 200 shuffles with seed 0."""
    r = reg.current()
    return X1Config(
        B=r.B,
        seed=r.seed,
        alpha=r.alpha,
        tasks=list(TASK_IDS),
        framings=list(FRAMING_IDS),
        models=list(MODELS),
        full=True,
        nested=True,
        shuffle_k=SHUFFLE_K,
        shuffle_seed=SHUFFLE_SEED,
    )


def config_deviations(cfg: X1Config) -> list[str]:
    """How a run's parameters depart from :func:`registered_config` (empty: none)."""
    want = registered_config()
    return [
        f"{name} {getattr(cfg, name)!r} is not {getattr(want, name)!r}"
        for name in _RUN_PARAMETERS
        if getattr(cfg, name) != getattr(want, name)
    ]


def configured_arms(cfg: X1Config) -> list[str]:
    """The arms a configuration runs, in :data:`ARM_ORDER`: every one of its framings under every
    one of its models (§1.5). :class:`X1DataError` for a framing or model X1 does not have, or
    one named twice."""
    unknown = [f for f in cfg.framings if f not in FRAMING_IDS] + [
        m for m in cfg.models if m not in MODELS
    ]
    twice = len(set(cfg.framings)) != len(cfg.framings) or len(set(cfg.models)) != len(cfg.models)
    if unknown or twice or not cfg.framings or not cfg.models:
        raise X1DataError(
            f"framings {cfg.framings} and models {cfg.models} are not a grid of X1's "
            f"{list(FRAMING_IDS)} × {list(MODELS)}"
        )
    return [
        a for a in ARM_ORDER if split_arm(a)[0] in cfg.framings and split_arm(a)[1] in cfg.models
    ]


def x1_deviations(
    cfg: X1Config,
    labels_sha256: str | None,
    labels_content_sha256: str | None,
    framings_lock_sha: str | None,
    fingerprints: Mapping[str, Mapping[str, str] | None] | None,
) -> list[str]:
    """Every way a run departs from the registration (§13.2; empty: it is the registered run):
    its parameters (:func:`config_deviations`), its labels
    (:func:`mesa_clm.bench.registered.data_deviations`), and the framings lock and each model's
    fingerprint it scored under (:func:`mesa_clm.bench.registered.identity_deviations`). The
    producer applies it to what it ran on and ``decide_from_json`` to what ``x1.json`` records,
    so both derive the same list."""
    return (
        config_deviations(cfg)
        + reg.data_deviations(labels_sha256, labels_content_sha256)
        + reg.identity_deviations(
            framings_lock_sha=framings_lock_sha, fingerprints=fingerprints, models=cfg.models
        )
    )


def _check_constants(cfg: X1Config) -> None:
    """Everything but the run parameters must be this code's constants (§12.6)."""
    want = registered_config().model_copy(
        update={name: getattr(cfg, name) for name in _RUN_PARAMETERS}
    )
    if cfg != want:
        raise X1ReproError(
            "the stored run used other constants than this code applies: "
            f"{_diff(cfg.model_dump(), want.model_dump(), 0.0, 'config')}"
        )


class ItemsFile(_Model):
    name: str
    sha256: str
    rows: int


class SnapshotInfo(_Model):
    """The labels a run read (§1.1)."""

    path: str | None
    labels_sha256: str
    labels_content_sha256: str


class X1Block(_Model):
    """The ``x1`` block of ``x1.json`` (§12.1)."""

    plan: str = PLAN
    config: X1Config
    registered: bool
    deviations: list[str]
    snapshot: SnapshotInfo
    framings_lock_sha: str
    question_keys: dict[str, dict[str, str]]
    models: dict[str, dict[str, Any]]
    store: dict[str, Any]
    manifest: dict[str, Any]
    latency: dict[str, Any] | None
    items_file: ItemsFile | None
    tasks: dict[str, TaskRecord]


class X1Results(_Model):
    """``bench/results/<date>/x1.json``: the :class:`~mesa_clm.bench.results.BenchResults`
    fields (the 16 arm cells, §12.1) plus the ``x1`` block. :func:`x1_cells` reads the cells back as
    a ``BenchResults``."""

    format: str = FORMAT
    date: str
    name: str
    mesa_clm: str
    labels_sha256: str
    labels_content_sha256: str
    environment: dict[str, Any]
    cells: dict[str, BenchCell]
    notes: list[str] = Field(default_factory=list)
    x1: X1Block


@dataclass
class X1Run:
    """A finished run: the results and the per-item tables :func:`write_x1` stores."""

    results: X1Results
    items: dict[str, TaskItems]


def _ci(ci: stats.BootstrapCI) -> ClusterCI:
    return ClusterCI(
        point=finite(ci.point),
        lower=finite(ci.lower),
        upper=finite(ci.upper),
        B=ci.B,
        seed=ci.seed,
        alpha=ci.alpha,
        n_clusters=ci.n_clusters,
        n_valid=ci.n_valid,
    )


def _check_task_items(task: Task, items: TaskItems) -> None:
    """The item table must be the task's, item for item: identities (target, option), cards,
    labels and weights, and the task at the shipped policy's ``min_weight`` (D9)."""
    if items.task != task.name or items.task_id != task.task_id or items.n != len(task.items):
        raise X1DataError(f"{task.name}: the item table is not this task's")
    if items.label.tolist() != task.labels or items.card != list(task.cards):
        raise X1DataError(f"{task.name}: the item table's labels or cards differ from the task's")
    targets = [t for t, _ in item_identities(task)]
    if items.target != targets or items.option != list(task.option_keys):
        raise X1DataError(f"{task.name}: the item table's targets or options are not the task's")
    if items.weight.tolist() != [float(w) for w in task.weights]:
        raise X1DataError(f"{task.name}: the item table's weights are not the task's (D20)")
    want = min_weight_for(task.task_id)
    if task.meta.get("min_weight") != want:
        raise X1DataError(
            f"{task.name}: min_weight {task.meta.get('min_weight')} is not the shipped "
            f"policy's {want} (D9)"
        )


def _summary(p_yes: FloatArray, labels: IntArray) -> CalibrationSummary:
    if len(p_yes) == 0:
        return CalibrationSummary(nll=None, ece=None, acc=None, brier=None, mean_p_minus_rate=None)
    probs = np.column_stack([p_yes, 1.0 - p_yes])
    return CalibrationSummary(
        nll=finite(stats.metric_value("nll", probs, labels)),
        ece=finite(stats.metric_value("ece", probs, labels)),
        acc=finite(stats.metric_value("acc", probs, labels)),
        brier=finite(stats.metric_value("brier", probs, labels)),
        mean_p_minus_rate=finite(float(np.mean(p_yes)) - float(np.mean(labels == 0))),
    )


def _arm_diagnostics(
    items: TaskItems,
    framing: str,
    model: str,
    latency: Sequence[float] | None,
    flags: Mapping[str, Sequence[bool]] | None = None,
) -> dict[str, Any]:
    """The frozen cell diagnostics in the units every M2 cell uses (§9.10) and X1's own (§11):
    ``mean_state_cos`` on the model's side, ``calls`` and ``input_tokens`` in total over the
    cell's items, ``ms_per_decision`` of the offline replay; per target the calls, the encoder
    tokens and the context tokens; the collapse diagnostic on both sides; the live timing run's
    ms per target (``flags``: its ``first`` and ``new_text`` flags, §11.3)."""
    arm = arm_id(framing, model)
    every = np.arange(items.n, dtype=np.int64)
    per_target = _target_tokens(items, framing, every) if framing in items.tokens else {}
    tc = items.tokens.get(framing)
    n_targets = len(set(items.target))
    calls = items.n if framing == CONTROL else n_targets
    context = None
    if tc is not None:
        if framing == CONTROL:
            context = float(tc.context.sum()) / n_targets
        else:
            first = {t: i for i, t in reversed(list(enumerate(items.target)))}
            context = float(np.mean([tc.context[i] for i in first.values()]))
    collapse = dict(items.collapse.get(framing, {}))
    side = "projected" if model == LATEST_MODEL else "raw"
    lat = (
        latency_summary(latency, (flags or {}).get("first"), (flags or {}).get("new_text"))
        if latency
        else None
    )
    return {
        "mean_state_cos": collapse.get(side),
        "calls": calls,
        "input_tokens": items.input_tokens.get(framing),
        "ms_per_decision": items.ms_per_item.get(arm),
        "timing_source": TIMING_SOURCE,
        "calls_per_target": calls / n_targets,
        "input_tokens_per_target": float(np.mean(list(per_target.values())))
        if per_target
        else None,
        "context_tokens_per_target": context,
        "collapse": collapse or None,
        "latency": lat,
    }


def _probe_record(
    items: TaskItems,
    probe: CandidateProbe,
    *,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
) -> CandidateProbeRecord:
    """The probe's record (§7.11): its pooled held-out p(Yes) (``items.p_probe``) scored for
    Yes (index 0), with the run's bootstrap settings."""
    pooled = np.flatnonzero(~np.isnan(probe.probs))
    labels = items.label[pooled]
    p = probe.probs[pooled]
    auroc = None
    if len(pooled) and len(set(items.card_arr[pooled].tolist())) >= 2:
        auroc = _ci(
            stats.metric_ci(
                "auroc",
                np.column_stack([p, 1.0 - p]),
                labels,
                [str(c) for c in items.card_arr[pooled].tolist()],
                B=B,
                seed=seed,
                alpha=alpha,
            )
        )
    s = _summary(p, labels)
    return CandidateProbeRecord(
        n=len(pooled),
        n_folds=len(probe.folds),
        skipped=dict(probe.skipped),
        auroc=auroc,
        nll=s.nll,
        acc=s.acc,
        ece=s.ece,
        brier=s.brier,
        folds={k: dict(v) for k, v in probe.folds.items()},
    )


def _shuffle_draws_record(items: TaskItems, k: int, seed: int) -> ShuffleDraws:
    idx = items.shuffle_index(k, seed)
    return ShuffleDraws(
        k=k,
        seed=seed,
        cards_with_one_target=[c for c, ts in items.card_targets.items() if len(ts) < 2],
        n_unshuffled=items.n_unshuffled,
        index_sha256=hashlib.sha256(np.ascontiguousarray(idx, dtype="<i8").tobytes()).hexdigest(),
    )


def evaluate_task(
    task: Task,
    items: TaskItems,
    *,
    labels_sha256: str,
    labels_content_sha256: str,
    registered: bool,
    fingerprints: Mapping[str, Mapping[str, str]] | None = None,
    latency_ms: Mapping[str, Sequence[float]] | None = None,
    latency_flags: Mapping[str, Mapping[str, Sequence[bool]]] | None = None,
    full: bool = True,
    nested: bool = True,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
    shuffle_k: int = SHUFFLE_K,
    shuffle_seed: int = SHUFFLE_SEED,
) -> tuple[TaskRecord, dict[str, BenchCell]]:
    """§4–§11 for one task over its item table: with ``full`` every arm's statistics and cell
    and the full decision (A1); with ``nested`` the nested selection. ``latency_ms`` maps an arm
    id to the ms per target of the live timing run (§11.3) and ``latency_flags`` to its
    ``first`` and ``new_text`` flags; ``fingerprints`` a model to its D5 bundle; ``registered``
    stamps the cells (``pre_registered``) and needs the full run and the nesting over every arm
    of the grid."""
    if not (full or nested):
        raise X1DataError("nothing to run: neither the full run nor the nesting")
    _check_task_items(task, items)
    if registered and (not (full and nested) or set(items.scores) != set(ARM_ORDER)):
        raise X1DataError(
            f"{task.name}: a registered run is the full run and the nesting over every arm of "
            f"the grid ({', '.join(ARM_ORDER)}; §13.1)"
        )
    ev = ItemEvidence(
        items, B=B, seed=seed, alpha=alpha, shuffle_k=shuffle_k, shuffle_seed=shuffle_seed
    )
    arms: dict[str, ArmRecord] = {}
    cells: dict[str, BenchCell] = {}
    decision: Decision | None = None
    probe: CandidateProbeRecord | None = None
    pool = ev.plan.pool
    if full:
        decision = decide(ev)
        if items.candidate_vectors is not None:
            cp = candidate_probe(items, items.candidate_vectors)
            items.p_probe = cp.probs
            probe = _probe_record(items, cp, B=B, seed=seed, alpha=alpha)
        for arm in ev.arms():
            arm_record, cell = _arm(
                task,
                items,
                ev,
                decision,
                arm,
                pool,
                fingerprints=fingerprints,
                latency_ms=latency_ms,
                latency_flags=latency_flags,
                labels_sha256=labels_sha256,
                labels_content_sha256=labels_content_sha256,
                registered=registered,
                B=B,
                seed=seed,
            )
            arms[arm] = arm_record
            cells[cell.key] = cell
    nested_rec = (
        nested_selection(
            items, B=B, seed=seed, alpha=alpha, shuffle_k=shuffle_k, shuffle_seed=shuffle_seed
        )
        if nested
        else None
    )
    record = TaskRecord(
        task=task.name,
        task_id=task.task_id,
        task_key=task.spec.key,
        n=items.n,
        n_targets=len(set(items.target)),
        class_counts=task.class_counts(),
        cards={c: int(np.sum(items.card_arr == c)) for c in sorted(set(items.card))},
        min_weight=task.meta.get("min_weight"),
        label_sources=dict(task.meta.get("label_sources", {})),
        excluded=dict(task.meta.get("excluded", {})),
        scales=dict(items.scale),
        loco_skipped=dict(ev.plan.skipped),
        collapse={f: dict(v) for f, v in items.collapse.items()},
        shuffle=_shuffle_draws_record(items, shuffle_k, shuffle_seed),
        candidate_probe=probe,
        arms=arms,
        decision=decision,
        a1=None if decision is None else decision.choice,
        nested=nested_rec,
    )
    return record, cells


def _arm(
    task: Task,
    items: TaskItems,
    ev: ItemEvidence,
    decision: Decision,
    arm: str,
    pool: IntArray,
    *,
    fingerprints: Mapping[str, Mapping[str, str]] | None,
    latency_ms: Mapping[str, Sequence[float]] | None,
    latency_flags: Mapping[str, Mapping[str, Sequence[bool]]] | None = None,
    labels_sha256: str,
    labels_content_sha256: str,
    registered: bool,
    B: int,
    seed: int,
) -> tuple[ArmRecord, BenchCell]:
    """One arm's record and its full-run cell (§12.1)."""
    framing, model = split_arm(arm)
    qkey = fr.framing(task.task_id, framing).question_key
    st = _arm_stats(items, ev, arm)
    p_full = st.p_full
    items.p_loco[arm] = p_full
    verdict = decision.arms[arm]
    diagnostics = _arm_diagnostics(
        items,
        framing,
        model,
        None if latency_ms is None else latency_ms.get(arm),
        None if latency_flags is None else latency_flags.get(arm),
    )
    inverted = st.loco.inverted
    mean_ctx = st.mean_context
    pooled = np.flatnonzero(~np.isnan(p_full))
    if not np.array_equal(pooled, pool):
        raise X1DataError(f"{task.name}/{arm}: the pooled items are not the fold plan's")
    folds = [
        Fold(c, items.card_items(c).tolist(), np.flatnonzero(items.card_arr != c).tolist())
        for c in sorted({items.card[i] for i in pooled.tolist()})
    ]
    cell = assemble_cell(
        task,
        pooled.tolist(),
        np.column_stack([p_full[pooled], 1.0 - p_full[pooled]]),
        folds,
        tier="calibrated",
        framing=framing,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        selection="full",
        pre_registered=registered,
        exploratory=True,
        servable=framing != CONTROL,
        variant=model,
        model=model,
        question_key=qkey,
        fingerprint=None if fingerprints is None else fingerprints.get(model),
        skipped_folds=ev.plan.skipped,
        fold_choices={},
        diagnostics={
            "x1_arm": arm,
            "auroc_s": st.auroc.model_dump(),
            "qualified": verdict.qualified,
            "failed": list(verdict.failed),
            "shuffle_auroc": None if st.shuffle is None else st.shuffle.auroc.model_dump(),
            "shuffle_rule_r": None if st.shuffle is None else st.shuffle.rule_r.model_dump(),
            "mean_context_auroc": None if mean_ctx is None else mean_ctx.auroc.model_dump(),
            "inverted_folds": inverted,
            "calibration": "platt",
            "fitters": _jsonable(calibrate.FITTER_CONSTANTS),
            **diagnostics,
        },
        notes=(
            f"X1 arm {arm}, full run on all cards: LOCO-Platt (weighted Platt per fold); "
            "exploratory (a full-data selection, D27); the X1 statistics are in diagnostics"
        ),
        B=B,
        seed=seed,
    )
    record = ArmRecord(
        arm=arm,
        framing=framing,
        model=model,
        question_key=qkey,
        scale=items.scale.get(model),
        cell=cell.key,
        auroc=st.auroc,
        shuffle=st.shuffle,
        mean_context=mean_ctx,
        loco=st.loco,
        sensitivity=st.sensitivity,
        qualified=verdict.qualified,
        failed=list(verdict.failed),
        diagnostics=diagnostics,
    )
    return record, cell


@dataclass(frozen=True)
class _ArmStats:
    """One arm's full-run statistics as :class:`ArmRecord` stores them, and ``p_full`` its
    LOCO-Platt p(Yes) per item (NaN off the pool)."""

    auroc: ClusterCI
    shuffle: ShuffleRecord | None
    mean_context: MeanContextRecord | None
    loco: LocoRecord
    sensitivity: SensitivityRecord | None
    p_full: FloatArray

    def records(self) -> dict[str, Any]:
        """The record fields, as JSON (what the recompute compares, §12.6 (b))."""
        return {
            name: None if value is None else value.model_dump(mode="json")
            for name, value in (
                ("auroc", self.auroc),
                ("shuffle", self.shuffle),
                ("mean_context", self.mean_context),
                ("loco", self.loco),
                ("sensitivity", self.sensitivity),
            )
        }


def _arm_stats(items: TaskItems, ev: ItemEvidence, arm: str) -> _ArmStats:
    """One arm's AUROC with its CI (§4), its shuffle record (§5), its mean-context record
    (§7.11), its LOCO-Platt record and §6.6's sensitivity, from the per-item data alone: what
    the run writes and what ``decide_from_json`` recomputes from the items file."""
    pool = ev.plan.pool
    lp = ev.loco(arm)
    p_full = np.full(items.n, np.nan)
    p_full[pool] = lp.probs
    sh = ev.shuffle(arm)
    labels = items.label[pool]
    mean_ctx: MeanContextRecord | None = None
    if items.scores[arm].mean_context is not None:
        s = _summary(ev.loco(arm, "mean_context").probs, labels)
        mean_ctx = MeanContextRecord(
            auroc=_ci(ev.auroc(arm, "mean_context")),
            n_loco=len(pool),
            nll=s.nll,
            acc=s.acc,
            ece=s.ece,
        )
    sensitivity = None
    if len(pool):
        sensitivity = SensitivityRecord(
            n=len(pool),
            weighted=_summary(lp.probs, labels),
            unweighted=_summary(ev.loco(arm, weighted=False).probs, labels),
        )
    return _ArmStats(
        auroc=_ci(ev.auroc(arm)),
        shuffle=None
        if sh is None
        else ShuffleRecord(
            k=ev.shuffle_k,
            seed=ev.shuffle_seed,
            auroc=_ci(sh[0]),
            rule_r=RuleRSummary.of(sh[1]),
            n_unshuffled=items.n_unshuffled,
        ),
        mean_context=mean_ctx,
        loco=LocoRecord(
            n=len(pool),
            nll=ev.loco_nll(arm),
            folds=dict(lp.fits),
            skipped=dict(ev.plan.skipped),
            inverted=sorted(c for c, f in lp.fits.items() if f.inverted),
        ),
        sensitivity=sensitivity,
        p_full=p_full,
    )


def _question_keys(
    framing_ids: Sequence[str], task_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    return {
        task_id: {f: fr.framing(task_id, f).question_key for f in framing_ids}
        for task_id in task_ids
    }


def run_x1_items(
    tasks: Mapping[str, Task],
    items: Mapping[str, TaskItems],
    *,
    date: str,
    labels_sha256: str,
    labels_content_sha256: str,
    snapshot: str | None = None,
    fingerprints: Mapping[str, Mapping[str, str]] | None = None,
    latency: tuple[Mapping[str, Mapping[str, Sequence[float]]], Mapping[str, Any]] | None = None,
    store_info: Mapping[str, Any] | None = None,
    manifest_info: Mapping[str, Any] | None = None,
    framings: Sequence[str] = FRAMING_IDS,
    models: Sequence[str] = MODELS,
    full: bool = True,
    nested: bool = True,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
    shuffle_k: int = SHUFFLE_K,
    shuffle_seed: int = SHUFFLE_SEED,
    name: str = NAME,
) -> X1Run:
    """X1 over item tables already scored (:func:`score_items`, or synthetic ones in tests):
    :func:`evaluate_task` per task, assembled into an :class:`X1Run`. ``latency`` is
    :func:`load_latency`'s ``(ms, info)``. The run is ``registered`` when its configuration is
    :func:`registered_config` and its labels the registered snapshot's
    (:func:`mesa_clm.bench.registered.data_deviations`); otherwise every cell is
    ``pre_registered: false`` and the deviations are recorded (§13). Labels that claim to be the
    registered snapshot must give its published counts per task
    (:func:`mesa_clm.bench.registered.check_task`; ``RegistrationError`` otherwise, §1.3)."""
    task_ids = [t for t in X1_TASKS if TASK_NAMES[t] in tasks]
    if not task_ids:
        raise X1DataError("no X1 task to run")
    config = X1Config(
        B=B,
        seed=seed,
        alpha=alpha,
        tasks=task_ids,
        framings=list(framings),
        models=list(models),
        full=full,
        nested=nested,
        shuffle_k=shuffle_k,
        shuffle_seed=shuffle_seed,
    )
    arms = configured_arms(config)
    for task_id in task_ids:
        task_name = TASK_NAMES[task_id]
        table = items.get(task_name)
        if table is None:
            raise X1DataError(f"{task_name}: no item table")
        if set(table.scores) != set(arms):
            raise X1DataError(
                f"{task_name}: the item table scores {sorted(table.scores)}, not the configured "
                f"arms {arms} (the framings × models the run records; §13.2)"
            )
        if not set(config.framings) <= set(table.tokens):
            raise X1DataError(f"{task_name}: no token counts for {sorted(config.framings)}")
    data = reg.data_deviations(labels_sha256, labels_content_sha256)
    if not data:
        # labels that claim to be the registered snapshot must give its published items (§1.3)
        for task_id in task_ids:
            reg.check_task(TASK_NAMES[task_id], tasks[TASK_NAMES[task_id]])
    deviations = x1_deviations(
        config, labels_sha256, labels_content_sha256, fr.lock_sha(), fingerprints
    )
    registered = not deviations
    ms = None if latency is None else latency[0]
    info = {} if latency is None else dict(latency[1])
    flags = {f: info.pop(f, None) or {} for f in LATENCY_FLAGS}
    records: dict[str, TaskRecord] = {}
    cells: dict[str, BenchCell] = {}
    for task_id in task_ids:
        task_name = TASK_NAMES[task_id]
        record, task_cells = evaluate_task(
            tasks[task_name],
            items[task_name],
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            registered=registered,
            fingerprints=fingerprints,
            latency_ms=None if ms is None else ms.get(task_id),
            latency_flags={
                arm: {
                    f: flags[f][task_id][arm]
                    for f in LATENCY_FLAGS
                    if arm in flags[f].get(task_id, {})
                }
                for arm in ARM_ORDER
            },
            full=full,
            nested=nested,
            B=B,
            seed=seed,
            alpha=alpha,
            shuffle_k=shuffle_k,
            shuffle_seed=shuffle_seed,
        )
        records[task_name] = record
        cells.update(task_cells)
    model_info = {
        m: {
            "scale": next((it.scale[m] for it in items.values() if m in it.scale), None),
            "clm_model_fp": None
            if fingerprints is None or m not in fingerprints
            else fingerprints[m].get("clm_model_fp"),
            "fingerprint": None
            if fingerprints is None or m not in fingerprints
            else dict(fingerprints[m]),
        }
        for m in models
    }
    block = X1Block(
        config=config,
        registered=registered,
        deviations=deviations,
        snapshot=SnapshotInfo(
            path=snapshot,
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
        ),
        framings_lock_sha=fr.lock_sha(),
        question_keys=_question_keys(framings, task_ids),
        models=model_info,
        store=dict(store_info or {}),
        manifest=dict(manifest_info or {}),
        latency=None if latency is None else info,
        items_file=None,
        tasks=records,
    )
    notes = [
        f"X1 framing A/B as pre-registered (DESIGN 'Pre-registration (G1)'), read by {PLAN}.",
        "Arm cells (<task>.calibrated.<framing>@<model>) are the full run on all 7 cards: "
        "exploratory, never citable (D27). X1 chooses; the citable nested cells "
        "<task>.<tier>.<A1 framing> are the tier run's (tiers.json), fed by this file's "
        "fold_choices.",
        "AUROC in the X1 statistics is of the raw score s; a cell's metrics.auroc is of its "
        "pooled LOCO-Platt probabilities. Silver labels are four-model agreement, not truth.",
    ]
    if not registered:
        notes.insert(
            0,
            "NOT the pre-registered run (every cell pre_registered: false, exploratory): "
            + "; ".join(deviations),
        )
    if latency is None:
        notes.append("No live timing file: p50 latency (§11.3) is not reported in this file.")
    env = environment()
    results = X1Results(
        date=date,
        name=name,
        mesa_clm=env["mesa_clm"],
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        environment=env,
        cells=cells,
        notes=notes,
        x1=block,
    )
    return X1Run(results=results, items={k: items[k] for k in records})


def run_x1(
    tasks: Mapping[str, Task],
    manifest: Manifest,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    *,
    date: str,
    labels_sha256: str,
    labels_content_sha256: str,
    fingerprints: Mapping[str, Mapping[str, str]] | None = None,
    latency: tuple[Mapping[str, Mapping[str, Sequence[float]]], Mapping[str, Any]] | None = None,
    framings: Sequence[str] = FRAMING_IDS,
    models: Sequence[str] = MODELS,
    full: bool = True,
    nested: bool = True,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
    shuffle_k: int = SHUFFLE_K,
    shuffle_seed: int = SHUFFLE_SEED,
    name: str = NAME,
    clock: Clock = time.perf_counter,
) -> X1Run:
    """The X1 run (§1–§12): score every X1 task's items from the store (:func:`score_items`) and
    evaluate them (:func:`run_x1_items`). ``labels_sha256`` must be the manifest's snapshot and
    ``labels_content_sha256`` that snapshot's label content (D30); both are checked first."""
    if manifest.labels_sha256 != labels_sha256:
        raise X1DataError(
            f"the manifest was built from snapshot {manifest.labels_sha256[:12]}…, not "
            f"{labels_sha256[:12]}… (D30)"
        )
    snap = Path(manifest.snapshot)
    if not snap.is_file():
        raise X1DataError(f"{snap}: the manifest's snapshot is not readable (D30)")
    content = snapshot_content_sha256(snap)
    if content != labels_content_sha256:
        raise X1DataError(
            f"labels_content_sha256 {labels_content_sha256[:12]}… is not the snapshot's "
            f"{content[:12]}… (D30)"
        )
    items = {
        name_: score_items(
            t, manifest, store, scorers, framings=framings, models=models, clock=clock
        )
        for name_, t in tasks.items()
        if t.task_id in X1_TASKS
    }
    meta = store.stats()
    store_info = {
        k: meta.get(k)
        for k in (
            "encoder_fp",
            "format",
            "vector_recipe_sha256",
            "serving_lock_sha",
            "texts",
            "vectors",
            "truncated",
        )
    }
    manifest_info = {
        "snapshot": manifest.snapshot,
        "labels_sha256": manifest.labels_sha256,
        "rows": len(manifest.rows),
        "texts": len(manifest.texts()),
        "context_conflicts": manifest.context_conflicts,
    }
    return run_x1_items(
        {k: v for k, v in tasks.items() if v.task_id in X1_TASKS},
        items,
        date=date,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        snapshot=manifest.snapshot,
        fingerprints=fingerprints,
        latency=latency,
        store_info=store_info,
        manifest_info=manifest_info,
        framings=framings,
        models=models,
        full=full,
        nested=nested,
        B=B,
        seed=seed,
        alpha=alpha,
        shuffle_k=shuffle_k,
        shuffle_seed=shuffle_seed,
        name=name,
    )


# -- the items file (§12.2) -----------------------------------------------------------------------

ITEM_COLUMNS: Final[dict[str, str]] = {
    "task": "VARCHAR",
    "task_id": "VARCHAR",
    "framing": "VARCHAR",
    "model": "VARCHAR",
    "item": "INTEGER",
    "target_sha256": "VARCHAR",
    "option_key": "VARCHAR",
    "card": "VARCHAR",
    "label_index": "INTEGER",
    "weight": "DOUBLE",
    "scale": "DOUBLE",
    "s": "DOUBLE",
    "s_contexts": "DOUBLE[]",
    "s_mean_context": "DOUBLE",
    "tokens_context": "INTEGER",
    "tokens_candidate": "INTEGER",
    "tokens_anchor": "INTEGER",
    "p_loco": "DOUBLE",
    "p_probe": "DOUBLE",
}
_JSON_TYPES: Final[dict[str, Any]] = {
    c: (["DOUBLE"] if t == "DOUBLE[]" else t) for c, t in ITEM_COLUMNS.items()
}


def _opt(arr: FloatArray | None, i: int) -> float | None:
    if arr is None:
        return None
    v = float(arr[i])
    return v if math.isfinite(v) else None


def _item_rows(items: Mapping[str, TaskItems]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for it in items.values():
        for arm in it.arms():
            framing, model = split_arm(arm)
            sc = it.scores[arm]
            tc = it.tokens.get(framing)
            p = it.p_loco.get(arm)
            for i in range(it.n):
                ctx = (
                    None
                    if sc.contexts is None
                    else [float(v) for v in sc.contexts[i, : int(it.card_size[i])]]
                )
                rows.append(
                    {
                        "task": it.task,
                        "task_id": it.task_id,
                        "framing": framing,
                        "model": model,
                        "item": i,
                        "target_sha256": it.target[i],
                        "option_key": it.option[i],
                        "card": it.card[i],
                        "label_index": int(it.label[i]),
                        "weight": float(it.weight[i]),
                        "scale": it.scale.get(model),
                        "s": float(sc.s[i]),
                        "s_contexts": ctx,
                        "s_mean_context": _opt(sc.mean_context, i),
                        "tokens_context": None if tc is None else int(tc.context[i]),
                        "tokens_candidate": None if tc is None else int(tc.candidate[i]),
                        "tokens_anchor": None if tc is None else int(tc.anchor[i]),
                        "p_loco": _opt(p, i),
                        "p_probe": _opt(it.p_probe, i),
                    }
                )
    return rows


def write_items(items: Mapping[str, TaskItems], path: str | Path) -> int:
    """Write the per-item rows (§12.2) to a Parquet file with DuckDB ``COPY`` (as ``labels
    snapshot`` does), ordered by task, framing, model and item; returns the row count. Rows reach
    DuckDB as one JSON document per chunk (DESIGN "Bulk sidecar inserts"); float64 values,
    the context lists included, round-trip exactly."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _item_rows(items)
    columns = ", ".join(f"{c} {t}" for c, t in ITEM_COLUMNS.items())
    structure = json.dumps([_JSON_TYPES], separators=(",", ":"))
    picked = ", ".join(f'r."{c}"' for c in ITEM_COLUMNS)
    con = duckdb.connect()
    try:
        con.execute(f"CREATE TABLE items ({columns})")
        for start in range(0, len(rows), 2000):
            chunk = json.dumps(rows[start : start + 2000], allow_nan=False)
            con.execute(
                f"INSERT INTO items SELECT {picked} FROM "  # noqa: S608 - fixed column names
                f"(SELECT unnest(from_json_strict(?, '{structure}')) AS r)",
                [chunk],
            )
        target = str(out).replace("'", "''")
        con.execute(
            "COPY (SELECT * FROM items ORDER BY task, framing, model, item) "  # noqa: S608
            f"TO '{target}' (FORMAT PARQUET)"
        )
    finally:
        con.close()
    return len(rows)


def read_items(path: str | Path) -> dict[str, TaskItems]:
    """The item tables of an ``x1_items.parquet`` (:func:`write_items`), by task; checks that the
    item columns agree across arms."""
    con = duckdb.connect()
    try:
        cursor = con.execute(
            f"SELECT {', '.join(ITEM_COLUMNS)} FROM read_parquet(?) "  # noqa: S608 - fixed names
            "ORDER BY task, framing, model, item",
            [str(Path(path))],
        )
        rows = cursor.fetchall()
    finally:
        con.close()
    names = list(ITEM_COLUMNS)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        r = dict(zip(names, raw, strict=True))
        by_task.setdefault(str(r["task"]), []).append(r)
    out: dict[str, TaskItems] = {}
    for task, task_rows in by_task.items():
        out[task] = _items_from_rows(task, task_rows)
    return out


def _items_from_rows(task: str, rows: Sequence[Mapping[str, Any]]) -> TaskItems:
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(arm_id(str(r["framing"]), str(r["model"])), []).append(r)
    arms = [a for a in ARM_ORDER if a in by_arm] + sorted(a for a in by_arm if a not in ARM_ORDER)
    base = sorted(by_arm[arms[0]], key=lambda r: int(r["item"]))
    n = len(base)

    def ident(r: Mapping[str, Any]) -> tuple[Any, ...]:
        return (r["target_sha256"], r["option_key"], r["card"], r["label_index"], r["weight"])

    if [int(r["item"]) for r in base] != list(range(n)):
        raise X1DataError(f"{task}: the items file's item numbers are not 0..{n - 1}")
    items = TaskItems(
        task=task,
        task_id=str(base[0]["task_id"]),
        target=[str(r["target_sha256"]) for r in base],
        option=[str(r["option_key"]) for r in base],
        card=[str(r["card"]) for r in base],
        label=np.asarray([int(r["label_index"]) for r in base], dtype=np.int64),
        weight=np.asarray([float(r["weight"]) for r in base], dtype=np.float64),
    )
    probe = [r["p_probe"] for r in base]
    if any(v is not None for v in probe):
        items.p_probe = np.asarray([np.nan if v is None else float(v) for v in probe])
    for arm in arms:
        arm_rows = sorted(by_arm[arm], key=lambda r: int(r["item"]))
        if [ident(r) for r in arm_rows] != [ident(r) for r in base]:
            raise X1DataError(f"{task}/{arm}: the item columns differ from the other arms'")
        framing, model = split_arm(arm)

        def column(name: str, rs: Sequence[Mapping[str, Any]] = arm_rows) -> FloatArray | None:
            values = [r[name] for r in rs]
            if all(v is None for v in values):
                return None
            return np.asarray([np.nan if v is None else float(v) for v in values])

        s = column("s")
        if s is None:
            raise X1DataError(f"{task}/{arm}: no scores")
        lists = [r["s_contexts"] for r in arm_rows]
        contexts = None
        if any(v is not None for v in lists):
            sizes = items.card_size
            if any(v is None or len(v) != int(m) for v, m in zip(lists, sizes, strict=True)):
                raise X1DataError(f"{task}/{arm}: a context list does not cover its card")
            contexts = np.full((n, int(sizes.max())), np.nan)
            for i, v in enumerate(lists):
                contexts[i, : len(v)] = v
        items.scores[arm] = ArmScores(s=s, contexts=contexts, mean_context=column("s_mean_context"))
        p = column("p_loco")
        items.p_loco[arm] = np.full(n, np.nan) if p is None else p
        scale = arm_rows[0]["scale"]
        if scale is not None:
            items.scale[model] = float(scale)
        if arm_rows[0]["tokens_context"] is not None:
            tc = TokenCounts(
                context=np.asarray([int(r["tokens_context"]) for r in arm_rows], dtype=np.int64),
                candidate=np.asarray(
                    [int(r["tokens_candidate"]) for r in arm_rows], dtype=np.int64
                ),
                anchor=np.asarray([int(r["tokens_anchor"]) for r in arm_rows], dtype=np.int64),
            )
            known = items.tokens.get(framing)
            if known is None:
                items.tokens[framing] = tc
            elif not all(
                np.array_equal(getattr(known, k), getattr(tc, k))
                for k in ("context", "candidate", "anchor")
            ):
                raise X1DataError(f"{task}/{framing}: token counts differ between models")
    items.__post_init__()
    return items


# -- writing, reading, replaying (§12) ------------------------------------------------------------


class X1Exists(FileExistsError):
    """``write_x1`` would overwrite an earlier run's files (§14.9: one run, committed as
    produced)."""


def write_x1(run: X1Run, out_dir: str | Path = DEFAULT_OUT_DIR, *, force: bool = False) -> Path:
    """Write ``<out_dir>/<date>/<name>_items.parquet`` (§12.2), then ``<name>.json`` stamped with
    its sha256 (§12.1) and ``<name>.md`` (§12.5); ``run.results`` becomes the results written.
    Refuses (:class:`X1Exists`) to replace an existing file unless ``force``. Returns the JSON
    path."""
    res = run.results
    path = results_path(out_dir, res.date, res.name)
    items_path = path.with_name(f"{res.name}{ITEMS_SUFFIX}")
    existing = [p for p in (path, items_path, path.with_suffix(".md")) if p.exists()]
    if existing and not force:
        raise X1Exists(f"{existing[0]} exists: one run per results file (pass --force to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = write_items(run.items, items_path)
    sha = hashlib.sha256(items_path.read_bytes()).hexdigest()
    block = res.x1.model_copy(
        update={"items_file": ItemsFile(name=items_path.name, sha256=sha, rows=rows)}
    )
    final = res.model_copy(update={"x1": block})
    run.results = final
    payload = final.model_dump(by_alias=True, mode="json")
    text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown(final), encoding="utf-8")
    return path


def load_x1(path: str | Path) -> X1Results:
    """An ``x1.json`` (:func:`write_x1`)."""
    return X1Results.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def x1_cells(path: str | Path) -> BenchResults:
    """The cells of an ``x1.json`` as a :class:`~mesa_clm.bench.results.BenchResults` (what a
    policy ``cite`` resolves against): the file without its ``x1`` block."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.pop("x1", None)
    raw["format"] = RESULTS_FORMAT
    return BenchResults.model_validate(raw)


def _diff(a: Any, b: Any, tol: float, where: str) -> str | None:
    """The first difference between two JSON values: exact for everything but floats, which
    may differ by ``tol`` relative to max(1, |x|)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{where}: keys {sorted(set(a) ^ set(b))} differ"
        for k in sorted(a):
            d = _diff(a[k], b[k], tol, f"{where}.{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{where}: {len(a)} vs {len(b)} entries"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = _diff(x, y, tol, f"{where}[{i}]")
            if d is not None:
                return d
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a is b else f"{where}: {a!r} != {b!r}"
    if (
        isinstance(a, float | int)
        and isinstance(b, float | int)
        and (isinstance(a, float) or isinstance(b, float))
    ):
        if abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b))):
            return None
        return f"{where}: {a!r} != {b!r}"
    return None if a == b else f"{where}: {a!r} != {b!r}"


Settings = tuple[int, int, float]


def _check_decision(
    decision: Decision, where: str, settings: Settings | None = None, arms: Sequence[str] = ()
) -> None:
    """Every stored rule R verdict of a trace follows from its numbers and used the run's
    bootstrap ``settings`` (:func:`check_rule`), the trace holds exactly the configured ``arms``
    (when given), and it replays bit for bit (§12.6 (a))."""
    if arms and list(decision.arms) != list(arms):
        raise X1ReproError(
            f"{where}: the trace's arms {list(decision.arms)} are not the configured {list(arms)}"
        )
    for arm, verdict in decision.arms.items():
        ev = verdict.evidence
        if (ev.arm, ev.framing, ev.model) != (arm, *split_arm(arm)):
            raise X1ReproError(f"{where}.arms.{arm}: the evidence names another arm")
        if ev.shuffle is not None:
            check_rule(ev.shuffle, f"{where}.arms.{arm}.shuffle", settings)
    for i, c in enumerate(decision.comparisons):
        check_rule(c.rule_r, f"{where}.comparisons[{i}] {c.a} ≻ {c.b}", settings)
    replay = decide(StoredEvidence(decision))
    if replay != decision:
        d = _diff(replay.model_dump(), decision.model_dump(), 0.0, where)
        raise X1ReproError(
            f"{where}: the stored decision does not follow from its statistics ({d})"
        )


def _ci_settings(ci: ClusterCI | None, cfg: X1Config, where: str) -> None:
    if ci is not None and (ci.B, ci.seed, ci.alpha) != (cfg.B, cfg.seed, cfg.alpha):
        raise X1ReproError(f"{where}: its bootstrap is not the run's B/seed/alpha")


def _dump(value: BaseModel | None) -> Any:
    return None if value is None else value.model_dump(mode="json")


def _check_arm_record(
    rec: TaskRecord, a: ArmRecord, verdict: ArmVerdict, cell: BenchCell, cfg: X1Config
) -> None:
    """An arm record says what its trace and its cell say (§12.6 (a)): the AUROC, the shuffle's
    rule R and the LOCO-Platt NLL the decision read, its own bootstrap settings, the folds and
    skips of the task's fold plan, and the cell's diagnostics (``auroc_s``, the shuffle's
    record, the mean-context AUROC, the inverted folds, the arm diagnostics), so nothing
    ``x1.md`` prints can contradict the decision that replays."""
    where = f"{rec.task}/{a.arm}"
    ev = verdict.evidence
    framing, model = split_arm(a.arm)
    if (a.framing, a.model) != (framing, model) or a.scale != rec.scales.get(model):
        raise X1ReproError(f"{where}: the arm record names another arm or scale")
    if (a.auroc.point, a.auroc.lower) != (ev.auroc, ev.auroc_lower):
        raise X1ReproError(f"{where}: the arm record's AUROC is not its trace's")
    if (a.shuffle is None) != (ev.shuffle is None) or (
        a.shuffle is not None and a.shuffle.rule_r != ev.shuffle
    ):
        raise X1ReproError(f"{where}: the arm record's shuffle rule R is not its trace's")
    if (a.loco.nll, a.loco.n) != (ev.nll, ev.n_loco):
        raise X1ReproError(f"{where}: the arm record's LOCO-Platt NLL is not its trace's")
    _ci_settings(a.auroc, cfg, where)
    if a.shuffle is not None:
        _ci_settings(a.shuffle.auroc, cfg, f"{where}.shuffle")
        if (a.shuffle.k, a.shuffle.seed, a.shuffle.n_unshuffled) != (
            cfg.shuffle_k,
            cfg.shuffle_seed,
            rec.shuffle.n_unshuffled,
        ):
            raise X1ReproError(f"{where}: the shuffle record is not the run's draws")
    if a.mean_context is not None:
        _ci_settings(a.mean_context.auroc, cfg, f"{where}.mean_context")
        if a.mean_context.n_loco != a.loco.n:
            raise X1ReproError(f"{where}: the mean-context record pools other items")
    if a.loco.skipped != rec.loco_skipped or set(a.loco.folds) & set(a.loco.skipped):
        raise X1ReproError(f"{where}: the LOCO record's folds are not the task's fold plan")
    if set(a.loco.folds) | set(a.loco.skipped) != set(rec.cards):
        raise X1ReproError(f"{where}: the LOCO record does not cover the task's cards")
    if a.loco.inverted != sorted(c for c, f in a.loco.folds.items() if f.inverted):
        raise X1ReproError(f"{where}: the inverted folds are not the fits'")
    sens = a.sensitivity
    if (sens is None) != (a.loco.n == 0) or (
        sens is not None and (sens.n, sens.weighted.nll) != (a.loco.n, a.loco.nll)
    ):
        raise X1ReproError(f"{where}: the sensitivity record is not the arm's LOCO-Platt")
    want = {
        "x1_arm": a.arm,
        "auroc_s": _dump(a.auroc),
        "qualified": a.qualified,
        "failed": a.failed,
        "shuffle_auroc": None if a.shuffle is None else _dump(a.shuffle.auroc),
        "shuffle_rule_r": None if a.shuffle is None else _dump(a.shuffle.rule_r),
        "mean_context_auroc": None if a.mean_context is None else _dump(a.mean_context.auroc),
        "inverted_folds": a.loco.inverted,
        "calibration": "platt",
        "fitters": _jsonable(calibrate.FITTER_CONSTANTS),
        **_jsonable(a.diagnostics),
    }
    got = {k: cell.diagnostics.get(k) for k in want}
    if got != want or set(cell.diagnostics) != set(want):
        d = _diff(got, want, 0.0, f"{where} cell.diagnostics")
        raise X1ReproError(f"{where}: the cell's diagnostics are not the arm record's ({d})")
    calls = rec.n if framing == CONTROL else rec.n_targets
    if (a.diagnostics.get("calls"), a.diagnostics.get("calls_per_target")) != (
        calls,
        calls / rec.n_targets,
    ):
        raise X1ReproError(f"{where}: the calls are not the task's")
    if rec.decision is not None and framing in rec.decision.tokens_per_target:
        d = _diff(
            a.diagnostics.get("input_tokens_per_target"),
            rec.decision.tokens_per_target[framing],
            REPRO_TOLERANCE,
            f"{where}.input_tokens_per_target",
        )
        if d is not None:
            raise X1ReproError(f"{where}: the tokens per target are not the trace's ({d})")


def _check_cell_numbers(
    cell: BenchCell, rec: TaskRecord, a: ArmRecord, cfg: X1Config, tolerance: float
) -> None:
    """An arm cell's counts and metrics follow from its own per-item predictions, and so does
    its novel-key block (the cell's predictions on its novel items); its pool and folds are the
    arm record's (§12.6 (a)). The lookup controls and ``beats_lookup_novel`` need the items'
    states and are report-only here (§12.6)."""
    where = f"{rec.task}/{a.arm} cell"
    items = cell.items or []
    task = Task(
        rec.task,
        TASKS[rec.task_id],
        [({}, int(it.label)) for it in items],
        "",
        "",
        cards=[it.card for it in items],
    )
    keys = [(it.target_sha256, it.option_key) for it in items]
    if len(set(keys)) != len(keys):
        raise X1ReproError(f"{where}: an item is listed twice")
    counts = CellCounts(
        n=len(items),
        n_neg=task.n_neg(),
        n_nonmodal=task.n_nonmodal(),
        class_counts=task.class_counts(),
    )
    cards = sorted(set(task.cards))
    if cell.counts != counts or len(items) != a.loco.n:
        raise X1ReproError(f"{where}: its counts are not its items'")
    if cards != sorted(a.loco.folds) or cell.n_folds != len(cards):
        raise X1ReproError(f"{where}: its folds are not the arm record's")
    if cell.skipped_folds != rec.loco_skipped:
        raise X1ReproError(f"{where}: its skipped folds are not the task's fold plan")
    if not items:
        if cell.metrics is not None or cell.baselines is not None:
            raise X1ReproError(f"{where}: metrics without a pooled item")
        return
    probs = np.asarray([it.probs for it in items], dtype=np.float64).reshape(len(items), task.k)
    labels = np.asarray(task.labels, dtype=np.int64)
    metrics = pooled_metrics(task, probs, labels, task.cards, B=cfg.B, seed=cfg.seed)
    if cell.metrics is None:
        raise X1ReproError(f"{where}: no metrics")
    d = _diff(
        metrics.model_dump(by_alias=True, mode="json"),
        cell.metrics.model_dump(by_alias=True, mode="json"),
        tolerance,
        f"{where}.metrics",
    )
    if d is not None:
        raise X1ReproError(f"{where}: its metrics are not its items' ({d})")
    if cell.baselines is None:
        raise X1ReproError(f"{where}: no baselines block")
    novel = np.asarray([it.novel for it in items], dtype=bool)
    n = len(items)
    own = LookupEval(
        idx=np.arange(n, dtype=np.int64),
        labels=labels,
        cards=list(task.cards),
        fold=list(task.cards),
        pred=np.zeros(n, dtype=np.int64),
        majority_pred=np.zeros(n, dtype=np.int64),
        probs=probs,
        novel=novel,
        conflicting=np.zeros(n, dtype=bool),
        tied=np.zeros(n, dtype=bool),
        n_folds=len(cards),
        folds=cards,
    )
    mine = novel_key_block(own, binary=task.binary, B=cfg.B, seed=cfg.seed).model_dump(mode="json")
    stored = cell.baselines.novel_key.model_dump(mode="json")
    mine.pop("majority_acc")  # the lookup's majority prediction needs the items' states
    stored.pop("majority_acc")
    d = _diff(mine, stored, tolerance, f"{where}.baselines.novel_key")
    lookup = cell.baselines.novel_key_lookup
    if d is not None or lookup is None or lookup.n != mine["n"]:
        raise X1ReproError(f"{where}: its novel-key block is not its items' ({d})")
    if (cell.baselines.beats_detail or {}).get("n_novel") != int(novel.sum()):
        raise X1ReproError(f"{where}: beats_detail counts other novel items")


def _check_cells(res: X1Results, rec: TaskRecord, cfg: X1Config, tolerance: float) -> None:
    """The task's arm cells and records say what its trace says (§12.6 (a)): one record and one
    cell per arm of the trace, ``qualified`` and ``failed`` as the verdict, the framing's
    ``question_key``, the run's labels, model fingerprint and registration, the record's
    statistics the trace's (:func:`_check_arm_record`), the cell's numbers its own items'
    (:func:`_check_cell_numbers`)."""
    if rec.decision is None:
        if rec.arms:
            raise X1ReproError(f"{rec.task}: arm records without a full run")
        return
    for arm, verdict in rec.decision.arms.items():
        framing, model = split_arm(arm)
        a = rec.arms.get(arm)
        cell = res.cells.get(a.cell) if a is not None else None
        if a is None or cell is None:
            raise X1ReproError(f"{rec.task}/{arm}: no arm cell for an arm of the trace")
        if a.arm != arm or a.cell != cell_key(rec.task, "calibrated", framing, model):
            raise X1ReproError(f"{rec.task}/{arm}: the arm record names another arm or cell")
        if (a.qualified, a.failed) != (verdict.qualified, verdict.failed) or (
            cell.diagnostics.get("qualified"),
            cell.diagnostics.get("failed"),
        ) != (verdict.qualified, verdict.failed):
            raise X1ReproError(f"{rec.task}/{arm}: the cell's rule (1) verdict is not the trace's")
        fp = res.x1.models.get(model, {}).get("fingerprint")
        qkey = fr.framing(rec.task_id, framing).question_key
        if (
            cell.question_key != qkey
            or a.question_key != qkey
            or cell.question_key != res.x1.question_keys.get(rec.task_id, {}).get(framing)
            or (cell.task, cell.task_id, cell.tier, cell.framing)
            != (
                rec.task,
                rec.task_id,
                "calibrated",
                framing,
            )
            or (cell.model, cell.variant) != (model, model)
            or cell.labels_sha256 != res.labels_sha256
            or cell.labels_content_sha256 != res.labels_content_sha256
            or cell.fingerprint != fp
            or cell.pre_registered != res.x1.registered
            or not cell.exploratory
            or cell.selection != "full"
        ):
            raise X1ReproError(f"{rec.task}/{arm}: the cell's identity fields are not the run's")
        _check_arm_record(rec, a, verdict, cell, cfg)
        _check_cell_numbers(cell, rec, a, cfg, tolerance)
    if set(rec.arms) != set(rec.decision.arms):
        raise X1ReproError(f"{rec.task}: the arm records are not the trace's arms")


def _check_nested(rec: TaskRecord, settings: Settings | None, arms: Sequence[str]) -> None:
    if rec.nested is None:
        return
    if set(rec.nested.folds) != set(rec.cards):
        raise X1ReproError(f"{rec.task}: the nested folds are not one per card")
    for card, fold in rec.nested.folds.items():
        if (fold.card, fold.n_test, fold.n_train) != (
            card,
            rec.cards[card],
            rec.n - rec.cards[card],
        ):
            raise X1ReproError(f"{rec.task} fold {card}: its sizes are not the task's")
        _check_decision(fold.decision, f"{rec.task} fold {card}", settings, arms)
        c = fold.decision.choice
        stored = rec.nested.fold_choices.get(card)
        if (
            stored is None
            or stored.outcome != fold.decision.outcome
            or stored.arm != (None if c is None else c.arm)
            or stored.framing != (None if c is None else c.framing)
            or stored.model != (None if c is None else c.model)
        ):
            raise X1ReproError(f"{rec.task} fold {card}: fold_choices disagree with its trace")
    if set(rec.nested.fold_choices) != set(rec.nested.folds):
        raise X1ReproError(f"{rec.task}: fold_choices are not one per outer fold")
    skips = {
        card: why
        for card, f in rec.nested.fold_choices.items()
        if (why := _inner_skip(f.outcome)) is not None
    }
    if skips != rec.nested.skipped_folds:
        raise X1ReproError(f"{rec.task}: skipped_folds are not the folds without a choice")


def _check_published(rec: TaskRecord, counts: Mapping[str, reg.TaskCounts]) -> None:
    """A run on the registered labels holds the published items (§1.3): its ``n``, class
    counts and items per card, as the registration publishes them."""
    want = counts.get(rec.task)
    if want is None:
        return
    if (
        rec.n != want.n
        or dict(rec.class_counts) != dict(want.class_counts)
        or dict(rec.cards) != dict(want.per_card)
    ):
        raise X1ReproError(
            f"{rec.task}: {rec.n} items {dict(rec.class_counts)} are not the published "
            f"{want.n} {dict(want.class_counts)} ({reg.current().published})"
        )


def _close(a: npt.ArrayLike, b: npt.ArrayLike, tolerance: float) -> bool:
    """Equal NaN patterns and the finite values within ``tolerance`` relative to max(1, |x|)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
        return False
    on = ~np.isnan(x)
    scale = np.maximum(1.0, np.maximum(np.abs(x[on]), np.abs(y[on])))
    return bool(np.all(np.abs(x[on] - y[on]) <= tolerance * scale))


def _check_cell_items(
    cell: BenchCell, items: TaskItems, p_full: FloatArray, tolerance: float, where: str
) -> None:
    """The arm cell's per-item predictions are the recomputed LOCO-Platt p(Yes) of exactly the
    pooled items, with the items file's cards and labels (§12.6 (b))."""
    index = {(t, o): i for i, (t, o) in enumerate(zip(items.target, items.option, strict=True))}
    seen: list[int] = []
    for it in cell.items or []:
        i = index.get((it.target_sha256, it.option_key))
        if i is None or it.card != items.card[i] or it.label != int(items.label[i]):
            raise X1ReproError(f"{where}: a cell item is not an item of the items file")
        if not _close(it.probs, [p_full[i], 1.0 - p_full[i]], tolerance):
            raise X1ReproError(f"{where}: a cell item's prediction is not the recomputed one")
        seen.append(i)
    if sorted(seen) != np.flatnonzero(~np.isnan(p_full)).tolist():
        raise X1ReproError(f"{where}: the cell's items are not the pooled items")


def _check_probe(rec: TaskRecord, items: TaskItems, cfg: X1Config, tolerance: float) -> None:
    """The candidate-only probe's record (§7.11) from the items file: its folds and skips from
    the labels and cards, its statistics from the ``p_probe`` column. The probe's fits need the
    candidate vectors, so ``p_probe`` itself and each fold's solver state are report-only."""
    cp = rec.candidate_probe
    p = items.p_probe
    if cp is None:
        if p is not None and np.any(~np.isnan(p)):
            raise X1ReproError(f"{rec.task}: p_probe without a probe record")
        return
    if p is None:
        raise X1ReproError(f"{rec.task}: a probe record without a p_probe column")
    run, skipped = probe_folds(items)
    sizes = {f.held_out: (len(f.test), len(f.train)) for f in run}
    if (
        cp.skipped != skipped
        or cp.n_folds != len(run)
        or {k: (v.get("n"), v.get("n_train")) for k, v in cp.folds.items()} != sizes
        or sorted(np.flatnonzero(~np.isnan(p)).tolist()) != sorted(i for f in run for i in f.test)
    ):
        raise X1ReproError(f"{rec.task}: the probe's folds are not the items file's")
    want = _probe_record(
        items,
        CandidateProbe(p, dict(cp.folds), dict(cp.skipped)),
        B=cfg.B,
        seed=cfg.seed,
        alpha=cfg.alpha,
    )
    d = _diff(_dump(want), _dump(cp), tolerance, f"{rec.task}.candidate_probe")
    if d is not None:
        raise X1ReproError(f"{rec.task}: the probe's statistics are not p_probe's ({d})")


def _recompute_task(
    res: X1Results,
    rec: TaskRecord,
    items: TaskItems,
    cfg: X1Config,
    arms: Sequence[str],
    counts: Mapping[str, reg.TaskCounts],
    tolerance: float,
) -> None:
    """§12.6 (b) for one task, from its items file table alone: the item set (the published
    counts when the labels are the registered ones), the task summary, the fold plan, the
    decision, every arm record's statistics, the ``p_loco`` column and the arm cells' per-item
    predictions, the probe's record, the nesting and the shuffle draws."""
    task = rec.task
    if items.task_id != rec.task_id or set(items.scores) != set(arms):
        raise X1ReproError(f"{task}: the items file's arms are not the configured {list(arms)}")
    per_card = {c: int(np.sum(items.card_arr == c)) for c in sorted(set(items.card))}
    labels = {
        int(k): int(v) for k, v in zip(*np.unique(items.label, return_counts=True), strict=True)
    }
    want = counts.get(task)
    if want is not None and (
        items.n != want.n or labels != dict(want.class_counts) or per_card != dict(want.per_card)
    ):
        raise X1ReproError(f"{task}: the items file does not hold the published items (§1.3)")
    if (
        rec.n != items.n
        or rec.n_targets != len(set(items.target))
        or dict(rec.class_counts) != labels
        or dict(rec.cards) != per_card
        or {m: items.scale.get(m) for m in cfg.models} != {m: rec.scales.get(m) for m in cfg.models}
    ):
        raise X1ReproError(f"{task}: the task summary is not the items file's")
    ev = ItemEvidence(
        items,
        B=cfg.B,
        seed=cfg.seed,
        alpha=cfg.alpha,
        shuffle_k=cfg.shuffle_k,
        shuffle_seed=cfg.shuffle_seed,
    )
    if dict(ev.plan.skipped) != dict(rec.loco_skipped):
        raise X1ReproError(f"{task}: the fold plan's skips are not the items file's")
    if rec.decision is not None:
        dec = decide(ev)
        d = _diff(dec.model_dump(), rec.decision.model_dump(), tolerance, f"{task}.decision")
        if d is not None:
            raise X1ReproError(f"{task}: recomputed from the items, the decision differs ({d})")
        for arm, a in rec.arms.items():
            st = _arm_stats(items, ev, arm)
            stored = {name: _dump(getattr(a, name)) for name in st.records()}
            d = _diff(st.records(), stored, tolerance, f"{task}.arms.{arm}")
            if d is not None:
                raise X1ReproError(
                    f"{task}: recomputed from the items, the arm record differs ({d})"
                )
            if not _close(items.p_loco.get(arm, np.full(items.n, np.nan)), st.p_full, tolerance):
                raise X1ReproError(f"{task}/{arm}: p_loco is not the recomputed LOCO-Platt")
            _check_cell_items(res.cells[a.cell], items, st.p_full, tolerance, f"{task}/{arm}")
        _check_probe(rec, items, cfg, tolerance)
    elif any(np.any(~np.isnan(p)) for p in items.p_loco.values()) or (
        items.p_probe is not None and np.any(~np.isnan(items.p_probe))
    ):
        raise X1ReproError(f"{task}: held-out probabilities without a full run")
    if rec.nested is not None:
        nested = nested_selection(
            items,
            B=cfg.B,
            seed=cfg.seed,
            alpha=cfg.alpha,
            shuffle_k=cfg.shuffle_k,
            shuffle_seed=cfg.shuffle_seed,
        )
        d = _diff(nested.model_dump(), rec.nested.model_dump(), tolerance, f"{task}.nested")
        if d is not None:
            raise X1ReproError(f"{task}: recomputed from the items, the nesting differs ({d})")
    drawn = _shuffle_draws_record(items, cfg.shuffle_k, cfg.shuffle_seed)
    if drawn != rec.shuffle:
        raise X1ReproError(f"{task}: the shuffle draws are not the recorded ones")


def decide_from_json(
    path: str | Path,
    *,
    recompute: bool = True,
    tolerance: float = REPRO_TOLERANCE,
    require_registered: bool = True,
) -> dict[str, Decision | None]:
    """``bench framing --decide --from x1.json`` (§12.6), from the files alone: (0) the
    configuration's constants are this code's, and the file's ``registered`` and ``deviations``
    follow from its configuration, its labels, the framings lock and the models' fingerprints it
    records (:func:`x1_deviations`); with ``require_registered`` (the default) a run that is not
    the pre-registered one is refused; the file holds exactly the configured tasks, each trace
    exactly the configured arms, and registered labels the published counts; (a) **replay**:
    every stored rule R verdict follows from its own numbers and the run's bootstrap settings,
    rules (1)-(5) re-run on the statistics stored in every trace (the full run and each nested
    fold) give the identical trace, and the arm records, the arm cells (their counts, metrics
    and novel-key block from their own items), ``a1``, ``fold_choices`` and ``skipped_folds``
    say what the traces say; (b) with ``recompute``, the items file (sha256 checked) rebuilds
    the decision, every arm record's statistics, the ``p_loco`` column and the cells' per-item
    predictions, the probe's record, the nesting and the shuffle draws, identical but for
    numbers within ``tolerance``. Returns the full-run decision per task (``None`` when the run
    had no full run); raises :class:`X1ReproError` on any mismatch. Reads neither the feature
    store nor a label store."""
    file = Path(path)
    res = load_x1(file)
    cfg = res.x1.config
    _check_constants(cfg)
    fingerprints = {m: (res.x1.models.get(m) or {}).get("fingerprint") for m in cfg.models}
    deviations = x1_deviations(
        cfg, res.labels_sha256, res.labels_content_sha256, res.x1.framings_lock_sha, fingerprints
    )
    if res.x1.deviations != deviations or res.x1.registered != (not deviations):
        raise X1ReproError(f"{file}: its registered status does not follow from its configuration")
    if require_registered and deviations:
        raise X1ReproError(
            f"{file}: not the pre-registered run ({'; '.join(deviations)}); its cells are "
            "exploratory"
        )
    snap = res.x1.snapshot
    if (snap.labels_sha256, snap.labels_content_sha256) != (
        res.labels_sha256,
        res.labels_content_sha256,
    ):
        raise X1ReproError(f"{file}: the snapshot block names other labels than the file")
    try:
        arms = configured_arms(cfg)
    except X1DataError as exc:
        raise X1ReproError(f"{file}: {exc}") from exc
    unknown = [t for t in cfg.tasks if t not in TASK_NAMES]
    want_tasks = {TASK_NAMES[t]: t for t in cfg.tasks if t in TASK_NAMES}
    if unknown or set(res.x1.tasks) != set(want_tasks):
        raise X1ReproError(
            f"{file}: the tasks recorded ({sorted(res.x1.tasks)}) are not the configured ones "
            f"({sorted(cfg.tasks)})"
        )
    if res.x1.question_keys != _question_keys(cfg.framings, cfg.tasks):
        raise X1ReproError(f"{file}: the question keys are not the configured framings' (D1)")
    settings: Settings = (cfg.B, cfg.seed, cfg.alpha)
    counts = (
        dict(reg.current().counts)
        if not reg.data_deviations(res.labels_sha256, res.labels_content_sha256)
        else {}
    )
    out: dict[str, Decision | None] = {}
    for task, rec in res.x1.tasks.items():
        if (rec.task, rec.task_id) != (task, want_tasks[task]):
            raise X1ReproError(f"{task}: the record names another task")
        if rec.decision is not None:
            _check_decision(rec.decision, f"{task}.decision", settings, arms)
        if rec.a1 != (None if rec.decision is None else rec.decision.choice):
            raise X1ReproError(f"{task}: a1 is not the full-run choice")
        if (rec.decision is None) == cfg.full or (rec.nested is None) == cfg.nested:
            raise X1ReproError(f"{task}: the record is not what the configuration ran")
        if (rec.shuffle.k, rec.shuffle.seed) != (cfg.shuffle_k, cfg.shuffle_seed):
            raise X1ReproError(f"{task}: the shuffle draws are not the configured ones")
        _check_published(rec, counts)
        _check_cells(res, rec, cfg, tolerance)
        _check_nested(rec, settings, arms)
        out[task] = rec.decision
    if {c for r in res.x1.tasks.values() for c in (a.cell for a in r.arms.values())} != set(
        res.cells
    ):
        raise X1ReproError(f"{file}: a cell belongs to no arm record")
    if not recompute:
        return out
    if res.x1.items_file is None:
        raise X1ReproError(f"{file}: no items file is recorded")
    items_path = file.with_name(res.x1.items_file.name)
    if not items_path.is_file():
        raise X1ReproError(f"{items_path}: the items file is missing")
    if hashlib.sha256(items_path.read_bytes()).hexdigest() != res.x1.items_file.sha256:
        raise X1ReproError(f"{items_path}: sha256 differs from the one x1.json records")
    try:
        tables = read_items(items_path)
    except X1DataError as exc:
        raise X1ReproError(f"{items_path}: {exc}") from exc
    if set(tables) != set(res.x1.tasks):
        raise X1ReproError(f"{items_path}: its tasks are not the recorded ones")
    for task, rec in res.x1.tasks.items():
        _recompute_task(res, rec, tables[task], cfg, arms, counts, tolerance)
    return out


# -- markdown (§12.5) -----------------------------------------------------------------------------


def _f(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _cif(ci: ClusterCI | None) -> str:
    if ci is None or ci.point is None:
        return ""
    return f"{_f(ci.point)} [{_f(ci.lower)}, {_f(ci.upper)}]"


def markdown(results: X1Results) -> str:
    """``x1.md``: per task the arms, the decision, the nested folds and the cells; every number
    names ``x1.json``."""
    src = f"bench/results/{results.date}/{results.name}.json"
    cfg = results.x1.config
    ci = f"[one-sided {100 * (1 - cfg.alpha):g}% bounds; {100 * (1 - 2 * cfg.alpha):g}% interval]"
    lines = [
        f"# X1 framing A/B · {results.date} · mesa-clm {results.mesa_clm}",
        "",
        f"Every number below names `{src}` (labels_sha256 `{results.labels_sha256[:12]}…`); the "
        f"analysis plan is `{results.x1.plan}`. Silver labels are four-model agreement, not truth. "
        f"Bootstraps: B={cfg.B}, seed {cfg.seed}, card clusters; every interval [lower, upper] "
        f"is the pair of one-sided {100 * (1 - cfg.alpha):g}% bounds (the {cfg.alpha:g} and "
        f"{1 - cfg.alpha:g} quantiles), a {100 * (1 - 2 * cfg.alpha):g}% interval; shuffle: "
        f"{cfg.shuffle_k} within-card derangements, seed {cfg.shuffle_seed}. `bench framing "
        "--decide --from` replays and recomputes every number of the arm tables and the cells' "
        "counts and metrics; the cells' lookup controls and beats_lookup_novel need the items' "
        "states and are report-only. Registered run: "
        f"{'yes' if results.x1.registered else 'NO (' + '; '.join(results.x1.deviations) + ')'}.",
    ]
    for name, rec in results.x1.tasks.items():
        d = rec.decision
        head = (
            "nested only"
            if d is None
            else f"{d.outcome} → {'—' if d.choice is None else d.choice.arm}"
        )
        lines += [
            "",
            f"## {name} ({rec.task_id}): {head}",
            "",
            f"{rec.n} items, {rec.n_targets} targets, {len(rec.cards)} cards.",
        ]
        if rec.arms:
            lines += [
                "",
                f"| arm | AUROC(s) {ci} | shuffle AUROC (mean of k) | Δ rule R | qualified "
                "| LOCO-Platt NLL | unweighted NLL | tokens/target | cos raw / proj |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for arm, a in rec.arms.items():
                sh = a.shuffle
                cos = a.diagnostics.get("collapse") or {}
                unw = a.sensitivity.unweighted.nll if a.sensitivity is not None else None
                lines.append(
                    f"| {arm} | {_cif(a.auroc)} | {'' if sh is None else _cif(sh.auroc)} | "
                    f"{'' if sh is None else sh.rule_r.reason} | "
                    f"{'control' if a.qualified is None else a.qualified} | {_f(a.loco.nll)} | "
                    f"{_f(unw)} | {_f(a.diagnostics.get('input_tokens_per_target'), 1)} | "
                    f"{_f(cos.get('raw'))} / {_f(cos.get('projected'))} |"
                )
        if rec.candidate_probe is not None:
            cp = rec.candidate_probe
            lines += [
                "",
                f"- candidate-only probe (report-only): AUROC {_cif(cp.auroc)}, NLL "
                f"{_f(cp.nll)}, {cp.n} items over {cp.n_folds} folds",
            ]
        if d is not None:
            lines.append("")
            if d.step2 is not None:
                lines.append(f"- rule (2): {d.step2.reason} → {d.step2.framing}")
            if d.step3 is not None:
                lines.append(f"- rule (3): {d.step3.reason} → {d.step3.model}")
            if d.step4 is not None:
                lines.append(
                    f"- rule (4): D3 amendment recommended: {d.step4.d3_amendment_recommended}"
                )
            lines += [f"- {note}" for note in d.notes]
        if rec.nested is not None:
            lines += ["", "| outer card | inner outcome | choice | skipped |", "|---|---|---|---|"]
            for card, c in rec.nested.fold_choices.items():
                lines.append(
                    f"| {card} | {c.outcome} | {c.arm or ''} | "
                    f"{rec.nested.skipped_folds.get(card, '')} |"
                )
    if results.cells:
        lines += [
            "",
            f"| cell | n | acc | nll | ece | auroc {ci} | beats_lookup_novel (report-only) |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, cell in results.cells.items():
            m = cell.metrics
            b = cell.baselines
            lines.append(
                f"| {key} | {cell.counts.n} | {_f(m.acc) if m else ''} | {_f(m.nll) if m else ''} | "
                f"{_f(m.ece) if m else ''} | {_cif(m.auroc_ci) if m else ''} | "
                f"{'' if b is None else b.beats_lookup_novel} |"
            )
    for note in results.notes:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"


# -- inputs and diagnostics for the CLI (§1, §11.3) ------------------------------------------------


def x1_tasks(store: LabelStore) -> dict[str, Task]:
    """The X1 bench tasks over ``store`` (§2.1): :func:`~mesa_clm.bench.tasks.neon.neon_task`
    for term.fits and column.ontology_fits at the shipped policy's ``min_weight`` (no override,
    D9) with the D30 fold filter. The M2 verbs build ``store`` from the registered snapshot
    itself (:mod:`mesa_clm.bench.run`)."""
    return {name: neon_task(store, name, task_id) for task_id, name in TASK_NAMES.items()}


def selection(record: TaskRecord) -> dict[str, Any]:
    """X1's outcome for one task as plain data: the keyword arguments of the tier cells'
    ``X1Selection.of`` (``bench/cells.py``). ``a1`` is the full-run choice ``{framing, model}``
    (``None`` unless the outcome is a choice); ``fold_choices`` maps each outer card with an
    inner choice to ``{framing, model, outcome, x1_trace}`` (``x1_trace`` is the JSON pointer of
    the inner trace in ``x1.json``, which the tier cell's ``fold_choices`` entry keeps, §9.5);
    ``fold_skips`` maps each outer card without one to ``inner_k1`` or ``inner_undecidable``
    (§8.5); ``note`` states the outcome. An outer guard is not a skip here: a cell applies its
    own guards."""
    if record.decision is None or record.nested is None:
        raise X1ReproError(f"{record.task}: the X1 record has no full run or no nesting")
    a1 = record.a1
    choices: dict[str, Any] = {}
    skips: dict[str, str] = {}
    for card, c in record.nested.fold_choices.items():
        why = _inner_skip(c.outcome)
        if why is not None:
            skips[card] = why
        else:
            choices[card] = {
                "framing": c.framing,
                "model": c.model,
                "outcome": c.outcome,
                "x1_trace": f"#/x1/tasks/{record.task}/nested/folds/{card}/decision",
            }
    outcome = record.decision.outcome if a1 is None else f"{record.decision.outcome} {a1.arm}"
    return {
        "a1": None if a1 is None else {"framing": a1.framing, "model": a1.model},
        "fold_choices": choices,
        "fold_skips": skips,
        "note": f"X1 {record.task}: {outcome} (full run, {PLAN})",
    }


def selections_from_json(
    path: str | Path, *, recompute: bool = True, require_registered: bool = True
) -> dict[str, dict[str, Any]]:
    """:func:`selection` for every task of an ``x1.json``, keyed by task id (what ``bench run``
    hands the tier cells). The file is first replayed and, with ``recompute`` (the default, what
    ``bench run`` does), recomputed from its items file (:func:`decide_from_json`, refusing an
    unregistered run unless ``require_registered`` is false), so a selection never comes from a
    trace that its own statistics or its items file do not support (§12.7, §14.6)."""
    decide_from_json(path, recompute=recompute, require_registered=require_registered)
    return {rec.task_id: selection(rec) for rec in load_x1(path).x1.tasks.values()}


def fingerprints_for_lock(
    lock: ServingLock, models: Sequence[str] = MODELS
) -> dict[str, dict[str, str]]:
    """The D5 bundle of each model under ``lock`` (the cells' ``fingerprint``)."""
    return {m: lock.fingerprint(m).as_dict() for m in models}


def scorers_for_lock(lock: ServingLock, head: HeadProjector) -> dict[str, OfflineScorer]:
    """The two X1 scorers: ``clm-latest`` through ``head`` (which must be the lock's pinned head,
    ``source_sha256``; K4) under its ``clm_model_fp``, and ``clm-raw``."""
    if head.source_sha256 != lock.head.sha256:
        raise X1DataError("the head export is not the serving lock's head (K4)")
    return {
        LATEST_MODEL: OfflineScorer(
            LATEST_MODEL, head, clm_model_fp=clm_model_fp(lock.model_spec(LATEST_MODEL))
        ),
        RAW_MODEL: OfflineScorer(RAW_MODEL),
    }


def latency_targets(
    manifest: Manifest, task_id: str, framing_id: str, n: int = LATENCY_TARGETS
) -> list[str]:
    """The label-free draw of the p50 timing run (§11.3): the manifest's distinct targets of
    (task, framing) in ``sha256("task|framing|target")`` order, the first ``n``."""
    targets = {
        r.target_sha256
        for r in manifest.rows
        if r.task_id == task_id and r.framing_id == framing_id
    }
    order = sorted(
        targets,
        key=lambda t: hashlib.sha256(f"{task_id}|{framing_id}|{t}".encode()).hexdigest(),
    )
    return order[:n]


def _percentiles(ms: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(ms), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "p50": None, "p95": None}
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def latency_summary(
    ms: Sequence[float],
    first: Sequence[bool] | None = None,
    new_text: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """``{n, p50, p95}`` of a timing run's ms per target (numpy linear percentiles); with
    ``first`` (per target: was this model asked first for it, §11.3) also the same for the
    targets asked first and for those asked second, and with ``new_text`` (per target: did its
    requests carry a text this run had not sent before) the share that did."""
    out: dict[str, Any] = _percentiles(ms)
    if first is not None:
        if len(first) != len(ms):
            raise X1DataError("the timing run's order flags are not one per target")
        out["first"] = _percentiles([v for v, f in zip(ms, first, strict=True) if f])
        out["second"] = _percentiles([v for v, f in zip(ms, first, strict=True) if not f])
    if new_text is not None:
        if len(new_text) != len(ms):
            raise X1DataError("the timing run's new-text flags are not one per target")
        out["new_text_share"] = float(np.mean(new_text)) if len(new_text) else None
    return out


LATENCY_FLAGS: Final[tuple[str, ...]] = ("first", "new_text")


def load_latency(
    path: str | Path, *, labels_sha256: str | None = None, lock_sha: str | None = None
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    """The live timing run's file (``scripts/x1_latency.py``, §11.3): ``ms`` per task id and
    arm, and ``{file, sha256, format, labels_sha256, lock_sha}`` for ``x1.json`` plus, when the
    file has them, the per-target ``first`` and ``new_text`` flags of every arm (what
    :func:`latency_summary` splits on). With ``labels_sha256`` or ``lock_sha`` (the X1 run's
    labels and serving lock) the file must name the same. :class:`X1DataError` for another
    format, an unknown task or arm, a negative time, flags that are not one per target, or
    another snapshot or serving lock."""
    file = Path(path)
    raw = file.read_bytes()
    data = json.loads(raw)
    if data.get("format") != LATENCY_FORMAT:
        raise X1DataError(f"{file}: format {data.get('format')!r}, expected {LATENCY_FORMAT!r}")
    recorded = {
        "labels_sha256": data.get("labels_sha256"),
        "lock_sha": (data.get("serving") or {}).get("lock_sha"),
    }
    for name, want in (("labels_sha256", labels_sha256), ("lock_sha", lock_sha)):
        if want is not None and recorded[name] != want:
            raise X1DataError(
                f"{file}: the timing run's {name} {str(recorded[name])[:12]} is not this run's "
                f"{want[:12]}"
            )
    out: dict[str, dict[str, list[float]]] = {}
    flags: dict[str, dict[str, dict[str, list[bool]]]] = {f: {} for f in LATENCY_FLAGS}
    for task_id, arms in dict(data.get("ms") or {}).items():
        if task_id not in X1_TASKS:
            raise X1DataError(f"{file}: {task_id!r} is not an X1 task")
        out[task_id] = {}
        for arm, values in dict(arms).items():
            if arm not in ARM_ORDER:
                raise X1DataError(f"{file}: {arm!r} is not an X1 arm")
            ms = [float(v) for v in values]
            if any(not math.isfinite(v) or v < 0 for v in ms):
                raise X1DataError(f"{file}: {task_id}/{arm} has a time that is not >= 0")
            out[task_id][arm] = ms
            for flag in LATENCY_FLAGS:
                given = (data.get(flag) or {}).get(task_id, {}).get(arm)
                if given is None:
                    continue
                if len(given) != len(ms) or not all(isinstance(v, bool) for v in given):
                    raise X1DataError(f"{file}: {task_id}/{arm} {flag} is not one flag per target")
                flags[flag].setdefault(task_id, {})[arm] = list(given)
    info: dict[str, Any] = {
        "file": str(file),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": LATENCY_FORMAT,
        **recorded,
    }
    info.update({f: v for f, v in flags.items() if v})
    return out, info
