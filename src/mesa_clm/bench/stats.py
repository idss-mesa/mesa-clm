"""Bench statistics: the card-cluster paired bootstrap, the card sign test, rule R,
non-inferiority, AUROC with a cluster CI, and Clopper-Pearson thresholds (DESIGN D8, D27; plan
§4.7, §5.4).

Seven cards are seven clusters: items of one card share its columns, sites and product, so
item-level intervals overstate what the bench knows. Every interval here resamples *cards*
(the held-out clusters), never items, with ``B=2000`` replicates and ``seed=0`` (plan §5.4), and
the paired versions resample the same cards for both arms so the difference is paired.

**Rule R** ("A ≻ B on metric m") needs both: (i) the one-sided 95% lower bound of the cluster
bootstrap of Δm (positive = A better) is > 0; (ii) A beats B on at least ⌈0.8·m_c⌉ of the m_c
held-out cards with at least 10 evaluable items; fewer than 4 such cards is
``insufficient_clusters``. "A ≽ B within δ" (non-inferiority) is a cluster lower bound > −δ.
Effect floors (AUROC ≥ 0.60, ECE ≤ 0.08) are always paired with one of these. When B is the
mean over K scorings of the same items (X1's within-card shuffles), :func:`rule_r_auroc_mean`
computes both conditions on Δ = AUROC(A) − mean_k AUROC(B_k), exactly, through the Mann-Whitney
pair kernel averaged over the K scorings (:func:`pair_kernel`, :func:`kernel_auroc`).

**Clopper-Pearson** (plan §4.7): a numeric ``auto`` must sit at or above ``threshold_cp[risk]``,
the smallest statistic ``t`` whose pooled held-out set ``{stat ≥ t}`` has at least 30 items and a
one-sided 95% Clopper-Pearson upper bound on its error rate at or below ``risk``. scipy is not a
dependency, so the beta quantile is computed here: the regularized incomplete beta by Lentz's
continued fraction (Numerical Recipes ``betacf``) and the quantile by bisection.

Probabilities are ``[n, K]`` arrays over a task's options, labels are option indices, and the
positive class of a two-class task is index 0 ("Yes"), as everywhere in mesa-clm. ``ece`` is the
vendored AnyJev top-1 equal-mass estimator (:mod:`mesa_clm.bench._metrics`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from mesa_clm.bench import _metrics
from mesa_clm.bench import metrics as tie_metrics

Metric = Literal["acc", "nll", "brier", "auroc", "ece"]

HIGHER_IS_BETTER: Final[dict[str, bool]] = {
    "acc": True,
    "auroc": True,
    "nll": False,
    "brier": False,
    "ece": False,
}

DEFAULT_B: Final = 2000
DEFAULT_SEED: Final = 0
DEFAULT_ALPHA: Final = 0.05
# Rule R (ii): a card counts when it has at least this many evaluable items, and A must win on
# at least this fraction of the counting cards; fewer than MIN_CLUSTERS counting cards is
# ``insufficient_clusters``.
SIGN_MIN_ITEMS: Final = 10
SIGN_FRACTION: Final = 0.8
MIN_CLUSTERS: Final = 4
# The default risks a cell reports thresholds for (plan §5.4 ``threshold_cp{0.05,0.10}``).
THRESHOLD_RISKS: Final[tuple[float, ...]] = (0.05, 0.10)
CP_MIN_N: Final = 30

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# -- pooled metrics --------------------------------------------------------------------------------


def _probs(probs: npt.ArrayLike) -> FloatArray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs must be a 2-d [n, K] array")
    return p


def _labels(labels: Sequence[int] | npt.ArrayLike, n: int) -> IntArray:
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (n,):
        raise ValueError(f"labels must have one entry per row ({y.shape} vs {n})")
    return y


def as_list(labels: IntArray) -> list[int]:
    """Labels as the ``Sequence[int]`` the vendored metrics are typed for."""
    return [int(v) for v in labels.tolist()]


def auroc(
    scores: npt.ArrayLike, labels: Sequence[int] | npt.ArrayLike, *, positive: int = 0
) -> float:
    """Area under the ROC curve of ``scores`` for the ``positive`` class: the Mann-Whitney
    statistic with ties counted one half. ``nan`` when a class is missing."""
    s = np.asarray(scores, dtype=float)
    y = _labels(labels, len(s))
    out = _weighted_auroc(s, y == positive, np.ones((1, len(s))))
    return float(out[0])


def metric_value(
    metric: Metric, probs: npt.ArrayLike, labels: Sequence[int] | npt.ArrayLike
) -> float:
    """One pooled metric over ``probs`` and ``labels`` (``auroc`` scores the positive class,
    index 0, of a two-class task; ``nan`` for a wider task)."""
    p = _probs(probs)
    y = _labels(labels, len(p))
    if len(y) == 0:
        return math.nan
    if metric == "auroc":
        return auroc(p[:, 0], y, positive=0) if p.shape[1] == 2 else math.nan
    if metric == "acc":
        return _metrics.accuracy(p, as_list(y))
    if metric == "nll":
        return _metrics.nll(p, as_list(y))
    if metric == "brier":
        return _metrics.brier(p, as_list(y))
    if metric == "ece":
        return tie_metrics.ece(p, as_list(y))
    raise ValueError(f"unknown metric {metric!r}")


# -- weighted (bootstrap) metrics -----------------------------------------------------------------


def _weighted_auroc(scores: FloatArray, pos: npt.NDArray[np.bool_], w: FloatArray) -> FloatArray:
    """AUROC per row of the weight matrix ``w`` ([B, n], item multiplicities): the weighted
    Mann-Whitney statistic Σ_pos Σ_neg w_i w_j [s_i > s_j] (+½ for ties) / (W_pos · W_neg),
    vectorised over B with one sort. ``nan`` where a row has no positive or no negative mass."""
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    pos_s = pos[order]
    w_s = w[:, order]
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    group = np.cumsum(np.concatenate(([True], s[1:] != s[:-1]))) - 1  # group id per sorted item
    neg_per_group = np.add.reduceat(w_s * ~pos_s, starts, axis=1)  # [B, G]
    below = np.cumsum(neg_per_group, axis=1) - neg_per_group  # strictly lower negative mass
    per_item = below[:, group] + 0.5 * neg_per_group[:, group]  # [B, n], for every sorted item
    num = np.sum(w_s * pos_s * per_item, axis=1)
    w_pos = np.sum(w_s * pos_s, axis=1)
    w_neg = np.sum(w_s * ~pos_s, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: FloatArray = np.where((w_pos > 0) & (w_neg > 0), num / (w_pos * w_neg), np.nan)
    return out


def weighted_metric(
    metric: Metric, probs: FloatArray, labels: IntArray, w: FloatArray
) -> FloatArray:
    """``metric`` on every weighted copy of the items: ``w`` is ``[B, n]`` (item multiplicities
    from a cluster resample); returns ``[B]``. ``acc``, ``nll``, ``brier`` and ``auroc`` are
    computed in closed form over the weights; ``ece`` materialises each resample (the equal-mass
    binning has no weighted form)."""
    n = len(labels)
    if metric == "ece":
        out = np.empty(len(w))
        for b in range(len(w)):
            idx = np.repeat(np.arange(n), w[b].astype(np.int64))
            out[b] = tie_metrics.ece(probs[idx], as_list(labels[idx])) if len(idx) else np.nan
        return out
    if metric == "auroc":
        if probs.shape[1] != 2:
            return np.full(len(w), np.nan)
        return _weighted_auroc(probs[:, 0], labels == 0, w)
    if metric == "acc":
        per_item = (np.argmax(probs, axis=1) == labels).astype(float)
    elif metric == "nll":
        per_item = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, None))
    elif metric == "brier":
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), labels] = 1.0
        per_item = np.sum((probs - onehot) ** 2, axis=1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    total = w.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result: FloatArray = np.where(total > 0, (w @ per_item) / total, np.nan)
    return result


def bootstrap_weights(
    clusters: Sequence[str], *, B: int = DEFAULT_B, seed: int = DEFAULT_SEED
) -> FloatArray:
    """Item multiplicities ``[B, n]`` of ``B`` cluster resamples: each replicate draws the
    ``m`` distinct clusters (sorted, so the draw is independent of item order) with
    replacement and keeps every item of a drawn cluster once per draw. Deterministic in
    ``seed`` (``numpy.random.default_rng``)."""
    names = sorted(set(clusters))
    index = {c: i for i, c in enumerate(names)}
    member = np.asarray([index[c] for c in clusters], dtype=np.int64)
    rng = np.random.default_rng(seed)
    m = len(names)
    draws = rng.integers(0, m, size=(B, m))  # [B, m] cluster ids drawn with replacement
    counts = np.zeros((B, m), dtype=np.int64)
    for j in range(m):
        counts[:, j] = np.sum(draws == j, axis=1)
    return counts[:, member].astype(float)


# -- intervals -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapCI:
    """A cluster-bootstrap interval. ``lower`` and ``upper`` are the one-sided ``1-alpha``
    bounds (the ``alpha`` and ``1-alpha`` quantiles of the replicates); ``n_valid`` counts the
    replicates whose statistic was defined (an AUROC resample can miss a class)."""

    point: float
    lower: float
    upper: float
    B: int
    seed: int
    alpha: float
    n_clusters: int
    n_valid: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "B": self.B,
            "seed": self.seed,
            "alpha": self.alpha,
            "n_clusters": self.n_clusters,
            "n_valid": self.n_valid,
        }


def _quantiles(values: FloatArray, alpha: float) -> tuple[float, float, int]:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return math.nan, math.nan, 0
    return float(np.quantile(valid, alpha)), float(np.quantile(valid, 1 - alpha)), len(valid)


def metric_ci(
    metric: Metric,
    probs: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> BootstrapCI:
    """The cluster-bootstrap interval of one metric (the ``auroc[cluster CI]`` of a cell)."""
    p = _probs(probs)
    y = _labels(labels, len(p))
    if len(clusters) != len(p):
        raise ValueError("clusters must have one entry per item")
    w = bootstrap_weights(clusters, B=B, seed=seed)
    values = weighted_metric(metric, p, y, w)
    lower, upper, n_valid = _quantiles(values, alpha)
    return BootstrapCI(
        point=metric_value(metric, p, y),
        lower=lower,
        upper=upper,
        B=B,
        seed=seed,
        alpha=alpha,
        n_clusters=len(set(clusters)),
        n_valid=n_valid,
    )


def improvement(metric: Metric, value_a: float, value_b: float) -> float:
    """Δm signed so that positive means A is better: ``a - b`` for a higher-is-better metric,
    ``b - a`` otherwise."""
    return value_a - value_b if HIGHER_IS_BETTER[metric] else value_b - value_a


def paired_bootstrap(
    metric: Metric,
    probs_a: npt.ArrayLike,
    probs_b: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    weights: FloatArray | None = None,
) -> BootstrapCI:
    """The card-cluster paired bootstrap of Δm = improvement(A over B) (plan §5.4, rule R (i)):
    both arms are scored on the same held-out items and the same resampled cards, so the
    ``lower`` bound is the one-sided 95% (``alpha=0.05``) lower bound of a paired difference.
    ``weights`` (``[B, n]`` multiplicities from :func:`bootstrap_weights` for these clusters)
    lets a simulation reuse one resample across many arms; they must match ``B``."""
    pa = _probs(probs_a)
    pb = _probs(probs_b)
    if pa.shape != pb.shape:
        raise ValueError("the two arms must score the same items with the same K")
    y = _labels(labels, len(pa))
    if len(clusters) != len(pa):
        raise ValueError("clusters must have one entry per item")
    w = bootstrap_weights(clusters, B=B, seed=seed) if weights is None else weights
    if w.shape != (B, len(pa)):
        raise ValueError(f"weights must be [B, n] = {(B, len(pa))}, got {w.shape}")
    va = weighted_metric(metric, pa, y, w)
    vb = weighted_metric(metric, pb, y, w)
    deltas = va - vb if HIGHER_IS_BETTER[metric] else vb - va
    lower, upper, n_valid = _quantiles(deltas, alpha)
    return BootstrapCI(
        point=improvement(metric, metric_value(metric, pa, y), metric_value(metric, pb, y)),
        lower=lower,
        upper=upper,
        B=B,
        seed=seed,
        alpha=alpha,
        n_clusters=len(set(clusters)),
        n_valid=n_valid,
    )


# -- sign test and rule R -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SignTest:
    """Rule R (ii): ``wins`` of ``m_c`` counting cards (≥ ``min_items`` evaluable items), the
    ``needed`` ⌈fraction·m_c⌉ and whether that was reached. ``reason`` is ``passed``,
    ``insufficient_clusters`` (m_c < 4) or ``sign_test_failed``."""

    m_c: int
    wins: int
    needed: int
    passed: bool
    reason: str
    per_card: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "m_c": self.m_c,
            "wins": self.wins,
            "needed": self.needed,
            "passed": self.passed,
            "reason": self.reason,
            "per_card": dict(sorted(self.per_card.items())),
        }


def card_sign_test(
    metric: Metric,
    probs_a: npt.ArrayLike,
    probs_b: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    min_items: int = SIGN_MIN_ITEMS,
    fraction: float = SIGN_FRACTION,
    min_clusters: int = MIN_CLUSTERS,
) -> SignTest:
    """Does A beat B on at least ⌈fraction·m_c⌉ of the m_c cards with at least ``min_items``
    evaluable items? An item is evaluable when the metric is defined on its card (for AUROC the
    card needs both classes); a card with fewer evaluable items does not count. A strict
    improvement is a win. ``per_card`` records Δm for every counting card."""
    pa = _probs(probs_a)
    pb = _probs(probs_b)
    y = _labels(labels, len(pa))
    if pa.shape != pb.shape or len(clusters) != len(pa):
        raise ValueError("the two arms and clusters must cover the same items")
    groups = np.asarray(clusters)
    per_card: dict[str, float] = {}
    for card in sorted(set(clusters)):
        idx = np.flatnonzero(groups == card)
        if len(idx) < min_items:
            continue
        va = metric_value(metric, pa[idx], y[idx])
        vb = metric_value(metric, pb[idx], y[idx])
        if math.isnan(va) or math.isnan(vb):
            continue  # the metric is undefined on this card (AUROC with one class)
        per_card[card] = improvement(metric, va, vb)
    return sign_verdict(per_card, fraction=fraction, min_clusters=min_clusters)


def sign_verdict(
    per_card: dict[str, float],
    *,
    fraction: float = SIGN_FRACTION,
    min_clusters: int = MIN_CLUSTERS,
) -> SignTest:
    """Rule R (ii) from the Δm of every counting card (positive = A better): a strict
    improvement is a win, A needs ⌈fraction·m_c⌉ wins, fewer than ``min_clusters`` counting
    cards is ``insufficient_clusters``."""
    m_c = len(per_card)
    wins = sum(1 for d in per_card.values() if d > 0)
    needed = math.ceil(fraction * m_c)
    if m_c < min_clusters:
        return SignTest(m_c, wins, needed, False, "insufficient_clusters", per_card)
    passed = wins >= needed
    return SignTest(m_c, wins, needed, passed, "passed" if passed else "sign_test_failed", per_card)


def rule_r_reason(sign: SignTest, lower_bound: float) -> str:
    """The first failing condition of rule R, in the order the verdict reports it:
    ``insufficient_clusters``, ``bootstrap_lower_bound_not_positive``, ``sign_test_failed``, or
    ``passed``."""
    if sign.reason == "insufficient_clusters":
        return "insufficient_clusters"
    if not (lower_bound > 0):
        return "bootstrap_lower_bound_not_positive"
    if not sign.passed:
        return "sign_test_failed"
    return "passed"


@dataclass(frozen=True)
class RuleR:
    """The verdict of rule R for ``metric``: ``passed`` iff the paired cluster lower bound is
    > 0 and the sign test passed. ``reason`` names the first failing condition."""

    metric: str
    delta: float
    lower_bound: float
    bootstrap: BootstrapCI
    sign: SignTest
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "delta": self.delta,
            "lower_bound": self.lower_bound,
            "bootstrap": self.bootstrap.as_dict(),
            "sign": self.sign.as_dict(),
            "passed": self.passed,
            "reason": self.reason,
        }


def rule_r(
    metric: Metric,
    probs_a: npt.ArrayLike,
    probs_b: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    weights: FloatArray | None = None,
) -> RuleR:
    """Rule R: "A ≻ B on ``metric``" (plan §5.4). Both conditions are always evaluated so the
    record shows how far each was from passing. ``weights`` as in :func:`paired_bootstrap`."""
    ci = paired_bootstrap(
        metric, probs_a, probs_b, labels, clusters, B=B, seed=seed, alpha=alpha, weights=weights
    )
    sign = card_sign_test(metric, probs_a, probs_b, labels, clusters)
    reason = rule_r_reason(sign, ci.lower)
    return RuleR(metric, ci.point, ci.lower, ci, sign, reason == "passed", reason)


# -- AUROC of a raw score (X1) ---------------------------------------------------------------------


def score_matrix(scores: npt.ArrayLike) -> FloatArray:
    """``[n, 2]`` with ``scores`` in column 0 and their negation in column 1: the shape the AUROC
    paths of :func:`metric_value`, :func:`weighted_metric`, :func:`paired_bootstrap` and
    :func:`card_sign_test` read, which look only at column 0 for ``auroc``. For a raw score such
    as X1's ``s_c`` (DESIGN X1), whose sigmoid rounds to exactly 1.0 in float64 above about 37 and
    would add ties the score does not have. Column 1 is not a probability: never pass the matrix
    to another metric."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be a 1-d array")
    out: FloatArray = np.column_stack([s, -s])
    return out


def auroc_ci(
    scores: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    weights: FloatArray | None = None,
) -> BootstrapCI:
    """The card-cluster bootstrap interval of the AUROC of a raw score for the positive class
    (index 0): :func:`metric_ci` with metric ``auroc`` on :func:`score_matrix`. ``weights``
    (``[B, n]`` multiplicities from :func:`bootstrap_weights` for these clusters) lets several
    scores of the same items share one resample, as in :func:`paired_bootstrap`; the result is
    identical to drawing it here with the same ``B`` and ``seed``."""
    p = score_matrix(scores)
    y = _labels(labels, len(p))
    if len(clusters) != len(p):
        raise ValueError("clusters must have one entry per item")
    w = bootstrap_weights(clusters, B=B, seed=seed) if weights is None else weights
    if w.shape != (B, len(p)):
        raise ValueError(f"weights must be [B, n] = {(B, len(p))}, got {w.shape}")
    values = weighted_metric("auroc", p, y, w)
    lower, upper, n_valid = _quantiles(values, alpha)
    return BootstrapCI(
        point=metric_value("auroc", p, y),
        lower=lower,
        upper=upper,
        B=B,
        seed=seed,
        alpha=alpha,
        n_clusters=len(set(clusters)),
        n_valid=n_valid,
    )


def rule_r_auroc(
    scores_a: npt.ArrayLike,
    scores_b: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    weights: FloatArray | None = None,
) -> RuleR:
    """Rule R "A ≻ B on AUROC" for two raw scores of the same items (X1's ΔAUROC(real −
    shuffle)): :func:`rule_r` with metric ``auroc`` on :func:`score_matrix` of each, so a card
    counts for the sign test when it holds at least 10 items and both classes."""
    return rule_r(
        "auroc",
        score_matrix(scores_a),
        score_matrix(scores_b),
        labels,
        clusters,
        B=B,
        seed=seed,
        alpha=alpha,
        weights=weights,
    )


# -- the mean AUROC over several scorings of the same items (X1's within-card shuffle) ----------------


def _rows(scores: npt.ArrayLike, n: int | None = None) -> FloatArray:
    s = np.asarray(scores, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2 or s.shape[0] == 0 or (n is not None and s.shape[1] != n):
        raise ValueError("score rows must be a non-empty [K, n] array over the same items")
    if not np.all(np.isfinite(s)):
        raise ValueError("score rows must be finite")
    return s


def pair_kernel(scores: npt.ArrayLike, positive: npt.ArrayLike) -> FloatArray:
    """``[P, N]``: for every (positive ``i``, negative ``j``) pair, the mean over the ``K`` rows
    of ``scores`` (``[K, n]``, or one ``[n]`` row) of ``H(s_i − s_j)``, with ``H`` 1 above 0, ½
    at 0, 0 below: the Mann-Whitney kernel. The AUROC of a row is the mean of its kernel over
    the pairs, so the mean of the ``K`` rows' AUROCs over any weighted set of items is the
    weighted mean of this one matrix (:func:`kernel_auroc`), exactly."""
    pos = np.asarray(positive, dtype=bool)
    s = _rows(scores, len(pos))
    p_idx, n_idx = np.flatnonzero(pos), np.flatnonzero(~pos)
    acc = np.zeros((len(p_idx), len(n_idx)), dtype=float)
    for row in s:
        d = row[p_idx][:, None] - row[n_idx][None, :]
        acc += (d > 0) + 0.5 * (d == 0)
    out: FloatArray = acc / len(s)
    return out


def kernel_auroc(kernel: FloatArray, positive: npt.ArrayLike, w: FloatArray) -> FloatArray:
    """The AUROC that ``kernel`` (:func:`pair_kernel`) gives under each row of item
    multiplicities ``w`` (``[B, n]``): Σ w_i w_j k_ij / (W_pos · W_neg); ``nan`` where a row has
    no positive or no negative mass (as :func:`weighted_metric`)."""
    pos = np.asarray(positive, dtype=bool)
    wp, wn = w[:, pos], w[:, ~pos]
    num = np.einsum("bp,pn,bn->b", wp, kernel, wn)
    den = wp.sum(axis=1) * wn.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: FloatArray = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def _kernel_point(kernel: FloatArray) -> float:
    return float(kernel.mean()) if kernel.size else math.nan


def mean_auroc_ci(
    scores: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    weights: FloatArray | None = None,
) -> BootstrapCI:
    """The card-cluster bootstrap interval of the mean AUROC (positive class index 0) of the
    ``K`` score rows of ``scores`` (``[K, n]``): in each replicate, the mean over the rows of
    their AUROC on the same resampled cards. With one row it equals :func:`auroc_ci`."""
    y = _labels(labels, np.asarray(scores).shape[-1])
    pos = y == 0
    kernel = pair_kernel(scores, pos)
    if len(clusters) != len(y):
        raise ValueError("clusters must have one entry per item")
    w = bootstrap_weights(clusters, B=B, seed=seed) if weights is None else weights
    if w.shape != (B, len(y)):
        raise ValueError(f"weights must be [B, n] = {(B, len(y))}, got {w.shape}")
    lower, upper, n_valid = _quantiles(kernel_auroc(kernel, pos, w), alpha)
    return BootstrapCI(
        point=_kernel_point(kernel),
        lower=lower,
        upper=upper,
        B=B,
        seed=seed,
        alpha=alpha,
        n_clusters=len(set(clusters)),
        n_valid=n_valid,
    )


def rule_r_auroc_mean(
    scores_a: npt.ArrayLike,
    scores_b: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    weights: FloatArray | None = None,
    min_items: int = SIGN_MIN_ITEMS,
    fraction: float = SIGN_FRACTION,
    min_clusters: int = MIN_CLUSTERS,
) -> RuleR:
    """Rule R "A ≻ B on AUROC" where B is the **mean AUROC of K scorings** of the same items
    (``scores_b`` ``[K, n]``; X1's ΔAUROC(real − shuffle) with K within-card shuffles,
    ``design/m2-analysis-plan.md`` §5.4): (i) the card-cluster paired bootstrap of
    Δ = AUROC(A) − mean_k AUROC(B_k), both on the same resampled cards in every replicate, has a
    one-sided lower bound > 0; (ii) Δ restricted
    to a card is > 0 on ⌈fraction·m_c⌉ of the m_c cards with at least ``min_items`` items and
    both classes. With ``K = 1`` it is :func:`rule_r_auroc` (to rounding)."""
    y = _labels(labels, np.asarray(scores_a).shape[-1])
    pos = y == 0
    a = _rows(scores_a, len(y))
    if a.shape[0] != 1:
        raise ValueError("scores_a must be one [n] score")
    b = _rows(scores_b, len(y))
    if len(clusters) != len(y):
        raise ValueError("clusters must have one entry per item")
    diff = pair_kernel(a, pos) - pair_kernel(b, pos)
    w = bootstrap_weights(clusters, B=B, seed=seed) if weights is None else weights
    if w.shape != (B, len(y)):
        raise ValueError(f"weights must be [B, n] = {(B, len(y))}, got {w.shape}")
    lower, upper, n_valid = _quantiles(kernel_auroc(diff, pos, w), alpha)
    ci = BootstrapCI(
        point=_kernel_point(diff),
        lower=lower,
        upper=upper,
        B=B,
        seed=seed,
        alpha=alpha,
        n_clusters=len(set(clusters)),
        n_valid=n_valid,
    )
    groups = np.asarray(clusters)
    p_idx, n_idx = np.flatnonzero(pos), np.flatnonzero(~pos)
    per_card: dict[str, float] = {}
    for card in sorted(set(clusters)):
        on = groups == card
        if int(on.sum()) < min_items:
            continue
        rows, cols = np.flatnonzero(on[p_idx]), np.flatnonzero(on[n_idx])
        if len(rows) == 0 or len(cols) == 0:
            continue  # AUROC is undefined on a card with one class
        per_card[card] = float(diff[np.ix_(rows, cols)].mean())
    sign = sign_verdict(per_card, fraction=fraction, min_clusters=min_clusters)
    reason = rule_r_reason(sign, ci.lower)
    return RuleR("auroc", ci.point, ci.lower, ci, sign, reason == "passed", reason)


@dataclass(frozen=True)
class NonInferiority:
    """ "A ≽ B within δ": the paired cluster lower bound of Δm is > −δ."""

    metric: str
    delta: float
    lower_bound: float
    margin: float
    passed: bool
    bootstrap: BootstrapCI

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "delta": self.delta,
            "lower_bound": self.lower_bound,
            "margin": self.margin,
            "passed": self.passed,
            "bootstrap": self.bootstrap.as_dict(),
        }


def non_inferior(
    metric: Metric,
    probs_a: npt.ArrayLike,
    probs_b: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    clusters: Sequence[str],
    margin: float,
    *,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> NonInferiority:
    """Non-inferiority of A to B on ``metric`` within ``margin`` (δ ≥ 0), e.g. K2(a)'s
    ``Δacc cluster-LB > −0.02``."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    ci = paired_bootstrap(metric, probs_a, probs_b, labels, clusters, B=B, seed=seed, alpha=alpha)
    return NonInferiority(metric, ci.point, ci.lower, margin, ci.lower > -margin, ci)


# -- Clopper-Pearson -----------------------------------------------------------------------------------

_CF_MAX_ITER: Final = 500
_CF_EPS: Final = 1e-15
_CF_FPMIN: Final = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for the incomplete beta function (Numerical Recipes 6.4)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_FPMIN:
        d = _CF_FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_FPMIN:
            d = _CF_FPMIN
        c = 1.0 + aa / c
        if abs(c) < _CF_FPMIN:
            c = _CF_FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_FPMIN:
            d = _CF_FPMIN
        c = 1.0 + aa / c
        if abs(c) < _CF_FPMIN:
            c = _CF_FPMIN
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _CF_EPS:
            break
    return h


def betainc(x: float, a: float, b: float) -> float:
    """The regularized incomplete beta function I_x(a, b) (the Beta(a, b) CDF at ``x``), by the
    continued fraction on the side of the symmetry point where it converges fast."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float, *, tol: float = 1e-13) -> float:
    """The Beta(a, b) quantile at ``q`` by bisection on :func:`betainc` (monotone in x)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def clopper_pearson_upper(k: int, n: int, alpha: float = DEFAULT_ALPHA) -> float:
    """The one-sided ``1-alpha`` Clopper-Pearson upper bound on a binomial proportion after
    ``k`` events in ``n`` trials: the ``1-alpha`` quantile of Beta(k+1, n-k); 1.0 when
    ``k == n``. ``k=0`` gives ``1 - alpha**(1/n)`` exactly."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if k >= n:
        return 1.0
    return beta_ppf(1.0 - alpha, k + 1.0, float(n - k))


def threshold_cp(
    stat: npt.ArrayLike,
    correct: npt.ArrayLike,
    risk: float,
    *,
    min_n: int = CP_MIN_N,
    alpha: float = DEFAULT_ALPHA,
) -> float | None:
    """The smallest threshold ``t`` (among the observed statistics) whose pooled held-out set
    ``{stat >= t}`` has at least ``min_n`` items and a one-sided ``1-alpha`` Clopper-Pearson
    upper bound on its error rate at or below ``risk`` (plan §4.7); ``None`` when no
    threshold qualifies. Smallest, because a lower threshold auto-writes more."""
    s = np.asarray(stat, dtype=float)
    c = np.asarray(correct, dtype=bool)
    if s.shape != c.shape or s.ndim != 1:
        raise ValueError("stat and correct must be parallel 1-d arrays")
    if not 0.0 < risk < 1.0:
        raise ValueError("risk must be in (0, 1)")
    order = np.argsort(-s, kind="stable")  # descending: the prefix {stat >= t} grows with t
    errors = np.cumsum(~c[order])
    sizes = np.arange(1, len(s) + 1)
    s_sorted = s[order]
    best: float | None = None
    for i in range(len(s)):
        if i + 1 < len(s) and s_sorted[i + 1] == s_sorted[i]:
            continue  # the set {stat >= t} ends at the last item of a tie group
        n = int(sizes[i])
        if n < min_n:
            continue
        if clopper_pearson_upper(int(errors[i]), n, alpha) <= risk:
            best = float(s_sorted[i])  # ties: lower t means larger coverage; keep scanning
    return best


def policy_stat(
    probs: npt.ArrayLike, labels: Sequence[int] | npt.ArrayLike, *, rank_fit: bool
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """The statistic a policy thresholds and whether an auto at that item would be right:
    ``p_fit`` (the probability of "Yes", index 0) and ``label == Yes`` for a rank_fit task;
    ``confidence = max(probs)`` and ``argmax == label`` for a closed choice (plan §4.7)."""
    p = _probs(probs)
    y = _labels(labels, len(p))
    if rank_fit:
        return p[:, 0], y == 0
    return np.max(p, axis=1), np.argmax(p, axis=1) == y


def thresholds_cp(
    probs: npt.ArrayLike,
    labels: Sequence[int] | npt.ArrayLike,
    *,
    rank_fit: bool,
    risks: Sequence[float] = THRESHOLD_RISKS,
    min_n: int = CP_MIN_N,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float | None]:
    """``{"0.05": t, "0.10": t}``: :func:`threshold_cp` at each risk over the policy statistic."""
    stat, correct = policy_stat(probs, labels, rank_fit=rank_fit)
    return {f"{r:.2f}": threshold_cp(stat, correct, r, min_n=min_n, alpha=alpha) for r in risks}
