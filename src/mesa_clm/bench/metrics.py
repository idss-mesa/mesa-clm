"""Tie-invariant ranking metrics: ``ece``, ``coverage_risk``, ``coverage_at_risk``, ``aurc``.

The vendored AnyJev metrics (``bench/_metrics.py``, byte-identical, never edited) sort items by
confidence with ``np.argsort``'s default, unstable quicksort. When confidences tie, the order of
the tied items is whatever the platform's sort kernel produces, and that order decides which tied
items fall into which equal-mass ECE bin and which prefix of the risk-coverage curve they join.
The lookup baseline (a handful of distinct Laplace frequencies) and a collapsed zero-shot CLM
(near-constant outputs, issue #15) are almost all ties, so the same inputs gave ECE 0.1452 on
aarch64 and 0.1473 on x86-64 CI (DESIGN D33, RESEARCH.md).

The versions here replace each item's correctness by the mean correctness of its tie group (the
items sharing its exact confidence) before binning or accumulating. That is the expectation over
uniformly random tie-breaking for the risk-coverage curve, and a canonical, order-free value for
ECE. On tie-free inputs every function returns exactly what the vendored one returns (a test pins
it), so numbers stay comparable with mesa-anyjev's committed cells, whose continuous L2
probabilities have no ties. Accuracy, macro-F1, Brier and NLL do not sort and are taken from the
vendored module unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from mesa_clm.bench import _metrics

FloatArray = npt.NDArray[np.float64]


def _conf_correct(probs: npt.ArrayLike, labels: Sequence[int]) -> tuple[FloatArray, FloatArray]:
    p = np.asarray(probs, dtype=float)
    conf = np.max(p, 1)
    correct = (np.argmax(p, 1) == np.asarray(labels)).astype(float)
    return conf, correct


def tie_averaged(conf: FloatArray, correct: FloatArray) -> FloatArray:
    """``correct`` with every value replaced by the mean over the items sharing its confidence."""
    uniq, inverse = np.unique(conf, return_inverse=True)
    sums = np.bincount(inverse, weights=correct, minlength=len(uniq))
    counts = np.bincount(inverse, minlength=len(uniq))
    out: FloatArray = (sums / counts)[inverse]
    return out


def ece(probs: npt.ArrayLike, labels: Sequence[int], n_bins: int = 15) -> float:
    """Expected calibration error on top-1 confidence with equal-mass bins (vendored formula),
    invariant to the order of tied confidences."""
    conf, correct = _conf_correct(probs, labels)
    correct = tie_averaged(conf, correct)
    order = np.argsort(conf, kind="stable")
    conf, correct = conf[order], correct[order]
    n = len(conf)
    total = 0.0
    for chunk in np.array_split(np.arange(n), min(n_bins, n)):
        if len(chunk) == 0:
            continue
        total += len(chunk) / n * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(total)


def coverage_risk(probs: npt.ArrayLike, labels: Sequence[int]) -> tuple[FloatArray, FloatArray]:
    """``(coverage, risk)`` sorted by confidence descending, invariant to tie order."""
    conf, correct = _conf_correct(probs, labels)
    correct = tie_averaged(conf, correct)
    order = np.argsort(-conf, kind="stable")
    n = len(conf)
    err: FloatArray = 1.0 - np.cumsum(correct[order]) / np.arange(1, n + 1)
    cov: FloatArray = np.arange(1, n + 1) / n
    return cov, err


def coverage_at_risk(probs: npt.ArrayLike, labels: Sequence[int], target: float = 0.05) -> float:
    """Largest coverage whose empirical risk is at or below ``target``."""
    cov, err = coverage_risk(probs, labels)
    ok = np.where(err <= target)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def aurc(probs: npt.ArrayLike, labels: Sequence[int]) -> float:
    """Area under the risk-coverage curve."""
    cov, err = coverage_risk(probs, labels)
    return float(np.trapezoid(err, cov))


def summarize(probs: npt.ArrayLike, labels: Sequence[int]) -> dict[str, float]:
    """The vendored ``summarize`` keys, with the ranking metrics taken from this module."""
    p = np.asarray(probs, dtype=float)
    return {
        "n": len(labels),
        "acc": float(_metrics.accuracy(p, labels)),
        "macro_f1": float(_metrics.macro_f1(p, labels)),
        "brier": float(_metrics.brier(p, labels)),
        "nll": float(_metrics.nll(p, labels)),
        "ece": ece(p, labels),
        "cov@5%": coverage_at_risk(p, labels, 0.05),
        "aurc": aurc(p, labels),
    }
