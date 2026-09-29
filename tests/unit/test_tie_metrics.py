"""Tie-invariant ranking metrics (``mesa_clm.bench.metrics``, DESIGN D33).

The vendored AnyJev metrics sort with an unstable quicksort, so tied confidences land in
platform-dependent bins; the committed baselines cell differed between aarch64 and x86-64 CI.
These tests pin the two properties that fix it: (1) the result never depends on the order of the
items, and (2) on tie-free inputs the value is exactly the vendored one."""

from __future__ import annotations

import numpy as np
import pytest

from mesa_clm.bench import _metrics
from mesa_clm.bench import metrics as tie_metrics


def _binary(conf_yes: np.ndarray) -> np.ndarray:
    return np.stack([conf_yes, 1.0 - conf_yes], axis=1)


def _tied_problem(seed: int = 0, n: int = 240) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    # a handful of distinct values, like Laplace key frequencies or a collapsed zero-shot head
    levels = np.array([0.2, 1 / 3, 0.5, 2 / 3, 0.75, 0.9])
    p_yes = levels[rng.integers(0, len(levels), n)]
    labels = [int(x) for x in (rng.random(n) > p_yes)]
    return _binary(p_yes), labels


@pytest.mark.parametrize("fn", ["ece", "aurc", "cov5", "cov10"])
def test_ranking_metrics_do_not_depend_on_item_order_under_ties(fn: str) -> None:
    probs, labels = _tied_problem()

    def value(p: np.ndarray, y: list[int]) -> float:
        if fn == "ece":
            return tie_metrics.ece(p, y)
        if fn == "aurc":
            return tie_metrics.aurc(p, y)
        return tie_metrics.coverage_at_risk(p, y, 0.05 if fn == "cov5" else 0.10)

    reference = value(probs, labels)
    rng = np.random.default_rng(1)
    for _ in range(25):
        perm = rng.permutation(len(labels))
        assert value(probs[perm], [labels[i] for i in perm]) == pytest.approx(reference, abs=1e-12)


def test_vendored_ece_is_order_dependent_under_ties() -> None:
    """The defect being fixed: the vendored ECE moves when tied items are reordered."""
    probs, labels = _tied_problem()
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(25):
        perm = rng.permutation(len(labels))
        seen.add(round(_metrics.ece(probs[perm], [labels[i] for i in perm]), 12))
    assert len(seen) > 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equal_to_vendored_on_tie_free_inputs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    p_yes = rng.random(n)  # continuous: no ties
    labels = [int(x) for x in (rng.random(n) > p_yes)]
    probs = _binary(p_yes)
    assert len(np.unique(np.max(probs, 1))) == n
    assert tie_metrics.ece(probs, labels) == pytest.approx(_metrics.ece(probs, labels), abs=1e-12)
    assert tie_metrics.aurc(probs, labels) == pytest.approx(_metrics.aurc(probs, labels), abs=1e-12)
    for target in (0.05, 0.10, 0.3):
        assert tie_metrics.coverage_at_risk(probs, labels, target) == _metrics.coverage_at_risk(
            probs, labels, target
        )
    ours, theirs = tie_metrics.summarize(probs, labels), _metrics.summarize(probs, labels)
    assert set(ours) == set(theirs)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key], abs=1e-12), key


def test_tie_averaged_replaces_correctness_by_group_mean() -> None:
    conf = np.array([0.5, 0.5, 0.9, 0.9, 0.9, 0.7])
    correct = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    out = tie_metrics.tie_averaged(conf, correct)
    assert out.tolist() == pytest.approx([0.5, 0.5, 2 / 3, 2 / 3, 2 / 3, 1.0])


def test_coverage_risk_curve_is_the_expected_curve_over_random_tie_breaking() -> None:
    """For a single tie group the risk at every prefix equals the group's error rate, the
    expectation over random orderings of the group."""
    probs = _binary(np.full(10, 0.8))
    labels = [0] * 7 + [1] * 3  # 7 correct (argmax is class 0), 3 wrong
    cov, err = tie_metrics.coverage_risk(probs, labels)
    assert cov.tolist() == pytest.approx([(i + 1) / 10 for i in range(10)])
    assert err.tolist() == pytest.approx([0.3] * 10)
