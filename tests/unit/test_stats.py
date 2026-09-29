"""Bench statistics (plan §4.7, §5.4): AUROC and its weighted form, the card-cluster bootstrap
(paired and single), the card sign test, rule R, non-inferiority, the pure-numpy regularized
incomplete beta, Clopper-Pearson upper bounds and the CP threshold rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mesa_clm.bench import _metrics, stats

RNG = np.random.default_rng(12345)


def _binary(p_yes: np.ndarray) -> np.ndarray:
    return np.stack([p_yes, 1.0 - p_yes], axis=1)


def _brute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 0]
    neg = scores[labels == 1]
    total = 0.0
    for p in pos:
        for q in neg:
            total += 1.0 if p > q else (0.5 if p == q else 0.0)
    return total / (len(pos) * len(neg))


# -- AUROC ------------------------------------------------------------------------------------------------


def test_auroc_basic_cases_and_brute_force_agreement() -> None:
    assert stats.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 1.0
    assert stats.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 0.0
    assert stats.auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5
    assert stats.auroc([0.9, 0.5, 0.5, 0.1], [0, 1, 0, 1]) == pytest.approx(
        3.5 / 4
    )  # one tie -> half
    assert math.isnan(stats.auroc([0.9, 0.1], [0, 0]))
    assert stats.auroc([1, 2, 3, 4], [1, 1, 0, 0], positive=1) == 0.0
    for _ in range(20):
        n = int(RNG.integers(5, 40))
        scores = np.round(RNG.random(n), 1)  # ties on purpose
        labels = RNG.integers(0, 2, n)
        if len(set(labels.tolist())) < 2:
            continue
        assert stats.auroc(scores, labels) == pytest.approx(_brute_auroc(scores, labels))
    probs = _binary(np.array([0.9, 0.2, 0.7, 0.4]))
    assert stats.metric_value("auroc", probs, [0, 1, 0, 1]) == 1.0
    assert math.isnan(stats.metric_value("auroc", np.full((4, 3), 1 / 3), [0, 1, 2, 0]))
    assert math.isnan(stats.metric_value("acc", np.zeros((0, 2)), []))
    with pytest.raises(ValueError, match="2-d"):
        stats.metric_value("acc", [0.5, 0.5], [0])
    with pytest.raises(ValueError, match="one entry per row"):
        stats.metric_value("acc", probs, [0, 1])
    with pytest.raises(ValueError, match="unknown metric"):
        stats.metric_value("f1", probs, [0, 1, 0, 1])  # type: ignore[arg-type]


def test_metric_value_matches_the_vendored_metrics() -> None:
    probs = _binary(RNG.random(50))
    labels = RNG.integers(0, 2, 50)
    y = labels.tolist()
    assert stats.metric_value("acc", probs, labels) == _metrics.accuracy(probs, y)
    assert stats.metric_value("nll", probs, labels) == _metrics.nll(probs, y)
    assert stats.metric_value("brier", probs, labels) == _metrics.brier(probs, y)
    assert stats.metric_value("ece", probs, labels) == _metrics.ece(probs, y)
    assert stats.improvement("acc", 0.8, 0.7) == pytest.approx(0.1)
    assert stats.improvement("nll", 0.8, 0.7) == pytest.approx(-0.1)


def test_weighted_metrics_equal_metrics_on_repeated_items() -> None:
    n = 30
    probs = _binary(np.round(RNG.random(n), 1))
    labels = RNG.integers(0, 2, n)
    w = RNG.integers(0, 4, size=(5, n)).astype(float)
    w[0] = 1.0
    for metric in ("acc", "nll", "brier", "auroc", "ece"):
        got = stats.weighted_metric(metric, probs, labels, w)  # type: ignore[arg-type]
        for b in range(5):
            idx = np.repeat(np.arange(n), w[b].astype(int))
            expected = stats.metric_value(metric, probs[idx], labels[idx])  # type: ignore[arg-type]
            assert got[b] == pytest.approx(expected, nan_ok=True), (metric, b)
    zero = stats.weighted_metric("acc", probs, labels, np.zeros((1, n)))
    assert math.isnan(zero[0])
    wide = stats.weighted_metric("auroc", np.full((n, 3), 1 / 3), labels, w)
    assert np.isnan(wide).all()
    with pytest.raises(ValueError, match="unknown metric"):
        stats.weighted_metric("f1", probs, labels, w)  # type: ignore[arg-type]


# -- bootstrap -------------------------------------------------------------------------------------------------


def test_bootstrap_weights_resample_clusters() -> None:
    clusters = ["a", "a", "b", "c", "c", "c", "b"]
    w = stats.bootstrap_weights(clusters, B=500, seed=0)
    assert w.shape == (500, 7)
    # items of one cluster always share a multiplicity, and every replicate draws 3 clusters
    assert np.array_equal(w[:, 0], w[:, 1]) and np.array_equal(w[:, 3], w[:, 4])
    per_cluster = np.stack([w[:, 0], w[:, 2], w[:, 3]], axis=1)
    assert np.array_equal(per_cluster.sum(axis=1), np.full(500, 3.0))
    assert np.array_equal(w, stats.bootstrap_weights(clusters, B=500, seed=0))
    assert not np.array_equal(w, stats.bootstrap_weights(clusters, B=500, seed=1))
    # item order does not change the draw: the same clusters in another order get the same counts
    w2 = stats.bootstrap_weights(["c", "b", "a", "a", "c", "c", "b"], B=500, seed=0)
    assert np.array_equal(w2[:, 2], w[:, 0]) and np.array_equal(w2[:, 1], w[:, 2])


def test_metric_ci_and_paired_bootstrap() -> None:
    n = 140
    clusters = [f"card{i % 7}" for i in range(n)]
    labels = RNG.integers(0, 2, n)
    good = _binary(np.clip(np.where(labels == 0, 0.8, 0.2) + RNG.normal(0, 0.1, n), 0.01, 0.99))
    prior = _binary(np.full(n, float(np.mean(labels == 0))))
    ci = stats.metric_ci("auroc", good, labels, clusters, B=400, seed=0)
    assert ci.point == pytest.approx(stats.auroc(good[:, 0], labels))
    assert ci.lower <= ci.point <= ci.upper and ci.lower > 0.9
    assert (
        ci.B == 400
        and ci.seed == 0
        and ci.alpha == 0.05
        and ci.n_clusters == 7
        and ci.n_valid == 400
    )
    assert ci.as_dict()["n_valid"] == 400
    paired = stats.paired_bootstrap("auroc", good, prior, labels, clusters, B=400, seed=0)
    assert paired.point == pytest.approx(ci.point - 0.5) and paired.lower > 0
    same = stats.paired_bootstrap("nll", good, good, labels, clusters, B=100)
    assert same.point == 0 and same.lower == 0 and same.upper == 0
    worse = stats.paired_bootstrap("nll", prior, good, labels, clusters, B=100)
    assert worse.point < 0 and worse.upper < 0  # lower NLL is better: the prior loses
    w = stats.bootstrap_weights(clusters, B=400, seed=0)
    again = stats.paired_bootstrap("auroc", good, prior, labels, clusters, B=400, weights=w)
    assert again == paired
    with pytest.raises(ValueError, match="weights must be"):
        stats.paired_bootstrap("auroc", good, prior, labels, clusters, B=10, weights=w)
    with pytest.raises(ValueError, match="same items"):
        stats.paired_bootstrap("auroc", good, prior[:10], labels, clusters)
    with pytest.raises(ValueError, match="clusters"):
        stats.metric_ci("acc", good, labels, clusters[:-1])


# -- sign test, rule R, non-inferiority -------------------------------------------------------------------------


def test_card_sign_test_counts_only_evaluable_cards() -> None:
    n = 70
    clusters = [f"c{i % 7}" for i in range(n)]
    labels = np.array([i % 2 for i in range(n)])
    a = _binary(np.where(labels == 0, 0.9, 0.1))
    b = _binary(np.full(n, 0.5))
    st = stats.card_sign_test("auroc", a, b, labels, clusters)
    assert st.m_c == 7 and st.wins == 7 and st.needed == 6 and st.passed and st.reason == "passed"
    assert set(st.per_card) == set(clusters) and all(
        d == pytest.approx(0.5) for d in st.per_card.values()
    )
    tie = stats.card_sign_test("acc", a, a, labels, clusters)
    assert tie.wins == 0 and not tie.passed and tie.reason == "sign_test_failed"
    few = stats.card_sign_test("acc", a, b, labels, [f"c{i % 3}" for i in range(n)])
    assert few.m_c == 3 and not few.passed and few.reason == "insufficient_clusters"
    thin = stats.card_sign_test(
        "acc", a[:68], b[:68], labels[:68], [f"c{i % 7}" for i in range(68)]
    )
    assert thin.m_c == 5  # 68 = 7 * 9 + 5: five cards have 10 items, two have 9 < 10
    one_class = np.zeros(n, dtype=int)
    one_class[:20] = (
        1  # cards c0..c6 all mixed except items >= 20 are all Yes -> per-card AUROC undefined
    )
    undefined = stats.card_sign_test("auroc", a, b, one_class, [f"c{i // 10}" for i in range(n)])
    assert undefined.m_c == 0 and undefined.reason == "insufficient_clusters"
    assert stats.card_sign_test("acc", a, b, labels, clusters, fraction=1.0).needed == 7
    assert undefined.as_dict()["m_c"] == 0


def test_rule_r_and_non_inferiority() -> None:
    n = 140
    clusters = [f"card{i % 7}" for i in range(n)]
    labels = RNG.integers(0, 2, n)
    strong = _binary(
        np.clip(np.where(labels == 0, 0.85, 0.15) + RNG.normal(0, 0.08, n), 0.01, 0.99)
    )
    prior = _binary(np.full(n, float(np.mean(labels == 0))))
    verdict = stats.rule_r("auroc", strong, prior, labels, clusters, B=400)
    assert verdict.passed and verdict.reason == "passed" and verdict.metric == "auroc"
    assert (
        verdict.delta == verdict.bootstrap.point
        and verdict.lower_bound == verdict.bootstrap.lower > 0
    )
    assert verdict.sign.passed and verdict.as_dict()["passed"] is True
    nll = stats.rule_r("nll", strong, prior, labels, clusters, B=400)
    assert nll.passed and nll.delta > 0
    null = stats.rule_r("acc", prior, prior, labels, clusters, B=100)
    assert not null.passed and null.reason == "bootstrap_lower_bound_not_positive"
    few = stats.rule_r("auroc", strong, prior, labels, [f"c{i % 3}" for i in range(n)], B=100)
    assert not few.passed and few.reason == "insufficient_clusters"
    # a gain on one card only: the bootstrap may pass but the sign test cannot
    lucky = prior.copy()
    card0 = np.array([c == "card0" for c in clusters])
    lucky[card0] = strong[card0]
    one = stats.rule_r("auroc", lucky, prior, labels, clusters, B=400)
    assert not one.passed and one.sign.wins == 1
    ni = stats.non_inferior("acc", prior, prior, labels, clusters, 0.02, B=100)
    assert (
        ni.passed and ni.lower_bound == 0 and ni.margin == 0.02 and ni.as_dict()["passed"] is True
    )
    inferior = stats.non_inferior("acc", prior, strong, labels, clusters, 0.02, B=400)
    assert not inferior.passed and inferior.delta < -0.02
    with pytest.raises(ValueError, match="margin"):
        stats.non_inferior("acc", prior, prior, labels, clusters, -0.1)


# -- incomplete beta and Clopper-Pearson ----------------------------------------------------------------------------


def _binom_tail(n: int, x: float, k: int) -> float:
    """P(Bin(n, x) >= k) = I_x(k, n - k + 1)."""
    return sum(math.comb(n, i) * x**i * (1 - x) ** (n - i) for i in range(k, n + 1))


def test_betainc_against_binomial_sums_symmetry_and_edges() -> None:
    for n, k in ((10, 3), (30, 0), (30, 29), (61, 2), (200, 100), (7, 6)):
        for x in (0.01, 0.2, 0.5, 0.77, 0.99):
            if k > 0:
                assert stats.betainc(x, k, n - k + 1) == pytest.approx(
                    _binom_tail(n, x, k), abs=1e-12
                )
            assert stats.betainc(x, k + 1, n - k) == pytest.approx(
                _binom_tail(n, x, k + 1), abs=1e-12
            )
    assert stats.betainc(0.5, 1, 1) == 0.5 and stats.betainc(0.3, 1, 1) == pytest.approx(0.3)
    assert stats.betainc(0.0, 2, 3) == 0.0 and stats.betainc(1.0, 2, 3) == 1.0
    assert stats.betainc(0.35, 2.5, 7.5) == pytest.approx(1 - stats.betainc(0.65, 7.5, 2.5))
    # non-integer parameters against a fine trapezoid of the Beta(2.5, 7.5) density
    grid = np.linspace(0.0, 0.35, 200_001)
    pdf = (
        grid**1.5
        * (1 - grid) ** 6.5
        / math.exp(math.lgamma(2.5) + math.lgamma(7.5) - math.lgamma(10.0))
    )
    assert stats.betainc(0.35, 2.5, 7.5) == pytest.approx(float(np.trapezoid(pdf, grid)), abs=1e-6)
    with pytest.raises(ValueError):
        stats.betainc(0.5, 0, 1)
    for q in (0.001, 0.05, 0.5, 0.95, 0.999):
        x = stats.beta_ppf(q, 3, 12)
        assert stats.betainc(x, 3, 12) == pytest.approx(q, abs=1e-10)
    assert stats.beta_ppf(0.0, 2, 2) == 0.0 and stats.beta_ppf(1.0, 2, 2) == 1.0
    with pytest.raises(ValueError):
        stats.beta_ppf(1.5, 2, 2)


def test_clopper_pearson_upper_known_values() -> None:
    # k = 0: 1 - alpha ** (1/n); k = n-1: (1 - alpha) ** (1/n); k = n: 1
    for n in (1, 10, 30, 59, 200):
        assert stats.clopper_pearson_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-10)
        assert stats.clopper_pearson_upper(n - 1, n) == pytest.approx(0.95 ** (1 / n), abs=1e-10)
        assert stats.clopper_pearson_upper(n, n) == 1.0
    assert stats.clopper_pearson_upper(0, 30) == pytest.approx(0.0950, abs=5e-5)
    assert stats.clopper_pearson_upper(0, 59) < 0.05 < stats.clopper_pearson_upper(0, 58)
    # the bound solves P(X <= k | p) = alpha exactly
    for k, n in ((1, 61), (2, 50), (5, 100), (17, 40)):
        p = stats.clopper_pearson_upper(k, n)
        assert 1 - _binom_tail(n, p, k + 1) == pytest.approx(0.05, abs=1e-9)
        assert (
            stats.clopper_pearson_upper(k, n, alpha=0.10)
            < p
            < stats.clopper_pearson_upper(k, n, alpha=0.01)
        )
    # the "rule of three": zero events in n trials bound close to 3 / n for large n
    assert stats.clopper_pearson_upper(0, 1000) == pytest.approx(3 / 1000, rel=0.01)
    bounds = [stats.clopper_pearson_upper(k, 40) for k in range(41)]
    assert bounds == sorted(bounds)
    for bad in ((-1, 10), (11, 10), (0, 0)):
        with pytest.raises(ValueError):
            stats.clopper_pearson_upper(*bad)
    with pytest.raises(ValueError):
        stats.clopper_pearson_upper(1, 10, alpha=1.0)


def _brute_threshold(
    stat: np.ndarray, correct: np.ndarray, risk: float, min_n: int = 30
) -> float | None:
    best = None
    for t in sorted(set(stat.tolist())):
        keep = stat >= t
        n = int(keep.sum())
        if n >= min_n and stats.clopper_pearson_upper(int((~correct[keep]).sum()), n) <= risk:
            best = t
            break
    return best


def test_threshold_cp_rule() -> None:
    stat = np.linspace(1.0, 0.01, 100)
    correct = np.zeros(100, dtype=bool)
    correct[:60] = True  # the 60 most confident items are right, the rest wrong
    # 0 errors in 59 -> CP 0.0495 <= 0.05; 60 -> ok; 61 has an error -> 0.0746 > 0.05
    assert stats.threshold_cp(stat, correct, 0.05) == pytest.approx(stat[59])
    assert stats.threshold_cp(stat, correct, 0.10) == pytest.approx(
        _brute_threshold(stat, correct, 0.10)
    )
    assert stats.threshold_cp(stat, correct, 0.05) == pytest.approx(
        _brute_threshold(stat, correct, 0.05)
    )
    assert stats.threshold_cp(stat[:50], correct[:50], 0.05, min_n=59) is None  # never 59 items
    assert stats.threshold_cp(stat, ~correct, 0.05) is None
    assert stats.threshold_cp(stat, correct, 0.05, min_n=30) == pytest.approx(stat[59])
    assert stats.threshold_cp(stat, correct, 0.05, min_n=61) is None
    # ties: {stat >= t} always takes the whole tie group
    tied = np.repeat([0.9, 0.6, 0.3], [40, 30, 30])
    ok = np.concatenate([np.ones(40, bool), np.ones(29, bool), np.zeros(31, bool)])
    assert stats.threshold_cp(tied, ok, 0.10) == pytest.approx(0.6)  # 70 items, 1 error -> 0.066
    assert stats.threshold_cp(tied, ok, 0.05) == pytest.approx(_brute_threshold(tied, ok, 0.05))
    for _ in range(20):
        s = np.round(RNG.random(80), 2)
        c = RNG.random(80) < 0.9
        for risk in (0.05, 0.10):
            assert stats.threshold_cp(s, c, risk) == pytest.approx(
                _brute_threshold(s, c, risk), nan_ok=True
            )
    with pytest.raises(ValueError, match="risk"):
        stats.threshold_cp(stat, correct, 1.5)
    with pytest.raises(ValueError, match="parallel"):
        stats.threshold_cp(stat, correct[:10], 0.05)


def test_policy_stat_and_thresholds_cp() -> None:
    probs = _binary(np.array([0.9, 0.8, 0.3, 0.6]))
    labels = np.array([0, 1, 1, 0])
    stat, correct = stats.policy_stat(probs, labels, rank_fit=True)
    np.testing.assert_allclose(stat, [0.9, 0.8, 0.3, 0.6])
    assert correct.tolist() == [True, False, False, True]  # p_fit thresholds a Yes
    stat2, correct2 = stats.policy_stat(probs, labels, rank_fit=False)
    np.testing.assert_allclose(stat2, [0.9, 0.8, 0.7, 0.6])
    assert correct2.tolist() == [True, False, True, True]
    out = stats.thresholds_cp(probs, labels, rank_fit=True)
    assert out == {"0.05": None, "0.10": None}
    big = _binary(np.linspace(0.99, 0.5, 80))
    y = np.zeros(80, dtype=int)
    out = stats.thresholds_cp(big, y, rank_fit=True, min_n=30)
    assert out["0.05"] == pytest.approx(stats.threshold_cp(big[:, 0], np.ones(80, bool), 0.05))
    assert out["0.05"] is not None and out["0.10"] is not None and out["0.10"] <= out["0.05"]
