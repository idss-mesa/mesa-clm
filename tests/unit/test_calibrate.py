"""The calibrated tier's fitters (``mesa_clm.learn.calibrate``; plan §5.3; DESIGN D6, D7, D20;
``design/m2-analysis-plan.md`` §6.4–§6.5), on synthetic data only: weighted Platt
recovers planted ``(a, b)``, weights act as sample weights (an integer weight equals repeated
rows), separable and one-class data stay finite, ``a < 0`` is flagged ``inverted``, both target
modes reach the optimum of their own objective (checked against scipy), temperature scaling
recovers a planted ``T`` and stops at a bound when the logits carry nothing; every fit is
deterministic and round-trips through JSON."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mesa_clm.learn import calibrate as cal
from mesa_clm.learn.calibrate import (
    PLATT_RIDGE,
    TEMPERATURE_BOUNDS,
    CalibrationError,
    PlattCalibrator,
    TemperatureCalibrator,
    apply_calibrator,
    fit_calibrator,
    fit_platt,
    fit_temperature,
    load_calibrator,
    logit_difference,
    sigmoid,
    softmax,
    zero_shot_probs,
)


def _planted_platt(n: int, a: float, b: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 2.0, n)
    y = rng.random(n) < sigmoid(a * s + b)
    return s, y


def _planted_temperature(n: int, k: int, t: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 4.0, (n, k))
    p = softmax(logits, t)
    u = rng.random(n)
    labels = (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    return logits, np.minimum(labels, k - 1)


# -- primitives -------------------------------------------------------------------------------------


def test_sigmoid_softmax_and_logit_difference_are_stable() -> None:
    z = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p)) and p[2] == 0.5 and p[0] == 0.0 and p[-1] == 1.0
    np.testing.assert_allclose(sigmoid(z[1:4]), 1 / (1 + np.exp(-z[1:4])), rtol=1e-15)
    sm = softmax(np.array([[1000.0, 999.0], [0.0, 0.0]]))
    np.testing.assert_allclose(sm, [[1 / (1 + np.exp(-1)), 1 - 1 / (1 + np.exp(-1))], [0.5, 0.5]])
    np.testing.assert_allclose(
        softmax(np.array([[2.0, 0.0]]), 2.0), softmax(np.array([[1.0, 0.0]]))
    )
    np.testing.assert_array_equal(logit_difference(np.array([[3.0, 1.0], [0.0, 2.0]])), [2.0, -2.0])
    with pytest.raises(CalibrationError):
        logit_difference(np.zeros((2, 3)))
    with pytest.raises(CalibrationError):
        softmax(np.zeros(3))
    with pytest.raises(CalibrationError):
        softmax(np.zeros((1, 2)), 0.0)


def test_zero_shot_is_sigma_of_s_c_and_the_option_softmax() -> None:
    s = np.array([-2.0, 0.0, 3.0])
    zs = zero_shot_probs("rank_fit", s)
    np.testing.assert_allclose(zs[:, 0], sigmoid(s))
    np.testing.assert_allclose(zs.sum(axis=1), 1.0)
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(zero_shot_probs("choice", logits), softmax(logits))


# -- Platt ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [(1.7, -0.4), (0.3, 1.1), (-0.9, 0.2)])
def test_platt_recovers_planted_parameters(a: float, b: float) -> None:
    s, y = _planted_platt(20000, a, b, seed=3)
    fit = fit_platt(s, y)
    assert fit.converged and fit.iterations <= 20
    assert fit.a == pytest.approx(a, abs=0.06) and fit.b == pytest.approx(b, abs=0.06)
    assert fit.inverted is (a < 0)
    assert fit.n == 20000 and fit.weight_sum == 20000.0 and fit.ridge == PLATT_RIDGE
    assert fit.kind == "platt" and fit.feature == "s_c" and fit.targets == "hard"
    np.testing.assert_allclose(fit.probs(s)[:, 0], sigmoid(fit.a * s + fit.b))


def test_platt_weights_are_sample_weights() -> None:
    """D20: an integer weight is the same fit as repeating the row; scaling every weight by a
    constant changes nothing (the objective is the weighted mean)."""
    s, y = _planted_platt(400, 1.2, -0.3, seed=5)
    w = np.random.default_rng(1).integers(1, 4, size=len(s)).astype(float)
    weighted = fit_platt(s, y, w)
    repeated = fit_platt(np.repeat(s, w.astype(int)), np.repeat(y, w.astype(int)))
    assert weighted.a == pytest.approx(repeated.a, abs=1e-9)
    assert weighted.b == pytest.approx(repeated.b, abs=1e-9)
    scaled = fit_platt(s, y, w * 0.37)
    assert scaled.a == pytest.approx(weighted.a, abs=1e-9)
    assert weighted.weight_sum == float(w.sum())
    # the label weights of the silver sources move the fit (D20: weights are not decoration)
    silver = np.where(y, 0.8, 0.5)
    assert fit_platt(s, y, silver).b != pytest.approx(fit_platt(s, y).b, abs=1e-3)


@pytest.mark.parametrize("targets", ["hard", "smoothed"])
def test_platt_reaches_the_optimum_of_its_objective(targets: str) -> None:
    """Both target modes against scipy's BFGS on the same weighted objective."""
    s, y = _planted_platt(300, 1.3, -0.2, seed=7)
    w = np.random.default_rng(2).choice([0.5, 0.6, 0.8], size=len(s))
    ridge = PLATT_RIDGE if targets == "hard" else 0.0
    fit = fit_platt(s, y, w, targets=targets, ridge=ridge)  # type: ignore[arg-type]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    t = (
        y.astype(float)
        if targets == "hard"
        else np.where(y, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    )

    def objective(theta: np.ndarray) -> float:
        z = theta[0] * s + theta[1]
        per = t * np.logaddexp(0, -z) + (1 - t) * np.logaddexp(0, z)
        return float(np.dot(w, per) / w.sum() + 0.5 * ridge * theta @ theta)

    optimize = pytest.importorskip("scipy.optimize")  # scikit-learn's dependency (bench extra)
    ref = optimize.minimize(objective, np.zeros(2), method="BFGS", options={"gtol": 1e-12}).x
    assert fit.a == pytest.approx(ref[0], abs=1e-6) and fit.b == pytest.approx(ref[1], abs=1e-6)
    assert fit.targets == targets and fit.converged


def test_platt_stays_finite_on_separable_and_one_class_data() -> None:
    s = np.array([-2.0, -1.0, 1.0, 2.0])
    sep = fit_platt(s, s > 0)
    assert sep.converged and 1.0 < sep.a < 1e4 and abs(sep.b) < 1e-6
    one = fit_platt(s, np.ones(4, dtype=bool))
    assert one.converged and np.isfinite(one.b) and one.b > 5
    flat = fit_platt(np.zeros(4), s > 0)
    assert flat.a == 0.0 and flat.b == 0.0 and flat.converged
    smoothed = fit_platt(s, s > 0, targets="smoothed", ridge=0.0)
    assert smoothed.converged and 0 < smoothed.a < 10  # Platt's targets keep separable data finite


def test_platt_flags_inverted_scores() -> None:
    s, y = _planted_platt(5000, -1.5, 0.0, seed=11)
    fit = fit_platt(s, y)
    assert fit.a < 0 and fit.inverted
    assert fit_platt(-s, y).inverted is False


def test_platt_refuses_bad_input() -> None:
    s = np.array([0.0, 1.0, 2.0])
    y = np.array([True, False, True])
    with pytest.raises(CalibrationError, match="one entry per score"):
        fit_platt(s, y[:2])
    with pytest.raises(CalibrationError, match="non-negative"):
        fit_platt(s, y, [1.0, -1.0, 1.0])
    with pytest.raises(CalibrationError, match="sum to zero"):
        fit_platt(s, y, [0.0, 0.0, 0.0])
    with pytest.raises(CalibrationError, match="one entry per item"):
        fit_platt(s, y, [1.0, 1.0])
    with pytest.raises(CalibrationError, match="non-finite"):
        fit_platt([0.0, np.nan, 1.0], y)
    with pytest.raises(CalibrationError, match="non-empty"):
        fit_platt([], [])
    with pytest.raises(CalibrationError, match="ridge"):
        fit_platt(s, y, ridge=-1.0)


# -- temperature ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "t"), [(8, 3.0), (4, 0.5), (4, 1.0)])
def test_temperature_recovers_a_planted_temperature(k: int, t: float) -> None:
    logits, labels = _planted_temperature(20000, k, t, seed=4)
    fit = fit_temperature(logits, labels)
    assert fit.temperature == pytest.approx(t, rel=0.05)
    assert not fit.at_bound and fit.iterations > 0
    assert (fit.lower, fit.upper) == TEMPERATURE_BOUNDS
    np.testing.assert_allclose(fit.probs(logits), softmax(logits, fit.temperature))
    # the fitted T minimises the NLL: nudging it either way never helps
    nll = fit.nll
    for other in (fit.temperature * 0.98, fit.temperature * 1.02):
        p = softmax(logits, other)[np.arange(len(labels)), labels]
        assert -np.mean(np.log(p)) >= nll - 1e-12


def test_temperature_weights_are_sample_weights() -> None:
    logits, labels = _planted_temperature(500, 4, 2.0, seed=8)
    w = np.random.default_rng(3).integers(1, 4, size=len(labels))
    weighted = fit_temperature(logits, labels, w.astype(float))
    repeated = fit_temperature(np.repeat(logits, w, axis=0), np.repeat(labels, w))
    assert weighted.temperature == pytest.approx(repeated.temperature, rel=1e-9)


def test_temperature_bounds_and_degenerate_logits() -> None:
    nothing = fit_temperature(np.zeros((6, 3)), np.array([0, 1, 2, 0, 1, 2]))
    assert nothing.at_bound and nothing.temperature == pytest.approx(TEMPERATURE_BOUNDS[1])
    # perfectly separating logits push T to its lower bound
    sharp = fit_temperature(np.array([[5.0, 0.0], [0.0, 5.0]]), np.array([0, 1]))
    assert sharp.at_bound and sharp.temperature == pytest.approx(TEMPERATURE_BOUNDS[0])
    with pytest.raises(CalibrationError, match="index the 3 options"):
        fit_temperature(np.zeros((2, 3)), np.array([0, 3]))
    with pytest.raises(CalibrationError, match="bounds"):
        fit_temperature(np.zeros((2, 3)), np.array([0, 1]), bounds=(2.0, 1.0))
    with pytest.raises(CalibrationError, match="K>=2"):
        fit_temperature(np.zeros((2, 1)), np.array([0, 0]))
    with pytest.raises(CalibrationError, match="non-finite"):
        fit_temperature(np.array([[np.inf, 0.0]]), np.array([0]))


def test_a_compressed_logit_scale_is_fitted_not_clamped() -> None:
    """Logits at CLM's scale 100 whose options differ by about 1e-4 in cosine (a collapsed
    zero-shot head) need T far below the old bound 0.01: the bounds [1e-4, 1e4] fit it."""
    assert TEMPERATURE_BOUNDS == (1e-4, 1e4)
    rng = np.random.default_rng(11)
    logits = rng.normal(0.0, 0.004, (4000, 4))  # 100 x cosine differences of about 4e-5
    p = softmax(logits, 0.002)
    labels = np.minimum((rng.random(4000)[:, None] > np.cumsum(p, axis=1)).sum(axis=1), 3)
    fit = fit_temperature(logits, labels)
    assert not fit.at_bound and fit.temperature == pytest.approx(0.002, rel=0.1)
    clamped = fit_temperature(logits, labels, bounds=(0.01, 100.0))
    assert clamped.at_bound and clamped.nll > fit.nll


# -- dispatch, determinism, JSON -------------------------------------------------------------------------


def test_fit_calibrator_dispatches_on_shape_and_k() -> None:
    s, y = _planted_platt(400, 1.0, 0.0, seed=9)
    labels = np.where(y, 0, 1)
    rank = fit_calibrator("rank_fit", s, labels)
    assert isinstance(rank, PlattCalibrator) and rank.feature == "s_c"
    assert rank.targets == cal.PLATT_TARGETS == "hard"  # the analysis plan's item 1.4
    assert rank == fit_platt(s, y, feature="s_c")
    two = np.stack([s, np.zeros_like(s)], axis=1) + 3.0  # logits whose difference is s
    pair = fit_calibrator("choice", two, labels)
    assert isinstance(pair, PlattCalibrator) and pair.feature == "logit_difference"
    assert pair.a == pytest.approx(rank.a, abs=1e-12) and pair.b == pytest.approx(rank.b, abs=1e-12)
    np.testing.assert_allclose(apply_calibrator(pair, two), apply_calibrator(rank, s))
    logits, k_labels = _planted_temperature(300, 4, 2.0)
    temp = fit_calibrator("choice", logits, k_labels)
    assert isinstance(temp, TemperatureCalibrator)
    with pytest.raises(CalibrationError, match="logits"):
        fit_calibrator("choice", s, labels)


def test_fits_are_deterministic_and_round_trip_through_json() -> None:
    s, y = _planted_platt(1000, 0.8, 0.3, seed=12)
    w = np.where(y, 0.8, 0.5)
    first, second = fit_platt(s, y, w), fit_platt(s, y, w)
    assert first == second  # bit for bit: no random start
    logits, labels = _planted_temperature(1000, 8, 3.0, seed=12)
    t1, t2 = fit_temperature(logits, labels, w), fit_temperature(logits, labels, w)
    assert t1 == t2
    for fitted in (first, t1):
        text = json.dumps(fitted.model_dump(mode="json"), allow_nan=False)
        back = load_calibrator(json.loads(text))
        assert back == fitted and type(back) is type(fitted)
    with pytest.raises(ValueError):
        load_calibrator({"kind": "isotonic"})
    assert cal.TemperatureCalibrator.model_fields["kind"].default == "temperature"
