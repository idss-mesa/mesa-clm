"""The probe tier's linear fitters (``mesa_clm.learn.linear``; plan §5.3; DESIGN D18, D20;
the M4 analysis plan "probe"), on synthetic data only: weighted L2 logistic regression (binary
and multinomial) reaches the optimum scikit-learn reaches for the same objective, shrinkage LDA
equals scikit-learn's ``lsqr`` LDA and the direct full-space formula when d > n, ridge equals
its normal equations; the SVD reduction is exact; weights are sample weights (an integer weight
equals repeated rows, other weights give another fit); standardization uses the training set's
weighted moments with the std floor; a constant column gets a zero coefficient; a fit that
reaches the iteration cap is used as it is and flagged; refusals; JSON round trip and
determinism; the grids and constants are the declared ones."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from mesa_clm.learn import linear as L
from mesa_clm.learn.calibrate import softmax
from mesa_clm.learn.linear import (
    GRIDS,
    STD_FLOOR,
    LinearError,
    LinearModel,
    Standardizer,
    fit_design,
    fit_linear,
    log_softmax,
    prepare,
)


def _binary(n: int, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    p = 1.0 / (1.0 + np.exp(-(x @ beta + 0.3)))
    y = np.where(rng.random(n) < p, 0, 1)  # Yes = 0 where the planted score is high
    w = rng.choice([0.5, 0.6, 0.8], size=n)
    return x, y, w


def _multi(n: int, d: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    b = rng.normal(size=(k, d))
    y = np.argmax(x @ b.T + rng.gumbel(size=(n, k)), axis=1)
    w = rng.choice([0.5, 0.6, 0.8], size=n)
    return x, y, w


# -- logistic regression -------------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [h["lambda"] for h in GRIDS["logreg"]])
def test_binary_logreg_is_the_sklearn_optimum(lam: float) -> None:
    """Weighted mean NLL + λ/2‖w‖² is sklearn's C·Σ wᵢ ℓᵢ + ½‖w‖² with C = 1/(λ·Σw) on the same
    standardized features; sklearn's class-1 vector is the negative of our Yes (class 0) row."""
    x, y, w = _binary(300, 5)
    m = fit_linear("logreg", x, y, w, k=2, hyper={"lambda": lam})
    assert m.kind == "logreg" and m.k == 2 and m.converged and 0 < m.iterations <= 20
    assert m.coef[1] == [0.0] * 5 and m.intercept[1] == 0.0  # the binary form: row 1 is zero
    xs = m.standardizer.transform(x)
    sk = LogisticRegression(C=1.0 / (lam * w.sum()), max_iter=5000, tol=1e-12)
    sk.fit(xs, y, sample_weight=w)
    np.testing.assert_allclose(np.array(m.coef[0]), -sk.coef_[0], atol=1e-6)
    assert m.intercept[0] == pytest.approx(-float(sk.intercept_[0]), abs=1e-6)
    np.testing.assert_allclose(m.probs(x)[:, 1], sk.predict_proba(xs)[:, 1], atol=1e-7)
    np.testing.assert_allclose(m.probs(x), softmax(m.scores(x)))
    assert m.n == 300 and m.weight_sum == pytest.approx(w.sum())


@pytest.mark.parametrize("lam", [0.1, 10.0])
def test_multinomial_logreg_is_the_symmetric_sklearn_optimum(lam: float) -> None:
    x, y, w = _multi(278, 20, 4)
    m = fit_linear("logreg", x, y, w, k=4, hyper={"lambda": lam})
    assert m.converged and len(m.coef) == 4 and m.intercept[3] == 0.0  # the gauge b[K-1] = 0
    xs = m.standardizer.transform(x)
    sk = LogisticRegression(C=1.0 / (lam * w.sum()), max_iter=20000, tol=1e-12)
    sk.fit(xs, y, sample_weight=w)
    np.testing.assert_allclose(np.array(m.coef), sk.coef_, atol=1e-6)
    np.testing.assert_allclose(m.probs(x), sk.predict_proba(xs), atol=1e-7)
    np.testing.assert_allclose(np.array(m.coef).sum(axis=0), 0.0, atol=1e-8)  # symmetric form


def test_the_reduction_is_exact_when_d_exceeds_n() -> None:
    """A 4097-d fit on 300 rows is solved in the ≤300-d row span; its probabilities are the
    full-space optimum (sklearn on the standardized matrix)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 4097))
    y = rng.integers(0, 2, 300)
    design = prepare(x)
    assert design.r == 299 and design.d == 4097 and design.v.shape == (4097, 299)
    m = fit_design(design, "logreg", y, k=2, hyper={"lambda": 1e-2})
    assert m.converged and len(m.coef[0]) == 4097
    xs = m.standardizer.transform(x)
    sk = LogisticRegression(C=1.0 / (1e-2 * 300), max_iter=10000, tol=1e-12).fit(xs, y)
    np.testing.assert_allclose(m.probs(x)[:, 1], sk.predict_proba(xs)[:, 1], atol=1e-6)
    # Unseen rows have components outside the span; the model still scores them finitely.
    assert np.all(np.isfinite(m.scores(rng.normal(size=(5, 4097)))))


def test_a_fit_at_the_iteration_cap_is_used_and_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y, w = _binary(120, 4, seed=5)
    monkeypatch.setattr(L, "LOGREG_MAX_ITER", 1)
    m = fit_linear("logreg", x, y, w, k=2, hyper={"lambda": 0.1})
    assert m.iterations == 1 and not m.converged
    assert np.all(np.isfinite(m.probs(x)))
    monkeypatch.setattr(L, "LOGREG_MAX_ITER", 100)
    full = fit_linear("logreg", x, y, w, k=2, hyper={"lambda": 0.1})
    assert full.converged and full.coef != m.coef


# -- LDA and ridge ------------------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [h["gamma"] for h in GRIDS["lda"]])
def test_lda_equals_sklearn_shrinkage_lda(gamma: float) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 6)) + np.repeat(rng.normal(size=(3, 6)), [70, 70, 60], axis=0)
    y = np.repeat([0, 1, 2], [70, 70, 60])
    xs = Standardizer.fit(x).transform(x)  # sklearn does not standardize; feed it ours
    m = fit_linear("lda", xs, y, None, k=3, hyper={"gamma": gamma})
    assert m.kind == "lda" and m.converged and m.iterations == 0
    sk = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=gamma).fit(xs, y)
    np.testing.assert_allclose(m.probs(xs), sk.predict_proba(xs), atol=1e-9)


def test_weighted_lda_equals_the_full_space_formula_when_d_exceeds_n() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 200))
    y = rng.integers(0, 3, 60)
    w = rng.choice([0.5, 1.0], 60)
    m = fit_linear("lda", x, y, w, k=3, hyper={"gamma": 0.5})
    xs = m.standardizer.transform(x)
    total = w.sum()
    mu = np.stack([(w[y == c] @ xs[y == c]) / w[y == c].sum() for c in range(3)])
    prior = np.array([w[y == c].sum() / total for c in range(3)])
    resid = xs - mu[y]
    cov = (resid.T @ (w[:, None] * resid)) / total
    nu = np.trace(cov) / 200
    shrunk = 0.5 * cov + 0.5 * nu * np.eye(200)
    coef = np.linalg.solve(shrunk, mu.T).T
    intercept = -0.5 * np.einsum("kr,kr->k", mu, coef) + np.log(prior)
    xt = rng.normal(size=(10, 200))
    want = softmax(m.standardizer.transform(xt) @ coef.T + intercept)
    np.testing.assert_allclose(m.probs(xt), want, atol=1e-12)
    np.testing.assert_allclose(np.array(m.coef), coef, atol=1e-9)


def test_lda_refuses_a_class_without_training_weight() -> None:
    x, y, w = _multi(80, 5, 4, seed=4)
    y[y == 3] = 2
    with pytest.raises(LinearError, match="class 3 has no training weight"):
        fit_linear("lda", x, y, w, k=4, hyper={"gamma": 0.5})
    w2 = w.copy()
    w2[y == 1] = 0.0  # present but weightless: the same refusal
    with pytest.raises(LinearError, match="class 1 has no training weight"):
        fit_linear("lda", x, y, w2, k=4, hyper={"gamma": 0.5})
    assert fit_linear("logreg", x, y, w, k=4, hyper={"lambda": 1.0}).k == 4  # fine for logreg
    assert fit_linear("ridge", x, y, w, k=4, hyper={"lambda": 1.0}).k == 4


@pytest.mark.parametrize("lam", [h["lambda"] for h in GRIDS["ridge"]])
def test_ridge_solves_its_normal_equations(lam: float) -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(100, 7))
    y = rng.integers(0, 3, 100)
    w = rng.choice([0.5, 1.0], 100)
    m = fit_linear("ridge", x, y, w, k=3, hyper={"lambda": lam})
    xs = m.standardizer.transform(x)
    total = w.sum()
    onehot = np.eye(3)[y]
    b = (w @ onehot) / total
    beta = np.linalg.solve(
        xs.T @ (w[:, None] * xs) / total + lam * np.eye(7),
        xs.T @ (w[:, None] * (onehot - b)) / total,
    ).T
    np.testing.assert_allclose(np.array(m.coef), beta, atol=1e-12)
    np.testing.assert_allclose(np.array(m.intercept), b, atol=1e-15)
    np.testing.assert_allclose(m.scores(x), xs @ beta.T + b, atol=1e-12)
    np.testing.assert_allclose(m.probs(x), softmax(m.scores(x)))
    two = fit_linear("ridge", x, (y == 0).astype(int), w, k=2, hyper={"lambda": lam})
    s = two.scores(x)
    np.testing.assert_allclose(two.probs(x)[:, 0], 1 / (1 + np.exp(-(s[:, 0] - s[:, 1]))))


# -- weights and standardization ---------------------------------------------------------------------


FITS = [("logreg", {"lambda": 0.1}), ("lda", {"gamma": 0.5}), ("ridge", {"lambda": 1.0})]


@pytest.mark.parametrize(("kind", "hyper"), FITS)
def test_an_integer_weight_equals_repeated_rows(kind: L.Kind, hyper: dict[str, float]) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, 50)
    dup = fit_linear(
        kind, np.vstack([x, x[:10]]), np.concatenate([y, y[:10]]), None, k=2, hyper=hyper
    )
    weighted = fit_linear(
        kind, x, y, np.concatenate([np.full(10, 2.0), np.ones(40)]), k=2, hyper=hyper
    )
    np.testing.assert_allclose(np.array(dup.coef), np.array(weighted.coef), atol=1e-10)
    np.testing.assert_allclose(dup.intercept, weighted.intercept, atol=1e-10)
    assert dup.standardizer.mean == pytest.approx(weighted.standardizer.mean)
    # The scale of the weights does not matter, their ratios do.
    scaled = fit_linear(
        kind, x, y, 3.0 * np.concatenate([np.full(10, 2.0), np.ones(40)]), k=2, hyper=hyper
    )
    np.testing.assert_allclose(np.array(scaled.coef), np.array(weighted.coef), atol=1e-10)
    other = fit_linear(kind, x, y, np.where(y == 0, 3.0, 0.5), k=2, hyper=hyper)
    assert other.coef != weighted.coef and other.intercept != weighted.intercept


def test_standardization_uses_the_weighted_training_moments_and_the_floor() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 3)) * [1.0, 5.0, 0.0] + [0.0, 2.0, 7.0]
    w = rng.choice([0.5, 1.0], 40)
    std = Standardizer.fit(x, w)
    assert std.mean == pytest.approx(((w @ x) / w.sum()).tolist())
    var = (w @ (x - np.asarray(std.mean)) ** 2) / w.sum()
    assert std.std[:2] == pytest.approx(np.sqrt(var[:2]).tolist())
    assert std.std[2] == STD_FLOOR  # a constant column
    xs = std.transform(x)
    np.testing.assert_allclose((w @ xs) / w.sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(xs[:, 2], 0.0)
    y = rng.integers(0, 2, 40)
    for kind, hyper in (
        ("logreg", {"lambda": 0.1}),
        ("lda", {"gamma": 0.5}),
        ("ridge", {"lambda": 1.0}),
    ):
        m = fit_linear(kind, x, y, w, k=2, hyper=hyper)  # type: ignore[arg-type]
        assert all(row[2] == 0.0 for row in m.coef), kind  # spans nothing: zero coefficient
    with pytest.raises(LinearError, match="expected 3 features"):
        std.transform(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="floored"):
        Standardizer(mean=[0.0], std=[0.0])


# -- refusals, JSON and determinism -----------------------------------------------------------------


def test_refusals() -> None:
    x, y, w = _binary(40, 3)
    with pytest.raises(LinearError, match="unknown fitter"):
        fit_linear("svm", x, y, w, k=2, hyper={"lambda": 1.0})  # type: ignore[arg-type]
    with pytest.raises(LinearError, match="takes the hyperparameter 'lambda'"):
        fit_linear("logreg", x, y, w, k=2, hyper={"gamma": 1.0})
    with pytest.raises(LinearError, match=r"gamma must be in \(0, 1\]"):
        fit_linear("lda", x, y, w, k=2, hyper={"gamma": 1.5})
    with pytest.raises(LinearError, match="lambda must be in"):
        fit_linear("ridge", x, y, w, k=2, hyper={"lambda": 0.0})
    with pytest.raises(LinearError, match="index the 2 options"):
        fit_linear("logreg", x, y + 1, w, k=2, hyper={"lambda": 1.0})
    with pytest.raises(LinearError, match="one entry per row"):
        fit_linear("logreg", x, y[:-1], w, k=2, hyper={"lambda": 1.0})
    with pytest.raises(LinearError, match="non-negative"):
        fit_linear("logreg", x, y, -w, k=2, hyper={"lambda": 1.0})
    with pytest.raises(LinearError, match="sum to zero"):
        fit_linear("logreg", x, y, 0 * w, k=2, hyper={"lambda": 1.0})
    with pytest.raises(LinearError, match=r"must be \[n, d\]"):
        fit_linear("logreg", x[:, 0], y, w, k=2, hyper={"lambda": 1.0})
    with pytest.raises(LinearError, match="non-finite"):
        fit_linear("logreg", np.where(x > 2, np.nan, x), y, w, k=2, hyper={"lambda": 1.0})
    with pytest.raises(LinearError, match="no rows"):
        prepare(np.zeros((0, 3)))
    with pytest.raises(LinearError, match="k must be at least 2"):
        fit_design(prepare(x), "logreg", y, k=1, hyper={"lambda": 1.0})


def test_json_round_trip_and_determinism() -> None:
    x, y, w = _multi(90, 6, 3, seed=9)
    for kind, hyper in (
        ("logreg", {"lambda": 0.1}),
        ("lda", {"gamma": 0.5}),
        ("ridge", {"lambda": 1.0}),
    ):
        m = fit_linear(kind, x, y, w, k=3, hyper=hyper)  # type: ignore[arg-type]
        again = fit_linear(kind, x, y, w, k=3, hyper=hyper)  # type: ignore[arg-type]
        a = json.dumps(m.model_dump(mode="json"), sort_keys=True)
        b = json.dumps(again.model_dump(mode="json"), sort_keys=True)
        assert a == b
        back = LinearModel.model_validate(json.loads(a))
        assert back == m
        np.testing.assert_array_equal(back.probs(x), m.probs(x))
    with pytest.raises(ValueError, match="rows"):
        LinearModel.model_validate({**m.model_dump(), "intercept": [0.0]})
    with pytest.raises(ValueError, match="entries"):
        LinearModel.model_validate({**m.model_dump(), "coef": [[0.0]] * 3})
    with pytest.raises(ValueError):
        LinearModel.model_validate({**m.model_dump(), "extra": 1})


def test_grids_and_constants_are_the_declared_ones() -> None:
    assert L.KINDS == ("logreg", "lda", "ridge")
    assert [h["lambda"] for h in GRIDS["logreg"]] == [1e-2, 1e-1, 1.0, 10.0]
    assert [h["gamma"] for h in GRIDS["lda"]] == [0.1, 0.5, 0.9]
    assert [h["lambda"] for h in GRIDS["ridge"]] == [1e-1, 1.0, 10.0]
    assert L.LOGREG_MAX_ITER == 100 and L.LOGREG_GTOL == 1e-8 and STD_FLOOR == 1e-8
    assert L.LOGREG_ARMIJO == 1e-4 and L.LOGREG_MAX_HALVINGS == 60 and L.LOGREG_QUADRATIC == 1e-12
    c = L.jsonable_constants()
    assert c["logreg"]["grid"] == [1e-2, 1e-1, 1.0, 10.0] and c["lda"]["grid"] == [0.1, 0.5, 0.9]
    assert c["standardization"]["std_floor"] == STD_FLOOR and c["reduction"]["eps"] == L.RANK_EPS
    assert L.grid_of("ridge") == GRIDS["ridge"]
    with pytest.raises(LinearError):
        L.grid_of("nope")  # type: ignore[arg-type]
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.exp(log_softmax(z)), softmax(z))
