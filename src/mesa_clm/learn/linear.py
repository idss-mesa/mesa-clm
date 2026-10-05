"""The probe tier's linear fitters: weighted L2 logistic regression, weighted shrinkage LDA and
weighted ridge on one-hot targets, in numpy float64 (plan §5.3 "probe", §5.4; DESIGN D18, D20,
D27; the M4 analysis plan ``design/m4-analysis-plan.md`` §2 (P.2)).

Every fitter takes a feature matrix ``X [n, d]``, option-index labels ``y [n]`` (Yes = 0 on a
two-class task) and **sample weights** ``w [n]`` (D20: the label weights; an integer weight
equals that many copies of the row), standardizes the features with the training set's weighted
mean and weighted standard deviation (std floored at :data:`STD_FLOOR`), and returns a frozen
:class:`LinearModel` whose ``scores(X)`` are ``[n, K]`` logits (``X_std · coefᵀ + intercept``)
and whose ``probs(X)`` is their row-wise softmax: the uncalibrated probe probabilities the inner
selection compares and the OOF calibrator (``learn.probe``) then rescales. The data term of
every objective is a **weighted mean** (``Σ wᵢ ℓᵢ / Σ wᵢ``, the convention of
``calibrate.fit_platt``), so a fit does not depend on the scale of the weights, only on their
ratios; the penalties act on the coefficients of the standardized features and never on an
intercept.

**Reduction.** Let ``X_std = U S Vᵀ`` be the thin SVD of the standardized training matrix and
``r`` its numerical rank (singular values above ``s_max · max(n, d) · ε``, numpy's
``matrix_rank`` rule). Every fitter works on ``Z = U S [n, r]`` and maps its coefficients back
with ``V``: for the L2-penalised logistic regression and the ridge this is exact (any component
of ``β`` orthogonal to the rows of ``X_std`` leaves the data term unchanged and only adds
penalty, so the optimum lies in the row span and ``‖Vγ‖ = ‖γ‖``), and for LDA the class means
and the pooled within-class covariance lie in that span while the shrinkage identity acts on
the complement as a scalar that every class shares and the softmax cancels. A 4097-d fit on 300
rows is therefore a 300-dimensional problem; the result does not depend on the basis the SVD
returns (the objectives are rotation-invariant in the reduced space), only on rounding.

**logreg** (``hyper["lambda"]``): ``J(β, b) = Σᵢ wᵢ (lse(ηᵢ) − η_{i,yᵢ}) / Σ wᵢ + λ/2 ‖β‖²_F``
with ``ηᵢ = β zᵢ + b``. K = 2 is the binary form: one weight vector (row 0, Yes against No),
row 1 identically zero, so ``probs = [σ(η₀), 1 − σ(η₀)]`` and the penalty is ``λ/2 ‖w‖²``.
K > 2 is the symmetric multinomial form (every class has a row, all K rows penalised, which
makes the coefficients identifiable and the fit invariant to the option order) with the
intercept gauge ``b_{K−1} = 0`` (the softmax is invariant to a common shift of the intercepts,
so the gauge changes no probability). Newton's method from ``(0, 0)`` with Armijo step halving
(constant :data:`LOGREG_ARMIJO`, at most :data:`LOGREG_MAX_HALVINGS` halvings; inside the
quadratic region, a predicted decrease below :data:`LOGREG_QUADRATIC` of the loss, the full step
is taken), at most :data:`LOGREG_MAX_ITER` iterations, stopping when every gradient component
is below :data:`LOGREG_GTOL`. **A fit that stops before the tolerance is used as it is** with
``converged: false`` (never refitted, never skipped), as the M2 calibrators are (§6.4).

**lda** (``hyper["gamma"]``): weighted class means ``μ_k = Σ_{i∈k} wᵢ zᵢ / W_k``, weighted
class priors ``π_k = W_k / W``, pooled weighted within-class covariance ``Σ = Σᵢ wᵢ (zᵢ −
μ_{yᵢ})(zᵢ − μ_{yᵢ})ᵀ / W`` (normalised by the total weight), shrunk toward the scaled identity
``Σ_γ = (1 − γ) Σ + γ ν I`` with ``ν = tr(Σ) / d`` (the trace over the full standardized space
divided by its dimension ``d``, the Ledoit-Wolf target), posteriors ``softmax_k(zᵀ Σ_γ⁻¹ μ_k −
½ μ_kᵀ Σ_γ⁻¹ μ_k + ln π_k)``. A class with no training weight has no mean and no prior: the
fit is refused (:class:`LinearError`), which the nested selection records as that
configuration being unfittable on that fold. Closed form: ``iterations 0``, ``converged``.

**ridge** (``hyper["lambda"]``): ``J(β, b) = Σᵢ wᵢ ‖β zᵢ + b − e_{yᵢ}‖² / (2 Σ wᵢ) + λ/2 ‖β‖²_F``
on the one-hot targets, solved by the √w row scaling of the normal equations ``(ZᵀWZ / W + λI) γ
= ZᵀW(Y − b) / W``; the unpenalised intercept is the weighted class frequency (the standardized
features have weighted mean zero, so it decouples). The scores are the fitted one-hot
regressions; their softmax (K = 2: ``σ`` of the score difference) is a ranking that only the
OOF calibrator makes a probability.

:data:`GRIDS` holds each fitter's hyperparameter grid in its **declared order**, which is the
tie order of the nested selection; :data:`FITTER_CONSTANTS` records every constant (Appendix B).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mesa_clm.learn.calibrate import softmax

__all__ = [
    "FITTER_CONSTANTS",
    "GRIDS",
    "KINDS",
    "LOGREG_ARMIJO",
    "LOGREG_GTOL",
    "LOGREG_MAX_HALVINGS",
    "LOGREG_MAX_ITER",
    "LOGREG_QUADRATIC",
    "RANK_EPS",
    "STD_FLOOR",
    "Design",
    "Kind",
    "LinearError",
    "LinearModel",
    "Standardizer",
    "fit_design",
    "fit_linear",
    "log_softmax",
    "prepare",
]

Kind = Literal["logreg", "lda", "ridge"]
KINDS: Final[tuple[Kind, ...]] = ("logreg", "lda", "ridge")
# The hyperparameter grids (M4 brief R1), in declared order = the nested selection's tie order.
GRIDS: Final[dict[Kind, tuple[dict[str, float], ...]]] = {
    "logreg": ({"lambda": 1e-2}, {"lambda": 1e-1}, {"lambda": 1.0}, {"lambda": 10.0}),
    "lda": ({"gamma": 0.1}, {"gamma": 0.5}, {"gamma": 0.9}),
    "ridge": ({"lambda": 1e-1}, {"lambda": 1.0}, {"lambda": 10.0}),
}
# The weighted standard deviation of a feature is floored here before dividing (a constant
# column maps to 0 on the training rows and gets a zero coefficient: it spans nothing).
STD_FLOOR: Final[float] = 1e-8
# Singular values of the standardized training matrix above s_max * max(n, d) * RANK_EPS span
# the reduced space (numpy.linalg.matrix_rank's default tolerance).
RANK_EPS: Final[float] = float(np.finfo(np.float64).eps)
LOGREG_MAX_ITER: Final[int] = 100
# Newton stops once every gradient component of the weighted-mean objective is below this.
LOGREG_GTOL: Final[float] = 1e-8
LOGREG_ARMIJO: Final[float] = 1e-4
LOGREG_MAX_HALVINGS: Final[int] = 60
LOGREG_QUADRATIC: Final[float] = 1e-12

FITTER_CONSTANTS: Final[dict[str, object]] = {
    "standardization": {
        "what": "weighted mean and weighted (biased) std of the training set per feature",
        "std_floor": STD_FLOOR,
        "weights": "label weights as sample weights (D20)",
    },
    "reduction": {
        "what": "thin SVD of the standardized training matrix; fits in its row span",
        "rank_tolerance": "s_max * max(n, d) * eps",
        "eps": RANK_EPS,
    },
    "logreg": {
        "implementation": "mesa_clm.learn.linear.fit_linear(kind='logreg')",
        "objective": "weighted mean NLL + lambda/2 * ||coef||^2 (intercept unpenalised)",
        "form": "binary (K = 2, row 0 vs 1); symmetric multinomial with gauge b[K-1] = 0 (K > 2)",
        "grid": [h["lambda"] for h in GRIDS["logreg"]],
        "max_iter": LOGREG_MAX_ITER,
        "gtol": LOGREG_GTOL,
        "armijo": LOGREG_ARMIJO,
        "max_halvings": LOGREG_MAX_HALVINGS,
        "quadratic": LOGREG_QUADRATIC,
        "start": "zero",
        "non_converged": "used as it is, flagged converged: false",
    },
    "lda": {
        "implementation": "mesa_clm.learn.linear.fit_linear(kind='lda')",
        "means": "weighted class means",
        "covariance": "pooled weighted within-class covariance / total weight",
        "shrinkage": "(1 - gamma) * Sigma + gamma * (tr(Sigma) / d) * I",
        "priors": "weighted class priors W_k / W",
        "grid": [h["gamma"] for h in GRIDS["lda"]],
        "absent_class": "refused (LinearError): unfittable on that fold",
    },
    "ridge": {
        "implementation": "mesa_clm.learn.linear.fit_linear(kind='ridge')",
        "objective": "weighted mean squared error on one-hot targets / 2 + lambda/2 * ||coef||^2",
        "intercept": "weighted class frequency (unpenalised)",
        "grid": [h["lambda"] for h in GRIDS["ridge"]],
        "probabilities": "softmax of the fitted scores (K = 2: sigma of the difference)",
    },
}

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class LinearError(ValueError):
    """Inputs a linear model cannot be fitted on or applied to: shapes, labels, weights, a
    hyperparameter outside its range, an LDA class without training weight, a non-finite fit."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# -- inputs -----------------------------------------------------------------------------------------


def _matrix(X: npt.ArrayLike, what: str = "X") -> FloatArray:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise LinearError(f"{what} must be [n, d], got shape {list(x.shape)}")
    if not np.all(np.isfinite(x)):
        raise LinearError(f"{what} contains a non-finite value")
    return x


def _weights(weights: npt.ArrayLike | None, n: int) -> FloatArray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise LinearError(f"weights must have one entry per row ({w.shape} vs {n})")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise LinearError("weights must be finite and non-negative")
    if not w.sum() > 0:
        raise LinearError("the weights sum to zero: nothing to fit")
    return w


def _labels(y: npt.ArrayLike, n: int, k: int) -> IntArray:
    labels = np.asarray(y, dtype=np.int64)
    if labels.shape != (n,):
        raise LinearError(f"labels must have one entry per row ({labels.shape} vs {n})")
    if np.any(labels < 0) or np.any(labels >= k):
        raise LinearError(f"labels must index the {k} options")
    return labels


def _hyper(kind: Kind, hyper: Mapping[str, float]) -> dict[str, float]:
    key = "gamma" if kind == "lda" else "lambda"
    if set(hyper) != {key}:
        raise LinearError(f"{kind} takes the hyperparameter {key!r}, got {sorted(hyper)}")
    value = float(hyper[key])
    if not np.isfinite(value) or value <= 0 or (kind == "lda" and value > 1):
        rng = "(0, 1]" if kind == "lda" else "(0, inf)"
        raise LinearError(f"{kind}: {key} must be in {rng}, got {value}")
    return {key: value}


def log_softmax(scores: npt.ArrayLike) -> FloatArray:
    """Row-wise ``scores − logsumexp(scores)`` of an ``[n, K]`` array (the log-probabilities
    the K > 2 OOF calibrator is fitted on)."""
    x = _matrix(scores, "scores")
    m = x.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))
    out: FloatArray = x - lse
    return out


# -- standardization and the reduced design ----------------------------------------------------


class Standardizer(_Frozen):
    """Per-feature weighted mean and weighted standard deviation of a training set (std floored
    at :data:`STD_FLOOR`); ``transform`` maps any ``[n, d]`` matrix to standardized features."""

    mean: list[float]
    std: list[float]

    @model_validator(mode="after")
    def _shapes(self) -> Standardizer:
        if len(self.mean) != len(self.std):
            raise ValueError("mean and std must have one entry per feature")
        if any(s < STD_FLOOR for s in self.std):
            raise ValueError(f"std is floored at {STD_FLOOR}")
        return self

    @property
    def d(self) -> int:
        return len(self.mean)

    @classmethod
    def fit(cls, X: npt.ArrayLike, weights: npt.ArrayLike | None = None) -> Standardizer:
        x = _matrix(X)
        w = _weights(weights, x.shape[0])
        total = float(w.sum())
        mean = (w @ x) / total
        var = (w @ (x - mean) ** 2) / total
        std = np.maximum(np.sqrt(np.maximum(var, 0.0)), STD_FLOOR)
        return cls(mean=[float(v) for v in mean], std=[float(v) for v in std])

    def transform(self, X: npt.ArrayLike) -> FloatArray:
        x = _matrix(X)
        if x.shape[1] != self.d:
            raise LinearError(f"expected {self.d} features, got {x.shape[1]}")
        out: FloatArray = (x - np.asarray(self.mean)) / np.asarray(self.std)
        return out


@dataclass(frozen=True)
class Design:
    """A training set prepared once for every fitter (module docstring "Reduction"):
    ``standardizer`` from its weighted moments, ``z [n, r]`` the standardized rows in the
    orthonormal basis ``v [d, r]`` of their span, ``w [n]`` the sample weights."""

    standardizer: Standardizer
    z: FloatArray
    v: FloatArray
    w: FloatArray

    @property
    def n(self) -> int:
        return int(self.z.shape[0])

    @property
    def r(self) -> int:
        return int(self.z.shape[1])

    @property
    def d(self) -> int:
        return self.standardizer.d

    @property
    def weight_sum(self) -> float:
        return float(self.w.sum())

    def lift(self, gamma: FloatArray) -> FloatArray:
        """``[K, d]`` coefficients over the standardized features from ``[K, r]`` ones."""
        out: FloatArray = np.asarray(gamma, dtype=np.float64) @ self.v.T
        return out


def prepare(X: npt.ArrayLike, weights: npt.ArrayLike | None = None) -> Design:
    """Standardize ``X`` with the weighted moments of its rows and reduce it to its row span."""
    x = _matrix(X)
    n = x.shape[0]
    if n == 0:
        raise LinearError("nothing to fit: no rows")
    w = _weights(weights, n)
    standardizer = Standardizer.fit(x, w)
    xs = standardizer.transform(x)
    u, s, vt = _thin_svd(xs)
    tol = float(s.max()) * max(xs.shape) * RANK_EPS if s.size else 0.0
    r = int(np.sum(s > tol))
    return Design(standardizer, u[:, :r] * s[:r], vt[:r].T.copy(), w)


def _thin_svd(xs: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``numpy.linalg.svd(xs, full_matrices=False)``, and when LAPACK's divide-and-conquer
    driver does not converge (``LinAlgError``; seen once on a stacked silver-plus-teacher
    design of the X4 run, every entry finite) the same thin SVD from the eigendecomposition of
    the Gram matrix ``xs xsᵀ`` (``n × n``, ``n`` the rows: ``U`` its eigenvectors, ``s`` the
    square roots of its eigenvalues, ``Vᵀ = diag(1/s) Uᵀ xs`` on the components above the rank
    tolerance, the rest zero). Only the failure path differs from numpy's; a design whose SVD
    converges is byte-identical to the one numpy returns."""
    try:
        u, s, vt = np.linalg.svd(xs, full_matrices=False)
    except np.linalg.LinAlgError:
        gram = xs @ xs.T
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        # The rank cut is made on the eigenvalues: a null direction of the Gram matrix carries
        # eigenvalue noise of order eps·λ_max, whose square root (√eps·s_max) would pass the
        # singular-value tolerance of :func:`prepare`; the rest of ``s`` is zeroed so that
        # ``prepare`` keeps exactly the components kept here.
        keep = evals > (float(evals.max()) * max(xs.shape) * RANK_EPS if evals.size else 0.0)
        s = np.where(keep, np.sqrt(np.clip(evals, 0.0, None)), 0.0)
        vt = np.zeros((s.size, xs.shape[1]), dtype=np.float64)
        vt[keep] = (evecs[:, keep].T @ xs) / s[keep][:, None]
        u = evecs
        return u, s, vt
    return u, s, vt


# -- the model ---------------------------------------------------------------------------------------


class LinearModel(_Frozen):
    """A fitted probe fitter (module docstring). ``coef`` is ``[K, d]`` over the standardized
    features (K = 2 logreg: row 0 is the Yes-vs-No weight vector, row 1 zero), ``intercept``
    ``[K]``; ``n`` and ``weight_sum`` describe the training set; ``converged`` and
    ``iterations`` the Newton run (closed forms: ``True``, ``0``)."""

    kind: Kind
    k: int = Field(ge=2)
    hyper: dict[str, float]
    coef: list[list[float]]
    intercept: list[float]
    standardizer: Standardizer
    n: int = Field(ge=1)
    weight_sum: float = Field(gt=0)
    converged: bool
    iterations: int = Field(ge=0)

    @model_validator(mode="after")
    def _shapes(self) -> LinearModel:
        d = self.standardizer.d
        if len(self.coef) != self.k or len(self.intercept) != self.k:
            raise ValueError(f"coef and intercept need {self.k} rows")
        if any(len(row) != d for row in self.coef):
            raise ValueError(f"every coef row needs {d} entries")
        return self

    @property
    def d(self) -> int:
        return self.standardizer.d

    def scores(self, X: npt.ArrayLike) -> FloatArray:
        """``[n, K]`` logits: ``X_std · coefᵀ + intercept`` (ridge: the fitted scores)."""
        xs = self.standardizer.transform(X)
        out: FloatArray = xs @ np.asarray(self.coef, dtype=np.float64).T + np.asarray(
            self.intercept, dtype=np.float64
        )
        return out

    def probs(self, X: npt.ArrayLike) -> FloatArray:
        """``[n, K]`` uncalibrated probabilities: the row-wise softmax of :meth:`scores`."""
        return softmax(self.scores(X))


def _model(
    design: Design,
    kind: Kind,
    k: int,
    hyper: dict[str, float],
    gamma: FloatArray,
    intercept: FloatArray,
    *,
    converged: bool,
    iterations: int,
) -> LinearModel:
    coef = design.lift(gamma)
    if not (np.all(np.isfinite(coef)) and np.all(np.isfinite(intercept))):
        raise LinearError(f"{kind}: the fit is not finite")
    return LinearModel(
        kind=kind,
        k=k,
        hyper=hyper,
        coef=[[float(v) for v in row] for row in coef],
        intercept=[float(v) for v in intercept],
        standardizer=design.standardizer,
        n=design.n,
        weight_sum=design.weight_sum,
        converged=converged,
        iterations=iterations,
    )


# -- logistic regression ------------------------------------------------------------------------


def _lse(eta: FloatArray) -> FloatArray:
    m = eta.max(axis=1)
    out: FloatArray = m + np.log(np.exp(eta - m[:, None]).sum(axis=1))
    return out


def _fit_logreg(
    design: Design, y: IntArray, k: int, lam: float
) -> tuple[FloatArray, FloatArray, bool, int]:
    n, r = design.z.shape
    zh = np.hstack([design.z, np.ones((n, 1))])  # [n, r + 1]: the intercept column last
    w = design.w / design.weight_sum
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0
    free = np.zeros((k, r + 1), dtype=bool)
    if k == 2:
        free[0, :] = True  # binary: one weight vector and one intercept
    else:
        free[:, :r] = True  # symmetric multinomial, intercept gauge b[K-1] = 0
        free[: k - 1, r] = True
    idx = np.flatnonzero(free.ravel())
    pen = np.zeros((k, r + 1), dtype=np.float64)
    pen[:, :r] = lam

    def loss(theta: FloatArray) -> float:
        eta = zh @ theta.T
        return float(w @ (_lse(eta) - eta[np.arange(n), y]) + 0.5 * np.sum(pen * theta * theta))

    def gradient(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        p = softmax(zh @ theta.T)
        g = (w[:, None] * (p - onehot)).T @ zh + pen * theta
        return g, p

    def hessian(p: FloatArray) -> FloatArray:
        m = r + 1
        h = np.zeros((k * m, k * m), dtype=np.float64)
        for a in range(k):
            for b in range(a, k):
                c = w * p[:, a] * ((1.0 if a == b else 0.0) - p[:, b])
                block = zh.T @ (c[:, None] * zh)
                h[a * m : (a + 1) * m, b * m : (b + 1) * m] = block
                if b != a:
                    h[b * m : (b + 1) * m, a * m : (a + 1) * m] = block
        h[np.diag_indices_from(h)] += pen.ravel()
        return h[np.ix_(idx, idx)]

    theta = np.zeros((k, r + 1), dtype=np.float64)
    value = loss(theta)
    iterations = 0
    while iterations < LOGREG_MAX_ITER:
        g, p = gradient(theta)
        gf = g.ravel()[idx]
        if float(np.max(np.abs(gf))) < LOGREG_GTOL:
            break
        iterations += 1
        try:
            step_f = -np.linalg.solve(hessian(p), gf)
        except np.linalg.LinAlgError:
            step_f = -gf
        step = np.zeros(k * (r + 1), dtype=np.float64)
        step[idx] = step_f
        step = step.reshape(k, r + 1)
        slope = float(gf @ step_f)
        if -slope < LOGREG_QUADRATIC * max(1.0, abs(value)):
            theta = theta + step
            value = loss(theta)
            continue
        t = 1.0
        accepted = False
        for _ in range(LOGREG_MAX_HALVINGS):
            trial = theta + t * step
            trial_value = loss(trial)
            if trial_value <= value + LOGREG_ARMIJO * t * slope:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        theta, value = trial, trial_value
    final, _ = gradient(theta)
    converged = float(np.max(np.abs(final.ravel()[idx]))) < LOGREG_GTOL
    return theta[:, :r], theta[:, r], converged, iterations


# -- LDA and ridge ------------------------------------------------------------------------------------


def _fit_lda(design: Design, y: IntArray, k: int, gamma: float) -> tuple[FloatArray, FloatArray]:
    z, w = design.z, design.w
    total = design.weight_sum
    r = design.r
    means = np.zeros((k, r), dtype=np.float64)
    priors = np.zeros(k, dtype=np.float64)
    for c in range(k):
        mask = y == c
        wc = float(w[mask].sum())
        if not wc > 0:
            raise LinearError(f"lda: class {c} has no training weight")
        means[c] = (w[mask] @ z[mask]) / wc
        priors[c] = wc / total
    resid = z - means[y]
    cov = (resid.T @ (w[:, None] * resid)) / total
    nu = float(np.trace(cov)) / design.d if r else 0.0
    if not nu > 0:
        raise LinearError("lda: no within-class scatter to invert")
    shrunk = (1.0 - gamma) * cov + gamma * nu * np.eye(r)
    coef = np.asarray(np.linalg.solve(shrunk, means.T).T, dtype=np.float64)  # Σ_γ⁻¹ μ_k
    intercept = -0.5 * np.einsum("kr,kr->k", means, coef) + np.log(priors)
    return coef, np.asarray(intercept, dtype=np.float64)


def _fit_ridge(design: Design, y: IntArray, k: int, lam: float) -> tuple[FloatArray, FloatArray]:
    z, w = design.z, design.w
    total = design.weight_sum
    n, r = z.shape
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0
    intercept = (w @ onehot) / total
    zw = z * (w / total)[:, None]
    system = z.T @ zw + lam * np.eye(r)
    coef = np.linalg.solve(system, zw.T @ (onehot - intercept)).T if r else np.zeros((k, 0))
    return np.asarray(coef, dtype=np.float64), np.asarray(intercept, dtype=np.float64)


# -- dispatch ----------------------------------------------------------------------------------------


def fit_design(
    design: Design, kind: Kind, y: npt.ArrayLike, *, k: int, hyper: Mapping[str, float]
) -> LinearModel:
    """Fit ``kind`` on a prepared :class:`Design` (one preparation serves every fitter and
    hyperparameter of a training set). Deterministic; :class:`LinearError` on bad inputs or an
    LDA class without training weight."""
    if kind not in KINDS:
        raise LinearError(f"unknown fitter {kind!r}; expected one of {KINDS}")
    if k < 2:
        raise LinearError("k must be at least 2")
    labels = _labels(y, design.n, k)
    params = _hyper(kind, hyper)
    if kind == "logreg":
        gamma, intercept, converged, iterations = _fit_logreg(design, labels, k, params["lambda"])
        return _model(
            design, kind, k, params, gamma, intercept, converged=converged, iterations=iterations
        )
    if kind == "lda":
        gamma, intercept = _fit_lda(design, labels, k, params["gamma"])
    else:
        gamma, intercept = _fit_ridge(design, labels, k, params["lambda"])
    return _model(design, kind, k, params, gamma, intercept, converged=True, iterations=0)


def fit_linear(
    kind: Kind,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    k: int,
    hyper: Mapping[str, float],
) -> LinearModel:
    """One fit of ``kind`` on ``X [n, d]``, labels ``y [n]`` (option indices below ``k``) and
    sample ``weights`` (``None`` = all 1): :func:`prepare` then :func:`fit_design`."""
    return fit_design(prepare(X, weights), kind, y, k=k, hyper=hyper)


def grid_of(kind: Kind) -> Sequence[dict[str, float]]:
    """The declared hyperparameter grid of ``kind`` (:data:`GRIDS`)."""
    if kind not in GRIDS:
        raise LinearError(f"unknown fitter {kind!r}; expected one of {KINDS}")
    return GRIDS[kind]


def jsonable_constants() -> dict[str, Any]:
    """:data:`FITTER_CONSTANTS` as plain JSON values (for a cell's diagnostics)."""
    import json

    out: dict[str, Any] = json.loads(json.dumps(FITTER_CONSTANTS))
    return out
