"""Client-side calibrators of the ``calibrated`` tier (plan §5.3; DESIGN D6, D7, D20;
``design/m2-analysis-plan.md`` §6.4–§6.6, §9.2–§9.3).

CLM answers with temperature 1 and calibration happens here, on the client (plan §4.1):

* **rank_fit** (``term.fits``, ``column.ontology_fits``): weighted **Platt** scaling of the
  set-independent score ``s_c`` (D2), ``p_fit = σ(a · s_c + b)`` (D7); ``a < 0`` means the
  zero-shot ranking is inverted on the training cards and is flagged ``inverted``, never fixed;
* **closed choice, K = 2** (``column.annotate``): weighted Platt on the logit difference
  ``d = logit_Yes − logit_No``; zero shot is the special case ``a = 1, b = 0``;
* **closed choice, K > 2** (``column.aspect``, ``avu.value_kind``): weighted **temperature**
  scaling, ``p = softmax(logits / T)``.

Label weights are sample weights (D20). Both fitters are deterministic (no random start, fixed
iteration caps) and return frozen pydantic models whose ``model_dump(mode="json")`` is the JSON a
``calibrators.json`` artifact (M4) and a bench cell's diagnostics record; :func:`load_calibrator`
reads one back.

**Platt** minimises the weighted mean negative log-likelihood plus ``λ/2 · (a² + b²)`` with
``λ = 1e-6`` by Newton's method with step halving (Armijo constant 1e-4, at most 60 halvings) from
``(0, 0)``, at most 100 iterations, stopping when every gradient component is below 1e-10. The ridge exists only
so that separable training data (or one class) still has a finite, unique optimum; at this size it
moves a well-posed fit by far less than its sampling error. The targets are the labels
(``targets="hard"``, the tier cells' setting) or Platt's (1999) smoothed targets
``(N₊+1)/(N₊+2)`` for a Yes and ``1/(N₋+2)`` for a No, ``N±`` the unweighted training class
counts (``targets="smoothed"``, what scikit-learn's sigmoid calibration uses); the bench and the
artifacts use :data:`PLATT_TARGETS` (``"hard"``), frozen with the M2 analysis plan.
**Temperature** minimises the weighted mean NLL over ``T ∈ [1e-4, 1e4]``: the NLL is convex in
``β = 1/T`` with derivative ``E_p[l] − l_y`` (monotone in ``β``), which is bisected in ``log β`` to
an interval below 1e-12 (at most 200 halvings); a ``T`` on a bound is reported (``at_bound``), for
instance when the logits carry no information. :data:`FITTER_CONSTANTS` records every constant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

__all__ = [
    "FITTER_CONSTANTS",
    "PLATT_ARMIJO",
    "PLATT_GTOL",
    "PLATT_MAX_HALVINGS",
    "PLATT_MAX_ITER",
    "PLATT_QUADRATIC",
    "PLATT_RIDGE",
    "PLATT_TARGETS",
    "TEMPERATURE_BOUNDS",
    "TEMPERATURE_MAX_ITER",
    "TEMPERATURE_TOL",
    "CalibrationError",
    "Calibrator",
    "PlattCalibrator",
    "Shape",
    "TemperatureCalibrator",
    "apply_calibrator",
    "fit_calibrator",
    "fit_platt",
    "fit_temperature",
    "load_calibrator",
    "logit_difference",
    "sigmoid",
    "softmax",
    "zero_shot_probs",
]

PLATT_RIDGE: Final[float] = 1e-6
PLATT_MAX_ITER: Final[int] = 100
# Newton stops once every gradient component is below this (the objective is O(1)).
PLATT_GTOL: Final[float] = 1e-10
PLATT_ARMIJO: Final[float] = 1e-4
PLATT_MAX_HALVINGS: Final[int] = 60
# Below this Newton decrement (relative to the loss) a line search compares rounding noise.
PLATT_QUADRATIC: Final[float] = 1e-12
# Wide enough that a compressed logit scale (T far below 1) or an uninformative one (T far above
# 1) is fitted rather than clamped; bisection in log(1/T) keeps it cheap (about 45 halvings).
TEMPERATURE_BOUNDS: Final[tuple[float, float]] = (1e-4, 1e4)
# Bisection in log(1/T) stops below this interval width (or after TEMPERATURE_MAX_ITER halvings).
TEMPERATURE_TOL: Final[float] = 1e-12
TEMPERATURE_MAX_ITER: Final[int] = 200

Shape = Literal["rank_fit", "choice"]
PlattFeature = Literal["s_c", "logit_difference"]
PlattTargets = Literal["hard", "smoothed"]
# The targets every calibrator of the bench and of M4's artifacts is fitted with
# (``fit_calibrator``, X1's LOCO-Platt): the labels themselves, frozen with the M2 analysis plan
# (``design/m2-analysis-plan.md`` §6.4). One constant, so two different Platts can never be fitted.
PLATT_TARGETS: Final[PlattTargets] = "hard"
# Every constant that shapes a fitted value, recorded by X1's configuration and every calibrated
# cell (``design/m2-analysis-plan.md`` §6.4, §6.5, §13): a change to any of them is visible in the
# results file and refused by ``bench framing --decide --from``.
FITTER_CONSTANTS: Final[dict[str, object]] = {
    "platt": {
        "implementation": "mesa_clm.learn.calibrate.fit_platt",
        "targets": PLATT_TARGETS,
        "ridge": PLATT_RIDGE,
        "max_iter": PLATT_MAX_ITER,
        "gtol": PLATT_GTOL,
        "armijo": PLATT_ARMIJO,
        "max_halvings": PLATT_MAX_HALVINGS,
        "quadratic": PLATT_QUADRATIC,
        "start": [0.0, 0.0],
        "weights": "label weights as sample weights (D20)",
    },
    "temperature": {
        "implementation": "mesa_clm.learn.calibrate.fit_temperature",
        "bounds": list(TEMPERATURE_BOUNDS),
        "tol": TEMPERATURE_TOL,
        "max_iter": TEMPERATURE_MAX_ITER,
        "search": "bisection of the NLL's derivative in log(1/T)",
        "weights": "label weights as sample weights (D20)",
    },
}

FloatArray = npt.NDArray[np.float64]


class CalibrationError(ValueError):
    """Inputs a calibrator cannot be fitted on or applied to (shapes, labels, weights)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- primitives ---------------------------------------------------------------------------------


def sigmoid(z: npt.ArrayLike) -> FloatArray:
    """``1 / (1 + e^-z)`` element-wise without overflow for large ``|z|``."""
    x = np.asarray(z, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(logits: npt.ArrayLike, temperature: float = 1.0) -> FloatArray:
    """Row-wise ``softmax(logits / T)`` of an ``[n, K]`` array, max-subtracted."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2:
        raise CalibrationError(f"logits must be [n, K], got shape {list(x.shape)}")
    if not temperature > 0:
        raise CalibrationError("temperature must be positive")
    z = x / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: FloatArray = e / e.sum(axis=1, keepdims=True)
    return out


def logit_difference(logits: npt.ArrayLike) -> FloatArray:
    """``logit_Yes − logit_No`` of a two-option ``[n, 2]`` logit array (Yes is index 0)."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise CalibrationError(f"a logit difference needs [n, 2] logits, got {list(x.shape)}")
    out: FloatArray = x[:, 0] - x[:, 1]
    return out


def _two_way(p_yes: FloatArray) -> FloatArray:
    out: FloatArray = np.stack([p_yes, 1.0 - p_yes], axis=1)
    return out


def _weights(weights: npt.ArrayLike | None, n: int) -> FloatArray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise CalibrationError(f"weights must have one entry per item ({w.shape} vs {n})")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise CalibrationError("weights must be finite and non-negative")
    if not w.sum() > 0:
        raise CalibrationError("the weights sum to zero: nothing to fit")
    return w


def _scores(scores: npt.ArrayLike) -> FloatArray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise CalibrationError(f"scores must be a non-empty 1-d array, got {list(s.shape)}")
    if not np.all(np.isfinite(s)):
        raise CalibrationError("scores contain a non-finite value")
    return s


# -- Platt ----------------------------------------------------------------------------------------


class PlattCalibrator(_Frozen):
    """``p_Yes = σ(a · x + b)`` where ``x`` is ``s_c`` (rank_fit) or the logit difference of a
    two-option choice (``feature``). ``nll`` is the weighted mean NLL on the training items at the
    optimum (without the ridge); ``converged`` says Newton met its gradient tolerance."""

    kind: Literal["platt"] = "platt"
    feature: PlattFeature = "s_c"
    targets: PlattTargets = "hard"
    a: float
    b: float
    n: int = Field(ge=1)
    weight_sum: float = Field(gt=0)
    ridge: float = Field(ge=0)
    iterations: int = Field(ge=0)
    converged: bool
    nll: float

    @property
    def inverted(self) -> bool:
        """``a < 0``: on the training cards a higher score meant a lower chance of Yes."""
        return self.a < 0

    def p_yes(self, x: npt.ArrayLike) -> FloatArray:
        """``σ(a · x + b)`` for the 1-d feature ``x``."""
        return sigmoid(self.a * _scores(x) + self.b)

    def probs(self, scores: npt.ArrayLike) -> FloatArray:
        """``[n, 2]`` = ``[p_Yes, 1 − p_Yes]``: ``scores`` is ``s_c [n]`` for ``feature='s_c'``
        and the ``[n, 2]`` option logits for ``feature='logit_difference'``."""
        x = logit_difference(scores) if self.feature == "logit_difference" else _scores(scores)
        return _two_way(self.p_yes(x))


def _platt_loss(
    theta: FloatArray, s: FloatArray, y: FloatArray, w: FloatArray, ridge: float
) -> float:
    z = theta[0] * s + theta[1]
    # y·softplus(−z) + (1 − y)·softplus(z) is −log σ(z) for a Yes and −log(1 − σ(z)) for a No.
    per = y * np.logaddexp(0.0, -z) + (1.0 - y) * np.logaddexp(0.0, z)
    return float(np.dot(w, per) / w.sum() + 0.5 * ridge * float(theta @ theta))


def fit_platt(
    scores: npt.ArrayLike,
    positive: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    feature: PlattFeature = "s_c",
    targets: PlattTargets = "hard",
    ridge: float = PLATT_RIDGE,
    max_iter: int = PLATT_MAX_ITER,
    gtol: float = PLATT_GTOL,
) -> PlattCalibrator:
    """Weighted Platt scaling (module docstring): ``positive[i]`` is true for a Yes label,
    ``weights`` are sample weights (D20; ``None`` = all 1), ``targets`` the labels themselves
    or Platt's smoothed targets. Deterministic: Newton from ``(0, 0)`` with step halving, at
    most ``max_iter`` iterations. A fit that stops before the gradient tolerance (the iteration
    cap, or no decrease along the Newton direction) is returned as it is with ``converged``
    false; the bench uses it unchanged and records the flag in the fold's calibrator, never
    refitting or skipping the fold (``design/m2-analysis-plan.md`` §6.4)."""
    s = _scores(scores)
    yes = np.asarray(positive, dtype=bool)
    if yes.shape != s.shape:
        raise CalibrationError(f"labels must have one entry per score ({yes.shape} vs {s.shape})")
    if ridge < 0:
        raise CalibrationError("ridge must be >= 0")
    if targets == "smoothed":
        n_pos = int(yes.sum())
        n_neg = len(yes) - n_pos
        y = np.where(yes, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    else:
        y = yes.astype(np.float64)
    w = _weights(weights, len(s))
    wsum = float(w.sum())

    def gradient(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        p = sigmoid(theta[0] * s + theta[1])
        r = w * (p - y) / wsum
        return np.array([float(r @ s), float(r.sum())]) + ridge * theta, p

    theta = np.zeros(2, dtype=np.float64)
    loss = _platt_loss(theta, s, y, w, ridge)
    iterations = 0
    while iterations < max_iter:
        grad, p = gradient(theta)
        if float(np.max(np.abs(grad))) < gtol:
            break
        iterations += 1
        h = w * p * (1.0 - p) / wsum
        hess = np.array(
            [[float(h @ (s * s)), float(h @ s)], [float(h @ s), float(h.sum())]]
        ) + ridge * np.eye(2)
        try:
            step = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = -grad  # a singular Hessian (no ridge, no spread): a gradient step instead
        slope = float(grad @ step)
        if -slope < PLATT_QUADRATIC * max(1.0, abs(loss)):
            # The predicted decrease is below the loss's rounding: inside Newton's quadratic
            # region, where comparing losses is noise; the full step is the right one.
            theta = theta + step
            loss = _platt_loss(theta, s, y, w, ridge)
            continue
        t = 1.0
        accepted = False
        for _ in range(PLATT_MAX_HALVINGS):
            trial = theta + t * step
            trial_loss = _platt_loss(trial, s, y, w, ridge)
            if trial_loss <= loss + PLATT_ARMIJO * t * slope:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break  # no decrease along the Newton direction: the optimum, to rounding
        theta, loss = trial, trial_loss
    final, _ = gradient(theta)
    converged = float(np.max(np.abs(final))) < gtol
    nll = _platt_loss(theta, s, y, w, 0.0)
    return PlattCalibrator(
        feature=feature,
        targets=targets,
        a=float(theta[0]),
        b=float(theta[1]),
        n=len(s),
        weight_sum=wsum,
        ridge=float(ridge),
        iterations=iterations,
        converged=converged,
        nll=nll,
    )


# -- temperature ----------------------------------------------------------------------------------


class TemperatureCalibrator(_Frozen):
    """``p = softmax(logits / T)``. ``lower``/``upper`` are the search bounds, ``at_bound``
    says the optimum was on one of them; ``nll`` is the weighted mean NLL on the training items
    at ``T``."""

    kind: Literal["temperature"] = "temperature"
    temperature: float = Field(gt=0)
    n: int = Field(ge=1)
    weight_sum: float = Field(gt=0)
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)
    iterations: int = Field(ge=0)
    at_bound: bool
    nll: float

    def probs(self, logits: npt.ArrayLike) -> FloatArray:
        """``softmax(logits / T)`` row-wise over ``[n, K]`` logits."""
        return softmax(logits, self.temperature)


def _labels(labels: npt.ArrayLike, n: int, k: int) -> npt.NDArray[np.int64]:
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (n,):
        raise CalibrationError(f"labels must have one entry per row ({y.shape} vs {n})")
    if np.any(y < 0) or np.any(y >= k):
        raise CalibrationError(f"labels must index the {k} options")
    return y


def _temperature_grad(beta: float, x: FloatArray, y: npt.NDArray[np.int64], w: FloatArray) -> float:
    """d/dβ of the weighted mean NLL of softmax(β·x): ``Σ w (E_p[x] − x_y) / Σ w``."""
    p = softmax(x * beta)
    expected = np.sum(p * x, axis=1)
    picked = x[np.arange(len(y)), y]
    return float(np.dot(w, expected - picked) / w.sum())


def _temperature_nll(beta: float, x: FloatArray, y: npt.NDArray[np.int64], w: FloatArray) -> float:
    z = x * beta
    m = z.max(axis=1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
    return float(np.dot(w, lse - z[np.arange(len(y)), y]) / w.sum())


def fit_temperature(
    logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> TemperatureCalibrator:
    """Weighted temperature scaling (module docstring): the ``T`` in ``bounds`` that minimises
    the weighted mean NLL of ``softmax(logits / T)``. Deterministic bisection in ``log(1/T)``.
    A ``T`` on a bound is returned as it is with ``at_bound`` true; the bench uses it unchanged
    and records the flag (``design/m2-analysis-plan.md`` §6.5)."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise CalibrationError(f"logits must be a non-empty [n, K>=2] array, got {list(x.shape)}")
    if not np.all(np.isfinite(x)):
        raise CalibrationError("logits contain a non-finite value")
    lo_t, hi_t = (float(bounds[0]), float(bounds[1]))
    if not 0 < lo_t < hi_t:
        raise CalibrationError("temperature bounds must satisfy 0 < lower < upper")
    y = _labels(labels, x.shape[0], x.shape[1])
    w = _weights(weights, x.shape[0])
    # β = 1/T: the NLL is convex in β, its derivative non-decreasing, so the minimiser is the
    # root of the derivative, or a bound when the derivative has one sign on the interval.
    lo, hi = math.log(1.0 / hi_t), math.log(1.0 / lo_t)
    iterations = 0
    if _temperature_grad(math.exp(lo), x, y, w) >= 0:
        beta, at_bound = math.exp(lo), True
    elif _temperature_grad(math.exp(hi), x, y, w) <= 0:
        beta, at_bound = math.exp(hi), True
    else:
        at_bound = False
        while hi - lo > TEMPERATURE_TOL and iterations < TEMPERATURE_MAX_ITER:
            iterations += 1
            mid = 0.5 * (lo + hi)
            if _temperature_grad(math.exp(mid), x, y, w) < 0:
                lo = mid
            else:
                hi = mid
        beta = math.exp(0.5 * (lo + hi))
    return TemperatureCalibrator(
        temperature=1.0 / beta,
        n=x.shape[0],
        weight_sum=float(w.sum()),
        lower=lo_t,
        upper=hi_t,
        iterations=iterations,
        at_bound=at_bound,
        nll=_temperature_nll(beta, x, y, w),
    )


# -- dispatch and JSON ----------------------------------------------------------------------------

Calibrator = Annotated[PlattCalibrator | TemperatureCalibrator, Field(discriminator="kind")]
_ADAPTER: Final[TypeAdapter[PlattCalibrator | TemperatureCalibrator]] = TypeAdapter(Calibrator)


def load_calibrator(data: Mapping[str, Any]) -> PlattCalibrator | TemperatureCalibrator:
    """A calibrator from its ``model_dump(mode="json")``, dispatched on ``kind``."""
    return _ADAPTER.validate_python(dict(data))


def fit_calibrator(
    shape: Shape,
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> PlattCalibrator | TemperatureCalibrator:
    """The plan §5.3 calibrator for a task shape: Platt on ``s_c`` (rank_fit; ``scores`` is
    ``[n]``), Platt on the logit difference (choice, K = 2) or temperature (choice, K > 2;
    ``scores`` is the ``[n, K]`` option logits). ``labels`` are option indices, Yes = 0."""
    y = np.asarray(labels, dtype=np.int64)
    if shape == "rank_fit":
        return fit_platt(scores, y == 0, weights, feature="s_c", targets=PLATT_TARGETS)
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2:
        raise CalibrationError(f"a choice calibrator takes [n, K] logits, got {list(x.shape)}")
    if x.shape[1] == 2:
        return fit_platt(
            logit_difference(x), y == 0, weights, feature="logit_difference", targets=PLATT_TARGETS
        )
    return fit_temperature(x, y, weights)


def apply_calibrator(
    calibrator: PlattCalibrator | TemperatureCalibrator, scores: npt.ArrayLike
) -> FloatArray:
    """The calibrated ``[n, K]`` distribution of ``scores`` (``s_c [n]`` or ``[n, K]`` logits,
    as the calibrator was fitted on)."""
    return calibrator.probs(scores)


def zero_shot_probs(shape: Shape, scores: npt.ArrayLike) -> FloatArray:
    """The zero-shot distribution (plan §5.3): ``[σ(s_c), 1 − σ(s_c)]`` for rank_fit (D7) and
    CLM's softmax over the option logits at temperature 1 for a closed choice."""
    if shape == "rank_fit":
        return _two_way(sigmoid(_scores(scores)))
    return softmax(scores)
