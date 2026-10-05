"""The probe tier: feature specs, the nested spec × fitter × hyperparameter selection, the OOF
calibrator and the probe artifact (plan §5.3 "probe", §5.4, §5.5; DESIGN D18, D19, D20, D27;
the M4 analysis plan ``design/m4-analysis-plan.md`` §2 (P.1–P.7),
whose item numbers are cited below).

**Specs** (:data:`SPECS`; every vector from the feature store of the registered encoder; ``zs``,
``zc`` the 512-d ``clm-latest`` projections, ``xs``, ``xc`` the raw L2-normalised 4096-d
vectors, ``s_latest`` / ``s_raw`` the zero-shot ``s_c`` under each model, ``⊕`` concatenation,
``⊙`` the elementwise product). rank_fit (one row per labelled (target, candidate) pair; the
anchor only forms ``s_c``): ``lowdim.v1 = [s_latest, s_raw]``; ``pair512.v1 = zs⊙zc ⊕ |zs−zc| ⊕
[s_latest]``; ``pair4096.v1 = xs⊙xc ⊕ [s_raw]``; ``joint4096@S1`` / ``joint4096@S1ns`` the raw
vector of the X2 joint context text. Closed choices (one row per labelled target):
``choice.state.v1 = zs ⊕`` the K zero-shot option logits under ``clm-latest``; ``choice.raw.v1 =
xs ⊕`` the K logits under ``clm-raw``. A spec's **model** is ``clm-latest`` when it reads a head
quantity, else ``clm-raw``. :class:`FeatureBuilder` builds the matrices from a
:class:`~mesa_clm.learn.features.FeatureStore`, a :class:`~mesa_clm.bench.cells.TextIndex` and
the :class:`~mesa_clm.learn.offline.OfflineScorer` of each model exactly as
:func:`mesa_clm.bench.cells.score_arm` reads vectors and scores (the same helpers), so its
``zero_shot_logits`` equal the tier cells' scores; :func:`rank_fit_features` and
:func:`choice_features` are the spec formulas themselves, shared with serving.

**Nested selection** (D27). For outer fold *c* (held-out card *c*, training cards *T*):
:func:`inner_select` runs a grouped leave-one-card-out over *T* (held-out card *t*, training
the other cards); each inner fold applies the 30/5 guard and the probe floor of 40 training
items to its training items; every configuration (spec, fitter, hyper) of the :class:`Grid` is
fitted (:mod:`mesa_clm.learn.linear`) on every passing inner fold and its out-of-fold
predictions pooled over *T*; the inner criterion is the **pooled OOF NLL of the uncalibrated
probe probabilities** (unweighted over the items, the quantity the cell's metrics and rule R
measure, as X1's inner LOCO-Platt NLL was; the label weights are the fits' sample weights);
the winner has the lowest inner NLL, ties going to the first configuration in the grid's
declared order (specs, then fitters, then hyperparameters). A configuration that cannot be
fitted on a passing inner fold (an LDA class without training weight) is recorded unfittable
and is not selectable on that fold. Then the **OOF calibrator** is fitted, weighted, on the
winner's pooled OOF predictions: K = 2 (rank_fit and ``column.annotate``) a Platt on the OOF
logit ``ln p/(1−p)`` (the difference of the two scores, ``feature = logit_difference``), K > 2 a
temperature on the OOF log-probabilities (``calibrate.fit_calibrator``); fewer than 100 OOF
items skip the outer fold ``below_floor n < 100 OOF items (probe calibrator)``. The winner is
refitted on all of *T* (guard 30/5 and floor 40 on *T*) and applied to card *c*, the calibrator
applied; held-out predictions are pooled, never averaged. :func:`nested_probe` returns them in
the form :func:`mesa_clm.bench.cells.assemble_cell` takes, with ``fold_choices`` and the
report-only diagnostics (the uncalibrated held-out metrics of every fitted fold, every
configuration's inner NLL, the calibrator records). With ``fixed`` (the ``@full`` cell) every
outer fold uses one configuration and only the calibrator and the refit are per fold.

**Teacher rows** (D19, plan §5.4): a :class:`TeacherRows` set is added to every training set,
outer and inner, except the rows whose ``leak_group`` equals the held-out card's product; they
are never in a test index (they are not items of the task), the guards and floors count the
task's items only, and every fold records how many teacher rows it trained on.

**Artifact.** :class:`ProbeArtifact` is ``probes/<question_key>.json`` (plan §5.5): the spec,
the model, the fitted :class:`~mesa_clm.learn.linear.LinearModel`, the calibrator in
``learn.calibrate``'s JSON form, the selection record and the label identity; ``predict(X)`` is
the calibrated ``[n, K]`` distribution of spec features ``X``. :func:`full_probe` builds the
full-data artifact: the configuration chosen by the inner LOCO over all cards, the calibrator
from those OOF predictions (same floor), the fit on every item (and every teacher row).
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from mesa_clm import framings
from mesa_clm.bench import stats
from mesa_clm.bench.cells import (
    CALIBRATED_FLOOR,
    PROBE_FLOOR,
    CellError,
    TextIndex,
    _unique_vectors,
    calibration_summary,
    item_identities,
    mean_pairwise_cosine,
)
from mesa_clm.bench.tasks.base import (
    MIN_TRAIN_PER_CLASS,
    Fold,
    Task,
    fold_guard,
)
from mesa_clm.clm.encoder import EMBEDDING_DIM
from mesa_clm.learn import calibrate
from mesa_clm.learn.calibrate import PlattCalibrator, TemperatureCalibrator, softmax
from mesa_clm.learn.features import FeatureStore
from mesa_clm.learn.linear import (
    GRIDS,
    KINDS,
    Design,
    Kind,
    LinearError,
    LinearModel,
    fit_design,
    jsonable_constants,
    log_softmax,
    prepare,
)
from mesa_clm.learn.offline import LATEST_MODEL, RAW_MODEL, OfflineScorer, pairwise_s_c
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import RANK_FIT_TASKS

__all__ = [
    "CALIBRATION_FLOOR",
    "DEFAULT_GRID",
    "FITTERS",
    "PROBE_FLOOR",
    "PROBE_FORMAT",
    "PROJECTION_DIM",
    "SPECS",
    "SPEC_BY_ID",
    "FeatureBuilder",
    "FeatureSource",
    "FoldFit",
    "Grid",
    "InnerFit",
    "NestedProbeResult",
    "ProbeArtifact",
    "ProbeConfig",
    "ProbeError",
    "SelectionRecord",
    "SpecInfo",
    "SpecShape",
    "TeacherRows",
    "calibrator_input",
    "choice_features",
    "full_probe",
    "inner_select",
    "load_probe",
    "nested_probe",
    "rank_fit_features",
    "spec_shape",
]

SpecShape = Literal["rank_fit", "choice"]
PROBE_FORMAT: Final[str] = "mesa-clm/probe/1"
# The released head projects to 512 dimensions (clm/headproj.py HeadConfig.projection_dim).
PROJECTION_DIM: Final[int] = 512
# The fewest OOF items an outer fold's probe calibrator is fitted on (M4 brief R1; the same 100
# as the calibrated tier's floor, applied to the pooled inner OOF items).
CALIBRATION_FLOOR: Final[int] = CALIBRATED_FLOOR
FITTERS: Final[tuple[Kind, ...]] = KINDS
_CLOCK_SOURCE: Final[str] = "offline_replay"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Clock = Callable[[], float]


class ProbeError(ValueError):
    """Inputs the probe procedure cannot honestly run on: an unknown spec, a grid of another
    shape, a builder that does not match the task, teacher rows without a leak group, a
    full-data fit below a floor or without a fittable configuration."""


# -- specs -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecInfo:
    """One registered feature spec: its id, the shape of task it serves, the model whose
    quantities it reads (``clm-latest`` for any head quantity, else ``clm-raw``) and its
    registered dimension (``base_dim`` plus ``K`` when ``per_option``)."""

    id: str
    shape: SpecShape
    model: str
    base_dim: int
    per_option: bool = False

    def dim(self, k: int) -> int:
        return self.base_dim + (k if self.per_option else 0)


SPECS: Final[tuple[SpecInfo, ...]] = (
    SpecInfo("lowdim.v1", "rank_fit", LATEST_MODEL, 2),
    SpecInfo("pair512.v1", "rank_fit", LATEST_MODEL, 2 * PROJECTION_DIM + 1),
    SpecInfo("pair4096.v1", "rank_fit", RAW_MODEL, EMBEDDING_DIM + 1),
    SpecInfo("joint4096@S1", "rank_fit", RAW_MODEL, EMBEDDING_DIM),
    SpecInfo("joint4096@S1ns", "rank_fit", RAW_MODEL, EMBEDDING_DIM),
    SpecInfo("choice.state.v1", "choice", LATEST_MODEL, PROJECTION_DIM, True),
    SpecInfo("choice.raw.v1", "choice", RAW_MODEL, EMBEDDING_DIM, True),
)
SPEC_BY_ID: Final[dict[str, SpecInfo]] = {s.id: s for s in SPECS}
_JOINT_SPECS: Final[frozenset[str]] = frozenset({"joint4096@S1", "joint4096@S1ns"})


def spec_info(spec_id: str) -> SpecInfo:
    try:
        return SPEC_BY_ID[spec_id]
    except KeyError:
        raise ProbeError(
            f"unknown spec {spec_id!r}; expected one of {sorted(SPEC_BY_ID)}"
        ) from None


def spec_shape(task: Task) -> SpecShape:
    """The spec shape of a bench task: rank_fit for the two pair tasks, choice otherwise."""
    return "rank_fit" if task.task_id in RANK_FIT_TASKS else "choice"


def _optional(value: npt.ArrayLike | None) -> FloatArray | None:
    return None if value is None else np.asarray(value, dtype=np.float64)


def _need(name: str, value: FloatArray | None, spec_id: str) -> FloatArray:
    if value is None:
        raise ProbeError(f"{spec_id} needs {name}")
    return np.asarray(value, dtype=np.float64)


def rank_fit_features(
    spec_id: str,
    *,
    s_latest: npt.ArrayLike | None = None,
    s_raw: npt.ArrayLike | None = None,
    zs: npt.ArrayLike | None = None,
    zc: npt.ArrayLike | None = None,
    xs: npt.ArrayLike | None = None,
    xc: npt.ArrayLike | None = None,
    joint: npt.ArrayLike | None = None,
) -> FloatArray:
    """The ``[n, d]`` rank_fit spec matrix from its inputs (module docstring; ``zs``/``zc``
    ``[n, 512]`` projections, ``xs``/``xc`` ``[n, 4096]`` raw vectors, ``s_*`` ``[n]``, ``joint``
    the joint context vectors). The one place the formulas live: the bench and serving share it."""
    info = spec_info(spec_id)
    if info.shape != "rank_fit":
        raise ProbeError(f"{spec_id} is a {info.shape} spec")
    arr = _optional
    if spec_id == "lowdim.v1":
        a, b = _need("s_latest", arr(s_latest), spec_id), _need("s_raw", arr(s_raw), spec_id)
        return np.stack([a, b], axis=1)
    if spec_id == "pair512.v1":
        s, c = _need("zs", arr(zs), spec_id), _need("zc", arr(zc), spec_id)
        sl = _need("s_latest", arr(s_latest), spec_id)
        return np.hstack([s * c, np.abs(s - c), sl[:, None]])
    if spec_id == "pair4096.v1":
        s, c = _need("xs", arr(xs), spec_id), _need("xc", arr(xc), spec_id)
        sr = _need("s_raw", arr(s_raw), spec_id)
        return np.hstack([s * c, sr[:, None]])
    return _need("joint", arr(joint), spec_id)


def choice_features(
    spec_id: str,
    *,
    zs: npt.ArrayLike | None = None,
    xs: npt.ArrayLike | None = None,
    logits: npt.ArrayLike | None = None,
) -> FloatArray:
    """The ``[n, d + K]`` closed-choice spec matrix: the state vector of the spec's model
    (``zs`` for ``choice.state.v1``, ``xs`` for ``choice.raw.v1``) ⊕ that model's ``[n, K]``
    zero-shot option logits."""
    info = spec_info(spec_id)
    if info.shape != "choice":
        raise ProbeError(f"{spec_id} is a {info.shape} spec")
    latest = info.model == LATEST_MODEL
    s = _need("zs" if latest else "xs", _optional(zs if latest else xs), spec_id)
    lg = _need("logits", _optional(logits), spec_id)
    if lg.ndim != 2 or lg.shape[0] != s.shape[0]:
        raise ProbeError(f"{spec_id}: logits must be [n, K] aligned with the states")
    return np.hstack([s, lg])


# -- the grid ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """One configuration of the grid: a spec, a fitter and its hyperparameters."""

    spec: str
    fitter: str
    hyper: dict[str, float]

    @property
    def model(self) -> str:
        return spec_info(self.spec).model

    def as_dict(self) -> dict[str, Any]:
        return {"spec": self.spec, "fitter": self.fitter, "hyper": dict(self.hyper)}

    @classmethod
    def of(cls, value: ProbeConfig | Mapping[str, Any]) -> ProbeConfig:
        if isinstance(value, ProbeConfig):
            return value
        return cls(
            str(value["spec"]),
            str(value["fitter"]),
            {str(k): float(v) for k, v in dict(value["hyper"]).items()},
        )


@dataclass(frozen=True)
class Grid:
    """The configurations of one shape in **declared order** (the tie order): ``specs`` in
    order, each with ``fitters`` in order, each with its ``hypers`` in order. ``restrict``
    keeps the specs of one model (the ``@latest`` / ``@raw`` cells)."""

    specs: tuple[str, ...]
    fitters: tuple[str, ...]
    hypers: Mapping[str, tuple[dict[str, float], ...]]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ProbeError("a grid needs at least one spec")
        shapes = {spec_info(s).shape for s in self.specs}
        if len(shapes) != 1:
            raise ProbeError(f"a grid's specs must share one shape, got {sorted(shapes)}")
        if len(set(self.specs)) != len(self.specs):
            raise ProbeError("a grid lists a spec twice")
        for f in self.fitters:
            if f not in KINDS:
                raise ProbeError(f"unknown fitter {f!r}; expected one of {KINDS}")
            if not self.hypers.get(f):
                raise ProbeError(f"fitter {f!r} has no hyperparameter grid")
        if not self.fitters:
            raise ProbeError("a grid needs at least one fitter")

    @property
    def shape(self) -> SpecShape:
        return spec_info(self.specs[0]).shape

    def configurations(self) -> tuple[ProbeConfig, ...]:
        return tuple(
            ProbeConfig(s, f, dict(h))
            for s in self.specs
            for f in self.fitters
            for h in self.hypers[f]
        )

    def restrict(self, model: str) -> Grid:
        specs = tuple(s for s in self.specs if spec_info(s).model == model)
        if not specs:
            raise ProbeError(f"no spec of {model!r} in the grid")
        return dataclasses.replace(self, specs=specs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "specs": list(self.specs),
            "fitters": list(self.fitters),
            "hypers": {f: [dict(h) for h in self.hypers[f]] for f in self.fitters},
            "order": "specs, then fitters, then hyperparameters; ties go to the first",
        }


def _default_grid(shape: SpecShape) -> Grid:
    return Grid(tuple(s.id for s in SPECS if s.shape == shape), FITTERS, dict(GRIDS.items()))


DEFAULT_GRID: Final[dict[SpecShape, Grid]] = {
    "rank_fit": _default_grid("rank_fit"),
    "choice": _default_grid("choice"),
}


# -- feature sources ----------------------------------------------------------------------------------


class FeatureSource(Protocol):
    """What the nested procedure reads: the spec matrices of one item set, in item order."""

    @property
    def shape(self) -> SpecShape: ...

    @property
    def k(self) -> int: ...

    @property
    def n(self) -> int: ...

    @property
    def question_key(self) -> str: ...

    @property
    def framing_id(self) -> str: ...

    def matrix(self, spec_id: str) -> FloatArray: ...


@dataclass
class _ArmVectors:
    """One model's vectors and zero-shot scores over the items (item order)."""

    model: str
    states: FloatArray  # [n, p] state-side vectors
    scores: FloatArray  # rank_fit [n] s_c; choice [n, K] option logits
    candidates: FloatArray | None  # rank_fit [n, p]
    diagnostics: dict[str, Any]


class FeatureBuilder:
    """Spec matrices of one task from the store, the manifest texts and the models' scorers
    (module docstring). Vectors are read the way :func:`~mesa_clm.bench.cells.score_arm` reads
    them (``_unique_vectors`` per side and model; projections cached in the store under the
    scorer's ``clm_model_fp``), scores computed with the same maths, matrices cached per spec."""

    def __init__(
        self,
        task: Task,
        index: TextIndex,
        store: FeatureStore,
        scorers: Mapping[str, OfflineScorer],
        framing_id: str,
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        f = framings.framing(task.task_id, framing_id)
        if f.control:
            raise CellError(f"{task.task_id}/{f.id}: a control framing is not a probe framing")
        if not task.items:
            raise CellError(f"{task.name}: no items to build features for")
        self.task = task
        self.framing = f
        self.index = index
        self.store = store
        self.scorers = dict(scorers)
        self.clock = clock
        self.ids = item_identities(task)
        self._shape: SpecShape = spec_shape(task)
        if (f.shape == "rank_fit") != (self._shape == "rank_fit"):
            raise CellError(f"{task.task_id}/{f.id}: the framing's shape is not the task's")
        self._contexts = [index.text(task.task_id, f.id, t, "", "context") for t, _ in self.ids]
        self._cands: list[str] = []
        self._anchors: list[str] = []
        self._options: list[list[tuple[str, str]]] = []
        if self._shape == "rank_fit":
            self._cands = [index.text(task.task_id, f.id, t, o, "candidate") for t, o in self.ids]
            self._anchors = [
                index.text(task.task_id, f.id, t, ANCHOR_KEY, "anchor") for t, _ in self.ids
            ]
        else:
            wire = list(f.closed_options or {})
            self._options = [index.options(task.task_id, f.id, t) for t, _ in self.ids]
            for opts in self._options:
                if [k for k, _ in opts] != wire:
                    raise CellError(
                        f"{task.task_id}/{f.id}: the options are not the closed options"
                    )
        self._arms: dict[str, _ArmVectors] = {}
        self._matrices: dict[str, FloatArray] = {}
        self._build_ms: dict[str, float] = {}

    # -- identity --------------------------------------------------------------------------------------

    @property
    def shape(self) -> SpecShape:
        return self._shape

    @property
    def k(self) -> int:
        return self.task.k

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def question_key(self) -> str:
        return self.framing.question_key

    @property
    def framing_id(self) -> str:
        return self.framing.id

    @property
    def task_id(self) -> str:
        return self.task.task_id

    # -- vectors ---------------------------------------------------------------------------------------

    def _scorer(self, model: str) -> OfflineScorer:
        try:
            return self.scorers[model]
        except KeyError:
            raise CellError(f"{self.task.name}: no scorer for model {model!r}") from None

    def _arm(self, model: str) -> _ArmVectors:
        arm = self._arms.get(model)
        if arm is not None:
            return arm
        scorer = self._scorer(model)
        started = self.clock()
        states, z_s = _unique_vectors(scorer, self.store, "state", self._contexts)
        zs = z_s[[states[t] for t in self._contexts]].astype(np.float64)
        if self._shape == "rank_fit":
            actions, z_a = _unique_vectors(
                scorer, self.store, "action", [*self._cands, *self._anchors]
            )
            zc = z_a[[actions[t] for t in self._cands]]
            za = z_a[[actions[t] for t in self._anchors]]
            scores = pairwise_s_c(scorer.scale, zs.astype(np.float32), zc, za)
            candidates: FloatArray | None = zc.astype(np.float64)
        else:
            actions, z_a = _unique_vectors(
                scorer, self.store, "action", [t for o in self._options for _, t in o]
            )
            scores = np.stack(
                [
                    scorer.logits(z_s[states[c]], z_a[[actions[t] for _, t in opts]])
                    for c, opts in zip(self._contexts, self._options, strict=True)
                ]
            ).astype(np.float64)
            candidates = None
        elapsed = self.clock() - started
        if not np.all(np.isfinite(scores)):
            raise CellError(f"{self.task.task_id}/{model}: a zero-shot score is not finite")
        tokens = self.store.token_counts([*states, *actions])
        n_targets = len({t for t, _ in self.ids})
        arm = _ArmVectors(
            model,
            zs,
            np.asarray(scores, dtype=np.float64),
            candidates,
            {
                "arm": f"{self.framing.id}@{model}",
                "n_items": self.n,
                "n_targets": n_targets,
                "mean_state_cos": mean_pairwise_cosine(z_s),
                "calls": n_targets,
                "input_tokens": int(sum(tokens)),
                "ms_per_decision": round(1000.0 * elapsed / max(self.n, 1), 4),
                "timing_source": _CLOCK_SOURCE,
            },
        )
        self._arms[model] = arm
        return arm

    @property
    def zero_shot_logits(self) -> dict[str, FloatArray]:
        """Per model: rank_fit ``s_c [n]``, choice the ``[n, K]`` option logits (the tier
        cells' scores, :func:`~mesa_clm.bench.cells.score_arm`)."""
        return {m: self._arm(m).scores for m in sorted(self.scorers)}

    def arm_diagnostics(self, model: str) -> dict[str, Any]:
        """``score_arm``'s diagnostics block for ``model`` (``mean_state_cos``, ``calls``,
        ``input_tokens``, ``ms_per_decision``)."""
        return dict(self._arm(model).diagnostics)

    # -- matrices --------------------------------------------------------------------------------------

    def matrix(self, spec_id: str) -> FloatArray:
        """The ``[n, d]`` float64 matrix of ``spec_id`` in item order (cached)."""
        cached = self._matrices.get(spec_id)
        if cached is not None:
            return cached
        info = spec_info(spec_id)
        if info.shape != self._shape:
            raise ProbeError(f"{self.task.name}: {spec_id} is a {info.shape} spec")
        started = self.clock()
        if self._shape == "rank_fit":
            if spec_id in _JOINT_SPECS:
                texts = [
                    self.index.text(self.task.task_id, spec_id, t, o, "context")
                    for t, o in self.ids
                ]
                x = rank_fit_features(spec_id, joint=self.store.get(texts).astype(np.float64))
            elif spec_id == "lowdim.v1":
                x = rank_fit_features(
                    spec_id,
                    s_latest=self._arm(LATEST_MODEL).scores,
                    s_raw=self._arm(RAW_MODEL).scores,
                )
            elif spec_id == "pair512.v1":
                arm = self._arm(LATEST_MODEL)
                x = rank_fit_features(
                    spec_id, zs=arm.states, zc=arm.candidates, s_latest=arm.scores
                )
            else:
                arm = self._arm(RAW_MODEL)
                x = rank_fit_features(spec_id, xs=arm.states, xc=arm.candidates, s_raw=arm.scores)
        else:
            arm = self._arm(info.model)
            x = choice_features(
                spec_id,
                zs=arm.states if info.model == LATEST_MODEL else None,
                xs=arm.states if info.model == RAW_MODEL else None,
                logits=arm.scores,
            )
        if x.shape[0] != self.n or not np.all(np.isfinite(x)):
            raise CellError(f"{self.task.name}: the {spec_id} matrix is malformed")
        self._build_ms[spec_id] = 1000.0 * (self.clock() - started)
        self._matrices[spec_id] = x
        return x

    def spec_diagnostics(self, spec_id: str) -> dict[str, Any]:
        """Report-only: the spec's model, dimension, the collapse diagnostic and the cost of
        its texts (the model's arm block; a joint spec's own texts), and the build time."""
        info = spec_info(spec_id)
        x = self.matrix(spec_id)
        out: dict[str, Any] = {"spec": spec_id, "model": info.model, "dim": int(x.shape[1])}
        if spec_id in _JOINT_SPECS:
            texts = [
                self.index.text(self.task.task_id, spec_id, t, o, "context") for t, o in self.ids
            ]
            distinct = list(dict.fromkeys(texts))
            out.update(
                mean_state_cos=mean_pairwise_cosine(self.store.get(distinct)),
                calls=len(distinct),
                calls_kind="/v1/embeddings, one joint text per labelled pair",
                input_tokens=int(sum(self.store.token_counts(distinct))),
            )
        else:
            arm = self._arm(info.model).diagnostics
            out.update(
                {k: arm[k] for k in ("mean_state_cos", "calls", "input_tokens", "ms_per_decision")}
            )
        out["build_ms"] = round(self._build_ms.get(spec_id, 0.0), 4)
        return out


# -- teacher rows -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRows:
    """Teacher labels (D19) added to training sets only: ``source`` builds their spec matrices
    (a :class:`FeatureBuilder` over the teacher item set, the same framing), ``labels`` their
    option indices, ``weights`` their down-weights, ``leak_groups`` the product code of each
    row; ``rows`` selects a subset of the source (``without`` drops a product)."""

    source: FeatureSource
    labels: IntArray
    weights: FloatArray
    leak_groups: tuple[str, ...]
    rows: IntArray | None = None

    def __post_init__(self) -> None:
        m = len(self.labels)
        if len(self.weights) != m or len(self.leak_groups) != m or self.source.n != m:
            raise ProbeError("teacher rows need one label, weight and leak_group per source row")
        if any(not g for g in self.leak_groups):
            raise ProbeError("every teacher row needs its leak_group (D19)")
        if np.any(self.labels < 0) or np.any(self.labels >= self.source.k):
            raise ProbeError("a teacher label does not index the task's options")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise ProbeError("teacher weights must be finite and non-negative")
        if self.rows is None:
            object.__setattr__(self, "rows", np.arange(m, dtype=np.int64))

    @classmethod
    def from_task(
        cls, task: Task, source: FeatureSource, weights: Sequence[float] | None = None
    ) -> TeacherRows:
        """From a bench :class:`Task` of teacher items (``products`` are the leak groups) and
        the feature source built over it; ``weights`` override the task's (X4's arms)."""
        if not task.products:
            raise ProbeError(f"{task.name}: teacher items need their leak_group (products)")
        w = list(weights) if weights is not None else (task.weights or [1.0] * len(task.items))
        return cls(
            source,
            np.asarray(task.labels, dtype=np.int64),
            np.asarray(w, dtype=np.float64),
            tuple(task.products),
        )

    @property
    def selected(self) -> IntArray:
        rows = self.rows
        if rows is None:  # pragma: no cover - __post_init__ always sets it
            raise ProbeError("teacher rows are not initialised")
        return rows

    @property
    def n(self) -> int:
        return int(self.selected.size)

    def without(self, leak_group: str | None) -> TeacherRows:
        """The rows whose ``leak_group`` is not ``leak_group`` (the held-out card's product)."""
        if leak_group is None:
            return self
        keep = np.asarray(
            [i for i in self.selected.tolist() if self.leak_groups[i] != leak_group],
            dtype=np.int64,
        )
        return dataclasses.replace(self, rows=keep)

    def matrix(self, spec_id: str) -> FloatArray:
        return self.source.matrix(spec_id)[self.selected]

    @property
    def y(self) -> IntArray:
        return self.labels[self.selected]

    @property
    def w(self) -> FloatArray:
        return self.weights[self.selected]


# -- the inner selection -------------------------------------------------------------------------------


def floor_reason(n_train: int, floor: int = PROBE_FLOOR, what: str = "probe") -> str | None:
    """``below_floor n < floor training items (what)`` or ``None`` (§6.3 wording)."""
    if n_train < floor:
        return f"below_floor {n_train} < {floor} training items ({what})"
    return None


@dataclass(frozen=True)
class InnerFit:
    """One configuration's pooled inner result: its OOF NLL and item count, or why it could
    not be fitted."""

    config: ProbeConfig
    nll: float | None
    n: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {**self.config.as_dict(), "nll": self.nll, "n": self.n, "error": self.error}


@dataclass(frozen=True)
class SelectionRecord:
    """The inner selection over one training set: the ``winner`` (``None`` with
    ``skip_reason``), its ``inner_nll`` over ``inner_n`` pooled OOF items, the skipped inner
    folds, every configuration's result (``inner_grid``), the OOF item indices (into the task)
    and the winner's OOF scores on them, and the teacher rows the training set carried."""

    winner: ProbeConfig | None
    inner_nll: float | None
    inner_n: int
    inner_skips: dict[str, str]
    inner_grid: tuple[InnerFit, ...]
    oof_idx: IntArray
    oof_scores: FloatArray | None
    skip_reason: str | None
    n_teacher: int
    inner_folds: int = 0

    def as_dict(self, *, grid: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "spec": None if self.winner is None else self.winner.spec,
            "fitter": None if self.winner is None else self.winner.fitter,
            "hyper": None if self.winner is None else dict(self.winner.hyper),
            "model": None if self.winner is None else self.winner.model,
            "inner_nll": self.inner_nll,
            "inner_n": self.inner_n,
            "inner_skips": dict(self.inner_skips),
            "inner_folds": self.inner_folds,
            "n_teacher": self.n_teacher,
        }
        if self.skip_reason is not None:
            out["skip_reason"] = self.skip_reason
        if grid:
            out["inner_grid"] = [f.as_dict() for f in self.inner_grid]
        return out


def _weights_of(task: Task) -> FloatArray:
    return np.asarray(task.weights or [1.0] * len(task.items), dtype=np.float64)


def _product_of(task: Task, idx: Sequence[int] | IntArray) -> str | None:
    if not task.products or not len(idx):
        return None
    return str(task.products[int(idx[0])])


def _check_source(task: Task, fb: FeatureSource) -> None:
    if fb.shape != spec_shape(task):
        raise ProbeError(
            f"{task.name}: the feature source is {fb.shape}, the task {spec_shape(task)}"
        )
    if fb.k != task.k or fb.n != len(task.items):
        raise ProbeError(f"{task.name}: the feature source does not match the task's items")


def _check_teacher(task: Task, teacher: TeacherRows | None) -> None:
    if teacher is None:
        return
    if teacher.source.k != task.k or teacher.source.shape != spec_shape(task):
        raise ProbeError(f"{task.name}: the teacher rows are not of this task's shape")
    if not task.products:
        raise ProbeError(f"{task.name}: teacher rows need the items' products (leak groups)")


def _design(
    x: FloatArray, y: IntArray, w: FloatArray, teacher: TeacherRows | None, spec: str
) -> tuple[Design, IntArray]:
    """The training design of ``x`` rows (labels ``y``, weights ``w``) with the teacher rows of
    ``spec`` appended, and the labels in the same order."""
    if teacher is None or teacher.n == 0:
        return prepare(x, w), y
    return (
        prepare(np.vstack([x, teacher.matrix(spec)]), np.concatenate([w, teacher.w])),
        np.concatenate([y, teacher.y]),
    )


def inner_select(
    task: Task,
    fb: FeatureSource,
    train_idx: Sequence[int] | npt.ArrayLike,
    *,
    grid: Grid,
    teacher: TeacherRows | None = None,
    probe_floor: int = PROBE_FLOOR,
    fixed: ProbeConfig | None = None,
) -> SelectionRecord:
    """The grouped inner LOCO over the cards of ``train_idx`` (module docstring): every
    configuration of ``grid`` (or only ``fixed``) fitted on every passing inner fold, OOF
    predictions pooled, the winner by pooled OOF NLL with ties to the first in grid order.
    ``teacher`` rows (already without the outer held-out product) are added to every inner
    training set except those of the inner held-out card's product. Nothing here reads an
    item outside ``train_idx``."""
    _check_source(task, fb)
    _check_teacher(task, teacher)
    if grid.shape != fb.shape:
        raise ProbeError(f"{task.name}: the grid is {grid.shape}, the task {fb.shape}")
    train = np.asarray(
        sorted(int(i) for i in np.asarray(train_idx, dtype=np.int64).tolist()), dtype=np.int64
    )
    if train.size == 0:
        raise ProbeError(f"{task.name}: an empty training set")
    labels = np.asarray(task.labels, dtype=np.int64)
    weights = _weights_of(task)
    cards = np.asarray(task.cards, dtype=object)
    configs = (fixed,) if fixed is not None else grid.configurations()
    if fixed is not None and spec_info(fixed.spec).shape != fb.shape:
        raise ProbeError(f"{task.name}: {fixed.spec} is not a {fb.shape} spec")
    n_teacher = 0 if teacher is None else teacher.n
    passing: list[tuple[str, IntArray, IntArray, TeacherRows | None]] = []
    skips: dict[str, str] = {}
    for card in sorted({str(c) for c in cards[train]}):
        on = cards[train] == card
        test, tr = train[on], train[~on]
        reason = fold_guard(task, Fold(card, test.tolist(), tr.tolist()))
        if reason is None:
            reason = floor_reason(len(tr), probe_floor)
        if reason is not None:
            skips[card] = reason
            continue
        t_rows = None if teacher is None else teacher.without(_product_of(task, test))
        passing.append((card, tr, test, t_rows))
    if not passing:
        return SelectionRecord(
            None,
            None,
            0,
            skips,
            tuple(InnerFit(c, None, 0, "no passing inner fold") for c in configs),
            np.zeros(0, dtype=np.int64),
            None,
            "inner_no_passing_fold",
            n_teacher,
        )
    oof_idx = np.sort(np.concatenate([test for _, _, test, _ in passing]))
    position = {int(i): j for j, i in enumerate(oof_idx.tolist())}
    k = task.k
    scores: dict[int, FloatArray] = {
        i: np.full((len(oof_idx), k), np.nan) for i in range(len(configs))
    }
    errors: dict[int, str] = {}
    by_spec: dict[str, list[int]] = {}
    for i, c in enumerate(configs):
        by_spec.setdefault(c.spec, []).append(i)
    for spec, members in by_spec.items():
        x = fb.matrix(spec)
        for _card, tr, test, t_rows in passing:
            design, y_all = _design(x[tr], labels[tr], weights[tr], t_rows, spec)
            rows = [position[int(i)] for i in test.tolist()]
            for i in members:
                if i in errors:
                    continue
                c = configs[i]
                try:
                    model = fit_design(design, c.fitter, y_all, k=k, hyper=c.hyper)  # type: ignore[arg-type]
                except LinearError as exc:
                    errors[i] = f"{_card}: {exc}"
                    continue
                scores[i][rows] = model.scores(x[test])
    y_oof = labels[oof_idx]
    fits: list[InnerFit] = []
    winner: int | None = None
    best = np.inf
    for i, c in enumerate(configs):
        if i in errors:
            fits.append(InnerFit(c, None, 0, errors[i]))
            continue
        nll = float(stats.metric_value("nll", softmax(scores[i]), y_oof))
        fits.append(InnerFit(c, nll, len(oof_idx)))
        if nll < best:  # strict: ties keep the first in declared order
            best, winner = nll, i
    if winner is None:
        return SelectionRecord(
            None,
            None,
            len(oof_idx),
            skips,
            tuple(fits),
            oof_idx,
            None,
            "inner_no_fittable_configuration",
            n_teacher,
            len(passing),
        )
    return SelectionRecord(
        configs[winner],
        best,
        len(oof_idx),
        skips,
        tuple(fits),
        oof_idx,
        scores[winner],
        None,
        n_teacher,
        len(passing),
    )


# -- the calibrator and the artifact -------------------------------------------------------------------


def calibrator_input(scores: npt.ArrayLike, k: int) -> FloatArray:
    """What the probe calibrator is fitted on and applied to: the ``[n, 2]`` scores themselves
    at K = 2 (Platt on their difference, the logit ``ln p/(1−p)``), the log-probabilities
    (``log_softmax``) at K > 2 (temperature)."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != k:
        raise ProbeError(f"scores must be [n, {k}], got {list(x.shape)}")
    return x if k == 2 else log_softmax(x)


def _fit_probe_calibrator(
    scores: FloatArray, y: IntArray, w: FloatArray, k: int
) -> PlattCalibrator | TemperatureCalibrator:
    return calibrate.fit_calibrator("choice", calibrator_input(scores, k), y, w)


def _calibrator_record(cal: PlattCalibrator | TemperatureCalibrator) -> dict[str, Any]:
    if isinstance(cal, PlattCalibrator):
        return {
            "kind": cal.kind,
            "params": {"a": cal.a, "b": cal.b},
            "n": cal.n,
            "converged": cal.converged,
            "inverted": cal.inverted,
            "nll": cal.nll,
        }
    return {
        "kind": cal.kind,
        "params": {"temperature": cal.temperature},
        "n": cal.n,
        "at_bound": cal.at_bound,
        "inverted": False,
        "nll": cal.nll,
    }


class ProbeArtifact(BaseModel):
    """``probes/<question_key>.json`` (module docstring "Artifact")."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["mesa-clm/probe/1"] = "mesa-clm/probe/1"
    question_key: str
    task_id: str
    framing_id: str
    spec: str
    model: str
    fingerprint: dict[str, str]
    linear: LinearModel
    calibrator: calibrate.Calibrator
    selection: dict[str, Any]
    labels_sha256: str
    labels_content_sha256: str
    n_train: int = Field(ge=1)

    @property
    def k(self) -> int:
        return self.linear.k

    def scores(self, X: npt.ArrayLike) -> FloatArray:
        """The uncalibrated ``[n, K]`` probe logits of spec features ``X``."""
        return self.linear.scores(X)

    def predict(self, X: npt.ArrayLike) -> FloatArray:
        """The calibrated ``[n, K]`` distribution (K = 2: ``[p_yes, p_no]``)."""
        return self.calibrator.probs(calibrator_input(self.scores(X), self.k))


def load_probe(data: Mapping[str, Any]) -> ProbeArtifact:
    """A :class:`ProbeArtifact` from its ``model_dump(mode="json")``."""
    return ProbeArtifact.model_validate(dict(data))


def _artifact(
    task: Task,
    fb: FeatureSource,
    config: ProbeConfig,
    linear: LinearModel,
    cal: PlattCalibrator | TemperatureCalibrator,
    selection: Mapping[str, Any],
    *,
    fingerprint: Mapping[str, str],
    labels_sha256: str,
    labels_content_sha256: str,
) -> ProbeArtifact:
    return ProbeArtifact(
        question_key=fb.question_key,
        task_id=task.task_id,
        framing_id=fb.framing_id,
        spec=config.spec,
        model=config.model,
        fingerprint=dict(fingerprint),
        linear=linear,
        calibrator=cal,
        selection=_json_safe(dict(selection)),
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        n_train=linear.n,
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_default))


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


# -- the outer procedure ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldFit:
    """What one evaluated outer fold fitted: the configuration, the refit on its training
    cards (teacher rows included) and the OOF calibrator."""

    config: ProbeConfig
    linear: LinearModel
    calibrator: PlattCalibrator | TemperatureCalibrator


@dataclass
class NestedProbeResult:
    """The pooled held-out probe predictions of a task in :func:`~mesa_clm.bench.cells
    .assemble_cell`'s form: ``probs [n_pooled, K]`` parallel to ``pooled`` (item indices),
    ``folds`` the evaluated :class:`Fold`s, ``skipped_folds`` with reasons, ``fold_choices``
    (every outer card: the configuration, model, inner NLL and count, inner skips, the
    calibrator record, ``evaluated``), ``diagnostics`` (report-only) and ``fits`` per
    evaluated card."""

    task_name: str
    k: int
    probs: FloatArray
    pooled: list[int]
    folds: list[Fold]
    skipped_folds: dict[str, str]
    fold_choices: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    fits: dict[str, FoldFit]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def artifact(
        self,
        card: str,
        task: Task,
        fb: FeatureSource,
        *,
        fingerprint: Mapping[str, str],
        labels_sha256: str,
        labels_content_sha256: str,
    ) -> ProbeArtifact:
        """Fold ``card``'s fit as a :class:`ProbeArtifact` (tests: ``predict`` on the card's
        spec features equals this result's held-out probabilities)."""
        fit = self.fits[card]
        return _artifact(
            task,
            fb,
            fit.config,
            fit.linear,
            fit.calibrator,
            {"fold": card, **{k: v for k, v in self.fold_choices[card].items() if k != "decision"}},
            fingerprint=fingerprint,
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
        )


def _no_choice(reason: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "decision": reason,
        "spec": None,
        "fitter": None,
        "hyper": None,
        "model": None,
        "clm_model_fp": None,
        "evaluated": False,
        **dict(extra or {}),
    }


def nested_probe(
    task: Task,
    fb: FeatureSource,
    *,
    grid: Grid,
    calibration_floor: int = CALIBRATION_FLOOR,
    probe_floor: int = PROBE_FLOOR,
    teacher: TeacherRows | None = None,
    fixed: ProbeConfig | None = None,
    clm_model_fps: Mapping[str, str] | None = None,
) -> NestedProbeResult:
    """The nested probe cell's predictions (module docstring). ``fixed`` evaluates one
    configuration in every outer fold (the ``@full`` cell); ``clm_model_fps`` (model ->
    fingerprint) names each fold's head in ``fold_choices``."""
    _check_source(task, fb)
    _check_teacher(task, teacher)
    if grid.shape != fb.shape:
        raise ProbeError(f"{task.name}: the grid is {grid.shape}, the task {fb.shape}")
    labels = np.asarray(task.labels, dtype=np.int64)
    weights = _weights_of(task)
    fps = dict(clm_model_fps or {})
    k = task.k
    pooled: list[int] = []
    parts: list[FloatArray] = []
    evaluated: list[Fold] = []
    skipped: dict[str, str] = {}
    chosen: dict[str, dict[str, Any]] = {}
    per_fold: dict[str, dict[str, Any]] = {}
    fits: dict[str, FoldFit] = {}
    uncal_idx: list[int] = []
    uncal_parts: list[FloatArray] = []
    outer = list(task.leave_one_card_out())
    for fold in outer:
        card = fold.held_out
        test = np.asarray(fold.test, dtype=np.int64)
        train = np.asarray(fold.train, dtype=np.int64)
        info: dict[str, Any] = {"n": len(fold.test), "n_train": len(fold.train)}
        t_fold = None if teacher is None else teacher.without(_product_of(task, test))
        info["n_teacher"] = 0 if t_fold is None else t_fold.n
        reason = fold_guard(task, fold)
        if reason is None:
            reason = floor_reason(len(fold.train), probe_floor)
        if reason is not None:
            skipped[card] = reason
            chosen[card] = _no_choice(
                reason, {"n_train": len(fold.train), "n_teacher": info["n_teacher"]}
            )
            per_fold[card] = info
            continue
        sel = inner_select(
            task, fb, train, grid=grid, teacher=t_fold, probe_floor=probe_floor, fixed=fixed
        )
        record = {"n_train": len(fold.train), **sel.as_dict(grid=False)}
        info["inner_grid"] = [f.as_dict() for f in sel.inner_grid]
        info.update({k_: v for k_, v in record.items() if k_ != "inner_grid"})
        if sel.winner is None:
            why = sel.skip_reason or "inner_no_choice"
            skipped[card] = why
            chosen[card] = _no_choice(why, record)
            per_fold[card] = info
            continue
        config = sel.winner
        x = fb.matrix(config.spec)
        design, y_all = _design(x[train], labels[train], weights[train], t_fold, config.spec)
        try:
            model = fit_design(design, config.fitter, y_all, k=k, hyper=config.hyper)  # type: ignore[arg-type]
        except LinearError as exc:
            why = f"unfittable {config.spec}/{config.fitter}: {exc}"
            skipped[card] = why
            chosen[card] = _no_choice(why, record)
            per_fold[card] = info
            continue
        held = model.scores(x[test])
        uncal = softmax(held)
        info["linear"] = {
            "converged": model.converged,
            "iterations": model.iterations,
            "d": model.d,
            "r": design.r,
            "n": model.n,
        }
        info["uncalibrated_heldout"] = calibration_summary(uncal, labels[test])
        uncal_idx.extend(fold.test)
        uncal_parts.append(uncal)
        entry: dict[str, Any] = {
            **record,
            "decision": "configuration",
            "clm_model_fp": fps.get(config.model),
            "evaluated": False,
            "converged": model.converged,
        }
        if sel.inner_n < calibration_floor:
            why = f"below_floor {sel.inner_n} < {calibration_floor} OOF items (probe calibrator)"
            skipped[card] = why
            entry["skip_reason"] = why
            chosen[card] = entry
            per_fold[card] = info
            continue
        if sel.oof_scores is None:  # pragma: no cover - a winner always has OOF scores
            raise ProbeError(f"{task.name}: the winner has no OOF scores")
        cal = _fit_probe_calibrator(sel.oof_scores, labels[sel.oof_idx], weights[sel.oof_idx], k)
        part = cal.probs(calibrator_input(held, k))
        entry["calibrator"] = _calibrator_record(cal)
        entry["evaluated"] = True
        info["calibrator"] = cal.model_dump(mode="json")
        chosen[card] = entry
        per_fold[card] = info
        fits[card] = FoldFit(config, model, cal)
        pooled.extend(fold.test)
        parts.append(part)
        evaluated.append(fold)
    if len(evaluated) + len(skipped) != len(outer) or set(chosen) != {f.held_out for f in outer}:
        raise ProbeError(f"{task.name}: a fold is neither evaluated nor skipped")
    probs = np.vstack(parts) if parts else np.zeros((0, k), dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "calibration": "platt" if k == 2 else "temperature",
        "selection": "fixed" if fixed is not None else "nested",
        "fixed": None if fixed is None else fixed.as_dict(),
        "grid": grid.as_dict(),
        "criterion": "pooled OOF NLL of the uncalibrated probe probabilities (unweighted); "
        "ties to the first configuration in grid order",
        "floors": {"probe": probe_floor, "calibrator_oof": calibration_floor},
        "per_fold": per_fold,
        "fitters": jsonable_constants(),
        "calibrators": _json_safe(calibrate.FITTER_CONSTANTS),
        "teacher": teacher is not None and teacher.n > 0,
        "teacher_in_test": False,
        "teacher_rows": 0 if teacher is None else teacher.n,
        "task_counts": {"n": len(task.items), "class_counts": task.class_counts()},
        "excluded": dict(task.meta.get("excluded", {})),
        "chosen_specs": sorted({c["spec"] for c in chosen.values() if c.get("spec")}),
    }
    if uncal_idx:
        order = np.argsort(np.asarray(uncal_idx, dtype=np.int64), kind="stable")
        y = labels[np.asarray(uncal_idx, dtype=np.int64)[order]]
        diagnostics["uncalibrated_pooled"] = {
            "what": "the uncalibrated probe predictions of every fold that fitted a model "
            "(evaluated folds and folds skipped by the calibrator floor alone); report-only",
            "n": len(uncal_idx),
            "cards": sorted(c for c in per_fold if "uncalibrated_heldout" in per_fold[c]),
            **calibration_summary(np.vstack(uncal_parts)[order], y),
        }
    return NestedProbeResult(
        task.name, k, probs, pooled, evaluated, skipped, chosen, _json_safe(diagnostics), fits
    )


# -- the full-data artifact ------------------------------------------------------------------------------


def full_probe(
    task: Task,
    fb: FeatureSource,
    *,
    grid: Grid,
    fingerprints: Mapping[str, Mapping[str, str]],
    labels_sha256: str,
    labels_content_sha256: str,
    teacher: TeacherRows | None = None,
    calibration_floor: int = CALIBRATION_FLOOR,
    probe_floor: int = PROBE_FLOOR,
) -> tuple[ProbeArtifact, dict[str, Any]]:
    """The full-data probe (module docstring "Artifact"; ``learn fit``, plan §5.5): the
    configuration chosen by :func:`inner_select` over every card, its calibrator from those
    pooled OOF predictions (the 100 floor), the fit on every item with every teacher row.
    Returns the artifact (its fingerprint the winner's model's from ``fingerprints``) and the
    report (the selection with every configuration's inner NLL). :class:`ProbeError` when the
    item set is below the probe floor or a two-class guard, or no configuration is selectable;
    the configuration it chose is the one every outer fold of the ``@full`` cell uses."""
    _check_source(task, fb)
    _check_teacher(task, teacher)
    n = len(task.items)
    reason = floor_reason(n, probe_floor)
    if reason is None and task.binary:
        counts = task.class_counts()
        if len(counts) < 2 or min(counts.values()) < MIN_TRAIN_PER_CLASS:
            reason = f"insufficient_train_per_class {counts}"
    if reason is not None:
        raise ProbeError(f"{task.name}: {reason}")
    sel = inner_select(task, fb, range(n), grid=grid, teacher=teacher, probe_floor=probe_floor)
    if sel.winner is None:
        raise ProbeError(f"{task.name}: {sel.skip_reason}")
    if sel.inner_n < calibration_floor:
        raise ProbeError(
            f"{task.name}: below_floor {sel.inner_n} < {calibration_floor} OOF items "
            "(probe calibrator)"
        )
    if sel.oof_scores is None:  # pragma: no cover - a winner always has OOF scores
        raise ProbeError(f"{task.name}: the winner has no OOF scores")
    labels = np.asarray(task.labels, dtype=np.int64)
    weights = _weights_of(task)
    cal = _fit_probe_calibrator(sel.oof_scores, labels[sel.oof_idx], weights[sel.oof_idx], task.k)
    config = sel.winner
    x = fb.matrix(config.spec)
    design, y_all = _design(x, labels, weights, teacher, config.spec)
    try:
        model = fit_design(design, config.fitter, y_all, k=task.k, hyper=config.hyper)  # type: ignore[arg-type]
    except LinearError as exc:
        raise ProbeError(f"{task.name}: unfittable {config.spec}/{config.fitter}: {exc}") from exc
    try:
        fingerprint = fingerprints[config.model]
    except KeyError:
        raise ProbeError(f"no fingerprint for model {config.model!r} (D5)") from None
    selection = {
        "selection": "full",
        "cards": sorted(set(task.cards)),
        **sel.as_dict(grid=False),
        "calibrator": _calibrator_record(cal),
        "linear": {"converged": model.converged, "iterations": model.iterations, "r": design.r},
    }
    artifact = _artifact(
        task,
        fb,
        config,
        model,
        cal,
        selection,
        fingerprint=fingerprint,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
    )
    report = {
        **sel.as_dict(grid=True),
        "grid": grid.as_dict(),
        "n_train": model.n,
        "n_items": n,
        "n_teacher": 0 if teacher is None else teacher.n,
        "calibrator": cal.model_dump(mode="json"),
        "linear": selection["linear"],
    }
    return artifact, _json_safe(report)
