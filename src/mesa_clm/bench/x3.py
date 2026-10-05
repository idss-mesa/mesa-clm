"""X3: the ``probe`` tier cells of every task (plan §5.3, §5.6 X3, §8 M4; DESIGN D18, D20, D27;
the M4 analysis plan's R1 reading, ``design/m4-analysis-plan.md``).

A probe is a weighted linear model (:mod:`mesa_clm.learn.linear`: ``logreg``, ``lda``, ``ridge``)
over a **feature spec** built from the feature store's vectors of the texts the task's active
framing renders (:class:`mesa_clm.learn.probe.FeatureBuilder`; the framing is X1's decision, A1,
not an X3 axis: F9 for ``column.ontology_fits``, F7 for ``term.fits`` and the closed choices).
The grid is "specs × fitters × hyperparameters" (``bench.registered.X3_GRID`` =
``learn.probe.DEFAULT_GRID``), and every choice in it is made **inside each outer fold** by
grouped inner leave-one-card-out over the fold's six training cards (D27:
``learn.probe.nested_probe``; the inner criterion is the pooled OOF NLL of the uncalibrated
probe, ties to the first configuration in the declared order), then the winner is refitted on
the six cards, its OOF calibrator (Platt at K = 2 and rank_fit, temperature at K > 2; fitted on
at least 100 OOF items, else the fold is skipped ``below_floor``) applied, and the held-out
predictions pooled, never averaged.

Four cells per task, keyed ``<task>.probe.<framing>[@<variant>]``:

* the **nested cell** (no variant): the full grid, ``selection: "nested"``, the only citable-form
  probe cell; its identity (``feature_spec``, ``model``, ``fingerprint.clm_model_fp``) is the
  configuration the **full-data** inner LOCO over all seven cards chooses (what ``learn fit``
  serves), with ``fold_choices`` recording each fold's own configuration and
  ``diagnostics.fold_agreement_with_full`` how many folds agree with it (the citation test asks
  for 5 of 7, R6), exactly as the M2 nested cells carry A1 and ``fold_agreement_with_a1``;
* ``@latest`` and ``@raw``: the same nesting with the grid restricted to the ``clm-latest``
  specs (``lowdim.v1``, ``pair512.v1``, ``choice.state.v1``) or the ``clm-raw`` specs
  (``pair4096.v1``, ``joint4096@S1``, ``joint4096@S1ns``, ``choice.raw.v1``); pre-registered,
  never citable (a variant), what K2's clm-raw clause compares (``bench/k2.py``); each carries
  its own full-data choice over its restricted grid as identity;
* ``@full``: the full-data configuration in every fold (``nested_probe(..., fixed=)``;
  ``selection: "full"``, ``exploratory: true``, D27), where each fold's inner LOCO only fits
  that configuration's OOF calibrator.

A spec's **model** is ``clm-latest`` if it reads any head quantity, else ``clm-raw``
(``learn.probe.SPECS``); the cell's ``model`` and ``fingerprint`` are the chosen spec's, and a
probe cell is ``servable`` (both models are served) whenever a configuration was chosen. A task
whose full-data selection chooses nothing (every inner fold guarded, as ``column.annotate``'s
98 items can be) has ``feature_spec`` and ``model`` ``None`` and is not servable; its
``diagnostics.full_selection`` says why. :func:`mesa_clm.bench.cells.assemble_cell` is the one
cell builder, so a probe cell carries every PR "Cell fields" field, ``items``, ``threshold_cp``,
the lookup controls and ``beats_lookup_novel`` exactly as a calibrated cell does. The producer
applies the M4 registration itself (R5): other labels, bootstrap settings, framings lock, model
fingerprints, grid, active framings or task set make the run unregistered whatever the caller
says.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from mesa_clm import framings
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.cells import (
    CALIBRATED_FLOOR,
    FULL_VARIANT,
    PROBE_FLOOR,
    SERVED_MODELS,
    CellError,
    TextIndex,
    assemble_cell,
    item_identities,
)
from mesa_clm.bench.results import BenchCell, BenchResults, Selection, environment
from mesa_clm.bench.tasks.base import Fold, Task
from mesa_clm.learn.features import FeatureStore
from mesa_clm.learn.offline import LATEST_MODEL, RAW_MODEL, OfflineScorer
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import RANK_FIT_TASKS

__all__ = [
    "CALIBRATION_FLOOR",
    "LATEST_VARIANT",
    "PROBE_FLOOR",
    "PROBE_TIER",
    "RAW_VARIANT",
    "VARIANTS",
    "Nested",
    "active_framing_id",
    "calibration_of",
    "default_grid",
    "feature_builder",
    "fold_entry",
    "full_selection",
    "probe_cell",
    "run_x3",
    "shape_of",
    "spec_model",
    "task_probe_cells",
    "unpack",
]

PROBE_TIER: Final[str] = "probe"
LATEST_VARIANT: Final[str] = "latest"
RAW_VARIANT: Final[str] = "raw"
# The cells of one task in results order: nested, @latest, @raw, @full.
VARIANTS: Final[tuple[str | None, ...]] = (None, LATEST_VARIANT, RAW_VARIANT, FULL_VARIANT)
CALIBRATION_FLOOR: Final[int] = CALIBRATED_FLOOR
TIMING_SOURCE: Final[str] = "offline_replay"

Shape = Literal["rank_fit", "choice"]
FloatArray = npt.NDArray[np.float64]
Clock = Callable[[], float]


# -- the grid and the specs -------------------------------------------------------------------------


def shape_of(task: Task) -> Shape:
    """``rank_fit`` for term.fits and column.ontology_fits, else ``choice``."""
    return "rank_fit" if task.task_id in RANK_FIT_TASKS else "choice"


def active_framing_id(task_id: str) -> str:
    """The framing a probe serves: the task's active framing after A1 (``framings.ACTIVE``)."""
    return framings.ACTIVE[task_id]


def calibration_of(task: Task) -> str:
    """The OOF calibrator a probe cell applies: Platt (rank_fit, K = 2), temperature (K > 2)."""
    return "platt" if task.task_id in RANK_FIT_TASKS or task.k == 2 else "temperature"


def spec_model(spec_id: str) -> str:
    """The served model a spec's features come from (``learn.probe.SPECS``)."""
    from mesa_clm.learn.probe import SPEC_BY_ID

    try:
        model = str(SPEC_BY_ID[spec_id].model)
    except KeyError:
        raise CellError(f"unknown probe spec {spec_id!r}") from None
    if model not in SERVED_MODELS:
        raise CellError(f"spec {spec_id!r} names a model that is not served: {model!r}")
    return model


def default_grid(shape: Shape) -> Any:
    """``learn.probe.DEFAULT_GRID`` for a shape."""
    from mesa_clm.learn.probe import DEFAULT_GRID

    return DEFAULT_GRID[shape]


def _grid_record(grid: Any) -> dict[str, Any]:
    out: dict[str, Any] = grid.as_dict()
    return out


# -- features --------------------------------------------------------------------------------------


def feature_builder(
    task: Task,
    index: TextIndex,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    framing_id: str | None = None,
) -> Any:
    """The task's :class:`~mesa_clm.learn.probe.FeatureBuilder` over its active framing's texts
    (both served models' scorers; the specs read what they need)."""
    from mesa_clm.learn.probe import FeatureBuilder

    missing = sorted(SERVED_MODELS - set(scorers))
    if missing:
        raise CellError(f"{task.name}: no scorer for {', '.join(missing)} (both models are served)")
    return FeatureBuilder(
        task, index, store, dict(scorers), framing_id or active_framing_id(task.task_id)
    )


def _input_tokens(task: Task, index: TextIndex, store: FeatureStore, framing_id: str) -> int:
    """The encoder tokens of the distinct texts the task's items need under ``framing_id``
    (contexts, candidates and the anchor, or the closed options; a cold cache), as
    ``cells.score_arm`` counts them."""
    f = framings.framing(task.task_id, framing_id)
    ids = item_identities(task)
    texts = [index.text(task.task_id, f.id, t, "", "context") for t, _ in ids]
    if f.shape == "rank_fit":
        texts += [index.text(task.task_id, f.id, t, o, "candidate") for t, o in ids]
        texts += [index.text(task.task_id, f.id, t, ANCHOR_KEY, "anchor") for t, _ in ids]
    else:
        texts += [t for target, _ in ids for _, t in index.options(task.task_id, f.id, target)]
    return int(sum(store.token_counts(list(dict.fromkeys(texts)))))


# -- the result of a nested probe --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Nested:
    """What a cell reads from a ``learn.probe.NestedProbeResult``: the pooled predictions, the
    evaluated folds, the skipped folds with reasons, the per-fold choices and the
    diagnostics."""

    idx: list[int]
    probs: FloatArray
    evaluated: list[Fold]
    skipped: dict[str, str]
    fold_choices: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]


def unpack(task: Task, result: Any) -> Nested:
    """A :class:`Nested` of a ``NestedProbeResult`` (``probs``, ``pooled``, ``folds``,
    ``skipped_folds``, ``fold_choices``, ``diagnostics``)."""
    idx = [int(i) for i in result.pooled]
    probs = np.asarray(result.probs, dtype=np.float64)
    return Nested(
        idx=idx,
        probs=probs.reshape(len(idx), task.k) if len(idx) else np.zeros((0, task.k)),
        evaluated=list(result.folds),
        skipped={str(c): str(r) for c, r in dict(result.skipped_folds).items()},
        fold_choices={str(c): dict(v) for c, v in dict(result.fold_choices).items()},
        diagnostics=dict(result.diagnostics),
    )


def full_selection(task: Task, fb: Any, grid: Any) -> dict[str, Any]:
    """The configuration the inner LOCO over all seven cards chooses (the cell's identity and
    the ``@full`` cell's configuration), as the selection's record (``spec``, ``fitter``,
    ``hyper`` and ``model`` are ``None`` with a ``skip_reason`` when nothing is selectable)."""
    from mesa_clm.learn.probe import inner_select

    out: dict[str, Any] = inner_select(task, fb, list(range(len(task.items))), grid=grid).as_dict(
        grid=True
    )
    return out


def fold_entry(
    card: str,
    record: Mapping[str, Any] | None,
    *,
    evaluated: bool,
    reason: str | None,
    framing_id: str,
    question_key: str,
) -> dict[str, Any]:
    """A ``fold_choices`` entry: ``nested_probe``'s record of the fold (its configuration,
    model, head, inner NLL and count, inner skips, calibrator, ``evaluated``) with the framing
    and ``question_key`` every probe fold serves; a fold without a record is a skip."""
    entry = dict(record or {})
    if not entry:
        entry = {"decision": reason or "no_fold_choice", "spec": None, "model": None}
    entry.setdefault("decision", "configuration" if evaluated else (reason or "skipped"))
    entry["framing"] = framing_id if entry.get("spec") else None
    entry["question_key"] = question_key if entry.get("spec") else None
    entry["evaluated"] = evaluated
    if not evaluated and reason and "skip_reason" not in entry:
        entry["skip_reason"] = reason
    if card != card.strip():  # pragma: no cover - cards never carry whitespace
        raise CellError(f"card name {card!r} is not normalised")
    return entry


# -- the cell ----------------------------------------------------------------------------------------


def _agreement(choices: Mapping[str, Mapping[str, Any]], chosen: Mapping[str, Any]) -> int:
    if not chosen.get("spec"):
        return 0
    return sum(
        1
        for c in choices.values()
        if c.get("evaluated")
        and c.get("spec") == chosen["spec"]
        and c.get("fitter") == chosen["fitter"]
        and dict(c.get("hyper") or {}) == dict(chosen["hyper"] or {})
    )


def probe_cell(
    task: Task,
    fb: Any,
    *,
    grid: Any,
    variant: str | None,
    fingerprints: Mapping[str, Mapping[str, str]],
    labels_sha256: str,
    labels_content_sha256: str | None,
    selection: Selection,
    pre_registered: bool,
    exploratory: bool,
    full_choice: Mapping[str, Any] | None = None,
    fixed: bool = False,
    index: TextIndex | None = None,
    store: FeatureStore | None = None,
    notes: str = "",
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> BenchCell:
    """One probe cell of ``task`` (module docstring): the nested procedure over ``grid``
    (``learn.probe.nested_probe``) assembled by :func:`~mesa_clm.bench.cells.assemble_cell`.
    ``full_choice`` is the full-data selection the cell's identity names (computed over ``grid``
    when not given); with ``fixed`` every fold uses it (the ``@full`` cell). ``index`` and
    ``store`` let the cell count its encoder tokens."""
    from mesa_clm.learn.probe import ProbeConfig, nested_probe

    framing_id = active_framing_id(task.task_id)
    frame = framings.framing(task.task_id, framing_id)
    fps = {m: dict(fp).get("clm_model_fp", "") for m, fp in fingerprints.items()}
    chosen = dict(full_choice) if full_choice is not None else full_selection(task, fb, grid)
    config = ProbeConfig.of(chosen) if fixed and chosen.get("spec") else None
    if fixed and config is None:
        raise CellError(
            f"{task.name}: no full-data configuration to fix ({chosen.get('skip_reason')})"
        )
    started = clock()
    nested = unpack(
        task,
        nested_probe(
            task,
            fb,
            grid=grid,
            calibration_floor=CALIBRATION_FLOOR,
            probe_floor=PROBE_FLOOR,
            fixed=config,
            clm_model_fps=fps,
        ),
    )
    elapsed = clock() - started
    outer = list(task.leave_one_card_out())
    evaluated_cards = {f.held_out for f in nested.evaluated}
    choices: dict[str, dict[str, Any]] = {}
    for fold in outer:
        card = fold.held_out
        on = card in evaluated_cards
        if not on and card not in nested.skipped:
            raise CellError(f"{task.name}: fold {card} is neither evaluated nor skipped")
        choices[card] = fold_entry(
            card,
            nested.fold_choices.get(card),
            evaluated=on,
            reason=nested.skipped.get(card),
            framing_id=framing_id,
            question_key=frame.question_key,
        )
    spec = chosen.get("spec")
    model = spec_model(str(spec)) if spec else None
    n_targets = len({t for t, _ in item_identities(task)})
    diagnostics: dict[str, Any] = {
        "arm": f"{framing_id}@probe",
        "n_items": len(task.items),
        "n_targets": n_targets,
        "calls": n_targets,
        "input_tokens": (
            _input_tokens(task, index, store, framing_id)
            if index is not None and store is not None
            else None
        ),
        "ms_per_decision": round(1000.0 * elapsed / max(len(task.items), 1), 4),
        "timing_source": TIMING_SOURCE,
        "full_selection": {
            **{k: v for k, v in chosen.items() if k != "inner_grid"},
            "clm_model_fp": None if model is None else fps.get(model),
        },
        "full_selection_grid": chosen.get("inner_grid"),
        "fold_agreement_with_full": {"agree": _agreement(choices, chosen), "folds": len(outer)},
        **nested.diagnostics,
    }
    return assemble_cell(
        task,
        nested.idx,
        nested.probs,
        nested.evaluated,
        tier="probe",
        framing=framing_id,
        variant=variant,
        model=model,
        question_key=frame.question_key,
        fingerprint=None if model is None else dict(fingerprints[model]),
        feature_spec=None if spec is None else str(spec),
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        selection=selection,
        pre_registered=pre_registered,
        exploratory=exploratory,
        servable=model is not None,
        skipped_folds=nested.skipped,
        fold_choices=choices,
        diagnostics=diagnostics,
        notes=notes,
        B=B,
        seed=seed,
    )


def task_probe_cells(
    task: Task,
    fb: Any,
    fingerprints: Mapping[str, Mapping[str, str]],
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    registered: bool,
    grid: Any = None,
    index: TextIndex | None = None,
    store: FeatureStore | None = None,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> dict[str, BenchCell]:
    """The four probe cells of one task (module docstring), keyed by ``cell_key``. ``grid``
    defaults to ``learn.probe.DEFAULT_GRID`` for the task's shape. ``registered`` says the run
    is the pre-registered one; otherwise every cell is ``pre_registered: false``, ``exploratory:
    true``. A task whose full-data selection chooses nothing has no ``@full`` cell (there is no
    configuration to fix) and says so in the nested cell's diagnostics."""
    grid = grid if grid is not None else default_grid(shape_of(task))
    deviation = "" if registered else "; not the pre-registered run (R5): exploratory"
    full_choice = full_selection(task, fb, grid)
    common: dict[str, Any] = {
        "fingerprints": fingerprints,
        "labels_sha256": labels_sha256,
        "labels_content_sha256": labels_content_sha256,
        "index": index,
        "store": store,
        "B": B,
        "seed": seed,
        "clock": clock,
    }
    out: dict[str, BenchCell] = {}
    nested = probe_cell(
        task,
        fb,
        grid=grid,
        variant=None,
        selection="nested",
        pre_registered=registered,
        exploratory=not registered,
        full_choice=full_choice,
        notes="X3 nested probe: every spec, fitter and hyperparameter chosen by inner LOCO "
        "inside each outer fold (D27); identity = the full-data choice" + deviation,
        **common,
    )
    out[nested.key] = nested
    for variant, model in ((LATEST_VARIANT, LATEST_MODEL), (RAW_VARIANT, RAW_MODEL)):
        if not any(spec_model(str(s)) == model for s in grid.specs):
            continue  # a (deviating) grid without this model's specs has no such variant
        sub = grid.restrict(model)
        cell = probe_cell(
            task,
            fb,
            grid=sub,
            variant=variant,
            selection="nested",
            pre_registered=registered,
            exploratory=not registered,
            full_choice=full_selection(task, fb, sub),
            notes=f"X3 nested probe over the {model} specs only (K2's clm-raw clause); a "
            "variant, never citable" + deviation,
            **common,
        )
        out[cell.key] = cell
    if full_choice.get("spec"):
        full = probe_cell(
            task,
            fb,
            grid=grid,
            variant=FULL_VARIANT,
            selection="full",
            pre_registered=registered,
            exploratory=True,
            full_choice=full_choice,
            fixed=True,
            notes="the full-data configuration (inner LOCO over all seven cards) in every fold: "
            "exploratory, never citable (D27)" + deviation,
            **common,
        )
        out[full.key] = full
    return out


# -- the results file ---------------------------------------------------------------------------------


def _own_deviations(
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    fingerprints: Mapping[str, Mapping[str, str]],
    scorers: Mapping[str, OfflineScorer],
    tasks: Mapping[str, Task],
    grids: Mapping[str, Any],
    B: int,
    seed: int,
) -> list[str]:
    """How an X3 run departs from the M4 registration (R5): labels and bootstrap settings, the
    framings lock of A1 and the model fingerprints, the active framings, the grid per shape, the
    task set, and the specs' models."""
    m4 = reg.current_m4()
    out = [
        *reg.run_deviations(
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            B=B,
            seed=seed,
        ),
        *reg.m4_identity_deviations(
            framings_lock_sha=framings.lock_sha(), fingerprints=fingerprints, models=sorted(scorers)
        ),
        *reg.task_set_deviations(list(tasks)),
    ]
    if dict(framings.ACTIVE) != dict(m4.active):
        out.append(f"active framings {dict(framings.ACTIVE)} are not {dict(m4.active)}")
    for shape, grid in grids.items():
        out.extend(reg.grid_deviations(grid, shape=shape))
        for spec in grid.specs:
            want = dict(m4.grid["spec_models"]).get(str(spec))
            if want is not None and spec_model(str(spec)) != want:
                out.append(f"spec {spec} reads {spec_model(str(spec))}, registered {want}")
    return out


def run_x3(
    tasks: Mapping[str, Task],
    index: TextIndex,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    fingerprints: Mapping[str, Mapping[str, str]],
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    date: str,
    registered: bool,
    deviations: Sequence[str] = (),
    grids: Mapping[str, Any] | None = None,
    name: str = "x3",
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> BenchResults:
    """The ``x3`` results file: :func:`task_probe_cells` for every task (bench task name ->
    task). ``grids`` maps a shape to the grid to run (default ``learn.probe.DEFAULT_GRID``; any
    other grid is a deviation). ``registered`` (with the ``deviations`` that made it false)
    stamps every cell; the producer's own deviations (:func:`_own_deviations`) make the run
    unregistered whatever the caller says, and a registered run's tasks must have their
    published counts."""
    if registered and deviations:
        raise CellError("a run with deviations cannot be the registered one")
    shapes: tuple[Shape, ...] = ("rank_fit", "choice")
    use: dict[str, Any] = {
        s: (grids[s] if grids is not None and s in grids else default_grid(s)) for s in shapes
    }
    own = _own_deviations(
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        fingerprints=fingerprints,
        scorers=scorers,
        tasks=tasks,
        grids=use,
        B=B,
        seed=seed,
    )
    if own:
        registered, deviations = False, [*deviations, *(d for d in own if d not in deviations)]
    if registered:
        for task_name, task in tasks.items():
            reg.check_task(task_name, task)
    cells: dict[str, BenchCell] = {}
    for task in tasks.values():
        fb = feature_builder(task, index, store, scorers)
        cells.update(
            task_probe_cells(
                task,
                fb,
                fingerprints,
                labels_sha256=labels_sha256,
                labels_content_sha256=labels_content_sha256,
                registered=registered,
                grid=use[shape_of(task)],
                index=index,
                store=store,
                B=B,
                seed=seed,
                clock=clock,
            )
        )
    notes = [
        "X3 probe cells for every task (plan §5.3, §5.6 X3, §8 M4), scored offline from the "
        "feature store; design/m4-analysis-plan.md (R1) says how every field is computed.",
        "A probe cell without @ is nested (fold_choices from the inner LOCO of each outer fold; "
        "the only citable-form probe cells); @latest and @raw restrict the grid to one model's "
        "specs (K2's clm-raw clause; variants, never citable); @full uses the full-data "
        "configuration in every fold and is exploratory (D27). The cell's identity "
        "(feature_spec, model, fingerprint) is the full-data choice; "
        "diagnostics.fold_agreement_with_full counts the folds that chose it.",
        "Silver labels are four-model agreement, not truth. ms_per_decision is the offline "
        "replay time of the whole nested procedure per item, not serving latency.",
    ]
    if not registered:
        notes.insert(
            0,
            "NOT the pre-registered run (every cell pre_registered: false, exploratory): "
            + "; ".join(deviations or ["registered=false"]),
        )
    return BenchResults(
        date=date,
        name=name,
        mesa_clm=environment()["mesa_clm"],
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        environment=environment(),
        cells=cells,
        notes=notes,
    )
