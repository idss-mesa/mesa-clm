"""X4: the teacher ablation on the two rank_fit tasks (plan §5.6 X4, §5.4; DESIGN D19, D20; the
M4 analysis plan's R3 reading, ``design/m4-analysis-plan.md``).

X4 runs the X3 nested probe procedure (``learn.probe.nested_probe``, the machinery of
:mod:`mesa_clm.bench.x3`) three times on **identical outer folds** (the task's
``leave_one_card_out``, a function of the silver items alone) with the teacher rows
(:class:`mesa_clm.learn.labels.TeacherRows`, from the teacher snapshot) added to **every
training set** (outer and inner) at the weights of each arm: ``off`` (no teacher rows),
``(0.5, 0.3)`` and ``(0.3, 0.1)`` for (``teacher``, ``teacher_implicit``). A teacher row whose
``leak_group`` equals the held-out card's product is dropped from that fold's training; no
teacher row is ever in a test fold (``teacher_in_test: false``, asserted: the pooled items are
the silver items of the registered snapshot and nothing else, and ``nested_probe`` records the
teacher rows each fold trained on). Each arm is scored twice: on the **silver** labels of the
registered snapshot (every pooled item) and on **silver-minus-Opus** (the registered items
whose silver label survives when claude-opus-5-5's proposals leave the consensus:
:func:`mesa_clm.learn.labels.surviving_identities` against the silver labels rebuilt with
``ingest_neon_eval(exclude_models=["claude-opus-5-5"])`` and committed as
``bench/snapshots/<date>-minus-opus.parquet``; the same predictions, the subset of items).

Cells, all variants and never citable: ``<task>.probe.<framing>@teacher-off``, ``@teacher-0.5``,
``@teacher-0.3`` (silver) and ``…-minus-opus`` (silver-minus-Opus); ``teacher: true`` on the
on-arms. The cells' identity (``feature_spec``, ``model``) is the off arm's full-data choice,
what production serves. **Decision rule**: keep teacher labels in production fits only if
``on ≻ off`` on **novel-key NLL** (rule R, the items whose lookup key no training card of their
fold carries) for **both** scorings, for the (0.5, 0.3) arm; the (0.3, 0.1) arm is reported.
Expected ≈ null with few teacher rows in the bench products' registry (stated in advance). The
producer applies the M4 registration itself (R5): the teacher and silver-minus-Opus snapshots by
their pins (unpinned: unregistered), the labels, the grid, B/seed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from mesa_clm import framings
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.baselines import evaluate_lookup
from mesa_clm.bench.cells import CellError, TextIndex, _json_safe, assemble_cell, item_identities
from mesa_clm.bench.results import BenchCell, BenchResults, environment
from mesa_clm.bench.tasks.base import Fold, Task
from mesa_clm.bench.x3 import (
    CALIBRATION_FLOOR,
    PROBE_FLOOR,
    Clock,
    active_framing_id,
    default_grid,
    feature_builder,
    fold_entry,
    full_selection,
    spec_model,
    unpack,
)
from mesa_clm.learn.features import FeatureStore
from mesa_clm.learn.labels import TeacherRows, surviving_identities
from mesa_clm.learn.offline import OfflineScorer
from mesa_clm.tasks import RANK_FIT_TASKS

__all__ = [
    "DECISION_ARM",
    "MINUS_OPUS_SUFFIX",
    "OPUS",
    "SCORINGS",
    "TEACHER_ARMS",
    "X4_TASKS",
    "arm_variant",
    "arm_weights",
    "minus_opus_subset",
    "run_x4",
    "task_x4_cells",
]

X4_TASKS: Final[tuple[str, ...]] = tuple(sorted(RANK_FIT_TASKS))
TEACHER_ARMS: Final[tuple[tuple[float, float] | None, ...]] = reg.X4_TEACHER_ARMS
DECISION_ARM: Final[tuple[float, float]] = reg.X4_DECISION_ARM
SCORINGS: Final[tuple[str, str]] = ("silver", "silver-minus-opus")
MINUS_OPUS_SUFFIX: Final[str] = "-minus-opus"
OPUS: Final[str] = "claude-opus-5-5"

FloatArray = npt.NDArray[np.float64]


def arm_variant(arm: tuple[float, float] | None) -> str:
    """``teacher-off`` / ``teacher-0.5`` / ``teacher-0.3`` (the ``teacher`` weight names the arm)."""
    return "teacher-off" if arm is None else f"teacher-{arm[0]:g}"


def arm_weights(arm: tuple[float, float] | None) -> dict[str, float] | None:
    """The source weights of an arm (``None`` for the off arm)."""
    return None if arm is None else {"teacher": float(arm[0]), "teacher_implicit": float(arm[1])}


def minus_opus_subset(task: Task, minus: Task | None) -> list[int] | None:
    """The silver-minus-Opus item subset of ``task`` (module docstring): the indices of its
    items whose identity the rebuilt task ``minus`` carries with the same label; ``None``
    without a rebuilt task."""
    if minus is None:
        return None
    mine = [(t, o, y) for (t, o), y in zip(item_identities(task), task.labels, strict=True)]
    theirs = [(t, o, y) for (t, o), y in zip(item_identities(minus), minus.labels, strict=True)]
    return surviving_identities(mine, theirs)


def _subset_folds(folds: Sequence[Fold], keep: set[int]) -> list[Fold]:
    """The evaluated folds restricted to ``keep`` in their test lists (a fold with no kept item
    is dropped), the training lists untouched (what the lookup controls may use)."""
    out: list[Fold] = []
    for f in folds:
        test = [i for i in f.test if i in keep]
        if test:
            out.append(Fold(f.held_out, test, list(f.train)))
    return out


def _novel_mask(task: Task, folds: Sequence[Fold], idx: Sequence[int]) -> npt.NDArray[np.bool_]:
    """Whether each of the pooled items ``idx`` (in the order given) was a novel key in its
    fold (``baselines.evaluate_lookup`` over the same folds)."""
    ev = evaluate_lookup(task, folds)
    novel = {int(i): bool(n) for i, n in zip(ev.idx, ev.novel, strict=True)}
    return np.asarray([novel[int(i)] for i in idx], dtype=bool)


def _probe_teacher(rows: TeacherRows, weights: Mapping[str, float]) -> Any:
    """The rows as ``learn.probe.TeacherRows`` at the arm's weights, over their feature
    builder (:meth:`mesa_clm.learn.labels.TeacherRows.with_builder`)."""
    from mesa_clm.learn.probe import TeacherRows as ProbeTeacherRows

    if rows.builder is None:
        raise CellError("the teacher rows carry no feature builder (run_x4 attaches it)")
    return ProbeTeacherRows.from_task(
        rows.as_task(), rows.builder, rows.reweighted(weights).weights
    )


def task_x4_cells(
    task: Task,
    fb: Any,
    teacher: TeacherRows | None,
    fingerprints: Mapping[str, Mapping[str, str]],
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    registered: bool,
    minus: Task | None = None,
    grid: Any = None,
    arms: Sequence[tuple[float, float] | None] = TEACHER_ARMS,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> tuple[dict[str, BenchCell], dict[str, Any]]:
    """The X4 cells of one rank_fit task and its decision record (module docstring).
    ``teacher`` carries the task's teacher rows with their feature builder attached
    (``None`` or empty: every on arm trains on nothing more than the off arm, counted);
    ``minus`` is the task rebuilt without Opus (``None``: no silver-minus-Opus scoring, a
    deviation the caller records)."""
    from mesa_clm.learn.probe import nested_probe

    if task.task_id not in RANK_FIT_TASKS:
        raise CellError(f"{task.name}: X4 runs on the rank_fit tasks only")
    grid = grid if grid is not None else default_grid("rank_fit")
    framing_id = active_framing_id(task.task_id)
    frame = framings.framing(task.task_id, framing_id)
    fps = {m: dict(fp).get("clm_model_fp", "") for m, fp in fingerprints.items()}
    chosen = full_selection(task, fb, grid)
    spec = chosen.get("spec")
    model = spec_model(str(spec)) if spec else None
    subset = minus_opus_subset(task, minus)
    deviation = "" if registered else "; not the pre-registered run (R5): exploratory"
    n_teacher = 0 if teacher is None else len(teacher)
    cells: dict[str, BenchCell] = {}
    pooled: dict[str, tuple[list[int], FloatArray, list[Fold]]] = {}
    for arm in arms:
        weights = arm_weights(arm)
        rows = (
            None
            if weights is None or teacher is None or not len(teacher)
            else _probe_teacher(teacher, weights)
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
                teacher=rows,
                clm_model_fps=fps,
            ),
        )
        elapsed = clock() - started
        if any(i < 0 or i >= len(task.items) for i in nested.idx):
            raise CellError(f"{task.name}: a pooled index is not a silver item (teacher_in_test)")
        variant = arm_variant(arm)
        pooled[variant] = (nested.idx, nested.probs, nested.evaluated)
        evaluated_cards = {f.held_out for f in nested.evaluated}
        choices = {
            f.held_out: fold_entry(
                f.held_out,
                nested.fold_choices.get(f.held_out),
                evaluated=f.held_out in evaluated_cards,
                reason=nested.skipped.get(f.held_out),
                framing_id=framing_id,
                question_key=frame.question_key,
            )
            for f in task.leave_one_card_out()
        }
        n_rows = 0 if rows is None else int(rows.n)
        for scoring in SCORINGS:
            if scoring == "silver":
                idx, probs, folds = nested.idx, nested.probs, nested.evaluated
                name = variant
            else:
                if subset is None:
                    continue
                keep = set(subset)
                sel = [j for j, i in enumerate(nested.idx) if i in keep]
                idx = [nested.idx[j] for j in sel]
                probs = nested.probs[sel] if sel else np.zeros((0, task.k))
                folds = _subset_folds(nested.evaluated, keep)
                name = variant + MINUS_OPUS_SUFFIX
            diagnostics: dict[str, Any] = {
                "arm": f"{framing_id}@probe",
                "teacher_arm": variant,
                "teacher_weights": weights,
                "teacher_rows_available": n_teacher,
                "teacher_rows_weighted": n_rows,
                "scoring": scoring,
                "subset_n": None if subset is None else len(subset),
                "full_selection": {
                    **{k: v for k, v in chosen.items() if k != "inner_grid"},
                    "clm_model_fp": None if model is None else fps.get(model),
                },
                "ms_per_decision": round(1000.0 * elapsed / max(len(task.items), 1), 4),
                "timing_source": "offline_replay",
                **nested.diagnostics,
            }
            cell = assemble_cell(
                task,
                idx,
                probs,
                folds,
                tier="probe",
                framing=framing_id,
                variant=name,
                model=model,
                question_key=frame.question_key,
                fingerprint=None if model is None else dict(fingerprints[model]),
                feature_spec=None if spec is None else str(spec),
                labels_sha256=labels_sha256,
                labels_content_sha256=labels_content_sha256,
                selection="nested",
                pre_registered=registered,
                exploratory=True,
                servable=model is not None,
                skipped_folds=nested.skipped,
                fold_choices=choices,
                diagnostics=diagnostics,
                notes=f"X4 teacher ablation, arm {variant}, scored on {scoring}: a variant, "
                "never citable; teacher rows in training only (D19)" + deviation,
                B=B,
                seed=seed,
            )
            cells[cell.key] = cell.model_copy(update={"teacher": weights is not None})
    decision = _decision(task, pooled, subset, B=B, seed=seed)
    for key, cell in cells.items():
        cells[key] = cell.model_copy(
            update={"diagnostics": {**cell.diagnostics, "teacher_decision": decision}}
        )
    return cells, decision


def _decision(
    task: Task,
    pooled: Mapping[str, tuple[list[int], FloatArray, list[Fold]]],
    subset: Sequence[int] | None,
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    """The X4 decision record: for each on arm and scoring, rule R of ``on ≻ off`` on the
    novel-key NLL over the items both arms pooled; ``keep`` reads the (0.5, 0.3) arm on both
    scorings."""
    off = pooled.get(arm_variant(None))
    out: dict[str, Any] = {
        "rule": "keep teacher labels iff on ≻ off (rule R) on novel-key NLL for both scorings, "
        "for the (0.5, 0.3) arm; the (0.3, 0.1) arm is reported",
        "decision_arm": arm_variant(DECISION_ARM),
        "arms": {},
        "keep": None,
    }
    if off is None:
        out["reason"] = "no off arm"
        return out
    off_idx, off_probs, off_folds = off
    off_pos = {int(i): j for j, i in enumerate(off_idx)}
    labels = np.asarray(task.labels, dtype=np.int64)
    kept = None if subset is None else set(subset)
    for variant, (idx, probs, _folds) in pooled.items():
        if variant == arm_variant(None):
            continue
        record: dict[str, Any] = {}
        common = [int(i) for i in idx if int(i) in off_pos]
        on_pos = {int(i): j for j, i in enumerate(idx)}
        for scoring in SCORINGS:
            if scoring == "silver":
                items: list[int] | None = common
            else:
                items = None if kept is None else [i for i in common if i in kept]
            if items is None:
                record[scoring] = {"reason": "no silver-minus-Opus subset"}
                continue
            if not items:
                record[scoring] = {"reason": "no common pooled items"}
                continue
            novel = _novel_mask(task, off_folds, items)
            if not novel.any():
                record[scoring] = {"reason": "no novel-key items", "n": len(items)}
                continue
            sel = [i for i, m in zip(items, novel, strict=True) if m]
            cards = [task.cards[i] for i in sel]
            if len(set(cards)) < 2:
                record[scoring] = {"reason": "fewer than two novel-key cards", "n_novel": len(sel)}
                continue
            rr = stats.rule_r(
                "nll",
                probs[[on_pos[i] for i in sel]],
                off_probs[[off_pos[i] for i in sel]],
                labels[sel],
                cards,
                B=B,
                seed=seed,
            )
            record[scoring] = {"n_novel": len(sel), "n": len(items), **_json_safe(rr.as_dict())}
        out["arms"][variant] = record
    decision = out["arms"].get(arm_variant(DECISION_ARM), {})
    verdicts = [bool(decision.get(s, {}).get("passed", False)) for s in SCORINGS]
    out["keep"] = all(verdicts) and len(verdicts) == len(SCORINGS)
    out["keep_detail"] = dict(zip(SCORINGS, verdicts, strict=True))
    return out


def _own_deviations(
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    fingerprints: Mapping[str, Mapping[str, str]],
    scorers: Mapping[str, OfflineScorer],
    tasks: Mapping[str, Task],
    grid: Any,
    teacher_hashes: tuple[str | None, str | None] | None,
    minus_opus_hashes: tuple[str | None, str | None] | None,
    arms: Sequence[tuple[float, float] | None],
    B: int,
    seed: int,
    teacher_corpus_sha256: str | None = None,
) -> list[str]:
    """How an X4 run departs from the M4 registration (empty: none). The teacher corpus hash
    is compared only when the caller hashed a corpus directory (``teacher_corpus_sha256`` not
    ``None``): the corpus is pinned through the teacher snapshot, and the live corpus may
    move."""
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
        *reg.grid_deviations(grid, shape="rank_fit"),
    ]
    if dict(framings.ACTIVE) != dict(m4.active):
        out.append(f"active framings {dict(framings.ACTIVE)} are not {dict(m4.active)}")
    got = {t.task_id for t in tasks.values()}
    if not set(X4_TASKS) <= got:
        out.append(f"tasks {sorted(set(X4_TASKS) - got)} are not run")
    if tuple(arms) != tuple(m4.teacher_arms):
        out.append(f"teacher arms {list(arms)} are not {list(m4.teacher_arms)}")
    if teacher_hashes is None:
        out.append("no teacher snapshot")
    else:
        out.extend(reg.pinned_snapshot_deviations("teacher", *teacher_hashes))
    if minus_opus_hashes is None:
        out.append("no silver-minus-Opus snapshot")
    else:
        out.extend(reg.pinned_snapshot_deviations("minus_opus", *minus_opus_hashes))
    if teacher_corpus_sha256 is not None:
        if not m4.teacher_corpus_sha256:
            out.append("teacher corpus is not pinned by the M4 registration")
        elif teacher_corpus_sha256 != m4.teacher_corpus_sha256:
            out.append(
                f"teacher corpus sha256 {teacher_corpus_sha256[:12]} is not the pinned "
                f"{m4.teacher_corpus_sha256[:12]}"
            )
    return out


def run_x4(
    tasks: Mapping[str, Task],
    index: TextIndex,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    fingerprints: Mapping[str, Mapping[str, str]],
    *,
    teacher: Mapping[str, TeacherRows],
    minus_opus: Mapping[str, Task] | None,
    labels_sha256: str,
    labels_content_sha256: str | None,
    date: str,
    registered: bool,
    deviations: Sequence[str] = (),
    teacher_hashes: tuple[str | None, str | None] | None = None,
    minus_opus_hashes: tuple[str | None, str | None] | None = None,
    teacher_corpus_sha256: str | None = None,
    grid: Any = None,
    arms: Sequence[tuple[float, float] | None] = TEACHER_ARMS,
    name: str = "x4",
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> BenchResults:
    """The ``x4`` results file: :func:`task_x4_cells` for the rank_fit tasks of ``tasks`` (bench
    task name -> task). ``teacher`` maps a task id to its teacher rows (features built here
    through ``index`` and ``store`` by the task's active framing); ``minus_opus`` maps a bench
    task name to the task rebuilt without Opus. ``teacher_hashes`` / ``minus_opus_hashes`` are
    the two snapshots' ``(labels_sha256, labels_content_sha256)`` the producer checks against
    the M4 pins, ``teacher_corpus_sha256`` the corpus hash when a corpus directory was given
    (compared with the pin only then); the producer's own deviations make the run unregistered
    whatever the caller says."""
    if registered and deviations:
        raise CellError("a run with deviations cannot be the registered one")
    grid = grid if grid is not None else default_grid("rank_fit")
    own = _own_deviations(
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        fingerprints=fingerprints,
        scorers=scorers,
        tasks=tasks,
        grid=grid,
        teacher_hashes=teacher_hashes,
        minus_opus_hashes=minus_opus_hashes,
        arms=arms,
        B=B,
        seed=seed,
        teacher_corpus_sha256=teacher_corpus_sha256,
    )
    if own:
        registered, deviations = False, [*deviations, *(d for d in own if d not in deviations)]
    if registered:
        for task_name, task in tasks.items():
            reg.check_task(task_name, task)
    cells: dict[str, BenchCell] = {}
    decisions: dict[str, Any] = {}
    for task in tasks.values():
        if task.task_id not in RANK_FIT_TASKS:
            continue
        fb = feature_builder(task, index, store, scorers)
        rows = teacher.get(task.task_id)
        if rows is not None and len(rows):
            rows = rows.with_builder(
                feature_builder(
                    rows.as_task(), index, store, scorers, active_framing_id(task.task_id)
                )
            )
        task_cells, decision = task_x4_cells(
            task,
            fb,
            rows,
            fingerprints,
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            registered=registered,
            minus=None if minus_opus is None else minus_opus.get(task.name),
            grid=grid,
            arms=arms,
            B=B,
            seed=seed,
            clock=clock,
        )
        cells.update(task_cells)
        decisions[task.name] = decision
    notes = [
        "X4 teacher ablation (plan §5.6 X4; design/m4-analysis-plan.md R3): the X3 nested probe "
        "three times on identical outer folds with the teacher rows in every training set at "
        "(teacher, teacher_implicit) weights off, (0.5, 0.3), (0.3, 0.1); teacher rows of the "
        "held-out card's product dropped from that fold; never in a test fold. Scored on silver "
        "and on silver-minus-Opus (the same predictions on the surviving items). Every cell is a "
        "variant, never citable; teacher: true on the on-arms.",
        "Decision (keep teacher labels in production fits): on ≻ off on novel-key NLL under rule "
        "R for both scorings, for the (0.5, 0.3) arm; the (0.3, 0.1) arm is reported. Expected ≈ "
        "null with few teacher rows in the bench products' registry (stated in advance).",
        *(
            f"{task}: keep={d.get('keep')} {d.get('keep_detail', {})}"
            for task, d in sorted(decisions.items())
        ),
        "Silver labels are four-model agreement, not truth.",
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
