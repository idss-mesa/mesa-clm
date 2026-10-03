"""X2 baselines in the bench cell schema (PR "X2 baselines"; plan §5.6; the analysis plan
``design/m2-analysis-plan.md`` §10, whose item numbers are cited below).

* **No-model controls** (§10.1): majority, ``lookup_prob``, novel-key and leave-one-product-out
  are the M0 cell ``<task>.baseline.lookup_prob`` (:func:`mesa_clm.bench.baselines.baselines_cell`),
  recomputed for every task from the same snapshot.
* **PR #13 replica** (§10.2): ``StandardScaler`` then ``LogisticRegression(C=1,
  max_iter=2000)`` (scikit-learn, the ``bench`` extra), fitted leave-one-card-out on the stored
  float32 4096-d vectors (CLM's ``TextCache``, which PR #13 read, holds float16 casts of them; the
  replica keeps float32) of the joint specs ``joint4096@S1`` (the mesa-anyjev per-candidate state
  with the task question appended) and ``joint4096@S1ns`` (without it), one row per labelled pair,
  for ``term.fits`` and ``column.ontology_fits``. The scaler and the model see the fold's training
  rows only. **Unweighted**: the frozen text names PR #13's recipe and no weights; D20's sample
  weights are for mesa-clm's own fitters. Folds pass the 30/5 guards and the probe floor of 40
  training items.
* **AnyJev L2** (§10.3): the per-item held-out predictions of
  ``bench/baselines/anyjev_l2_2026-09-29.json`` (``scripts/anyjev_l2_predictions.py``), joined to
  the bench items by the D1 identity ``(task_key, target_sha256, option_key)`` derived from each
  dump item's ``state_json``. A duplicate identity or a matched item held out on another card is
  refused; an unmatched bench item is left out (counted and listed, never imputed) and an
  unmatched dump item dropped (counted), so the cell, and the K2(a) comparison built on its
  ``items``, is paired by identity.

Every cell here is ``tier: baseline``, never ``servable`` and keyed ``<task>.baseline.<framing>``
(``pr13@S1``, ``pr13@S1ns``, ``anyjev_l2``); :func:`mesa_clm.bench.cells.assemble_cell` fills the
schema, so the lookup controls, the novel-key blocks and ``beats_lookup_novel`` are computed the
same way as for a model tier (reported; X2 gates nothing).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.baselines import baselines_cell
from mesa_clm.bench.cells import (
    PROBE_FLOOR,
    CellError,
    TextIndex,
    assemble_cell,
    item_identities,
    mean_pairwise_cosine,
)
from mesa_clm.bench.results import BenchCell, BenchResults, environment
from mesa_clm.bench.tasks.base import Fold, Task, fold_guard
from mesa_clm.identity import IdentityError, identity
from mesa_clm.learn.features import X2_SPECS, FeatureStore
from mesa_clm.learn.offline import LATEST_MODEL
from mesa_clm.states import state_sha256

__all__ = [
    "ANYJEV_FORMAT",
    "ANYJEV_FRAMING",
    "ANYJEV_TASKS",
    "REPLICA_FRAMING",
    "REPLICA_TASKS",
    "SPEC_VARIANTS",
    "AnyJevDump",
    "AnyJevItem",
    "AnyJevJoin",
    "anyjev_cell",
    "join_anyjev",
    "load_anyjev",
    "replica_cell",
    "replica_pipeline",
    "run_x2",
]

REPLICA_FRAMING: Final[str] = "pr13"
REPLICA_C: Final[float] = 1.0
REPLICA_MAX_ITER: Final[int] = 2000
REPLICA_RECIPE: Final[str] = "StandardScaler + LogisticRegression(C=1, max_iter=2000), unweighted"
REPLICA_TASKS: Final[tuple[str, ...]] = ("term.fits", "column.ontology_fits")
SPEC_VARIANTS: Final[dict[str, str]] = {"joint4096@S1": "S1", "joint4096@S1ns": "S1ns"}
ANYJEV_FORMAT: Final[str] = "mesa-clm/anyjev-l2-predictions/1"
ANYJEV_FRAMING: Final[str] = "anyjev_l2"
ANYJEV_TASKS: Final[dict[str, str]] = {
    "neon_term_fits": "term.fits",
    "neon_ontology_fits": "column.ontology_fits",
}
JOIN_KEY: Final[str] = (
    "(task_key, target_sha256, option_key) derived from each dump item's state_json (D1)"
)
_LISTED: Final[int] = 20

FloatArray = npt.NDArray[np.float64]


# -- the PR #13 replica ----------------------------------------------------------------------------


def replica_pipeline() -> Any:
    """PR #13's recipe, unfitted: ``StandardScaler`` then ``LogisticRegression(C=1,
    max_iter=2000)`` with scikit-learn's other defaults (L2, lbfgs, ``tol=1e-4``)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - the bench extra is part of every dev install
        raise CellError("the PR #13 replica needs scikit-learn: install mesa-clm[bench]") from exc
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=REPLICA_C, max_iter=REPLICA_MAX_ITER)
    )


def _sklearn_version() -> str | None:
    try:
        import sklearn
    except ImportError:  # pragma: no cover
        return None
    return str(sklearn.__version__)


def _replica_texts(task: Task, index: TextIndex, spec: str) -> list[str]:
    return [index.text(task.task_id, spec, t, o, "context") for t, o in item_identities(task)]


def replica_cell(
    task: Task,
    index: TextIndex,
    store: FeatureStore,
    spec: str,
    *,
    fingerprint: Mapping[str, str] | None,
    labels_sha256: str,
    labels_content_sha256: str | None,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchCell:
    """The ``<task>.baseline.pr13@<S1|S1ns>`` cell (§10.2). ``fingerprint`` is the live D5
    bundle; the replica reads encoder vectors and no head, so ``clm_model_fp`` is dropped. A fold
    whose fit does not converge (``max_iter`` reached, a ``ConvergenceWarning``) is used as it
    is and flagged in ``diagnostics.per_fold`` (``n_iter``, ``converged``, ``warnings``), never
    refitted or skipped."""
    if task.task_id not in REPLICA_TASKS:
        raise CellError(f"{task.name}: the PR #13 replica covers {', '.join(REPLICA_TASKS)}")
    if spec not in SPEC_VARIANTS:
        raise CellError(f"unknown joint spec {spec!r}; expected one of {', '.join(X2_SPECS)}")
    texts = _replica_texts(task, index, spec)
    x = np.asarray(store.get(texts), dtype=np.float64)
    y = np.asarray(task.labels, dtype=np.int64)
    pooled: list[int] = []
    parts: list[FloatArray] = []
    evaluated: list[Fold] = []
    skipped: dict[str, str] = {}
    per_fold: dict[str, dict[str, Any]] = {}
    for fold in task.leave_one_card_out():
        reason = fold_guard(task, fold)
        if reason is None and len(fold.train) < PROBE_FLOOR:
            reason = f"below_floor {len(fold.train)} < {PROBE_FLOOR} training items (probe)"
        if reason is not None:
            skipped[fold.held_out] = reason
            continue
        train = np.asarray(fold.train, dtype=np.int64)
        test = np.asarray(fold.test, dtype=np.int64)
        model = replica_pipeline()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x[train], y[train])
        raw = np.asarray(model.predict_proba(x[test]), dtype=np.float64)
        part = np.zeros((len(test), task.k), dtype=np.float64)
        for j, cls in enumerate(model.classes_):
            part[:, int(cls)] = raw[:, j]
        n_iter = int(np.max(model[-1].n_iter_))
        per_fold[fold.held_out] = {
            "n": len(fold.test),
            "n_train": len(fold.train),
            "n_iter": n_iter,
            "converged": n_iter < REPLICA_MAX_ITER,
            "warnings": sorted({type(w.message).__name__ for w in caught}),
        }
        pooled.extend(fold.test)
        parts.append(part)
        evaluated.append(fold)
    distinct = list(dict.fromkeys(texts))
    diagnostics: dict[str, Any] = {
        "recipe": REPLICA_RECIPE,
        "weighted": False,
        "spec": spec,
        "sklearn": _sklearn_version(),
        "mean_state_cos": mean_pairwise_cosine(store.get(distinct)),
        "calls": len(distinct),
        "calls_kind": "/v1/embeddings, one text per request (no /v1/systemone)",
        "input_tokens": int(sum(store.token_counts(distinct))),
        "per_fold": per_fold,
        "task_counts": {"n": len(task.items), "class_counts": task.class_counts()},
    }
    fp = (
        None
        if fingerprint is None
        else {k: v for k, v in fingerprint.items() if k != "clm_model_fp"}
    )
    probs = np.vstack(parts) if parts else np.zeros((0, task.k), dtype=np.float64)
    return assemble_cell(
        task,
        pooled,
        probs,
        evaluated,
        tier="baseline",
        framing=REPLICA_FRAMING,
        variant=SPEC_VARIANTS[spec],
        fingerprint=fp,
        feature_spec=spec,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        selection="none",
        pre_registered=True,
        exploratory=False,
        servable=False,
        skipped_folds=skipped,
        diagnostics=diagnostics,
        notes="PR #13 replica (X2): a baseline, never servable; gates nothing.",
        B=B,
        seed=seed,
    )


# -- AnyJev L2 -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnyJevItem:
    """One held-out AnyJev L2 prediction with its D1 identity (derived from its ``state_json``)."""

    task_id: str
    task_key: str
    target_sha256: str
    option_key: str
    card: str
    fold: str
    state_sha256: str
    label: int
    p_yes: float


@dataclass(frozen=True)
class AnyJevDump:
    """A loaded ``mesa-clm/anyjev-l2-predictions/1`` file: items per mesa-clm task id."""

    path: str
    sha256: str
    meta: dict[str, Any]
    items: dict[str, list[AnyJevItem]]
    question_keys: dict[str, str] = field(default_factory=dict)


def load_anyjev(path: str | Path) -> AnyJevDump:
    """Read the AnyJev L2 dump (§10.3). :class:`CellError` for another format, an unknown
    task, an item without ``state_json`` (a ``--no-states`` dump cannot be joined), a state
    without a D1 identity, or an item without a probability."""
    file = Path(path).expanduser()
    raw = file.read_bytes()
    data = json.loads(raw)
    if data.get("format") != ANYJEV_FORMAT:
        raise CellError(f"{file}: format {data.get('format')!r}, expected {ANYJEV_FORMAT!r}")
    items: dict[str, list[AnyJevItem]] = {}
    keys: dict[str, str] = {}
    for name, block in dict(data.get("tasks") or {}).items():
        task_id = ANYJEV_TASKS.get(str(name))
        if task_id is None:
            raise CellError(
                f"{file}: unknown AnyJev task {name!r}; expected {sorted(ANYJEV_TASKS)}"
            )
        keys[task_id] = str(block.get("question_key") or "")
        out: list[AnyJevItem] = []
        for n, it in enumerate(block.get("items") or []):
            state = it.get("state_json")
            if not isinstance(state, dict):
                raise CellError(f"{file}: {name} item {n} has no state_json to derive its identity")
            try:
                ident = identity(task_id, state)
            except IdentityError as exc:
                raise CellError(f"{file}: {name} item {n}: {exc}") from exc
            p_yes = it.get("p_yes")
            if p_yes is None or not 0.0 <= float(p_yes) <= 1.0:
                raise CellError(f"{file}: {name} item {n} has no probability in [0, 1]")
            out.append(
                AnyJevItem(
                    task_id=task_id,
                    task_key=ident.task_key,
                    target_sha256=ident.target_sha256,
                    option_key=ident.option_key,
                    card=str(it.get("card") or ""),
                    fold=str(it.get("fold") or it.get("card") or ""),
                    state_sha256=str(it.get("state_sha256") or ""),
                    label=int(it.get("label", -1)),
                    p_yes=float(p_yes),
                )
            )
        items[task_id] = out
    meta = {
        k: data.get(k)
        for k in ("format", "mesa_anyjev", "level", "questions_lock_sha", "labels_file_sha256")
    }
    return AnyJevDump(str(file), hashlib.sha256(raw).hexdigest(), meta, items, keys)


@dataclass(frozen=True)
class AnyJevJoin:
    """The bench items that have an AnyJev prediction (``idx``, item order) with ``p_yes``
    parallel to them, and the join report (§10.3)."""

    idx: list[int]
    p_yes: FloatArray
    report: dict[str, Any]


def join_anyjev(task: Task, items: Sequence[AnyJevItem]) -> AnyJevJoin:
    """Join AnyJev predictions to ``task``'s items by D1 identity (§10.3). :class:`CellError`
    for a dump item of another task key, a duplicate identity, or a matched item whose held-out
    card is not the bench item's card."""
    by_key: dict[tuple[str, str], AnyJevItem] = {}
    for it in items:
        if it.task_key != task.spec.key:
            raise CellError(
                f"{task.name}: a dump item has task_key {it.task_key}, not {task.spec.key}"
            )
        key = (it.target_sha256, it.option_key)
        if key in by_key:
            raise CellError(f"{task.name}: the dump has identity {key[0][:12]}/{key[1]} twice")
        by_key[key] = it
    idx: list[int] = []
    p_yes: list[float] = []
    unmatched: list[tuple[str, str]] = []
    used: set[tuple[str, str]] = set()
    state_differs = label_differs = 0
    for i, key in enumerate(item_identities(task)):
        hit = by_key.get(key)
        if hit is None:
            unmatched.append(key)
            continue
        card = task.cards[i]
        if hit.card != card or hit.fold != card:
            raise CellError(
                f"{task.name}: identity {key[0][:12]}/{key[1]} was held out on {hit.fold!r} by "
                f"AnyJev and belongs to {card!r} here; the folds are not the same"
            )
        state_differs += hit.state_sha256 != state_sha256(task.items[i][0])
        label_differs += hit.label != task.items[i][1]
        idx.append(i)
        p_yes.append(hit.p_yes)
        used.add(key)
    report = {
        "key": JOIN_KEY,
        "bench_items": len(task.items),
        "dump_items": len(items),
        "matched": len(idx),
        "unmatched_bench": len(unmatched),
        "unmatched_bench_identities": [f"{t[:12]}/{o}" for t, o in unmatched[:_LISTED]],
        "unmatched_dump": len(by_key) - len(used),
        "state_sha256_differs": state_differs,
        "label_differs": label_differs,
        "full_set": not unmatched,
    }
    return AnyJevJoin(idx, np.asarray(p_yes, dtype=np.float64), report)


def anyjev_cell(
    task: Task,
    join: AnyJevJoin,
    dump: AnyJevDump,
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchCell:
    """The ``<task>.baseline.anyjev_l2`` cell over the joined items (§10.3): ``[p_yes,
    1 − p_yes]`` pooled over AnyJev's folds (the bench cards); the lookup controls see every
    item of the other cards, as AnyJev's training did."""
    matched = set(join.idx)
    folds: list[Fold] = []
    skipped: dict[str, str] = {}
    for fold in task.leave_one_card_out():
        test = [i for i in fold.test if i in matched]
        if not test:
            skipped[fold.held_out] = "no_anyjev_prediction"
            continue
        folds.append(Fold(fold.held_out, test, fold.train))
    probs = np.stack([join.p_yes, 1.0 - join.p_yes], axis=1)
    return assemble_cell(
        task,
        join.idx,
        probs,
        folds,
        tier="baseline",
        framing=ANYJEV_FRAMING,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        selection="none",
        pre_registered=True,
        exploratory=False,
        servable=False,
        skipped_folds=skipped,
        diagnostics={
            "join": join.report,
            "anyjev": {
                **dump.meta,
                "question_key": dump.question_keys.get(task.task_id),
                "path": dump.path,
                "sha256": dump.sha256,
            },
        },
        notes="AnyJev L2 held-out predictions (X2), paired by D1 identity; never servable.",
        B=B,
        seed=seed,
    )


# -- the results file -------------------------------------------------------------------------------


def run_x2(
    tasks: Mapping[str, Task],
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    date: str,
    index: TextIndex | None = None,
    store: FeatureStore | None = None,
    fingerprint: Mapping[str, str] | None = None,
    anyjev: AnyJevDump | None = None,
    registered: bool,
    deviations: Sequence[str] = (),
    specs: Sequence[str] = X2_SPECS,
    name: str = "x2",
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchResults:
    """The ``x2`` results file (§12.4, §10): the M0 control for every task; the replica cells
    when ``index`` and ``store`` are given; the AnyJev cells when ``anyjev`` is. ``registered``
    (with the ``deviations`` that made it false) stamps every cell (§13): an unregistered run's
    cells are ``pre_registered: false`` and ``exploratory: true``. Labels or bootstrap settings
    other than the registration's (:func:`mesa_clm.bench.registered.run_deviations`), a run
    without the replica or with another encoder or serving lock in ``fingerprint`` (``clm-latest``'s
    bundle; the replica reads no head, so its ``clm_model_fp`` is not compared), a missing AnyJev
    dump or one whose sha256 is not the registered one
    (:func:`mesa_clm.bench.registered.anyjev_deviations`), or a subset of the registered tasks
    make the run unregistered whatever the caller says, and a registered run's tasks must have
    their published counts (§1.3)."""
    if registered and deviations:
        raise CellError("a run with deviations cannot be the registered one")
    own = [
        *reg.run_deviations(
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            B=B,
            seed=seed,
        ),
        *(
            ["no PR #13 replica (no manifest index or feature store)"]
            if index is None or store is None
            else reg.identity_deviations(
                framings_lock_sha=None,
                fingerprints=None if fingerprint is None else {LATEST_MODEL: fingerprint},
                models=[LATEST_MODEL],
                ignore=("clm_model_fp",),
            )
        ),
        *reg.anyjev_deviations(None if anyjev is None else anyjev.sha256),
        *(
            []
            if tuple(specs) == tuple(X2_SPECS)
            else [f"specs {list(specs)} are not {list(X2_SPECS)}"]
        ),
        *reg.task_set_deviations(list(tasks)),
    ]
    if own:
        registered, deviations = False, [*deviations, *(d for d in own if d not in deviations)]
    if registered:
        for task_name, task in tasks.items():
            reg.check_task(task_name, task)
    cells: dict[str, BenchCell] = {}
    for task in tasks.values():
        base = baselines_cell(
            task,
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            B=B,
            seed=seed,
        )
        cells[base.key] = base
        if task.task_id in REPLICA_TASKS and index is not None and store is not None:
            for spec in specs:
                cell = replica_cell(
                    task,
                    index,
                    store,
                    spec,
                    fingerprint=fingerprint,
                    labels_sha256=labels_sha256,
                    labels_content_sha256=labels_content_sha256,
                    B=B,
                    seed=seed,
                )
                cells[cell.key] = cell
        if anyjev is not None and task.task_id in anyjev.items:
            join = join_anyjev(task, anyjev.items[task.task_id])
            cell = anyjev_cell(
                task,
                join,
                anyjev,
                labels_sha256=labels_sha256,
                labels_content_sha256=labels_content_sha256,
                B=B,
                seed=seed,
            )
            cells[cell.key] = cell
    notes = [
        "X2 baselines (PR X2, plan §5.6): the M0 no-model controls, the PR #13 replica "
        f"({REPLICA_RECIPE}) on joint4096@S1/@S1ns, and AnyJev L2's held-out predictions "
        "joined by D1 identity; design/m2-analysis-plan.md §10.",
        "Every X2 cell is a baseline: never servable, never cited.",
    ]
    if not registered:
        cells = {
            k: c.model_copy(update={"pre_registered": False, "exploratory": True})
            for k, c in cells.items()
        }
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
        environment={**environment(), "sklearn": _sklearn_version()},
        cells=cells,
        notes=notes,
    )
