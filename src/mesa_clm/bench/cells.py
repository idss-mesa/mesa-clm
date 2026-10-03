"""Leave-one-card-out cells of the ``zero_shot`` and ``calibrated`` tiers for every task, scored
offline from the feature store (plan §5.3, §5.4, §8 M2; DESIGN D6, D7, D20, D27, D30; the
analysis plan ``design/m2-analysis-plan.md`` §9, whose item numbers are cited below).

**Scores** (§9.1). A cell scores exactly the texts ``features build`` embedded: a
:class:`TextIndex` over :func:`mesa_clm.learn.features.manifest` rows maps each item's D1
identity to its context and option texts, and :func:`score_arm` reads their vectors from the
float32 store (``offline.store_vectors``: raw for ``clm-raw``, the cached head projections
otherwise) with :mod:`mesa_clm.learn.offline`'s maths for one *arm* (a framing and a served
model): ``s_c = scale · (zs·zc − zs·za)`` per (target, candidate) for ``term.fits`` and
``column.ontology_fits`` (set-independent, D2), and the ``K`` option logits ``scale · zs·z_k`` of
a closed choice. A text without a vector (a truncated one has none) stops the cell with the
store's :class:`~mesa_clm.learn.features.FeatureMissing`, never a silent drop.

**Tiers** (§9.2–§9.4). ``zero_shot`` is ``[σ(s_c), 1 − σ(s_c)]`` or CLM's softmax over the
options; it fits nothing, so it evaluates every fold. ``calibrated`` fits, per fold and on the
fold's training cards only, the plan §5.3 calibrator (:func:`mesa_clm.learn.calibrate
.fit_calibrator`: weighted Platt on ``s_c``, Platt on the logit difference at K = 2, temperature
at K > 2) with the label weights as sample weights (D20), and skips a fold that fails the 30/5
guards (two-class tasks) or has fewer than 100 training items (the calibrated floor: no
calibrator is ever fitted on fewer). Predictions are pooled, never fold-averaged. Every
calibrated cell also reports, report-only, the same folds with an unweighted calibrator (§6.6).

**Selection** (§9.5). The cell code never chooses an arm. For the rank_fit tasks the caller
passes X1's outcome (:class:`X1Selection`, from ``framing.selections_from_json``: the full-data
A1 arm and the nested per-fold arms that ``bench/framing.py`` chose by grouped inner CV on each
fold's six training cards). The *nested* cell ``<task>.<tier>.<A1 framing>`` scores fold *c*
with fold *c*'s arm (``selection: "nested"``); these are the only citable-form rank_fit cells of
M2 and this module is their one producer. The *full-data* cell ``…@full`` uses A1 in every fold
(``exploratory: true``, D27). Without an A1 (X1's K1 or undecidable outcome) the production
default arm F7 / ``clm-latest`` is benched for audit only (``selection: "none"``,
``pre_registered: false``). The closed choices have one framing, F7: ``<task>.<tier>.F7`` is
``clm-latest`` (``selection: "none"``: nothing was chosen), other models are reported as
``…@<model>`` and are not pre-registered. A run that is not the registered one (``registered``
false) stamps every cell ``pre_registered: false``, ``exploratory: true``.

**Cell fields** (§9.6–§9.9): :func:`assemble_cell` fills the plan §5.4 schema of
:class:`~mesa_clm.bench.results.BenchCell` for any pooled predictions (``bench/x2.py`` and X1's
arm cells use it too): pooled metrics (:func:`mesa_clm.bench.baselines.pooled_metrics`), the
lookup controls on the cell's own folds (:func:`~mesa_clm.bench.baselines.evaluate_lookup`), the
cell's novel-key block next to ``lookup_prob``'s, ``beats_lookup_novel`` (novel-key AUROC
cluster lower bound > 0.5 and rule R on NLL), the per-item predictions and the diagnostics.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from mesa_clm import framings
from mesa_clm.avu import pre_rule_value_kind
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.baselines import (
    LookupEval,
    baselines_block,
    evaluate_lookup,
    guard_skipped_folds,
    novel_key_block,
    pooled_metrics,
)
from mesa_clm.bench.results import (
    BenchCell,
    BenchResults,
    CellCounts,
    CellItem,
    NovelKey,
    Selection,
    Tier,
    environment,
    finite,
)
from mesa_clm.bench.tasks.base import Fold, Task, fold_guard
from mesa_clm.clm.headproj import Side
from mesa_clm.identity import target_sha256
from mesa_clm.learn import calibrate
from mesa_clm.learn.features import FeatureStore, Manifest, ManifestRow
from mesa_clm.learn.offline import (
    LATEST_MODEL,
    RAW_MODEL,
    OfflineScorer,
    pairwise_s_c,
    store_vectors,
)
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import RANK_FIT_TASKS

__all__ = [
    "CALIBRATED_FLOOR",
    "DEFAULT_ARM",
    "FOLD_CHOICE_KEYS",
    "FULL_VARIANT",
    "PROBE_FLOOR",
    "SERVED_MODELS",
    "TIERS",
    "Arm",
    "ArmScores",
    "CellError",
    "FoldChoice",
    "TextIndex",
    "X1Selection",
    "assemble_cell",
    "beats_lookup_novel",
    "item_identities",
    "mean_pairwise_cosine",
    "run_tier_cells",
    "score_arm",
    "task_cells",
    "tier_cell",
]

TierName = Literal["zero_shot", "calibrated"]
TIERS: Final[tuple[TierName, ...]] = ("zero_shot", "calibrated")
# PR "LOCO folds": floors 100 (calibrated) / 40 (probe/head): the fewest training items any
# calibrator (probe) is fitted on (design/m2-analysis-plan.md §6.3).
CALIBRATED_FLOOR: Final[int] = 100
PROBE_FLOOR: Final[int] = 40
SERVED_MODELS: Final[frozenset[str]] = frozenset({LATEST_MODEL, RAW_MODEL})
FULL_VARIANT: Final[str] = "full"
CHOICE_FRAMING: Final[str] = "F7"
TIMING_SOURCE: Final[str] = "offline_replay"
TEACHER_SOURCES: Final[frozenset[str]] = frozenset({"teacher", "teacher_implicit"})
BEATS_RULE: Final[str] = (
    "novel-key AUROC cluster lower bound > 0.5 and novel-key NLL beats lookup_prob under rule R"
)
_SUM_TOL: Final[float] = 1e-6

FloatArray = npt.NDArray[np.float64]
Clock = Callable[[], float]


class CellError(ValueError):
    """Inputs a cell cannot honestly be computed from: an item the manifest does not know, a
    text without a vector, a control framing, an arm that was not scored, a teacher label in a
    pooled item, predictions that are not distributions."""


# -- arms and X1's outcome --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Arm:
    """A (framing, served model) pair: what X1 chooses and what a cell's scores come from."""

    framing: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.framing}@{self.model}"

    @classmethod
    def of(cls, value: Arm | Mapping[str, Any]) -> Arm:
        """An arm from itself or a ``{"framing", "model"}`` mapping."""
        if isinstance(value, Arm):
            return value
        return cls(str(value["framing"]), str(value["model"]))


DEFAULT_ARM: Final[Arm] = Arm(framings.ACTIVE["term.fits"], LATEST_MODEL)


@dataclass(frozen=True)
class FoldChoice:
    """One outer fold's arm as X1's grouped inner CV chose it on the fold's six training cards,
    with the selector's own record (JSON-serialisable; copied into ``fold_choices``)."""

    arm: Arm
    record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, value: FoldChoice | Arm | Mapping[str, Any]) -> FoldChoice:
        """From itself, an :class:`Arm`, or a mapping with ``framing``, ``model`` and any other
        keys (the selector's record)."""
        if isinstance(value, FoldChoice):
            return value
        if isinstance(value, Arm):
            return cls(value)
        rest = {k: v for k, v in value.items() if k not in ("framing", "model")}
        return cls(Arm.of(value), rest)


@dataclass(frozen=True)
class X1Selection:
    """X1's outcome for one rank_fit task (``bench/framing.py``): ``a1`` the arm chosen on all
    seven cards (``None`` under K1), ``fold_choices`` the nested arm per held-out card,
    ``fold_skips`` the held-out cards whose inner selection chose no arm, with its outcome
    (``inner_k1``, ``inner_undecidable``), and ``note`` what the selector wants recorded (the
    K1 verdict, the decision record's path)."""

    a1: Arm | None
    fold_choices: Mapping[str, FoldChoice] = field(default_factory=dict)
    fold_skips: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        both = set(self.fold_choices) & set(self.fold_skips)
        if both:
            raise CellError(f"a fold has both an arm and a skip: {sorted(both)}")

    @classmethod
    def of(
        cls,
        a1: Arm | Mapping[str, Any] | None,
        fold_choices: Mapping[str, FoldChoice | Arm | Mapping[str, Any]] | None = None,
        fold_skips: Mapping[str, str] | None = None,
        note: str = "",
    ) -> X1Selection:
        return cls(
            None if a1 is None else Arm.of(a1),
            {str(card): FoldChoice.of(v) for card, v in (fold_choices or {}).items()},
            {str(card): str(why) for card, why in (fold_skips or {}).items()},
            note,
        )


# -- texts ----------------------------------------------------------------------------------------

TextKey = tuple[str, str, str, str, str]


class TextIndex:
    """Manifest rows by ``(task_id, framing_id, target_sha256, option_key, role)``, plus the
    closed options of each (task, framing, target) in wire order. The same key twice must carry
    the same text (two manifests of one snapshot always agree)."""

    def __init__(self, rows: Iterable[ManifestRow]) -> None:
        self._text: dict[TextKey, str] = {}
        self._options: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
        for r in rows:
            key = (r.task_id, r.framing_id, r.target_sha256, r.option_key, r.role)
            known = self._text.get(key)
            if known is not None:
                if known != r.text:
                    raise CellError(f"two manifest rows disagree on {key[0]}/{key[1]} {key[4]}")
                continue
            self._text[key] = r.text
            if r.role == "option":
                self._options.setdefault(key[:3], []).append((r.option_key, r.text))

    @classmethod
    def from_manifests(cls, *manifests: Manifest) -> TextIndex:
        """One index over several manifests of the same snapshot (labels_sha256 must agree)."""
        shas = {m.labels_sha256 for m in manifests}
        if len(shas) > 1:
            raise CellError("the manifests come from different snapshots")
        return cls(r for m in manifests for r in m.rows)

    def __len__(self) -> int:
        return len(self._text)

    def text(self, task_id: str, framing_id: str, target: str, option: str, role: str) -> str:
        try:
            return self._text[(task_id, framing_id, target, option, role)]
        except KeyError:
            raise CellError(
                f"{task_id}/{framing_id}: the manifest has no {role} text for target "
                f"{target[:12]} option {option!r}; build it from the same snapshot"
            ) from None

    def options(self, task_id: str, framing_id: str, target: str) -> list[tuple[str, str]]:
        """``[(wire key, option text)]`` of a closed choice's target, in wire order."""
        try:
            return list(self._options[(task_id, framing_id, target)])
        except KeyError:
            raise CellError(
                f"{task_id}/{framing_id}: the manifest has no options for target {target[:12]}"
            ) from None


def item_identities(task: Task) -> list[tuple[str, str]]:
    """``(target_sha256, option_key)`` of every item (D1; ``option_key`` ``''`` for a choice)."""
    opts = task.option_keys or [""] * len(task.items)
    return [
        (target_sha256(task.task_id, state), str(opt))
        for (state, _), opt in zip(task.items, opts, strict=True)
    ]


# -- scoring ---------------------------------------------------------------------------------------


def mean_pairwise_cosine(z: npt.ArrayLike) -> float | None:
    """Mean cosine over distinct pairs of the unit rows of ``z`` (the collapse diagnostic);
    ``None`` below two rows."""
    x = np.asarray(z, dtype=np.float64)
    m = x.shape[0]
    if m < 2:
        return None
    norms = np.linalg.norm(x, axis=1)
    u = x / np.maximum(norms, 1e-12)[:, None]
    gram = u @ u.T
    return float((gram.sum() - np.trace(gram)) / (m * (m - 1)))


@dataclass(frozen=True)
class ArmScores:
    """One arm's scores on every item of a task, in item order: ``x`` is ``s_c`` ``[n]``
    (rank_fit) or the option logits ``[n, K]`` (closed choice)."""

    task_id: str
    arm: Arm
    shape: calibrate.Shape
    x: FloatArray
    question_key: str
    diagnostics: dict[str, Any]

    def zero_shot(self, idx: Sequence[int]) -> FloatArray:
        """The zero-shot distribution of items ``idx`` (§9.2, §9.3)."""
        return calibrate.zero_shot_probs(self.shape, self.x[np.asarray(idx, dtype=np.int64)])


def _unique_vectors(
    scorer: OfflineScorer, store: FeatureStore, side: Side, texts: Sequence[str]
) -> tuple[dict[str, int], npt.NDArray[np.float32]]:
    order = list(dict.fromkeys(texts))
    if scorer.projector is not None and scorer.clm_model_fp is None:
        raise CellError(f"{scorer.model}: reading cached projections needs its clm_model_fp")
    vectors = store_vectors(scorer, store, side, order)
    return {t: i for i, t in enumerate(order)}, vectors


def score_arm(
    task: Task,
    index: TextIndex,
    store: FeatureStore,
    scorer: OfflineScorer,
    framing_id: str,
    *,
    clock: Clock = time.perf_counter,
) -> ArmScores:
    """Score every item of ``task`` under ``framing_id`` and ``scorer``'s model (§9.1).
    :class:`CellError` for a control framing (F1 is X1's noul control, never a tier arm) or an
    item the index does not know; :class:`~mesa_clm.learn.features.FeatureMissing` for a text
    without a vector (a truncated one included)."""
    f = framings.framing(task.task_id, framing_id)
    if f.control:
        raise CellError(f"{task.task_id}/{f.id}: a control framing is not a tier arm")
    if not task.items:
        raise CellError(f"{task.name}: no items to score")
    arm = Arm(f.id, scorer.model)
    ids = item_identities(task)
    started = clock()
    contexts = [index.text(task.task_id, f.id, t, "", "context") for t, _ in ids]
    if f.shape == "rank_fit":
        cands = [index.text(task.task_id, f.id, t, o, "candidate") for t, o in ids]
        anchors = [index.text(task.task_id, f.id, t, ANCHOR_KEY, "anchor") for t, _ in ids]
        states, z_s = _unique_vectors(scorer, store, "state", contexts)
        actions, z_a = _unique_vectors(scorer, store, "action", [*cands, *anchors])
        x = pairwise_s_c(
            scorer.scale,
            z_s[[states[t] for t in contexts]],
            z_a[[actions[t] for t in cands]],
            z_a[[actions[t] for t in anchors]],
        )
    else:
        wire = list(f.closed_options or {})
        options = [index.options(task.task_id, f.id, t) for t, _ in ids]
        for opts in options:
            if [k for k, _ in opts] != wire:
                raise CellError(f"{task.task_id}/{f.id}: the options are not the closed options")
        states, z_s = _unique_vectors(scorer, store, "state", contexts)
        actions, z_a = _unique_vectors(scorer, store, "action", [t for o in options for _, t in o])
        x = np.stack(
            [
                scorer.logits(z_s[states[c]], z_a[[actions[t] for _, t in opts]])
                for c, opts in zip(contexts, options, strict=True)
            ]
        ).astype(np.float64)
    elapsed = clock() - started
    if not np.all(np.isfinite(x)):
        raise CellError(f"{task.task_id}/{arm.name}: a score is not finite")
    n = len(ids)
    tokens = store.token_counts([*states, *actions])
    return ArmScores(
        task_id=task.task_id,
        arm=arm,
        shape="rank_fit" if f.shape == "rank_fit" else "choice",
        x=np.asarray(x, dtype=np.float64),
        question_key=f.question_key,
        diagnostics={
            "arm": arm.name,
            "n_items": n,
            "n_targets": len({t for t, _ in ids}),
            "mean_state_cos": mean_pairwise_cosine(z_s),
            # One /v1/systemone Choice per target (its labelled candidates, or the closed options).
            "calls": len({t for t, _ in ids}),
            "input_tokens": int(sum(tokens)),
            "ms_per_decision": round(1000.0 * elapsed / max(n, 1), 4),
            "timing_source": TIMING_SOURCE,
        },
    )


# -- the cell -------------------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """``value`` with every float made finite or ``None`` and numpy scalars made plain."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return finite(value)
    return value


def beats_lookup_novel(
    task: Task,
    ev: LookupEval,
    probs: FloatArray,
    mine: NovelKey,
    *,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> tuple[bool, dict[str, Any]]:
    """PR "Lookup as a probability model": the cell's novel-key AUROC cluster lower bound is
    above 0.5 **and** the cell beats ``lookup_prob`` on novel-key NLL under rule R (§9.9).
    ``ev`` is the lookup evaluation aligned with ``probs``; ``mine`` the cell's novel-key block.
    Returns the verdict and its record (both conditions, the rule R record, the reason)."""
    novel = ev.novel
    detail: dict[str, Any] = {"rule": BEATS_RULE, "n_novel": int(novel.sum())}
    if not novel.any():
        detail["reason"] = "no_novel_items"
        return False, detail
    cards = [c for c, m in zip(ev.cards, novel, strict=True) if m]
    detail["n_novel_clusters"] = len(set(cards))
    if len(set(cards)) < 2:
        detail["reason"] = "fewer_than_two_novel_clusters"
        return False, detail
    rr = stats.rule_r("nll", probs[novel], ev.probs[novel], ev.labels[novel], cards, B=B, seed=seed)
    detail["nll_rule_r"] = _json_safe(rr.as_dict())
    if not task.binary:
        detail["auroc_condition"] = None
        detail["reason"] = f"auroc_not_applicable_k{task.k}"
        return False, detail
    lower = mine.auroc_ci.lower if mine.auroc_ci is not None else None
    detail["auroc_lower"] = lower
    detail["auroc_condition"] = lower is not None and lower > 0.5
    if lower is None:
        detail["reason"] = "auroc_undefined_on_novel_items"  # one class among them
    elif not detail["auroc_condition"]:
        detail["reason"] = "auroc_lower_bound_not_above_0.5"
    elif not rr.passed:
        detail["reason"] = f"nll_{rr.reason}"
    else:
        detail["reason"] = "passed"
    return detail["reason"] == "passed", detail


def assemble_cell(
    task: Task,
    idx: Sequence[int],
    probs: npt.ArrayLike,
    folds: Sequence[Fold],
    *,
    tier: Tier,
    framing: str,
    labels_sha256: str,
    labels_content_sha256: str | None,
    selection: Selection,
    pre_registered: bool,
    exploratory: bool,
    servable: bool,
    variant: str | None = None,
    model: str | None = None,
    question_key: str | None = None,
    fingerprint: Mapping[str, str] | None = None,
    feature_spec: str | None = None,
    skipped_folds: Mapping[str, str] | None = None,
    fold_choices: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    notes: str = "",
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchCell:
    """The plan §5.4 cell of pooled held-out predictions (§9.6–§9.9). ``idx`` are the pooled
    item indices and ``probs`` their ``[n, K]`` distributions (parallel, any order); ``folds``
    are the evaluated folds, whose ``test`` lists together are exactly ``idx`` and whose
    ``train`` lists are what the lookup controls may use. A cell with no pooled item has
    ``metrics`` and ``baselines`` ``None``."""
    sources = task.meta.get("label_sources", {}) or {}
    if any(s in TEACHER_SOURCES for s in sources):
        raise CellError(
            f"{task.name}: a teacher label is among the items (D19: never in a test fold)"
        )
    order = np.argsort(np.asarray(idx, dtype=np.int64), kind="stable")
    pooled = np.asarray(idx, dtype=np.int64)[order]
    p = np.asarray(probs, dtype=np.float64).reshape(len(pooled), task.k)[order]
    if len(set(pooled.tolist())) != len(pooled):
        raise CellError(f"{task.name}: an item is pooled twice")
    tested = sorted(i for fold in folds for i in fold.test)
    if tested != pooled.tolist():
        raise CellError(f"{task.name}: the folds' held-out items are not the pooled items")
    if len(pooled) and (
        not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(np.abs(p.sum(axis=1) - 1) > _SUM_TOL)
    ):
        raise CellError(f"{task.name}: the predictions are not probability distributions")
    sel = pooled.tolist()
    metrics = baselines = None
    items: list[CellItem] = []
    if sel:
        labels = np.asarray(task.labels, dtype=np.int64)[pooled]
        cards = [task.cards[i] for i in sel]
        metrics = pooled_metrics(task, p, labels, cards, B=B, seed=seed)
        ev = evaluate_lookup(task, folds)
        if ev.idx.tolist() != sel:
            raise CellError(f"{task.name}: the lookup evaluation is not aligned with the pool")
        base = baselines_block(task, ev, B=B, seed=seed)
        mine = novel_key_block(dataclasses.replace(ev, probs=p), binary=task.binary, B=B, seed=seed)
        beats, detail = beats_lookup_novel(task, ev, p, mine, B=B, seed=seed)
        baselines = base.model_copy(
            update={
                "novel_key": mine,
                "novel_key_lookup": base.novel_key,
                "beats_lookup_novel": beats,
                "beats_detail": _json_safe(detail),
            }
        )
        ids = item_identities(task)
        items = [
            CellItem(
                target_sha256=ids[i][0],
                option_key=ids[i][1],
                card=task.cards[i],
                label=int(labels[j]),
                probs=[float(v) for v in p[j]],
                novel=bool(ev.novel[j]),
            )
            for j, i in enumerate(sel)
        ]
    return BenchCell(
        task=task.name,
        task_id=task.task_id,
        task_key=task.spec.key,
        tier=tier,
        framing=framing,
        question_key=question_key,
        fingerprint=None if fingerprint is None else dict(fingerprint),
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        feature_spec=feature_spec,
        label_sources=dict(sources),
        min_weight=float(task.meta.get("min_weight", 0.0)),
        teacher=False,
        teacher_in_test=False,
        masked=bool(task.meta.get("masked", False)),
        mask=task.meta.get("mask"),
        loco=True,
        selection=selection,
        pre_registered=pre_registered,
        exploratory=exploratory,
        servable=servable,
        n_folds=len(folds),
        skipped_folds=dict(skipped_folds or {}),
        guard_skipped_folds=guard_skipped_folds(task),
        fold_choices=_json_safe(dict(fold_choices or {})),
        counts=CellCounts(
            n=len(sel),
            n_neg=task.n_neg(sel),
            n_nonmodal=task.n_nonmodal(sel),
            class_counts=task.class_counts(sel),
        ),
        metrics=metrics,
        baselines=baselines,
        diagnostics=_json_safe(dict(diagnostics or {})),
        notes=notes,
        model=model,
        variant=variant,
        items=items,
    )


# -- tier cells --------------------------------------------------------------------------------------


def _calibration(task: Task) -> str:
    """The calibration a calibrated cell of ``task`` applies (plan §5.3)."""
    return "platt" if task.task_id in RANK_FIT_TASKS or task.k == 2 else "temperature"


def _pre_rule_count(task: Task) -> int | None:
    """Items serving would answer by ``pre_rule_value_kind`` before asking CLM (§9.12)."""
    if task.task_id != "avu.value_kind":
        return None
    return sum(
        1 for state, _ in task.items if pre_rule_value_kind(str(state.get("aspect")), "column")
    )


# The keys a ``fold_choices`` entry of a nested cell sets itself; the selector's record (X1's
# inner outcome and the JSON pointer of its trace, §9.5) is kept beside them and may not use one.
FOLD_CHOICE_KEYS: Final[tuple[str, ...]] = (
    "decision",
    "framing",
    "model",
    "question_key",
    "clm_model_fp",
    "evaluated",
)


def _decision(choice: FoldChoice, task: Task, fps: Mapping[str, str]) -> dict[str, Any]:
    """A fold's ``fold_choices`` entry: its arm, that arm's ``question_key`` and head, and the
    selector's record (X1's ``outcome`` and ``x1_trace``, the JSON pointer of the inner trace)
    as it was given. :class:`CellError` for a record key the entry sets itself, which would be
    overwritten silently."""
    clash = sorted(set(choice.record) & set(FOLD_CHOICE_KEYS))
    if clash:
        raise CellError(
            f"{task.name}: the selector's fold record uses the cell's own key(s) {clash}"
        )
    return {
        **dict(choice.record),
        "decision": "arm",
        "framing": choice.arm.framing,
        "model": choice.arm.model,
        "question_key": framings.framing(task.task_id, choice.arm.framing).question_key,
        "clm_model_fp": fps.get(choice.arm.model),
        "evaluated": True,
    }


def _no_arm(why: str) -> dict[str, Any]:
    return {"decision": why, "framing": None, "model": None, "evaluated": False}


def tier_cell(
    task: Task,
    scores: Mapping[Arm, ArmScores],
    tier: TierName,
    *,
    arm: Arm,
    fingerprint: Mapping[str, str] | None,
    labels_sha256: str,
    labels_content_sha256: str | None,
    selection: Selection,
    pre_registered: bool,
    exploratory: bool,
    x1: X1Selection | None = None,
    clm_model_fps: Mapping[str, str] | None = None,
    variant: str | None = None,
    notes: str = "",
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchCell:
    """One ``zero_shot`` or ``calibrated`` cell (§9.2–§9.5). ``arm`` is the cell's identity
    (the served framing and model). With ``x1`` the cell is nested: fold *c* is scored (and, at
    calibrated, fitted) under ``x1.fold_choices[c]``'s arm, and a fold for which X1 chose no arm
    (``x1.fold_skips``, or simply absent) is skipped with that reason, whatever the guards say;
    every outer fold is listed in ``fold_choices`` and is either evaluated or skipped. Without
    ``x1``, ``arm`` scores every fold. ``scores`` must hold every arm used; ``clm_model_fps``
    (model -> fingerprint) names each fold's head. A calibrated cell also carries, report-only,
    the same folds fitted without the label weights (§6.6)."""
    if arm not in scores:
        raise CellError(f"{task.name}: the cell's arm {arm.name} was not scored")
    frame = framings.framing(task.task_id, arm.framing)
    labels = np.asarray(task.labels, dtype=np.int64)
    weights = np.asarray(task.weights or [1.0] * len(task.items), dtype=np.float64)
    fps = dict(clm_model_fps or {})
    pooled: list[int] = []
    parts: list[FloatArray] = []
    unweighted: list[FloatArray] = []
    raw: list[FloatArray] = []
    evaluated: list[Fold] = []
    skipped: dict[str, str] = {}
    per_fold: dict[str, dict[str, Any]] = {}
    chosen: dict[str, dict[str, Any]] = {}
    outer = list(task.leave_one_card_out())
    for fold in outer:
        card = fold.held_out
        fold_arm = arm
        if x1 is not None:
            choice = x1.fold_choices.get(card)
            if choice is None:
                why = x1.fold_skips.get(card, "no_fold_choice")
                skipped[card] = why
                chosen[card] = _no_arm(why)
                continue
            fold_arm = choice.arm
            chosen[card] = _decision(choice, task, fps)
        sc = scores.get(fold_arm)
        if sc is None:
            raise CellError(f"{task.name}: fold {card}'s arm {fold_arm.name} was not scored")
        test = np.asarray(fold.test, dtype=np.int64)
        info: dict[str, Any] = {"n": len(fold.test), "arm": fold_arm.name}
        if tier == "calibrated":
            reason = fold_guard(task, fold)
            if reason is None and len(fold.train) < CALIBRATED_FLOOR:
                reason = (
                    f"below_floor {len(fold.train)} < {CALIBRATED_FLOOR} training items "
                    "(calibrated)"
                )
            if reason is not None:
                skipped[card] = reason
                if card in chosen:
                    chosen[card]["evaluated"] = False
                continue
            train = np.asarray(fold.train, dtype=np.int64)
            cal = calibrate.fit_calibrator(sc.shape, sc.x[train], labels[train], weights[train])
            part = calibrate.apply_calibrator(cal, sc.x[test])
            plain = calibrate.fit_calibrator(sc.shape, sc.x[train], labels[train], None)
            unweighted.append(calibrate.apply_calibrator(plain, sc.x[test]))
            info["n_train"] = len(fold.train)
            info["calibrator"] = cal.model_dump(mode="json")
            if isinstance(cal, calibrate.PlattCalibrator):
                info["inverted"] = cal.inverted
        else:
            part = sc.zero_shot(fold.test)
            if sc.shape == "rank_fit":
                raw.append(sc.x[test])
            elif task.k == 2:
                raw.append(calibrate.logit_difference(sc.x[test]))
        pooled.extend(fold.test)
        parts.append(part)
        evaluated.append(fold)
        per_fold[card] = info
    if len(evaluated) + len(skipped) != len(outer) or (
        x1 is not None and set(chosen) != {f.held_out for f in outer}
    ):
        raise CellError(f"{task.name}: a fold is neither evaluated nor skipped")
    probs = np.vstack(parts) if parts else np.zeros((0, task.k), dtype=np.float64)
    diagnostics: dict[str, Any] = {
        **scores[arm].diagnostics,
        "calibration": "uncalibrated" if tier == "zero_shot" else _calibration(task),
        "per_fold": per_fold,
        "task_counts": {"n": len(task.items), "class_counts": task.class_counts()},
        "excluded": dict(task.meta.get("excluded", {})),
    }
    if tier == "calibrated":
        diagnostics["inverted_folds"] = sorted(c for c, i in per_fold.items() if i.get("inverted"))
        diagnostics["fitters"] = _jsonable(calibrate.FITTER_CONSTANTS)
        if pooled:
            ordered = np.argsort(np.asarray(pooled, dtype=np.int64), kind="stable")
            y = labels[np.asarray(pooled, dtype=np.int64)]
            diagnostics["sensitivity_unweighted"] = {
                "what": "the same folds with the calibrator fitted without label weights "
                "(report-only, design/m2-analysis-plan.md §6.6)",
                "n": len(pooled),
                "weighted": calibration_summary(probs[ordered], y[ordered]),
                "unweighted": calibration_summary(np.vstack(unweighted)[ordered], y[ordered]),
            }
    elif raw and pooled:
        diagnostics.update(_raw_score_auroc(task, pooled, np.concatenate(raw), probs, B, seed))
    if x1 is not None:
        agree = sum(
            1 for c in chosen.values() if (c["framing"], c["model"]) == (arm.framing, arm.model)
        )
        diagnostics["fold_agreement_with_a1"] = {"agree": agree, "folds": len(outer)}
    n_pre = _pre_rule_count(task)
    if n_pre is not None:
        diagnostics["n_pre_rule"] = n_pre
    return assemble_cell(
        task,
        pooled,
        probs,
        evaluated,
        tier=tier,
        framing=arm.framing,
        variant=variant,
        model=arm.model,
        question_key=frame.question_key,
        fingerprint=fingerprint,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        selection=selection,
        pre_registered=pre_registered,
        exploratory=exploratory,
        servable=not frame.control and arm.model in SERVED_MODELS,
        skipped_folds=skipped,
        fold_choices=chosen,
        diagnostics=diagnostics,
        notes=notes,
        B=B,
        seed=seed,
    )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def calibration_summary(probs: FloatArray, labels: npt.ArrayLike) -> dict[str, Any]:
    """Pooled NLL, ECE, accuracy and Brier of ``[n, K]`` predictions, and the calibration in the
    large: per option the mean predicted probability minus the option's frequency (index 0 is
    Yes on a two-class task), the quantity the D20 weights move (§2.4, §6.6)."""
    y = np.asarray(labels, dtype=np.int64)
    if len(y) == 0:
        return {"nll": None, "ece": None, "acc": None, "brier": None, "in_the_large": None}
    freq = np.bincount(y, minlength=probs.shape[1]) / len(y)
    return {
        "nll": finite(stats.metric_value("nll", probs, y)),
        "ece": finite(stats.metric_value("ece", probs, y)),
        "acc": finite(stats.metric_value("acc", probs, y)),
        "brier": finite(stats.metric_value("brier", probs, y)),
        "in_the_large": [finite(v) for v in (probs.mean(axis=0) - freq).tolist()],
    }


def _raw_score_auroc(
    task: Task, pooled: Sequence[int], score: FloatArray, probs: FloatArray, B: int, seed: int
) -> dict[str, Any]:
    """The zero-shot AUROC of the raw score (``s_c``, or the logit difference at K = 2) with its
    cluster CI, next to the cell's AUROC of ``σ(score)``: float64 rounds ``σ`` to exactly 1.0
    above a score of about 37, which can only add ties (``n_saturated`` counts them)."""
    if not task.binary:
        return {}
    order = np.argsort(np.asarray(pooled, dtype=np.int64), kind="stable")
    idx = np.asarray(pooled, dtype=np.int64)[order]
    labels = np.asarray(task.labels, dtype=np.int64)[idx]
    cards = [task.cards[i] for i in idx]
    out: dict[str, Any] = {
        "n_saturated": int(np.sum((probs[:, 0] == 0.0) | (probs[:, 0] == 1.0))),
        "auroc_raw_score": {
            "score": "s_c" if task.task_id in RANK_FIT_TASKS else "logit_difference",
            "point": finite(stats.auroc(score[order], labels, positive=0)),
        },
    }
    if len(set(cards)) >= 2:
        ci = stats.auroc_ci(score[order], labels, cards, B=B, seed=seed)
        out["auroc_raw_score"].update(_json_safe(ci.as_dict()))
    return out


def _fingerprint(fingerprints: Mapping[str, Mapping[str, str]], model: str) -> dict[str, str]:
    try:
        return dict(fingerprints[model])
    except KeyError:
        raise CellError(
            f"no fingerprint for model {model!r} (D5: every cell carries one)"
        ) from None


def task_cells(
    task: Task,
    index: TextIndex,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    fingerprints: Mapping[str, Mapping[str, str]],
    *,
    labels_sha256: str,
    labels_content_sha256: str | None,
    registered: bool,
    selection: X1Selection | None = None,
    tiers: Sequence[TierName] = TIERS,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> dict[str, BenchCell]:
    """Every tier cell of one task, keyed by ``cell_key`` (§9.5). A rank_fit task needs
    ``selection`` (X1's outcome; ``a1=None`` is X1's K1 or undecidable outcome); a closed choice
    benches F7 under every model in ``scorers``, ``clm-latest`` as the cell and the others as
    ``@<model>`` variants. ``registered`` says the run is the pre-registered one (§13); when it
    is false every cell is ``pre_registered: false`` and ``exploratory: true``."""
    fps = {m: _fingerprint(fingerprints, m).get("clm_model_fp", "") for m in fingerprints}
    deviation = "" if registered else "; not the pre-registered run (§13): exploratory"
    common: dict[str, Any] = {
        "labels_sha256": labels_sha256,
        "labels_content_sha256": labels_content_sha256,
        "B": B,
        "seed": seed,
    }
    cache: dict[Arm, ArmScores] = {}

    def scored(arms: Iterable[Arm]) -> dict[Arm, ArmScores]:
        for a in arms:
            if a not in cache:
                if a.model not in scorers:
                    raise CellError(f"{task.name}: no scorer for model {a.model!r}")
                cache[a] = score_arm(task, index, store, scorers[a.model], a.framing, clock=clock)
        return cache

    out: dict[str, BenchCell] = {}
    if task.task_id in RANK_FIT_TASKS:
        if selection is None:
            raise CellError(f"{task.name}: a rank_fit task's cells need X1's selection (§9.5)")
        if selection.a1 is None:  # K1 or undecidable: audit-only cells of the default arm
            note = (
                (selection.note or "K1: no X1 arm qualified")
                + (
                    "; no A1: the zero_shot/calibrated tiers are audit-only (PR K1), reported for the "
                    "production default arm F7@clm-latest"
                )
                + deviation
            )
            for tier in tiers:
                cell = tier_cell(
                    task,
                    scored([DEFAULT_ARM]),
                    tier,
                    arm=DEFAULT_ARM,
                    fingerprint=_fingerprint(fingerprints, DEFAULT_ARM.model),
                    selection="none",
                    pre_registered=False,
                    exploratory=True,
                    notes=note,
                    **common,
                )
                out[cell.key] = cell
            return out
        a1 = selection.a1
        arms = {a1, *(c.arm for c in selection.fold_choices.values())}
        all_scores = scored(sorted(arms))
        for tier in tiers:
            nested = tier_cell(
                task,
                all_scores,
                tier,
                arm=a1,
                fingerprint=_fingerprint(fingerprints, a1.model),
                selection="nested",
                pre_registered=registered,
                exploratory=not registered,
                x1=selection,
                clm_model_fps=fps,
                notes=selection.note + deviation,
                **common,
            )
            full = tier_cell(
                task,
                all_scores,
                tier,
                arm=a1,
                fingerprint=_fingerprint(fingerprints, a1.model),
                selection="full",
                pre_registered=registered,
                exploratory=True,
                variant=FULL_VARIANT,
                notes="A1 chosen on all seven cards: exploratory, never citable (D27)" + deviation,
                **common,
            )
            out[nested.key] = nested
            out[full.key] = full
        return out
    for model in sorted(scorers, key=lambda m: (m != LATEST_MODEL, m)):
        arm = Arm(CHOICE_FRAMING, model)
        production = model == LATEST_MODEL
        for tier in tiers:
            cell = tier_cell(
                task,
                scored([arm]),
                tier,
                arm=arm,
                fingerprint=_fingerprint(fingerprints, model),
                selection="none",
                pre_registered=production and registered,
                exploratory=not (production and registered),
                variant=None if production else model,
                notes=(
                    ""
                    if production
                    else f"{model} on a closed choice: reported, not pre-registered (§9.5)"
                )
                + deviation,
                **common,
            )
            out[cell.key] = cell
    return out


def run_tier_cells(
    tasks: Mapping[str, Task],
    index: TextIndex,
    store: FeatureStore,
    scorers: Mapping[str, OfflineScorer],
    fingerprints: Mapping[str, Mapping[str, str]],
    *,
    selections: Mapping[str, X1Selection],
    labels_sha256: str,
    labels_content_sha256: str | None,
    date: str,
    registered: bool,
    deviations: Sequence[str] = (),
    name: str = "tiers",
    tiers: Sequence[TierName] = TIERS,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    clock: Clock = time.perf_counter,
) -> BenchResults:
    """The ``tiers`` results file (§12.3): :func:`task_cells` for every task in ``tasks``
    (bench task name -> task); ``selections`` maps a rank_fit task id to X1's outcome;
    ``registered`` (with the ``deviations`` that made it false) stamps every cell (§13). Labels
    or bootstrap settings other than the registration's
    (:func:`mesa_clm.bench.registered.run_deviations`), another framings lock or model
    fingerprint (:func:`mesa_clm.bench.registered.identity_deviations`), a subset of the tiers
    or of the registered tasks make the run unregistered whatever the caller says, and a
    registered run's tasks must have their published counts (§1.3)."""
    if registered and deviations:
        raise CellError("a run with deviations cannot be the registered one")
    own = [
        *reg.run_deviations(
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            B=B,
            seed=seed,
        ),
        *reg.identity_deviations(
            framings_lock_sha=framings.lock_sha(), fingerprints=fingerprints, models=sorted(scorers)
        ),
        *([] if tuple(tiers) == TIERS else [f"tiers {list(tiers)} is not {list(TIERS)}"]),
        *reg.task_set_deviations(list(tasks)),
    ]
    if own:
        registered, deviations = False, [*deviations, *(d for d in own if d not in deviations)]
    if registered:
        for task_name, task in tasks.items():
            reg.check_task(task_name, task)
    cells: dict[str, BenchCell] = {}
    for task in tasks.values():
        cells.update(
            task_cells(
                task,
                index,
                store,
                scorers,
                fingerprints,
                labels_sha256=labels_sha256,
                labels_content_sha256=labels_content_sha256,
                registered=registered,
                selection=selections.get(task.task_id),
                tiers=tiers,
                B=B,
                seed=seed,
                clock=clock,
            )
        )
    notes = [
        "zero_shot and calibrated cells for every task (plan §8 M2, X3), scored offline from "
        "the feature store; design/m2-analysis-plan.md §9 says how every field is computed.",
        "A rank_fit cell without @ is nested (fold_choices from X1's inner CV; the only "
        "citable-form rank_fit cells of M2); @full uses A1 in every fold and is exploratory "
        "(D27). Without an A1 (X1's K1 or undecidable outcome) the default arm F7@clm-latest is "
        "reported for audit only. A closed choice is F7 under clm-latest; @clm-raw is reported, "
        "not pre-registered.",
        "Silver labels are four-model agreement, not truth. ms_per_decision is the offline "
        "replay time, not serving latency.",
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
