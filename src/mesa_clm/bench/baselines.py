"""No-model baselines: majority, the deterministic lookup, ``lookup_prob``, the novel-key
subset and leave-one-product-out (DESIGN D8; plan §5.4, X2).

A read-only check during planning found that copying the label of the same (scope, column,
CURIE) from the other cards already scores leave-one-card-out accuracy 0.772 on ``term.fits``
and 0.800 on ``column.ontology_fits`` (plan §1), above AnyJev L2's 0.765. Every citable cell
must therefore beat the lookup where the lookup knows nothing: on the *novel keys*, the
held-out items whose key no training card carries. This module reproduces those controls from
the label store so the numbers are re-derived, never copied (AGENTS.md ground rule 1).

**Lookup key** (plan §5.4): ``(task_id, scope, target, option_key)`` where ``scope`` is the
state's scope (or the task's), ``target`` the column name for a column or AVU scope, the site
code for a site scope and ``''`` for the dataset itself, and ``option_key`` the candidate a
rank_fit label names (the CURIE for ``term.fits``, the registry id for ``column.ontology_fits``)
or ``''`` for a closed choice. Card, aspect and candidate text are *not* part of the key.

**Deterministic lookup:** for a held-out item, the most common label of its key among the
training cards; when two labels tie, the label of the alphabetically first training card
carrying the key (``collections.Counter.most_common`` over rows taken in sorted card order is
stable on ties); an unseen key gets the training majority. This is the rule the planning-time
measurement used and it reproduces 220/285 and 152/190 exactly; a symmetric tie rule gives
221/151.

**``lookup_prob``** (plan §5.4): Laplace-smoothed (α=1) label frequency of the key over the
training cards, ``(n_k + 1) / (n + K)``, falling back to the empirical training prior for an
unseen key, so on novel keys its AUROC is 0.5 up to the per-fold priors and its NLL is the prior
cross-entropy. ``beats_lookup_novel`` := a model's novel-key AUROC cluster lower bound > 0.5 and
its novel-key NLL ≻ ``lookup_prob`` under rule R (:mod:`mesa_clm.bench.stats`); novel-key
accuracy against the majority is reported, not gated.

Baselines are evaluated on every fold (a lookup needs no fit and has no guard); the folds the
30/5 guards would skip for a fitted tier are recorded in ``guard_skipped_folds`` so a model cell
can be compared on the same pool by passing ``folds=`` explicitly.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from mesa_clm.bench import _metrics, stats
from mesa_clm.bench import metrics as tie_metrics
from mesa_clm.bench.results import (
    BenchCell,
    BenchResults,
    CellBaselines,
    CellCounts,
    CellMetrics,
    ClusterCI,
    Lopo,
    NovelKey,
    cell_key,
    environment,
    finite,
    labels_content_sha256,
)
from mesa_clm.bench.tasks.base import Fold, Task, fold_guard
from mesa_clm.bench.tasks.neon import tasks_from_store
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.tasks import RANK_FIT_TASKS
from mesa_clm.tasks import TASKS as SPECS

LookupKey = tuple[str, str, str, str]

LAPLACE_ALPHA: Final = 1.0
LOOKUP_KEY_DOC: Final = (
    "(task_id, scope, target: column name | site code | '' for the dataset, option_key)"
)
LOOKUP_RULE_DOC: Final = (
    "most common training label of the key; ties -> the label of the alphabetically first "
    "training card carrying the key; unseen key -> training majority"
)
BASELINE_TIER: Final = "baseline"
BASELINE_FRAMING: Final = "lookup_prob"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def lookup_key(task_id: str, state: dict[str, Any], option_key: str = "") -> LookupKey:
    """The lookup key of a label on ``state`` (module docstring). ``KeyError`` when a column or
    site scope state lacks its target; ``ValueError`` for an unknown scope."""
    scope = str(state.get("scope") or SPECS[task_id].scope)
    if scope in ("column", "avu"):
        target = str(state["column"]["name"])
    elif scope == "site":
        target = str(state["site"]["code"])
    elif scope == "dataset":
        target = ""
    else:
        raise ValueError(f"{task_id}: unknown scope {scope!r}")
    return (task_id, scope, target, option_key)


def task_keys(task: Task) -> list[LookupKey]:
    """One lookup key per item of ``task``."""
    opts = task.option_keys or [""] * len(task.items)
    return [
        lookup_key(task.task_id, state, opt)
        for (state, _), opt in zip(task.items, opts, strict=True)
    ]


class Lookup:
    """A lookup table fitted on the training items of one fold (module docstring).

    ``table[key]`` counts labels in *card order*, so ``most_common`` breaks a tie towards the
    alphabetically first card; ``prior`` counts every training label.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.table: dict[LookupKey, collections.Counter[int]] = {}
        self.prior: collections.Counter[int] = collections.Counter()

    @classmethod
    def fit(
        cls, task: Task, train: Sequence[int], keys: Sequence[LookupKey] | None = None
    ) -> Lookup:
        keys = keys if keys is not None else task_keys(task)
        cards = task.cards or [""] * len(task.items)
        self = cls(task.k)
        for i in sorted(train, key=lambda j: (cards[j], j)):
            label = task.items[i][1]
            self.table.setdefault(keys[i], collections.Counter())[label] += 1
            self.prior[label] += 1
        return self

    @property
    def n_train(self) -> int:
        return sum(self.prior.values())

    @property
    def majority(self) -> int:
        """The training majority label (ties: the label seen first in card order)."""
        if not self.prior:
            return 0
        return self.prior.most_common(1)[0][0]

    @property
    def prior_probs(self) -> FloatArray:
        """The empirical training class frequencies (uniform for an empty training set)."""
        if not self.prior:
            uniform: FloatArray = np.full(self.k, 1.0 / self.k, dtype=float)
            return uniform
        counts = np.asarray([self.prior.get(c, 0) for c in range(self.k)], dtype=float)
        probs: FloatArray = counts / counts.sum()
        return probs

    def seen(self, key: LookupKey) -> bool:
        return key in self.table

    def conflicting(self, key: LookupKey) -> bool:
        """Training cards disagree on the label of ``key``."""
        return len(self.table.get(key, ())) > 1

    def tied(self, key: LookupKey) -> bool:
        """The two most common training labels of ``key`` have the same count."""
        top = self.table[key].most_common(2) if key in self.table else []
        return len(top) > 1 and top[0][1] == top[1][1]

    def predict(self, key: LookupKey) -> int:
        """The deterministic lookup label."""
        counts = self.table.get(key)
        if counts is None:
            return self.majority
        return counts.most_common(1)[0][0]

    def prob(self, key: LookupKey, alpha: float = LAPLACE_ALPHA) -> FloatArray:
        """``lookup_prob``: Laplace(α) label frequency of ``key``, or the training prior."""
        counts = self.table.get(key)
        if counts is None:
            return self.prior_probs
        c = np.asarray([counts.get(i, 0) for i in range(self.k)], dtype=float)
        smoothed: FloatArray = (c + alpha) / (c.sum() + alpha * self.k)
        return smoothed


@dataclass
class LookupEval:
    """Pooled held-out predictions of the lookups over a set of folds: parallel to the items
    that were held out at least once (every item, for leave-one-*-out); ``idx`` maps rows back
    to ``Task.items``."""

    idx: IntArray
    labels: IntArray
    cards: list[str]
    fold: list[str]
    pred: IntArray
    majority_pred: IntArray
    probs: FloatArray
    novel: npt.NDArray[np.bool_]
    conflicting: npt.NDArray[np.bool_]
    tied: npt.NDArray[np.bool_]
    n_folds: int
    folds: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.idx)

    @property
    def label_list(self) -> list[int]:
        return stats.as_list(self.labels)

    def acc(self) -> float:
        return float(np.mean(self.pred == self.labels)) if self.n else float("nan")

    def majority_acc(self) -> float:
        return float(np.mean(self.majority_pred == self.labels)) if self.n else float("nan")

    def subset(self, mask: npt.NDArray[np.bool_]) -> LookupEval:
        return LookupEval(
            idx=self.idx[mask],
            labels=self.labels[mask],
            cards=[c for c, m in zip(self.cards, mask, strict=True) if m],
            fold=[f for f, m in zip(self.fold, mask, strict=True) if m],
            pred=self.pred[mask],
            majority_pred=self.majority_pred[mask],
            probs=self.probs[mask],
            novel=self.novel[mask],
            conflicting=self.conflicting[mask],
            tied=self.tied[mask],
            n_folds=self.n_folds,
            folds=list(self.folds),
        )


def evaluate_lookup(
    task: Task, folds: Iterable[Fold] | None = None, *, alpha: float = LAPLACE_ALPHA
) -> LookupEval:
    """Fit a :class:`Lookup` on every fold's training items and score its held-out items;
    ``folds`` defaults to ``task.leave_one_card_out()``. Predictions are pooled in item order."""
    keys = task_keys(task)
    cards = task.cards or [""] * len(task.items)
    labels = task.labels
    rows: list[tuple[int, str, int, int, FloatArray, bool, bool, bool]] = []
    names: list[str] = []
    for fold in folds if folds is not None else task.leave_one_card_out():
        names.append(fold.held_out)
        lk = Lookup.fit(task, fold.train, keys)
        for i in fold.test:
            key = keys[i]
            rows.append(
                (
                    i,
                    fold.held_out,
                    lk.predict(key),
                    lk.majority,
                    lk.prob(key, alpha),
                    not lk.seen(key),
                    lk.conflicting(key),
                    lk.tied(key),
                )
            )
    rows.sort(key=lambda r: r[0])
    idx = np.asarray([r[0] for r in rows], dtype=np.int64)
    return LookupEval(
        idx=idx,
        labels=np.asarray([labels[i] for i in idx], dtype=np.int64),
        cards=[cards[i] for i in idx],
        fold=[r[1] for r in rows],
        pred=np.asarray([r[2] for r in rows], dtype=np.int64),
        majority_pred=np.asarray([r[3] for r in rows], dtype=np.int64),
        probs=np.asarray([r[4] for r in rows], dtype=float).reshape(len(rows), task.k),
        novel=np.asarray([r[5] for r in rows], dtype=bool),
        conflicting=np.asarray([r[6] for r in rows], dtype=bool),
        tied=np.asarray([r[7] for r in rows], dtype=bool),
        n_folds=len(names),
        folds=names,
    )


def _ci_model(ci: stats.BootstrapCI) -> ClusterCI:
    return ClusterCI(
        point=finite(ci.point),
        lower=finite(ci.lower),
        upper=finite(ci.upper),
        B=ci.B,
        seed=ci.seed,
        alpha=ci.alpha,
        n_clusters=ci.n_clusters,
        n_valid=ci.n_valid,
    )


def _auroc_ci(ev: LookupEval, *, binary: bool, B: int, seed: int) -> ClusterCI | None:
    if not binary or ev.n == 0 or len(set(ev.cards)) < 2:
        return None
    return _ci_model(stats.metric_ci("auroc", ev.probs, ev.labels, ev.cards, B=B, seed=seed))


def novel_key_block(ev: LookupEval, *, binary: bool, B: int, seed: int) -> NovelKey:
    """The ``novel_key`` block of the ``lookup_prob`` cell over the items of ``ev`` whose key
    was unseen in their fold's training cards."""
    nov = ev.subset(ev.novel)
    counts = dict(sorted(collections.Counter(nov.labels.tolist()).items()))
    if nov.n == 0:
        return NovelKey(
            n=0,
            class_counts={},
            majority_acc=None,
            acc=None,
            auroc=None,
            auroc_ci=None,
            nll=None,
            brier=None,
        )
    return NovelKey(
        n=nov.n,
        class_counts=counts,
        majority_acc=finite(nov.majority_acc()),
        acc=finite(_metrics.accuracy(nov.probs, nov.label_list)),
        auroc=finite(stats.metric_value("auroc", nov.probs, nov.labels)),
        auroc_ci=_auroc_ci(nov, binary=binary, B=B, seed=seed),
        nll=finite(_metrics.nll(nov.probs, nov.label_list)),
        brier=finite(_metrics.brier(nov.probs, nov.label_list)),
    )


def lopo_block(task: Task) -> Lopo:
    """The leave-one-product-out control (``None`` metrics when the task has one product)."""
    if not task.products or len(set(task.products)) < 2:
        return Lopo(n_folds=0, n=0, acc=None, nll=None, novel_n=0, majority_acc=None)
    ev = evaluate_lookup(task, task.leave_one_product_out())
    return Lopo(
        n_folds=ev.n_folds,
        n=ev.n,
        acc=finite(ev.acc()),
        nll=finite(_metrics.nll(ev.probs, ev.label_list)),
        novel_n=int(ev.novel.sum()),
        majority_acc=finite(ev.majority_acc()),
    )


def baselines_block(
    task: Task, ev: LookupEval, *, B: int = stats.DEFAULT_B, seed: int = stats.DEFAULT_SEED
) -> CellBaselines:
    """The ``baselines`` block of a cell from the leave-one-card-out lookup evaluation ``ev``
    (and a fresh leave-one-product-out run)."""
    return CellBaselines(
        lookup_key=LOOKUP_KEY_DOC,
        lookup_rule=LOOKUP_RULE_DOC,
        majority_acc=ev.majority_acc(),
        lookup_acc=ev.acc(),
        lookup_nll=_metrics.nll(ev.probs, ev.label_list),
        lookup_prob_acc=_metrics.accuracy(ev.probs, ev.label_list),
        novel_key=novel_key_block(ev, binary=task.binary, B=B, seed=seed),
        lopo=lopo_block(task),
        beats_lookup_novel=None,
    )


def pooled_metrics(
    task: Task,
    probs: FloatArray,
    labels: IntArray,
    cards: Sequence[str],
    *,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> CellMetrics:
    """The plan §5.4 metrics block over pooled held-out probabilities (any tier)."""
    y = stats.as_list(labels)
    summary = tie_metrics.summarize(probs, y)
    auroc = stats.metric_value("auroc", probs, labels) if task.binary else None
    ci = (
        _ci_model(stats.metric_ci("auroc", probs, labels, cards, B=B, seed=seed))
        if task.binary and len(set(cards)) >= 2
        else None
    )
    return CellMetrics(
        acc=float(summary["acc"]),
        macro_f1=float(summary["macro_f1"]),
        brier=float(summary["brier"]),
        nll=float(summary["nll"]),
        ece=float(summary["ece"]),
        cov_at_5=float(summary["cov@5%"]),
        cov_at_10=float(tie_metrics.coverage_at_risk(probs, y, target=0.10)),
        aurc=float(summary["aurc"]),
        auroc=finite(auroc),
        auroc_ci=ci,
        threshold_cp=stats.thresholds_cp(probs, labels, rank_fit=task.task_id in RANK_FIT_TASKS),
    )


def guard_skipped_folds(task: Task) -> dict[str, str]:
    """The leave-one-card-out folds the 30/5 guards would skip for a fitted tier."""
    out: dict[str, str] = {}
    for fold in task.leave_one_card_out():
        reason = fold_guard(task, fold)
        if reason is not None:
            out[fold.held_out] = reason
    return out


def baselines_cell(
    task: Task,
    *,
    labels_sha256: str,
    labels_content_sha256: str | None = None,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchCell:
    """The ``<task>.baseline.lookup_prob`` cell: ``lookup_prob``'s pooled leave-one-card-out
    metrics, the baselines block and the lookup diagnostics."""
    ev = evaluate_lookup(task)
    return BenchCell(
        task=task.name,
        task_id=task.task_id,
        task_key=task.spec.key,
        tier=BASELINE_TIER,
        framing=BASELINE_FRAMING,
        question_key=None,
        fingerprint=None,
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        feature_spec=None,
        label_sources=dict(task.meta.get("label_sources", {})),
        min_weight=float(task.meta["min_weight"]),
        teacher=False,
        teacher_in_test=False,
        masked=bool(task.meta.get("masked", False)),
        mask=task.meta.get("mask"),
        loco=True,
        selection="none",
        pre_registered=True,
        exploratory=False,
        servable=False,
        n_folds=ev.n_folds,
        skipped_folds={},
        guard_skipped_folds=guard_skipped_folds(task),
        fold_choices={},
        counts=CellCounts(
            n=len(task.items),
            n_neg=task.n_neg(),
            n_nonmodal=task.n_nonmodal(),
            class_counts=task.class_counts(),
        ),
        metrics=pooled_metrics(task, ev.probs, ev.labels, ev.cards, B=B, seed=seed),
        baselines=baselines_block(task, ev, B=B, seed=seed),
        diagnostics={
            "laplace_alpha": LAPLACE_ALPHA,
            "n_novel": int(ev.novel.sum()),
            "n_conflicting_keys": int(ev.conflicting.sum()),
            "n_tied_keys": int(ev.tied.sum()),
            "excluded": dict(task.meta.get("excluded", {})),
            "per_fold_n": {name: int(sum(1 for f in ev.fold if f == name)) for name in ev.folds},
        },
        notes=task.notes,
    )


def run_baselines(
    store: LabelStore,
    *,
    labels_sha256: str,
    date: str,
    name: str = "baselines",
    policy_path: str | Path | None = None,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> BenchResults:
    """Baseline cells for every neon task in ``store`` (``labels_sha256`` is the hash of the
    snapshot the store was frozen to, D30)."""
    content = labels_content_sha256(store)
    tasks = tasks_from_store(store, policy_path=policy_path)
    cells = {
        cell_key(t.name, BASELINE_TIER, BASELINE_FRAMING): baselines_cell(
            t, labels_sha256=labels_sha256, labels_content_sha256=content, B=B, seed=seed
        )
        for t in tasks.values()
    }
    return BenchResults(
        date=date,
        name=name,
        mesa_clm=environment()["mesa_clm"],
        labels_sha256=labels_sha256,
        labels_content_sha256=content,
        environment=environment(),
        cells=cells,
        notes=[
            "No-model controls (plan §5.4, X2). `lookup_acc` copies the most common training label "
            "of (task, scope, target, option_key) from the other cards; `lookup_prob` is its "
            "Laplace(α=1) frequency with the training prior on unseen keys. Novel keys are the "
            "held-out items no training card had a key for; `lopo` holds out a whole NEON product.",
            "Every cell is a leave-one-card-out pool over all 7 cards (a lookup needs no fit guard); "
            "`guard_skipped_folds` lists the folds a fitted tier would skip under the 30/5 guards.",
        ],
    )
