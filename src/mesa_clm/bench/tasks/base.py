"""The AnyJev bench ``Task`` pattern (bench/tasks/base.py at 795a497, Apache-2.0), ported from
mesa-anyjev ``bench/tasks/base.py`` (``6159281``) without ``anyjev.Question``: a bench task
carries a frozen :class:`mesa_clm.tasks.Task` instead, and its items are ``(state, label)``
pairs whose label indexes that task's options.

Two things changed in the port. Folds are yielded as :class:`Fold` index triples instead of
item lists, so a fitter or a baseline can read the parallel ``cards``, ``weights``,
``products`` and ``option_keys`` of a fold without searching for them; and
``leave_one_product_out`` joins ``leave_one_card_out`` (plan §5.4: the leave-one-product-out
control that shows how much of a lookup's accuracy is the sibling tables of one NEON product).
Leave-one-card-out remains the only split that may set a threshold or promote an artifact
(DESIGN D8, D27). The fold guards (30 per class in training, 5 per class held out) are the
ones mesa-anyjev's ``bench/run.py`` and ``learn/fit.py`` shared; they apply to the two-class
tasks only, as there.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

from mesa_clm.tasks import Task as TaskSpec

TASKS: dict[str, Callable[[], Task]] = {}

Item = tuple[dict[str, Any], int]  # (state, label index into spec.options)

# Fold guards (mesa-anyjev ``bench/run.py:33-34``; plan §5.4): a leave-one-card-out fold of a
# two-class task is skipped when any class has fewer training or held-out items than these.
MIN_TRAIN_PER_CLASS: Final = 30
MIN_HELDOUT_PER_CLASS: Final = 5

# The tiers a task can be benched at (plan §4.5 minus ``none``, which is never a bench tier).
ALL_TIERS: Final[tuple[str, ...]] = ("zero_shot", "calibrated", "probe", "head")


class Fold(NamedTuple):
    """One held-out group: ``held_out`` names the card (or product), ``test`` and ``train`` are
    item indices into ``Task.items`` (and its parallel lists), each in item order."""

    held_out: str
    test: list[int]
    train: list[int]


@dataclass
class Task:
    """A labelled item set for one frozen task.

    ``items[i] = (state, label)``; ``cards[i]``, ``weights[i]``, ``products[i]`` (the
    ``leak_group``, a NEON product code) and ``option_keys[i]`` (the candidate a rank_fit label
    names, ``''`` for a closed choice) are parallel to it when given. ``meta`` carries what the
    cell records about the item set (``task_id``, ``min_weight``, ``class_counts``, ``masked``).
    """

    name: str
    spec: TaskSpec
    items: list[Item]
    license: str
    source: str
    notes: str = ""
    tiers_supported: tuple[str, ...] = ALL_TIERS
    cards: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    option_keys: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.items)
        for attr in ("cards", "weights", "products", "option_keys"):
            values = getattr(self, attr)
            if values and len(values) != n:
                raise ValueError(
                    f"{self.name}: {attr} needs one entry per item ({len(values)} vs {n})"
                )
        for _, label in self.items:
            if not 0 <= label < self.spec.k:
                raise ValueError(f"{self.name}: label {label} is not an option of {self.spec.id}")

    # -- shape ----------------------------------------------------------------------------------
    @property
    def task_id(self) -> str:
        return self.spec.id

    @property
    def k(self) -> int:
        return self.spec.k

    @property
    def binary(self) -> bool:
        """A two-class task (``noul`` in AnyJev's vocabulary: index 0 is "Yes", the positive)."""
        return self.spec.kind == "noul"

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.items]

    @property
    def states(self) -> list[dict[str, Any]]:
        return [state for state, _ in self.items]

    def subset(self, idx: Sequence[int]) -> list[Item]:
        return [self.items[i] for i in idx]

    # -- splits -----------------------------------------------------------------------------------
    def split(self, n_test: int, n_calib: int, seed: int = 0) -> tuple[list[Item], list[Item]]:
        """A shuffled exploratory split (never a citable one, DESIGN D8)."""
        rng = random.Random(seed)  # noqa: S311 - reproducible exploratory split, not security
        idx = list(range(len(self.items)))
        rng.shuffle(idx)
        test = [self.items[i] for i in idx[:n_test]]
        calib = [self.items[i] for i in idx[n_test : n_test + n_calib]]
        return test, calib

    def _folds(self, groups: Sequence[str], what: str) -> Iterator[Fold]:
        if len(groups) != len(self.items):
            raise ValueError(f"{what} needs one group per item")
        for held_out in sorted(set(groups)):
            test = [i for i, g in enumerate(groups) if g == held_out]
            train = [i for i, g in enumerate(groups) if g != held_out]
            yield Fold(held_out, test, train)

    def leave_one_card_out(self) -> Iterator[Fold]:
        """One :class:`Fold` per distinct card, in sorted card order (plan §5.4). The only split
        that may set a threshold or promote an artifact (DESIGN D8)."""
        return self._folds(self.cards, "leave_one_card_out")

    def leave_one_product_out(self) -> Iterator[Fold]:
        """One :class:`Fold` per distinct product (``leak_group``), in sorted order: the
        leave-one-product-out control of plan §5.4 (report-only, never citable)."""
        return self._folds(self.products, "leave_one_product_out")

    # -- counts ------------------------------------------------------------------------------------
    def class_counts(self, idx: Sequence[int] | None = None) -> dict[int, int]:
        """``{label: n}`` over all items, or over the items at ``idx``, sorted by label."""
        counts: dict[int, int] = {}
        for i in idx if idx is not None else range(len(self.items)):
            label = self.items[i][1]
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def n_neg(self, idx: Sequence[int] | None = None) -> int | None:
        """Items whose label is not the positive (index 0) of a two-class task; ``None`` for a
        closed choice, whose guard is ``n_nonmodal`` (plan §4.7)."""
        if not self.binary:
            return None
        return sum(n for label, n in self.class_counts(idx).items() if label != 0)

    def n_nonmodal(self, idx: Sequence[int] | None = None) -> int:
        """Items outside the most frequent class (the choice-task analogue of ``n_neg``)."""
        counts = self.class_counts(idx)
        return sum(counts.values()) - max(counts.values(), default=0)


def fold_guard(
    task: Task,
    fold: Fold,
    *,
    min_train: int = MIN_TRAIN_PER_CLASS,
    min_heldout: int = MIN_HELDOUT_PER_CLASS,
) -> str | None:
    """Why a fold of a two-class task must be skipped (``insufficient_train_per_class {…}`` or
    ``insufficient_heldout_per_class {…}``), or ``None`` when it may run. A closed-choice task
    passes unconditionally, as in mesa-anyjev's ``_fit_eval_loco`` (its guard is the
    ``n_nonmodal`` floor of the citation rule, plan §4.7)."""
    if not task.binary:
        return None
    tr = task.class_counts(fold.train)
    te = task.class_counts(fold.test)
    if len(tr) < 2 or min(tr.values(), default=0) < min_train:
        return f"insufficient_train_per_class {tr}"
    if len(te) < 2 or min(te.values(), default=0) < min_heldout:
        return f"insufficient_heldout_per_class {te}"
    return None


def register(name: str) -> Callable[[Callable[[], Task]], Callable[[], Task]]:
    """Register a zero-argument task loader under ``name`` (importing a task module registers
    its tasks; the neon tasks need a label store and are built by ``tasks_from_store``)."""

    def deco(fn: Callable[[], Task]) -> Callable[[], Task]:
        TASKS[name] = fn
        return fn

    return deco


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()
