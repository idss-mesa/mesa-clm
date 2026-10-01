"""Bench tasks over the neon-avu-eval labels (NEON data CC BY 4.0; eval outputs MIT).

Ported from mesa-anyjev ``bench/tasks/neon.py`` (``6159281``). Every task reads its items from
the ``mesa_clm.labels`` table through :func:`mesa_clm.learn.labels.labelled_targets`, so the
bench and the fits see the same targets, and its ``min_weight`` comes from
:func:`mesa_clm.policy_defaults.min_weight_for` and nowhere else (DESIGN D9: mesa-anyjev kept
0.6 in the policy and 0.5 in the bench, which at 0.6 would have left ``column.annotate`` with no
No, ``avu.value_kind`` without class 3 and ``column.ontology_fits`` with 76 Yes / 0 No).

Five tasks: ``neon_term_fits``, ``neon_ontology_fits``, ``neon_annotate``, ``neon_aspect`` and
``neon_value_kind`` (plan §4.2). mesa-anyjev's ``neon_ontology_for_column`` and ``neon_keep_avu``
have no labels in mesa-clm (``column.ontology`` is asked as ``ontology_fits``, ``avu.keep`` is a
rule, D25) and the ``neon_term_choice26`` control measured the logprob top-20 cap, which CLM
does not have. Rows that may not enter a fold are dropped here, once, and counted in
``meta["excluded"]``: ``fold_eligible=False`` (teacher and agent labels, D19, D21) and curator
labels on a bench card (``bench_card=True``, or a curator source on one of the fixed
:data:`~mesa_clm.learn.labels.BENCH_CARDS` whatever its tag, D30). They are dropped *before*
the per-identity highest-weight selection (``labelled_targets(fold_only=True)``), so a curator
answer can neither remove a pre-registered item nor change its label. Every task sets
``meta["masked"]`` (the anyjev-learn-bench §5 defect: no cell could ever be cited because no
task set it): the item set carries every option restriction serving applies, which for
``column.ontology_fits`` is the aspect mask (labels exist only for aspect-allowed registry
entries, ``mask_for_aspect``) and for the other tasks is none.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Final

from mesa_clm.bench.tasks.base import Task
from mesa_clm.learn.labels import LabelledSet, labelled_targets
from mesa_clm.policy_defaults import min_weight_for
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.tasks import TASKS as SPECS

LICENSE: Final = (
    "NEON data CC BY 4.0 (DP1.10003.001, DP1.10022.001, RELEASE-2026); "
    "labels are agreement between four agentic models"
)
SOURCE: Final = "neon-avu-eval/results/validated.json"

# Bench task name -> frozen task id, in plan §4.2 order.
NEON_TASKS: Final[dict[str, str]] = {
    "neon_annotate": "column.annotate",
    "neon_aspect": "column.aspect",
    "neon_ontology_fits": "column.ontology_fits",
    "neon_term_fits": "term.fits",
    "neon_value_kind": "avu.value_kind",
}

# The option restriction serving applies per task (``meta["mask"]``); ``None`` means none.
MASKS: Final[dict[str, str | None]] = {
    "column.annotate": None,
    "column.aspect": None,
    "column.ontology_fits": "aspect",
    "term.fits": None,
    "avu.value_kind": None,
}


def _eligible(ls: LabelledSet) -> tuple[list[int], collections.Counter[str]]:
    """Indices of the rows a pre-registered fold may use and why the rows were dropped: the
    counts ``labelled_targets(fold_only=True)`` made before its selection, plus (defensively)
    any selected entry that still is not fold-eligible or on a bench card."""
    keep: list[int] = []
    excluded: collections.Counter[str] = collections.Counter(ls.excluded)
    for i in range(len(ls)):
        if ls.fold_eligible and not ls.fold_eligible[i]:
            excluded["not_fold_eligible"] += 1
        elif ls.bench_card and ls.bench_card[i]:
            excluded["bench_card"] += 1
        else:
            keep.append(i)
    return keep, excluded


def neon_task(
    store: LabelStore,
    name: str,
    task_id: str,
    *,
    policy_path: str | Path | None = None,
    notes: str = "",
) -> Task:
    """The bench task ``name`` over the labels of ``task_id`` at the policy ``min_weight``.

    ``policy_path`` names another ``policy_defaults.yaml`` (tests); the default is the shipped
    file. The returned task's ``meta`` records ``task_id``, ``task_key``, ``min_weight``, the
    class counts and label sources of the items, what was excluded, ``masked`` and ``mask``.
    """
    spec = SPECS[task_id]
    min_weight = min_weight_for(task_id, policy_path)
    ls = labelled_targets(store, task_id, min_weight=min_weight, fold_only=True)
    keep, excluded = _eligible(ls)
    items = [(ls.states[i], ls.labels[i]) for i in keep]
    sources = (
        collections.Counter(ls.sources[i] for i in keep) if ls.sources else collections.Counter()
    )
    task = Task(
        name,
        spec,
        items,
        LICENSE,
        SOURCE,
        notes=notes or f"{len(items)} labelled targets from the eval",
        cards=[ls.cards[i] for i in keep],
        weights=[ls.weights[i] for i in keep],
        products=[ls.leak_group[i] for i in keep] if ls.leak_group else [],
        option_keys=[ls.option_key[i] for i in keep],
        meta={
            "task_id": task_id,
            "task_key": spec.key,
            "min_weight": min_weight,
            "class_counts": {},
            "label_sources": dict(sorted(sources.items())),
            "excluded": dict(sorted(excluded.items())),
            "masked": True,
            "mask": MASKS.get(task_id),
        },
    )
    task.meta["class_counts"] = task.class_counts()
    task.notes = (
        notes
        or f"{len(items)} labelled targets from the eval; class counts {task.meta['class_counts']}"
    )
    return task


def tasks_from_store(
    store: LabelStore, *, policy_path: str | Path | None = None
) -> dict[str, Task]:
    """Every neon task that has at least one item, keyed by bench task name."""
    out: dict[str, Any] = {
        name: neon_task(store, name, task_id, policy_path=policy_path)
        for name, task_id in NEON_TASKS.items()
    }
    return {name: task for name, task in out.items() if task.items}
