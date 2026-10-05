"""``bench e2e --loco``: the end-to-end leave-one-card-out measure of the whole annotate
pipeline (plan §8 M4 "Report-only with CIs: e2e consensus-all recall vs AnyJev static 0.25;
anchor-abstain precision"; the M4 analysis plan's R8 reading, ``design/m4-analysis-plan.md``).

**What it measures, report-only.** Per held-out card, :class:`mesa_clm.pipeline.Annotator`
annotates the card with an offline provider that serves the fold's nested probe (or calibrator)
fitted on the six other cards (candidates the feature store does not hold go to the encoder,
label-free; the OLS fixtures replay), and two numbers are read from the run:

* **consensus-all recall**: the fraction of the card's ``consensus_all`` Yes items (the
  ``term.fits`` CURIEs every neon-avu-eval model proposed, the only silver with four-model
  agreement) whose CURIE the run proposes (``AnnotationRun.proposals``), against AnyJev's static
  0.25 (:data:`ANYJEV_STATIC_RECALL`, plan §8 M4);
* **anchor-abstain precision**: the fraction of the run's anchor-won groups (an abstain with
  ``reason == "anchor_won"``) whose target has no ``consensus_all`` Yes among its candidates,
  i.e. the abstain was right as far as the silver says.

Both are pooled over the cards with a card-cluster bootstrap interval
(:func:`mesa_clm.bench.stats.bootstrap_weights`, B = 2000, seed 0, the one-sided 95% bounds).
Nothing here gates anything and no cell is written: ``e2e.json`` is a report.

**Interface (first wave) and implementation (second wave).** :func:`consensus_all_curies`,
:func:`fold_outcome` and :func:`measure` are the label-free-to-write measurement half: they
read a label store's ``consensus_all`` rows and an :class:`~mesa_clm.pipeline.AnnotationRun`.
The provider that serves a fold's probe from the store (``FoldProvider``: the fold's
``learn.probe`` artifact behind the ``DecisionProvider`` protocol, with the encoder for texts
outside the store) is :func:`run_e2e`'s job and is **not implemented in the pre-run commit**;
the plan says so, and it ships by amendment with its own disclosure (R8). ``bench e2e`` says
the same and exits with a usage error until then.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from mesa_clm.bench import stats
from mesa_clm.provenance.labels import LabelStore

__all__ = [
    "ANCHOR_REASON",
    "ANYJEV_STATIC_RECALL",
    "FORMAT",
    "E2EResult",
    "FoldOutcome",
    "NotImplementedYet",
    "consensus_all_curies",
    "fold_outcome",
    "measure",
    "run_e2e",
]

FORMAT: Final[str] = "mesa-clm/e2e/1"
# AnyJev's static consensus-all recall the e2e recall is compared with (plan §8 M4).
ANYJEV_STATIC_RECALL: Final[float] = 0.25
ANCHOR_REASON: Final[str] = "anchor_won"
CONSENSUS_ALL: Final[str] = "consensus_all"


class NotImplementedYet(NotImplementedError):
    """The second-wave part of ``bench e2e`` (module docstring)."""


@dataclass(frozen=True)
class FoldOutcome:
    """One held-out card: the consensus-all CURIEs, how many the run proposed, the anchor-won
    groups and how many of them were clean (no consensus-all Yes among the group's
    candidates)."""

    card: str
    consensus_all: int
    recalled: int
    anchor_groups: int
    anchor_clean: int

    @property
    def recall(self) -> float:
        return self.recalled / self.consensus_all if self.consensus_all else math.nan

    @property
    def precision(self) -> float:
        return self.anchor_clean / self.anchor_groups if self.anchor_groups else math.nan


@dataclass(frozen=True)
class E2EResult:
    """The pooled numbers with their cluster intervals and the per-card outcomes."""

    folds: tuple[FoldOutcome, ...]
    recall: dict[str, Any]
    anchor_precision: dict[str, Any]
    anyjev_static_recall: float = ANYJEV_STATIC_RECALL
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "folds": [
                {
                    "card": f.card,
                    "consensus_all": f.consensus_all,
                    "recalled": f.recalled,
                    "recall": None if math.isnan(f.recall) else f.recall,
                    "anchor_groups": f.anchor_groups,
                    "anchor_clean": f.anchor_clean,
                    "anchor_precision": None if math.isnan(f.precision) else f.precision,
                }
                for f in self.folds
            ],
            "recall": self.recall,
            "anchor_precision": self.anchor_precision,
            "anyjev_static_recall": self.anyjev_static_recall,
            "notes": list(self.notes),
        }


def consensus_all_curies(store: LabelStore, task_id: str = "term.fits") -> dict[str, set[str]]:
    """The ``consensus_all`` Yes CURIEs per card of ``task_id`` in ``store`` (the e2e truth):
    ``{card: {curie}}``."""
    out: dict[str, set[str]] = {}
    for row in store.labels_for(task_id, sources=[CONSENSUS_ALL]):
        if int(row.get("label_index", 1)) != 0:
            continue
        out.setdefault(str(row["card"]), set()).add(str(row["option_key"]))
    return out


def _proposed_curies(run: Any) -> set[str]:
    out: set[str] = set()
    for p in getattr(run, "proposals", []) or []:
        cand = getattr(p, "candidate", None)
        curie = getattr(cand, "curie", None) if cand is not None else None
        if curie:
            out.add(str(curie))
    return out


def _anchor_groups(run: Any) -> list[Mapping[str, Any]]:
    return [
        a
        for a in getattr(run, "abstained", []) or []
        if isinstance(a, Mapping) and str(a.get("reason")) == ANCHOR_REASON
    ]


def fold_outcome(
    card: str,
    run: Any,
    truth: Iterable[str],
    *,
    group_candidates: Mapping[str, Iterable[str]] | None = None,
) -> FoldOutcome:
    """The :class:`FoldOutcome` of one card from its :class:`~mesa_clm.pipeline.AnnotationRun`
    and the card's consensus-all CURIEs. ``group_candidates`` maps an anchor-won group id to
    the CURIEs it ranked (an anchor group counts as clean when none is a consensus-all Yes;
    without the mapping every anchor group counts as clean only when the card has no
    consensus-all Yes at all, the conservative reading)."""
    want = set(truth)
    proposed = _proposed_curies(run)
    anchors = _anchor_groups(run)
    clean = 0
    for a in anchors:
        gid = str(a.get("group_id"))
        if group_candidates is not None and gid in group_candidates:
            clean += not (set(group_candidates[gid]) & want)
        else:
            clean += not want
    return FoldOutcome(
        card=card,
        consensus_all=len(want),
        recalled=len(want & proposed),
        anchor_groups=len(anchors),
        anchor_clean=clean,
    )


def _ratio_ci(
    num: Sequence[int], den: Sequence[int], cards: Sequence[str], *, B: int, seed: int, alpha: float
) -> dict[str, Any]:
    """The pooled ratio Σnum/Σden with its card-cluster bootstrap bounds."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    total = float(d.sum())
    point = float(n.sum() / total) if total > 0 else None
    if point is None or len(set(cards)) < 2:
        return {"point": point, "lower": None, "upper": None, "B": B, "seed": seed, "alpha": alpha,
                "n_clusters": len(set(cards)), "n": int(total)}  # fmt: skip
    w = stats.bootstrap_weights(list(cards), B=B, seed=seed)
    dens = w @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(dens > 0, (w @ n) / np.where(dens > 0, dens, 1.0), np.nan)
    valid = ratios[~np.isnan(ratios)]
    return {
        "point": point,
        "lower": float(np.quantile(valid, alpha)) if len(valid) else None,
        "upper": float(np.quantile(valid, 1 - alpha)) if len(valid) else None,
        "B": B,
        "seed": seed,
        "alpha": alpha,
        "n_clusters": len(set(cards)),
        "n": int(total),
        "n_valid": len(valid),
    }


def measure(
    folds: Sequence[FoldOutcome],
    *,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
) -> E2EResult:
    """The pooled consensus-all recall and anchor-abstain precision over ``folds`` with their
    card-cluster bootstrap bounds (module docstring)."""
    cards = [f.card for f in folds]
    recall = _ratio_ci(
        [f.recalled for f in folds],
        [f.consensus_all for f in folds],
        cards,
        B=B,
        seed=seed,
        alpha=alpha,
    )
    precision = _ratio_ci(
        [f.anchor_clean for f in folds],
        [f.anchor_groups for f in folds],
        cards,
        B=B,
        seed=seed,
        alpha=alpha,
    )
    return E2EResult(
        folds=tuple(folds),
        recall={**recall, "anyjev_static": ANYJEV_STATIC_RECALL},
        anchor_precision=precision,
        notes=(
            "Report-only (plan §8 M4): consensus-all recall of the whole annotate pipeline per "
            "held-out card against AnyJev's static 0.25, and the precision of the anchor "
            "abstains; card-cluster bootstrap bounds, one-sided 95%.",
        ),
    )


def run_e2e(*args: Any, **kwargs: Any) -> E2EResult:
    """The second wave (module docstring): the fold provider that serves a fold's nested probe
    from the store through the pipeline is not in the pre-run commit."""
    raise NotImplementedYet(
        "bench e2e --loco: the fold provider (a fold's nested probe served through the "
        "annotate pipeline) is second-wave work (R8) and ships by amendment with its own "
        "disclosure; the measurement half (consensus_all_curies, fold_outcome, measure) is here"
    )
