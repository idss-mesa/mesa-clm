"""Minimum detectable effect under rule R (DESIGN D27; plan §5.4 ``bench mde``).

Seven cards and a few hundred labels bound what any cell can show. Before the first model cell
is run, this module says how large an improvement over ``lookup_prob`` rule R can detect at the
realised n, base rates and card clusters, so an inconclusive X1–X4 result can be read as
"under-powered" or "no effect" honestly (committed with the pre-registration, G1).

The simulation uses **label counts only**: per card, how many positives and negatives the task
(or its novel-key subset) holds. No model output enters. Model A is a calibrated binormal
scorer: negatives score ``N(0, 1)``, positives ``N(d, 1)``, so its AUROC is ``Φ(d/√2)``, and its
probability is the exact posterior ``σ(logit π + d·s − d²/2)`` at the population base rate ``π``.
Model B is ``lookup_prob`` on novel keys: the constant prior ``π`` (AUROC 0.5 by the tie rule,
NLL the prior cross-entropy). For every AUROC on a grid, ``n_sims`` datasets are drawn with
deterministic seeds and rule R (:func:`mesa_clm.bench.stats.rule_r`, B=2000 cluster bootstrap
plus the card sign test) is applied to A vs B for the AUROC and the NLL metric. The *power* at
a grid point is the fraction of simulations rule R passes; the **MDE** is the smallest grid
AUROC whose power reaches ``power_target`` (0.8), reported as ΔAUROC over 0.5 and, for NLL, as
the mean ΔNLL at that point.

Populations: the whole task (every card, every item) and the novel-key subset the baselines
found (:func:`mesa_clm.bench.baselines.evaluate_lookup`), which is the population
``beats_lookup_novel`` is judged on. Only two-class tasks have an AUROC; the choice tasks are
reported as not applicable.
"""

from __future__ import annotations

import collections
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from mesa_clm.bench import stats
from mesa_clm.bench.baselines import evaluate_lookup
from mesa_clm.bench.results import DEFAULT_OUT_DIR, environment, finite
from mesa_clm.bench.tasks.base import Task

FORMAT: Final = "mesa-clm/bench-mde/1"
DEFAULT_GRID: Final[tuple[float, ...]] = tuple(round(0.55 + 0.025 * i, 3) for i in range(13))
DEFAULT_N_SIMS: Final = 200
POWER_TARGET: Final = 0.8
METRICS: Final[tuple[stats.Metric, ...]] = ("auroc", "nll")

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PowerPoint(_Model):
    """Rule R's power at one simulated effect."""

    auroc: float
    separation: float
    delta_mean: float | None
    power: float
    passed: int
    n_sims: int
    reasons: dict[str, int]


class MdeCurve(_Model):
    """The power curve of one metric on one population and the MDE read off it."""

    metric: str
    population: str
    task: str
    n: int
    n_pos: int
    n_neg: int
    base_rate: float
    n_cards: int
    per_card: dict[str, dict[str, int]]
    cards_counting: int
    power_target: float
    grid: list[PowerPoint]
    mde_auroc: float | None
    mde_delta: float | None
    not_applicable: str | None = None


class MdeResults(_Model):
    format: str = FORMAT
    date: str
    name: str
    mesa_clm: str
    labels_sha256: str
    environment: dict[str, Any]
    n_sims: int
    B: int
    seed: int
    grid: list[float]
    rule: str
    curves: list[MdeCurve]
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Population:
    """Per-card positive and negative counts of a two-class item set (labels only)."""

    name: str
    task: str
    cards: tuple[str, ...]
    n_pos: tuple[int, ...]
    n_neg: tuple[int, ...]

    @classmethod
    def from_task(
        cls, task: Task, idx: Sequence[int] | None = None, *, name: str = "full"
    ) -> Population:
        """The counts of ``task`` (or of its items at ``idx``); ``ValueError`` for a task that
        is not two-class or has no cards."""
        if not task.binary:
            raise ValueError(f"{task.name}: the MDE simulation needs a two-class task")
        if not task.cards:
            raise ValueError(f"{task.name}: the MDE simulation needs one card per item")
        pos: collections.Counter[str] = collections.Counter()
        neg: collections.Counter[str] = collections.Counter()
        for i in idx if idx is not None else range(len(task.items)):
            (pos if task.items[i][1] == 0 else neg)[task.cards[i]] += 1
        cards = tuple(sorted(set(pos) | set(neg)))
        return cls(
            name, task.name, cards, tuple(pos[c] for c in cards), tuple(neg[c] for c in cards)
        )

    @property
    def n(self) -> int:
        return sum(self.n_pos) + sum(self.n_neg)

    @property
    def base_rate(self) -> float:
        return sum(self.n_pos) / self.n if self.n else math.nan

    def labels(self) -> IntArray:
        """Item labels in card order (0 = positive), the layout every simulation uses."""
        parts = [np.repeat([0, 1], [p, q]) for p, q in zip(self.n_pos, self.n_neg, strict=True)]
        return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)

    def clusters(self) -> list[str]:
        out: list[str] = []
        for card, p, q in zip(self.cards, self.n_pos, self.n_neg, strict=True):
            out.extend([card] * (p + q))
        return out

    def per_card(self) -> dict[str, dict[str, int]]:
        return {
            c: {"n_pos": p, "n_neg": q}
            for c, p, q in zip(self.cards, self.n_pos, self.n_neg, strict=True)
        }

    def cards_counting(self, min_items: int = stats.SIGN_MIN_ITEMS) -> int:
        """Cards the sign test can count: at least ``min_items`` items and both classes."""
        return sum(
            1
            for p, q in zip(self.n_pos, self.n_neg, strict=True)
            if p + q >= min_items and p > 0 and q > 0
        )


# -- the binormal model -----------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(q: float, *, tol: float = 1e-12) -> float:
    """The standard normal quantile by bisection on :func:`norm_cdf`."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def separation_for_auroc(auroc: float) -> float:
    """The binormal separation ``d`` with ``Φ(d/√2) = auroc``."""
    if not 0.5 <= auroc < 1.0:
        raise ValueError("auroc must be in [0.5, 1)")
    return math.sqrt(2.0) * norm_ppf(auroc) if auroc > 0.5 else 0.0


def simulate_arms(
    labels: IntArray, separation: float, base_rate: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """One draw of model A (calibrated binormal scorer at ``separation``) and model B (the
    constant prior) as ``[n, 2]`` probability arrays over (positive, negative)."""
    n = len(labels)
    scores = rng.standard_normal(n) + np.where(labels == 0, separation, 0.0)
    logit_prior = math.log(base_rate / (1.0 - base_rate))
    logit_a = logit_prior + separation * scores - separation**2 / 2.0
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    probs_a = np.stack([p_a, 1.0 - p_a], axis=1)
    probs_b = np.tile([base_rate, 1.0 - base_rate], (n, 1))
    return probs_a, probs_b


# -- power and MDE ---------------------------------------------------------------------------------------


def power_at(
    pop: Population,
    auroc: float,
    *,
    metrics: Sequence[stats.Metric] = METRICS,
    n_sims: int = DEFAULT_N_SIMS,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    grid_index: int = 0,
    weights: FloatArray | None = None,
) -> dict[str, PowerPoint]:
    """Rule R's power for each metric at one simulated AUROC. Simulation ``j`` of grid point
    ``i`` draws from ``default_rng([seed, i, j])``, so any point is reproducible on its own;
    the bootstrap resample (``weights``) is the rule's own seed-0 draw for these clusters."""
    labels = pop.labels()
    clusters = pop.clusters()
    d = separation_for_auroc(auroc)
    w = stats.bootstrap_weights(clusters, B=B, seed=seed) if weights is None else weights
    passed: dict[str, int] = dict.fromkeys(metrics, 0)
    deltas: dict[str, list[float]] = {m: [] for m in metrics}
    reasons: dict[str, collections.Counter[str]] = {m: collections.Counter() for m in metrics}
    for j in range(n_sims):
        rng = np.random.default_rng([seed, grid_index, j])
        probs_a, probs_b = simulate_arms(labels, d, pop.base_rate, rng)
        for m in metrics:
            verdict = stats.rule_r(m, probs_a, probs_b, labels, clusters, B=B, seed=seed, weights=w)
            passed[m] += int(verdict.passed)
            deltas[m].append(verdict.delta)
            reasons[m][verdict.reason] += 1
    return {
        m: PowerPoint(
            auroc=auroc,
            separation=d,
            delta_mean=finite(float(np.nanmean(deltas[m]))) if deltas[m] else None,
            power=passed[m] / n_sims if n_sims else 0.0,
            passed=passed[m],
            n_sims=n_sims,
            reasons=dict(sorted(reasons[m].items())),
        )
        for m in metrics
    }


def mde_curves(
    pop: Population,
    *,
    metrics: Sequence[stats.Metric] = METRICS,
    grid: Sequence[float] = DEFAULT_GRID,
    power_target: float = POWER_TARGET,
    n_sims: int = DEFAULT_N_SIMS,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> list[MdeCurve]:
    """One :class:`MdeCurve` per metric over ``grid``: the smallest grid AUROC whose power
    reaches ``power_target`` is the MDE (``None`` when no grid point does)."""
    clusters = pop.clusters()
    weights = stats.bootstrap_weights(clusters, B=B, seed=seed) if pop.n else None
    points: dict[str, list[PowerPoint]] = {m: [] for m in metrics}
    for i, auroc in enumerate(grid):
        if pop.n == 0 or pop.n_pos == () or sum(pop.n_pos) == 0 or sum(pop.n_neg) == 0:
            break
        at = power_at(
            pop,
            auroc,
            metrics=metrics,
            n_sims=n_sims,
            B=B,
            seed=seed,
            grid_index=i,
            weights=weights,
        )
        for m in metrics:
            points[m].append(at[m])
    out: list[MdeCurve] = []
    for m in metrics:
        hit = next((p for p in points[m] if p.power >= power_target), None)
        out.append(
            MdeCurve(
                metric=m,
                population=pop.name,
                task=pop.task,
                n=pop.n,
                n_pos=sum(pop.n_pos),
                n_neg=sum(pop.n_neg),
                base_rate=finite(pop.base_rate) or 0.0,
                n_cards=len(pop.cards),
                per_card=pop.per_card(),
                cards_counting=pop.cards_counting(),
                power_target=power_target,
                grid=points[m],
                mde_auroc=hit.auroc if hit else None,
                mde_delta=(hit.auroc - 0.5 if m == "auroc" else hit.delta_mean) if hit else None,
                not_applicable=None if points[m] else "population has no items of one class",
            )
        )
    return out


def populations_for(task: Task) -> list[Population]:
    """The full item set and the novel-key subset of a two-class task."""
    full = Population.from_task(task, name="full")
    ev = evaluate_lookup(task)
    novel = Population.from_task(task, ev.idx[ev.novel].tolist(), name="novel_key")
    return [full, novel]


def run_mde(
    tasks: Mapping[str, Task],
    *,
    labels_sha256: str,
    date: str,
    name: str = "mde",
    grid: Sequence[float] = DEFAULT_GRID,
    power_target: float = POWER_TARGET,
    n_sims: int = DEFAULT_N_SIMS,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> MdeResults:
    """MDE curves for every two-class task in ``tasks`` (full and novel-key populations); a
    choice task gets one ``not_applicable`` curve."""
    curves: list[MdeCurve] = []
    for task in tasks.values():
        if not task.binary:
            curves.append(
                MdeCurve(
                    metric="auroc",
                    population="full",
                    task=task.name,
                    n=len(task.items),
                    n_pos=0,
                    n_neg=0,
                    base_rate=0.0,
                    n_cards=len(set(task.cards)),
                    per_card={},
                    cards_counting=0,
                    power_target=power_target,
                    grid=[],
                    mde_auroc=None,
                    mde_delta=None,
                    not_applicable=f"{task.task_id} has {task.k} classes; AUROC and the binormal model need two",
                )
            )
            continue
        for pop in populations_for(task):
            curves.extend(
                mde_curves(pop, grid=grid, power_target=power_target, n_sims=n_sims, B=B, seed=seed)
            )
    return MdeResults(
        date=date,
        name=name,
        mesa_clm=environment()["mesa_clm"],
        labels_sha256=labels_sha256,
        environment=environment(),
        n_sims=n_sims,
        B=B,
        seed=seed,
        grid=list(grid),
        rule=(
            "rule R: paired card-cluster bootstrap lower bound > 0 (B=2000, seed 0, one-sided 95%) "
            "and A beats B on >= ceil(0.8*m_c) of the m_c cards with >= 10 evaluable items "
            "(m_c < 4 -> insufficient_clusters)"
        ),
        curves=curves,
        notes=[
            "Label counts only: model A is a calibrated binormal scorer at the grid AUROC, model B "
            "the constant training prior (lookup_prob on novel keys). Power = fraction of simulated "
            "datasets on which rule R passes; MDE = smallest grid AUROC with power >= target.",
            "The novel_key population is the subset beats_lookup_novel is judged on; its per-card "
            "counts decide how many cards the sign test can count.",
        ],
    )


def write_mde(results: MdeResults, out_dir: str | Path = DEFAULT_OUT_DIR) -> Path:
    """Write ``<out_dir>/<date>/<name>.json`` and return its path."""
    path = Path(out_dir) / results.date / f"{results.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_mde(path: str | Path) -> MdeResults:
    return MdeResults.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
