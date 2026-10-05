"""K2: the value verdict per task (plan §8 "K2 value per task"; DESIGN "Kill and pivot criteria";
the M4 analysis plan's R2 reading, ``design/m4-analysis-plan.md``).

K2 judges each task's **best servable tier on nested cells**. The candidates are the task's
citable-form nested cells: ``calibrated`` from the committed ``bench/results/2026-10-03/tiers.json``
(``<task>.calibrated.<active framing>`` with ``selection: "nested"``; only ``column.ontology_fits``
has one, the term.fits K1 audit cells are ``selection: "none"`` and the closed choices' F7 cells
are fixed arms) and ``probe`` from ``x3.json`` (``<task>.probe.<active framing>``). The best tier
is the candidate with the lowest pooled NLL; whether the probe is ≻ the calibrated tier under
rule R is recorded beside it (promotion, R4, needs ≻; K2 does not).

The conditions, every one with its numbers, on the best tier's cell:

* **K2(a) auto-eligible** := ``beats_lookup_novel`` ∧ paired non-inferiority to AnyJev L2 on the
  full item set ∧ ECE ≤ 0.08 with cluster upper bound ≤ 0.12. The AnyJev side is
  ``x2.json#/cells/<task>.baseline.anyjev_l2.items`` joined by ``(target_sha256, option_key)``;
  the tier must pool every item of the task and every one must match (the committed join is
  full). Non-inferiority on accuracy is two conditions: Δacc = acc(tier) − acc(AnyJev) has a
  card-cluster paired bootstrap lower bound > −0.02 (``stats.non_inferior``), **and** the
  card-sign condition, read **literally** as rule R's sign test (ii): on at least ⌈0.8·m_c⌉ of
  the m_c held-out cards with ≥ 10 items the per-card Δacc > 0 (``stats.card_sign_test("acc",
  …)`` as is; m_c < 4 is ``insufficient_clusters`` and fails). The margin-shifted alternative
  (per-card Δacc > −0.02) is **reported, not gated**: ``numbers["card_sign_margin_report"]``,
  computed exactly in integers by :func:`card_sign_margin_report` (a card "wins within the
  margin" iff ``50·(correct_tier − correct_anyjev) > −n_card``, which is Δacc > −0.02 with no
  float rounding). ECE's cluster upper bound is ``stats.metric_ci("ece").upper``, the one-sided
  95% bound.
* **K2(b) proposer-only** := ``beats_lookup_novel`` alone.
* **K2(c)** otherwise (no servable candidate with pooled items included).

Strict inequalities are strict. The closed choices with K > 2 (``column.aspect``,
``avu.value_kind``) have no AUROC, so ``beats_lookup_novel`` is false and they are K2(c) by
construction; ``column.annotate`` (K = 2) has an AUROC, and its K2(c) follows from the frozen MDE
statement (one card with ≥ 10 novel-key items → ``insufficient_clusters`` on the novel-key test)
and the 100-OOF calibrator floor, not from "no AUROC" (all stated in advance; under DESIGN A6
those steps are rules anyway). **The clm-raw clause** ("a ``clm-raw`` probe ≽ a ``clm-latest`` probe"): on the ``@raw``
and ``@latest`` cells of ``x3.json``, ``@latest`` is **not** ≻ ``@raw`` on NLL (rule R) **and**
``@raw`` ≽ ``@latest`` on accuracy within 0.01 (cluster lower bound of Δacc > −0.01); recorded per
rank_fit task as ``head_adds_nothing``. No production configuration changes in the run; the
integrator records the consequences as an amendment (A7). The producer applies the M4
registration itself (R5): the committed inputs by sha256, the labels, B/seed/α.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from mesa_clm import __version__, framings
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.cells import CellError
from mesa_clm.bench.results import (
    DEFAULT_OUT_DIR,
    BenchCell,
    BenchResults,
    ResultsExist,
    environment,
    finite,
    results_path,
)
from mesa_clm.tasks import RANK_FIT_TASKS

__all__ = [
    "ACC_MARGIN",
    "ECE_MAX",
    "ECE_UPPER_MAX",
    "FORMAT",
    "HEAD_ACC_MARGIN",
    "NAME",
    "Candidate",
    "Condition",
    "HeadClause",
    "K2Results",
    "TaskVerdict",
    "Verdict",
    "card_sign_margin_report",
    "evaluate_k2",
    "load_k2",
    "markdown_k2",
    "paired_items",
    "write_k2",
]

FORMAT: Final[str] = "mesa-clm/k2/1"
NAME: Final[str] = "k2"
ACC_MARGIN: Final[float] = 0.02
HEAD_ACC_MARGIN: Final[float] = 0.01
ECE_MAX: Final[float] = 0.08
ECE_UPPER_MAX: Final[float] = 0.12
ANYJEV_FRAMING: Final[str] = "anyjev_l2"
CALIBRATED_TIER: Final[str] = "calibrated"
PROBE_TIER: Final[str] = "probe"

Verdict = Literal["a", "b", "c"]
FloatArray = npt.NDArray[np.float64]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(_Model):
    """One candidate tier: its cell's identity and the numbers K2 reads. ``eligible`` says it
    may be the best tier (a nested, servable cell with pooled items); ``reason`` why not."""

    tier: str
    key: str
    cite: str
    n: int
    nll: float | None
    acc: float | None
    ece: float | None
    beats_lookup_novel: bool | None
    selection: str
    pre_registered: bool
    exploratory: bool
    servable: bool
    feature_spec: str | None
    model: str | None
    question_key: str | None
    fingerprint: dict[str, str] | None
    eligible: bool
    reason: str | None = None


class Condition(_Model):
    """One K2 condition with every number it read."""

    name: str
    passed: bool
    reason: str
    numbers: dict[str, Any] = Field(default_factory=dict)


class HeadClause(_Model):
    """The clm-raw clause of one rank_fit task (module docstring)."""

    evaluated: bool
    reason: str | None
    latest_cite: str | None
    raw_cite: str | None
    n_paired: int
    latest_not_better_nll: dict[str, Any] | None
    raw_noninferior_acc: dict[str, Any] | None
    head_adds_nothing: bool | None


class TaskVerdict(_Model):
    task: str
    task_id: str
    shape: str
    framing: str
    candidates: list[Candidate]
    probe_vs_calibrated_nll: dict[str, Any] | None
    promotion_probe_over_calibrated: bool | None
    best: str | None
    best_cite: str | None
    best_reason: str
    conditions: dict[str, Condition]
    verdict: Verdict
    reason: str
    head_adds_nothing: HeadClause | None


class K2Results(_Model):
    format: str = FORMAT
    date: str
    name: str = NAME
    mesa_clm: str
    labels_sha256: str
    labels_content_sha256: str | None
    registered: bool
    deviations: list[str]
    inputs: dict[str, dict[str, str]]
    constants: dict[str, Any]
    environment: dict[str, Any]
    tasks: dict[str, TaskVerdict]
    notes: list[str] = Field(default_factory=list)


# -- pairing -------------------------------------------------------------------------------------


def _by_identity(cell: BenchCell) -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for it in cell.items or []:
        key = (it.target_sha256, it.option_key)
        if key in out:
            raise CellError(f"{cell.key}: identity {key[0][:12]}/{key[1]} is pooled twice")
        out[key] = it
    return out


def paired_items(
    a: BenchCell, b: BenchCell
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64], list[str], dict[str, int]]:
    """The items both cells pooled, paired by ``(target_sha256, option_key)`` in identity
    order: ``(probs_a, probs_b, labels, cards, report)``. The labels and cards of a paired item
    must agree (both come from the snapshot); the report counts each side and the pairs."""
    ia, ib = _by_identity(a), _by_identity(b)
    keys = sorted(set(ia) & set(ib))
    pa: list[list[float]] = []
    pb: list[list[float]] = []
    labels: list[int] = []
    cards: list[str] = []
    for key in keys:
        x, y = ia[key], ib[key]
        if x.label != y.label:
            raise CellError(f"{a.key} and {b.key} disagree on the label of {key[0][:12]}/{key[1]}")
        if x.card != y.card:
            raise CellError(f"{a.key} and {b.key} disagree on the card of {key[0][:12]}/{key[1]}")
        pa.append(list(x.probs))
        pb.append(list(y.probs))
        labels.append(int(x.label))
        cards.append(str(x.card))
    k = max((len(p) for p in pa), default=0)
    report = {
        "n_a": len(ia),
        "n_b": len(ib),
        "n_paired": len(keys),
        "unmatched_a": len(ia) - len(keys),
        "unmatched_b": len(ib) - len(keys),
    }
    return (
        np.asarray(pa, dtype=np.float64).reshape(len(keys), k),
        np.asarray(pb, dtype=np.float64).reshape(len(keys), k),
        np.asarray(labels, dtype=np.int64),
        cards,
        report,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return finite(value)
    return value


# -- the conditions ----------------------------------------------------------------------------------


def card_sign_margin_report(
    probs_a: npt.ArrayLike,
    probs_b: npt.ArrayLike,
    labels: npt.ArrayLike,
    clusters: Sequence[str],
    margin: float = ACC_MARGIN,
    *,
    min_items: int = stats.SIGN_MIN_ITEMS,
    fraction: float = stats.SIGN_FRACTION,
    min_clusters: int = stats.MIN_CLUSTERS,
) -> dict[str, Any]:
    """**Report-only** (never a gate): the margin-shifted card-sign variant the plan names and
    rejects as the K2(a) condition — on ≥ ⌈fraction·m_c⌉ of the m_c counting cards (≥
    ``min_items`` items) the per-card Δacc > −``margin``. Computed exactly in integers: with
    ``margin = p/q`` in lowest terms (0.02 = 1/50) and ``correct_a``, ``correct_b`` the
    correctly answered items of a card of ``n`` items, the card wins within the margin iff
    ``q·(correct_a − correct_b) > −p·n`` (for 0.02: ``50·(correct_a − correct_b) > −n``), which
    is Δacc > −margin with no float rounding. Returns the per-card counts and the verdict in
    :class:`stats.SignTest`'s vocabulary (``m_c``, ``wins``, ``needed``, ``passed``, ``reason``)."""
    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if pa.shape != pb.shape or len(y) != len(pa) or len(clusters) != len(pa):
        raise ValueError("the two arms, labels and clusters must cover the same items")
    frac = Fraction(margin).limit_denominator(10**6)
    p, q = frac.numerator, frac.denominator
    groups = np.asarray(clusters)
    hit_a = np.argmax(pa, axis=1) == y if len(pa) else np.zeros(0, dtype=bool)
    hit_b = np.argmax(pb, axis=1) == y if len(pb) else np.zeros(0, dtype=bool)
    per_card: dict[str, dict[str, Any]] = {}
    for card in sorted(set(clusters)):
        idx = np.flatnonzero(groups == card)
        n = len(idx)
        if n < min_items:
            continue
        ca, cb = int(hit_a[idx].sum()), int(hit_b[idx].sum())
        per_card[card] = {
            "n": n,
            "correct_tier": ca,
            "correct_anyjev": cb,
            "wins_within_margin": q * (ca - cb) > -p * n,
        }
    m_c = len(per_card)
    wins = sum(1 for c in per_card.values() if c["wins_within_margin"])
    needed = math.ceil(fraction * m_c)
    if m_c < min_clusters:
        passed, reason = False, "insufficient_clusters"
    else:
        passed = wins >= needed
        reason = "passed" if passed else "sign_test_failed"
    return {
        "what": "REPORT-ONLY: the margin-shifted card-sign variant (not the K2(a) gate)",
        "rule": f"per-card Δacc > −{margin} on ≥ ⌈{fraction}·m_c⌉ of the m_c cards with ≥ "
        f"{min_items} items, in integers: {q}·(correct_tier − correct_anyjev) > −{p}·n_card",
        "margin": margin,
        "m_c": m_c,
        "wins": wins,
        "needed": needed,
        "passed": passed,
        "reason": reason,
        "per_card": per_card,
    }


def _cell_arrays(cell: BenchCell) -> tuple[FloatArray, npt.NDArray[np.int64], list[str]]:
    items = cell.items or []
    k = cell.metrics and len(items[0].probs) if items else 0
    probs = np.asarray([it.probs for it in items], dtype=np.float64).reshape(len(items), k or 0)
    labels = np.asarray([it.label for it in items], dtype=np.int64)
    return probs, labels, [it.card for it in items]


def _candidate(
    tier: str, cell: BenchCell | None, cite: str, *, required_selection: str = "nested"
) -> Candidate | None:
    """The candidate of ``tier`` (``None`` without a cell): eligible only as a nested,
    ``pre_registered: true``, servable, non-exploratory cell with pooled items (an exploratory
    or unregistered cell is ineligible with the reason)."""
    if cell is None:
        return None
    m = cell.metrics
    eligible = True
    reason = None
    if cell.selection != required_selection:
        eligible, reason = False, f"selection {cell.selection!r} is not {required_selection!r}"
    elif cell.pre_registered is not True:
        eligible, reason = False, "not pre_registered"
    elif not cell.servable:
        eligible, reason = False, "not servable"
    elif m is None or not cell.items:
        eligible, reason = False, "no pooled items"
    elif cell.exploratory:
        eligible, reason = False, "exploratory"
    return Candidate(
        tier=tier,
        key=cell.key,
        cite=cite,
        n=cell.counts.n,
        nll=None if m is None else finite(m.nll),
        acc=None if m is None else finite(m.acc),
        ece=None if m is None else finite(m.ece),
        beats_lookup_novel=None if cell.baselines is None else cell.baselines.beats_lookup_novel,
        selection=cell.selection,
        pre_registered=cell.pre_registered,
        exploratory=cell.exploratory,
        servable=cell.servable,
        feature_spec=cell.feature_spec,
        model=cell.model,
        question_key=cell.question_key,
        fingerprint=cell.fingerprint,
        eligible=eligible,
        reason=reason,
    )


def _task_n(cell: BenchCell) -> int | None:
    """The task's item count the full-set condition compares with: the cell's own
    ``diagnostics.task_counts.n``, else the registered count; ``None`` when neither exists
    (the condition then fails closed with ``task_count_unknown``)."""
    counts = cell.diagnostics.get("task_counts") if cell.diagnostics else None
    if isinstance(counts, Mapping) and "n" in counts:
        return int(counts["n"])
    want = reg.current().counts.get(cell.task)
    return None if want is None else int(want.n)


def _conditions(
    best: BenchCell, anyjev: BenchCell | None, *, B: int, seed: int, alpha: float
) -> dict[str, Condition]:
    out: dict[str, Condition] = {}
    b = best.baselines
    beats = bool(b.beats_lookup_novel) if b is not None else False
    out["beats_lookup_novel"] = Condition(
        name="beats_lookup_novel",
        passed=beats,
        reason="passed"
        if beats
        else str((b.beats_detail or {}).get("reason", "false") if b else "no_baselines"),
        numbers=_json_safe({"beats_detail": b.beats_detail if b is not None else None}),
    )
    probs, labels, cards = _cell_arrays(best)
    # -- non-inferiority to AnyJev L2 on the full item set -------------------------------------
    n_task = _task_n(best)
    if anyjev is None:
        ni = Condition(name="non_inferior_acc", passed=False, reason="no_anyjev_cell")
        cs = Condition(name="card_sign", passed=False, reason="no_anyjev_cell")
    else:
        pa, pb, y, cl, report = paired_items(best, anyjev)
        numbers: dict[str, Any] = {"join": report, "n_task": n_task, "margin": ACC_MARGIN}
        if n_task is None:
            # Fail closed: without the task's count "the full item set" cannot be shown.
            why = "task_count_unknown"
            ni = Condition(name="non_inferior_acc", passed=False, reason=why, numbers=numbers)
            cs = Condition(name="card_sign", passed=False, reason=why, numbers=numbers)
        elif best.counts.n != n_task or report["n_paired"] != n_task:
            why = (
                "tier_not_on_full_item_set" if best.counts.n != n_task else "anyjev_join_incomplete"
            )
            ni = Condition(name="non_inferior_acc", passed=False, reason=why, numbers=numbers)
            cs = Condition(name="card_sign", passed=False, reason=why, numbers=numbers)
        elif report["n_paired"] == 0:
            ni = Condition(
                name="non_inferior_acc", passed=False, reason="no_paired_items", numbers=numbers
            )
            cs = Condition(
                name="card_sign", passed=False, reason="no_paired_items", numbers=numbers
            )
        else:
            res = stats.non_inferior("acc", pa, pb, y, cl, ACC_MARGIN, B=B, seed=seed, alpha=alpha)
            ni = Condition(
                name="non_inferior_acc",
                passed=res.passed,
                reason="passed" if res.passed else "acc_lower_bound_not_above_minus_margin",
                numbers=_json_safe(
                    {
                        **numbers,
                        "acc_tier": stats.metric_value("acc", pa, y),
                        "acc_anyjev": stats.metric_value("acc", pb, y),
                        **res.as_dict(),
                    }
                ),
            )
            # The gate: rule R's sign test (ii) verbatim (per-card Δacc > 0). The margin-shifted
            # variant is reported beside it, never gated.
            sign = stats.card_sign_test("acc", pa, pb, y, cl)
            cs = Condition(
                name="card_sign",
                passed=sign.passed,
                reason=sign.reason,
                numbers=_json_safe(
                    {
                        "join": report,
                        "n_task": n_task,
                        "rule": "rule R (ii) verbatim: per-card Δacc > 0 on ≥ ⌈0.8·m_c⌉ of the "
                        "m_c cards with ≥ 10 items; m_c < 4 is insufficient_clusters",
                        **sign.as_dict(),
                        "card_sign_margin_report": card_sign_margin_report(
                            pa, pb, y, cl, ACC_MARGIN
                        ),
                    }
                ),
            )
    out["non_inferior_acc"] = ni
    out["card_sign"] = cs
    # -- ECE ---------------------------------------------------------------------------------------
    m = best.metrics
    if m is None or len(labels) == 0:
        out["ece"] = Condition(name="ece", passed=False, reason="no_pooled_items")
    else:
        ece = float(m.ece)
        ci = (
            stats.metric_ci("ece", probs, labels, cards, B=B, seed=seed, alpha=alpha)
            if len(set(cards)) >= 2
            else None
        )
        upper = None if ci is None else ci.upper
        ok_point = ece <= ECE_MAX
        ok_upper = upper is not None and not math.isnan(upper) and upper <= ECE_UPPER_MAX
        reason = (
            "passed"
            if ok_point and ok_upper
            else ("ece_above_max" if not ok_point else "ece_upper_bound_above_max")
        )
        out["ece"] = Condition(
            name="ece",
            passed=ok_point and ok_upper,
            reason=reason,
            numbers=_json_safe(
                {
                    "ece": ece,
                    "ece_max": ECE_MAX,
                    "ece_upper": upper,
                    "ece_upper_max": ECE_UPPER_MAX,
                    "bootstrap": None if ci is None else ci.as_dict(),
                }
            ),
        )
    return out


def _head_clause(
    latest: BenchCell | None,
    raw: BenchCell | None,
    cite: Mapping[str, str],
    *,
    B: int,
    seed: int,
    alpha: float,
) -> HeadClause:
    if latest is None or raw is None:
        return HeadClause(
            evaluated=False,
            reason="missing @latest or @raw cell",
            latest_cite=None if latest is None else cite["latest"],
            raw_cite=None if raw is None else cite["raw"],
            n_paired=0,
            latest_not_better_nll=None,
            raw_noninferior_acc=None,
            head_adds_nothing=None,
        )
    if not latest.items or not raw.items:
        return HeadClause(
            evaluated=False,
            reason="a cell has no pooled items",
            latest_cite=cite["latest"],
            raw_cite=cite["raw"],
            n_paired=0,
            latest_not_better_nll=None,
            raw_noninferior_acc=None,
            head_adds_nothing=None,
        )
    pl, pr, y, cl, report = paired_items(latest, raw)
    better = stats.rule_r("nll", pl, pr, y, cl, B=B, seed=seed, alpha=alpha)
    ni = stats.non_inferior("acc", pr, pl, y, cl, HEAD_ACC_MARGIN, B=B, seed=seed, alpha=alpha)
    return HeadClause(
        evaluated=True,
        reason=None,
        latest_cite=cite["latest"],
        raw_cite=cite["raw"],
        n_paired=int(report["n_paired"]),
        latest_not_better_nll=_json_safe({**better.as_dict(), "join": report}),
        raw_noninferior_acc=_json_safe(ni.as_dict()),
        head_adds_nothing=(not better.passed) and ni.passed,
    )


def _verdict(conditions: Mapping[str, Condition]) -> tuple[Verdict, str]:
    a = all(
        conditions[k].passed for k in ("beats_lookup_novel", "non_inferior_acc", "card_sign", "ece")
    )
    if a:
        return "a", "auto-eligible: every K2(a) condition passed"
    if conditions["beats_lookup_novel"].passed:
        failed = [k for k in ("non_inferior_acc", "card_sign", "ece") if not conditions[k].passed]
        return "b", "proposer-only: beats_lookup_novel passed; failed " + ", ".join(failed)
    return "c", "killed: beats_lookup_novel failed (" + conditions[
        "beats_lookup_novel"
    ].reason + ")"


# -- the evaluation -------------------------------------------------------------------------------------


def _cite(path: str, key: str) -> str:
    return f"{path}#{key}"


def evaluate_task(
    task: str,
    task_id: str,
    *,
    tiers: BenchResults,
    x2: BenchResults,
    x3: BenchResults,
    cites: Mapping[str, str],
    B: int,
    seed: int,
    alpha: float,
) -> TaskVerdict:
    """K2 for one task (module docstring)."""
    framing_id = framings.ACTIVE[task_id]
    shape = "rank_fit" if task_id in RANK_FIT_TASKS else "choice"
    cal_key = f"{task}.{CALIBRATED_TIER}.{framing_id}"
    probe_key = f"{task}.{PROBE_TIER}.{framing_id}"
    cal_cell = tiers.cells.get(cal_key)
    probe_cell = x3.cells.get(probe_key)
    candidates = [
        c
        for c in (
            _candidate(CALIBRATED_TIER, cal_cell, _cite(cites["tiers"], cal_key)),
            _candidate(PROBE_TIER, probe_cell, _cite(cites["x3"], probe_key)),
        )
        if c is not None
    ]
    cells = {CALIBRATED_TIER: cal_cell, PROBE_TIER: probe_cell}
    # -- rule R probe vs calibrated on NLL (recorded; promotion needs it, K2 does not) ----------
    vs: dict[str, Any] | None = None
    promotion: bool | None = None
    if probe_cell is not None and cal_cell is not None and probe_cell.items and cal_cell.items:
        pp, pc, y, cl, report = paired_items(probe_cell, cal_cell)
        if len(y):
            rr = stats.rule_r("nll", pp, pc, y, cl, B=B, seed=seed, alpha=alpha)
            vs = _json_safe({**rr.as_dict(), "join": report})
            promotion = rr.passed
    # -- the best servable tier: lowest pooled NLL among the eligible candidates ---------------
    eligible = [c for c in candidates if c.eligible and c.nll is not None]
    best_cand = min(eligible, key=lambda c: (float(c.nll or 0.0), c.tier)) if eligible else None
    if best_cand is None:
        why = "no eligible candidate: " + (
            "; ".join(f"{c.tier} ({c.reason})" for c in candidates) or "no nested cell"
        )
        conditions = {
            name: Condition(name=name, passed=False, reason="no_candidate")
            for name in ("beats_lookup_novel", "non_inferior_acc", "card_sign", "ece")
        }
        verdict: Verdict = "c"
        reason = "killed: " + why
        best_cell = None
    else:
        best_cell = cells[best_cand.tier]
        if best_cell is None:  # pragma: no cover - a candidate always has its cell
            raise CellError(f"{task}: the best tier {best_cand.tier} has no cell")
        why = f"lowest pooled NLL among {[c.tier for c in eligible]}"
        anyjev = x2.cells.get(f"{task}.baseline.{ANYJEV_FRAMING}")
        conditions = _conditions(best_cell, anyjev, B=B, seed=seed, alpha=alpha)
        verdict, reason = _verdict(conditions)
    head: HeadClause | None = None
    if shape == "rank_fit":
        latest_key, raw_key = f"{probe_key}@latest", f"{probe_key}@raw"
        head = _head_clause(
            x3.cells.get(latest_key),
            x3.cells.get(raw_key),
            {"latest": _cite(cites["x3"], latest_key), "raw": _cite(cites["x3"], raw_key)},
            B=B,
            seed=seed,
            alpha=alpha,
        )
    return TaskVerdict(
        task=task,
        task_id=task_id,
        shape=shape,
        framing=framing_id,
        candidates=candidates,
        probe_vs_calibrated_nll=vs,
        promotion_probe_over_calibrated=promotion,
        best=None if best_cand is None else best_cand.tier,
        best_cite=None if best_cand is None else best_cand.cite,
        best_reason=why,
        conditions=conditions,
        verdict=verdict,
        reason=reason,
        head_adds_nothing=head,
    )


def _task_ids(x3: BenchResults, tiers: BenchResults) -> dict[str, str]:
    out: dict[str, str] = {}
    for res in (tiers, x3):
        for cell in res.cells.values():
            out.setdefault(cell.task, cell.task_id)
    return dict(sorted(out.items()))


def evaluate_k2(
    *,
    tiers: BenchResults,
    x2: BenchResults,
    x3: BenchResults,
    date: str,
    inputs: Mapping[str, Mapping[str, str]],
    registered: bool,
    deviations: Sequence[str] = (),
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
    alpha: float = stats.DEFAULT_ALPHA,
) -> K2Results:
    """The ``k2`` results (module docstring) over the three results files. ``inputs`` maps
    ``tiers``, ``x2`` and ``x3`` to ``{"path", "sha256"}`` (the cites name the paths).
    ``registered`` (with ``deviations``) says the inputs are the pinned ones; the producer's own
    checks (the three files about the same labels, ``x3`` pre-registered, B/seed/α) can only
    make it false."""
    own: list[str] = [
        *reg.run_deviations(
            labels_sha256=x3.labels_sha256,
            labels_content_sha256=x3.labels_content_sha256,
            B=B,
            seed=seed,
            alpha=alpha,
        )
    ]
    for name, res in (("tiers", tiers), ("x2", x2)):
        if res.labels_sha256 != x3.labels_sha256:
            own.append(f"{name}.json labels_sha256 {res.labels_sha256[:12]} is not x3.json's")
        if res.labels_content_sha256 != x3.labels_content_sha256:
            own.append(f"{name}.json labels_content_sha256 is not x3.json's")
    if any(not c.pre_registered for c in x3.cells.values()):
        own.append("x3.json is not the pre-registered run")
    if own:
        registered, deviations = False, [*deviations, *(d for d in own if d not in deviations)]
    cites = {k: str(v["path"]) for k, v in inputs.items()}
    tasks = {
        task: evaluate_task(
            task, task_id, tiers=tiers, x2=x2, x3=x3, cites=cites, B=B, seed=seed, alpha=alpha
        )
        for task, task_id in _task_ids(x3, tiers).items()
    }
    notes = [
        "K2 per task (plan §8; design/m4-analysis-plan.md R2): the best servable tier among the "
        "citable-form nested cells (calibrated from tiers.json, probe from x3.json) is the one "
        "with the lowest pooled NLL; (a) auto-eligible needs beats_lookup_novel, paired "
        "non-inferiority to AnyJev L2 on the full item set (Δacc cluster lower bound > −0.02 and "
        "the card-sign condition read literally: rule R's sign test, per-card Δacc > 0 on ≥ "
        "⌈0.8·m_c⌉ of the cards with ≥ 10 items; the margin-shifted variant, per-card Δacc > "
        "−0.02 in integers, is reported as card_sign_margin_report and never gated) and ECE ≤ "
        "0.08 with cluster upper bound ≤ 0.12; (b) proposer-only needs beats_lookup_novel "
        "alone; (c) otherwise.",
        "column.aspect and avu.value_kind (K > 2) have no AUROC, so beats_lookup_novel is false "
        "and they are K2(c) by construction; column.annotate (K = 2) has an AUROC and its K2(c) "
        "follows from the frozen MDE statement (one card with ≥ 10 novel-key items → "
        "insufficient_clusters) and the 100-OOF calibrator floor (all stated in advance). "
        "head_adds_nothing: @latest not ≻ @raw on NLL and @raw ≽ @latest on acc within 0.01, "
        "per rank_fit task. No production configuration changes here; the integrator records "
        "the consequences as an amendment.",
        "Silver labels are four-model agreement, not truth.",
    ]
    if not registered:
        notes.insert(
            0,
            "NOT the pre-registered evaluation: " + "; ".join(deviations or ["registered=false"]),
        )
    return K2Results(
        date=date,
        mesa_clm=__version__,
        labels_sha256=x3.labels_sha256,
        labels_content_sha256=x3.labels_content_sha256,
        registered=registered,
        deviations=list(deviations),
        inputs={k: dict(v) for k, v in inputs.items()},
        constants={
            "acc_margin": ACC_MARGIN,
            "head_acc_margin": HEAD_ACC_MARGIN,
            "ece_max": ECE_MAX,
            "ece_upper_max": ECE_UPPER_MAX,
            "sign_fraction": stats.SIGN_FRACTION,
            "sign_min_items": stats.SIGN_MIN_ITEMS,
            "min_clusters": stats.MIN_CLUSTERS,
            "B": B,
            "seed": seed,
            "alpha": alpha,
        },
        environment=environment(),
        tasks=tasks,
        notes=notes,
    )


# -- files ---------------------------------------------------------------------------------------------


def markdown_k2(results: K2Results) -> str:
    """The ``.md`` beside ``k2.json``: one row per task, every number naming its JSON."""
    lines = [
        f"# bench {results.date} · k2 · mesa-clm {results.mesa_clm}",
        "",
        f"Every number below names `bench/results/{results.date}/k2.json` (labels_sha256 "
        f"`{results.labels_sha256[:12]}…`). Silver labels are four-model agreement, not truth.",
        "",
        "| task | best tier | NLL | acc | ECE | beats_lookup_novel | non-inferior acc (LB) | "
        "card sign | ECE UB | verdict | head_adds_nothing |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task, t in results.tasks.items():
        best = next((c for c in t.candidates if c.tier == t.best), None)
        c = t.conditions
        ni = c["non_inferior_acc"].numbers.get("lower_bound")
        ub = c["ece"].numbers.get("ece_upper")
        head = "" if t.head_adds_nothing is None else str(t.head_adds_nothing.head_adds_nothing)
        lines.append(
            f"| {task} | {t.best or '-'} | {_f(best.nll if best else None)} | "
            f"{_f(best.acc if best else None)} | {_f(best.ece if best else None)} | "
            f"{c['beats_lookup_novel'].passed} | {c['non_inferior_acc'].passed} "
            f"({_f(ni)}) | {c['card_sign'].passed} ({c['card_sign'].reason}) | {_f(ub)} | "
            f"**{t.verdict}** | {head} |"
        )
    for note in results.notes:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"


def _f(value: Any, digits: int = 3) -> str:
    return (
        "" if value is None or not isinstance(value, int | float) else f"{float(value):.{digits}f}"
    )


def write_k2(
    results: K2Results, out_dir: str | Path = DEFAULT_OUT_DIR, *, force: bool = False
) -> Path:
    """Write ``<out_dir>/<date>/k2.json`` and its ``.md``; refuses to replace an existing file
    unless ``force`` (one run per file)."""
    path = results_path(out_dir, results.date, results.name)
    existing = [p for p in (path, path.with_suffix(".md")) if p.exists()]
    if existing and not force:
        raise ResultsExist(
            f"{existing[0]} exists: one run per results file (pass --force to replace)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown_k2(results), encoding="utf-8")
    return path


def load_k2(path: str | Path) -> K2Results:
    return K2Results.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
