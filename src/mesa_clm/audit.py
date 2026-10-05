"""Curator audits of production decisions (plan §4.7 "auto_requires_audit", §8 M4 "Production
audit"; DESIGN D21, D30; ``design/m4-analysis-plan.md`` C4).

``audit sample`` draws, from the sidecar's decisions of annotate runs on **non-bench** cards
(any :data:`~mesa_clm.learn.labels.BENCH_CARDS` card or a card with silver labels is refused:
an audit on a bench card would mint curator labels on a pre-registered item), ``n`` decisions
in three strata of equal shares: **would-be-auto** (the policy statistic at or above the task's
cited cell's ``threshold_cp[risk]`` when the task cites a cell, else the top decile of the
task's statistics in the pool), **proposed** (outcome ``proposed``, not would-be-auto; the
``ols_rank`` proposals of K1 among them) and **anchor-abstain** (outcome ``abstain`` with reason
``anchor_won``). By default the pool is the decisions that carry an ``artifact_version`` (the
promoted artifacts' decisions: only those can feed an ``audits`` row, C4.3), and the top-decile
rule is computed per task over that pool alone; ``--all-tiers`` (``artifact_only=False``) adds
the ``zero_shot`` and ``ols_rank`` decisions for the report-only precision (a ``zero_shot``
``p_fit`` is ``σ(s_c)``, uncalibrated, and saturates near 1, so its "top decile" says nothing).
The draw is seeded (:data:`SAMPLE_SEED`) and must span at least ``min_cards`` cards. The sample
file (:data:`FORMAT`) holds ids only, no card content, the sampling mode (``artifact_only``) and
the reviewer's checklist (:data:`CHECKLIST`).

``audit review --file`` (the CLI, at a terminal only, like ``review``) shows each decision with
what a curator needs to judge it (:func:`context_lines`: the scope and target, the column's
description, dtype and unit from the stored ``state_json``, the aspect and ontology the group
searched and its OLS queries, for ``column.ontology_fits`` that the candidates are ontologies,
and the tier marked plainly by :func:`tier_note`), then its candidates, and records ``correct``
or ``incorrect``: a curator label row (``curator`` via ``cli``, origin ``audit:<audit_id>``,
``fold_eligible=false``, ``bench_card=false``) for the answered option of a rank_fit decision
(Yes when correct, No when incorrect) or for a correct closed choice (an incorrect one names no
true class, so it counts as an error without a label), written through
:func:`labels_for_verdict`.

``audit record --file`` writes one ``audits`` row per ``(task_key, artifact_version)`` among the
reviewed would-be-auto decisions (:func:`audit_rows`: ``n``, ``n_cards``, ``cards``, ``reviewer``,
``n_errors``, ``cp95_upper = clopper_pearson_upper(n_errors, n)``, ``risk`` from the policy,
``passed`` = ``n >= 50 ∧ n_cards >= 3 ∧ cp95_upper <= 2·risk``, enforced by
:class:`~mesa_clm.provenance.models.AuditRow` too). Decisions that apply no artifact
(``zero_shot``, ``ols_rank``) get no row: an audit is of an artifact; a sample drawn with
``--all-tiers`` reports how many reviewed items had none (:func:`summarize`,
``reviewed_without_artifact``). The proposed-precision Clopper-Pearson interval is printed,
report-only (:func:`summarize`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mesa_clm.bench.stats import clopper_pearson_upper
from mesa_clm.identity import identity
from mesa_clm.learn.labels import WEIGHTS, product_code_of
from mesa_clm.perms import write_private_text
from mesa_clm.policy import Policy, parse_cite
from mesa_clm.provenance.labels import LabelRow
from mesa_clm.provenance.models import AuditRow, audit_passes
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import TASKS

__all__ = [
    "CHECKLIST",
    "DEFAULT_MIN_CARDS",
    "DEFAULT_N",
    "FORMAT",
    "SAMPLE_SEED",
    "STRATA",
    "TOP_DECILE",
    "AuditError",
    "AuditItem",
    "AuditSample",
    "audit_rows",
    "build_sample",
    "candidates",
    "context_lines",
    "labels_for_verdict",
    "load_sample",
    "proposed_precision",
    "stratified_sample",
    "summarize",
    "threshold_for",
    "tier_note",
    "write_sample",
]

FORMAT: Final = "mesa-clm/audit-sample/1"
Stratum = Literal["would_be_auto", "proposed", "anchor_abstain"]
STRATA: Final[tuple[Stratum, ...]] = ("would_be_auto", "proposed", "anchor_abstain")
Verdict = Literal["correct", "incorrect"]
# Plan §8 M4: ``audit sample --n 100`` over at least 5 non-bench cards; the draw's seed.
DEFAULT_N: Final[int] = 100
DEFAULT_MIN_CARDS: Final[int] = 5
SAMPLE_SEED: Final[int] = 0
# Without a cited cell the would-be-auto stratum is the top decile of the task's statistics.
TOP_DECILE: Final[float] = 0.90
# The outcomes a would-be-auto or proposed decision may carry (a later human answer leaves the
# decision's own outcome; an audit reviews what the policy produced).
_POLICY_OUTCOMES: Final[frozenset[str]] = frozenset({"proposed", "auto"})
_NO_DISTRIBUTION: Final[frozenset[str]] = frozenset(
    {"rule", "planner", "claude:structured_output", "unavailable"}
)
CHECKLIST: Final[tuple[str, ...]] = (
    "Judge the decision as the card shows it: is the answered option the right annotation for "
    "this column, site or dataset?",
    "A rank_fit decision is correct when its answered CURIE is a right term for the target; "
    "incorrect when a better term exists among the offered candidates or the anchor was right.",
    "An anchor abstain is correct when none of the offered candidates fits; incorrect when one "
    "of them (shown) was the right term.",
    "A closed choice is correct when its answered option is the right class for the column.",
    "Skip what you cannot judge; a skipped item is neither correct nor incorrect.",
    "Your answers are curator labels (weight 1.0) outside every fold; never review a bench card.",
)


class AuditError(ValueError):
    """The sample cannot be drawn or recorded as asked (a bench card, too few cards, a file that
    is not a sample, an unreviewed sample)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuditItem(_Model):
    """One sampled decision: ids and the stratum (never card content), and once reviewed the
    verdict, when, and the labels it minted."""

    decision_id: str
    run_id: str
    group_id: str | None = None
    task_id: str
    task_key: str
    artifact_version: str | None = None
    level: str
    method: str
    outcome: str
    stratum: Stratum
    card: str
    stat: float | None = None
    verdict: Verdict | None = None
    reviewed_at: str | None = None
    labels_written: int = 0


class AuditSample(_Model):
    """The sample file (:data:`FORMAT`)."""

    format: Literal["mesa-clm/audit-sample/1"] = FORMAT
    audit_id: str
    created_at: str
    n: int = Field(ge=1)
    min_cards: int = Field(ge=1)
    seed: int
    runs: list[str]
    cards: list[str]
    strata: dict[str, int]
    thresholds: dict[str, float | None] = Field(default_factory=dict)
    # The sampling mode: ``True`` (the default) draws only decisions that carry an
    # ``artifact_version``; ``False`` (``--all-tiers``) draws every decision with a distribution.
    artifact_only: bool = True
    items: list[AuditItem]
    checklist: list[str] = Field(default_factory=lambda: list(CHECKLIST))
    reviewer: str | None = None

    @property
    def reviewed(self) -> list[AuditItem]:
        return [i for i in self.items if i.verdict is not None]


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


# -- the pool ---------------------------------------------------------------------------------


def threshold_for(policy: Policy, task_id: str, results_root: str | Path) -> float | None:
    """The task's would-be-auto threshold: the cited cell's ``threshold_cp[risk]`` when the
    policy cites one (``None`` when it does not, when the file or cell is missing, or when the
    cell has no threshold at that risk)."""
    from mesa_clm.bench.results import load_results

    t = policy.thresholds(task_id)
    ref = parse_cite(t.cite)
    if ref is None:
        return None
    path = Path(results_root) / ref.path
    if not path.is_file():
        return None
    try:
        cell = load_results(path).cells.get(f"{ref.task}.{ref.tier}.{ref.framing}")
    except (ValueError, OSError):
        return None
    if cell is None or cell.metrics is None:
        return None
    return cell.metrics.threshold_cp.get(f"{t.risk:.2f}")


def _stat(row: Mapping[str, Any]) -> float | None:
    value = row.get("p_fit") if row.get("shape") == "rank_fit" else row.get("confidence")
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def candidates(
    store: Any,
    runs: Iterable[Mapping[str, Any]],
    *,
    policy: Policy,
    results_root: str | Path,
    is_bench_card: Callable[[str], bool],
    artifact_only: bool = True,
) -> tuple[list[AuditItem], dict[str, float | None]]:
    """Every decision of ``runs`` eligible for an audit, with its stratum (module docstring),
    and the would-be-auto threshold applied per task (``None``: the top decile, computed per
    task over the returned pool). A run on a bench card is :class:`AuditError`; decisions
    without a distribution (rules, planners, unavailable) are left out, and so are, by default
    (``artifact_only``), decisions without an ``artifact_version`` (``zero_shot``,
    ``ols_rank``): they can feed no ``audits`` row."""
    thresholds: dict[str, float | None] = {}
    rows: list[tuple[Mapping[str, Any], str]] = []
    for run in runs:
        card = str(run.get("card_name") or "")
        if is_bench_card(card):
            raise AuditError(
                f"run {run.get('run_id')} is on {card or 'an unnamed card'}: a bench card (or one "
                "with silver labels) is never audited (D30)"
            )
        for d in store.decisions(run["run_id"]):
            if d.get("method") in _NO_DISTRIBUTION:
                continue
            if d.get("task_id") not in TASKS:
                continue
            if artifact_only and not d.get("artifact_version"):
                continue
            rows.append((d, card))
    stats_by_task: dict[str, list[float]] = {}
    for d, _ in rows:
        s = _stat(d)
        if s is not None and d.get("outcome") in _POLICY_OUTCOMES:
            stats_by_task.setdefault(str(d["task_id"]), []).append(s)
    cut: dict[str, float | None] = {}
    for task_id, values in stats_by_task.items():
        thresholds[task_id] = threshold_for(policy, task_id, results_root)
        cut[task_id] = (
            thresholds[task_id]
            if thresholds[task_id] is not None
            else float(np.quantile(np.asarray(values, dtype=np.float64), TOP_DECILE))
        )
    out: list[AuditItem] = []
    for d, card in rows:
        task_id = str(d["task_id"])
        outcome = str(d.get("outcome"))
        stat = _stat(d)
        stratum: Stratum | None = None
        if outcome == "abstain" and d.get("reason") == "anchor_won":
            stratum = "anchor_abstain"
        elif outcome in _POLICY_OUTCOMES:
            limit = cut.get(task_id)
            if stat is not None and limit is not None and stat >= limit:
                stratum = "would_be_auto"
            elif outcome == "proposed":
                stratum = "proposed"
        if stratum is None:
            continue
        out.append(
            AuditItem(
                decision_id=str(d["decision_id"]),
                run_id=str(d["run_id"]),
                group_id=None if d.get("group_id") is None else str(d["group_id"]),
                task_id=task_id,
                task_key=str(d["task_key"]),
                artifact_version=d.get("artifact_version"),
                level=str(d.get("level")),
                method=str(d.get("method")),
                outcome=outcome,
                stratum=stratum,
                card=card,
                stat=stat,
            )
        )
    return out, thresholds


def stratified_sample(
    pool: Sequence[AuditItem], *, n: int, min_cards: int, seed: int = SAMPLE_SEED
) -> list[AuditItem]:
    """``n`` items in equal shares per stratum (the remainder to the first strata; a stratum
    short of its share gives the rest to the others, in order), drawn with
    ``numpy.random.default_rng(seed)`` from each stratum sorted by decision id and then
    filled card-spanning (:func:`_take_spanning`: each pick prefers a card the sample has not
    got yet, so a draw spans the cards the pool allows and the ``min_cards`` check
    fails only when the pool itself cannot span them); refused when the pool is empty or the
    sample spans fewer than ``min_cards`` cards."""
    if n < 1 or min_cards < 1:
        raise AuditError("n and min_cards must be positive")
    if not pool:
        raise AuditError("no eligible decision to audit")
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[AuditItem]] = {s: [] for s in STRATA}
    for item in sorted(pool, key=lambda i: i.decision_id):
        by_stratum[item.stratum].append(item)
    shuffled = {
        s: [items[i] for i in rng.permutation(len(items))] for s, items in by_stratum.items()
    }
    shares = {s: n // len(STRATA) + (1 if i < n % len(STRATA) else 0) for i, s in enumerate(STRATA)}
    seen: set[str] = set()
    chosen: dict[str, list[AuditItem]] = {}
    for s in STRATA:
        chosen[s] = _take_spanning(shuffled[s], shares[s], seen)
    spare = n - sum(len(c) for c in chosen.values())
    for s in STRATA:  # a stratum short of its share passes the rest on, in order
        rest = [i for i in shuffled[s] if i not in chosen[s]]
        extra = _take_spanning(rest, min(spare, len(rest)), seen)
        chosen[s] = chosen[s] + extra
        spare -= len(extra)
    out = [i for s in STRATA for i in chosen[s]]
    cards = {i.card for i in out}
    if len(cards) < min_cards:
        raise AuditError(
            f"the sample spans {len(cards)} card(s), fewer than --min-cards {min_cards}; "
            "audit more runs"
        )
    return out


def _take_spanning(items: Sequence[AuditItem], k: int, seen: set[str]) -> list[AuditItem]:
    """The first ``k`` of ``items`` (a shuffled stratum) in a card-spanning order: each pick is
    the earliest remaining item whose card the sample has not got yet (``seen``, updated), else
    the earliest remaining item. So a sample spans as many distinct cards as its strata's pools
    allow and the ``min_cards`` check fails only when the pool itself cannot span them.
    Deterministic in ``items``."""
    remaining = list(items)
    out: list[AuditItem] = []
    while remaining and len(out) < k:
        pick = next((i for i in remaining if i.card not in seen), remaining[0])
        remaining.remove(pick)
        seen.add(pick.card)
        out.append(pick)
    return out


def build_sample(
    pool: Sequence[AuditItem],
    *,
    runs: Sequence[str],
    thresholds: Mapping[str, float | None],
    n: int = DEFAULT_N,
    min_cards: int = DEFAULT_MIN_CARDS,
    seed: int = SAMPLE_SEED,
    artifact_only: bool = True,
) -> AuditSample:
    """The sample file's content from a pool :func:`candidates` drew in the same mode
    (``artifact_only`` is recorded, not applied here). An empty artifact-only pool names the
    way out."""
    if artifact_only and not pool:
        raise AuditError(
            "no artifact-backed decision to audit in these runs (only decisions of a promoted "
            "artifact can feed an audits row); --all-tiers includes the zero_shot and ols_rank "
            "decisions for the report-only precision"
        )
    items = stratified_sample(pool, n=n, min_cards=min_cards, seed=seed)
    strata = {str(s): sum(1 for i in items if i.stratum == s) for s in STRATA}
    return AuditSample(
        audit_id=str(uuid4()),
        created_at=_now(),
        n=n,
        min_cards=min_cards,
        seed=seed,
        runs=list(runs),
        cards=sorted({i.card for i in items}),
        strata=strata,
        thresholds=dict(thresholds),
        artifact_only=artifact_only,
        items=items,
    )


def write_sample(path: str | Path, sample: AuditSample) -> Path:
    """The sample file, owner-only (ids only, no card content)."""
    return write_private_text(
        Path(path).expanduser(),
        json.dumps(sample.model_dump(mode="json"), indent=1, sort_keys=True) + "\n",
    )


def load_sample(path: str | Path) -> AuditSample:
    p = Path(path).expanduser()
    try:
        return AuditSample.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise AuditError(f"{p}: no such sample file") from None
    except (OSError, ValueError, ValidationError) as exc:  # JSON and UTF-8 errors are ValueErrors
        raise AuditError(f"{p}: not an audit sample ({type(exc).__name__})") from None


# -- what the reviewer sees ------------------------------------------------------------------

TASK_ONTOLOGY_FITS: Final = "column.ontology_fits"
_ZERO_SHOT_RANK_NOTE: Final = "uncalibrated: p_fit is σ(s_c), saturates near 1"
_ZERO_SHOT_CHOICE_NOTE: Final = "uncalibrated: confidence is the raw softmax maximum"


def _text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "-"


def tier_note(decision: Mapping[str, Any]) -> str:
    """The tier of a decision, marked plainly for a reviewer: ``probe/clm v1 (platt)`` for a
    decision of a promoted artifact (level/method, the artifact version, the calibration);
    ``zero_shot/clm (uncalibrated: p_fit is σ(s_c), saturates near 1)`` for a zero-shot
    rank_fit (a choice names its raw softmax instead); ``ols_rank (degraded: OLS rank order,
    no p_fit)`` for K1's fallback."""
    level = _text(decision.get("level"))
    method = _text(decision.get("method"))
    if method == "ols_rank":
        return "ols_rank (degraded: OLS rank order, no p_fit)"
    version = decision.get("artifact_version")
    if version:
        return f"{level}/{method} {version} ({_text(decision.get('calibration'))})"
    if level == "zero_shot":
        note = (
            _ZERO_SHOT_RANK_NOTE if decision.get("shape") == "rank_fit" else _ZERO_SHOT_CHOICE_NOTE
        )
        return f"{level}/{method} ({note})"
    return f"{level}/{method} ({_text(decision.get('calibration'))}, no artifact)"


def context_lines(decision: Mapping[str, Any], group: Mapping[str, Any] | None = None) -> list[str]:
    """What a curator needs before the candidates (module docstring): the scope and target
    (the column name, the site code or the dataset), the column's description, dtype and unit
    from the decision's ``state_json`` (its ``column`` block; a site's name, domain and habitat
    for a site), the aspect and ontology the group searched and its OLS ``queries`` (from the
    group's ``search_json``; the unit table and a D24 refinement are named as such), for
    ``column.ontology_fits`` the aspect and that the candidates are ontologies, and the tier
    (:func:`tier_note`). Two-space indented lines, ready for the terminal."""
    state: Mapping[str, Any] = decision.get("state_json") or {}
    column: Mapping[str, Any] = state.get("column") or {}
    site: Mapping[str, Any] = state.get("site") or {}
    card: Mapping[str, Any] = state.get("card") or {}
    scope = _text(decision.get("scope") or state.get("scope"))
    column_name = decision.get("column_name") or column.get("name")
    site_code = decision.get("site_code") or site.get("code")
    lines: list[str] = []
    if column_name:
        lines.append(f"  target: column {column_name}  (scope {scope})")
        lines.append(f"  description: {_text(column.get('description'))}")
        lines.append(f"  dtype: {_text(column.get('dtype'))}  unit: {_text(column.get('unit'))}")
    elif site_code:
        lines.append(f"  target: site {site_code}  (scope {scope})")
        lines.append(
            f"  site: {_text(site.get('name'))}  domain: {_text(site.get('domain'))}  "
            f"habitat: {_text(site.get('habitat'))}"
        )
    else:
        lines.append(f"  target: dataset {_text(card.get('dataset'))}  (scope {scope})")
        lines.append(f"  product: {_text(card.get('product_title'))}")
    task_id = str(decision.get("task_id") or "")
    search: Mapping[str, Any] = (group or {}).get("search_json") or {}
    aspect = (group or {}).get("aspect") or state.get("aspect") or search.get("aspect")
    if task_id == TASK_ONTOLOGY_FITS:
        lines.append(
            f"  aspect: {_text(aspect)}  (the candidates are ontologies of the registry, "
            "not terms; the question is which ontology fits this column for the aspect)"
        )
    elif decision.get("shape") == "rank_fit":
        ontology = (group or {}).get("ontology_id") or search.get("ontology_id")
        lines.append(f"  aspect: {_text(aspect)}  ontology: {_text(ontology)}")
        if search.get("specificity_of"):
            lines.append(
                f"  refines: {search['specificity_of']} (D24: the candidates are the winner "
                "and its OLS children)"
            )
        queries = [str(q) for q in (search.get("queries") or []) if q]
        if search.get("table"):
            lines.append(f"  OLS queries: unit table lookup of {' | '.join(queries) or '-'}")
        elif group is not None:
            lines.append(f"  OLS queries: {' | '.join(queries) if queries else '-'}")
    lines.append(f"  tier: {tier_note(decision)}")
    return lines


# -- verdicts --------------------------------------------------------------------------------


def labels_for_verdict(
    decision: Mapping[str, Any],
    run: Mapping[str, Any],
    verdict: Verdict,
    *,
    actor: str,
    audit_id: str,
) -> list[LabelRow]:
    """The curator label rows a verdict mints (module docstring): the answered option of a
    rank_fit decision (Yes/No), a correct closed choice's answer; nothing for an incorrect
    closed choice, an abstain without an answer, or an anchor answer judged correct (the
    anchor-positive row would need the whole offered set, which ``review`` records)."""
    task_id = str(decision["task_id"])
    task = TASKS[task_id]
    answer = str(decision.get("answer") or "")
    if not answer or int(decision.get("answer_index", -1)) < 0:
        return []
    state: dict[str, Any] = dict(decision.get("state_json") or {})
    card = str(run.get("card_name") or "")
    product = product_code_of(card)
    if decision.get("shape") == "rank_fit":
        if answer == ANCHOR_KEY:
            return []
        option, index = answer, (0 if verdict == "correct" else 1)
    else:
        if verdict != "correct":
            return []
        option, index = "", task.index_of(answer)
    ident = identity(task_id, state, option)
    return [
        LabelRow(
            task_id=task_id,
            task_key=ident.task_key,
            target_sha256=ident.target_sha256,
            option_key=ident.option_key,
            label_source="curator",
            label=task.options[index],
            label_index=index,
            weight=WEIGHTS["curator"],
            state_sha256=str(decision["state_sha256"]),
            state_json=state,
            card=card,
            product_code=product,
            leak_group=product,
            fold_eligible=False,
            bench_card=False,
            origin=f"audit:{audit_id}",
            actor=actor,
        )
    ]


# -- the record ------------------------------------------------------------------------------


def proposed_precision(items: Sequence[AuditItem]) -> dict[str, Any]:
    """The precision of the reviewed items with its Clopper-Pearson bounds (report-only): the
    point, the one-sided 95% lower bound (``1 − cp95_upper(errors)``) and upper bound
    (``cp95_upper(correct)``); ``None`` without reviewed items."""
    reviewed = [i for i in items if i.verdict is not None]
    n = len(reviewed)
    if n == 0:
        return {"n": 0, "n_errors": 0, "precision": None, "lower": None, "upper": None}
    errors = sum(1 for i in reviewed if i.verdict == "incorrect")
    return {
        "n": n,
        "n_errors": errors,
        "precision": (n - errors) / n,
        "lower": 1.0 - clopper_pearson_upper(errors, n),
        "upper": clopper_pearson_upper(n - errors, n),
    }


def summarize(sample: AuditSample) -> dict[str, Any]:
    """Per stratum the reviewed counts and precision (:func:`proposed_precision`), and per
    ``(task_key, artifact_version)`` the would-be-auto audit numbers."""
    by_stratum = {
        s: proposed_precision([i for i in sample.items if i.stratum == s]) for s in STRATA
    }
    groups: dict[str, dict[str, Any]] = {}
    for i in sample.items:
        if i.stratum != "would_be_auto" or i.verdict is None:
            continue
        key = f"{i.task_key}@{i.artifact_version or 'none'}"
        g = groups.setdefault(
            key,
            {
                "task_id": i.task_id,
                "task_key": i.task_key,
                "artifact_version": i.artifact_version,
                "n": 0,
                "n_errors": 0,
                "cards": set(),
            },
        )
        g["n"] += 1
        g["n_errors"] += int(i.verdict == "incorrect")
        g["cards"].add(i.card)
    return {
        "audit_id": sample.audit_id,
        "artifact_only": sample.artifact_only,
        "reviewed": len(sample.reviewed),
        "reviewed_without_artifact": sum(1 for i in sample.reviewed if not i.artifact_version),
        "strata": by_stratum,
        "would_be_auto": {
            k: {**g, "cards": sorted(g["cards"]), "n_cards": len(g["cards"])}
            for k, g in sorted(groups.items())
        },
    }


def audit_rows(
    sample: AuditSample, *, reviewer: str, risk_of: Callable[[str], float]
) -> tuple[list[AuditRow], dict[str, str]]:
    """The ``audits`` rows of a reviewed sample (module docstring): one per
    ``(task_key, artifact_version)`` of the reviewed would-be-auto decisions that apply an
    artifact, and why the others got none."""
    rows: list[AuditRow] = []
    skipped: dict[str, str] = {}
    for key, g in summarize(sample)["would_be_auto"].items():
        version = g["artifact_version"]
        if not version:
            skipped[key] = (
                f"{g['task_id']}: {g['n']} reviewed would-be-auto decision(s) apply no artifact "
                "(zero_shot or ols_rank): an audits row is of an artifact"
            )
            continue
        n, errors = int(g["n"]), int(g["n_errors"])
        risk = float(risk_of(str(g["task_id"])))
        upper = clopper_pearson_upper(errors, n)
        rows.append(
            AuditRow(
                task_key=str(g["task_key"]),
                artifact_version=str(version),
                n=n,
                n_cards=int(g["n_cards"]),
                cards=list(g["cards"]),
                reviewer=reviewer,
                n_errors=errors,
                cp95_upper=upper,
                risk=risk,
                passed=audit_passes(n, int(g["n_cards"]), upper, risk),
            )
        )
    return rows, skipped
