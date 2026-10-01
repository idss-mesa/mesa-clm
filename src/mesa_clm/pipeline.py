"""The decide phase: ``Annotator.annotate(card) -> AnnotationRun`` (DESIGN D2, D10, D22-D25, D28;
plan §4.2).

Nothing here writes iRODS or the MESA history: annotate decides, proposes and records, and
``apply`` (M3) later writes only what is already accepted (D10). Ported from mesa-anyjev
``pipeline.py`` (``6159281``; DESIGN U1) with the AnyJev logprob readout replaced by rank-first
CLM questions. The steps, per card:

* **Q1 ``column.annotate``** (choice K=2 over ``registry.ANNOTATE_OPTIONS``). Identifier columns
  (``cards.is_identifier``) and columns the planner marks ``annotate=False`` get ``rule`` rows
  first and are never sent. A column is annotated when CLM's top-1 is ``Yes`` (rank-and-cap
  while uncalibrated, D28; a calibrated tier must also clear ``t.propose``). A column CLM could
  not answer for (unavailable, truncated) stays in play: nothing ranked it out.
* **Q2 ``column.aspect``** (choice K=8) goes in the *same* request as Q1 (they share
  ``column_state``, plan §4.1): top-1 when proposed, else top-2, the planner's aspect appended,
  ``other`` dropped. The aspect answer of a column Q1 left out is recorded ``rejected``
  (``column_not_annotated``). Without an answer only the planner's aspect remains; a column with
  no aspect at all is reported (``no_aspect``) and gets no ontology groups.
* **Q3 ``column.ontology_fits``**: one rank_fit request per (column, aspect) over the twelve
  registry entries plus the anchor, masked after scoring by the aspect (the provider applies the
  framing's ``mask_rule``) and by the ontologies in play (:func:`mesa_clm.policy.masked_for_aspect`,
  idempotent over the provider's mask). The top two in-play ontologies by ``p_fit`` are kept
  whatever the outcome (rank-and-cap, D28); ``unit`` is ``uo`` by a ``rule`` row; the planner's
  ontology is appended.
* **S** (``OLSLayer.search_candidates``: planner queries then ``queries_for_column``, at most
  three, ``size=20``, capped at ``max_candidates``; ``uo`` through ``unit_candidate`` first) and
  **Q4 ``term.fits``**: one rank_fit request per (column, ontology) group over its candidates
  (``"{label}: {description[:300]}"`` through the framing) plus the anchor, contexts from
  ``target_state`` so every group of one target shares a request. **Q5** site biomes
  (``OLSLayer.biome_candidates``) and **Q6** the dataset taxon (top two) are the same question
  over their own targets.
* **Q4b specificity** (D24): for a proposed winner with children, one rank over ``{parent, <=10
  OLS children, anchor}``; a child replaces the parent when ``p_fit(child) >= p_fit(parent) +
  delta`` and is at most ``proposed`` (:func:`mesa_clm.policy.cap_at_proposed`).
* **Q7 ``avu.value_kind``** (choice K=4): ``avu.pre_rule_value_kind`` first, else asked over
  ``value_kind_state`` of the proposal's *own* column (defect (b): mesa-anyjev recorded the value
  kind under the last column of an earlier loop), ``"the term label"`` unless proposed.
* **Q8** the ``avu.keep`` rule (D25): exact-triple dedup, then a cap of ``max_avus`` (25) by
  ``p_fit`` (:func:`mesa_clm.policy.dedup_and_cap`); what it drops rejects its group.

**Storage.** A rank_fit question is *one* CLM Choice over a group, so it is one ``decisions``
row (the answered candidate's ``s_c``/``p_fit`` on the row) with one ``decision_options`` row per
candidate and one for the anchor (``option_key``, ``s_c``, ``p_fit``, ``prob``, ``raw_prob``,
``masked``, ``rank``), plus a ``decision_groups`` row (``search_json`` with the OLS log and the
candidates, ``top_p_fit``, ``group_margin``, ``anchor_won``, the outcome). Every decision, rule
rows included, goes through :meth:`mesa_clm.policy.Policy.verdict`; ``decisions.reason`` is the
verdict's reason and an abstain for ``decider_unavailable`` is stored as that outcome. A policy
``auto`` becomes a link ``accepted`` by ``policy`` (D10); ``proposed`` a link ``proposed``.
Everything is buffered in a :class:`~mesa_clm.provenance.store.RunBuffer` and committed in one
transaction (D11), a failed run with status ``failed``.

**Degraded mode (D28).** A term or ontology group whose CLM record is ``unavailable``
(``decider_unavailable``, or ``truncated``) is recorded and then decided by ``ols_rank``: the
OLS top-1 as ``proposed``, no numbers, never ``auto``. The same happens for every group of a
task in ``ols_rank_tasks`` (K1/K2(c)) or for all rank_fit tasks when the tier is ``ols_rank``.
Anchor-won groups are recorded as ``abstain`` (``anchor_won``) and stay pending for review.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final, Literal
from uuid import UUID, uuid4

from mesa_clm import __version__
from mesa_clm import framings as fr
from mesa_clm.avu import (
    VALUE_KIND_LABEL,
    Avu,
    build_avu,
    pre_rule_value_kind,
    top_profile_value,
    triple,
    value_for,
)
from mesa_clm.cards import ColumnInfo, DatasetCard, SiteInfo, is_identifier
from mesa_clm.clm.fingerprint import Fingerprint
from mesa_clm.config import Config, config_sha256
from mesa_clm.ols import OLS_ERRORS, Candidate, OLSLayer
from mesa_clm.planner.base import ColumnHint, Planner, PlanResult
from mesa_clm.planner.static_planner import habitat_queries, queries_for_column, taxon_queries
from mesa_clm.policy import (
    Policy,
    PolicyError,
    Thresholds,
    Verdict,
    cap_at_proposed,
    dedup_and_cap,
    group_margin,
    link_status,
    masked_for_aspect,
    specific_child,
)
from mesa_clm.provenance.models import (
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    RunRow,
)
from mesa_clm.provenance.store import ProvenanceStore, RunBuffer
from mesa_clm.providers.base import (
    CLM_METHODS,
    ClmCall,
    DecisionProvider,
    DecisionRecord,
    base_fields,
    framing_options,
    sha256_text,
)
from mesa_clm.providers.tiered import CLM_COMMIT, DecisionRequest, ols_rank_record
from mesa_clm.registry import ASPECTS, ONTOLOGY_REGISTRY, allowed_for_aspect
from mesa_clm.states import column_state, target_state, value_kind_state
from mesa_clm.tasks import RANK_FIT_TASKS, Scope
from mesa_clm.vocab import Calibration, Level, Method, Outcome

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_CHILDREN",
    "MAX_TAXON_AVUS",
    "ONTOLOGY_KEEP",
    "TIERS",
    "AnnotationRun",
    "Annotator",
    "Proposal",
    "rule_record",
    "settle",
]

TASK_ANNOTATE: Final[str] = "column.annotate"
TASK_ASPECT: Final[str] = "column.aspect"
TASK_ONTOLOGY: Final[str] = "column.ontology_fits"
TASK_TERM: Final[str] = "term.fits"
TASK_VALUE_KIND: Final[str] = "avu.value_kind"

# Q6 keeps the dataset's two best taxa (mesa-anyjev MAX_TAXON_AVUS); Q3 keeps two ontologies per
# aspect (rank-and-cap, D28); at most three OLS queries per group; D24 ranks <= 10 children.
MAX_TAXON_AVUS: Final[int] = 2
ONTOLOGY_KEEP: Final[int] = 2
MAX_QUERIES: Final[int] = 3
MAX_CHILDREN: Final[int] = 10
# What ``Annotator(tier=...)`` and ``decider.tier`` accept (``config.Tier``, asserted equal by
# the tests): the CLM tiers plus ``ols_rank``, the degraded method for every rank_fit task (D28).
TIERS: Final[tuple[str, ...]] = ("auto", "zero_shot", "calibrated", "probe", "head", "ols_rank")
ANNOTATE_YES: Final[str] = "Yes"
_PROPOSING: Final[frozenset[str]] = frozenset({"auto", "proposed"})

_TargetScope = Literal["column", "site", "dataset"]


# -- results ----------------------------------------------------------------------------------


@dataclass
class Proposal:
    """One AVU the run proposes: the candidate term, where it came from (group, decision,
    target, aspect), how sure the deciding record was and the AVU built from it (Q7).
    ``outcome`` is ``proposed`` or ``auto`` (``escalated`` after a disagreeing second opinion,
    ``rejected`` once the keep rule drops it). ``column_name``/``site_code`` are ``''`` when the
    target is not a column/site, as the link stores them."""

    link_id: UUID
    group_id: UUID
    decision_id: UUID
    candidate: Candidate
    scope: _TargetScope
    column_name: str
    site_code: str
    aspect: str
    ontology_id: str
    outcome: Outcome
    p_fit: float | None
    confidence: float | None
    level: Level
    calibration: Calibration
    method: Method
    task_key: str
    rationale: str = ""
    value_kind: str = VALUE_KIND_LABEL
    value_kind_decision_id: UUID | None = None
    avu: Avu = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The ``mesa_clm_annotate`` proposal shape (plan §7.1)."""
        return {
            "link_id": str(self.link_id),
            "group_id": str(self.group_id),
            "decision_id": str(self.decision_id),
            "option_id": self.candidate.curie,
            "attribute": self.avu.get("attribute", ""),
            "value": self.avu.get("value", ""),
            "unit": self.avu.get("unit", ""),
            "label": self.candidate.label,
            "iri": self.candidate.iri,
            "ontology_id": self.ontology_id,
            "aspect": self.aspect,
            "outcome": self.outcome,
            "p_fit": self.p_fit,
            "confidence": self.confidence,
            "level": self.level,
            "calibration": self.calibration,
            "method": self.method,
            "task_key": self.task_key,
            "column_name": self.column_name,
            "site_code": self.site_code,
            "value_kind": self.value_kind,
            "rationale": self.rationale,
        }


@dataclass
class AnnotationRun:
    """What one ``annotate`` produced. ``proposals`` are the kept AVUs (each with a link row);
    ``abstained`` lists what was decided but proposes nothing (``{group_id, reason, task_id,
    scope, column_name, site_code, ontology_id}``; ``group_id`` is ``None`` for a column without
    an aspect). ``n_calls``/``input_tokens`` count the requests to clm-serve and the encoder
    tokens it reported, ``n_failed_calls`` those that got no answer; ``degraded`` is true when
    any group fell back to ``ols_rank`` or a CLM answer was unavailable (D28). ``buffer`` holds
    the rows (committed when a store was given)."""

    run_id: UUID
    owner: str
    actor: str
    card: DatasetCard
    plan: PlanResult
    proposals: list[Proposal]
    abstained: list[dict[str, Any]]
    n_decisions: int
    n_calls: int
    input_tokens: int
    outcomes: dict[str, int]
    degraded: bool
    seconds: float
    fingerprint: dict[str, str]
    clm_model: str = ""
    buffer: RunBuffer | None = None
    n_failed_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The ``mesa_clm_annotate`` output (plan §7.1) minus the tool's ``next_step``."""
        return {
            "run_id": str(self.run_id),
            "owner": self.owner,
            "actor": self.actor,
            "card": self.card.name,
            "fingerprint": dict(self.fingerprint),
            "proposals": [p.as_dict() for p in self.proposals],
            "abstained": [dict(a) for a in self.abstained],
            "n_decisions": self.n_decisions,
            "n_calls": self.n_calls,
            "n_failed_calls": self.n_failed_calls,
            "input_tokens": self.input_tokens,
            "outcomes": dict(self.outcomes),
            "degraded": self.degraded,
            "seconds": self.seconds,
        }

    def to_eval_result(self) -> dict[str, Any]:
        """The neon-avu-eval result shape (as mesa-anyjev's), so its scoring scripts run
        unchanged: ``avus`` with ``attribute, value, unit, ontology_id, curie, iri, label,
        aspect, column, rationale`` plus the decision's ``p`` (= ``p_fit``), level and outcome.
        ``n_prompts`` is the number of CLM requests; ``missing_labels`` is always 0 (no
        logprob readout)."""
        return {
            "model": f"mesa-clm/{__version__}:{self.clm_model}",
            "family": "mesa-clm",
            "card": self.card.name,
            "run_id": str(self.run_id),
            "parse_ok": True,
            "avus": [
                {
                    "attribute": p.avu["attribute"],
                    "value": p.avu["value"],
                    "unit": p.avu["unit"],
                    "ontology_id": p.candidate.ontology_id,
                    "curie": p.candidate.curie,
                    "iri": p.candidate.iri,
                    "label": p.candidate.label,
                    "aspect": p.aspect,
                    "column": p.column_name or None,
                    "rationale": p.rationale,
                    "p": p.p_fit,
                    "level": p.level,
                    "method": p.method,
                    "outcome": p.outcome,
                    "link_id": str(p.link_id),
                }
                for p in self.proposals
            ],
            "n_decisions": self.n_decisions,
            "n_prompts": self.n_calls,
            "missing_labels": 0,
            "seconds": self.seconds,
            "outcomes": dict(self.outcomes),
            "degraded": self.degraded,
            "fingerprint": dict(self.fingerprint),
        }


# -- records the pipeline builds itself -------------------------------------------------------


def rule_record(
    framing: fr.Framing,
    state: dict[str, Any],
    answer: str,
    fingerprint: Fingerprint,
    *,
    candidates: Sequence[fr.CandidateLike] | None = None,
    model: str = "rule",
    reason: str | None = None,
) -> DecisionRecord:
    """A deterministic ``rule`` answer to ``framing`` over ``state`` (an identifier column is
    not annotated, a ``unit`` aspect is searched in ``uo``): level and calibration ``none``, no
    distribution, the context hashed but never sent (``context_tokens = 0``). ``answer`` is an
    option key of the framing (an anyjev label for a closed choice, a candidate key for a
    rank_fit, which then needs its ``candidates``)."""
    opts = framing_options(framing, candidates)
    if answer not in opts.options:
        raise ValueError(f"{framing.task_id}: {answer!r} is not one of its options")
    f = base_fields(
        framing,
        state,
        fingerprint,
        provider="rule",
        method="rule",
        model=model,
        context_sha256=sha256_text(fr.context_text(framing, state)),
        context_tokens=0,
    )
    idx = opts.options.index(answer)
    diagnostics: dict[str, Any] = {"context_sent": False}
    if reason is not None:
        diagnostics["rule"] = reason
    f.update(
        options=opts.options,
        option_texts=opts.option_texts,
        anchor_index=opts.anchor_index,
        level="none",
        calibration="none",
        answer_index=idx,
        answer=opts.options[idx],
        diagnostics=diagnostics,
    )
    return DecisionRecord.model_validate(f)


def settle(v: Verdict) -> tuple[Outcome, str | None]:
    """The stored ``(outcome, reason)`` of a verdict: an abstain because clm-serve could not
    answer is stored as ``decider_unavailable`` (the vocabulary has it), everything else as the
    policy said."""
    if v.outcome == "abstain" and v.reason == "decider_unavailable":
        return "decider_unavailable", v.reason
    return v.outcome, v.reason


def _candidate_json(c: Candidate) -> dict[str, Any]:
    """What a group keeps of an OLS candidate so a later pick can build its AVU (the service
    reads it back): the identity, the label and the OLS rank; the description is in the option
    text."""
    return {
        "curie": c.curie,
        "label": c.label,
        "iri": c.iri,
        "ontology_id": c.ontology_id,
        "has_children": c.has_children,
        "rank": c.rank,
    }


def _framing_candidates(cands: Sequence[Candidate]) -> list[fr.FramingCandidate]:
    """OLS candidates as the framings take them, in OLS order (the key is the CURIE)."""
    return [fr.FramingCandidate.from_term(c.as_state()) for c in cands]


def _in_play(record: DecisionRecord) -> list[int]:
    """Indices of the candidates still in play (not the anchor, not masked), best ``p_fit``
    first (request order within ties); by request order when the record has no scores."""
    flags = record.masked or [False] * record.k
    idx = [i for i in range(record.k) if i != record.anchor_index and not flags[i]]
    p_fit = record.p_fit
    if p_fit is None:
        return idx
    return sorted(idx, key=lambda i: -p_fit[i])


def _best_p_fit(record: DecisionRecord) -> float | None:
    order = _in_play(record)
    if record.p_fit is None or not order:
        return None
    return record.p_fit[order[0]]


def _score_ranks(record: DecisionRecord) -> list[int | None]:
    """1-based rank of every option by the record's own score (``p_fit`` for a rank_fit,
    ``probs`` for a choice), masked options unranked; all ``None`` without a score."""
    scores = record.p_fit if record.p_fit is not None else record.probs
    ranks: list[int | None] = [None] * record.k
    if scores is None:
        return ranks
    flags = record.masked or [False] * record.k
    order = sorted((i for i in range(record.k) if not flags[i]), key=lambda i: -scores[i])
    for pos, i in enumerate(order):
        ranks[i] = pos + 1
    return ranks


def _option_rows(
    decision_id: UUID, record: DecisionRecord, ols_ranks: Sequence[int | None] | None
) -> list[DecisionOptionRow]:
    """One ``decision_options`` row per option. ``rank`` is the rank by the record's own score;
    for an ``ols_rank`` record (no score) it is the OLS position."""
    use_ols = record.method == "ols_rank" and ols_ranks is not None
    ranks = list(ols_ranks) if use_ols and ols_ranks is not None else _score_ranks(record)
    flags = record.masked or [False] * record.k

    def at(values: Sequence[float] | None, i: int) -> float | None:
        return None if values is None else float(values[i])

    return [
        DecisionOptionRow(
            decision_id=decision_id,
            option_index=i,
            option_key=record.options[i],
            option_text=record.option_texts[i],
            rank=ranks[i],
            s_c=at(record.s_c, i),
            p_fit=at(record.p_fit, i),
            prob=at(record.probs, i),
            raw_prob=at(record.raw_probs, i),
            masked=bool(flags[i]),
            action_sha256=sha256_text(record.option_texts[i]),
        )
        for i in range(record.k)
    ]


def _answer_rank(record: DecisionRecord, ols_ranks: Sequence[int | None] | None) -> int | None:
    """``decisions.rank``: the OLS rank for ``ols_rank`` (D28); for a CLM rank_fit that answered
    a candidate, that candidate's OLS position (where OLS had CLM's pick)."""
    if record.method == "ols_rank":
        return record.rank
    if ols_ranks is None or record.answer_index < 0 or record.anchor_won:
        return None
    return ols_ranks[record.answer_index]


# -- the recorder -----------------------------------------------------------------------------


class _Recorder:
    """Turns records into sidecar rows in the run buffer and keeps the sequence number and the
    outcome counts."""

    def __init__(self, buffer: RunBuffer, policy: Policy) -> None:
        self.buffer = buffer
        self.policy = policy
        self.seq = 0
        self.outcomes: Counter[str] = Counter()

    def _thresholds(self, task_id: str) -> Thresholds | None:
        try:
            return self.policy.thresholds(task_id)
        except PolicyError:
            return None

    def verdict(self, record: DecisionRecord) -> tuple[Verdict, Outcome, str | None]:
        """The policy's verdict on ``record`` and its stored ``(outcome, reason)``."""
        v = self.policy.verdict(record)
        outcome, reason = settle(v)
        return v, outcome, reason

    def record(
        self,
        record: DecisionRecord,
        *,
        scope: Scope,
        outcome: Outcome,
        reason: str | None,
        column_name: str | None = None,
        site_code: str | None = None,
        group_id: UUID | None = None,
        parent_decision_id: UUID | None = None,
        ols_ranks: Sequence[int | None] | None = None,
    ) -> DecisionRow:
        self.seq += 1
        t = self._thresholds(record.task_id)
        row = DecisionRow(
            run_id=self.buffer.run_id,
            seq=self.seq,
            group_id=group_id,
            parent_decision_id=parent_decision_id,
            task_id=record.task_id,
            task_key=record.task_key,
            question_key=record.question_key,
            framing_id=record.framing_id,
            shape=record.shape,
            k=record.k,
            scope=scope,
            column_name=column_name,
            site_code=site_code,
            state_sha256=record.state_sha256,
            state_json=record.state,
            target_sha256=record.target_sha256,
            context_sha256=record.context_sha256,
            context_tokens=record.context_tokens,
            truncated=record.truncated,
            method=record.method,
            model=record.model,
            level=record.level,
            calibration=record.calibration,
            probs=record.probs,
            raw_probs=record.raw_probs,
            confidence=record.confidence,
            clm_confidence=record.clm_confidence,
            s_c=record.answer_s_c,
            p_fit=record.answer_p_fit,
            margin=record.margin,
            anchor_index=record.anchor_index,
            answer_index=record.answer_index,
            answer=record.answer,
            options=list(record.options),
            rank=_answer_rank(record, ols_ranks),
            artifact_version=record.artifact_version,
            feature_spec=record.feature_spec,
            encoder_fp=record.encoder_fp,
            clm_model_fp=record.clm_model_fp,
            schema_sha256=record.schema_sha256,
            latency_ms=record.latency_ms,
            threshold_auto=t.auto if t is not None else None,
            threshold_propose=t.propose if t is not None else None,
            outcome=outcome,
            reason=reason,
        )
        self.buffer.insert_decisions([row], _option_rows(row.decision_id, record, ols_ranks))
        self.outcomes[outcome] += 1
        return row


# -- the annotator ----------------------------------------------------------------------------


@dataclass
class _Group:
    """One term.fits candidate group on its way through Q4: the target, the OLS search and the
    target-last context every group of that target shares."""

    group_id: UUID
    scope: _TargetScope
    column: ColumnInfo | None
    site: SiteInfo | None
    aspect: str
    ontology_id: str
    candidates: list[Candidate]
    log: dict[str, Any]
    state: dict[str, Any]
    keep_top: int = 1

    @property
    def column_name(self) -> str | None:
        return self.column.name if self.column is not None else None

    @property
    def site_code(self) -> str | None:
        return self.site.code if self.site is not None else None


class Annotator:
    """The collaborators of one annotate call; :meth:`annotate` runs one card.

    ``provider`` answers the CLM questions (a :class:`~mesa_clm.providers.tiered.TieredProvider`
    or :class:`~mesa_clm.providers.tiered.FakeProvider`; its ``fingerprint`` is stamped on every
    record, its ``on_call`` hook is borrowed during the run to collect ``clm_calls``). ``planner``
    plans (D22), ``ols`` searches, ``policy`` gives every outcome, ``store`` (optional) receives
    the committed run. ``owner`` is the identity the run belongs to (D21; explain, feedback and
    apply check it) and ``actor`` who asked for it. ``tier`` is a :data:`TIERS` value (default
    ``cfg.decider.tier``); ``ols_rank`` sends every rank_fit task to the degraded method, and
    ``ols_rank_tasks`` does so per task (K1/K2(c)). ``second_opinion`` is an optional Claude
    provider asked about proposed winners; it never decides (D22).
    """

    def __init__(
        self,
        *,
        provider: DecisionProvider,
        planner: Planner,
        ols: OLSLayer,
        policy: Policy,
        cfg: Config,
        owner: str,
        store: ProvenanceStore | None = None,
        actor: str | None = None,
        second_opinion: DecisionProvider | None = None,
        tier: str | None = None,
        ols_rank_tasks: Collection[str] = (),
        encoder_model: str | None = None,
    ) -> None:
        if not owner or not owner.strip():
            raise ValueError("a run needs an owner (D21)")
        chosen = tier or cfg.decider.tier
        if chosen not in TIERS:
            raise ValueError(f"unknown tier {chosen!r}; expected one of {TIERS}")
        degraded_tasks = frozenset(ols_rank_tasks) | (
            RANK_FIT_TASKS if chosen == "ols_rank" else frozenset()
        )
        unknown = degraded_tasks - RANK_FIT_TASKS
        if unknown:
            raise ValueError(f"ols_rank ranks candidate groups only; not {sorted(unknown)} (D28)")
        self.provider = provider
        self.planner = planner
        self.ols = ols
        self.policy = policy
        self.cfg = cfg
        self.owner = owner
        self.actor = actor or owner
        self.store = store
        self.second_opinion = second_opinion
        self.tier = chosen
        # What the provider is asked for: its own best tier unless one was named.
        self.clm_tier: str | None = None if chosen in ("auto", "ols_rank") else chosen
        self.ols_rank_tasks: frozenset[str] = degraded_tasks
        self.encoder_model = encoder_model

    def annotate(self, card: DatasetCard) -> AnnotationRun:
        """Decide one card (module docstring) and commit its rows to the store."""
        return _Pass(self, card).run()


class _Pass:
    """One annotate pass over one card: the per-run state the steps share."""

    def __init__(self, ann: Annotator, card: DatasetCard) -> None:
        self.a = ann
        self.card = card
        self.cfg = ann.cfg
        self.started = time.monotonic()
        self.plan_result = ann.planner.plan(card)
        self.plan = self.plan_result.plan
        self.in_play: frozenset[str] = (
            frozenset(self.plan.ontologies)
            if self.plan.ontologies
            else frozenset(e.id for e in ONTOLOGY_REGISTRY)
        )
        self.fp: Fingerprint = ann.provider.fingerprint
        self.framing = {t: fr.active_framing(t) for t in fr.FRAMINGS}
        self.run_row = self._run_row()
        self.buffer = RunBuffer(self.run_row)
        self.rec = _Recorder(self.buffer, ann.policy)
        self.calls: list[ClmCall] = []
        self.degraded = False
        self.proposals: list[Proposal] = []
        self.abstained: list[dict[str, Any]] = []
        # specificity group -> the group it refined (Q4b), so the keep rule can reject both
        self.refined_from: dict[UUID, UUID] = {}

    # -- run bookkeeping --------------------------------------------------------------------

    def _run_row(self) -> RunRow:
        a, p = self.a, self.a.provider
        artifacts = getattr(p, "artifacts", None)
        method = getattr(p, "method", "")
        encoder_model = a.encoder_model or (
            "fake-ngram" if method == "fake" else a.cfg.encoder.model
        )
        return RunRow(
            owner=a.owner,
            card_name=self.card.name,
            card_sha256=self.card.sha256,
            planner=self.plan_result.planner,
            planner_model=self.plan_result.model,
            planner_prompt_sha256=self.plan_result.prompt_sha256,
            plan_json=self.plan.model_dump(mode="json"),
            planner_fallback=self.plan_result.fallback,
            provider=p.name,
            tier=a.tier,
            clm_model=p.model,
            encoder_model=encoder_model,
            clm_commit=CLM_COMMIT if method in CLM_METHODS else "",
            schema_sha256=self.fp.schema_sha256,
            encoder_fp=self.fp.encoder_fp,
            clm_model_fp=self.fp.clm_model_fp,
            serving_lock_sha=self.fp.serving_lock_sha,
            framings_lock_sha=fr.lock_sha(),
            artifacts_version=getattr(artifacts, "version", None),
            mesa_clm_version=__version__,
            policy_profile=a.policy.profile.name,
            config_sha256=config_sha256(a.cfg),
            vm_id=a.cfg.vm_id,
        )

    @contextmanager
    def _collect_calls(self) -> Iterator[None]:
        """Borrow the provider's ``on_call`` hook for the run (chained to any hook it had), so
        every clm-serve request lands in ``clm_calls`` with this run's id."""
        provider: Any = self.a.provider  # the hook is TieredProvider's, not the protocol's
        if not hasattr(provider, "on_call"):
            yield
            return
        previous: Callable[[ClmCall], None] | None = provider.on_call

        def hook(call: ClmCall) -> None:
            self.calls.append(call)
            if previous is not None:
                previous(call)

        provider.on_call = hook
        try:
            yield
        finally:
            provider.on_call = previous

    def run(self) -> AnnotationRun:
        try:
            with self._collect_calls():
                columns = self._columns()
                picks = self._ontologies(columns)
                groups = self._term_groups(picks)
                self._rank_groups(groups)
                self._value_kinds()
                kept = self._keep()
                self.buffer.insert_links([self._link(p) for p in kept])
            return self._finish(kept)
        except Exception:
            self._fail()
            raise

    def _call_rows(self) -> list[ClmCallRow]:
        return [
            ClmCallRow(
                run_id=self.buffer.run_id,
                endpoint=c.endpoint,
                model=c.model or "unknown",
                n_questions=c.n_questions,
                n_candidates=c.n_candidates,
                input_tokens=c.input_tokens,
                latency_ms=max(0.0, float(c.latency_ms)),
                status=c.status,
            )
            for c in self.calls
        ]

    def _finish(self, kept: list[Proposal]) -> AnnotationRun:
        seconds = time.monotonic() - self.started
        input_tokens = sum(int(c.input_tokens or 0) for c in self.calls)
        self.buffer.insert_clm_calls(self._call_rows())
        self.buffer.finish_run(
            "decided",
            degraded=self.degraded,
            n_encoder_tokens=input_tokens,
            seconds=seconds,
        )
        if self.a.store is not None:
            self.a.store.commit_run(self.buffer)
        return AnnotationRun(
            run_id=self.buffer.run_id,
            owner=self.a.owner,
            actor=self.a.actor,
            card=self.card,
            plan=self.plan_result,
            proposals=kept,
            abstained=self.abstained,
            n_decisions=len(self.buffer.decisions),
            n_calls=len(self.calls),
            n_failed_calls=sum(c.status != "ok" for c in self.calls),
            input_tokens=input_tokens,
            outcomes=dict(self.rec.outcomes),
            degraded=self.degraded,
            seconds=seconds,
            fingerprint={**self.fp.as_dict(), "framings_lock_sha": self.run_row.framings_lock_sha},
            clm_model=self.a.provider.model,
            buffer=self.buffer,
        )

    def _fail(self) -> None:
        """Commit what the failed run decided, status ``failed``, for the post-mortem; never
        mask the original exception."""
        if self.a.store is None or self.buffer.committed:
            return
        try:
            self.buffer.insert_clm_calls(self._call_rows())
            self.buffer.finish_run(
                "failed", degraded=self.degraded, seconds=time.monotonic() - self.started
            )
            self.a.store.commit_run(self.buffer)
        except Exception as exc:  # the original error is what the caller needs to see
            logger.warning("could not record the failed run (%s)", type(exc).__name__)

    # -- deciding ---------------------------------------------------------------------------

    def _decide(self, requests: Sequence[DecisionRequest]) -> list[DecisionRecord]:
        """One record per request, in order: ``decide_many`` when the provider has it (requests
        sharing a context go in one call, plan §4.1), else one ``decide`` per framing."""
        if not requests:
            return []
        provider = self.a.provider
        many = getattr(provider, "decide_many", None)
        if callable(many):
            records = list(many(requests, tier=self.a.clm_tier))
        else:
            slots: list[DecisionRecord | None] = [None] * len(requests)
            by_framing: dict[tuple[str, str], list[int]] = {}
            for i, req in enumerate(requests):
                by_framing.setdefault((req.framing.task_id, req.framing.id), []).append(i)
            for idxs in by_framing.values():
                framing = requests[idxs[0]].framing
                groups = (
                    None
                    if framing.shape == "choice"
                    else [requests[i].candidates or () for i in idxs]
                )
                got = provider.decide(
                    framing, [requests[i].state for i in idxs], groups, tier=self.a.clm_tier
                )
                for i, r in zip(idxs, got, strict=True):
                    slots[i] = r
            records = [r for r in slots if r is not None]
        if len(records) != len(requests):
            raise RuntimeError("the provider returned a different number of records")
        for r in records:
            if r.method == "unavailable":
                self.degraded = True
        return records

    def _rule(
        self,
        task_id: str,
        state: dict[str, Any],
        answer: str,
        *,
        scope: Scope,
        reason: str,
        column_name: str | None = None,
        candidates: Sequence[fr.CandidateLike] | None = None,
        model: str = "rule",
    ) -> DecisionRow:
        rec = rule_record(
            self.framing[task_id],
            state,
            answer,
            self.fp,
            candidates=candidates,
            model=model,
            reason=reason,
        )
        _, outcome, _ = self.rec.verdict(rec)  # 'rule': a rule needs no thresholds
        return self.rec.record(
            rec, scope=scope, outcome=outcome, reason=reason, column_name=column_name
        )

    def _abstain(
        self,
        reason: str,
        *,
        task_id: str,
        scope: str,
        group_id: UUID | None = None,
        column_name: str | None = None,
        site_code: str | None = None,
        ontology_id: str | None = None,
    ) -> None:
        self.abstained.append(
            {
                "group_id": str(group_id) if group_id is not None else None,
                "reason": reason,
                "task_id": task_id,
                "scope": scope,
                "column_name": column_name or "",
                "site_code": site_code or "",
                "ontology_id": ontology_id or "",
            }
        )

    # -- Q1 + Q2 ----------------------------------------------------------------------------

    def _columns(self) -> list[tuple[ColumnInfo, list[str]]]:
        """Q1 and Q2 over every column: the columns to annotate with their aspects."""
        live: list[ColumnInfo] = []
        for col in self.card.columns:
            hint = self.plan.columns.get(col.name)
            state = column_state(self.card, col)
            if is_identifier(col) and not (hint and hint.annotate is True):
                self._rule(
                    TASK_ANNOTATE,
                    state,
                    "No",
                    scope="column",
                    reason="is_identifier",
                    column_name=col.name,
                )
                continue
            if hint and hint.annotate is False:
                self._rule(
                    TASK_ANNOTATE,
                    state,
                    "No",
                    scope="column",
                    reason="planner_annotate_false",
                    column_name=col.name,
                    model="planner",
                )
                continue
            live.append(col)
        requests: list[DecisionRequest] = []
        for col in live:
            state = column_state(self.card, col)
            requests.append(DecisionRequest(self.framing[TASK_ANNOTATE], state))
            requests.append(DecisionRequest(self.framing[TASK_ASPECT], state))
        records = self._decide(requests)
        out: list[tuple[ColumnInfo, list[str]]] = []
        for i, col in enumerate(live):
            r_annotate, r_aspect = records[2 * i], records[2 * i + 1]
            hint = self.plan.columns.get(col.name)
            _, o1, why1 = self.rec.verdict(r_annotate)
            self.rec.record(
                r_annotate, scope="column", outcome=o1, reason=why1, column_name=col.name
            )
            if r_annotate.answer_index < 0:
                annotate = True  # nothing ranked the column out: keep it (D28)
            else:
                annotate = r_annotate.answer == ANNOTATE_YES and (
                    o1 in _PROPOSING or r_annotate.calibration == "uncalibrated"
                )
            if not annotate:
                # The aspect was asked in the same request; recorded, but Q1 made it moot.
                self.rec.verdict(r_aspect)  # every record meets the policy, even a moot one
                self.rec.record(
                    r_aspect,
                    scope="column",
                    outcome="rejected",
                    reason="column_not_annotated",
                    column_name=col.name,
                )
                continue
            _, o2, why2 = self.rec.verdict(r_aspect)
            self.rec.record(r_aspect, scope="column", outcome=o2, reason=why2, column_name=col.name)
            aspects = self._aspects(r_aspect, o2, hint)
            if not aspects:
                self._abstain(
                    "no_aspect", task_id=TASK_ASPECT, scope="column", column_name=col.name
                )
                continue
            out.append((col, aspects))
        return out

    @staticmethod
    def _aspects(record: DecisionRecord, outcome: Outcome, hint: ColumnHint | None) -> list[str]:
        """Q2's aspects: top-1 when proposed, else top-2 (rank-and-cap), the planner's aspect
        appended, ``other`` dropped."""
        chosen: list[str] = []
        probs = record.probs
        if probs is not None and record.answer_index >= 0:
            order = sorted(range(record.k), key=lambda i: -probs[i])
            n = 1 if outcome in _PROPOSING else 2
            chosen = [ASPECTS[i] for i in order[:n]]
        if hint is not None and hint.aspect and hint.aspect not in chosen:
            chosen.append(hint.aspect)
        return [a for a in chosen if a != "other"]

    # -- Q3 ---------------------------------------------------------------------------------

    def _ontologies(
        self, columns: Sequence[tuple[ColumnInfo, list[str]]]
    ) -> list[tuple[ColumnInfo, str, str]]:
        """Q3: ``(column, ontology, aspect)`` for every group S will search, in column order."""
        framing = self.framing[TASK_ONTOLOGY]
        registry = fr.ontology_candidates()
        picks: dict[str, list[tuple[str, str]]] = {}
        pending: list[tuple[ColumnInfo, str, frozenset[str]]] = []
        for col, aspects in columns:
            picks[col.name] = []
            for aspect in aspects:
                if aspect == "unit":
                    self._rule(
                        TASK_ONTOLOGY,
                        target_state(self.card, "column", "unit", column=col),
                        "uo",
                        scope="column",
                        reason="unit_aspect",
                        column_name=col.name,
                        candidates=registry,
                    )
                    self._add_pick(picks[col.name], "uo", "unit")
                    continue
                allowed = allowed_for_aspect(aspect) & self.in_play
                if allowed:
                    pending.append((col, aspect, allowed))
        records: list[DecisionRecord | None]
        if TASK_ONTOLOGY in self.a.ols_rank_tasks:
            records = [None] * len(pending)
        else:
            records = list(
                self._decide(
                    [
                        DecisionRequest(
                            framing,
                            target_state(self.card, "column", aspect, column=col),
                            registry,
                        )
                        for col, aspect, _ in pending
                    ]
                )
            )
        for (col, aspect, allowed), record in zip(pending, records, strict=True):
            for ont in self._rank_ontologies(col, aspect, allowed, record):
                self._add_pick(picks[col.name], ont, aspect)
        out: list[tuple[ColumnInfo, str, str]] = []
        for col, aspects in columns:
            chosen = picks[col.name]
            hint = self.plan.columns.get(col.name)
            if hint and hint.ontology and hint.ontology in self.in_play:
                self._add_pick(chosen, hint.ontology, aspects[0])
            out.extend((col, ont, aspect) for ont, aspect in chosen)
        return out

    @staticmethod
    def _add_pick(chosen: list[tuple[str, str]], ont: str, aspect: str) -> None:
        if all(o != ont for o, _ in chosen):
            chosen.append((ont, aspect))

    def _rank_ontologies(
        self,
        col: ColumnInfo,
        aspect: str,
        allowed: frozenset[str],
        record: DecisionRecord | None,
    ) -> list[str]:
        """One Q3 group: record the rank (or its ``ols_rank`` fallback) and return the two
        in-play ontologies it keeps."""
        framing = self.framing[TASK_ONTOLOGY]
        state = target_state(self.card, "column", aspect, column=col)
        group_id = uuid4()
        parent: UUID | None = None
        if record is not None:
            record = masked_for_aspect(record, aspect, self.in_play)
        if record is None or record.method == "unavailable":
            if record is not None:
                _, o, why = self.rec.verdict(record)
                parent = self.rec.record(
                    record,
                    scope="column",
                    outcome=o,
                    reason=why,
                    column_name=col.name,
                    group_id=group_id,
                ).decision_id
            self.degraded = True
            cands = fr.ontology_candidates(allowed)
            record = ols_rank_record(framing, state, cands, self.fp)
            ols_ranks: list[int | None] | None = [*range(1, len(cands) + 1), None]
            kept = [c.key for c in cands[:ONTOLOGY_KEEP]]
        else:
            ols_ranks = None
            kept = [record.options[i] for i in _in_play(record)[:ONTOLOGY_KEEP]]
        _, outcome, reason = self.rec.verdict(record)
        margin = group_margin(record)
        outcome = self.a.policy.demote_for_group_margin(outcome, margin, TASK_ONTOLOGY)
        row = self.rec.record(
            record,
            scope="column",
            outcome=outcome,
            reason=reason,
            column_name=col.name,
            group_id=group_id,
            parent_decision_id=parent,
            ols_ranks=ols_ranks,
        )
        self.buffer.insert_group(
            DecisionGroupRow(
                group_id=group_id,
                run_id=self.buffer.run_id,
                task_id=TASK_ONTOLOGY,
                task_key=framing.task_key,
                question_key=record.question_key,
                scope="column",
                column_name=col.name,
                aspect=aspect,
                search_json={
                    "registry": [e.id for e in ONTOLOGY_REGISTRY],
                    "aspect": aspect,
                    "in_play": sorted(self.in_play),
                    "allowed": sorted(allowed),
                    "kept": kept,
                    "rule": "rank_and_cap_top2",
                },
                n_candidates=len(allowed),
                winner_decision_id=row.decision_id,
                top_p_fit=_best_p_fit(record),
                group_margin=margin,
                level=record.level,
                method=record.method,
                outcome=outcome,
                anchor_won=record.anchor_won,
            )
        )
        return kept

    # -- S, Q5, Q6 --------------------------------------------------------------------------

    def _queries(self, col: ColumnInfo) -> list[str]:
        hint = self.plan.columns.get(col.name)
        queries = list(hint.queries) if hint and hint.queries else []
        for q in queries_for_column(col):
            if q not in queries:
                queries.append(q)
        return queries[:MAX_QUERIES]

    def _term_groups(self, picks: Sequence[tuple[ColumnInfo, str, str]]) -> list[_Group]:
        """The OLS searches: one group per (column, ontology), per site (Q5) and for the
        dataset's taxon (Q6)."""
        ols = self.a.ols
        groups: list[_Group] = []
        for col, ont, aspect in picks:
            unit = ols.unit_candidate(col.unit) if ont == "uo" and col.unit else None
            if unit is not None:
                cands: list[Candidate] = [unit]
                log: dict[str, Any] = {
                    "ontology_id": "uo",
                    "table": True,
                    "queries": [col.unit],
                    "n_candidates": 1,
                }
            elif ont == "uo":
                cands, log = ols.search_candidates([col.unit or "unit"], "uo")
            else:
                cands, log = ols.search_candidates(self._queries(col), ont)
            groups.append(
                _Group(
                    uuid4(),
                    "column",
                    col,
                    None,
                    aspect,
                    ont,
                    cands,
                    log,
                    target_state(self.card, "column", aspect, column=col),
                )
            )
        for site in self.card.sites:
            hint = self.plan.sites.get(site.code)
            queries = (
                list(hint.environment_queries) if hint and hint.environment_queries else []
            ) or habitat_queries(site)
            cands, log = ols.biome_candidates(queries)
            groups.append(
                _Group(
                    uuid4(),
                    "site",
                    None,
                    site,
                    "environment",
                    "envo",
                    cands,
                    log,
                    target_state(self.card, "site", "environment", site=site),
                )
            )
        taxon_q = list(self.plan.taxon_queries) or taxon_queries(self.card)
        if taxon_q and "ncbitaxon" in self.in_play:
            cands, log = ols.search_candidates(taxon_q[:MAX_QUERIES], "ncbitaxon")
            groups.append(
                _Group(
                    uuid4(),
                    "dataset",
                    None,
                    None,
                    "taxon",
                    "ncbitaxon",
                    cands,
                    log,
                    target_state(self.card, "dataset", "taxon"),
                    keep_top=MAX_TAXON_AVUS,
                )
            )
        return groups

    # -- Q4 ---------------------------------------------------------------------------------

    def _group_row(
        self,
        g: _Group,
        *,
        group_id: UUID,
        search_json: dict[str, Any],
        n_candidates: int,
        outcome: Outcome,
        record: DecisionRecord | None = None,
        winner: UUID | None = None,
        margin: float | None = None,
        top_p_fit: float | None = None,
        escalated_from: UUID | None = None,
    ) -> DecisionGroupRow:
        framing = self.framing[TASK_TERM]
        return DecisionGroupRow(
            group_id=group_id,
            run_id=self.buffer.run_id,
            task_id=TASK_TERM,
            task_key=framing.task_key,
            question_key=record.question_key if record is not None else framing.question_key,
            scope=g.scope,
            column_name=g.column_name,
            site_code=g.site_code,
            aspect=g.aspect,
            ontology_id=g.ontology_id,
            search_json=search_json,
            n_candidates=n_candidates,
            winner_decision_id=winner,
            top_p_fit=top_p_fit,
            group_margin=margin,
            level=record.level if record is not None else None,
            method=record.method if record is not None else None,
            outcome=outcome,
            anchor_won=record.anchor_won if record is not None else False,
            escalated_from=escalated_from,
        )

    def _rank_groups(self, groups: Sequence[_Group]) -> None:
        todo = [g for g in groups if g.candidates]
        for g in groups:
            if not g.candidates:
                self.buffer.insert_group(
                    self._group_row(
                        g,
                        group_id=g.group_id,
                        search_json={**g.log, "candidates": []},
                        n_candidates=0,
                        outcome="abstain",
                    )
                )
                self._abstain(
                    "no_candidates",
                    task_id=TASK_TERM,
                    scope=g.scope,
                    group_id=g.group_id,
                    column_name=g.column_name,
                    site_code=g.site_code,
                    ontology_id=g.ontology_id,
                )
        records: list[DecisionRecord | None]
        if TASK_TERM in self.a.ols_rank_tasks:
            records = [None] * len(todo)
        else:
            framing = self.framing[TASK_TERM]
            records = list(
                self._decide(
                    [
                        DecisionRequest(framing, g.state, _framing_candidates(g.candidates))
                        for g in todo
                    ]
                )
            )
        for g, record in zip(todo, records, strict=True):
            self._settle_group(g, record)

    def _settle_group(self, g: _Group, record: DecisionRecord | None) -> None:
        """Record one Q4/Q5/Q6 group, turn its winner (and, for the taxon, its runner-up) into
        proposals, refine the winner (Q4b) and ask the second opinion."""
        framing = self.framing[TASK_TERM]
        fcands = _framing_candidates(g.candidates)
        ols_ranks: list[int | None] = [*range(1, len(fcands) + 1), None]
        parent: UUID | None = None
        common: dict[str, Any] = {
            "scope": g.scope,
            "column_name": g.column_name,
            "site_code": g.site_code,
            "group_id": g.group_id,
            "ols_ranks": ols_ranks,
        }
        if record is None or record.method == "unavailable":
            if record is not None:
                _, o, why = self.rec.verdict(record)
                parent = self.rec.record(record, outcome=o, reason=why, **common).decision_id
            self.degraded = True
            record = ols_rank_record(framing, g.state, fcands, self.fp)
        _, outcome, reason = self.rec.verdict(record)
        margin = group_margin(record)
        outcome = self.a.policy.demote_for_group_margin(outcome, margin, TASK_TERM)
        row = self.rec.record(
            record, outcome=outcome, reason=reason, parent_decision_id=parent, **common
        )
        self.buffer.insert_group(
            self._group_row(
                g,
                group_id=g.group_id,
                search_json={**g.log, "candidates": [_candidate_json(c) for c in g.candidates]},
                n_candidates=len(g.candidates),
                outcome=outcome,
                record=record,
                winner=row.decision_id,
                margin=margin,
                top_p_fit=_best_p_fit(record),
            )
        )
        if outcome not in _PROPOSING or record.answer_index < 0 or record.anchor_won:
            self._abstain(
                reason or outcome,
                task_id=TASK_TERM,
                scope=g.scope,
                group_id=g.group_id,
                column_name=g.column_name,
                site_code=g.site_code,
                ontology_id=g.ontology_id,
            )
            return
        winner = self._proposal(g, row, record, record.answer_index, outcome)
        props = [winner]
        for i in self._runner_ups(record, g.keep_top - 1):
            props.append(self._proposal(g, row, record, i, "proposed", runner_up=True))
        if (
            self.cfg.policy.specificity
            and record.method in CLM_METHODS
            and winner.candidate.has_children
        ):
            props[0] = self._specificity(g, row, record, winner)
        if self.a.second_opinion is not None and props[0].outcome == "proposed":
            self._second_opinion(g, props[0])
        self.proposals.extend(props)

    def _runner_ups(self, record: DecisionRecord, n: int) -> list[int]:
        """Up to ``n`` candidates after the winner that still out-rank the anchor (``probs``
        above the anchor's: ``s_c > 0`` at zero shot, ``a·s_c + b > 0`` under Platt) and clear
        ``t.propose``; only a CLM record has them (an ``ols_rank`` rank 2 is below
        ``ols_rank_top``). Capped at ``proposed``: the policy judged only the winner."""
        probs, p_fit, anchor = record.probs, record.p_fit, record.anchor_index
        if n <= 0 or probs is None or p_fit is None or anchor is None:
            return []
        t = self.a.policy.thresholds(TASK_TERM)
        order = [i for i in _in_play(record) if i != record.answer_index]
        return [i for i in order if probs[i] > probs[anchor] and p_fit[i] >= t.propose][:n]

    def _proposal(
        self,
        g: _Group,
        row: DecisionRow,
        record: DecisionRecord,
        index: int,
        outcome: Outcome,
        *,
        runner_up: bool = False,
    ) -> Proposal:
        cand = g.candidates[index]
        p_fit = record.p_fit[index] if record.p_fit is not None else None
        if record.method == "ols_rank":
            rationale = f"OLS rank {record.rank} (ols_rank, D28)"
        else:
            rationale = f"p_fit={(p_fit or 0.0):.2f} {record.level}"
        if runner_up:
            rationale += "; runner-up (capped at proposed)"
        return Proposal(
            link_id=uuid4(),
            group_id=g.group_id,
            decision_id=row.decision_id,
            candidate=cand,
            scope=g.scope,
            column_name=g.column_name or "",
            site_code=g.site_code or "",
            aspect=g.aspect,
            ontology_id=g.ontology_id,
            outcome=cap_at_proposed(outcome) if runner_up else outcome,
            p_fit=p_fit,
            confidence=record.confidence,
            level=record.level,
            calibration=record.calibration,
            method=record.method,
            task_key=record.task_key,
            rationale=rationale,
        )

    # -- Q4b --------------------------------------------------------------------------------

    def _specificity(
        self, g: _Group, parent_row: DecisionRow, parent: DecisionRecord, winner: Proposal
    ) -> Proposal:
        """D24: one rank over the winner and its OLS children; a child that beats the parent's
        ``p_fit`` by ``specificity_delta`` replaces it, at most ``proposed``."""
        try:
            children = self.a.ols.children(g.ontology_id, winner.candidate.iri)
        except OLS_ERRORS as exc:
            logger.info("specificity: no children for %s (%s)", winner.candidate.curie, exc)
            return winner
        seen = {winner.candidate.curie}
        kids: list[Candidate] = []
        for c in children:
            if c.curie not in seen:
                seen.add(c.curie)
                kids.append(c)
        kids = kids[:MAX_CHILDREN]
        if not kids:
            return winner
        cands = [winner.candidate, *kids]
        [record] = self._decide(
            [DecisionRequest(self.framing[TASK_TERM], g.state, _framing_candidates(cands))]
        )
        group_id = uuid4()
        search_json = {
            "specificity_of": winner.candidate.curie,
            "delta": self.cfg.policy.specificity_delta,
            "candidates": [_candidate_json(c) for c in cands],
        }
        _, outcome, reason = self.rec.verdict(record)
        outcome = cap_at_proposed(outcome)  # D24: an unbenched heuristic only ever proposes
        row = self.rec.record(
            record,
            scope=g.scope,
            outcome=outcome,
            reason=reason,
            column_name=g.column_name,
            site_code=g.site_code,
            group_id=group_id,
            parent_decision_id=parent_row.decision_id,
        )
        child = (
            specific_child(record, winner.candidate.curie, delta=self.cfg.policy.specificity_delta)
            if record.p_fit is not None
            else None
        )
        replaces = child is not None and record.answer_index == child and outcome in _PROPOSING
        p_parent = record.p_fit[0] if record.p_fit is not None else None
        order = _in_play(record)
        best = child if child is not None else (order[0] if order else None)
        p_best = record.p_fit[best] if record.p_fit is not None and best is not None else None
        margin = p_best - p_parent if p_best is not None and p_parent is not None else None
        self.buffer.insert_group(
            self._group_row(
                g,
                group_id=group_id,
                search_json=search_json,
                n_candidates=len(cands),
                # Not replaced: the refinement is rejected and the parent stands; a rank CLM
                # could not answer keeps its own outcome (decider_unavailable, truncated).
                outcome=outcome if replaces or record.answer_index < 0 else "rejected",
                record=record,
                winner=row.decision_id,
                margin=margin,
                top_p_fit=p_best,
                escalated_from=g.group_id,
            )
        )
        if not replaces or child is None or record.p_fit is None:
            return winner
        self.refined_from[group_id] = g.group_id
        p_child = record.p_fit[child]
        return Proposal(
            link_id=uuid4(),
            group_id=group_id,
            decision_id=row.decision_id,
            candidate=cands[child],
            scope=g.scope,
            column_name=g.column_name or "",
            site_code=g.site_code or "",
            aspect=g.aspect,
            ontology_id=g.ontology_id,
            outcome="proposed",
            p_fit=p_child,
            confidence=record.confidence,
            level=record.level,
            calibration=record.calibration,
            method=record.method,
            task_key=record.task_key,
            rationale=(
                f"p_fit={p_child:.2f} {record.level}; more specific than "
                f"{winner.candidate.curie} (p_fit={(p_parent or 0.0):.2f}, D24)"
            ),
        )

    # -- the second opinion (D22) -----------------------------------------------------------

    def _second_opinion(self, g: _Group, prop: Proposal) -> None:
        """Claude answers the same question over the proposal's group (the winner and the best
        of the rest); recorded at level none under the winner. Agreement is noted; a different
        answer escalates the proposal and its group to a human."""
        so = self.a.second_opinion
        if so is None:
            return
        cands = self._group_candidates(g, prop)
        top_k = max(1, int(self.cfg.claude.second_opinion_top_k))
        cands = cands[:top_k]
        [record] = so.decide(self.framing[TASK_TERM], [g.state], [_framing_candidates(cands)])
        _, outcome, reason = self.rec.verdict(record)
        self.rec.record(
            record,
            scope=g.scope,
            outcome=outcome,
            reason=reason,
            column_name=g.column_name,
            site_code=g.site_code,
            group_id=prop.group_id,
            parent_decision_id=prop.decision_id,
        )
        if record.answer_index < 0:
            prop.rationale += "; claude gave no answer"
        elif record.answer == prop.candidate.curie:
            prop.rationale += "; claude agrees"
        else:
            prop.outcome = "escalated"
            prop.rationale += f"; claude prefers {record.answer} (escalated)"
            self.buffer.update_group(prop.group_id, outcome="escalated")

    def _group_candidates(self, g: _Group, prop: Proposal) -> list[Candidate]:
        """The proposal's candidate first, then the rest of its original group by OLS rank."""
        rest = [c for c in g.candidates if c.curie != prop.candidate.curie]
        return [prop.candidate, *rest]

    # -- Q7 ---------------------------------------------------------------------------------

    def _value_kinds(self) -> None:
        """Q7 and the AVUs. Every value kind is decided for the proposal's own column
        (defect (b))."""
        framing = self.framing[TASK_VALUE_KIND]
        ask: list[tuple[Proposal, dict[str, Any]]] = []
        for prop in self.proposals:
            pre = pre_rule_value_kind(prop.aspect, prop.scope)
            if pre is not None or not prop.column_name:
                prop.value_kind = pre or VALUE_KIND_LABEL
                continue
            pcol = self.card.column(prop.column_name)
            ask.append(
                (prop, value_kind_state(self.card, pcol, prop.candidate.as_state(), prop.aspect))
            )
        records = self._decide([DecisionRequest(framing, state) for _, state in ask])
        for (prop, _), record in zip(ask, records, strict=True):
            _, outcome, reason = self.rec.verdict(record)
            row = self.rec.record(
                record,
                scope="avu",
                outcome=outcome,
                reason=reason,
                column_name=prop.column_name,  # defect (b): the proposal's own column
                group_id=prop.group_id,
                parent_decision_id=prop.decision_id,
            )
            prop.value_kind_decision_id = row.decision_id
            proposed = outcome in _PROPOSING and record.answer_index >= 0
            prop.value_kind = record.answer if proposed else VALUE_KIND_LABEL
        for prop in self.proposals:
            vcol = self.card.column(prop.column_name) if prop.column_name else None
            value = value_for(
                prop.value_kind,
                term_label=prop.candidate.label,
                column_name=prop.column_name or None,
                site_code=prop.site_code or None,
                top_value=top_profile_value(vcol.profile) if vcol is not None else None,
            )
            try:
                prop.avu = build_avu(prop.candidate, value)
            except ValueError as exc:
                logger.info("no AVU for %s (%s)", prop.candidate.curie, exc)
                prop.avu = {}

    # -- Q8 ---------------------------------------------------------------------------------

    def _keep(self) -> list[Proposal]:
        """The keep rule (D25): exact-triple dedup, then the ``max_avus`` best by ``p_fit``. A
        group none of whose proposals survive is ``rejected``."""
        built = [p for p in self.proposals if p.avu]
        for p in self.proposals:
            if not p.avu:
                p.outcome = "rejected"
                self._drop(p, "avu_unbuildable")

        def key(p: Proposal) -> tuple[str, str, str]:
            return triple(p.avu)

        def score(p: Proposal) -> float | None:
            return p.p_fit

        unique = dedup_and_cap(built, triple=key, p_fit=score, cap=len(built))
        kept = dedup_and_cap(built, triple=key, p_fit=score, cap=self.cfg.policy.max_avus)
        unique_ids = {id(p) for p in unique}
        kept_ids = {id(p) for p in kept}
        for p in built:
            if id(p) in kept_ids:
                continue
            p.outcome = "rejected"
            self._drop(p, "duplicate" if id(p) not in unique_ids else "over_cap")
        surviving = {p.group_id for p in kept}
        for gid in {p.group_id for p in self.proposals} - surviving:
            self.buffer.update_group(gid, outcome="rejected")
            parent = self.refined_from.get(gid)
            if parent is not None and parent not in surviving:
                self.buffer.update_group(parent, outcome="rejected")
        return kept

    def _drop(self, p: Proposal, reason: str) -> None:
        self._abstain(
            reason,
            task_id=TASK_TERM,
            scope=p.scope,
            group_id=p.group_id,
            column_name=p.column_name,
            site_code=p.site_code,
            ontology_id=p.ontology_id,
        )

    def _link(self, p: Proposal) -> AvuLinkRow:
        """The link row of a kept proposal (D10): ``auto`` -> ``accepted`` by ``policy``,
        ``proposed`` (and an ``escalated`` one, which waits for a human) -> ``proposed``."""
        status = link_status("proposed" if p.outcome == "escalated" else p.outcome)
        if status is None:  # pragma: no cover - only proposing outcomes are kept
            raise RuntimeError(f"a kept proposal cannot have outcome {p.outcome!r}")
        write_status, accepted_by = status
        return AvuLinkRow(
            link_id=p.link_id,
            run_id=self.buffer.run_id,
            group_id=p.group_id,
            decision_id=p.decision_id,
            attribute=p.avu["attribute"],
            value=p.avu["value"],
            unit=p.avu["unit"],
            term_curie=p.candidate.curie,
            term_iri=p.candidate.iri or None,
            term_label=p.candidate.label,
            ontology_id=p.candidate.ontology_id,
            aspect=p.aspect,
            column_name=p.column_name,
            site_code=p.site_code,
            value_kind=p.value_kind,
            source=f"mesa-clm:annotate:{p.outcome}",
            write_status=write_status,
            accepted_by=accepted_by,
        )
