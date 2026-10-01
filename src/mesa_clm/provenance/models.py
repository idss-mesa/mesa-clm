"""Pydantic rows mirroring the ``mesa_clm`` sidecar DDL (DESIGN D5, D6, D7, D10, D11, D21, D28;
plan §4.6).

One model per table, field order = DDL column order (``tests/unit/test_provenance_ddl.py``
asserts it), every model ``extra="forbid"`` so a misspelled column fails before it reaches SQL.
The validators mirror the DB CHECK constraints one for one and nothing more: the decision-record
invariants that need the whole record (``confidence == max(probs)``, ``raw_probs`` sums to one,
the artifact fingerprint match) belong to ``providers.base._honest``; what the database can
enforce on a row is repeated here so a bad row fails with a field name instead of a DuckDB
``Constraint Error`` naming an expression.

Vocabularies come from :mod:`mesa_clm.vocab` (levels, calibrations, methods, shapes, outcomes,
write statuses, ``accepted_by``, ``via``, label sources, run statuses); the ones only the
sidecar knows (override actions, the resolved history backend, CLM call statuses) are defined
here. ``LabelRow`` is :class:`mesa_clm.provenance.labels.LabelRow`, re-exported so the sidecar
API has one models module. Ids are UUIDs in Python and ``TEXT`` in both dialects (D11 keeps
the two DDLs column-identical).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, Literal, Self, get_args
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mesa_clm import __version__
from mesa_clm.provenance.labels import LabelRow
from mesa_clm.tasks import Scope
from mesa_clm.vocab import (
    CALIBRATIONS,
    LEVEL_RANK,
    LEVELS,
    METHODS_WITHOUT_PROBS,
    AcceptedBy,
    Calibration,
    LabelSource,
    Level,
    Method,
    Outcome,
    RunStatus,
    Shape,
    Via,
    WriteStatus,
)

__all__ = [
    "CALIBRATED_CALIBRATIONS",
    "CALIBRATED_LEVELS",
    "CALL_STATUSES",
    "GROUP_SUMMARY_COLUMNS",
    "HISTORY_BACKENDS",
    "JSON_COLUMNS",
    "LINK_OPS",
    "OVERRIDE_ACTIONS",
    "RUN_FINISH_COLUMNS",
    "SCOPES",
    "AuditRow",
    "AvuLinkRow",
    "CallStatus",
    "ClmCallRow",
    "DecisionGroupRow",
    "DecisionOptionRow",
    "DecisionRow",
    "HistoryBackend",
    "HumanOverrideRow",
    "LabelRow",
    "LinkOp",
    "OverrideAction",
    "RunRow",
]

# -- sidecar-only vocabularies ---------------------------------------------------------------------

# What a human (or an agent, ``via='tool'``) did with a group: ``pick`` one offered candidate,
# an explicit ``none`` ("none of these"; an anchor-positive label row, D21), ``accept`` or
# ``reject`` the winner, ``decline`` to answer (says nothing about the candidates), ``restore``
# a reverted link.
OverrideAction = Literal["pick", "none", "accept", "reject", "decline", "restore"]
OVERRIDE_ACTIONS: Final[tuple[str, ...]] = get_args(OverrideAction)

# The history backend a run's apply *resolved to* (D12). ``auto`` is a configuration knob, never
# a stored value; NULL means apply has not run (or wrote nothing).
HistoryBackend = Literal["direct", "spool", "none"]
HISTORY_BACKENDS: Final[tuple[str, ...]] = get_args(HistoryBackend)

# How a call to clm-serve or the encoder ended: ``ok``; ``error`` (a non-2xx or a malformed
# body); ``timeout``; ``unavailable`` (connection refused or the circuit breaker open).
CallStatus = Literal["ok", "error", "timeout", "unavailable"]
CALL_STATUSES: Final[tuple[str, ...]] = get_args(CallStatus)

LinkOp = Literal["add", "delete"]
LINK_OPS: Final[tuple[str, ...]] = get_args(LinkOp)

SCOPES: Final[tuple[str, ...]] = get_args(Scope)

# Plan §4.6 invariant 2: these levels need a real calibration (D6).
CALIBRATED_LEVELS: Final[tuple[str, ...]] = tuple(
    level for level in LEVELS if LEVEL_RANK[level] >= LEVEL_RANK["calibrated"]
)
CALIBRATED_CALIBRATIONS: Final[tuple[str, ...]] = tuple(
    c for c in CALIBRATIONS if c not in ("none", "uncalibrated")
)

# Columns ``finish_run`` may set besides ``status`` (the rest of a run row is written once).
RUN_FINISH_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "finished_at",
        "terminal_at",
        "exported_at",
        "irods_path",
        "project_id",
        "artifacts_version",
        "labels_sha256",
        "history_backend",
        "history_waiver_actor",
        "degraded",
        "n_decisions",
        "n_clm_calls",
        "n_encoder_tokens",
        "seconds",
    }
)

# Columns ``update_group`` may set: a group summarises a ranking, set once the batch returns.
GROUP_SUMMARY_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "question_key",
        "n_candidates",
        "winner_decision_id",
        "top_p_fit",
        "group_margin",
        "level",
        "method",
        "outcome",
        "anchor_won",
        "escalated_from",
    }
)

# Columns stored as JSON text in DuckDB (JSONB in Postgres); ``_select`` parses them back.
JSON_COLUMNS: Final[frozenset[str]] = frozenset(
    {"plan_json", "state_json", "probs", "raw_probs", "options", "search_json", "offered", "cards"}
)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _non_blank(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("must be a non-empty string (provenance is mandatory)")
    return value


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- runs --------------------------------------------------------------------------------------------


class RunRow(_Row):
    """One ``annotate`` run: who ran it (``owner``, D21), on which card, under which plan,
    serving stack and fingerprints (D5), and how it ended.

    ``provider`` is the method family that answered (``clm`` or ``fake``); ``tier`` the tier
    requested; ``clm_model`` the served head (``clm-latest``, ``clm-raw`` or a promoted head's
    unique name) and ``encoder_model`` the encoder's served name. ``clm_commit``,
    ``schema_sha256``, ``encoder_fp``, ``clm_model_fp``, ``serving_lock_sha`` and
    ``framings_lock_sha`` pin what every decision of the run was made under;
    ``artifacts_version`` and ``labels_sha256`` say which learned artifacts (M4) and label
    snapshot were in force. ``history_backend`` is the backend apply resolved to and
    ``history_waiver_actor`` who accepted ``none`` (D12). ``exported_at`` and ``terminal_at``
    drive prune (plan §7.3 step 9). ``degraded`` marks a run that fell back to ``ols_rank``
    (D28).
    """

    run_id: UUID = Field(default_factory=uuid4)
    owner: str
    status: RunStatus = "running"
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    terminal_at: datetime | None = None
    exported_at: datetime | None = None
    card_name: str
    card_sha256: str
    irods_path: str | None = None
    project_id: str | None = None
    planner: str
    planner_model: str | None = None
    planner_prompt_sha256: str | None = None
    plan_json: dict[str, Any] = Field(default_factory=dict)
    planner_fallback: bool = False
    provider: str
    tier: str = "auto"
    clm_model: str
    encoder_model: str
    clm_commit: str = ""
    schema_sha256: str
    encoder_fp: str
    clm_model_fp: str
    serving_lock_sha: str | None = None
    framings_lock_sha: str
    artifacts_version: str | None = None
    labels_sha256: str | None = None
    mesa_clm_version: str = __version__
    policy_profile: str
    config_sha256: str
    history_backend: HistoryBackend | None = None
    history_waiver_actor: str | None = None
    vm_id: str
    degraded: bool = False
    n_decisions: int | None = None
    n_clm_calls: int | None = None
    n_encoder_tokens: int | None = None
    seconds: float | None = None

    _owner = field_validator("owner", "card_name", "planner", "provider", "vm_id")(_non_blank)


# -- decisions ---------------------------------------------------------------------------------------


class DecisionRow(_Row):
    """One answered question (plan §4.6; one row per candidate for a rank_fit group plus the
    group's summary in :class:`DecisionGroupRow`).

    ``options`` lists the option keys in request order (CURIEs or registry ids plus
    ``__none__`` for rank_fit; the option texts for a choice); ``k`` is their count.
    ``answer_index`` indexes ``options`` (``-1`` = abstain) and ``anchor_index`` the anchor for
    rank_fit (NULL for a choice). ``probs`` is the calibrated distribution (NULL iff
    ``calibration='none'``), ``raw_probs`` CLM's softmax as served, ``confidence = max(probs)``
    computed locally and ``clm_confidence`` CLM's own field (D7). ``s_c`` and ``p_fit`` are
    the winner's set-independent score and fit probability (D2). ``rank`` is the winner's
    position in the ranking (the OLS rank for ``method='ols_rank'``, D28). ``context_*`` and
    ``truncated`` describe the rendered context (D23); ``reason`` says why a record abstained
    or was rejected (``anchor_won``, ``truncated``, ``numeric_underflow``, ``masked``, ...).
    """

    decision_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    seq: int
    group_id: UUID | None = None
    parent_decision_id: UUID | None = None
    task_id: str
    task_key: str
    question_key: str
    framing_id: str
    shape: Shape
    k: int = Field(ge=0)
    scope: Scope
    column_name: str | None = None
    site_code: str | None = None
    state_sha256: str
    state_json: dict[str, Any]
    target_sha256: str
    context_sha256: str
    context_tokens: int = Field(ge=0)
    truncated: bool = False
    method: Method
    model: str = ""
    level: Level
    calibration: Calibration
    probs: list[float] | None = None
    raw_probs: list[float] | None = None
    confidence: float | None = None
    clm_confidence: float | None = None
    s_c: float | None = None
    p_fit: float | None = None
    margin: float | None = None
    anchor_index: int | None = None
    answer_index: int
    answer: str
    options: list[str] = Field(default_factory=list)
    rank: int | None = None
    artifact_version: str | None = None
    feature_spec: str | None = None
    encoder_fp: str
    clm_model_fp: str
    schema_sha256: str
    latency_ms: float | None = None
    threshold_auto: float | None = None
    threshold_propose: float | None = None
    outcome: Outcome
    reason: str | None = None
    ts: datetime = Field(default_factory=_now)

    _text = field_validator("task_id", "task_key", "question_key", "framing_id")(_non_blank)

    @model_validator(mode="after")
    def _mirror_the_checks(self) -> Self:
        # CHECK ((probs IS NULL) = (calibration = 'none'))
        if (self.probs is None) != (self.calibration == "none"):
            raise ValueError("probs must be NULL exactly when calibration is 'none' (D6)")
        # CHECK (method NOT IN (rule, planner, ols_rank, claude:*, unavailable) OR
        #        (probs IS NULL AND level = 'none'))
        if self.method in METHODS_WITHOUT_PROBS and (
            self.probs is not None or self.level != "none"
        ):
            raise ValueError(f"method {self.method!r} carries no distribution: level 'none'")
        # CHECK (level <> 'zero_shot' OR calibration = 'uncalibrated')
        if self.level == "zero_shot" and self.calibration != "uncalibrated":
            raise ValueError("a zero_shot record is 'uncalibrated' (D6)")
        # CHECK (level NOT IN (calibrated, probe, head) OR calibration IN (platt, temperature))
        if self.level in CALIBRATED_LEVELS and self.calibration not in CALIBRATED_CALIBRATIONS:
            raise ValueError(f"level {self.level!r} needs a platt or temperature calibration")
        # CHECK (method <> 'ols_rank' OR (outcome IN (proposed, abstain) AND rank IS NOT NULL))
        if self.method == "ols_rank" and (
            self.outcome not in ("proposed", "abstain") or self.rank is None
        ):
            raise ValueError("ols_rank proposes or abstains and carries the OLS rank (D28)")
        # CHECK (shape <> 'choice' OR anchor_index IS NULL)
        if self.shape == "choice" and self.anchor_index is not None:
            raise ValueError("a choice has no anchor")
        return self


class DecisionOptionRow(_Row):
    """One option of a decision: its request position, key and text, its rank by score, the
    set-independent ``s_c`` and ``p_fit``, the calibrated ``prob`` and served ``raw_prob``,
    whether the aspect mask removed it after scoring (``masked``, plan Q3) and the sha256 of
    the candidate text CLM saw (``action_sha256``, the feature-cache key)."""

    decision_id: UUID
    option_index: int = Field(ge=0)
    option_key: str
    option_text: str
    rank: int | None = None
    s_c: float | None = None
    p_fit: float | None = None
    prob: float | None = None
    raw_prob: float | None = None
    masked: bool = False
    action_sha256: str | None = None

    _key = field_validator("option_key")(_non_blank)


class DecisionGroupRow(_Row):
    """One rank_fit group (a candidate set for one target): the OLS search that produced it
    (``search_json``), the winner and the summary the policy reads (``top_p_fit``,
    ``group_margin``), and ``anchor_won`` when the abstain anchor out-ranked every candidate
    (D2). ``outcome='escalated'`` marks a group waiting for a human; an answer rewrites it
    (``human``/``rejected``)."""

    group_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    task_id: str
    task_key: str
    question_key: str | None = None
    scope: Scope
    column_name: str | None = None
    site_code: str | None = None
    aspect: str | None = None
    ontology_id: str | None = None
    search_json: dict[str, Any] = Field(default_factory=dict)
    n_candidates: int = Field(default=0, ge=0)
    winner_decision_id: UUID | None = None
    top_p_fit: float | None = None
    group_margin: float | None = None
    level: Level | None = None
    method: Method | None = None
    outcome: Outcome
    anchor_won: bool = False
    escalated_from: UUID | None = None
    ts: datetime = Field(default_factory=_now)

    _text = field_validator("task_id", "task_key")(_non_blank)


# -- links -------------------------------------------------------------------------------------------


class AvuLinkRow(_Row):
    """The AVU a decision proposes and what became of it (D10, D12, D13).

    ``write_status`` follows :data:`mesa_clm.vocab.WRITE_STATUSES`; ``accepted_by`` says who
    accepted it (``policy`` for a stored auto, ``human``, ``agent``) and is mandatory once
    accepted (CHECK). ``snapshot_id`` is the mesa-ducklake snapshot the write reached (direct
    mode, or filled by ``reconcile`` after the recorder coalesced a spool batch);
    ``spool_batch_id`` the ``mesa-spool/1`` batch. ``source`` is the history source tag
    (``mesa-clm:<tool>:<outcome>``). ``unit``, ``column_name`` and ``site_code`` are ``''``
    sentinels, never NULL, so ``UNIQUE (run_id, attribute, value, unit, column_name,
    site_code)`` holds in DuckDB 1.5.x (no ``NULLS NOT DISTINCT``; D11).
    """

    link_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    group_id: UUID | None = None
    decision_id: UUID | None = None
    project_id: str | None = None
    snapshot_id: int | None = None
    spool_batch_id: str | None = None
    irods_path: str | None = None
    target_type: str = "data_object"
    attribute: str
    value: str
    unit: str = ""
    op: LinkOp = "add"
    term_curie: str | None = None
    term_iri: str | None = None
    term_label: str | None = None
    ontology_id: str | None = None
    aspect: str | None = None
    column_name: str = ""
    site_code: str = ""
    value_kind: str | None = None
    source: str | None = None
    write_status: WriteStatus = "proposed"
    accepted_by: AcceptedBy | None = None
    duplicate_of: UUID | None = None
    written_at: datetime | None = None

    _avu = field_validator("attribute", "value")(_non_blank)

    @model_validator(mode="after")
    def _accepted_needs_a_by(self) -> Self:
        # CHECK (write_status <> 'accepted' OR accepted_by IS NOT NULL)  (D10)
        if self.write_status == "accepted" and self.accepted_by is None:
            raise ValueError("an accepted link records accepted_by (policy|human|agent; D10)")
        return self


# -- human overrides ---------------------------------------------------------------------------------


class HumanOverrideRow(_Row):
    """What a reviewer did with a group and how (D21): ``via='elicitation'`` (MRTR) or ``cli``
    produce curator labels; ``via='tool'`` (a plain ``mesa_clm_feedback`` call) can only produce
    ``agent_pick`` labels (CHECK). ``offered`` records the candidates as presented (decision
    id, option key, rank, ``p_fit``, level) so a pick can be audited against what was shown.
    ``label_source`` and ``labels_written`` say what the pick minted."""

    override_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    group_id: UUID | None = None
    decision_id: UUID | None = None
    link_id: UUID | None = None
    actor: str
    via: Via
    action: OverrideAction
    chosen_decision_id: UUID | None = None
    chosen_option_key: str | None = None
    elicitation_key: str | None = None
    label_source: LabelSource | None = None
    labels_written: int = Field(default=0, ge=0)
    offered: list[dict[str, Any]] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now)

    _actor = field_validator("actor")(_non_blank)

    @model_validator(mode="after")
    def _tool_picks_are_agent_picks(self) -> Self:
        # CHECK (via <> 'tool' OR label_source IS NULL OR label_source = 'agent_pick')  (D21)
        if self.via == "tool" and self.label_source not in (None, "agent_pick"):
            raise ValueError("a tool pick can only mint agent_pick labels (D21, defect g)")
        return self


# -- audits ------------------------------------------------------------------------------------------


class AuditRow(_Row):
    """A curator audit of would-be-auto decisions (plan §4.7): ``n`` decisions from ``n_cards``
    non-bench cards (``cards`` lists them), ``n_errors`` found by ``reviewer``, the one-sided
    95% Clopper-Pearson upper bound on the error rate and the ``risk`` it was held against;
    ``passed`` when ``cp95_upper <= 2 * risk`` with ``n >= 50`` and ``n_cards >= 3``."""

    audit_id: UUID = Field(default_factory=uuid4)
    task_key: str
    artifact_version: str
    n: int = Field(ge=0)
    n_cards: int = Field(ge=0)
    cards: list[str] = Field(default_factory=list)
    reviewer: str
    n_errors: int = Field(ge=0)
    cp95_upper: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    passed: bool
    created_at: datetime = Field(default_factory=_now)

    _text = field_validator("task_key", "artifact_version", "reviewer")(_non_blank)

    @model_validator(mode="after")
    def _errors_within_n(self) -> Self:
        # CHECK (n_errors <= n)
        if self.n_errors > self.n:
            raise ValueError("n_errors cannot exceed n")
        return self


# -- clm calls ---------------------------------------------------------------------------------------


class ClmCallRow(_Row):
    """One HTTP call to clm-serve or the encoder: endpoint, served model, how many questions
    and candidate texts it carried, the input tokens (when the server reports them), latency,
    whether clm-serve's action cache answered (``cache_hit``) and how it ended. ``run_id`` is
    NULL for calls outside a run (``features build``, the doctor)."""

    call_id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    endpoint: str
    model: str
    n_questions: int = Field(default=0, ge=0)
    n_candidates: int = Field(default=0, ge=0)
    input_tokens: int | None = None
    latency_ms: float = Field(ge=0.0)
    cache_hit: bool | None = None
    status: CallStatus = "ok"
    ts: datetime = Field(default_factory=_now)

    _text = field_validator("endpoint", "model")(_non_blank)
