"""The provider contract: one :class:`DecisionRecord` per answered question, honest about its
numbers (DESIGN D2, D6, D7, D22, D28; plan §4.1, §4.5, §4.6).

Every provider (the CLM tiers, the degraded ``ols_rank`` method, Claude's second opinion)
returns ``DecisionRecord``\\ s and the sidecar stores them. Two fields say how far to trust the
numbers: ``level`` names what produced them (``zero_shot``, ``calibrated``, ``probe`` and
``head`` for CLM tiers; ``none`` for everything that is not a CLM distribution) and
``calibration`` how they were mapped (``uncalibrated`` at zero shot, ``platt`` or
``temperature`` once an artifact is applied, ``none`` when there is no distribution at all).
``probs`` is ``None`` exactly when ``calibration`` is ``none``: a rule, the OLS rank or Claude
never fabricate a distribution, and nothing is ever one-hot by construction.

Shapes (D2): a ``rank_fit`` record answers one ``/v1/systemone`` Choice over a candidate group
plus the fixed abstain anchor ``registry.ANCHOR_KEY``. Its ``s_c`` is the per-option
``ln p_c - ln p_anchor`` recovered from CLM's served distribution (``raw_probs``), ``0`` for the
anchor, and independent of which other candidates were offered; ``p_fit = σ(s_c)`` at zero shot
and ``σ(a·s_c + b)`` once a Platt calibrator applies (D7). A ``choice`` record answers a closed
option set and carries no anchor, ``s_c`` or ``p_fit``. In both, ``confidence = max(probs)`` is
computed here; CLM's own ``confidence`` field (``p_top - mean(p_rest)``) is kept only as
``clm_confidence`` and never read by the policy.

:meth:`DecisionRecord._honest` enforces the eight invariants of plan §4.6 (mirrored by the
sidecar CHECKs in ``provenance.models``) plus the consistency they rely on: the answer is the
argmax of ``probs``, a zero-shot record carries CLM's distribution untransformed, ``s_c`` really
is the log-ratio of ``raw_probs``, the identity hashes match the state they claim to hash.
Records are frozen; :meth:`DecisionRecord.revise` is the only way to change one and it
re-validates. :func:`apply_mask` is the pure masking step (the AnyJev 2048 pattern: zero the
out-of-play options, renormalise, log the masked mass; a fully masked record abstains) that the
tiered provider applies for ``mask_rule='aspect'`` and ``policy.masked`` wraps.
"""

from __future__ import annotations

import hashlib
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NamedTuple, Protocol, Self, get_args, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mesa_clm.clm.fingerprint import Fingerprint
from mesa_clm.framings import CandidateLike, Framing, FramingError, build_question
from mesa_clm.identity import IdentityError
from mesa_clm.identity import target_sha256 as _target_sha256
from mesa_clm.registry import ANCHOR_KEY, allowed_for_aspect
from mesa_clm.states import state_sha256 as _state_sha256
from mesa_clm.tasks import RANK_FIT_TASKS, TASKS, Kind
from mesa_clm.vocab import LEVEL_RANK, METHODS_WITHOUT_PROBS, Calibration, Level, Method, Shape

__all__ = [
    "CALL_STATUSES",
    "CLM_METHODS",
    "FEATURE_SPEC_LEVELS",
    "REASONS",
    "ArtifactRef",
    "CallStatus",
    "ClmCall",
    "DeciderRefused",
    "DecisionProvider",
    "DecisionRecord",
    "FramingOptions",
    "Scalar",
    "TierUnavailable",
    "apply_mask",
    "aspect_keep",
    "base_fields",
    "check_batch",
    "framing_options",
    "meets_level",
    "rank_probs",
    "sha256_text",
    "sigmoid",
]

Scalar = float | int | str | bool | None

# The methods whose records carry CLM's served distribution (``raw_probs``) and a CLM tier.
CLM_METHODS: Final[frozenset[str]] = frozenset({"clm", "fake"})

# Tolerances: a served distribution sums to one within 1e-3 (plan §4.6 invariant 1); every
# number derived here from another (confidence, margin, p_fit at zero shot, a renormalised
# probability) must match its derivation to 1e-9; ``s_c`` must match the log-ratio of the
# served probabilities to 1e-6 relative (it is computed from them, so the slack is generous).
SUM_TOL: Final[float] = 1e-3
DERIVED_TOL: Final[float] = 1e-9
SC_TOL: Final[float] = 1e-6

# The provider-level reasons a record abstains, kept in ``diagnostics["reason"]`` (the sidecar
# ``decisions.reason`` column is the policy's, which may add its own: anchor_won, below_propose).
REASONS: Final[tuple[str, ...]] = (
    "truncated",  # the context exceeds the encoder window: never sent (D23)
    "decider_unavailable",  # clm-serve unreachable, refused the request or the breaker is open
    "numeric_underflow",  # a served probability is not > 0 (plan §4.1): no s_c, abstain
    "malformed_answer",  # the answer does not cover exactly the offered options
    "mask_empty",  # every option the mask may remove was removed
    "refusal",  # Claude declined (stop_reason refusal)
    "unparsed",  # Claude answered something that is not one of the options
    "error",  # the second-opinion call failed
)

_HEX: Final[frozenset[str]] = frozenset(string.hexdigits.lower())
# Plan §4.6 invariant 8: the identity fields and their hex widths (task/question keys 16,
# encoder/model fingerprints 12, the sha256s 64).
_HEX_FIELDS: Final[dict[str, int]] = {
    "task_key": 16,
    "question_key": 16,
    "encoder_fp": 12,
    "clm_model_fp": 12,
    "schema_sha256": 64,
    "state_sha256": 64,
    "target_sha256": 64,
    "context_sha256": 64,
}
# Plan §4.6 invariant 2: the calibrations each level may carry.
_LEVEL_CALIBRATIONS: Final[dict[str, frozenset[str]]] = {
    "none": frozenset({"none"}),
    "zero_shot": frozenset({"uncalibrated"}),
    "calibrated": frozenset({"platt", "temperature"}),
    "probe": frozenset({"platt", "temperature"}),
    "head": frozenset({"platt", "temperature"}),
}

# The levels whose numbers come from a feature specification (plan §5.3): ``feature_spec`` is
# set on exactly these records (M4; ``_check_feature_spec``).
FEATURE_SPEC_LEVELS: Final[frozenset[str]] = frozenset({"probe", "head"})

# How a call to clm-serve ended; the same vocabulary as ``provenance.models.CallStatus``
# (``tests/unit/test_tiered_provider.py`` asserts they agree).
CallStatus = Literal["ok", "error", "timeout", "unavailable"]
CALL_STATUSES: Final[tuple[str, ...]] = get_args(CallStatus)


class TierUnavailable(RuntimeError):
    """The requested tier cannot be served honestly by this provider (no artifact for the
    question key, a tier that lands in a later milestone, a tier the provider does not have)."""


class DeciderRefused(RuntimeError):
    """clm-serve refused a request for a reason no fallback may paper over: the key
    (401/403) or the route (404). ``status`` is the HTTP status. A transport error, a 5xx or a
    timeout is not this: those records become ``unavailable`` and the group falls back to
    ``ols_rank`` (D28)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# -- small maths ------------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """``1 / (1 + e^-x)`` without overflow for large ``|x|``."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sha256_text(text: str) -> str:
    """Full hex sha256 of ``text`` as UTF-8 (``context_sha256``, prompt hashes)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rank_probs(probs: Sequence[float]) -> tuple[int, float, float]:
    """``(argmax, confidence, margin)`` of a distribution: the first index of the largest
    probability (CLM's own tie rule), that probability, and its gap to the runner-up (``0``
    for a single option). ``ValueError`` on an empty sequence."""
    if not probs:
        raise ValueError("rank_probs needs at least one probability")
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top = float(probs[order[0]])
    second = float(probs[order[1]]) if len(order) > 1 else top
    return order[0], top, top - second


# -- the record -------------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRef(_Frozen):
    """The learned artifact that transformed a record's numbers (plan §4.6 invariant 5): its
    version and the ``(question_key, encoder_fp, clm_model_fp)`` it was fitted under, which must
    equal the record's own (D5, K4)."""

    version: str = Field(min_length=1)
    question_key: str
    encoder_fp: str
    clm_model_fp: str


class DecisionRecord(_Frozen):
    """One answered question (one candidate group for a rank_fit), ready for the sidecar.

    Identity: ``task_id``/``task_key``/``kind`` name the frozen task (``kind`` is its
    mesa-anyjev kind: ``term.fits`` is a ``noul`` task asked as a ``rank_fit``);
    ``question_key``/``framing_id``/``shape`` the framing it was asked under; ``state`` is the
    builder state the caller passed (``state_sha256`` hashes it, ``target_sha256`` is its D1
    label identity) and ``context_sha256``/``context_tokens`` describe the exact text the state
    head saw (the prompt, for Claude). ``encoder_fp``, ``clm_model_fp`` and ``schema_sha256``
    are the D5 fingerprints of the serving stack the run used.

    Options: ``options`` are the option keys in request order: candidate CURIEs or registry ids
    followed by ``__none__`` for a rank_fit (``anchor_index`` points at it), the mesa-anyjev
    option labels for a closed choice. ``option_texts`` are the texts CLM embedded, in the same
    order. ``answer_index`` is ``-1`` for an abstain (``answer`` is then ``''``); an anchor
    answer is a real answer (``answer == '__none__'``) that the policy turns into an abstain.

    Numbers (per option, request order): ``raw_probs`` CLM's softmax as served; ``probs`` the
    distribution after the tier's calibrator and any mask; ``s_c``/``p_fit`` the rank_fit
    scores; ``masked`` which options a mask removed. ``confidence = max(probs)`` and
    ``margin = top1 - top2`` of ``probs``; ``clm_confidence`` is CLM's field. ``rank`` is the
    OLS rank of the answered candidate for ``method='ols_rank'`` (D28). ``diagnostics`` holds
    scalars only; ``diagnostics['reason']`` says why a record abstained at the provider.
    """

    task_id: str
    task_key: str
    question_key: str
    framing_id: str = Field(min_length=1)
    shape: Shape
    kind: Kind
    options: list[str]
    option_texts: list[str]
    state: dict[str, Any]
    state_sha256: str
    target_sha256: str
    context_sha256: str
    context_tokens: int = Field(ge=0)
    truncated: bool = False
    provider: str = Field(min_length=1)
    method: Method
    model: str
    served_model: str | None = None
    level: Level
    calibration: Calibration
    probs: list[float] | None = None
    raw_probs: list[float] | None = None
    s_c: list[float] | None = None
    p_fit: list[float] | None = None
    masked: list[bool] | None = None
    anchor_index: int | None = None
    answer_index: int
    answer: str
    rank: int | None = None
    confidence: float | None = None
    clm_confidence: float | None = None
    margin: float | None = None
    artifact_version: str | None = None
    artifact: ArtifactRef | None = None
    encoder_fp: str
    clm_model_fp: str
    schema_sha256: str
    latency_ms: float | None = Field(default=None, ge=0.0)
    feature_spec: str | None = None
    diagnostics: dict[str, Scalar] = Field(default_factory=dict)

    # -- derived views --------------------------------------------------------------------------

    @property
    def k(self) -> int:
        """The number of options (candidates plus the anchor for a rank_fit)."""
        return len(self.options)

    @property
    def anchor_won(self) -> bool:
        """The abstain anchor is the answer (D2): the policy abstains (``anchor_won``)."""
        return self.anchor_index is not None and self.answer_index == self.anchor_index

    @property
    def abstained(self) -> bool:
        """No answer at all (``answer_index == -1``): truncated, unavailable, fully masked."""
        return self.answer_index < 0

    @property
    def reason(self) -> str | None:
        """The provider-level abstain reason (``diagnostics['reason']``), if any."""
        value = self.diagnostics.get("reason")
        return None if value is None else str(value)

    @property
    def answer_s_c(self) -> float | None:
        """``s_c`` of the answered option (rank_fit with an answer), else ``None``."""
        if self.s_c is None or self.answer_index < 0:
            return None
        return self.s_c[self.answer_index]

    @property
    def answer_p_fit(self) -> float | None:
        """``p_fit`` of the answered option: the statistic a rank_fit threshold reads."""
        if self.p_fit is None or self.answer_index < 0:
            return None
        return self.p_fit[self.answer_index]

    def revise(self, **update: Any) -> DecisionRecord:
        """A copy with ``update`` applied and every invariant re-checked (``model_copy`` would
        skip validation, which is how a dishonest record gets written)."""
        return DecisionRecord.model_validate({**self.model_dump(), **update})

    # -- plan §4.6 ------------------------------------------------------------------------------

    @model_validator(mode="after")
    def _honest(self) -> Self:
        _check_identity(self)
        _check_options(self)
        _check_levels(self)
        _check_distributions(self)
        _check_shape(self)
        _check_mask(self)
        _check_artifact(self)
        _check_feature_spec(self)
        _check_abstain(self)
        _check_ols_rank(self)
        return self


def _is_hex(value: str, width: int) -> bool:
    return len(value) == width and all(c in _HEX for c in value)


def _check_identity(r: DecisionRecord) -> None:
    """Invariant 8: every record carries the task, framing, fingerprint and hash identity, and
    each hash really is the hash it claims to be."""
    for name, width in _HEX_FIELDS.items():
        value = getattr(r, name)
        if not _is_hex(value, width):
            raise ValueError(
                f"invariant 8: {name} must be {width} lower-case hex characters, got {value!r}"
            )
    task = TASKS.get(r.task_id)
    if task is None:
        raise ValueError(f"invariant 8: unknown task {r.task_id!r}")
    if r.task_key != task.key:
        raise ValueError(f"invariant 8: task_key {r.task_key} is not {r.task_id}'s ({task.key})")
    if r.kind != task.kind:
        raise ValueError(f"invariant 8: {r.task_id} is a {task.kind} task, not {r.kind}")
    expected_shape = "rank_fit" if r.task_id in RANK_FIT_TASKS else "choice"
    if r.shape != expected_shape:
        raise ValueError(f"invariant 8: {r.task_id} is asked as a {expected_shape}")
    if r.state_sha256 != _state_sha256(r.state):
        raise ValueError("invariant 8: state_sha256 does not hash the recorded state")
    try:
        target = _target_sha256(r.task_id, r.state)
    except IdentityError as exc:
        raise ValueError(f"invariant 8: the state has no target identity: {exc}") from None
    if r.target_sha256 != target:
        raise ValueError("invariant 8: target_sha256 is not the D1 identity of the state")


def _check_options(r: DecisionRecord) -> None:
    k = r.k
    if k < 2:
        raise ValueError("a decision offers at least two options")
    if any(not o for o in r.options) or len(set(r.options)) != k:
        raise ValueError("options must be unique non-empty keys")
    if len(r.option_texts) != k:
        raise ValueError("option_texts must have one entry per option")
    if not -1 <= r.answer_index < k:
        raise ValueError(f"answer_index {r.answer_index} out of range for {k} options")
    expected = r.options[r.answer_index] if r.answer_index >= 0 else ""
    if r.answer != expected:
        raise ValueError(f"answer {r.answer!r} is not option {r.answer_index} ({expected!r})")
    if r.rank is not None and r.rank < 1:
        raise ValueError("rank is 1-based")


def _check_levels(r: DecisionRecord) -> None:
    """Invariant 2: zero_shot is uncalibrated, the learned tiers are platt|temperature, and
    every method without a CLM distribution sits at level none."""
    if r.calibration not in _LEVEL_CALIBRATIONS[r.level]:
        raise ValueError(
            f"invariant 2: level {r.level!r} cannot carry calibration {r.calibration!r} "
            f"(allowed: {sorted(_LEVEL_CALIBRATIONS[r.level])})"
        )
    if r.method in METHODS_WITHOUT_PROBS and r.level != "none":
        raise ValueError(f"invariant 2: method {r.method!r} is level 'none' (D6)")
    if r.method in CLM_METHODS and r.level == "none":
        raise ValueError(f"invariant 2: a {r.method!r} decision carries a CLM tier, not 'none'")


def _check_distribution(name: str, values: list[float], k: int) -> None:
    if len(values) != k:
        raise ValueError(f"invariant 1: {name} must have one entry per option ({k})")
    if any(not math.isfinite(p) or p < 0.0 for p in values):
        raise ValueError(f"invariant 1: {name} must be finite and non-negative")
    if abs(math.fsum(values) - 1.0) > SUM_TOL:
        raise ValueError(f"invariant 1: {name} must sum to 1 (±{SUM_TOL})")


def _check_distributions(r: DecisionRecord) -> None:
    """Invariant 1 (probs iff calibrated at all; CLM methods carry the served distribution) and
    invariant 4 (confidence and margin are read off ``probs``, the answer is its argmax)."""
    if (r.probs is None) != (r.calibration == "none"):
        raise ValueError("invariant 1: probs must be None exactly when calibration is 'none'")
    if r.method in CLM_METHODS and r.raw_probs is None:
        raise ValueError(f"invariant 1: a {r.method!r} decision carries raw_probs as served")
    if r.method not in CLM_METHODS and r.raw_probs is not None:
        raise ValueError(f"invariant 1: method {r.method!r} has no served distribution")
    if r.raw_probs is not None:
        _check_distribution("raw_probs", r.raw_probs, r.k)
        if min(r.raw_probs) <= 0.0:
            # CLM's scale is clamped at 100, so the smallest served probability is ~e^-200 > 0;
            # a zero means underflow, which the provider records as an abstain (plan §4.1).
            raise ValueError("invariant 1: a served probability is not > 0 (numeric_underflow)")
    if r.clm_confidence is not None and (r.raw_probs is None or not 0.0 <= r.clm_confidence <= 1.0):
        raise ValueError("clm_confidence is CLM's field in [0, 1], only with raw_probs (D7)")
    if r.probs is None:
        if r.confidence is not None or r.margin is not None:
            raise ValueError("invariant 4: confidence and margin need probs")
        return
    _check_distribution("probs", r.probs, r.k)
    _, top, margin = rank_probs(r.probs)
    if r.confidence is None or abs(r.confidence - top) > DERIVED_TOL:
        raise ValueError("invariant 4: confidence must equal max(probs), computed locally (D7)")
    if r.margin is None or abs(r.margin - margin) > DERIVED_TOL:
        raise ValueError("invariant 4: margin must equal top1 - top2 of probs")
    if r.answer_index >= 0 and r.probs[r.answer_index] < top - DERIVED_TOL:
        raise ValueError("invariant 4: the answer must be the argmax of probs")


def _check_shape(r: DecisionRecord) -> None:
    """Invariant 3: a rank_fit carries the anchor and per-option ``s_c``/``p_fit`` (when CLM
    answered); a choice carries neither. At the served tiers ``s_c`` must be the log-ratio of
    ``raw_probs`` and, at zero shot, ``p_fit`` exactly ``σ(s_c)``."""
    k = r.k
    if r.shape == "choice":
        if r.anchor_index is not None or ANCHOR_KEY in r.options:
            raise ValueError("invariant 3: a closed choice has no anchor (plan §4.1)")
        if r.s_c is not None or r.p_fit is not None:
            raise ValueError("invariant 3: s_c and p_fit belong to rank_fit decisions")
        return
    a = r.anchor_index
    if a is None or not 0 <= a < k or r.options[a] != ANCHOR_KEY:
        raise ValueError(f"invariant 3: a rank_fit carries the {ANCHOR_KEY!r} anchor (D2)")
    if r.raw_probs is None:
        if r.s_c is not None or r.p_fit is not None:
            raise ValueError("invariant 3: s_c and p_fit need a served distribution")
        return
    if r.s_c is None or r.p_fit is None or len(r.s_c) != k or len(r.p_fit) != k:
        raise ValueError("invariant 3: a rank_fit carries s_c and p_fit per option")
    if any(not math.isfinite(s) for s in r.s_c) or r.s_c[a] != 0.0:
        raise ValueError("invariant 3: s_c is finite and 0 for the anchor")
    if any(not 0.0 <= p <= 1.0 for p in r.p_fit):
        raise ValueError("invariant 3: p_fit is a probability")
    if r.level in ("zero_shot", "calibrated"):
        # Served tiers: s_c comes from CLM's distribution and nothing else (D2).
        log_a = math.log(r.raw_probs[a])
        for s, p in zip(r.s_c, r.raw_probs, strict=True):
            expected = math.log(p) - log_a
            if abs(s - expected) > SC_TOL * max(1.0, abs(expected)):
                raise ValueError("invariant 3: s_c must be ln p_c - ln p_anchor of raw_probs")
    if r.level == "zero_shot" and any(
        abs(p - sigmoid(s)) > DERIVED_TOL for s, p in zip(r.s_c, r.p_fit, strict=True)
    ):
        raise ValueError("invariant 3: at zero_shot p_fit is σ(s_c), uncalibrated (D7)")


def _mask_is_empty(r: DecisionRecord, masked: list[bool]) -> bool:
    return all(m for i, m in enumerate(masked) if i != r.anchor_index)


def _check_mask(r: DecisionRecord) -> None:
    """A mask removes options after scoring: removed options carry probability 0, the anchor is
    never removed, a fully masked record abstains, and at zero shot ``probs`` is still CLM's
    distribution, renormalised over what the masks kept (no other transform).

    A fully masked record keeps the distribution it had before the mask that emptied it
    (:func:`apply_mask` never renormalises onto the anchor alone). That distribution may itself
    be the renormalisation of an *earlier* mask (the provider masks by aspect, the pipeline then
    by the ontologies in play), so the zero-shot check reads the support off ``probs`` instead of
    off the final flags: ``probs`` must be ``raw_probs`` restricted to its own support and
    renormalised (``raw_probs`` itself when nothing was removed), and that support must contain
    every option the flags keep. For a partially masked record the support is exactly the kept
    options (served probabilities are ``> 0``), which is the old rule."""
    masked = r.masked
    if masked is not None:
        if len(masked) != r.k:
            raise ValueError("masked must have one flag per option")
        if r.anchor_index is not None and masked[r.anchor_index]:
            raise ValueError("the abstain anchor is never masked")
    empty = masked is not None and _mask_is_empty(r, masked)
    if empty and r.answer_index >= 0:
        raise ValueError("a fully masked record abstains")
    if masked is not None and r.answer_index >= 0 and masked[r.answer_index]:
        raise ValueError("a masked option cannot be the answer")
    renormalised = masked is not None and not empty and r.reason != "mask_empty"
    if (
        renormalised
        and masked is not None
        and r.probs is not None
        and any(p != 0.0 for p, m in zip(r.probs, masked, strict=True) if m)
    ):
        raise ValueError("a masked option carries probability 0")
    if r.level == "zero_shot" and r.probs is not None and r.raw_probs is not None:
        _check_zero_shot_probs(r.probs, r.raw_probs, masked)


def _check_zero_shot_probs(
    probs: list[float], raw_probs: list[float], masked: list[bool] | None
) -> None:
    """At zero shot the only transforms are masks: ``probs`` is ``raw_probs`` renormalised over
    the options it still gives mass to, and no option the flags keep has lost its mass."""
    support = [p > 0.0 for p in probs]
    kept_by_flags = [True] * len(probs) if masked is None else [not m for m in masked]
    if any(k and not s for k, s in zip(kept_by_flags, support, strict=True)):
        raise ValueError("a zero_shot record's probs drop an option no mask removed")
    if sum(support) < 2:
        # A mask always keeps a candidate next to the anchor, and an emptying mask leaves the
        # previous distribution: all mass on one option is fabricated, never renormalised.
        raise ValueError("a zero_shot record's probs put all mass on one option")
    if all(support):
        expected = list(raw_probs)  # nothing removed: CLM's distribution bit for bit
    else:
        kept = math.fsum(p for p, s in zip(raw_probs, support, strict=True) if s)
        expected = [p / kept if s else 0.0 for p, s in zip(raw_probs, support, strict=True)]
    if any(abs(p - q) > DERIVED_TOL for p, q in zip(probs, expected, strict=True)):
        raise ValueError("a zero_shot record's probs are CLM's raw_probs, untransformed")


def _check_artifact(r: DecisionRecord) -> None:
    """Invariant 5: a learned tier names the artifact that transformed it, fitted under this
    record's own question key and fingerprints; zero_shot and none name none."""
    if LEVEL_RANK[r.level] <= LEVEL_RANK["zero_shot"]:
        if r.artifact_version is not None or r.artifact is not None:
            raise ValueError(f"invariant 5: a {r.level!r} record applies no artifact")
        return
    if r.artifact_version is None or r.artifact is None:
        raise ValueError(f"invariant 5: level {r.level!r} needs artifact_version and artifact")
    art = r.artifact
    if art.version != r.artifact_version:
        raise ValueError("invariant 5: artifact.version must equal artifact_version")
    mine = (r.question_key, r.encoder_fp, r.clm_model_fp)
    theirs = (art.question_key, art.encoder_fp, art.clm_model_fp)
    if mine != theirs:
        raise ValueError(
            "invariant 5: the artifact's (question_key, encoder_fp, clm_model_fp) "
            f"{theirs} differs from the record's {mine} (K4)"
        )


def _check_feature_spec(r: DecisionRecord) -> None:
    """``feature_spec`` names the probe (or head) feature specification a record's numbers came
    from (plan §5.3), so it is set exactly when the level is ``probe`` or ``head`` (M4, R4). The
    sidecar's DDL carries no CHECK for it: the column is free text and the rule lives here."""
    learned = r.level in FEATURE_SPEC_LEVELS
    if learned and not r.feature_spec:
        raise ValueError(f"level {r.level!r} names its feature_spec (plan §5.3)")
    if not learned and r.feature_spec is not None:
        raise ValueError(f"a {r.level!r} record applies no feature spec (feature_spec is None)")


def _check_abstain(r: DecisionRecord) -> None:
    """Invariant 6 (the record side): a truncated context never yields an answer."""
    if r.truncated and r.answer_index >= 0:
        raise ValueError("invariant 6: a truncated context abstains (D23)")


def _check_ols_rank(r: DecisionRecord) -> None:
    """Invariant 7 (the record side): the degraded method proposes an OLS-ranked candidate,
    with its rank and without any numbers; the policy keeps it at proposed|abstain."""
    if r.method != "ols_rank":
        return
    if r.shape != "rank_fit":
        raise ValueError("invariant 7: ols_rank ranks a candidate group (rank_fit)")
    if r.rank is None:
        raise ValueError("invariant 7: an ols_rank record carries the OLS rank (D28)")
    if r.answer_index < 0 or r.answer_index == r.anchor_index:
        raise ValueError("invariant 7: ols_rank answers a candidate, never the anchor")


def meets_level(record: DecisionRecord, required: Level) -> bool:
    """Whether ``record``'s tier is at or above ``required`` (``LEVEL_RANK``, D6)."""
    return LEVEL_RANK[record.level] >= LEVEL_RANK[required]


# -- masking (plan §4.2 Q3) -------------------------------------------------------------------


def aspect_keep(
    options: Sequence[str], aspect: str, in_play: frozenset[str] | None = None
) -> list[bool]:
    """Per-option "in play" flags for ``column.ontology_fits`` under an aspect: a registry
    ontology id is kept when ``registry.allowed_for_aspect(aspect)`` (intersected with
    ``in_play`` when given) contains it; the anchor is always kept; any other key is out."""
    allowed = allowed_for_aspect(aspect)
    if in_play is not None:
        allowed = allowed & frozenset(i.lower() for i in in_play)
    return [o == ANCHOR_KEY or o.lower() in allowed for o in options]


def apply_mask(record: DecisionRecord, keep: Sequence[bool]) -> DecisionRecord:
    """Remove the options ``keep`` marks out of play, after scoring (the 2048 pattern).

    Removed options get probability 0, ``probs`` is renormalised over the rest, the answer,
    ``confidence`` and ``margin`` follow, and ``diagnostics['masked_mass']`` accumulates the
    probability mass removed (relative to the unmasked distribution, over repeated masks).
    ``s_c`` and ``p_fit`` are untouched: they are set-independent (D2). The anchor is never
    removed. When every removable option is out, the record abstains (``answer_index=-1``,
    ``reason='mask_empty'``) and ``probs`` is left as it was rather than renormalised onto the
    anchor alone. A record without a distribution (``ols_rank``, Claude, unavailable) comes
    back unchanged: there is nothing to renormalise, so its caller filters candidates first.
    """
    if len(keep) != record.k:
        raise ValueError(f"the mask needs one flag per option ({record.k}), got {len(keep)}")
    if record.probs is None:
        return record
    in_play = [bool(x) for x in keep]
    if record.anchor_index is not None:
        in_play[record.anchor_index] = True
    before = record.masked or [False] * record.k
    masked = [m or not kp for m, kp in zip(before, in_play, strict=True)]
    diagnostics: dict[str, Scalar] = dict(record.diagnostics)
    if masked == before:
        # Nothing newly removed: record the flags, leave the distribution bit-for-bit alone.
        diagnostics.setdefault("masked_mass", 0.0)
        return record.revise(masked=masked, diagnostics=diagnostics)
    kept = math.fsum(p for p, m in zip(record.probs, masked, strict=True) if not m)
    removed_now = min(max(1.0 - kept, 0.0), 1.0)
    prior = diagnostics.get("masked_mass")
    prior_mass = float(prior) if isinstance(prior, int | float) else 0.0
    diagnostics["masked_mass"] = round(1.0 - (1.0 - prior_mass) * (1.0 - removed_now), 12)
    empty = all(m for i, m in enumerate(masked) if i != record.anchor_index)
    if empty or kept <= 0.0:
        diagnostics["mask_empty"] = True
        diagnostics["reason"] = "mask_empty"
        return record.revise(masked=masked, answer_index=-1, answer="", diagnostics=diagnostics)
    probs = [0.0 if m else p / kept for p, m in zip(record.probs, masked, strict=True)]
    idx, top, margin = rank_probs(probs)
    return record.revise(
        probs=probs,
        masked=masked,
        answer_index=idx,
        answer=record.options[idx],
        confidence=top,
        margin=margin,
        diagnostics=diagnostics,
    )


# -- building records -------------------------------------------------------------------------


class FramingOptions(NamedTuple):
    """What one question offers: the wire question, its criteria keys in request order, the
    record's option keys and texts (same order) and the anchor's index (rank_fit only)."""

    question: dict[str, Any]
    wire_keys: list[str]
    options: list[str]
    option_texts: list[str]
    anchor_index: int | None


def framing_options(framing: Framing, candidates: Sequence[CandidateLike] | None) -> FramingOptions:
    """The options of ``framing`` over ``candidates`` (``None`` for a closed choice), through
    ``framings.build_question`` so the texts are exactly what CLM embeds. A rank_fit's option
    keys are its wire keys (CURIE / registry id, then ``__none__``); a closed choice's are the
    mesa-anyjev labels its ``label_map`` maps the wire keys to. The F1 noul control is asked
    offline by the X1 bench, never by a provider: :class:`FramingError`."""
    if framing.control:
        raise FramingError(
            f"{framing.task_id}/{framing.id} is the X1 noul control: it is scored offline by "
            "`bench framing`, not asked by a decision provider"
        )
    question = build_question(framing, candidates)
    criteria: dict[str, Any] = question["criteria"]
    wire_keys = [str(k) for k in criteria]
    texts = [str(v) for v in criteria.values()]
    if framing.shape == "rank_fit":
        return FramingOptions(question, wire_keys, wire_keys, texts, wire_keys.index(ANCHOR_KEY))
    label_map = framing.label_map or {}
    return FramingOptions(question, wire_keys, [label_map[k] for k in wire_keys], texts, None)


def base_fields(
    framing: Framing,
    state: Mapping[str, Any],
    fingerprint: Fingerprint,
    *,
    provider: str,
    method: Method,
    model: str,
    context_sha256: str,
    context_tokens: int,
    question_key: str | None = None,
) -> dict[str, Any]:
    """The identity half of a record (invariant 8) shared by every provider: task and framing
    keys, the state and its hashes, the D5 fingerprints. ``question_key`` overrides the
    framing's (Claude asks through its own prompt, so its decisions get their own key)."""
    task = framing.task
    st = dict(state)
    return {
        "task_id": framing.task_id,
        "task_key": task.key,
        "question_key": question_key or framing.question_key,
        "framing_id": framing.id,
        "shape": framing.shape,
        "kind": task.kind,
        "state": st,
        "state_sha256": _state_sha256(st),
        "target_sha256": _target_sha256(framing.task_id, st),
        "context_sha256": context_sha256,
        "context_tokens": int(context_tokens),
        "provider": provider,
        "method": method,
        "model": model,
        "encoder_fp": fingerprint.encoder_fp,
        "clm_model_fp": fingerprint.clm_model_fp,
        "schema_sha256": fingerprint.schema_sha256,
    }


# -- calls and the protocol -------------------------------------------------------------------


@dataclass(frozen=True)
class ClmCall:
    """One request to clm-serve as the sidecar's ``clm_calls`` table records it (the pipeline
    adds ``run_id``): endpoint, served model, questions and candidate texts carried,
    ``billing_units`` and ``input_tokens`` as the server reported them (``input_tokens`` counts
    encoder tokens on cache misses only), ``latency_ms`` (the server's ``X-CLM-Latency-Ms``
    when sent, else the client's wall time), the wall time, and how it ended."""

    endpoint: str
    model: str
    n_questions: int
    n_candidates: int
    latency_ms: float
    wall_ms: float
    status: CallStatus = "ok"
    billing_units: int | None = None
    input_tokens: int | None = None
    error: str | None = None


@runtime_checkable
class DecisionProvider(Protocol):
    """What answers a framing's question over a batch of states.

    ``decide(framing, states, candidates_per_state, tier=)`` returns one record per state, in
    order. ``candidates_per_state`` is required for a rank_fit (one candidate group per state)
    and must be ``None`` for a closed choice. ``tier`` is ``None``/``"auto"`` (the best tier
    this provider can serve honestly for the framing's question key, :meth:`resolve_tier`) or
    an explicit tier; a tier the provider cannot serve raises :class:`TierUnavailable`.
    ``fingerprint`` is stamped on every record (D5).
    """

    name: str
    model: str
    served_model: str | None
    fingerprint: Fingerprint

    def decide(
        self,
        framing: Framing,
        states: Sequence[Mapping[str, Any]],
        candidates_per_state: Sequence[Sequence[CandidateLike]] | None = None,
        *,
        tier: str | None = None,
    ) -> list[DecisionRecord]: ...

    def supports_tier(self, task_id: str, tier: str) -> bool: ...

    def resolve_tier(self, question_key: str) -> Level: ...


def check_batch(
    framing: Framing,
    states: Sequence[Mapping[str, Any]],
    candidates_per_state: Sequence[Sequence[CandidateLike]] | None,
) -> list[Sequence[CandidateLike] | None]:
    """The per-state candidate groups of a ``decide`` call, checked against the framing's
    shape: a rank_fit needs one group per state, a closed choice none."""
    if framing.shape == "rank_fit":
        if candidates_per_state is None or len(candidates_per_state) != len(states):
            raise ValueError(f"{framing.task_id}: a rank_fit needs one candidate group per state")
        return list(candidates_per_state)
    if candidates_per_state is not None:
        raise ValueError(f"{framing.task_id}: a closed choice takes no candidates")
    return [None] * len(states)
