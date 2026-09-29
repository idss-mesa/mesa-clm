"""CLM tiers behind one provider (DESIGN D2, D5, D6, D7, D23, D28; plan §4.1, §4.3, §5.3).

:class:`TieredProvider` asks clm-serve rank-first questions and turns each answer into an
honest :class:`~mesa_clm.providers.base.DecisionRecord`:

* **Requests.** Every question comes from ``framings.build_question`` and every context from
  ``framings.build_context``, so the wire bodies are exactly what the framing registry and
  ``framings.lock.json`` describe. Questions whose contexts are identical go in *one*
  ``/v1/systemone`` request (plan §4.1): CLM embeds the context once, and a batch of candidate
  groups for the same target costs one call. Requests always use the server's temperature 1;
  calibration is client-side.
* **Token guard (D23, plan §4.3).** Each context (plus the framing's instructions) is counted
  through the encoder's ``/tokenize`` when an :class:`~mesa_clm.clm.encoder.EncoderClient` is
  given, else ``ceil(chars / 2)`` (an over-count on the NEON cards). A context over
  ``max_len - 16`` is never sent: its record is ``method='unavailable'``, ``truncated=True``.
* **s_c recovery (D2).** For a rank_fit, ``s_c = ln p_c - ln p_anchor`` from the served
  distribution, ``0`` for the anchor. The softmax normaliser cancels, so ``s_c`` does not move
  when other candidates join or leave the group (``tests/unit/test_delta_recovery.py``). A served
  probability that is not ``> 0`` is ``numeric_underflow``: the record abstains.
* **Tiers.** ``zero_shot``: ``p_fit = σ(s_c)``, ``probs`` = CLM's distribution, calibration
  ``uncalibrated``. ``calibrated``: a :class:`PlattCalibrator` ``(a, b)`` or
  :class:`TemperatureCalibrator` ``T`` from an :class:`ArtifactBundle` (``calibrators.json``)
  fitted under the same question key and fingerprints; for a rank_fit ``p_fit = σ(a·s_c + b)``
  and ``probs`` is the softmax of those logits against the anchor's 0, so each candidate's
  pairwise odds against the anchor are exactly its ``p_fit``. ``probe`` (M4) and ``head`` (M7)
  raise :class:`~mesa_clm.providers.base.TierUnavailable` until they land.
* **Mask (plan §4.2 Q3).** A framing with ``mask_rule='aspect'`` (``column.ontology_fits``) is
  masked after scoring by the state's aspect through the pure
  :func:`~mesa_clm.providers.base.apply_mask`.
* **Calls.** Every request is reported to ``on_call`` as a
  :class:`~mesa_clm.providers.base.ClmCall` (latency, ``input_tokens``, ``billing_units``,
  status) for the sidecar's ``clm_calls``; a failed request yields ``method='unavailable'``
  records (``reason='decider_unavailable'``) instead of an exception, so the pipeline can fall
  back to ``ols_rank`` per group (D28).

:class:`FakeProvider` is the same provider over :class:`~mesa_clm.clm.fake.FakeClm` with
``method='fake'``, in process or through the fake transport. :class:`OlsRankProvider` is the
degraded D28 method: the OLS top-1 of a group as a record with ``rank`` set, no numbers, level
``none``; the policy only ever proposes it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from mesa_clm import render
from mesa_clm.clm.encoder import GUARD_MARGIN, chars_estimate
from mesa_clm.clm.fake import LATEST_MODEL, FakeClm, FakeClmError
from mesa_clm.clm.fingerprint import (
    ClmModelSpec,
    EncoderSpec,
    Fingerprint,
    FingerprintMismatch,
    canonical_json,
    fingerprint,
    sha256_hex,
)
from mesa_clm.clm.http import (
    ChoiceAnswer,
    ClmError,
    Question,
    SystemOneResponse,
    question_to_dict,
)
from mesa_clm.framings import (
    FRAMINGS,
    CandidateLike,
    Framing,
    FramingError,
    active_framing,
    build_context,
    context_text,
)
from mesa_clm.net import EndpointError
from mesa_clm.providers.base import (
    ArtifactRef,
    CallStatus,
    ClmCall,
    DecisionRecord,
    FramingOptions,
    Scalar,
    TierUnavailable,
    apply_mask,
    aspect_keep,
    base_fields,
    check_batch,
    framing_options,
    rank_probs,
    sha256_text,
    sigmoid,
)
from mesa_clm.vocab import Level

logger = logging.getLogger(__name__)

__all__ = [
    "CLM_COMMIT",
    "ArtifactBundle",
    "ArtifactError",
    "Calibrator",
    "CalibratorError",
    "DecisionRequest",
    "FakeClmClient",
    "FakeProvider",
    "OlsRankProvider",
    "PlattCalibrator",
    "SystemOneClient",
    "TemperatureCalibrator",
    "TieredProvider",
    "TokenCounter",
    "fake_fingerprint",
    "load_calibrator",
    "ols_rank_record",
]

# The CLM commit mesa-clm pins (plan §6.2): the engine whose maths the fake reproduces.
CLM_COMMIT: Final[str] = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
SYSTEMONE: Final[str] = "/v1/systemone"
DEFAULT_MAX_LEN: Final[int] = 4096
# Contexts counted once per provider; cleared wholesale when full (a run touches far fewer).
_TOKEN_CACHE_MAX: Final[int] = 4096
_ERROR_CHARS: Final[int] = 160


# -- calibrators (the Calibrator interface M4 fits and serialises) -----------------------------


class CalibratorError(ValueError):
    """A calibrator applied where it has no meaning (Platt on a closed choice with K > 2)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("must be finite")
    return value


def _softmax(logits: Sequence[float]) -> list[float]:
    return render.softmax(list(logits))


class PlattCalibrator(_Frozen):
    """Weighted Platt scaling ``σ(a·x + b)`` (plan §5.3). On a rank_fit ``x = s_c``; on a closed
    choice with K = 2, ``x`` is the logit difference ``ln p_0 - ln p_1``. ``a < 0`` means the
    raw scores ran backwards on the fit data (#15's inverted calibration): kept, but flagged."""

    kind: Literal["platt"] = "platt"
    a: float
    b: float
    n: int | None = Field(default=None, ge=0)  # labels it was fitted on (informational)

    _fa = field_validator("a", "b")(_finite)

    @property
    def inverted(self) -> bool:
        return self.a < 0.0

    def rank_fit(self, s_c: Sequence[float], anchor_index: int) -> tuple[list[float], list[float]]:
        """``(p_fit, probs)``: ``p_fit_i = σ(a·s_i + b)`` for every option (the anchor gets
        ``σ(b)``, the fit of a candidate scoring exactly like it) and ``probs`` the softmax over
        ``a·s_c + b`` for candidates and ``0`` for the anchor."""
        logits = [0.0 if i == anchor_index else self.a * s + self.b for i, s in enumerate(s_c)]
        return [sigmoid(self.a * s + self.b) for s in s_c], _softmax(logits)

    def choice(self, raw: Sequence[float]) -> list[float]:
        if len(raw) != 2:
            raise CalibratorError(
                f"Platt on a closed choice needs K=2 (got K={len(raw)}); a K>2 choice is "
                "calibrated by temperature (plan §5.3)"
            )
        p0 = sigmoid(self.a * (math.log(raw[0]) - math.log(raw[1])) + self.b)
        return [p0, 1.0 - p0]


class TemperatureCalibrator(_Frozen):
    """Temperature scaling (plan §5.3, K > 2 choices): ``softmax(ln p / T)``. On a rank_fit it is
    Platt with ``a = 1/T, b = 0``."""

    kind: Literal["temperature"] = "temperature"
    T: float = Field(gt=0.0)
    n: int | None = Field(default=None, ge=0)

    _ft = field_validator("T")(_finite)

    @property
    def inverted(self) -> bool:
        return False

    def rank_fit(self, s_c: Sequence[float], anchor_index: int) -> tuple[list[float], list[float]]:
        scaled = [s / self.T for s in s_c]
        return [sigmoid(x) for x in scaled], _softmax(scaled)

    def choice(self, raw: Sequence[float]) -> list[float]:
        return _softmax([math.log(p) / self.T for p in raw])


Calibrator = Annotated[PlattCalibrator | TemperatureCalibrator, Field(discriminator="kind")]
_CALIBRATOR: Final[TypeAdapter[PlattCalibrator | TemperatureCalibrator]] = TypeAdapter(Calibrator)


def load_calibrator(data: Mapping[str, Any]) -> PlattCalibrator | TemperatureCalibrator:
    """A calibrator from its JSON object (``{"kind": "platt", "a": …, "b": …}`` or
    ``{"kind": "temperature", "T": …}``)."""
    return _CALIBRATOR.validate_python(dict(data))


class ArtifactError(ValueError):
    """``calibrators.json`` is missing, not JSON, or not a valid bundle."""


def _hex_width(value: str, width: int, what: str) -> str:
    if len(value) != width or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{what} must be {width} lower-case hex characters")
    return value


class ArtifactBundle(_Frozen):
    """The calibrated tier's artifact (``calibrators.json`` of an artifacts version, plan §5.5):
    one calibrator per question key, fitted under one ``(encoder_fp, clm_model_fp)``.

    The provider refuses a bundle whose fingerprints differ from the live ones (K4) and applies
    a calibrator only to decisions of its exact ``question_key`` (a rotated framing has no
    artifact until it is re-benched). Probes and heads join this bundle in M4 and M7.
    """

    version: str = Field(min_length=1)
    encoder_fp: str
    clm_model_fp: str
    framings_lock_sha: str | None = None
    calibrators: dict[str, Calibrator] = Field(default_factory=dict)

    @field_validator("encoder_fp", "clm_model_fp")
    @classmethod
    def _fp(cls, v: str) -> str:
        return _hex_width(v, 12, "fingerprint")

    @field_validator("calibrators")
    @classmethod
    def _keys(
        cls, v: dict[str, PlattCalibrator | TemperatureCalibrator]
    ) -> dict[str, PlattCalibrator | TemperatureCalibrator]:
        for key in v:
            _hex_width(key, 16, f"calibrator key {key!r} (a question_key)")
        return v

    @classmethod
    def load(cls, path: str | Path) -> ArtifactBundle:
        """Read and validate a ``calibrators.json``; :class:`ArtifactError` names the file and
        the problem."""
        p = Path(path).expanduser()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ArtifactError(f"{p}: artifact bundle not found") from None
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{p}: not valid JSON ({exc.msg} at line {exc.lineno})") from None
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ArtifactError(f"{p}: {exc.error_count()} validation error(s): {exc}") from None

    def calibrator_for(self, question_key: str) -> PlattCalibrator | TemperatureCalibrator | None:
        return self.calibrators.get(question_key)

    def ref(self, question_key: str) -> ArtifactRef:
        """The record-side reference to this bundle for one question key (invariant 5)."""
        return ArtifactRef(
            version=self.version,
            question_key=question_key,
            encoder_fp=self.encoder_fp,
            clm_model_fp=self.clm_model_fp,
        )

    def check(self, live: Fingerprint) -> None:
        """Raise :class:`~mesa_clm.clm.fingerprint.FingerprintMismatch` unless the bundle was
        fitted under the live encoder and head (K4)."""
        diffs = [
            f"{name}: bundle {mine!r}, live {theirs!r}"
            for name, mine, theirs in (
                ("encoder_fp", self.encoder_fp, live.encoder_fp),
                ("clm_model_fp", self.clm_model_fp, live.clm_model_fp),
            )
            if mine != theirs
        ]
        if diffs:
            raise FingerprintMismatch(
                f"artifacts {self.version} were fitted under another serving stack "
                f"(K4: re-bench before use): {'; '.join(diffs)}"
            )


# -- clients ----------------------------------------------------------------------------------


class SystemOneClient(Protocol):
    """What the provider needs from clm-serve: :class:`~mesa_clm.clm.http.ClmHttpClient` or the
    in-process :class:`FakeClmClient`."""

    model: str

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse: ...


class TokenCounter(Protocol):
    """The token guard's counter: :meth:`mesa_clm.clm.encoder.EncoderClient.count_tokens`."""

    def count_tokens(self, text: str) -> tuple[int, str]: ...


class FakeClmClient:
    """:class:`~mesa_clm.clm.fake.FakeClm` behind the ``system_one`` call, in process: the
    same question dicts, the same response model, a refused request as ``ClmError(422)`` (what
    the fake transport answers). ``latency_ms`` stands in for the server header."""

    def __init__(
        self,
        clm: FakeClm | None = None,
        *,
        model: str = LATEST_MODEL,
        latency_ms: float | None = None,
    ) -> None:
        self.clm = clm or FakeClm()
        self.model = model
        self.latency_ms = latency_ms
        self.calls = 0

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        body = {k: question_to_dict(q) for k, q in questions.items()}
        try:
            out = self.clm.answer(
                state,
                body,
                model=model or self.model,
                temperature=1.0 if temperature is None else temperature,
            )
        except FakeClmError as exc:
            raise ClmError(422, str(exc)) from None
        self.calls += 1
        return SystemOneResponse.model_validate({**out, "latency_ms": self.latency_ms})


def fake_fingerprint(
    model: str = LATEST_MODEL, *, seed: int = 0, max_len: int = DEFAULT_MAX_LEN
) -> Fingerprint:
    """The D5 bundle of the fake stack: an encoder spec with ``route='fake'`` (so a fake
    artifact can never pass for a real one), the named head without a file sha, the pinned CLM
    commit, and a ``serving_lock_sha`` hashed from those."""
    enc = EncoderSpec(
        model="fake-ngram",
        revision=f"seed-{seed}",
        dtype="float32",
        max_len=max_len,
        route="fake",
    )
    head = ClmModelSpec(head_name=model, head_sha256="", clm_commit=CLM_COMMIT)
    lock = sha256_hex(canonical_json({"fake": True, "encoder": enc.as_dict(), **head.as_dict()}))
    return fingerprint(enc, head, serving_lock_sha=lock)


# -- the tiered provider ----------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionRequest:
    """One question to decide: a framing, the state to ask it over and, for a rank_fit, the
    candidate group. :meth:`TieredProvider.decide_many` packs requests with identical contexts
    into one call."""

    framing: Framing
    state: Mapping[str, Any]
    candidates: Sequence[CandidateLike] | None = None


@dataclass
class _Item:
    """A request on its way through ``decide_many``."""

    index: int
    request: DecisionRequest
    level: Level
    calibrator: PlattCalibrator | TemperatureCalibrator | None
    opts: FramingOptions
    context: Any
    context_key: str
    context_sha256: str
    tokens: int
    token_source: str
    truncated: bool
    qid: str = ""


class TieredProvider:
    """CLM tiers over one clm-serve client (module docstring).

    ``client`` is a :class:`~mesa_clm.clm.http.ClmHttpClient` (or anything with its
    ``system_one``); ``encoder`` counts context tokens for the guard (``None``: chars/2);
    ``fingerprint`` is the live D5 bundle stamped on every record; ``artifacts`` the calibrated
    tier's bundle, refused unless fitted under that fingerprint. ``method`` is ``clm`` for the
    real server, ``fake`` for :class:`~mesa_clm.clm.fake.FakeClm` (never cited). ``model`` is
    the served head (default: the client's). ``on_call`` receives a
    :class:`~mesa_clm.providers.base.ClmCall` per request.
    """

    def __init__(
        self,
        client: SystemOneClient,
        encoder: TokenCounter | None,
        fingerprint: Fingerprint,
        artifacts: ArtifactBundle | None = None,
        *,
        method: Literal["clm", "fake"] = "clm",
        name: str | None = None,
        model: str | None = None,
        max_len: int | None = None,
        on_call: Callable[[ClmCall], None] | None = None,
    ) -> None:
        if artifacts is not None:
            artifacts.check(fingerprint)  # K4: refuse, never silently mix stacks
        self.client = client
        self.encoder = encoder
        self.fingerprint = fingerprint
        self.artifacts = artifacts
        self.method: Literal["clm", "fake"] = method
        self.name: str = name or method
        self.model: str = model or client.model
        self.served_model: str | None = None
        limit = max_len if max_len is not None else getattr(encoder, "max_len", DEFAULT_MAX_LEN)
        self.max_len = int(limit)
        if self.max_len <= GUARD_MARGIN:
            raise ValueError(f"max_len must exceed the guard margin ({GUARD_MARGIN})")
        self.on_call = on_call
        self.n_calls = 0
        self.input_tokens = 0
        self._tokens: dict[str, tuple[int, str]] = {}

    # -- tiers ----------------------------------------------------------------------------------

    def _calibrator(self, question_key: str) -> PlattCalibrator | TemperatureCalibrator | None:
        return None if self.artifacts is None else self.artifacts.calibrator_for(question_key)

    def resolve_tier(self, question_key: str) -> Level:
        """The best tier servable for ``question_key``: ``calibrated`` when the bundle has its
        calibrator, else ``zero_shot`` (probe and head land in M4/M7)."""
        return "calibrated" if self._calibrator(question_key) is not None else "zero_shot"

    def supports_tier(self, task_id: str, tier: str) -> bool:
        """Whether ``tier`` can be served for ``task_id``'s active framing."""
        if task_id not in FRAMINGS:
            return False
        if tier in ("auto", "zero_shot"):
            return True
        if tier == "calibrated":
            return self._calibrator(active_framing(task_id).question_key) is not None
        return False

    def _tier(
        self, framing: Framing, tier: str | None
    ) -> tuple[Level, PlattCalibrator | TemperatureCalibrator | None]:
        qk = framing.question_key
        requested = tier or "auto"
        if requested == "auto":
            level = self.resolve_tier(qk)
            return level, self._calibrator(qk) if level == "calibrated" else None
        if requested == "zero_shot":
            return "zero_shot", None
        if requested == "calibrated":
            cal = self._calibrator(qk)
            if cal is None:
                where = f"artifacts {self.artifacts.version}" if self.artifacts else "no artifacts"
                raise TierUnavailable(
                    f"{framing.task_id}/{framing.id}: no calibrator for question_key {qk} "
                    f"({where}); fit and promote one (`mesa-clm learn fit`, M4) or ask zero_shot"
                )
            return "calibrated", cal
        if requested == "probe":
            raise TierUnavailable(
                "the probe tier (a local numpy probe over encoder features, plan §5.3) lands in "
                "M4; until then ask zero_shot or calibrated"
            )
        if requested == "head":
            raise TierUnavailable(
                "the head tier (a fine-tuned head promoted to clm-serve, plan §5.3) lands in "
                "M7; until then ask zero_shot or calibrated"
            )
        if requested == "none":
            raise TierUnavailable(
                "'none' is not a CLM tier: use OlsRankProvider (D28) or a rule for level none"
            )
        raise ValueError(f"unknown tier {requested!r}")

    # -- deciding -------------------------------------------------------------------------------

    def decide(
        self,
        framing: Framing,
        states: Sequence[Mapping[str, Any]],
        candidates_per_state: Sequence[Sequence[CandidateLike]] | None = None,
        *,
        tier: str | None = None,
    ) -> list[DecisionRecord]:
        """One record per state (see :class:`~mesa_clm.providers.base.DecisionProvider`)."""
        groups = check_batch(framing, states, candidates_per_state)
        return self.decide_many(
            [DecisionRequest(framing, s, g) for s, g in zip(states, groups, strict=True)],
            tier=tier,
        )

    def decide_many(
        self, requests: Sequence[DecisionRequest], *, tier: str | None = None
    ) -> list[DecisionRecord]:
        """One record per request, in order, with every set of requests whose contexts are
        identical asked in one ``/v1/systemone`` call (plan §4.1). Requests may mix framings
        and tasks (``column.annotate`` and ``column.aspect`` share ``column_state``)."""
        tiers: dict[tuple[str, str], tuple[Level, Any]] = {}
        items: list[_Item] = []
        for i, req in enumerate(requests):
            fkey = (req.framing.task_id, req.framing.id)
            if fkey not in tiers:
                tiers[fkey] = self._tier(req.framing, tier)
            items.append(self._prepare(i, req, *tiers[fkey]))
        out: list[DecisionRecord | None] = [None] * len(items)
        groups: dict[str, list[_Item]] = {}
        for it in items:
            if it.truncated:
                out[it.index] = self._unavailable(it, reason="truncated")
            else:
                groups.setdefault(it.context_key, []).append(it)
        for group in groups.values():
            for rec_index, rec in self._ask(group):
                out[rec_index] = rec
        records = [r for r in out if r is not None]
        if len(records) != len(out):  # pragma: no cover - every item is sent or abstains above
            raise RuntimeError("a decision request was neither asked nor recorded")
        return records

    def _count(self, text: str) -> tuple[int, str]:
        hit = self._tokens.get(text)
        if hit is not None:
            return hit
        counted: tuple[int, str]
        if self.encoder is None:
            counted = (chars_estimate(text), "chars")
        else:
            try:
                n, source = self.encoder.count_tokens(text)
                counted = (int(n), str(source))
            except (ClmError, EndpointError) as exc:
                # The guard must not fail the run: the conservative estimate over-counts.
                logger.info("token count fell back to chars (%s)", type(exc).__name__)
                counted = (chars_estimate(text), "chars")
        if len(self._tokens) >= _TOKEN_CACHE_MAX:
            self._tokens.clear()
        self._tokens[text] = counted
        return counted

    def _prepare(
        self,
        index: int,
        req: DecisionRequest,
        level: Level,
        calibrator: PlattCalibrator | TemperatureCalibrator | None,
    ) -> _Item:
        opts = framing_options(req.framing, req.candidates)
        if (
            isinstance(calibrator, PlattCalibrator)
            and opts.anchor_index is None
            and len(opts.options) != 2
        ):
            # Refused before any request is spent, with the calibrator's own message.
            raise CalibratorError(
                f"{req.framing.task_id}: Platt on a closed choice needs K=2 "
                f"(got K={len(opts.options)}); a K>2 choice is calibrated by temperature"
            )
        context = build_context(req.framing, req.state)
        text = context_text(req.framing, req.state)
        tokens, source = self._count(text)
        return _Item(
            index=index,
            request=req,
            level=level,
            calibrator=calibrator,
            opts=opts,
            context=context,
            context_key=canonical_json(context),
            context_sha256=sha256_text(text),
            tokens=tokens,
            token_source=source,
            truncated=tokens > self.max_len - GUARD_MARGIN,
        )

    def _fields(self, it: _Item) -> dict[str, Any]:
        f = base_fields(
            it.request.framing,
            it.request.state,
            self.fingerprint,
            provider=self.name,
            method=self.method,
            model=self.model,
            context_sha256=it.context_sha256,
            context_tokens=it.tokens,
        )
        f.update(
            options=it.opts.options,
            option_texts=it.opts.option_texts,
            anchor_index=it.opts.anchor_index,
            truncated=it.truncated,
        )
        return f

    def _unavailable(self, it: _Item, *, reason: str, error: str | None = None) -> DecisionRecord:
        """A record without a CLM answer: never sent (truncated) or not answered."""
        diag: dict[str, Scalar] = {"reason": reason, "token_source": it.token_source}
        if error is not None:
            diag["error"] = error[:_ERROR_CHARS]
        f = self._fields(it)
        f.update(
            method="unavailable",
            level="none",
            calibration="none",
            answer_index=-1,
            answer="",
            diagnostics=diag,
        )
        return DecisionRecord.model_validate(f)

    def _ask(self, group: list[_Item]) -> list[tuple[int, DecisionRecord]]:
        """One request for a group of items sharing a context."""
        questions: dict[str, Question] = {}
        seen: dict[str, int] = {}
        for it in group:
            base = it.request.framing.task_id
            n = seen.get(base, 0)
            seen[base] = n + 1
            it.qid = base if n == 0 else f"{base}#{n}"
            questions[it.qid] = it.opts.question
        n_candidates = sum(len(it.opts.wire_keys) for it in group)
        started = time.monotonic()
        try:
            response = self.client.system_one(group[0].context, questions, model=self.model)
        except (ClmError, EndpointError) as exc:
            wall = (time.monotonic() - started) * 1000.0
            status = _call_status(exc)
            message = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "clm-serve request for %d question(s) failed (%s)", len(questions), status
            )
            self._report(
                ClmCall(
                    endpoint=SYSTEMONE,
                    model=self.model,
                    n_questions=len(questions),
                    n_candidates=n_candidates,
                    latency_ms=wall,
                    wall_ms=wall,
                    status=status,
                    error=message[:_ERROR_CHARS],
                )
            )
            return [
                (it.index, self._unavailable(it, reason="decider_unavailable", error=message))
                for it in group
            ]
        wall = (time.monotonic() - started) * 1000.0
        latency = response.latency_ms if response.latency_ms is not None else wall
        self.n_calls += 1
        self.input_tokens += int(response.usage.input_tokens or 0)
        self.served_model = response.model
        self._report(
            ClmCall(
                endpoint=SYSTEMONE,
                model=self.model,
                n_questions=len(questions),
                n_candidates=n_candidates,
                latency_ms=latency,
                wall_ms=wall,
                billing_units=response.usage.billing_units,
                input_tokens=response.usage.input_tokens,
            )
        )
        return [(it.index, self._record(it, response, latency, len(questions))) for it in group]

    def _report(self, call: ClmCall) -> None:
        if self.on_call is not None:
            self.on_call(call)

    def _record(
        self, it: _Item, response: SystemOneResponse, latency: float, n_questions: int
    ) -> DecisionRecord:
        answer = response.answers.get(it.qid)
        keys = it.opts.wire_keys
        if not isinstance(answer, ChoiceAnswer) or set(answer.probabilities) != set(keys):
            return self._unavailable(
                it, reason="malformed_answer", error=f"answer to {it.qid} does not cover its keys"
            )
        raw = [float(answer.probabilities[k]) for k in keys]
        if any(not math.isfinite(p) or p <= 0.0 for p in raw):
            # Plan §4.1: no underflow re-query; the record abstains.
            return self._unavailable(it, reason="numeric_underflow")
        if abs(math.fsum(raw) - 1.0) > 1e-3:
            return self._unavailable(
                it, reason="malformed_answer", error="probabilities do not sum to 1"
            )
        framing = it.request.framing
        s_c: list[float] | None = None
        p_fit: list[float] | None = None
        a = it.opts.anchor_index
        if a is not None:
            log_a = math.log(raw[a])
            s_c = [0.0 if i == a else math.log(p) - log_a for i, p in enumerate(raw)]
        cal = it.calibrator
        if cal is None:
            probs = list(raw)
            if s_c is not None:
                p_fit = [sigmoid(s) for s in s_c]
        elif s_c is not None and a is not None:
            p_fit, probs = cal.rank_fit(s_c, a)
        else:
            probs = cal.choice(raw)
        idx, top, margin = rank_probs(probs)
        diag: dict[str, Scalar] = {
            "qid": it.qid,
            "n_questions": n_questions,
            "token_source": it.token_source,
        }
        if answer.choice != keys[rank_probs(raw)[0]]:
            diag["clm_choice"] = answer.choice  # never expected; recorded if CLM disagrees
        if cal is not None and cal.inverted:
            diag["inverted"] = True
        f = self._fields(it)
        f.update(
            served_model=response.model,
            level=it.level,
            calibration="uncalibrated" if cal is None else cal.kind,
            probs=probs,
            raw_probs=raw,
            s_c=s_c,
            p_fit=p_fit,
            answer_index=idx,
            answer=it.opts.options[idx],
            confidence=top,
            clm_confidence=answer.confidence,
            margin=margin,
            latency_ms=latency,
            diagnostics=diag,
        )
        if cal is not None and self.artifacts is not None:
            f.update(
                artifact_version=self.artifacts.version,
                artifact=self.artifacts.ref(framing.question_key),
            )
        rec = DecisionRecord.model_validate(f)
        if framing.mask_rule == "aspect":
            rec = apply_mask(rec, aspect_keep(rec.options, str(it.request.state["aspect"])))
        return rec


def _call_status(exc: Exception) -> CallStatus:
    """How a failed request ended: no response at all is ``unavailable`` (``timeout`` when the
    transport timed out), an open breaker or refused URL ``unavailable``, an HTTP error
    ``error``."""
    if isinstance(exc, ClmError):
        if exc.status == 0:
            return "timeout" if "Timeout" in exc.message else "unavailable"
        return "error"
    return "unavailable"


class FakeProvider(TieredProvider):
    """:class:`TieredProvider` over :class:`~mesa_clm.clm.fake.FakeClm`, ``method='fake'``.

    By default the fake engine runs in process (:class:`FakeClmClient`); pass ``client`` (a
    :class:`~mesa_clm.clm.http.ClmHttpClient` over ``tests/fakes/clm_transport.py``) to go
    through the wire instead. The fingerprint defaults to :func:`fake_fingerprint`. What it
    decides exercises the pipeline, the policy and the sidecar; it is never evidence.
    """

    def __init__(
        self,
        clm: FakeClm | None = None,
        *,
        client: SystemOneClient | None = None,
        encoder: TokenCounter | None = None,
        fingerprint: Fingerprint | None = None,
        artifacts: ArtifactBundle | None = None,
        model: str = LATEST_MODEL,
        seed: int = 0,
        max_len: int | None = None,
        on_call: Callable[[ClmCall], None] | None = None,
    ) -> None:
        if client is None:
            client = FakeClmClient(clm or FakeClm(seed=seed), model=model)
        super().__init__(
            client,
            encoder,
            fingerprint or fake_fingerprint(model, seed=seed),
            artifacts,
            method="fake",
            name="fake",
            model=model,
            max_len=max_len,
            on_call=on_call,
        )


# -- the degraded method (D28) ----------------------------------------------------------------


def _ols_rank_of(candidate: CandidateLike, position: int) -> int:
    rank = getattr(candidate, "rank", None)
    return rank if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1 else position


def ols_rank_record(
    framing: Framing,
    state: Mapping[str, Any],
    candidates: Sequence[CandidateLike],
    fingerprint: Fingerprint,
    *,
    pick: int = 0,
    provider: str = "ols_rank",
) -> DecisionRecord:
    """The ``ols_rank`` record for candidate ``pick`` of an OLS-ordered group (D28): options are
    the group's keys plus the anchor, the answer is that candidate, ``rank`` its OLS rank (the
    candidate's own ``rank`` when it carries a positive one, else ``pick + 1``), level and
    calibration ``none``, no numbers. The context is hashed but never sent
    (``context_tokens = 0``). The policy proposes it only while ``rank <= ols_rank_top``."""
    if framing.shape != "rank_fit":
        raise FramingError(f"{framing.task_id}: ols_rank ranks an OLS candidate group (rank_fit)")
    if not candidates:
        raise FramingError(f"{framing.task_id}: ols_rank needs at least one candidate")
    if not 0 <= pick < len(candidates):
        raise ValueError(f"pick {pick} out of range for {len(candidates)} candidates")
    opts = framing_options(framing, candidates)
    f = base_fields(
        framing,
        state,
        fingerprint,
        provider=provider,
        method="ols_rank",
        model="ols",
        context_sha256=sha256_text(context_text(framing, state)),
        context_tokens=0,
    )
    f.update(
        options=opts.options,
        option_texts=opts.option_texts,
        anchor_index=opts.anchor_index,
        level="none",
        calibration="none",
        answer_index=pick,
        answer=opts.options[pick],
        rank=_ols_rank_of(candidates[pick], pick + 1),
        diagnostics={"context_sent": False},
    )
    return DecisionRecord.model_validate(f)


class OlsRankProvider:
    """The degraded provider (D28): each group's OLS top-1 as an ``ols_rank`` record. It serves
    only level ``none`` and only rank_fit tasks; the pipeline uses it when a task has no
    trustworthy tier (K1, K2(c)) or clm-serve is down, and :func:`ols_rank_record` for deeper
    ranks (rank-and-cap)."""

    name: str = "ols_rank"
    model: str = "ols"
    served_model: str | None = None

    def __init__(self, fingerprint: Fingerprint) -> None:
        self.fingerprint = fingerprint

    def resolve_tier(self, question_key: str) -> Level:
        return "none"

    def supports_tier(self, task_id: str, tier: str) -> bool:
        framings = FRAMINGS.get(task_id)
        if not framings or tier not in ("auto", "none"):
            return False
        return next(iter(framings.values())).shape == "rank_fit"

    def decide(
        self,
        framing: Framing,
        states: Sequence[Mapping[str, Any]],
        candidates_per_state: Sequence[Sequence[CandidateLike]] | None = None,
        *,
        tier: str | None = None,
    ) -> list[DecisionRecord]:
        if tier not in (None, "auto", "none"):
            raise TierUnavailable(f"ols_rank serves level 'none' only, not {tier!r} (D28)")
        groups = check_batch(framing, states, candidates_per_state)
        return [
            ols_rank_record(framing, s, g or (), self.fingerprint)
            for s, g in zip(states, groups, strict=True)
        ]
