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
  pairwise odds against the anchor are exactly its ``p_fit``. ``probe`` (M4, plan §4.1, §5.3,
  §5.5; ``design/m4-analysis-plan.md`` C1): a promoted :class:`ServedProbe` of the bundle, scored
  locally and never through clm-serve (below). ``head`` (M7) raises
  :class:`~mesa_clm.providers.base.TierUnavailable` until it lands.
* **Promotion.** A bundle built from ``CURRENT.json`` (:mod:`mesa_clm.artifacts`) carries a
  promotion table (``promoted``: task -> tier, question key, version); then only promoted
  entries are served, at ``auto`` or by name (an unpromoted calibrator or probe of the same
  version is ``TierUnavailable``: serving requires ``learn promote``, plan §5.5). A bundle
  without one (``calibrators.json`` loaded directly, the M1 path) serves its calibrators as
  before. :meth:`TieredProvider.resolve_tier` is ``probe`` for a promoted probe, else
  ``calibrated`` for a servable calibrator, else ``zero_shot``.
* **The probe path.** The state text (``framings.context_text``) and the option texts (the
  wire criteria, the anchor last) go to the encoder (``EncoderClient.embed``) through a
  :class:`VectorCache` (an in-memory LRU, and the live feature store read-only when one is
  configured: the provider never writes it); the head export projects them
  (:class:`~mesa_clm.clm.headproj.HeadProjector`, as ``health._head`` loads it); the probe's
  feature specification (:func:`spec_features`, the R1 definitions, mirrored exactly from the
  bench's ``learn.probe.FeatureBuilder``) builds one row per candidate (plus the anchor's row,
  the candidate side being the anchor's own vector: the probe analogue of Platt's ``σ(b)``, a
  candidate scoring exactly like the anchor) or one row for a closed choice; the probe's
  ``predict`` gives the calibrated probabilities. The record: ``method='clm'`` (``fake`` under
  the fake stack), ``level='probe'``, ``calibration`` the probe calibrator's kind,
  ``raw_probs`` the local zero-shot softmax over the same vectors under the served model (the
  M1 parity with clm-serve, <= 1e-4), ``s_c`` its log-ratios (anchor 0), ``p_fit`` the probe's
  calibrated per-candidate probability, ``probs`` the softmax over ``logit(p_fit)`` for the
  candidates and 0 for the anchor (the mirror of :meth:`PlattCalibrator.rank_fit`),
  ``confidence = max(probs)``, ``artifact_version``/``artifact`` the probe's and
  ``feature_spec`` its spec. An encoder that does not answer gives ``unavailable`` records
  (``decider_unavailable``) exactly like clm-serve; a rejected key refuses the run.
* **Mask (plan §4.2 Q3).** A framing with ``mask_rule='aspect'`` (``column.ontology_fits``) is
  masked after scoring by the state's aspect through the pure
  :func:`~mesa_clm.providers.base.apply_mask`.
* **Calls.** Every request is reported to ``on_call`` as a
  :class:`~mesa_clm.providers.base.ClmCall` (latency, ``input_tokens``, ``billing_units``,
  status) for the sidecar's ``clm_calls``; a failed request (no response, a timeout, a 5xx, a
  422) yields ``method='unavailable'`` records (``reason='decider_unavailable'``) instead of an
  exception, so the pipeline can fall back to ``ols_rank`` per group (D28). A 401, 403 or 404
  (:data:`REFUSED_STATUSES`: a rejected key or a wrong route, which would answer every group
  the same way) raises :class:`~mesa_clm.providers.base.DeciderRefused` instead, so a
  misconfigured run fails (and is recorded ``failed``) rather than quietly proposing
  ``ols_rank`` under the requested tier; so does a clm-serve port held by another account's
  socket (:class:`mesa_clm.net.ListenerOwnerError`, the key not sent; DESIGN A5).

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
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from mesa_clm import render
from mesa_clm.clm.encoder import GUARD_MARGIN, chars_estimate
from mesa_clm.clm.fake import LATEST_MODEL, RAW_MODEL, FakeClm, FakeClmError
from mesa_clm.clm.fingerprint import (
    ClmModelSpec,
    EncoderSpec,
    Fingerprint,
    FingerprintMismatch,
    canonical_json,
    fingerprint,
    sha256_hex,
)
from mesa_clm.clm.headproj import RAW_SCALE, HeadProjector
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
    control_state,
)
from mesa_clm.framings import (
    framing as framing_of,
)
from mesa_clm.net import EndpointError, ListenerOwnerError
from mesa_clm.providers.base import (
    ArtifactRef,
    CallStatus,
    ClmCall,
    DeciderRefused,
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
    "CHOICE_SPECS",
    "CLM_COMMIT",
    "EMBEDDINGS",
    "PROBE_LOGIT_CLIP",
    "RANK_FIT_SPECS",
    "REFUSED_STATUSES",
    "SPEC_MODELS",
    "SPEC_NEEDS_HEAD",
    "VECTOR_CACHE_MAX",
    "ArtifactBundle",
    "ArtifactError",
    "Calibrator",
    "CalibratorError",
    "DecisionRequest",
    "Embedder",
    "FakeClmClient",
    "FakeProvider",
    "OlsRankProvider",
    "PlattCalibrator",
    "Promotion",
    "ServedProbe",
    "SystemOneClient",
    "TemperatureCalibrator",
    "TieredProvider",
    "TokenCounter",
    "VectorCache",
    "VectorSource",
    "choice_features",
    "fake_fingerprint",
    "joint_texts",
    "load_calibrator",
    "ols_rank_record",
    "probe_from_json",
    "probe_rank_probs",
    "rank_fit_features",
    "spec_dim",
    "spec_features",
]

# The CLM commit mesa-clm pins (plan §6.2): the engine whose maths the fake reproduces.
CLM_COMMIT: Final[str] = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
SYSTEMONE: Final[str] = "/v1/systemone"
# The encoder route the probe path calls (``clm_calls.endpoint`` of a probe decision).
EMBEDDINGS: Final[str] = "/v1/embeddings"
# Statuses that refuse the run instead of one group (module docstring, "Calls").
REFUSED_STATUSES: Final[frozenset[int]] = frozenset({401, 403, 404})
DEFAULT_MAX_LEN: Final[int] = 4096
# Contexts counted once per provider; cleared wholesale when full (a run touches far fewer).
_TOKEN_CACHE_MAX: Final[int] = 4096
_ERROR_CHARS: Final[int] = 160
# The probe path's in-memory LRU of encoder vectors (texts): a card's run touches far fewer.
VECTOR_CACHE_MAX: Final[int] = 2048
# ``logit(p_fit)`` of a probe probability at exactly 0 or 1 is clipped here (the NLL clip of the
# vendored metrics), so the group softmax stays finite.
PROBE_LOGIT_CLIP: Final[float] = 1e-12

# The probe feature specifications of plan §5.3 as the brief (R1) reads them: which served
# model each one belongs to (``clm-latest`` when it reads any head quantity, else ``clm-raw``),
# which ones need the head export, and the shape they serve. ``learn.probe.SPECS`` (builder A)
# declares the same ids; ``tests/unit/test_m4_plan_constants.py`` holds the two together.
RANK_FIT_SPECS: Final[tuple[str, ...]] = (
    "lowdim.v1",
    "pair512.v1",
    "pair4096.v1",
    "joint4096@S1",
    "joint4096@S1ns",
)
CHOICE_SPECS: Final[tuple[str, ...]] = ("choice.state.v1", "choice.raw.v1")
SPEC_MODELS: Final[dict[str, str]] = {
    "lowdim.v1": LATEST_MODEL,
    "pair512.v1": LATEST_MODEL,
    "pair4096.v1": RAW_MODEL,
    "joint4096@S1": RAW_MODEL,
    "joint4096@S1ns": RAW_MODEL,
    "choice.state.v1": LATEST_MODEL,
    "choice.raw.v1": RAW_MODEL,
}
SPEC_NEEDS_HEAD: Final[frozenset[str]] = frozenset({"lowdim.v1", "pair512.v1", "choice.state.v1"})
# The X1 control framing whose per-candidate anyjev state the joint specs embed (features.py).
_JOINT_FRAMING: Final[str] = "F1"

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


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
    """A runtime calibrator from its JSON object: its own form (``{"kind": "platt", "a", "b"[,
    "n"]}`` / ``{"kind": "temperature", "T"[, "n"]}``) or the form :mod:`mesa_clm.learn.calibrate`
    fits and the bench records (``PlattCalibrator``/``TemperatureCalibrator`` with ``feature``,
    ``targets``, ``weight_sum``, ``ridge``, ``iterations``, ``converged``, ``nll``; the
    temperature under ``temperature``), of which only the applied parameters are kept. Any
    other ``kind`` or a parameter out of range is ``ValueError``."""
    body = dict(data)
    kind = body.get("kind")
    if kind == "temperature" and "T" not in body and "temperature" in body:
        body["T"] = body.pop("temperature")
    if kind in ("platt", "temperature"):
        keep = ("kind", "a", "b", "n") if kind == "platt" else ("kind", "T", "n")
        body = {k: body[k] for k in keep if k in body}
    return _CALIBRATOR.validate_python(body)


class ArtifactError(ValueError):
    """An artifact file is missing, not JSON, not valid, tampered with, or not the one the
    serving stack may use (``calibrators.json``, ``manifest.json``, a probe, ``CURRENT.json``)."""


def _hex_width(value: str, width: int, what: str) -> str:
    if len(value) != width or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{what} must be {width} lower-case hex characters")
    return value


# -- probes as the provider serves them (M4) ----------------------------------------------------


class ServedProbe:
    """A probe artifact (``probes/<question_key>.json``, :class:`mesa_clm.learn.probe
    .ProbeArtifact`) as the provider applies it: its identity (question key, task, framing,
    spec, model, fingerprint), the number of options it scores, the kind of its calibrator and
    its ``predict``; ``version`` is the artifacts version the file came from. Anything with the
    artifact's fields and a ``predict(X) -> [n, K]`` serves (the tests pass a stand-in)."""

    def __init__(self, artifact: Any, *, version: str) -> None:
        if not version:
            raise ArtifactError("a served probe needs its artifacts version")
        self.artifact = artifact
        self.version = str(version)
        self.question_key = _hex_width(str(artifact.question_key), 16, "probe question_key")
        self.task_id = str(artifact.task_id)
        self.framing_id = str(artifact.framing_id)
        self.spec = str(artifact.spec)
        self.model = str(artifact.model)
        fp = artifact.fingerprint
        self.fingerprint: dict[str, str] = {str(k): str(v) for k, v in dict(fp).items()}
        self.k = _probe_k(artifact)
        self.calibration = _calibration_kind(artifact.calibrator)
        if self.spec not in SPEC_MODELS:
            raise ArtifactError(f"probe {self.question_key}: unknown feature spec {self.spec!r}")
        if SPEC_MODELS[self.spec] != self.model:
            raise ArtifactError(
                f"probe {self.question_key}: spec {self.spec} belongs to {SPEC_MODELS[self.spec]}, "
                f"not {self.model} (R1)"
            )
        if self.calibration not in ("platt", "temperature"):
            raise ArtifactError(
                f"probe {self.question_key}: calibrator kind {self.calibration!r} is not platt or "
                "temperature (D6)"
            )
        predict = getattr(artifact, "predict", None)
        if not callable(predict):
            raise ArtifactError(f"probe {self.question_key}: the artifact has no predict")
        self._predict: Callable[[F64], Any] = predict

    @property
    def shape(self) -> Literal["rank_fit", "choice"]:
        return "rank_fit" if self.spec in RANK_FIT_SPECS else "choice"

    @property
    def needs_head(self) -> bool:
        return self.spec in SPEC_NEEDS_HEAD

    def predict(self, x: F64) -> F64:
        """``[n, K]`` calibrated probabilities (the artifact's own ``predict``), checked; a
        ``predict`` that refuses the features (a dimension mismatch: ``LinearError``,
        ``ProbeError``, both ``ValueError``) is :class:`ArtifactError`."""
        try:
            p = np.asarray(self._predict(np.asarray(x, dtype=np.float64)), dtype=np.float64)
        except ValueError as exc:
            raise ArtifactError(
                f"probe {self.question_key}: predict refused the features: {exc}"
            ) from None
        if p.ndim != 2 or p.shape[0] != x.shape[0] or p.shape[1] != self.k:
            raise ArtifactError(
                f"probe {self.question_key}: predict gave shape {list(p.shape)}, "
                f"expected [{x.shape[0]}, {self.k}]"
            )
        if not np.all(np.isfinite(p)) or np.any(p < 0.0):
            raise ArtifactError(f"probe {self.question_key}: predict gave a non-probability")
        return p

    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            version=self.version,
            question_key=self.question_key,
            encoder_fp=self.fingerprint.get("encoder_fp", ""),
            clm_model_fp=self.fingerprint.get("clm_model_fp", ""),
        )

    def __repr__(self) -> str:
        return f"ServedProbe({self.question_key}, {self.spec}, {self.model}, {self.version})"


def _probe_k(artifact: Any) -> int:
    k = getattr(artifact, "k", None)
    if k is None:
        linear = getattr(artifact, "linear", None)
        k = getattr(linear, "k", None)
    if not isinstance(k, int) or k < 2:
        raise ArtifactError("a probe artifact names the number of options it scores (k >= 2)")
    return k


def _calibration_kind(calibrator: Any) -> str:
    if isinstance(calibrator, Mapping):
        return str(calibrator.get("kind", ""))
    return str(getattr(calibrator, "kind", ""))


def probe_from_json(data: Mapping[str, Any], *, version: str) -> ServedProbe:
    """A :class:`ServedProbe` from a ``probes/<question_key>.json`` object, validated by
    :class:`mesa_clm.learn.probe.ProbeArtifact` (imported here: the learn core lands beside
    this module in M4); :class:`ArtifactError` when the model is unavailable or refuses it."""
    try:
        from mesa_clm.learn.probe import ProbeArtifact
    except ImportError as exc:  # pragma: no cover - the learn core is part of M4
        raise ArtifactError(
            f"the probe artifact model (learn/probe.py) is not available: {exc}"
        ) from None
    try:
        artifact = ProbeArtifact.model_validate(dict(data))
    except ValidationError as exc:
        raise ArtifactError(
            f"probe artifact: {exc.error_count()} validation error(s): {exc}"
        ) from None
    return ServedProbe(artifact, version=version)


# -- vectors for the probe path -----------------------------------------------------------------


class Embedder(Protocol):
    """``EncoderClient.embed`` (or the fake encoder's): ``(float32 [n, dim], tokens charged)``."""

    def __call__(self, texts: Sequence[str], /) -> tuple[F32, int]: ...


class VectorSource(Protocol):
    """What the probe path reads vectors from: a :class:`~mesa_clm.learn.features.FeatureStore`
    (``exists``, ``missing``, ``get``), read-only here."""

    def exists(self) -> bool: ...
    def missing(self, texts: Sequence[str]) -> list[str]: ...
    def get(self, texts: Sequence[str]) -> F32: ...


@dataclass(frozen=True)
class Vectors:
    """What :meth:`VectorCache.vectors` returns: the rows in input order, the encoder tokens
    charged and how many texts were embedded (0: every row came from the caches)."""

    rows: F32
    tokens: int
    embedded: int


class VectorCache:
    """Encoder vectors by text for the probe path: an in-memory LRU (``max_entries``), then
    the configured feature store when it has the text (read-only: the provider never writes a
    store), then ``embed`` for the rest in one request."""

    def __init__(
        self,
        embed: Embedder,
        *,
        store: VectorSource | None = None,
        max_entries: int = VECTOR_CACHE_MAX,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.embed = embed
        self.store = store
        self.max_entries = int(max_entries)
        self._lru: OrderedDict[str, F32] = OrderedDict()
        self.store_hits = 0
        self.embed_calls = 0

    def __len__(self) -> int:
        return len(self._lru)

    def _remember(self, text: str, row: F32) -> None:
        self._lru[text] = row
        self._lru.move_to_end(text)
        while len(self._lru) > self.max_entries:
            self._lru.popitem(last=False)

    def _from_store(self, texts: list[str]) -> dict[str, F32]:
        if self.store is None or not texts:
            return {}
        try:
            if not self.store.exists():
                return {}
            missing = set(self.store.missing(texts))
            known = [t for t in texts if t not in missing]
            if not known:
                return {}
            rows = self.store.get(known)
        except Exception as exc:  # a store problem never fails a decision: the encoder answers
            logger.info("feature store read skipped on the probe path (%s)", type(exc).__name__)
            return {}
        self.store_hits += len(known)
        return {t: np.asarray(r, dtype=np.float32) for t, r in zip(known, rows, strict=True)}

    def vectors(self, texts: Sequence[str]) -> Vectors:
        order = list(dict.fromkeys(texts))
        found: dict[str, F32] = {}
        for t in order:
            hit = self._lru.get(t)
            if hit is not None:
                self._lru.move_to_end(t)
                found[t] = hit
        rest = [t for t in order if t not in found]
        found.update(self._from_store(rest))
        rest = [t for t in order if t not in found]
        tokens = 0
        if rest:
            rows, tokens = self.embed(rest)
            arr = np.asarray(rows, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != len(rest):
                raise ValueError(f"the encoder returned {arr.shape} for {len(rest)} texts")
            self.embed_calls += 1
            for t, r in zip(rest, arr, strict=True):
                found[t] = r
        for t in order:
            self._remember(t, found[t])
        return Vectors(np.stack([found[t] for t in texts]), int(tokens), len(rest))


# -- the feature specifications (plan §5.3; the brief's R1) --------------------------------------


def spec_dim(spec: str, *, k: int = 2, projection_dim: int = 512, dim: int = 4096) -> int:
    """The feature dimension of ``spec`` (R1): 2; 2·512 + 1; 4096 + 1; 4096; 512 + K; 4096 + K."""
    if spec == "lowdim.v1":
        return 2
    if spec == "pair512.v1":
        return 2 * projection_dim + 1
    if spec == "pair4096.v1":
        return dim + 1
    if spec in ("joint4096@S1", "joint4096@S1ns"):
        return dim
    if spec == "choice.state.v1":
        return projection_dim + k
    if spec == "choice.raw.v1":
        return dim + k
    raise ValueError(f"unknown feature spec {spec!r}")


def _f64(x: npt.ArrayLike) -> F64:
    return np.asarray(x, dtype=np.float64)


def rank_fit_features(
    spec: str,
    *,
    xs: npt.ArrayLike,
    xc: npt.ArrayLike,
    xa: npt.ArrayLike,
    head: HeadProjector | None = None,
    xj: npt.ArrayLike | None = None,
) -> F64:
    """The ``[m, d]`` feature rows of ``m`` candidates for a rank_fit spec (R1), through the
    bench's own formulas (:func:`mesa_clm.learn.probe.rank_fit_features`: the one place they
    live). ``xs`` is the state's raw 4096-d L2 vector, ``xc`` the candidates' ``[m, 4096]``,
    ``xa`` the anchor's; ``head`` the served head (``lowdim.v1``, ``pair512.v1``: ``zs``/``zc``/
    ``za`` are its projections, ``s_latest = scale·(zs·zc − zs·za)``); ``xj`` the ``[m, 4096]``
    raw vectors of the candidates' joint context texts (``joint4096@*``, :func:`joint_texts`).
    ``s_raw = 100·(xs·xc − xs·xa)`` (``clm-raw``). Cosines are float64 dot products of the
    float32 vectors, as :mod:`mesa_clm.learn.offline` computes them."""
    from mesa_clm.learn.probe import rank_fit_features as shared

    s = _f64(xs).reshape(-1)
    c = _f64(xc)
    a = _f64(xa).reshape(-1)
    if c.ndim != 2 or c.shape[1] != s.shape[0] or a.shape[0] != s.shape[0]:
        raise ValueError("xs, xc and xa must share one dimension")
    if spec not in RANK_FIT_SPECS:
        raise ValueError(f"{spec!r} is not a rank_fit feature spec")
    m = c.shape[0]
    if spec in ("joint4096@S1", "joint4096@S1ns"):
        if xj is None:
            raise ValueError(f"{spec} needs the joint context vectors (xj)")
        j = _f64(xj)
        if j.shape != c.shape:
            raise ValueError("xj must be [m, dim] like xc")
        return shared(spec, joint=j)
    s_raw = RAW_SCALE * (c @ s - float(a @ s))
    if spec == "pair4096.v1":
        return shared(spec, xs=np.repeat(s[None, :], m, axis=0), xc=c, s_raw=s_raw)
    if head is None:
        raise ValueError(f"{spec} needs the served head's projections")
    zs = _f64(head.project_states(s[None, :].astype(np.float32))[0])
    zc = _f64(head.project_actions(c.astype(np.float32)))
    za = _f64(head.project_actions(a[None, :].astype(np.float32))[0])
    s_latest = head.scale * (zc @ zs - float(za @ zs))
    return shared(spec, s_latest=s_latest, s_raw=s_raw, zs=np.repeat(zs[None, :], m, axis=0), zc=zc)


def choice_features(
    spec: str, *, xs: npt.ArrayLike, xo: npt.ArrayLike, head: HeadProjector | None = None
) -> F64:
    """The ``[1, d]`` feature row of a closed choice (R1), through the bench's own formula
    (:func:`mesa_clm.learn.probe.choice_features`): ``choice.state.v1`` = ``zs`` ⊕ the K
    zero-shot option logits under ``clm-latest`` (``scale·zs·z_k``); ``choice.raw.v1`` = ``xs``
    ⊕ the K logits under ``clm-raw`` (``100·xs·x_k``). ``xo`` is ``[K, 4096]``."""
    from mesa_clm.learn.probe import choice_features as shared

    s = _f64(xs).reshape(-1)
    o = _f64(xo)
    if o.ndim != 2 or o.shape[1] != s.shape[0]:
        raise ValueError("xs and xo must share one dimension")
    if spec == "choice.raw.v1":
        return shared(spec, xs=s[None, :], logits=(RAW_SCALE * (o @ s))[None, :])
    if spec != "choice.state.v1":
        raise ValueError(f"{spec!r} is not a closed-choice feature spec")
    if head is None:
        raise ValueError(f"{spec} needs the served head's projections")
    zs = _f64(head.project_states(s[None, :].astype(np.float32))[0])
    zo = _f64(head.project_actions(o.astype(np.float32)))
    return shared(spec, zs=zs[None, :], logits=(head.scale * (zo @ zs))[None, :])


def spec_features(
    spec: str,
    *,
    xs: npt.ArrayLike,
    xo: npt.ArrayLike,
    anchor_index: int | None,
    head: HeadProjector | None = None,
    xj: npt.ArrayLike | None = None,
) -> F64:
    """The feature rows of one decision: for a rank_fit (``anchor_index`` set) one row per
    option in request order, the anchor's row built with the anchor's own vector on the
    candidate side (module docstring); for a closed choice one row. ``xo`` is the ``[k, 4096]``
    option vectors in request order (the anchor among them), ``xj`` the ``[k, 4096]`` joint
    context vectors (``joint4096@*``; the anchor's row is its own vector)."""
    o = _f64(xo)
    if anchor_index is None:
        return choice_features(spec, xs=xs, xo=o, head=head)
    xa = o[anchor_index]
    return rank_fit_features(spec, xs=xs, xc=o, xa=xa, head=head, xj=xj)


def joint_texts(
    spec: str, framing: Framing, state: Mapping[str, Any], candidates: Sequence[CandidateLike]
) -> list[str]:
    """The joint context texts of ``candidates`` for a ``joint4096@*`` spec: the X1 control
    framing's per-candidate anyjev state (``framings.control_state`` with ``n_candidates`` the
    group's size) rendered by CLM's ``state_text``, with the task question appended for ``@S1``
    and without it for ``@S1ns`` (exactly ``learn.features._joint_rows``)."""
    control = framing_of(framing.task_id, _JOINT_FRAMING)
    suffix = control.instructions if spec == "joint4096@S1" else None
    return [
        render.state_text(
            build_context(control, control_state(control, state, c, len(candidates))), suffix
        )
        for c in candidates
    ]


def probe_rank_probs(p_fit: Sequence[float], anchor_index: int) -> list[float]:
    """The group distribution of a rank_fit probe decision: the softmax over
    ``logit(p_fit)`` for the candidates and ``0`` for the anchor (the mirror of
    :meth:`PlattCalibrator.rank_fit`, whose candidate logits are ``a·s_c + b``), ``p_fit``
    clipped to ``[PROBE_LOGIT_CLIP, 1 - PROBE_LOGIT_CLIP]`` before the logit."""
    logits = []
    for i, p in enumerate(p_fit):
        if i == anchor_index:
            logits.append(0.0)
            continue
        q = min(max(float(p), PROBE_LOGIT_CLIP), 1.0 - PROBE_LOGIT_CLIP)
        logits.append(math.log(q) - math.log1p(-q))
    return _softmax(logits)


# -- the bundle -------------------------------------------------------------------------------


class Promotion(_Frozen):
    """One task's production artifact (``CURRENT.json``, plan §5.5): the tier, the question
    key it serves and the artifacts version it comes from; ``cite`` the cell it was promoted on."""

    tier: Literal["calibrated", "probe", "head"]
    question_key: str
    version: str = Field(min_length=1)
    cite: str | None = None

    @field_validator("question_key")
    @classmethod
    def _qk(cls, v: str) -> str:
        return _hex_width(v, 16, "question_key")


class ArtifactBundle(_Frozen):
    """The learned artifacts a provider serves (plan §5.5): the calibrators of
    ``calibrators.json`` and, from M4, the probes (``probes/<question_key>.json``) and the
    promotion table of ``CURRENT.json``, all fitted under one ``(encoder_fp, clm_model_fp)``.

    The provider refuses a bundle whose fingerprints differ from the live ones (K4) and applies
    an artifact only to decisions of its exact ``question_key`` (a rotated framing has no
    artifact until it is re-benched). ``promoted`` (task -> :class:`Promotion`) is the
    promotion table: when present, only promoted entries are served (module docstring);
    ``versions`` names the artifacts version of an entry that came from another version than
    ``version`` (a promotion table spans versions). Heads join in M7.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    version: str = Field(min_length=1)
    encoder_fp: str
    clm_model_fp: str
    framings_lock_sha: str | None = None
    calibrators: dict[str, Calibrator] = Field(default_factory=dict)
    probes: dict[str, ServedProbe] = Field(default_factory=dict)
    promoted: dict[str, Promotion] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def _consistent(self) -> ArtifactBundle:
        mine = {"encoder_fp": self.encoder_fp, "clm_model_fp": self.clm_model_fp}
        for key, probe in self.probes.items():
            if probe.question_key != key:
                raise ValueError(f"probe {probe.question_key} is filed under {key!r}")
            theirs = {k: probe.fingerprint.get(k) for k in mine}
            if theirs != mine:
                raise ValueError(
                    f"probe {key} was fitted under {theirs}, the bundle under {mine} (K4)"
                )
        for task_id, promo in self.promoted.items():
            if promo.tier == "probe" and promo.question_key not in self.probes:
                raise ValueError(
                    f"{task_id}: the promoted probe {promo.question_key} is not loaded"
                )
            if promo.tier == "calibrated" and promo.question_key not in self.calibrators:
                raise ValueError(
                    f"{task_id}: the promoted calibrator {promo.question_key} is not loaded"
                )
            if promo.tier == "head":
                raise ValueError(f"{task_id}: a promoted head cannot be served before M7")
        return self

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

    def probe_for(self, question_key: str) -> ServedProbe | None:
        return self.probes.get(question_key)

    def promotion_for(self, question_key: str) -> Promotion | None:
        """The promotion naming ``question_key``, if any."""
        for promo in self.promoted.values():
            if promo.question_key == question_key:
                return promo
        return None

    def version_of(self, question_key: str) -> str:
        """The artifacts version the entry for ``question_key`` came from."""
        probe = self.probes.get(question_key)
        if probe is not None:
            return probe.version
        return self.versions.get(question_key, self.version)

    def ref(self, question_key: str) -> ArtifactRef:
        """The record-side reference to this bundle for one question key (invariant 5)."""
        return ArtifactRef(
            version=self.version_of(question_key),
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
    text: str = ""
    probe: ServedProbe | None = None
    qid: str = ""


_Served = tuple[Level, PlattCalibrator | TemperatureCalibrator | None, ServedProbe | None]


class TieredProvider:
    """CLM tiers over one clm-serve client (module docstring).

    ``client`` is a :class:`~mesa_clm.clm.http.ClmHttpClient` (or anything with its
    ``system_one``); ``encoder`` counts context tokens for the guard (``None``: chars/2) and,
    when it also embeds (:class:`~mesa_clm.clm.encoder.EncoderClient`), is the probe path's
    encoder unless ``embed`` names another; ``fingerprint`` is the live D5 bundle stamped on
    every record; ``artifacts`` the learned bundle, refused unless fitted under that
    fingerprint. ``head`` is the served head's export (the probe path's projector, needed by
    the specs that read head quantities) and ``feature_store`` the live store, read-only.
    ``method`` is ``clm`` for the real server, ``fake`` for :class:`~mesa_clm.clm.fake.FakeClm`
    (never cited). ``model`` is the served head (default: the client's). ``on_call`` receives
    a :class:`~mesa_clm.providers.base.ClmCall` per request (clm-serve or the encoder).
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
        embed: Embedder | None = None,
        head: HeadProjector | None = None,
        feature_store: VectorSource | None = None,
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
        embedder = embed if embed is not None else getattr(encoder, "embed", None)
        self.head = head
        self.vectors: VectorCache | None = (
            VectorCache(embedder, store=feature_store) if callable(embedder) else None
        )
        if artifacts is not None:
            self._check_probes(artifacts)

    def _check_probes(self, artifacts: ArtifactBundle) -> None:
        """A served probe must belong to the served model and have what it needs (the encoder,
        its head), before any request: a promoted probe that cannot be served is a setup error,
        never a silent zero_shot (K4)."""
        for key, probe in artifacts.probes.items():
            promo = artifacts.promotion_for(key)
            if promo is None or promo.tier != "probe":
                continue  # not served: loaded for inspection only
            if probe.model != self.model:
                raise ArtifactError(
                    f"probe {key} belongs to {probe.model}; this provider serves {self.model}"
                )
            if self.vectors is None:
                raise ArtifactError(
                    f"probe {key} is promoted but no encoder embeds for this provider "
                    "(the probe tier needs EncoderClient.embed)"
                )
            if probe.needs_head and self.head is None:
                raise ArtifactError(
                    f"probe {key} ({probe.spec}) needs the served head's export "
                    "(heads/npz/<sha8>.npz), which this provider was not given"
                )

    # -- tiers ----------------------------------------------------------------------------------

    def _calibrator(self, question_key: str) -> PlattCalibrator | TemperatureCalibrator | None:
        """The servable calibrator of ``question_key``: under a promotion table only a promoted
        one (module docstring), else whatever the bundle has."""
        if self.artifacts is None:
            return None
        if self.artifacts.promoted:
            promo = self.artifacts.promotion_for(question_key)
            if promo is None or promo.tier != "calibrated":
                return None
        return self.artifacts.calibrator_for(question_key)

    def _probe(self, question_key: str) -> ServedProbe | None:
        """The promoted probe of ``question_key`` (a probe is served only once promoted)."""
        if self.artifacts is None:
            return None
        promo = self.artifacts.promotion_for(question_key)
        if promo is None or promo.tier != "probe":
            return None
        return self.artifacts.probe_for(question_key)

    def resolve_tier(self, question_key: str) -> Level:
        """The best tier servable for ``question_key``: ``probe`` when a promoted probe exists
        (``CURRENT.json``), else ``calibrated`` when a servable calibrator exists, else
        ``zero_shot`` (heads land in M7)."""
        if self._probe(question_key) is not None:
            return "probe"
        return "calibrated" if self._calibrator(question_key) is not None else "zero_shot"

    def supports_tier(self, task_id: str, tier: str) -> bool:
        """Whether ``tier`` can be served for ``task_id``'s active framing."""
        if task_id not in FRAMINGS:
            return False
        if tier in ("auto", "zero_shot"):
            return True
        qk = active_framing(task_id).question_key
        if tier == "calibrated":
            return self._calibrator(qk) is not None
        if tier == "probe":
            return self._probe(qk) is not None
        return False

    def _tier(self, framing: Framing, tier: str | None) -> _Served:
        qk = framing.question_key
        requested = tier or "auto"
        if requested == "auto":
            level = self.resolve_tier(qk)
            if level == "probe":
                return level, None, self._probe(qk)
            return level, self._calibrator(qk) if level == "calibrated" else None, None
        if requested == "zero_shot":
            return "zero_shot", None, None
        where = f"artifacts {self.artifacts.version}" if self.artifacts else "no artifacts"
        if requested == "calibrated":
            cal = self._calibrator(qk)
            if cal is None:
                raise TierUnavailable(
                    f"{framing.task_id}/{framing.id}: no calibrator for question_key {qk} is "
                    f"promoted ({where}); fit and promote one (`mesa-clm learn fit`, `learn "
                    "promote`) or ask zero_shot"
                )
            return "calibrated", cal, None
        if requested == "probe":
            probe = self._probe(qk)
            if probe is None:
                raise TierUnavailable(
                    f"{framing.task_id}/{framing.id}: no probe for question_key {qk} is "
                    f"promoted ({where}); fit and promote one (`mesa-clm learn fit`, `learn "
                    "promote --tier probe`) or ask auto"
                )
            return "probe", None, probe
        if requested == "head":
            raise TierUnavailable(
                "the head tier (a fine-tuned head promoted to clm-serve, plan §5.3) lands in "
                "M7; until then ask zero_shot, calibrated or probe"
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
        and tasks (``column.annotate`` and ``column.aspect`` share ``column_state``). A request
        at the probe tier is scored locally (module docstring), never sent to clm-serve."""
        tiers: dict[tuple[str, str], _Served] = {}
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
            elif it.probe is not None:
                out[it.index] = self._probe_decide(it)
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
        probe: ServedProbe | None = None,
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
        if probe is not None:
            expected = "rank_fit" if opts.anchor_index is not None else "choice"
            if probe.shape != expected:
                raise ArtifactError(
                    f"{req.framing.task_id}: probe {probe.question_key} ({probe.spec}) scores a "
                    f"{probe.shape}, the framing asks a {expected}"
                )
            if expected == "choice" and probe.k != len(opts.options):
                raise ArtifactError(
                    f"{req.framing.task_id}: probe {probe.question_key} scores K={probe.k} "
                    f"options, the framing offers {len(opts.options)}"
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
            text=text,
            probe=probe,
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
            if isinstance(exc, ListenerOwnerError):
                # Another account holds clm-serve's port: the key was not sent, and every
                # group would meet the same socket (DESIGN A5).
                raise DeciderRefused(0, f"clm-serve refused before /v1/systemone: {exc}") from None
            if isinstance(exc, ClmError) and exc.status in REFUSED_STATUSES:
                # A wrong key or route answers every group the same way: refuse the run rather
                # than degrade it to ols_rank group by group (module docstring).
                raise DeciderRefused(
                    exc.status,
                    f"clm-serve refused /v1/systemone ({exc.status}): "
                    + ("the key was rejected" if exc.status in (401, 403) else "no such route"),
                ) from None
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

    # -- the probe tier (M4) ----------------------------------------------------------------------

    def _zero_shot_logits(self, xs: F32, xo: F32) -> F64:
        """The served model's zero-shot logits over the option vectors: CLM's maths on the
        same vectors (``clm-raw``: ``100·cos``; a head: ``scale·cos`` of its projections)."""
        if self.model == RAW_MODEL:
            return np.asarray(RAW_SCALE * (xo.astype(np.float64) @ xs.astype(np.float64)))
        if self.head is None:
            raise ArtifactError(
                f"the probe path needs {self.model}'s head export for the zero-shot parity"
            )
        return np.asarray(
            self.head.logits(self.head.project_states(xs[None, :]), self.head.project_actions(xo))[
                0
            ]
        )

    def _probe_decide(self, it: _Item) -> DecisionRecord:
        """One probe decision (module docstring): the vectors, the zero-shot parity
        distribution, the spec features, the probe's calibrated probabilities."""
        probe = it.probe
        if probe is None or self.vectors is None:  # pragma: no cover - checked at construction
            raise ArtifactError("a probe decision needs a probe and an encoder")
        a = it.opts.anchor_index
        texts = [it.text, *it.opts.option_texts]
        joint: list[str] = []
        if probe.spec.startswith("joint4096@"):
            # The candidates' joint contexts; the anchor's row is its own option text.
            joint = joint_texts(
                probe.spec, it.request.framing, it.request.state, it.request.candidates or ()
            )
        started = time.monotonic()
        try:
            got = self.vectors.vectors([*texts, *joint])
        except (ClmError, EndpointError) as exc:
            wall = (time.monotonic() - started) * 1000.0
            status = _call_status(exc)
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("the encoder did not answer the probe path (%s)", status)
            self._report(
                ClmCall(
                    endpoint=EMBEDDINGS,
                    model=self.model,
                    n_questions=1,
                    n_candidates=len(it.opts.wire_keys),
                    latency_ms=wall,
                    wall_ms=wall,
                    status=status,
                    error=message[:_ERROR_CHARS],
                )
            )
            if isinstance(exc, ListenerOwnerError):
                raise DeciderRefused(0, f"the encoder refused before {EMBEDDINGS}: {exc}") from None
            if isinstance(exc, ClmError) and exc.status in REFUSED_STATUSES:
                raise DeciderRefused(
                    exc.status,
                    f"the encoder refused {EMBEDDINGS} ({exc.status}): "
                    + ("the key was rejected" if exc.status in (401, 403) else "no such route"),
                ) from None
            return self._unavailable(it, reason="decider_unavailable", error=message)
        wall = (time.monotonic() - started) * 1000.0
        if got.embedded:
            self.input_tokens += got.tokens
            self._report(
                ClmCall(
                    endpoint=EMBEDDINGS,
                    model=self.model,
                    n_questions=1,
                    n_candidates=len(it.opts.wire_keys),
                    latency_ms=wall,
                    wall_ms=wall,
                    input_tokens=got.tokens,
                )
            )
        k = len(it.opts.options)
        xs = got.rows[0]
        xo = got.rows[1 : 1 + k]
        xj: F32 | None = None
        if joint:
            # Candidate rows in request order, the anchor's row its own option vector.
            rows = list(got.rows[1 + k :])
            cand = iter(rows)
            xj = np.stack([xo[i] if i == a else next(cand) for i in range(k)])
        logits = self._zero_shot_logits(xs, xo)
        raw = render.softmax([float(x) for x in logits])
        if any(not math.isfinite(p) or p <= 0.0 for p in raw):
            return self._unavailable(it, reason="numeric_underflow")
        features = spec_features(probe.spec, xs=xs, xo=xo, anchor_index=a, head=self.head, xj=xj)
        try:
            predicted = probe.predict(features)
        except ArtifactError as exc:
            return self._unavailable(it, reason="malformed_answer", error=str(exc))
        s_c: list[float] | None = None
        p_fit: list[float] | None = None
        if a is not None:
            log_a = math.log(raw[a])
            s_c = [0.0 if i == a else math.log(p) - log_a for i, p in enumerate(raw)]
            p_fit = [min(max(float(p), 0.0), 1.0) for p in predicted[:, 0]]
            probs = probe_rank_probs(p_fit, a)
        else:
            probs = [float(p) for p in predicted[0]]
            total = math.fsum(probs)
            if total <= 0.0:
                return self._unavailable(it, reason="malformed_answer", error="probe gave no mass")
            probs = [p / total for p in probs]
        idx, top, margin = rank_probs(probs)
        diag: dict[str, Scalar] = {
            "probe": True,
            "spec": probe.spec,
            "feature_dim": int(features.shape[1]),
            "n_embedded": got.embedded,
            "token_source": it.token_source,
        }
        f = self._fields(it)
        f.update(
            served_model=None,
            level="probe",
            calibration=probe.calibration,
            probs=probs,
            raw_probs=raw,
            s_c=s_c,
            p_fit=p_fit,
            answer_index=idx,
            answer=it.opts.options[idx],
            confidence=top,
            margin=margin,
            latency_ms=wall,
            feature_spec=probe.spec,
            artifact_version=probe.version,
            artifact=probe.ref(),
            diagnostics=diag,
        )
        rec = DecisionRecord.model_validate(f)
        if it.request.framing.mask_rule == "aspect":
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
        embed: Embedder | None = None,
        head: HeadProjector | None = None,
        feature_store: VectorSource | None = None,
    ) -> None:
        engine = clm if clm is not None else (FakeClm(seed=seed) if client is None else None)
        if client is None:
            client = FakeClmClient(engine, model=model)
        # The probe path over the fake stack: the fake encoder embeds and the fake head
        # projects (the same vectors and maths the in-process engine answers with), unless the
        # caller names an encoder over the wire or its own embed/head.
        if engine is not None:
            if embed is None and encoder is None:
                embed = engine.encoder.embed
            if head is None:
                head = engine.heads.get(LATEST_MODEL)
        self.clm = engine
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
            embed=embed,
            head=head,
            feature_store=feature_store,
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
