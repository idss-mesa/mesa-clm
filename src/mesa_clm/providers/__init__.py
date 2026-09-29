"""Decision providers: what answers a framing's question and how honestly it reports its tier
(DESIGN D2, D6, D7, D22, D28; plan §4.1, §4.5, §4.6).

* :mod:`mesa_clm.providers.base` - :class:`DecisionRecord` and its ``_honest`` invariants,
  :class:`DecisionProvider`, :func:`apply_mask`, :func:`rank_probs`, :class:`ClmCall`;
* :mod:`mesa_clm.providers.tiered` - :class:`TieredProvider` (zero_shot and calibrated CLM
  tiers, s_c recovery, one request per shared context), :class:`FakeProvider`, the
  :class:`PlattCalibrator` / :class:`TemperatureCalibrator` interface and
  :class:`ArtifactBundle`, and the degraded :class:`OlsRankProvider` (D28);
* :mod:`mesa_clm.providers.claude_provider` - :class:`ClaudeStructuredProvider`, a recorded
  second opinion at level ``none`` that never decides (D22).
"""

from __future__ import annotations

from mesa_clm.providers.base import (
    CALL_STATUSES,
    CLM_METHODS,
    REASONS,
    ArtifactRef,
    CallStatus,
    ClmCall,
    DecisionProvider,
    DecisionRecord,
    FramingOptions,
    Scalar,
    TierUnavailable,
    apply_mask,
    aspect_keep,
    base_fields,
    check_batch,
    framing_options,
    meets_level,
    rank_probs,
    sha256_text,
    sigmoid,
)
from mesa_clm.providers.claude_provider import ClaudeStructuredProvider, claude_question_key
from mesa_clm.providers.tiered import (
    CLM_COMMIT,
    ArtifactBundle,
    ArtifactError,
    Calibrator,
    CalibratorError,
    DecisionRequest,
    FakeClmClient,
    FakeProvider,
    OlsRankProvider,
    PlattCalibrator,
    SystemOneClient,
    TemperatureCalibrator,
    TieredProvider,
    TokenCounter,
    fake_fingerprint,
    load_calibrator,
    ols_rank_record,
)

__all__ = [
    "CALL_STATUSES",
    "CLM_COMMIT",
    "CLM_METHODS",
    "REASONS",
    "ArtifactBundle",
    "ArtifactError",
    "ArtifactRef",
    "Calibrator",
    "CalibratorError",
    "CallStatus",
    "ClaudeStructuredProvider",
    "ClmCall",
    "DecisionProvider",
    "DecisionRecord",
    "DecisionRequest",
    "FakeClmClient",
    "FakeProvider",
    "FramingOptions",
    "OlsRankProvider",
    "PlattCalibrator",
    "Scalar",
    "SystemOneClient",
    "TemperatureCalibrator",
    "TierUnavailable",
    "TieredProvider",
    "TokenCounter",
    "apply_mask",
    "aspect_keep",
    "base_fields",
    "check_batch",
    "claude_question_key",
    "fake_fingerprint",
    "framing_options",
    "load_calibrator",
    "meets_level",
    "ols_rank_record",
    "rank_probs",
    "sha256_text",
    "sigmoid",
]
