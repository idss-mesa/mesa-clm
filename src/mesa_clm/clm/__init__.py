"""The torch-free CLM serving clients and their fingerprints (plan §3 ``clm/*``; DESIGN D5, D16).

* :mod:`mesa_clm.clm.http` - ``ClmHttpClient`` over clm-serve (``/v1/systemone``, ``/v1/rank``,
  ``/v1/models``, ``/health``), the ``Noul``/``Choice``/``Score`` questions and typed answers;
* :mod:`mesa_clm.clm.encoder` - ``EncoderClient`` over the vLLM pooling encoder
  (``/v1/embeddings`` with left truncation, ``/tokenize``, the token guard);
* :mod:`mesa_clm.clm.headproj` - CLM's head MLP in numpy over an exported ``.npz``;
* :mod:`mesa_clm.clm.fingerprint` - ``encoder_fp``, ``clm_model_fp``, ``Fingerprint``,
  ``serving.lock.json``;
* :mod:`mesa_clm.clm.fake` - the deterministic ``FakeEncoder`` / ``FakeClm`` for hermetic tests.
"""

from __future__ import annotations

from mesa_clm.clm.encoder import EncoderClient, TokenCount, chars_estimate, token_guard
from mesa_clm.clm.fake import FakeClm, FakeClmError, FakeEncoder
from mesa_clm.clm.fingerprint import (
    ClmModelSpec,
    EncoderSpec,
    Fingerprint,
    FingerprintMismatch,
    LockError,
    ServingLock,
    canonical_json,
    clm_model_fp,
    encoder_fp,
    fingerprint,
    load_serving_lock,
    parse_serving_lock,
    sign_lock_body,
)
from mesa_clm.clm.headproj import HeadConfig, HeadError, HeadProjector, random_head
from mesa_clm.clm.http import (
    Choice,
    ChoiceAnswer,
    ClmError,
    ClmHttpClient,
    HttpEndpoint,
    Noul,
    NoulAnswer,
    Question,
    RankedCandidate,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    question_to_dict,
)

__all__ = [
    "Choice",
    "ChoiceAnswer",
    "ClmError",
    "ClmHttpClient",
    "ClmModelSpec",
    "EncoderClient",
    "EncoderSpec",
    "FakeClm",
    "FakeClmError",
    "FakeEncoder",
    "Fingerprint",
    "FingerprintMismatch",
    "HeadConfig",
    "HeadError",
    "HeadProjector",
    "HttpEndpoint",
    "LockError",
    "Noul",
    "NoulAnswer",
    "Question",
    "RankedCandidate",
    "Score",
    "ScoreAnswer",
    "ServingLock",
    "SystemOneResponse",
    "TokenCount",
    "Usage",
    "canonical_json",
    "chars_estimate",
    "clm_model_fp",
    "encoder_fp",
    "fingerprint",
    "load_serving_lock",
    "parse_serving_lock",
    "question_to_dict",
    "random_head",
    "sign_lock_body",
    "token_guard",
]
