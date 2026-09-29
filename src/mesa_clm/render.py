"""Typed facade over the vendored CLM text contract (DESIGN D4).

Strict code never imports ``mesa_clm._vendor.clm.schema`` directly: everything that must agree
byte-for-byte with CLM's serving and training (how a state and a question become the state text
and the candidate texts, and how logits become an answer) goes through this module, so a change
in the vendored file is visible in exactly one place and the ``schema_sha256`` fingerprint
(D5) names the contract every decision was made under.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any, Final, cast

from mesa_clm._vendor.clm import schema as _schema

QUESTION_TYPES: Final[tuple[str, ...]] = tuple(_schema.QUESTION_TYPES)
NOUL_KEYS: Final[tuple[str, str]] = ("false", "true")

Question = dict[str, Any]
Answer = dict[str, Any]


def schema_sha256() -> str:
    """sha256 of the vendored ``schema.py`` bytes (the text contract every record cites)."""
    data = resources.files("mesa_clm._vendor.clm").joinpath("schema.py").read_bytes()
    return hashlib.sha256(data).hexdigest()


def to_text(x: Any, indent: int = 0) -> str:
    """Render a state or description as the prose CLM's heads were trained on."""
    return str(_schema.to_text(x, indent))


def state_text(state: Any, instructions: Any) -> str:
    """Context first, question last; ``instructions`` None or empty gives the state alone."""
    return str(_schema.state_text(state, instructions))


def candidates(question: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """``(keys, candidate_texts)`` the action head sees for one wire question."""
    keys, texts = _schema.candidates(dict(question))
    return list(keys), list(texts)


def build_pairs(
    state: Any, questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[str, list[str], list[str]]]:
    """``{qid: (state_text, keys, candidate_texts)}`` exactly as the CLM engine builds them."""
    pairs = _schema.build_pairs(state, {k: dict(v) for k, v in questions.items()})
    return {qid: (str(s), list(keys), list(texts)) for qid, (s, keys, texts) in pairs.items()}


def softmax(logits: Sequence[float]) -> list[float]:
    return [float(p) for p in _schema.softmax(list(logits))]


def clm_confidence(probs: Sequence[float]) -> float:
    """CLM's own confidence field (top minus mean of the rest); stored only as ``clm_confidence`` (D7)."""
    return float(_schema.confidence(list(probs)))


def answer_from_logits(
    question: Mapping[str, Any], keys: Sequence[str], logits: Sequence[float]
) -> Answer:
    return cast(Answer, _schema.answer_from_logits(dict(question), list(keys), list(logits)))


def answer_from_probs(
    question: Mapping[str, Any], keys: Sequence[str], probs: Sequence[float]
) -> Answer:
    return cast(Answer, _schema.answer_from_probs(dict(question), list(keys), list(probs)))


def label_of(answer: Mapping[str, Any]) -> str:
    return str(_schema.label_of(dict(answer)))


def probabilities_of(answer: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in _schema.probabilities_of(dict(answer)).items()}
