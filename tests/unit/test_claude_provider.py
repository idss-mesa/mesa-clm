"""Claude as a recorded second opinion (``providers.claude_provider``; DESIGN D6, D22) with a fake
SDK client: the structured-output request (the context CLM saw, never the raw card; a
``Literal`` answer model over the option ids; adaptive thinking; the effort knob), records at
level ``none`` with no distribution and their own ``question_key``, and the abstaining records
for a refusal, an unparsable answer, an SDK error and a missing ``anthropic`` package (pattern:
``tests/unit/test_planner_claude.py``)."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from mesa_clm import framings as fr
from mesa_clm.cards import DatasetCard
from mesa_clm.config import ClaudeConfig
from mesa_clm.providers import (
    ClaudeStructuredProvider,
    DecisionRecord,
    TierUnavailable,
    claude_question_key,
    fake_fingerprint,
    sha256_text,
)
from mesa_clm.providers.claude_provider import (
    METHOD,
    MIN_MAX_TOKENS,
    SYSTEM,
    answer_model,
    render_prompt,
)
from mesa_clm.registry import ANCHOR_KEY, ASPECT_OPTIONS
from mesa_clm.states import column_state, target_state
from tests.conftest import CARD_TEXT

FP = fake_fingerprint()
TERM = fr.active_framing("term.fits")
ASPECT = fr.active_framing("column.aspect")
CANDS = [
    fr.FramingCandidate("PATO:0000040", "distance", "A 1-D extent quality between two points."),
    fr.FramingCandidate("UO:0000008", "meter", "A length unit equal to the SI base unit."),
]
USAGE = SimpleNamespace(input_tokens=812, output_tokens=40)
SECRET = "sk-ant-not-a-real-key-0123456789"


class FakeMessages:
    """``messages.parse``: records the call; answers with ``answer`` (validated through the
    request's own ``output_format``), a scripted response, or an error."""

    def __init__(
        self,
        answer: str | None = None,
        *,
        response: Any = None,
        error: Exception | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        self.answer = answer
        self.response = response
        self.error = error
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        parsed = kwargs["output_format"](answer=self.answer) if self.answer is not None else None
        return SimpleNamespace(
            parsed_output=parsed,
            stop_reason=self.stop_reason,
            usage=USAGE,
            model="claude-opus-5",
            content=[],
        )


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def _provider(messages: FakeMessages, **cfg: Any) -> ClaudeStructuredProvider:
    return ClaudeStructuredProvider(ClaudeConfig(**cfg), FP, client=FakeClient(messages))


def _col(card: DatasetCard, name: str = "observerDistance") -> Any:
    return next(c for c in card.columns if c.name == name)


@pytest.fixture
def target(card: DatasetCard) -> dict[str, Any]:
    return target_state(card, "column", "measurement", column=_col(card))


def _assert_no_distribution(rec: DecisionRecord) -> None:
    assert (rec.provider, rec.method, rec.level, rec.calibration) == (
        "claude",
        METHOD,
        "none",
        "none",
    )
    assert rec.probs is None and rec.raw_probs is None and rec.s_c is None and rec.p_fit is None
    assert rec.confidence is None and rec.margin is None and rec.clm_confidence is None


def test_rank_fit_second_opinion_record(target: dict[str, Any]) -> None:
    messages = FakeMessages("UO:0000008")
    provider = _provider(messages)
    [rec] = provider.decide(TERM, [target], [CANDS])
    _assert_no_distribution(rec)
    assert rec.options == ["PATO:0000040", "UO:0000008", ANCHOR_KEY] and rec.anchor_index == 2
    assert rec.answer == "UO:0000008" and rec.answer_index == 1 and not rec.anchor_won
    assert rec.model == "claude-opus-5" and rec.served_model == "claude-opus-5"
    assert provider.served_model == "claude-opus-5" and provider.requests == 1
    # Its own question key, derived from the framing's (never mixed with CLM decisions).
    assert rec.question_key == claude_question_key(TERM) != TERM.question_key
    assert rec.diagnostics["base_question_key"] == TERM.question_key
    assert rec.framing_id == TERM.id and rec.task_key == TERM.task_key
    assert (rec.encoder_fp, rec.clm_model_fp) == (FP.encoder_fp, FP.clm_model_fp)
    [call] = messages.calls
    prompt = call["messages"][0]["content"]
    assert rec.context_sha256 == sha256_text(SYSTEM + "\n\n" + prompt)
    assert rec.context_tokens == 812 and rec.diagnostics["output_tokens"] == 40


def test_the_request_shape(target: dict[str, Any]) -> None:
    messages = FakeMessages(ANCHOR_KEY)
    [rec] = _provider(messages, effort="medium", max_tokens=64).decide(TERM, [target], [CANDS])
    assert rec.anchor_won and rec.answer == ANCHOR_KEY
    [call] = messages.calls
    assert call["model"] == "claude-opus-5" and call["system"] == SYSTEM
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["max_tokens"] == MIN_MAX_TOKENS  # adaptive thinking needs room
    assert "tool_choice" not in call  # never forced (rejected on the newest models)
    prompt = call["messages"][0]["content"]
    assert prompt == render_prompt(TERM, target, rec.options, rec.option_texts)
    assert fr.context_text(TERM, target) in prompt  # the context CLM's state head saw
    assert f"- {ANCHOR_KEY}: " in prompt and "- PATO:0000040: distance: " in prompt
    assert TERM.task.text in prompt
    # Never the raw card: no card header line, no raw column row.
    assert "# Dataset:" not in prompt and "## Columns" not in prompt
    for line in CARD_TEXT.splitlines():
        if line.startswith("- ") and "|" in line:
            assert line not in prompt
    schema = call["output_format"].model_json_schema()
    assert schema["properties"]["answer"]["enum"] == rec.options
    assert schema.get("additionalProperties") is False


def test_closed_choice_second_opinion(card: DatasetCard) -> None:
    messages = FakeMessages(ASPECT_OPTIONS[3])
    [rec] = _provider(messages).decide(ASPECT, [column_state(card, _col(card))])
    _assert_no_distribution(rec)
    assert rec.options == list(ASPECT_OPTIONS) and rec.anchor_index is None
    assert rec.answer == ASPECT_OPTIONS[3] and rec.kind == "choice"
    prompt = messages.calls[0]["messages"][0]["content"]
    assert f"Question: {ASPECT.task.text}" in prompt and ANCHOR_KEY not in prompt


def test_answer_model_accepts_only_the_options() -> None:
    model = answer_model(["a", "b"])
    assert model(answer="b").model_dump() == {"answer": "b"}
    with pytest.raises(ValidationError):
        model(answer="c")
    with pytest.raises(ValidationError):
        model(answer="a", why="because")


def test_refusal_and_unparsed_answers_abstain(target: dict[str, Any]) -> None:
    [refused] = _provider(FakeMessages("UO:0000008", stop_reason="refusal")).decide(
        TERM, [target], [CANDS]
    )
    assert refused.answer_index == -1 and refused.answer == "" and refused.reason == "refusal"
    assert refused.diagnostics["stop_reason"] == "refusal"
    [unparsed] = _provider(FakeMessages(stop_reason="max_tokens")).decide(TERM, [target], [CANDS])
    assert unparsed.answer_index == -1 and unparsed.reason == "unparsed"
    assert unparsed.diagnostics["stop_reason"] == "max_tokens"
    odd = SimpleNamespace(
        parsed_output=SimpleNamespace(answer="GO:0008150"), stop_reason="end_turn"
    )
    [other] = _provider(FakeMessages(response=odd)).decide(TERM, [target], [CANDS])
    assert other.reason == "unparsed" and other.diagnostics["unparsed"] == "GO:0008150"
    assert other.context_tokens == 0 and other.served_model is None  # no usage, no model


def test_sdk_errors_abstain_and_never_leak_secrets(
    target: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    provider = _provider(FakeMessages(error=RuntimeError("503 overloaded")))
    with caplog.at_level(logging.WARNING, logger="mesa_clm.providers.claude_provider"):
        [rec] = provider.decide(TERM, [target], [CANDS])
    assert rec.answer_index == -1 and rec.reason == "error"
    assert rec.diagnostics["error"] == "RuntimeError: 503 overloaded"
    assert "claude second opinion failed (RuntimeError)" in caplog.text
    assert "503 overloaded" not in caplog.text  # only the exception type is logged
    for record in caplog.records:
        assert SECRET not in record.getMessage()


def test_missing_sdk_abstains(target: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic -> ImportError
    provider = ClaudeStructuredProvider(ClaudeConfig(), FP)
    [rec] = provider.decide(TERM, [target], [CANDS])
    assert rec.answer_index == -1 and rec.reason == "error"
    assert "ModuleNotFoundError" in str(rec.diagnostics["error"])  # an ImportError


def test_lazy_client_uses_the_sdk(target: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    messages = FakeMessages("PATO:0000040")
    fake_sdk = SimpleNamespace(Anthropic=lambda: FakeClient(messages))
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
    provider = ClaudeStructuredProvider(ClaudeConfig(model="claude-opus-5-5"), FP)
    [rec] = provider.decide(TERM, [target], [CANDS])
    assert rec.answer == "PATO:0000040" and rec.model == "claude-opus-5-5"
    provider.decide(TERM, [target], [CANDS])
    assert len(messages.calls) == 2  # one client, built once


def test_tiers_and_batches(target: dict[str, Any], card: DatasetCard) -> None:
    provider = _provider(FakeMessages("PATO:0000040"))
    assert provider.resolve_tier(TERM.question_key) == "none"
    assert provider.supports_tier("term.fits", "none") and provider.supports_tier(
        "column.aspect", "auto"
    )
    assert not provider.supports_tier("term.fits", "zero_shot")
    assert not provider.supports_tier("nope", "none")
    with pytest.raises(TierUnavailable, match="second opinion at level 'none'"):
        provider.decide(TERM, [target], [CANDS], tier="calibrated")
    recs = provider.decide(TERM, [target, target], [CANDS, CANDS[:1]])
    assert [r.options for r in recs] == [
        ["PATO:0000040", "UO:0000008", ANCHOR_KEY],
        ["PATO:0000040", ANCHOR_KEY],
    ]
    with pytest.raises(ValueError, match="takes no candidates"):
        provider.decide(ASPECT, [column_state(card, _col(card))], [CANDS])


def test_question_key_is_stable_and_framing_specific() -> None:
    assert claude_question_key(TERM) == claude_question_key(fr.active_framing("term.fits"))
    assert claude_question_key(TERM) != claude_question_key(fr.framing("term.fits", "F4"))
    assert claude_question_key(TERM) != claude_question_key(ASPECT)
    assert len(claude_question_key(TERM)) == 16
