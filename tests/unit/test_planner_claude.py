"""The Claude planner with a fake SDK client: the structured-output request, the parsed plan,
and the static fallback on refusal, no parse, SDK error and a missing ``anthropic`` package."""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from mesa_clm.cards import DatasetCard
from mesa_clm.config import Config, PlannerConfig
from mesa_clm.planner import make_planner
from mesa_clm.planner.base import ColumnHint, Plan
from mesa_clm.planner.claude_planner import MAX_TOKENS, ClaudePlanner
from mesa_clm.planner.gateway_planner import SYSTEM, plan_prompt, prompt_sha256
from mesa_clm.planner.static_planner import StaticPlanner

GOOD_PLAN = Plan(
    ontologies=["envo", "ncbitaxon"],
    columns={"scientificName": ColumnHint(aspect="taxon", ontology="ncbitaxon", queries=["bird"])},
    taxon_queries=["Aves"],
    notes="ok",
)


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.messages = FakeMessages(response, error)


USAGE = SimpleNamespace(input_tokens=1200, output_tokens=300)


def response(
    parsed: Any = None,
    *,
    stop_reason: str = "end_turn",
    content: list[Any] | None = None,
    usage: Any = USAGE,
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=parsed, stop_reason=stop_reason, content=content or [], usage=usage
    )


def test_success_request_and_result(card: DatasetCard) -> None:
    cfg = PlannerConfig(claude_model="claude-opus-5-5", claude_effort="medium", timeout=42.0)
    client = FakeClient(response(GOOD_PLAN))
    planner = ClaudePlanner(cfg, client=client)
    assert (planner.name, planner.model) == ("claude", "claude-opus-5-5")

    result = planner.plan(card)
    assert (result.planner, result.model, result.fallback) == ("claude", "claude-opus-5-5", False)
    assert result.plan is GOOD_PLAN
    assert result.usage == {"input_tokens": 1200, "output_tokens": 300}
    assert result.raw_text is None
    assert result.prompt_sha256 == prompt_sha256(plan_prompt(card))

    (call,) = client.messages.calls
    assert call["model"] == "claude-opus-5-5" and call["max_tokens"] == MAX_TOKENS
    assert call["system"] == SYSTEM
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["output_format"] is Plan
    assert call["messages"] == [{"role": "user", "content": plan_prompt(card)}]
    assert "tool_choice" not in call  # no forced tool use, no prefill


def _assert_static_fallback(result: Any, card: DatasetCard) -> None:
    assert result.fallback is True and result.planner == "claude"
    assert result.plan == StaticPlanner().plan(card).plan
    assert result.prompt_sha256 == prompt_sha256(plan_prompt(card))


def test_refusal_falls_back(card: DatasetCard, caplog: pytest.LogCaptureFixture) -> None:
    client = FakeClient(response(GOOD_PLAN, stop_reason="refusal"))
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.claude_planner"):
        result = ClaudePlanner(PlannerConfig(), client=client).plan(card)
    _assert_static_fallback(result, card)
    assert result.usage == {"input_tokens": 1200, "output_tokens": 300}  # tokens were spent
    assert result.raw_text is None and result.model == "claude-opus-5"
    assert "refusal" in caplog.text


def test_no_parsed_plan_keeps_raw_text(card: DatasetCard, caplog: pytest.LogCaptureFixture) -> None:
    blocks = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text="I cannot plan this."),
    ]
    client = FakeClient(response(None, content=blocks, usage=None))
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.claude_planner"):
        result = ClaudePlanner(PlannerConfig(), client=client).plan(card)
    _assert_static_fallback(result, card)
    assert result.raw_text == "I cannot plan this."
    assert result.usage == {"input_tokens": None, "output_tokens": None}
    assert "no parsed plan" in caplog.text
    # A parsed output of the wrong type counts as no plan as well.
    other = ClaudePlanner(
        PlannerConfig(), client=FakeClient(response({"ontologies": ["envo"]}))
    ).plan(card)
    assert other.fallback is True


@pytest.mark.parametrize(
    "error",
    [TimeoutError("read timed out"), RuntimeError("SDK error"), ValueError("bad schema")],
    ids=["timeout", "runtime", "value"],
)
def test_sdk_errors_fall_back(
    card: DatasetCard, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.claude_planner"):
        result = ClaudePlanner(PlannerConfig(), client=FakeClient(error=error)).plan(card)
    _assert_static_fallback(result, card)
    assert result.usage == {} and result.raw_text is None
    assert f"claude planner failed ({type(error).__name__}: {error})" in caplog.text


def test_missing_sdk_falls_back(
    monkeypatch: pytest.MonkeyPatch, card: DatasetCard, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the `claude` extra the import fails inside plan(): a fallback, not a crash."""
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise
    planner = ClaudePlanner(PlannerConfig())
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.claude_planner"):
        result = planner.plan(card)
    _assert_static_fallback(result, card)
    assert "ModuleNotFoundError" in caplog.text or "ImportError" in caplog.text
    assert planner._client is None  # nothing cached; a later install would be picked up


def test_lazy_sdk_client_gets_the_timeout(
    monkeypatch: pytest.MonkeyPatch, card: DatasetCard
) -> None:
    """The SDK is imported on first use and built with the planner timeout only: credentials
    are the SDK's business (ANTHROPIC_API_KEY / `ant auth`), never this package's config."""
    made: list[dict[str, Any]] = []
    fake_client = FakeClient(response(GOOD_PLAN))

    def anthropic_ctor(**kwargs: Any) -> FakeClient:
        made.append(kwargs)
        return fake_client

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = anthropic_ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-read-by-mesa-clm")

    planner = ClaudePlanner(PlannerConfig(timeout=33.0))
    assert planner._client is None
    result = planner.plan(card)
    assert not result.fallback and result.plan is GOOD_PLAN
    assert made == [{"timeout": 33.0}]
    assert planner._client is fake_client
    planner.plan(card)
    assert len(made) == 1  # the client is built once and reused


def test_make_planner_claude_kind() -> None:
    cfg = Config.model_validate({"planner": {"kind": "claude", "claude_model": "claude-opus-5-5"}})
    planner = make_planner(cfg)
    assert isinstance(planner, ClaudePlanner) and planner.model == "claude-opus-5-5"
    assert planner.cfg is cfg.planner
    assert isinstance(make_planner(Config(), "claude"), ClaudePlanner)
