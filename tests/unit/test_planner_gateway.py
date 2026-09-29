"""The gateway planner over ``httpx.MockTransport``: the streamed request shape, the SSE parse,
nudges, the static fallback on every failure class, and the loopback gate. No network."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from mesa_clm.cards import DatasetCard
from mesa_clm.config import Config, PlannerConfig
from mesa_clm.net import EndpointError
from mesa_clm.planner import make_planner
from mesa_clm.planner.gateway_planner import (
    MAX_ATTEMPTS,
    MAX_TOKENS,
    NUDGE,
    SYSTEM,
    GatewayPlanner,
    plan_prompt,
    prompt_sha256,
)
from mesa_clm.planner.static_planner import StaticPlanner

GOOD_PLAN = {
    "ontologies": ["ENVO", "uo", "bogus", "envo"],
    "columns": {
        "observerDistance": {
            "annotate": True,
            "aspect": "measurement",
            "ontology": "PATO",
            "queries": ["distance", "radial distance"],
        },
        "uid": {"annotate": False},
    },
    "sites": {"HARV": {"environment_queries": ["temperate deciduous forest biome"]}},
    "taxon_queries": ["Aves"],
    "notes": "birds",
}


def sse(text: str, usage: dict[str, int] | None = None, *, pieces: int = 3) -> bytes:
    """An OpenAI-style SSE stream: the text split over several deltas, an optional usage chunk,
    a comment line and ``[DONE]``."""
    step = max(1, -(-len(text) // pieces))
    lines = [": keep-alive"]
    for i in range(0, len(text), step):
        chunk = {"choices": [{"delta": {"content": text[i : i + step]}}]}
        lines.append("data: " + json.dumps(chunk))
    if usage is not None:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


Reply = httpx.Response | Exception


class Gateway:
    """A scripted gateway: each request pops the next reply (a Response or an exception to
    raise) and records the parsed request body and headers."""

    def __init__(self, replies: Iterable[Reply]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def planner(self, **kw: Any) -> GatewayPlanner:
        kw.setdefault("api_key", "sk-test-key")
        return GatewayPlanner(
            "http://127.0.0.1:8000", transport=httpx.MockTransport(self), timeout=7.5, **kw
        )


def test_success_request_shape_and_parse(card: DatasetCard) -> None:
    text = "<think>plan...</think>```json\n" + json.dumps(GOOD_PLAN) + "\n```"
    gw = Gateway(
        [httpx.Response(200, content=sse(text, {"prompt_tokens": 900, "completion_tokens": 120}))]
    )
    result = gw.planner().plan(card)

    assert (result.planner, result.model, result.fallback) == ("gateway", "carc-tools", False)
    assert result.usage == {"prompt_tokens": 900, "completion_tokens": 120, "llm_calls": 1}
    assert result.raw_text == text
    assert result.prompt_sha256 == prompt_sha256(plan_prompt(card))
    plan = result.plan
    assert plan.ontologies == ["envo", "uo"]  # lower-cased, unknown dropped, deduplicated
    hint = plan.columns["observerDistance"]
    assert (hint.annotate, hint.aspect, hint.ontology) == (True, "measurement", "pato")
    assert hint.queries == ["distance", "radial distance"]
    assert plan.columns["uid"].annotate is False and plan.columns["uid"].queries == []
    assert plan.sites["HARV"].environment_queries == ["temperate deciduous forest biome"]
    assert plan.taxon_queries == ["Aves"] and plan.notes == "birds"

    (request,) = gw.requests
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:8000/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test-key"
    assert request.headers["Content-Type"] == "application/json"
    (body,) = gw.bodies
    assert body["model"] == "carc-tools" and body["temperature"] == 0.0
    assert body["max_tokens"] == MAX_TOKENS
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM
    prompt = body["messages"][1]["content"]
    assert prompt.startswith("Dataset: DP1.10003.001.brd_countdata\n")
    assert "- observerDistance | Radial distance" in prompt and "- envo: ENVO:" in prompt
    assert "- taxon: the organisms" in prompt


def test_think_flag_and_no_key(card: DatasetCard) -> None:
    gw = Gateway([httpx.Response(200, content=sse(json.dumps(GOOD_PLAN)))])
    result = gw.planner(api_key=None, think=True, model="carc-fast").plan(card)
    assert result.model == "carc-fast" and not result.fallback
    assert "Authorization" not in gw.requests[0].headers
    assert gw.bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert result.usage["prompt_tokens"] == 0  # no usage chunk: counted as zero, one call
    assert result.usage["llm_calls"] == 1


def test_nudge_after_prose_then_success(card: DatasetCard) -> None:
    gw = Gateway(
        [
            httpx.Response(
                200,
                content=sse(
                    "Sure! Here is my thinking about the dataset.",
                    {"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ),
            httpx.Response(
                200,
                content=sse(json.dumps(GOOD_PLAN), {"prompt_tokens": 20, "completion_tokens": 7}),
            ),
        ]
    )
    result = gw.planner().plan(card)
    assert not result.fallback and result.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "llm_calls": 2,
    }
    first, second = gw.bodies
    assert len(first["messages"]) == 2 and len(second["messages"]) == 4
    assert second["messages"][2] == {
        "role": "assistant",
        "content": "Sure! Here is my thinking about the dataset.",
    }
    assert second["messages"][3] == {"role": "user", "content": NUDGE}


def test_nudge_after_rejected_plan(card: DatasetCard, caplog: pytest.LogCaptureFixture) -> None:
    """JSON that is an object but not a Plan (ints where strings belong) is nudged, not raised."""
    gw = Gateway(
        [
            httpx.Response(200, content=sse(json.dumps({"ontologies": [1, 2]}))),
            httpx.Response(200, content=sse(json.dumps(GOOD_PLAN))),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.gateway_planner"):
        result = gw.planner().plan(card)
    assert not result.fallback and result.usage["llm_calls"] == 2
    assert "plan rejected" in caplog.text
    assert gw.bodies[1]["messages"][2]["content"] == json.dumps({"ontologies": [1, 2]})


def test_fallback_after_three_bad_answers(card: DatasetCard) -> None:
    replies = [httpx.Response(200, content=sse("", {"prompt_tokens": 1, "completion_tokens": 0}))]
    replies += [
        httpx.Response(
            200, content=sse("still no json", {"prompt_tokens": 1, "completion_tokens": 2})
        )
    ] * 2
    gw = Gateway(replies)
    result = gw.planner().plan(card)

    assert len(gw.requests) == MAX_ATTEMPTS == 3
    assert (result.planner, result.model, result.fallback) == ("gateway", "carc-tools", True)
    assert result.plan == StaticPlanner().plan(card).plan  # the static plan, verbatim
    assert result.raw_text == "still no json"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 4, "llm_calls": 3}
    assert result.prompt_sha256 == prompt_sha256(plan_prompt(card))
    # An empty first answer is nudged as "(no answer)".
    assert gw.bodies[1]["messages"][2] == {"role": "assistant", "content": "(no answer)"}


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, text="upstream exploded"),
        httpx.Response(401, json={"error": "invalid key"}),
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
        httpx.Response(200, content=b"data: {not json\n\ndata: [DONE]\n"),
        httpx.Response(200, content=b"data: [1, 2, 3]\n\ndata: [DONE]\n"),
    ],
    ids=["http500", "http401", "connect-error", "read-timeout", "bad-sse-json", "non-object-chunk"],
)
def test_fallback_on_transport_and_protocol_failures(
    card: DatasetCard, reply: Reply, caplog: pytest.LogCaptureFixture
) -> None:
    gw = Gateway([reply])
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.gateway_planner"):
        result = gw.planner().plan(card)
    assert result.fallback is True and result.planner == "gateway"
    assert result.plan == StaticPlanner().plan(card).plan
    assert result.raw_text is None and result.usage["llm_calls"] == 0
    assert len(gw.requests) == 1  # a hard failure is not nudged
    assert "falling back to static rules" in caplog.text
    assert "sk-test-key" not in caplog.text


def test_http_error_message_carries_status_and_body_prefix(
    card: DatasetCard, caplog: pytest.LogCaptureFixture
) -> None:
    gw = Gateway([httpx.Response(503, text="x" * 1000)])
    with caplog.at_level(logging.WARNING, logger="mesa_clm.planner.gateway_planner"):
        gw.planner().plan(card)
    assert "RuntimeError: HTTP 503" in caplog.text
    assert "x" * 300 in caplog.text and "x" * 301 not in caplog.text


def test_loopback_gate_and_client_settings() -> None:
    planner = GatewayPlanner("http://localhost:8000/", "k")
    assert planner.base_url == "http://localhost:8000"
    assert planner._client.base_url == httpx.URL("http://localhost:8000")
    assert planner._client.headers["Authorization"] == "Bearer k"
    assert planner._client.timeout == httpx.Timeout(600.0, connect=5.0)
    assert planner._client.trust_env is False
    with pytest.raises(EndpointError, match="planner gateway must be reached over loopback"):
        GatewayPlanner("http://192.0.2.10:8000", None)
    with pytest.raises(EndpointError, match="/v1"):
        GatewayPlanner("http://127.0.0.1:8000/v1", None)
    with pytest.raises(EndpointError, match="https"):
        GatewayPlanner("http://llm.example.org:8000", None, allow_remote=True)
    remote = GatewayPlanner("https://llm.example.org/", None, allow_remote=True)
    assert remote.base_url == "https://llm.example.org"


def test_injected_client_is_used_as_is(card: DatasetCard) -> None:
    gw = Gateway([httpx.Response(200, content=sse(json.dumps(GOOD_PLAN)))])
    client = httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(gw))
    result = GatewayPlanner(
        "http://127.0.0.1:8000", "ignored-when-client-given", client=client
    ).plan(card)
    assert not result.fallback
    assert "Authorization" not in gw.requests[0].headers  # the caller owns the client's headers


def test_from_config_resolves_the_secret(card: DatasetCard) -> None:
    cfg = PlannerConfig(
        gateway_base_url="http://127.0.0.1:18000/",
        gateway_api_key=SecretStr("sk-from-config"),
        gateway_model="carc-fast",
        timeout=12.0,
    )
    gw = Gateway([httpx.Response(200, content=sse(json.dumps(GOOD_PLAN)))])
    planner = GatewayPlanner.from_config(cfg, transport=httpx.MockTransport(gw))
    assert (planner.base_url, planner.model, planner.timeout) == (
        "http://127.0.0.1:18000",
        "carc-fast",
        12.0,
    )
    assert planner.plan(card).model == "carc-fast"
    assert gw.requests[0].headers["Authorization"] == "Bearer sk-from-config"
    assert str(gw.requests[0].url) == "http://127.0.0.1:18000/v1/chat/completions"
    # No key configured: no Authorization header at all (never "Bearer None").
    anonymous = GatewayPlanner.from_config(
        PlannerConfig(), transport=httpx.MockTransport(Gateway([]))
    )
    assert "Authorization" not in anonymous._client.headers


def test_make_planner_gateway_kind() -> None:
    cfg = Config.model_validate({"planner": {"kind": "gateway", "gateway_api_key": "k"}})
    planner = make_planner(cfg)
    assert isinstance(planner, GatewayPlanner) and planner.model == "carc-tools"
    assert isinstance(make_planner(Config(), "gateway"), GatewayPlanner)  # the flag wins
    remote = Config.model_validate(
        {"planner": {"gateway_base_url": "https://llm.example.org"}, "clm": {"allow_remote": True}}
    )
    assert isinstance(make_planner(remote, "gateway"), GatewayPlanner)
    with pytest.raises(EndpointError, match="loopback"):
        make_planner(
            Config.model_validate({"planner": {"gateway_base_url": "https://llm.example.org"}}),
            "gateway",
        )


def test_planner_never_reads_proxy_env(monkeypatch: pytest.MonkeyPatch, card: DatasetCard) -> None:
    """trust_env=False: an HTTP(S)_PROXY in the environment must not divert the request."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    gw = Gateway([httpx.Response(200, content=sse(json.dumps(GOOD_PLAN)))])
    assert not gw.planner().plan(card).fallback
    assert str(gw.requests[0].url).startswith("http://127.0.0.1:8000/")
