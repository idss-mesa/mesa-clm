"""``ClmHttpClient`` over the fake transport: typed answers, the latency header, ``/v1/rank``,
``/v1/models``, ``/health``, retries with backoff on the transient statuses, the circuit breaker,
``ClmError`` semantics (no retry on 401/422, no secret in messages), the loopback gate and
``from_config``. No network."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from mesa_clm.clm.http import (
    RETRY_STATUSES,
    Choice,
    ChoiceAnswer,
    ClmError,
    ClmHttpClient,
    HttpEndpoint,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from mesa_clm.config import ClmConfig
from mesa_clm.net import BreakerOpenError, CircuitBreaker, EndpointError
from mesa_clm.registry import ANCHOR_KEY, ANCHORS
from tests.fakes.clm_transport import CLM_URL, FakeClmServer

KEY = "test-key-not-a-secret"
STATE = {
    "card": {"dataset": "d"},
    "column": {"name": "observerDistance", "description": "Radial distance"},
}
FIT = Choice(
    criteria={
        "PATO:0000040": "distance: a 1-D extent",
        "PATO:0000122": "length",
        ANCHOR_KEY: ANCHORS["term"],
    }
)


@pytest.fixture
def server() -> FakeClmServer:
    return FakeClmServer(clm_api_key=KEY, latency_ms=7.5)


def client(server: FakeClmServer, **kw: Any) -> ClmHttpClient:
    kw.setdefault("sleep", lambda _s: None)
    return ClmHttpClient(CLM_URL, KEY, transport=server.transport(), **kw)


# -- system_one -----------------------------------------------------------------------------------


def test_system_one_typed_answers_and_latency(server: FakeClmServer) -> None:
    with client(server) as c:
        r = c.system_one(
            STATE,
            {
                "fit": FIT,
                "n": Noul(instructions="Is it a measurement?"),
                "s": Score(criteria=["a", "b"], instructions="how"),
            },
        )
    assert isinstance(r, SystemOneResponse) and r.model == "clm-latest"
    fit = r.answers["fit"]
    assert isinstance(fit, ChoiceAnswer) and fit.choice in FIT.criteria
    assert set(fit.probabilities) == set(FIT.criteria)
    assert sum(fit.probabilities.values()) == pytest.approx(1.0) and 0.0 <= fit.confidence <= 1.0
    n = r.answers["n"]
    assert isinstance(n, NoulAnswer) and 0.0 < n.noul < 1.0
    assert n.probabilities == {"false": 1.0 - n.noul, "true": n.noul}
    s = r.answers["s"]
    assert isinstance(s, ScoreAnswer) and 0.0 <= s.score <= 1.0 and s.legend == {"0": "a", "1": "b"}
    assert r.usage == Usage(billing_units=3, input_tokens=r.usage.input_tokens, output_tokens=0)
    assert r.usage.input_tokens is not None and r.usage.input_tokens > 0
    assert r.latency_ms == 7.5
    body = server.bodies[-1]
    assert list(body) == ["state", "model", "questions"] and body["model"] == "clm-latest"
    assert server.requests[-1].headers["Authorization"] == f"Bearer {KEY}"
    assert c.calls == 1


def test_model_and_temperature_overrides(server: FakeClmServer) -> None:
    c = client(server, model="clm-raw")
    r = c.system_one(STATE, {"fit": FIT})
    assert r.model == "clm-raw" and server.bodies[-1]["model"] == "clm-raw"
    r2 = c.system_one(STATE, {"fit": FIT}, model="clm-latest", temperature=3.0)
    assert r2.model == "clm-latest"
    assert list(server.bodies[-1]) == ["state", "model", "questions", "temperature"]
    assert server.bodies[-1]["temperature"] == 3.0
    # Flatter at T=3: the top probability drops.
    r1 = c.system_one(STATE, {"fit": FIT}, model="clm-latest")
    top1 = max(r1.answers["fit"].probabilities.values())  # type: ignore[union-attr]
    top3 = max(r2.answers["fit"].probabilities.values())  # type: ignore[union-attr]
    assert top3 < top1


def test_wire_dict_questions_pass_through(server: FakeClmServer) -> None:
    q = {"type": "choice", "instructions": "pick", "criteria": {"a": "alpha", "b": "beta"}}
    r = client(server).system_one("state text", {"q": q})
    assert server.bodies[-1]["questions"]["q"] == q and server.bodies[-1]["state"] == "state text"
    assert isinstance(r.answers["q"], ChoiceAnswer)


def test_missing_latency_header_and_null_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "clm-latest",
                "answers": {"n": {"type": "noul", "noul": 0.25}},
                "usage": {"billing_units": None},
            },
        )

    c = ClmHttpClient(CLM_URL, None, transport=httpx.MockTransport(handler))
    r = c.system_one("s", {"n": Noul("q")})
    assert r.latency_ms is None and r.usage.billing_units == 0 and r.usage.input_tokens is None
    assert "Authorization" not in c.endpoint._client.headers


def test_malformed_and_drifted_responses_are_errors() -> None:
    payloads: list[Any] = [
        {
            "model": "m",
            "answers": {"n": {"type": "noul", "noul": 0.2, "extra": 1}},
        },  # drift: unknown field
        {"model": "m", "answers": {"n": {"type": "bogus"}}},
        {
            "model": "m",
            "answers": {"n": {"type": "choice", "choice": "a"}},
        },  # missing probabilities
        {"model": "m", "answers": {}, "usage": {}, "surprise": True},
        "not an object",
    ]
    for payload in payloads:
        c = ClmHttpClient(
            CLM_URL,
            None,
            transport=httpx.MockTransport(lambda _r, p=payload: httpx.Response(200, json=p)),
        )
        with pytest.raises(ClmError, match="malformed") as info:
            c.system_one("s", {"n": Noul("q")})
        assert info.value.status == 200
    c = ClmHttpClient(
        CLM_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"<html>")),
    )
    with pytest.raises(ClmError, match="non-JSON"):
        c.system_one("s", {"n": Noul("q")})


# -- rank / models / health -----------------------------------------------------------------------


def test_rank(server: FakeClmServer) -> None:
    c = client(server)
    ranked = c.rank(STATE, None, ["distance", "length", "colour"])
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert {r.candidate for r in ranked} == {"distance", "length", "colour"}
    assert ranked[0].prob >= ranked[1].prob >= ranked[2].prob
    assert sum(r.prob for r in ranked) == pytest.approx(1.0)
    body = server.bodies[-1]
    assert list(body) == ["context", "question", "answers", "model"] and body["question"] is None
    c.rank("ctx", "which unit?", ("m", "s"), model="clm-raw", temperature=0.5)
    assert (
        server.bodies[-1]["question"] == "which unit?" and server.bodies[-1]["temperature"] == 0.5
    )
    assert server.bodies[-1]["answers"] == ["m", "s"] and server.bodies[-1]["model"] == "clm-raw"
    with pytest.raises(ClmError) as info:
        c.rank("ctx", None, [])
    assert info.value.status == 422
    bad = ClmHttpClient(
        CLM_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"model": "m"})),
    )
    with pytest.raises(ClmError, match="malformed /v1/rank"):
        bad.rank("ctx", None, ["a"])


def test_models_and_health(server: FakeClmServer) -> None:
    c = client(server)
    names = [m["name"] for m in c.models()]
    assert names[0] == "clm-latest" and names[-1] == "clm-raw"
    assert c.health() is True
    assert server.paths[-1] == "/health" and "Authorization" in server.requests[-1].headers
    # Health never raises: a down server, a 500, a non-JSON body are all False.
    server.fail_next = [httpx.ConnectError("refused")]
    assert c.health() is False
    server.fail_next = [500]
    assert c.health() is False
    c2 = ClmHttpClient(
        CLM_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"ok": False})),
    )
    assert c2.health() is False
    c3 = ClmHttpClient(
        CLM_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"models": "x"})),
    )
    with pytest.raises(ClmError, match="malformed /v1/models"):
        c3.models()
    c4 = ClmHttpClient(
        CLM_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"nope": 1})),
    )
    with pytest.raises(ClmError, match="malformed /v1/models"):
        c4.models()


# -- errors, retries, breaker ---------------------------------------------------------------------


def test_401_and_422_raise_without_retry(server: FakeClmServer) -> None:
    wrong = ClmHttpClient(CLM_URL, "wrong-key", transport=server.transport(), sleep=lambda _s: None)
    with pytest.raises(ClmError) as info:
        wrong.system_one(STATE, {"fit": FIT})
    assert info.value.status == 401 and info.value.message == "invalid API key"
    assert str(info.value) == "401: invalid API key"
    assert len(server.requests) == 1 and wrong.breaker.consecutive_failures == 0
    c = client(server)
    with pytest.raises(ClmError) as info2:
        c.system_one(STATE, {"bad": {"type": "score", "criteria": ["only one"]}})
    assert info2.value.status == 422 and "score" in info2.value.message
    with pytest.raises(ClmError) as info3:
        c.system_one(STATE, {"fit": FIT}, model="no-such-head")
    assert info3.value.status == 422 and "no-such-head" in info3.value.message
    assert len(server.requests) == 3 and c.breaker.consecutive_failures == 0


def test_retries_transient_statuses_with_backoff(server: FakeClmServer) -> None:
    slept: list[float] = []
    c = ClmHttpClient(
        CLM_URL, KEY, transport=server.transport(), retries=4, backoff=0.1, sleep=slept.append
    )
    server.fail_next = [503, 429, httpx.ReadTimeout("slow")]
    r = c.system_one(STATE, {"fit": FIT})
    assert isinstance(r.answers["fit"], ChoiceAnswer)
    assert len(server.requests) == 4 and slept == [0.1, 0.2, 0.4]
    assert c.breaker.consecutive_failures == 0  # success resets
    assert c.calls == 1


def test_retry_after_header_is_honoured_and_capped() -> None:
    slept: list[float] = []
    replies = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}, json={"detail": "slow down"}),
            httpx.Response(503, headers={"Retry-After": "9999"}, json={"detail": "busy"}),
            httpx.Response(
                502, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, json={"detail": "gw"}
            ),
            httpx.Response(200, json={"models": []}),
        ]
    )
    c = ClmHttpClient(
        CLM_URL,
        None,
        transport=httpx.MockTransport(lambda _r: next(replies)),
        retries=4,
        backoff=0.5,
        sleep=slept.append,
    )
    assert c.models() == []
    assert slept == [
        2.0,
        30.0,
        2.0,
    ]  # max(backoff, hint), capped at 30 s; a date falls back to backoff


def test_exhausted_retries_raise_last_status(server: FakeClmServer) -> None:
    c = client(server, retries=2)
    server.fail_next = [500, 502]
    with pytest.raises(ClmError) as info:
        c.system_one(STATE, {"fit": FIT})
    assert info.value.status == 502 and "after 2 attempt(s)" in info.value.message
    assert "scripted failure 502" in info.value.message
    server.fail_next = [httpx.ConnectError("refused"), httpx.ConnectError("refused")]
    with pytest.raises(ClmError) as info2:
        c.system_one(STATE, {"fit": FIT})
    assert info2.value.status == 0 and "unreachable" in info2.value.message
    assert "127.0.0.1:8700" in info2.value.message
    assert {429, 500, 502, 503, 504, 529} == RETRY_STATUSES


def test_breaker_opens_and_recovers(server: FakeClmServer) -> None:
    now = [1000.0]
    breaker = CircuitBreaker(failures=2, open_for=60.0, name="CLM server", clock=lambda: now[0])
    c = client(server, retries=1, breaker=breaker)
    server.fail_next = [503, 504]
    for _ in range(2):
        with pytest.raises(ClmError):
            c.system_one(STATE, {"fit": FIT})
    assert breaker.is_open
    with pytest.raises(BreakerOpenError, match="circuit breaker is open"):
        c.system_one(STATE, {"fit": FIT})
    assert len(server.requests) == 2  # the open breaker sent nothing
    now[0] += 61.0
    assert isinstance(c.system_one(STATE, {"fit": FIT}).answers["fit"], ChoiceAnswer)
    assert not breaker.is_open


def test_no_secret_in_errors_or_logs(
    server: FakeClmServer, caplog: pytest.LogCaptureFixture
) -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"detail": f"bad request with {request.headers['Authorization']}"}
        )

    c = ClmHttpClient(CLM_URL, KEY, transport=httpx.MockTransport(echo))
    with (
        caplog.at_level(logging.DEBUG, logger="mesa_clm.clm.http"),
        pytest.raises(ClmError) as info,
    ):
        c.system_one(STATE, {"fit": FIT})
    assert KEY not in str(info.value) and "<redacted>" in info.value.message
    assert KEY not in caplog.text and "Bearer" not in caplog.text
    assert any("/v1/systemone" in rec.message for rec in caplog.records)
    # A key echoed by a transport error is redacted too.
    c2 = ClmHttpClient(
        CLM_URL,
        KEY,
        transport=httpx.MockTransport(
            lambda _r: (_ for _ in ()).throw(httpx.ConnectError(f"boom {KEY}"))
        ),
        retries=1,
    )
    with pytest.raises(ClmError) as info2:
        c2.models()
    assert KEY not in str(info2.value)
    # Detail extraction: FastAPI detail, vLLM error object, plain text.
    ep = HttpEndpoint(
        CLM_URL,
        KEY,
        what="x",
        timeout=1.0,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200)),
    )
    assert ep.detail(httpx.Response(400, json={"error": {"message": "m"}})) == "m"
    assert ep.detail(httpx.Response(400, json={"error": "plain"})) == "plain"
    assert ep.detail(httpx.Response(400, json={"other": 1})) == '{"other":1}'
    assert ep.detail(httpx.Response(400, text="x" * 400)) == "x" * 300


# -- construction ---------------------------------------------------------------------------------


def test_loopback_gate_and_remote_rules() -> None:
    with pytest.raises(EndpointError, match="loopback"):
        ClmHttpClient("http://clm.example.org:8700", KEY)
    with pytest.raises(EndpointError, match="https"):
        ClmHttpClient("http://clm.example.org:8700", KEY, allow_remote=True)
    with pytest.raises(EndpointError, match="/v1"):
        ClmHttpClient("http://127.0.0.1:8700/v1", KEY)
    remote = ClmHttpClient(
        "https://clm.example.org/",
        KEY,
        allow_remote=True,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"ok": True})),
    )
    assert remote.base_url == "https://clm.example.org" and remote.health()
    assert ClmHttpClient(CLM_URL + "/", KEY).base_url == CLM_URL
    with pytest.raises(ValueError, match="retries"):
        ClmHttpClient(CLM_URL, KEY, retries=0)


def test_client_is_proxy_proof_and_from_config(
    server: FakeClmServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    cfg = ClmConfig(base_url=CLM_URL, api_key=SecretStr(KEY), model="clm-raw", timeout=42.0)
    c = ClmHttpClient.from_config(cfg, transport=server.transport())
    assert c.model == "clm-raw" and c.endpoint.timeout == 42.0
    assert c.endpoint._client.trust_env is False
    assert c.health() and server.requests[-1].headers["Authorization"] == f"Bearer {KEY}"
    remote_cfg = ClmConfig(base_url="https://clm.example.org", allow_remote=True)
    assert ClmHttpClient.from_config(remote_cfg).base_url == "https://clm.example.org"
    with pytest.raises(EndpointError):
        ClmHttpClient.from_config(ClmConfig(base_url="https://clm.example.org"))
