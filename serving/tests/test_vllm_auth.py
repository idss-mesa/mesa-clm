"""``serving/vllm_auth.py``: the bearer guard the encoder container loads with ``--middleware``
(DESIGN A4). The app below has the shapes of vLLM 0.27.1's pooling server (plain routes, a
FastAPI route outside ``/v1``, a mounted ``/metrics`` sub-application, CORS, the docs routes)
and installs the guard the way vLLM's ``build_app`` does: resolve ``vllm_auth.require_api_key``
by name, check ``inspect.iscoroutinefunction``, pass it to ``app.middleware("http")``."""

# ruff: noqa: S101

from __future__ import annotations

import hmac
import importlib
import inspect
import re
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

vllm_auth: ModuleType = pytest.importorskip("vllm_auth")  # needs starlette
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

KEY = "k" * 43
OTHER = "o" * 43
MIDDLEWARE = "vllm_auth.require_api_key"  # the --middleware value of mesa-clm-encoder-run

# (method, path) of the routes the stub app serves; every one but /health needs the key.
ROUTES = [
    ("GET", "/v1/models"),
    ("POST", "/v1/embeddings"),
    ("POST", "/pooling"),
    ("POST", "/invocations"),
    ("POST", "/score"),
    ("POST", "/rerank"),
    ("POST", "/tokenize"),
    ("POST", "/detokenize"),
    ("GET", "/version"),
    ("GET", "/load"),
    ("GET", "/ping"),
    ("GET", "/metrics"),
    ("GET", "/metrics/anything"),
    ("GET", "/openapi.json"),
    ("GET", "/docs"),
]


def _app(guard: Any) -> Any:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.routing import Mount

    app = FastAPI()

    # response_model=None: the handlers' annotations name a locally imported class, which
    # FastAPI would otherwise try to resolve as a response model.
    @app.get("/health", response_model=None)
    def health() -> JSONResponse:
        return JSONResponse({}, status_code=200)

    def ok() -> JSONResponse:
        return JSONResponse({"ok": True})

    for method, path in ROUTES:
        if path.startswith(("/metrics", "/openapi", "/docs")):
            continue
        app.add_api_route(path, ok, methods=[method], response_model=None)

    async def metrics_app(scope: Any, receive: Any, send: Any) -> None:
        await JSONResponse({"metrics": True})(scope, receive, send)

    mount = Mount("/metrics", metrics_app)
    mount.path_regex = re.compile("^/metrics(?P<path>.*)$")  # vLLM's 307 workaround
    app.routes.append(mount)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    if not inspect.iscoroutinefunction(guard):  # vLLM's own test before app.middleware("http")
        raise AssertionError("the guard is not a coroutine function")
    app.middleware("http")(guard)
    return app


def _client(app: Any) -> Any:
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def from_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """``vllm_auth.require_api_key`` resolved as vLLM resolves ``--middleware``, with the key in
    ``VLLM_API_KEY``; the cached guard is dropped afterwards."""
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    monkeypatch.delitem(vllm_auth.__dict__, "require_api_key", raising=False)
    module_path, object_name = MIDDLEWARE.rsplit(".", 1)
    guard = getattr(importlib.import_module(module_path), object_name)
    yield guard
    vllm_auth.__dict__.pop("require_api_key", None)


@pytest.fixture
def client(from_env: Any) -> Any:
    return _client(_app(from_env))


def _call(client: Any, method: str, path: str, auth: str | None = None) -> Any:
    headers = {"Authorization": auth} if auth is not None else {}
    return client.request(method, path, headers=headers, json={} if method == "POST" else None)


# -- the route matrix ----------------------------------------------------------------------------


def test_health_is_the_only_open_path(client: Any) -> None:
    assert client.get("/health").status_code == 200
    for method, path in ROUTES:
        r = _call(client, method, path)
        assert r.status_code == 401, (method, path)
        assert r.json() == {"error": "Unauthorized"}, (method, path)


@pytest.mark.parametrize(
    "auth",
    [
        f"Bearer {OTHER}",
        f"Bearer {KEY}x",
        f"Bearer {KEY[:-1]}",
        f"Basic {KEY}",
        "Bearer ",
        "Bearer",
        KEY,
        "",
    ],
)
def test_wrong_credentials_are_refused_everywhere(client: Any, auth: str) -> None:
    for method, path in ROUTES:
        assert _call(client, method, path, auth).status_code == 401, (method, path, auth[:8])


def test_the_key_opens_every_route(client: Any) -> None:
    for method, path in ROUTES:
        if path.startswith(("/openapi", "/docs")):
            continue  # FastAPI's own routes: present here, absent with --disable-fastapi-docs
        r = _call(client, method, path, f"Bearer {KEY}")
        assert r.status_code == 200, (method, path)
    assert _call(client, "GET", "/v1/models", f"bearer {KEY}").status_code == 200
    assert _call(client, "GET", "/openapi.json", f"Bearer {KEY}").status_code == 200


def test_unknown_paths_and_lookalikes_need_the_key(client: Any) -> None:
    for path in ("/nonexistent", "/health/", "/healthz", "/v3/anything", "/HEALTH"):
        assert client.get(path).status_code == 401, path
    assert client.get("/nonexistent", headers={"Authorization": f"Bearer {KEY}"}).status_code == 404


def test_preflight_and_head_need_the_key(client: Any) -> None:
    preflight = {"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
    assert client.options("/v1/embeddings", headers=preflight).status_code == 401
    assert client.head("/v1/models").status_code == 401
    assert client.head("/health").status_code in (200, 405)  # /health stays reachable


def test_compares_in_constant_time(from_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bytes] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append(b)
        return bool(real(a, b))

    monkeypatch.setattr(vllm_auth.hmac, "compare_digest", spy)
    client = _client(_app(from_env))
    assert _call(client, "GET", "/v1/models", f"Bearer {OTHER}").status_code == 401
    assert _call(client, "GET", "/v1/models", f"Bearer {KEY}").status_code == 200
    assert calls == [KEY.encode()] * 2


# -- construction ----------------------------------------------------------------------------------


def test_resolution_needs_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(vllm_auth.__dict__, "require_api_key", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_API_KEY is empty or unset"):
        _ = vllm_auth.require_api_key
    monkeypatch.setenv("VLLM_API_KEY", "")
    with pytest.raises(RuntimeError, match="refusing"):
        _ = vllm_auth.require_api_key
    with pytest.raises(AttributeError, match="no attribute 'other'"):
        _ = vllm_auth.other
    with pytest.raises(RuntimeError):
        vllm_auth.make_middleware("")


def test_resolution_is_cached_and_never_echoes_the_key(from_env: Any) -> None:
    assert vllm_auth.require_api_key is from_env
    assert inspect.iscoroutinefunction(from_env)
    assert KEY not in repr(from_env) and KEY not in str(vars(vllm_auth).get("__doc__"))


def test_helpers() -> None:
    key = KEY.encode()
    assert vllm_auth.bearer_ok(f"Bearer {KEY}", key)
    assert vllm_auth.bearer_ok(f"BEARER {KEY}", key)
    assert not vllm_auth.bearer_ok(None, key)
    assert not vllm_auth.bearer_ok(f"Bearer  {KEY}", key)
    assert vllm_auth.route_path({"path": "/health"}) == "/health"
    assert vllm_auth.route_path({"path": "/x/health", "root_path": "/x"}) == "/health"
    assert vllm_auth.route_path({"path": "/xhealth", "root_path": "/x"}) == "health"
    assert vllm_auth.route_path({}) == ""
    assert sorted(vllm_auth.OPEN_PATHS) == ["/health"]
