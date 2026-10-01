"""``serving/vllm_routes.py``: the route-table walk and the per-route guard probe that
``scripts/serving_probes.py`` runs inside the encoder container (DESIGN A3), exercised here on a
FastAPI app with the bearer guard installed as vLLM installs it (no vllm needed)."""

# ruff: noqa: S101

from __future__ import annotations

import re
from types import ModuleType
from typing import Any

import pytest

vllm_auth: ModuleType = pytest.importorskip("vllm_auth")
vllm_routes: ModuleType = pytest.importorskip("vllm_routes")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

KEY = "k" * 43


def _app(guarded: bool) -> Any:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from starlette.routing import Mount

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    def ok() -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_api_route("/health", ok, methods=["GET"], response_model=None)
    app.add_api_route("/v1/models", ok, methods=["GET"], response_model=None)
    app.add_api_route("/pooling", ok, methods=["POST"], response_model=None)
    app.add_api_route("/ping", ok, methods=["GET", "POST"], response_model=None)

    async def metrics(scope: Any, receive: Any, send: Any) -> None:
        await JSONResponse({})(scope, receive, send)

    mount = Mount("/metrics", metrics)
    mount.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(mount)
    if guarded:
        app.middleware("http")(vllm_auth.make_middleware(KEY))
    return app


def test_route_table_and_middleware_order() -> None:
    app = _app(guarded=True)
    table = vllm_routes.route_table(app)
    assert [(r["kind"], r["path"]) for r in table] == [
        ("APIRoute", "/health"),
        ("APIRoute", "/v1/models"),
        ("APIRoute", "/pooling"),
        ("APIRoute", "/ping"),
        ("Mount", "/metrics"),
    ]
    assert table[3]["methods"] == ["GET", "POST"] and not any(r["websocket"] for r in table)
    assert vllm_routes.middleware_order(app) == ["BaseHTTPMiddleware:require_api_key"]


def test_probe_guard_passes_with_the_guard_and_fails_without() -> None:
    open_paths = frozenset({"/health"})
    app = _app(guarded=True)
    results = vllm_routes.probe_guard(app, vllm_routes.route_table(app))
    assert [(r["method"], r["path"]) for r in results] == [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/pooling"),
        ("GET", "/ping"),
        ("POST", "/ping"),
        ("GET", "/metrics"),
    ]
    verdict = vllm_routes.summarize(results, open_paths)
    assert verdict == {
        "n_requests": 6,
        "n_guarded": 5,
        "unguarded": [],
        "open_paths_reachable": True,
        "pass": True,
    }
    bare = _app(guarded=False)
    leaky = vllm_routes.summarize(
        vllm_routes.probe_guard(bare, vllm_routes.route_table(bare)), open_paths
    )
    assert not leaky["pass"] and "POST /pooling" in leaky["unguarded"]
    assert "GET /metrics" in leaky["unguarded"]


def test_main_needs_one_json_argument(capsys: pytest.CaptureFixture[str]) -> None:
    assert vllm_routes.main([]) == 2
    assert "usage" in capsys.readouterr().err
