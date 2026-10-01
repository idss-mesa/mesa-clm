"""A bearer-key guard for every route of the vLLM encoder container except ``/health``.

vLLM 0.27.1's own API-key check (``VLLM_API_KEY``; ``AuthenticationMiddleware`` in
``vllm/entrypoints/serve/utils/server_utils.py``) guards only the ``/v1``, ``/v2``, ``/inference``
and ``/cohere`` prefixes. Under ``--runner pooling`` the same server also answers ``POST
/pooling``, ``/invocations``, ``/score``, ``/rerank``, ``/detokenize``, ``/tokenize`` and ``GET
/metrics``, ``/version``, ``/load``, ``/ping`` without a key
(``bench/results/2026-09-29/serving_m1.json``), so any account on the serving host could have
embedded or scored text through loopback. This module closes that gap (DESIGN A4):

* the container loads it with ``--middleware vllm_auth.require_api_key`` (the module is mounted
  read-only at ``/opt/mesa-clm-auth`` and put on ``PYTHONPATH``; ``deploy/bin/mesa-clm-encoder-run``).
  vLLM's ``build_app`` hands an async function to FastAPI's ``app.middleware("http")``, after
  its own middleware, so this check is the outermost layer: it runs before routing, CORS,
  vLLM's prefix check and every handler, including mounted sub-applications (``/metrics``) and
  paths no route matches;
* every HTTP request must carry ``Authorization: Bearer <VLLM_API_KEY>`` (scheme
  case-insensitive, as vLLM's own check; the token compared with :func:`hmac.compare_digest`),
  otherwise the answer is ``401 {"error": "Unauthorized"}``. ``OPTIONS`` is no exception (no
  browser client is served);
* the one exemption is a request whose route path is exactly ``/health``: the readiness wait of
  ``mesa-clm-serve.service`` (``mesa-clm-wait-http``) and the doctor probe it without a key, and
  it answers only 200/503.

The function form covers HTTP only. The pooling runner registers no WebSocket route (the route
list is enumerated by ``scripts/serving_probes.py``; DESIGN A4), and a WebSocket handshake to an
unknown path is refused by the router anyway.

It runs inside the container (Python 3.12) with the standard library and starlette only. The key
is read from ``VLLM_API_KEY`` (``--env-file ~/.mesa/clm/secrets/encoder.env``) when vLLM resolves
``require_api_key`` at startup; when the variable is empty or unset the lookup raises and vLLM
refuses to start instead of serving unguarded. The key lives only in the guard's closure: it is
never logged, echoed, hashed or written.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

KEY_ENV: Final[str] = "VLLM_API_KEY"
# The attribute vLLM's ``--middleware vllm_auth.require_api_key`` resolves (module __getattr__).
MIDDLEWARE_NAME: Final[str] = "require_api_key"
OPEN_PATHS: Final[frozenset[str]] = frozenset({"/health"})
UNAUTHORIZED_BODY: Final[dict[str, str]] = {"error": "Unauthorized"}

CallNext = Callable[[Request], Awaitable[Response]]
Middleware = Callable[[Request, CallNext], Awaitable[Response]]


def route_path(scope: Mapping[str, Any]) -> str:
    """The path the router matches: ``scope["path"]`` without the ``root_path`` prefix, computed
    as vLLM's own ``AuthenticationMiddleware`` does (``root_path`` is empty in mesa-clm's
    recipe; a path that does not carry it is checked as it is, which can only deny)."""
    path = str(scope.get("path") or "")
    root = str(scope.get("root_path") or "")
    return path.removeprefix(root) if root else path


def bearer_ok(authorization: str | None, key: bytes) -> bool:
    """``Authorization: Bearer <key>``: scheme compared case-insensitively, the token in
    constant time. A missing header, another scheme or an empty token is ``False``."""
    if not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), key)


def make_middleware(key: str) -> Middleware:
    """The guard for ``key`` as an ``async (request, call_next)`` function, the shape FastAPI's
    ``app.middleware("http")`` takes (and ``inspect.iscoroutinefunction`` accepts, which is
    vLLM's test for a function middleware). An empty key is refused."""
    if not key:
        raise RuntimeError(
            f"vllm_auth: {KEY_ENV} is empty or unset; refusing to build the bearer guard "
            "(the encoder must not serve without a key)"
        )
    expected = key.encode("utf-8")

    async def require_api_key(request: Request, call_next: CallNext) -> Response:
        if route_path(request.scope) in OPEN_PATHS or bearer_ok(
            request.headers.get("authorization"), expected
        ):
            return await call_next(request)
        return JSONResponse(UNAUTHORIZED_BODY, status_code=401)

    return require_api_key


def __getattr__(name: str) -> Middleware:
    """``vllm_auth.require_api_key``, built from ``VLLM_API_KEY`` on first lookup (PEP 562) and
    cached, so importing the module (tests, CI) needs no key while the container's lookup at
    startup fails loudly without one."""
    if name != MIDDLEWARE_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    guard = make_middleware(os.environ.get(KEY_ENV, ""))
    globals()[name] = guard
    return guard
