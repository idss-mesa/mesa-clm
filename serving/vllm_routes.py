"""The vLLM encoder's real route table and the bearer guard on every route (DESIGN A4).

Runs inside a throwaway copy of the pinned container: no model is loaded and the container has
no network. ``scripts/serving_probes.py`` starts it like this (the dummy key is not a secret; the
real key never enters this container)::

    sg docker -c "docker run --rm -i --gpus all --network none --user <uid>:<gid> -e HOME=/tmp \\
        -e VLLM_API_KEY=<dummy> -e PYTHONPATH=/opt/mesa-clm-auth \\
        -v ~/.mesa/clm/serve/vllm-auth:/opt/mesa-clm-auth:ro -v <repo>/serving:/work:ro \\
        --entrypoint python3 vllm/vllm-openai@sha256:<digest> /work/vllm_routes.py '<args json>'"

It parses the recipe's ``vllm serve`` arguments (``serving.lock.json`` ``recipe.args``) with
vLLM's own parser, builds the application with ``build_app`` for the tasks ``--runner pooling``
resolves for Qwen3-8B (``embed``, ``token_embed``; a stub model config whose pooling task is
``embed``), and prints one JSON object: every route (``APIRoute``, ``Route``, ``Mount``,
``WebSocketRoute``) with its methods, the middleware order (outermost first), and the status of
each route requested without a key and with a wrong key through starlette's ``TestClient``.
Every route but ``/health`` must answer 401 to both. ``--gpus all`` is there only because vLLM's
argument parser infers the device type from the platform.

The helpers below take any Starlette/FastAPI application, so the serve-side tests exercise them
without vllm.
"""

from __future__ import annotations

import json
import sys
import warnings
from typing import Any, Final

WRONG_KEY: Final[str] = "mesa-clm-route-probe-wrong-key"
# Bodies that pass FastAPI's parsing far enough to show the guard answered, not a 422.
POST_BODY: Final[dict[str, Any]] = {"model": "qwen3-8b", "input": "probe", "prompt": "probe"}


def route_table(app: Any) -> list[dict[str, Any]]:
    """Every route of ``app`` (and of mounted routers, one level down) as
    ``{kind, path, methods, websocket}``, in registration order."""
    out: list[dict[str, Any]] = []
    for route in app.routes:
        kind = type(route).__name__
        out.append(
            {
                "kind": kind,
                "path": str(getattr(route, "path", "")),
                "methods": sorted(getattr(route, "methods", None) or []),
                "websocket": kind == "WebSocketRoute",
            }
        )
    return out


def middleware_order(app: Any) -> list[str]:
    """``app.user_middleware`` outermost first; a function middleware is shown as
    ``BaseHTTPMiddleware:<function name>``."""
    names = []
    for item in app.user_middleware:
        name = getattr(item.cls, "__name__", repr(item.cls))
        dispatch = item.kwargs.get("dispatch")
        names.append(f"{name}:{dispatch.__name__}" if dispatch is not None else name)
    return names


def probe_guard(app: Any, routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each route's status without a key and with :data:`WRONG_KEY` (``TestClient``, no
    network). A mount is requested with ``GET``; a route with several methods once per method."""
    from starlette.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    results = []
    for r in routes:
        if r["websocket"]:
            results.append({**r, "no_key": None, "wrong_key": None})
            continue
        for method in r["methods"] or ["GET"]:
            if method == "HEAD":
                continue
            body = POST_BODY if method in ("POST", "PUT", "PATCH") else None
            statuses = {}
            for label, headers in (
                ("no_key", {}),
                ("wrong_key", {"Authorization": f"Bearer {WRONG_KEY}"}),
            ):
                resp = client.request(method, r["path"], headers=headers, json=body)
                statuses[label] = resp.status_code
            results.append({"kind": r["kind"], "method": method, "path": r["path"], **statuses})
    return results


def summarize(results: list[dict[str, Any]], open_paths: frozenset[str]) -> dict[str, Any]:
    """The verdict: every route outside ``open_paths`` answered 401 without a key and with a
    wrong one; the open paths did not answer 401."""
    guarded = [r for r in results if r["path"] not in open_paths and r.get("no_key") is not None]
    leaks = [
        f"{r['method']} {r['path']}" for r in guarded if r["no_key"] != 401 or r["wrong_key"] != 401
    ]
    opened = [r for r in results if r["path"] in open_paths]
    return {
        "n_requests": len(results),
        "n_guarded": len(guarded),
        "unguarded": leaks,
        "open_paths_reachable": all(r["no_key"] != 401 for r in opened),
        "pass": not leaks and bool(guarded),
    }


def build_vllm_app(argv: list[str]) -> Any:
    """vLLM's application for ``argv`` (the ``vllm serve`` arguments after the image)."""
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:  # older layouts
        from vllm.utils import FlexibleArgumentParser

    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)

    class StubModelConfig:
        """What the pooling and SageMaker routers read from the model config at build time."""

        io_processor_plugin = None
        hf_config = None

        def get_pooling_task(self, tasks: Any) -> str:
            return "embed"

    return build_app(args, supported_tasks=("embed", "token_embed"), model_config=StubModelConfig())


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: vllm_routes.py '<vllm serve arguments as a JSON list>'", file=sys.stderr)
        return 2
    vllm_args = [str(a) for a in json.loads(args[0])]
    app = build_vllm_app(vllm_args)
    routes = route_table(app)
    results = probe_guard(app, routes)
    payload = {
        "vllm_args": vllm_args,
        "routes": routes,
        "websocket_routes": [r["path"] for r in routes if r["websocket"]],
        "middleware_outermost_first": middleware_order(app),
        "probes": results,
        "verdict": summarize(results, frozenset({"/health"})),
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
