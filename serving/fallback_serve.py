#!/usr/bin/env python3
"""The in-process fallback for the vLLM encoder container (plan §6.6; DESIGN D16; K0 step 2).

One process holds Qwen3-8B (:class:`cuda_encoder.CudaEncoder`) and serves two loopback APIs:

* ``127.0.0.1:8700`` - CLM's own app (the patched ``clm.server.create_app``) over an in-process
  ``Engine(embedder=encoder, device="cpu", action_cache="512MiB")``: the heads run on CPU as in
  ``mesa-clm-serve.service`` (DESIGN D17) and ``Engine`` only ever calls ``embed`` and
  ``healthy`` on its embedder;
* ``127.0.0.1:8090`` - a vLLM-compatible subset for the encoder clients (``mesa_clm``'s
  ``EncoderClient``, CLM's ``Embedder``): ``POST /v1/embeddings`` (string, string-list or
  token-id inputs; ``encoding_format`` ``float`` or ``base64`` little-endian float32;
  ``truncate_prompt_tokens`` with ``-1`` meaning the model length; ``truncation_side`` ``left``
  or ``right``, default ``right`` as in vLLM, while every mesa-clm and patched CLM client sends
  ``left``), ``GET /v1/models``, ``POST /tokenize`` and ``GET /health``.

Keys: ``CLM_API_KEY`` guards :8700 (``create_app``'s own check, constant-time after patch 0006)
and the encoder key (``VLLM_API_KEY``, else ``CLM_EMB_API_KEY``; ``clm.env`` carries the latter)
guards every ``/v1/*`` route on :8090 with ``hmac.compare_digest``; ``/health`` and
``/tokenize`` stay open as in vLLM. The server refuses to start without both keys unless
``--allow-anonymous`` is given, and refuses a non-loopback bind without a key in any case.

The fallback never runs next to the vLLM container (stop both units first; they hold the same
ports). ``route: transformers`` enters ``encoder_fp`` (DESIGN D5), so nothing it produces is
shared with the vLLM route before a recorded parity run. Run it in the serve venv, e.g.::

    systemd-run --user --unit mesa-clm-fallback -p EnvironmentFile=$HOME/.mesa/clm/secrets/clm.env \
        ~/.mesa/clm/serve/.venv/bin/python <mesa-clm>/serving/fallback_serve.py

``clm``, torch and transformers are imported lazily, so the module (and the embeddings app with
a stub encoder) loads without them. There is deliberately no ``from __future__ import
annotations``: FastAPI resolves endpoint annotations from module globals, and the endpoints
below close over a locally imported ``Request``.
"""

import argparse
import asyncio
import base64
import hmac
import ipaddress
import os
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

DEFAULT_HOST: Final[str] = "127.0.0.1"
EMB_PORT: Final[int] = 8090
CLM_PORT: Final[int] = 8700
SERVED_MODEL: Final[str] = "qwen3-8b"
HEADS: Final[str] = "~/.mesa/clm/heads"
F32 = npt.NDArray[np.float32]


class EncoderLike(Protocol):
    """What the :8090 app needs from an encoder (``CudaEncoder`` or a test stub)."""

    model_id: str
    max_len: int

    def tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def encode(
        self, inputs: Sequence[str] | Sequence[Sequence[int]], *, truncate: int | None, side: Any
    ) -> tuple[F32, int]: ...

    def healthy(self) -> bool: ...


class RequestError(ValueError):
    """A malformed request: the handler answers 400 with the message."""


def is_loopback(host: str) -> bool:
    """``localhost`` or a loopback IP; anything else (``0.0.0.0``, hostnames) is reachable."""
    host = host.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def bearer_ok(authorization: str | None, key: str | None) -> bool:
    """Constant-time check of ``Authorization: Bearer <key>``; no key configured means open."""
    if not key:
        return True
    return hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode())


def parse_inputs(value: Any) -> list[str] | list[list[int]]:
    """vLLM's accepted ``input`` shapes: a string, strings, one token list, token lists."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value:
        if all(isinstance(v, str) for v in value):
            return [str(v) for v in value]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return [[int(v) for v in value]]
        if all(
            isinstance(v, list)
            and v
            and all(isinstance(t, int) and not isinstance(t, bool) for t in v)
            for v in value
        ):
            return [[int(t) for t in v] for v in value]
    raise RequestError("input must be a string, a list of strings, or token-id lists")


def parse_truncation(body: dict[str, Any], max_len: int) -> tuple[int | None, str]:
    """``(limit or None, side)`` from ``truncate_prompt_tokens`` and ``truncation_side``."""
    raw = body.get("truncate_prompt_tokens")
    limit: int | None
    if raw is None:
        limit = None
    elif isinstance(raw, int) and not isinstance(raw, bool):
        limit = max_len if raw == -1 else raw
        if not 0 < limit <= max_len:
            raise RequestError(f"truncate_prompt_tokens must be -1 or in 1..{max_len}")
    else:
        raise RequestError("truncate_prompt_tokens must be an integer")
    side = body.get("truncation_side") or "right"
    if side not in ("left", "right"):
        raise RequestError("truncation_side must be 'left' or 'right'")
    return limit, str(side)


def encode_vector(vec: F32, fmt: str) -> str | list[float]:
    if fmt == "base64":
        return base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode("ascii")
    return [float(x) for x in vec]


def build_embeddings_app(
    encoder: EncoderLike, *, api_key: str | None, served_model: str = SERVED_MODEL
) -> Any:
    """The :8090 FastAPI app over ``encoder`` (module docstring)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="mesa-clm fallback encoder", version="1")

    def error(status: int, message: str) -> JSONResponse:
        kind = {400: "BadRequestError", 401: "Unauthorized", 404: "NotFoundError"}.get(
            status, "Error"
        )
        return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status)

    @app.middleware("http")
    async def guard_v1(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/v1") and not bearer_ok(
            request.headers.get("authorization"), api_key
        ):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health() -> Response:
        return Response(status_code=200 if encoder.healthy() else 503)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mesa-clm-fallback",
                    "root": encoder.model_id,
                    "max_model_len": encoder.max_len,
                }
            ],
        }

    async def json_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except ValueError:
            raise RequestError("body is not JSON") from None
        if not isinstance(body, dict):
            raise RequestError("body must be a JSON object")
        model = body.get("model")
        if model not in (None, served_model):
            raise LookupError(f"The model `{model}` does not exist.")
        return body

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        try:
            body = await json_body(request)
            inputs = parse_inputs(body.get("input"))
            limit, side = parse_truncation(body, encoder.max_len)
            fmt = body.get("encoding_format") or "float"
            if fmt not in ("float", "base64"):
                raise RequestError("encoding_format must be 'float' or 'base64'")
            vecs, tokens = await asyncio.to_thread(
                encoder.encode, inputs, truncate=limit, side=side
            )
        except LookupError as exc:
            return error(404, str(exc))
        except ValueError as exc:  # RequestError and the encoder's refusals
            return error(400, str(exc))
        data = [
            {"index": i, "object": "embedding", "embedding": encode_vector(v, str(fmt))}
            for i, v in enumerate(vecs)
        ]
        return JSONResponse(
            {
                "id": f"embd-{uuid.uuid4().hex}",
                "object": "list",
                "created": int(time.time()),
                "model": served_model,
                "data": data,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "completion_tokens": 0},
            }
        )

    @app.post("/tokenize")
    async def tokenize(request: Request) -> JSONResponse:
        try:
            body = await json_body(request)
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise RequestError("prompt must be a string")
            special = bool(body.get("add_special_tokens", True))
            ids = await asyncio.to_thread(encoder.tokenize, prompt, add_special_tokens=special)
        except LookupError as exc:
            return error(404, str(exc))
        except ValueError as exc:
            return error(400, str(exc))
        return JSONResponse({"count": len(ids), "max_model_len": encoder.max_len, "tokens": ids})

    return app


def build_clm_app(
    encoder: Any,
    *,
    api_key: str | None,
    ckpt: Path,
    ckpt_dir: Path | None,
    action_cache: str,
) -> Any:
    """CLM's patched app over an in-process ``Engine`` whose embedder is ``encoder``."""
    from clm.engine import Engine
    from clm.server import create_app

    engine = Engine(
        embedder=encoder,
        checkpoint=str(ckpt),
        checkpoint_dir=str(ckpt_dir) if ckpt_dir else None,
        device="cpu",
        action_cache=action_cache,
    )
    if not engine.heads:
        raise SystemExit(f"fallback_serve: no head loaded from {ckpt}")
    return create_app(engine, api_key, ui=False, cors=False)


def resolve_keys(allow_anonymous: bool, hosts: Sequence[str]) -> tuple[str | None, str | None]:
    """``(clm key, encoder key)`` from the environment, applying the refusal rules."""
    clm_key = os.environ.get("CLM_API_KEY") or None
    emb_key = os.environ.get("VLLM_API_KEY") or os.environ.get("CLM_EMB_API_KEY") or None
    missing = [
        name
        for name, key in (("CLM_API_KEY", clm_key), ("VLLM_API_KEY or CLM_EMB_API_KEY", emb_key))
        if not key
    ]
    if missing and any(not is_loopback(h) for h in hosts):
        raise SystemExit(
            f"fallback_serve: refusing a non-loopback bind without {', '.join(missing)}"
        )
    if missing and not allow_anonymous:
        raise SystemExit(
            f"fallback_serve: {', '.join(missing)} not set (EnvironmentFile=~/.mesa/clm/secrets/"
            "clm.env); pass --allow-anonymous to serve loopback without keys"
        )
    return clm_key, emb_key


async def serve_all(apps: Sequence[tuple[Any, str, int]]) -> None:
    """Run the apps in one event loop; SIGTERM stops them all (uvicorn re-raises it LIFO)."""
    import uvicorn

    servers = [
        uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        for app, host, port in apps
    ]
    await asyncio.gather(*(s.serve() for s in servers))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="In-process fallback for the mesa-clm encoder.")
    ap.add_argument("--model", default=None, help="HF model id (default: the pinned Qwen/Qwen3-8B)")
    ap.add_argument("--revision", default=None, help="HF revision (default: the pinned commit)")
    ap.add_argument("--served-model-name", default=SERVED_MODEL)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for the K0 3b throughput run")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--native-triton", action="store_true", help="keep torch's Triton overrides")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--emb-port", type=int, default=EMB_PORT)
    ap.add_argument("--clm-port", type=int, default=CLM_PORT)
    ap.add_argument("--ckpt", type=Path, default=Path(HEADS) / "CLM_v0.1-8B.pt")
    ap.add_argument("--ckpt-dir", type=Path, default=Path(HEADS) / "served")
    ap.add_argument("--action-cache", default="512MiB")
    ap.add_argument("--no-clm", action="store_true", help="serve only the :8090 encoder API")
    ap.add_argument("--allow-anonymous", action="store_true")
    args = ap.parse_args(argv)

    clm_key, emb_key = resolve_keys(args.allow_anonymous, [args.host])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cuda_encoder

    encoder = cuda_encoder.CudaEncoder(
        args.model or cuda_encoder.MODEL,
        args.revision or cuda_encoder.REVISION,
        max_len=args.max_len,
        device=args.device,
        dtype=args.dtype,
        batch=args.batch,
        native_triton=args.native_triton,
    )
    apps: list[tuple[Any, str, int]] = [
        (
            build_embeddings_app(encoder, api_key=emb_key, served_model=args.served_model_name),
            args.host,
            args.emb_port,
        )
    ]
    if not args.no_clm:
        ckpt_dir = args.ckpt_dir.expanduser()
        clm_app = build_clm_app(
            encoder,
            api_key=clm_key,
            ckpt=args.ckpt.expanduser(),
            ckpt_dir=ckpt_dir if ckpt_dir.is_dir() else None,
            action_cache=args.action_cache,
        )
        apps.append((clm_app, args.host, args.clm_port))
    ports = ", ".join(f"{host}:{port}" for _, host, port in apps)
    print(
        f"fallback_serve: {encoder.model_id}@{encoder.revision[:12]} on {args.device} "
        f"(route transformers, triton overrides {'off' if encoder.triton_deregistered else 'on'}); "
        f"serving {ports}",
        flush=True,
    )
    asyncio.run(serve_all(apps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
