"""clm-serve and the vLLM encoder as one ``httpx.MockTransport`` handler (plan §9 "Fake CLM").

:class:`FakeClmServer` answers the routes the core calls, with the shapes of CLM ``server.py``
(``bb42c6c5``) and vLLM's OpenAI-compatible server:

* clm-serve (port ``clm_port``, default 8700): ``POST /v1/systemone`` and ``POST /v1/rank``
  scored by :class:`mesa_clm.clm.fake.FakeClm` (vendored ``build_pairs`` /
  ``answer_from_logits`` over hashed n-grams), ``GET /v1/models`` (``{"models": [...]}``),
  ``GET /health`` (``{"ok", "embedder", "models", "cache"}``, unguarded); errors ``401``
  (bad bearer key on ``/v1/*``), ``422`` (malformed body, unknown model, bad temperature) as
  ``{"detail": ...}``; every 200 carries ``X-CLM-Latency-Ms``;
* the encoder (port ``encoder_port``, default 8090) behind the bearer guard of DESIGN A4
  (``serving/vllm_auth.py``): every path but exactly ``/health`` answers ``401`` without the
  right key, unknown paths included; with it ``POST /v1/embeddings`` (base64 float32 from
  :class:`mesa_clm.clm.fake.FakeEncoder`, honouring ``truncate_prompt_tokens`` with
  ``truncation_side: left`` on whitespace tokens; an input may be a text or a list of token ids
  from ``/tokenize``), ``GET /v1/models`` (OpenAI ``{"data": [...]}`` with ``owned_by`` and
  ``root``, settable to impersonate the in-process fallback), ``POST /tokenize``
  (``count`` = ``len(text.split())`` and one stable id per word, or 404 when
  ``tokenize_supported=False``); errors ``404`` for an unknown model or route, ``400`` for a
  malformed body, all as vLLM's ``{"error": {...}}``.

The two servers share one handler and are told apart by the request's port, so a test points
``ClmHttpClient`` at ``http://127.0.0.1:8700`` and ``EncoderClient`` at ``http://127.0.0.1:8090``
with the same ``transport``. ``fail_next`` scripts transient failures (a status or an exception
raised before the route runs) for the retry and breaker tests; ``requests`` records every
request the handler saw. :func:`requests_session` mounts the same handler behind a
``requests.Session`` so the vendored ``tests/_vendor/clm_client.py`` (the wire oracle) can run
over it too.
"""

from __future__ import annotations

import base64
import hmac
import json
from collections.abc import Sequence
from typing import Any

import httpx
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from mesa_clm.clm.fake import FakeClm, FakeClmError, FakeEncoder

CLM_PORT = 8700
ENCODER_PORT = 8090
CLM_URL = f"http://127.0.0.1:{CLM_PORT}"
ENCODER_URL = f"http://127.0.0.1:{ENCODER_PORT}"
LATENCY_HEADER = "X-CLM-Latency-Ms"

Failure = int | Exception


def _vllm_error(status: int, message: str, kind: str = "BadRequestError") -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"message": message, "type": kind, "param": None, "code": status}}
    )


def _detail(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"detail": message})


class FakeClmServer:
    """The handler (module docstring). Construct, then pass ``transport()`` to the clients."""

    def __init__(
        self,
        *,
        clm: FakeClm | None = None,
        encoder: FakeEncoder | None = None,
        clm_api_key: str | None = None,
        encoder_api_key: str | None = None,
        encoder_model: str = "qwen3-8b",
        max_model_len: int = 4096,
        latency_ms: float = 12.5,
        tokenize_supported: bool = True,
        clm_port: int = CLM_PORT,
        encoder_port: int = ENCODER_PORT,
        encoder_root: str | None = "Qwen/Qwen3-8B",
        encoder_owned_by: str = "vllm",
    ) -> None:
        self.clm = clm or FakeClm(encoder)
        self.encoder = encoder or self.clm.encoder
        self.clm_api_key = clm_api_key
        self.encoder_api_key = encoder_api_key
        self.encoder_model = encoder_model
        self.max_model_len = max_model_len
        self.latency_ms = latency_ms
        self.tokenize_supported = tokenize_supported
        self.clm_port = clm_port
        self.encoder_port = encoder_port
        self.encoder_root = encoder_root
        self.encoder_owned_by = encoder_owned_by
        self.requests: list[httpx.Request] = []
        # Scripted failures consumed one per request before any route runs.
        self.fail_next: list[Failure] = []
        self.embedded_texts: list[str] = []
        # /tokenize ids: one stable id per distinct word, so token-id inputs decode back.
        self._word_ids: dict[str, int] = {}
        self._id_words: dict[int, str] = {}

    # -- helpers for tests -----------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.content]

    @property
    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def requests_session(self) -> requests.Session:
        """A ``requests.Session`` whose http:// traffic goes through this handler."""
        s = requests.Session()
        s.mount("http://", _RequestsOverHandler(self))
        return s

    # -- dispatch --------------------------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_next:
            failure = self.fail_next.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return _detail(failure, f"scripted failure {failure}")
        if request.url.port == self.encoder_port:
            return self._encoder(request)
        return self._clm(request)

    def _authorised(self, request: httpx.Request, key: str | None) -> bool:
        if key is None:
            return True
        header = request.headers.get("Authorization", "")
        return header.startswith("Bearer ") and hmac.compare_digest(header[7:], key)

    @staticmethod
    def _body(request: httpx.Request) -> dict[str, Any] | None:
        try:
            body = json.loads(request.content)
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    # -- clm-serve -------------------------------------------------------------------------------

    def _clm(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/health" and method == "GET":
            return self._ok(
                {
                    "ok": True,
                    "embedder": True,
                    "models": [m["name"] for m in self.clm.models()],
                    "cache": {"entries": 0},
                }
            )
        if not path.startswith("/v1/"):
            return _detail(404, "Not Found")
        if not self._authorised(request, self.clm_api_key):
            return _detail(401, "invalid API key")
        if path == "/v1/models" and method == "GET":
            return self._ok({"models": self.clm.models()})
        if path == "/v1/systemone" and method == "POST":
            body = self._body(request)
            if body is None or "state" not in body or not isinstance(body.get("questions"), dict):
                return _detail(422, "body needs 'state' and a 'questions' object")
            try:
                out = self.clm.answer(
                    body["state"],
                    body["questions"],
                    model=str(body.get("model", "clm-latest")),
                    temperature=float(body.get("temperature", 1.0)),
                )
            except (FakeClmError, TypeError, ValueError) as exc:
                return _detail(422, str(exc))
            return self._ok(out)
        if path == "/v1/rank" and method == "POST":
            body = self._body(request)
            if body is None or "context" not in body or not isinstance(body.get("answers"), list):
                return _detail(422, "body needs 'context' and an 'answers' list")
            model = str(body.get("model", "clm-latest"))
            try:
                ranked = self.clm.rank(
                    body["context"],
                    body["answers"],
                    instructions=body.get("question"),
                    model=model,
                    temperature=float(body.get("temperature", 1.0)),
                )
            except (FakeClmError, TypeError, ValueError) as exc:
                return _detail(422, str(exc))
            return self._ok({"model": model, "ranked": ranked})
        return _detail(404, "Not Found")

    def _ok(self, payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json=payload, headers={LATENCY_HEADER: f"{self.latency_ms:.1f}"})

    # -- the encoder -----------------------------------------------------------------------------

    def _word_id(self, word: str) -> int:
        if word not in self._word_ids:
            self._word_ids[word] = len(self._word_ids) + 1
            self._id_words[self._word_ids[word]] = word
        return self._word_ids[word]

    def _encoder(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/health":
            return httpx.Response(200, content=b"")
        # The bearer guard (serving/vllm_auth.py, DESIGN A4): everything else needs the key.
        if not self._authorised(request, self.encoder_api_key):
            return httpx.Response(401, json={"error": "Unauthorized"})
        if path == "/tokenize" and method == "POST":
            if not self.tokenize_supported:
                return _vllm_error(404, "Not Found", "NotFoundError")
            body = self._body(request)
            if body is None or not isinstance(body.get("prompt"), str):
                return _vllm_error(400, "tokenize needs a string 'prompt'")
            ids = [self._word_id(w) for w in body["prompt"].split()]
            return httpx.Response(
                200,
                json={"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids},
            )
        if path == "/v1/models" and method == "GET":
            entry: dict[str, Any] = {
                "id": self.encoder_model,
                "object": "model",
                "owned_by": self.encoder_owned_by,
                "max_model_len": self.max_model_len,
            }
            if self.encoder_root is not None:
                entry["root"] = self.encoder_root
            return httpx.Response(200, json={"object": "list", "data": [entry]})
        if path == "/v1/embeddings" and method == "POST":
            return self._embeddings(request)
        return _vllm_error(404, "Not Found", "NotFoundError")

    def _embeddings(self, request: httpx.Request) -> httpx.Response:
        body = self._body(request)
        if body is None:
            return _vllm_error(400, "body must be a JSON object")
        if body.get("model") != self.encoder_model:
            return _vllm_error(
                404, f"The model `{body.get('model')}` does not exist.", "NotFoundError"
            )
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, list):
            # Token-id inputs (lists of ids from /tokenize) decode back to their words.
            inputs = [
                " ".join(self._id_words.get(int(i), "?") for i in t) if isinstance(t, list) else t
                for t in inputs
            ]
        if not isinstance(inputs, list) or not all(isinstance(t, str) for t in inputs):
            return _vllm_error(400, "'input' must be a string, a list of strings or of id lists")
        fmt = body.get("encoding_format", "float")
        if fmt not in ("float", "base64"):
            return _vllm_error(400, "encoding_format must be 'float' or 'base64'")
        limit = body.get("truncate_prompt_tokens")
        side = body.get("truncation_side", "right")
        texts: list[str] = []
        for t in inputs:
            if isinstance(limit, int) and limit > 0 and side == "left":
                t = FakeEncoder.truncate(t, limit)
            elif isinstance(limit, int) and limit > 0:
                t = " ".join(t.split()[:limit])
            texts.append(t)
        self.embedded_texts.extend(texts)
        vectors, tokens = self.encoder.embed(texts)
        data = []
        for i, vec in enumerate(vectors):
            emb: Any = (
                base64.b64encode(vec.astype("<f4").tobytes()).decode("ascii")
                if fmt == "base64"
                else [float(x) for x in vec]
            )
            data.append({"object": "embedding", "index": i, "embedding": emb})
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": self.encoder_model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )


class _RequestsOverHandler(BaseAdapter):
    """A ``requests`` transport adapter that hands each prepared request to the httpx handler."""

    def __init__(self, handler: FakeClmServer) -> None:
        super().__init__()
        self.handler = handler

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: Any = True,
        cert: Any = None,
        proxies: Any = None,
    ) -> requests.Response:
        body = request.body
        content = body.encode("utf-8") if isinstance(body, str) else (body or b"")
        req = httpx.Request(
            request.method or "GET",
            str(request.url),
            headers=dict(request.headers),
            content=content,
        )
        resp = self.handler(req)
        resp.read()
        out = requests.Response()
        out.status_code = resp.status_code
        out.headers = CaseInsensitiveDict(dict(resp.headers))
        out._content = resp.content
        out.encoding = "utf-8"
        out.url = str(request.url)
        out.request = request
        return out

    def close(self) -> None:
        pass


def wire_json(body: Any) -> str:
    """The exact bytes ``ClmHttpClient`` puts on the wire for ``body`` (httpx's ``json=``
    encoding: compact separators, non-ASCII kept), as a string."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def golden_questions(anchor: str, candidates: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """A rank_fit Choice over ``candidates`` ``(curie, text)`` plus the anchor, wire-shaped."""
    criteria: dict[str, Any] = {curie: text for curie, text in candidates}
    criteria["__none__"] = anchor
    return {"type": "choice", "instructions": None, "criteria": criteria}
