"""The vLLM pooling encoder behind ``POST /v1/embeddings`` (plan §3 ``clm/encoder.py``, §4.3,
§6.1; DESIGN D16, D23).

:class:`EncoderClient` is the mesa-clm counterpart of CLM's ``Embedder`` (``embedder.py``) with
PR #6 applied: every request carries ``truncate_prompt_tokens = max_len`` **and**
``truncation_side = "left"``, so a state longer than the window loses its head, never the target
and question at its tail (last-token pooling reads the tail; RESEARCH.md). Vectors come back
``encoding_format: base64`` (little-endian float32), are L2-normalised exactly as ``embedder.py``
``l2`` does, and are returned as one ``float32 [n, dim]`` array together with the prompt tokens
the server charged. Inputs are sent in chunks of ``batch``.

The token guard (plan §4.3) counts a context's tokens before it is embedded so a truncated
context becomes ``truncated=true`` and an abstain instead of a silently clipped decision. It
uses, in order: vLLM's ``POST /tokenize`` when the server has it (``None`` from
:meth:`EncoderClient.tokenize` on 404/405/501 means it does not; the probe result is
remembered), the ``tokenize`` extra (``tokenizers``, imported lazily, over the
``tokenizer.json`` of the pinned Qwen3 revision passed as ``tokenizer_json``), else the
conservative ``ceil(chars / 2.0)`` (the dense NEON cards measure 2.4-2.7 chars per token, so this
over-counts and never under-counts). ``truncated`` is ``tokens > max_len - margin``.

Raw cards are never sent to the encoder (D23, D26): callers pass rendered contexts and
candidate texts only. The transport rules are :class:`mesa_clm.clm.http.HttpEndpoint`'s.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, Self

import httpx
import numpy as np
import numpy.typing as npt

from mesa_clm.clm.headproj import l2
from mesa_clm.clm.http import DEFAULT_BACKOFF_S, DEFAULT_RETRIES, ClmError, HttpEndpoint
from mesa_clm.config import EncoderConfig
from mesa_clm.net import CircuitBreaker

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_MODEL: Final[str] = "qwen3-8b"
EMBEDDING_DIM: Final[int] = 4096
CHARS_PER_TOKEN: Final[float] = 2.0
GUARD_MARGIN: Final[int] = 16
TRUNCATION_SIDE: Final[str] = "left"
# vLLM answers 404 on an unknown route, 405 on a known path without POST; 501 is "not implemented".
_UNSUPPORTED: Final[frozenset[int]] = frozenset({404, 405, 501})

TokenSource = Literal["server", "tokenizers", "chars"]
F32 = npt.NDArray[np.float32]


class TokenCount(NamedTuple):
    """One text's token guard result: the count, whether it would be truncated, and which
    counter produced it (``server`` is exact, ``tokenizers`` exact, ``chars`` an over-estimate)."""

    tokens: int
    truncated: bool
    source: TokenSource


def chars_estimate(text: str) -> int:
    """The conservative fallback count: ``ceil(len(text) / CHARS_PER_TOKEN)``."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def token_guard(
    texts: Sequence[str],
    *,
    max_len: int,
    margin: int = GUARD_MARGIN,
    counter: Callable[[str], tuple[int, TokenSource]],
) -> list[TokenCount]:
    """Per-text ``(tokens, truncated, source)`` with ``truncated = tokens > max_len - margin``.

    ``counter`` returns a count and its source; :meth:`EncoderClient.count_tokens` is the usual
    one, the fake transport's ``len(text.split())`` another.
    """
    if margin < 0 or margin >= max_len:
        raise ValueError("margin must be in [0, max_len)")
    limit = max_len - margin
    out: list[TokenCount] = []
    for text in texts:
        n, source = counter(text)
        out.append(TokenCount(n, n > limit, source))
    return out


def decode_base64_f32(data: str, *, dim: int | None = None) -> F32:
    """One base64 float32 vector as vLLM's ``encoding_format: base64`` returns it."""
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"embedding is not valid base64: {exc}") from None
    if len(raw) % 4:
        raise ValueError(f"embedding has {len(raw)} bytes, not a multiple of 4")
    vec = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if dim is not None and vec.shape[0] != dim:
        raise ValueError(f"embedding has {vec.shape[0]} dimensions, expected {dim}")
    return vec


class EncoderClient:
    """``POST /v1/embeddings`` with left truncation, base64 float32, L2 (module docstring)."""

    def __init__(
        self,
        url: str,
        api_key: str | None,
        *,
        model: str = DEFAULT_ENCODER_MODEL,
        max_len: int = 4096,
        batch: int = 32,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
        allow_remote: bool = False,
        expect_dim: int | None = EMBEDDING_DIM,
        tokenizer_json: str | Path | None = None,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF_S,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_len <= GUARD_MARGIN:
            raise ValueError(f"max_len must exceed the guard margin ({GUARD_MARGIN})")
        if batch < 1:
            raise ValueError("batch must be at least 1")
        self.model = model
        self.max_len = int(max_len)
        self.batch = int(batch)
        self.expect_dim = expect_dim
        self.tokenizer_json = Path(tokenizer_json).expanduser() if tokenizer_json else None
        self.endpoint = HttpEndpoint(
            url,
            api_key,
            what="encoder",
            timeout=timeout,
            transport=transport,
            allow_remote=allow_remote,
            retries=retries,
            backoff=backoff,
            breaker=breaker,
            sleep=sleep,
        )
        # None: not probed yet; False: the server has no /tokenize (remembered, not re-probed).
        self._tokenize_supported: bool | None = None
        self._local_tokenizer: Any | None = None
        self._local_tokenizer_tried = False

    @classmethod
    def from_config(
        cls,
        cfg: EncoderConfig,
        *,
        allow_remote: bool = False,
        transport: httpx.BaseTransport | None = None,
        tokenizer_json: str | Path | None = None,
    ) -> EncoderClient:
        """A client from the ``encoder`` section; ``allow_remote`` is the ``clm`` section's one
        switch for the serving pair (plan §6.4)."""
        return cls(
            cfg.url,
            cfg.resolved_api_key(),
            model=cfg.model,
            max_len=cfg.max_len,
            timeout=cfg.timeout,
            transport=transport,
            allow_remote=allow_remote,
            tokenizer_json=tokenizer_json,
        )

    @property
    def url(self) -> str:
        return self.endpoint.base_url

    @property
    def breaker(self) -> CircuitBreaker:
        return self.endpoint.breaker

    @property
    def calls(self) -> int:
        return self.endpoint.calls

    def close(self) -> None:
        self.endpoint.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- embeddings -------------------------------------------------------------------------------

    def request_body(self, texts: Sequence[str]) -> dict[str, Any]:
        """The wire body for one chunk (PR #6 shape; key order as ``embedder.py`` builds it)."""
        return {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "base64",
            "truncate_prompt_tokens": self.max_len,
            "truncation_side": TRUNCATION_SIDE,
        }

    def embed(self, texts: Sequence[str]) -> tuple[F32, int]:
        """``(float32 [n, dim] L2-normalised vectors, prompt tokens charged)`` for ``texts``.

        Empty input costs no request. Rows follow the input order whatever order the server
        lists ``data`` in (``index`` is honoured). A response with the wrong number of vectors or
        an unexpected dimension is a :class:`ClmError` with the 200 status it came with.
        """
        if not texts:
            return np.zeros((0, self.expect_dim or EMBEDDING_DIM), dtype=np.float32), 0
        rows: list[F32] = []
        tokens = 0
        for start in range(0, len(texts), self.batch):
            chunk = texts[start : start + self.batch]
            vectors, n = self._embed_chunk(chunk)
            rows.extend(vectors)
            tokens += n
        dims = {v.shape[0] for v in rows}
        if len(dims) != 1:
            raise ClmError(200, f"encoder returned vectors of mixed dimensions {sorted(dims)}")
        return l2(np.stack(rows)), tokens

    def _embed_chunk(self, texts: Sequence[str]) -> tuple[list[F32], int]:
        j, response = self.endpoint.post_json("/v1/embeddings", self.request_body(texts))
        status = response.status_code
        try:
            data = j["data"]
            if not isinstance(data, list) or len(data) != len(texts):
                got = len(data) if isinstance(data, list) else "no"
                raise ClmError(status, f"encoder returned {got} vectors for {len(texts)} inputs")
            by_index: dict[int, F32] = {}
            for item in data:
                emb = item["embedding"]
                vec = (
                    decode_base64_f32(emb, dim=self.expect_dim)
                    if isinstance(emb, str)
                    else np.asarray(emb, dtype=np.float32)
                )
                if self.expect_dim is not None and vec.shape != (self.expect_dim,):
                    raise ClmError(
                        status,
                        f"encoder vector has shape {vec.shape}, expected ({self.expect_dim},)",
                    )
                by_index[int(item["index"])] = vec
            if sorted(by_index) != list(range(len(texts))):
                raise ClmError(status, "encoder response indices do not cover the inputs")
            usage = j.get("usage") or {}
            tokens = int(usage.get("prompt_tokens") or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ClmError(status, f"malformed /v1/embeddings response: {exc}") from None
        return [by_index[i] for i in range(len(texts))], tokens

    # -- discovery --------------------------------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        """The OpenAI-style ``GET /v1/models`` list (``data``); the served name must appear."""
        j = self.endpoint.get_json("/v1/models")
        try:
            data = j["data"]
        except (KeyError, TypeError) as exc:
            raise ClmError(200, f"malformed /v1/models response: {exc}") from None
        if not isinstance(data, list):
            raise ClmError(200, "malformed /v1/models response: 'data' is not a list")
        return [dict(m) for m in data]

    def healthy(self) -> bool:
        """``GET /v1/models`` answers 200 (CLM's ``Embedder.healthy``); any failure is ``False``."""
        try:
            self.endpoint.request("GET", "/v1/models", timeout=5.0, retry=False)
        except Exception:  # a probe; the reason belongs to the doctor
            return False
        return True

    # -- token guard ------------------------------------------------------------------------------

    def tokenize(self, text: str) -> int | None:
        """Exact token count from vLLM's ``POST /tokenize``, or ``None`` when the server has no
        such route (remembered after the first 404/405/501). Other errors raise."""
        if self._tokenize_supported is False:
            return None
        body = {"model": self.model, "prompt": text, "add_special_tokens": True}
        try:
            j, _ = self.endpoint.post_json("/tokenize", body)
        except ClmError as exc:
            if exc.status in _UNSUPPORTED:
                self._tokenize_supported = False
                logger.info(
                    "encoder at %s has no /tokenize; the token guard falls back",
                    self.endpoint.shown,
                )
                return None
            raise
        try:
            count = int(j["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClmError(200, f"malformed /tokenize response: {exc}") from None
        self._tokenize_supported = True
        return count

    def count_tokens(self, text: str) -> tuple[int, TokenSource]:
        """The best available count: server, then the ``tokenize`` extra, then ``chars / 2``."""
        n = self.tokenize(text)
        if n is not None:
            return n, "server"
        tok = self._tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False).ids), "tokenizers"
        return chars_estimate(text), "chars"

    def token_guard(
        self, texts: Sequence[str], *, max_len: int | None = None, margin: int = GUARD_MARGIN
    ) -> list[TokenCount]:
        """:func:`token_guard` over :meth:`count_tokens` with this client's ``max_len``."""
        return token_guard(
            texts,
            max_len=max_len if max_len is not None else self.max_len,
            margin=margin,
            counter=self.count_tokens,
        )

    def _tokenizer(self) -> Any | None:
        """The ``tokenizers`` fast tokenizer over ``tokenizer_json``, loaded once; ``None`` when
        the extra is absent or no file was given (never a network download)."""
        if self._local_tokenizer_tried:
            return self._local_tokenizer
        self._local_tokenizer_tried = True
        if self.tokenizer_json is None:
            return None
        try:
            import tokenizers
        except ImportError:
            logger.info(
                "the 'tokenize' extra is not installed; the token guard uses chars/%.1f",
                CHARS_PER_TOKEN,
            )
            return None
        try:
            self._local_tokenizer = tokenizers.Tokenizer.from_file(str(self.tokenizer_json))
        except Exception as exc:  # a bad file must not break annotation
            logger.warning(
                "could not load tokenizer %s (%s); using chars/%.1f",
                self.tokenizer_json,
                type(exc).__name__,
                CHARS_PER_TOKEN,
            )
            self._local_tokenizer = None
        return self._local_tokenizer
