"""The patched clm-serve (0001-0006) and the fallback encoder app, with stubs (plan §9)."""

# ruff: noqa: S101

from __future__ import annotations

import base64
import hmac
import importlib
import inspect
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cuda_encoder
import fallback_serve
import numpy as np
import pytest

KEY = "k" * 43


@pytest.fixture
def clm() -> Iterator[ModuleType]:
    """The patched ``clm`` package from ``MESA_CLM_CLM_SRC`` (skip when absent)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("requests")
    src = os.environ.get("MESA_CLM_CLM_SRC")
    if not src or not (Path(src) / "clm" / "server.py").is_file():
        pytest.skip("MESA_CLM_CLM_SRC does not point at the patched CLM clone's src/")
    module = importlib.import_module("clm")
    yield module
    for name in [m for m in sys.modules if m == "clm" or m.startswith("clm.")]:
        sys.modules.pop(name, None)


def _client(app: Any) -> Any:
    from fastapi.testclient import TestClient

    return TestClient(app)


class StubEngine:
    """What ``create_app`` touches: ``embedder.healthy()``, ``models()``, ``arena``."""

    class _Embedder:
        def healthy(self) -> bool:
            return True

    def __init__(self) -> None:
        self.embedder = self._Embedder()
        self.arena = None

    def models(self) -> list[dict[str, str]]:
        return [{"name": "clm-latest", "description": "stub", "release_date": "2026-09-19"}]


# -- patched CLM ---------------------------------------------------------------------------------


def test_create_app_guards_v1_and_leaves_health_open(clm: ModuleType) -> None:
    server = importlib.import_module("clm.server")
    client = _client(server.create_app(StubEngine(), api_key=KEY, ui=False))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert client.post("/v1/systemone", json={}).status_code == 401


def test_create_app_compares_in_constant_time(clm: ModuleType, monkeypatch: Any) -> None:
    server = importlib.import_module("clm.server")
    calls: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return bool(real(a, b))

    monkeypatch.setattr(hmac, "compare_digest", spy)
    client = _client(server.create_app(StubEngine(), api_key=KEY, ui=False))
    client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert calls and calls[-1][1] == f"Bearer {KEY}".encode()


def test_main_refuses_an_exposed_unkeyed_server(clm: ModuleType, monkeypatch: Any) -> None:
    server = importlib.import_module("clm.server")
    monkeypatch.delenv("CLM_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["clm-serve", "--host", "0.0.0.0", "--no-download"])  # noqa: S104
    with pytest.raises(SystemExit, match="refusing to serve"):
        server.main()
    assert server.is_loopback("127.0.0.1") and server.is_loopback("localhost")
    assert not server.is_loopback("0.0.0.0")  # noqa: S104


def test_main_passes_the_encoder_key_and_cache_size(clm: ModuleType, monkeypatch: Any) -> None:
    server = importlib.import_module("clm.server")
    embedder_mod = importlib.import_module("clm.embedder")
    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    class RecordingEmbedder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

    def engine(embedder: Any, **kwargs: Any) -> Any:
        raise Stop

    monkeypatch.setenv("CLM_EMB_API_KEY", KEY)
    monkeypatch.setenv("CLM_EMB_CACHE_SIZE", "20000")
    monkeypatch.delenv("CLM_EMB_MAX_TOKENS", raising=False)
    monkeypatch.setattr(embedder_mod, "Embedder", RecordingEmbedder)
    monkeypatch.setattr(server, "Engine", engine)
    monkeypatch.setattr(
        sys, "argv", ["clm-serve", "--ckpt", "/nonexistent.pt", "--no-download", "--device", "cpu"]
    )
    with pytest.raises(Stop):
        server.main()
    assert seen["api_key"] == KEY
    assert seen["cache_size"] == 20000
    assert seen["max_tokens"] == 2048  # the unit passes --max-tokens 4096 explicitly


def test_embedder_keeps_the_tail(clm: ModuleType, monkeypatch: Any) -> None:
    embedder_mod = importlib.import_module("clm.embedder")
    emb = embedder_mod.Embedder(max_tokens=4096, api_key=KEY)
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            vec = base64.b64encode(np.ones(4, dtype="<f4").tobytes()).decode()
            return {"data": [{"index": 0, "embedding": vec}], "usage": {"prompt_tokens": 3}}

    def post(url: str, json: dict[str, Any], timeout: float) -> Response:
        captured.update(json)
        return Response()

    monkeypatch.setattr(emb.session, "post", post)
    emb.embed(["some state"])
    assert captured["truncate_prompt_tokens"] == 4096
    assert captured["truncation_side"] == "left"
    assert emb.session.headers["Authorization"] == f"Bearer {KEY}"


def test_arena_reuses_the_slot_of_a_concurrent_miss(clm: ModuleType) -> None:
    cache = importlib.import_module("clm.cache")
    pool = cache.Pool(np.zeros((4, 2), dtype=np.float32), 2)
    first = pool.claim("a")
    assert pool.claim("a") == first
    assert len(pool.free) == 3


def test_heads_load_with_weights_only(clm: ModuleType) -> None:
    heads = Path(importlib.import_module("clm.heads").__file__ or "")
    assert "weights_only=True" in heads.read_text(encoding="utf-8")


# -- fallback encoder app ------------------------------------------------------------------------


class StubEncoder:
    model_id = "Qwen/Qwen3-8B"
    max_len = 16

    def __init__(self) -> None:
        self.calls: list[tuple[Any, int | None, str]] = []

    def tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def encode(
        self, inputs: Sequence[str] | Sequence[Sequence[int]], *, truncate: int | None, side: Any
    ) -> tuple[Any, int]:
        self.calls.append((list(inputs), truncate, side))
        vecs = np.arange(len(inputs) * 4, dtype=np.float32).reshape(len(inputs), 4) / 10
        return vecs, 7 * len(inputs)

    def healthy(self) -> bool:
        return True


@pytest.fixture
def fallback() -> tuple[Any, StubEncoder]:
    pytest.importorskip("fastapi")
    enc = StubEncoder()
    return _client(fallback_serve.build_embeddings_app(enc, api_key=KEY)), enc


AUTH = {"Authorization": f"Bearer {KEY}"}


def test_fallback_guards_everything_but_health(fallback: tuple[Any, StubEncoder]) -> None:
    """Every :8090 route needs the encoder key except ``/health``, as in the vLLM container with
    ``serving/vllm_auth.py`` (DESIGN A4)."""
    client, _ = fallback
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 401
    assert client.post("/tokenize", json={"prompt": "abc"}).status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/nonexistent").status_code == 401
    assert client.post("/tokenize", json={"prompt": "abc"}, headers=AUTH).json()["count"] == 3
    models = client.get("/v1/models", headers=AUTH).json()
    assert models["data"][0]["id"] == "qwen3-8b"
    assert models["data"][0]["max_model_len"] == 16


def test_fallback_embeddings_base64_and_truncation(fallback: tuple[Any, StubEncoder]) -> None:
    client, enc = fallback
    body = {
        "model": "qwen3-8b",
        "input": ["a", "b"],
        "encoding_format": "base64",
        "truncate_prompt_tokens": 16,
        "truncation_side": "left",
    }
    r = client.post("/v1/embeddings", json=body, headers=AUTH)
    assert r.status_code == 200
    out = r.json()
    vec = np.frombuffer(base64.b64decode(out["data"][1]["embedding"]), dtype="<f4")
    assert np.allclose(vec, np.array([4, 5, 6, 7], dtype=np.float32) / 10)
    assert out["usage"]["prompt_tokens"] == 14
    r = client.post(
        "/v1/embeddings", json={"input": [1, 2, 3], "truncate_prompt_tokens": -1}, headers=AUTH
    )
    assert r.status_code == 200 and isinstance(r.json()["data"][0]["embedding"], list)
    assert enc.calls == [(["a", "b"], 16, "left"), ([[1, 2, 3]], 16, "right")]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"input": "x", "model": "other"}, 404),
        ({"input": "x", "truncate_prompt_tokens": 17}, 400),
        ({"input": "x", "truncation_side": "middle"}, 400),
        ({"input": "x", "encoding_format": "bytes"}, 400),
        ({"input": []}, 400),
        ({"input": [True]}, 400),
    ],
)
def test_fallback_rejects_bad_requests(
    fallback: tuple[Any, StubEncoder], body: dict[str, Any], status: int
) -> None:
    client, _ = fallback
    assert client.post("/v1/embeddings", json=body, headers=AUTH).status_code == status


def test_fallback_compares_in_constant_time(monkeypatch: Any) -> None:
    calls: list[bytes] = []

    def spy(a: bytes, b: bytes) -> bool:
        calls.append(b)
        return a == b

    monkeypatch.setattr(hmac, "compare_digest", spy)
    assert fallback_serve.bearer_ok(f"Bearer {KEY}", KEY)
    assert not fallback_serve.bearer_ok(None, KEY)
    assert fallback_serve.bearer_ok(None, None)
    assert calls == [f"Bearer {KEY}".encode()] * 2


def test_fallback_key_rules(monkeypatch: Any) -> None:
    for name in ("CLM_API_KEY", "VLLM_API_KEY", "CLM_EMB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="non-loopback"):
        fallback_serve.resolve_keys(True, ["0.0.0.0"])  # noqa: S104
    with pytest.raises(SystemExit, match="allow-anonymous"):
        fallback_serve.resolve_keys(False, ["127.0.0.1"])
    assert fallback_serve.resolve_keys(True, ["127.0.0.1"]) == (None, None)
    monkeypatch.setenv("CLM_API_KEY", "c" * 43)
    monkeypatch.setenv("CLM_EMB_API_KEY", KEY)
    assert fallback_serve.resolve_keys(False, ["0.0.0.0"]) == ("c" * 43, KEY)  # noqa: S104


def test_cuda_encoder_defaults_to_one_sequence_and_the_capped_window(
    monkeypatch: Any,
) -> None:
    """Batch 1 (no padding) and ``embed`` truncating to ``max_len - 1`` (DESIGN A3, A4), checked on
    the signature and on ``embed`` with the model replaced (no torch here)."""
    sig = inspect.signature(cuda_encoder.CudaEncoder.__init__)
    assert sig.parameters["batch"].default == 1
    assert sig.parameters["max_len"].default == cuda_encoder.MAX_LEN == 4096
    seen: list[tuple[Any, ...]] = []

    def encode(texts: Any, truncate: Any, side: Any) -> tuple[int, int]:
        seen.append((texts, truncate, side))
        return 0, 0

    enc = object.__new__(cuda_encoder.CudaEncoder)
    enc.max_len = 4096
    monkeypatch.setattr(enc, "encode", encode)
    enc.embed(["a"])
    assert seen == [(["a"], 4095, "left")]


def test_truncate_ids_keeps_the_requested_side() -> None:
    ids = list(range(10))
    assert cuda_encoder.truncate_ids(ids, 4, "left") == [6, 7, 8, 9]
    assert cuda_encoder.truncate_ids(ids, 4, "right") == [0, 1, 2, 3]
    assert cuda_encoder.truncate_ids(ids, None, "left") == ids
    assert cuda_encoder.truncate_ids(ids, 20, "left") == ids


@pytest.mark.parametrize("batch", ["8", "0", "2"])
def test_fallback_refuses_a_batched_encoder(batch: str, capsys: Any) -> None:
    """The fallback's encoder_fp says ``serial`` (DESIGN A3): a server batching sequences would
    stamp it on vectors it does not describe, so ``--batch`` other than 1 stops at the arguments,
    before any key, model or port is touched."""
    with pytest.raises(SystemExit) as err:
        fallback_serve.main(["--batch", batch, "--no-clm"])
    assert err.value.code == 2
    assert "one sequence per forward pass only" in capsys.readouterr().err
