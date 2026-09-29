"""``EncoderClient`` over the fake transport: the PR #6 request body (``truncate_prompt_tokens`` +
``truncation_side: left``), base64 float32 decoding, L2, batching, ``models``/``healthy``,
``/tokenize`` with the 404 fallback, the token guard's three counters, auth and the loopback gate.
No network, no tokenizer download."""

from __future__ import annotations

import base64
import json
import math
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from mesa_clm.clm.encoder import (
    CHARS_PER_TOKEN,
    GUARD_MARGIN,
    EncoderClient,
    TokenCount,
    chars_estimate,
    decode_base64_f32,
    token_guard,
)
from mesa_clm.clm.fake import FakeEncoder
from mesa_clm.clm.http import ClmError
from mesa_clm.config import EncoderConfig
from mesa_clm.net import EndpointError
from tests.fakes.clm_transport import ENCODER_URL, FakeClmServer

KEY = "encoder-key-not-a-secret"
TEXTS = ["Radial distance between the observer and the individual", "site code", "the term label"]


@pytest.fixture
def server() -> FakeClmServer:
    return FakeClmServer(encoder_api_key=KEY, max_model_len=4096)


def client(server: FakeClmServer, **kw: Any) -> EncoderClient:
    kw.setdefault("sleep", lambda _s: None)
    return EncoderClient(ENCODER_URL, KEY, transport=server.transport(), **kw)


# -- embed ----------------------------------------------------------------------------------------


def test_embed_body_shape_and_vectors(server: FakeClmServer) -> None:
    with client(server) as c:
        vectors, tokens = c.embed(TEXTS)
    assert vectors.shape == (3, 4096) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert tokens == sum(len(t.split()) for t in TEXTS)
    body = server.bodies[-1]
    assert body == {
        "model": "qwen3-8b",
        "input": TEXTS,
        "encoding_format": "base64",
        "truncate_prompt_tokens": 4096,
        "truncation_side": "left",
    }
    assert list(body) == [
        "model",
        "input",
        "encoding_format",
        "truncate_prompt_tokens",
        "truncation_side",
    ]
    assert server.requests[-1].headers["Authorization"] == f"Bearer {KEY}"
    # The same vectors the fake produces directly: base64 round trip is lossless.
    expect, _ = FakeEncoder().embed(TEXTS)
    np.testing.assert_array_equal(vectors, expect)
    assert c.calls == 1


def test_embed_batches_and_keeps_order(server: FakeClmServer) -> None:
    c = client(server, batch=2)
    texts = [f"text number {i}" for i in range(5)]
    vectors, tokens = c.embed(texts)
    assert vectors.shape == (5, 4096) and tokens == 15
    assert [len(b["input"]) for b in server.bodies] == [2, 2, 1]
    expect, _ = FakeEncoder().embed(texts)
    np.testing.assert_array_equal(vectors, expect)
    empty, n = c.embed([])
    assert empty.shape == (0, 4096) and n == 0 and len(server.requests) == 3


def test_embed_honours_shuffled_indices_and_float_format() -> None:
    enc = FakeEncoder()
    vecs, _ = enc.embed(["a b", "c d"])

    def handler(request: httpx.Request) -> httpx.Response:
        data = [
            {"object": "embedding", "index": 1, "embedding": [float(x) for x in vecs[1]]},
            {
                "object": "embedding",
                "index": 0,
                "embedding": base64.b64encode(vecs[0].astype("<f4").tobytes()).decode(),
            },
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": "qwen3-8b",
                "usage": {"prompt_tokens": 4},
            },
        )

    c = EncoderClient(ENCODER_URL, None, transport=httpx.MockTransport(handler))
    out, tokens = c.embed(["a b", "c d"])
    np.testing.assert_allclose(out, vecs, atol=1e-7)
    assert tokens == 4


def test_embed_left_truncation_reaches_the_fake(server: FakeClmServer) -> None:
    """The fake honours ``truncate_prompt_tokens`` on whitespace tokens from the left, so a long
    text embeds like its tail: what PR #6 guarantees on the real server."""
    c = client(server, max_len=20)
    words = [f"w{i}" for i in range(50)]
    long_text = " ".join(words)
    tail = " ".join(words[-20:])
    v_long, _ = c.embed([long_text])
    v_tail, _ = c.embed([tail])
    np.testing.assert_array_equal(v_long, v_tail)
    assert server.embedded_texts[0] == tail
    assert server.bodies[-1]["truncate_prompt_tokens"] == 20


def test_embed_response_validation() -> None:
    good = base64.b64encode(np.ones(4096, "<f4").tobytes()).decode()
    cases: list[tuple[Any, str]] = [
        ({"data": [{"index": 0, "embedding": good}]}, "1 vectors for 2 inputs"),
        ({"data": [{"index": 0, "embedding": good}, {"index": 0, "embedding": good}]}, "indices"),
        ({"data": [{"index": 0, "embedding": good}, {"index": 1, "embedding": "!!!"}]}, "base64"),
        (
            {
                "data": [
                    {"index": 0, "embedding": good},
                    {"index": 1, "embedding": base64.b64encode(b"123").decode()},
                ]
            },
            "multiple of 4",
        ),
        (
            {"data": [{"index": 0, "embedding": good}, {"index": 1, "embedding": [1.0, 2.0]}]},
            "shape",
        ),
        ({"data": "nope"}, "no vectors"),
        ({"nodata": 1}, "malformed"),
    ]
    for payload, match in cases:
        c = EncoderClient(
            ENCODER_URL,
            None,
            transport=httpx.MockTransport(lambda _r, p=payload: httpx.Response(200, json=p)),
        )
        with pytest.raises(ClmError, match=match) as info:
            c.embed(["a", "b"])
        assert info.value.status == 200
    short = base64.b64encode(np.ones(8, "<f4").tobytes()).decode()
    c = EncoderClient(
        ENCODER_URL,
        None,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"data": [{"index": 0, "embedding": short}]})
        ),
    )
    with pytest.raises(ClmError, match="dimensions"):
        c.embed(["a"])
    # expect_dim=None accepts any consistent width (a future encoder).
    c2 = EncoderClient(
        ENCODER_URL,
        None,
        expect_dim=None,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"data": [{"index": 0, "embedding": short}]})
        ),
    )
    out, _ = c2.embed(["a"])
    assert out.shape == (1, 8) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert decode_base64_f32(short).shape == (8,)
    with pytest.raises(ValueError, match="dimensions"):
        decode_base64_f32(short, dim=4096)


def test_auth_and_unknown_model_errors(server: FakeClmServer) -> None:
    wrong = EncoderClient(ENCODER_URL, "nope", transport=server.transport())
    with pytest.raises(ClmError) as info:
        wrong.embed(["x"])
    assert (
        info.value.status == 401
        and "nope" not in str(info.value)
        and wrong.breaker.consecutive_failures == 0
    )
    c = client(server, model="other-model")
    with pytest.raises(ClmError) as info2:
        c.embed(["x"])
    assert info2.value.status == 404 and "other-model" in info2.value.message
    server.fail_next = [503, httpx.ConnectError("x")]
    slept: list[float] = []
    retrying = EncoderClient(ENCODER_URL, KEY, transport=server.transport(), sleep=slept.append)
    out, _ = retrying.embed(["x"])
    assert out.shape == (1, 4096) and slept == [0.5, 1.0]


# -- models / healthy -----------------------------------------------------------------------------


def test_models_and_healthy(server: FakeClmServer) -> None:
    c = client(server)
    models = c.models()
    assert [m["id"] for m in models] == ["qwen3-8b"]
    assert c.healthy() is True and server.paths[-1] == "/v1/models"
    server.fail_next = [httpx.ConnectError("down")]
    assert c.healthy() is False
    assert EncoderClient(ENCODER_URL, "wrong", transport=server.transport()).healthy() is False
    bad = EncoderClient(
        ENCODER_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"data": 5})),
    )
    with pytest.raises(ClmError, match="malformed /v1/models"):
        bad.models()
    bad2 = EncoderClient(
        ENCODER_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"x": 5})),
    )
    with pytest.raises(ClmError, match="malformed /v1/models"):
        bad2.models()


# -- tokenize and the guard -----------------------------------------------------------------------


def test_tokenize_server_route(server: FakeClmServer) -> None:
    c = client(server)
    assert c.tokenize("one two three") == 3
    body = server.bodies[-1]
    assert body["prompt"] == "one two three" and body["model"] == "qwen3-8b"
    assert (
        "Authorization" in server.requests[-1].headers
    )  # sent, though vLLM does not guard /tokenize
    assert c.count_tokens("a b") == (2, "server")
    bad = EncoderClient(
        ENCODER_URL,
        None,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"tokens": []})),
    )
    with pytest.raises(ClmError, match="malformed /tokenize"):
        bad.tokenize("x")


def test_tokenize_unsupported_is_remembered(server: FakeClmServer) -> None:
    server.tokenize_supported = False
    c = client(server)
    assert c.tokenize("a b c") is None
    assert c.tokenize("d e") is None
    assert server.paths.count("/tokenize") == 1  # the 404 is remembered
    n, source = c.count_tokens("a" * 10)
    assert (n, source) == (5, "chars")
    for status in (405, 501):
        cc = EncoderClient(
            ENCODER_URL,
            None,
            transport=httpx.MockTransport(
                lambda _r, s=status: httpx.Response(s, json={"error": "no"})
            ),
        )
        assert cc.tokenize("x") is None
    boom = EncoderClient(
        ENCODER_URL,
        None,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(400, json={"error": {"message": "bad"}})
        ),
    )
    with pytest.raises(ClmError) as info:
        boom.tokenize("x")
    assert info.value.status == 400


def test_token_guard_helper_and_client(server: FakeClmServer) -> None:
    c = client(server, max_len=32)
    texts = [" ".join(["w"] * 10), " ".join(["w"] * 16), " ".join(["w"] * 17), " ".join(["w"] * 40)]
    got = c.token_guard(texts)
    assert got == [
        TokenCount(10, False, "server"),
        TokenCount(16, False, "server"),
        TokenCount(17, True, "server"),  # > max_len - 16
        TokenCount(40, True, "server"),
    ]
    assert c.token_guard(texts[:1], max_len=20, margin=0) == [TokenCount(10, False, "server")]
    assert c.token_guard(texts[2:3], max_len=64)[0].truncated is False
    with pytest.raises(ValueError, match="margin"):
        c.token_guard(texts, margin=32)
    # The pure helper with a custom counter.
    words = token_guard(
        ["a b c", "a b c d"], max_len=20, margin=17, counter=lambda t: (len(t.split()), "server")
    )
    assert words == [TokenCount(3, False, "server"), TokenCount(4, True, "server")]
    assert chars_estimate("x" * 11) == math.ceil(11 / CHARS_PER_TOKEN) == 6 and GUARD_MARGIN == 16


def test_local_tokenizer_fallback(
    server: FakeClmServer, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the ``tokenize`` extra (or without a file) the guard falls back to chars/2; with a
    tokenizer.json the ``tokenizers`` path is used when the package imports. The tokenizer here is
    a tiny whitespace model written in-test, never downloaded."""
    server.tokenize_supported = False
    c = client(server)
    assert c.count_tokens("abcd") == (2, "chars")
    tok_json = tmp_path / "tokenizer.json"
    tok_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "a": 1, "b": 2},
                    "unk_token": "[UNK]",
                },
            }
        )
    )
    c2 = client(server, tokenizer_json=tok_json)
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        assert c2.count_tokens("a b zz") == (3, "chars")
    else:
        assert c2.count_tokens("a b zz") == (3, "tokenizers")
        assert c2.count_tokens("a b zz")[1] == "tokenizers"  # loaded once
    broken = client(server, tokenizer_json=tmp_path / "missing.json")
    assert broken.count_tokens("abcd")[1] == "chars"
    # With the extra hidden the file is ignored.
    import builtins

    real_import = builtins.__import__

    def no_tokenizers(name: str, *a: Any, **k: Any) -> Any:
        if name == "tokenizers":
            raise ImportError("hidden")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tokenizers)
    c3 = client(server, tokenizer_json=tok_json)
    assert c3.count_tokens("a b zz") == (3, "chars")


# -- construction ---------------------------------------------------------------------------------


def test_construction_rules_and_from_config(server: FakeClmServer) -> None:
    with pytest.raises(EndpointError, match="loopback"):
        EncoderClient("http://enc.example.org:8090", KEY)
    with pytest.raises(EndpointError, match="https"):
        EncoderClient("http://enc.example.org:8090", KEY, allow_remote=True)
    with pytest.raises(ValueError, match="max_len"):
        EncoderClient(ENCODER_URL, KEY, max_len=16)
    with pytest.raises(ValueError, match="batch"):
        EncoderClient(ENCODER_URL, KEY, batch=0)
    cfg = EncoderConfig(
        url=ENCODER_URL, api_key=SecretStr(KEY), model="qwen3-8b", max_len=2048, timeout=9.0
    )
    c = EncoderClient.from_config(cfg, transport=server.transport())
    assert c.max_len == 2048 and c.endpoint.timeout == 9.0 and c.url == ENCODER_URL
    assert c.healthy()
    assert server.bodies == [] and server.requests[-1].headers["Authorization"] == f"Bearer {KEY}"
    remote = EncoderClient.from_config(
        EncoderConfig(url="https://enc.example.org"), allow_remote=True
    )
    assert remote.url == "https://enc.example.org"
    with pytest.raises(EndpointError):
        EncoderClient.from_config(EncoderConfig(url="https://enc.example.org"))
