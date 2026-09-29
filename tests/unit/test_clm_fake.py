"""The deterministic fake (``mesa_clm.clm.fake``): platform-stable hashed n-gram vectors, the
``collapse`` knob reproducing issue #15's suffix dominance, and ``FakeClm`` implementing CLM's
engine maths through the vendored contract (``render.build_pairs`` -> scaled cosine ->
``render.answer_from_logits``) with clm-serve's usage accounting, ``clm-raw``, named heads and
``rank``. Plus the fake transport's routes and error codes."""

from __future__ import annotations

import base64
import json
import math

import httpx
import numpy as np
import pytest

from mesa_clm import render
from mesa_clm.clm.fake import FakeClm, FakeClmError, FakeEncoder
from mesa_clm.clm.headproj import RAW_SCALE, random_head
from mesa_clm.registry import ANCHOR_KEY, ANCHORS
from tests.fakes.clm_transport import CLM_URL, ENCODER_URL, LATENCY_HEADER, FakeClmServer

STATE = {
    "card": {"dataset": "DP1.10003.001.brd_countdata"},
    "scope": "column",
    "aspect": "measurement",
    "column": {
        "name": "observerDistance",
        "description": "Radial distance between the observer and the bird",
    },
}
FIT = {
    "type": "choice",
    "instructions": None,
    "criteria": {
        "PATO:0000040": "distance: A 1-D extent quality equal to the distance between two points.",
        "PATO:0000014": "color: A composite chromatic quality.",
        ANCHOR_KEY: ANCHORS["term"],
    },
}


# -- FakeEncoder ----------------------------------------------------------------------------------


def test_encoder_vectors_are_unit_deterministic_and_platform_stable() -> None:
    enc = FakeEncoder()
    v, tokens = enc.embed(
        [
            "Radial distance between the observer",
            "Radial distance between the observer",
            "site code",
        ]
    )
    assert v.shape == (3, 4096) and v.dtype == np.float32 and tokens == 5 + 5 + 2
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(v[0], v[1])
    assert float(v[0] @ v[2]) < 0.3  # unrelated texts are far apart
    again, _ = FakeEncoder().embed(["Radial distance between the observer"])
    np.testing.assert_array_equal(again[0], v[0])
    # A committed spot check: blake2b hashing (not Python's salted hash) makes this stable
    # across processes and platforms.
    idx = np.flatnonzero(v[2])
    assert (
        len(idx) > 0 and idx.tolist() == np.flatnonzero(FakeEncoder().vector("site code")).tolist()
    )
    assert FakeEncoder(seed=1).vector("site code") @ v[2] < 0.5  # the seed changes the hash space
    assert enc.healthy() and enc.calls == 1 and enc.texts_embedded == 3


def test_similar_texts_are_closer() -> None:
    enc = FakeEncoder()
    a = enc.vector("distance between the observer and the bird")
    b = enc.vector("distance between the observer and the individual")
    c = enc.vector("taxonomic identification of breeding landbirds")
    assert float(a @ b) > float(a @ c) + 0.2
    # Whitespace and case do not matter; an empty text is the zero vector, not NaN.
    np.testing.assert_array_equal(enc.vector("  Site   CODE "), enc.vector("site code"))
    assert not np.isnan(enc.vector("")).any() and float(np.linalg.norm(enc.vector(""))) == 0.0


def test_collapse_knob_reproduces_issue_15() -> None:
    """#15: mean cosine between *different* states 0.958 with a shared suffix vs 0.452 without.
    ``collapse`` moves the fake between those regimes so collapse diagnostics can be tested."""
    texts = [
        "Radial distance between the observer and the bird",
        "Scientific name associated with the taxon identifier",
        "Start date-time of the sampling event",
        "How the individual was first detected by the observer",
        "Unique identifier of the record in the database",
        "Mean air temperature over thirty minutes in degrees Celsius",
        "Soil moisture volumetric water content at depth",
        "Count of individuals observed within the plot boundary",
    ]

    def mean_pairwise(collapse: float) -> float:
        v, _ = FakeEncoder(collapse=collapse).embed(texts)
        g = v @ v.T
        return float(g[np.triu_indices(len(texts), 1)].mean())

    plain, mid, collapsed = mean_pairwise(0.0), mean_pairwise(0.5), mean_pairwise(0.95)
    assert plain < 0.5 < mid < 0.9 < collapsed <= 1.0
    # Collapse never breaks determinism or normalisation.
    v, _ = FakeEncoder(collapse=0.9).embed(texts[:2])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="collapse"):
            FakeEncoder(collapse=bad)
    with pytest.raises(ValueError):
        FakeEncoder(dim=1)
    with pytest.raises(ValueError):
        FakeEncoder(ngram=0)


def test_tokens_and_truncation_helpers() -> None:
    assert FakeEncoder.count_tokens("a  b\nc") == 3 and FakeEncoder.count_tokens("") == 0
    assert (
        FakeEncoder.truncate("a b c d e", 3) == "c d e" and FakeEncoder.truncate("a b", 3) == "a b"
    )


# -- FakeClm --------------------------------------------------------------------------------------


def test_answer_is_engine_maths_through_the_vendored_contract() -> None:
    clm = FakeClm(seed=0)
    qs = {"fit": FIT, "n": {"type": "noul", "instructions": "Is this a measurement?"}}
    out = clm.answer(STATE, qs)
    assert set(out) == {"model", "answers", "usage"} and out["model"] == "clm-latest"
    fit = out["answers"]["fit"]
    assert fit["type"] == "choice" and set(fit["probabilities"]) == set(FIT["criteria"])
    assert sum(fit["probabilities"].values()) == pytest.approx(1.0)
    # Re-derive by hand: pairs -> head projections -> scale*cos -> answer_from_logits.
    pairs = render.build_pairs(STATE, qs)
    head = clm.heads["clm-latest"]
    for qid, (s_text, keys, texts) in pairs.items():
        xs, _ = clm.encoder.embed([s_text])
        xa, _ = clm.encoder.embed(texts)
        logits = head.logits(head.project_states(xs), head.project_actions(xa))[0]
        expect = render.answer_from_logits(qs[qid], keys, logits.tolist())
        got = out["answers"][qid]
        assert got["type"] == expect["type"]
        if got["type"] == "noul":
            assert got["noul"] == pytest.approx(expect["noul"], abs=1e-12)
        else:
            for k in keys:
                assert got["probabilities"][k] == pytest.approx(
                    expect["probabilities"][k], abs=1e-12
                )
            assert got["choice"] == expect["choice"] and got["confidence"] == pytest.approx(
                expect["confidence"]
            )
    assert head.scale == 100.0  # the released head's clamped scale
    assert clm.calls == 1


def test_usage_mirrors_clm_serve_cache_semantics() -> None:
    clm = FakeClm()
    qs = {"fit": FIT, "n": {"type": "noul", "instructions": "q"}}
    first = clm.answer(STATE, qs)["usage"]
    pairs = render.build_pairs(STATE, qs)
    expected_tokens = sum(FakeEncoder.count_tokens(s) for s in {s for s, _, _ in pairs.values()})
    expected_tokens += sum(
        FakeEncoder.count_tokens(t) for _, _, texts in pairs.values() for t in texts
    )
    assert first == {"billing_units": 2, "input_tokens": expected_tokens, "output_tokens": 0}
    second = clm.answer(STATE, qs)["usage"]
    assert second == {"billing_units": 2, "input_tokens": 0, "output_tokens": 0}  # misses only
    clm.reset_cache()
    assert clm.answer(STATE, qs)["usage"]["input_tokens"] == expected_tokens
    # A shared state text across questions is embedded once (build_pairs de-duplicates).
    clm.reset_cache()
    two = {
        "a": FIT,
        "b": {**FIT, "criteria": {"x": "some other option", ANCHOR_KEY: ANCHORS["term"]}},
    }
    usage = clm.answer(STATE, two)["usage"]
    state_tokens = FakeEncoder.count_tokens(render.state_text(STATE, None))
    assert usage["input_tokens"] < 2 * state_tokens + sum(
        FakeEncoder.count_tokens(t) for q in two.values() for t in render.candidates(q)[1]
    )


def test_raw_named_heads_models_and_temperature() -> None:
    clm = FakeClm(seed=3)
    latest = clm.answer(STATE, {"fit": FIT})["answers"]["fit"]["probabilities"]
    raw = clm.answer(STATE, {"fit": FIT}, model="clm-raw")["answers"]["fit"]["probabilities"]
    assert latest != raw
    # clm-raw is cosine in the raw space at RAW_SCALE.
    xs, _ = clm.encoder.embed([render.state_text(STATE, None)])
    keys, texts = render.candidates(FIT)
    xa, _ = clm.encoder.embed(texts)
    logits = (RAW_SCALE * (xa.astype(np.float64) @ xs[0].astype(np.float64))).tolist()
    expect = render.answer_from_logits(FIT, keys, logits)["probabilities"]
    for k in keys:
        assert raw[k] == pytest.approx(expect[k], abs=1e-12)
    clm.add_head("mesa-term-v1", random_head(99, width=32))
    promoted = clm.answer(STATE, {"fit": FIT}, model="mesa-term-v1")["answers"]["fit"][
        "probabilities"
    ]
    assert promoted != latest
    assert [m["name"] for m in clm.models()] == ["clm-latest", "mesa-term-v1", "clm-raw"]
    assert clm.models()[-1]["kind"] == "raw" and clm.has("clm-raw") and not clm.has("nope")
    flat = clm.answer(STATE, {"fit": FIT}, temperature=10.0)["answers"]["fit"]["probabilities"]
    assert max(flat.values()) < max(latest.values())
    with pytest.raises(FakeClmError, match="unknown model"):
        clm.answer(STATE, {"fit": FIT}, model="nope")
    for t in (0.0, 101.0):
        with pytest.raises(FakeClmError, match="temperature"):
            clm.answer(STATE, {"fit": FIT}, temperature=t)
    with pytest.raises(FakeClmError):
        clm.answer(STATE, {})
    with pytest.raises(FakeClmError, match="question type"):
        clm.answer(STATE, {"x": {"type": "essay"}})
    with pytest.raises(FakeClmError, match="criteria"):
        clm.answer(STATE, {"x": {"type": "choice", "criteria": {}}})
    with pytest.raises(ValueError, match="raw"):
        clm.add_head("clm-raw", random_head(1, width=8))
    with pytest.raises(ValueError, match="encoder"):
        clm.add_head("small", random_head(1, width=8, hidden_size=16))
    with pytest.raises(ValueError, match="expects"):
        FakeClm(FakeEncoder(dim=64), heads={"clm-latest": random_head(1, width=8, hidden_size=16)})


def test_rank_matches_engine_rank_semantics() -> None:
    clm = FakeClm()
    cands = ["distance", "colour", "length"]
    ranked = clm.rank(STATE, cands, instructions=None)
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert sorted(r["candidate"] for r in ranked) == sorted(cands)
    probs = [r["prob"] for r in ranked]
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
    q = {
        "type": "choice",
        "instructions": None,
        "criteria": {str(i): c for i, c in enumerate(cands)},
    }
    direct = clm.answer(STATE, {"rank": q})["answers"]["rank"]["probabilities"]
    assert {r["candidate"]: r["prob"] for r in ranked} == {
        cands[int(k)]: p for k, p in direct.items()
    }
    with_q = clm.rank(STATE, cands, instructions="Which quality?")
    assert with_q != ranked  # the instructions change the state text
    for bad in ([], [""], ["a", 3]):
        with pytest.raises(FakeClmError, match="answers"):
            clm.rank(STATE, bad)  # type: ignore[arg-type]


def test_fake_is_deterministic_across_instances() -> None:
    a = FakeClm(seed=5).answer(STATE, {"fit": FIT})
    b = FakeClm(seed=5).answer(STATE, {"fit": FIT})
    assert a == b
    c = FakeClm(seed=6).answer(STATE, {"fit": FIT})
    assert c["answers"]["fit"]["probabilities"] != a["answers"]["fit"]["probabilities"]


# -- the transport --------------------------------------------------------------------------------


def _get(server: FakeClmServer, url: str, **headers: str) -> httpx.Response:
    return server(httpx.Request("GET", url, headers=headers))


def _post(server: FakeClmServer, url: str, body: object, **headers: str) -> httpx.Response:
    content = json.dumps(body).encode() if not isinstance(body, bytes) else body
    return server(httpx.Request("POST", url, content=content, headers=headers))


def test_transport_routes_and_codes() -> None:
    server = FakeClmServer(clm_api_key="k", encoder_api_key="e", latency_ms=3.0)
    bearer = {"Authorization": "Bearer k"}
    assert _get(server, f"{CLM_URL}/health").json()["ok"] is True  # unguarded
    assert _get(server, f"{CLM_URL}/v1/models").status_code == 401
    assert _get(server, f"{CLM_URL}/v1/models", Authorization="Bearer wrong").status_code == 401
    r = _get(server, f"{CLM_URL}/v1/models", **bearer)
    assert r.status_code == 200 and r.headers[LATENCY_HEADER] == "3.0"
    assert [m["name"] for m in r.json()["models"]] == ["clm-latest", "clm-raw"]
    ok = _post(
        server, f"{CLM_URL}/v1/systemone", {"state": STATE, "questions": {"fit": FIT}}, **bearer
    )
    assert ok.status_code == 200 and ok.json()["answers"]["fit"]["type"] == "choice"
    assert _post(server, f"{CLM_URL}/v1/systemone", {"state": STATE}, **bearer).status_code == 422
    assert _post(server, f"{CLM_URL}/v1/systemone", b"{not json", **bearer).status_code == 422
    assert (
        _post(
            server,
            f"{CLM_URL}/v1/systemone",
            {"state": STATE, "questions": {"fit": FIT}, "model": "x"},
            **bearer,
        ).status_code
        == 422
    )
    assert (
        _post(
            server,
            f"{CLM_URL}/v1/systemone",
            {"state": STATE, "questions": {"fit": FIT}, "temperature": 0},
            **bearer,
        ).status_code
        == 422
    )
    assert (
        _post(
            server, f"{CLM_URL}/v1/rank", {"context": "c", "answers": ["a", "b"]}, **bearer
        ).json()["ranked"][0]["rank"]
        == 1
    )
    assert _post(server, f"{CLM_URL}/v1/rank", {"context": "c"}, **bearer).status_code == 422
    assert _get(server, f"{CLM_URL}/nothing").status_code == 404
    assert _get(server, f"{CLM_URL}/v1/nothing", **bearer).status_code == 404
    # The encoder side, told apart by port.
    assert _get(server, f"{ENCODER_URL}/health").status_code == 200
    assert _get(server, f"{ENCODER_URL}/v1/models").status_code == 401
    assert (
        _get(server, f"{ENCODER_URL}/v1/models", Authorization="Bearer e").json()["data"][0]["id"]
        == "qwen3-8b"
    )
    emb = _post(
        server,
        f"{ENCODER_URL}/v1/embeddings",
        {"model": "qwen3-8b", "input": "a b", "encoding_format": "base64"},
        Authorization="Bearer e",
    )
    vec = np.frombuffer(base64.b64decode(emb.json()["data"][0]["embedding"]), "<f4")
    assert (
        emb.status_code == 200
        and vec.shape == (4096,)
        and emb.json()["usage"]["prompt_tokens"] == 2
    )
    assert (
        _post(
            server,
            f"{ENCODER_URL}/v1/embeddings",
            {"model": "other", "input": "a"},
            Authorization="Bearer e",
        ).status_code
        == 404
    )
    assert (
        _post(
            server,
            f"{ENCODER_URL}/v1/embeddings",
            {"model": "qwen3-8b", "input": 5},
            Authorization="Bearer e",
        ).status_code
        == 400
    )
    assert (
        _post(
            server,
            f"{ENCODER_URL}/v1/embeddings",
            {"model": "qwen3-8b", "input": "a", "encoding_format": "hex"},
            Authorization="Bearer e",
        ).status_code
        == 400
    )
    assert (
        _post(server, f"{ENCODER_URL}/v1/embeddings", b"nope", Authorization="Bearer e").status_code
        == 400
    )
    tok = _post(
        server, f"{ENCODER_URL}/tokenize", {"model": "qwen3-8b", "prompt": "a b c"}
    )  # unguarded, like vLLM
    assert tok.json() == {"count": 3, "max_model_len": 4096, "tokens": [0, 1, 2]}
    assert _post(server, f"{ENCODER_URL}/tokenize", {"model": "qwen3-8b"}).status_code == 400
    assert _get(server, f"{ENCODER_URL}/v1/other", Authorization="Bearer e").status_code == 404
    assert _get(server, f"{ENCODER_URL}/other").status_code == 404
    right = _post(
        server,
        f"{ENCODER_URL}/v1/embeddings",
        {"model": "qwen3-8b", "input": ["a b c d"], "truncate_prompt_tokens": 2},
        Authorization="Bearer e",
    )
    assert (
        right.status_code == 200 and server.embedded_texts[-1] == "a b"
    )  # right truncation without the PR #6 field
    server.tokenize_supported = False
    assert _post(server, f"{ENCODER_URL}/tokenize", {"prompt": "x"}).status_code == 404
    assert len(server.requests) == len(server.paths)
    open_server = FakeClmServer()  # no keys configured: everything is open
    assert _get(open_server, f"{CLM_URL}/v1/models").status_code == 200


def test_transport_scripted_failures() -> None:
    server = FakeClmServer()
    server.fail_next = [503, RuntimeError("boom")]
    assert _get(server, f"{CLM_URL}/health").status_code == 503
    with pytest.raises(RuntimeError, match="boom"):
        _get(server, f"{CLM_URL}/health")
    assert _get(server, f"{CLM_URL}/health").status_code == 200
    assert math.isclose(float(_get(server, f"{CLM_URL}/health").headers[LATENCY_HEADER]), 12.5)
