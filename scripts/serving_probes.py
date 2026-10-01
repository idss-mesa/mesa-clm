#!/usr/bin/env python3
"""Live serving probes for milestone M1 track A (plan §6.1, §6.4, §6.6, §6.8, §8 M1-A).

Runs against the two loopback units on the GPU host (``mesa-clm-encoder.service`` on :8090,
``mesa-clm-serve.service`` on :8700) with mesa-clm's own clients (``EncoderClient``,
``ClmHttpClient``, ``HeadProjector``) and writes one JSON record::

    uv run python scripts/serving_probes.py            # full run -> bench/results/<date>/serving_m1.json
    uv run python scripts/serving_probes.py --quick    # binds, 401s, health, one systemone

Sections of the full run: (a) loopback binds; (b) the auth matrix (every route with no key, a
wrong key, the other unit's key and the right key; open routes); (c) served models; (d) upstream
drift probes (the CLM README quickstart at ``bb42c6c5`` and the model-card tides rank); (e) the
5,000-token left-truncation probe; (f) golden encoder vectors (``.local/serving/
encoder_golden.npz``); (g) clm-serve ``/v1/systemone`` vs the local head projection over 50
(state, question) pairs for ``clm-latest`` and ``clm-raw``; (h) memory footprint; (i) latency
p50/p95; plus versions, the image digest and ``encoder_fp``.

Keys are read from the 0600 files under ``~/.mesa/clm/secrets`` and never printed, logged or
written: before anything is written the serialised output is checked for both key values and
the run aborts if either appears. Only loopback addresses are recorded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np

from mesa_clm import framings, render
from mesa_clm.cards import load_card
from mesa_clm.clm.encoder import EncoderClient, decode_base64_f32
from mesa_clm.clm.fingerprint import EncoderSpec, encoder_fp
from mesa_clm.clm.headproj import RAW_SCALE, HeadProjector
from mesa_clm.clm.http import Choice, ClmHttpClient, Noul, Score
from mesa_clm.registry import ANCHOR_KEY, ANCHORS, ONTOLOGY_REGISTRY
from mesa_clm.states import target_state
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-29"
ENC_URL = "http://127.0.0.1:8090"
CLM_URL = "http://127.0.0.1:8700"
SECRETS = Path("~/.mesa/clm/secrets").expanduser()
HEAD_NPZ = Path("~/.mesa/clm/heads/npz/b2b4a8c9.npz").expanduser()
SERVE_PY = Path("~/.mesa/clm/serve/.venv/bin/python").expanduser()
TOKENIZER_JSON = Path(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
).expanduser()
SNAPSHOT = ROOT / "bench/snapshots/2026-09-29.parquet"
LOCK = ROOT / "serving/serving.lock.json"
CARDS = ROOT / "tests/fixtures/cards"
OLS = ROOT / "tests/fixtures/ols"
MODEL = "qwen3-8b"
MAX_LEN = 4096
WRONG_KEY = "mesa-clm-probe-wrong-key"
# Dotted quads allowed in the output: loopback, and the wildcard peer column of `ss`.
LOOPBACK_OR_WILDCARD = frozenset({"127.0.0.1", "0.0.0.0"})  # noqa: S104

# Footprint reference points recorded during the M1-A bring-up (caller-provided, same host).
MEMAVAILABLE_PRE_START_GIB = 107.8
MEMAVAILABLE_ENCODER_ONLY_GIB = 80.3

# Upstream drift references (plan §6.8; RESEARCH.md issue #15 and the model card).
ISSUE15 = {"urgency": 0.8366, "billing": 0.9888, "frustration": 2.0000, "tolerance": 0.01}
README_PRINTED = {"urgency": 0.41022, "billing": 0.93878, "frustration": 1.98386}
TIDES_CARD = {"prob": 0.993, "tolerance": 0.005, "readme_prob": 0.997}
TIDES = (
    "What causes tides on Earth?",
    ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."],
)

# Twenty fixed short texts for the golden encoder vectors (plan §6.8 "golden cosine >= 0.9999").
GOLDEN_TEXTS: tuple[str, ...] = (
    "hello world",
    "Customer: my invoice was charged twice and nobody answers the phone!",
    "What causes tides on Earth?",
    "The Moon's gravitational pull.",
    "None of these terms is the right concept for this target.",
    "None of these ontologies has a suitable term for this target.",
    "NEON dataset Breeding landbird point counts. Column observerDistance: Radial distance "
    "between the observer and the individual(s) being observed (meter). measurement ontology "
    "term:",
    "distance: A spatial quality inhering in a bearer by virtue of the bearer's distance from "
    "another entity.",
    "scope: column\n\naspect: measurement",
    "Santa Rita Experimental Range NEON, D14, semi-arid desert grassland/shrubland",
    "Harvard Forest & Quabbin Watershed NEON",
    "scientificName: the full scientific name of the organism, including authorship",
    "Is the candidate ontology term a correct annotation for the described column, site or "
    "dataset: the right concept at an appropriate specificity, not merely related?",
    "true: Yes. This is true: Is this urgent?",
    "false: No. This is false: Is this urgent?",
    "ENVO: environments, biomes, habitats and environmental materials",
    "Carabid beetles collected in pitfall traps, sorted and identified by parataxonomists",
    "Température moyenne de l'air: 21,5 °C; précipitations 3 mm",
    "numeric, n=14670, min=1, median=59, max=999",
    "x",
)


# -- small helpers -------------------------------------------------------------------------------


def read_key(name: str) -> str:
    """A bearer key from its 0600 file; the value never leaves this process."""
    return (SECRETS / f"{name}.key").read_text(encoding="utf-8").strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: Sequence[str], timeout: float = 60.0) -> str:
    """stdout of a read-only command ('' on failure; the failure is not a secret)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<{type(exc).__name__}>"
    return out.stdout.strip()


def sg_docker(command: str) -> str:
    return run(["sg", "docker", "-c", command])


def meminfo_gib(field: str = "MemAvailable") -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(field + ":"):
            return round(int(line.split()[1]) / 1024**2, 2)
    raise KeyError(field)


def percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "min_ms": round(float(arr.min()), 1),
        "max_ms": round(float(arr.max()), 1),
        "mean_ms": round(float(arr.mean()), 1),
    }


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))


class Http:
    """Raw status probes (no retries, no proxies) with an explicit Authorization choice."""

    def __init__(self) -> None:
        self.client = httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0, connect=5.0))

    def status(self, method: str, url: str, key: str | None = None, body: Any = None) -> int:
        headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
        try:
            r = self.client.request(method, url, headers=headers, json=body)
        except httpx.TransportError:
            return -1
        return r.status_code


# -- (a) binds -----------------------------------------------------------------------------------


def probe_binds() -> dict[str, Any]:
    out = run(["ss", "-Hltn", "( sport = :8090 or sport = :8700 )"])
    listeners = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            listeners.append(parts[3])
    loopback = [a for a in listeners if a.startswith(("127.0.0.1:", "[::1]:"))]
    return {
        "command": "ss -Hltn '( sport = :8090 or sport = :8700 )'",
        "listeners": sorted(listeners),
        "loopback_only": bool(listeners) and len(loopback) == len(listeners),
        "ports_seen": sorted({a.rsplit(":", 1)[1] for a in listeners}),
        "docker_port": sg_docker("docker port mesa-clm-encoder"),
    }


# -- (b) auth matrix -----------------------------------------------------------------------------


def probe_auth(http: Http, enc_key: str, clm_key: str) -> dict[str, Any]:
    emb_body = {"model": MODEL, "input": ["probe"]}
    so_body = {
        "state": "probe",
        "model": "clm-latest",
        "questions": {"q": {"type": "noul", "instructions": "Is this a probe?"}},
    }
    rank_body = {"context": "probe", "question": None, "answers": ["a", "b"], "model": "clm-latest"}
    guarded = [
        ("encoder", "GET", f"{ENC_URL}/v1/models", None, enc_key, clm_key),
        ("encoder", "POST", f"{ENC_URL}/v1/embeddings", emb_body, enc_key, clm_key),
        ("clm-serve", "GET", f"{CLM_URL}/v1/models", None, clm_key, enc_key),
        ("clm-serve", "POST", f"{CLM_URL}/v1/systemone", so_body, clm_key, enc_key),
        ("clm-serve", "POST", f"{CLM_URL}/v1/rank", rank_body, clm_key, enc_key),
    ]
    rows = []
    for unit, method, url, body, right, other in guarded:
        rows.append(
            {
                "unit": unit,
                "route": f"{method} {url}",
                "no_key": http.status(method, url, None, body),
                "wrong_key": http.status(method, url, WRONG_KEY, body),
                "other_units_key": http.status(method, url, other, body),
                "right_key": http.status(method, url, right, body),
            }
        )
    all_401 = all(
        r["no_key"] == 401 and r["wrong_key"] == 401 and r["other_units_key"] == 401 for r in rows
    )
    tok_body = {"model": MODEL, "prompt": "hello world"}
    open_routes = {
        "encoder GET /health": http.status("GET", f"{ENC_URL}/health"),
        "encoder GET /metrics": http.status("GET", f"{ENC_URL}/metrics"),
        "encoder GET /version": http.status("GET", f"{ENC_URL}/version"),
        "encoder POST /tokenize (no key)": http.status(
            "POST", f"{ENC_URL}/tokenize", None, tok_body
        ),
        "encoder POST /tokenize (key)": http.status(
            "POST", f"{ENC_URL}/tokenize", enc_key, tok_body
        ),
        "clm-serve GET /health": http.status("GET", f"{CLM_URL}/health"),
        "clm-serve GET / (--no-ui)": http.status("GET", f"{CLM_URL}/"),
    }
    # Routes the pooling runner registers outside /v1 and /v2 (startup log), unauthenticated.
    unguarded_bodies = {
        "POST /pooling": {"model": MODEL, "input": "probe"},
        "POST /invocations": {"model": MODEL, "input": "probe"},
        "POST /score": {"model": MODEL, "text_1": "a", "text_2": "b"},
        "POST /rerank": {"model": MODEL, "query": "a", "documents": ["b"]},
        "POST /detokenize": {"model": MODEL, "tokens": [14990]},
        "POST /v1/score": {"model": MODEL, "text_1": "a", "text_2": "b"},
        "POST /v2/embed": {
            "model": MODEL,
            "texts": ["probe"],
            "input_type": "search_query",
            "embedding_types": ["float"],
        },
    }
    non_v1 = {
        f"encoder {route} (no key)": http.status(
            route.split()[0], f"{ENC_URL}{route.split()[1]}", None, body
        )
        for route, body in unguarded_bodies.items()
    }
    non_v1["encoder GET /load (no key)"] = http.status("GET", f"{ENC_URL}/load")
    non_v1["encoder GET /openapi.json (no key)"] = http.status("GET", f"{ENC_URL}/openapi.json")
    return {
        "guarded": rows,
        "named_v1_routes_all_401_without_valid_key": all_401,
        "right_key_all_200": all(r["right_key"] == 200 for r in rows),
        "open_routes": open_routes,
        "tokenize": {
            "works_under_runner_pooling": open_routes["encoder POST /tokenize (no key)"] == 200,
            "guarded_by_key": open_routes["encoder POST /tokenize (no key)"] == 401,
        },
        "encoder_routes_outside_v1_v2_without_key": non_v1,
        "note": (
            "vLLM's API-key middleware guards only the /v1, /v2, /inference and /cohere prefixes; "
            "a 200 above without a key means that route is reachable by any local user on "
            "loopback."
        ),
    }


# -- (c) models ----------------------------------------------------------------------------------


def probe_models(enc: EncoderClient, clm: ClmHttpClient, http: Http) -> dict[str, Any]:
    enc_models = enc.models()
    clm_models = clm.models()
    health = http.client.get(f"{CLM_URL}/health").json()
    return {
        "encoder": [
            {k: m.get(k) for k in ("id", "root", "max_model_len", "owned_by")} for m in enc_models
        ],
        "encoder_has_qwen3_8b": any(m.get("id") == MODEL for m in enc_models),
        "clm_serve": [m.get("name") for m in clm_models],
        "clm_serve_has_latest_and_raw": {"clm-latest", "clm-raw"}
        <= {m.get("name") for m in clm_models},
        "clm_serve_health": {
            "ok": health.get("ok"),
            "embedder": health.get("embedder"),
            "models": health.get("models"),
            "cache_device": (health.get("cache") or {}).get("device"),
            "cache_reserved_mb": (health.get("cache") or {}).get("reserved_mb"),
        },
    }


# -- (d) upstream drift --------------------------------------------------------------------------


def probe_drift(clm: ClmHttpClient) -> dict[str, Any]:
    # The README quickstart at bb42c6c5, verbatim (Noul/Choice/Score are wire-identical to
    # CLM's client.py; tests/unit/test_wire_parity.py).
    state = "Customer: my invoice was charged twice and nobody answers the phone!"
    questions = {
        "urgency": Noul(instructions="Is this urgent?"),
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"},
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    }
    r = clm.system_one(state, questions, model="clm-latest")
    urgency = r.answers["urgency"].noul  # type: ignore[union-attr]
    dept = r.answers["department"]
    frus = r.answers["frustration"]
    observed = {
        "urgency": round(float(urgency), 4),
        "department": dept.choice,  # type: ignore[union-attr]
        "billing": round(float(dept.probabilities["billing"]), 4),
        "frustration": round(float(frus.score), 4),  # type: ignore[union-attr]
        "input_tokens": r.usage.input_tokens,
        "latency_ms_header": r.latency_ms,
    }
    tol = ISSUE15["tolerance"]
    within = {
        k: abs(observed[k] - ISSUE15[k]) <= tol for k in ("urgency", "billing", "frustration")
    }
    ranked = clm.rank(TIDES[0], None, TIDES[1], model="clm-latest")
    top = ranked[0]
    return {
        "source": (
            "README.md at Contrastive-LM/CLM bb42c6c5 (gh api .../contents/README.md?ref=bb42c6c5...)"
            "; issue #15 comment 2026-09-28 (vLLM 0.30.0, max-model-len 8192, GB10)"
        ),
        "quickstart": {
            "observed": observed,
            "issue15": {k: ISSUE15[k] for k in ("urgency", "billing", "frustration")},
            "readme_printed": README_PRINTED,
            "tolerance": tol,
            "within_tolerance_of_issue15": within,
            "pass": all(within.values()),
        },
        "tides_rank": {
            "top_candidate": top.candidate,
            "top_prob": round(float(top.prob), 4),
            "ranked": [
                {"rank": c.rank, "candidate": c.candidate, "prob": round(float(c.prob), 6)}
                for c in ranked
            ],
            "model_card": TIDES_CARD["prob"],
            "readme": TIDES_CARD["readme_prob"],
            "tolerance": TIDES_CARD["tolerance"],
            "pass": top.candidate == TIDES[1][0]
            and abs(float(top.prob) - TIDES_CARD["prob"]) <= TIDES_CARD["tolerance"],
        },
    }


# -- (e) truncation ------------------------------------------------------------------------------


def _embed_raw(enc: EncoderClient, body: dict[str, Any]) -> tuple[np.ndarray, int]:
    j, _ = enc.endpoint.post_json("/v1/embeddings", body)
    vec = decode_base64_f32(j["data"][0]["embedding"], dim=4096)
    return vec, int((j.get("usage") or {}).get("prompt_tokens") or 0)


def _tokenize(http: Http, text: str) -> list[int]:
    r = http.client.post(f"{ENC_URL}/tokenize", json={"model": MODEL, "prompt": text})
    r.raise_for_status()
    return [int(t) for t in r.json()["tokens"]]


def _bounded(enc_key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """One /v1/embeddings request with a hard timeout and no retry: status and seconds, or
    ``timeout`` (a hung request; vLLM aborts it when the client disconnects)."""
    t0 = time.perf_counter()
    try:
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(timeout, connect=5.0)) as c:
            r = c.post(
                f"{ENC_URL}/v1/embeddings",
                json=body,
                headers={"Authorization": f"Bearer {enc_key}"},
            )
        outcome: dict[str, Any] = {"status": r.status_code}
        if r.status_code == 200:
            outcome["prompt_tokens"] = (r.json().get("usage") or {}).get("prompt_tokens")
    except httpx.TimeoutException:
        outcome = {"status": "timeout"}
    outcome["seconds"] = round(time.perf_counter() - t0, 2)
    time.sleep(3.0)  # let the server abort a disconnected request before the next one
    return outcome


def probe_truncation(enc: EncoderClient, http: Http, enc_key: str) -> dict[str, Any]:
    """Left truncation (PR #6) on a ~5,000-token text, and the max-model-len boundary.

    At ``truncate_prompt_tokens = 4096 = --max-model-len`` the truncated request never
    completes on this image (found in M1-A), so the boundary is recorded with hard timeouts and
    the left-truncation proof runs at 4095 through ``EncoderClient(max_len=4095)``.
    """
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    corpus = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(CARDS.glob("*.md")))
    head_ids = _tokenize(http, corpus)[:4990]
    text = (
        tok.decode(head_ids)
        + "\n\nTarget column: observerDistance (meter), the radial distance to the bird."
    )
    ids = _tokenize(http, text)
    b64 = {"model": MODEL, "encoding_format": "base64"}

    boundary: dict[str, Any] = {}
    for n in (2048, 2049, MAX_LEN - 1, MAX_LEN):
        boundary[f"token_ids_{n}"] = [
            _bounded(enc_key, {**b64, "input": [ids[:n]]}, 30.0) for _ in range(2)
        ]
    boundary["text_truncate_4096_left"] = _bounded(enc_key, {**enc.request_body([text])}, 60.0)
    boundary["text_no_truncation"] = _bounded(enc_key, {**b64, "input": [text]}, 30.0)

    limit = MAX_LEN - 1
    left_client = EncoderClient(ENC_URL, enc_key, max_len=limit, retries=1)
    v_left, t_left = left_client.embed([text])
    v_left = v_left[0]
    v_tail_ids, t_tail = _embed_raw(enc, {**b64, "input": [ids[-limit:]]})
    tail_text = tok.decode(ids[-limit:])
    v_tail_text, t_tail_text = _embed_raw(enc, {**b64, "input": [tail_text]})
    v_right, t_right = _embed_raw(enc, {**b64, "input": [text], "truncate_prompt_tokens": limit})
    v_head_ids, _ = _embed_raw(enc, {**b64, "input": [ids[:limit]]})
    c_left_tail = cos(v_left, v_tail_ids)
    hangs = [
        k
        for k, v in boundary.items()
        if any(o["status"] == "timeout" for o in (v if isinstance(v, list) else [v]))
    ]
    return {
        "text_sha256": sha256_text(text),
        "text_tokens": len(ids),
        "corpus": "tests/fixtures/cards/*.md concatenated, first 4990 tokens, plus a target line",
        "max_model_len_boundary": boundary,
        "requests_that_timed_out": hangs,
        "left": {
            "request": f"EncoderClient(max_len={limit}).embed: truncate_prompt_tokens={limit}, "
            "truncation_side=left",
            "prompt_tokens_charged": t_left,
            f"cos_vs_last_{limit}_token_ids": round(c_left_tail, 7),
            "cos_vs_decoded_tail_text": round(cos(v_left, v_tail_text), 7),
            "decoded_tail_text_tokens_charged": t_tail_text,
            "tail_ids_tokens_charged": t_tail,
            "gate": 0.9999,
            "pass": c_left_tail >= 0.9999,
        },
        "right_default": {
            "request": f"truncate_prompt_tokens={limit} without truncation_side",
            "prompt_tokens_charged": t_right,
            f"cos_vs_first_{limit}_token_ids": round(cos(v_right, v_head_ids), 7),
            "cos_vs_left_truncated": round(cos(v_right, v_left), 7),
        },
    }


# -- (f) goldens ---------------------------------------------------------------------------------


def probe_goldens(enc: EncoderClient, out: Path) -> dict[str, Any]:
    texts = list(GOLDEN_TEXTS)
    batch, tokens = enc.embed(texts)
    single = np.stack([enc.embed([t])[0][0] for t in texts])
    single2 = np.stack([enc.embed([t])[0][0] for t in texts])
    again, _ = enc.embed(texts)
    shas = [sha256_text(t) for t in texts]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        np.savez(
            fh,
            texts=np.array(texts),
            text_sha256=np.array(shas),
            vectors=batch.astype(np.float32),
            encoder_fp=np.array(json.loads(LOCK.read_text())["encoder_fp"]),
            route=np.array("vllm"),
        )
    cs_single = [cos(batch[i], single[i]) for i in range(len(texts))]
    cs_again = [cos(batch[i], again[i]) for i in range(len(texts))]
    cs_single2 = [cos(single[i], single2[i]) for i in range(len(texts))]
    return {
        "file": str(out.relative_to(ROOT)),
        "file_sha256": sha256_file(out),
        "n": len(texts),
        "dtype": "float32",
        "prompt_tokens": tokens,
        "texts_sha256": shas,
        "batch_vs_one_by_one": {
            "min_cos": round(min(cs_single), 7),
            "mean_cos": round(float(np.mean(cs_single)), 7),
        },
        "batch_vs_batch_repeat": {
            "min_cos": round(min(cs_again), 7),
            "mean_cos": round(float(np.mean(cs_again)), 7),
        },
        "one_by_one_vs_one_by_one_repeat": {
            "min_cos": round(min(cs_single2), 7),
            "mean_cos": round(float(np.mean(cs_single2)), 7),
            "bitwise_identical": int(np.sum(np.all(single == single2, axis=1))),
        },
        "golden_gate_0_9999": {
            "batch_vs_one_by_one_pass": min(cs_single) >= 0.9999,
            "batch_vs_batch_repeat_pass": min(cs_again) >= 0.9999,
            "one_by_one_repeat_pass": min(cs_single2) >= 0.9999,
        },
    }


# -- (g) head-projection parity ------------------------------------------------------------------


def _snapshot_rows(task_id: str) -> list[tuple[str, dict[str, Any], str]]:
    con = duckdb.connect()
    rows = con.execute(
        "select target_sha256, state_json, label_id from read_parquet(?) where task_id = ? "
        "order by target_sha256, label_id",
        [str(SNAPSHOT), task_id],
    ).fetchall()
    con.close()
    return [(r[0], json.loads(r[1]), r[2]) for r in rows]


def parity_pairs(n_term: int = 30, n_onto: int = 10, n_noul: int = 10) -> list[dict[str, Any]]:
    """50 real (state, questions) requests from the committed label snapshot: term.fits F7
    rank_fit groups, column.ontology_fits F7 groups (aspect-masked registry) and F1 noul
    controls; deterministic order (target_sha256, label_id)."""
    pairs: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for tsha, state, _ in _snapshot_rows("term.fits"):
        groups.setdefault(tsha, []).append(state)
    f7 = framings.framing("term.fits", "F7")
    f1 = framings.framing("term.fits", "F1")
    for tsha in sorted(groups)[:n_term]:
        states = groups[tsha]
        cands, seen = [], set()
        for s in states:
            c = framings.FramingCandidate.from_term(s["candidate"])
            if c.key not in seen:
                seen.add(c.key)
                cands.append(c)
        ctx, qs = framings.build_request(f7, states[0], cands)
        pairs.append({"id": f"term.fits/F7/{tsha[:12]}", "state": ctx, "questions": qs})
    for tsha in sorted(groups)[n_term : n_term + n_noul]:
        s = groups[tsha][0]
        c = framings.FramingCandidate.from_term(s["candidate"])
        ctx, qs = framings.build_requests(f1, s, [c])[0]
        pairs.append({"id": f"term.fits/F1/{tsha[:12]}", "state": ctx, "questions": qs})
    onto: dict[str, dict[str, Any]] = {}
    for tsha, state, _ in _snapshot_rows("column.ontology_fits"):
        onto.setdefault(tsha, state)
    fo = framings.framing("column.ontology_fits", "F7")
    for tsha in sorted(onto)[:n_onto]:
        s = {**onto[tsha], "scope": onto[tsha].get("scope") or TASKS["column.ontology_fits"].scope}
        allowed = [e.id for e in ONTOLOGY_REGISTRY if s["aspect"] in e.aspects]
        ctx, qs = framings.build_request(fo, s, framings.ontology_candidates(allowed))
        pairs.append({"id": f"column.ontology_fits/F7/{tsha[:12]}", "state": ctx, "questions": qs})
    return pairs


def _probs(answer: Any) -> dict[str, float]:
    """Probabilities of a typed clm-serve answer or a vendored answer dict."""
    if isinstance(answer, dict):
        return render.probabilities_of(answer)
    return {str(k): float(v) for k, v in answer.probabilities.items()}


def _softmax(logits: np.ndarray) -> list[float]:
    return render.softmax([float(x) for x in logits])


def local_answers(
    enc: EncoderClient,
    head: HeadProjector,
    state: Any,
    questions: dict[str, Any],
    model: str,
    sink: list[tuple[np.ndarray, ...]] | None = None,
) -> dict[str, dict[str, Any]]:
    """What clm-serve computes, done locally: build_pairs -> encoder (4096-d, L2) -> heads ->
    scale * cos -> answer_from_logits (clm-raw: 100 * cos in the raw space). ``sink`` collects
    (state vector, candidate vectors, projections, probabilities) for the head-only check."""
    out = {}
    for qid, (stext, keys, texts) in render.build_pairs(state, questions).items():
        vs, _ = enc.embed([stext])
        va, _ = enc.embed(texts)
        if model == "clm-raw":
            logits = RAW_SCALE * (va.astype(np.float64) @ vs[0].astype(np.float64))
        else:
            zs, za = head.project_states(vs), head.project_actions(va)
            logits = head.logits(zs, za)[0]
            if sink is not None:
                sink.append((vs[0], va, zs[0], za, np.asarray(_softmax(logits))))
        out[qid] = render.answer_from_logits(questions[qid], keys, [float(x) for x in logits])
    return out


def _diff_stats(
    a: list[dict[str, dict[str, float]]], b: list[dict[str, dict[str, float]]], ids: list[str]
) -> dict[str, Any]:
    diffs, agree, worst = [], 0, ("", -1.0)
    for pid, qa, qb in zip(ids, a, b, strict=True):
        for qid, pa in qa.items():
            pb = qb[qid]
            d = max(abs(pa[k] - pb[k]) for k in pa)
            diffs.append(d)
            agree += max(pa, key=pa.__getitem__) == max(pb, key=pb.__getitem__)
            if d > worst[1]:
                worst = (pid, d)
    return {
        "max_abs_prob_diff": float(max(diffs)),
        "mean_abs_prob_diff": float(np.mean(diffs)),
        "p95_abs_prob_diff": float(np.percentile(diffs, 95)),
        "top_choice_agreement": f"{agree}/{len(diffs)}",
        "worst_pair": worst[0],
    }


def head_check(sink: list[tuple[np.ndarray, ...]], out: Path) -> dict[str, Any]:
    """The same encoder vectors through CLM's own torch ``HeadPair`` (serve venv, CPU): the
    head-projection parity with encoder noise removed (plan: headproj <= 1e-5)."""
    offsets = np.cumsum([0] + [len(item[1]) for item in sink])
    with out.open("wb") as fh:
        np.savez(
            fh,
            states=np.stack([item[0] for item in sink]).astype(np.float32),
            actions=np.concatenate([item[1] for item in sink]).astype(np.float32),
            offsets=offsets.astype(np.int64),
            zs=np.stack([item[2] for item in sink]).astype(np.float64),
            za=np.concatenate([item[3] for item in sink]).astype(np.float64),
            probs=np.concatenate([item[4] for item in sink]).astype(np.float64),
        )
    proc = subprocess.run(
        [str(SERVE_PY), str(ROOT / "scripts/fallback_parity.py"), "head", "--vectors", str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip().splitlines()[-1:] or ["failed"]}
    return {"vectors_file": str(out.relative_to(ROOT)), **json.loads(proc.stdout)}


def probe_parity(
    enc: EncoderClient, clm: ClmHttpClient, head: HeadProjector, pairs_out: Path
) -> dict[str, Any]:
    """clm-serve vs the local route, the local route against itself (the encoder's own
    run-to-run noise floor), and the head alone on identical vectors."""
    pairs = parity_pairs()
    ids = [p["id"] for p in pairs]
    result: dict[str, Any] = {
        "n_pairs": len(pairs),
        "mix": {
            "term.fits F7": sum(i.startswith("term.fits/F7") for i in ids),
            "term.fits F1 noul": sum(i.startswith("term.fits/F1") for i in ids),
            "column.ontology_fits F7": sum(i.startswith("column.ontology") for i in ids),
        },
        "head_npz_sha256": sha256_file(HEAD_NPZ),
        "head_source_sha256": head.source_sha256,
        "head_scale": head.scale,
        "gate": 1e-4,
    }
    recorded: list[dict[str, Any]] = []
    sink: list[tuple[np.ndarray, ...]] = []
    for model in ("clm-latest", "clm-raw"):
        served, local_a, local_b = [], [], []
        for p in pairs:
            r = clm.system_one(p["state"], p["questions"], model=model)
            served.append({q: _probs(a) for q, a in r.answers.items()})
            a = local_answers(
                enc,
                head,
                p["state"],
                p["questions"],
                model,
                sink if model == "clm-latest" else None,
            )
            b = local_answers(enc, head, p["state"], p["questions"], model)
            local_a.append({q: _probs(v) for q, v in a.items()})
            local_b.append({q: _probs(v) for q, v in b.items()})
        e2e = _diff_stats(served, local_a, ids)
        result[model] = {
            "served_vs_local": {**e2e, "pass": e2e["max_abs_prob_diff"] <= 1e-4},
            "local_vs_local_repeat_noise_floor": _diff_stats(local_a, local_b, ids),
        }
        if model == "clm-latest":
            recorded = [{"id": i, "served": s} for i, s in zip(ids, served, strict=True)]
    result["head_only_clm_latest"] = head_check(sink, pairs_out.with_name("parity_vectors.npz"))
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.write_text(
        json.dumps({"pairs": pairs, "served_clm_latest": recorded}, indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result["pairs_file"] = str(pairs_out.relative_to(ROOT))
    return result


# -- (h) footprint -------------------------------------------------------------------------------


def _unit_prop(unit: str, prop: str) -> str:
    return run(["systemctl", "--user", "show", "-p", prop, "--value", unit])


def _proc_status(pid: str) -> dict[str, str]:
    out = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
                out[key] = value.strip()
    except OSError:
        pass
    return out


def probe_footprint() -> dict[str, Any]:
    now = meminfo_gib()
    stats_raw = sg_docker("docker stats --no-stream --format '{{json .}}' mesa-clm-encoder")
    try:
        stats = json.loads(stats_raw)
        stats = {k: stats[k] for k in ("MemUsage", "MemPerc", "CPUPerc", "PIDs") if k in stats}
    except json.JSONDecodeError:
        stats = {"raw": stats_raw}
    pid = _unit_prop("mesa-clm-serve.service", "MainPID")
    serve_status = _proc_status(pid)
    apps = run(
        [
            "nvidia-smi",
            "--query-compute-apps=process_name,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    return {
        "memavailable_now_gib": now,
        "memavailable_pre_start_gib": MEMAVAILABLE_PRE_START_GIB,
        "memavailable_encoder_only_gib": MEMAVAILABLE_ENCODER_ONLY_GIB,
        "delta_both_units_gib": round(MEMAVAILABLE_PRE_START_GIB - now, 2),
        "delta_encoder_gib": round(MEMAVAILABLE_PRE_START_GIB - MEMAVAILABLE_ENCODER_ONLY_GIB, 2),
        "delta_clm_serve_gib": round(MEMAVAILABLE_ENCODER_ONLY_GIB - now, 2),
        "budget_vllm_gib": 24.3,
        "docker_stats": stats,
        "nvidia_smi_compute_apps": apps.splitlines(),
        "nvidia_smi_memory_used": run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]
        ),
        "clm_serve_proc_status": serve_status,
        "clm_serve_unit_memory_current_bytes": _unit_prop(
            "mesa-clm-serve.service", "MemoryCurrent"
        ),
        "clm_serve_unit_memory_peak_bytes": _unit_prop("mesa-clm-serve.service", "MemoryPeak"),
        "encoder_log": encoder_log_facts(),
    }


def encoder_log_facts() -> dict[str, Any]:
    """Selected facts from the encoder's most recent start (messages only, no addresses)."""
    since = _unit_prop("mesa-clm-encoder.service", "ActiveEnterTimestamp")
    log = run(
        [
            "journalctl",
            "--user",
            "-u",
            "mesa-clm-encoder.service",
            "--no-pager",
            "-o",
            "cat",
            "--since",
            " ".join(since.split()[1:3]) if since else "today",
        ]
    )
    facts: dict[str, Any] = {}
    patterns = {
        "model_loading": r"Model loading took ([\d.]+ GiB memory and [\d.]+ seconds)",
        "kv_cache": r"Available KV cache memory: ([\d.]+ GiB)",
        "free_on_startup": r"Free memory on device \(([\d./]+ GiB)\) on startup",
        "seq_pooling_type": r"seq_pooling_type='(\w+)'",
        "enable_prefix_caching": r"enable_prefix_caching=(\w+)",
        "resolved_pooling": r"Resolved pooling config: (pooling_type=[^,]+, [^,]+, use_activation=[^,]+)",
        "version": r"Initializing a V1 LLM engine \((v[\d.]+)\)",
    }
    for name, pat in patterns.items():
        m = re.search(pat, log)
        facts[name] = m.group(1) if m else None
    routes = sorted(set(re.findall(r"Route: (\S+), Methods: ([A-Z, ]+)", log)))
    facts["routes"] = [f"{methods.strip()} {path}" for path, methods in routes]
    return facts


# -- (i) latency ---------------------------------------------------------------------------------


@functools.cache
def pato_distance_criteria() -> dict[str, str]:
    """12 PATO 'distance' OLS fixture terms as '<label>: <definition>' plus the anchor."""
    wanted = {"ontology_id": "pato", "query": "distance", "size": 20}
    for path in sorted(OLS.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("method") == "search_terms" and fixture.get("args") == wanted:
            break
    else:
        raise SystemExit("PATO 'distance' search fixture not found under tests/fixtures/ols")
    resp = fixture["response"]
    items = resp if isinstance(resp, list) else resp["results"]
    cands = [t for t in items if t.get("description")][:12]
    criteria = {t["curie"]: f"{t['label']}: {t['description'][:300]}" for t in cands}
    criteria[ANCHOR_KEY] = ANCHORS["term"]
    return criteria


def rank_fit_request(nonce: int | None) -> tuple[dict[str, Any], Choice]:
    """observerDistance (brd_countdata fixture card) as a target_state and the PATO candidates.
    ``nonce`` changes the card's ``rows`` so the state text (and every cache key) is new."""
    card = load_card(CARDS / "DP1.10003.001.brd_countdata.md")
    col = next(c for c in card.columns if c.name == "observerDistance")
    state = target_state(card, "column", "measurement", column=col)
    if nonce is not None:
        state["card"] = {**state["card"], "rows": int(state["card"]["rows"]) + nonce}
    return state, Choice(criteria=pato_distance_criteria(), instructions=None)


def _time(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def probe_latency(enc: EncoderClient, clm: ClmHttpClient, repeats: int) -> dict[str, Any]:
    nonce0 = int(time.time()) % 1_000_000 * 1000  # new cache keys on every script run
    state, q = rank_fit_request(None)
    for _ in range(2):
        clm.system_one(state, {"fit": q})
    warm, warm_hdr = [], []
    for _ in range(repeats):
        ms, r = _time(lambda: clm.system_one(state, {"fit": q}))
        warm.append(ms)
        warm_hdr.append(r.latency_ms or 0.0)
    cold, cold_hdr, cold_tokens = [], [], []
    for i in range(repeats):
        st, _ = rank_fit_request(nonce0 + i + 1)
        ms, r = _time(lambda st=st: clm.system_one(st, {"fit": q}))
        cold.append(ms)
        cold_hdr.append(r.latency_ms or 0.0)
        cold_tokens.append(r.usage.input_tokens or 0)
    texts = []
    for _, s, _ in _snapshot_rows("term.fits"):
        t = render.to_text(framings.project("target_state", s))
        if t not in texts:
            texts.append(t)
        if len(texts) == 32:
            break
    enc.embed(texts)
    batch, tokens = [], 0
    for _ in range(repeats):
        ms, (_, tokens) = _time(lambda: enc.embed(texts))
        batch.append(ms)
    return {
        "rank_fit_request": {
            "state": "target_state of DP1.10003.001.brd_countdata observerDistance (fixture card)",
            "state_tokens": cold_tokens[0] if cold_tokens else None,
            "candidates": "12 PATO 'distance' OLS fixture terms '<label>: <definition>' + __none__",
            "model": "clm-latest",
        },
        "systemone_warm_same_state": {**percentiles(warm), "server_header": percentiles(warm_hdr)},
        "systemone_cold_new_state": {
            **percentiles(cold),
            "server_header": percentiles(cold_hdr),
            "input_tokens_per_request": sorted(set(cold_tokens)),
            "note": "new state text each time (card rows varied); candidate vectors cached",
        },
        "embeddings_batch_32": {
            **percentiles(batch),
            "prompt_tokens": tokens,
            "texts": "first 32 distinct term.fits target_state contexts of the snapshot",
        },
    }


# -- versions ------------------------------------------------------------------------------------


def probe_versions(http: Http) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    image_id = sg_docker("docker inspect mesa-clm-encoder --format {{.Image}}")
    config_image = sg_docker("docker inspect mesa-clm-encoder --format {{.Config.Image}}")
    repo_digests = sg_docker(
        f"docker image inspect {image_id} --format '{{{{json .RepoDigests}}}}'"
    )
    digests = json.loads(repo_digests) if repo_digests.startswith("[") else [repo_digests]
    spec = EncoderSpec(**lock["encoder"])
    return {
        "vllm_version": http.client.get(f"{ENC_URL}/version").json().get("version"),
        "image_id": image_id,
        "container_image_ref": config_image,
        "image_repo_digests": digests,
        "lock_image_digest": lock["image"]["digest"],
        "image_digest_matches_lock": any(lock["image"]["digest"] in d for d in digests)
        and lock["image"]["digest"] in config_image,
        "encoder_fp_lock": lock["encoder_fp"],
        "encoder_fp_recomputed": encoder_fp(spec),
        "lock_sha": lock["lock_sha"],
        "head_npz_sha256": sha256_file(HEAD_NPZ),
        "units_enabled": {
            u: run(["systemctl", "--user", "is-enabled", u])
            for u in ("mesa-clm-encoder.service", "mesa-clm-serve.service")
        },
        "units_active": {
            u: run(["systemctl", "--user", "is-active", u])
            for u in ("mesa-clm-encoder.service", "mesa-clm-serve.service")
        },
    }


# -- main ----------------------------------------------------------------------------------------


def dump_checked(payload: dict[str, Any], secrets: Sequence[str]) -> str:
    text = json.dumps(payload, indent=1, sort_keys=False, default=float, ensure_ascii=False)
    if any(s and s in text for s in secrets):
        raise SystemExit("refusing to write: a key value appears in the output")
    quads = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)) - LOOPBACK_OR_WILDCARD
    if quads:
        raise SystemExit("refusing to write: a non-loopback address appears in the output")
    return text + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out",
        type=Path,
        help=f"default bench/results/{DATE}/serving_m1.json (--quick: .local/serving/quick_probe.json)",
    )
    ap.add_argument("--golden-out", type=Path, default=ROOT / ".local/serving/encoder_golden.npz")
    ap.add_argument("--pairs-out", type=Path, default=ROOT / ".local/serving/systemone_pairs.json")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="binds, 401s, health, one systemone")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = ROOT / (
            ".local/serving/quick_probe.json"
            if args.quick
            else f"bench/results/{DATE}/serving_m1.json"
        )

    enc_key, clm_key = read_key("encoder"), read_key("clm")
    http = Http()
    enc = EncoderClient(
        ENC_URL, enc_key, max_len=MAX_LEN, timeout=120.0, retries=1, tokenizer_json=TOKENIZER_JSON
    )
    clm = ClmHttpClient(CLM_URL, clm_key, timeout=120.0, retries=1)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "format": "mesa-clm/serving-probes/1",
        "started_at": started,
        "host": "sparky-1",
        "command": "uv run python scripts/serving_probes.py" + (" --quick" if args.quick else ""),
    }
    t0 = time.perf_counter()

    def section(name: str, fn: Callable[[], Any]) -> None:
        print(f"== {name}", file=sys.stderr)
        payload[name] = fn()

    section("binds", probe_binds)
    section("auth", lambda: probe_auth(http, enc_key, clm_key))
    if args.quick:
        section("health", lambda: {"encoder": enc.healthy(), "clm_serve": clm.health()})
        state, q = rank_fit_request(None)
        section(
            "systemone",
            lambda: {
                k: round(v, 6)
                for k, v in clm.system_one(state, {"fit": q}).answers["fit"].probabilities.items()  # type: ignore[union-attr]
            },
        )
    else:
        head = HeadProjector.from_npz(HEAD_NPZ)
        section("versions", lambda: probe_versions(http))
        section("models", lambda: probe_models(enc, clm, http))
        section("drift", lambda: probe_drift(clm))
        section("truncation", lambda: probe_truncation(enc, http, enc_key))
        section("goldens", lambda: probe_goldens(enc, args.golden_out))
        section("parity", lambda: probe_parity(enc, clm, head, args.pairs_out))
        section("latency", lambda: probe_latency(enc, clm, args.repeats))
        section("footprint", probe_footprint)
    payload["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_checked(payload, [enc_key, clm_key]), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
