#!/usr/bin/env python3
"""Live serving probes for milestone M1 (plan §6.1, §6.4, §6.6, §6.8, §8 M1-A; DESIGN A3, A4).

Runs against the two loopback units on the GPU host (``mesa-clm-encoder.service`` on :8090,
``mesa-clm-serve.service`` on :8700) with mesa-clm's own clients (``EncoderClient``,
``ClmHttpClient``, ``HeadProjector``) and writes one JSON record::

    uv run python scripts/serving_probes.py --mem-pre-start GIB --mem-encoder-only GIB \
        --out bench/results/2026-10-01/serving_m1c.json \
        --golden-out .local/serving/m1c/encoder_golden_<encoder_fp>.npz
        # full run (serving_m1b.json is the A3/A4 recipe's, serving_m1c.json DESIGN A5's)
    uv run python scripts/serving_probes.py --quick    # binds, 401s, health, one systemone

Sections of the full run: (a) loopback binds; (b) routes and auth: the encoder's real route
table, enumerated by building vLLM's app for the locked arguments in a throwaway copy of the
pinned container (``serving/vllm_routes.py``; no model, no network), then every route on the
live :8090 requested with no key, a wrong key, the other unit's key and (for the routes that only
read or compute) the right key - every route but ``/health`` must answer 401 without the right
key (DESIGN A4); the same matrix on clm-serve; the container's docker-bridge address (not
recorded) probed too, since a container port is reachable there from any local account (under
DESIGN A5 the container has no network and so no bridge address), and the container's own
network namespace (interfaces and every listening socket, ``/proc/<pid>/net``), which ``ss`` on
the host cannot see; (c)
served models; (d) upstream drift probes (the CLM README quickstart at ``bb42c6c5`` and the
model-card tides rank); (e) a ~5,000-token text through ``EncoderClient`` (truncated to
``max_len - 1``) and as a clm-serve ``/v1/systemone`` state, both completing, the left-truncated
vector against the embedding of the last 4,095 token ids; (f) golden encoder vectors and
determinism (one text per request twice, a batch twice, against the M1-A goldens and, bitwise,
against the existing reference of the same ``encoder_fp`` when there is one: a recipe change that
keeps ``encoder_fp`` must leave it unchanged) written to ``--golden-out``; (g) clm-serve ``/v1/systemone`` vs the local
head projection over 50 (state, question) pairs for ``clm-latest`` and ``clm-raw``; (h) memory
footprint; (i) latency p50/p95; (j) last, one request of exactly 4,096 token ids with a hard
timeout (the window boundary the cap avoids) and the engine's request gauges after it; plus
versions, the image digest, ``encoder_fp`` and ``verify_serving_lock(require_live=True)``.

Keys are read from the 0600 files under ``~/.mesa/clm/secrets`` and never printed, logged or
written: before anything is written the serialised output is checked for both key values and
the run aborts if either appears. Only loopback addresses are recorded. Every keyed request,
through mesa-clm's clients or the raw ``httpx`` clients here (:func:`owner_hook`), first asks
who holds the loopback port (:func:`mesa_clm.net.assert_listener_owner`, DESIGN A5): run while
the units are down or restarting, another account's socket on :8090 or :8700 gets no key and
the run stops.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import pwd
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np

from mesa_clm import framings, render, serving
from mesa_clm.cards import load_card
from mesa_clm.clm.encoder import EncoderClient, decode_base64_f32
from mesa_clm.clm.fingerprint import EncoderSpec, encoder_fp
from mesa_clm.clm.headproj import RAW_SCALE, HeadProjector
from mesa_clm.clm.http import Choice, ClmHttpClient, Noul, Score, question_to_dict
from mesa_clm.health import parse_listener_details
from mesa_clm.net import assert_listener_owner
from mesa_clm.registry import ANCHOR_KEY, ANCHORS, ONTOLOGY_REGISTRY
from mesa_clm.states import target_state
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-01"
ENC_URL = "http://127.0.0.1:8090"
CLM_URL = "http://127.0.0.1:8700"
SECRETS = Path("~/.mesa/clm/secrets").expanduser()
HEAD_NPZ = Path("~/.mesa/clm/heads/npz/b2b4a8c9.npz").expanduser()
SERVE_PY = Path("~/.mesa/clm/serve/.venv/bin/python").expanduser()
AUTH_DIR = Path("~/.mesa/clm/serve/vllm-auth").expanduser()
M1A_GOLDENS = ROOT / ".local/serving/encoder_golden.npz"
TOKENIZER_JSON = Path(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
).expanduser()
SNAPSHOT = ROOT / "bench/snapshots/2026-09-29.parquet"
LOCK = ROOT / "serving/serving.lock.json"
CARDS = ROOT / "tests/fixtures/cards"
OLS = ROOT / "tests/fixtures/ols"
# Plan §9's non-bench live-smoke card: the long-input probe and the latency probe's rank-fit
# question use it (DESIGN, "G1 freeze", item 7); until 2026-10-01 they used the bench card
# brd_countdata (serving_m1.json, serving_m1b.json, serving_m1c.json, batch_invariance.json).
# The parity section (g) and the latency probe's 32-text batch still ask and embed labelled
# snapshot items, label-free, and the served answers go to --pairs-out under .local/ (item 4).
SMOKE_CARD = ROOT / "tests/fixtures/cards-srer/DP1.00004.001.BP_30min.md"
MODEL = "qwen3-8b"
MAX_LEN = 4096
CAP = MAX_LEN - 1  # truncate_prompt_tokens every client sends (DESIGN A4)
WRONG_KEY = "mesa-clm-probe-wrong-key"
# The throwaway enumeration container's key: a fixed dummy, never the encoder's key.
ROUTE_DUMMY_KEY = "mesa-clm-route-enumeration-dummy"
# Dotted quads allowed in the output: loopback, and the wildcard peer column of `ss`.
LOOPBACK_OR_WILDCARD = frozenset({"127.0.0.1", "0.0.0.0"})  # noqa: S104

# Footprint reference points: MemAvailable with both units stopped and with the encoder alone,
# recorded by whoever restarts the units before the run (--mem-pre-start, --mem-encoder-only).
MEM_REFERENCE: dict[str, float | None] = {"pre_start": None, "encoder_only": None}

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


def owner_hook(request: httpx.Request) -> None:
    """An ``httpx`` request hook: a request that carries a key first asks who holds the
    loopback port and raises :class:`~mesa_clm.net.ListenerOwnerError` (nothing sent) when it is
    another account's socket, as ``EncoderClient`` and ``ClmHttpClient`` do (DESIGN A5)."""
    if "authorization" in request.headers:
        assert_listener_owner(str(request.url), what="serving probe")


def raw_client(timeout: float) -> httpx.Client:
    """A raw client for the probes: no proxies from the environment, the owner check on every
    keyed request."""
    return httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(timeout, connect=5.0),
        event_hooks={"request": [owner_hook]},
    )


class Http:
    """Raw status probes (no retries, no proxies) with an explicit Authorization choice."""

    def __init__(self) -> None:
        self.client = raw_client(120.0)

    def status(self, method: str, url: str, key: str | None = None, body: Any = None) -> int:
        headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
        try:
            r = self.client.request(method, url, headers=headers, json=body)
        except httpx.TransportError:
            return -1
        return r.status_code


# -- (a) binds -----------------------------------------------------------------------------------


def probe_binds() -> dict[str, Any]:
    out = run(["ss", "-ltne", "( sport = :8090 or sport = :8700 )"])
    pairs = [(x, port) for port in (8090, 8700) for x in parse_listener_details(out, port)]
    listeners = [f"{x.host}:{port}" for x, port in pairs]
    loopback = [a for a in listeners if a.startswith(("127.0.0.1:", "[::1]:"))]
    return {
        "command": "ss -ltne '( sport = :8090 or sport = :8700 )'",
        "listeners": sorted(listeners),
        "loopback_only": bool(listeners) and len(loopback) == len(listeners),
        "ports_seen": sorted({a.rsplit(":", 1)[1] for a in listeners}),
        # The owner of each listener (ss omits uid:0, read as root's): another account's socket
        # on a port would be sent the keys (DESIGN A5); owner_hook refuses that per request.
        "owners_are_this_account": bool(pairs) and all(x.uid == os.getuid() for x, _ in pairs),
        "docker_port": sg_docker("docker port mesa-clm-encoder"),
    }


# -- (b) auth matrix -----------------------------------------------------------------------------


def enumerate_routes() -> dict[str, Any]:
    """The encoder's route table for the locked ``vllm serve`` arguments, from vLLM's own
    ``build_app`` in a throwaway copy of the pinned container (``serving/vllm_routes.py``): the
    installed guard on ``PYTHONPATH``, a dummy key, no network, no model (``--gpus all`` only
    because vLLM's argument parser infers the device)."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    uid = os.getuid()
    gid = pwd.getpwuid(uid).pw_gid
    image = f"{lock['image']['ref']}@{lock['image']['digest']}"
    argv = [
        *("docker", "run", "--rm", "--gpus", "all", "--network", "none", "--user", f"{uid}:{gid}"),
        *("-e", "HOME=/tmp", "-e", f"VLLM_API_KEY={ROUTE_DUMMY_KEY}"),
        *("-e", "PYTHONPATH=/opt/mesa-clm-auth", "-e", "VLLM_NO_USAGE_STATS=1"),
        *("-e", "DO_NOT_TRACK=1", "-e", "HF_HUB_OFFLINE=1"),
        *("-v", f"{AUTH_DIR}:/opt/mesa-clm-auth:ro"),
        *("-v", f"{ROOT / 'serving' / 'vllm_routes.py'}:/work/vllm_routes.py:ro"),
        *("--entrypoint", "python3", image, "/work/vllm_routes.py"),
        json.dumps(lock["recipe"]["args"]),
    ]
    out = run(["sg", "docker", "-c", shlex.join(argv)], timeout=600.0)
    try:
        table: dict[str, Any] = json.loads(out.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": "route enumeration failed", "tail": out.splitlines()[-3:]}
    table["command"] = (
        "docker run --rm --gpus all --network none ... --entrypoint python3 <pinned image> "
        "/work/vllm_routes.py <recipe.args>"
    )
    return table


# Requests that only read or compute, sent with the right key to show the route still works.
SAFE_WITH_KEY: dict[tuple[str, str], Any] = {
    ("GET", "/health"): None,
    ("GET", "/load"): None,
    ("GET", "/version"): None,
    ("GET", "/metrics"): None,
    ("GET", "/ping"): None,
    ("POST", "/ping"): None,
    ("GET", "/v1/models"): None,
    ("POST", "/tokenize"): {"model": MODEL, "prompt": "hello world"},
    ("POST", "/detokenize"): {"model": MODEL, "tokens": [14990, 1879]},
    ("POST", "/v1/embeddings"): {"model": MODEL, "input": ["probe"]},
    ("POST", "/pooling"): {"model": MODEL, "input": "probe"},
    ("POST", "/invocations"): {"model": MODEL, "input": "probe"},
    ("POST", "/score"): {"model": MODEL, "text_1": "a", "text_2": "b"},
    ("POST", "/v1/score"): {"model": MODEL, "text_1": "a", "text_2": "b"},
    ("POST", "/rerank"): {"model": MODEL, "query": "a", "documents": ["b"]},
    ("POST", "/v1/rerank"): {"model": MODEL, "query": "a", "documents": ["b"]},
    ("POST", "/v2/rerank"): {"model": MODEL, "query": "a", "documents": ["b"]},
    ("POST", "/v2/embed"): {
        "model": MODEL,
        "texts": ["probe"],
        "input_type": "search_query",
        "embedding_types": ["float"],
    },
}
UNKNOWN_PATHS = (
    "/nonexistent",
    "/metrics/anything",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/health/",
)


def _bridge_url() -> str | None:
    """The encoder container's docker-bridge base URL (kept in memory, never recorded)."""
    ip = sg_docker(
        "docker inspect mesa-clm-encoder --format '{{range .NetworkSettings.Networks}}"
        "{{.IPAddress}}{{end}}'"
    )
    return f"http://{ip}:8090" if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", ip) else None


def probe_netns() -> dict[str, Any]:
    """The encoder container's network namespace (DESIGN A5): its interfaces and every listening
    TCP socket, read from ``/proc/<pid>/net`` of the container's main process (the container
    runs as this user; nothing is connected to)."""
    pid_text = sg_docker("docker inspect mesa-clm-encoder --format '{{.State.Pid}}'")
    network = sg_docker("docker inspect mesa-clm-encoder --format '{{.HostConfig.NetworkMode}}'")
    if not pid_text.isdigit():
        return {"error": "the encoder container is not running", "network_mode": network}
    state = serving.read_netns(int(pid_text))
    if state is None:
        return {"error": "/proc/<pid>/net is not readable", "network_mode": network}
    return {
        "network_mode": network,
        "interfaces": state.interfaces,
        "listeners": state.listeners,
        "problems": serving.netns_problems(state),
        "isolated": state.interfaces == ["lo"],
    }


def probe_auth(http: Http, enc_key: str, clm_key: str, routes: dict[str, Any]) -> dict[str, Any]:
    """Every enumerated encoder route, unknown paths, the bridge address and clm-serve's routes
    with no key, a wrong key, the other unit's key and (read/compute routes) the right key."""
    pairs = sorted({(p["method"], p["path"]) for p in routes.get("probes", [])})
    rows = []
    for method, path in pairs:
        body = SAFE_WITH_KEY.get((method, path), {"model": MODEL, "input": "probe"})
        body = None if method == "GET" else body
        url = f"{ENC_URL}{path}"
        row: dict[str, Any] = {
            "route": f"{method} {path}",
            "no_key": http.status(method, url, None, body),
            "wrong_key": http.status(method, url, WRONG_KEY, body),
            "other_units_key": http.status(method, url, clm_key, body),
        }
        if (method, path) in SAFE_WITH_KEY:
            row["right_key"] = http.status(method, url, enc_key, body)
        rows.append(row)
    guarded = [r for r in rows if r["route"] != "GET /health"]
    unguarded = [
        r["route"]
        for r in guarded
        if not (r["no_key"] == r["wrong_key"] == r["other_units_key"] == 401)
    ]
    unknown = {
        f"GET {path}": {
            "no_key": http.status("GET", f"{ENC_URL}{path}"),
            "right_key": http.status("GET", f"{ENC_URL}{path}", enc_key),
        }
        for path in UNKNOWN_PATHS
    }
    bridge_base = _bridge_url()
    bridge = (
        {
            "GET /health (no key)": http.status("GET", f"{bridge_base}/health"),
            "GET /version (no key)": http.status("GET", f"{bridge_base}/version"),
            "POST /pooling (no key)": http.status(
                "POST", f"{bridge_base}/pooling", None, {"model": MODEL, "input": "probe"}
            ),
            "GET /v1/models (key)": http.status("GET", f"{bridge_base}/v1/models", enc_key),
        }
        if bridge_base
        else {"none": "the container has no bridge address (no network, DESIGN A5)"}
    )
    emb_body = {"model": MODEL, "input": ["probe"]}
    so_body = {
        "state": "probe",
        "model": "clm-latest",
        "questions": {"q": {"type": "noul", "instructions": "Is this a probe?"}},
    }
    rank_body = {"context": "probe", "question": None, "answers": ["a", "b"], "model": "clm-latest"}
    clm_rows = []
    for unit, method, url, body, right, other in (
        ("encoder", "GET", f"{ENC_URL}/v1/models", None, enc_key, clm_key),
        ("encoder", "POST", f"{ENC_URL}/v1/embeddings", emb_body, enc_key, clm_key),
        ("clm-serve", "GET", f"{CLM_URL}/v1/models", None, clm_key, enc_key),
        ("clm-serve", "POST", f"{CLM_URL}/v1/systemone", so_body, clm_key, enc_key),
        ("clm-serve", "POST", f"{CLM_URL}/v1/rank", rank_body, clm_key, enc_key),
    ):
        clm_rows.append(
            {
                "unit": unit,
                "route": f"{method} {url}",
                "no_key": http.status(method, url, None, body),
                "wrong_key": http.status(method, url, WRONG_KEY, body),
                "other_units_key": http.status(method, url, other, body),
                "right_key": http.status(method, url, right, body),
            }
        )
    clm_open = {
        f"clm-serve GET {path}": http.status("GET", f"{CLM_URL}{path}")
        for path in ("/health", "/", "/docs", "/openapi.json", "/redoc")
    }
    verdict = routes.get("verdict") or {}
    return {
        "enumeration": {
            k: routes.get(k)
            for k in (
                "command",
                "vllm_args",
                "routes",
                "websocket_routes",
                "middleware_outermost_first",
                "verdict",
                "error",
            )
            if k in routes
        },
        "n_routes": len(routes.get("routes") or []),
        "encoder_routes": rows,
        "encoder_unguarded_without_valid_key": unguarded,
        "encoder_every_route_but_health_401": bool(guarded) and not unguarded,
        "encoder_health_open": next(
            (r["no_key"] == 200 for r in rows if r["route"] == "GET /health"), False
        ),
        "encoder_right_key_status": {
            r["route"]: r.get("right_key") for r in rows if "right_key" in r
        },
        "unknown_paths": unknown,
        "container_enumeration_pass": bool(verdict.get("pass")),
        "docker_bridge_address": {
            "note": "a container on the docker bridge is reachable on its bridge address from any "
            "local account (address not recorded); under DESIGN A5 (--network none) there is none",
            **bridge,
        },
        "named_routes": clm_rows,
        "named_v1_routes_all_401_without_valid_key": all(
            r["no_key"] == r["wrong_key"] == r["other_units_key"] == 401 for r in clm_rows
        ),
        "right_key_all_200": all(r["right_key"] == 200 for r in clm_rows),
        "clm_serve_open_routes": clm_open,
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
    # Compared unrounded (the record shows 4 decimals): rounding first can flip an edge case.
    exact = {
        "urgency": float(urgency),
        "billing": float(dept.probabilities["billing"]),  # type: ignore[union-attr]
        "frustration": float(frus.score),  # type: ignore[union-attr]
    }
    abs_delta = {k: abs(v - ISSUE15[k]) for k, v in exact.items()}
    within = {k: d <= tol for k, d in abs_delta.items()}
    ranked = clm.rank(TIDES[0], None, TIDES[1], model="clm-latest")
    top = ranked[0]
    return {
        "source": (
            "README.md at Contrastive-LM/CLM bb42c6c5 (gh api .../contents/README.md?ref=bb42c6c5...)"
            "; issue #15 comment 2026-09-28 (vLLM 0.30.0, max-model-len 8192, GB10)"
        ),
        "quickstart": {
            "observed": observed,
            "abs_delta_unrounded": {k: round(d, 7) for k, d in abs_delta.items()},
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


def _tokenize(http: Http, text: str, key: str) -> list[int]:
    r = http.client.post(
        f"{ENC_URL}/tokenize",
        json={"model": MODEL, "prompt": text},
        headers={"Authorization": f"Bearer {key}"},
    )
    r.raise_for_status()
    return [int(t) for t in r.json()["tokens"]]


def _bounded(enc_key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """One /v1/embeddings request with a hard timeout and no retry: status and seconds, or
    ``timeout`` (a hung request; vLLM aborts it when the client disconnects)."""
    t0 = time.perf_counter()
    try:
        with raw_client(timeout) as c:
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
    return outcome


def long_text(http: Http, enc_key: str) -> tuple[str, list[int]]:
    """~5,000 tokens: the non-bench smoke card repeated, cut at 4,990 tokens, plus a target
    line."""
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    corpus = "\n\n".join([SMOKE_CARD.read_text(encoding="utf-8")] * 6)
    head_ids = _tokenize(http, corpus, enc_key)[:4990]
    text = (
        tok.decode(head_ids)
        + "\n\nTarget column: staPresMean (kilopascal), the mean station pressure."
    )
    return text, _tokenize(http, text, enc_key)


def probe_truncation(
    enc: EncoderClient, http: Http, enc_key: str, clm: ClmHttpClient, head: HeadProjector
) -> dict[str, Any]:
    """A ~5,000-token text completes through ``EncoderClient`` (``truncate_prompt_tokens`` =
    ``max_len - 1``, left) and as a clm-serve state (``--max-tokens 4095``); the left-truncated
    vector equals the embedding of the last 4,095 token ids (DESIGN A4)."""
    text, ids = long_text(http, enc_key)
    b64 = {"model": MODEL, "encoding_format": "base64"}
    t0 = time.perf_counter()
    v_client, t_client = enc.embed([text])
    client_s = time.perf_counter() - t0
    v_tail, t_tail = _embed_raw(enc, {**b64, "input": [ids[-CAP:]]})
    v_head, _ = _embed_raw(enc, {**b64, "input": [ids[:CAP]]})
    c_tail = cos(v_client[0], v_tail)
    choice = Choice(
        criteria={
            "pressure": "pressure: A physical quality that inheres in a bearer by virtue of the "
            "bearer's amount of force per unit area it exerts.",
            "temperature": "temperature: A physical quality of the thermal energy of a system.",
            ANCHOR_KEY: ANCHORS["term"],
        }
    )
    question = {"fit": question_to_dict(choice)}  # the wire form both routes take
    # A run nonce keeps the state out of clm-serve's caches, so the request is a cold one.
    state = f"Probe run {time.time_ns()}.\n\n{text}"
    t1 = time.perf_counter()
    served = clm.system_one(state, question, model="clm-latest")
    served_s = time.perf_counter() - t1
    local = local_answers(enc, head, state, question, "clm-latest")
    sp, lp = _probs(served.answers["fit"]), _probs(local["fit"])
    return {
        "text_sha256": sha256_text(text),
        "text_tokens": len(ids),
        "corpus": "tests/fixtures/cards-srer/DP1.00004.001.BP_30min.md (non-bench) repeated, "
        "first 4990 tokens, plus a target line",
        "encoder_client": {
            "request": f"EncoderClient(max_len={MAX_LEN}).embed: truncate_prompt_tokens="
            f"{enc.truncate_prompt_tokens}, truncation_side=left",
            "completed": True,
            "seconds": round(client_s, 2),
            "prompt_tokens_charged": t_client,
            f"cos_vs_last_{CAP}_token_ids": round(c_tail, 9),
            "bitwise_equal_to_tail_ids": bool(np.array_equal(v_client[0], v_tail)),
            "tail_ids_tokens_charged": t_tail,
            f"cos_vs_first_{CAP}_token_ids": round(cos(v_client[0], v_head), 7),
            "gate": 0.9999,
            "pass": c_tail >= 0.9999 and t_client == CAP,
        },
        "clm_serve_systemone": {
            "request": "the same text behind a run nonce (never cached) as the state of one "
            "Choice (2 candidates + __none__), clm-latest; clm-serve --max-tokens 4095",
            "completed": True,
            "seconds": round(served_s, 2),
            "server_latency_ms": served.latency_ms,
            "input_tokens": served.usage.input_tokens,
            "probabilities": {k: round(v, 6) for k, v in sp.items()},
            "max_abs_prob_diff_vs_local_route": max(abs(sp[k] - lp[k]) for k in sp),
        },
    }


def probe_boundary(enc_key: str, http: Http, text_ids: list[int]) -> dict[str, Any]:
    """Last: one request of exactly ``MAX_LEN`` token ids (the window the cap stays below) with a
    30 s timeout, then the engine's request gauges from ``/metrics`` (with the key) a few seconds
    later, to show whether the abandoned request was aborted."""
    b64 = {"model": MODEL, "encoding_format": "base64"}
    outcome = _bounded(enc_key, {**b64, "input": [text_ids[:MAX_LEN]]}, 30.0)
    time.sleep(5.0)
    metrics = http.client.get(
        f"{ENC_URL}/metrics", headers={"Authorization": f"Bearer {enc_key}"}
    ).text
    gauges = {
        name: float(m.group(1))
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting")
        if (m := re.search(rf"^{re.escape(name)}{{[^}}]*}} ([0-9.e+]+)$", metrics, re.M))
    }
    after = _bounded(enc_key, {**b64, "input": [text_ids[: MAX_LEN - 1]]}, 60.0)
    return {
        f"token_ids_{MAX_LEN}": outcome,
        "gauges_5s_after": gauges,
        f"token_ids_{MAX_LEN - 1}_afterwards": after,
        "note": "the clients never send this: truncate_prompt_tokens is max_len - 1 (DESIGN A4)",
    }


# -- (f) goldens ---------------------------------------------------------------------------------


def probe_goldens(enc: EncoderClient, out: Path, reference: Path | None) -> dict[str, Any]:
    """Golden vectors and determinism: one text per request twice (the doctor's reference), the
    20 texts as one batch twice, the M1-A goldens (batched, ``852efc921a8a``) for scale and,
    bitwise, the existing reference of this ``encoder_fp`` (``reference``) when there is one."""
    texts = list(GOLDEN_TEXTS)
    single = np.stack([enc.embed([t])[0][0] for t in texts])
    single2 = np.stack([enc.embed([t])[0][0] for t in texts])
    batch, tokens = enc.embed(texts)
    batch2, _ = enc.embed(texts)
    shas = [sha256_text(t) for t in texts]
    lock = json.loads(LOCK.read_text())
    previous: dict[str, Any] | None = None
    if reference is not None and reference.is_file():
        with np.load(reference, allow_pickle=False) as npz:
            ref_texts = [str(t) for t in npz["texts"]]
            ref_vectors = np.asarray(npz["vectors"], dtype=np.float32)
            ref_fp = str(npz["encoder_fp"])
        same = ref_texts == texts and ref_fp == lock["encoder_fp"]
        previous = {
            "file_sha256": sha256_file(reference),
            "encoder_fp": ref_fp,
            "same_texts_and_encoder_fp": same,
            "bitwise_equal_rows": int(np.sum(np.all(single == ref_vectors, axis=1))) if same else 0,
            "n": len(texts),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        np.savez(
            fh,
            texts=np.array(texts),
            text_sha256=np.array(shas),
            vectors=single.astype(np.float32),
            encoder_fp=np.array(lock["encoder_fp"]),
            route=np.array("vllm"),
            pattern=np.array("one text per request"),
        )

    def stats(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        cs = [cos(x[i], y[i]) for i in range(len(texts))]
        return {
            "min_cos": round(min(cs), 9),
            "mean_cos": round(float(np.mean(cs)), 9),
            "bitwise_identical": int(np.sum(np.all(x == y, axis=1))),
        }

    with np.load(M1A_GOLDENS, allow_pickle=False) as npz:
        m1a = np.asarray(npz["vectors"], dtype=np.float32)
        m1a_fp = str(npz["encoder_fp"])
    return {
        "file": str(out.relative_to(ROOT)),
        "file_sha256": sha256_file(out),
        "encoder_fp": lock["encoder_fp"],
        "n": len(texts),
        "dtype": "float32",
        "prompt_tokens": tokens,
        "texts_sha256": shas,
        "one_by_one_vs_repeat": stats(single, single2),
        "batch_vs_one_by_one": stats(batch, single),
        "batch_vs_batch_repeat": stats(batch, batch2),
        "vs_m1a_goldens": {"m1a_encoder_fp": m1a_fp, **stats(single, m1a)},
        "vs_reference_same_encoder_fp": previous,
        "golden_gate_0_9999": {
            "one_by_one_repeat_pass": stats(single, single2)["min_cos"] >= 0.9999,
            "batch_vs_one_by_one_pass": stats(batch, single)["min_cos"] >= 0.9999,
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
    pre, enc_only = MEM_REFERENCE["pre_start"], MEM_REFERENCE["encoder_only"]
    return {
        "memavailable_now_gib": now,
        "memavailable_pre_start_gib": pre,
        "memavailable_encoder_only_gib": enc_only,
        "delta_both_units_gib": round(pre - now, 2) if pre is not None else None,
        "delta_encoder_gib": round(pre - enc_only, 2)
        if pre is not None and enc_only is not None
        else None,
        "delta_clm_serve_gib": round(enc_only - now, 2) if enc_only is not None else None,
        "reference_points": "MemAvailable logged by the restart before this run "
        "(both units stopped; encoder alone)",
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
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "kv_cache_reserved": r"reserved ([\d.]+ GiB) memory for KV Cache as specified",
        "desired_utilization": r"Desired GPU memory utilization is \(([\d.]+, [\d.]+ GiB)\)",
        "listen": r"Starting vLLM server on (unix:\S+)",
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
    """staPresMean of the non-bench smoke card as a target_state and the 12 PATO candidates (no
    labelled pair: the latency probe never answers a bench item). ``nonce`` changes the card's
    ``rows`` so the state text (and every cache key) is new."""
    card = load_card(SMOKE_CARD)
    col = next(c for c in card.columns if c.name == "staPresMean")
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
            "state": "target_state of DP1.00004.001.BP_30min staPresMean (non-bench smoke card)",
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
    enc_key = read_key("encoder")
    problems = serving.verify_serving_lock(require_live=True)
    return {
        "vllm_version": http.client.get(
            f"{ENC_URL}/version", headers={"Authorization": f"Bearer {enc_key}"}
        )
        .json()
        .get("version"),
        "image_id": image_id,
        "container_image_ref": config_image,
        "image_repo_digests": digests,
        "lock_image_digest": lock["image"]["digest"],
        "image_digest_matches_lock": any(lock["image"]["digest"] in d for d in digests)
        and lock["image"]["digest"] in config_image,
        "encoder_fp_lock": lock["encoder_fp"],
        "encoder_fp_recomputed": encoder_fp(spec),
        "encoder_spec": lock["encoder"],
        "lock_sha": lock["lock_sha"],
        "recipe": {k: v for k, v in lock.get("recipe", {}).items() if k != "args"},
        "recipe_args": " ".join(lock.get("recipe", {}).get("args", [])),
        "verify_serving_lock_require_live": problems,
        "verify_serving_lock_pass": problems == [],
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
        help=f"default bench/results/{DATE}/serving_m1b.json (--quick: .local/serving/quick_probe.json)",
    )
    ap.add_argument("--golden-out", type=Path)
    ap.add_argument(
        "--golden-reference",
        type=Path,
        help="existing goldens of this encoder_fp to compare bitwise "
        "(default .local/serving/encoder_golden_<encoder_fp>.npz)",
    )
    ap.add_argument(
        "--pairs-out", type=Path, default=ROOT / ".local/serving/m1b/systemone_pairs.json"
    )
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--mem-pre-start", type=float, help="MemAvailable GiB with both units stopped")
    ap.add_argument("--mem-encoder-only", type=float, help="MemAvailable GiB, encoder alone up")
    ap.add_argument("--no-boundary", action="store_true", help="skip the 4096-token-id request")
    ap.add_argument("--quick", action="store_true", help="binds, 401s, health, one systemone")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = ROOT / (
            ".local/serving/quick_probe.json"
            if args.quick
            else f"bench/results/{DATE}/serving_m1b.json"
        )
    MEM_REFERENCE.update(pre_start=args.mem_pre_start, encoder_only=args.mem_encoder_only)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    golden_out = (
        args.golden_out or ROOT / f".local/serving/encoder_golden_{lock['encoder_fp']}.npz"
    ).resolve()
    golden_ref: Path | None = args.golden_reference or (
        ROOT / f".local/serving/encoder_golden_{lock['encoder_fp']}.npz"
    )
    if golden_ref is not None and golden_ref.resolve() == golden_out.resolve():
        golden_ref = None  # the run would compare the new file with itself

    enc_key, clm_key = read_key("encoder"), read_key("clm")
    http = Http()
    enc = EncoderClient(
        ENC_URL, enc_key, max_len=MAX_LEN, timeout=120.0, retries=1, tokenizer_json=TOKENIZER_JSON
    )
    clm = ClmHttpClient(CLM_URL, clm_key, timeout=300.0, retries=1)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "format": "mesa-clm/serving-probes/2",
        "started_at": started,
        "host": "sparky-1",
        "command": "uv run python scripts/serving_probes.py" + (" --quick" if args.quick else ""),
    }
    t0 = time.perf_counter()

    def section(name: str, fn: Callable[[], Any]) -> None:
        print(f"== {name}", file=sys.stderr)
        payload[name] = fn()

    section("binds", probe_binds)
    section("netns", probe_netns)
    section("routes", enumerate_routes)
    section("auth", lambda: probe_auth(http, enc_key, clm_key, payload["routes"]))
    del payload["routes"]  # recorded under auth.enumeration
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
        section("truncation", lambda: probe_truncation(enc, http, enc_key, clm, head))
        section("goldens", lambda: probe_goldens(enc, golden_out, golden_ref))
        section("parity", lambda: probe_parity(enc, clm, head, args.pairs_out))
        section("latency", lambda: probe_latency(enc, clm, args.repeats))
        section("footprint", probe_footprint)
        if not args.no_boundary:
            _, ids = long_text(http, enc_key)
            section("boundary", lambda: probe_boundary(enc_key, http, ids))
    payload["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_checked(payload, [enc_key, clm_key]), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
