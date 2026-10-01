"""The serving host's keys, systemd units and pins, seen from the torch-free core (plan §6).

The serve side itself (the vLLM encoder container, the patched ``clm-serve`` venv, the fallback
encoder) lives under ``serving/`` and ``deploy/`` and never imports :mod:`mesa_clm`. This module
is the core's view of it, for the doctor and the planned ``mesa-clm serve
keys|units|lock`` verbs:

* :func:`init_keys` - the two bearer keys (plan §6.4). ``clm.key`` guards clm-serve (:8700) and
  ``encoder.key`` the encoder (:8090); both are raw 0600 files the core reads through
  ``MESA_CLM_CLM__API_KEY_FILE`` / ``MESA_CLM_ENCODER__API_KEY_FILE``. Two env files are derived
  from them for the units: ``clm.env`` (``CLM_API_KEY``, ``CLM_EMB_API_KEY`` = the encoder key,
  ``CLM_EMB_CACHE_SIZE``; read by ``mesa-clm-serve.service`` through ``EnvironmentFile=``) and
  ``encoder.env`` (``VLLM_API_KEY`` = the encoder key; ``docker run --env-file``). Every file is
  written atomically with mode 0600 in a 0700 directory. Nothing returned or logged carries a key
  or any hash of one; the result names paths and whether keys were created or rotated.
* :func:`render_units` - the systemd ``--user`` units (plan §6.1, §6.2): the encoder, clm-serve,
  the headroom check and its timer, and the encoder's loopback endpoint (DESIGN A5: the
  container has no network and serves on a unix socket; ``mesa-clm-encoder-proxy.socket`` owns
  127.0.0.1:8090 and the bootstrap-installed ``mesa-clm-encoder-proxy``,
  ``serving/encoder_proxy.py``, relays each connection to that socket, never through a symlink).
  The encoder requires the endpoint socket and starts after it (``Requires=``, ``After=``: a port
  another account holds fails the encoder and clm-serve instead of leaving them to talk to that
  account's socket) and pulls in the headroom timer (``Wants=``); both stop with it
  (``PartOf=``). The encoder is started at most three times an hour, manual starts included.
  ``%h`` is left for systemd to expand; ``deploy/systemd/`` holds the default rendering and a
  unit test keeps the two identical. Installing or enabling a unit is always a separate user
  action.
* :func:`read_netns` / :func:`netns_problems` - the encoder container's network namespace read
  from ``/proc/<pid>/net`` (interfaces and listening sockets), for the doctor and the probes:
  a listener on a non-loopback address in a namespace with another interface than ``lo`` is a
  problem (DESIGN A5).
* :func:`build_lock_body` / :func:`render_lock` - ``serving/serving.lock.json`` (plan §6.7) from
  the pins below, the sha256 of each carried patch and of the encoder's bearer guard
  (``serving/vllm_auth.py``), and the container recipe (:func:`encoder_recipe`: the ``vllm
  serve`` arguments, the non-secret environment, the guard, the clients' truncation cap; DESIGN
  A3, A4), signed with :func:`mesa_clm.clm.fingerprint.sign_lock_body`.
* :func:`check_serving_lock` / :func:`verify_serving_lock` - the lock against this host (plan
  §6.8): the patch files, the copy the bootstrap installed, the serve clone (commit, applied
  series, ``schema.py``), the serve venv's ``clm.schema``, the head file, the installed bearer
  guard, and the pinned image and running encoder container (arguments, environment, the guard's
  read-only mount; through ``sg docker`` when the session lacks the docker group). Anything
  *absent* (no clone yet, no docker, not the serving host) is a ``skip`` unless
  ``require_live=True``; anything *present and different* is a ``fail``. The container's
  environment is read through a ``docker inspect`` template that prints the values of the
  locked names only, so the bearer key in ``VLLM_API_KEY`` never reaches this process.
"""

from __future__ import annotations

import grp
import hashlib
import ipaddress
import json
import logging
import os
import posixpath
import pwd
import re
import secrets
import shlex
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

from mesa_clm.clm.fingerprint import (
    BatchInvariance,
    EncoderSpec,
    LockError,
    LockRecipe,
    ServingLock,
    load_serving_lock,
    sign_lock_body,
)
from mesa_clm.net import proc_address as _proc_address
from mesa_clm.perms import private_dir, write_private_text
from mesa_clm.secrets import SecretError, read_secret_file

logger = logging.getLogger(__name__)

# -- pins (plan §6.1-§6.3, §6.7; RESEARCH.md) --------------------------------------------------

CLM_REPO_URL: Final[str] = "https://github.com/Contrastive-LM/CLM.git"
CLM_COMMIT: Final[str] = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
# sha256 of src/clm/schema.py at CLM_COMMIT; equal to the vendored copy (vendored.sha256).
SCHEMA_SHA256: Final[str] = "52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335"
HEAD_REPO: Final[str] = "Contrastive-LM/CLM-v0.1-8B"
HEAD_REVISION: Final[str] = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
HEAD_FILE: Final[str] = "CLM_v0.1-8B.pt"
HEAD_SIZE: Final[int] = 75_557_149
HEAD_SHA256: Final[str] = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
# How the vLLM route keeps a text's vector independent of its batch (DESIGN A3): vLLM's
# batch-invariant kernels (VLLM_BATCH_INVARIANT=1), adopted by the pre-registered experiment in
# bench/results/2026-10-01/batch_invariance.json.
BATCH_INVARIANCE: Final[BatchInvariance] = "kernels"
ENCODER: Final[EncoderSpec] = EncoderSpec(
    model="Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
    dtype="bfloat16",
    pooling="LAST",
    normalize=True,
    max_len=4096,
    truncation_side="left",
    prefix_caching=False,
    route="vllm",
    batch_invariance=BATCH_INVARIANCE,
)
# The in-process fallback (serving/cuda_encoder.py, fallback_serve.py): one text per forward pass,
# so no padding and no batch dependence (DESIGN A3, D16).
FALLBACK_ENCODER: Final[EncoderSpec] = ENCODER.model_copy(
    update={"route": "transformers", "batch_invariance": "serial"}
)
# What every client sends as truncate_prompt_tokens (EncoderClient; clm-serve --max-tokens): one
# below the window, because vLLM 0.27.1 never completes an input of exactly --max-model-len
# tokens (bench/results/2026-09-29/serving_m1.json); CLM's training cap is max_len - 1 as well.
TRUNCATE_PROMPT_TOKENS: Final[int] = ENCODER.max_len - 1
IMAGE_REF: Final[str] = "vllm/vllm-openai"
IMAGE_TAG: Final[str] = "v0.27.1"  # the image's VLLM_IMAGE_TAG is "vllm/vllm-openai:v0.27.1"
IMAGE_DIGEST: Final[str] = "sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
ENCODER_CONTAINER: Final[str] = "mesa-clm-encoder"
ENCODER_PORT: Final[int] = 8090
CLM_PORT: Final[int] = 8700
SERVED_MODEL: Final[str] = "qwen3-8b"
# Host LRU of clm-serve's Embedder (patch 0005): 20,000 fp32 4096-d vectors, about 0.33 GB.
EMB_CACHE_SIZE: Final[int] = 20_000

# The encoder container's bearer guard (serving/vllm_auth.py; DESIGN A4): installed by the
# bootstrap under <home>/serve/vllm-auth (0700, the module owner-only), bind-mounted read-only at
# AUTH_MOUNT, on PYTHONPATH, loaded with --middleware. Only AUTH_OPEN_PATHS answer without a key.
AUTH_MODULE: Final[str] = "vllm_auth.py"
AUTH_MIDDLEWARE: Final[str] = "vllm_auth.require_api_key"
AUTH_MOUNT: Final[str] = "/opt/mesa-clm-auth"
AUTH_DIR: Final[str] = "serve/vllm-auth"
AUTH_OPEN_PATHS: Final[tuple[str, ...]] = ("/health",)
MAX_NUM_SEQS: Final[int] = 1 if BATCH_INVARIANCE == "serial" else 8
GPU_MEMORY_UTILIZATION: Final[str] = "0.20"
# The KV cache is pinned to what --max-num-seqs sequences of the whole window need (DESIGN A5;
# plan §6.5's budget "KV 8 x 4096 x 144 KiB = 4.5 GiB"): Qwen3-8B keeps K and V for 36 layers x 8
# KV heads x 128 dimensions in bf16, 147,456 bytes per token (RESEARCH.md, "Encoder model").
# Without the pin vLLM sizes the cache to fill its 0.20 share (9.38-9.87 GiB, varying between
# starts), which put the process above the plan's 24.3 GiB (bench/results/2026-10-01/serving_m1b.json).
KV_BYTES_PER_TOKEN: Final[int] = 2 * 36 * 8 * 128 * 2
KV_CACHE_MEMORY_BYTES: Final[int] = MAX_NUM_SEQS * ENCODER.max_len * KV_BYTES_PER_TOKEN

# DESIGN A5: the container has no network (docker --network none), so nothing its processes bind
# (the engine's TCPStore and Gloo listeners included) is reachable from the host. vLLM serves on a
# unix socket in <home>/run (0700, bind-mounted at SOCKET_MOUNT); the socket unit
# mesa-clm-encoder-proxy.socket owns 127.0.0.1:8090 and hands its connections to
# mesa-clm-encoder-proxy (serving/encoder_proxy.py, installed by the bootstrap into <home>/bin),
# which relays each connection of this account (the kernel's socket diagnostics name the client
# socket's owner; another account's is closed at once) to that socket after checking, without
# following a symlink, that the entry is a socket of this account (the container writes the
# directory). It holds at most PROXY_CONNECTIONS_MAX connections, two descriptors each, under
# LimitNOFILE=PROXY_NOFILE, and runs in a systemd sandbox (the unit's comment says which
# directives a rootless user manager can set up).
ENCODER_NETWORK: Final[str] = "none"
RUN_DIR: Final[str] = "run"
SOCKET_MOUNT: Final[str] = "/run/mesa-clm"
SOCKET_NAME: Final[str] = "encoder.sock"
ENCODER_SOCKET: Final[str] = f"{SOCKET_MOUNT}/{SOCKET_NAME}"
PROXY_SCRIPT: Final[str] = "mesa-clm-encoder-proxy"
PROXY_CONNECTIONS_MAX: Final[int] = 4096
PROXY_NOFILE: Final[int] = 16384

# The container's non-secret environment (deploy/bin/mesa-clm-encoder-run; VLLM_API_KEY arrives
# through --env-file and is never named here). VLLM_NO_USAGE_STATS / DO_NOT_TRACK stop vLLM's
# default usage report to stats.vllm.ai; VLLM_HOST_IP and GLOO_SOCKET_IFNAME put the engine's
# single-process rendezvous on loopback (it would otherwise look for an outbound address).
ENCODER_ENV: Final[dict[str, str]] = {
    "HOME": "/tmp",  # noqa: S108 - the container's own /tmp, not the host's
    "HF_HOME": "/hf",
    "HF_HUB_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "PYTHONPATH": AUTH_MOUNT,
    **({"VLLM_BATCH_INVARIANT": "1"} if BATCH_INVARIANCE == "kernels" else {}),
    "VLLM_HOST_IP": "127.0.0.1",
    "GLOO_SOCKET_IFNAME": "lo",
}
# `vllm serve` arguments after the image, exactly as deploy/bin/mesa-clm-encoder-run passes them.
ENCODER_ARGS: Final[tuple[str, ...]] = (
    ENCODER.model,
    *("--revision", ENCODER.revision, "--served-model-name", SERVED_MODEL),
    *("--runner", "pooling", "--dtype", ENCODER.dtype, "--max-model-len", str(ENCODER.max_len)),
    *("--max-num-seqs", str(MAX_NUM_SEQS), "--no-enable-prefix-caching"),
    *("--gpu-memory-utilization", GPU_MEMORY_UTILIZATION),
    *("--kv-cache-memory-bytes", str(KV_CACHE_MEMORY_BYTES), "--enforce-eager"),
    *("--uds", ENCODER_SOCKET),
    *("--middleware", AUTH_MIDDLEWARE, "--disable-fastapi-docs"),
)


class PatchPin(NamedTuple):
    """One carried patch: ``serving/patches/<id>.patch``; ``pr`` is None for the local ones."""

    id: str
    pr: int | None
    pr_head_sha: str | None


PATCHES: Final[tuple[PatchPin, ...]] = (
    PatchPin("0001-embedder-truncation-side-left", 6, "11211fcab1c27e4779323ff646714dbce475568e"),
    PatchPin("0002-server-loopback-default", 11, "fc4d8d85b41bbe4dc6cf17c6260f7de3c3ff69e8"),
    PatchPin("0003-torch-load-weights-only", 10, "cf230d2a5337a98706a9497e96663f9d9f150b33"),
    PatchPin("0004-cache-arena-slot-leak", 23, "b618c1d8994042bd4cc205f293476e2b295f298c"),
    PatchPin("0005-server-embedder-key-and-cache-size", None, None),
    PatchPin("0006-server-constant-time-auth", None, None),
)

# -- layout under the serving home (default ~/.mesa/clm) ----------------------------------------

DEFAULT_HOME: Final[str] = "~/.mesa/clm"
SECRETS_DIR: Final[str] = "secrets"
CLM_KEY_FILE: Final[str] = "clm.key"
ENCODER_KEY_FILE: Final[str] = "encoder.key"
CLM_ENV_FILE: Final[str] = "clm.env"
ENCODER_ENV_FILE: Final[str] = "encoder.env"
INSTALLED_LOCK: Final[str] = "serving.lock.json"  # the copy the bootstrap installs
SERVE_DIR: Final[str] = "serve"
CLONE_DIR: Final[str] = "serve/CLM"
VENV_PYTHON: Final[str] = "serve/.venv/bin/python"
PATCH_STAMP: Final[str] = "serve/patches.applied.json"  # written by mesa-clm-serve-bootstrap
HEADS_DIR: Final[str] = "heads"
AUTH_INSTALLED: Final[str] = f"{AUTH_DIR}/{AUTH_MODULE}"  # written by mesa-clm-serve-bootstrap

_PRIVATE_DIR_MODE: Final[int] = 0o700
# Keys land in env files read by docker --env-file (no quoting) and systemd EnvironmentFile=
# (quotes and backslashes are special), so only characters both read literally are accepted.
_ENV_SAFE_KEY = re.compile(r"[A-Za-z0-9._~+/=-]{16,512}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IMAGE_REF = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")


class ServingError(RuntimeError):
    """A serving-side file or directory cannot be used; the message never carries a key."""


def serving_home(home: str | Path | None = None) -> Path:
    """The serving home (``~/.mesa/clm`` unless given), home directory expanded."""
    return Path(home if home is not None else DEFAULT_HOME).expanduser()


# -- keys (plan §6.4) ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeysResult:
    """What :func:`init_keys` did: paths and flags only, never a key or a hash of one."""

    secrets_dir: Path
    clm_key: Path
    encoder_key: Path
    clm_env: Path
    encoder_env: Path
    created: tuple[str, ...]
    rotated: bool


def init_keys(secrets_dir: str | Path | None = None, *, rotate: bool = False) -> KeysResult:
    """Create (or, with ``rotate``, replace) the two keys and re-derive both env files.

    Without ``rotate`` existing keys are kept and only missing ones are created, so the call is
    idempotent and also repairs env files left stale by an interrupted run. With ``rotate`` both
    keys are replaced; the caller restarts both units afterwards (the running encoder and
    clm-serve keep the old keys until then). An existing key file must pass the 0600 rule of
    :func:`mesa_clm.secrets.read_secret_file` and contain only env-safe characters; otherwise a
    :class:`~mesa_clm.secrets.SecretError` / :class:`ServingError` names the file and the fix
    (``rotate``), never its content.
    """
    directory = (
        Path(secrets_dir).expanduser() if secrets_dir is not None else serving_home() / SECRETS_DIR
    )
    _ensure_private_dir(directory)
    keys: dict[str, str] = {}
    created: list[str] = []
    for name in (CLM_KEY_FILE, ENCODER_KEY_FILE):
        path = directory / name
        if rotate or not os.path.lexists(path):
            value = secrets.token_urlsafe(32)
            _write_private(path, value + "\n")
            created.append(name)
        else:
            value = read_secret_file(path)
            if not _ENV_SAFE_KEY.fullmatch(value):
                raise ServingError(
                    f"{path}: the key has characters an env file cannot carry literally "
                    "(or is shorter than 16); replace it with init_keys(rotate=True) "
                    "(the planned `mesa-clm serve keys --rotate`)"
                )
        keys[name] = value
    _write_private(directory / CLM_ENV_FILE, _clm_env(keys[CLM_KEY_FILE], keys[ENCODER_KEY_FILE]))
    _write_private(directory / ENCODER_ENV_FILE, _encoder_env(keys[ENCODER_KEY_FILE]))
    keys.clear()
    action = "rotated" if rotate else ("created " + ", ".join(created) if created else "kept")
    logger.info("serving keys %s in %s; env files re-derived", action, directory)
    return KeysResult(
        secrets_dir=directory,
        clm_key=directory / CLM_KEY_FILE,
        encoder_key=directory / ENCODER_KEY_FILE,
        clm_env=directory / CLM_ENV_FILE,
        encoder_env=directory / ENCODER_ENV_FILE,
        created=tuple(created),
        rotated=rotate,
    )


def _clm_env(clm_key: str, encoder_key: str) -> str:
    return (
        "# Generated by mesa-clm (serving.init_keys) from clm.key and encoder.key; do not edit.\n"
        "# mesa-clm-serve.service reads it through EnvironmentFile= (plan §6.2).\n"
        f"CLM_API_KEY={clm_key}\n"
        f"CLM_EMB_API_KEY={encoder_key}\n"
        f"CLM_EMB_CACHE_SIZE={EMB_CACHE_SIZE}\n"
    )


def _encoder_env(encoder_key: str) -> str:
    return (
        "# Generated by mesa-clm (serving.init_keys) from encoder.key; do not edit.\n"
        "# docker run --env-file for the encoder container (plan §6.1).\n"
        f"VLLM_API_KEY={encoder_key}\n"
    )


def _ensure_private_dir(directory: Path) -> None:
    """``directory`` exists, is a real directory owned by this user, and has mode 0700. Missing
    components (``~/.mesa/clm`` itself on a fresh host) are created 0700 too, whatever the umask
    (:func:`mesa_clm.perms.private_dir`)."""
    if not os.path.lexists(directory):
        private_dir(directory)
    st = os.lstat(directory)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ServingError(f"{directory}: the secrets directory must be a real directory")
    if st.st_uid != os.getuid():
        raise ServingError(f"{directory}: owned by uid {st.st_uid}, not the current user")
    if stat.S_IMODE(st.st_mode) != _PRIVATE_DIR_MODE:
        os.chmod(directory, _PRIVATE_DIR_MODE)
        logger.info("secrets directory %s set to mode 0700", directory)


def _write_private(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` at mode 0600 (temp file, fsync, rename)."""
    write_private_text(path, text)


# -- systemd --user units (plan §6.1, §6.2) -----------------------------------------------------

ENCODER_UNIT: Final[str] = "mesa-clm-encoder.service"
SERVE_UNIT: Final[str] = "mesa-clm-serve.service"
HEADROOM_SERVICE: Final[str] = "mesa-clm-headroom.service"
HEADROOM_TIMER: Final[str] = "mesa-clm-headroom.timer"
PROXY_SOCKET: Final[str] = "mesa-clm-encoder-proxy.socket"
PROXY_SERVICE: Final[str] = "mesa-clm-encoder-proxy.service"
UNIT_NAMES: Final[tuple[str, ...]] = (
    ENCODER_UNIT,
    SERVE_UNIT,
    HEADROOM_SERVICE,
    HEADROOM_TIMER,
    PROXY_SOCKET,
    PROXY_SERVICE,
)
DOCS_URL: Final[str] = "https://idss-mesa.github.io/mesa-clm/deploy/serving/"
# MemAvailable floors in GiB: before the encoder starts, and every 5 minutes while it runs.
START_HEADROOM_GIB: Final[int] = 32
RUN_HEADROOM_GIB: Final[int] = 8

_UNIT_HEADER = (
    "# Generated by mesa_clm.serving.render_units(); deploy/systemd/ holds this rendering.\n"
    "# Install into ~/.config/systemd/user/ and `systemctl --user daemon-reload`; installing does\n"
    "# not enable it (enabling starts it at boot and waits for the CARC GPU allocation).\n"
)

_TEMPLATES: Final[dict[str, str]] = {
    ENCODER_UNIT: """\
# mesa-clm encoder: vLLM pooling Qwen/Qwen3-8B behind 127.0.0.1:8090, pinned by digest (plan §6.1).
{header}
[Unit]
Description=mesa-clm encoder (vLLM pooling Qwen3-8B behind 127.0.0.1:8090)
Documentation={docs}
# The container has no network and serves on a unix socket; {proxy_socket} owns
# 127.0.0.1:8090 (DESIGN A5). The encoder requires it and starts after it, so a port that cannot
# be bound (another account holds it) fails the encoder, and clm-serve with it, instead of
# leaving clm-serve to send the encoder key to that account. The headroom timer runs while the
# encoder does (plan §6.5); both stop with it (PartOf=).
Requires={proxy_socket}
After={proxy_socket}
Wants={headroom_timer}
# Every start counts, manual ones included: a fourth start within an hour is refused
# (start-limit-hit), which keeps a recipe that fails after the model load (about 70-80 s and
# 14 GiB on the shared GPU each time) from reloading every few minutes. Run
# `systemctl --user reset-failed {encoder_unit}` before restarting it by hand.
StartLimitIntervalSec=1h
StartLimitBurst=3

[Service]
Type=simple
# The user manager does not carry the docker group, hence sg (RESEARCH.md, sparky-1 host facts).
ExecStartPre=-/usr/bin/sg docker -c "docker rm -f {container}"
ExecStartPre={home}/bin/mesa-clm-check-headroom {start_gib}
ExecStart=/usr/bin/sg docker -c "exec {home}/bin/mesa-clm-encoder-run"
ExecStop=/usr/bin/sg docker -c "docker stop -t 30 {container}"
SuccessExitStatus=143
Restart=on-failure
RestartSec=30
TimeoutStartSec=900
TimeoutStopSec=60
SyslogIdentifier={container}

[Install]
WantedBy=default.target
""",
    SERVE_UNIT: """\
# mesa-clm clm-serve: patched CLM on 127.0.0.1:8700, heads on CPU (plan §6.2; DESIGN D17).
{header}
[Unit]
Description=mesa-clm clm-serve (patched CLM System One API on 127.0.0.1:8700)
Documentation={docs}
Requires={encoder_unit}
After={encoder_unit}

[Service]
Type=simple
EnvironmentFile={home}/secrets/clm.env
Environment=HF_HUB_OFFLINE=1
Environment=PYTHONUNBUFFERED=1
ExecStartPre={home}/bin/mesa-clm-wait-http http://127.0.0.1:{encoder_port}/health 600
ExecStart={home}/serve/.venv/bin/clm-serve --host 127.0.0.1 --port {clm_port} \\
    --emb-url http://127.0.0.1:{encoder_port}/v1/embeddings --emb-model {served_model} \\
    --max-tokens {truncate} --ckpt {home}/heads/{head_file} --ckpt-dir {home}/heads/served \\
    --device cpu --action-cache 512MiB --no-download --no-ui
Restart=on-failure
RestartSec=10
TimeoutStartSec=660
SyslogIdentifier=mesa-clm-serve

[Install]
WantedBy=default.target
""",
    HEADROOM_SERVICE: """\
# mesa-clm headroom check: fails when MemAvailable < {run_gib} GiB (plan §6.5; GB10 unified memory).
{header}
[Unit]
Description=mesa-clm headroom check (MemAvailable >= {run_gib} GiB)
Documentation={docs}

[Service]
Type=oneshot
ExecStart={home}/bin/mesa-clm-check-headroom {run_gib}
""",
    HEADROOM_TIMER: """\
# mesa-clm headroom check every 5 minutes (plan §6.2); started and stopped with the encoder.
{header}
[Unit]
Description=mesa-clm headroom check every 5 minutes
Documentation={docs}
PartOf={encoder_unit}

[Timer]
OnActiveSec=1min
OnUnitActiveSec=5min
AccuracySec=30s
Unit={headroom_service}

[Install]
WantedBy=timers.target
""",
    PROXY_SOCKET: """\
# mesa-clm encoder endpoint: 127.0.0.1:{encoder_port}, forwarded to the container's unix socket (DESIGN A5).
{header}
# Started before {encoder_unit}, which requires it (Requires=, After=), and stopped with
# it (PartOf=); a port it cannot bind fails the encoder. Without an install section it can never
# be enabled on its own.
[Unit]
Description=mesa-clm encoder endpoint (127.0.0.1:{encoder_port})
Documentation={docs}
PartOf={encoder_unit}

[Socket]
ListenStream=127.0.0.1:{encoder_port}
NoDelay=true
""",
    PROXY_SERVICE: """\
# mesa-clm encoder proxy: connections on 127.0.0.1:{encoder_port} to {home}/{run_dir}/{socket_name} (DESIGN A5).
{header}
# Socket-activated by {proxy_socket}. The encoder container creates
# its socket in a 0700 directory it can write; {proxy_script}
# (serving/encoder_proxy.py, installed by the bootstrap) relays only this account's connections
# (the kernel's socket diagnostics name the client socket's owner), connects only to a socket of
# this account there, never through a symlink, and holds at most {proxy_max} connections (two
# descriptors each, hence LimitNOFILE).
[Unit]
Description=mesa-clm encoder proxy (127.0.0.1:{encoder_port} to the encoder's unix socket)
Documentation={docs}
Requires={proxy_socket}
After={proxy_socket}
PartOf={encoder_unit}

[Service]
Type=simple
ExecStart={home}/serve/.venv/bin/python -I {home}/bin/{proxy_script} \\
    --connections-max {proxy_max} {home}/{run_dir}/{socket_name}
LimitNOFILE={proxy_nofile}
# The sandbox (systemd.exec(5)); the port is reachable from every local account. The proxy reads
# the run directory's socket and its own code, asks the kernel's socket diagnostics (AF_NETLINK)
# and relays; it writes nothing. Only what needs no namespace: in a rootless user manager every
# mount-namespace option (ProtectSystem=, ProtectHome=, PrivateTmp=, ProtectProc=, ...) implies
# PrivateUsers=, and Ubuntu's AppArmor confines a process in an unprivileged user namespace
# (unprivileged_userns), which refuses its connect() to the socket the container bound
# ("disconnected path"): every connection to the port would fail. PrivateDevices=,
# ProtectKernelModules=, ProtectKernelLogs=, ProtectClock= and an empty CapabilityBoundingSet=
# cannot be set up there at all (status 218).
NoNewPrivileges=yes
KeyringMode=private
LockPersonality=yes
MemoryDenyWriteExecute=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
RestrictNamespaces=yes
RestrictAddressFamilies=AF_UNIX AF_NETLINK
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
UMask=0077
SyslogIdentifier=mesa-clm-encoder-proxy
""",
}


def unit_home(home: str | Path = DEFAULT_HOME) -> str:
    """How unit files spell the serving home: ``%h/<rel>`` under the user's home directory (so
    systemd expands it), else the absolute path. Paths with whitespace are refused."""
    text = str(home)
    if text == "%h" or text.startswith("%h/"):
        spelled = text.rstrip("/")
    else:
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise ServingError(f"serving home {text!r} must be absolute or start with ~ or %h")
        try:
            rel = path.relative_to(Path.home())
        except ValueError:
            spelled = str(path)
        else:
            spelled = "%h" if str(rel) == "." else f"%h/{rel.as_posix()}"
    if any(c.isspace() for c in spelled):
        raise ServingError(f"serving home {text!r} contains whitespace; unit files cannot use it")
    return spelled


def render_units(home: str | Path = DEFAULT_HOME) -> dict[str, str]:
    """Unit file name -> text for the encoder, clm-serve, the headroom check and timer, and the
    encoder's loopback endpoint (the proxy socket and service, DESIGN A5).

    ``home`` is the serving home (default ``~/.mesa/clm``), spelled through :func:`unit_home` so
    the texts keep ``%h`` for systemd. The default rendering is committed as ``deploy/systemd/``.
    """
    values = {
        "header": _UNIT_HEADER.rstrip("\n"),
        "docs": DOCS_URL,
        "home": unit_home(home),
        "container": ENCODER_CONTAINER,
        "encoder_unit": ENCODER_UNIT,
        "headroom_service": HEADROOM_SERVICE,
        "headroom_timer": HEADROOM_TIMER,
        "proxy_socket": PROXY_SOCKET,
        "proxy_script": PROXY_SCRIPT,
        "proxy_max": PROXY_CONNECTIONS_MAX,
        "proxy_nofile": PROXY_NOFILE,
        "run_dir": RUN_DIR,
        "socket_name": SOCKET_NAME,
        "encoder_port": ENCODER_PORT,
        "clm_port": CLM_PORT,
        "served_model": SERVED_MODEL,
        "truncate": TRUNCATE_PROMPT_TOKENS,
        "head_file": HEAD_FILE,
        "start_gib": START_HEADROOM_GIB,
        "run_gib": RUN_HEADROOM_GIB,
    }
    return {name: template.format(**values) for name, template in _TEMPLATES.items()}


def unit_install_dir() -> Path:
    """Where ``systemctl --user`` looks for units a user installs by hand."""
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config) / "systemd" / "user"


# -- serving.lock.json (plan §6.7) --------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder_recipe(auth_module: str | Path) -> dict[str, Any]:
    """The lock's ``recipe`` block (:class:`mesa_clm.clm.fingerprint.LockRecipe`; DESIGN A3,
    A5): the container arguments and non-secret environment above, the bearer guard with the
    sha256 of ``auth_module`` (``serving/vllm_auth.py``), the clients' truncation cap and the
    container's network mode."""
    path = Path(auth_module)
    if not path.is_file():
        raise ServingError(f"{path}: the encoder's bearer guard is missing")
    return {
        "args": list(ENCODER_ARGS),
        "env": dict(ENCODER_ENV),
        "auth": {
            "middleware": AUTH_MIDDLEWARE,
            "file": AUTH_MODULE,
            "sha256": sha256_file(path),
            "mount": AUTH_MOUNT,
            "open_paths": list(AUTH_OPEN_PATHS),
        },
        "truncate_prompt_tokens": TRUNCATE_PROMPT_TOKENS,
        "network": ENCODER_NETWORK,
    }


def build_lock_body(
    patches_dir: str | Path, auth_module: str | Path | None = None
) -> dict[str, Any]:
    """The signed lock document: the pins above, the sha256 of every patch file and the
    container recipe. ``auth_module`` defaults to ``vllm_auth.py`` next to ``patches_dir``
    (``serving/``)."""
    directory = Path(patches_dir)
    guard = Path(auth_module) if auth_module is not None else directory.parent / AUTH_MODULE
    patches = []
    for pin in PATCHES:
        path = directory / f"{pin.id}.patch"
        if not path.is_file():
            raise ServingError(f"{path}: patch file missing")
        patches.append(
            {
                "id": pin.id,
                "pr": pin.pr,
                "pr_head_sha": pin.pr_head_sha,
                "sha256": sha256_file(path),
            }
        )
    body: dict[str, Any] = {
        "clm_commit": CLM_COMMIT,
        "patches": patches,
        "schema_sha256": SCHEMA_SHA256,
        "head": {
            "repo": HEAD_REPO,
            "revision": HEAD_REVISION,
            "file": HEAD_FILE,
            "size": HEAD_SIZE,
            "sha256": HEAD_SHA256,
        },
        "encoder": ENCODER.as_dict(),
        "image": {"ref": IMAGE_REF, "tag": IMAGE_TAG, "digest": IMAGE_DIGEST},
        "recipe": encoder_recipe(guard),
    }
    return sign_lock_body(body)


def render_lock(patches_dir: str | Path, auth_module: str | Path | None = None) -> str:
    """``serving.lock.json`` as committed: two-space JSON, key order as built, final newline."""
    body = build_lock_body(patches_dir, auth_module)
    return json.dumps(body, indent=2, ensure_ascii=False) + "\n"


def repo_root() -> Path | None:
    """The mesa-clm checkout this module runs from (it holds ``serving/serving.lock.json``), or
    ``None`` in an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "serving" / "serving.lock.json").is_file() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    return None


def default_lock_path(home: str | Path | None = None) -> Path:
    """The checkout's ``serving/serving.lock.json``, else the copy the bootstrap installed."""
    root = repo_root()
    if root is not None:
        return root / "serving" / "serving.lock.json"
    return serving_home(home) / INSTALLED_LOCK


def load_lock(path: str | Path | None = None) -> ServingLock:
    """Read and verify a serving lock (:func:`mesa_clm.clm.fingerprint.load_serving_lock`)."""
    return load_serving_lock(path if path is not None else default_lock_path())


# -- live verification (plan §6.8) --------------------------------------------------------------

CheckStatus = Literal["ok", "fail", "skip"]


@dataclass(frozen=True)
class LockCheck:
    name: str
    status: CheckStatus
    detail: str


class CommandResult(NamedTuple):
    returncode: int
    stdout: str


Runner = Callable[[Sequence[str]], CommandResult]
DockerArgv = Callable[[Sequence[str]], "list[str] | None"]


def run_command(argv: Sequence[str], *, timeout: float = 60.0) -> CommandResult:
    """Run a read-only command detached from the terminal (``sg`` can never prompt); 127 when
    the program is not installed, 124 on timeout or an OS error."""
    exe = shutil.which(argv[0])
    if exe is None:
        return CommandResult(127, "")
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell; docker args are shlex-quoted
            [exe, *argv[1:]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CommandResult(124, "")
    return CommandResult(proc.returncode, proc.stdout)


def docker_argv(args: Sequence[str]) -> list[str] | None:
    """``docker <args>`` directly when this process has the docker group, through ``sg docker``
    when the user is listed in it (login sessions and the user manager lack the gid on the
    serving host), else ``None``."""
    if shutil.which("docker") is None:
        return None
    try:
        group = grp.getgrnam("docker")
    except KeyError:
        return None
    if group.gr_gid in os.getgroups():
        return ["docker", *args]
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return None
    if user in group.gr_mem and shutil.which("sg") is not None:
        return ["sg", "docker", "-c", shlex.join(["docker", *args])]
    return None


def check_serving_lock(
    lock_path: str | Path | None = None,
    *,
    home: str | Path | None = None,
    repo: Path | None = None,
    require_live: bool = False,
    runner: Runner | None = None,
    docker: DockerArgv | None = None,
) -> list[LockCheck]:
    """Compare the serving lock with this host (module docstring); one entry per check.

    ``repo`` is the mesa-clm checkout holding ``serving/patches`` (default: the one this module
    runs from, if any); ``home`` the serving home. ``runner`` and ``docker`` are injectable for
    tests. The lock itself failing to verify is the only check that stops the rest.
    """
    run = runner or run_command
    dock = docker or docker_argv
    root = repo if repo is not None else repo_root()
    base = serving_home(home)
    checks: list[LockCheck] = []
    absent: CheckStatus = "fail" if require_live else "skip"
    path = Path(lock_path) if lock_path is not None else default_lock_path(home)
    try:
        lock = load_serving_lock(path)
    except LockError as exc:
        return [LockCheck("lock", "fail", str(exc))]
    checks.append(LockCheck("lock", "ok", f"{path}: lock_sha {lock.lock_sha[:12]}"))
    checks.append(_check_patch_files(lock, root))
    checks.append(_check_installed_lock(lock, base, absent))
    checks.extend(_check_clone(lock, base, absent, run))
    checks.append(_check_venv_schema(lock, base, absent, run))
    checks.append(_check_head(lock, base, absent))
    checks.append(_check_auth_module(lock, base, absent))
    checks.extend(_check_image(lock, absent, run, dock))
    return checks


def check_encoder_container(
    lock: ServingLock, *, runner: Runner | None = None, docker: DockerArgv | None = None
) -> list[LockCheck]:
    """Only the image and running-container checks of :func:`check_serving_lock`: the pinned
    image present and the encoder container running it with the lock's arguments, environment,
    mounts and network. Docker unreachable, the pinned image absent or no container is a
    ``skip``; a container that departs from the lock is a ``fail``, also when the pinned image is
    absent (a container not created from the pinned digest, or with another recipe). This is what
    ``features build`` and the annotate pre-flight ask before trusting the lock's ``encoder_fp``
    for what the encoder returns (``/v1/models`` cannot tell the recipes apart)."""
    return _check_image(lock, "skip", runner or run_command, docker or docker_argv)


def verify_serving_lock(
    lock_path: str | Path | None = None,
    *,
    home: str | Path | None = None,
    repo: Path | None = None,
    require_live: bool = False,
    runner: Runner | None = None,
    docker: DockerArgv | None = None,
) -> list[str]:
    """The problems :func:`check_serving_lock` found, as ``"<check>: <detail>"`` lines."""
    return [
        f"{c.name}: {c.detail}"
        for c in check_serving_lock(
            lock_path,
            home=home,
            repo=repo,
            require_live=require_live,
            runner=runner,
            docker=docker,
        )
        if c.status == "fail"
    ]


def _check_patch_files(lock: ServingLock, root: Path | None) -> LockCheck:
    if root is None or not (root / "serving" / "patches").is_dir():
        return LockCheck("patch files", "skip", "no mesa-clm checkout with serving/patches")
    directory = root / "serving" / "patches"
    problems = []
    for patch in lock.patches:
        path = directory / f"{patch.id}.patch"
        if not path.is_file():
            problems.append(f"{patch.id}: missing")
        elif sha256_file(path) != patch.sha256:
            problems.append(f"{patch.id}: sha256 differs from the lock")
    listed = {f"{p.id}.patch" for p in lock.patches}
    extra = sorted(p.name for p in directory.glob("*.patch") if p.name not in listed)
    if extra:
        problems.append(f"not in the lock: {', '.join(extra)}")
    if problems:
        return LockCheck("patch files", "fail", "; ".join(problems))
    return LockCheck("patch files", "ok", f"{len(lock.patches)} patches match the lock")


def _check_installed_lock(lock: ServingLock, base: Path, absent: CheckStatus) -> LockCheck:
    path = base / INSTALLED_LOCK
    if not path.is_file():
        return LockCheck("installed lock", absent, f"{path}: not installed (run the bootstrap)")
    try:
        installed = load_serving_lock(path)
    except LockError as exc:
        return LockCheck("installed lock", "fail", str(exc))
    if installed.lock_sha != lock.lock_sha:
        return LockCheck(
            "installed lock",
            "fail",
            f"{path}: lock_sha {installed.lock_sha[:12]} differs from the checkout's "
            f"{lock.lock_sha[:12]}; re-run mesa-clm-serve-bootstrap",
        )
    return LockCheck("installed lock", "ok", f"{path}: same lock_sha")


def _check_clone(
    lock: ServingLock, base: Path, absent: CheckStatus, run: Runner
) -> list[LockCheck]:
    clone = base / CLONE_DIR
    if not (clone / ".git").exists():
        return [LockCheck("serve clone", absent, f"{clone}: no clone (run the bootstrap)")]
    checks: list[LockCheck] = []
    head = run(["git", "-C", str(clone), "rev-parse", "HEAD"])
    commit = head.stdout.strip()
    if head.returncode != 0:
        checks.append(LockCheck("serve clone", "skip", f"{clone}: git rev-parse failed"))
    elif commit != lock.clm_commit:
        checks.append(
            LockCheck(
                "serve clone", "fail", f"{clone}: HEAD {commit[:12]}, lock {lock.clm_commit[:12]}"
            )
        )
    else:
        checks.append(LockCheck("serve clone", "ok", f"{clone}: HEAD {commit[:12]}"))
    schema = clone / "src" / "clm" / "schema.py"
    if not schema.is_file():
        checks.append(LockCheck("clone schema", "fail", f"{schema}: missing"))
    elif sha256_file(schema) != lock.schema_sha256:
        checks.append(LockCheck("clone schema", "fail", f"{schema}: sha256 differs from the lock"))
    else:
        checks.append(LockCheck("clone schema", "ok", f"{schema}: sha256 matches"))
    checks.append(_check_stamp(lock, base / PATCH_STAMP, clone, run))
    return checks


def _check_stamp(lock: ServingLock, stamp: Path, clone: Path, run: Runner) -> LockCheck:
    """The bootstrap's record of the applied series, and the clone's files still equal to the
    tree it recorded (``git diff`` against that tree, without taking the index lock)."""
    if not stamp.is_file():
        return LockCheck("applied patches", "fail", f"{stamp}: missing (re-run the bootstrap)")
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
        applied = [(str(p["id"]), str(p["sha256"])) for p in data["patches"]]
        commit, tree = str(data["clm_commit"]), str(data["tree"])
    except (OSError, ValueError, KeyError, TypeError):
        return LockCheck("applied patches", "fail", f"{stamp}: unreadable")
    expected = [(p.id, p.sha256) for p in lock.patches]
    if commit != lock.clm_commit or applied != expected:
        return LockCheck(
            "applied patches",
            "fail",
            f"{stamp}: the applied series differs from the lock; re-run mesa-clm-serve-bootstrap",
        )
    if not _GIT_OBJECT.fullmatch(tree):
        return LockCheck("applied patches", "fail", f"{stamp}: malformed tree id")
    diff = run(["git", "--no-optional-locks", "-C", str(clone), "diff", "--quiet", tree, "--"])
    if diff.returncode == 1:
        return LockCheck(
            "applied patches",
            "fail",
            f"{clone}: files differ from the patched tree {tree[:12]} (edited after bootstrap)",
        )
    if diff.returncode != 0:
        return LockCheck("applied patches", "skip", f"{clone}: git diff against {tree[:12]} failed")
    return LockCheck("applied patches", "ok", f"{len(applied)} patches applied as locked")


_SCHEMA_ORIGIN = (
    "import importlib.util; s = importlib.util.find_spec('clm.schema'); "
    "print(s.origin if s and s.origin else '')"
)


def _check_venv_schema(
    lock: ServingLock, base: Path, absent: CheckStatus, run: Runner
) -> LockCheck:
    python = base / VENV_PYTHON
    if not python.exists():
        return LockCheck("venv schema", absent, f"{python}: no serve venv (run the bootstrap)")
    res = run([str(python), "-c", _SCHEMA_ORIGIN])
    origin = res.stdout.strip()
    if res.returncode != 0 or not origin:
        return LockCheck("venv schema", "fail", f"{python}: clm.schema is not importable")
    try:
        digest = sha256_file(origin)
    except OSError:
        return LockCheck("venv schema", "fail", f"{origin}: unreadable")
    if digest != lock.schema_sha256:
        return LockCheck("venv schema", "fail", f"{origin}: sha256 differs from the lock")
    return LockCheck("venv schema", "ok", f"{origin}: sha256 matches")


def _check_head(lock: ServingLock, base: Path, absent: CheckStatus) -> LockCheck:
    path = base / HEADS_DIR / lock.head.file
    if not path.is_file():
        return LockCheck("head", absent, f"{path}: not downloaded (run the bootstrap)")
    size = path.stat().st_size
    if size != lock.head.size:
        return LockCheck("head", "fail", f"{path}: {size} bytes, lock {lock.head.size}")
    if sha256_file(path) != lock.head.sha256:
        return LockCheck("head", "fail", f"{path}: sha256 differs from the lock")
    return LockCheck("head", "ok", f"{path}: size and sha256 match")


def _check_auth_module(lock: ServingLock, base: Path, absent: CheckStatus) -> LockCheck:
    """The encoder's bearer guard as the bootstrap installed it: owner-only, the locked sha256."""
    recipe = lock.recipe
    if recipe is None:
        return LockCheck("auth module", "skip", "the lock predates the encoder's bearer guard")
    path = base / AUTH_DIR / recipe.auth.file
    if not path.is_file():
        return LockCheck("auth module", absent, f"{path}: not installed (run the bootstrap)")
    loose = [
        p for p in (path.parent, path) if stat.S_IMODE(p.stat().st_mode) & 0o077 or not _owned(p)
    ]
    if loose:
        return LockCheck(
            "auth module", "fail", f"{loose[0]}: must be owned by this user and private (0700/0600)"
        )
    if sha256_file(path) != recipe.auth.sha256:
        return LockCheck("auth module", "fail", f"{path}: sha256 differs from the lock")
    return LockCheck("auth module", "ok", f"{path}: private, sha256 matches")


def _owned(path: Path) -> bool:
    return path.stat().st_uid == os.getuid()


def _check_image(
    lock: ServingLock, absent: CheckStatus, run: Runner, dock: DockerArgv
) -> list[LockCheck]:
    ref, digest, tag = lock.image.ref, lock.image.digest, lock.image.tag
    if not _IMAGE_REF.fullmatch(ref) or not _DIGEST.fullmatch(digest):
        return [LockCheck("image", "fail", "the lock's image ref or digest is malformed")]
    version = dock(["version", "--format", "{{.Server.Version}}"])
    if version is None or run(version).returncode != 0:
        return [LockCheck("image", absent, "docker is not reachable from this session")]
    pinned = f"{ref}@{digest}"
    inspect = dock(["image", "inspect", pinned, "--format", "{{.Id}}\t{{json .Config.Env}}"])
    res = run(inspect) if inspect is not None else CommandResult(127, "")
    if res.returncode != 0:
        # Docker answers but the pinned image is not here: an encoder container that runs is
        # still compared, or one started from another image would pass as "not verified".
        return [
            LockCheck("image", absent, f"{pinned}: not present (docker pull it by digest)"),
            _container_without_pinned_image(lock, pinned, run, dock),
        ]
    image_id, _, env_json = res.stdout.strip().partition("\t")
    checks = [_image_tag_check(pinned, tag, env_json)]
    checks.append(_container_check(lock, image_id, run, dock))
    return checks


def created_from_pinned(created_from: str, ref: str, digest: str) -> bool:
    """Whether a container's ``Config.Image`` (what ``docker run`` was given) names the pinned
    digest of ``ref``: ``ref@digest`` or ``ref:tag@digest``, with or without ``docker.io/``."""
    name, at, got = created_from.strip().partition("@")
    if not at or got != digest:
        return False
    for prefix in ("docker.io/library/", "docker.io/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]
    return name == ref


def _container_without_pinned_image(
    lock: ServingLock, pinned: str, run: Runner, dock: DockerArgv
) -> LockCheck:
    """The encoder container when the pinned image is absent: none is a ``skip``; one created
    from another image a ``fail``; one created from the pinned digest (since removed here) is
    compared like any other, without the image id."""
    argv = dock(["inspect", ENCODER_CONTAINER, "--format", "{{.Config.Image}}"])
    res = run(argv) if argv is not None else CommandResult(127, "")
    if res.returncode != 0:
        return LockCheck("encoder container", "skip", f"{ENCODER_CONTAINER}: not running")
    created_from = res.stdout.strip()
    if not created_from_pinned(created_from, lock.image.ref, lock.image.digest):
        return LockCheck(
            "encoder container",
            "fail",
            f"{ENCODER_CONTAINER}: runs {created_from or 'an unnamed image'}, not the pinned "
            f"{pinned} (not present here)",
        )
    return _container_check(lock, None, run, dock)


def _image_tag_check(pinned: str, tag: str, env_json: str) -> LockCheck:
    try:
        env = json.loads(env_json) if env_json else []
    except ValueError:
        env = []
    tags = [
        str(e).split("=", 1)[1]
        for e in env
        if isinstance(e, str) and e.startswith("VLLM_IMAGE_TAG=")
    ]
    if tags and not any(t == tag or t.endswith(f":{tag}") for t in tags):
        return LockCheck("image", "fail", f"{pinned}: VLLM_IMAGE_TAG {tags[0]}, lock {tag}")
    return LockCheck("image", "ok", f"{pinned}: present" + (f" ({tags[0]})" if tags else ""))


# Environment names whose values the container check reads besides the locked ones: a stray
# VLLM_BATCH_INVARIANT changes the vectors (DESIGN A3). VLLM_API_KEY is only ever seen by name.
_WATCHED_ENV: Final[frozenset[str]] = frozenset({"VLLM_BATCH_INVARIANT"})
_KEY_ENV: Final[str] = "VLLM_API_KEY"


def inspect_format(names: Iterable[str]) -> str:
    """The ``docker inspect --format`` template of the container check: image id, ``Cmd`` as
    JSON, the environment as ``NAME=VALUE;`` for ``names`` and ``NAME;`` for every other entry
    (so a key's value never leaves docker), the mounts as ``DESTINATION=RW;`` and the network
    mode (``HostConfig.NetworkMode``)."""
    wanted = " ".join(json.dumps(n) for n in sorted(names))
    return (
        "{{.Image}}\t{{json .Config.Cmd}}\t"
        '{{range .Config.Env}}{{$k := index (split . "=") 0}}'
        "{{if eq $k " + wanted + "}}{{.}}{{else}}{{$k}}{{end}};{{end}}\t"
        "{{range .Mounts}}{{.Destination}}={{.RW}};{{end}}\t"
        "{{.HostConfig.NetworkMode}}"
    )


class ContainerState(NamedTuple):
    """What :func:`inspect_format` reports: values only for the watched names."""

    image: str
    cmd: list[str]
    env: dict[str, str | None]  # name -> value (watched names) or None (name only)
    mounts: dict[str, bool]  # destination -> read-write
    network: str = ""  # docker's HostConfig.NetworkMode


def parse_inspect(stdout: str) -> ContainerState | None:
    """:func:`inspect_format` output, or ``None`` when it is not that shape."""
    parts = stdout.strip("\n").split("\t")
    if len(parts) != 5:
        return None
    image, cmd_json, env_text, mounts_text, network = parts
    try:
        cmd = [str(a) for a in json.loads(cmd_json)] if cmd_json else []
    except (ValueError, TypeError):
        return None
    env: dict[str, str | None] = {}
    for entry in filter(None, env_text.split(";")):
        name, sep, value = entry.partition("=")
        env[name] = value if sep else None
    mounts: dict[str, bool] = {}
    for entry in filter(None, mounts_text.split(";")):
        dest, _, rw = entry.rpartition("=")
        mounts[dest] = rw == "true"
    return ContainerState(image, cmd, env, mounts, network.strip())


def _container_check(
    lock: ServingLock, image_id: str | None, run: Runner, dock: DockerArgv
) -> LockCheck:
    """The running encoder container against the lock's recipe; ``image_id`` is the pinned
    image's id (``None``: not present here, the container was created from the pinned digest)."""
    recipe = lock.recipe
    watched = _WATCHED_ENV | set(recipe.env if recipe is not None else ())
    argv = dock(["inspect", ENCODER_CONTAINER, "--format", inspect_format(watched)])
    res = run(argv) if argv is not None else CommandResult(127, "")
    if res.returncode != 0:
        return LockCheck("encoder container", "skip", f"{ENCODER_CONTAINER}: not running")
    state = parse_inspect(res.stdout)
    if state is None:
        return LockCheck("encoder container", "fail", f"{ENCODER_CONTAINER}: unreadable inspect")
    if image_id is not None and state.image != image_id:
        return LockCheck(
            "encoder container",
            "fail",
            f"{ENCODER_CONTAINER}: runs image {state.image[:19]}, not the pinned digest",
        )
    wrong = recipe_mismatches(lock.encoder, state.cmd, recipe)
    if recipe is not None:
        wrong.extend(environment_mismatches(recipe, state))
    if wrong:
        return LockCheck("encoder container", "fail", f"{ENCODER_CONTAINER}: {'; '.join(wrong)}")
    return LockCheck("encoder container", "ok", f"{ENCODER_CONTAINER}: pinned image and recipe")


def environment_mismatches(recipe: LockRecipe, state: ContainerState) -> list[str]:
    """The running container's environment, mounts and network against the lock's recipe: every
    locked variable with its value, no stray ``VLLM_BATCH_INVARIANT``, a key present (by name),
    the guard mounted read-only and, under DESIGN A5, the network mode and the socket directory
    mounted read-write."""
    wrong = []
    for name, want in sorted(recipe.env.items()):
        got = state.env.get(name)
        if got != want:
            wrong.append(f"env {name} {got!r}, lock {want!r}")
    for name in sorted(_WATCHED_ENV - set(recipe.env)):
        if name in state.env:
            wrong.append(f"env {name} is set; the lock does not set it")
    if _KEY_ENV not in state.env:
        wrong.append(f"env {_KEY_ENV} is missing (the bearer guard cannot start without it)")
    rw = state.mounts.get(recipe.auth.mount)
    if rw is None:
        wrong.append(f"{recipe.auth.mount} is not mounted")
    elif rw:
        wrong.append(f"{recipe.auth.mount} is mounted read-write")
    if recipe.network is not None and state.network != recipe.network:
        wrong.append(f"network {state.network or None!r}, lock {recipe.network!r}")
    uds = recipe.flag("--uds")
    if uds is not None:
        socket_dir = posixpath.dirname(uds)
        if not state.mounts.get(socket_dir):
            wrong.append(f"{socket_dir} (the API socket's directory) is not mounted read-write")
    return wrong


# -- the encoder's network namespace (DESIGN A5) ---------------------------------------------------

# /proc/net/tcp{,6} ``st`` column: 0A is TCP_LISTEN.
_TCP_LISTEN: Final[str] = "0A"


class NetnsState(NamedTuple):
    """The encoder container's network namespace as ``/proc/<pid>/net`` shows it: its
    interfaces and every listening TCP socket (``address:port``, IPv6 in brackets)."""

    pid: int
    interfaces: list[str]
    listeners: list[str]


def _loopback(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped.is_loopback
    return addr.is_loopback


def parse_proc_listeners(
    text: str,
) -> list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]]:
    """The listening sockets of one ``/proc/<pid>/net/tcp`` or ``tcp6`` file."""
    out: list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4 or parts[3] != _TCP_LISTEN:
            continue
        host, _, port = parts[1].partition(":")
        try:
            out.append((_proc_address(host), int(port, 16)))
        except ValueError:
            continue
    return out


def read_netns(pid: int, proc: Path = Path("/proc")) -> NetnsState | None:
    """The interfaces (``net/dev``) and listening sockets (``net/tcp``, ``net/tcp6``) of
    ``pid``'s network namespace, read-only; ``None`` when they cannot be read."""
    base = proc / str(pid) / "net"
    try:
        dev = (base / "dev").read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    interfaces = sorted(
        line.split(":", 1)[0].strip() for line in dev.splitlines()[2:] if ":" in line
    )
    listeners: list[str] = []
    for name in ("tcp", "tcp6"):
        try:
            text = (base / name).read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        for addr, port in parse_proc_listeners(text):
            shown = f"[{addr}]" if isinstance(addr, ipaddress.IPv6Address) else str(addr)
            listeners.append(f"{shown}:{port}")
    return NetnsState(pid, interfaces, sorted(set(listeners)))


def netns_problems(state: NetnsState) -> list[str]:
    """Listeners another account could reach: a socket bound to a non-loopback address (a
    wildcard included) in a namespace that has an interface besides ``lo``. In a namespace whose
    only interface is ``lo`` (docker ``--network none``) a wildcard bind reaches nothing."""
    exposed = []
    for entry in state.listeners:
        host = entry.rsplit(":", 1)[0].strip("[]")
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            exposed.append(entry)
            continue
        if not _loopback(addr):
            exposed.append(entry)
    others = [i for i in state.interfaces if i != "lo"]
    if exposed and others:
        return [
            f"{len(exposed)} listener(s) on non-loopback addresses ({', '.join(exposed)}) in a "
            f"namespace with {', '.join(others)}"
        ]
    return []


def container_pid(run: Runner, dock: DockerArgv) -> int | None:
    """The encoder container's main pid on the host (``docker inspect``), ``None`` when docker
    is unreachable or the container is not running."""
    argv = dock(["inspect", ENCODER_CONTAINER, "--format", "{{.State.Pid}}"])
    if argv is None:
        return None
    res = run(argv)
    text = res.stdout.strip()
    if res.returncode != 0 or not text.isdigit() or int(text) <= 0:
        return None
    return int(text)


def recipe_mismatches(
    spec: EncoderSpec, args: Iterable[str], recipe: LockRecipe | None = None
) -> list[str]:
    """Where a ``vllm serve`` argument list departs from the locked encoder recipe (D5) and, with
    ``recipe``, from the locked argument list itself (DESIGN A3)."""
    argv = list(args)

    def value(flag: str) -> str | None:
        return argv[argv.index(flag) + 1] if flag in argv[:-1] else None

    wrong = []
    if not argv or argv[0] != spec.model:
        wrong.append(f"model {argv[0] if argv else None!r}, lock {spec.model!r}")
    expected = {
        "--revision": spec.revision,
        "--dtype": spec.dtype,
        "--max-model-len": str(spec.max_len),
        "--runner": "pooling",
    }
    for flag, want in expected.items():
        got = value(flag)
        if got != want:
            wrong.append(f"{flag} {got!r}, lock {want!r}")
    caching_off = "--no-enable-prefix-caching" in argv
    if spec.prefix_caching == caching_off:
        wrong.append(
            f"prefix caching {'off' if caching_off else 'default'}, lock {spec.prefix_caching}"
        )
    if recipe is not None and argv != recipe.args:
        extra = sorted(set(argv) - set(recipe.args))
        missing = sorted(set(recipe.args) - set(argv))
        detail = "; ".join(
            part
            for part in (
                f"not in the lock: {' '.join(extra)}" if extra else "",
                f"missing: {' '.join(missing)}" if missing else "",
            )
            if part
        )
        wrong.append(f"arguments differ from the lock ({detail or 'order or repetition'})")
    return wrong


__all__ = [
    "AUTH_MIDDLEWARE",
    "CLM_COMMIT",
    "ENCODER",
    "ENCODER_ARGS",
    "ENCODER_ENV",
    "FALLBACK_ENCODER",
    "PATCHES",
    "TRUNCATE_PROMPT_TOKENS",
    "UNIT_NAMES",
    "CommandResult",
    "ContainerState",
    "KeysResult",
    "LockCheck",
    "NetnsState",
    "PatchPin",
    "SecretError",
    "ServingError",
    "build_lock_body",
    "check_encoder_container",
    "check_serving_lock",
    "container_pid",
    "created_from_pinned",
    "default_lock_path",
    "docker_argv",
    "encoder_recipe",
    "environment_mismatches",
    "init_keys",
    "inspect_format",
    "load_lock",
    "netns_problems",
    "parse_inspect",
    "parse_proc_listeners",
    "read_netns",
    "recipe_mismatches",
    "render_lock",
    "render_units",
    "repo_root",
    "run_command",
    "serving_home",
    "sha256_file",
    "unit_home",
    "unit_install_dir",
    "verify_serving_lock",
]
