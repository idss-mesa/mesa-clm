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
* :func:`render_units` - the systemd ``--user`` units (plan §6.1, §6.2). ``%h`` is left for
  systemd to expand; ``deploy/systemd/`` holds the default rendering and a unit test keeps the two
  identical. Installing or enabling a unit is always a separate user action.
* :func:`build_lock_body` / :func:`render_lock` - ``serving/serving.lock.json`` (plan §6.7) from
  the pins below and the sha256 of each carried patch, signed with
  :func:`mesa_clm.clm.fingerprint.sign_lock_body`.
* :func:`check_serving_lock` / :func:`verify_serving_lock` - the lock against this host (plan
  §6.8): the patch files, the copy the bootstrap installed, the serve clone (commit, applied
  series, ``schema.py``), the serve venv's ``clm.schema``, the head file, and the pinned image and
  running encoder container (through ``sg docker`` when the session lacks the docker group).
  Anything *absent* (no clone yet, no docker, not the serving host) is a ``skip`` unless
  ``require_live=True``; anything *present and different* is a ``fail``.
"""

from __future__ import annotations

import contextlib
import grp
import hashlib
import json
import logging
import os
import pwd
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

from mesa_clm.clm.fingerprint import (
    EncoderSpec,
    LockError,
    ServingLock,
    load_serving_lock,
    sign_lock_body,
)
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
)
IMAGE_REF: Final[str] = "vllm/vllm-openai"
IMAGE_TAG: Final[str] = "v0.27.1"  # the image's VLLM_IMAGE_TAG is "vllm/vllm-openai:v0.27.1"
IMAGE_DIGEST: Final[str] = "sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
ENCODER_CONTAINER: Final[str] = "mesa-clm-encoder"
ENCODER_PORT: Final[int] = 8090
CLM_PORT: Final[int] = 8700
SERVED_MODEL: Final[str] = "qwen3-8b"
# Host LRU of clm-serve's Embedder (patch 0005): 20,000 fp32 4096-d vectors, about 0.33 GB.
EMB_CACHE_SIZE: Final[int] = 20_000


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

_PRIVATE_DIR_MODE: Final[int] = 0o700
_PRIVATE_FILE_MODE: Final[int] = 0o600
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
    """``directory`` exists, is a real directory owned by this user, and has mode 0700."""
    try:
        st = os.lstat(directory)
    except FileNotFoundError:
        directory.mkdir(mode=_PRIVATE_DIR_MODE, parents=True)
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
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), _PRIVATE_FILE_MODE)
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# -- systemd --user units (plan §6.1, §6.2) -----------------------------------------------------

ENCODER_UNIT: Final[str] = "mesa-clm-encoder.service"
SERVE_UNIT: Final[str] = "mesa-clm-serve.service"
HEADROOM_SERVICE: Final[str] = "mesa-clm-headroom.service"
HEADROOM_TIMER: Final[str] = "mesa-clm-headroom.timer"
UNIT_NAMES: Final[tuple[str, ...]] = (ENCODER_UNIT, SERVE_UNIT, HEADROOM_SERVICE, HEADROOM_TIMER)
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
# mesa-clm encoder: vLLM pooling Qwen/Qwen3-8B on 127.0.0.1:8090, pinned by digest (plan §6.1).
{header}
[Unit]
Description=mesa-clm encoder (vLLM pooling Qwen3-8B on 127.0.0.1:8090)
Documentation={docs}

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
    --max-tokens {max_len} --ckpt {home}/heads/{head_file} --ckpt-dir {home}/heads/served \\
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
# mesa-clm headroom check every 5 minutes (plan §6.2).
{header}
[Unit]
Description=mesa-clm headroom check every 5 minutes
Documentation={docs}

[Timer]
OnActiveSec=1min
OnUnitActiveSec=5min
AccuracySec=30s
Unit={headroom_service}

[Install]
WantedBy=timers.target
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
    """Unit file name -> text for the encoder, clm-serve and the headroom check and timer.

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
        "encoder_port": ENCODER_PORT,
        "clm_port": CLM_PORT,
        "served_model": SERVED_MODEL,
        "max_len": ENCODER.max_len,
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


def build_lock_body(patches_dir: str | Path) -> dict[str, Any]:
    """The signed lock document: the pins above plus the sha256 of every patch file."""
    directory = Path(patches_dir)
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
    }
    return sign_lock_body(body)


def render_lock(patches_dir: str | Path) -> str:
    """``serving.lock.json`` as committed: two-space JSON, key order as built, final newline."""
    return json.dumps(build_lock_body(patches_dir), indent=2, ensure_ascii=False) + "\n"


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
    checks.extend(_check_image(lock, absent, run, dock))
    return checks


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
        return [LockCheck("image", absent, f"{pinned}: not present (docker pull it by digest)")]
    image_id, _, env_json = res.stdout.strip().partition("\t")
    checks = [_image_tag_check(pinned, tag, env_json)]
    checks.append(_container_check(lock, image_id, run, dock))
    return checks


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


def _container_check(lock: ServingLock, image_id: str, run: Runner, dock: DockerArgv) -> LockCheck:
    argv = dock(["inspect", ENCODER_CONTAINER, "--format", "{{.Image}}\t{{json .Config.Cmd}}"])
    res = run(argv) if argv is not None else CommandResult(127, "")
    if res.returncode != 0:
        return LockCheck("encoder container", "skip", f"{ENCODER_CONTAINER}: not running")
    running_image, _, cmd_json = res.stdout.strip().partition("\t")
    if running_image != image_id:
        return LockCheck(
            "encoder container",
            "fail",
            f"{ENCODER_CONTAINER}: runs image {running_image[:19]}, not the pinned digest",
        )
    try:
        cmd = [str(a) for a in json.loads(cmd_json)] if cmd_json else []
    except (ValueError, TypeError):
        cmd = []
    wrong = recipe_mismatches(lock.encoder, cmd)
    if wrong:
        return LockCheck("encoder container", "fail", f"{ENCODER_CONTAINER}: {'; '.join(wrong)}")
    return LockCheck("encoder container", "ok", f"{ENCODER_CONTAINER}: pinned image and recipe")


def recipe_mismatches(spec: EncoderSpec, args: Iterable[str]) -> list[str]:
    """Where a ``vllm serve`` argument list departs from the locked encoder recipe (D5)."""
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
    return wrong


__all__ = [
    "CLM_COMMIT",
    "ENCODER",
    "PATCHES",
    "UNIT_NAMES",
    "CommandResult",
    "KeysResult",
    "LockCheck",
    "PatchPin",
    "SecretError",
    "ServingError",
    "build_lock_body",
    "check_serving_lock",
    "default_lock_path",
    "docker_argv",
    "init_keys",
    "load_lock",
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
