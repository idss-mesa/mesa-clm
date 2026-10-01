"""``mesa-clm doctor``: what this host can run, honestly (plan §6.8; the M0 and M1 subset).

Every check is one :class:`Check` with a status ``ok``, ``warn`` or ``fail``; a report is ``ok``
when nothing failed and the CLI exits 1 otherwise. M0 checks the pins and the ground the later
milestones stand on:

* the vendored files against ``vendored.sha256`` (package-relative: the manifest lives at the
  repository root, so the check is skipped, and says so, when mesa-clm runs from a wheel);
* the effective configuration (its secret-free ``config_sha256``, profile, tier, planner);
* the Python, duckdb, mesa-mcp and mesa-ducklake versions, with the installed git commits of the
  two MESA packages compared against the DESIGN D0 pins, and duckdb against the 1.5.x window the
  shared catalog is written with (D11);
* mesa-mcp's plugin API, ``load_plugins`` and ``register_tool(meta=)`` (D14), and the ``clm``
  entry point this package declares;
* the policy defaults (D9: they load, every active task has thresholds, the profile exists);
* the provenance path (D11: the DuckDB file's directory is writable or can be created), the
  label store, any configured key file and the serving pair's default ones
  (``~/.mesa/clm/secrets/{clm,encoder}.key``, named "(default)"; stat-ed for mode 0600, never
  opened), the eval root and the OLS fixture directory;
* ``permissions``: group or other bits on mesa-clm's state (the data home, ``locks/``,
  ``secrets/``, the sidecar with its ``.wal`` and lock, the feature stores) are a warning that
  names the paths and the ``chmod`` that fixes them (:mod:`mesa_clm.perms`; read with ``lstat``).

M1 adds:

* ``framings lock``: the framing keys against ``framings.lock.json`` (``framings --check``);
* ``schema sha256``: :func:`mesa_clm.render.schema_sha256` against the ``vendored.sha256`` entry
  and the serving pin (D4, D5);
* ``sidecar schema``: the ``mesa_clm`` schema version of the sidecar, read without creating
  anything but the D11 lock file (a DuckDB file is opened read-only under the shared flock;
  Postgres through :func:`mesa_clm.provenance.migrate.current_version`);
* ``feature store``: the store of the live ``encoder_fp`` (format, the vector recipe it was
  built under against the live lock's, the float16 export gate, vectors embedded under other
  ``lock_sha`` values); a format-1 store or another vector recipe fails (DESIGN D5, the
  implementation note "The feature store");
* ``serving lock``: :func:`mesa_clm.serving.check_serving_lock` when the serving home
  (``~/.mesa/clm``) exists (skipped by ``--quick`` outside serve mode);
* ``host`` (``MemAvailable``; warn below the 8 GiB running floor, plan §6.5) and ``gpu_budget``
  (active CARC vLLM backends, ``systemctl list-units carc-vllm@*``, read-only);
* the serving endpoints. Every configured URL first passes the clients' loopback rule
  (:func:`mesa_clm.net.assert_loopback`: loopback, or ``allow_remote`` and https, never userinfo)
  and is printed redacted; one that does not is a failure and is never called. Without serve
  mode one ``serving`` line says whether both ``/health`` routes answer (``warn`` when not:
  annotate falls back to ``--provider fake`` or fails). **Serve mode** (``--serve``, or
  automatically when both ``mesa-clm-*`` user units are active) runs the live probes: the
  **binds** (``ss -ltne``: loopback only, sockets of this account, the encoder's port held by the
  user manager's ``mesa-clm-encoder-proxy.socket`` while that unit is active; DESIGN A5); the
  **encoder socket** (``<serving home>/run`` holds nothing but ``encoder.sock``, a socket of this
  account, never a symlink); the **units** (``systemctl --user show``: the six units systemd
  loaded are the checkout's rendering, none awaits a ``daemon-reload``, and the running proxy
  has the rendered command line and its sandbox, ``/proc/<pid>/status``; a host that pulled a
  fix without installing the units fails); the **encoder network**: the container's own
  namespace read from ``/proc/<pid>/net``, where a listener on a non-loopback address next to an
  interface besides ``lo`` fails (DESIGN A5; ``ss`` on the host cannot see it); the **headroom
  timer** active (warn); ``/health`` 200 on both ports without a key; **the 401 matrix**: without
  a key every encoder route vLLM registers (17, DESIGN A4) and an unknown path, and clm-serve's
  ``/v1/models``, ``/v1/systemone`` and ``/v1/rank``, must answer 401; the authenticated model
  lists (``qwen3-8b`` with the lock's ``root``, window and route; ``clm-latest`` and
  ``clm-raw``) and one golden ``/v1/systemone`` call (its shape). The full probes run under
  ``--serve`` or in serve mode without ``--quick``: **encoder goldens** (the texts of
  ``encoder_golden_<encoder_fp>.npz`` one per request, bitwise against the reference; as one batch
  within ``1 - 1e-6`` under a batch-invariant recipe, DESIGN A3), the **long-input probe** (a text
  over the window completes, truncated at ``max_len - 1`` from the left, with cosine >= 0.9999
  against its last ``max_len - 1`` token ids), the **systemone parity** (the golden question
  recomputed on the local route, encoder plus the pinned head's numpy projection, within 1e-4
  of clm-serve for ``clm-latest`` when the head export is present, and ``clm-raw``) and the
  **upstream drift** probes (warn only: the README quickstart against issue #15 with the
  unrounded differences, the tides rank against the model card). An unreachable endpoint or a
  failed full probe fails under ``--serve`` (or auto-detected serve mode without ``--quick``)
  and is a warning otherwise; an open route is always a failure.

Everything that touches the host or the network goes through :class:`ServeProbes` (serving
home, httpx transport, command runner, ``/proc/meminfo``, ``/proc``, the feature root), so the
tests inject fakes and the hermetic suite never probes a port. No key is ever read into a report
line.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shlex
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple
from urllib.parse import urlsplit

import duckdb
import httpx

import mesa_clm
from mesa_clm import __version__
from mesa_clm.clm.fingerprint import ServingLock
from mesa_clm.config import Config, config_sha256, duckdb_path, expand_path, redact_dsn
from mesa_clm.net import EndpointError, assert_loopback, redact_url
from mesa_clm.secrets import SecretError, check_secret_file_stat
from mesa_clm.serving import (
    ENCODER_UNIT,
    HEADROOM_TIMER,
    PROXY_SERVICE,
    PROXY_SOCKET,
    RUN_DIR,
    RUN_HEADROOM_GIB,
    SOCKET_NAME,
    START_HEADROOM_GIB,
    UNIT_NAMES,
    CommandResult,
    docker_argv,
    run_command,
    serving_home,
)

Status = Literal["ok", "warn", "fail"]

# The git commits pyproject.toml pins (DESIGN D0). A different installed commit is a warning,
# not a failure: it runs, but the pin bump that changes it must update this table as well.
PINNED_COMMITS: Final[dict[str, str]] = {
    "mesa-mcp": "c74f3aa1a3caade558f063f4f637a9127dba82d9",
    "mesa-ducklake": "7bc143fefb093f0a2ace748a7036598147533ca0",
}
# The shared MESA catalog is written by duckdb 1.5.x everywhere on the stack (DESIGN D11).
DUCKDB_MIN: Final = (1, 5, 5)
DUCKDB_MAX: Final = (1, 6)
PYTHON_MIN: Final = (3, 11)
VENDORED_MANIFEST: Final = "vendored.sha256"
PLUGIN_ENTRY_POINT: Final = "clm"
# (check name, config section, the *_api_key_file field) of every key file the doctor stats.
_KEY_FILES: Final[tuple[tuple[str, str, str], ...]] = (
    ("clm", "clm", "api_key_file"),
    ("encoder", "encoder", "api_key_file"),
    ("planner gateway", "planner", "gateway_api_key_file"),
)


@dataclass
class Check:
    """One line of the report; ``ok`` is anything but a failure (a warning still passes)."""

    name: str
    status: Status
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "fail"


@dataclass
class HealthReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def counts(self) -> dict[str, int]:
        out = {"ok": 0, "warn": 0, "fail": 0}
        for c in self.checks:
            out[c.status] += 1
        return out

    def lines(self) -> list[str]:
        tags = {"ok": "ok", "warn": "WARN", "fail": "FAIL"}
        return [
            f"[{tags[c.status]}] {c.name}{': ' + c.detail if c.detail else ''}" for c in self.checks
        ]

    def as_dict(self) -> dict[str, Any]:
        """The ``--json`` shape (and the body of ``mesa_clm_health`` from M3)."""
        return {
            "ok": self.ok,
            "mesa_clm": __version__,
            "summary": self.counts(),
            "checks": [asdict(c) for c in self.checks],
        }


@dataclass
class ServeProbes:
    """What the host and serving checks touch, injectable for tests: the serving home (``None``:
    ``~/.mesa/clm``), the httpx transport of every probe (``None``: the network), the command
    runner (``ss``, ``systemctl``, git and the venv through :func:`mesa_clm.serving.run_command`),
    the docker argv builder, the meminfo file and the directories searched for the encoder
    goldens (``None``: the serving home's ``goldens/``, then the checkout's ``.local/serving/``,
    where ``scripts/serving_probes.py`` writes them)."""

    home: Path | None = None
    transport: httpx.BaseTransport | None = None
    runner: Callable[[Sequence[str]], CommandResult] = run_command
    docker: Callable[[Sequence[str]], list[str] | None] = docker_argv
    meminfo: Path = Path("/proc/meminfo")
    golden_dirs: tuple[Path, ...] | None = None
    # /proc, read for the encoder container's network namespace (DESIGN A5).
    proc: Path = Path("/proc")
    # The feature stores' root; None: the configured features.dir.
    features_dir: Path | None = None

    @classmethod
    def offline(cls, home: Path | None = None) -> ServeProbes:
        """No serving home, every connection refused, every command missing (exit 127), no
        meminfo: what the hermetic test suite runs the doctor with."""

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline (ServeProbes.offline)", request=request)

        return cls(
            home=home if home is not None else Path("/nonexistent/mesa-clm-serving-home"),
            transport=httpx.MockTransport(refuse),
            runner=lambda argv: CommandResult(127, ""),
            docker=lambda args: None,
            meminfo=Path("/nonexistent/meminfo"),
            golden_dirs=(),
            proc=Path("/nonexistent/proc"),
            features_dir=Path("/nonexistent/mesa-clm-features"),
        )


def default_probes() -> ServeProbes:
    """The real host (the hermetic test suite replaces this with :meth:`ServeProbes.offline`)."""
    return ServeProbes()


def units_active(runner: Callable[[Sequence[str]], CommandResult]) -> bool:
    """Both serving units active under the user manager (``systemctl --user is-active``)."""
    units = UNIT_NAMES[:2]
    res = runner(["systemctl", "--user", "is-active", *units])
    states = res.stdout.split()
    return len(states) == len(units) and all(s == "active" for s in states)


def doctor(
    cfg: Config,
    *,
    quick: bool = False,
    serve: bool | None = False,
    probes: ServeProbes | None = None,
) -> HealthReport:
    """Run every check against ``cfg`` (module docstring). ``quick`` skips the vendored-file
    hashing and the serving-lock verification; ``serve`` ``True`` runs the live serving probes,
    ``None`` runs them when both serving units are active (the CLI's default), ``False`` never."""
    probes = probes if probes is not None else default_probes()
    rep = HealthReport()
    _config_check(cfg, rep)
    _python_check(rep)
    _duckdb_check(rep)
    for name in ("mesa-mcp", "mesa-ducklake"):
        _dependency_check(name, rep)
    _plugin_api_check(rep)
    _entry_point_check(rep)
    if not quick:
        _vendored_check(rep)
    _framings_check(rep)
    _schema_sha_check(rep)
    _policy_check(cfg, rep)
    _store_checks(cfg, rep)
    _sidecar_check(cfg, rep)
    _feature_store_check(cfg, rep, probes)
    _secret_checks(cfg, rep)
    _permissions_check(cfg, rep, probes)
    _path_checks(cfg, rep)
    _serving_checks(cfg, rep, quick=quick, serve=serve, probes=probes)
    return rep


# -- configuration and runtime ------------------------------------------------------------------


def _config_check(cfg: Config, rep: HealthReport) -> None:
    rep.add(
        "config",
        "ok",
        f"sha256 {config_sha256(cfg)[:12]}; profile={cfg.policy.profile} tier={cfg.decider.tier} "
        f"planner={cfg.planner.kind} secrets={cfg.secrets} vm_id={cfg.vm_id}",
    )


def _python_check(rep: HealthReport) -> None:
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info[:2] >= PYTHON_MIN
    rep.add("python", "ok" if ok else "fail", version if ok else f"{version}: needs >= 3.11")


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """``"1.5.6"`` -> ``(1, 5, 6)``; ``None`` when the text is not dotted integers."""
    try:
        return tuple(int(part) for part in text.split(".")[:3])
    except ValueError:
        return None


def _duckdb_check(rep: HealthReport) -> None:
    version = duckdb.__version__
    parsed = _version_tuple(version)
    if parsed is None:
        rep.add("duckdb", "warn", f"{version}: unparseable version; the stack pins 1.5.x (D11)")
    elif DUCKDB_MIN <= parsed < DUCKDB_MAX:
        rep.add("duckdb", "ok", f"{version} (shared catalog window 1.5.5 <= v < 1.6)")
    else:
        rep.add(
            "duckdb",
            "warn",
            f"{version}: the shared MESA catalog is written by duckdb 1.5.x (D11); "
            "direct history mode may not open it",
        )


def dist_info(name: str) -> tuple[str, str | None] | None:
    """``(version, git commit)`` of an installed distribution, the commit from the
    ``direct_url.json`` a git install writes; ``None`` when it is not installed."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    commit: str | None = None
    raw = dist.read_text("direct_url.json")
    if raw:
        try:
            info = json.loads(raw)
            commit = info.get("vcs_info", {}).get("commit_id") or None
        except (ValueError, AttributeError):
            commit = None
    return dist.version, commit


def _dependency_check(name: str, rep: HealthReport) -> None:
    info = dist_info(name)
    if info is None:
        rep.add(name, "fail", "not installed")
        return
    version, commit = info
    pinned = PINNED_COMMITS.get(name)
    if commit is None:
        # A local editable checkout (the live venv installs -e ~/.mesa/repos/...) records no commit.
        note = f"; pin {pinned[:7]} unverified" if pinned else ""
        rep.add(name, "ok", f"{version} (no git commit recorded{note})")
    elif pinned and commit != pinned:
        rep.add(name, "warn", f"{version} @ {commit[:7]}; pyproject.toml pins {pinned[:7]} (D0)")
    else:
        rep.add(name, "ok", f"{version} @ {commit[:7]} (pinned)")


def _plugin_api_check(rep: HealthReport) -> None:
    """mesa-mcp >= c74f3aa: the entry-point loader and ``register_tool(meta=)`` (DESIGN D14)."""
    try:
        server = importlib.import_module("mesa_mcp.server")
    except Exception as exc:
        rep.add(
            "mesa-mcp plugin api",
            "fail",
            f"cannot import mesa_mcp.server: {type(exc).__name__}: {exc}",
        )
        return
    missing: list[str] = []
    if not callable(getattr(server, "load_plugins", None)):
        missing.append("load_plugins")
    register = getattr(server, "register_tool", None)
    if not callable(register):
        missing.append("register_tool")
    else:
        try:
            names = set(inspect.signature(register).parameters)
        except (TypeError, ValueError):
            names = set()
        if "meta" not in names:
            missing.append("register_tool(meta=)")
    if missing:
        rep.add(
            "mesa-mcp plugin api",
            "fail",
            f"missing {', '.join(missing)}: this mesa-mcp predates c74f3aa and cannot load the "
            "clm plugin (D14)",
        )
        return
    group = getattr(server, "PLUGIN_ENTRY_POINT_GROUP", "mesa_mcp.tools")
    rep.add(
        "mesa-mcp plugin api", "ok", f"load_plugins and register_tool(meta=) present; group {group}"
    )


def _entry_point_check(rep: HealthReport) -> None:
    eps = {ep.name: ep.value for ep in importlib.metadata.entry_points(group="mesa_mcp.tools")}
    target = eps.get(PLUGIN_ENTRY_POINT)
    if target is None:
        rep.add(
            "plugin entry point",
            "warn",
            f"no mesa_mcp.tools entry point {PLUGIN_ENTRY_POINT!r}: mesa-clm is not installed as a "
            "distribution, so mesa-mcp will not load it",
        )
    else:
        rep.add(
            "plugin entry point",
            "ok",
            f"{PLUGIN_ENTRY_POINT} = {target} (registers no tools until M3)",
        )


# -- vendored files -----------------------------------------------------------------------------


def repo_root() -> Path | None:
    """The mesa-clm checkout this package runs from (two directories above the package), or
    ``None`` for an installed wheel, which ships neither ``vendored.sha256`` nor ``tests/``."""
    root = Path(mesa_clm.__file__).resolve().parents[2]
    if (root / VENDORED_MANIFEST).is_file() and (root / "src" / "mesa_clm").is_dir():
        return root
    return None


def parse_manifest(text: str) -> list[tuple[str, str]]:
    """``(sha256, relative path)`` per entry of a ``sha256sum`` manifest (blank lines and ``#``
    comments skipped; a ``*`` binary marker before the path is dropped)."""
    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        out.append((digest.strip().lower(), name.strip().lstrip("*")))
    return out


def verify_vendored(root: Path) -> tuple[list[str], list[str]]:
    """``(matching, problems)`` for ``<root>/vendored.sha256``: the paths whose sha256 equals the
    manifest's, and one line per file that is missing or differs (digests, never contents)."""
    matching: list[str] = []
    problems: list[str] = []
    for expected, name in parse_manifest((root / VENDORED_MANIFEST).read_text(encoding="utf-8")):
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"{name}: sha256 {actual[:12]} != {expected[:12]}")
        else:
            matching.append(name)
    return matching, problems


def _vendored_check(rep: HealthReport) -> None:
    root = repo_root()
    if root is None:
        rep.add(
            "vendored files",
            "ok",
            f"skipped: no {VENDORED_MANIFEST} beside the package (installed from a wheel)",
        )
        return
    matching, problems = verify_vendored(root)
    if problems:
        rep.add("vendored files", "fail", "; ".join(problems))
    else:
        rep.add("vendored files", "ok", f"{len(matching)} files match {VENDORED_MANIFEST}")


# -- policy, stores, secrets, paths -------------------------------------------------------------


def _policy_check(cfg: Config, rep: HealthReport) -> None:
    from mesa_clm.policy_defaults import DEFAULTS_PATH, load_policy_defaults

    path = expand_path(cfg.policy.policy_path) if cfg.policy.policy_path else DEFAULTS_PATH
    try:
        policy = load_policy_defaults(path)
    except Exception as exc:
        rep.add("policy defaults", "fail", f"{path}: {type(exc).__name__}: {exc}")
        return
    numeric = sorted(t for t, th in policy.tasks.items() if th.auto is not None)
    detail = (
        f"{path.name}: {len(policy.tasks)} tasks, numeric auto: "
        f"{', '.join(numeric) or 'none (proposed-only)'}; profile {cfg.policy.profile}"
    )
    if cfg.policy.profile in policy.profiles:
        rep.add("policy defaults", "ok", detail)
    else:
        rep.add("policy defaults", "fail", detail + " is not defined in the policy file")


def _store_checks(cfg: Config, rep: HealthReport) -> None:
    dsn = cfg.provenance.dsn
    path = duckdb_path(dsn)
    if path is None:
        rep.add(
            "provenance path",
            "warn",
            f"{redact_dsn(dsn)}: not a duckdb:/// DSN; no local path to probe (the Postgres "
            "sidecar is checked under 'sidecar schema'; the M0 label verbs need DuckDB)",
        )
        return
    if path.exists():
        if path.is_file() and os.access(path, os.W_OK):
            rep.add("provenance path", "ok", str(path))
        else:
            rep.add("provenance path", "fail", f"{path}: exists but is not a writable file")
    else:
        # The nearest existing ancestor must be writable so the directory chain can be created.
        probe = path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.is_dir() and os.access(probe, os.W_OK):
            note = "" if path.parent.exists() else f" ({path.parent} will be created)"
            rep.add("provenance path", "ok", f"{path}{note}")
        else:
            rep.add("provenance path", "fail", f"{path}: {probe} is not a writable directory")
    # The M0 labels table lives in the same DuckDB file (mesa_clm.labels, D11).
    if path.is_file():
        rep.add("labels store", "ok", f"{path} ({path.stat().st_size / 1024:.0f} KiB)")
    else:
        rep.add("labels store", "ok", "none yet (`mesa-clm labels ingest-neon-eval` creates it)")


def _secret_checks(cfg: Config, rep: HealthReport) -> None:
    """Key files are stat-ed for the 0600 rule of :mod:`mesa_clm.secrets`, never opened. The
    serving pair's default files (``~/.mesa/clm/secrets/{clm,encoder}.key``, used when no key is
    configured and the file exists) are checked and named as defaults."""
    for label, section, attr in _KEY_FILES:
        conf = getattr(cfg, section)
        file: str | None
        default = False
        if section in ("clm", "encoder"):
            file, default = conf.effective_api_key_file()
        else:
            file = getattr(conf, attr)
        if not file:
            continue
        path = expand_path(file)
        name = f"{label} key file"
        where = f"{path} (default)" if default else str(path)
        try:
            check_secret_file_stat(path, path.stat())
        except FileNotFoundError:
            rep.add(name, "fail", f"{where}: not found")
        except (SecretError, OSError) as exc:
            rep.add(name, "fail", str(exc))
        else:
            rep.add(name, "ok", f"{where}: regular file, mode 0600 or stricter (not read)")
    if cfg.secrets == "keyring":
        present = importlib.util.find_spec("keyring") is not None
        rep.add(
            "keyring",
            "ok" if present else "fail",
            "keyring package importable"
            if present
            else "MESA_CLM_SECRETS=keyring but the keyring package is not installed",
        )


def _permission_targets(cfg: Config, probes: ServeProbes) -> list[Path]:
    """What mesa-clm keeps on disk and must keep owner-only (:mod:`mesa_clm.perms`): the data
    home, its ``locks/`` and ``secrets/`` (with the key and env files), the configured sidecar
    with its ``.wal``, lock directory and lock file, and every feature store. Read with
    ``lstat`` only; the serve venv, the CLM clone and the heads are the bootstrap's."""
    from mesa_clm.learn.features import DB_FILE, LOCK_FILE

    home = serving_home(probes.home)
    paths: list[Path] = [home, home / "locks", home / "locks" / "provenance.lock"]
    paths.append(home / RUN_DIR)  # the encoder's unix socket directory (DESIGN A5)
    secrets = home / "secrets"
    paths.append(secrets)
    if secrets.is_dir():
        paths.extend(sorted(secrets.iterdir()))
    db = duckdb_path(cfg.provenance.dsn)
    if db is not None:
        lock = db.parent / "locks" / "provenance.lock"
        paths.extend([db, db.with_name(db.name + ".wal"), lock.parent, lock])
    features = expand_path(cfg.features.dir)
    paths.append(features)
    if features.is_dir():
        for store in sorted(p for p in features.iterdir() if p.is_dir()):
            paths.extend([store, store / DB_FILE, store / (DB_FILE + ".wal"), store / LOCK_FILE])
    return list(dict.fromkeys(paths))


def _permissions_check(cfg: Config, rep: HealthReport, probes: ServeProbes) -> None:
    """Group or other bits on anything :func:`_permission_targets` lists: a warning naming the
    paths and the fix (the next write through mesa-clm also tightens its own lock and store
    files; directories the user made are never chmod-ed by mesa-clm)."""
    from mesa_clm.perms import loose

    found = loose(_permission_targets(cfg, probes))
    if not found:
        rep.add(
            "permissions", "ok", "mesa-clm's state is owner-only (directories 0700, files 0600)"
        )
        return
    dirs = [str(p) for p, _ in found if p.is_dir()]
    files = [str(p) for p, _ in found if not p.is_dir()]
    fix = "; ".join(
        part
        for part in (
            f"chmod 700 {' '.join(dirs)}" if dirs else "",
            f"chmod 600 {' '.join(files)}" if files else "",
        )
        if part
    )
    rep.add(
        "permissions",
        "warn",
        ", ".join(f"{p} {mode:04o}" for p, mode in found) + f" (owner-only state; fix: {fix})",
    )


def _path_checks(cfg: Config, rep: HealthReport) -> None:
    if cfg.eval_root:
        # The same rule `labels ingest-neon-eval` applies before it opens anything.
        from mesa_clm.learn.labels import check_eval_root

        try:
            root = check_eval_root(cfg.eval_root)
        except FileNotFoundError as exc:
            rep.add("eval root", "fail", str(exc))
        else:
            rep.add("eval root", "ok", str(root))
    if cfg.ols.fixtures != "off":
        fixtures = expand_path(cfg.ols.fixtures_dir)
        if fixtures.is_dir():
            n = sum(1 for _ in fixtures.glob("*.json"))
            rep.add(
                "ols fixtures", "ok", f"{cfg.ols.fixtures}: {n} recorded responses under {fixtures}"
            )
        elif cfg.ols.fixtures == "replay":
            rep.add("ols fixtures", "fail", f"replay: {fixtures} does not exist")
        else:
            rep.add("ols fixtures", "ok", f"{cfg.ols.fixtures}: {fixtures} will be created")


# -- M1: framings lock, schema contract, sidecar schema -----------------------------------------

_SCHEMA_ENTRY: Final = "src/mesa_clm/_vendor/clm/schema.py"
_MAX_PROBLEMS: Final = 5


def _framings_check(rep: HealthReport) -> None:
    """The framing keys against ``framings.lock.json`` (the ``framings --check`` rule)."""
    from mesa_clm import framings

    if not framings.LOCK_PATH.is_file() and repo_root() is None:
        rep.add(
            "framings lock",
            "ok",
            f"skipped: no {framings.LOCK_PATH.name} beside the package (installed from a wheel)",
        )
        return
    problems = framings.lock_drift()
    if problems:
        more = f" (+{len(problems) - _MAX_PROBLEMS} more)" if len(problems) > _MAX_PROBLEMS else ""
        rep.add(
            "framings lock",
            "fail",
            "; ".join(problems[:_MAX_PROBLEMS])
            + more
            + "; after a deliberate framing change run `mesa-clm framings --update-lock`",
        )
        return
    rep.add(
        "framings lock",
        "ok",
        f"{framings.LOCK_PATH.name} in sync; lock_sha {framings.lock_sha()[:12]}",
    )


def _schema_sha_check(rep: HealthReport) -> None:
    """``render.schema_sha256()`` (what every record cites, D5) against the ``vendored.sha256``
    entry and the serving lock's pin (the schema clm-serve renders with, D4)."""
    from mesa_clm import render
    from mesa_clm.serving import SCHEMA_SHA256

    actual = render.schema_sha256()
    problems: list[str] = []
    sources = ["serving pin"]
    root = repo_root()
    if root is not None:
        entries = {
            name: sha for sha, name in parse_manifest((root / VENDORED_MANIFEST).read_text())
        }
        expected = entries.get(_SCHEMA_ENTRY)
        if expected is None:
            problems.append(f"{VENDORED_MANIFEST} has no entry for {_SCHEMA_ENTRY}")
        elif expected != actual:
            problems.append(f"{VENDORED_MANIFEST} pins {expected[:12]}")
        sources.insert(0, f"{VENDORED_MANIFEST} entry")
    if actual != SCHEMA_SHA256:
        problems.append(f"the serving pin is {SCHEMA_SHA256[:12]}")
    if problems:
        rep.add(
            "schema sha256", "fail", f"render.schema_sha256() {actual[:12]}: " + "; ".join(problems)
        )
    else:
        rep.add("schema sha256", "ok", f"{actual[:12]} = {' = '.join(sources)}")


def _duckdb_schema(path: Path) -> tuple[int | None, set[str]]:
    """``(schema version or None, table names)`` of the ``mesa_clm`` schema in a DuckDB file,
    opened read-only under the sidecar's shared flock (nothing is created)."""
    from mesa_clm.provenance.store import SCHEMA, DuckDBStore

    with DuckDBStore(path).connect(read_only=True) as con:
        tables = {
            str(r[0])
            for r in con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = ?", [SCHEMA]
            ).fetchall()
        }
        if "schema_versions" not in tables:
            return None, tables
        got = con.execute(f"SELECT max(version) FROM {SCHEMA}.schema_versions").fetchone()  # noqa: S608
    return (int(got[0]) if got and got[0] is not None else None), tables


def _sidecar_check(cfg: Config, rep: HealthReport) -> None:
    """The ``mesa_clm`` schema version of the configured sidecar, without creating anything."""
    from mesa_clm.provenance.store import RUN_TABLES, SCHEMA_VERSION, is_postgres_dsn

    dsn = cfg.provenance.dsn
    path = duckdb_path(dsn)
    expected = {*RUN_TABLES, "labels", "audits", "schema_versions"}
    if path is not None:
        if not path.is_file():
            rep.add(
                "sidecar schema",
                "ok",
                f"none yet (the first annotate creates schema v{SCHEMA_VERSION})",
            )
            return
        try:
            version, tables = _duckdb_schema(path)
        except (duckdb.Error, OSError) as exc:
            rep.add(
                "sidecar schema", "fail", f"{path}: cannot open read-only ({type(exc).__name__})"
            )
            return
        if version is None:
            status: Status = "ok" if tables <= {"labels"} else "warn"
            rep.add(
                "sidecar schema",
                status,
                f"{path}: no schema_versions ({', '.join(sorted(tables)) or 'no tables'}); the "
                f"next write or `mesa-clm provenance migrate` creates schema v{SCHEMA_VERSION}",
            )
            return
        missing = sorted(expected - tables)
    elif is_postgres_dsn(dsn):
        from mesa_clm.provenance.migrate import current_version

        try:
            version = current_version(dsn)
        except ImportError:
            rep.add(
                "sidecar schema", "warn", "Postgres DSN but psycopg is not installed (pg extra)"
            )
            return
        except Exception as exc:  # connection refused, auth, DNS: named, never the password
            rep.add("sidecar schema", "fail", f"{redact_dsn(dsn)}: {type(exc).__name__}")
            return
        missing = []
    else:
        rep.add("sidecar schema", "fail", f"{redact_dsn(dsn)}: unsupported DSN")
        return
    where = str(path) if path is not None else redact_dsn(dsn)
    if version > SCHEMA_VERSION:
        rep.add(
            "sidecar schema",
            "fail",
            f"{where}: schema v{version} is newer than this mesa-clm (v{SCHEMA_VERSION})",
        )
    elif version < SCHEMA_VERSION or missing:
        gap = f"; missing {', '.join(missing)}" if missing else ""
        rep.add(
            "sidecar schema",
            "warn",
            f"{where}: schema v{version}{gap}; run `mesa-clm provenance migrate`",
        )
    else:
        rep.add("sidecar schema", "ok", f"{where}: mesa_clm schema v{version}")


def _feature_store_check(cfg: Config, rep: HealthReport, probes: ServeProbes) -> None:
    """``feature store`` (plan §6.8 "Stores"; DESIGN D5, the implementation note "The feature
    store"): the store of the live ``encoder_fp`` under ``features.dir``, read under the shared
    flock without creating anything. A store of another format (format 1 kept float16 vectors
    only) or built under another vector recipe than the live lock's is a failure (every read
    would be refused, K4); a float16 round trip below the export gate is a warning; vectors
    embedded under other ``lock_sha`` values of the same vector recipe are reported. Stores of
    other fingerprints are listed, never read."""
    import re

    from mesa_clm.learn.features import DB_FILE, FP16_MIN_COSINE, FeatureStore, FeatureStoreError

    root = probes.features_dir if probes.features_dir is not None else expand_path(cfg.features.dir)
    lock = _live_lock(probes)
    live_fp = lock.encoder_fp if lock is not None else None
    others = (
        sorted(
            p.name
            for p in root.iterdir()
            if re.fullmatch(r"[0-9a-f]{12}", p.name)
            and p.name != live_fp
            and (p / DB_FILE).is_file()
        )
        if root.is_dir()
        else []
    )
    tail = f"; stores of other fingerprints, never read here: {', '.join(others)}" if others else ""
    if lock is None:
        rep.add("feature store", "ok", f"skipped: no serving lock verifies here{tail}")
        return
    store = FeatureStore.for_lock(root, lock)
    if not store.exists():
        rep.add(
            "feature store",
            "ok",
            f"none yet for encoder_fp {lock.encoder_fp} under {root} (`mesa-clm features build "
            f"--snapshot <labels snapshot>` creates it){tail}",
        )
        return
    try:
        info = store.stats()
    except FeatureStoreError as exc:
        rep.add("feature store", "fail", f"{exc}{tail}")
        return
    except (duckdb.Error, OSError) as exc:
        rep.add("feature store", "fail", f"{store.path}: cannot read ({type(exc).__name__}){tail}")
        return
    gate = info.get("fp16_min_cosine")
    by_lock: dict[str, int] = info.get("vectors_by_lock_sha") or {}
    elsewhere = sum(n for sha, n in by_lock.items() if sha != lock.lock_sha)
    detail = (
        f"{store.path}: {info.get('format')}, {info['vectors']} vectors, {info['truncated']} "
        f"truncated texts; vector recipe {str(info.get('vector_recipe_sha256'))[:12]} is the "
        "live lock's; fp16 round trip min cosine "
        + ("-" if gate is None else f"{gate:.7f}")
        + f" (export gate >= {FP16_MIN_COSINE})"
        + (
            f"; {elsewhere} vectors embedded under another lock_sha of the same vector recipe"
            if elsewhere
            else ""
        )
    )
    status: Status = "warn" if gate is not None and gate < FP16_MIN_COSINE else "ok"
    rep.add("feature store", status, detail + tail)


# -- M1: serving (plan §6.4, §6.5, §6.8) ------------------------------------------------------------

PROBE_TIMEOUT_S: Final = 5.0
# The long-input probe, the goldens and the systemone parity embed up to 4,095 tokens per call.
SLOW_PROBE_TIMEOUT_S: Final = 60.0
_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "[::1]", "localhost"})
# Every route of the encoder but exactly /health answers 401 without the key (DESIGN A4): the 17
# routes vLLM's own build_app registers for the locked arguments
# (bench/results/2026-10-01/serving_m1b.json#/auth/enumeration) plus a path no route has, since
# the guard is the outermost layer. clm-serve guards its /v1 routes in each handler, so each is
# asked separately. An empty JSON body is enough: without the key no handler runs.
ENCODER_GUARDED_ROUTES: Final[tuple[tuple[str, str], ...]] = (
    ("GET", "/v1/models"),
    ("POST", "/v1/embeddings"),
    ("POST", "/pooling"),
    ("POST", "/invocations"),
    ("POST", "/score"),
    ("POST", "/v1/score"),
    ("POST", "/rerank"),
    ("POST", "/v1/rerank"),
    ("POST", "/v2/rerank"),
    ("POST", "/v2/embed"),
    ("POST", "/tokenize"),
    ("POST", "/detokenize"),
    ("GET", "/metrics"),
    ("GET", "/version"),
    ("GET", "/load"),
    ("GET", "/ping"),
    ("POST", "/ping"),
    ("GET", "/mesa-clm-doctor-no-such-route"),
)
CLM_GUARDED_ROUTES: Final[tuple[tuple[str, str], ...]] = (
    ("GET", "/v1/models"),
    ("POST", "/v1/systemone"),
    ("POST", "/v1/rank"),
)
# One fixed /v1/systemone question the doctor asks (plan §6.8 "golden /v1/systemone"): a NEON
# column as a target state and two candidate terms plus the term anchor. The light probe checks
# the answer's shape; the full one recomputes it on the local route (encoder vectors, the pinned
# head's numpy projection) and requires clm-serve to agree within GOLDEN_SYSTEMONE_GATE. The
# column is the non-bench SRER card's (plan §9's live-smoke card, tests/fixtures/cards-srer), so
# a routine doctor run never answers a pre-registered bench item; until 2026-10-01 it asked
# brd_countdata's observerDistance, a labelled term.fits target (DESIGN, "G1 freeze"). Terms and
# definitions as EMBL-EBI OLS gives them.
GOLDEN_STATE: Final[dict[str, Any]] = {
    "card": {
        "dataset": "DP1.00004.001.BP_30min",
        "product_title": "Barometric pressure",
    },
    "scope": "column",
    "aspect": "measurement",
    "column": {
        "name": "staPresMean",
        "description": "Arithmetic mean of station pressure",
        "dtype": "real",
        "unit": "kilopascal",
    },
}
GOLDEN_CRITERIA: Final[dict[str, str]] = {
    "PATO:0001025": "pressure: A physical quality that inheres in a bearer by virtue of the "
    "bearer's amount of force per unit area it exerts.",
    "UO:0000110": "pascal: A pressure unit which is equal to the pressure or stress on a surface "
    "caused by a force of 1 newton spread over a surface of 1 m^[2].",
    "__none__": "None of these terms is the right concept for this target.",
}
# Plan §5.6/§8: clm-serve against the local route, in probability.
GOLDEN_SYSTEMONE_GATE: Final = 1e-4
# Plan §6.8: the encoder goldens and the long-input probe, in cosine; batch invariance (DESIGN
# A3: a batch agrees with one text per request to 1 - 3.3e-15 under the adopted kernels).
GOLDEN_COSINE_GATE: Final = 0.9999
BATCH_COSINE_GATE: Final = 1.0 - 1e-6
# The encoder goldens: <serving home>/goldens/ or the checkout's .local/serving/, one file per
# encoder_fp, written by scripts/serving_probes.py (texts, text_sha256, vectors: one text per
# request on the recipe the lock names).
GOLDEN_DIR: Final = "goldens"
GOLDEN_FILE: Final = "encoder_golden_{fp}.npz"
# Upstream drift (plan §6.8; warn only): the CLM README quickstart at bb42c6c5 against issue #15's
# GB10 report, and the model card's tides rank (RESEARCH.md, issues #15 and #3). The references
# were measured on another stack; DESIGN A3 records why a kernel change can move urgency past the
# tolerance, so the doctor reports the unrounded difference and never re-tunes either.
QUICKSTART_STATE: Final = "Customer: my invoice was charged twice and nobody answers the phone!"
QUICKSTART_REFERENCE: Final[dict[str, float]] = {
    "urgency": 0.8366,
    "billing": 0.9888,
    "frustration": 2.0,
}
QUICKSTART_TOLERANCE: Final = 0.01
TIDES_QUESTION: Final = "What causes tides on Earth?"
TIDES_ANSWERS: Final[tuple[str, ...]] = (
    "The Moon's gravitational pull.",
    "Photosynthesis in plants.",
    "Because the Earth is round.",
)
TIDES_REFERENCE: Final = 0.993
TIDES_TOLERANCE: Final = 0.005


def meminfo_gib(path: Path, field_name: str = "MemAvailable") -> float | None:
    """A ``/proc/meminfo`` field in GiB, ``None`` when the file or field is missing."""
    try:
        text = path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name.strip() == field_name:
            parts = rest.split()
            if parts and parts[0].isdigit():
                return int(parts[0]) / (1024 * 1024)
    return None


def _host_checks(rep: HealthReport, probes: ServeProbes) -> None:
    """``host``: MemAvailable against the running floor; ``gpu_budget``: CARC's co-tenancy."""
    avail = meminfo_gib(probes.meminfo)
    if avail is None:
        rep.add("host", "ok", f"skipped: no MemAvailable in {probes.meminfo}")
    elif avail < RUN_HEADROOM_GIB:
        rep.add(
            "host",
            "warn",
            f"MemAvailable {avail:.1f} GiB < {RUN_HEADROOM_GIB} GiB, the encoder's running floor "
            "(plan §6.5: stop both mesa-clm units)",
        )
    else:
        rep.add(
            "host",
            "ok",
            f"MemAvailable {avail:.1f} GiB (floors: {START_HEADROOM_GIB} GiB to start the encoder, "
            f"{RUN_HEADROOM_GIB} GiB while it runs)",
        )
    res = probes.runner(
        ["systemctl", "list-units", "--no-legend", "--plain", "--state=active", "carc-vllm@*"]
    )
    if res.returncode == 127:
        rep.add("gpu_budget", "ok", "systemctl not available; CARC presence unknown")
        return
    active = [line.split()[0] for line in res.stdout.splitlines() if line.strip()]
    if not active:
        rep.add("gpu_budget", "ok", "no CARC vLLM backend active (carc-vllm@*)")
        return
    tight = avail is not None and avail < START_HEADROOM_GIB
    rep.add(
        "gpu_budget",
        "warn" if tight else "ok",
        f"CARC vLLM active: {', '.join(active)}"
        + (
            f"; MemAvailable {avail:.1f} GiB < {START_HEADROOM_GIB} GiB, the encoder could not start"
            if tight
            else ""
        )
        + "; the encoder's 0.20 share needs the written CARC allocation (plan §6.5)",
    )


def _port(url: str, default: int) -> int:
    try:
        return urlsplit(url).port or default
    except ValueError:
        return default


class Listener(NamedTuple):
    """One listening socket of ``ss -ltne`` output: its local address, the owning uid (``ss``
    omits ``uid:0``, so a missing field is root) and its cgroup when the kernel reports one."""

    host: str
    uid: int
    cgroup: str | None


def parse_listener_details(ss_output: str, port: int) -> list[Listener]:
    """The sockets listening on ``port`` in ``ss -ltne`` output, with owner and cgroup."""
    out: list[Listener] = []
    for line in ss_output.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "LISTEN":
            continue
        host, _, p = parts[3].rpartition(":")
        if p != str(port):
            continue
        uid, cgroup = 0, None
        for token in parts[4:]:
            if token.startswith("uid:") and token[4:].isdigit():
                uid = int(token[4:])
            elif token.startswith("cgroup:"):
                cgroup = token[len("cgroup:") :]
        out.append(Listener(host.split("%", 1)[0], uid, cgroup))
    return out


def parse_listeners(ss_output: str, port: int) -> list[str]:
    """The local addresses listening on ``port`` in ``ss -ltn`` output (``127.0.0.1``,
    ``[::1]``, ``0.0.0.0``, ``*`` ...)."""
    return [listener.host for listener in parse_listener_details(ss_output, port)]


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _unit_states(probes: ServeProbes, units: Sequence[str]) -> dict[str, str] | None:
    """``systemctl --user is-active`` of ``units`` (one state per line), ``None`` when
    systemctl is missing or does not answer one line per unit."""
    res = probes.runner(["systemctl", "--user", "is-active", *units])
    states = res.stdout.split()
    if res.returncode == 127 or len(states) != len(units):
        return None
    return dict(zip(units, states, strict=True))


def _binds_check(
    rep: HealthReport, probes: ServeProbes, ports: dict[str, int], bad: Status
) -> None:
    """``serving binds`` (DESIGN A5): each port listens on loopback only, in sockets of this
    account (another account's socket on a free port would get the key: the units are not
    enabled at boot), and while ``mesa-clm-encoder-proxy.socket`` is active the encoder's port
    belongs to it (the user manager's ``init.scope`` cgroup, when ``ss`` reports cgroups); the
    encoder unit active without that socket unit is a failure too."""
    res = probes.runner(["ss", "-ltne"])
    if res.returncode != 0:
        rep.add("serving binds", "warn", "`ss -ltne` not available; binds not checked")
        return
    me = os.getuid()
    manager = f"/user@{me}.service/init.scope"
    units = _unit_states(probes, (PROXY_SOCKET, ENCODER_UNIT))
    failed: list[str] = []
    missing: list[str] = []
    seen: list[str] = []
    for name, port in ports.items():
        found = parse_listener_details(res.stdout, port)
        if not found:
            missing.append(f"nothing listens on :{port} ({name})")
            continue
        wrong: list[str] = []
        exposed = [x.host for x in found if not _is_loopback(x.host)]
        if exposed:
            wrong.append(f"{name} :{port} listens on {', '.join(exposed)} (not loopback)")
        foreign = sorted({x.uid for x in found if x.uid != me})
        if foreign:
            wrong.append(
                f"{name} :{port} is held by a socket of uid {', '.join(map(str, foreign))}, "
                f"not of this account (uid {me}): a client's key would go to it (DESIGN A5)"
            )
        via = ""
        if name == "encoder" and units is not None:
            if units[PROXY_SOCKET] == "active":
                others = sorted({x.cgroup for x in found if x.cgroup and manager not in x.cgroup})
                if others:
                    wrong.append(
                        f"encoder :{port} is held from {others[0]}, not by the user manager's "
                        f"{PROXY_SOCKET}"
                    )
                via = f" ({PROXY_SOCKET})"
            elif units[ENCODER_UNIT] == "active":
                wrong.append(
                    f"{ENCODER_UNIT} is active but {PROXY_SOCKET} is {units[PROXY_SOCKET]}: "
                    f":{port} is not the socket unit's (install the units of deploy/systemd/)"
                )
        failed.extend(wrong)
        if not wrong:
            hosts = ", ".join(sorted({x.host for x in found}))
            seen.append(f"{name} {hosts}:{port}{via}")
    if failed:
        rep.add("serving binds", "fail", "; ".join(failed + missing))
    elif missing:
        rep.add("serving binds", bad, "; ".join(missing))
    else:
        rep.add("serving binds", "ok", f"loopback only, sockets of uid {me}: " + "; ".join(seen))


def _encoder_socket_check(rep: HealthReport, probes: ServeProbes) -> None:
    """``encoder socket`` (DESIGN A5): the container writes ``<serving home>/run``, the
    directory of its API socket, so the doctor checks what the endpoint proxy will connect to:
    a real owner-only directory holding nothing but ``encoder.sock``, and that entry a socket of
    this account (never a symlink, which would redirect the host's :8090 to another socket).
    No socket yet is a warning (the model is loading, or the fallback serves :8090)."""
    run = serving_home(probes.home) / RUN_DIR
    me = os.getuid()
    try:
        st = os.lstat(run)
    except FileNotFoundError:
        rep.add(
            "encoder socket",
            "warn",
            f"{run}: missing (the units have not started, or the in-process fallback serves :8090)",
        )
        return
    except OSError as exc:
        rep.add("encoder socket", "fail", f"{run}: cannot be read ({type(exc).__name__})")
        return
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != me or stat.S_IMODE(st.st_mode) & 0o077:
        rep.add(
            "encoder socket",
            "fail",
            f"{run}: must be a directory of this account with mode 0700 (the run script makes it)",
        )
        return
    entries = sorted(os.listdir(run))
    others = [e for e in entries if e != SOCKET_NAME]
    sock = run / SOCKET_NAME
    problems: list[str] = []
    if others:
        problems.append(f"unexpected entries in {run}: {', '.join(others[:5])}")
    if SOCKET_NAME in entries:
        sst = os.lstat(sock)
        if not stat.S_ISSOCK(sst.st_mode) or sst.st_uid != me:
            kind = "a symlink" if stat.S_ISLNK(sst.st_mode) else "not a socket of this account"
            problems.append(f"{sock} is {kind}: the endpoint proxy must reach the encoder only")
    if problems:
        rep.add(
            "encoder socket",
            "fail",
            "; ".join(problems) + " (the container can write this directory; restart the "
            "encoder unit, which recreates it, and look for what changed it)",
        )
    elif SOCKET_NAME not in entries:
        rep.add("encoder socket", "warn", f"{sock}: no socket yet (the model may still be loading)")
    else:
        rep.add("encoder socket", "ok", f"{sock}: a socket of uid {me}, alone in a 0700 directory")


def _encoder_network_check(rep: HealthReport, probes: ServeProbes) -> None:
    """``encoder network`` (DESIGN A5): the encoder container's network namespace, read from
    ``/proc/<pid>/net`` (interfaces, listening sockets). ``ss`` on the host cannot see it, and
    the engine's rendezvous and collective sockets listen there without a key: a listener on a
    non-loopback address in a namespace with an interface besides ``lo`` fails (any local
    account reaches it through the docker bridge). A warning when the container cannot be
    inspected (no docker from this session, the container not running, the fallback serving)."""
    from mesa_clm.serving import container_pid, netns_problems, read_netns

    pid = container_pid(probes.runner, probes.docker)
    if pid is None:
        rep.add(
            "encoder network",
            "warn",
            "not checked: the encoder container cannot be inspected from this session (docker "
            "unreachable, or the container is not running)",
        )
        return
    state = read_netns(pid, probes.proc)
    if state is None:
        rep.add("encoder network", "warn", f"not checked: {probes.proc}/{pid}/net is not readable")
        return
    problems = netns_problems(state)
    if problems:
        rep.add(
            "encoder network",
            "fail",
            "; ".join(problems) + " (reachable without the key from any local account; the "
            "lock's recipe runs the container with --network none, DESIGN A5)",
        )
        return
    where = (
        "inside a namespace whose only interface is lo"
        if state.interfaces == ["lo"]
        else "all on loopback"
    )
    rep.add(
        "encoder network",
        "ok",
        f"container namespace interfaces {', '.join(state.interfaces) or 'none'}; "
        f"{len(state.listeners)} listening socket(s), {where}",
    )


def _headroom_timer_check(rep: HealthReport, probes: ServeProbes) -> None:
    """``headroom timer`` (plan §6.5): while the units serve, the 5-minute MemAvailable check
    should run; the encoder unit pulls it in (``Wants=``), so a stopped timer means a unit file
    from before DESIGN A5 or a manual stop. A warning, never a failure."""
    res = probes.runner(["systemctl", "--user", "is-active", HEADROOM_TIMER])
    state = (res.stdout.split() or [""])[0]
    if res.returncode == 127:
        rep.add("headroom timer", "ok", "skipped: systemctl not available")
    elif state == "active":
        rep.add("headroom timer", "ok", f"{HEADROOM_TIMER} active (MemAvailable every 5 min)")
    else:
        rep.add(
            "headroom timer",
            "warn",
            f"{HEADROOM_TIMER} is {state or 'not active'}: the 5-minute MemAvailable check is off "
            f"while serving (plan §6.5); `systemctl --user start {HEADROOM_TIMER}` (start only)",
        )


# What `systemctl --user show` is asked for each unit by the `serving units` check.
_UNIT_PROPS: Final[tuple[str, ...]] = (
    "Id",
    "LoadState",
    "ActiveState",
    "FragmentPath",
    "NeedDaemonReload",
    "DropInPaths",
)


def _show_units(probes: ServeProbes, units: Sequence[str]) -> dict[str, dict[str, str]] | None:
    """``systemctl --user show`` of ``units`` (one block of properties per unit, keyed by
    ``Id``), ``None`` when systemctl is missing or prints nothing usable."""
    res = probes.runner(
        ["systemctl", "--user", "show", *units, *(f"--property={p}" for p in _UNIT_PROPS)]
    )
    if res.returncode != 0:
        return None
    out: dict[str, dict[str, str]] = {}
    for block in res.stdout.split("\n\n"):
        props = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if props.get("Id"):
            out[props["Id"]] = props
    return out or None


def _exec_start_argv(unit_text: str) -> list[str]:
    """The ``ExecStart=`` argv of a rendered unit, ``%h`` expanded as systemd expands it."""
    joined = unit_text.replace("\\\n", " ")
    (line,) = [x for x in joined.splitlines() if x.startswith("ExecStart=")]
    return shlex.split(line.split("=", 1)[1].replace("%h", str(Path.home())))


def _proxy_process(probes: ServeProbes, expected: list[str]) -> tuple[list[str], str]:
    """Problems with the running proxy process (``MainPID`` of the proxy service) and a short
    description of it: its command line must be the rendered unit's and it must run in the
    sandbox (``NoNewPrivs: 1``, ``Seccomp: 2``), which a process started before the units were
    installed lacks. No process yet is no problem: the socket unit starts it on the first
    connection."""
    res = probes.runner(["systemctl", "--user", "show", PROXY_SERVICE, "--property=MainPID"])
    text = res.stdout.strip()
    pid = text.split("=", 1)[1] if text.startswith("MainPID=") else ""
    if res.returncode != 0 or not pid.isdigit():
        return [], "the proxy process was not inspected (no MainPID)"
    if pid == "0":
        return [], "the proxy has not started yet (the socket unit starts it on a connection)"
    try:
        argv = [
            a.decode("utf-8", "replace")
            for a in (probes.proc / pid / "cmdline").read_bytes().split(b"\0")[:-1]
        ]
        status = (probes.proc / pid / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], f"the proxy process {pid} was not inspected ({probes.proc}/{pid} unreadable)"
    fields = dict(
        (k.strip(), v.strip()) for k, _, v in (ln.partition(":") for ln in status.splitlines())
    )
    problems: list[str] = []
    if argv != expected:
        problems.append(
            f"the running proxy (pid {pid}) is not the rendered unit's command "
            f"({' '.join(argv[:3]) or 'unknown'} ...): restart the units after installing them"
        )
    if fields.get("NoNewPrivs") != "1" or fields.get("Seccomp") != "2":
        problems.append(
            f"the running proxy (pid {pid}) is not sandboxed (NoNewPrivs "
            f"{fields.get('NoNewPrivs', '?')}, Seccomp {fields.get('Seccomp', '?')}): it started "
            "before the units of deploy/systemd/ were installed; restart the units"
        )
    return problems, f"proxy pid {pid} sandboxed with the rendered command line"


def _units_check(rep: HealthReport, probes: ServeProbes) -> None:
    """``serving units`` (DESIGN A5, the review of 70dbefe): the binds and the run directory
    look the same under the old wiring (``systemd-socket-proxyd``, an encoder that only
    ``Wants=`` its socket) and the revised one, so the doctor compares what systemd loaded with
    the checkout's rendering (:func:`mesa_clm.serving.render_units` for the serving home): each
    unit's ``FragmentPath`` text, nothing pending a ``daemon-reload``, and the running proxy
    started from it (its command line, and the sandbox in ``/proc/<pid>/status``). A unit that
    differs or a stale proxy fails; units that are not installed (the in-process fallback
    serving) or drop-ins that may override the rendering are warnings."""
    from mesa_clm.serving import ServingError, render_units

    home = serving_home(probes.home)
    try:
        rendered = render_units(home)
    except ServingError as exc:
        rep.add("serving units", "warn", f"not checked: {exc}")
        return
    shown = _show_units(probes, UNIT_NAMES)
    if shown is None:
        rep.add("serving units", "warn", "not checked: `systemctl --user show` not available")
        return
    failed: list[str] = []
    warned: list[str] = []
    missing = [u for u in UNIT_NAMES if shown.get(u, {}).get("LoadState") != "loaded"]
    for unit in UNIT_NAMES:
        props = shown.get(unit, {})
        if unit in missing:
            continue
        fragment = props.get("FragmentPath", "")
        try:
            text = Path(fragment).read_text(encoding="utf-8") if fragment else None
        except OSError:
            text = None
        if text is None:
            failed.append(f"{unit}: its unit file {fragment or '(none)'} cannot be read")
        elif text != rendered[unit]:
            failed.append(
                f"{unit}: {fragment} is not the checkout's rendering (install deploy/systemd/ "
                "and `systemctl --user daemon-reload`, docs/deploy/serving.md)"
            )
        if props.get("NeedDaemonReload") == "yes":
            failed.append(
                f"{unit}: changed on disk since loaded (`systemctl --user daemon-reload`)"
            )
        if props.get("DropInPaths"):
            warned.append(f"{unit}: drop-ins may override the rendering ({props['DropInPaths']})")
    proxy_note = ""
    if PROXY_SERVICE not in missing:
        problems, proxy_note = _proxy_process(probes, _exec_start_argv(rendered[PROXY_SERVICE]))
        failed.extend(problems)
    if missing and len(missing) < len(UNIT_NAMES):
        failed.append(f"not installed: {', '.join(missing)} (install all six of deploy/systemd/)")
    elif missing:
        warned.append(
            "no mesa-clm unit is installed (the in-process fallback may be serving the ports)"
        )
    if failed:
        rep.add("serving units", "fail", "; ".join(failed + warned))
    elif warned:
        rep.add("serving units", "warn", "; ".join(warned))
    else:
        rep.add(
            "serving units",
            "ok",
            f"the {len(UNIT_NAMES)} units loaded are the checkout's rendering for {home}; "
            + proxy_note,
        )


def _http(probes: ServeProbes) -> httpx.Client:
    return httpx.Client(timeout=PROBE_TIMEOUT_S, trust_env=False, transport=probes.transport)


def _status_of(client: httpx.Client, url: str, method: str = "GET") -> int | None:
    """The HTTP status of an unauthenticated request (an empty JSON body for a POST), ``None``
    when nothing answered."""
    try:
        if method == "POST":
            return client.post(url, json={}).status_code
        return client.get(url).status_code
    except httpx.HTTPError:
        return None


def _endpoints(cfg: Config, rep: HealthReport) -> dict[str, str]:
    """The two configured base URLs the probes may call: each passes the loopback rule every
    client applies (:func:`mesa_clm.net.assert_loopback`: loopback, or ``allow_remote`` and
    https, never userinfo) or is reported as a failure by its redacted form and not called."""
    out: dict[str, str] = {}
    for name, url, what in (
        ("encoder", cfg.encoder.url, "encoder"),
        ("clm-serve", cfg.clm.base_url, "CLM server"),
    ):
        try:
            out[name] = assert_loopback(url, allow_remote=cfg.clm.allow_remote, what=what)
        except EndpointError as exc:
            rep.add(f"{name} url", "fail", f"{redact_url(url)}: not probed ({exc})")
    return out


def _reachability_check(cfg: Config, rep: HealthReport, probes: ServeProbes) -> None:
    """Without serve mode: do both ``/health`` routes answer? A warning when not."""
    endpoints = _endpoints(cfg, rep)
    if not endpoints:
        return
    with _http(probes) as client:
        got = {
            f"{name} :{_port(url, 8090 if name == 'encoder' else 8700)}": _status_of(
                client, f"{url}/health"
            )
            for name, url in endpoints.items()
        }
    down = [
        f"{name} {'unreachable' if st is None else f'/health {st}'}"
        for name, st in got.items()
        if st != 200
    ]
    if down:
        rep.add(
            "serving",
            "warn",
            "; ".join(down) + " (annotate --provider clm needs both; --provider fake runs offline; "
            "live probes: `mesa-clm doctor --serve`)",
        )
    else:
        rep.add(
            "serving",
            "ok",
            " and ".join(got) + " answer /health (live probes: `mesa-clm doctor --serve`)",
        )


def _model_names(models: list[dict[str, Any]], key: str) -> list[str]:
    return [str(m.get(key)) for m in models if m.get(key)]


def _auth_matrix(
    client: httpx.Client, name: str, url: str, routes: Sequence[tuple[str, str]], rep: HealthReport
) -> None:
    """Every guarded route answers 401 without a key (DESIGN A4; plan §6.4)."""
    statuses = {f"{m} {p}": _status_of(client, f"{url}{p}", m) for m, p in routes}
    open_ = {route: st for route, st in statuses.items() if st != 401}
    if open_:
        rep.add(
            f"{name} auth",
            "fail",
            "without a key: "
            + ", ".join(f"{route} -> {st}" for route, st in open_.items())
            + " (every route but /health must require the key; DESIGN A4)",
        )
    else:
        rep.add(
            f"{name} auth",
            "ok",
            f"{len(routes)} routes answer 401 without a key"
            + (" (an unknown path too)" if name == "encoder" else ""),
        )


def _live_lock(probes: ServeProbes) -> ServingLock | None:
    """The serving lock this host runs (installed copy, else the checkout's), or ``None``."""
    from mesa_clm.clm.fingerprint import LockError, load_serving_lock
    from mesa_clm.providers.live import live_lock_path

    try:
        return load_serving_lock(live_lock_path(probes.home))
    except LockError:
        return None


def _golden_path(probes: ServeProbes, encoder_fp: str) -> Path | None:
    name = GOLDEN_FILE.format(fp=encoder_fp)
    if probes.golden_dirs is not None:
        dirs = list(probes.golden_dirs)
    else:
        dirs = [serving_home(probes.home) / GOLDEN_DIR]
        root = repo_root()
        if root is not None:
            dirs.append(root / ".local" / "serving")
    return next((d / name for d in dirs if (d / name).is_file()), None)


def _cos(a: Any, b: Any) -> float:
    import numpy as np

    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def _golden_check(
    enc: Any, lock: ServingLock, probes: ServeProbes, rep: HealthReport, bad: Status
) -> None:
    """The encoder goldens (plan §6.8; serving agent's M1-B request): each text one per request
    bitwise equal to the reference of this ``encoder_fp``, and, under a batch-invariant recipe,
    the texts as one batch within ``1 - 1e-6`` of one by one (DESIGN A3)."""
    import numpy as np

    path = _golden_path(probes, lock.encoder_fp)
    if path is None:
        rep.add(
            "encoder goldens",
            "warn",
            f"no golden reference for encoder_fp {lock.encoder_fp} "
            f"({GOLDEN_DIR}/{GOLDEN_FILE.format(fp=lock.encoder_fp)}; scripts/serving_probes.py "
            "writes one)",
        )
        return
    with np.load(path, allow_pickle=False) as npz:
        texts = [str(t) for t in npz["texts"]]
        ref = np.asarray(npz["vectors"], dtype=np.float32)
        fp = str(npz["encoder_fp"]) if "encoder_fp" in npz.files else lock.encoder_fp
    if fp != lock.encoder_fp or ref.shape[0] != len(texts):
        rep.add("encoder goldens", "fail", f"{path}: recorded for encoder_fp {fp}, not this lock's")
        return
    single = np.stack([enc.embed([t])[0][0] for t in texts])
    bitwise = int(np.sum(np.all(single == ref, axis=1)))
    min_cos = min(_cos(single[i], ref[i]) for i in range(len(texts)))
    parts = [f"{bitwise}/{len(texts)} bitwise one text per request (min cos {min_cos:.7f})"]
    status: Status = "ok" if bitwise == len(texts) else "warn"
    if min_cos < GOLDEN_COSINE_GATE:
        status = bad
    if lock.encoder.batch_invariance != "none":
        batch, _ = enc.embed(texts)
        batch_cos = min(_cos(batch[i], single[i]) for i in range(len(texts)))
        parts.append(f"one batch vs one by one min cos 1 - {1.0 - batch_cos:.1e}")
        if batch_cos < BATCH_COSINE_GATE:
            status = bad
            parts.append(f"below 1 - 1e-6 (batch_invariance {lock.encoder.batch_invariance})")
    if status == "warn":
        parts.append(f"moved within the {GOLDEN_COSINE_GATE} tolerance; D5 asks exact")
    rep.add("encoder goldens", status, f"{path.name}: " + "; ".join(parts))


def long_probe_text(lines: int = 300) -> str:
    """A deterministic text well over 4,096 tokens (the long-input probe; DESIGN A4)."""
    return "\n".join(
        f"Line {i}: the observer recorded breeding landbird number {i} at a radial distance of "
        f"{i % 97} meters from the point."
        for i in range(lines)
    )


def _long_input_check(enc: Any, rep: HealthReport, bad: Status) -> None:
    """A text longer than the window completes, truncated from the left at ``max_len - 1``
    tokens, and its vector is the vector of its last ``max_len - 1`` token ids (plan §6.8
    "5,000-token tail probe >= 0.9999"; DESIGN A4)."""
    text = long_probe_text()
    ids = enc.token_ids(text)
    cap = int(enc.truncate_prompt_tokens)
    if ids is None:
        rep.add("encoder long input", "warn", "the encoder has no /tokenize; not checked")
        return
    if len(ids) <= cap:
        rep.add("encoder long input", "warn", f"the probe text is {len(ids)} tokens, not > {cap}")
        return
    vec, charged = enc.embed([text])
    tail, _ = enc.embed_ids(ids[-cap:])
    cos = _cos(vec[0], tail)
    ok = cos >= GOLDEN_COSINE_GATE and charged == cap
    rep.add(
        "encoder long input",
        "ok" if ok else bad,
        f"a {len(ids)}-token text completed with {charged} tokens charged (cap {cap}); cos with "
        f"its last {cap} token ids {cos:.7f} (gate {GOLDEN_COSINE_GATE})",
    )


def _head(probes: ServeProbes, lock: ServingLock) -> Any | None:
    """The pinned head's numpy export (``heads/npz/<sha8>.npz``, plan §6.2) when present and
    exported from the lock's checkpoint; ``None`` otherwise."""
    from mesa_clm.clm.headproj import HeadError, HeadProjector
    from mesa_clm.serving import HEADS_DIR

    path = serving_home(probes.home) / HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    if not path.is_file():
        return None
    try:
        head = HeadProjector.from_npz(path)
    except HeadError:
        return None
    return head if head.source_sha256 == lock.head.sha256 else None


def _local_probabilities(enc: Any, head: Any, model: str) -> dict[str, float]:
    """What clm-serve computes for the golden question, done here: ``build_pairs``, the encoder,
    the head projection (``clm-raw``: raw cosines at CLM's scale 100), ``answer_from_logits``."""
    import numpy as np

    from mesa_clm import render
    from mesa_clm.clm.headproj import RAW_SCALE
    from mesa_clm.clm.http import Choice

    question = {"fit": Choice(criteria=GOLDEN_CRITERIA).to_dict()}
    stext, keys, texts = render.build_pairs(GOLDEN_STATE, question)["fit"]
    vs, _ = enc.embed([stext])
    va, _ = enc.embed(texts)
    if model == "clm-raw":
        logits = RAW_SCALE * (va.astype(np.float64) @ vs[0].astype(np.float64))
    else:
        logits = head.logits(head.project_states(vs), head.project_actions(va))[0]
    answer = render.answer_from_logits(question["fit"], keys, [float(x) for x in logits])
    return render.probabilities_of(answer)


def _systemone_parity_check(
    clm: Any, enc: Any, lock: ServingLock, probes: ServeProbes, rep: HealthReport, bad: Status
) -> None:
    """The golden ``/v1/systemone`` against the local route within 1e-4 for ``clm-latest`` (when
    the head export is present) and ``clm-raw`` (plan §5.6, §8; exact under DESIGN A3)."""
    from mesa_clm.clm.http import Choice, ChoiceAnswer

    head = _head(probes, lock)
    parts: list[str] = []
    status: Status = "ok"
    for model in ("clm-latest", "clm-raw"):
        if model == "clm-latest" and head is None:
            parts.append("clm-latest skipped (no head export of the lock's checkpoint)")
            continue
        served = clm.system_one(
            GOLDEN_STATE, {"fit": Choice(criteria=GOLDEN_CRITERIA)}, model=model
        )
        answer = served.answers.get("fit")
        if not isinstance(answer, ChoiceAnswer):
            parts.append(f"{model}: no choice answer")
            status = bad
            continue
        local = _local_probabilities(enc, head, model)
        diff = max(abs(float(answer.probabilities[k]) - local[k]) for k in local)
        parts.append(f"{model} max |Δp| {diff:.1e}")
        if diff > GOLDEN_SYSTEMONE_GATE:
            status = bad
    rep.add(
        "clm parity",
        status,
        "golden /v1/systemone vs the local route: "
        + "; ".join(parts)
        + f" (gate {GOLDEN_SYSTEMONE_GATE})",
    )


def _drift_check(clm: Any, rep: HealthReport) -> None:
    """Upstream drift (warn only, plan §6.8): the README quickstart against issue #15 with the
    unrounded differences, the tides rank against the model card."""
    from mesa_clm.clm.http import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

    r = clm.system_one(
        QUICKSTART_STATE,
        {
            "urgency": Noul(instructions="Is this urgent?"),
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"},
            ),
            "frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
        model="clm-latest",
    )
    urgency, dept, frus = (r.answers.get(k) for k in ("urgency", "department", "frustration"))
    if not (
        isinstance(urgency, NoulAnswer)
        and isinstance(dept, ChoiceAnswer)
        and isinstance(frus, ScoreAnswer)
    ):
        rep.add("clm drift", "warn", "the quickstart answers do not have the README's shapes")
        return
    observed = {
        "urgency": float(urgency.noul),
        "billing": float(dept.probabilities.get("billing", float("nan"))),
        "frustration": float(frus.score),
    }
    deltas = {k: abs(v - QUICKSTART_REFERENCE[k]) for k, v in observed.items()}
    ranked = clm.rank(TIDES_QUESTION, None, list(TIDES_ANSWERS), model="clm-latest")
    top = ranked[0] if ranked else None
    tides_ok = (
        top is not None
        and top.candidate == TIDES_ANSWERS[0]
        and abs(float(top.prob) - TIDES_REFERENCE) <= TIDES_TOLERANCE
    )
    within = all(d <= QUICKSTART_TOLERANCE for d in deltas.values()) and tides_ok
    quick = ", ".join(
        f"{k} {observed[k]:.4f} (|Δ| {deltas[k]:.7f})"
        for k in ("urgency", "billing", "frustration")
    )
    tides = (
        f"tides top {top.candidate!r} {float(top.prob):.4f}" if top is not None else "tides: none"
    )
    rep.add(
        "clm drift",
        "ok" if within else "warn",
        f"quickstart {quick} against issue #15 ± {QUICKSTART_TOLERANCE}; {tides} against the "
        f"model card {TIDES_REFERENCE} ± {TIDES_TOLERANCE} (warn only; DESIGN A3)",
    )


def _live_checks(
    cfg: Config, rep: HealthReport, probes: ServeProbes, *, strict: bool, full: bool
) -> None:
    """Serve mode: binds, the 401 matrix, health, model lists (and the encoder against the
    lock) and one golden /v1/systemone call; with ``full`` also the encoder goldens, the
    long-input probe, the systemone parity and the drift probes."""
    from mesa_clm.clm.encoder import EncoderClient
    from mesa_clm.clm.http import Choice, ChoiceAnswer, ClmError, ClmHttpClient
    from mesa_clm.providers.live import encoder_problems

    bad: Status = "fail" if strict else "warn"
    endpoints = _endpoints(cfg, rep)
    _binds_check(
        rep,
        probes,
        {"encoder": _port(cfg.encoder.url, 8090), "clm-serve": _port(cfg.clm.base_url, 8700)},
        bad,
    )
    _encoder_socket_check(rep, probes)
    _encoder_network_check(rep, probes)
    _headroom_timer_check(rep, probes)
    up: dict[str, bool] = {}
    with _http(probes) as client:
        for name, url in endpoints.items():
            st = _status_of(client, f"{url}/health")
            up[name] = st == 200
            shown = redact_url(f"{url}/health")
            if st == 200:
                rep.add(f"{name} health", "ok", f"{shown} 200 (open without a key)")
            else:
                rep.add(f"{name} health", bad, f"{shown} {'unreachable' if st is None else st}")
        for name, url in endpoints.items():
            if up.get(name):
                routes = ENCODER_GUARDED_ROUTES if name == "encoder" else CLM_GUARDED_ROUTES
                _auth_matrix(client, name, url, routes, rep)
    # After the /health probes, which start the socket-activated proxy if nothing had yet.
    _units_check(rep, probes)
    keys: dict[str, str | None] = {}
    for name, section in (("encoder", cfg.encoder), ("clm-serve", cfg.clm)):
        if not up.get(name):
            continue
        try:
            keys[name] = section.resolved_api_key()
        except SecretError as exc:
            rep.add(f"{name} models", "fail", str(exc))
            continue
        if not keys[name]:
            rep.add(f"{name} models", bad, f"no key configured (set {_KEY_HINTS[name]})")
    lock = _live_lock(probes)
    enc_ok = False
    timeout = SLOW_PROBE_TIMEOUT_S if full else PROBE_TIMEOUT_S
    enc: EncoderClient | None = None
    if keys.get("encoder"):
        enc = EncoderClient(
            endpoints["encoder"],
            keys["encoder"],
            model=cfg.encoder.model,
            max_len=cfg.encoder.max_len,
            timeout=timeout,
            transport=probes.transport,
            allow_remote=cfg.clm.allow_remote,
            retries=1,
        )
        try:
            models = enc.models()
        except (ClmError, EndpointError) as exc:
            rep.add("encoder models", "fail", _key_hint(exc, "encoder"))
        else:
            names = _model_names(models, "id")
            problems = encoder_problems(models, cfg.encoder.model, lock) if lock else []
            if cfg.encoder.model not in names:
                problems.insert(0, f"{cfg.encoder.model} missing")
            enc_ok = not problems
            rep.add(
                "encoder models",
                "ok" if enc_ok else "fail",
                f"{', '.join(names) or 'none'}"
                + ("; " + "; ".join(problems) if problems else "")
                + (f"; route {lock.encoder.route} as the lock" if lock and enc_ok else ""),
            )
    clm: ClmHttpClient | None = None
    try:
        if keys.get("clm-serve"):
            clm = ClmHttpClient(
                endpoints["clm-serve"],
                keys["clm-serve"],
                timeout=max(timeout, 30.0),
                transport=probes.transport,
                allow_remote=cfg.clm.allow_remote,
                retries=1,
            )
            try:
                names = _model_names(clm.models(), "name")
                missing = [m for m in ("clm-latest", "clm-raw") if m not in names]
                rep.add(
                    "clm-serve models",
                    "fail" if missing else "ok",
                    f"{', '.join(names) or 'none'}"
                    + (f"; missing {', '.join(missing)}" if missing else ""),
                )
                response = clm.system_one(
                    GOLDEN_STATE, {"fit": Choice(criteria=GOLDEN_CRITERIA)}, model="clm-latest"
                )
            except (ClmError, EndpointError) as exc:
                rep.add("clm golden", "fail", _key_hint(exc, "clm-serve"))
                clm.close()
                clm = None
            else:
                answer = response.answers.get("fit")
                ok = (
                    isinstance(answer, ChoiceAnswer)
                    and set(answer.probabilities) == set(GOLDEN_CRITERIA)
                    and all(p > 0 for p in answer.probabilities.values())
                    and abs(sum(answer.probabilities.values()) - 1.0) <= 1e-3
                )
                if isinstance(answer, ChoiceAnswer) and ok:
                    latency = (
                        f"{response.latency_ms:.0f} ms, " if response.latency_ms is not None else ""
                    )
                    rep.add(
                        "clm golden",
                        "ok",
                        f"/v1/systemone clm-latest: {latency}answer {answer.choice}, "
                        f"{response.usage.input_tokens} input tokens",
                    )
                else:
                    rep.add(
                        "clm golden", "fail", "/v1/systemone answer does not cover the golden keys"
                    )
        if full:
            _full_probes(enc if enc_ok else None, clm, lock, probes, rep, bad)
    finally:
        if enc is not None:
            enc.close()
        if clm is not None:
            clm.close()


def _full_probes(
    enc: Any,
    clm: Any,
    lock: ServingLock | None,
    probes: ServeProbes,
    rep: HealthReport,
    bad: Status,
) -> None:
    """The slow serve-mode probes (``doctor --serve``, or serve mode without ``--quick``); each
    failure names itself, none hides another."""
    from mesa_clm.clm.http import ClmError

    if lock is None:
        rep.add("encoder goldens", bad, "no serving lock verifies on this host")
        return
    probes_run: list[tuple[str, Callable[[], None]]] = []
    if enc is not None:
        probes_run.append(("encoder goldens", lambda: _golden_check(enc, lock, probes, rep, bad)))
        probes_run.append(("encoder long input", lambda: _long_input_check(enc, rep, bad)))
    if enc is not None and clm is not None:
        probes_run.append(
            ("clm parity", lambda: _systemone_parity_check(clm, enc, lock, probes, rep, bad))
        )
    if clm is not None:
        probes_run.append(("clm drift", lambda: _drift_check(clm, rep)))
    for name, run in probes_run:
        try:
            run()
        except (ClmError, EndpointError, ValueError, OSError) as exc:
            rep.add(name, bad, f"{type(exc).__name__}: {exc}")


_KEY_HINTS: Final[dict[str, str]] = {
    "encoder": "MESA_CLM_ENCODER__API_KEY_FILE, default ~/.mesa/clm/secrets/encoder.key",
    "clm-serve": "MESA_CLM_CLM__API_KEY_FILE, default ~/.mesa/clm/secrets/clm.key",
}


def _key_hint(exc: Exception, name: str) -> str:
    status = getattr(exc, "status", None)
    if status == 401:
        return (
            f"{name} rejected the configured key (401): it differs from the unit's; re-read "
            "`mesa-clm serve keys --init` and restart the units if the keys were rotated"
        )
    return f"{type(exc).__name__}: {exc}"


def _serving_checks(
    cfg: Config, rep: HealthReport, *, quick: bool, serve: bool | None, probes: ServeProbes
) -> None:
    from mesa_clm.serving import check_serving_lock

    home = serving_home(probes.home)
    active = units_active(probes.runner) if serve is None else False
    serve_mode = serve if serve is not None else active
    strict = serve is True or (serve is None and active and not quick)
    if not home.is_dir():
        rep.add("serving lock", "ok", f"skipped: no serving home at {home} (not the serving host)")
    elif quick and not serve_mode:
        rep.add("serving lock", "ok", "skipped (--quick; `mesa-clm serve lock --check` runs it)")
    else:
        checks = check_serving_lock(
            home=home, require_live=serve_mode, runner=probes.runner, docker=probes.docker
        )
        failed = [f"{c.name}: {c.detail}" for c in checks if c.status == "fail"]
        n_ok = sum(c.status == "ok" for c in checks)
        n_skip = sum(c.status == "skip" for c in checks)
        if failed:
            rep.add("serving lock", "fail", "; ".join(failed))
        else:
            lock = next((c.detail for c in checks if c.name == "lock"), "")
            rep.add("serving lock", "ok", f"{lock}; {n_ok} ok, {n_skip} skipped")
    _host_checks(rep, probes)
    if serve_mode:
        _live_checks(cfg, rep, probes, strict=strict, full=serve is True or not quick)
    else:
        _reachability_check(cfg, rep, probes)
