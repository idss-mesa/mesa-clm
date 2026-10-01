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
  label store, any configured key file (stat-ed for mode 0600, never opened), the eval root and
  the OLS fixture directory.

M1 adds:

* ``framings lock``: the framing keys against ``framings.lock.json`` (``framings --check``);
* ``schema sha256``: :func:`mesa_clm.render.schema_sha256` against the ``vendored.sha256`` entry
  and the serving pin (D4, D5);
* ``sidecar schema``: the ``mesa_clm`` schema version of the sidecar, read without creating
  anything (a DuckDB file is opened read-only under the shared flock; Postgres through
  :func:`mesa_clm.provenance.migrate.current_version`);
* ``serving lock``: :func:`mesa_clm.serving.check_serving_lock` when the serving home
  (``~/.mesa/clm``) exists (skipped by ``--quick`` unless ``--serve``);
* ``host`` (``MemAvailable``; warn below the 8 GiB running floor, plan §6.5) and ``gpu_budget``
  (active CARC vLLM backends, ``systemctl list-units carc-vllm@*``, read-only);
* the serving endpoints. Without serve mode one ``serving`` line says whether both ``/health``
  routes answer (``warn`` when not: annotate falls back to ``--provider fake`` or fails). **Serve
  mode** (``--serve``, or automatically when both ``mesa-clm-*`` user units are active) runs the
  live probes: loopback binds (``ss -ltn``), unauthenticated ``GET /v1/models`` answered 401 on
  both ports, ``/health`` 200 on both, the authenticated model lists (``qwen3-8b``;
  ``clm-latest`` and ``clm-raw``) and one golden ``/v1/systemone`` call. An unreachable endpoint
  fails under ``--serve`` (or auto-detected serve mode without ``--quick``) and is a warning
  otherwise.

Everything that touches the host or the network goes through :class:`ServeProbes` (serving
home, httpx transport, command runner, ``/proc/meminfo``), so the tests inject fakes and the
hermetic suite never probes a port. No key is ever read into a report line.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

import duckdb
import httpx

import mesa_clm
from mesa_clm import __version__
from mesa_clm.config import Config, config_sha256, duckdb_path, expand_path, redact_dsn
from mesa_clm.secrets import SecretError, check_secret_file_stat
from mesa_clm.serving import (
    RUN_HEADROOM_GIB,
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
    the docker argv builder and the meminfo file."""

    home: Path | None = None
    transport: httpx.BaseTransport | None = None
    runner: Callable[[Sequence[str]], CommandResult] = run_command
    docker: Callable[[Sequence[str]], list[str] | None] = docker_argv
    meminfo: Path = Path("/proc/meminfo")

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
    _secret_checks(cfg, rep)
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
    """Key files are stat-ed for the 0600 rule of :mod:`mesa_clm.secrets`, never opened."""
    for label, section, attr in _KEY_FILES:
        file: str | None = getattr(getattr(cfg, section), attr)
        if not file:
            continue
        path = expand_path(file)
        name = f"{label} key file"
        try:
            check_secret_file_stat(path, path.stat())
        except FileNotFoundError:
            rep.add(name, "fail", f"{path}: not found")
        except (SecretError, OSError) as exc:
            rep.add(name, "fail", str(exc))
        else:
            rep.add(name, "ok", f"{path}: regular file, mode 0600 or stricter (not read)")
    if cfg.secrets == "keyring":
        present = importlib.util.find_spec("keyring") is not None
        rep.add(
            "keyring",
            "ok" if present else "fail",
            "keyring package importable"
            if present
            else "MESA_CLM_SECRETS=keyring but the keyring package is not installed",
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


# -- M1: serving (plan §6.4, §6.5, §6.8) ------------------------------------------------------------

PROBE_TIMEOUT_S: Final = 5.0
_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "[::1]", "localhost"})
# One fixed /v1/systemone question the doctor asks (plan §6.8 "golden /v1/systemone"): a NEON
# column as a target state and two candidate terms plus the term anchor. What is checked is the
# answer's shape (every key answered, p > 0, sum 1), not its numbers.
GOLDEN_STATE: Final[dict[str, Any]] = {
    "card": {
        "dataset": "DP1.10003.001.brd_countdata",
        "product_title": "Breeding landbird point counts",
    },
    "scope": "column",
    "aspect": "measurement",
    "column": {
        "name": "observerDistance",
        "description": "Radial distance between the observer and the individual(s) being observed",
        "dtype": "real",
        "unit": "meter",
    },
}
GOLDEN_CRITERIA: Final[dict[str, str]] = {
    "PATO:0000040": "distance: A 1-D extent quality which is equal to the distance between two "
    "points.",
    "UO:0000008": "meter: A length unit which is equal to the length of the path travelled by "
    "light in vacuum during a time interval of 1/299 792 458 of a second.",
    "__none__": "None of these terms is the right concept for this target.",
}


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


def parse_listeners(ss_output: str, port: int) -> list[str]:
    """The local addresses listening on ``port`` in ``ss -ltn`` output (``127.0.0.1``,
    ``[::1]``, ``0.0.0.0``, ``*`` ...)."""
    out: list[str] = []
    for line in ss_output.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "LISTEN":
            continue
        host, _, p = parts[3].rpartition(":")
        if p == str(port):
            out.append(host.split("%", 1)[0])
    return out


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _binds_check(
    rep: HealthReport, probes: ServeProbes, ports: dict[str, int], bad: Status
) -> None:
    res = probes.runner(["ss", "-ltn"])
    if res.returncode != 0:
        rep.add("serving binds", "warn", "`ss -ltn` not available; binds not checked")
        return
    problems, seen = [], []
    for name, port in ports.items():
        hosts = parse_listeners(res.stdout, port)
        exposed = [h for h in hosts if not _is_loopback(h)]
        if not hosts:
            problems.append(f"nothing listens on :{port} ({name})")
        elif exposed:
            problems.append(f"{name} :{port} listens on {', '.join(exposed)} (not loopback)")
        else:
            seen.append(f"{name} {', '.join(sorted(set(hosts)))}:{port}")
    if any("not loopback" in p for p in problems):
        rep.add("serving binds", "fail", "; ".join(problems))
    elif problems:
        rep.add("serving binds", bad, "; ".join(problems))
    else:
        rep.add("serving binds", "ok", "loopback only: " + "; ".join(seen))


def _http(probes: ServeProbes) -> httpx.Client:
    return httpx.Client(timeout=PROBE_TIMEOUT_S, trust_env=False, transport=probes.transport)


def _status_of(client: httpx.Client, url: str) -> int | None:
    """The HTTP status of an unauthenticated GET, ``None`` when nothing answered."""
    try:
        return client.get(url).status_code
    except httpx.HTTPError:
        return None


def _reachability_check(cfg: Config, rep: HealthReport, probes: ServeProbes) -> None:
    """Without serve mode: do both ``/health`` routes answer? A warning when not."""
    with _http(probes) as client:
        got = {
            f"encoder :{_port(cfg.encoder.url, 8090)}": _status_of(
                client, f"{cfg.encoder.url}/health"
            ),
            f"clm-serve :{_port(cfg.clm.base_url, 8700)}": _status_of(
                client, f"{cfg.clm.base_url}/health"
            ),
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


def _live_checks(cfg: Config, rep: HealthReport, probes: ServeProbes, *, strict: bool) -> None:
    """Serve mode: binds, 401s, health, model lists and one golden /v1/systemone call."""
    from mesa_clm.clm.encoder import EncoderClient
    from mesa_clm.clm.http import Choice, ChoiceAnswer, ClmError, ClmHttpClient
    from mesa_clm.net import EndpointError

    bad: Status = "fail" if strict else "warn"
    endpoints = {"encoder": cfg.encoder.url, "clm-serve": cfg.clm.base_url}
    _binds_check(
        rep,
        probes,
        {"encoder": _port(cfg.encoder.url, 8090), "clm-serve": _port(cfg.clm.base_url, 8700)},
        bad,
    )
    up: dict[str, bool] = {}
    with _http(probes) as client:
        for name, url in endpoints.items():
            st = _status_of(client, f"{url}/health")
            up[name] = st == 200
            if st == 200:
                rep.add(f"{name} health", "ok", f"{url}/health 200")
            else:
                rep.add(
                    f"{name} health", bad, f"{url}/health {'unreachable' if st is None else st}"
                )
        for name, url in endpoints.items():
            if not up[name]:
                continue
            st = _status_of(client, f"{url}/v1/models")
            if st == 401:
                rep.add(f"{name} auth", "ok", "GET /v1/models without a key -> 401")
            else:
                rep.add(
                    f"{name} auth",
                    "fail",
                    f"GET /v1/models without a key -> {st}: the route must require the key "
                    "(plan §6.4)",
                )
    hints = {
        "encoder": "MESA_CLM_ENCODER__API_KEY_FILE=~/.mesa/clm/secrets/encoder.key",
        "clm-serve": "MESA_CLM_CLM__API_KEY_FILE=~/.mesa/clm/secrets/clm.key",
    }
    keys: dict[str, str | None] = {}
    for name, section in (("encoder", cfg.encoder), ("clm-serve", cfg.clm)):
        if not up[name]:
            continue
        try:
            keys[name] = section.resolved_api_key()
        except SecretError as exc:
            rep.add(f"{name} models", "fail", str(exc))
            continue
        if not keys[name]:
            rep.add(f"{name} models", bad, f"no key configured (set {hints[name]})")
    if keys.get("encoder"):
        try:
            with EncoderClient(
                cfg.encoder.url,
                keys["encoder"],
                model=cfg.encoder.model,
                timeout=PROBE_TIMEOUT_S,
                transport=probes.transport,
                allow_remote=cfg.clm.allow_remote,
                retries=1,
            ) as enc:
                names = _model_names(enc.models(), "id")
        except (ClmError, EndpointError) as exc:
            rep.add("encoder models", "fail", _key_hint(exc, "encoder"))
        else:
            want = cfg.encoder.model
            rep.add(
                "encoder models",
                "ok" if want in names else "fail",
                f"{', '.join(names) or 'none'}" + ("" if want in names else f"; {want} missing"),
            )
    if keys.get("clm-serve"):
        try:
            with ClmHttpClient(
                cfg.clm.base_url,
                keys["clm-serve"],
                timeout=max(PROBE_TIMEOUT_S, 30.0),
                transport=probes.transport,
                allow_remote=cfg.clm.allow_remote,
                retries=1,
            ) as clm:
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
                rep.add("clm golden", "fail", "/v1/systemone answer does not cover the golden keys")


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
        _live_checks(cfg, rep, probes, strict=strict)
    else:
        _reachability_check(cfg, rep, probes)
