"""``mesa-clm doctor``: what this host can run, honestly (plan §6.8; the M0 subset).

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

``quick=True`` skips the file hashing; from M1 it also skips the serving probes (encoder :8090,
clm-serve :8700, goldens and the serving lock) that ``mesa_clm_health`` does not run. Nothing
here touches the network or reads a secret.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import duckdb

import mesa_clm
from mesa_clm import __version__
from mesa_clm.config import Config, config_sha256, duckdb_path, expand_path, redact_dsn
from mesa_clm.secrets import SecretError, check_secret_file_stat

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


def doctor(cfg: Config, *, quick: bool = False) -> HealthReport:
    """Run every M0 check against ``cfg``; ``quick`` skips the vendored-file hashing."""
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
    _policy_check(cfg, rep)
    _store_checks(cfg, rep)
    _secret_checks(cfg, rep)
    _path_checks(cfg, rep)
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
            f"{redact_dsn(dsn)}: not a duckdb:/// DSN; not probed in M0 (the Postgres dialect "
            "lands in M1)",
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
