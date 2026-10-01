"""``mesa_clm.health.doctor``: the M0 checks are green on this checkout, every failure path is
forced through a fake or a temporary file, and no secret ever reaches a report line."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import duckdb
import pytest
import yaml

import mesa_clm
from mesa_clm import __version__, health
from mesa_clm.config import Config, load_config
from mesa_clm.health import (
    PINNED_COMMITS,
    Check,
    HealthReport,
    dist_info,
    doctor,
    parse_manifest,
    repo_root,
    verify_vendored,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"

M0_CHECKS = [
    "config",
    "python",
    "duckdb",
    "mesa-mcp",
    "mesa-ducklake",
    "mesa-mcp plugin api",
    "plugin entry point",
    "vendored files",
    "framings lock",
    "schema sha256",
    "policy defaults",
    "provenance path",
    "labels store",
    "sidecar schema",
    "feature store",
    "permissions",
    "serving lock",
    "host",
    "gpu_budget",
    "serving",
]


def _cfg(**env: str) -> Config:
    """A config from an explicit environment only, so the developer's shell never leaks in."""
    return load_config(env=env)


def _by_name(rep: HealthReport) -> dict[str, Check]:
    return {c.name: c for c in rep.checks}


def _dsn(tmp_path: Path) -> str:
    return f"duckdb:///{tmp_path / 'provenance.duckdb'}"


def test_report_lines_counts_and_dict() -> None:
    rep = HealthReport()
    assert rep.ok and rep.counts() == {"ok": 0, "warn": 0, "fail": 0}
    rep.add("a", "ok", "fine")
    rep.add("b", "warn")
    rep.add("c", "fail", "bad")
    assert not rep.ok and rep.counts() == {"ok": 1, "warn": 1, "fail": 1}
    assert rep.lines() == ["[ok] a: fine", "[WARN] b", "[FAIL] c: bad"]
    d = rep.as_dict()
    assert d["ok"] is False and d["mesa_clm"] == __version__
    assert d["summary"] == {"ok": 1, "warn": 1, "fail": 1}
    assert d["checks"][2] == {"name": "c", "status": "fail", "detail": "bad"}
    assert json.loads(json.dumps(d)) == d
    # A warning still passes; only a failure fails the report.
    assert Check("x", "warn").ok and not Check("x", "fail").ok
    rep.checks.pop()
    assert rep.ok


def test_doctor_is_green_on_this_checkout(tmp_path: Path) -> None:
    rep = doctor(_cfg(MESA_CLM_PROVENANCE__DSN=_dsn(tmp_path)))
    assert rep.ok, rep.lines()
    assert [c.name for c in rep.checks] == M0_CHECKS
    by = _by_name(rep)
    assert by["config"].detail.startswith("sha256 ") and "planner=static" in by["config"].detail
    assert by["vendored files"].detail == "4 files match vendored.sha256"
    assert f"@ {PINNED_COMMITS['mesa-mcp'][:7]} (pinned)" in by["mesa-mcp"].detail
    assert f"@ {PINNED_COMMITS['mesa-ducklake'][:7]} (pinned)" in by["mesa-ducklake"].detail
    assert "register_tool(meta=)" in by["mesa-mcp plugin api"].detail
    assert "clm = mesa_clm.mcp_tools" in by["plugin entry point"].detail
    assert "6 tasks" in by["policy defaults"].detail
    assert "proposed-only" in by["policy defaults"].detail
    assert by["provenance path"].status == "ok" and str(tmp_path) in by["provenance path"].detail
    assert by["labels store"].detail.startswith("none yet")
    # M1: the pins and the (offline) serving view; the conftest makes every probe offline.
    assert by["framings lock"].status == "ok" and "in sync" in by["framings lock"].detail
    assert by["schema sha256"].detail.startswith("52cec58afbf4 = vendored.sha256 entry")
    assert by["sidecar schema"].detail.startswith("none yet")
    assert by["feature store"].status == "ok" and by["feature store"].detail.startswith("none yet")
    assert by["serving lock"].detail.startswith("skipped: no serving home")
    assert by["host"].detail.startswith("skipped") and by["gpu_budget"].status == "ok"
    assert by["serving"].status == "warn" and "unreachable" in by["serving"].detail


def test_quick_skips_the_hashing(tmp_path: Path) -> None:
    rep = doctor(_cfg(MESA_CLM_PROVENANCE__DSN=_dsn(tmp_path)), quick=True)
    assert rep.ok
    assert [c.name for c in rep.checks] == [n for n in M0_CHECKS if n != "vendored files"]


# -- vendored files --------------------------------------------------------------------------------


def test_repo_root_and_the_committed_manifest() -> None:
    root = repo_root()
    assert root == REPO and (root / "vendored.sha256").is_file()
    entries = parse_manifest((root / "vendored.sha256").read_text(encoding="utf-8"))
    assert [name for _, name in entries] == [
        "src/mesa_clm/_vendor/clm/schema.py",
        "src/mesa_clm/_vendor/clm/LICENSE",
        "src/mesa_clm/bench/_metrics.py",
        "tests/_vendor/clm_client.py",
    ]
    assert all(len(digest) == 64 for digest, _ in entries)
    matching, problems = verify_vendored(root)
    assert problems == [] and len(matching) == 4


def test_parse_manifest_skips_comments_and_binary_markers() -> None:
    text = "# vendored\n\nABCDEF  *x.py\n0123  dir/y.txt  \n"
    assert parse_manifest(text) == [("abcdef", "x.py"), ("0123", "dir/y.txt")]


def test_vendored_mismatch_and_missing_are_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "src" / "mesa_clm").mkdir(parents=True)
    (root / "good.py").write_text("x", encoding="utf-8")
    (root / "bad.py").write_text("y", encoding="utf-8")
    manifest = (
        f"{hashlib.sha256(b'x').hexdigest()}  good.py\n{'0' * 64}  bad.py\n{'1' * 64}  gone.py\n"
    )
    (root / "vendored.sha256").write_text(manifest, encoding="utf-8")
    matching, problems = verify_vendored(root)
    assert matching == ["good.py"]
    assert problems[0].startswith("bad.py: sha256 ") and "!= 000000000000" in problems[0]
    assert problems[1] == "gone.py: missing"

    monkeypatch.setattr(health, "repo_root", lambda: root)
    rep = doctor(_cfg(MESA_CLM_PROVENANCE__DSN=_dsn(tmp_path)))
    vendored = _by_name(rep)["vendored files"]
    assert not rep.ok and vendored.status == "fail" and "gone.py: missing" in vendored.detail
    # A wheel install has no manifest beside the package: skipped and said so, never failed.
    monkeypatch.setattr(health, "repo_root", lambda: None)
    rep = doctor(_cfg(MESA_CLM_PROVENANCE__DSN=_dsn(tmp_path)))
    vendored = _by_name(rep)["vendored files"]
    assert rep.ok and vendored.status == "ok" and "skipped" in vendored.detail


def test_repo_root_is_none_outside_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pkg = tmp_path / "site-packages" / "mesa_clm"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(mesa_clm, "__file__", str(fake_pkg / "__init__.py"))
    assert repo_root() is None


# -- runtime and dependencies ----------------------------------------------------------------------


def test_python_check(monkeypatch: pytest.MonkeyPatch) -> None:
    rep = HealthReport()
    health._python_check(rep)
    assert rep.checks[0].status == "ok" and rep.checks[0].detail.startswith("3.")
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    rep = HealthReport()
    health._python_check(rep)
    assert rep.checks[0].status == "fail" and "needs >= 3.11" in rep.checks[0].detail


def test_duckdb_window(monkeypatch: pytest.MonkeyPatch) -> None:
    for version, status in (
        ("1.5.5", "ok"),
        ("1.5.6", "ok"),
        ("1.5.99", "ok"),
        ("1.5.4", "warn"),
        ("1.4.1", "warn"),
        ("1.6.0", "warn"),
        ("2.0.0", "warn"),
        ("weird", "warn"),
    ):
        monkeypatch.setattr(duckdb, "__version__", version)
        rep = HealthReport()
        health._duckdb_check(rep)
        assert rep.checks[0].status == status, version
        assert rep.checks[0].detail.startswith(version)


def test_dist_info_reads_the_git_commit() -> None:
    assert dist_info("definitely-not-installed-xyz") is None
    info = dist_info("mesa-mcp")
    assert info is not None and info[1] == PINNED_COMMITS["mesa-mcp"]
    wheel = dist_info("duckdb")  # a PyPI wheel records no VCS commit
    assert wheel == (duckdb.__version__, None)


def test_dependency_check_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    rep = HealthReport()
    monkeypatch.setattr(health, "dist_info", lambda name: None)
    health._dependency_check("mesa-mcp", rep)
    monkeypatch.setattr(health, "dist_info", lambda name: ("0.2.0", "deadbeef" * 5))
    health._dependency_check("mesa-mcp", rep)
    monkeypatch.setattr(health, "dist_info", lambda name: ("0.2.0", None))
    health._dependency_check("mesa-ducklake", rep)
    monkeypatch.setattr(health, "dist_info", lambda name: ("1.0", None))
    health._dependency_check("something-unpinned", rep)
    assert [c.status for c in rep.checks] == ["fail", "warn", "ok", "ok"]
    assert rep.checks[0].detail == "not installed"
    assert "pyproject.toml pins c74f3aa" in rep.checks[1].detail
    assert "pin 7bc143f unverified" in rep.checks[2].detail
    assert rep.checks[3].detail == "1.0 (no git commit recorded)"


def test_plugin_api_check_detects_an_old_mesa_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    rep = HealthReport()
    health._plugin_api_check(rep)
    assert rep.checks[0].status == "ok" and "mesa_mcp.tools" in rep.checks[0].detail

    # 8fbaedf-era server: register_tool without meta=, no load_plugins at all.
    old = types.SimpleNamespace(register_tool=lambda name, description: None)
    monkeypatch.setitem(sys.modules, "mesa_mcp.server", old)
    rep = HealthReport()
    health._plugin_api_check(rep)
    assert rep.checks[0].status == "fail"
    assert "load_plugins" in rep.checks[0].detail
    assert "register_tool(meta=)" in rep.checks[0].detail
    # No register_tool at all.
    monkeypatch.setitem(
        sys.modules, "mesa_mcp.server", types.SimpleNamespace(load_plugins=lambda: {})
    )
    rep = HealthReport()
    health._plugin_api_check(rep)
    assert rep.checks[0].status == "fail" and "missing register_tool:" in rep.checks[0].detail
    # An import failure is a failure with the exception type in the detail.
    monkeypatch.setitem(sys.modules, "mesa_mcp.server", None)
    rep = HealthReport()
    health._plugin_api_check(rep)
    assert rep.checks[0].status == "fail" and "cannot import" in rep.checks[0].detail


def test_entry_point_check(monkeypatch: pytest.MonkeyPatch) -> None:
    rep = HealthReport()
    health._entry_point_check(rep)
    assert rep.checks[0].status == "ok" and "mesa_clm.mcp_tools" in rep.checks[0].detail
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: [])
    rep = HealthReport()
    health._entry_point_check(rep)
    assert rep.checks[0].status == "warn"
    assert "not installed as a distribution" in rep.checks[0].detail


# -- stores, policy, secrets, paths -----------------------------------------------------------------


def _store_checks(dsn: str) -> dict[str, Check]:
    rep = HealthReport()
    health._store_checks(_cfg(MESA_CLM_PROVENANCE__DSN=dsn), rep)
    return _by_name(rep)


def test_provenance_path_checks(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "p.duckdb"
    by = _store_checks(f"duckdb:///{nested}")
    assert by["provenance path"].status == "ok"
    assert "will be created" in by["provenance path"].detail
    assert by["labels store"].status == "ok" and by["labels store"].detail.startswith("none yet")

    existing = tmp_path / "p.duckdb"
    existing.write_bytes(b"x" * 2048)
    by = _store_checks(f"duckdb:///{existing}")
    assert by["provenance path"] == Check("provenance path", "ok", str(existing))
    assert by["labels store"] == Check("labels store", "ok", f"{existing} (2 KiB)")

    adir = tmp_path / "dir.duckdb"
    adir.mkdir()
    by = _store_checks(f"duckdb:///{adir}")
    assert by["provenance path"].status == "fail"
    assert "not a writable file" in by["provenance path"].detail

    by = _store_checks("postgresql://user:s3cret@db.example/clm")
    assert by["provenance path"].status == "warn"
    assert "no local path to probe" in by["provenance path"].detail
    assert "s3cret" not in by["provenance path"].detail and "labels store" not in by

    if os.getuid() != 0:  # root ignores directory modes
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        try:
            by = _store_checks(f"duckdb:///{ro / 'sub' / 'p.duckdb'}")
            assert by["provenance path"].status == "fail"
            assert "not a writable directory" in by["provenance path"].detail
        finally:
            ro.chmod(0o700)


def test_policy_checks(tmp_path: Path) -> None:
    rep = HealthReport()
    health._policy_check(_cfg(), rep)
    assert rep.checks[0].status == "ok"
    assert "policy_defaults.yaml: 6 tasks" in rep.checks[0].detail

    empty = tmp_path / "empty.yaml"
    empty.write_text("tasks: {}\nprofiles: {}\n", encoding="utf-8")
    rep = HealthReport()
    health._policy_check(_cfg(MESA_CLM_POLICY__POLICY_PATH=str(empty)), rep)
    assert rep.checks[0].status == "fail" and "ValueError" in rep.checks[0].detail

    missing = tmp_path / "nope.yaml"
    rep = HealthReport()
    health._policy_check(_cfg(MESA_CLM_POLICY__POLICY_PATH=str(missing)), rep)
    assert rep.checks[0].status == "fail" and "FileNotFoundError" in rep.checks[0].detail

    # The shipped file minus the configured profile: loads, but the profile is not defined.
    data = yaml.safe_load((REPO / "policy_defaults.yaml").read_text(encoding="utf-8"))
    data["profiles"].pop("dev")
    prod_only = tmp_path / "prod_only.yaml"
    prod_only.write_text(yaml.safe_dump(data), encoding="utf-8")
    rep = HealthReport()
    health._policy_check(
        _cfg(MESA_CLM_POLICY__POLICY_PATH=str(prod_only), MESA_CLM_POLICY__PROFILE="dev"), rep
    )
    assert rep.checks[0].status == "fail"
    assert "profile dev is not defined" in rep.checks[0].detail


def test_secret_file_checks_never_read_the_key(tmp_path: Path) -> None:
    key = tmp_path / "clm.key"
    key.write_text("super-secret-value\n", encoding="utf-8")
    key.chmod(0o644)
    cfg = _cfg(
        MESA_CLM_CLM__API_KEY_FILE=str(key),
        MESA_CLM_ENCODER__API_KEY_FILE=str(tmp_path / "missing.key"),
        MESA_CLM_PLANNER__GATEWAY_API_KEY_FILE=str(tmp_path),  # a directory, not a file
    )
    rep = HealthReport()
    health._secret_checks(cfg, rep)
    by = _by_name(rep)
    assert by["clm key file"].status == "fail" and "0644" in by["clm key file"].detail
    assert by["encoder key file"].status == "fail"
    assert "not found" in by["encoder key file"].detail
    assert by["planner gateway key file"].status == "fail"
    assert "not a regular file" in by["planner gateway key file"].detail
    assert "super-secret-value" not in json.dumps(rep.as_dict())

    key.chmod(0o600)
    rep = HealthReport()
    health._secret_checks(cfg, rep)
    by = _by_name(rep)
    assert by["clm key file"].status == "ok" and "not read" in by["clm key file"].detail
    assert "super-secret-value" not in json.dumps(rep.as_dict())

    rep = HealthReport()
    health._secret_checks(_cfg(), rep)
    assert rep.checks == []

    rep = HealthReport()
    health._secret_checks(_cfg(MESA_CLM_SECRETS="keyring"), rep)
    present = importlib.util.find_spec("keyring") is not None
    assert rep.checks == [
        Check(
            "keyring",
            "ok" if present else "fail",
            "keyring package importable"
            if present
            else "MESA_CLM_SECRETS=keyring but the keyring package is not installed",
        )
    ]


def test_path_checks(tmp_path: Path) -> None:
    rep = HealthReport()
    health._path_checks(
        _cfg(
            MESA_CLM_EVAL_ROOT=str(FIXTURES / "neon-avu-eval"),
            MESA_CLM_OLS__FIXTURES="replay",
            MESA_CLM_OLS__FIXTURES_DIR=str(FIXTURES / "ols"),
        ),
        rep,
    )
    by = _by_name(rep)
    assert by["eval root"] == Check("eval root", "ok", str(FIXTURES / "neon-avu-eval"))
    assert by["ols fixtures"].status == "ok"
    assert by["ols fixtures"].detail.startswith("replay: ")
    assert int(by["ols fixtures"].detail.split()[1]) >= 777

    rep = HealthReport()
    health._path_checks(
        _cfg(
            MESA_CLM_EVAL_ROOT=str(tmp_path / "nope"),
            MESA_CLM_OLS__FIXTURES="replay",
            MESA_CLM_OLS__FIXTURES_DIR=str(tmp_path / "nofix"),
        ),
        rep,
    )
    by = _by_name(rep)
    assert by["eval root"].status == "fail" and "validated.json" in by["eval root"].detail
    assert by["ols fixtures"].status == "fail" and "does not exist" in by["ols fixtures"].detail

    rep = HealthReport()
    health._path_checks(
        _cfg(MESA_CLM_OLS__FIXTURES="record", MESA_CLM_OLS__FIXTURES_DIR=str(tmp_path / "new")),
        rep,
    )
    assert rep.checks == [
        Check("ols fixtures", "ok", f"record: {tmp_path / 'new'} will be created")
    ]

    rep = HealthReport()
    health._path_checks(_cfg(), rep)  # fixtures off, no eval root: nothing to say
    assert rep.checks == []


def test_doctor_fails_on_a_missing_eval_root(tmp_path: Path) -> None:
    rep = doctor(
        _cfg(MESA_CLM_PROVENANCE__DSN=_dsn(tmp_path), MESA_CLM_EVAL_ROOT=str(tmp_path / "gone"))
    )
    assert not rep.ok and _by_name(rep)["eval root"].status == "fail"
    assert rep.counts()["fail"] == 1
