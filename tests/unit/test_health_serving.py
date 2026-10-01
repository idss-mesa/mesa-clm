"""The M1 doctor checks (plan §6.8): framings lock, schema contract, sidecar schema, serving
lock, host and GPU budget, and the serving probes, each driven through injected
:class:`~mesa_clm.health.ServeProbes` (the fake CLM transport, scripted ``ss``/``systemctl``
output, a meminfo file). Nothing here opens a port or runs a command; no key reaches a report.
The live path is ``tests/engine/test_doctor_live.py`` (marker ``engine``)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pytest

from mesa_clm import framings, health, render
from mesa_clm.config import Config, load_config
from mesa_clm.health import (
    Check,
    HealthReport,
    ServeProbes,
    doctor,
    meminfo_gib,
    parse_listeners,
    units_active,
)
from mesa_clm.provenance.store import SCHEMA, DuckDBStore
from mesa_clm.serving import CommandResult, LockCheck
from tests.fakes.clm_transport import FakeClmServer

CLM_KEY = "clm-key-0123456789abcdef"
ENC_KEY = "enc-key-0123456789abcdef"
SS_LOOPBACK = (
    "LISTEN 0 2048 127.0.0.1:8700 0.0.0.0:*\n"
    "LISTEN 0 4096 127.0.0.1:8090 0.0.0.0:*\n"
    "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n"
)
CARC = "carc-vllm@qwen.service loaded active running CARC vLLM backend\n"


def _cfg(tmp_path: Path, **env: str) -> Config:
    base = {"MESA_CLM_PROVENANCE__DSN": f"duckdb:///{tmp_path / 'prov.duckdb'}"}
    return load_config(env={**base, **env})


def _keyed(tmp_path: Path, **env: str) -> Config:
    keys = {"MESA_CLM_CLM__API_KEY": CLM_KEY, "MESA_CLM_ENCODER__API_KEY": ENC_KEY}
    return _cfg(tmp_path, **{**keys, **env})


def _by(rep: HealthReport) -> dict[str, Check]:
    return {c.name: c for c in rep.checks}


class Runner:
    """Scripted command output by program name; records every argv."""

    def __init__(self, **out: tuple[int, str]) -> None:
        self.out = out
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        self.calls.append(list(argv))
        if argv[0] == "systemctl" and "--user" in argv:
            return CommandResult(*self.out.get("units", (3, "inactive\ninactive\n")))
        if argv[0] == "systemctl":
            return CommandResult(*self.out.get("carc", (0, "")))
        return CommandResult(*self.out.get(argv[0], (127, "")))


def _meminfo(tmp_path: Path, gib: float) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: 127000000 kB\nMemAvailable: {int(gib * 1024 * 1024)} kB\n")
    return path


def _probes(tmp_path: Path, server: FakeClmServer | None = None, **runner: Any) -> ServeProbes:
    offline = ServeProbes.offline(home=tmp_path / "no-serving-home")
    return ServeProbes(
        home=offline.home,
        transport=server.transport() if server is not None else offline.transport,
        runner=Runner(**runner),
        docker=lambda args: None,
        meminfo=_meminfo(tmp_path, 64),
    )


# -- framings lock and the schema contract ------------------------------------------------------


def test_framings_lock_drift_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rep = HealthReport()
    health._framings_check(rep)
    assert rep.checks[0].status == "ok" and framings.lock_sha()[:12] in rep.checks[0].detail
    lock = tmp_path / "framings.lock.json"
    data = json.loads(framings.LOCK_PATH.read_text(encoding="utf-8"))
    data["schema_sha256"] = "0" * 64
    lock.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(framings, "LOCK_PATH", lock)
    monkeypatch.setattr(framings, "lock_drift", lambda path=lock: [f"p{i}" for i in range(7)])
    rep = HealthReport()
    health._framings_check(rep)
    c = rep.checks[0]
    assert c.status == "fail" and "(+2 more)" in c.detail and "--update-lock" in c.detail


def test_framings_lock_is_skipped_in_a_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(framings, "LOCK_PATH", tmp_path / "framings.lock.json")
    monkeypatch.setattr(health, "repo_root", lambda: None)
    rep = HealthReport()
    health._framings_check(rep)
    assert rep.checks[0].status == "ok" and rep.checks[0].detail.startswith("skipped")


def test_schema_sha_against_manifest_and_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    rep = HealthReport()
    health._schema_sha_check(rep)
    assert rep.checks[0] == Check(
        "schema sha256", "ok", "52cec58afbf4 = vendored.sha256 entry = serving pin"
    )
    monkeypatch.setattr(render, "schema_sha256", lambda: "f" * 64)
    rep = HealthReport()
    health._schema_sha_check(rep)
    c = rep.checks[0]
    assert c.status == "fail" and "vendored.sha256 pins 52cec58afbf4" in c.detail
    assert "serving pin is 52cec58afbf4" in c.detail
    monkeypatch.setattr(health, "repo_root", lambda: None)  # a wheel: the pin only
    rep = HealthReport()
    health._schema_sha_check(rep)
    assert rep.checks[0].status == "fail" and "vendored" not in rep.checks[0].detail


# -- sidecar schema ------------------------------------------------------------------------------


def _sidecar(cfg: Config) -> Check:
    rep = HealthReport()
    health._sidecar_check(cfg, rep)
    return rep.checks[0]


def test_sidecar_schema_versions(tmp_path: Path) -> None:
    path = tmp_path / "prov.duckdb"
    cfg = _cfg(tmp_path)
    assert _sidecar(cfg).detail.startswith("none yet") and not path.exists()
    DuckDBStore(path).ensure_schema()
    c = _sidecar(cfg)
    assert c == Check("sidecar schema", "ok", f"{path}: mesa_clm schema v1")
    con = duckdb.connect(str(path))
    con.execute(f"INSERT INTO {SCHEMA}.schema_versions (version) VALUES (2)")  # noqa: S608
    con.close()
    c = _sidecar(cfg)
    assert c.status == "fail" and "newer than this mesa-clm" in c.detail
    con = duckdb.connect(str(path))
    con.execute(f"DELETE FROM {SCHEMA}.schema_versions")  # noqa: S608
    con.execute(f"INSERT INTO {SCHEMA}.schema_versions (version) VALUES (0)")  # noqa: S608
    con.execute(f"DROP TABLE {SCHEMA}.audits")
    con.close()
    c = _sidecar(cfg)
    assert c.status == "warn" and "missing audits" in c.detail and "provenance migrate" in c.detail


def test_sidecar_schema_of_an_m0_labels_file_and_a_broken_file(tmp_path: Path) -> None:
    from mesa_clm.provenance.labels import LabelStore

    path = tmp_path / "prov.duckdb"
    LabelStore(path).ensure_schema()
    c = _sidecar(_cfg(tmp_path))
    assert c.status == "ok" and "no schema_versions (labels)" in c.detail
    path.write_bytes(b"not a duckdb file" * 100)
    c = _sidecar(_cfg(tmp_path))
    assert c.status == "fail" and "cannot open read-only" in c.detail


def test_sidecar_schema_on_postgres_and_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mesa_clm.provenance import migrate

    dsn = "postgresql://clm:s3cret@db.example/clm"
    cfg = _cfg(tmp_path, MESA_CLM_PROVENANCE__DSN=dsn)
    monkeypatch.setattr(migrate, "current_version", lambda d: 1)
    c = _sidecar(cfg)
    assert c.status == "ok" and "schema v1" in c.detail and "s3cret" not in c.detail

    def refused(d: str) -> int:
        raise ConnectionRefusedError("s3cret")

    monkeypatch.setattr(migrate, "current_version", refused)
    c = _sidecar(cfg)
    assert c.status == "fail" and "ConnectionRefusedError" in c.detail and "s3cret" not in c.detail

    def no_driver(d: str) -> int:
        raise ImportError("psycopg")

    monkeypatch.setattr(migrate, "current_version", no_driver)
    assert _sidecar(cfg).status == "warn"
    c = _sidecar(_cfg(tmp_path, MESA_CLM_PROVENANCE__DSN="sqlite:///x.db"))
    assert c.status == "fail" and "unsupported" in c.detail


# -- serving lock, host, gpu budget --------------------------------------------------------------


def test_serving_lock_runs_on_a_serving_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mesa_clm import serving

    home = tmp_path / "home"
    home.mkdir()
    seen: dict[str, Any] = {}

    def fake_check(**kw: Any) -> list[LockCheck]:
        seen.update(kw)
        return [
            LockCheck("lock", "ok", "serving.lock.json: lock_sha abcdef012345"),
            LockCheck("head", "skip", "absent"),
            LockCheck("clone", "ok", "fine"),
        ]

    monkeypatch.setattr(serving, "check_serving_lock", fake_check)
    probes = _probes(tmp_path)
    probes.home = home
    rep = doctor(_cfg(tmp_path), quick=True, probes=probes)
    assert _by(rep)["serving lock"].detail.startswith("skipped (--quick")
    rep = doctor(_cfg(tmp_path), probes=probes)
    assert _by(rep)["serving lock"] == Check(
        "serving lock", "ok", "serving.lock.json: lock_sha abcdef012345; 2 ok, 1 skipped"
    )
    assert seen["home"] == home and seen["require_live"] is False

    def failing(**kw: Any) -> list[LockCheck]:
        return [LockCheck("lock", "ok", "x"), LockCheck("image", "fail", "digest differs")]

    monkeypatch.setattr(serving, "check_serving_lock", failing)
    rep = doctor(_cfg(tmp_path), probes=probes)
    assert _by(rep)["serving lock"] == Check("serving lock", "fail", "image: digest differs")
    assert not rep.ok


def test_meminfo_and_host_floor(tmp_path: Path) -> None:
    assert meminfo_gib(_meminfo(tmp_path, 12.5)) == pytest.approx(12.5)
    assert meminfo_gib(tmp_path / "missing") is None
    (tmp_path / "odd").write_text("MemAvailable: lots\n")
    assert meminfo_gib(tmp_path / "odd") is None
    probes = _probes(tmp_path)
    probes.meminfo = _meminfo(tmp_path, 4)
    rep = HealthReport()
    health._host_checks(rep, probes)
    by = _by(rep)
    assert by["host"].status == "warn" and "< 8 GiB" in by["host"].detail
    assert by["gpu_budget"] == Check(
        "gpu_budget", "ok", "no CARC vLLM backend active (carc-vllm@*)"
    )
    probes.meminfo = _meminfo(tmp_path, 64)
    rep = HealthReport()
    health._host_checks(rep, probes)
    assert _by(rep)["host"].status == "ok" and "64.0 GiB" in _by(rep)["host"].detail


def test_gpu_budget_notes_carc(tmp_path: Path) -> None:
    probes = _probes(tmp_path, carc=(0, CARC))
    rep = HealthReport()
    health._host_checks(rep, probes)
    c = _by(rep)["gpu_budget"]
    assert c.status == "ok" and "carc-vllm@qwen.service" in c.detail
    probes.meminfo = _meminfo(tmp_path, 20)
    rep = HealthReport()
    health._host_checks(rep, probes)
    c = _by(rep)["gpu_budget"]
    assert c.status == "warn" and "could not start" in c.detail
    runner = probes.runner
    assert isinstance(runner, Runner)
    assert [
        "systemctl",
        "list-units",
        "--no-legend",
        "--plain",
        "--state=active",
        "carc-vllm@*",
    ] in runner.calls
    rep = HealthReport()
    health._host_checks(rep, ServeProbes.offline())
    assert _by(rep)["gpu_budget"].detail.startswith("systemctl not available")


def test_units_active() -> None:
    assert units_active(Runner(units=(0, "active\nactive\n")))
    assert not units_active(Runner(units=(3, "active\ninactive\n")))
    assert not units_active(Runner(units=(127, "")))


def test_parse_listeners() -> None:
    text = (
        SS_LOOPBACK
        + "LISTEN 0 4096 [::1]:8090 [::]:*\nLISTEN 0 5 *:8700 *:*\nESTAB 0 0 1.2.3.4:8090 x\n"
    )
    assert parse_listeners(text, 8090) == ["127.0.0.1", "[::1]"]
    assert parse_listeners(text, 8700) == ["127.0.0.1", "*"]
    assert parse_listeners(text, 9999) == []


# -- serving endpoints ---------------------------------------------------------------------------


def test_without_serve_mode_one_reachability_line(tmp_path: Path) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    rep = doctor(_cfg(tmp_path), quick=True, probes=_probes(tmp_path, server))
    c = _by(rep)["serving"]
    assert c.status == "ok" and "answer /health" in c.detail and "doctor --serve" in c.detail
    assert server.paths == ["/health", "/health"]  # nothing guarded was touched
    rep = doctor(_cfg(tmp_path), quick=True, probes=_probes(tmp_path))
    c = _by(rep)["serving"]
    assert c.status == "warn" and c.detail.count("unreachable") == 2 and rep.ok


def test_serve_mode_is_green_against_the_fake_stack(tmp_path: Path) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    rep = doctor(
        _keyed(tmp_path),
        quick=True,
        serve=True,
        probes=_probes(tmp_path, server, ss=(0, SS_LOOPBACK)),
    )
    by = _by(rep)
    assert rep.ok, rep.lines()
    assert "serving" not in by
    for name in ("serving binds", "encoder health", "clm-serve health", "encoder auth",
                 "clm-serve auth", "encoder models", "clm-serve models", "clm golden"):  # fmt: skip
        assert by[name].status == "ok", (name, by[name])
    assert by["serving binds"].detail.startswith("loopback only")
    assert by["encoder auth"].detail == "GET /v1/models without a key -> 401"
    assert by["encoder models"].detail == "qwen3-8b"
    assert (
        "clm-latest" in by["clm-serve models"].detail and "clm-raw" in by["clm-serve models"].detail
    )
    assert by["clm golden"].detail.startswith("/v1/systemone clm-latest: 12 ms, answer ")
    golden = next(b for b in server.bodies if "questions" in b)
    assert set(golden["questions"]["fit"]["criteria"]) == set(health.GOLDEN_CRITERIA)
    text = json.dumps(rep.as_dict())
    assert CLM_KEY not in text and ENC_KEY not in text


def test_serve_mode_failures(tmp_path: Path) -> None:
    # An open /v1 route, an exposed bind and a key the server does not know.
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=None)
    exposed = SS_LOOPBACK.replace("127.0.0.1:8090", "0.0.0.0:8090")
    cfg = _keyed(tmp_path, MESA_CLM_CLM__API_KEY="not-the-" + CLM_KEY)
    rep = doctor(cfg, quick=True, serve=True, probes=_probes(tmp_path, server, ss=(0, exposed)))
    by = _by(rep)
    assert not rep.ok
    assert by["serving binds"].status == "fail" and "not loopback" in by["serving binds"].detail
    assert by["encoder auth"].status == "fail" and "-> 200" in by["encoder auth"].detail
    assert by["clm golden"].status == "fail" and "(401)" in by["clm golden"].detail
    assert "not-the-" not in json.dumps(rep.as_dict())


def test_serve_mode_without_keys_names_the_setting(tmp_path: Path) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    rep = doctor(
        _cfg(tmp_path),
        quick=True,
        serve=True,
        probes=_probes(tmp_path, server, ss=(0, SS_LOOPBACK)),
    )
    by = _by(rep)
    assert "MESA_CLM_ENCODER__API_KEY_FILE" in by["encoder models"].detail
    assert "MESA_CLM_CLM__API_KEY_FILE" in by["clm-serve models"].detail
    assert "clm golden" not in by


def test_unreachable_is_fail_under_serve_and_warn_when_auto_quick(tmp_path: Path) -> None:
    probes = _probes(tmp_path, ss=(0, ""), units=(0, "active\nactive\n"))
    rep = doctor(_keyed(tmp_path), quick=True, serve=True, probes=probes)
    by = _by(rep)
    assert by["encoder health"].status == "fail" and by["clm-serve health"].status == "fail"
    assert by["serving binds"].status == "fail" and "nothing listens" in by["serving binds"].detail
    assert "encoder auth" not in by and not rep.ok
    # Auto-detected serve mode (both units active) under --quick: warnings only.
    rep = doctor(_keyed(tmp_path), quick=True, serve=None, probes=probes)
    by = _by(rep)
    assert by["encoder health"].status == "warn" and rep.ok
    # Auto-detected without --quick: an unreachable endpoint fails.
    rep = doctor(_keyed(tmp_path), serve=None, probes=probes)
    assert _by(rep)["encoder health"].status == "fail"
    # Units inactive: no serve mode, the single warning line.
    rep = doctor(_keyed(tmp_path), serve=None, probes=_probes(tmp_path))
    assert _by(rep)["serving"].status == "warn" and "encoder health" not in _by(rep)


def test_ss_missing_is_a_warning(tmp_path: Path) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    rep = doctor(_keyed(tmp_path), quick=True, serve=True, probes=_probes(tmp_path, server))
    assert _by(rep)["serving binds"].status == "warn" and rep.ok
