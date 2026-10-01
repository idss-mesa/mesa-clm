"""The live serving stack (marker ``engine``; ``MESA_CLM_ENGINE=1``): ``doctor --serve`` is green
and a zero-shot annotate of a fixture card goes through the real encoder :8090 and clm-serve
:8700 with the fingerprint of the serving lock the host runs. Keys come from the configuration
(``MESA_CLM_CLM__API_KEY_FILE``, ``MESA_CLM_ENCODER__API_KEY_FILE``); OLS replays the fixtures,
so nothing but loopback is contacted."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mesa_clm.cards import load_card
from mesa_clm.config import Config, load_config
from mesa_clm.health import ServeProbes, doctor
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers import live
from mesa_clm.service import DecisionService

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        os.environ.get("MESA_CLM_ENGINE") != "1", reason="set MESA_CLM_ENGINE=1 (live serving)"
    ),
]

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "tests" / "fixtures" / "cards" / "DP1.10003.001.brd_countdata.md"
LIVE_CHECKS = (
    "serving binds",
    "encoder health",
    "clm-serve health",
    "encoder auth",
    "clm-serve auth",
    "encoder models",
    "clm-serve models",
    "clm golden",
)


def _keyed_config() -> Config:
    cfg = load_config()
    if not (cfg.clm.resolved_api_key() and cfg.encoder.resolved_api_key()):
        pytest.skip("configure MESA_CLM_CLM__API_KEY_FILE and MESA_CLM_ENCODER__API_KEY_FILE")
    return cfg


def test_doctor_serve_is_green() -> None:
    cfg = _keyed_config()
    rep = doctor(cfg, quick=True, serve=True, probes=ServeProbes())
    by = {c.name: c for c in rep.checks}
    assert rep.ok, rep.lines()
    for name in LIVE_CHECKS:
        assert by[name].status == "ok", by[name]


def test_live_zero_shot_annotate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(ROOT / "tests" / "fixtures" / "ols"))
    cfg = _keyed_config()
    stack = live.clm_provider(cfg)
    try:
        assert live.clm_status(stack.client)[0]
        store = DuckDBStore(tmp_path / "prov.duckdb")
        svc = DecisionService.from_config(cfg, stack.provider, store=store)
        run = svc.annotate(load_card(CARD), "engine-test", owner="engine-test", tier="zero_shot")
    finally:
        stack.close()
    assert run.proposals and not run.degraded
    assert all(p.outcome != "auto" for p in run.proposals)
    assert run.fingerprint["serving_lock_sha"] == stack.lock.lock_sha
    assert run.fingerprint["encoder_fp"] == stack.lock.encoder_fp
    calls = store.clm_calls(run.run_id)
    assert len(calls) == run.n_calls > 0 and all(c["status"] == "ok" for c in calls)
    assert {d["method"] for d in store.decisions(run.run_id)} >= {"clm"}
