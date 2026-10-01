"""The live provider's refusals and the annotate pre-flight (``mesa_clm.providers.live``;
DESIGN D5, D16, D28, K4): a lock that contradicts the vendored schema or the checkout's lock, a
missing or rejected key, an unserved model, an encoder that is not the lock's route, and a
mid-run 401 all stop the run instead of degrading it to ``ols_rank`` under the requested tier.
Hermetic: the fake transport answers both ports, locks live under ``tmp_path``."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from mesa_clm import serving
from mesa_clm.cli import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, main
from mesa_clm.clm.fingerprint import sign_lock_body
from mesa_clm.config import load_config
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers import live
from tests.fakes.clm_transport import FakeClmServer

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "tests" / "fixtures" / "cards" / "DP1.10003.001.brd_countdata.md"
LOCK = ROOT / "serving" / "serving.lock.json"
CLM_KEY = "clm-key-0123456789abcdef"
ENC_KEY = "enc-key-0123456789abcdef"
KEYS = {"MESA_CLM_CLM__API_KEY": CLM_KEY, "MESA_CLM_ENCODER__API_KEY": ENC_KEY}


def _lock(tmp_path: Path, name: str = "serving.lock.json", **changes: Any) -> Path:
    body = json.loads(LOCK.read_text(encoding="utf-8"))
    body.update(changes)
    path = tmp_path / name
    path.write_text(json.dumps(sign_lock_body(body)), encoding="utf-8")
    return path


def _stack(cfg: Any, server: FakeClmServer, **kw: Any) -> live.LiveStack:
    return live.clm_provider(cfg, transport=server.transport(), **kw)


def test_a_lock_with_another_schema_is_refused(tmp_path: Path) -> None:
    drifted = _lock(tmp_path, schema_sha256="0" * 64)
    cfg = load_config(env=KEYS)
    with pytest.raises(live.ProviderSetupError, match="schema_sha256 000000000000"):
        live.clm_provider(cfg, lock_path=drifted)


def test_an_installed_lock_that_differs_from_the_checkout_is_refused(tmp_path: Path) -> None:
    home = Path(serving.DEFAULT_HOME)
    home.mkdir(parents=True)
    head = dict(json.loads(LOCK.read_text(encoding="utf-8"))["head"])
    installed = _lock(home, head={**head, "revision": "f" * 40})
    assert live.live_lock_path() == installed
    cfg = load_config(env=KEYS)
    with pytest.raises(live.ProviderSetupError, match="re-run deploy/bin/mesa-clm-serve-bootstrap"):
        live.clm_provider(cfg)
    # Naming the lock explicitly is the caller's override; the same file as the checkout's is fine.
    live.clm_provider(cfg, lock_path=installed).close()
    installed.write_text(LOCK.read_text(encoding="utf-8"), encoding="utf-8")
    live.clm_provider(cfg).close()


def test_preflight_passes_on_a_matching_stack(tmp_path: Path) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    stack = _stack(load_config(env=KEYS), server)
    try:
        answering, detail = live.preflight(stack, load_config(env=KEYS))
    finally:
        stack.close()
    assert answering and "matches the lock" in detail and "models: clm-latest, clm-raw" in detail
    assert {"/health", "/v1/models"} <= set(server.paths)


@pytest.mark.parametrize(
    ("env", "server_kw", "match"),
    [
        ({}, {}, "no clm key configured"),
        ({"MESA_CLM_CLM__API_KEY": CLM_KEY}, {}, "no encoder key configured"),
        ({**KEYS, "MESA_CLM_CLM__API_KEY": "wrong-" + CLM_KEY}, {}, "rejected the configured key"),
        ({**KEYS, "MESA_CLM_ENCODER__API_KEY": "wrong-" + ENC_KEY}, {}, "encoder at"),
        (KEYS, {"encoder_owned_by": "mesa-clm-fallback"}, "not the lock's route 'vllm'"),
        (KEYS, {"encoder_root": "Qwen/Qwen3-0.6B"}, "the lock pins Qwen/Qwen3-8B"),
        (KEYS, {"max_model_len": 2048}, "max_model_len 2048 is not the lock's window 4096"),
        (KEYS, {"encoder_model": "other"}, "does not serve 'qwen3-8b'"),
    ],
)
def test_preflight_refuses_what_no_fallback_may_hide(
    env: dict[str, str], server_kw: dict[str, Any], match: str
) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY, **server_kw)
    cfg = load_config(env=env)
    stack = _stack(cfg, server)
    try:
        with pytest.raises(live.PreflightError, match=match) as exc:
            live.preflight(stack, cfg)
    finally:
        stack.close()
    assert CLM_KEY not in str(exc.value) and ENC_KEY not in str(exc.value)


def test_preflight_refuses_an_unserved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    monkeypatch.setattr(server.clm, "models", lambda: [{"name": "clm-latest", "kind": "head"}])
    cfg = load_config(env={**KEYS, "MESA_CLM_CLM__MODEL": "clm-raw"})
    stack = _stack(cfg, server)
    try:
        with pytest.raises(live.PreflightError, match=r"does not serve clm\.model 'clm-raw'"):
            live.preflight(stack, cfg)
    finally:
        stack.close()


def test_preflight_reports_an_unreachable_pair_as_not_answering() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    cfg = load_config(env=KEYS)
    stack = live.clm_provider(cfg, transport=httpx.MockTransport(refuse))
    try:
        answering, detail = live.preflight(stack, cfg)
    finally:
        stack.close()
    assert not answering and "unreachable" in detail


# -- through the CLI --------------------------------------------------------------------------------


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for key in list(os.environ):
        if key.startswith(("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(ROOT / "tests" / "fixtures" / "ols"))
    monkeypatch.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    db = tmp_path / "prov.duckdb"
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{db}")
    monkeypatch.setattr("mesa_clm.clm.http.time.sleep", lambda s: None)
    return db


def _serve(monkeypatch: pytest.MonkeyPatch, server: FakeClmServer) -> None:
    monkeypatch.setattr(
        live, "clm_provider", functools.partial(live.clm_provider, transport=server.transport())
    )


@pytest.mark.parametrize("configured", [None, "wrong-" + CLM_KEY])
def test_annotate_with_a_missing_or_wrong_key_refuses_before_any_run(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    configured: str | None,
) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    _serve(monkeypatch, server)
    monkeypatch.setenv("MESA_CLM_ENCODER__API_KEY", ENC_KEY)
    if configured:
        monkeypatch.setenv("MESA_CLM_CLM__API_KEY", configured)
    monkeypatch.setenv("MESA_CLM_DECIDER__OLS_RANK_FALLBACK", "true")  # never degrades a key
    for tier in ("zero_shot", "auto"):
        assert main(["annotate", "--card", str(CARD), "--tier", tier, "--out", "-"]) == EXIT_CONFIG
        captured = capsys.readouterr()
        assert captured.out == "" and ("no clm key" in captured.err or "rejected" in captured.err)
        assert CLM_KEY not in captured.err
    assert "/v1/systemone" not in server.paths
    assert not env.exists() or not DuckDBStore(env).runs()


def test_a_mid_run_401_fails_the_run_instead_of_degrading_it(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Rotated(FakeClmServer):
        """The key is rotated between the pre-flight and the first question."""

        def _clm(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/systemone":
                return httpx.Response(401, json={"detail": "invalid API key"})
            return super()._clm(request)

    server = Rotated(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    _serve(monkeypatch, server)
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    assert main(["annotate", "--card", str(CARD), "--tier", "zero_shot"]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "refused /v1/systemone (401)" in err and "recorded with status failed" in err
    [run] = DuckDBStore(env).runs()
    assert run["status"] == "failed"
    calls = DuckDBStore(env).clm_calls(run["run_id"])
    assert calls and calls[-1]["status"] == "error"


def test_annotate_counts_failed_calls(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    _serve(monkeypatch, server)
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    real = server.__call__
    seen = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/systemone":
            seen["n"] += 1
            if seen["n"] == 1:  # one request the server cannot answer: that group degrades
                return httpx.Response(422, json={"detail": "cannot answer"})
        return real(request)

    monkeypatch.setattr(server, "transport", lambda: httpx.MockTransport(flaky))
    _serve(monkeypatch, server)
    assert main(["annotate", "--card", str(CARD), "--tier", "zero_shot", "--out", "-"]) == EXIT_OK
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["n_failed_calls"] == 1 and data["degraded"]
    assert "CLM calls (1 failed)" in captured.err


# -- the encoder container against the lock (finding: /v1/models cannot tell recipes apart) --------


def _host(container_args: list[str] | None, **env_over: str | None) -> Any:
    """A scripted docker host (``tests.unit.test_serving.FakeHost``) running the encoder with
    ``container_args`` (``None``: not running) and the locked environment changed by
    ``env_over`` (a ``None`` value removes the variable)."""
    from tests.unit.test_serving import GOOD_ENV, IMAGE_ID, FakeHost

    host = FakeHost()
    if container_args is not None:
        host.container = (IMAGE_ID, container_args)
    env = dict(GOOD_ENV)
    for name, value in env_over.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    host.container_env = env
    return host


def _locked_args() -> list[str]:
    recipe = serving.load_lock(LOCK).recipe
    assert recipe is not None
    return list(recipe.args)


def _docker(args: Any) -> list[str]:
    return ["docker", *args]


def test_container_check_verified_departed_and_unverified() -> None:
    lock = serving.load_lock(LOCK)
    ok = live.container_check(lock, runner=_host(_locked_args()).run, docker=_docker)
    assert ok.ok and ok.verified and "pinned image and recipe" in ok.note()
    # The B0 arm of the batch-invariance experiment: same image and flags, no kernel switch.
    b0 = _host(_locked_args(), VLLM_BATCH_INVARIANT=None)
    bad = live.container_check(lock, runner=b0.run, docker=_docker)
    assert not bad.ok and "env VLLM_BATCH_INVARIANT None, lock '1'" in bad.note()
    eager = [a for a in _locked_args() if a != "--enforce-eager"]
    assert not live.container_check(lock, runner=_host(eager).run, docker=_docker).ok
    stopped = live.container_check(lock, runner=_host(None).run, docker=_docker)
    assert stopped.ok and not stopped.verified and "not running" in stopped.note()
    unreachable = live.container_check(lock, docker=lambda args: None)
    assert unreachable.ok and not unreachable.verified
    assert "not verified (docker is not reachable" in unreachable.note()


def test_annotate_refuses_an_encoder_container_with_another_recipe(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    _serve(monkeypatch, server)
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    host = _host(_locked_args(), VLLM_BATCH_INVARIANT=None)
    monkeypatch.setattr(live, "docker_argv", _docker)
    monkeypatch.setattr(live, "run_command", host.run)
    assert main(["annotate", "--card", str(CARD), "--tier", "zero_shot"]) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "encoder container:" in err and "VLLM_BATCH_INVARIANT" in err and "D5" in err
    assert "/v1/systemone" not in server.paths
    assert not env.exists() or not DuckDBStore(env).runs()


def test_annotate_records_whether_the_container_was_verified(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    _serve(monkeypatch, server)
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    assert main(["annotate", "--card", str(CARD), "--tier", "zero_shot", "--out", "-"]) == EXIT_OK
    captured = capsys.readouterr()
    notes = json.loads(captured.out)["preflight"]
    assert any(n.startswith("encoder container: not verified") for n in notes), notes
    assert "encoder container: not verified" in captured.err
    host = _host(_locked_args())
    monkeypatch.setattr(live, "docker_argv", _docker)
    monkeypatch.setattr(live, "run_command", host.run)
    assert main(["annotate", "--card", str(CARD), "--tier", "zero_shot", "--out", "-"]) == EXIT_OK
    notes = json.loads(capsys.readouterr().out)["preflight"]
    assert "encoder container: mesa-clm-encoder: pinned image and recipe" in notes
