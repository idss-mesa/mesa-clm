"""The ``features`` verbs of the console script (plan §5.2, §7.2): ``build`` over a small labels
snapshot through the fake encoder transport (one text per request by default, only missing texts,
exact token counts, truncated texts recorded and never embedded), ``export-npz`` read back with
CLM's own ``TextCache`` reader, ``project`` with a head exported to the serving home, ``stats``;
the refusals and their exit codes. Hermetic: the developer's ``MESA_CLM_*`` environment is
cleared, the feature root, the serving home and the snapshots live under ``tmp_path``, the live
fingerprint is the checkout's serving lock, and no key is ever printed."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest

from mesa_clm import serving
from mesa_clm.cli import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, main
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.fingerprint import clm_model_fp, load_serving_lock
from mesa_clm.clm.headproj import HeadProjector, random_head
from mesa_clm.learn import features as feat
from mesa_clm.providers import live
from tests.fakes.clm_transport import FakeClmServer
from tests.unit.test_features import TextCache

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
CLM_KEY = "clm-key-0123456789abcdef"
ENC_KEY = "enc-key-0123456789abcdef"
_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")
# Some other encoder's fingerprint: derived, never a literal (EncoderSpec may gain fields).
FOREIGN_FP = hashlib.sha256(b"another encoder").hexdigest()[:12]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A clean environment: the feature root and the serving home under ``tmp_path``, keys for
    both fake endpoints, no retry sleeps. Returns the feature root."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    root = tmp_path / "features"
    monkeypatch.setenv("MESA_CLM_FEATURES__DIR", str(root))
    monkeypatch.setenv("MESA_CLM_CLM__API_KEY", CLM_KEY)
    monkeypatch.setenv("MESA_CLM_ENCODER__API_KEY", ENC_KEY)
    # No installed serving lock here: the live lock is the checkout's serving/serving.lock.json.
    monkeypatch.setattr(serving, "DEFAULT_HOME", str(tmp_path / "serving-home"))
    monkeypatch.setattr("mesa_clm.clm.http.time.sleep", lambda s: None)
    return root


def _lock() -> Any:
    return load_serving_lock(live.live_lock_path())


def _wire(monkeypatch: pytest.MonkeyPatch, server: FakeClmServer | None) -> None:
    """The encoder client over the fake transport (or one that refuses every connection)."""
    if server is not None:
        transport: httpx.BaseTransport = server.transport()
    else:

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(
        EncoderClient,
        "from_config",
        functools.partial(EncoderClient.from_config, transport=transport),
    )


def _server(**kw: Any) -> FakeClmServer:
    return FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY, **kw)


def _snapshot(tmp_path: Path, *, long_description: bool = False) -> Path:
    """The rows of three column targets of the committed snapshot (the first two of term.fits,
    the first of column.ontology_fits; a target key carries no task, so a (column, aspect) both
    tasks label brings both tasks' rows). ``long_description`` stretches one column description
    past the token guard (identity is unchanged: the target key holds the column name, not its
    description)."""
    out = tmp_path / ("long.parquet" if long_description else "small.parquet")
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE t AS SELECT * FROM read_parquet(?) WHERE target_sha256 IN ("
            "SELECT target_sha256 FROM (SELECT DISTINCT task_id, target_sha256 FROM read_parquet(?) "
            "WHERE task_id IN ('term.fits', 'column.ontology_fits') "
            "AND json_extract_string(state_json, '$.scope') IS DISTINCT FROM 'dataset') "
            "QUALIFY row_number() OVER (PARTITION BY task_id ORDER BY target_sha256) <= "
            "CASE task_id WHEN 'term.fits' THEN 2 ELSE 1 END)",
            [str(SNAPSHOT), str(SNAPSHOT)],
        )
        if long_description:
            label_id, state_json = con.execute(
                "SELECT label_id, state_json FROM t WHERE task_id = 'term.fits' "
                "ORDER BY label_id LIMIT 1"
            ).fetchone() or ("", "{}")
            state = json.loads(state_json)
            state["column"]["description"] = "word " * 5000
            con.execute(
                "UPDATE t SET state_json = ? WHERE label_id = ?",
                [json.dumps(state, sort_keys=True), label_id],
            )
        con.execute(f"COPY t TO '{out}' (FORMAT PARQUET)")
    finally:
        con.close()
    return out


def _store(root: Path) -> feat.FeatureStore:
    return feat.FeatureStore(root, _lock().encoder_fp)


def _embed_bodies(server: FakeClmServer) -> list[dict[str, Any]]:
    return [
        json.loads(r.content)
        for r in server.requests
        if r.url.path == "/v1/embeddings" and r.content
    ]


def _build(snapshot: Path, *extra: str) -> int:
    return main(["features", "build", "--snapshot", str(snapshot), *extra])


def test_build_embeds_each_missing_text_once(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = _server()
    _wire(monkeypatch, server)
    snap = _snapshot(tmp_path)
    expected = feat.manifest(snap)
    assert _build(snap) == EXIT_OK
    captured = capsys.readouterr()
    store = _store(env)
    stats = store.stats()
    n = len(expected.texts())
    assert stats["vectors"] == n and stats["truncated"] == 0
    assert store.missing(expected.texts()) == []
    bodies = _embed_bodies(server)
    assert len(bodies) == n and all(len(b["input"]) == 1 for b in bodies)  # one per request
    assert "/tokenize" in server.paths and "/v1/models" in server.paths
    assert f"labels_sha256 {expected.labels_sha256}" in captured.out
    assert f"embedded {n} ({n} new vectors)" in captured.out
    assert f"encoder_fp {_lock().encoder_fp}" in captured.out
    assert f"embedded {n}/{n} texts" in captured.err
    assert json.loads(store.meta()["encoder_spec"]) == _lock().encoder.as_dict()
    assert CLM_KEY not in captured.out + captured.err and ENC_KEY not in captured.out + captured.err
    # A second build finds everything and embeds nothing.
    before = len(bodies)
    assert _build(snap) == EXIT_OK
    assert len(_embed_bodies(server)) == before
    assert f"already stored {n}, embedded 0 (0 new vectors)" in capsys.readouterr().out


def test_build_batches_and_records_truncated_texts(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = _server()
    _wire(monkeypatch, server)
    snap = _snapshot(tmp_path, long_description=True)
    texts = feat.manifest(snap, framing_ids="F7,F1").texts()
    long = [t for t in texts if len(t.split()) > 4096 - 16]
    assert long  # the stretched column's F7 and F1 contexts
    assert _build(snap, "--framings", "F7,F1", "--batch", "4") == EXIT_OK
    out = capsys.readouterr().out
    stats = _store(env).stats()
    assert stats["truncated"] == len(long) and stats["vectors"] == len(texts) - len(long)
    assert f"truncated {len(long)} (recorded, not embedded)" in out
    sizes = [len(b["input"]) for b in _embed_bodies(server)]
    assert max(sizes) == 4 and sum(sizes) == len(texts) - len(long)
    assert not any(t in server.embedded_texts for t in long)


def test_build_refusals(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snap = _snapshot(tmp_path)
    _wire(monkeypatch, _server())
    assert _build(snap, "--tasks", "column.aspect") == EXIT_CONFIG
    assert "unknown task" in capsys.readouterr().err
    assert _build(snap, "--framings", "F2") == EXIT_CONFIG
    assert _build(tmp_path / "missing.parquet") == EXIT_CONFIG
    assert _build(snap, "--batch", "0") == EXIT_CONFIG
    capsys.readouterr()
    # The encoder's window must be the lock's.
    monkeypatch.setenv("MESA_CLM_ENCODER__MAX_LEN", "2048")
    assert _build(snap) == EXIT_CONFIG
    assert "not the serving lock's" in capsys.readouterr().err
    monkeypatch.delenv("MESA_CLM_ENCODER__MAX_LEN")
    # The encoder must be on loopback (plan §6.4), and the live lock must verify (K4).
    monkeypatch.setenv("MESA_CLM_ENCODER__URL", "http://encoder.example.org:8090")
    assert _build(snap) == EXIT_CONFIG
    assert "loopback" in capsys.readouterr().err
    monkeypatch.delenv("MESA_CLM_ENCODER__URL")
    home = serving.serving_home()
    home.mkdir(parents=True)
    (home / serving.INSTALLED_LOCK).write_text("{}", encoding="utf-8")
    assert _build(snap) == EXIT_FAIL
    assert "serving lock" in capsys.readouterr().err
    (home / serving.INSTALLED_LOCK).unlink()
    # Unreachable encoder, another served model, no exact token counter.
    _wire(monkeypatch, None)
    assert _build(snap) == EXIT_FAIL
    assert "unreachable" in capsys.readouterr().err
    _wire(monkeypatch, _server(encoder_model="another-model"))
    assert _build(snap) == EXIT_FAIL
    assert "does not serve 'qwen3-8b'" in capsys.readouterr().err
    _wire(monkeypatch, _server(tokenize_supported=False))
    assert _build(snap) == EXIT_FAIL
    assert "exact token counts only" in capsys.readouterr().err
    assert not _store(env).embedded_texts()
    # A store that records another encoder_fp is refused before anything is embedded.
    store = _store(env)
    store.ensure()
    con = duckdb.connect(str(store.path))
    try:
        con.execute("UPDATE meta SET value = ? WHERE key = 'encoder_fp'", [FOREIGN_FP])
    finally:
        con.close()
    server = _server()
    _wire(monkeypatch, server)
    assert _build(snap) == EXIT_FAIL
    assert "refused" in capsys.readouterr().err and not _embed_bodies(server)


def test_build_stops_cleanly_mid_way_and_resumes(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = _server()
    _wire(monkeypatch, server)
    snap = _snapshot(tmp_path)
    texts = feat.manifest(snap).texts()
    real = server._embeddings
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] > 3:
            return httpx.Response(400, json={"error": {"message": "scripted", "code": 400}})
        return real(request)

    monkeypatch.setattr(server, "_embeddings", flaky)
    assert _build(snap) == EXIT_FAIL
    assert "stopped after 3 of" in capsys.readouterr().err
    assert _store(env).stats()["vectors"] == 3
    monkeypatch.setattr(server, "_embeddings", real)
    assert _build(snap) == EXIT_OK
    assert _store(env).stats()["vectors"] == len(texts)


def test_export_npz_is_read_by_textcache(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "npz"
    assert main(["features", "export-npz", "--out", str(out_dir)]) == EXIT_FAIL
    assert "features build" in capsys.readouterr().err
    _wire(monkeypatch, _server())
    snap = _snapshot(tmp_path)
    assert _build(snap) == EXIT_OK
    capsys.readouterr()
    assert main(["features", "export-npz", "--out", str(out_dir)]) == EXIT_OK
    out = capsys.readouterr().out
    lock = _lock()
    path = out_dir / feat.textcache_filename(lock.encoder.model, lock.encoder.max_len)
    assert f"wrote {path}" in out and "--embed-cache" in out
    texts = feat.manifest(snap).texts()
    cache = TextCache(str(path))
    assert cache.missing(texts) == []
    store = _store(env)
    for text in texts[:5]:
        assert (cache[text] == store.get([text])[0].astype("float16")).all()


def _export_head(source_sha256: str) -> Path:
    h = random_head(3, hidden_size=4096)
    head = HeadProjector(h.cfg, h.state, h.action, h.logit_scale, source_sha256)
    path = serving.serving_home() / serving.HEADS_DIR / "npz" / f"{_lock().head.sha256[:8]}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return head.to_npz(path)


def test_project_caches_the_pinned_head(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["features", "project"]) == EXIT_FAIL  # no store yet
    _wire(monkeypatch, _server())
    assert _build(_snapshot(tmp_path)) == EXIT_OK
    capsys.readouterr()
    assert main(["features", "project", "--model", "clm-raw"]) == EXIT_OK
    assert "nothing to project" in capsys.readouterr().out
    assert main(["features", "project"]) == EXIT_FAIL  # the head export is missing
    assert "head export not found" in capsys.readouterr().err
    _export_head("0" * 64)
    assert main(["features", "project"]) == EXIT_FAIL  # exported from another checkpoint
    assert "the serving lock pins" in capsys.readouterr().err
    lock = _lock()
    _export_head(lock.head.sha256)
    assert main(["features", "project"]) == EXIT_OK
    out = capsys.readouterr().out
    fp = clm_model_fp(lock.model_spec("clm-latest"))
    stats = _store(env).stats()
    n = stats["vectors"]
    assert stats["projections"] == {fp: {"action": n, "state": n}}
    assert f"clm_model_fp {fp}" in out and f"state {n}, action {n}" in out
    assert main(["features", "project"]) == EXIT_OK  # cached: nothing recomputed


def test_stats_lists_every_store(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["features", "stats"]) == EXIT_OK
    assert "none yet" in capsys.readouterr().out
    _wire(monkeypatch, _server())
    assert _build(_snapshot(tmp_path)) == EXIT_OK
    foreign = feat.FeatureStore(env, FOREIGN_FP)
    foreign.ensure()
    (env / hashlib.sha256(b"empty").hexdigest()[:12]).mkdir()  # no database: not a store
    con = duckdb.connect(str(foreign.path))
    try:
        con.execute("UPDATE meta SET value = '4' WHERE key = 'dim'")
    finally:
        con.close()
    capsys.readouterr()
    assert main(["features", "stats", "--json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    lock = _lock()
    assert data["root"] == str(env) and data["live_encoder_fp"] == lock.encoder_fp
    by_fp = {s["encoder_fp"]: s for s in data["stores"]}
    assert set(by_fp) == {lock.encoder_fp, FOREIGN_FP}
    assert by_fp[lock.encoder_fp]["vectors"] > 0 and "error" in by_fp[FOREIGN_FP]
    assert main(["features", "stats"]) == EXIT_OK
    text = capsys.readouterr().out
    assert f"{lock.encoder_fp} (live):" in text and f"{FOREIGN_FP}: " in text


def _rewriting(
    monkeypatch: pytest.MonkeyPatch, server: FakeClmServer, path: str, rewrite: Any
) -> None:
    """The fake transport with one encoder route's response replaced by ``rewrite(response)``."""

    def handler(request: httpx.Request) -> httpx.Response:
        response = server(request)
        if request.url.path == path:
            return rewrite(response)
        return response

    monkeypatch.setattr(
        EncoderClient,
        "from_config",
        functools.partial(EncoderClient.from_config, transport=httpx.MockTransport(handler)),
    )


def test_build_checks_what_the_encoder_reports(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snap = _snapshot(tmp_path)
    _wire(monkeypatch, _server(max_model_len=2048))
    assert _build(snap) == EXIT_FAIL
    assert "max_model_len 2048 is not the lock's" in capsys.readouterr().err

    def other_root(response: httpx.Response) -> httpx.Response:
        body = response.json()
        body["data"][0]["root"] = "Qwen/Qwen3-0.6B"
        return httpx.Response(200, json=body)

    _rewriting(monkeypatch, _server(), "/v1/models", other_root)
    assert _build(snap) == EXIT_FAIL
    assert "the lock pins Qwen/Qwen3-8B" in capsys.readouterr().err

    def broken(response: httpx.Response) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded", "code": 503}})

    _rewriting(monkeypatch, _server(), "/tokenize", broken)
    assert _build(snap) == EXIT_FAIL
    assert "token guard:" in capsys.readouterr().err
    assert not _store(env).embedded_texts()


def test_export_gate_second_head_and_stats_notes(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wire(monkeypatch, _server())
    assert _build(_snapshot(tmp_path)) == EXIT_OK
    lock = _lock()
    _export_head(lock.head.sha256)
    assert main(["features", "project"]) == EXIT_OK
    capsys.readouterr()
    # Projection lines in the plain stats output.
    assert main(["features", "stats"]) == EXIT_OK
    fp = clm_model_fp(lock.model_spec("clm-latest"))
    assert f"projections {fp}: action " in capsys.readouterr().out
    # Another export of the pinned checkpoint with different weights: the cache refuses it.
    h = random_head(4, hidden_size=4096)
    path = serving.serving_home() / serving.HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    HeadProjector(h.cfg, h.state, h.action, h.logit_scale, lock.head.sha256).to_npz(path)
    assert main(["features", "project"]) == EXIT_FAIL
    assert "another head" in capsys.readouterr().err
    # The fp16 gate guards the export.
    monkeypatch.setattr(feat, "fp16_roundtrip_min_cosine", lambda store: 0.5)
    assert main(["features", "export-npz", "--out", str(tmp_path / "npz")]) == EXIT_FAIL
    assert "gate" in capsys.readouterr().err
    # A lock that does not verify: stats still lists the stores, with a note.
    home = serving.serving_home()
    (home / serving.INSTALLED_LOCK).write_text("{}", encoding="utf-8")
    assert main(["features", "stats"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "live encoder_fp ?" in out and "serving lock" in out


def test_build_refuses_an_encoder_container_with_another_recipe(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``/v1/models`` reads the same for the B0 container (no ``VLLM_BATCH_INVARIANT``), whose
    vectors differ: the container is inspected before a vector is stored, and a store built
    under one vector recipe is refused under another."""
    from tests.unit.test_live_provider import _docker, _host, _locked_args

    server = _server()
    _wire(monkeypatch, server)
    snap = _snapshot(tmp_path)
    monkeypatch.setattr(live, "docker_argv", _docker)
    monkeypatch.setattr(live, "run_command", _host(_locked_args(), VLLM_BATCH_INVARIANT=None).run)
    assert _build(snap) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "encoder container:" in err and "VLLM_BATCH_INVARIANT" in err
    assert not _embed_bodies(server) and not _store(env).exists()
    monkeypatch.setattr(live, "run_command", _host(_locked_args()).run)
    assert _build(snap) == EXIT_OK
    out = capsys.readouterr().out
    assert "encoder container: mesa-clm-encoder: pinned image and recipe" in out
    meta = _store(env).meta()
    assert meta["vector_recipe_sha256"] == _lock().vector_recipe_sha256()
    assert meta["serving_lock_sha"] == _lock().lock_sha
    assert set(_store(env).stats()["vectors_by_lock_sha"]) == {_lock().lock_sha}
    # The same directory under a lock whose recipe differs (another image) is refused.
    con = duckdb.connect(str(_store(env).path))
    try:
        con.execute("UPDATE meta SET value = 'f' || value WHERE key = 'vector_recipe_sha256'")
    finally:
        con.close()
    assert _build(snap) == EXIT_FAIL
    assert "vector recipe" in capsys.readouterr().err
    assert main(["features", "stats"]) == EXIT_OK
    assert "NOT the live lock's" in capsys.readouterr().out
