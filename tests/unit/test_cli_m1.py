"""The M1 verbs of the ``mesa-clm`` console script (plan §7.2): ``framings``, ``annotate``
(fake and clm providers, the unreachable-clm-serve rule), ``explain``, ``review`` (interactive
and ``--pick``/``--decline``), ``feedback``, ``provenance migrate|export|import|prune``,
``serve keys|units|lock`` and ``doctor --serve``. Hermetic: OLS replayed from the fixtures,
clm-serve and the encoder answered by the fake transport, DuckDB and key files under
``tmp_path``, the developer's ``MESA_CLM_*`` environment cleared first."""

from __future__ import annotations

import functools
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from mesa_clm import framings, serving
from mesa_clm.cli import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, main, review_interactive
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers import live
from mesa_clm.registry import ANCHOR_KEY
from tests.fakes.clm_transport import FakeClmServer
from tests.fakes.pipeline import FAKE_SEED

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "tests" / "fixtures" / "cards" / "DP1.10003.001.brd_countdata.md"
OLS_DIR = ROOT / "tests" / "fixtures" / "ols"
CLM_KEY = "clm-key-0123456789abcdef"
ENC_KEY = "enc-key-0123456789abcdef"
_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A clean environment: replayed OLS, a sidecar under tmp_path, actor alice. Returns the
    sidecar path."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(OLS_DIR))
    monkeypatch.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    db = tmp_path / "prov.duckdb"
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{db}")
    monkeypatch.setenv("USER", "alice")
    # No installed serving lock on this machine: the checkout's serving/serving.lock.json.
    monkeypatch.setattr(serving, "DEFAULT_HOME", str(tmp_path / "serving-home"))
    return db


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def _annotate(capsys: pytest.CaptureFixture[str], *extra: str) -> dict[str, Any]:
    code = main(
        [
            "annotate",
            "--card",
            str(CARD),
            "--provider",
            "fake",
            "--fake-seed",
            str(FAKE_SEED),
            "--out",
            "-",
            *extra,
        ]
    )
    assert code == EXIT_OK
    return dict(json.loads(_out(capsys)))


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, dict[str, Any]]]:
    """One fake run of the fixture card through the CLI (owner alice), copied per test."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("cli-m1")
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            mp.delenv(key, raising=False)
    db = tmp / "prov.duckdb"
    mp.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    mp.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(OLS_DIR))
    mp.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    mp.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{db}")
    out = tmp / "run.json"
    try:
        code = main(
            [
                "--actor",
                "alice",
                "annotate",
                "--card",
                str(CARD),
                "--provider",
                "fake",
                "--fake-seed",
                str(FAKE_SEED),
                "--out",
                str(out),
            ]
        )
        assert code == EXIT_OK
        yield db, json.loads(out.read_text(encoding="utf-8"))
    finally:
        mp.undo()


@pytest.fixture
def run(template: tuple[Path, dict[str, Any]], env: Path) -> dict[str, Any]:
    shutil.copy(template[0], env)
    return template[1]


def _store(env: Path) -> DuckDBStore:
    return DuckDBStore(env)


def _pending(env: Path, run_id: str) -> list[dict[str, Any]]:
    from mesa_clm.cli import _reader_service
    from mesa_clm.config import load_config

    svc = _reader_service(load_config(), _store(env))
    return list(svc.run_summary(UUID(run_id))["pending"])


# -- parser and framings ---------------------------------------------------------------------------


def test_help_lists_the_m1_verbs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = _out(capsys)
    for verb in ("framings", "annotate", "explain", "review", "feedback", "provenance", "serve"):
        assert verb in out
    for argv, needle in (
        (["provenance", "--help"], "{migrate,export,import,prune}"),
        (["serve", "--help"], "{keys,units,lock}"),
    ):
        with pytest.raises(SystemExit):
            main(argv)
        assert needle in _out(capsys)


def test_framings_check_and_update(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["framings", "--check"]) == EXIT_OK
    assert f"in sync; lock_sha {framings.lock_sha()}" in _out(capsys)
    lock = tmp_path / "framings.lock.json"
    data = json.loads(framings.LOCK_PATH.read_text(encoding="utf-8"))
    first = next(iter(data["tasks"]))
    data["tasks"][first]["task_key"] = "0" * 16
    lock.write_text(json.dumps(data), encoding="utf-8")
    assert main(["framings", "--check", "--lock", str(lock)]) == EXIT_FAIL
    out = _out(capsys)
    assert "does not match" in out and "--update-lock" in out
    assert main(["framings", "--update-lock", "--lock", str(lock)]) == EXIT_OK
    assert main(["framings", "--check", "--lock", str(lock)]) == EXIT_OK
    with pytest.raises(SystemExit) as exc:
        main(["framings"])
    assert exc.value.code == EXIT_CONFIG


# -- annotate ----------------------------------------------------------------------------------


def test_annotate_fake_writes_the_run_and_the_eval_result(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run.json"
    code = main(
        [
            "annotate",
            "--card",
            str(CARD),
            "--provider",
            "fake",
            "--fake-seed",
            str(FAKE_SEED),
            "--out",
            str(out),
        ]
    )
    assert code == EXIT_OK
    report = _out(capsys)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert report.startswith(f"run {data['run_id']} (owner alice, card DP1.10003.001")
    assert "abstained: " in report and f"next: mesa-clm review --run-id {data['run_id']}" in report
    assert data["owner"] == "alice" and data["tier"] == "auto" and data["proposals"]
    assert all(p["outcome"] == "proposed" for p in data["proposals"])
    row = _store(env).run(UUID(data["run_id"]))
    assert row is not None and row["provider"] == "fake" and row["owner"] == "alice"
    evaluated = _annotate(capsys, "--eval-result", "--owner", "bob")
    assert evaluated["family"] == "mesa-clm" and evaluated["missing_labels"] == 0
    assert evaluated["avus"] and "next_step" not in evaluated
    assert _store(env).run(UUID(evaluated["run_id"]))["owner"] == "bob"  # type: ignore[index]


def test_annotate_tier_ols_rank_and_unservable_tiers(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _annotate(capsys, "--tier", "ols_rank")
    assert data["tier"] == "ols_rank" and data["degraded"]
    assert all(p["method"] == "ols_rank" for p in data["proposals"])
    assert main(
        ["annotate", "--card", str(CARD), "--provider", "fake", "--tier", "calibrated"]
    ) == (EXIT_FAIL)
    assert "not servable" in capsys.readouterr().err


def test_annotate_bad_cards_and_replay_misses(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["annotate", "--card", str(tmp_path / "nope.md"), "--provider", "fake"]) == (
        EXIT_CONFIG
    )
    assert "no such file" in capsys.readouterr().err
    bad = tmp_path / "bad.md"
    bad.write_text("not a card\n", encoding="utf-8")
    assert main(["annotate", "--card", str(bad), "--provider", "fake"]) == EXIT_CONFIG
    assert "not a dataset card" in capsys.readouterr().err
    empty = tmp_path / "empty-ols"
    empty.mkdir()
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(empty))
    code = main(["annotate", "--card", str(CARD), "--provider", "fake", "--fake-seed", "5"])
    assert code == EXIT_FAIL
    err = capsys.readouterr().err
    assert "OLS replay miss" in err and "status failed" in err
    runs = _store(env).runs()
    assert [r["status"] for r in runs] == ["failed"]


def _clm_env(monkeypatch: pytest.MonkeyPatch, server: FakeClmServer | None) -> None:
    """``--provider clm`` over the fake transport (or a refusing one when ``server`` is None)."""
    import httpx

    monkeypatch.setenv("MESA_CLM_CLM__API_KEY", CLM_KEY)
    monkeypatch.setenv("MESA_CLM_ENCODER__API_KEY", ENC_KEY)
    if server is not None:
        transport: httpx.BaseTransport = server.transport()
    else:

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(
        live, "clm_provider", functools.partial(live.clm_provider, transport=transport)
    )
    monkeypatch.setattr("mesa_clm.clm.http.time.sleep", lambda s: None)


def test_annotate_clm_through_the_wire(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeClmServer(clm_api_key=CLM_KEY, encoder_api_key=ENC_KEY)
    _clm_env(monkeypatch, server)
    code = main(["annotate", "--card", str(CARD), "--tier", "zero_shot", "--out", "-"])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    lock = serving.load_lock()
    assert data["fingerprint"]["serving_lock_sha"] == lock.lock_sha
    assert data["fingerprint"]["encoder_fp"] == lock.encoder_fp
    assert f"serving lock {ROOT / 'serving' / 'serving.lock.json'}" in captured.err
    assert "clm-serve at http://127.0.0.1:8700 ok" in captured.err
    assert data["n_calls"] > 0 and data["input_tokens"] > 0 and not data["degraded"]
    assert all(p["outcome"] != "auto" for p in data["proposals"])
    store = _store(env)
    run_id = UUID(data["run_id"])
    calls = store.clm_calls(run_id)
    assert len(calls) == data["n_calls"] and all(c["status"] == "ok" for c in calls)
    methods = {d["method"] for d in store.decisions(run_id)}
    assert "clm" in methods and "fake" not in methods
    assert "/tokenize" in server.paths and "/health" in server.paths
    assert CLM_KEY not in captured.out + captured.err and ENC_KEY not in captured.out + captured.err


def test_annotate_clm_unreachable_refuses_or_degrades(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clm_env(monkeypatch, None)
    assert main(["annotate", "--card", str(CARD)]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "unreachable" in err and "--tier ols_rank" in err and "ols_rank_fallback" in err
    assert not _store(env).path.exists() or not _store(env).runs()
    monkeypatch.setenv("MESA_CLM_DECIDER__OLS_RANK_FALLBACK", "true")
    # An explicit CLM tier still refuses.
    assert main(["annotate", "--card", str(CARD), "--tier", "zero_shot"]) == EXIT_FAIL
    capsys.readouterr()
    assert main(["annotate", "--card", str(CARD), "--out", "-"]) == EXIT_OK
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["tier"] == "ols_rank" and data["degraded"] and data["proposals"]
    assert all(p["method"] == "ols_rank" and p["outcome"] == "proposed" for p in data["proposals"])
    assert "degraded to ols_rank (decider.ols_rank_fallback" in captured.err


def test_annotate_clm_refuses_a_promoted_head_and_a_bad_lock(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MESA_CLM_CLM__MODEL", "mesa-term-fits-v1")
    assert main(["annotate", "--card", str(CARD)]) == EXIT_FAIL
    assert "promoted heads arrive with M7" in capsys.readouterr().err
    monkeypatch.delenv("MESA_CLM_CLM__MODEL")
    home = Path(serving.DEFAULT_HOME)
    home.mkdir(parents=True)
    (home / serving.INSTALLED_LOCK).write_text("{}", encoding="utf-8")
    assert live.live_lock_path() == home / serving.INSTALLED_LOCK
    assert main(["annotate", "--card", str(CARD)]) == EXIT_FAIL
    assert "serving lock" in capsys.readouterr().err
    monkeypatch.setenv("MESA_CLM_CLM__BASE_URL", "http://clm.example.org:8700")
    (home / serving.INSTALLED_LOCK).unlink()
    assert main(["annotate", "--card", str(CARD)]) == EXIT_CONFIG
    assert "loopback" in capsys.readouterr().err


# -- explain, review, feedback -----------------------------------------------------------------------


def test_explain_by_prefix_path_and_owner(
    run: dict[str, Any], env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["explain", "--run-id", run["run_id"][:8], "--limit", "2"]) == EXIT_OK
    data = json.loads(_out(capsys))
    assert data["run"]["run_id"] == run["run_id"] and len(data["decisions"]) == 2
    assert data["pending_groups"]
    assert main(["explain", "--irods-path", "/zone/home/alice/x.csv"]) == EXIT_OK
    assert json.loads(_out(capsys)) == {"irods_path": "/zone/home/alice/x.csv", "runs": []}
    assert main(["--actor", "bob", "explain", "--run-id", run["run_id"]]) == EXIT_FAIL
    assert "another owner" in capsys.readouterr().err
    # bob has no run with that prefix
    assert main(["--actor", "bob", "explain", "--run-id", run["run_id"][:8]]) == EXIT_FAIL
    assert main(["explain", "--run-id", "zz"]) == EXIT_CONFIG


def test_explain_without_a_sidecar(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", "--run-id", "0" * 8]) == EXIT_FAIL
    assert "no sidecar" in capsys.readouterr().err and not env.exists()


def _terminal(monkeypatch: pytest.MonkeyPatch, attached: bool) -> None:
    """stdin is (or is not) an interactive terminal: the DESIGN A2 condition for via='cli'."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: attached)


def _labels(env: Path, run_id: str) -> list[dict[str, Any]]:
    from mesa_clm.provenance.labels import LabelStore

    rows = []
    for task in ("term.fits", "column.ontology_fits"):
        rows.extend(LabelStore(env).labels_for(task))
    return [r for r in rows if str(r["origin"]).startswith("override:")]


def _offered(env: Path, group_id: str) -> list[str]:
    from mesa_clm.cli import _reader_service
    from mesa_clm.config import load_config

    svc = _reader_service(load_config(), _store(env))
    return [c["option_key"] for c in svc.candidates_for_group(UUID(group_id)) if not c["is_anchor"]]


def test_review_at_a_terminal_records_curator_labels(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    _terminal(monkeypatch, True)
    pending = _pending(env, run["run_id"])
    assert len(pending) >= 3
    g1, g2, g3 = (p["group_id"] for p in pending[:3])
    key = _offered(env, g1)[0]
    # A typo in a later answer records nothing at all.
    code = main(
        ["review", "--run-id", run["run_id"], "--pick", f"{g1[:8]}={key}", "--pick", f"{g2}=NOPE:1"]
    )
    assert code == EXIT_FAIL and "is not offered" in capsys.readouterr().err
    assert not _store(env).overrides(UUID(run["run_id"]))
    code = main(
        [
            "review",
            "--run-id",
            run["run_id"][:8],
            "--pick",
            f"{g1[:8]}={key}",
            "--pick",
            f"{g2}=none",
            "--decline",
            g3,
        ]
    )
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "via=tool" not in captured.err
    results = json.loads(captured.out)
    assert [r["action"] for r in results] == ["pick", "none", "decline"]
    assert [r["outcome"] for r in results] == ["human", "rejected", "declined"]
    assert results[0]["label_source"] == "curator" and results[0]["labels_written"] >= 1
    assert {r["via"] for r in results} == {"cli"}
    overrides = _store(env).overrides(UUID(run["run_id"]))
    assert {o["via"] for o in overrides} == {"cli"} and len(overrides) == 3
    labels = _labels(env, run["run_id"])
    assert {r["label_source"] for r in labels} == {"curator", "curator_implicit"}
    assert all(r["fold_eligible"] and r["weight"] > 0 for r in labels)
    left = {p["group_id"] for p in _pending(env, run["run_id"])}
    assert g1 not in left and g2 not in left and g3 in left  # a decline leaves it pending
    assert main(["review", "--run-id", run["run_id"], "--pick", "nonsense"]) == EXIT_CONFIG


def test_review_without_a_terminal_records_an_agents_answers(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    _terminal(monkeypatch, False)
    pending = _pending(env, run["run_id"])
    g1, g2 = pending[0]["group_id"], pending[1]["group_id"]
    key = _offered(env, g1)[0]
    code = main(
        ["review", "--run-id", run["run_id"], "--pick", f"{g1}={key}", "--pick", f"{g2}=none"]
    )
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "stdin is not a terminal" in captured.err and "via=tool" in captured.err
    assert "DESIGN A2" in captured.err
    results = json.loads(captured.out)
    assert [r["via"] for r in results] == ["tool", "tool"]
    assert {r["label_source"] for r in results} == {"agent_pick"}
    overrides = _store(env).overrides(UUID(run["run_id"]))
    assert {o["via"] for o in overrides} == {"tool"}
    labels = _labels(env, run["run_id"])
    assert labels and {r["label_source"] for r in labels} == {"agent_pick"}
    assert all(r["weight"] == 0.0 and not r["fold_eligible"] for r in labels)
    accepted = [
        link
        for link in _store(env).links(UUID(run["run_id"]))
        if link["write_status"] == "accepted"
    ]
    assert accepted and {link["accepted_by"] for link in accepted} == {"agent"}
    # An agent's answer leaves both groups pending for a curator.
    still = {p["group_id"]: p for p in _pending(env, run["run_id"])}
    assert still[g1]["agent_answered"] and still[g2]["agent_answered"]


def test_review_needs_a_terminal_for_the_interactive_walk(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    _terminal(monkeypatch, False)
    assert main(["review", "--run-id", run["run_id"]]) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "needs a terminal" in err and "via=tool" in err
    assert main(["--actor", "bob", "review", "--run-id", run["run_id"]]) == EXIT_FAIL


def test_review_answers_pending_groups_once_each(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    _terminal(monkeypatch, True)
    from mesa_clm.cli import _reader_service
    from mesa_clm.config import load_config

    summary = _reader_service(load_config(), _store(env)).run_summary(UUID(run["run_id"]))
    onto = next(g for g in summary["groups"] if g["task_id"] == "column.ontology_fits")
    gid = str(onto["group_id"])
    key = _offered(env, gid)[0]
    assert main(["review", "--run-id", run["run_id"], "--pick", f"{gid}={key}"]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "does not wait for a reviewer" in err and "feedback --group-id" in err
    g = _pending(env, run["run_id"])[0]["group_id"]
    a, *rest = _offered(env, g)
    second = rest[0] if rest else "none"
    code = main(
        ["review", "--run-id", run["run_id"], "--pick", f"{g}={a}", "--pick", f"{g}={second}"]
    )
    assert code == EXIT_CONFIG and "more than once" in capsys.readouterr().err
    code = main(["review", "--run-id", run["run_id"], "--pick", f"{g}={a}", "--decline", g])
    assert code == EXIT_CONFIG
    assert not _store(env).overrides(UUID(run["run_id"]))


def test_a_curator_answer_supersedes_an_agents_and_is_final(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    g = next(
        p["group_id"]
        for p in _pending(env, run["run_id"])
        if len(_offered(env, p["group_id"])) >= 2
    )
    a, b = _offered(env, g)[:2]
    _terminal(monkeypatch, False)  # the agent picks a
    assert main(["review", "--run-id", run["run_id"], "--pick", f"{g}={a}"]) == EXIT_OK
    capsys.readouterr()
    # A different agent answer is refused; the same one is idempotent.
    assert main(["feedback", "--group-id", g, "--action", "pick", "--option-key", b]) == EXIT_FAIL
    assert "already has an agent answer" in capsys.readouterr().err
    assert main(["feedback", "--group-id", g, "--action", "pick", "--option-key", a]) == EXIT_OK
    capsys.readouterr()
    _terminal(monkeypatch, True)  # the curator picks b
    assert main(["review", "--run-id", run["run_id"], "--pick", f"{g}={b}"]) == EXIT_OK
    capsys.readouterr()
    links = [
        link
        for link in _store(env).links(UUID(run["run_id"]))
        if str(link["group_id"]) == g and link["write_status"] == "accepted"
    ]
    assert [(link["term_curie"], link["accepted_by"]) for link in links] == [(b, "human")]
    curator = {
        (r["option_key"], r["label"])
        for r in _labels(env, run["run_id"])
        if r["label_source"] == "curator"
    }
    assert (b, "Yes") in curator and (a, "Yes") not in curator
    assert g not in {p["group_id"] for p in _pending(env, run["run_id"])}
    # The curator's answer is final in M1: another answer is refused, the same one is a no-op.
    assert main(["review", "--run-id", run["run_id"], "--pick", f"{g}={a}"]) == EXIT_FAIL
    capsys.readouterr()
    assert main(["feedback", "--group-id", g, "--action", "reject"]) == EXIT_FAIL
    assert "already has a curator answer" in capsys.readouterr().err
    assert main(["feedback", "--group-id", g, "--action", "pick", "--option-key", b]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["labels_written"] == 0


def test_review_interactive(
    run: dict[str, Any],
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["huh", "1", "0", "d", "", "q"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["review", "--run-id", run["run_id"]]) == EXIT_OK
    out = _out(capsys)
    assert "wait for a reviewer" in out and "0. none of these (the anchor)  p_fit" in out
    assert "'huh' is not an answer" in out and "p_fit" in out
    assert "picked 1, none 1, declined 1, skipped 1" in out
    actions = sorted(o["action"] for o in _store(env).overrides(UUID(run["run_id"])))
    assert actions == ["decline", "none", "pick"]


def test_review_interactive_stops_at_end_of_input(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    _terminal(monkeypatch, True)

    def eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["review", "--run-id", run["run_id"]]) == EXIT_OK
    assert "picked 0, none 0, declined 0, skipped 0" in _out(capsys)

    def interrupted(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    assert main(["review", "--run-id", run["run_id"]]) == EXIT_OK
    assert not _store(env).overrides(UUID(run["run_id"]))


def test_review_interactive_helper_directly(run: dict[str, Any], env: Path) -> None:
    from mesa_clm.cli import _reader_service
    from mesa_clm.config import load_config

    svc = _reader_service(load_config(), _store(env))
    pending = svc.run_summary(UUID(run["run_id"]))["pending"]
    said: list[str] = []
    done = review_interactive(
        svc, pending[:1], owner="alice", actor="alice", ask=lambda p: "n", say=said.append
    )
    assert done == {"picked": 0, "none": 1, "declined": 0, "skipped": 0}
    assert said[0].startswith("[1/1] term.fits") and any("recorded none" in s for s in said)


def test_feedback(
    run: dict[str, Any], env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    pending = _pending(env, run["run_id"])
    g, g2 = pending[0]["group_id"], pending[1]["group_id"]
    _terminal(monkeypatch, True)
    assert main(["feedback", "--group-id", g, "--action", "reject"]) == EXIT_OK
    res = json.loads(_out(capsys))
    assert res["action"] == "reject" and res["outcome"] == "rejected"
    assert res["label_source"] == "curator" and res["via"] == "cli"
    override = _store(env).overrides(UUID(run["run_id"]))[0]
    assert override["via"] == "cli" and override["chosen_option_key"] == ANCHOR_KEY
    # --action pick needs --option-key; 'none' is none of these; group ids take a prefix.
    assert main(["feedback", "--group-id", g2, "--action", "pick"]) == EXIT_CONFIG
    assert "needs --option-key" in capsys.readouterr().err
    assert main(["feedback", "--group-id", g2[:8], "--action", "pick", "--option-key", "none"]) == 0
    res = json.loads(_out(capsys))
    assert res["action"] == "none" and res["group_id"] == g2
    assert main(["feedback", "--group-id", "x", "--action", "pick", "--option-key", "a"]) == 2
    assert main(["feedback", "--group-id", "ffffffff", "--action", "decline"]) == EXIT_FAIL
    assert main(["--actor", "bob", "feedback", "--group-id", g, "--action", "decline"]) == EXIT_FAIL
    g3 = pending[2]["group_id"]
    assert main(["feedback", "--group-id", g3, "--action", "pick", "--option-key", "NOPE:1"]) == (
        EXIT_FAIL
    )
    # Without a terminal the same verb is an agent's answer.
    _terminal(monkeypatch, False)
    key = _offered(env, g3)[0]
    assert main(["feedback", "--group-id", g3, "--action", "pick", "--option-key", key]) == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["via"] == "tool" and res["label_source"] == "agent_pick"
    assert "via=tool" in captured.err


def test_review_verbs_on_a_shared_postgres_sidecar_act_as_the_os_account(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mesa_clm import cli

    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", "postgresql://u:pw@db.example/clm")
    monkeypatch.setattr(cli, "_os_user", lambda: "alice")
    for argv in (
        ["--actor", "bob", "explain", "--run-id", "abcd1234"],
        ["--actor", "bob", "review", "--run-id", "abcd1234"],
        ["--actor", "bob", "feedback", "--group-id", "abcd1234", "--action", "decline"],
    ):
        assert main(argv) == EXIT_CONFIG
        err = capsys.readouterr().err
        assert "OS account (alice)" in err and "pw" not in err


def test_blank_actor_and_negative_ttl_are_usage_errors(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (
        ["--actor", " ", "annotate", "--card", str(CARD), "--provider", "fake"],
        ["provenance", "prune", "--ttl-days", "-1"],
        ["provenance", "prune", "--ttl-days", "soon"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "must not be blank" in err and "must be >= 0" in err


# -- provenance ----------------------------------------------------------------------------------


def test_provenance_verbs(
    run: dict[str, Any], env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    assert main(["provenance", "migrate"]) == EXIT_OK
    assert "mesa_clm schema v1" in _out(capsys)
    other = tmp_path / "other.duckdb"
    assert main(["provenance", "migrate", "--dsn", f"duckdb:///{other}"]) == EXIT_OK
    assert other.is_file()
    capsys.readouterr()
    out_dir = tmp_path / "export"
    assert main(["provenance", "export", "--run-id", run["run_id"][:8], "--out", str(out_dir)]) == 0
    manifest = out_dir / run["run_id"] / "manifest.json"
    assert manifest.is_file() and str(manifest) in _out(capsys)
    assert main(["provenance", "prune", "--dry-run", "--ttl-days", "1"]) == EXIT_OK
    report = json.loads(_out(capsys))
    assert report["dry_run"] is True and run["run_id"] in report["skipped"]
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{other}")
    assert main(["provenance", "import", str(out_dir / run["run_id"])]) == EXIT_OK
    assert DuckDBStore(other).run(UUID(run["run_id"])) is not None
    assert main(["provenance", "import", str(out_dir / run["run_id"])]) == EXIT_FAIL
    assert "already exists" in capsys.readouterr().err
    assert main(["provenance", "import", str(tmp_path / "missing")]) == EXIT_CONFIG
    assert main(["provenance", "migrate", "--dsn", "sqlite:///x"]) == EXIT_CONFIG


def test_provenance_error_paths_exit_cleanly(
    run: dict[str, Any], env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = UUID(run["run_id"])
    # An export into a file path (or an unwritable directory) fails without marking the run.
    assert main(["provenance", "export", "--run-id", run["run_id"], "--out", str(env)]) == 1
    assert "cannot write the export (NotADirectoryError)" in capsys.readouterr().err
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        assert main(["provenance", "export", "--run-id", run["run_id"], "--out", str(blocked)]) == 1
    finally:
        blocked.chmod(0o700)
    assert "was not marked exported" in capsys.readouterr().err
    assert _store(env).run(run_id)["exported_at"] is None  # type: ignore[index]
    assert not (blocked / run["run_id"]).exists()
    # An unsupported sidecar DSN for import is a usage error, not a traceback.
    argv = ["--provenance", "sqlite:///x.db", "provenance", "import", str(tmp_path)]
    assert main(argv) == EXIT_CONFIG
    assert "unsupported provenance DSN" in capsys.readouterr().err
    # A bare *.duckdb path is a DuckDB DSN everywhere; review verbs never create one.
    bare = tmp_path / "bare.duckdb"
    assert main(["--provenance", str(bare), "explain", "--run-id", "abcd"]) == EXIT_FAIL
    assert "no sidecar" in capsys.readouterr().err and not bare.exists()
    # A directory given as the serving lock is a failed check, not a traceback.
    assert main(["serve", "lock", "--check", "--lock", str(tmp_path)]) == EXIT_FAIL
    assert "cannot read the serving lock" in _out(capsys)


def test_annotate_expands_a_home_relative_fixture_dir(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    home = tmp_path / "home"
    home.mkdir()
    (home / "ols").symlink_to(OLS_DIR)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", "~/ols")
    code = main(["annotate", "--card", str(CARD), "--provider", "fake", "--fake-seed", "5"])
    assert code == EXIT_OK, capsys.readouterr().err
    assert not (Path.cwd() / "~").exists()


def test_a_reviewed_run_round_trips_through_export_and_import(
    run: dict[str, Any], env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    from mesa_clm.provenance.export import run_rows

    _terminal(monkeypatch, True)
    picks: list[str] = []
    for p in _pending(env, run["run_id"])[:4]:  # the last offered candidate: new links too
        picks += ["--pick", f"{p['group_id']}={_offered(env, p['group_id'])[-1]}"]
    assert main(["review", "--run-id", run["run_id"], *picks]) == EXIT_OK
    out = tmp_path / "exp"
    assert main(["provenance", "export", "--run-id", run["run_id"], "--out", str(out)]) == EXIT_OK
    other = tmp_path / "other.duckdb"
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{other}")
    assert main(["provenance", "import", str(out / run["run_id"])]) == EXIT_OK
    capsys.readouterr()
    mine = run_rows(_store(env), UUID(run["run_id"]))
    theirs = run_rows(DuckDBStore(other), UUID(run["run_id"]))
    assert mine.keys() == theirs.keys()
    for table, rows in mine.items():
        assert rows == theirs[table], table
    assert mine["human_overrides"] and mine["runs"][0]["exported_at"] is not None


# -- serve ---------------------------------------------------------------------------------------


def test_serve_keys_never_print_a_key(
    tmp_path: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secrets = tmp_path / "secrets"
    assert main(["serve", "keys", "--init", "--secrets-dir", str(secrets)]) == EXIT_OK
    out = _out(capsys)
    keys = [(secrets / n).read_text(encoding="utf-8").strip() for n in ("clm.key", "encoder.key")]
    assert "created clm.key, encoder.key" in out and "systemctl" not in out
    assert f"MESA_CLM_CLM__API_KEY_FILE={secrets / 'clm.key'}" in out
    assert not any(k in out for k in keys)
    assert main(["serve", "keys", "--init", "--secrets-dir", str(secrets)]) == EXIT_OK
    assert "kept (both keys already present)" in _out(capsys)
    assert main(["serve", "keys", "--rotate", "--secrets-dir", str(secrets)]) == EXIT_OK
    out = _out(capsys)
    new = [(secrets / n).read_text(encoding="utf-8").strip() for n in ("clm.key", "encoder.key")]
    assert new != keys and not any(k in out for k in keys + new)
    assert "systemctl --user restart mesa-clm-encoder.service mesa-clm-serve.service" in out
    (secrets / "clm.key").chmod(0o644)
    assert main(["serve", "keys", "--init", "--secrets-dir", str(secrets)]) == EXIT_FAIL
    assert not any(k in capsys.readouterr().err for k in new)


def test_serve_keys_in_the_default_home_are_read_by_default(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from mesa_clm.config import load_config

    assert main(["serve", "keys", "--init"]) == EXIT_OK
    out = _out(capsys)
    assert "reads these key files by default" in out and "export MESA_CLM" not in out
    secrets = Path(serving.DEFAULT_HOME) / serving.SECRETS_DIR
    cfg = load_config()
    key = (secrets / "clm.key").read_text(encoding="utf-8").strip()
    assert cfg.clm.resolved_api_key() == key and key not in out
    assert cfg.encoder.effective_api_key_file() == (str(secrets / "encoder.key"), True)


def test_serve_units_and_lock(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Hermetic: no docker and no commands on this host, whatever runs here.
    monkeypatch.setattr(serving, "docker_argv", lambda args: None)
    monkeypatch.setattr(serving, "run_command", lambda argv, **kw: serving.CommandResult(127, ""))
    assert main(["serve", "units"]) == EXIT_OK
    out = _out(capsys)
    for name in serving.UNIT_NAMES:
        assert f"# ---- {name} ----" in out
    assert "ExecStart=%h/.mesa/clm/serve/.venv/bin/clm-serve" in out
    assert main(["serve", "units", "--home", "relative/path"]) == EXIT_CONFIG
    assert main(["serve", "lock", "--check", "--home", str(tmp_path / "absent")]) == EXIT_OK
    out = _out(capsys)
    assert out.startswith("[ok] lock: ") and "[skip] serve clone" in out
    assert (
        main(["serve", "lock", "--check", "--home", str(tmp_path / "absent"), "--require-live"])
        == EXIT_FAIL
    )
    bad = tmp_path / "serving.lock.json"
    data = json.loads((ROOT / "serving" / "serving.lock.json").read_text(encoding="utf-8"))
    data["clm_commit"] = "f" * 40
    bad.write_text(json.dumps(data), encoding="utf-8")
    assert main(["serve", "lock", "--check", "--lock", str(bad)]) == EXIT_FAIL
    assert "[FAIL] lock" in _out(capsys)


def test_doctor_serve_flag(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mesa_clm import health

    seen: list[Any] = []
    real = health.doctor

    def spy(cfg: Any, **kw: Any) -> Any:
        seen.append(kw)
        return real(cfg, **kw)

    monkeypatch.setattr(health, "doctor", spy)
    assert main(["doctor", "--quick"]) == EXIT_OK
    assert main(["doctor", "--quick", "--serve"]) == EXIT_FAIL  # offline: unreachable fails
    out = _out(capsys)
    assert "[FAIL] encoder health" in out
    assert [k["serve"] for k in seen] == [None, True]
