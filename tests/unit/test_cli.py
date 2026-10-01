"""The ``mesa-clm`` console script: parser shape, exit codes and the M0 verbs end to end on the
committed fixtures. Hermetic: OLS replayed from ``tests/fixtures/ols``, DuckDB under ``tmp_path``,
the cwd moved to ``tmp_path`` whenever a verb writes ``bench/`` paths, and the developer's
``MESA_CLM_*`` environment cleared first."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pytest

from mesa_clm import __version__
from mesa_clm.bench import mde, stats
from mesa_clm.cards import DatasetCard
from mesa_clm.cli import (
    DEFAULT_B,
    DEFAULT_N_SIMS,
    DEFAULT_SEED,
    EXIT_CONFIG,
    EXIT_FAIL,
    EXIT_OK,
    UsageError,
    build_parser,
    label_store,
    main,
)
from mesa_clm.config import load_config
from mesa_clm.identity import identity
from mesa_clm.learn.labels import TermResolver, ingest_neon_eval
from mesa_clm.ols import RecordingOLS
from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.states import candidate_state, state_sha256
from mesa_clm.tasks import TASKS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EVAL_ROOT = FIXTURES / "neon-avu-eval"
OLS_DIR = FIXTURES / "ols"
# ``ingest_neon_eval`` over the committed fixtures (pinned in test_labels.py): 1083 derived rows,
# 149 of which share an identity with an earlier row (D1) and are dropped before the insert as
# ``collapsed_identity`` (28 with another label: ``conflicts``), so 934 rows are submitted.
N_FIXTURE_LABELS = 934
N_COLLAPSED = 149
N_CONFLICTS = 28
CELLS = {
    "neon_annotate.baseline.lookup_prob",
    "neon_aspect.baseline.lookup_prob",
    "neon_ontology_fits.baseline.lookup_prob",
    "neon_term_fits.baseline.lookup_prob",
    "neon_value_kind.baseline.lookup_prob",
}
CAND = {
    "label": "distance",
    "curie": "PATO:0000040",
    "iri": "http://purl.obolibrary.org/obo/PATO_0000040",
    "ontology_id": "pato",
    "description": "d",
    "synonyms": [],
    "has_children": False,
}

_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(OLS_DIR))


@pytest.fixture
def dsn(tmp_path: Path) -> str:
    return f"duckdb:///{tmp_path / 'labels.duckdb'}"


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One ingestion of the fixtures per module (through the library); the CLI tests read it."""
    path = tmp_path_factory.mktemp("labels") / "labels.duckdb"
    report = ingest_neon_eval(
        LabelStore(path), EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay"))
    )
    assert report.inserted == N_FIXTURE_LABELS
    return path


@pytest.fixture
def store_dsn(store_path: Path) -> str:
    return f"duckdb:///{store_path}"


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


# -- parser and exit codes --------------------------------------------------------------------------


def test_help_lists_the_m0_verbs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = _out(capsys)
    assert "{labels,bench,doctor," in out and "--provenance" in out and "--actor" in out
    for verb, choices in (
        ("labels", "{ingest-neon-eval,import-anyjev,snapshot,stats}"),
        ("bench", "{baselines,mde}"),
    ):
        with pytest.raises(SystemExit) as sub:
            main([verb, "--help"])
        assert sub.value.code == 0 and choices in _out(capsys)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert _out(capsys).strip() == f"mesa-clm {__version__}"


def test_no_verb_and_unknown_verb_are_argparse_errors(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in ([], ["frobnicate"], ["labels"], ["bench", "run"]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2, argv
    assert "invalid choice" in capsys.readouterr().err


def test_bad_config_is_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], clean_env: None
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    assert main(["--config", str(bad), "doctor"]) == EXIT_CONFIG
    assert "config error" in capsys.readouterr().err
    assert main(["--config", str(tmp_path / "missing.yaml"), "doctor"]) == EXIT_CONFIG


@pytest.mark.parametrize(
    ("variable", "value", "field"),
    [
        # A key on a mistyped name: extra="forbid" rejects it; the value must stay out of stderr.
        ("MESA_CLM_CLM__API_KEYX", "hunter2-secret-value", "clm.api_keyx"),
        ("MESA_CLM_ENCODER__APIKEY", "hunter2-secret-value", "encoder.apikey"),
        ("MESA_CLM_PLANNER__GATEWAY_API_KEY_FILEX", "/run/hunter2-secret-value", "planner"),
        # A key that fails its type check inside a whole-section YAML value.
        ("MESA_CLM_CLM", "{api_key: 987654321}", "clm.api_key"),
        ("MESA_CLM_ENCODER", "{api_key: [hunter2-secret-value]}", "encoder.api_key"),
        # A whole section given as a scalar.
        ("MESA_CLM_PLANNER", "hunter2-secret-value", "planner"),
    ],
)
def test_config_validation_error_never_echoes_the_value(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variable: str,
    value: str,
    field: str,
) -> None:
    """A mistyped ``*_API_KEY`` name or a badly typed key value is a config error (exit 2) that
    names the field and the problem, never the input (``hide_input_in_errors``; CLAUDE.md: no
    secret is ever printed)."""
    monkeypatch.setenv(variable, value)
    assert main(["doctor", "--quick"]) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert err.startswith("config error: ") and field in err
    assert "hunter2" not in err and "987654321" not in err
    assert "input_value" not in err and "https://" not in err


def test_parser_defaults_mirror_the_bench_constants() -> None:
    assert (DEFAULT_B, DEFAULT_SEED, DEFAULT_N_SIMS) == (
        stats.DEFAULT_B,
        stats.DEFAULT_SEED,
        mde.DEFAULT_N_SIMS,
    )
    args = build_parser().parse_args(["bench", "mde"])
    assert (args.B, args.seed, args.n_sims, args.out_dir) == (2000, 0, 200, "bench/results")
    assert args.snapshot is None and args.date is None and args.verb == "mde"


def test_actor_default_follows_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "curator")
    assert build_parser().parse_args(["doctor"]).actor == "curator"
    monkeypatch.delenv("USER")
    assert build_parser().parse_args(["doctor"]).actor == "mesa-clm"


def test_global_options_before_and_after_the_verb(
    clean_env: None, dsn: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Global options go before the verb (plan §7.2); ``--provenance`` and ``--actor`` are also
    accepted after it, as mesa-anyjev's users type them, and mean the same thing."""
    assert main(["--provenance", dsn, "labels", "stats"]) == EXIT_OK
    before = json.loads(_out(capsys))
    assert main(["labels", "stats", "--provenance", dsn]) == EXIT_OK
    after = json.loads(_out(capsys))
    assert before == after == {"n_labels": 0, "tasks": {}}
    parser = build_parser()
    args = parser.parse_args(["--actor", "a", "--provenance", "x", "labels", "stats"])
    assert (args.actor, args.provenance) == ("a", "x")
    args = parser.parse_args(
        ["--actor", "a", "labels", "stats", "--actor", "b", "--provenance", "y"]
    )
    assert (args.actor, args.provenance) == ("b", "y")  # the later one wins
    args = parser.parse_args(["labels", "stats"])
    assert args.provenance is None


def test_postgres_dsn_is_a_usage_error_in_m0(
    clean_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--provenance", "postgresql://user:s3cret@db.example/clm", "labels", "stats"])
    assert code == EXIT_CONFIG
    err = capsys.readouterr().err
    assert err.startswith("mesa-clm: ") and "duckdb:///" in err and "s3cret" not in err


def test_label_store_follows_the_config(clean_env: None) -> None:
    cfg = load_config(env={})
    store = label_store(cfg)
    assert store.path == Path("~/.mesa/clm/provenance.duckdb").expanduser()
    # DESIGN D11: the sidecar flock is ~/.mesa/clm/locks/provenance.lock, the path the M1
    # sidecar store must share, not a private <file>.lock beside the DuckDB file.
    assert store.lock_path == Path("~/.mesa/clm/locks/provenance.lock").expanduser()
    # Three slashes then the path, as in mesa-anyjev: a fourth slash makes it absolute.
    custom = label_store(cfg, "duckdb:////tmp/x.duckdb")
    assert custom.path == Path("/tmp/x.duckdb")
    assert custom.lock_path == Path("/tmp/locks/provenance.lock")
    assert label_store(cfg, "duckdb:///rel/x.duckdb").path == Path("rel/x.duckdb")
    with pytest.raises(UsageError) as exc:
        label_store(cfg, "postgresql://h/db")
    assert exc.value.code == EXIT_CONFIG


# -- doctor ---------------------------------------------------------------------------------------


def test_doctor_is_green_and_honours_provenance(
    clean_env: None, dsn: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--provenance", dsn, "doctor"]) == EXIT_OK
    out = _out(capsys)
    lines = out.strip().splitlines()
    assert all(line.startswith(("[ok] ", "[WARN] ")) for line in lines) and "[FAIL]" not in out
    assert any(line.startswith("[ok] vendored files: 4 files match") for line in lines)
    assert any(
        line.startswith("[ok] provenance path: ") and dsn[len("duckdb:///") :] in line
        for line in lines
    )

    assert main(["--provenance", dsn, "doctor", "--quick", "--json"]) == EXIT_OK
    report = json.loads(_out(capsys))
    assert report["ok"] is True and report["summary"]["fail"] == 0
    names = [c["name"] for c in report["checks"]]
    assert "vendored files" not in names and "provenance path" in names and "labels store" in names


def test_doctor_failure_is_exit_1(
    clean_env: None,
    dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MESA_CLM_EVAL_ROOT", str(tmp_path / "gone"))
    assert main(["--provenance", dsn, "doctor", "--quick"]) == EXIT_FAIL
    assert "[FAIL] eval root" in _out(capsys)


# -- labels ---------------------------------------------------------------------------------------


def test_ingest_needs_an_eval_root(
    clean_env: None, dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["labels", "ingest-neon-eval", "--provenance", dsn]) == EXIT_CONFIG
    assert "--eval-root" in capsys.readouterr().err


def test_ingest_refuses_a_root_without_the_eval_layout(
    clean_env: None, dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing directory, or one without ``results/validated.json`` and ``cards/``, is a usage
    error (``mesa-clm: ...``, exit 2), not a ``FileNotFoundError`` traceback; nothing is opened."""
    for root in (tmp_path / "nonexistent", tmp_path):
        argv = ["labels", "ingest-neon-eval", "--eval-root", str(root), "--provenance", dsn]
        assert main(argv) == EXIT_CONFIG, root
        err = capsys.readouterr().err
        assert err.startswith(f"mesa-clm: {root}: not a neon-avu-eval checkout")
        assert "results/validated.json" in err and "cards/" in err and "Traceback" not in err
    (tmp_path / "cards").mkdir()
    argv = ["labels", "ingest-neon-eval", "--eval-root", str(tmp_path), "--provenance", dsn]
    assert main(argv) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "missing results/validated.json)" in err and "cards/" not in err.split("missing")[1]
    assert not Path(dsn[len("duckdb:///") :]).exists()


def test_ingest_is_idempotent_and_reports_the_gaz_gap(
    clean_env: None, store_dsn: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second ingestion into an already ingested store inserts nothing (INSERT OR IGNORE on the
    identity key), reports the within-batch identity collapses the same way as the first, and
    still reports the 18 (card, CURIE) pairs of the five GAZ root terms."""
    argv = ["labels", "ingest-neon-eval", "--eval-root", str(EVAL_ROOT), "--provenance", store_dsn]
    assert main(argv) == EXIT_OK
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert (summary["inserted"], summary["skipped_existing"]) == (0, N_FIXTURE_LABELS)
    assert summary["skipped"] == {"collapsed_identity": N_COLLAPSED}
    assert summary["conflicts"] == N_CONFLICTS
    assert sum(sum(c.values()) for c in summary["per_task"].values()) == N_FIXTURE_LABELS
    assert summary["per_task"]["avu.value_kind"] == {
        "consensus_majority": 219,
        "consensus_negative": 82,
    }
    assert summary["terms_resolved"] == 285 and summary["terms_missing"] == 18
    assert len(summary["terms_missing_curies"]) == 5
    assert all(c.startswith("GAZ:") for c in summary["terms_missing_curies"])
    assert captured.err.startswith("unresolved CURIEs (5): GAZ:")


def test_ingest_with_excluded_models_reads_eval_root_from_the_config(
    clean_env: None, dsn: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MESA_CLM_EVAL_ROOT", str(EVAL_ROOT))
    argv = [
        "labels",
        "ingest-neon-eval",
        "--provenance",
        dsn,
        "--exclude-models",
        "claude-opus-5-5, ",
    ]
    assert main(argv) == EXIT_OK
    summary = json.loads(capsys.readouterr().out)
    assert 0 < summary["inserted"] < N_FIXTURE_LABELS
    assert summary["source_ref"].endswith(" minus:claude-opus-5-5")


def test_stats_and_snapshot(
    clean_env: None, store_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["labels", "stats", "--provenance", store_dsn]) == EXIT_OK
    st = json.loads(_out(capsys))
    assert st["n_labels"] == N_FIXTURE_LABELS
    assert set(st["tasks"]) == {
        "term.fits",
        "column.annotate",
        "column.aspect",
        "column.ontology_fits",
        "avu.value_kind",
    }
    term = st["tasks"]["term.fits"]
    assert term["n"] == 285 and term["labels"] == {"No": 199, "Yes": 86}
    assert term["sources"] == {
        "consensus_all": 16,
        "consensus_majority": 70,
        "consensus_negative": 199,
    }
    assert term["weights"] == {"0.5": 199, "0.6": 70, "0.8": 16}

    snap = tmp_path / "snap" / "labels.parquet"
    assert main(["labels", "snapshot", "--out", str(snap), "--provenance", store_dsn]) == EXIT_OK
    line = _out(capsys).strip()
    assert line.startswith(f"wrote {snap} ({N_FIXTURE_LABELS} labels); labels_sha256 ")
    sha = line.rsplit(" ", 1)[1]
    assert snap.is_file() and sha == hashlib.sha256(snap.read_bytes()).hexdigest()
    con = duckdb.connect()
    try:
        got = con.execute("SELECT count(*) FROM read_parquet(?)", [str(snap)]).fetchone()
    finally:
        con.close()
    assert got is not None and got[0] == N_FIXTURE_LABELS


def test_snapshot_default_path_and_empty_store(
    clean_env: None,
    dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["labels", "snapshot", "--provenance", dsn]) == EXIT_FAIL
    assert "no labels" in capsys.readouterr().err
    assert not (tmp_path / "bench").exists()


def _anyjev_sidecar(path: Path, rows: list[dict[str, Any]]) -> None:
    """A mesa-anyjev ``mesa_anyjev.labels`` table (``provenance/store.py`` DDL at ``6159281``)."""
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS mesa_anyjev")
        con.execute(
            """CREATE TABLE IF NOT EXISTS mesa_anyjev.labels (
            label_id TEXT PRIMARY KEY, question_id TEXT NOT NULL, question_key TEXT NOT NULL,
            state_sha256 TEXT NOT NULL, state_json JSON NOT NULL, label_index INTEGER NOT NULL,
            label_source TEXT NOT NULL, weight REAL NOT NULL, source_ref TEXT, card TEXT,
            decision_id TEXT, batch_id TEXT, consumed_in_bundle TEXT, ts TIMESTAMPTZ NOT NULL,
            UNIQUE (question_key, state_sha256, label_source))"""
        )
        for r in rows:
            con.execute(
                "INSERT INTO mesa_anyjev.labels (label_id, question_id, question_key, state_sha256, "
                "state_json, label_index, label_source, weight, source_ref, card, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())",
                [
                    str(uuid4()),
                    r["question_id"],
                    TASKS[r["question_id"]].key,
                    state_sha256(r["state"]),
                    json.dumps(r["state"], sort_keys=True, separators=(",", ":")),
                    r["label_index"],
                    r["label_source"],
                    r["weight"],
                    "neon-avu-eval/results/validated.json@deadbeef0000",
                    r["state"]["card"]["dataset"],
                ],
            )
    finally:
        con.close()


def test_import_anyjev(
    clean_env: None, dsn: str, tmp_path: Path, card: DatasetCard, capsys: pytest.CaptureFixture[str]
) -> None:
    col = card.column("observerDistance")
    src = tmp_path / "anyjev.duckdb"
    _anyjev_sidecar(
        src,
        [
            {
                "question_id": "term.fits",
                "state": candidate_state(card, "column", col, "measurement", CAND, 5),
                "label_index": 0,
                "label_source": "curator",
                "weight": 1.0,
            },
            {  # the same pair at another group size collapses onto one identity (D1)
                "question_id": "term.fits",
                "state": candidate_state(card, "column", col, "measurement", CAND, 9),
                "label_index": 0,
                "label_source": "curator",
                "weight": 1.0,
            },
        ],
    )
    assert (
        main(["labels", "import-anyjev", "--dsn", f"duckdb:///{src}", "--provenance", dsn])
        == EXIT_OK
    )
    summary = json.loads(_out(capsys))
    assert summary["inserted"] == 1 and summary["per_task"] == {"term.fits": {"curator": 1}}
    assert summary["skipped"] == {"collapsed_identity": 1} and summary["conflicts"] == 0
    assert summary["source_ref"].startswith("anyjev-import:anyjev.duckdb@")
    # A plain path works too; a second import is idempotent.
    assert main(["labels", "import-anyjev", "--dsn", str(src), "--provenance", dsn]) == EXIT_OK
    assert json.loads(_out(capsys))["inserted"] == 0

    assert main(["labels", "import-anyjev", "--provenance", dsn]) == EXIT_CONFIG
    assert "--dsn" in capsys.readouterr().err
    missing = tmp_path / "missing.duckdb"
    assert (
        main(["labels", "import-anyjev", "--dsn", str(missing), "--provenance", dsn]) == EXIT_CONFIG
    )
    assert str(missing) in capsys.readouterr().err
    junk = tmp_path / "junk.duckdb"
    junk.write_bytes(b"not a database at all")
    assert main(["labels", "import-anyjev", "--dsn", str(junk), "--provenance", dsn]) == EXIT_FAIL
    assert "not a readable mesa-anyjev sidecar" in capsys.readouterr().err


# -- bench ----------------------------------------------------------------------------------------


def test_bench_baselines_and_mde_stamp_the_snapshot(
    clean_env: None, store_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The M0 bench sequence (plan §9): snapshot -> baselines -> mde, every results file stamped
    with the sha256 of the frozen snapshot it was computed from (DESIGN D30)."""
    snap = tmp_path / "labels.parquet"
    assert main(["labels", "snapshot", "--out", str(snap), "--provenance", store_dsn]) == EXIT_OK
    sha = _out(capsys).strip().rsplit(" ", 1)[1]
    out_dir = tmp_path / "results"
    common = ["--provenance", store_dsn, "--snapshot", str(snap), "--out-dir", str(out_dir)]

    argv = ["bench", "baselines", *common, "--date", "2026-01-01", "--B", "20"]
    assert main(argv) == EXIT_OK
    captured = capsys.readouterr()
    path = out_dir / "2026-01-01" / "baselines.json"
    assert captured.out.splitlines()[0] == str(path) and "| cell |" in captured.out
    assert f"labels snapshot {snap}; labels_sha256 {sha}" in captured.err
    results = json.loads(path.read_text(encoding="utf-8"))
    assert results["labels_sha256"] == sha and set(results["cells"]) == CELLS
    assert results["date"] == "2026-01-01" and results["name"] == "baselines"
    assert path.with_suffix(".md").is_file()
    term = results["cells"]["neon_term_fits.baseline.lookup_prob"]
    assert term["counts"]["n"] == 285 and term["baselines"]["lookup_acc"] == pytest.approx(
        0.772, abs=5e-4
    )

    argv = ["bench", "mde", *common, "--date", "2026-01-01", "--B", "10", "--n-sims", "2"]
    assert main(argv) == EXIT_OK
    out = _out(capsys)
    mpath = out_dir / "2026-01-01" / "mde.json"
    assert out.splitlines()[0] == str(mpath)
    assert "neon_term_fits" in out and "8 classes" in out and "mde_auroc=" in out
    m = json.loads(mpath.read_text(encoding="utf-8"))
    assert m["labels_sha256"] == sha and (m["n_sims"], m["B"], m["seed"]) == (2, 10, 0)


def test_bench_writes_the_default_snapshot_then_reuses_it(
    clean_env: None,
    store_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        main(["bench", "baselines", "--provenance", store_dsn, "--date", "2026-01-02", "--B", "10"])
        == EXIT_OK
    )
    snap = tmp_path / "bench" / "snapshots" / "2026-01-02.parquet"
    assert snap.is_file()
    sha = hashlib.sha256(snap.read_bytes()).hexdigest()
    res = json.loads((tmp_path / "bench" / "results" / "2026-01-02" / "baselines.json").read_text())
    assert res["labels_sha256"] == sha
    capsys.readouterr()
    argv = [
        "bench",
        "mde",
        "--provenance",
        store_dsn,
        "--date",
        "2026-01-02",
        "--B",
        "10",
        "--n-sims",
        "1",
    ]
    assert main(argv) == EXIT_OK
    assert hashlib.sha256(snap.read_bytes()).hexdigest() == sha  # reused, not rewritten
    m = json.loads((tmp_path / "bench" / "results" / "2026-01-02" / "mde.json").read_text())
    assert m["labels_sha256"] == sha


def test_bench_refuses_an_empty_store_and_a_missing_snapshot(
    clean_env: None, dsn: str, store_dsn: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["bench", "baselines", "--provenance", dsn, "--out-dir", str(tmp_path)]) == EXIT_FAIL
    )
    assert "no labels" in capsys.readouterr().err
    argv = ["bench", "mde", "--provenance", store_dsn, "--snapshot", str(tmp_path / "nope.parquet")]
    assert main(argv) == EXIT_CONFIG
    assert "no such file" in capsys.readouterr().err


def test_bench_refuses_a_snapshot_the_store_has_outgrown(
    clean_env: None,
    store_path: Path,
    tmp_path: Path,
    card: DatasetCard,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """D30: an existing snapshot must still describe the store; a label added after the freeze
    changes the content hash and the run is refused with exit 1."""
    copy = tmp_path / "copy.duckdb"
    shutil.copy(store_path, copy)
    dsn = f"duckdb:///{copy}"
    snap = tmp_path / "frozen.parquet"
    assert main(["labels", "snapshot", "--out", str(snap), "--provenance", dsn]) == EXIT_OK
    capsys.readouterr()
    state = candidate_state(card, "column", card.column("observerDistance"), "measurement", CAND, 3)
    ident = identity("term.fits", state)
    LabelStore(copy).insert_labels(
        [
            LabelRow(
                task_id="term.fits",
                task_key=ident.task_key,
                target_sha256=ident.target_sha256,
                option_key=ident.option_key,
                label_source="curator",
                label="Yes",
                label_index=0,
                weight=1.0,
                state_sha256=state_sha256(state),
                state_json=state,
                card=card.name,
            )
        ]
    )
    argv = [
        "bench",
        "baselines",
        "--provenance",
        dsn,
        "--snapshot",
        str(snap),
        "--out-dir",
        str(tmp_path),
    ]
    assert main(argv) == EXIT_FAIL
    assert "changed since the snapshot" in capsys.readouterr().err
    assert not (tmp_path / "2026-01-01").exists()
