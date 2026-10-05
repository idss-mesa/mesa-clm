"""The M4 bench verbs of the console script (``bench x3``, ``bench k2``, ``bench x4``, ``bench
e2e``, ``bench table`` over the new files, ``labels ingest-teacher``; design/m4-analysis-plan.md)
on the **synthetic** world of ``test_bench_m2_cli`` (the committed snapshot's identity and state
columns with generated labels, the fake encoder, a random head, a stand-in registration) plus a
synthetic teacher corpus. The grid is a small stand-in pinned by the stand-in M4 registration
(two specs per shape, ``ridge`` with one λ: the raw specs are 4096-d and a Newton logreg on them
is what made this file slow), so the runs are registered and fast; the stand-in registration
leaves every teacher pin UNPINNED, so nothing here reaches the committed X4 snapshots. No silver
label and no model output on a bench item is involved.

Checked: ``bench x3`` writes the four probe cells of every task registered and refuses another
snapshot and a second run; ``bench k2`` needs ``x3.json``, reads the pinned ``tiers.json`` and
``x2.json`` (refusing other bytes), writes ``k2.json`` and ``k2.md``; ``bench table`` lists the
X3 cells and the K2 verdicts; ``bench e2e`` says it is second-wave work; ``labels
ingest-teacher`` writes the teacher rows through the replayed OLS layer; ``bench x4`` runs the
three arms registered when the teacher and silver-minus-Opus snapshots are the pinned ones,
unregistered without the pins, and refuses a teacher snapshot whose silver rows differ."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pytest

from mesa_clm import framings, serving
from mesa_clm.bench import registered as reg
from mesa_clm.bench import run as m2run
from mesa_clm.bench import x2 as x2_module
from mesa_clm.bench.cells import Arm, X1Selection, run_tier_cells
from mesa_clm.bench.k2 import load_k2
from mesa_clm.bench.results import load_results, snapshot_content_sha256, write_results
from mesa_clm.cli import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, main
from mesa_clm.clm.fake import FakeEncoder
from mesa_clm.learn import features as feat
from mesa_clm.learn import probe
from mesa_clm.learn.labels import snapshot
from mesa_clm.learn.probe import Grid
from mesa_clm.ols import RecordingOLS
from mesa_clm.provenance.labels import TABLE, LabelRow, LabelStore
from mesa_clm.providers import live
from tests.unit.test_bench_m2_cli import (  # noqa: F401 - the fixture is used by name
    _ENV_PREFIXES,
    DATE,
    ROOT,
    _quick_pipeline,
    world,
)
from tests.unit.test_teacher import _card, _item, _Resolver, _validated

SMALL_RANK = Grid(("lowdim.v1", "pair4096.v1"), ("ridge",), {"ridge": ({"lambda": 1.0},)})
SMALL_CHOICE = Grid(("choice.state.v1", "choice.raw.v1"), ("ridge",), {"ridge": ({"lambda": 1.0},)})
SMALL_GRID: dict[str, Any] = {
    **dict(reg.X3_GRID),
    "specs": {"rank_fit": SMALL_RANK.specs, "choice": SMALL_CHOICE.specs},
    "fitters": ("ridge",),
    "hypers": {"ridge": (1.0,)},
}
UNPINNED_FIELDS = (
    "teacher_corpus_sha256",
    "neon_ducklake_commit",
    "teacher_snapshot",
    "teacher_labels_sha256",
    "teacher_labels_content_sha256",
    "minus_opus_snapshot",
    "minus_opus_labels_sha256",
    "minus_opus_labels_content_sha256",
)


def _unpinned_m4() -> reg.RegistrationM4:
    """The stand-in M4 registration: this checkout's framings lock, the small grid, and every
    teacher pin UNPINNED (the pinned branches set their own pins)."""
    return reg.RegistrationM4(
        framings_lock_sha=framings.lock_sha(),
        grid=SMALL_GRID,
        **dict.fromkeys(UNPINNED_FIELDS, reg.UNPINNED),
    )


@pytest.fixture
def m4(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:  # noqa: F811
    """The synthetic world in force with a stand-in M4 registration; returns the results root."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_FEATURES__DIR", str(world["root"] / "features"))
    monkeypatch.setattr(serving, "DEFAULT_HOME", str(world["home"]))
    monkeypatch.setattr(reg, "REGISTERED", world["registration"])
    monkeypatch.setattr(reg, "REGISTERED_M4", _unpinned_m4())
    monkeypatch.setattr(probe, "DEFAULT_GRID", {"rank_fit": SMALL_RANK, "choice": SMALL_CHOICE})
    monkeypatch.setattr(x2_module, "replica_pipeline", _quick_pipeline)
    assert live.live_lock_path() == ROOT / "serving" / "serving.lock.json"
    return tmp_path / "results"


def _committed_m2(out: Path) -> tuple[Path, Path]:
    """``tiers.json`` and ``x2.json`` of the synthetic world, written by the library (X1's
    outcome stood in by a fixed per-fold arm), as the committed M2 inputs K2 reads."""
    from mesa_clm.bench.cells import TextIndex
    from mesa_clm.config import load_config

    with m2run.scratch() as tmp:
        inputs = m2run.load_inputs(None, tmp)
        sv = m2run.serving(load_config(None))
        index = TextIndex.from_manifests(
            feat.manifest(inputs.snapshot, feat.X1_TASKS, "F4,F7,F9"),
            feat.manifest(inputs.snapshot, feat.CHOICE_TASKS, "F7"),
        )
        selections = {}
        for task_id in feat.X1_TASKS:
            arm = Arm(framings.ACTIVE[task_id], "clm-latest")
            cards = sorted(
                set(
                    inputs.tasks[
                        {
                            "term.fits": "neon_term_fits",
                            "column.ontology_fits": "neon_ontology_fits",
                        }[task_id]
                    ].cards
                )
            )
            selections[task_id] = X1Selection.of(arm, {c: arm for c in cards})
        tiers = run_tier_cells(
            inputs.tasks, index, sv.store, sv.require_scorers(), sv.fingerprints,
            selections=selections, labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256, date=DATE, registered=True,
            B=reg.current().B, seed=reg.current().seed,
        )  # fmt: skip
        tiers_path = write_results(tiers, out)
        dump = x2_module.load_anyjev(reg.current().anyjev_path())
        x2 = x2_module.run_x2(
            inputs.tasks, labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256, date=DATE,
            index=TextIndex.from_manifests(feat.manifest(inputs.snapshot, feat.X1_TASKS, "X2")),
            store=sv.store, fingerprint=sv.lock.fingerprint("clm-latest").as_dict(), anyjev=dump,
            registered=True, B=reg.current().B, seed=reg.current().seed,
        )  # fmt: skip
        x2_path = write_results(x2, out)
    return tiers_path, x2_path


def test_x3_k2_table_and_e2e(
    m4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for verb in ("x3", "k2", "x4", "e2e"):
        with pytest.raises(SystemExit) as exc:
            main(["bench", verb, "--help"])
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--B" not in out and "--grid" not in out and "--teacher-snapshot" in out
    # k2 before x3: usage
    assert main(["bench", "k2", "--date", DATE, "--out-dir", str(m4)]) == EXIT_CONFIG
    assert "no X3 results file" in capsys.readouterr().err
    # x3: registered, four cells per task
    assert main(["bench", "x3", "--date", DATE, "--out-dir", str(m4)]) == EXIT_OK
    printed = capsys.readouterr().out
    x3_path = m4 / DATE / "x3.json"
    assert printed.splitlines()[0] == str(x3_path) and (m4 / DATE / "x3.md").is_file()
    x3 = load_results(x3_path)
    assert x3.labels_sha256 == reg.current().labels_sha256 and not x3.notes[0].startswith("NOT")
    tasks = {c.task for c in x3.cells.values()}
    assert tasks == {
        "neon_term_fits",
        "neon_ontology_fits",
        "neon_annotate",
        "neon_aspect",
        "neon_value_kind",
    }
    for name, framing in (
        ("neon_term_fits", "F7"),
        ("neon_ontology_fits", "F9"),
        ("neon_aspect", "F7"),
    ):
        cell = x3.cells[f"{name}.probe.{framing}"]
        assert cell.selection == "nested" and cell.pre_registered and not cell.exploratory
        assert (
            cell.framing == framing
            and cell.question_key == framings.framing(cell.task_id, framing).question_key
        )
        assert (
            f"{name}.probe.{framing}@latest" in x3.cells
            and f"{name}.probe.{framing}@raw" in x3.cells
        )
    onto = x3.cells["neon_ontology_fits.probe.F9"]
    assert onto.metrics is not None and onto.items and onto.feature_spec in SMALL_RANK.specs
    assert onto.fingerprint == reg.current().fingerprints[onto.model or ""]
    full = x3.cells.get("neon_ontology_fits.probe.F9@full")
    assert full is not None and full.selection == "full" and full.exploratory
    # one run per file; another snapshot refused before anything runs
    assert main(["bench", "x3", "--date", DATE, "--out-dir", str(m4)]) == EXIT_FAIL
    assert "pass --force" in capsys.readouterr().err
    other = tmp_path / "other.parquet"
    other.write_bytes(b"not the snapshot")
    assert (
        main(
            [
                "bench",
                "x3",
                "--date",
                DATE,
                "--out-dir",
                str(tmp_path / "o"),
                "--snapshot",
                str(other),
            ]
        )
        == EXIT_FAIL
    )
    assert "not the pre-registered snapshot" in capsys.readouterr().err
    # k2 over the committed M2 inputs: the pins must name the files' bytes
    tiers_path, x2_path = _committed_m2(tmp_path / "m2")
    pins = {
        "tiers_results": str(tiers_path),
        "tiers_sha256": hashlib.sha256(tiers_path.read_bytes()).hexdigest(),
        "x2_results": str(x2_path),
        "x2_sha256": hashlib.sha256(x2_path.read_bytes()).hexdigest(),
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            reg, "REGISTERED_M4", replace(reg.current_m4(), **{**pins, "x2_sha256": "e" * 64})
        )
        assert main(["bench", "k2", "--date", DATE, "--out-dir", str(m4)]) == EXIT_FAIL
    assert (
        "not the pinned x2.json" in capsys.readouterr().err and not (m4 / DATE / "k2.json").exists()
    )
    monkeypatch.setattr(reg, "REGISTERED_M4", replace(reg.current_m4(), **pins))
    assert main(["bench", "k2", "--date", DATE, "--out-dir", str(m4)]) == EXIT_OK
    printed = capsys.readouterr().out
    k2 = load_k2(m4 / DATE / "k2.json")
    assert printed.splitlines()[:2] == [str(m4 / DATE / "k2.json"), "registered"]
    assert k2.registered and set(k2.tasks) == tasks and (m4 / DATE / "k2.md").is_file()
    assert (
        k2.inputs["tiers"]["sha256"] == pins["tiers_sha256"]
        and k2.inputs["x3"]["path"] == x3_path.as_posix()
    )
    for name in ("neon_annotate", "neon_aspect", "neon_value_kind"):
        assert k2.tasks[name].verdict == "c" and k2.tasks[name].head_adds_nothing is None
    t = k2.tasks["neon_ontology_fits"]
    assert [c.tier for c in t.candidates] == ["calibrated", "probe"] and t.best in (
        "calibrated",
        "probe",
    )
    assert t.conditions["non_inferior_acc"].numbers.get("join", {}).get("n_paired") == onto.counts.n
    assert t.head_adds_nothing is not None and t.head_adds_nothing.evaluated
    assert main(["bench", "k2", "--date", DATE, "--out-dir", str(m4)]) == EXIT_FAIL  # exists
    capsys.readouterr()
    # the table lists the X3 cells and the K2 verdicts beside the M2 files
    assert main(["bench", "table", "--date", DATE, "--out-dir", str(m4)]) == EXIT_OK
    table = capsys.readouterr().out
    assert f"{x3_path}#neon_ontology_fits.probe.F9 |" in table
    assert (
        f"{m4 / DATE / 'k2.json'}#neon_ontology_fits | {t.best} |" in table
        and "no cells" not in table
    )
    # e2e: --loco required; the implementation is second-wave work
    assert main(["bench", "e2e", "--date", DATE, "--out-dir", str(m4)]) == EXIT_CONFIG
    assert "--loco" in capsys.readouterr().err
    assert main(["bench", "e2e", "--loco", "--date", DATE, "--out-dir", str(m4)]) == EXIT_CONFIG
    assert "second-wave" in capsys.readouterr().err


def _teacher_corpus(root: Path, product: str, table: str, columns: list[str]) -> None:
    """A synthetic corpus for one product whose card has ``columns``: one in-registry item per
    column (ENVO, measurement) and one dropped item."""
    _card(root, product, table, columns)
    _validated(
        root,
        product,
        [_item(f"ENVO:{i:07d}", "measured_property", c) for i, c in enumerate(columns, 1)]
        + [_item("STATO:0000001", "measured_property", columns[0])],
    )


def test_ingest_teacher_and_x4(
    m4: Path,
    world: dict[str, Any],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # -- labels ingest-teacher on a synthetic corpus, the OLS layer replaying recorded fixtures --
    neon = tmp_path / "neon"
    _teacher_corpus(neon, "DP9.00001.001", "tA", ["windSpeedMean", "temp"])
    _teacher_corpus(neon, "DP9.00002.001", "tB", ["depth"])
    fixtures = tmp_path / "ols-fixtures"
    recorder = RecordingOLS(_Resolver(), fixtures, "record")  # type: ignore[arg-type]
    for curie in ("ENVO:0000001", "ENVO:0000002"):
        recorder.get_term("envo", f"http://purl.obolibrary.org/obo/{curie.replace(':', '_')}")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(fixtures))
    dsn = f"duckdb:///{tmp_path / 'labels.duckdb'}"
    assert main(["--provenance", dsn, "labels", "ingest-teacher"]) == EXIT_CONFIG
    assert "--neon-root" in capsys.readouterr().err
    assert (
        main(["--provenance", dsn, "labels", "ingest-teacher", "--neon-root", str(neon)]) == EXIT_OK
    )
    out = capsys.readouterr()
    summary = json.loads(out.out)
    # ENVO:0000001 twice (both products' first column), ENVO:0000002 once, both replayed from
    # the recorded fixtures; the STATO items are out of the registry
    assert summary["dropped"] == {"out_of_registry": 2} and summary["terms_missing"] == []
    assert summary["per_task"] == {
        "column.ontology_fits": {"teacher": 3},
        "term.fits": {"teacher": 3},
    }
    assert out.err == ""
    store = LabelStore(tmp_path / "labels.duckdb")
    assert store.count() == 6 and all(not r["fold_eligible"] for r in store.labels_for("term.fits"))
    # -- a teacher snapshot: the registered (synthetic) silver rows plus teacher rows whose
    # states are bench targets re-homed on another product, and their texts in the store ------
    con = duckdb.connect()
    try:
        silver = con.execute(
            "SELECT * FROM read_parquet(?) WHERE task_id = 'term.fits' ORDER BY target_sha256 LIMIT 12",
            [str(world["snapshot"])],
        ).fetchall()
        names = [d[0] for d in con.description or []]
    finally:
        con.close()
    teacher_rows: list[LabelRow] = []
    for values in silver:
        row = dict(zip(names, values, strict=True))
        state = (
            json.loads(row["state_json"])
            if isinstance(row["state_json"], str)
            else dict(row["state_json"])
        )
        state["card"] = {**state["card"], "dataset": "DP9.00001.001.teach"}
        from mesa_clm.identity import identity

        ident = identity("term.fits", state)
        teacher_rows.append(
            LabelRow(
                task_id="term.fits",
                task_key=ident.task_key,
                target_sha256=ident.target_sha256,
                option_key=ident.option_key,
                label_source="teacher",
                label="Yes",
                label_index=0,
                weight=0.5,
                state_sha256=row["state_sha256"],
                state_json=state,
                card="DP9.00001.001.teach",
                product_code="DP9.00001.001",
                leak_group="DP9.00001.001",
                fold_eligible=False,
                bench_card=False,
                origin="teacher:000000000000",
                actor="t",
            )
        )
    tstore = LabelStore(tmp_path / "teacher.duckdb")
    tstore.ensure_schema()
    cols = ", ".join(name for name, _ in tstore.columns())
    with tstore.connect() as c:
        c.execute(
            f"INSERT INTO {TABLE} ({cols}) SELECT {cols} FROM read_parquet(?)",  # noqa: S608
            [str(world["snapshot"])],
        )
    tstore.insert_labels(teacher_rows)
    teacher_snapshot = tmp_path / "teacher.parquet"
    teacher_sha = snapshot(tstore, teacher_snapshot)
    man = feat.manifest(teacher_snapshot, feat.X1_TASKS, m2run.M4_FRAMINGS)
    fstore = feat.FeatureStore.for_lock(world["root"] / "features", live_lock())
    missing = fstore.missing(man.texts())
    if missing:
        vecs, _ = FakeEncoder().embed(missing)
        fstore.add(missing, vecs, [max(1, len(t.split())) for t in missing])
    # -- the silver-minus-Opus snapshot: the registered rows minus a few, two labels flipped ---
    minus_store = LabelStore(tmp_path / "minus.duckdb")
    minus_store.ensure_schema()
    with minus_store.connect() as c:
        c.execute(
            f"INSERT INTO {TABLE} ({cols}) SELECT {cols} FROM read_parquet(?)",  # noqa: S608
            [str(world["snapshot"])],
        )
        first = "(SELECT target_sha256 FROM @T WHERE task_id = ? ORDER BY target_sha256 LIMIT ?)"
        first = first.replace("@T", TABLE)
        c.execute(
            f"DELETE FROM {TABLE} WHERE task_id = ? AND target_sha256 IN {first}",  # noqa: S608
            ["term.fits", "term.fits", 3],
        )
        c.execute(
            f"UPDATE {TABLE} SET label_index = 1 - label_index, "  # noqa: S608
            "label = CASE label WHEN 'Yes' THEN 'No' ELSE 'Yes' END "
            f"WHERE task_id = ? AND target_sha256 IN {first}",
            ["column.ontology_fits", "column.ontology_fits", 2],
        )
    minus_snapshot = tmp_path / "minus.parquet"
    minus_sha = snapshot(minus_store, minus_snapshot)
    # -- x4 without the pins: unregistered, named; with the pins: registered ------------------
    monkeypatch.setattr(reg, "REGISTERED_M4", _unpinned_m4())  # every pin UNPINNED, explicitly
    run = [
        "bench",
        "x4",
        "--date",
        DATE,
        "--out-dir",
        str(m4),
        "--teacher-snapshot",
        str(teacher_snapshot),
        "--minus-opus-snapshot",
        str(minus_snapshot),
    ]
    assert main(run) == EXIT_OK
    printed = capsys.readouterr().out
    x4 = load_results(m4 / DATE / "x4.json")
    assert (
        x4.notes[0].startswith("NOT the pre-registered run")
        and "teacher snapshot is not pinned" in x4.notes[0]
    )
    assert all(not c.pre_registered for c in x4.cells.values()) and ": keep=" in printed
    pins = {
        "teacher_snapshot": str(teacher_snapshot), "teacher_labels_sha256": teacher_sha,
        "teacher_labels_content_sha256": snapshot_content_sha256(teacher_snapshot),
        "minus_opus_snapshot": str(minus_snapshot), "minus_opus_labels_sha256": minus_sha,
        "minus_opus_labels_content_sha256": snapshot_content_sha256(minus_snapshot),
        "teacher_corpus_sha256": "c" * 64,
    }  # fmt: skip
    monkeypatch.setattr(reg, "REGISTERED_M4", replace(reg.current_m4(), **pins))
    assert main([*run, "--force"]) == EXIT_OK
    capsys.readouterr()
    x4 = load_results(m4 / DATE / "x4.json")
    assert not x4.notes[0].startswith("NOT") and all(
        c.pre_registered and c.exploratory for c in x4.cells.values()
    )
    keys = set(x4.cells)
    for task, framing in (("neon_term_fits", "F7"), ("neon_ontology_fits", "F9")):
        for arm in ("off", "0.5", "0.3"):
            assert f"{task}.probe.{framing}@teacher-{arm}" in keys
            assert f"{task}.probe.{framing}@teacher-{arm}-minus-opus" in keys
    on = x4.cells["neon_term_fits.probe.F7@teacher-0.5"]
    off = x4.cells["neon_term_fits.probe.F7@teacher-off"]
    assert on.teacher and not off.teacher and on.diagnostics["teacher_rows_available"] == 12
    assert all(not c.teacher_in_test for c in x4.cells.values())
    assert on.counts.n == off.counts.n and on.label_sources == off.label_sources
    minus_cell = x4.cells["neon_term_fits.probe.F7@teacher-off-minus-opus"]
    assert (
        minus_cell.counts.n < off.counts.n
        and minus_cell.diagnostics["scoring"] == "silver-minus-opus"
    )
    assert any(
        "teacher snapshot" in n and "non-teacher rows equal the registered snapshot" in n
        for n in x4.notes
    )
    # the pinned teacher snapshot is checked by bytes; other silver rows are refused outright
    monkeypatch.setattr(
        reg, "REGISTERED_M4", replace(reg.current_m4(), teacher_labels_sha256="f" * 64)
    )
    assert main([*run, "--force"]) == EXIT_FAIL
    assert "not the pinned teacher snapshot" in capsys.readouterr().err
    monkeypatch.setattr(
        reg, "REGISTERED_M4", replace(reg.current_m4(), teacher_labels_sha256=minus_sha)
    )
    assert (
        main(
            [
                "bench",
                "x4",
                "--date",
                DATE,
                "--out-dir",
                str(m4),
                "--teacher-snapshot",
                str(minus_snapshot),
                "--force",
            ]
        )
        == EXIT_FAIL
    )
    assert "non-teacher rows" in capsys.readouterr().err
    # no pin and no option: a usage error
    monkeypatch.setattr(reg, "REGISTERED_M4", _unpinned_m4())
    assert main(["bench", "x4", "--date", DATE, "--out-dir", str(tmp_path / "none")]) == EXIT_CONFIG
    assert "no teacher snapshot" in capsys.readouterr().err
    assert main(["bench", "table", "--date", DATE, "--out-dir", str(m4)]) == EXIT_OK
    assert "x4.json#neon_term_fits.probe.F7@teacher-0.5 |" in capsys.readouterr().out


def live_lock() -> Any:
    from mesa_clm.clm.fingerprint import load_serving_lock

    return load_serving_lock(ROOT / "serving" / "serving.lock.json")
