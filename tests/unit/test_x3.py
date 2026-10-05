"""X3 (``mesa_clm.bench.x3``; plan §5.3, §5.6 X3; the M4 analysis plan R1) on synthetic worlds:
the feature-store worlds of ``test_cells`` (planted vectors, generated labels) and the planted
feature sources of ``test_probe``. Checked: the four cells of a task with every frozen cell
field, ``items``, ``threshold_cp``, the lookup controls and ``beats_lookup_novel``; the cell's
identity is the full-data choice and ``fold_agreement_with_full`` counts the folds that chose
it; ``@latest`` / ``@raw`` restrict the grid to one model's specs; ``@full`` fixes the
configuration; a planted signal passes ``beats_lookup_novel``; the registration (another grid,
other labels, the task set) stamps every cell exploratory; a stand-in registration makes the
run registered. No silver label meets a model output here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings
from mesa_clm.bench import registered as reg
from mesa_clm.bench import x3 as X
from mesa_clm.bench.cells import FULL_VARIANT, CellError
from mesa_clm.bench.results import load_results, write_results
from mesa_clm.learn.offline import LATEST_MODEL, RAW_MODEL
from mesa_clm.learn.probe import DEFAULT_GRID, Grid
from tests.unit.test_cells import FINGERPRINTS, ZERO, choice_world, rank_world
from tests.unit.test_probe import FakeSource, _with_joint_rows, planted_source, rank_task

B = 200
IDENTITY: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64}
# A reduced grid (two specs; logreg with one λ, lda with one γ) over the small-vector worlds of
# test_cells: the registered grid itself is held to learn.probe's default by the registration
# and deviation tests below, on a tiny world.
SMALL_HYPERS: dict[str, tuple[dict[str, float], ...]] = {
    "logreg": ({"lambda": 1.0},),
    "lda": ({"gamma": 0.5},),
}
SMALL_RANK = Grid(("lowdim.v1", "pair4096.v1"), ("logreg", "lda"), SMALL_HYPERS)
SMALL_CHOICE = Grid(("choice.state.v1", "choice.raw.v1"), ("logreg", "lda"), SMALL_HYPERS)


def _scorers(store: Any) -> dict[str, Any]:
    from tests.unit.test_cells import _scorers as make

    return make(store)


def _standin(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grid_rank: Grid = SMALL_RANK,
    grid_choice: Grid = SMALL_CHOICE,
) -> None:
    """A registration whose labels, B, framings lock, fingerprints and grid are the test's."""
    monkeypatch.setattr(
        reg,
        "REGISTERED",
        reg.Registration(
            snapshot="s",
            labels_sha256=IDENTITY["labels_sha256"],
            labels_content_sha256=IDENTITY["labels_content_sha256"],
            published="p",
            B=B,
            framings_lock_sha=framings.lock_sha(),
            fingerprints={m: dict(fp) for m, fp in FINGERPRINTS.items()},
        ),
    )
    grid = {
        **dict(reg.X3_GRID),
        "specs": {"rank_fit": grid_rank.specs, "choice": grid_choice.specs},
        "fitters": grid_rank.fitters,
        "hypers": {
            f: tuple(next(iter(h.values())) for h in grid_rank.hypers[f]) for f in grid_rank.fitters
        },
    }
    monkeypatch.setattr(
        reg, "REGISTERED_M4", reg.RegistrationM4(framings_lock_sha=framings.lock_sha(), grid=grid)
    )


def test_the_four_cells_of_a_rank_fit_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _standin(monkeypatch)
    w = rank_world(tmp_path, framing_ids=("F7",), n_targets=14, seed=2)
    _with_joint_rows(w, np.random.default_rng(5))
    fb = X.feature_builder(w.task, w.index, w.store, _scorers(w.store))
    cells = X.task_probe_cells(
        w.task,
        fb,
        FINGERPRINTS,
        registered=True,
        grid=SMALL_RANK,
        index=w.index,
        store=w.store,
        B=B,
        clock=ZERO,
        **IDENTITY,
    )
    keys = {
        None: "neon_term_fits.probe.F7",
        "latest": "neon_term_fits.probe.F7@latest",
        "raw": "neon_term_fits.probe.F7@raw",
        FULL_VARIANT: "neon_term_fits.probe.F7@full",
    }
    assert set(cells) == set(keys.values())
    nested = cells[keys[None]]
    assert (
        nested.tier == "probe"
        and nested.selection == "nested"
        and nested.pre_registered
        and not nested.exploratory
    )
    assert (
        nested.framing == "F7"
        and nested.question_key == framings.framing("term.fits", "F7").question_key
    )
    assert nested.loco and nested.servable and not nested.teacher and not nested.teacher_in_test
    assert nested.feature_spec in SMALL_RANK.specs and nested.model == X.spec_model(
        nested.feature_spec
    )
    assert nested.fingerprint == FINGERPRINTS[nested.model]
    assert (
        nested.n_folds == 7 and nested.skipped_folds == {} and nested.counts.n == len(w.task.items)
    )
    assert nested.metrics is not None and nested.baselines is not None and nested.items is not None
    assert (
        nested.metrics.threshold_cp.keys() == {"0.05", "0.10"} and nested.metrics.auroc is not None
    )
    assert (
        nested.baselines.beats_lookup_novel is not None
        and nested.baselines.novel_key_lookup is not None
    )
    assert len(nested.items) == len(w.task.items) and all(
        abs(sum(i.probs) - 1) < 1e-9 for i in nested.items
    )
    # fold_choices: every outer card, the configuration, the framing and question_key of the probe
    assert set(nested.fold_choices) == set(w.task.cards)
    for entry in nested.fold_choices.values():
        assert entry["evaluated"] and entry["decision"] == "configuration"
        assert entry["spec"] in SMALL_RANK.specs and entry["fitter"] in SMALL_RANK.fitters
        assert (
            entry["model"] == X.spec_model(entry["spec"])
            and entry["clm_model_fp"] == FINGERPRINTS[entry["model"]]["clm_model_fp"]
        )
        assert entry["framing"] == "F7" and entry["question_key"] == nested.question_key
        assert "calibrator" in entry and "inner_nll" in entry
    d = nested.diagnostics
    assert (
        d["full_selection"]["spec"] == nested.feature_spec
        and d["full_selection"]["clm_model_fp"] == nested.fingerprint["clm_model_fp"]
    )
    agree = sum(1 for e in nested.fold_choices.values() if e["spec"] == nested.feature_spec)
    assert d["fold_agreement_with_full"] == {"agree": agree, "folds": 7}
    assert d["calibration"] == "platt" and d["grid"]["specs"] == list(SMALL_RANK.specs)
    assert d["input_tokens"] > 0 and d["calls"] == 14 * 7 and d["timing_source"] == "offline_replay"
    assert "per_fold" in d and "fitters" in d and d["teacher"] is False
    # @latest and @raw: one model's specs only, identity = that restriction's full-data choice
    latest, raw = cells[keys["latest"]], cells[keys["raw"]]
    assert (
        latest.feature_spec == "lowdim.v1"
        and latest.model == LATEST_MODEL
        and latest.variant == "latest"
    )
    assert raw.feature_spec == "pair4096.v1" and raw.model == RAW_MODEL and raw.variant == "raw"
    assert all(e["spec"] == "lowdim.v1" for e in latest.fold_choices.values())
    assert all(e["spec"] == "pair4096.v1" for e in raw.fold_choices.values())
    assert latest.pre_registered and not latest.exploratory and latest.selection == "nested"
    # @full: the full-data configuration in every fold, exploratory
    full = cells[keys[FULL_VARIANT]]
    assert (
        full.selection == "full"
        and full.exploratory
        and full.pre_registered
        and full.variant == FULL_VARIANT
    )
    assert full.feature_spec == nested.feature_spec and all(
        e["spec"] == nested.feature_spec for e in full.fold_choices.values()
    )
    assert (
        full.diagnostics["selection"] == "fixed"
        and full.diagnostics["fixed"]["spec"] == nested.feature_spec
    )
    assert full.diagnostics["fold_agreement_with_full"] == {"agree": 7, "folds": 7}
    # the planted signal (the F7 scores drive the labels) beats the lookup on novel keys at @latest
    assert latest.baselines is not None and latest.baselines.beats_lookup_novel is True
    assert (
        latest.metrics is not None
        and latest.metrics.auroc is not None
        and latest.metrics.auroc > 0.7
    )


def test_closed_choice_cells_and_the_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _standin(monkeypatch)
    w = choice_world(tmp_path, "column.aspect", n_targets=30, seed=1, novel=True)
    fb = X.feature_builder(w.task, w.index, w.store, _scorers(w.store))
    cells = X.task_probe_cells(
        w.task, fb, FINGERPRINTS, registered=True, grid=SMALL_CHOICE, B=B, clock=ZERO, **IDENTITY
    )
    nested = cells["neon_aspect.probe.F7"]
    assert nested.framing == "F7" and nested.diagnostics["calibration"] == "temperature"
    assert (
        nested.metrics is not None and nested.metrics.auroc is None and nested.counts.n_neg is None
    )
    assert nested.baselines is not None and nested.baselines.beats_lookup_novel is False
    assert nested.baselines.beats_detail["reason"] == "auroc_not_applicable_k8"
    assert cells["neon_aspect.probe.F7@latest"].feature_spec == "choice.state.v1"
    assert cells["neon_aspect.probe.F7@raw"].feature_spec == "choice.raw.v1"
    assert len(cells) == 4
    # a grid of the other shape is refused by the probe
    with pytest.raises(Exception, match="rank_fit"):
        X.task_probe_cells(
            w.task, fb, FINGERPRINTS, registered=True, grid=SMALL_RANK, B=B, **IDENTITY
        )


def test_a_planted_source_recovers_its_spec_and_beats_the_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _standin(monkeypatch)
    task, s = rank_task(40, seed=4)
    fb = planted_source(s, signal="lowdim.v1")
    grid = Grid(
        ("lowdim.v1", "pair4096.v1"),
        ("logreg", "ridge"),
        {"logreg": ({"lambda": 0.1},), "ridge": ({"lambda": 1.0},)},
    )
    cell = X.probe_cell(
        task,
        fb,
        grid=grid,
        variant=None,
        fingerprints=FINGERPRINTS,
        selection="nested",
        pre_registered=True,
        exploratory=False,
        B=B,
        clock=ZERO,
        **IDENTITY,
    )
    assert cell.feature_spec == "lowdim.v1" and cell.model == LATEST_MODEL
    assert all(e["spec"] == "lowdim.v1" for e in cell.fold_choices.values())
    assert cell.diagnostics["fold_agreement_with_full"] == {"agree": 7, "folds": 7}
    assert cell.baselines is not None and cell.baselines.beats_lookup_novel is True
    assert cell.metrics is not None and cell.metrics.ece < 0.1
    # no signal anywhere: the cell is honest about it
    noise = FakeSource(
        {k: np.random.default_rng(9).normal(size=v.shape) for k, v in fb.mats.items()}
    )
    plain = X.probe_cell(
        task,
        noise,
        grid=grid,
        variant=None,
        fingerprints=FINGERPRINTS,
        selection="nested",
        pre_registered=True,
        exploratory=False,
        B=B,
        clock=ZERO,
        **IDENTITY,
    )
    assert plain.baselines is not None and plain.baselines.beats_lookup_novel is False


def test_run_x3_applies_the_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _standin(monkeypatch)
    w = rank_world(tmp_path, framing_ids=("F7",), n_targets=10, seed=6)
    _with_joint_rows(w, np.random.default_rng(1))
    scorers = _scorers(w.store)
    common: dict[str, Any] = {"date": "2026-10-05", "B": B, "clock": ZERO, **IDENTITY}
    grids = {"rank_fit": SMALL_RANK, "choice": SMALL_CHOICE}
    # the registration's task set is empty (a stand-in): the one task is the whole set
    res = X.run_x3(
        {w.task.name: w.task},
        w.index,
        w.store,
        scorers,
        FINGERPRINTS,
        registered=True,
        grids=grids,
        **common,
    )
    assert all(c.pre_registered for c in res.cells.values()) and len(res.cells) == 4
    assert res.name == "x3" and not res.notes[0].startswith("NOT")
    path = write_results(res, tmp_path / "out")
    assert load_results(path) == res
    # another grid than the registered one: unregistered, every cell exploratory, the grid named
    other = {
        "rank_fit": Grid(SMALL_RANK.specs, ("ridge",), {"ridge": ({"lambda": 1.0},)}),
        "choice": SMALL_CHOICE,
    }
    res2 = X.run_x3(
        {w.task.name: w.task},
        w.index,
        w.store,
        scorers,
        FINGERPRINTS,
        registered=True,
        grids=other,
        **common,
    )
    assert all(not c.pre_registered and c.exploratory for c in res2.cells.values())
    assert res2.notes[0].startswith("NOT the pre-registered run") and "fitters" in res2.notes[0]
    # a grid without one model's specs has no cell of that variant
    latest_only = {
        "rank_fit": DEFAULT_GRID["rank_fit"].restrict(LATEST_MODEL),
        "choice": SMALL_CHOICE,
    }
    res_l = X.run_x3(
        {w.task.name: w.task},
        w.index,
        w.store,
        scorers,
        FINGERPRINTS,
        registered=True,
        grids=latest_only,
        **common,
    )
    assert set(res_l.cells) == {
        "neon_term_fits.probe.F7",
        "neon_term_fits.probe.F7@latest",
        "neon_term_fits.probe.F7@full",
    }
    # other labels: unregistered too; a caller cannot register a run with deviations
    res3 = X.run_x3(
        {w.task.name: w.task},
        w.index,
        w.store,
        scorers,
        FINGERPRINTS,
        registered=True,
        grids=grids,
        **{**common, "labels_sha256": "e" * 64},
    )
    assert "labels_sha256" in res3.notes[0]
    with pytest.raises(CellError, match="deviations"):
        X.run_x3(
            {w.task.name: w.task},
            w.index,
            w.store,
            scorers,
            FINGERPRINTS,
            registered=True,
            deviations=["x"],
            grids=grids,
            **common,
        )
    # the registered grid is learn.probe's default grid, fitter by fitter
    monkeypatch.setattr(reg, "REGISTERED_M4", reg.RegistrationM4())
    for shape in ("rank_fit", "choice"):
        assert reg.grid_deviations(DEFAULT_GRID[shape], shape=shape) == []  # type: ignore[arg-type]
    assert reg.grid_deviations(SMALL_RANK, shape="rank_fit")
    # every registered spec's model is the one the registration pins
    for spec, model in reg.X3_GRID["spec_models"].items():
        assert X.spec_model(spec) == model
    with pytest.raises(CellError, match="unknown probe spec"):
        X.spec_model("nope")
