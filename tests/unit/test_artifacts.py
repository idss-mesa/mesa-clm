"""Learned artifacts on disk (``mesa_clm.artifacts``; plan §5.5; the M4 brief R4): the version
layout and manifest, the identity the manifest records of a cited cell equals the bench cell's,
immutability and owner-only modes, strict load refusals (tampering, another fingerprint,
another framings lock, a stale question key), ``CURRENT.json`` and ``bundle_from_current``,
promotion's four rules, ``learn fit``'s writer with an injected probe fitter, the export copy,
and the publish → pull round trip with a tampered pull refused. Synthetic only: a probe fitted on
random features, planted cells, a snapshot file that exists only to be hashed."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import artifacts as art
from mesa_clm import framings as fr
from mesa_clm.artifacts import (
    CALIBRATORS_FILE,
    CURRENT_FILE,
    MANIFEST_FILE,
    PROBES_DIR,
    ArtifactError,
    ArtifactLayout,
    CurrentEntry,
    ManifestEntry,
    bundle_from_current,
    cell_identity,
    entry_key,
    export_copy,
    fold_agreement,
    k2_verdict,
    load_manifest,
    load_version,
    promote,
    publish,
    pull,
    verify_files,
    write_version,
)
from mesa_clm.bench import stats
from mesa_clm.clm.fingerprint import EncoderSpec, Fingerprint, encoder_fp
from mesa_clm.providers.tiered import PlattCalibrator, TemperatureCalibrator, load_calibrator
from tests.fakes import m4
from tests.fakes.m4 import CONTENT_SHA, FP, LABELS_SHA, cell, planted_items, synthetic_probe

TERM = fr.active_framing("term.fits")
ASPECT = fr.active_framing("column.aspect")
ONTOLOGY = fr.active_framing("column.ontology_fits")
LOCK = fr.lock_sha()
DATE = "2026-10-05"


def _layout(root: Path, fp: Fingerprint = FP) -> ArtifactLayout:
    return ArtifactLayout.for_fingerprint(root / "artifacts", fp)


def _results(root: Path) -> tuple[str, str]:
    """x3.json with the probe cells and tiers.json with a calibrated cell under ``root``."""
    x3 = m4.results_file(
        root,
        DATE,
        "x3",
        [
            cell("term.fits", "probe", spec="lowdim.v1"),
            cell("column.aspect", "probe", spec="choice.state.v1"),
        ],
    )
    tiers = m4.results_file(root, DATE, "tiers", [cell("column.ontology_fits", "calibrated")])
    return x3, tiers


def _entries(x3: str, tiers: str, root: Path) -> tuple[dict[str, ManifestEntry], dict, dict]:
    """A calibrated entry (ontology_fits) and two probe entries (term.fits, aspect), cited."""
    probe = synthetic_probe("lowdim.v1")
    choice = synthetic_probe("choice.state.v1", task_id="column.aspect", seed=1)
    cal = PlattCalibrator(a=1.5, b=-0.25, n=190)
    x3_cells = art.load_results(root / x3).cells
    tier_cells = art.load_results(root / tiers).cells
    entries = {
        entry_key("calibrated", ONTOLOGY.question_key): ManifestEntry(
            question_key=ONTOLOGY.question_key,
            task_id="column.ontology_fits",
            framing_id=ONTOLOGY.id,
            tier="calibrated",
            file=CALIBRATORS_FILE,
            calibrator=cal.model_dump(mode="json"),
            cite=f"{tiers}#neon_ontology_fits.calibrated.{ONTOLOGY.id}",
            cell=cell_identity(
                tiers,
                tier_cells[f"neon_ontology_fits.calibrated.{ONTOLOGY.id}"],
                config={"framing": ONTOLOGY.id, "model": "clm-latest"},
            ),
        ),
        entry_key("probe", TERM.question_key): ManifestEntry(
            question_key=TERM.question_key,
            task_id="term.fits",
            framing_id=TERM.id,
            tier="probe",
            file=f"{PROBES_DIR}/{TERM.question_key}.json",
            spec="lowdim.v1",
            fitter="logreg",
            hyper={"lambda": 1.0},
            calibrator=probe.calibrator.model_dump(mode="json"),
            cite=f"{x3}#neon_term_fits.probe.{TERM.id}",
            cell=cell_identity(
                x3,
                x3_cells[f"neon_term_fits.probe.{TERM.id}"],
                config={"spec": "lowdim.v1", "model": "clm-latest"},
            ),
        ),
        entry_key("probe", ASPECT.question_key): ManifestEntry(
            question_key=ASPECT.question_key,
            task_id="column.aspect",
            framing_id=ASPECT.id,
            tier="probe",
            file=f"{PROBES_DIR}/{ASPECT.question_key}.json",
            spec="choice.state.v1",
            fitter="logreg",
            hyper={"lambda": 1.0},
            calibrator=choice.calibrator.model_dump(mode="json"),
            cite=f"{x3}#neon_aspect.probe.{ASPECT.id}",
            cell=cell_identity(
                x3,
                x3_cells[f"neon_aspect.probe.{ASPECT.id}"],
                config={"spec": "choice.state.v1", "model": "clm-latest"},
            ),
        ),
    }
    calibrators = {ONTOLOGY.question_key: cal}
    probes = {
        TERM.question_key: probe.model_dump(mode="json"),
        ASPECT.question_key: choice.model_dump(mode="json"),
    }
    return entries, calibrators, probes


def _write(root: Path, **over: Any) -> tuple[ArtifactLayout, int, Path, str, str]:
    x3, tiers = _results(root)
    entries, calibrators, probes = _entries(x3, tiers, root)
    layout = _layout(root)
    n, path = write_version(
        layout,
        entries=entries,
        calibrators=calibrators,
        probes=probes,
        labels_sha256=LABELS_SHA,
        labels_content_sha256=CONTENT_SHA,
        notes=["synthetic"],
        **over,
    )
    return layout, n, path, x3, tiers


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# -- writing and the manifest -------------------------------------------------------------------


def test_write_load_round_trip_and_the_manifest_records_the_cell(tmp_path: Path) -> None:
    layout, n, path, x3, _ = _write(tmp_path)
    assert n == 1 and path == layout.version_dir(1) and path.name == "v1"
    assert path.parent == layout.dir and layout.dir.name == LOCK[:8]
    manifest = load_manifest(path)
    assert manifest.version == "v1" and manifest.number == 1 and manifest.format == art.FORMAT
    assert (manifest.encoder_fp, manifest.clm_model_fp) == (FP.encoder_fp, FP.clm_model_fp)
    assert manifest.framings_lock_sha == LOCK and manifest.labels_sha256 == LABELS_SHA
    assert set(manifest.files) == {
        CALIBRATORS_FILE,
        f"{PROBES_DIR}/{TERM.question_key}.json",
        f"{PROBES_DIR}/{ASPECT.question_key}.json",
    }
    assert verify_files(path, manifest) == []
    # test_artifact_identity: the manifest's recorded cell equals the bench cell.
    probe_cell = art.load_results(tmp_path / x3).cells[f"neon_term_fits.probe.{TERM.id}"]
    recorded = manifest.entries[entry_key("probe", TERM.question_key)].cell
    assert recorded is not None
    assert recorded.question_key == probe_cell.question_key == TERM.question_key
    assert recorded.fingerprint == probe_cell.fingerprint == FP.as_dict()
    assert recorded.labels_sha256 == probe_cell.labels_sha256 == manifest.labels_sha256
    assert recorded.key == probe_cell.key and recorded.tier == "probe"
    assert recorded.fold_agreement.agree == 7 and recorded.fold_agreement.folds == 7
    assert recorded == cell_identity(
        x3, probe_cell, config={"spec": "lowdim.v1", "model": "clm-latest"}
    )
    cal_entry = manifest.entries[entry_key("calibrated", ONTOLOGY.question_key)]
    assert cal_entry.file == CALIBRATORS_FILE and cal_entry.cell is not None
    assert cal_entry.cell.fold_agreement.agree == 7
    # Loading back gives runtime calibrators and served probes.
    loaded = load_version(layout, 1, live=FP)
    assert loaded.calibrators == {ONTOLOGY.question_key: PlattCalibrator(a=1.5, b=-0.25, n=190)}
    assert set(loaded.probes) == {TERM.question_key, ASPECT.question_key}
    served = loaded.probes[TERM.question_key]
    assert (served.spec, served.model, served.version, served.k) == (
        "lowdim.v1",
        "clm-latest",
        "v1",
        2,
    )
    assert served.calibration == "platt" and loaded.probes[ASPECT.question_key].k == 8
    assert loaded.dropped == {}
    # The calibrators.json is the provider's bundle form (runtime calibrators only).
    data = json.loads((path / CALIBRATORS_FILE).read_text())
    assert set(data) == {
        "version",
        "encoder_fp",
        "clm_model_fp",
        "framings_lock_sha",
        "calibrators",
    }
    assert data["calibrators"][ONTOLOGY.question_key] == {
        "kind": "platt",
        "a": 1.5,
        "b": -0.25,
        "n": 190,
    }


def test_fold_agreement_counts_only_agreeing_evaluated_folds() -> None:
    probe_cell = cell("term.fits", "probe", spec="lowdim.v1", choices=m4.fold_choices(
        tier="probe", model="clm-latest", spec="lowdim.v1", agree=5
    ))  # fmt: skip
    assert fold_agreement(probe_cell, config={"spec": "lowdim.v1", "model": "clm-latest"}) == (
        art.FoldAgreement(agree=5, folds=7)
    )
    assert (
        fold_agreement(probe_cell, config={"spec": "pair512.v1", "model": "clm-latest"}).agree == 0
    )
    cal_cell = cell("column.ontology_fits", "calibrated")
    assert (
        fold_agreement(cal_cell, config={"framing": ONTOLOGY.id, "model": "clm-latest"}).agree == 7
    )
    assert fold_agreement(cal_cell, config={"framing": "F7", "model": "clm-latest"}).agree == 0


def test_immutability_and_owner_only_modes(tmp_path: Path) -> None:
    layout, _, path, x3, tiers = _write(tmp_path)
    assert _mode(path) == 0o700 and _mode(path / PROBES_DIR) == 0o700
    for p in path.rglob("*"):
        if p.is_file():
            assert _mode(p) == 0o600, p
    assert layout.versions() == [1] and layout.next_version() == 2
    # A second write is a new version; v1 stays byte for byte.
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    entries, calibrators, probes = _entries(x3, tiers, tmp_path)
    n2, path2 = write_version(
        layout,
        entries=entries,
        calibrators=calibrators,
        probes=probes,
        labels_sha256=LABELS_SHA,
        labels_content_sha256=CONTENT_SHA,
    )
    assert (n2, path2.name) == (2, "v2") and layout.versions() == [1, 2]
    assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == before
    # Nothing ever copies over an existing version.
    with pytest.raises(ArtifactError, match="never rewritten"):
        art._copy_version(path2, path, verify=True)
    # An entry without its artifact, or an artifact without its entry, is refused up front.
    with pytest.raises(ArtifactError, match="have no artifact"):
        write_version(
            layout,
            entries=entries,
            calibrators={},
            probes=probes,
            labels_sha256=LABELS_SHA,
            labels_content_sha256=CONTENT_SHA,
        )
    assert layout.versions() == [1, 2]  # the refused write left nothing behind
    assert not [p for p in layout.dir.iterdir() if p.name.startswith(".")]


# -- strict load ---------------------------------------------------------------------------------


def test_tampering_is_refused(tmp_path: Path) -> None:
    layout, _, path, _, _ = _write(tmp_path)
    target = path / PROBES_DIR / f"{TERM.question_key}.json"
    data = json.loads(target.read_text())
    data["n_train"] = 9999
    target.write_text(json.dumps(data))
    with pytest.raises(ArtifactError, match="tampered"):
        load_version(layout, 1, live=FP)
    with pytest.raises(ArtifactError, match="tampered"):
        load_version(layout, 1, live=FP, strict=False)  # verification is never relaxed
    assert verify_files(path, load_manifest(path)) == [
        f"{PROBES_DIR}/{TERM.question_key}.json: sha256 "
        + art.file_sha256(target)[:12]
        + " is not the manifest's "
        + load_manifest(path).files[f"{PROBES_DIR}/{TERM.question_key}.json"][:12]
    ]
    (path / "extra.json").write_text("{}")
    assert any(
        "extra.json: not in the manifest" in p for p in verify_files(path, load_manifest(path))
    )


def test_strict_load_refuses_other_stacks_locks_and_stale_keys(tmp_path: Path) -> None:
    layout, _, path, x3, tiers = _write(tmp_path)
    other = Fingerprint(
        encoder_fp=encoder_fp(
            EncoderSpec(model="Qwen/Qwen3-8B", revision="r", route="transformers")
        ),
        clm_model_fp=FP.clm_model_fp,
        schema_sha256=FP.schema_sha256,
        serving_lock_sha=FP.serving_lock_sha,
    )
    with pytest.raises(ArtifactError, match="K4"):
        load_version(layout, 1, live=other)
    dropped = load_version(layout, 1, live=other, strict=False)
    assert dropped.probes == {} and dropped.calibrators == {}
    assert set(dropped.dropped) == set(load_manifest(path).entries)
    # Another framings lock: the same files read through a layout of another lock.
    rotated = ArtifactLayout(layout.root, layout.encoder_fp, layout.clm_model_fp, "f" * 64)
    rotated.dir.mkdir(parents=True)
    os.rename(path, rotated.version_dir(1))
    with pytest.raises(ArtifactError, match="framings lock"):
        load_version(rotated, 1, live=FP)
    os.rename(rotated.version_dir(1), path)
    # A stale question key: an entry for a framing that is not the task's active one.
    entries, calibrators, probes = _entries(x3, tiers, tmp_path)
    stale_framing = fr.framing("term.fits", "F4")
    stale_probe = synthetic_probe("lowdim.v1").model_copy(
        update={"question_key": stale_framing.question_key, "framing_id": "F4"}
    )
    stale = ManifestEntry(
        question_key=stale_framing.question_key,
        task_id="term.fits",
        framing_id="F4",
        tier="probe",
        file=f"{PROBES_DIR}/{stale_framing.question_key}.json",
        spec="lowdim.v1",
    )
    n, _ = write_version(
        layout,
        entries={**entries, entry_key("probe", stale.question_key): stale},
        calibrators=calibrators,
        probes={**probes, stale.question_key: stale_probe.model_dump(mode="json")},
        labels_sha256=LABELS_SHA,
        labels_content_sha256=CONTENT_SHA,
    )
    with pytest.raises(ArtifactError, match="stale"):
        load_version(layout, n, live=FP)
    relaxed = load_version(layout, n, live=FP, strict=False)
    assert set(relaxed.dropped) == {entry_key("probe", stale.question_key)}
    assert set(relaxed.probes) == {TERM.question_key, ASPECT.question_key}
    with pytest.raises(ArtifactError, match="no such artifacts version"):
        load_version(layout, 9, live=FP)
    # A probe whose fingerprint is not the layout's never gets written.
    foreign = synthetic_probe("lowdim.v1", fp=other).model_dump(mode="json")
    with pytest.raises(ArtifactError, match="K4"):
        write_version(
            layout,
            entries={
                entry_key("probe", TERM.question_key): entries[
                    entry_key("probe", TERM.question_key)
                ]
            },
            calibrators={},
            probes={TERM.question_key: foreign},
            labels_sha256=LABELS_SHA,
            labels_content_sha256=CONTENT_SHA,
        )


# -- promotion and CURRENT.json ------------------------------------------------------------------


def test_promote_rules_and_bundle_from_current(tmp_path: Path) -> None:
    layout, _, _, x3, tiers = _write(tmp_path)
    assert layout.read_current() is None and bundle_from_current(layout, live=FP) is None
    k2 = m4.k2_json({"term.fits": ("probe", "b"), "column.aspect": ("probe", "c")})
    # (ii) a c tier is never promoted; a tier K2 did not judge has no verdict.
    with pytest.raises(ArtifactError, match=r"K2\(c\)"):
        promote(
            layout, version=1, task_id="column.aspect", tier="probe", results_root=tmp_path, k2=k2
        )
    with pytest.raises(ArtifactError, match="no verdict"):
        promote(
            layout,
            version=1,
            task_id="column.ontology_fits",
            tier=None,
            results_root=tmp_path,
            k2=k2,
        )
    assert (
        k2_verdict(k2, "term.fits", "probe") == "b"
        and k2_verdict(k2, "term.fits", "calibrated") is None
    )
    assert (
        k2_verdict(
            {"tasks": {"x": {"task_id": "term.fits", "best": "probe", "verdict": "a"}}},
            "term.fits",
            "probe",
        )
        == "a"
    )
    # Only best == tier has a verdict: a per-tier block in a file is not read (no hook).
    hooked = {
        "tasks": {
            "term.fits": {
                "task_id": "term.fits",
                "best": "probe",
                "verdict": "a",
                "tiers": {"calibrated": {"verdict": "a"}},
            }
        }
    }
    assert k2_verdict(hooked, "term.fits", "calibrated") is None
    assert k2_verdict(hooked, "term.fits", "probe") == "a"
    # (iv) heads are M7's.
    with pytest.raises(ArtifactError, match="M7"):
        promote(layout, version=1, task_id="term.fits", tier="head", results_root=tmp_path, k2=k2)
    # A promotion that qualifies writes CURRENT.json and nothing else.
    report = promote(
        layout,
        version=1,
        task_id="term.fits",
        tier=None,
        results_root=tmp_path,
        k2=k2,
        now="2026-10-05T00:00:00+00:00",
    )
    assert (report.tier, report.version, report.k2_verdict, report.replaced) == (
        "probe",
        "v1",
        "b",
        None,
    )
    current = layout.read_current()
    assert current is not None and current.tasks == {
        "term.fits": CurrentEntry(
            version="v1",
            tier="probe",
            question_key=TERM.question_key,
            promoted_at="2026-10-05T00:00:00+00:00",
            cite=f"{x3}#neon_term_fits.probe.{TERM.id}",
        )
    }
    assert _mode(layout.current_path) == 0o600
    with pytest.raises(ArtifactError, match="already promoted"):
        promote(layout, version=1, task_id="term.fits", tier="probe", results_root=tmp_path, k2=k2)
    bundle = bundle_from_current(layout, live=FP)
    assert bundle is not None and bundle.version == "v1"
    assert set(bundle.probes) == {TERM.question_key} and bundle.calibrators == {}
    assert bundle.promoted["term.fits"].tier == "probe" and bundle.versions == {}
    assert bundle.ref(TERM.question_key).version == "v1"
    # (i) the cite must name a pre-registered nested cell with the artifact's identity.
    bad = m4.results_file(
        tmp_path,
        "2026-10-06",
        "x3",
        [cell("term.fits", "probe", spec="lowdim.v1", selection="full", exploratory=True)],
    )
    entries, calibrators, probes = _entries(x3, tiers, tmp_path)
    probe_key = entry_key("probe", TERM.question_key)
    entries[probe_key] = entries[probe_key].model_copy(
        update={"cite": f"{bad}#neon_term_fits.probe.{TERM.id}"}
    )
    n2, _ = write_version(
        layout, entries=entries, calibrators=calibrators, probes=probes,
        labels_sha256=LABELS_SHA, labels_content_sha256=CONTENT_SHA,
    )  # fmt: skip
    with pytest.raises(ArtifactError, match="rule i"):
        promote(layout, version=n2, task_id="term.fits", tier="probe", results_root=tmp_path, k2=k2)
    # (iii) a newer artifact replaces the current one only when its cell ≻ on NLL and ≽ on acc.
    worse = m4.results_file(
        tmp_path, "2026-10-07", "x3",
        [cell("term.fits", "probe", spec="lowdim.v1", items=planted_items(p_correct=0.7, seed=3))],
    )  # fmt: skip
    better = m4.results_file(
        tmp_path, "2026-10-08", "x3",
        [cell("term.fits", "probe", spec="lowdim.v1", items=planted_items(p_correct=0.95, seed=4))],
    )  # fmt: skip
    for cite_path in (worse, better):
        entries[probe_key] = entries[probe_key].model_copy(
            update={"cite": f"{cite_path}#neon_term_fits.probe.{TERM.id}"}
        )
        write_version(
            layout, entries=entries, calibrators=calibrators, probes=probes,
            labels_sha256=LABELS_SHA, labels_content_sha256=CONTENT_SHA,
        )  # fmt: skip
    assert layout.versions() == [1, 2, 3, 4]
    with pytest.raises(ArtifactError, match="rule iii"):
        promote(layout, version=3, task_id="term.fits", tier="probe", results_root=tmp_path, k2=k2)
    assert layout.read_current() == current  # nothing written on a refusal
    report = promote(
        layout, version=4, task_id="term.fits", tier="probe", results_root=tmp_path, k2=k2
    )
    assert report.replaced == current.tasks["term.fits"] and report.version == "v4"
    assert (
        report.comparison["nll_rule_r"]["passed"]
        and report.comparison["acc_non_inferior"]["passed"]
    )
    now = layout.read_current()
    assert now is not None and now.tasks["term.fits"].version == "v4"
    bundle = bundle_from_current(layout, live=FP)
    assert (
        bundle is not None
        and bundle.version == "v4"
        and bundle.probes[TERM.question_key].version == "v4"
    )
    # A second task promoted from an older version: the bundle spans versions.
    k2b = m4.k2_json({"column.ontology_fits": ("calibrated", "a")})
    promote(
        layout,
        version=1,
        task_id="column.ontology_fits",
        tier="calibrated",
        results_root=tmp_path,
        k2=k2b,
    )
    bundle = bundle_from_current(layout, live=FP)
    assert bundle is not None and set(bundle.calibrators) == {ONTOLOGY.question_key}
    assert bundle.versions == {ONTOLOGY.question_key: "v1"} and bundle.version == "v4"
    assert bundle.ref(ONTOLOGY.question_key).version == "v1"
    # A CURRENT.json naming what the version does not hold is refused.
    broken = now.model_copy(
        update={
            "tasks": {
                **now.tasks,
                "column.aspect": now.tasks["term.fits"].model_copy(update={"tier": "calibrated"}),
            }
        }
    )  # fmt: skip
    layout.write_current(broken)
    with pytest.raises(ArtifactError, match="holds no such entry"):
        bundle_from_current(layout, live=FP)
    layout.current_path.write_text("{nope")
    with pytest.raises(ArtifactError, match=r"CURRENT\.json"):
        layout.read_current()


def _worse_acc_items(p_correct: float, wrong_every: int, seed: int) -> list:
    """Planted items whose label gets ``p_correct`` except every ``wrong_every``-th item per
    card, which is answered wrongly but mildly (the label at 0.45): a lower NLL than a flat
    0.7 cell and a lower accuracy."""
    items = []
    for i, it in enumerate(planted_items(p_correct=p_correct, seed=seed)):
        if i % wrong_every == 0:
            probs = [0.45, 0.55] if it.label == 0 else [0.55, 0.45]
            it = it.model_copy(update={"probs": probs})
        items.append(it)
    return items


def test_promote_refuses_a_cell_better_on_nll_but_worse_on_acc(tmp_path: Path) -> None:
    """Rule (iii) is two conditions: ≻ on NLL (rule R) and ≽ on acc within 0.01. A cell that
    beats the current one on NLL on every card but is several points worse on accuracy is
    refused, and CURRENT.json is untouched."""
    flat = m4.results_file(
        tmp_path, "2026-10-09", "x3",
        [cell("term.fits", "probe", spec="lowdim.v1", items=planted_items(p_correct=0.7, seed=3))],
    )  # fmt: skip
    sharper = m4.results_file(
        tmp_path, "2026-10-10", "x3",
        [cell("term.fits", "probe", spec="lowdim.v1", items=_worse_acc_items(0.95, 12, 4))],
    )  # fmt: skip
    cur = art.load_results(tmp_path / flat).cells[f"neon_term_fits.probe.{TERM.id}"]
    new = art.load_results(tmp_path / sharper).cells[f"neon_term_fits.probe.{TERM.id}"]
    # (the fake cell's metrics block is fixed synthetic numbers; the items are what rule iii
    # reads, so the planted direction is checked on them)
    cur_p, new_p = art._paired(new, cur)[1], art._paired(new, cur)[0]
    nll = {
        k: stats.metric_value("nll", p.probs, p.labels) for k, p in (("cur", cur_p), ("new", new_p))
    }
    acc = {
        k: stats.metric_value("acc", p.probs, p.labels) for k, p in (("cur", cur_p), ("new", new_p))
    }
    assert nll["new"] < nll["cur"] and acc["new"] < acc["cur"] - 0.03
    x3, tiers = _results(tmp_path)
    entries, calibrators, probes = _entries(x3, tiers, tmp_path)
    key = entry_key("probe", TERM.question_key)
    layout = ArtifactLayout.for_fingerprint(tmp_path / "acc-arts", FP)
    for cite_path in (flat, sharper):
        entries[key] = entries[key].model_copy(
            update={"cite": f"{cite_path}#neon_term_fits.probe.{TERM.id}"}
        )
        write_version(
            layout, entries=entries, calibrators=calibrators, probes=probes,
            labels_sha256=LABELS_SHA, labels_content_sha256=CONTENT_SHA,
        )  # fmt: skip
    k2 = m4.k2_json({"term.fits": ("probe", "b")})
    promote(layout, version=1, task_id="term.fits", tier="probe", results_root=tmp_path, k2=k2)
    before = layout.read_current()
    with pytest.raises(ArtifactError, match=r"not ≽ the current one on acc within 0\.01"):
        promote(layout, version=2, task_id="term.fits", tier="probe", results_root=tmp_path, k2=k2)
    assert layout.read_current() == before


# -- learn fit's writer, the export copy --------------------------------------------------------


def test_fit_version_with_an_injected_probe_fitter_and_export_copy(tmp_path: Path) -> None:
    from mesa_clm.bench.cells import score_arm
    from tests.unit.test_cells import _scorers, rank_world

    x3, tiers = _results(tmp_path)
    world = rank_world(tmp_path / "world", n_targets=12)
    scores = score_arm(
        world.task, world.index, world.store, _scorers(world.store)["clm-latest"], "F7"
    )
    layouts = {
        "clm-latest": _layout(tmp_path),
        "clm-raw": ArtifactLayout.for_fingerprint(
            tmp_path / "artifacts", m4.fake_fingerprint("clm-raw")
        ),
    }
    probe = synthetic_probe("lowdim.v1")
    raw_probe = synthetic_probe("pair4096.v1", fp=m4.fake_fingerprint("clm-raw"))
    calls: list[str] = []

    def fitter(task: Any) -> tuple[Any, dict[str, Any]]:
        calls.append(task.name)
        return (probe if not calls[1:] else raw_probe), {"inner_nll": 0.4}

    result = art.fit_version(
        layouts,
        tasks={"neon_term_fits": world.task, "again": world.task},
        arm_scores={"neon_term_fits": scores},
        probe_fitter=fitter,
        labels_sha256=LABELS_SHA,
        labels_content_sha256=CONTENT_SHA,
        results_root=tmp_path,
        x3_path=x3,
        tiers_path=tiers,
        registered=True,
        registration={"framings_lock_sha": LOCK, "inputs": {"tiers": {"path": tiers}}},
    )
    assert calls == ["neon_term_fits", "neon_term_fits"]
    assert set(result.written) == {"clm-latest", "clm-raw"}
    latest = result.manifests["clm-latest"]
    assert {e.tier for e in latest.entries.values()} == {"calibrated", "probe"}
    # The manifest records the registration learn fit applied and the tasks it fitted.
    for man in result.manifests.values():
        assert man.registered is True and man.registration["framings_lock_sha"] == LOCK
        assert man.tasks_fitted == ["term.fits"]
    assert load_manifest(_write(tmp_path / "plain")[2]).registered is False
    # A calibrated and a probe artifact of one framing share the question key and never a
    # manifest key (entry_key); the calibrated one cites nothing (tiers.json has no term.fits
    # cell), the probe cites x3.json's nested cell.
    cal_entry = latest.entries[entry_key("calibrated", TERM.question_key)]
    probe_entry = latest.entries[entry_key("probe", TERM.question_key)]
    assert cal_entry.tier == "calibrated" and cal_entry.cite is None and cal_entry.cell is None
    assert cal_entry.calibrator["kind"] == "platt" and cal_entry.n_train == len(world.task.items)
    assert probe_entry.spec == "lowdim.v1" and probe_entry.cite is not None
    assert probe_entry.cell is not None and probe_entry.cell.fold_agreement.agree == 7
    raw = result.manifests["clm-raw"]
    assert next(iter(raw.entries.values())).spec == "pair4096.v1"
    assert result.skipped == {"again.calibrated": "no A1 arm (K1): no calibrator fitted"}
    only_cal = art.fit_version(
        {"clm-latest": ArtifactLayout.for_fingerprint(tmp_path / "arts2", FP)},
        tasks={"neon_term_fits": world.task},
        arm_scores={"neon_term_fits": scores},
        probe_fitter=None,
        labels_sha256=LABELS_SHA,
        labels_content_sha256=CONTENT_SHA,
        results_root=tmp_path,
        x3_path=x3,
        tiers_path=tiers,
    )
    entry = only_cal.manifests["clm-latest"].entries[entry_key("calibrated", TERM.question_key)]
    assert entry.tier == "calibrated" and entry.cite is None
    # A calibrated cell that exists but is not a pre-registered nested one (the closed
    # choices' fixed arms, term.fits' K1 audit cells: selection none) gives cite: null too.
    fixed_arm = m4.results_file(
        tmp_path, "2026-10-06", "tiers",
        [cell("term.fits", "calibrated", selection="none", pre_registered=False, exploratory=True)],
    )  # fmt: skip
    none_entry, _ = art.calibrated_entry(
        world.task, scores, results_root=tmp_path, tiers_path=fixed_arm
    )
    assert none_entry.cite is None and none_entry.cell is None
    loaded = load_version(layouts["clm-latest"], 1, live=FP)
    assert isinstance(load_calibrator(entry.calibrator), PlattCalibrator)
    assert set(loaded.probes) == {TERM.question_key}
    # The export copy: manifest and probes only, once.
    exported = export_copy(result.written["clm-latest"][1], tmp_path / "bench" / "results", DATE)
    assert exported == tmp_path / "bench" / "results" / DATE / "artifacts_v1" / FP.clm_model_fp
    assert (exported / MANIFEST_FILE).read_bytes() == (
        result.written["clm-latest"][1] / MANIFEST_FILE
    ).read_bytes()
    assert sorted(p.name for p in (exported / PROBES_DIR).iterdir()) == [
        f"{TERM.question_key}.json"
    ]
    assert not (exported / CALIBRATORS_FILE).exists()
    with pytest.raises(ArtifactError, match="one export per version"):
        export_copy(result.written["clm-latest"][1], tmp_path / "bench" / "results", DATE)


# -- publish and pull ---------------------------------------------------------------------------


def test_publish_pull_round_trip_and_tamper_refused(tmp_path: Path) -> None:
    layout, _, path, _, _ = _write(tmp_path)
    remote = tmp_path / "remote"
    published = publish(layout, 1, remote)
    assert published == remote / FP.encoder_fp / FP.clm_model_fp / LOCK[:8] / "v1"
    assert verify_files(published, load_manifest(published)) == []
    with pytest.raises(ArtifactError, match="never rewritten"):
        publish(layout, 1, remote)
    with pytest.raises(ArtifactError, match="no such artifacts version"):
        publish(layout, 2, remote)
    other = ArtifactLayout.for_fingerprint(tmp_path / "other-host", FP)
    pulled = pull(other, 1, remote)
    assert pulled == other.version_dir(1) and load_manifest(pulled) == load_manifest(path)
    assert {p.relative_to(pulled): p.read_bytes() for p in pulled.rglob("*") if p.is_file()} == {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }
    assert _mode(pulled) == 0o700 and all(
        _mode(p) == 0o600 for p in pulled.rglob("*") if p.is_file()
    )
    assert not (other.dir / CURRENT_FILE).exists()  # promotion is local
    with pytest.raises(ArtifactError, match="never rewritten"):
        pull(other, 1, remote)
    # A tampered remote is refused on pull and leaves nothing behind.
    target = published / PROBES_DIR / f"{ASPECT.question_key}.json"
    target.write_text(target.read_text().replace("logreg", "ridge "))
    third = ArtifactLayout.for_fingerprint(tmp_path / "third-host", FP)
    with pytest.raises(ArtifactError, match="do not match the manifest"):
        pull(third, 1, remote)
    assert third.versions() == [] and not any(third.dir.iterdir()) if third.dir.exists() else True
    # ``--verify`` off copies the tampered version (the strict load refuses it later).
    pull(third, 1, remote, verify=False)
    with pytest.raises(ArtifactError, match="tampered"):
        load_version(third, 1, live=FP)


def test_load_calibrator_accepts_the_learned_form() -> None:
    from mesa_clm.learn import calibrate

    learned = calibrate.PlattCalibrator(
        a=1.0, b=0.5, n=10, weight_sum=10.0, ridge=1e-6, iterations=3, converged=True, nll=0.4
    )
    assert load_calibrator(learned.model_dump(mode="json")) == PlattCalibrator(a=1.0, b=0.5, n=10)
    temp = calibrate.TemperatureCalibrator(
        temperature=2.5, n=10, weight_sum=10.0, lower=1e-4, upper=1e4, iterations=5,
        at_bound=False, nll=0.4,
    )  # fmt: skip
    assert load_calibrator(temp.model_dump(mode="json")) == TemperatureCalibrator(T=2.5, n=10)
    with pytest.raises(ValueError):
        load_calibrator({"kind": "isotonic"})
