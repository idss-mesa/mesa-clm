"""Appendix B of the M4 analysis plan against the code (the M2 pattern of
``test_m2_plan_constants.py``): the K2 constants (name by name against ``k2.py``/``stats.py``),
the artifact and audit format strings, the probe grids and floors (the registered X3 grid is
``learn.probe.DEFAULT_GRID`` for both shapes), the citation test's numbers, the serving
constants, and the M4 registration's filled pins: the teacher corpus hash and neon-ducklake
commit, the two X4 snapshots by name and by the sha256 of their bytes (hashed, never opened for
their labels). Reads the committed plan and hashes the snapshot files only; no label, no
model."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from mesa_clm import artifacts, audit, policy
from mesa_clm.bench import k2, registered, stats
from mesa_clm.learn import linear, probe
from mesa_clm.provenance import models
from mesa_clm.providers import base, tiered

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "design" / "m4-analysis-plan.md"
DRAFT = ROOT / "design" / "m4" / "serving.md"
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])")


# -- the K2 constants (plan §8 K2; brief R2) --------------------------------------------------


def test_k2_constants() -> None:
    assert (k2.ACC_MARGIN, k2.HEAD_ACC_MARGIN, k2.ECE_MAX, k2.ECE_UPPER_MAX) == (
        0.02,
        0.01,
        0.08,
        0.12,
    )
    assert (stats.SIGN_FRACTION, stats.SIGN_MIN_ITEMS, stats.MIN_CLUSTERS) == (0.8, 10, 4)
    # Every K2 key of the registration, by name, is the code's constant.
    assert dict(registered.K2_CONSTANTS) == {
        "acc_margin": k2.ACC_MARGIN,
        "head_acc_margin": k2.HEAD_ACC_MARGIN,
        "ece_max": k2.ECE_MAX,
        "ece_upper_max": k2.ECE_UPPER_MAX,
        "sign_fraction": stats.SIGN_FRACTION,
        "sign_min_items": stats.SIGN_MIN_ITEMS,
        "min_clusters": stats.MIN_CLUSTERS,
    }
    assert registered.K2_CONSTANTS["acc_margin"] == k2.ACC_MARGIN
    assert registered.K2_CONSTANTS["head_acc_margin"] == k2.HEAD_ACC_MARGIN
    assert registered.K2_CONSTANTS["ece_max"] == k2.ECE_MAX
    assert registered.K2_CONSTANTS["ece_upper_max"] == k2.ECE_UPPER_MAX
    assert registered.K2_CONSTANTS["sign_fraction"] == stats.SIGN_FRACTION
    assert registered.K2_CONSTANTS["sign_min_items"] == stats.SIGN_MIN_ITEMS
    assert registered.K2_CONSTANTS["min_clusters"] == stats.MIN_CLUSTERS


# -- formats, grids, floors ----------------------------------------------------------------------


def test_format_strings() -> None:
    assert artifacts.FORMAT == "mesa-clm/artifacts/1"
    assert artifacts.CURRENT_FORMAT == "mesa-clm/artifacts-current/1"
    assert probe.PROBE_FORMAT == "mesa-clm/probe/1"
    assert audit.FORMAT == "mesa-clm/audit-sample/1"
    assert (
        artifacts.LOCK_SHA8 == 8
        and artifacts.VERSION_RE.match("v1")
        and not artifacts.VERSION_RE.match("v0")
    )
    assert artifacts.EXPORT_DIR.format(n=3) == "artifacts_v3"


def test_probe_grids_and_floors() -> None:
    assert [h["lambda"] for h in linear.GRIDS["logreg"]] == [1e-2, 1e-1, 1.0, 10.0]
    assert [h["gamma"] for h in linear.GRIDS["lda"]] == [0.1, 0.5, 0.9]
    assert [h["lambda"] for h in linear.GRIDS["ridge"]] == [1e-1, 1.0, 10.0]
    assert linear.STD_FLOOR == 1e-8 and probe.FITTERS == ("logreg", "lda", "ridge")
    assert probe.CALIBRATION_FLOOR == 100 and probe.PROBE_FLOOR == 40
    assert [s.id for s in probe.SPECS] == list(tiered.SPEC_MODELS)
    for s in probe.SPECS:
        assert tiered.SPEC_MODELS[s.id] == s.model
        assert (s.id in tiered.SPEC_NEEDS_HEAD) == (s.model == "clm-latest")
    assert probe.PROJECTION_DIM == 512
    grid = dict(registered.X3_GRID)
    assert grid, "the M4 registration pins the X3 grid"
    # The registered grid is learn.probe's default grid for both shapes, fitter by fitter.
    for shape in ("rank_fit", "choice"):
        assert registered.grid_deviations(probe.DEFAULT_GRID[shape], shape=shape) == []  # type: ignore[arg-type]
    assert grid["calibration_floor"] == probe.CALIBRATION_FLOOR
    assert grid["probe_floor"] == probe.PROBE_FLOOR and grid["std_floor"] == linear.STD_FLOOR


# -- the citation test, promotion, audits, serving --------------------------------------------


def test_citation_promotion_audit_and_serving_constants() -> None:
    assert (
        policy.MIN_NESTED_FOLDS == 5 and policy.FOLD_AGREEMENT_MIN == 5 and policy.GUARD_MIN == 30
    )
    assert policy.SNAPSHOTS_DIR == "bench/snapshots"
    assert policy.DEFAULT_RESULTS_HOME == "~/.mesa/clm/results"
    assert "audit_required" in policy.AUTO_BLOCKERS and policy.DEMOTION_REASONS == (
        "audit_required",
    )
    assert artifacts.PROMOTE_ACC_MARGIN == 0.01 and {"a", "b"} == artifacts.PROMOTABLE_VERDICTS
    assert (models.AUDIT_MIN_N, models.AUDIT_MIN_CARDS, models.AUDIT_RISK_FACTOR) == (50, 3, 2.0)
    assert (audit.DEFAULT_N, audit.DEFAULT_MIN_CARDS, audit.SAMPLE_SEED) == (100, 5, 0)
    assert audit.TOP_DECILE == 0.9 and audit.STRATA == (
        "would_be_auto",
        "proposed",
        "anchor_abstain",
    )
    assert tiered.VECTOR_CACHE_MAX == 2048 and tiered.PROBE_LOGIT_CLIP == 1e-12
    assert {"probe", "head"} == base.FEATURE_SPEC_LEVELS
    assert tiered.spec_dim("pair512.v1") == 1025 and tiered.spec_dim("choice.raw.v1", k=4) == 4100


# -- the draft's constants table holds to the code -----------------------------------------------


def _table(path: Path, heading: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    assert text.count(heading) == 1, f"{path} must have one {heading!r}"
    rows: dict[str, str] = {}
    for line in text.split(heading, 1)[1].splitlines():
        if line.startswith("## "):
            break
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if cells[0] != "constant":
            rows[cells[0]] = cells[1]
    return rows


def _numbers(value: str) -> list[float]:
    return [float(n) for n in NUMBER.findall(value)]


EXPECTED_DRAFT: dict[str, list[float]] = {
    "promotion acc margin": [artifacts.PROMOTE_ACC_MARGIN],
    "vector cache": [tiered.VECTOR_CACHE_MAX],
    "probe logit clip": [tiered.PROBE_LOGIT_CLIP],
    "citation: nested folds": [policy.MIN_NESTED_FOLDS],
    "citation: fold agreement": [policy.FOLD_AGREEMENT_MIN],
    "citation: guards": [policy.GUARD_MIN, policy.GUARD_MIN],
    "audit pass rule": [models.AUDIT_MIN_N, models.AUDIT_MIN_CARDS, models.AUDIT_RISK_FACTOR],
    "audit sample": [
        audit.DEFAULT_N,
        audit.DEFAULT_MIN_CARDS,
        audit.SAMPLE_SEED,
        3,
        audit.TOP_DECILE,
    ],
    "lock directory": [artifacts.LOCK_SHA8],
}


@pytest.mark.parametrize("name", list(EXPECTED_DRAFT))
def test_the_drafts_constants_table_is_the_codes(name: str) -> None:
    source = PLAN if PLAN.is_file() else DRAFT
    heading = "## Appendix B" if source == PLAN else "## C5. Constants"
    rows = _table(source, heading)
    if name not in rows:
        pytest.skip(f"{source.name} does not list {name!r} yet (the integrator merges the draft)")
    got = _numbers(rows[name])
    want = EXPECTED_DRAFT[name]
    assert len(got) == len(want), (name, rows[name], got)
    for g, w in zip(got, want, strict=True):
        assert math.isclose(g, float(w), rel_tol=0, abs_tol=0), (name, rows[name])


# -- the M4 registration's pins (filled 2026-10-05 after the label-free ingest) -------------------

TEACHER_CORPUS_SHA256 = "d90387928bed0a9c196a8efc4dce3cbf40b1e2e8cc879bfb50c5e3451b208785"
NEON_DUCKLAKE_COMMIT = "b1fa52a8d3ffc56173a09794253ad86b0fd10111"
TEACHER_SNAPSHOT = "bench/snapshots/2026-10-04-teacher.parquet"
MINUS_OPUS_SNAPSHOT = "bench/snapshots/2026-10-04-minus-opus.parquet"


def test_registration_pins() -> None:
    reg = registered.current_m4()
    assert dict(reg.k2) == dict(registered.K2_CONSTANTS)
    assert dict(reg.grid) == dict(registered.X3_GRID)
    assert reg.framings_lock_sha == registered.FRAMINGS_LOCK_SHA_M4
    assert dict(reg.active) == dict(registered.ACTIVE_FRAMINGS_M4)
    assert reg.teacher_arms == registered.X4_TEACHER_ARMS
    # The filled pins, value by value (no UNPINNED left).
    assert reg.teacher_corpus_sha256 == TEACHER_CORPUS_SHA256
    assert reg.neon_ducklake_commit == NEON_DUCKLAKE_COMMIT
    assert reg.teacher_snapshot == TEACHER_SNAPSHOT
    assert reg.minus_opus_snapshot == MINUS_OPUS_SNAPSHOT
    for name in (
        "teacher_corpus_sha256",
        "neon_ducklake_commit",
        "teacher_snapshot",
        "teacher_labels_sha256",
        "teacher_labels_content_sha256",
        "minus_opus_snapshot",
        "minus_opus_labels_sha256",
        "minus_opus_labels_content_sha256",
    ):
        assert getattr(reg, name) not in (registered.UNPINNED, None), name
    assert len(reg.teacher_corpus_sha256) == 64 and len(reg.neon_ducklake_commit) == 40
    # The two X4 snapshots: the sha256 of each file's bytes is its pin (hashed, never opened).
    teacher_path = reg.teacher_snapshot_path(ROOT)
    minus_path = reg.minus_opus_snapshot_path(ROOT)
    assert teacher_path is not None and teacher_path.is_file()
    assert minus_path is not None and minus_path.is_file()
    assert registered.file_sha256(teacher_path) == reg.teacher_labels_sha256
    assert registered.file_sha256(minus_path) == reg.minus_opus_labels_sha256
    assert (
        registered.pinned_snapshot_deviations(
            "teacher", reg.teacher_labels_sha256, reg.teacher_labels_content_sha256
        )
        == []
    )
    assert (
        registered.pinned_snapshot_deviations(
            "minus_opus", reg.minus_opus_labels_sha256, reg.minus_opus_labels_content_sha256
        )
        == []
    )
    # The committed M2 inputs K2 reads, by sha256.
    assert (
        registered.committed_input_deviations("tiers.json", reg.tiers_path(ROOT), reg.tiers_sha256)
        == []
    )
    assert registered.committed_input_deviations("x2.json", reg.x2_path(ROOT), reg.x2_sha256) == []
