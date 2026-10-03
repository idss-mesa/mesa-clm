"""The pre-registered M2 inputs (``mesa_clm.bench.registered``; ``design/m2-analysis-plan.md``
§1, §13) against the committed files, by digest and published count only: the snapshot's
sha256 and label-content digest, the counts ``bench baselines`` published in M0, the AnyJev dump's
sha256. No label value is read beyond those published counts, and no model output is involved.
The checks themselves are exercised on synthetic tasks and files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mesa_clm.bench import registered as reg
from mesa_clm.bench.results import snapshot_content_sha256
from mesa_clm.bench.tasks.base import Task
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "bench" / "results" / "2026-09-29" / "baselines.json"


def test_the_registration_is_the_published_snapshot() -> None:
    r = reg.REGISTERED
    snap = r.snapshot_path()
    assert snap == ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    assert hashlib.sha256(snap.read_bytes()).hexdigest() == r.labels_sha256
    assert published["labels_sha256"] == r.labels_sha256
    assert published["labels_content_sha256"] == r.labels_content_sha256
    assert snapshot_content_sha256(snap) == r.labels_content_sha256  # a digest, no value read
    assert reg.check_snapshot(snap) == (r.labels_sha256, r.labels_content_sha256)
    assert reg.data_deviations(r.labels_sha256, r.labels_content_sha256) == []
    assert hashlib.sha256(r.anyjev_path().read_bytes()).hexdigest() == r.anyjev_sha256
    assert reg.check_anyjev(r.anyjev_path()) == r.anyjev_sha256
    assert (r.B, r.seed, r.alpha) == (2000, 0, 0.05)
    cells = published["cells"]
    assert set(r.counts) == {k.split(".")[0] for k in cells}
    for name, want in r.counts.items():
        cell = cells[f"{name}.baseline.lookup_prob"]
        assert cell["counts"]["n"] == want.n
        assert {int(k): v for k, v in cell["counts"]["class_counts"].items()} == dict(
            want.class_counts
        )
        assert cell["diagnostics"]["per_fold_n"] == dict(want.per_card)
        assert cell["min_weight"] == want.min_weight


def _task(name: str, task_id: str, cards: dict[str, list[int]], min_weight: float) -> Task:
    items: list[tuple[dict[str, Any], int]] = []
    card_of: list[str] = []
    for card, labels in cards.items():
        for label in labels:
            items.append(({}, label))
            card_of.append(card)
    return Task(
        name, TASKS[task_id], items, "s", "s", cards=card_of, meta={"min_weight": min_weight}
    )


def test_the_checks_refuse_another_snapshot_and_other_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other.parquet"
    other.write_bytes(b"not the snapshot")
    with pytest.raises(reg.RegistrationError, match="not the pre-registered snapshot"):
        reg.check_snapshot(other)
    with pytest.raises(reg.RegistrationError, match="no such labels snapshot"):
        reg.check_snapshot(tmp_path / "missing.parquet")
    with pytest.raises(reg.RegistrationError, match="not the registered AnyJev"):
        reg.check_anyjev(other)
    assert len(reg.data_deviations("x" * 64, "y" * 64)) == 2
    counts = {"neon_term_fits": reg.TaskCounts(4, {0: 2, 1: 2}, {"a": 2, "b": 2}, 0.5)}
    monkeypatch.setattr(
        reg,
        "REGISTERED",
        reg.Registration(snapshot="s", labels_sha256="a", labels_content_sha256="b",
                         published="synthetic", counts=counts),
    )  # fmt: skip
    good = _task("neon_term_fits", "term.fits", {"a": [0, 1], "b": [0, 1]}, 0.5)
    reg.check_tasks({"neon_term_fits": good})
    with pytest.raises(reg.RegistrationError, match="gives no items"):
        reg.check_tasks({})
    bad = {
        "min_weight": _task("neon_term_fits", "term.fits", {"a": [0, 1], "b": [0, 1]}, 0.6),
        "published": _task("neon_term_fits", "term.fits", {"a": [0, 0], "b": [0, 1]}, 0.5),
        "per card": _task("neon_term_fits", "term.fits", {"a": [0, 1, 0], "b": [1]}, 0.5),
    }
    for what, task in bad.items():
        with pytest.raises(reg.RegistrationError, match=what):
            reg.check_tasks({"neon_term_fits": task})


def test_the_registered_framings_and_models_are_the_committed_locks() -> None:
    """§1.4-§1.6, §13.1: the framings lock of G1 and each X1 model's D5 fingerprint under the
    serving lock of G1 are the checkout's (``framings.lock.json``, ``serving/serving.lock.json``).
    Hashes of committed files only."""
    from mesa_clm import framings as fr
    from mesa_clm.bench.framing import MODELS, fingerprints_for_lock
    from mesa_clm.clm.fingerprint import load_serving_lock

    r = reg.REGISTERED
    lock = load_serving_lock(ROOT / "serving" / "serving.lock.json")
    assert r.framings_lock_sha == reg.FRAMINGS_LOCK_SHA == fr.lock_sha()
    assert r.serving_lock_sha == reg.SERVING_LOCK_SHA == lock.lock_sha
    assert sorted(r.fingerprints) == sorted(MODELS)
    assert {m: dict(fp) for m, fp in r.fingerprints.items()} == fingerprints_for_lock(lock)
    assert r.fingerprints["clm-latest"]["clm_model_fp"] == "78be8c462b2e"
    assert r.fingerprints["clm-latest"]["encoder_fp"] == "c3b3d5e1a283"
    assert (
        reg.identity_deviations(
            framings_lock_sha=fr.lock_sha(), fingerprints=fingerprints_for_lock(lock), models=MODELS
        )
        == []
    )


def test_identity_dump_and_task_deviations() -> None:
    """Every other framings lock, fingerprint field, model or dump is a deviation; a missing one
    too; ``ignore`` leaves a field out (X2's replica reads no head)."""
    r = reg.REGISTERED
    good = {m: dict(fp) for m, fp in r.fingerprints.items()}
    assert (
        reg.identity_deviations(framings_lock_sha=None, fingerprints=good, models=["clm-raw"]) == []
    )
    assert reg.identity_deviations(
        framings_lock_sha="0" * 64, fingerprints=good, models=["clm-latest"]
    ) == ["framings lock_sha 000000000000 is not b432d32a7536"]
    for field in ("encoder_fp", "clm_model_fp", "schema_sha256", "serving_lock_sha"):
        bad = {**good, "clm-latest": {**good["clm-latest"], field: "f" * 12}}
        got = reg.identity_deviations(
            framings_lock_sha=None, fingerprints=bad, models=["clm-latest"]
        )
        assert len(got) == 1 and f"{field} ffffffffffff (registered " in got[0], field
    no_head = {**good, "clm-latest": {**good["clm-latest"], "clm_model_fp": "f" * 12}}
    assert (
        reg.identity_deviations(
            framings_lock_sha=None,
            fingerprints=no_head,
            models=["clm-latest"],
            ignore=("clm_model_fp",),
        )
        == []
    )
    missing = reg.identity_deviations(framings_lock_sha=None, fingerprints=None, models=["clm-raw"])
    assert missing and missing[0].startswith("clm-raw: no fingerprint")
    assert reg.identity_deviations(
        framings_lock_sha=None, fingerprints=good, models=["clm-other"]
    ) == ["clm-other is not a registered model"]
    assert reg.anyjev_deviations(r.anyjev_sha256) == []
    assert reg.anyjev_deviations("e" * 64) == [
        "AnyJev L2 dump sha256 eeeeeeeeeeee is not 557c53b72847"
    ]
    assert reg.anyjev_deviations(None)[0].startswith("no AnyJev L2 dump")
    assert reg.task_set_deviations(list(r.counts)) == []
    assert reg.task_set_deviations(["neon_term_fits"]) == [
        "tasks ['neon_annotate', 'neon_aspect', 'neon_ontology_fits', 'neon_value_kind'] are not run"
    ]
