"""``policy_defaults.yaml`` and its typed loader (DESIGN D8, D9; plan §4.7): shipped
proposed-only, one source of ``min_weight``, the right statistic per shape, the two profiles.
Ported from mesa-anyjev ``tests/test_policy_citations.py`` (the cell-citation half lands with
``policy.py`` in M1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mesa_clm.policy_defaults import (
    DEFAULTS_PATH,
    LEVEL_RANK,
    PolicyDefaults,
    Profile,
    Thresholds,
    load_policy_defaults,
    min_weight_for,
)
from mesa_clm.tasks import ACTIVE_TASKS, RANK_FIT_TASKS


def _raw() -> dict:  # type: ignore[type-arg]
    data: dict = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))  # type: ignore[type-arg]
    return data


def test_defaults_file_is_found_and_ships_proposed_only() -> None:
    assert DEFAULTS_PATH.name == "policy_defaults.yaml" and DEFAULTS_PATH.exists()
    policy = load_policy_defaults()
    assert all(t.auto is None for t in policy.tasks.values())
    assert all(t.cite is None for t in policy.tasks.values())


def test_every_active_task_has_thresholds_with_the_right_stat() -> None:
    policy = load_policy_defaults()
    assert set(policy.tasks) == set(ACTIVE_TASKS)
    for task_id, t in policy.tasks.items():
        expected = "p_fit" if task_id in RANK_FIT_TASKS else "confidence"
        assert t.stat == expected, f"{task_id}: rank_fit reads p_fit, a choice reads confidence"
        assert 0.0 <= t.propose <= 1.0 and 0.0 <= t.margin <= 1.0
        assert t.min_level == "calibrated", "zero_shot never autos (D6)"
        assert t.risk == 0.05 and t.ols_rank_top == 1


def test_min_weight_has_one_source_d9() -> None:
    expected = {
        "term.fits": 0.5,
        "column.ontology_fits": 0.5,
        "column.annotate": 0.5,
        "avu.value_kind": 0.5,
        "column.aspect": 0.6,
        "column.ontology": 0.6,
    }
    policy = load_policy_defaults()
    for task_id, weight in expected.items():
        assert policy.min_weight(task_id) == weight, task_id
        assert min_weight_for(task_id) == weight
        assert policy.thresholds(task_id).min_weight == weight
    with pytest.raises(KeyError):
        min_weight_for("avu.keep")


def test_propose_and_margin_carried_from_anyjev() -> None:
    policy = load_policy_defaults()
    carried = {
        "column.annotate": (0.5, 0.0),
        "column.aspect": (0.45, 0.10),
        "column.ontology": (0.40, 0.0),
        "column.ontology_fits": (0.40, 0.0),
        "term.fits": (0.50, 0.15),
        "avu.value_kind": (0.50, 0.0),
    }
    for task_id, (propose, margin) in carried.items():
        t = policy.thresholds(task_id)
        assert (t.propose, t.margin) == (propose, margin), task_id


def test_profiles_per_plan_4_7() -> None:
    policy = load_policy_defaults()
    prod, dev = policy.profile("prod"), policy.profile("dev")
    assert prod.name == "prod" and dev.name == "dev"
    assert prod.min_level_write == "calibrated" and dev.min_level_write == "zero_shot"
    assert prod.auto_calibrations == ("platt", "temperature")
    assert prod.auto_requires_audit is True and prod.allow_history_none is False
    assert prod.allow_auto_write is True
    assert dev.allow_auto_write is False and dev.allow_history_none is True
    assert (
        LEVEL_RANK["probe"]
        == LEVEL_RANK["head"]
        > LEVEL_RANK["calibrated"]
        > LEVEL_RANK["zero_shot"]
        > LEVEL_RANK["none"]
    )


def test_models_forbid_unknown_keys_and_bad_values() -> None:
    with pytest.raises(ValidationError):
        Thresholds(stat="p_fit", propose=0.5, min_weight=0.5, typo=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Thresholds(stat="p_fit", propose=1.5, min_weight=0.5)
    with pytest.raises(ValidationError, match="min_level 'none'"):
        Thresholds(stat="p_fit", propose=0.5, min_weight=0.5, min_level="none")
    with pytest.raises(ValidationError, match="only platt or temperature"):
        Profile(name="p", min_level_write="calibrated", auto_calibrations=("uncalibrated",))
    with pytest.raises(ValidationError):
        PolicyDefaults(tasks={}, profiles={}, extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Thresholds(stat="p_true", propose=0.5, min_weight=0.5)  # type: ignore[arg-type]


def test_loader_rejects_missing_task_wrong_stat_and_uncited_auto(tmp_path: Path) -> None:
    raw = _raw()

    def write(data: dict) -> Path:  # type: ignore[type-arg]
        p = tmp_path / "policy.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        return p

    missing = {**raw, "tasks": {k: v for k, v in raw["tasks"].items() if k != "term.fits"}}
    with pytest.raises(ValueError, match="no thresholds for active task"):
        load_policy_defaults(write(missing))
    wrong_stat = {**raw, "tasks": dict(raw["tasks"])}
    wrong_stat["tasks"]["term.fits"] = {**raw["tasks"]["term.fits"], "stat": "confidence"}
    with pytest.raises(ValueError, match=r"term\.fits must read p_fit"):
        load_policy_defaults(write(wrong_stat))
    uncited = {**raw, "tasks": dict(raw["tasks"])}
    uncited["tasks"]["term.fits"] = {**raw["tasks"]["term.fits"], "auto": 0.9}
    with pytest.raises(ValueError, match="without a cite"):
        load_policy_defaults(write(uncited))
    cited = {**raw, "tasks": dict(raw["tasks"])}
    cited["tasks"]["term.fits"] = {
        **raw["tasks"]["term.fits"],
        "auto": 0.9,
        "cite": "bench/results/2026-10-01/x.json#term.fits.probe.F7",
    }
    assert load_policy_defaults(write(cited)).thresholds("term.fits").auto == 0.9
    with pytest.raises(ValueError, match="must be a mapping"):
        load_policy_defaults(write([1, 2]))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        typo = {**raw, "profiles": {**raw["profiles"], "prod": {**raw["profiles"]["prod"], "x": 1}}}
        load_policy_defaults(write(typo))


def test_models_are_frozen() -> None:
    t = load_policy_defaults().thresholds("term.fits")
    with pytest.raises(ValidationError):
        t.auto = 0.5  # type: ignore[misc]
