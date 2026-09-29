"""The typed view of ``policy_defaults.yaml`` (DESIGN D8, D9; plan §4.7).

``policy_defaults.yaml`` holds per-task thresholds and the write profiles. It is the *one*
source of ``min_weight`` (D9): the label loaders, the fitters and the bench all call
:func:`min_weight_for`, so mesa-anyjev's split (0.6 in the policy, 0.5 in the bench) cannot
recur. ``outcome()``, ``masked()`` and the citation test that reads a numeric ``auto`` land with
``policy.py`` in M1; this module only loads and validates the file.

Every model forbids unknown keys, so a typo in the YAML fails loudly instead of silently
keeping a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mesa_clm.tasks import ACTIVE_TASKS, RANK_FIT_TASKS

Stat = Literal["p_fit", "confidence"]
Level = Literal["none", "zero_shot", "calibrated", "probe", "head"]
Calibration = Literal["none", "uncalibrated", "platt", "temperature"]

# Plan §4.5: ``probe`` and ``head`` rank equal; ``none`` never reaches any floor.
LEVEL_RANK: Final[dict[str, int]] = {
    "none": -1,
    "zero_shot": 0,
    "calibrated": 1,
    "probe": 2,
    "head": 2,
}

# The repo-root file (the sdist ships it there); a copy inside the package wins when a wheel
# carries one, so an installed mesa-clm needs no checkout.
_PACKAGED: Final = Path(__file__).resolve().parent / "policy_defaults.yaml"
_REPO_ROOT: Final = Path(__file__).resolve().parents[2] / "policy_defaults.yaml"
DEFAULTS_PATH: Final[Path] = _PACKAGED if _PACKAGED.exists() else _REPO_ROOT


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Thresholds(_Model):
    """One task's write thresholds.

    ``stat`` names the statistic the thresholds read: ``p_fit`` for a rank_fit task, ``confidence``
    (``max(probs)``, computed locally, D7) for a closed choice. ``auto`` is ``None`` for
    proposed-only; a number must cite a bench cell in ``cite`` (D8). ``min_level`` is the lowest
    tier that may reach ``auto`` (``zero_shot`` never does, D6). ``min_weight`` is the lowest label
    weight that enters a fit or a bench cell for this task (D9). ``risk`` is the Clopper-Pearson
    error bound a numeric ``auto`` must respect, and ``ols_rank_top`` the deepest OLS rank the
    degraded ``ols_rank`` method still proposes (D28).
    """

    stat: Stat
    auto: float | None = Field(default=None, ge=0.0, le=1.0)
    propose: float = Field(ge=0.0, le=1.0)
    margin: float = Field(default=0.0, ge=0.0, le=1.0)
    min_level: Level = "calibrated"
    min_weight: float = Field(ge=0.0, le=1.0)
    risk: float = Field(default=0.05, gt=0.0, lt=1.0)
    cite: str | None = None
    ols_rank_top: int = Field(default=1, ge=1)

    @field_validator("min_level")
    @classmethod
    def _never_none(cls, value: Level) -> Level:
        if value == "none":
            raise ValueError("min_level 'none' would let a rule or planner answer auto-write")
        return value


class Profile(_Model):
    """A write profile (plan §4.7): the floor a tier must reach to auto-write, which
    calibrations may auto, whether an auto needs a passing audit row, whether auto-writes are
    allowed at all and whether a run may proceed without a history backend."""

    name: str
    min_level_write: Level
    auto_calibrations: tuple[Calibration, ...] = ("platt", "temperature")
    auto_requires_audit: bool = True
    allow_auto_write: bool = True
    allow_history_none: bool = False

    @field_validator("auto_calibrations")
    @classmethod
    def _calibrated_only(cls, value: tuple[Calibration, ...]) -> tuple[Calibration, ...]:
        if any(c in ("none", "uncalibrated") for c in value):
            raise ValueError("only platt or temperature calibrations may auto-write (D6)")
        return value


class PolicyDefaults(_Model):
    """The whole file: thresholds per task and the named profiles."""

    tasks: dict[str, Thresholds]
    profiles: dict[str, Profile]

    def thresholds(self, task_id: str) -> Thresholds:
        """``KeyError`` when the task has no entry."""
        return self.tasks[task_id]

    def profile(self, name: str) -> Profile:
        return self.profiles[name]

    def min_weight(self, task_id: str) -> float:
        return self.tasks[task_id].min_weight


def load_policy_defaults(path: str | Path | None = None) -> PolicyDefaults:
    """Load and validate ``policy_defaults.yaml`` (``path=None``: the shipped file).

    Beyond the schema, the file must name every active task, give every rank_fit task
    ``stat: p_fit`` and every other task ``stat: confidence`` (a noul-style ``confidence`` would
    accept confident rejections; mesa-anyjev A7), and cite a cell for every numeric ``auto``.
    """
    file = Path(path).expanduser() if path is not None else DEFAULTS_PATH
    raw: Any = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: the policy file must be a mapping")
    profiles = {
        str(name): Profile(name=str(name), **(spec or {}))
        for name, spec in (raw.get("profiles") or {}).items()
    }
    policy = PolicyDefaults(tasks=raw.get("tasks") or {}, profiles=profiles)
    missing = [t for t in ACTIVE_TASKS if t not in policy.tasks]
    if missing:
        raise ValueError(f"{file}: no thresholds for active task(s) {', '.join(missing)}")
    for task_id, t in policy.tasks.items():
        expected: Stat = "p_fit" if task_id in RANK_FIT_TASKS else "confidence"
        if t.stat != expected:
            raise ValueError(f"{file}: {task_id} must read {expected}, not {t.stat}")
        if t.auto is not None and not t.cite:
            raise ValueError(f"{file}: {task_id} has auto={t.auto} without a cite (D8)")
    return policy


def min_weight_for(task_id: str, path: str | Path | None = None) -> float:
    """The single source of a task's ``min_weight`` (D9); ``KeyError`` for a task without
    thresholds."""
    return load_policy_defaults(path).min_weight(task_id)
