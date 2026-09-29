"""Frozen vocabularies shared by the decision records, the policy and the sidecar CHECKs.

One module so that ``providers.base`` (the Pydantic validators), ``policy`` (the outcome rules)
and ``provenance`` (the SQL CHECK constraints in both dialects) can never drift apart: the DDL
is generated from these tuples and a test asserts it. Values follow DESIGN D6 (levels and
calibrations), D10/D28 (outcomes), D12/D13 (write status), D21 (label sources, ``via``) and
§4.5 of the plan (methods, shapes).
"""

from __future__ import annotations

from typing import Final, Literal, get_args

Level = Literal["none", "zero_shot", "calibrated", "probe", "head"]
Calibration = Literal["none", "uncalibrated", "platt", "temperature"]
Method = Literal[
    "clm", "fake", "rule", "planner", "ols_rank", "claude:structured_output", "unavailable"
]
Shape = Literal["rank_fit", "choice"]
Outcome = Literal[
    "auto", "proposed", "human", "escalated", "abstain", "rejected", "rule", "decider_unavailable"
]
WriteStatus = Literal[
    "proposed",
    "accepted",
    "dry_run",
    "writing",
    "written",
    "spooled",
    "reverted",
    "mirror_failed",
    "local_only",
]
AcceptedBy = Literal["policy", "human", "agent"]
Via = Literal["elicitation", "cli", "tool"]
LabelSource = Literal[
    "curator",
    "curator_implicit",
    "agent_pick",
    "consensus_all",
    "consensus_majority",
    "consensus_negative",
    "teacher",
    "teacher_implicit",
    "gold",
]
RunStatus = Literal["running", "decided", "applied", "partial", "failed", "abandoned"]

LEVELS: Final[tuple[str, ...]] = get_args(Level)
CALIBRATIONS: Final[tuple[str, ...]] = get_args(Calibration)
METHODS: Final[tuple[str, ...]] = get_args(Method)
SHAPES: Final[tuple[str, ...]] = get_args(Shape)
OUTCOMES: Final[tuple[str, ...]] = get_args(Outcome)
WRITE_STATUSES: Final[tuple[str, ...]] = get_args(WriteStatus)
ACCEPTED_BY: Final[tuple[str, ...]] = get_args(AcceptedBy)
VIAS: Final[tuple[str, ...]] = get_args(Via)
LABEL_SOURCES: Final[tuple[str, ...]] = get_args(LabelSource)
RUN_STATUSES: Final[tuple[str, ...]] = get_args(RunStatus)

# Rank order used by the policy (D6): a probe and a head are equally trusted tiers.
LEVEL_RANK: Final[dict[str, int]] = {
    "none": -1,
    "zero_shot": 0,
    "calibrated": 1,
    "probe": 2,
    "head": 2,
}

# Methods whose records carry no distribution (``probs IS NULL``, calibration ``none``).
METHODS_WITHOUT_PROBS: Final[frozenset[str]] = frozenset(
    {"rule", "planner", "ols_rank", "claude:structured_output", "unavailable"}
)


def sql_in_list(values: tuple[str, ...]) -> str:
    """Render a vocabulary as a SQL ``IN (...)`` list for a CHECK constraint (both dialects)."""
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)
