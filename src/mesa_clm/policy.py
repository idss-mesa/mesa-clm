"""Write policy: one decision record in, one outcome out (DESIGN D6, D8, D9, D10, D24, D25, D28;
plan §4.7).

Ported from mesa-anyjev ``policy.py`` (``6159281``; DESIGN U1). The thresholds and profiles are
no longer parsed here: :mod:`mesa_clm.policy_defaults` owns ``policy_defaults.yaml`` and is the
single source of ``min_weight`` (D9); :class:`Policy` is a thin wrapper that looks thresholds up
by ``task_id`` and adds the citation hook. Changes against the source:

* **Statistic.** A threshold reads ``p_fit`` of the winning candidate for a ``rank_fit`` and
  ``confidence = max(probs)`` for a closed ``choice`` (D7); CLM's own ``clm_confidence`` is never
  read. A threshold whose ``stat`` disagrees with the record's shape is a programming error
  (:class:`PolicyError`), not an abstain.
* **Explicit method branches (step 0).** ``method='rule'`` is the ``rule`` outcome;
  ``method='ols_rank'`` (D28) is ``proposed`` while the OLS rank is within
  ``t.ols_rank_top`` and ``abstain`` below it, and never ``auto`` whatever the thresholds, the
  profile or the citation say. Without this branch step 1 would abstain on every ``ols_rank``
  record (its probs are NULL) and the K1/K2(c) pivots would emit nothing.
* **Abstains (step 1).** No distribution, no answer, a truncated context (D23), the abstain
  anchor winning (D2) or a fully masked option set.
* **auto (step 2)** needs every gate: a numeric ``t.auto``; a profile that allows auto-writes
  at all (``dev`` does not); a level other than ``zero_shot`` (D6) at or above both floors,
  ``max(LEVEL_RANK[t.min_level], LEVEL_RANK[profile.min_level_write])``; a calibration the
  profile lists; ``stat >= t.auto``; ``margin >= t.margin``; and a citation that a
  :class:`CitationValidator` accepted as valid and fingerprint-matched (D8). The shipped
  validator (:class:`RefuseAllCitations`) refuses every cite until M4 implements the check, so
  a numeric ``auto`` cannot write before the bench backs it; the module-level :func:`outcome`
  refuses unless the caller passes ``cite_ok=True``.
* **proposed / abstain (steps 3-4)** at ``stat >= t.propose``, else ``abstain``
  (``below_propose``).
* **Demotions only.** The group margin (the ``p_fit`` gap to the runner-up candidate) and the
  unbenched specificity heuristic (D24) can turn an ``auto`` into ``proposed``, never the other
  way (:func:`demote_for_group_margin`, :func:`cap_at_proposed`); no citation is needed to be
  more careful.
* **Two phases (D10).** Annotate stores a policy ``auto`` as a link ``accepted`` with
  ``accepted_by='policy'`` (:func:`link_status`); ``apply`` never calls this module.

:func:`masked` is the pure masking step (mesa-anyjev's 2048 pattern) re-exported over
:func:`mesa_clm.providers.base.apply_mask`, and :func:`dedup_and_cap` is the ``avu.keep`` rule
(D25: exact-triple dedup, then a cap of 25 by ``p_fit``) that replaced the unlearnable question.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from mesa_clm.policy_defaults import (
    PolicyDefaults,
    Profile,
    Thresholds,
    load_policy_defaults,
)
from mesa_clm.providers.base import DecisionRecord, apply_mask, aspect_keep
from mesa_clm.vocab import LEVEL_RANK, AcceptedBy, Outcome, WriteStatus

if TYPE_CHECKING:
    from mesa_clm.config import PolicyConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ABSTAIN_REASONS",
    "AUTO_BLOCKERS",
    "CITE_RE",
    "MAX_AVUS",
    "SPECIFICITY_DELTA",
    "CitationValidator",
    "CiteCheck",
    "CiteRef",
    "Policy",
    "PolicyDefaults",
    "PolicyError",
    "Profile",
    "RefuseAllCitations",
    "Thresholds",
    "Verdict",
    "cap_at_proposed",
    "dedup_and_cap",
    "demote_for_group_margin",
    "fingerprint_mismatches",
    "group_margin",
    "link_status",
    "load_policy_defaults",
    "masked",
    "masked_for_aspect",
    "outcome",
    "parse_cite",
    "specific_child",
    "statistic",
    "verdict",
]

T = TypeVar("T")

# D24: a child term replaces its parent at p_fit(child) >= p_fit(parent) + 0.10 (the config's
# ``policy.specificity_delta`` defaults to the same number).
SPECIFICITY_DELTA: Final[float] = 0.10
# D25: at most 25 AVUs per card survive the keep rule (the config's ``policy.max_avus``).
MAX_AVUS: Final[int] = 25
# Float slack for the specificity boundary (0.2 + 0.1 is 0.30000000000000004).
_EPS: Final[float] = 1e-12

# Why the policy abstained (the sidecar's ``decisions.reason``). A provider-level reason
# (``providers.base.REASONS``: decider_unavailable, numeric_underflow, ...) is passed through
# when the record carries one, so this tuple lists only the reasons the policy adds.
ABSTAIN_REASONS: Final[tuple[str, ...]] = (
    "truncated",  # the context exceeds the encoder window (D23)
    "mask_empty",  # the mask removed every candidate (plan §4.2 Q3)
    "no_answer",  # answer_index < 0 without a provider reason
    "no_distribution",  # probs IS NULL (a planner/Claude record): nothing to threshold
    "anchor_won",  # the abstain anchor out-ranked every candidate (D2)
    "no_statistic",  # the threshold's statistic is missing or not finite
    "below_propose",  # stat < t.propose
    "no_rank",  # an ols_rank record without its rank (the record invariant forbids it)
    "ols_rank_below_top",  # an ols_rank record deeper than t.ols_rank_top (D28)
)

# The step-2 gates, in evaluation order; ``Verdict.auto_blockers`` names the ones that failed.
AUTO_BLOCKERS: Final[tuple[str, ...]] = (
    "auto_null",  # t.auto is None: proposed-only (every shipped task)
    "ols_rank",  # the degraded method never autos (D28)
    "auto_write_disabled",  # the profile forbids auto-writes (dev)
    "zero_shot",  # zero_shot never autos (D6)
    "level_below_floor",  # LEVEL_RANK[level] < max(t.min_level, profile.min_level_write)
    "calibration_not_allowed",  # calibration not in profile.auto_calibrations
    "below_auto",  # stat < t.auto
    "margin_below",  # margin < t.margin (or no margin)
    "cite_refused",  # no valid, fingerprint-matched citation (D8)
)

# plan §4.7: ``bench/results/<date>/<file>.json#<task>.<tier>.<framing>``; the fragment is
# ``bench.results.cell_key``.
CITE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<path>bench/results/\d{4}-\d{2}-\d{2}/[\w.\-]+\.json)"
    r"#(?P<task>[\w\-]+)\.(?P<tier>[a-z_]+)\.(?P<framing>[\w\-]+)$"
)

# The record fields a cited cell's fingerprint must equal (D5). ``serving_lock_sha`` is not on
# a record; the M4 validator compares it against the live Fingerprint instead.
_RECORD_FP_FIELDS: Final[tuple[str, ...]] = ("encoder_fp", "clm_model_fp", "schema_sha256")


class PolicyError(ValueError):
    """The policy was asked something it cannot answer honestly: thresholds for another shape,
    a task or profile without an entry."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- citations (D8) ------------------------------------------------------------------------------


class CiteRef(NamedTuple):
    """A parsed ``cite``: the results file and the ``<task>.<tier>.<framing>`` cell in it."""

    path: str
    task: str
    tier: str
    framing: str


def parse_cite(cite: str | None) -> CiteRef | None:
    """The parts of a well-formed ``cite`` (:data:`CITE_RE`), else ``None``. Syntax only: that
    the file exists and the cell qualifies is the validator's job (M4)."""
    if not cite:
        return None
    m = CITE_RE.match(cite)
    if m is None:
        return None
    return CiteRef(m["path"], m["task"], m["tier"], m["framing"])


class CiteCheck(_Frozen):
    """A validator's answer: ``ok`` only for a valid, fingerprint-matched cell; ``reason`` says
    why not (or what was checked)."""

    ok: bool
    reason: str = ""


@runtime_checkable
class CitationValidator(Protocol):
    """The D8 hook: is ``thresholds.cite`` a valid citation for ``record``?

    ``validator(task_id, thresholds, record)`` must return ``CiteCheck(ok=True)`` only when the
    cited cell exists and is ``loco``, ``pre_registered``, ``selection='nested'``, not
    ``exploratory``, ``servable``, with ``n_folds >= 5``, ``teacher_in_test`` false, ``masked``
    as served, the committed snapshot's ``labels_sha256``, the negatives guard,
    ``beats_lookup_novel``, ``thresholds.auto >= threshold_cp[risk]``, and a ``question_key``
    and fingerprint equal to the record's (:func:`fingerprint_mismatches`). M4 implements it
    against ``bench.results``; until then :class:`RefuseAllCitations` is the only validator
    shipped. The policy calls it only once every other auto gate has passed.
    """

    def __call__(
        self, task_id: str, thresholds: Thresholds, record: DecisionRecord, /
    ) -> CiteCheck: ...


class RefuseAllCitations:
    """The default :class:`CitationValidator`: refuses every cite (the M4 check does not exist
    yet), with a reason that says whether the cite is at least well-formed."""

    def __call__(
        self, task_id: str, thresholds: Thresholds, record: DecisionRecord, /
    ) -> CiteCheck:
        if thresholds.auto is None:
            return CiteCheck(ok=False, reason=f"{task_id}: no numeric auto, nothing to cite")
        if not thresholds.cite:
            return CiteCheck(ok=False, reason=f"{task_id}: auto={thresholds.auto} cites nothing")
        if parse_cite(thresholds.cite) is None:
            return CiteCheck(
                ok=False,
                reason=f"{task_id}: cite {thresholds.cite!r} is not "
                "bench/results/<date>/<file>.json#<task>.<tier>.<framing>",
            )
        return CiteCheck(
            ok=False,
            reason=f"{task_id}: citation validation lands in M4 (D8); refused until then",
        )


def fingerprint_mismatches(
    record: DecisionRecord,
    *,
    question_key: str | None,
    fingerprint: dict[str, str] | None,
) -> list[str]:
    """What differs between a cited cell's ``question_key``/``fingerprint`` and ``record``'s
    (empty when they match; D5, K4). A cell without a fingerprint (a baseline) never matches."""
    diffs: list[str] = []
    if question_key != record.question_key:
        diffs.append(f"question_key: cell {question_key!r}, record {record.question_key!r}")
    if fingerprint is None:
        diffs.append("fingerprint: the cell has none (a baseline cell is never servable)")
        return diffs
    for name in _RECORD_FP_FIELDS:
        theirs = fingerprint.get(name)
        mine = getattr(record, name)
        if theirs != mine:
            diffs.append(f"{name}: cell {theirs!r}, record {mine!r}")
    return diffs


# -- the outcome (plan §4.7) --------------------------------------------------------------------


class Verdict(_Frozen):
    """An outcome with its explanation: ``reason`` is why the record abstained (``None``
    otherwise), ``stat`` the statistic the thresholds read, ``auto_blockers`` the step-2 gates
    that kept it from ``auto`` (``()`` for an auto and for records that never reached step 2;
    the citation is checked only once every other gate passed) and ``cite`` the validator's
    reason when it was consulted."""

    outcome: Outcome
    reason: str | None = None
    stat: float | None = None
    auto_blockers: tuple[str, ...] = ()
    cite: str | None = None


def statistic(record: DecisionRecord) -> float | None:
    """The number a threshold reads: ``p_fit`` of the answered candidate for a ``rank_fit``,
    ``confidence = max(probs)`` for a ``choice`` (D7). ``None`` when there is none."""
    if record.shape == "rank_fit":
        return record.answer_p_fit
    return record.confidence


def _expected_stat(record: DecisionRecord) -> str:
    return "p_fit" if record.shape == "rank_fit" else "confidence"


def _fully_masked(record: DecisionRecord) -> bool:
    flags = record.masked
    if flags is None:
        return False
    return all(m for i, m in enumerate(flags) if i != record.anchor_index)


def _abstain_reason(record: DecisionRecord) -> str | None:
    """Step 1: why the record cannot be proposed at all, most specific reason first."""
    if record.truncated:
        return "truncated"
    if _fully_masked(record):
        return "mask_empty"
    if record.answer_index < 0:
        return record.reason or "no_answer"
    if record.probs is None:
        return record.reason or "no_distribution"
    if record.anchor_won:
        return "anchor_won"
    return None


def _ols_rank(record: DecisionRecord, t: Thresholds) -> Verdict:
    """Step 0, D28: the OLS top ``t.ols_rank_top`` is proposed, deeper ranks abstain; never auto
    (the sidecar CHECK repeats it)."""
    if record.rank is None:
        return Verdict(outcome="abstain", reason="no_rank")
    if record.answer_index < 0:
        return Verdict(outcome="abstain", reason=record.reason or "no_answer")
    if record.anchor_won:
        return Verdict(outcome="abstain", reason="anchor_won")
    if record.rank <= t.ols_rank_top:
        return Verdict(outcome="proposed", auto_blockers=("ols_rank",))
    return Verdict(outcome="abstain", reason="ols_rank_below_top", auto_blockers=("ols_rank",))


def _auto_blockers(
    record: DecisionRecord,
    t: Thresholds,
    profile: Profile,
    stat: float,
) -> list[str]:
    """The step-2 gates other than the citation that ``record`` fails."""
    if t.auto is None:
        return ["auto_null"]
    blockers: list[str] = []
    if not profile.allow_auto_write:
        blockers.append("auto_write_disabled")
    if record.level == "zero_shot":
        blockers.append("zero_shot")
    # ``t.min_level`` is never ``none`` (policy_defaults refuses it), so the floor is at least
    # zero_shot and a level-``none`` record always fails it.
    floor = max(LEVEL_RANK[t.min_level], LEVEL_RANK[profile.min_level_write])
    if LEVEL_RANK[record.level] < floor:
        blockers.append("level_below_floor")
    # The Profile validator already keeps none/uncalibrated out of auto_calibrations; checked
    # again so an unvalidated profile cannot let a raw distribution through.
    if record.calibration not in profile.auto_calibrations or record.calibration in (
        "none",
        "uncalibrated",
    ):
        blockers.append("calibration_not_allowed")
    if stat < t.auto:
        blockers.append("below_auto")
    margin = record.margin
    if margin is None or not math.isfinite(margin) or margin < t.margin:
        blockers.append("margin_below")
    return blockers


def _decide(
    record: DecisionRecord,
    t: Thresholds,
    profile: Profile,
    check_cite: Callable[[], CiteCheck],
) -> Verdict:
    # Step 0: the explicit method branches.
    if record.method == "rule":
        return Verdict(outcome="rule")
    expected = _expected_stat(record)
    if t.stat != expected:
        raise PolicyError(
            f"{record.task_id} is a {record.shape}: its thresholds must read {expected}, "
            f"not {t.stat} (thresholds for another task?)"
        )
    if record.method == "ols_rank":
        return _ols_rank(record, t)
    # Step 1: nothing to threshold.
    reason = _abstain_reason(record)
    if reason is not None:
        return Verdict(outcome="abstain", reason=reason)
    stat = statistic(record)
    if stat is None or not math.isfinite(stat):
        return Verdict(outcome="abstain", reason="no_statistic")
    # Step 2: auto only through every gate, the citation last (it may read files).
    blockers = _auto_blockers(record, t, profile, stat)
    cite_reason: str | None = None
    if not blockers:
        check = check_cite()
        cite_reason = check.reason or None
        if not check.ok:
            blockers.append("cite_refused")
    if not blockers:
        return Verdict(outcome="auto", stat=stat, cite=cite_reason)
    # Steps 3-4.
    if stat >= t.propose:
        return Verdict(
            outcome="proposed", stat=stat, auto_blockers=tuple(blockers), cite=cite_reason
        )
    return Verdict(
        outcome="abstain",
        reason="below_propose",
        stat=stat,
        auto_blockers=tuple(blockers),
        cite=cite_reason,
    )


def _caller_cite(cite_ok: bool | None) -> CiteCheck:
    if cite_ok is True:
        return CiteCheck(ok=True, reason="validated by the caller")
    if cite_ok is False:
        return CiteCheck(ok=False, reason="the caller's citation check failed")
    return CiteCheck(ok=False, reason="no validated citation supplied (D8)")


def verdict(
    record: DecisionRecord,
    t: Thresholds,
    profile: Profile,
    *,
    cite_ok: bool | None = None,
) -> Verdict:
    """:func:`outcome` with its reason, statistic and failed auto gates (module docstring).

    ``cite_ok`` is the caller's citation verdict for ``t.cite`` against this record: only
    ``True`` lets a record ``auto``; ``None`` (nothing validated) and ``False`` refuse it.
    """
    return _decide(record, t, profile, lambda: _caller_cite(cite_ok))


def outcome(
    record: DecisionRecord,
    t: Thresholds,
    profile: Profile,
    *,
    cite_ok: bool | None = None,
) -> Outcome:
    """The outcome of one decision under ``t`` and ``profile`` (plan §4.7 steps 0-4)."""
    return verdict(record, t, profile, cite_ok=cite_ok).outcome


# -- demotions (group margin, D24) ------------------------------------------------------------


def cap_at_proposed(result: Outcome) -> Outcome:
    """``auto`` becomes ``proposed``; every other outcome is unchanged (D24: an unbenched
    heuristic may only ever propose)."""
    return "proposed" if result == "auto" else result


def demote_for_group_margin(result: Outcome, group_margin: float | None, t: Thresholds) -> Outcome:
    """Demote an ``auto`` whose candidate group is too close to call: ``group_margin < t.margin``
    (or unmeasured, ``None``) turns it into ``proposed``. Nothing else changes: the group margin
    can only make the policy more careful, so it needs no citation."""
    if result != "auto":
        return result
    if group_margin is None or not math.isfinite(group_margin) or group_margin < t.margin:
        return "proposed"
    return result


def group_margin(record: DecisionRecord) -> float | None:
    """``p_fit`` of the answered candidate minus the best ``p_fit`` among the other in-play
    candidates (``0`` when it stands alone), as the sidecar's ``decision_groups.group_margin``.
    ``None`` unless a rank_fit answered a candidate."""
    p_fit = record.p_fit
    if record.shape != "rank_fit" or p_fit is None or record.answer_index < 0:
        return None
    if record.anchor_won:
        return None
    flags = record.masked or [False] * record.k
    others = [
        p
        for i, p in enumerate(p_fit)
        if i not in (record.answer_index, record.anchor_index) and not flags[i]
    ]
    return p_fit[record.answer_index] - (max(others) if others else 0.0)


def specific_child(
    record: DecisionRecord, parent_key: str, *, delta: float = SPECIFICITY_DELTA
) -> int | None:
    """D24: in one rank over ``{parent, children, anchor}``, the index of the child that
    replaces the parent (the best in-play child with ``p_fit >= p_fit(parent) + delta``), else
    ``None``. Its outcome is at most ``proposed`` (:func:`cap_at_proposed`)."""
    p_fit = record.p_fit
    if record.shape != "rank_fit" or p_fit is None:
        return None
    if parent_key not in record.options:
        raise PolicyError(f"specificity: the parent {parent_key!r} is not an option of the rank")
    parent = record.options.index(parent_key)
    flags = record.masked or [False] * record.k
    best: int | None = None
    for i in range(record.k):
        if i in (parent, record.anchor_index) or flags[i]:
            continue
        if best is None or p_fit[i] > p_fit[best]:
            best = i
    if best is None or p_fit[best] < p_fit[parent] + delta - _EPS:
        return None
    return best


# -- links (D10) and the keep rule (D25) ------------------------------------------------------------


def link_status(result: Outcome) -> tuple[WriteStatus, AcceptedBy | None] | None:
    """How annotate stores an outcome's AVU link (D10): a policy ``auto`` is ``accepted`` by
    ``policy`` (apply then writes it without deciding), ``proposed`` waits for a reviewer, and
    every other outcome makes no link."""
    if result == "auto":
        return "accepted", "policy"
    if result == "proposed":
        return "proposed", None
    return None


def dedup_and_cap(
    items: Sequence[T],
    *,
    triple: Callable[[T], tuple[str, str, str]],
    p_fit: Callable[[T], float | None],
    cap: int = MAX_AVUS,
) -> list[T]:
    """The ``avu.keep`` rule (D25): one item per exact ``(attribute, value, unit)`` triple (the
    highest ``p_fit``, the first on a tie), then the ``cap`` best by ``p_fit`` (``None`` last,
    input order within ties)."""
    if cap < 0:
        raise ValueError("cap must be >= 0")

    def score(item: T) -> float:
        p = p_fit(item)
        return p if p is not None and math.isfinite(p) else -math.inf

    kept: dict[tuple[str, str, str], tuple[int, T]] = {}
    for i, item in enumerate(items):
        key = triple(item)
        seen = kept.get(key)
        if seen is None:
            kept[key] = (i, item)
        elif score(item) > score(seen[1]):
            kept[key] = (seen[0], item)
    ranked = sorted(kept.values(), key=lambda pair: (-score(pair[1]), pair[0]))
    return [item for _, item in ranked[:cap]]


# -- masking (plan §4.2 Q3) ---------------------------------------------------------------------


def masked(record: DecisionRecord, keep: Sequence[bool] | npt.NDArray[np.bool_]) -> DecisionRecord:
    """Remove the options ``keep`` marks out of play, after scoring: probability 0, renormalise,
    ``diagnostics['masked_mass']``; a fully masked record abstains (``reason='mask_empty'``).
    The pure step is :func:`mesa_clm.providers.base.apply_mask`; this is its policy name."""
    return apply_mask(record, [bool(x) for x in keep])


def masked_for_aspect(
    record: DecisionRecord, aspect: str, in_play: frozenset[str] | None = None
) -> DecisionRecord:
    """:func:`masked` by an aspect's allowed ontologies (``column.ontology_fits``), optionally
    intersected with the ontologies in play."""
    return apply_mask(record, aspect_keep(record.options, aspect, in_play))


# -- the wrapper ------------------------------------------------------------------------------


class Policy:
    """Thresholds from :mod:`mesa_clm.policy_defaults` (the one source, D9), a write profile and
    the citation hook, applied per record by its ``task_id``.

    ``thresholds`` is the loaded :class:`~mesa_clm.policy_defaults.PolicyDefaults`; ``profile``
    a profile name in it (or a :class:`~mesa_clm.policy_defaults.Profile`); ``validator`` the
    :class:`CitationValidator` (default :class:`RefuseAllCitations`, so no numeric ``auto``
    writes until M4 validates its cell). A validator that raises counts as a refusal.
    """

    def __init__(
        self,
        thresholds: PolicyDefaults,
        *,
        profile: str | Profile = "prod",
        validator: CitationValidator | None = None,
    ) -> None:
        self.defaults = thresholds
        if isinstance(profile, Profile):
            self.profile = profile
        else:
            try:
                self.profile = thresholds.profile(profile)
            except KeyError:
                raise PolicyError(
                    f"no profile {profile!r} in the policy file "
                    f"(known: {', '.join(sorted(thresholds.profiles)) or 'none'})"
                ) from None
        self.validator: CitationValidator = validator or RefuseAllCitations()

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        profile: str | Profile = "prod",
        validator: CitationValidator | None = None,
    ) -> Policy:
        """:func:`~mesa_clm.policy_defaults.load_policy_defaults` (``None``: the shipped file)
        wrapped with ``profile`` and ``validator``."""
        return cls(load_policy_defaults(path), profile=profile, validator=validator)

    @classmethod
    def from_config(
        cls, section: PolicyConfig, *, validator: CitationValidator | None = None
    ) -> Policy:
        """From the config's ``policy`` section: its ``policy_path`` (``None``: the shipped
        file) and ``profile``."""
        path = Path(section.policy_path).expanduser() if section.policy_path else None
        return cls.load(path, profile=section.profile, validator=validator)

    def thresholds(self, task_id: str) -> Thresholds:
        """``task_id``'s thresholds; :class:`PolicyError` for a task without an entry."""
        try:
            return self.defaults.thresholds(task_id)
        except KeyError:
            raise PolicyError(f"no thresholds for task {task_id!r} in the policy file") from None

    def min_weight(self, task_id: str) -> float:
        """The single source of ``task_id``'s ``min_weight`` (D9)."""
        return self.thresholds(task_id).min_weight

    def _cite(self, task_id: str, t: Thresholds, record: DecisionRecord) -> CiteCheck:
        try:
            return self.validator(task_id, t, record)
        except Exception as exc:  # a broken validator must fail closed, never auto
            logger.warning("citation validator failed for %s (%s)", task_id, type(exc).__name__)
            return CiteCheck(ok=False, reason=f"the validator raised {type(exc).__name__}")

    def verdict(self, record: DecisionRecord, *, profile: Profile | None = None) -> Verdict:
        """:func:`verdict` under this policy: a ``rule`` record needs no thresholds, every other
        record reads its task's, and the citation goes through the validator."""
        if record.method == "rule":
            return Verdict(outcome="rule")
        t = self.thresholds(record.task_id)
        return _decide(
            record, t, profile or self.profile, lambda: self._cite(record.task_id, t, record)
        )

    def outcome(self, record: DecisionRecord, *, profile: Profile | None = None) -> Outcome:
        """The outcome of ``record`` under this policy (plan §4.7)."""
        return self.verdict(record, profile=profile).outcome

    def demote_for_group_margin(
        self, result: Outcome, group_margin: float | None, task_id: str
    ) -> Outcome:
        """:func:`demote_for_group_margin` with ``task_id``'s thresholds."""
        return demote_for_group_margin(result, group_margin, self.thresholds(task_id))
