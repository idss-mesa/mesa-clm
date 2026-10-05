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
  :class:`CitationValidator` accepted as valid and fingerprint-matched (D8). The default
  validator is :class:`CellCitationValidator` (M4): it reads the cited results file under
  ``policy.results_root`` and checks every field of the citation test (its docstring);
  :class:`RefuseAllCitations` refuses every cite. The module-level :func:`outcome` refuses
  unless the caller passes ``cite_ok=True``.
* **audit (step 2, prod).** Under a profile with ``auto_requires_audit`` an ``auto`` also
  needs a passing ``audits`` row for the record's ``(task_key, artifact_version)``, read from
  the :class:`AuditCheck` the caller supplies (the sidecar's, :class:`StoreAuditCheck`);
  without one the verdict is ``proposed`` with reason ``audit_required`` (M4; ``apply`` is
  M3's, so the pipeline records the blocker now).
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
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from mesa_clm.clm.fingerprint import Fingerprint
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
    "DEFAULT_RESULTS_HOME",
    "DEMOTION_REASONS",
    "FOLD_AGREEMENT_MIN",
    "GUARD_MIN",
    "MAX_AVUS",
    "MIN_NESTED_FOLDS",
    "SNAPSHOTS_DIR",
    "SPECIFICITY_DELTA",
    "AuditCheck",
    "CellCitationValidator",
    "CitationValidator",
    "CiteCheck",
    "CiteRef",
    "Policy",
    "PolicyDefaults",
    "PolicyError",
    "Profile",
    "RefuseAllCitations",
    "StoreAuditCheck",
    "Thresholds",
    "Verdict",
    "cap_at_proposed",
    "dedup_and_cap",
    "default_results_root",
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
    "audit_required",  # the profile needs a passing audits row and none exists (plan §4.7, M4)
)

# The reasons a demotion from auto records on a proposed row (``decisions.reason``; M4).
DEMOTION_REASONS: Final[tuple[str, ...]] = ("audit_required",)

# The citation test's constants (plan §4.7, PR "Citation test"; M4, R6).
MIN_NESTED_FOLDS: Final[int] = 5  # n_folds >= 5
FOLD_AGREEMENT_MIN: Final[int] = 5  # fold_choices agree with the served configuration in >= 5/7
GUARD_MIN: Final[int] = 30  # counts.n_neg (rank_fit) / n_nonmodal (choice) >= 30
# The committed snapshots a cited cell's labels_sha256 must be one of (hashed under results_root).
SNAPSHOTS_DIR: Final[str] = "bench/snapshots"
# Where cites resolve outside a src/ checkout (``policy.results_root`` unset).
DEFAULT_RESULTS_HOME: Final[str] = "~/.mesa/clm/results"

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


def default_results_root() -> Path:
    """Where a cite's ``bench/results/...`` path is resolved when ``policy.results_root`` is
    unset (M4): the checkout root when :mod:`mesa_clm` is imported from a ``src/`` checkout
    (the directory holding ``src/`` and ``pyproject.toml``), else ``~/.mesa/clm/results``."""
    here = Path(__file__).resolve()
    if len(here.parents) >= 3 and here.parents[1].name == "src":
        root = here.parents[2]
        if (root / "pyproject.toml").is_file():
            return root
    return Path(DEFAULT_RESULTS_HOME).expanduser()


def _snapshot_hashes(root: Path, snapshots_dir: str) -> dict[str, str]:
    """``{labels_sha256: file name}`` of the committed snapshots under ``root`` (the sha256 of
    each ``*.parquet``'s bytes, D30; the files are hashed, never opened for their labels)."""
    import hashlib

    out: dict[str, str] = {}
    directory = root / snapshots_dir
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.parquet")):
        out[hashlib.sha256(path.read_bytes()).hexdigest()] = path.name
    return out


class CellCitationValidator:
    """The D8 citation test (plan §4.7, PR "Citation test"; M4, R6): a numeric ``auto`` cites a
    pre-registered nested LOCO cell that qualifies, field by field, for the record it is applied
    to. ``results_root`` resolves the cite's path (default :func:`default_results_root`);
    ``labels_sha256s`` is the set of committed snapshot hashes (default: the ``*.parquet`` under
    ``<results_root>/bench/snapshots``, hashed once); ``live`` is the live
    :class:`~mesa_clm.clm.fingerprint.Fingerprint` the cell must equal, ``serving_lock_sha``
    included (bound by the pipeline through :meth:`bind`; unbound, every cite is refused:
    ``serving_lock_sha`` is not on a record and cannot be checked).

    Every check, in order (the first failure is the reason): the cite is well-formed; the
    results file exists and loads; the cell exists and is the record's task's; ``loco``,
    ``pre_registered``, ``selection == "nested"``, not ``exploratory``, ``servable``,
    ``n_folds >= 5``, ``teacher_in_test is False``; the masking is the record's framing's, read
    as M2's cells record it (``bench.cells.assemble_cell``: every cell has ``masked: true`` and
    ``mask`` the task's option restriction, ``"aspect"`` or ``None``): ``cell.masked is True``
    and ``cell.mask == framing.mask_rule`` (``None == None`` for a task without a mask); a cell
    with ``masked: false`` or a ``mask`` other than the framing's rule is refused;
    ``labels_sha256`` is a committed snapshot's; the guards
    ``counts.n_neg >= 30`` (rank_fit) or ``n_nonmodal >= 30`` (choice); ``question_key`` equals
    the record's; the fingerprint equals the record's fields and the live fingerprint;
    ``fold_choices`` agree with the served configuration in at least 5 folds (a calibrated
    cell: the fold's arm is the cell's; a probe cell: the fold's ``model`` is the record's and
    its ``spec`` the record's ``feature_spec``); ``baselines.beats_lookup_novel is True``;
    ``thresholds.auto >= metrics.threshold_cp[risk]`` (``None`` refuses). The cell's tier must
    be the record's level (the brief's "tier rank >= level" read at its most conservative:
    equality, which satisfies it), checked right after the task so a mismatch is named as such.
    """

    def __init__(
        self,
        results_root: str | Path | None = None,
        *,
        live: Fingerprint | None = None,
        labels_sha256s: Collection[str] | None = None,
        snapshots_dir: str = SNAPSHOTS_DIR,
    ) -> None:
        self.results_root = (
            Path(results_root).expanduser() if results_root is not None else default_results_root()
        )
        self.live = live
        self.snapshots_dir = snapshots_dir
        self._given: frozenset[str] | None = (
            None if labels_sha256s is None else frozenset(labels_sha256s)
        )
        self._hashed: dict[str, str] | None = None
        self._loaded: dict[str, Any] = {}

    def bind(self, live: Fingerprint | None) -> None:
        """Name the live fingerprint the cells must equal (the pipeline binds the provider's)."""
        self.live = live

    def committed_snapshots(self) -> frozenset[str]:
        if self._given is not None:
            return self._given
        if self._hashed is None:
            self._hashed = _snapshot_hashes(self.results_root, self.snapshots_dir)
        return frozenset(self._hashed)

    def _results(self, path: str) -> Any:
        from mesa_clm.bench.results import load_results

        if path not in self._loaded:
            self._loaded[path] = load_results(self.results_root / path)
        return self._loaded[path]

    def __call__(
        self, task_id: str, thresholds: Thresholds, record: DecisionRecord, /
    ) -> CiteCheck:
        why = self._refusal(task_id, thresholds, record)
        if why is not None:
            return CiteCheck(ok=False, reason=f"{task_id}: {why}")
        return CiteCheck(ok=True, reason=f"{task_id}: {thresholds.cite} qualifies (D8)")

    def _refusal(self, task_id: str, t: Thresholds, record: DecisionRecord) -> str | None:
        if t.auto is None:
            return "no numeric auto, nothing to cite"
        if not t.cite:
            return f"auto={t.auto} cites nothing"
        ref = parse_cite(t.cite)
        if ref is None:
            return (
                f"cite {t.cite!r} is not bench/results/<date>/<file>.json#<task>.<tier>.<framing>"
            )
        path = self.results_root / ref.path
        if not path.is_file():
            return f"{ref.path}: no such results file under {self.results_root}"
        try:
            results = self._results(ref.path)
        except Exception as exc:
            return f"{ref.path}: not a results file ({type(exc).__name__})"
        key = f"{ref.task}.{ref.tier}.{ref.framing}"
        cell = results.cells.get(key)
        if cell is None:
            return f"{ref.path}: no cell {key}"
        if cell.task_id != task_id or record.task_id != task_id:
            return f"cell {key} is {cell.task_id}'s, the record is {record.task_id}'s"
        if cell.tier != record.level:
            return f"cell {key} is a {cell.tier} cell; the record's level is {record.level}"
        if not cell.loco:
            return f"cell {key} is not leave-one-card-out"
        if not cell.pre_registered:
            return f"cell {key} is not pre_registered"
        if cell.selection != "nested":
            return f"cell {key} selection is {cell.selection!r}, not 'nested' (D27)"
        if cell.exploratory:
            return f"cell {key} is exploratory (D27)"
        if not cell.servable:
            return f"cell {key} is not servable"
        if cell.n_folds < MIN_NESTED_FOLDS:
            return f"cell {key} has {cell.n_folds} folds, fewer than {MIN_NESTED_FOLDS}"
        if cell.teacher_in_test is not False:
            return f"cell {key} does not record teacher_in_test false (D19)"
        expected_mask = _framing_mask_rule(record)
        if cell.masked is not True:
            return f"cell {key} masked={cell.masked!r}: every servable cell records masked true"
        if cell.mask != expected_mask:
            return (
                f"cell {key} mask={cell.mask!r} is not the record's framing's mask rule "
                f"{expected_mask!r} (plan §4.2 Q3)"
            )
        if cell.labels_sha256 not in self.committed_snapshots():
            return f"cell {key} labels_sha256 {cell.labels_sha256[:12]} is not a committed snapshot"
        if record.shape == "rank_fit":
            if cell.counts.n_neg is None or cell.counts.n_neg < GUARD_MIN:
                return f"cell {key} n_neg {cell.counts.n_neg} is below {GUARD_MIN}"
        elif cell.counts.n_nonmodal < GUARD_MIN:
            return f"cell {key} n_nonmodal {cell.counts.n_nonmodal} is below {GUARD_MIN}"
        diffs = fingerprint_mismatches(
            record, question_key=cell.question_key, fingerprint=cell.fingerprint
        )
        if diffs:
            return f"cell {key}: " + "; ".join(diffs)
        if self.live is None:
            return "no live fingerprint is bound: serving_lock_sha cannot be checked (K4)"
        if cell.fingerprint is None or not self.live.matches(cell.fingerprint):
            return f"cell {key} fingerprint is not the live one (serving_lock_sha included; K4)"
        agree = _fold_agreement(cell, record)
        if agree < FOLD_AGREEMENT_MIN:
            return (
                f"cell {key} fold_choices agree with the served configuration in {agree} folds, "
                f"fewer than {FOLD_AGREEMENT_MIN}"
            )
        if cell.baselines is None or cell.baselines.beats_lookup_novel is not True:
            return f"cell {key} does not beat the lookup on novel keys (beats_lookup_novel)"
        if cell.metrics is None:
            return f"cell {key} has no metrics"
        risk_key = f"{t.risk:.2f}"
        cp = cell.metrics.threshold_cp.get(risk_key)
        if cp is None:
            return f"cell {key} has no Clopper-Pearson threshold at risk {risk_key}"
        if t.auto < cp:
            return f"auto {t.auto} is below the cell's threshold_cp[{risk_key}] = {cp}"
        return None


def _framing_mask_rule(record: DecisionRecord) -> str | None:
    """The record's framing's option restriction after scoring (``mask_rule``: ``"aspect"`` or
    ``None``), what a cell's ``mask`` must equal (``bench.tasks.neon.MASKS`` per task)."""
    from mesa_clm.framings import FRAMINGS

    framing = FRAMINGS.get(record.task_id, {}).get(record.framing_id)
    return None if framing is None else framing.mask_rule


def _fold_agreement(cell: Any, record: DecisionRecord) -> int:
    """The outer folds of ``cell`` whose ``fold_choices`` entry chose the served configuration:
    a calibrated cell's arm (``framing``, ``model``) or, for a probe cell, the record's
    ``model`` and ``feature_spec``."""
    folds = cell.fold_choices or {}
    agree = 0
    for choice in folds.values():
        if not isinstance(choice, Mapping) or not choice.get("evaluated", True):
            continue
        if cell.tier == "probe":
            ok = (
                record.feature_spec is not None
                and choice.get("model") == record.model
                and choice.get("spec") == record.feature_spec
            )
        else:
            ok = choice.get("framing") == cell.framing and choice.get("model") == cell.model
        agree += int(bool(ok))
    return agree


# -- audits (plan §4.7 "auto_requires_audit"; M4, R7) --------------------------------------------


@runtime_checkable
class AuditCheck(Protocol):
    """Whether a passing ``audits`` row exists for ``(task_key, artifact_version)``: what the
    ``prod`` profile's ``auto_requires_audit`` reads (the caller supplies it: the sidecar's)."""

    def __call__(self, task_key: str, artifact_version: str, /) -> bool: ...


class StoreAuditCheck:
    """:class:`AuditCheck` over a provenance store (``store.audits(task_key)``): a row with
    ``passed`` for that ``artifact_version``."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def __call__(self, task_key: str, artifact_version: str, /) -> bool:
        rows = self.store.audits(task_key)
        return any(
            bool(r.get("passed")) and str(r.get("artifact_version")) == artifact_version
            for r in rows
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
    :class:`CitationValidator` (default a :class:`CellCitationValidator` over
    ``results_root``). A validator that raises counts as a refusal. ``audit_check`` is the
    :class:`AuditCheck` a profile with ``auto_requires_audit`` reads (``None``: no audit
    exists, every auto is demoted to ``proposed`` with reason ``audit_required``).
    """

    def __init__(
        self,
        thresholds: PolicyDefaults,
        *,
        profile: str | Profile = "prod",
        validator: CitationValidator | None = None,
        results_root: str | Path | None = None,
        audit_check: AuditCheck | None = None,
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
        self.validator: CitationValidator = validator or CellCitationValidator(results_root)
        self.audit_check: AuditCheck | None = audit_check

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        profile: str | Profile = "prod",
        validator: CitationValidator | None = None,
        results_root: str | Path | None = None,
        audit_check: AuditCheck | None = None,
    ) -> Policy:
        """:func:`~mesa_clm.policy_defaults.load_policy_defaults` (``None``: the shipped file)
        wrapped with ``profile``, ``validator`` (default the cell validator over
        ``results_root``) and ``audit_check``."""
        return cls(
            load_policy_defaults(path),
            profile=profile,
            validator=validator,
            results_root=results_root,
            audit_check=audit_check,
        )

    @classmethod
    def from_config(
        cls,
        section: PolicyConfig,
        *,
        validator: CitationValidator | None = None,
        audit_check: AuditCheck | None = None,
    ) -> Policy:
        """From the config's ``policy`` section: its ``policy_path`` (``None``: the shipped
        file), ``profile`` and ``results_root``."""
        path = Path(section.policy_path).expanduser() if section.policy_path else None
        return cls.load(
            path,
            profile=section.profile,
            validator=validator,
            results_root=section.results_root,
            audit_check=audit_check,
        )

    def bind(
        self, *, live: Fingerprint | None = None, audit_check: AuditCheck | None = None
    ) -> None:
        """What the pipeline knows and the policy does not: the live fingerprint the cited
        cells must equal (given to a validator with ``bind``) and the sidecar's audit check."""
        bind = getattr(self.validator, "bind", None)
        if live is not None and callable(bind):
            bind(live)
        if audit_check is not None:
            self.audit_check = audit_check

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
        p = profile or self.profile
        v = _decide(record, t, p, lambda: self._cite(record.task_id, t, record))
        if v.outcome == "auto" and p.auto_requires_audit and not self._audited(record):
            return Verdict(
                outcome="proposed",
                reason="audit_required",
                stat=v.stat,
                auto_blockers=("audit_required",),
                cite=v.cite,
            )
        return v

    def _audited(self, record: DecisionRecord) -> bool:
        """A passing audit for the record's ``(task_key, artifact_version)`` (module
        docstring); a record without an artifact version has none, and a check that raises
        counts as none."""
        if self.audit_check is None or not record.artifact_version:
            return False
        try:
            return bool(self.audit_check(record.task_key, record.artifact_version))
        except Exception as exc:  # fail closed, never auto
            logger.warning("audit check failed for %s (%s)", record.task_id, type(exc).__name__)
            return False

    def outcome(self, record: DecisionRecord, *, profile: Profile | None = None) -> Outcome:
        """The outcome of ``record`` under this policy (plan §4.7)."""
        return self.verdict(record, profile=profile).outcome

    def demote_for_group_margin(
        self, result: Outcome, group_margin: float | None, task_id: str
    ) -> Outcome:
        """:func:`demote_for_group_margin` with ``task_id``'s thresholds."""
        return demote_for_group_margin(result, group_margin, self.thresholds(task_id))
