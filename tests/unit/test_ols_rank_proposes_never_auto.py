"""The degraded ``ols_rank`` method proposes and never autos (DESIGN D28; plan §4.6 invariant 7,
§4.7 step 0).

While a task has no trustworthy tier (K1, K2(c)) or clm-serve is down, the pipeline proposes the
OLS top-1 of each candidate group as an ``ols_rank`` record: level ``none``, no distribution,
its OLS ``rank`` set. Step 1 would abstain on every such record (``probs IS NULL``), so the policy
has an explicit branch in front of it: ``proposed`` while ``rank <= t.ols_rank_top``, else
``abstain``, and never ``auto``, however permissive the thresholds, the profile or the citation
validator, and even for a copy that was tampered with to look calibrated. The sidecar CHECK
(``provenance.models.DecisionRow``) repeats the rule.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pytest

from mesa_clm import framings as fr
from mesa_clm.cards import DatasetCard
from mesa_clm.policy import (
    CiteCheck,
    Policy,
    Verdict,
    cap_at_proposed,
    demote_for_group_margin,
    link_status,
    outcome,
    verdict,
)
from mesa_clm.policy_defaults import PolicyDefaults, Profile, Thresholds, load_policy_defaults
from mesa_clm.providers import (
    DecisionRecord,
    OlsRankProvider,
    fake_fingerprint,
    ols_rank_record,
)
from mesa_clm.states import target_state
from mesa_clm.vocab import LEVELS

FP = fake_fingerprint()
TERM = fr.active_framing("term.fits")
ONTOLOGY = fr.active_framing("column.ontology_fits")
CANDS = [
    fr.FramingCandidate("PATO:0000040", "distance", "A 1-D extent quality between two points."),
    fr.FramingCandidate("PATO:0000014", "color", "A composite chromatic quality."),
    fr.FramingCandidate("UO:0000008", "meter", "A length unit equal to the SI base unit."),
    fr.FramingCandidate("ENVO:00000428", "biome", "An environmental system."),
]
CITE = "bench/results/2026-09-29/term_fits.json#neon_term_fits.calibrated.F7"
SHIPPED = load_policy_defaults()
PROD, DEV = SHIPPED.profile("prod"), SHIPPED.profile("dev")
OPEN = Profile(name="open", min_level_write="zero_shot", auto_requires_audit=False)
PROFILES = (PROD, DEV, OPEN)


@dataclass(frozen=True)
class RankedCandidate:
    """An OLS search hit that carries its own search rank (``ols.Candidate`` does)."""

    key: str
    label: str
    description: str
    rank: int


def _permissive(task_id: str = "term.fits", **over: Any) -> Thresholds:
    """Thresholds every CLM record would auto under: auto at 0, no margin, zero_shot floor."""
    base = SHIPPED.thresholds(task_id).model_dump()
    return Thresholds.model_validate(
        {**base, "auto": 0.0, "propose": 0.0, "margin": 0.0, "min_level": "zero_shot", **over}
    )


class AcceptAll:
    """A citation validator that accepts everything and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, task_id: str, thresholds: Thresholds, record: DecisionRecord, /
    ) -> CiteCheck:
        self.calls += 1
        return CiteCheck(ok=True, reason="accepted")


@pytest.fixture
def target(card: DatasetCard) -> dict[str, Any]:
    column = next(c for c in card.columns if c.name == "observerDistance")
    return target_state(card, "column", "measurement", column=column)


def test_ols_rank_proposes_within_the_top_and_never_autos(target: dict[str, Any]) -> None:
    records = [ols_rank_record(TERM, target, CANDS, FP, pick=i) for i in range(len(CANDS))]
    assert [r.rank for r in records] == [1, 2, 3, 4]
    for rec, top, profile, cite_ok in itertools.product(
        records, (1, 2, 4), PROFILES, (None, False, True)
    ):
        assert rec.method == "ols_rank" and rec.probs is None and rec.level == "none"
        t = _permissive(ols_rank_top=top, cite=CITE)
        v = verdict(rec, t, profile, cite_ok=cite_ok)
        assert rec.rank is not None
        if rec.rank <= top:
            assert v == Verdict(outcome="proposed", auto_blockers=("ols_rank",))
        else:
            assert v == Verdict(
                outcome="abstain", reason="ols_rank_below_top", auto_blockers=("ols_rank",)
            )
        assert outcome(rec, t, profile, cite_ok=cite_ok) != "auto"


def test_the_shipped_default_proposes_only_the_ols_top_1(target: dict[str, Any]) -> None:
    policy = Policy.load()
    assert policy.thresholds("term.fits").ols_rank_top == 1
    [top1] = OlsRankProvider(FP).decide(TERM, [target], [CANDS])
    assert top1.rank == 1 and policy.outcome(top1) == "proposed"
    deeper = ols_rank_record(TERM, target, CANDS, FP, pick=1)
    assert policy.verdict(deeper).reason == "ols_rank_below_top"
    # The degraded method covers column.ontology_fits too (a rank_fit over the registry).
    [onto] = OlsRankProvider(FP).decide(ONTOLOGY, [target], [fr.ontology_candidates()])
    assert policy.outcome(onto) == "proposed"


def test_the_candidates_own_ols_rank_is_what_counts(target: dict[str, Any]) -> None:
    hits = [RankedCandidate(c.key, c.label, c.description, rank=3 + i) for i, c in enumerate(CANDS)]
    rec = ols_rank_record(TERM, target, hits, FP, pick=0)
    assert rec.rank == 3
    assert outcome(rec, _permissive(), PROD, cite_ok=True) == "abstain"
    assert outcome(rec, _permissive(ols_rank_top=3), PROD, cite_ok=True) == "proposed"


def test_the_citation_validator_is_never_asked(target: dict[str, Any]) -> None:
    validator = AcceptAll()
    defaults = PolicyDefaults(
        tasks={**SHIPPED.tasks, "term.fits": _permissive(cite=CITE)}, profiles=SHIPPED.profiles
    )
    for profile in PROFILES:
        policy = Policy(defaults, profile=profile, validator=validator)
        for rec in OlsRankProvider(FP).decide(TERM, [target, target], [CANDS, CANDS[1:]]):
            assert policy.outcome(rec) == "proposed"
    assert validator.calls == 0


def test_a_tampered_ols_rank_record_still_never_autos(target: dict[str, Any]) -> None:
    """Step 0 reads the method before any number: a copy that skipped ``_honest`` and claims a
    calibrated distribution is still only proposed."""
    rec = ols_rank_record(TERM, target, CANDS, FP)
    k = rec.k
    probs = [0.97] + [0.03 / (k - 1)] * (k - 1)
    for level in LEVELS:
        forged = rec.model_copy(
            update={
                "level": level,
                "calibration": "platt",
                "probs": probs,
                "p_fit": [0.99] * k,
                "confidence": probs[0],
                "margin": probs[0] - probs[1],
            }
        )
        for profile in PROFILES:
            assert outcome(forged, _permissive(cite=CITE), profile, cite_ok=True) == "proposed"


def test_demotions_and_links_leave_an_ols_rank_proposal_alone(target: dict[str, Any]) -> None:
    rec = ols_rank_record(TERM, target, CANDS, FP)
    t = _permissive(margin=0.5)
    result = outcome(rec, t, PROD, cite_ok=True)
    assert demote_for_group_margin(result, 0.0, t) == "proposed"
    assert demote_for_group_margin(result, None, t) == "proposed"
    assert cap_at_proposed(result) == "proposed"
    # D10: a proposal waits for a reviewer; nothing is accepted by policy.
    assert link_status(result) == ("proposed", None)


def test_an_ols_rank_record_without_rank_or_answer_abstains(target: dict[str, Any]) -> None:
    """The record invariant forbids both; the policy still fails closed on an unvalidated copy."""
    rec = ols_rank_record(TERM, target, CANDS, FP)
    t = _permissive(cite=CITE)
    no_rank = rec.model_copy(update={"rank": None})
    assert verdict(no_rank, t, PROD, cite_ok=True) == Verdict(outcome="abstain", reason="no_rank")
    no_answer = rec.model_copy(update={"answer_index": -1, "answer": ""})
    assert verdict(no_answer, t, PROD, cite_ok=True).reason == "no_answer"
    anchor = rec.model_copy(update={"answer_index": rec.anchor_index})
    assert verdict(anchor, t, OPEN, cite_ok=True) == Verdict(outcome="abstain", reason="anchor_won")
