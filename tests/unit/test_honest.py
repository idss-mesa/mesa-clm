"""The honest-record contract (``providers.base.DecisionRecord._honest``; DESIGN D2, D6, D7, D28;
plan §4.6 invariants 1-8): valid records come from the real providers over the fake engine, and
each rejection test breaks exactly one invariant of an otherwise valid record. Also the pure
helpers the records rely on: ``rank_probs``, ``sigmoid``, ``meets_level``, ``revise`` and the
masking step ``apply_mask`` (pattern: mesa-anyjev ``tests/test_levels.py``)."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from mesa_clm import framings as fr
from mesa_clm.cards import DatasetCard
from mesa_clm.identity import target_sha256
from mesa_clm.providers import (
    ArtifactBundle,
    DecisionRecord,
    FakeProvider,
    OlsRankProvider,
    PlattCalibrator,
    apply_mask,
    aspect_keep,
    fake_fingerprint,
    meets_level,
    rank_probs,
    sigmoid,
)
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.states import column_state, state_sha256, target_state

FP = fake_fingerprint()
TERM = fr.active_framing("term.fits")
ASPECT = fr.active_framing("column.aspect")
CANDS = [
    fr.FramingCandidate("PATO:0000040", "distance", "A 1-D extent quality between two points."),
    fr.FramingCandidate("PATO:0000014", "color", "A composite chromatic quality."),
    fr.FramingCandidate("UO:0000008", "meter", "A length unit equal to the SI base unit."),
    fr.FramingCandidate("ENVO:00000428", "biome", "An environmental system."),
]


def _column(card: DatasetCard, name: str = "observerDistance") -> Any:
    return next(c for c in card.columns if c.name == name)


@pytest.fixture
def target(card: DatasetCard) -> dict[str, Any]:
    return target_state(card, "column", "measurement", column=_column(card))


@pytest.fixture
def rank_fit(target: dict[str, Any]) -> DecisionRecord:
    return FakeProvider().decide(TERM, [target], [CANDS])[0]


@pytest.fixture
def choice(card: DatasetCard) -> DecisionRecord:
    return FakeProvider().decide(ASPECT, [column_state(card, _column(card))])[0]


@pytest.fixture
def calibrated(target: dict[str, Any]) -> DecisionRecord:
    bundle = ArtifactBundle(
        version="v1",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        calibrators={TERM.question_key: PlattCalibrator(a=3.0, b=-0.2)},
    )
    return FakeProvider(artifacts=bundle).decide(TERM, [target], [CANDS])[0]


@pytest.fixture
def ols(target: dict[str, Any]) -> DecisionRecord:
    return OlsRankProvider(FP).decide(TERM, [target], [CANDS])[0]


def _data(rec: DecisionRecord, **over: Any) -> dict[str, Any]:
    data = rec.model_dump()
    data.update(over)
    return data


def _reject(rec: DecisionRecord, match: str, **over: Any) -> None:
    with pytest.raises(ValidationError, match=match):
        DecisionRecord.model_validate(_data(rec, **over))


# -- valid records --------------------------------------------------------------------------------


def test_provider_records_are_valid_and_round_trip(
    rank_fit: DecisionRecord,
    choice: DecisionRecord,
    calibrated: DecisionRecord,
    ols: DecisionRecord,
) -> None:
    for rec in (rank_fit, choice, calibrated, ols):
        assert DecisionRecord.model_validate(rec.model_dump()) == rec
        assert DecisionRecord.model_validate_json(rec.model_dump_json()) == rec
    assert rank_fit.level == "zero_shot" and rank_fit.calibration == "uncalibrated"
    assert rank_fit.options[-1] == ANCHOR_KEY and rank_fit.anchor_index == len(CANDS)
    assert rank_fit.s_c is not None and rank_fit.s_c[-1] == 0.0
    assert calibrated.level == "calibrated" and calibrated.calibration == "platt"
    assert calibrated.artifact_version == "v1" and calibrated.artifact is not None
    assert choice.anchor_index is None and choice.s_c is None and choice.kind == "choice"
    assert ols.method == "ols_rank" and ols.probs is None and ols.rank == 1 and ols.level == "none"
    assert rank_fit.kind == "noul"  # term.fits is a mesa-anyjev noul task asked as a rank_fit


# -- invariant 1: probs iff calibrated; CLM methods carry the served distribution -----------------


def test_inv1_probs_without_a_calibration_are_rejected(rank_fit: DecisionRecord) -> None:
    _reject(
        rank_fit,
        "invariant 1: probs must be None exactly when calibration is 'none'",
        method="unavailable",
        level="none",
        calibration="none",
    )


def test_inv1_a_calibrated_record_without_probs_is_rejected(rank_fit: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 1: probs must be None", probs=None)


def test_inv1_a_clm_method_needs_raw_probs(rank_fit: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 1: a 'fake' decision carries raw_probs", raw_probs=None)


def test_inv1_raw_probs_must_be_a_distribution(rank_fit: DecisionRecord) -> None:
    raw = rank_fit.raw_probs or []
    _reject(rank_fit, "invariant 1: raw_probs must sum to 1", raw_probs=[p * 2 for p in raw])
    _reject(rank_fit, "invariant 1: raw_probs must have one entry", raw_probs=raw[:-1])
    _reject(rank_fit, "invariant 1: raw_probs must be finite", raw_probs=[math.nan, *raw[1:]])


def test_inv1_probs_must_have_one_entry_per_option(rank_fit: DecisionRecord) -> None:
    probs = rank_fit.probs or []
    _reject(rank_fit, "invariant 1: probs must have one entry per option", probs=probs[:-1])


def test_inv1_a_zero_served_probability_is_underflow(rank_fit: DecisionRecord) -> None:
    raw = list(rank_fit.raw_probs or [])
    raw[0], raw[1] = 0.0, raw[1] + raw[0]
    _reject(rank_fit, "numeric_underflow", raw_probs=raw)


def test_inv1_methods_without_a_distribution_never_carry_one(
    ols: DecisionRecord, rank_fit: DecisionRecord
) -> None:
    _reject(ols, "invariant 1: method 'ols_rank' has no served distribution", raw_probs=[0.25] * 5)
    _reject(ols, "invariant 1: probs must be None", probs=[0.2] * 5, confidence=0.2, margin=0.0)
    _reject(ols, "clm_confidence is CLM's field", clm_confidence=0.3)


# -- invariant 2: levels and calibrations ---------------------------------------------------------


def test_inv2_zero_shot_is_uncalibrated(rank_fit: DecisionRecord) -> None:
    _reject(
        rank_fit,
        "invariant 2: level 'zero_shot' cannot carry calibration 'platt'",
        calibration="platt",
    )


def test_inv2_learned_tiers_need_platt_or_temperature(calibrated: DecisionRecord) -> None:
    _reject(
        calibrated,
        "invariant 2: level 'calibrated' cannot carry calibration 'uncalibrated'",
        calibration="uncalibrated",
    )
    _reject(
        calibrated, "invariant 2: level 'probe' cannot carry", level="probe", calibration="none"
    )


def test_inv2_methods_without_probs_sit_at_level_none(ols: DecisionRecord) -> None:
    _reject(
        ols,
        "invariant 2: method 'ols_rank' is level 'none'",
        level="zero_shot",
        calibration="uncalibrated",
    )


def test_inv2_clm_methods_carry_a_tier(rank_fit: DecisionRecord) -> None:
    _reject(
        rank_fit,
        "invariant 2: a 'fake' decision carries a CLM tier",
        level="none",
        calibration="none",
        probs=None,
        confidence=None,
        margin=None,
    )


# -- invariant 3: shapes --------------------------------------------------------------------------


def test_inv3_rank_fit_needs_the_anchor(rank_fit: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 3: a rank_fit carries the '__none__' anchor", anchor_index=None)
    _reject(rank_fit, "invariant 3: a rank_fit carries the '__none__' anchor", anchor_index=0)


def test_inv3_choice_has_no_anchor(choice: DecisionRecord) -> None:
    _reject(choice, "invariant 3: a closed choice has no anchor", anchor_index=0)


def test_inv3_rank_fit_needs_s_c_and_p_fit(rank_fit: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 3: a rank_fit carries s_c and p_fit per option", p_fit=None)
    _reject(rank_fit, "invariant 3: a rank_fit carries s_c and p_fit per option", s_c=None)


def test_inv3_anchor_s_c_is_zero(rank_fit: DecisionRecord) -> None:
    s_c = list(rank_fit.s_c or [])
    s_c[-1] = 0.1
    _reject(rank_fit, "invariant 3: s_c is finite and 0 for the anchor", s_c=s_c)


def test_inv3_p_fit_is_a_probability(rank_fit: DecisionRecord) -> None:
    p_fit = [1.5, *(rank_fit.p_fit or [])[1:]]
    _reject(rank_fit, "invariant 3: p_fit is a probability", p_fit=p_fit)


def test_inv3_s_c_cannot_be_fabricated(rank_fit: DecisionRecord) -> None:
    """A shifted score (still 0 at the anchor) is not the log-ratio CLM served."""
    s_c = [s + 1.0 if i != rank_fit.anchor_index else 0.0 for i, s in enumerate(rank_fit.s_c or [])]
    p_fit = [sigmoid(s) for s in s_c]
    _reject(rank_fit, "invariant 3: s_c must be ln p_c - ln p_anchor", s_c=s_c, p_fit=p_fit)


def test_inv3_zero_shot_p_fit_is_the_plain_sigmoid(rank_fit: DecisionRecord) -> None:
    p_fit = [sigmoid(2.0 * s + 0.3) for s in rank_fit.s_c or []]
    _reject(rank_fit, "invariant 3: at zero_shot p_fit is σ\\(s_c\\)", p_fit=p_fit)


def test_inv3_scores_belong_to_rank_fit_with_a_distribution(
    choice: DecisionRecord, ols: DecisionRecord
) -> None:
    _reject(choice, "invariant 3: s_c and p_fit belong to rank_fit", s_c=[0.0] * choice.k)
    _reject(ols, "invariant 3: s_c and p_fit need a served distribution", p_fit=[0.5] * ols.k)


# -- invariant 4: confidence is max(probs), computed locally --------------------------------------


def test_inv4_confidence_is_max_probs_not_clms_field(rank_fit: DecisionRecord) -> None:
    assert rank_fit.clm_confidence is not None
    assert rank_fit.clm_confidence != pytest.approx(rank_fit.confidence)
    _reject(
        rank_fit,
        "invariant 4: confidence must equal max\\(probs\\)",
        confidence=rank_fit.clm_confidence,
    )
    _reject(rank_fit, "invariant 4: confidence must equal max", confidence=None)


def test_inv4_margin_is_top1_minus_top2(rank_fit: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 4: margin must equal top1 - top2", margin=0.99)


def test_inv4_the_answer_is_the_argmax(rank_fit: DecisionRecord) -> None:
    other = next(i for i in range(rank_fit.k) if i != rank_fit.answer_index)
    _reject(
        rank_fit,
        "invariant 4: the answer must be the argmax of probs",
        answer_index=other,
        answer=rank_fit.options[other],
    )


def test_inv4_no_confidence_without_probs(ols: DecisionRecord) -> None:
    _reject(ols, "invariant 4: confidence and margin need probs", confidence=0.9)


# -- invariant 5: learned tiers name their artifact -----------------------------------------------


def test_inv5_a_calibrated_record_names_its_artifact(calibrated: DecisionRecord) -> None:
    _reject(
        calibrated, "invariant 5: level 'calibrated' needs artifact_version", artifact_version=None
    )
    _reject(calibrated, "invariant 5: level 'calibrated' needs", artifact=None)


def test_inv5_the_artifact_matches_the_record_fingerprints(calibrated: DecisionRecord) -> None:
    art = calibrated.model_dump()["artifact"]
    _reject(
        calibrated,
        "invariant 5: the artifact's .* differs .* \\(K4\\)",
        artifact={**art, "encoder_fp": "0" * 12},
    )
    _reject(calibrated, "invariant 5: the artifact's", artifact={**art, "question_key": "1" * 16})
    _reject(calibrated, "invariant 5: artifact.version must equal", artifact_version="v2")


def test_inv5_zero_shot_applies_no_artifact(rank_fit: DecisionRecord, calibrated: Any) -> None:
    _reject(
        rank_fit, "invariant 5: a 'zero_shot' record applies no artifact", artifact_version="v1"
    )
    _reject(
        rank_fit,
        "invariant 5: a 'zero_shot' record applies no artifact",
        artifact=calibrated.model_dump()["artifact"],
    )


# -- invariant 6: a truncated context abstains ----------------------------------------------------


def test_inv6_a_truncated_context_has_no_answer(rank_fit: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 6: a truncated context abstains", truncated=True)


def test_anchor_answer_is_an_answer_the_policy_abstains_on(rank_fit: DecisionRecord) -> None:
    """The anchor answer stays visible (``answer='__none__'``) so the policy can say
    ``anchor_won``; an answer cannot be relabelled away from its option."""
    a = rank_fit.anchor_index
    assert a is not None
    assert rank_fit.anchor_won == (rank_fit.answer_index == a)
    assert rank_fit.anchor_won == (rank_fit.answer == ANCHOR_KEY)
    wrong = next(o for o in rank_fit.options if o != rank_fit.answer)
    _reject(rank_fit, "is not option", answer=wrong)


# -- invariant 7: ols_rank --------------------------------------------------------------------------


def test_inv7_ols_rank_carries_its_rank(ols: DecisionRecord) -> None:
    _reject(ols, "invariant 7: an ols_rank record carries the OLS rank", rank=None)
    _reject(ols, "rank is 1-based", rank=0)


def test_inv7_ols_rank_ranks_a_candidate_group(choice: DecisionRecord) -> None:
    _reject(
        choice,
        "invariant 7: ols_rank ranks a candidate group",
        method="ols_rank",
        level="none",
        calibration="none",
        probs=None,
        raw_probs=None,
        confidence=None,
        margin=None,
        clm_confidence=None,
        rank=1,
    )


def test_inv7_ols_rank_proposes_a_candidate_never_the_anchor(ols: DecisionRecord) -> None:
    a = ols.anchor_index
    assert a is not None
    _reject(ols, "invariant 7: ols_rank answers a candidate", answer_index=a, answer=ANCHOR_KEY)
    _reject(ols, "invariant 7: ols_rank answers a candidate", answer_index=-1, answer="")


# -- invariant 8: identity --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("task_key", ""),
        ("question_key", ""),
        ("question_key", "Z" * 16),
        ("encoder_fp", "abc"),
        ("clm_model_fp", "0" * 13),
        ("schema_sha256", "0" * 63),
        ("state_sha256", "A" * 64),
        ("target_sha256", ""),
        ("context_sha256", "g" * 64),
    ],
)
def test_inv8_every_identity_field_is_present_and_well_formed(
    rank_fit: DecisionRecord, field: str, bad: str
) -> None:
    _reject(rank_fit, f"invariant 8: {field} must be", **{field: bad})


def test_inv8_hashes_must_hash_what_they_claim(rank_fit: DecisionRecord, card: DatasetCard) -> None:
    state = {**rank_fit.state, "aspect": "taxon"}
    _reject(rank_fit, "invariant 8: state_sha256 does not hash the recorded state", state=state)
    other = target_state(card, "column", "measurement", column=_column(card, "siteID"))
    _reject(
        rank_fit,
        "invariant 8: target_sha256 is not the D1 identity",
        target_sha256=target_sha256("term.fits", other),
    )
    bare = {"card": {"dataset": "x"}}
    _reject(
        rank_fit,
        "invariant 8: the state has no target identity",
        state=bare,
        state_sha256=state_sha256(bare),
    )


def test_inv8_task_identity_is_consistent(rank_fit: DecisionRecord, choice: DecisionRecord) -> None:
    _reject(rank_fit, "invariant 8: task_key .* is not term.fits'", task_key=choice.task_key)
    _reject(rank_fit, "invariant 8: term.fits is a noul task, not choice", kind="choice")
    _reject(rank_fit, "invariant 8: term.fits is asked as a rank_fit", shape="choice")
    _reject(rank_fit, "invariant 8: unknown task", task_id="nope")
    _reject(rank_fit, "framing_id", framing_id="")


# -- options, masks, model config -------------------------------------------------------------------


def test_options_answer_and_texts_are_consistent(rank_fit: DecisionRecord) -> None:
    opts = list(rank_fit.options)
    _reject(rank_fit, "options must be unique", options=[opts[1], *opts[1:]])
    _reject(rank_fit, "option_texts must have one entry", option_texts=rank_fit.option_texts[:-1])
    _reject(rank_fit, "answer_index 99 out of range", answer_index=99)
    _reject(rank_fit, "at least two options", options=opts[:1], option_texts=["x"])


def test_zero_shot_probs_are_clms_distribution_untransformed(rank_fit: DecisionRecord) -> None:
    raw = rank_fit.raw_probs or []
    sharpened = [p**2 for p in raw]
    z = math.fsum(sharpened)
    probs = [p / z for p in sharpened]
    idx, top, margin = rank_probs(probs)
    _reject(
        rank_fit,
        "a zero_shot record's probs are CLM's raw_probs, untransformed",
        probs=probs,
        confidence=top,
        margin=margin,
        answer_index=idx,
        answer=rank_fit.options[idx],
    )


def test_mask_flags_are_honest(choice: DecisionRecord, rank_fit: DecisionRecord) -> None:
    masked = [False] * choice.k
    masked[choice.answer_index] = True
    _reject(choice, "a masked option cannot be the answer", masked=masked)
    other = [
        i != choice.answer_index and i == (choice.answer_index + 1) % choice.k
        for i in range(choice.k)
    ]
    _reject(choice, "a masked option carries probability 0", masked=other)
    _reject(choice, "masked must have one flag per option", masked=[False])
    flags = [False] * rank_fit.k
    flags[-1] = True
    _reject(rank_fit, "the abstain anchor is never masked", masked=flags)
    _reject(choice, "a fully masked record abstains", masked=[True] * choice.k)


def test_records_are_frozen_and_forbid_extras(rank_fit: DecisionRecord) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        rank_fit.answer_index = -1  # type: ignore[misc]
    _reject(rank_fit, "Extra inputs are not permitted", outcome="auto")


def test_revise_revalidates(rank_fit: DecisionRecord) -> None:
    with pytest.raises(ValidationError, match="invariant 4"):
        rank_fit.revise(confidence=0.123)
    same = rank_fit.revise(latency_ms=1.0)
    assert same.latency_ms == 1.0 and same.s_c == rank_fit.s_c


# -- helpers --------------------------------------------------------------------------------------


def test_rank_probs() -> None:
    idx, conf, margin = rank_probs([0.2, 0.7, 0.1])
    assert (idx, conf) == (1, 0.7) and margin == pytest.approx(0.5)
    assert rank_probs([0.4, 0.4, 0.2])[0] == 0  # ties: the first index, as CLM's argmax
    assert rank_probs([1.0]) == (0, 1.0, 0.0)
    with pytest.raises(ValueError):
        rank_probs([])


def test_sigmoid_is_stable_and_symmetric() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(800.0) == 1.0 and sigmoid(-800.0) == 0.0  # no OverflowError
    for x in (-3.0, -0.2, 0.7, 12.0):
        assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0, abs=1e-15)
        assert sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)))


def test_meets_level_and_derived_views(
    rank_fit: DecisionRecord, calibrated: DecisionRecord, ols: DecisionRecord
) -> None:
    assert meets_level(calibrated, "calibrated") and meets_level(calibrated, "zero_shot")
    assert not meets_level(rank_fit, "calibrated") and not meets_level(ols, "zero_shot")
    assert meets_level(ols, "none")
    assert rank_fit.answer_p_fit == (rank_fit.p_fit or [])[rank_fit.answer_index]
    assert rank_fit.answer_s_c == (rank_fit.s_c or [])[rank_fit.answer_index]
    assert ols.answer_p_fit is None and ols.answer_s_c is None and not ols.abstained
    assert rank_fit.reason is None


# -- apply_mask (the pure masking step policy.masked wraps) ----------------------------------------


def test_apply_mask_renormalises_and_logs_the_mass(choice: DecisionRecord) -> None:
    keep = [i in (1, 3, 5) for i in range(choice.k)]
    raw = choice.raw_probs or []
    out = apply_mask(choice, keep)
    kept = math.fsum(raw[i] for i in (1, 3, 5))
    assert out.masked == [not k for k in keep]
    assert out.answer_index in (1, 3, 5) and out.answer == out.options[out.answer_index]
    assert math.fsum(out.probs or []) == pytest.approx(1.0)
    assert out.probs is not None and out.probs[0] == 0.0
    assert out.probs[1] == pytest.approx(raw[1] / kept)
    assert out.confidence == max(out.probs)
    assert out.diagnostics["masked_mass"] == pytest.approx(1.0 - kept)
    assert out.raw_probs == choice.raw_probs  # the served distribution is never touched
    # A second mask accumulates: the mass removed relative to the unmasked distribution.
    again = apply_mask(out, [i in (1, 3) for i in range(choice.k)])
    assert again.masked == [i not in (1, 3) for i in range(choice.k)]
    assert again.diagnostics["masked_mass"] == pytest.approx(1.0 - raw[1] - raw[3])
    assert again.probs is not None and again.probs[1] == pytest.approx(raw[1] / (raw[1] + raw[3]))


def test_apply_mask_fully_masked_abstains(choice: DecisionRecord) -> None:
    out = apply_mask(choice, [False] * choice.k)
    assert out.answer_index == -1 and out.answer == "" and out.abstained
    assert out.reason == "mask_empty" and out.diagnostics["mask_empty"] is True
    assert out.probs == choice.probs  # not renormalised onto nothing
    assert out.diagnostics["masked_mass"] == pytest.approx(1.0)


def test_apply_mask_keeps_the_anchor_and_the_scores(rank_fit: DecisionRecord) -> None:
    keep = [False] * rank_fit.k
    keep[0] = True
    out = apply_mask(rank_fit, keep)
    a = rank_fit.anchor_index
    assert a is not None and out.masked is not None and not out.masked[a]
    assert out.s_c == rank_fit.s_c and out.p_fit == rank_fit.p_fit  # set-independent (D2)
    assert out.answer_index in (0, a)
    # Every candidate out: the anchor alone is not a distribution to renormalise onto.
    empty = apply_mask(rank_fit, [False] * rank_fit.k)
    assert empty.answer_index == -1 and empty.reason == "mask_empty"
    assert empty.masked == [True] * (rank_fit.k - 1) + [False]


def test_apply_mask_edges(rank_fit: DecisionRecord, ols: DecisionRecord) -> None:
    with pytest.raises(ValueError, match="one flag per option"):
        apply_mask(rank_fit, [True])
    assert apply_mask(ols, [False] * ols.k) is ols  # nothing to renormalise
    assert apply_mask(rank_fit, [True] * rank_fit.k).probs == rank_fit.probs


def test_aspect_keep_follows_the_registry() -> None:
    options = ["envo", "ncbitaxon", "pato", "uo", ANCHOR_KEY]
    assert aspect_keep(options, "unit") == [False, False, False, True, True]
    assert aspect_keep(options, "measurement") == [True, False, True, False, True]
    assert aspect_keep(options, "measurement", frozenset({"PATO"})) == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert aspect_keep([*options, "nope"], "other")[-1] is False
