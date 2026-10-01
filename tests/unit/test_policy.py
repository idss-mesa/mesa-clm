"""The write policy (``mesa_clm.policy``; DESIGN D6, D8, D9, D10, D24, D25, D28; plan §4.7).

Records come from the real providers: :class:`TieredProvider` over a scripted clm-serve (so the
numbers are known exactly) and the fake engine, revised or copied only where a test needs a tier
M1 cannot serve yet (probe, head) or a record the ``_honest`` validator would refuse (the
defence-in-depth half of the matrix). Pattern: mesa-anyjev ``tests/test_policy_citations.py``
and ``tests/test_levels.py``. The ``ols_rank`` branch has its own file,
``test_ols_rank_proposes_never_auto.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
import yaml

from mesa_clm import framings as fr
from mesa_clm import render
from mesa_clm.cards import DatasetCard
from mesa_clm.clm.http import ClmError, Question, SystemOneResponse, question_to_dict
from mesa_clm.config import PolicyConfig
from mesa_clm.policy import (
    ABSTAIN_REASONS,
    AUTO_BLOCKERS,
    CITE_RE,
    MAX_AVUS,
    SPECIFICITY_DELTA,
    CitationValidator,
    CiteCheck,
    CiteRef,
    Policy,
    PolicyError,
    RefuseAllCitations,
    Verdict,
    cap_at_proposed,
    dedup_and_cap,
    demote_for_group_margin,
    fingerprint_mismatches,
    group_margin,
    link_status,
    masked,
    masked_for_aspect,
    outcome,
    parse_cite,
    specific_child,
    statistic,
    verdict,
)
from mesa_clm.policy_defaults import (
    DEFAULTS_PATH,
    PolicyDefaults,
    Profile,
    Thresholds,
    load_policy_defaults,
    min_weight_for,
)
from mesa_clm.provenance.models import AvuLinkRow
from mesa_clm.providers import (
    REASONS,
    ArtifactBundle,
    DecisionRecord,
    FakeProvider,
    PlattCalibrator,
    TemperatureCalibrator,
    TieredProvider,
    apply_mask,
    aspect_keep,
    fake_fingerprint,
)
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.states import column_state, target_state
from mesa_clm.vocab import CALIBRATIONS, LEVEL_RANK, LEVELS, OUTCOMES

FP = fake_fingerprint()
TERM = fr.active_framing("term.fits")
ASPECT = fr.active_framing("column.aspect")
ONTOLOGY = fr.active_framing("column.ontology_fits")
DISTANCE, COLOR, METER = "PATO:0000040", "PATO:0000014", "UO:0000008"
CANDS = [
    fr.FramingCandidate(DISTANCE, "distance", "A 1-D extent quality between two points."),
    fr.FramingCandidate(COLOR, "color", "A composite chromatic quality."),
    fr.FramingCandidate(METER, "meter", "A length unit equal to the SI base unit."),
]
CITE = "bench/results/2026-09-29/term_fits.json#neon_term_fits.calibrated.F7"
SHIPPED = load_policy_defaults()
PROD, DEV = SHIPPED.profile("prod"), SHIPPED.profile("dev")


class ScriptedClm:
    """clm-serve with scripted answers: each question's distribution is its criteria keys'
    ``weights`` (default 1) normalised; ``fail`` makes every request a 503."""

    model = "clm-latest"

    def __init__(self, weights: Mapping[str, float] | None = None, *, fail: bool = False) -> None:
        self.weights = dict(weights or {})
        self.fail = fail

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        if self.fail:
            raise ClmError(503, "clm-serve is down")
        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            body = question_to_dict(q)
            keys = list(body["criteria"])
            w = [self.weights.get(k, 1.0) for k in keys]
            answers[qid] = render.answer_from_probs(body, keys, [x / sum(w) for x in w])
        return SystemOneResponse.model_validate(
            {
                "model": model or self.model,
                "answers": answers,
                "usage": {"billing_units": len(questions), "input_tokens": 1},
            }
        )


def _col(card: DatasetCard, name: str = "observerDistance") -> Any:
    return next(c for c in card.columns if c.name == name)


def _target(card: DatasetCard) -> dict[str, Any]:
    return target_state(card, "column", "measurement", column=_col(card))


def _bundle(calibrator: PlattCalibrator | TemperatureCalibrator) -> ArtifactBundle:
    return ArtifactBundle(
        version="v1",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        calibrators={TERM.question_key: calibrator},
    )


def _rank_fit(
    card: DatasetCard,
    weights: Mapping[str, float],
    *,
    calibrator: PlattCalibrator | TemperatureCalibrator | None = None,
    candidates: list[fr.FramingCandidate] | None = None,
    **kw: Any,
) -> DecisionRecord:
    bundle = _bundle(calibrator) if calibrator is not None else None
    provider = TieredProvider(ScriptedClm(weights, **kw), None, FP, bundle, method="clm")
    return provider.decide(TERM, [_target(card)], [candidates or CANDS])[0]


def _choice(card: DatasetCard, weights: Mapping[str, float]) -> DecisionRecord:
    provider = TieredProvider(ScriptedClm(weights), None, FP, method="clm")
    return provider.decide(ASPECT, [column_state(card, _col(card))])[0]


def _t(task_id: str = "term.fits", **over: Any) -> Thresholds:
    """The shipped thresholds of ``task_id`` with ``over`` applied (re-validated)."""
    return Thresholds.model_validate({**SHIPPED.thresholds(task_id).model_dump(), **over})


def _profile(name: str, **over: Any) -> Profile:
    return Profile.model_validate({**PROD.model_dump(), "name": name, **over})


def _with_task(t: Thresholds, task_id: str = "term.fits") -> PolicyDefaults:
    return PolicyDefaults(tasks={**SHIPPED.tasks, task_id: t}, profiles=SHIPPED.profiles)


@pytest.fixture
def calibrated(card: DatasetCard) -> DecisionRecord:
    """A Platt-calibrated term.fits record whose candidate wins clearly: p_fit 0.973,
    margin 0.897."""
    return _rank_fit(card, {DISTANCE: 6.0}, calibrator=PlattCalibrator(a=2.0, b=0.0))


def _passing(rec: DecisionRecord, **over: Any) -> Thresholds:
    """Thresholds the numbers of ``rec`` pass exactly (auto = its stat, margin = its margin)."""
    stat = statistic(rec)
    assert stat is not None and rec.margin is not None
    return _t(**{"auto": stat, "margin": rec.margin, "propose": 0.5, "cite": CITE, **over})


# -- the shipped policy ---------------------------------------------------------------------------


def test_shipped_policy_is_proposed_only(card: DatasetCard, calibrated: DecisionRecord) -> None:
    policy = Policy.load()
    assert policy.defaults == SHIPPED and policy.profile == PROD
    assert isinstance(policy.validator, RefuseAllCitations)
    assert all(t.auto is None for t in policy.defaults.tasks.values())
    fake = FakeProvider().decide(TERM, [_target(card)], [CANDS])[0]
    for rec in (fake, calibrated, _rank_fit(card, {DISTANCE: 6.0}), _choice(card, {"unit": 9})):
        v = policy.verdict(rec)
        assert v.outcome in ("proposed", "abstain"), rec
        if v.outcome == "proposed":
            assert v.auto_blockers == ("auto_null",)
    assert policy.outcome(calibrated) == "proposed"


# -- step 0: rule -------------------------------------------------------------------------------


def test_rule_records_are_rule_whatever_the_thresholds(card: DatasetCard) -> None:
    choice = _choice(card, {"measurement": 3.0})
    rule = DecisionRecord.model_validate(
        {
            **choice.model_dump(),
            "provider": "rule",
            "method": "rule",
            "level": "none",
            "calibration": "none",
            "probs": None,
            "raw_probs": None,
            "confidence": None,
            "clm_confidence": None,
            "margin": None,
        }
    )
    t = _t("column.aspect", auto=0.0, propose=0.0, cite=CITE, min_level="zero_shot")
    for profile in (PROD, DEV):
        for cite_ok in (None, False, True):
            assert outcome(rule, t, profile, cite_ok=cite_ok) == "rule"
    # The wrapper needs no thresholds for a rule, and refuses anything else without them.
    bare = Policy(PolicyDefaults(tasks={}, profiles=SHIPPED.profiles))
    assert bare.verdict(rule) == Verdict(outcome="rule")
    with pytest.raises(PolicyError, match=r"no thresholds for task 'column\.aspect'"):
        bare.outcome(choice)


# -- step 1: abstains -----------------------------------------------------------------------------


def _planner(rec: DecisionRecord) -> DecisionRecord:
    """``rec`` as a record without a distribution that still names an answer."""
    return rec.revise(
        provider="planner",
        method="planner",
        level="none",
        calibration="none",
        probs=None,
        raw_probs=None,
        s_c=None,
        p_fit=None,
        confidence=None,
        clm_confidence=None,
        margin=None,
        served_model=None,
    )


def test_step1_abstains_even_when_every_auto_gate_would_pass(card: DatasetCard) -> None:
    permissive = _t(auto=0.0, propose=0.0, margin=0.0, cite=CITE, min_level="zero_shot")
    zero = _rank_fit(card, {DISTANCE: 6.0})
    # max_len 17 leaves one token past the guard margin: every context is truncated (D23).
    tiny = TieredProvider(ScriptedClm({DISTANCE: 6.0}), None, FP, method="clm", max_len=17)
    cases = {
        "anchor_won": _rank_fit(card, {ANCHOR_KEY: 9.0}),
        "truncated": tiny.decide(TERM, [_target(card)], [CANDS])[0],
        "decider_unavailable": _rank_fit(card, {}, fail=True),
        "mask_empty": masked(zero, [False] * zero.k),
        "no_distribution": _planner(zero),
    }
    for reason, rec in cases.items():
        for profile in (PROD, DEV, _profile("open", min_level_write="zero_shot")):
            v = verdict(rec, permissive, profile, cite_ok=True)
            assert v == Verdict(outcome="abstain", reason=reason), (reason, v)
        assert reason in ABSTAIN_REASONS or reason in REASONS


def test_step1_reason_falls_back_when_the_provider_gave_none(card: DatasetCard) -> None:
    rec = _rank_fit(card, {DISTANCE: 6.0})
    no_answer = rec.model_copy(update={"answer_index": -1, "answer": "", "diagnostics": {}})
    assert verdict(no_answer, _t(), PROD).reason == "no_answer"
    no_stat = rec.model_copy(update={"p_fit": None})
    assert verdict(no_stat, _t(), PROD) == Verdict(outcome="abstain", reason="no_statistic")
    nan = rec.model_copy(update={"p_fit": [float("nan")] * rec.k})
    assert verdict(nan, _t(), PROD).reason == "no_statistic"


# -- the statistic --------------------------------------------------------------------------------


def test_rank_fit_reads_p_fit_and_choice_reads_confidence(card: DatasetCard) -> None:
    rank = _rank_fit(card, {DISTANCE: 6.0})  # zero_shot: p_fit = 6/7, confidence = 2/3
    assert statistic(rank) == pytest.approx(6 / 7) and rank.confidence == pytest.approx(2 / 3)
    assert outcome(rank, _t(propose=0.8), PROD) == "proposed"  # p_fit clears, confidence not
    v = verdict(rank, _t(propose=0.9), PROD)
    assert (v.outcome, v.reason, v.stat) == ("abstain", "below_propose", pytest.approx(6 / 7))
    assert v.auto_blockers == ("auto_null",)
    choice = _choice(card, {"measurement": 7.0})  # confidence 7/14
    assert statistic(choice) == pytest.approx(0.5) and choice.p_fit is None
    assert outcome(choice, _t("column.aspect", propose=0.5), PROD) == "proposed"
    assert outcome(choice, _t("column.aspect", propose=0.51), PROD) == "abstain"


def test_thresholds_for_the_other_shape_are_refused(card: DatasetCard) -> None:
    rank = _rank_fit(card, {DISTANCE: 6.0})
    with pytest.raises(PolicyError, match="must read p_fit, not confidence"):
        outcome(rank, _t("column.aspect"), PROD)
    with pytest.raises(PolicyError, match="must read confidence, not p_fit"):
        outcome(_choice(card, {}), _t(), PROD)


# -- step 2: auto and each of its gates ---------------------------------------------------------


def test_auto_when_every_gate_passes_at_the_boundary(calibrated: DecisionRecord) -> None:
    t = _passing(calibrated)
    v = verdict(calibrated, t, PROD, cite_ok=True)
    assert v == Verdict(outcome="auto", stat=statistic(calibrated), cite="validated by the caller")
    assert outcome(calibrated, t, PROD, cite_ok=True) == "auto"


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({"t": {"auto": None}}, "auto_null"),
        ({"profile": "dev"}, "auto_write_disabled"),
        ({"t": {"min_level": "probe"}}, "level_below_floor"),
        ({"profile_over": {"min_level_write": "head"}}, "level_below_floor"),
        ({"profile_over": {"auto_calibrations": ("temperature",)}}, "calibration_not_allowed"),
        ({"stat_delta": 1e-6}, "below_auto"),
        ({"margin_delta": 1e-6}, "margin_below"),
        ({"cite_ok": None}, "cite_refused"),
        ({"cite_ok": False}, "cite_refused"),
    ],
)
def test_each_auto_gate_alone_keeps_a_record_proposed(
    calibrated: DecisionRecord, change: dict[str, Any], blocker: str
) -> None:
    stat, margin = statistic(calibrated), calibrated.margin
    assert stat is not None and margin is not None
    t_over: dict[str, Any] = dict(change.get("t", {}))
    if "stat_delta" in change:
        t_over["auto"] = stat + change["stat_delta"]
    if "margin_delta" in change:
        t_over["margin"] = margin + change["margin_delta"]
    t = _passing(calibrated, **t_over)
    profile = DEV if change.get("profile") == "dev" else PROD
    if "profile_over" in change:
        profile = _profile("custom", **change["profile_over"])
    cite_ok = change.get("cite_ok", True)
    v = verdict(calibrated, t, profile, cite_ok=cite_ok)
    assert v.outcome == "proposed" and v.auto_blockers == (blocker,), v
    assert blocker in AUTO_BLOCKERS
    if blocker == "cite_refused":
        assert v.cite is not None and ("no validated" in v.cite or "failed" in v.cite)


def test_zero_shot_never_autos_even_with_a_zero_shot_floor(card: DatasetCard) -> None:
    rec = _rank_fit(card, {DISTANCE: 6.0})
    open_profile = _profile("open", min_level_write="zero_shot")
    v = verdict(rec, _passing(rec, min_level="zero_shot"), open_profile, cite_ok=True)
    assert v.outcome == "proposed"
    assert v.auto_blockers == ("zero_shot", "calibration_not_allowed")


# -- the matrix: levels x calibrations x profiles x citations ------------------------------------

MATRIX_PROFILES = {
    "prod": PROD,
    "dev": DEV,
    "open": _profile("open", min_level_write="zero_shot"),
    "probe_floor": _profile("probe_floor", min_level_write="probe"),
    "temperature_only": _profile("temperature_only", auto_calibrations=("temperature",)),
}


def _expect_auto(level: str, calibration: str, profile: Profile, cite_ok: bool | None) -> bool:
    return (
        cite_ok is True
        and profile.allow_auto_write
        and level != "zero_shot"
        and LEVEL_RANK[level] >= LEVEL_RANK[profile.min_level_write]
        and calibration in ("platt", "temperature")
        and calibration in profile.auto_calibrations
    )


def test_matrix_over_every_level_calibration_profile_and_cite(calibrated: DecisionRecord) -> None:
    """Every (level, calibration) pair, including the ones ``_honest`` refuses (copied without
    validation: defence in depth), under every profile and every citation state, with numbers
    that pass exactly and a zero_shot task floor so only the profile and the record decide."""
    t = _passing(calibrated, min_level="zero_shot")
    autos = 0
    for level, calibration, (pname, profile), cite_ok in itertools.product(
        LEVELS, CALIBRATIONS, MATRIX_PROFILES.items(), (None, False, True)
    ):
        rec = calibrated.model_copy(update={"level": level, "calibration": calibration})
        got = outcome(rec, t, profile, cite_ok=cite_ok)
        want = "auto" if _expect_auto(level, calibration, profile, cite_ok) else "proposed"
        assert got == want, (level, calibration, pname, cite_ok)
        if level == "zero_shot" or cite_ok is not True:
            assert got != "auto"
        autos += got == "auto"
    # Not vacuous: calibrated/probe/head x platt/temperature autos under prod and open, and
    # probe/head under probe_floor, x temperature only under temperature_only.
    assert autos == 3 * 2 + 3 * 2 + 2 * 2 + 3 * 1


def test_matrix_over_honest_records(card: DatasetCard, calibrated: DecisionRecord) -> None:
    """The valid records of every tier: zero_shot (served), calibrated by Platt and by
    temperature (served), probe and head (revised: they land in M4/M7)."""
    temperature = _rank_fit(card, {DISTANCE: 6.0}, calibrator=TemperatureCalibrator(T=0.5))
    records = {
        ("zero_shot", "uncalibrated"): _rank_fit(card, {DISTANCE: 6.0}),
        ("calibrated", "platt"): calibrated,
        ("calibrated", "temperature"): temperature,
        ("probe", "platt"): calibrated.revise(level="probe"),
        ("head", "temperature"): temperature.revise(level="head"),
    }
    for (level, calibration), rec in records.items():
        assert (rec.level, rec.calibration) == (level, calibration)
        t = _passing(rec, min_level="zero_shot")
        for pname, profile in MATRIX_PROFILES.items():
            for cite_ok in (None, False, True):
                got = outcome(rec, t, profile, cite_ok=cite_ok)
                want = _expect_auto(level, calibration, profile, cite_ok)
                assert (got == "auto") == want, (level, calibration, pname, cite_ok)
                assert got in ("auto", "proposed")


# -- citations through the wrapper (D8) ---------------------------------------------------------


class AcceptAll:
    """A validator that accepts every cite and records its calls."""

    def __init__(self, check: CiteCheck | None = None) -> None:
        self.check = check or CiteCheck(ok=True, reason="cell ok")
        self.calls: list[tuple[str, Thresholds, DecisionRecord]] = []

    def __call__(
        self, task_id: str, thresholds: Thresholds, record: DecisionRecord, /
    ) -> CiteCheck:
        self.calls.append((task_id, thresholds, record))
        return self.check


class Broken:
    def __call__(
        self, task_id: str, thresholds: Thresholds, record: DecisionRecord, /
    ) -> CiteCheck:
        raise RuntimeError("cannot read the results file")


def test_missing_or_invalid_cite_blocks_auto_even_when_the_numbers_pass(
    calibrated: DecisionRecord,
) -> None:
    t = _passing(calibrated)
    defaults = _with_task(t)
    # The shipped validator refuses a well-formed cite: validation lands in M4.
    v = Policy(defaults).verdict(calibrated)
    assert v.outcome == "proposed" and v.auto_blockers == ("cite_refused",)
    assert v.cite is not None and "lands in M4" in v.cite
    # A malformed cite is refused with its own reason.
    bad = Thresholds.model_validate({**t.model_dump(), "cite": "results.json#term"})
    v = Policy(_with_task(bad)).verdict(calibrated)
    assert v.auto_blockers == ("cite_refused",) and v.cite is not None
    assert "is not bench/results/" in v.cite
    # A validator that says no (fingerprint mismatch) and one that raises both refuse.
    mismatch = AcceptAll(CiteCheck(ok=False, reason="encoder_fp: cell 'aaa', record 'bbb'"))
    v = Policy(defaults, validator=mismatch).verdict(calibrated)
    assert v.outcome == "proposed" and v.cite == "encoder_fp: cell 'aaa', record 'bbb'"
    v = Policy(defaults, validator=Broken()).verdict(calibrated)
    assert v.outcome == "proposed" and v.cite == "the validator raised RuntimeError"
    # Only a validator that accepts lets the same record auto.
    ok = AcceptAll()
    policy = Policy(defaults, validator=ok)
    assert isinstance(ok, CitationValidator)
    assert policy.verdict(calibrated) == Verdict(
        outcome="auto", stat=statistic(calibrated), cite="cell ok"
    )
    assert ok.calls == [("term.fits", t, calibrated)]
    # ... and never under a profile that forbids auto-writes.
    assert Policy(defaults, profile="dev", validator=ok).outcome(calibrated) == "proposed"


def test_the_validator_is_consulted_only_once_the_numbers_pass(
    card: DatasetCard, calibrated: DecisionRecord
) -> None:
    ok = AcceptAll()
    Policy(SHIPPED, validator=ok).outcome(calibrated)  # auto: null
    Policy(_with_task(_passing(calibrated, margin=0.99)), validator=ok).outcome(calibrated)
    Policy(_with_task(_passing(calibrated)), validator=ok).outcome(_rank_fit(card, {}))  # zero_shot
    assert ok.calls == []


def test_refuse_all_citations_reasons(calibrated: DecisionRecord) -> None:
    refuse = RefuseAllCitations()
    assert isinstance(refuse, CitationValidator)
    none = refuse("term.fits", _t(), calibrated)
    assert not none.ok and "no numeric auto" in none.reason
    uncited = Thresholds.model_validate({**_t().model_dump(), "auto": 0.9})
    assert refuse("term.fits", uncited, calibrated) == CiteCheck(
        ok=False, reason="term.fits: auto=0.9 cites nothing"
    )
    assert not refuse("term.fits", _passing(calibrated), calibrated).ok


def test_policy_load_from_a_file_with_a_numeric_auto(
    tmp_path: Path, calibrated: DecisionRecord
) -> None:
    raw = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    raw["tasks"]["term.fits"].update(auto=0.9, margin=0.5, cite=CITE)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert Policy.load(path).outcome(calibrated) == "proposed"  # refused citation
    assert Policy.load(path, validator=AcceptAll()).outcome(calibrated) == "auto"
    cfg = PolicyConfig(policy_path=str(path), profile="dev")
    dev = Policy.from_config(cfg, validator=AcceptAll())
    assert dev.profile == DEV and dev.outcome(calibrated) == "proposed"


def test_parse_cite_and_fingerprint_mismatches(calibrated: DecisionRecord) -> None:
    assert parse_cite(CITE) == CiteRef(
        "bench/results/2026-09-29/term_fits.json", "neon_term_fits", "calibrated", "F7"
    )
    assert CITE_RE.match(CITE) is not None
    for bad in (None, "", "bench/results/2026-9-29/x.json#a.b.c", "x.json#a.b.c", CITE + ".x"):
        assert parse_cite(bad) is None, bad
    fp = {k: getattr(calibrated, k) for k in ("encoder_fp", "clm_model_fp", "schema_sha256")}
    assert fingerprint_mismatches(calibrated, question_key=TERM.question_key, fingerprint=fp) == []
    diffs = fingerprint_mismatches(
        calibrated, question_key="0" * 16, fingerprint={**fp, "encoder_fp": "f" * 12}
    )
    assert [d.split(":")[0] for d in diffs] == ["question_key", "encoder_fp"]
    baseline = fingerprint_mismatches(calibrated, question_key=None, fingerprint=None)
    assert len(baseline) == 2 and "never servable" in baseline[1]


# -- demotions: the group margin and specificity (D24) -------------------------------------------


def test_demote_for_group_margin_only_demotes_an_auto() -> None:
    t = _t(margin=0.15)
    assert demote_for_group_margin("auto", 0.15, t) == "auto"
    assert demote_for_group_margin("auto", 0.1499, t) == "proposed"
    assert demote_for_group_margin("auto", None, t) == "proposed"
    assert demote_for_group_margin("auto", float("nan"), t) == "proposed"
    for other in OUTCOMES:
        if other != "auto":
            for gm in (None, 0.0, 1.0):
                assert demote_for_group_margin(other, gm, t) == other  # type: ignore[arg-type]
    policy = Policy(SHIPPED)
    assert policy.demote_for_group_margin("auto", 0.1, "term.fits") == "proposed"
    assert policy.demote_for_group_margin("auto", 0.2, "term.fits") == "auto"


def test_group_margin(card: DatasetCard) -> None:
    rec = _rank_fit(card, {DISTANCE: 6.0, COLOR: 3.0})  # p_fit 6/7 and 3/4, zero shot
    gm = group_margin(rec)
    assert gm == pytest.approx(6 / 7 - 3 / 4)
    without_color = masked(rec, [o != COLOR for o in rec.options])
    assert group_margin(without_color) == pytest.approx(6 / 7 - 1 / 2)
    alone = _rank_fit(card, {DISTANCE: 6.0}, candidates=CANDS[:1])
    assert group_margin(alone) == pytest.approx(6 / 7)
    assert group_margin(_rank_fit(card, {ANCHOR_KEY: 9.0})) is None
    assert group_margin(_choice(card, {})) is None
    assert group_margin(_rank_fit(card, {}, fail=True)) is None


def test_specificity_child_and_cap(card: DatasetCard) -> None:
    # One rank over {parent, children, anchor} (D24): parent = distance (p_fit 2/3 at w=2).
    near = _rank_fit(card, {DISTANCE: 2.0, COLOR: 3.0})  # 3/4 < 2/3 + 0.10
    assert specific_child(near, DISTANCE) is None
    assert specific_child(near, DISTANCE, delta=0.05) == near.options.index(COLOR)
    far = _rank_fit(card, {DISTANCE: 2.0, COLOR: 4.0, METER: 3.0})  # 4/5 >= 2/3 + 0.10
    assert specific_child(far, DISTANCE) == far.options.index(COLOR)
    # With the best child masked out, the next (meter, 3/4) is short of 2/3 + 0.10.
    masked_out = masked(far, [o != COLOR for o in far.options])
    assert specific_child(masked_out, DISTANCE) is None
    assert specific_child(masked_out, DISTANCE, delta=0.08) == far.options.index(METER)
    assert SPECIFICITY_DELTA == 0.10
    assert specific_child(_choice(card, {}), "measurement") is None
    with pytest.raises(PolicyError, match="not an option"):
        specific_child(far, "ENVO:00000428")
    assert cap_at_proposed("auto") == "proposed"
    assert [cap_at_proposed(o) for o in ("proposed", "abstain", "rule")] == [
        "proposed",
        "abstain",
        "rule",
    ]


# -- links (D10) and the keep rule (D25) ----------------------------------------------------------


def test_link_status_matches_the_sidecar_check() -> None:
    assert link_status("auto") == ("accepted", "policy")
    assert link_status("proposed") == ("proposed", None)
    for other in ("abstain", "rule", "rejected", "human", "escalated", "decider_unavailable"):
        assert link_status(other) is None  # type: ignore[arg-type]
    for result in ("auto", "proposed"):
        status = link_status(result)  # type: ignore[arg-type]
        assert status is not None
        row = AvuLinkRow(
            run_id=uuid4(),
            attribute="pato.distance",
            value="observerDistance",
            write_status=status[0],
            accepted_by=status[1],
        )
        assert row.write_status == status[0]


def test_dedup_and_cap() -> None:
    items = [
        ("a", "1", "", 0.5),
        ("a", "1", "", 0.9),  # duplicate triple, higher p_fit wins, keeps the first position
        ("a", "1", "", 0.1),  # a lower duplicate changes nothing
        ("b", "1", "", None),
        ("c", "1", "UO:1", 0.7),
        ("c", "1", "UO:2", 0.7),  # different unit: a different triple
    ]
    kept = dedup_and_cap(items, triple=lambda x: (x[0], x[1], x[2]), p_fit=lambda x: x[3])
    assert kept == [items[1], items[4], items[5], items[3]]
    assert dedup_and_cap(items, triple=lambda x: x[:3], p_fit=lambda x: x[3], cap=2) == kept[:2]
    many = [(str(i), "v", "", i / 100) for i in range(40)]
    capped = dedup_and_cap(many, triple=lambda x: x[:3], p_fit=lambda x: x[3])
    assert len(capped) == MAX_AVUS and capped[0] == many[-1]
    with pytest.raises(ValueError, match="cap"):
        dedup_and_cap(items, triple=lambda x: x[:3], p_fit=lambda x: x[3], cap=-1)


# -- masking ---------------------------------------------------------------------------------------


def test_masked_is_the_pure_apply_mask(card: DatasetCard) -> None:
    rec = _rank_fit(card, {DISTANCE: 6.0, COLOR: 3.0})
    keep = [o != DISTANCE for o in rec.options]
    out = masked(rec, keep)
    assert out == apply_mask(rec, keep) == masked(rec, np.array(keep, dtype=bool))
    assert out.answer == COLOR and out.masked == [True, False, False, False]
    assert outcome(out, _t(propose=0.7), PROD) == "proposed"  # p_fit(color) = 3/4, unchanged
    # column.ontology_fits comes back masked by the state's aspect (measurement); narrowing it to
    # the ontologies in play re-masks and renormalises.
    onto = FakeProvider().decide(ONTOLOGY, [_target(card)], [fr.ontology_candidates()])[0]
    in_play = frozenset({"pato", "envo"})
    narrowed = masked_for_aspect(onto, "measurement", in_play)
    assert narrowed == apply_mask(onto, aspect_keep(onto.options, "measurement", in_play))
    assert narrowed.masked == [o not in in_play and o != ANCHOR_KEY for o in onto.options]
    assert narrowed.answer in in_play | {ANCHOR_KEY}
    # No registry ontology serves a term.fits CURIE: the whole group is masked away.
    empty = masked_for_aspect(rec, "measurement")
    assert empty.reason == "mask_empty" and verdict(empty, _t(), PROD).reason == "mask_empty"


# -- the wrapper (D9) -----------------------------------------------------------------------------


def test_policy_wrapper_reads_the_one_policy_file() -> None:
    policy = Policy.from_config(PolicyConfig())
    assert policy.profile == PROD and policy.defaults == SHIPPED
    for task_id in SHIPPED.tasks:
        assert policy.thresholds(task_id) == SHIPPED.thresholds(task_id)
        assert policy.min_weight(task_id) == min_weight_for(task_id)
    assert Policy(SHIPPED, profile=DEV).profile == DEV
    with pytest.raises(PolicyError, match="no profile 'staging'"):
        Policy(SHIPPED, profile="staging")
    with pytest.raises(PolicyError, match=r"no thresholds for task 'avu\.keep'"):
        policy.thresholds("avu.keep")  # removed: a rule now (D25)


def test_every_verdict_reason_is_documented(card: DatasetCard, calibrated: DecisionRecord) -> None:
    records = [
        calibrated,
        _rank_fit(card, {}),
        _rank_fit(card, {ANCHOR_KEY: 9.0}),
        _rank_fit(card, {}, fail=True),
        _choice(card, {}),
    ]
    for rec in records:
        t = _t("column.aspect") if rec.shape == "choice" else _passing(calibrated)
        for profile in MATRIX_PROFILES.values():
            for cite_ok in (None, True):
                v = verdict(rec, t, profile, cite_ok=cite_ok)
                assert v.outcome in OUTCOMES
                assert v.reason is None or v.reason in ABSTAIN_REASONS or v.reason in REASONS
                assert set(v.auto_blockers) <= set(AUTO_BLOCKERS)
