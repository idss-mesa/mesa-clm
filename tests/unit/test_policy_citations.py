"""The citation test (``policy.CellCitationValidator``; DESIGN D8; plan §4.7; PR "Citation
test"; the M4 brief R6) on a synthetic results file: a cell that qualifies lets a passing
record ``auto``, and every field of the test, negated once, refuses it with a reason naming the
field. Also the audit blocker (``auto_requires_audit``: a passing ``audits`` row for the
record's artifact version, else ``proposed`` with reason ``audit_required``), ``Policy.load``'s
default validator and the pipeline's binding. No real results file or snapshot is read: the
results root is a temporary directory and the snapshot file exists only to be hashed."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import framings as fr
from mesa_clm.cards import DatasetCard
from mesa_clm.clm.fingerprint import Fingerprint
from mesa_clm.policy import (
    AUTO_BLOCKERS,
    DEMOTION_REASONS,
    FOLD_AGREEMENT_MIN,
    GUARD_MIN,
    MIN_NESTED_FOLDS,
    AuditCheck,
    CellCitationValidator,
    Policy,
    StoreAuditCheck,
    default_results_root,
)
from mesa_clm.policy_defaults import PolicyDefaults, Thresholds, load_policy_defaults
from mesa_clm.provenance.models import AuditRow
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers import ArtifactBundle, DecisionRecord, PlattCalibrator, TieredProvider
from mesa_clm.states import target_state
from tests.fakes import m4
from tests.fakes.m4 import FP, cell, planted_items
from tests.unit.test_policy import CANDS, DISTANCE, ScriptedClm, _target

TERM = fr.active_framing("term.fits")
ONTOLOGY = fr.active_framing("column.ontology_fits")
SHIPPED = load_policy_defaults()
DATE = "2026-10-05"
NAME = "tiers"
CITE = f"bench/results/{DATE}/{NAME}.json#neon_term_fits.calibrated.{TERM.id}"


def _record(card: DatasetCard) -> DecisionRecord:
    """A Platt-calibrated term.fits record whose candidate wins clearly (p_fit 0.973)."""
    bundle = ArtifactBundle(
        version="v1",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        calibrators={TERM.question_key: PlattCalibrator(a=2.0, b=0.0)},
    )
    provider = TieredProvider(ScriptedClm({DISTANCE: 6.0}), None, FP, bundle, method="clm")
    state = target_state(
        card,
        "column",
        "measurement",
        column=next(c for c in card.columns if c.name == "observerDistance"),
    )
    return provider.decide(TERM, [state], [CANDS])[0]


def _thresholds(**over: Any) -> Thresholds:
    base = SHIPPED.thresholds("term.fits").model_dump()
    return Thresholds.model_validate({**base, "auto": 0.95, "margin": 0.0, "cite": CITE, **over})


def _defaults(t: Thresholds) -> PolicyDefaults:
    return PolicyDefaults(tasks={**SHIPPED.tasks, "term.fits": t}, profiles=SHIPPED.profiles)


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, str, Callable[..., None]]:
    """A results root with the committed snapshot (hashed) and a writer of the cited cell."""
    _, sha = m4.snapshot_file(tmp_path)

    def write(**over: Any) -> None:
        params: dict[str, Any] = {"labels_sha256": sha, **over}
        c = cell("term.fits", params.pop("tier", "calibrated"), **params)
        m4.results_file(tmp_path, DATE, NAME, [c], labels_sha256=sha)

    write()
    return tmp_path, sha, write


def _validator(root: Path, live: Fingerprint | None = FP) -> CellCitationValidator:
    return CellCitationValidator(root, live=live)


def test_a_qualifying_cell_lets_a_passing_record_auto(
    card: DatasetCard, world: tuple[Path, str, Callable[..., None]]
) -> None:
    root, sha, _ = world
    rec = _record(card)
    v = _validator(root)
    check = v("term.fits", _thresholds(), rec)
    assert check.ok and "qualifies" in check.reason
    assert v.committed_snapshots() == {sha}
    policy = Policy(_defaults(_thresholds()), validator=v, audit_check=lambda k, ver: True)
    verdict = policy.verdict(rec)
    assert verdict.outcome == "auto" and verdict.cite is not None and "qualifies" in verdict.cite
    # The threshold at the boundary passes; below the cell's threshold_cp it does not.
    assert v("term.fits", _thresholds(auto=0.9), rec).ok
    assert not v("term.fits", _thresholds(auto=0.89), rec).ok
    # The stat below auto is not the citation's business: the policy blocks before it.
    assert Policy(_defaults(_thresholds(auto=0.99)), validator=v).verdict(rec).auto_blockers == (
        "below_auto",
    )


NEGATIONS: list[tuple[str, dict[str, Any], str]] = [
    ("loco", {"loco": False}, "leave-one-card-out"),
    ("pre_registered", {"pre_registered": False}, "pre_registered"),
    ("selection", {"selection": "full"}, "selection"),
    ("exploratory", {"exploratory": True}, "exploratory"),
    ("servable", {"servable": False}, "servable"),
    ("n_folds", {"n_folds": MIN_NESTED_FOLDS - 1}, "folds"),
    ("teacher_in_test", {"teacher_in_test": True}, "teacher_in_test"),
    # M2's cells record masked: true everywhere and mask = the task's rule ("aspect" or None);
    # term.fits has no mask, so masked: false and a mask of "aspect" are both refused.
    ("masked", {"masked": False}, "masked"),
    ("mask", {"mask": "aspect"}, "mask"),
    ("labels_sha256", {"labels_sha256": "c" * 64}, "committed snapshot"),
    ("n_neg", {"items": planted_items(per_card=4)}, "n_neg"),
    ("question_key", {"question_key": "0" * 16}, "question_key"),
    ("fingerprint", {"fp": {**FP.as_dict(), "encoder_fp": "0" * 12}}, "encoder_fp"),
    (
        "serving_lock_sha",
        {"fp": {**FP.as_dict(), "serving_lock_sha": "e" * 64}},
        "serving_lock_sha",
    ),
    (
        "fold_choices",
        {"choices": m4.fold_choices(framing=TERM.id, agree=FOLD_AGREEMENT_MIN - 1)},
        "fold_choices",
    ),
    ("beats_lookup_novel", {"baselines": None}, "beats_lookup_novel"),
    ("threshold_cp_none", {"threshold_cp": {"0.05": None, "0.10": 0.5}}, "Clopper-Pearson"),
    ("tier", {"tier": "probe", "spec": "lowdim.v1"}, "level"),
]


@pytest.mark.parametrize(("field", "over", "reason"), NEGATIONS, ids=[n[0] for n in NEGATIONS])
def test_each_field_negated_once_refuses(
    card: DatasetCard,
    world: tuple[Path, str, Callable[..., None]],
    field: str,
    over: dict[str, Any],
    reason: str,
) -> None:
    root, _, write = world
    if field == "beats_lookup_novel":
        base = cell("term.fits", "calibrated", labels_sha256=world[1])
        assert base.baselines is not None
        over = {"baselines": base.baselines.model_copy(update={"beats_lookup_novel": False})}
    write(**over)
    rec = _record(card)
    t = _thresholds()
    if field == "tier":  # the probe cell is keyed by its tier: cite it as such
        t = _thresholds(cite=f"bench/results/{DATE}/{NAME}.json#neon_term_fits.probe.{TERM.id}")
    check = _validator(root)("term.fits", t, rec)
    assert not check.ok and reason in check.reason, check.reason
    assert Policy(_defaults(t), validator=_validator(root)).verdict(rec).auto_blockers == (
        "cite_refused",
    )


def test_guard_counts_and_the_other_refusals(
    card: DatasetCard, world: tuple[Path, str, Callable[..., None]]
) -> None:
    root, _, _ = world
    rec = _record(card)
    v = _validator(root)
    assert "nothing to cite" in v("term.fits", _thresholds(auto=None, cite=None), rec).reason
    assert "cites nothing" in v("term.fits", _thresholds(cite=None), rec).reason
    assert "is not bench/results" in v("term.fits", _thresholds(cite="x.json#a.b.c"), rec).reason
    missing = _thresholds(
        cite=f"bench/results/{DATE}/none.json#neon_term_fits.calibrated.{TERM.id}"
    )
    assert "no such results file" in v("term.fits", missing, rec).reason
    no_cell = _thresholds(cite=f"bench/results/{DATE}/{NAME}.json#neon_term_fits.calibrated.F4")
    assert "no cell" in v("term.fits", no_cell, rec).reason
    assert "cell" in v("column.ontology_fits", _thresholds(), rec).reason  # another task's record
    assert "below" in v("term.fits", _thresholds(auto=0.5), rec).reason
    assert GUARD_MIN == 30 and MIN_NESTED_FOLDS == 5 and FOLD_AGREEMENT_MIN == 5
    # Unbound (no live fingerprint): serving_lock_sha cannot be checked, so refused.
    unbound = _validator(root, live=None)
    assert "no live fingerprint" in unbound("term.fits", _thresholds(), rec).reason
    unbound.bind(FP)
    assert unbound("term.fits", _thresholds(), rec).ok
    # An unreadable results file is a refusal, not an exception.
    (root / "bench" / "results" / DATE / f"{NAME}.json").write_text("{nope")
    fresh = _validator(root)
    assert "not a results file" in fresh("term.fits", _thresholds(), rec).reason


def test_a_probe_cell_is_matched_by_model_and_spec(
    card: DatasetCard, world: tuple[Path, str, Callable[..., None]]
) -> None:
    root, sha, _ = world
    probe_cell = cell("term.fits", "probe", labels_sha256=sha, spec="lowdim.v1")
    m4.results_file(root, DATE, "x3", [probe_cell], labels_sha256=sha)
    cite = f"bench/results/{DATE}/x3.json#neon_term_fits.probe.{TERM.id}"
    rec = _record(card).revise(level="probe", feature_spec="lowdim.v1")
    v = _validator(root)
    assert v("term.fits", _thresholds(cite=cite), rec).ok
    other_spec = rec.revise(feature_spec="pair512.v1")
    assert "fold_choices" in v("term.fits", _thresholds(cite=cite), other_spec).reason
    # A calibrated record cannot cite the probe cell, nor a probe record the calibrated cell.
    assert "level" in v("term.fits", _thresholds(cite=cite), _record(card)).reason
    assert "level" in v("term.fits", _thresholds(), rec).reason
    # A probe cell whose folds name no spec never matches a record (no None == None loophole).
    bare = cell("term.fits", "probe", labels_sha256=sha, choices=m4.fold_choices(framing=TERM.id))
    m4.results_file(root, DATE, "x3", [bare], labels_sha256=sha)
    assert (
        "fold_choices"
        in CellCitationValidator(root, live=FP)("term.fits", _thresholds(cite=cite), rec).reason
    )


def _ontology_record(card: DatasetCard) -> DecisionRecord:
    """A Platt-calibrated column.ontology_fits record (its options come back masked by the
    state's aspect, as served; ``pato`` wins clearly)."""
    bundle = ArtifactBundle(
        version="v1",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        calibrators={ONTOLOGY.question_key: PlattCalibrator(a=2.0, b=0.0)},
    )
    provider = TieredProvider(ScriptedClm({"pato": 6.0}), None, FP, bundle, method="clm")
    return provider.decide(ONTOLOGY, [_target(card)], [fr.ontology_candidates()])[0]


def test_masking_is_read_as_the_cells_record_it(
    card: DatasetCard, world: tuple[Path, str, Callable[..., None]]
) -> None:
    """``tiers.json#neon_term_fits.*`` cells carry ``masked: true, mask: null``;
    ``neon_ontology_fits.*`` cells ``masked: true, mask: "aspect"`` (``assemble_cell`` from
    ``bench.tasks.neon.MASKS``). Each qualifies for its own task's record; a cell whose
    ``masked`` is false, or whose ``mask`` differs from the framing's rule, is refused."""
    root, sha, write = world
    v = _validator(root)
    term = _record(card)
    term_cell = cell("term.fits", "calibrated", labels_sha256=sha)
    assert term_cell.masked is True and term_cell.mask is None and TERM.mask_rule is None
    assert v("term.fits", _thresholds(), term).ok
    # column.ontology_fits: masked true with the aspect mask, as the F9 cells record it.
    onto_cell = cell("column.ontology_fits", "calibrated", labels_sha256=sha)
    assert onto_cell.masked is True and onto_cell.mask == "aspect" == ONTOLOGY.mask_rule
    m4.results_file(root, DATE, "onto", [onto_cell], labels_sha256=sha)
    onto_cite = f"bench/results/{DATE}/onto.json#neon_ontology_fits.calibrated.{ONTOLOGY.id}"
    base = SHIPPED.thresholds("column.ontology_fits").model_dump()
    t_onto = Thresholds.model_validate(
        {**base, "auto": 0.95, "margin": 0.0, "cite": onto_cite, "risk": 0.05}
    )
    rec = _ontology_record(card)
    assert rec.level == "calibrated" and rec.masked is not None and any(rec.masked)
    check = v("column.ontology_fits", t_onto, rec)
    assert check.ok, check.reason
    # The same cell with the mask missing, or masked: false, is refused for that record.
    for over, word in (({"mask": None}, "mask"), ({"masked": False}, "masked")):
        bad = cell("column.ontology_fits", "calibrated", labels_sha256=sha, **over)
        m4.results_file(root, DATE, "onto", [bad], labels_sha256=sha)
        reason = CellCitationValidator(root, live=FP)("column.ontology_fits", t_onto, rec).reason
        assert word in reason, reason
    # And a term.fits cell that claims the aspect mask does not fit a term.fits record.
    write(mask="aspect")
    assert (
        "mask='aspect'"
        in CellCitationValidator(root, live=FP)("term.fits", _thresholds(), term).reason
    )


# -- the audit blocker (plan §4.7) ---------------------------------------------------------------


def test_prod_needs_a_passing_audit_for_the_artifact_version(
    card: DatasetCard, world: tuple[Path, str, Callable[..., None]], tmp_path: Path
) -> None:
    root, _, _ = world
    rec = _record(card)
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    check = StoreAuditCheck(store)
    assert isinstance(check, AuditCheck)
    policy = Policy(_defaults(_thresholds()), validator=_validator(root), audit_check=check)
    v = policy.verdict(rec)
    assert (v.outcome, v.reason, v.auto_blockers) == (
        "proposed",
        "audit_required",
        ("audit_required",),
    )
    assert "audit_required" in AUTO_BLOCKERS and DEMOTION_REASONS == ("audit_required",)
    # A failed audit, or one of another version, does not unblock.
    failed = AuditRow(
        task_key=rec.task_key, artifact_version="v1", n=10, n_cards=1, cards=["c"],
        reviewer="r", n_errors=0, cp95_upper=0.26, risk=0.05, passed=False,
    )  # fmt: skip
    store.insert_audit(failed)
    assert policy.outcome(rec) == "proposed"
    store.insert_audit(
        AuditRow(
            task_key=rec.task_key,
            artifact_version="v2",
            n=60,
            n_cards=4,
            cards=list("abcd"),
            reviewer="r",
            n_errors=1,
            cp95_upper=0.077,
            risk=0.05,
            passed=True,
        )
    )
    assert policy.outcome(rec) == "proposed"
    store.insert_audit(
        AuditRow(
            task_key=rec.task_key,
            artifact_version="v1",
            n=60,
            n_cards=4,
            cards=list("abcd"),
            reviewer="r",
            n_errors=1,
            cp95_upper=0.077,
            risk=0.05,
            passed=True,
        )
    )
    assert policy.outcome(rec) == "auto"
    # dev never autos, and a check that raises fails closed.
    assert policy.outcome(rec, profile=SHIPPED.profile("dev")) == "proposed"

    def broken(key: str, version: str) -> bool:
        raise RuntimeError("no sidecar")

    assert (
        Policy(_defaults(_thresholds()), validator=_validator(root), audit_check=broken).outcome(
            rec
        )
        == "proposed"
    )
    # The audit rule itself is enforced by the row model.
    with pytest.raises(ValueError, match="passed must be"):
        AuditRow(
            task_key="k", artifact_version="v", n=10, n_cards=1, cards=["c"], reviewer="r",
            n_errors=0, cp95_upper=0.26, risk=0.05, passed=True,
        )  # fmt: skip


def test_policy_load_wires_the_validator_by_default_and_the_pipeline_binds(tmp_path: Path) -> None:
    from tests.fakes.pipeline import annotator

    policy = Policy.load()
    assert isinstance(policy.validator, CellCitationValidator)
    assert policy.validator.results_root == default_results_root()
    assert policy.validator.live is None and policy.audit_check is None
    assert (default_results_root() / "pyproject.toml").is_file()  # this checkout
    rooted = Policy.load(results_root=tmp_path)
    assert isinstance(rooted.validator, CellCitationValidator)
    assert rooted.validator.results_root == tmp_path
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    ann = annotator(store)
    bound = ann.policy
    assert isinstance(bound.validator, CellCitationValidator)
    assert bound.validator.live == ann.provider.fingerprint
    assert isinstance(bound.audit_check, StoreAuditCheck)
