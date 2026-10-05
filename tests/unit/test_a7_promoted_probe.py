"""DESIGN A7 in the decide phase: a promoted probe decides a task ``decider.ols_rank_tasks``
names.

With the shipped configuration (``ols_rank_tasks: [term.fits]``, K1) and a provider whose
promotion table (``CURRENT.json``) names a probe for ``term.fits``' active framing, the probe
decides every Q4, Q5 and Q6 group at level ``probe`` (the calibrator's kind, ``feature_spec``
and ``artifact_version`` set, proposed or abstain, never ``auto``), D24's refinement is asked
again where the winner has OLS children (the group carries ``p_fit``), the run lists the task
under ``probe_tasks`` and not ``ols_rank_tasks`` and is not ``degraded``. Without a promoted
probe the K1 behaviour is unchanged (``test_k1_default.py``); ``--tier ols_rank`` still sends
every candidate group to ``ols_rank``; an explicit ``zero_shot`` keeps K1; a promoted probe for a
task outside ``ols_rank_tasks`` (``column.ontology_fits``) decides it at ``auto`` as before.
Synthetic only: a probe fitted on random features under the fake stack's fingerprint
(``tests/fakes/m4.synthetic_probe``), the fake provider (never evidence), OLS replayed from the
fixtures, DuckDB under ``tmp_path``."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from mesa_clm import framings as fr
from mesa_clm.cli import EXIT_OK, _ols_rank_note, main
from mesa_clm.pipeline import SPECIFICITY_NOT_ASKED, AnnotationRun
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers.tiered import (
    ArtifactBundle,
    FakeProvider,
    Promotion,
    ServedProbe,
    fake_fingerprint,
)
from mesa_clm.tasks import RANK_FIT_TASKS
from tests.fakes.m4 import synthetic_probe
from tests.fakes.pipeline import FAKE_SEED, SERVICE_CARD, annotator, card, shipped_config

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "tests" / "fixtures" / "cards" / f"{SERVICE_CARD}.md"
OLS_DIR = ROOT / "tests" / "fixtures" / "ols"
TERM = fr.active_framing("term.fits")
ONTOLOGY = fr.active_framing("column.ontology_fits")
_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")


def promoted_bundle(*tasks: str, version: str = "v1") -> ArtifactBundle:
    """A bundle under the fake stack's fingerprint whose promotion table names a synthetic
    ``lowdim.v1`` probe for each task's active framing (what ``CURRENT.json`` gives the live
    provider after ``learn promote --tier probe``)."""
    fp = fake_fingerprint(seed=FAKE_SEED)
    probes = [synthetic_probe("lowdim.v1", task_id=t, fp=fp) for t in tasks]
    return ArtifactBundle(
        version=version,
        encoder_fp=fp.encoder_fp,
        clm_model_fp=fp.clm_model_fp,
        probes={p.question_key: ServedProbe(p, version=version) for p in probes},
        promoted={
            p.task_id: Promotion(tier="probe", question_key=p.question_key, version=version)
            for p in probes
        },
    )


def promoted_provider(*tasks: str) -> FakeProvider:
    return FakeProvider(seed=FAKE_SEED, artifacts=promoted_bundle(*tasks))


def _rows(store: DuckDBStore, run: AnnotationRun) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in store.decisions(run.run_id):
        by_task[str(d["task_id"])].append(d)
    return {
        "run": store.run(run.run_id) or {},
        "by_task": by_task,
        "groups": store.groups(run.run_id),
        "links": store.links(run.run_id),
    }


@pytest.fixture(scope="module")
def promoted(tmp_path_factory: pytest.TempPathFactory) -> tuple[DuckDBStore, AnnotationRun]:
    """The fixture card annotated once under the shipped decider defaults with a promoted
    ``term.fits`` probe (fake head)."""
    store = DuckDBStore(tmp_path_factory.mktemp("a7") / "prov.duckdb")
    run = annotator(store, provider=promoted_provider("term.fits"), cfg=shipped_config()).annotate(
        card(SERVICE_CARD)
    )
    return store, run


# -- the resolution the pipeline asks for -------------------------------------------------------


def test_the_provider_resolves_the_promoted_probe_for_the_active_framing() -> None:
    provider = promoted_provider("term.fits")
    assert provider.resolve_tier(TERM.question_key) == "probe"
    assert provider.resolve_tier(ONTOLOGY.question_key) == "zero_shot"
    assert provider.supports_tier("term.fits", "probe")
    assert not provider.supports_tier("column.ontology_fits", "probe")
    # The shipped default is unchanged: the provider, not the configuration, lifts the task.
    assert shipped_config().decider.ols_rank_tasks == ["term.fits"]
    ann = annotator(None, provider=provider, cfg=shipped_config())
    assert ann.ols_rank_tasks == frozenset() and ann.probe_tasks == frozenset({"term.fits"})
    plain = annotator(None, cfg=shipped_config())
    assert plain.ols_rank_tasks == frozenset({"term.fits"}) and plain.probe_tasks == frozenset()


# -- the decide phase under a promoted term.fits probe -----------------------------------------


def test_term_fits_is_decided_by_the_promoted_probe(
    promoted: tuple[DuckDBStore, AnnotationRun],
) -> None:
    store, run = promoted
    rows = _rows(store, run)
    assert run.ols_rank_tasks == () and run.probe_tasks == ("term.fits",)
    data = run.to_dict()
    assert data["ols_rank_tasks"] == [] and data["probe_tasks"] == ["term.fits"]
    # The probe is the design, not a fallback: nothing degraded, nothing ols_rank.
    assert not run.degraded and rows["run"]["degraded"] is False
    assert rows["run"]["artifacts_version"] == "v1"
    terms = rows["by_task"]["term.fits"]
    assert terms and {d["method"] for d in terms} == {"fake"}
    for d in terms:
        assert (d["level"], d["calibration"]) == ("probe", "platt")
        assert d["feature_spec"] == "lowdim.v1" and d["artifact_version"] == "v1"
        assert d["outcome"] in ("proposed", "abstain", "rejected")
        assert d["outcome"] != "auto"
        assert (d["framing_id"], d["question_key"]) == (TERM.id, TERM.question_key)
    assert all(
        d["feature_spec"] is None
        for t, ds in rows["by_task"].items()
        if t != "term.fits"
        for d in ds
    )
    assert not any(g["method"] == "ols_rank" for g in rows["groups"])
    assert run.proposals
    for p in run.proposals:
        assert (p.method, p.level, p.calibration, p.outcome) == (
            "fake",
            "probe",
            "platt",
            "proposed",
        )
        assert p.p_fit is not None and "ols_rank" not in p.rationale
    assert {link["write_status"] for link in rows["links"]} == {"proposed"}
    assert "auto" not in run.outcomes
    # Q3 is still the zero-shot F9 answer: only term.fits has a promoted probe here.
    q3 = rows["by_task"]["column.ontology_fits"]
    assert q3 and {d["level"] for d in q3 if d["method"] == "fake"} == {"zero_shot"}


def test_d24_is_asked_again_for_probe_groups(
    promoted: tuple[DuckDBStore, AnnotationRun],
) -> None:
    """A probe record carries ``p_fit``, so the D24 refinement runs where a proposed winner has
    OLS children: a refinement group escalated from the parent exists, also decided by the probe
    and capped at ``proposed``; no group records that specificity was not asked."""
    store, run = promoted
    rows = _rows(store, run)
    groups = [g for g in rows["groups"] if g["task_id"] == "term.fits"]
    assert groups
    assert not any(g["search_json"].get("specificity") == SPECIFICITY_NOT_ASKED for g in groups)
    refined = {UUID(str(g["escalated_from"])) for g in groups if g["escalated_from"]}
    assert refined, "the fixture card must have a proposed probe winner with OLS children"
    escalated = {UUID(str(g["group_id"])) for g in groups if g["escalated_from"]}
    by_group: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
    for d in rows["by_task"]["term.fits"]:
        by_group[UUID(str(d["group_id"]))].append(d)
    for gid in escalated:
        for d in by_group[gid]:
            assert d["level"] == "probe" and d["p_fit"] is not None
            assert d["outcome"] in ("proposed", "abstain", "rejected")
    for p in run.proposals:
        if p.candidate.has_children and p.group_id not in escalated:
            assert p.group_id in refined, p.candidate.curie
        assert "D24) not asked" not in p.rationale


# -- what does not change --------------------------------------------------------------------


def test_the_ols_rank_tier_overrides_the_promoted_probe(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(
        store, provider=promoted_provider("term.fits"), cfg=shipped_config(), tier="ols_rank"
    ).annotate(card(SERVICE_CARD))
    assert run.degraded and run.ols_rank_tasks == tuple(sorted(RANK_FIT_TASKS))
    assert run.probe_tasks == () and run.to_dict()["probe_tasks"] == []
    rows = _rows(store, run)
    for task_id in RANK_FIT_TASKS:
        # The unit aspect's UO answer is M2's rule record (``unit_aspect``), never a tier's; since
        # DESIGN A8 the fallback gives the card's unit-bearing numeric columns that aspect.
        methods = {d["method"] for d in rows["by_task"][task_id] if d["reason"] != "unit_aspect"}
        assert methods == {"ols_rank"}, task_id
    assert all(p.method == "ols_rank" and p.p_fit is None for p in run.proposals)


def test_an_explicit_zero_shot_tier_keeps_k1(tmp_path: Path) -> None:
    """``--tier zero_shot`` cannot serve the probe, so the K1 default stands: ``term.fits`` is
    ``ols_rank`` (its zero-shot answers stay audit-only, A1), not a zero-shot proposal."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(
        store, provider=promoted_provider("term.fits"), cfg=shipped_config(), tier="zero_shot"
    ).annotate(card(SERVICE_CARD))
    assert run.ols_rank_tasks == ("term.fits",) and run.probe_tasks == ()
    assert not run.degraded
    assert {d["method"] for d in _rows(store, run)["by_task"]["term.fits"]} == {"ols_rank"}


def test_an_audit_run_asks_the_probe_too(tmp_path: Path) -> None:
    """``ols_rank_tasks: []`` with a promoted probe: the probe decides (tier auto), and the run
    still names it under ``probe_tasks``."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(
        store, provider=promoted_provider("term.fits"), cfg=shipped_config(), ols_rank_tasks=()
    ).annotate(card(SERVICE_CARD))
    assert run.ols_rank_tasks == () and run.probe_tasks == ("term.fits",)
    assert {d["level"] for d in _rows(store, run)["by_task"]["term.fits"]} == {"probe"}


def test_a_promoted_probe_outside_ols_rank_tasks_decides_at_auto(tmp_path: Path) -> None:
    """``column.ontology_fits`` is not in ``ols_rank_tasks``: its promoted probe decides Q3 at
    tier ``auto`` (level ``probe``, the aspect mask applied after scoring), while ``term.fits``
    stays ``ols_rank`` under K1 and the run says which is which."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(
        store, provider=promoted_provider("column.ontology_fits"), cfg=shipped_config()
    ).annotate(card(SERVICE_CARD))
    assert run.ols_rank_tasks == ("term.fits",)
    assert run.probe_tasks == ("column.ontology_fits",)
    assert not run.degraded
    rows = _rows(store, run)
    q3 = [d for d in rows["by_task"]["column.ontology_fits"] if d["method"] == "fake"]
    assert q3 and {(d["level"], d["calibration"]) for d in q3} == {("probe", "platt")}
    assert {d["feature_spec"] for d in q3} == {"lowdim.v1"}
    assert all(d["outcome"] != "auto" for d in q3)
    assert {d["method"] for d in rows["by_task"]["term.fits"]} == {"ols_rank"}
    assert run.proposals and all(p.method == "ols_rank" for p in run.proposals)
    # Both promoted: nothing is ols_rank any more.
    both = DuckDBStore(tmp_path / "both.duckdb")
    run2 = annotator(
        both, provider=promoted_provider("term.fits", "column.ontology_fits"), cfg=shipped_config()
    ).annotate(card(SERVICE_CARD))
    assert run2.ols_rank_tasks == () and run2.probe_tasks == tuple(sorted(RANK_FIT_TASKS))
    assert not any(g["method"] == "ols_rank" for g in _rows(both, run2)["groups"])


# -- the command line --------------------------------------------------------------------------


def test_the_summary_note_names_both_sets() -> None:
    assert "promoted probe decides: term.fits (DESIGN A7" in _ols_rank_note(
        (), "auto", ("term.fits",)
    )
    assert "audit run" not in _ols_rank_note((), "auto", ("term.fits",))
    assert "audit run" in _ols_rank_note((), "auto", ())
    assert _ols_rank_note((), "ols_rank", ("term.fits",)).startswith("ols_rank: every")
    k1 = _ols_rank_note(("term.fits",), "auto", ("column.ontology_fits",))
    assert "K1, DESIGN A1" in k1 and "promoted probe decides: column.ontology_fits" in k1


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(OLS_DIR))
    monkeypatch.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    db = tmp_path / "prov.duckdb"
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{db}")
    monkeypatch.setenv("USER", "alice")
    return db


def test_cli_annotate_reports_the_promoted_probe(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fake provider the CLI builds carries no artifacts; stand one with a promoted probe in
    for it (what ``live.clm_provider`` builds from ``CURRENT.json``) and check the summary and
    the ``--out`` JSON."""
    provider = promoted_provider("term.fits")
    monkeypatch.setattr(
        "mesa_clm.cli._annotate_provider",
        lambda cfg, args, tier: (provider, tier, ["provider fake (promoted probe)"], lambda: None),
    )
    base = ["annotate", "--card", str(CARD), "--provider", "fake", "--out", "-"]
    assert main(base) == EXIT_OK
    out = capsys.readouterr()
    data = json.loads(out.out)
    assert data["ols_rank_tasks"] == [] and data["probe_tasks"] == ["term.fits"]
    assert not data["degraded"] and data["tier"] == "auto"
    assert data["proposals"] and {p["level"] for p in data["proposals"]} == {"probe"}
    assert all(p["outcome"] == "proposed" and p["p_fit"] is not None for p in data["proposals"])
    assert "promoted probe decides: term.fits (DESIGN A7" in out.err
    assert "audit run" not in out.err
    assert main([*base, "--tier", "ols_rank"]) == EXIT_OK
    out = capsys.readouterr()
    data = json.loads(out.out)
    assert data["probe_tasks"] == [] and data["degraded"]
    assert {p["method"] for p in data["proposals"]} == {"ols_rank"}
    assert "ols_rank: every candidate group" in out.err
