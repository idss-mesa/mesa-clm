"""DESIGN A1 in the decide phase: the shipped defaults after M2's registered run.

X1 (``bench/results/2026-10-03/x1.json``) found no qualifying arm for ``term.fits`` (K1) and
chose F9 on ``clm-latest`` for ``column.ontology_fits``. So, by default: every ``term.fits`` group
(Q4, Q5, Q6) is decided by ``ols_rank`` (``decider.ols_rank_tasks``: the OLS top-1, proposed,
probabilities null, never auto; D28), D24's refinement is not asked for those groups and the
group records why, the run is not ``degraded`` by it, and CLM is asked about ``term.fits`` only
in an explicitly requested audit run (``decider.ols_rank_tasks: []``, ``annotate
--ols-rank-tasks none``); ``column.ontology_fits`` (Q3) is asked with F9; the closed choices are
unchanged. Hermetic: the fake provider (never evidence), OLS replayed from the fixtures, DuckDB
under ``tmp_path``."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, get_args
from uuid import UUID

import pytest
from pydantic import ValidationError

from mesa_clm import framings as fr
from mesa_clm.cli import EXIT_CONFIG, EXIT_OK, main
from mesa_clm.config import RankFitTask, load_config
from mesa_clm.pipeline import SPECIFICITY_NOT_ASKED, AnnotationRun
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers.base import sha256_text
from mesa_clm.tasks import RANK_FIT_TASKS
from tests.fakes.pipeline import (
    FAKE_SEED,
    SERVICE_CARD,
    annotator,
    card,
    config,
    service,
    shipped_config,
)

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "tests" / "fixtures" / "cards" / f"{SERVICE_CARD}.md"
OLS_DIR = ROOT / "tests" / "fixtures" / "ols"
F9_ONTOLOGY = fr.framing("column.ontology_fits", "F9")
F7_TERM = fr.framing("term.fits", "F7")
_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")


def _rows(store: DuckDBStore, run: AnnotationRun) -> dict[str, Any]:
    decisions = store.decisions(run.run_id)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        by_task[str(d["task_id"])].append(d)
    return {
        "run": store.run(run.run_id) or {},
        "by_task": by_task,
        "groups": store.groups(run.run_id),
        "links": store.links(run.run_id),
    }


@pytest.fixture(scope="module")
def shipped(tmp_path_factory: pytest.TempPathFactory) -> tuple[DuckDBStore, AnnotationRun]:
    """The fixture card annotated once under the shipped decider defaults (fake head)."""
    store = DuckDBStore(tmp_path_factory.mktemp("k1") / "prov.duckdb")
    run = service(store, cfg=shipped_config()).annotate(
        card(SERVICE_CARD), "agent-x", owner="alice"
    )
    return store, run


# -- the configuration -------------------------------------------------------------------------


def test_the_shipped_decider_sends_term_fits_to_ols_rank() -> None:
    assert load_config(env={}).decider.ols_rank_tasks == ["term.fits"]
    assert frozenset(get_args(RankFitTask)) == RANK_FIT_TASKS
    for text, want in (
        ("[]", []),
        ("none", []),
        ("term.fits,column.ontology_fits", ["term.fits", "column.ontology_fits"]),
        ("[term.fits, term.fits]", ["term.fits"]),
    ):
        cfg = load_config(env={"MESA_CLM_DECIDER__OLS_RANK_TASKS": text})
        assert cfg.decider.ols_rank_tasks == want, text
    with pytest.raises(ValidationError):
        load_config(env={"MESA_CLM_DECIDER__OLS_RANK_TASKS": "column.aspect"})


def test_a_task_outside_rank_fit_is_refused_by_the_annotator() -> None:
    with pytest.raises(ValueError, match="candidate groups"):
        annotator(None, cfg=shipped_config(), ols_rank_tasks={"avu.value_kind"})


# -- the decide phase under the shipped defaults ---------------------------------------------


def test_term_fits_proposals_are_ols_rank_by_default(
    shipped: tuple[DuckDBStore, AnnotationRun],
) -> None:
    store, run = shipped
    rows = _rows(store, run)
    assert run.ols_rank_tasks == ("term.fits",)
    assert run.to_dict()["ols_rank_tasks"] == ["term.fits"]
    # K1 is the design, not a fallback: the run is not degraded by it (D28).
    assert not run.degraded and rows["run"]["degraded"] is False
    terms = rows["by_task"]["term.fits"]
    assert terms and {d["method"] for d in terms} == {"ols_rank"}
    for d in terms:
        assert (d["level"], d["calibration"]) == ("none", "none")
        assert d["probs"] is None and d["p_fit"] is None
        assert d["outcome"] in ("proposed", "abstain") and d["rank"] is not None
        # term.fits keeps F7 as its active framing (its audit records' question_key).
        assert (d["framing_id"], d["question_key"]) == ("F7", F7_TERM.question_key)
    assert run.proposals
    term_groups = {UUID(str(g["group_id"])) for g in rows["groups"] if g["task_id"] == "term.fits"}
    for p in run.proposals:
        assert p.group_id in term_groups
        assert (p.method, p.level, p.calibration, p.outcome) == (
            "ols_rank",
            "none",
            "none",
            "proposed",
        )
        assert p.p_fit is None and p.rationale.startswith("OLS rank 1 (ols_rank, D28)")
    assert {link["write_status"] for link in rows["links"]} == {"proposed"}
    assert not any(o == "auto" for o in run.outcomes)
    # The scopes K1 covers: column terms (Q4) and the site biomes (Q5) or the taxon (Q6).
    assert {p.scope for p in run.proposals} >= {"column"}
    assert {g["scope"] for g in rows["groups"] if g["task_id"] == "term.fits"} >= {
        "column",
        "site",
    }


def test_d24_is_not_asked_for_ols_rank_groups_and_says_so(
    shipped: tuple[DuckDBStore, AnnotationRun],
) -> None:
    """D24's refinement needs ``p_fit``: no refinement group exists, and every proposed group
    whose winner has OLS children records that the step was not asked, as does its proposal."""
    store, run = shipped
    rows = _rows(store, run)
    groups = [g for g in rows["groups"] if g["task_id"] == "term.fits"]
    assert not any(g["escalated_from"] for g in groups)
    marked = {UUID(str(g["group_id"])) for g in groups if "specificity" in g["search_json"]}
    assert marked, "the fixture card must have a proposed winner with OLS children"
    for g in groups:
        if UUID(str(g["group_id"])) in marked:
            assert g["search_json"]["specificity"] == SPECIFICITY_NOT_ASKED
            assert g["method"] == "ols_rank" and g["outcome"] == "proposed"
    by_group = {p.group_id: p for p in run.proposals}
    for gid in marked:
        if gid in by_group:  # the keep rule may have dropped it (D25)
            assert by_group[gid].candidate.has_children
            assert "specificity (D24) not asked: ols_rank has no p_fit" in by_group[gid].rationale
    for p in run.proposals:
        if p.group_id not in marked:
            assert "D24" not in p.rationale and not p.candidate.has_children


def test_column_ontology_fits_is_asked_with_f9(shipped: tuple[DuckDBStore, AnnotationRun]) -> None:
    """Q3 under A1: the fake CLM answers the F9 context (the query-shaped template), keyed by
    F9's ``question_key``; the closed choices still use F7."""
    store, run = shipped
    rows = _rows(store, run)
    assert fr.active_framing("column.ontology_fits") is F9_ONTOLOGY
    q3 = rows["by_task"]["column.ontology_fits"]
    methods = {d["method"] for d in q3}
    assert q3 and methods <= {"fake", "rule"} and "fake" in methods
    for d in q3:
        assert (d["framing_id"], d["question_key"]) == ("F9", "c95785008b523fd0")
        if d["method"] == "fake":
            text = fr.context_text(F9_ONTOLOGY, d["state_json"])
            assert text.startswith("NEON dataset ") and text.endswith(" ontology term:")
            assert d["context_sha256"] == sha256_text(text)
    for g in rows["groups"]:
        if g["task_id"] == "column.ontology_fits":
            assert g["question_key"] == F9_ONTOLOGY.question_key
    for task_id in ("column.annotate", "column.aspect", "avu.value_kind"):
        for d in rows["by_task"][task_id]:
            assert d["framing_id"] == "F7"
            assert d["question_key"] == fr.framing(task_id, "F7").question_key


def test_term_fits_is_asked_of_clm_only_in_an_explicit_audit_run(tmp_path: Path) -> None:
    """``decider.ols_rank_tasks: []`` (or ``ols_rank_tasks=()`` from the caller) asks CLM for
    ``term.fits`` again: its decisions are zero_shot (never auto, D6), the K1 audit path."""
    for i, (cfg, kw) in enumerate(
        (
            (shipped_config(DECIDER__OLS_RANK_TASKS="[]"), {}),
            (shipped_config(), {"ols_rank_tasks": ()}),
            (config(), {}),  # the pipeline tests' helper is an audit configuration
        )
    ):
        store = DuckDBStore(tmp_path / f"prov{i}.duckdb")
        run = annotator(store, cfg=cfg, **kw).annotate(card(SERVICE_CARD))
        terms = _rows(store, run)["by_task"]["term.fits"]
        assert run.ols_rank_tasks == () and terms
        assert {d["method"] for d in terms} == {"fake"}
        assert {d["level"] for d in terms} == {"zero_shot"}
        assert all(p.method == "fake" and p.outcome != "auto" for p in run.proposals)


def test_the_ols_rank_tier_is_still_a_degraded_run(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, cfg=shipped_config(), tier="ols_rank").annotate(card(SERVICE_CARD))
    assert run.degraded and run.ols_rank_tasks == tuple(sorted(RANK_FIT_TASKS))


# -- the command line --------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A clean environment with the shipped decider defaults: replayed OLS, a sidecar under
    tmp_path, actor alice."""
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


def _annotate(capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[dict[str, Any], str]:
    code = main(
        [
            "annotate",
            "--card",
            str(CARD),
            "--provider",
            "fake",
            "--fake-seed",
            str(FAKE_SEED),
            "--out",
            "-",
            *extra,
        ]
    )
    assert code == EXIT_OK
    out = capsys.readouterr()
    return dict(json.loads(out.out)), out.err


def test_cli_annotate_defaults_to_k1(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, summary = _annotate(capsys)
    assert data["ols_rank_tasks"] == ["term.fits"] and not data["degraded"]
    assert data["proposals"] and {p["method"] for p in data["proposals"]} == {"ols_rank"}
    assert all(p["outcome"] == "proposed" and p["p_fit"] is None for p in data["proposals"])
    assert "ols_rank tasks: term.fits (term.fits: K1, DESIGN A1" in summary
    store = DuckDBStore(cli_env)
    [row] = store.runs()
    methods = Counter(
        (d["task_id"], d["method"]) for d in store.decisions(UUID(str(row["run_id"])))
    )
    assert {m for (t, m) in methods if t == "term.fits"} == {"ols_rank"}
    assert {m for (t, m) in methods if t == "column.ontology_fits"} <= {"fake", "rule"}


def test_cli_annotate_audit_run_and_bad_tasks(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, summary = _annotate(capsys, "--ols-rank-tasks", "none")
    assert data["ols_rank_tasks"] == [] and {p["method"] for p in data["proposals"]} == {"fake"}
    assert "term.fits asked of CLM: an audit run" in summary
    both, _ = _annotate(capsys, "--ols-rank-tasks", "term.fits,column.ontology_fits")
    assert both["ols_rank_tasks"] == ["column.ontology_fits", "term.fits"]
    evaluated, _ = _annotate(capsys, "--eval-result")
    assert evaluated["ols_rank_tasks"] == ["term.fits"]
    assert {a["method"] for a in evaluated["avus"]} == {"ols_rank"}
    for bad in ("column.aspect", "term.fits,avu.keep", ""):
        code = main(
            ["annotate", "--card", str(CARD), "--provider", "fake", "--ols-rank-tasks", bad]
        )
        assert code == EXIT_CONFIG, bad
        assert "--ols-rank-tasks takes rank_fit tasks" in capsys.readouterr().err
