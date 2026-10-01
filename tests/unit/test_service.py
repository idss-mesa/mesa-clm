"""``DecisionService`` on the fake provider with replayed OLS (plan §7.1, §8 M1-B): the decider
lock, the run summary and its pending groups, the offered candidates rebuilt from the sidecar,
explain, links built by a pick (defect (c)), and the Claude second opinion (D22).

Each test copies one annotated sidecar file (``tests/fakes/pipeline.annotated_template``)
instead of annotating again; picks and link changes stay local to the test.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from mesa_clm.avu import VALUE_KIND_LABEL, VALUE_KIND_TOP, build_avu
from mesa_clm.ols import Candidate
from mesa_clm.pipeline import AnnotationRun
from mesa_clm.provenance.models import AvuLinkRow
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.service import DeciderBusy, DecisionService
from tests.fakes.pipeline import (
    SERVICE_CARD,
    FakeClaude,
    annotated_template,
    card,
    copy_store,
    service,
)


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, AnnotationRun]:
    return annotated_template(tmp_path_factory.mktemp("template"))


@pytest.fixture
def svc(template: tuple[Path, AnnotationRun], tmp_path: Path) -> DecisionService:
    return service(copy_store(template[0], tmp_path), bench_cards=())


@pytest.fixture
def run_id(template: tuple[Path, AnnotationRun]) -> UUID:
    return template[1].run_id


def _groups(svc: DecisionService, run_id: UUID, **match: Any) -> list[dict[str, Any]]:
    return [g for g in svc.store.groups(run_id) if all(g.get(k) == v for k, v in match.items())]


def _proposed_group(svc: DecisionService, run_id: UUID) -> dict[str, Any]:
    """A proposed term group with a link and at least three candidates."""
    for g in _groups(svc, run_id, task_id="term.fits", outcome="proposed"):
        links = [k for k in svc.store.links(run_id) if str(k["group_id"]) == str(g["group_id"])]
        if links and g["n_candidates"] >= 3:
            return g
    raise AssertionError("the fixture card has a proposed group with three candidates")


# -- the lock and annotate --------------------------------------------------------------------


def test_annotate_records_owner_and_actor(template: tuple[Path, AnnotationRun]) -> None:
    run = template[1]
    assert run.owner == "alice" and run.actor == "agent-x" and run.proposals
    assert run.to_dict()["owner"] == "alice"


def test_a_busy_decider_raises_decider_busy(svc: DecisionService) -> None:
    svc.max_wait_s = 0.01
    assert svc.lock.acquire(timeout=1)
    try:
        with pytest.raises(DeciderBusy, match="busy"):
            svc.annotate(card(SERVICE_CARD), "agent-x", owner="alice")
    finally:
        svc.lock.release()
    assert not svc.store.runs(owner="bob")


def test_annotate_holds_the_lock_while_it_runs(svc: DecisionService) -> None:
    seen: list[bool] = []
    real = svc.planner.plan

    def plan(c: Any) -> Any:
        seen.append(svc.lock.locked())
        return real(c)

    svc.planner.plan = plan  # type: ignore[method-assign]
    t = threading.Thread(target=svc.annotate, args=(card(SERVICE_CARD), "a"), kwargs={"owner": "a"})
    t.start()
    t.join(timeout=60)
    assert seen == [True] and not svc.lock.locked()


# -- reads ------------------------------------------------------------------------------------


def test_run_summary_lists_what_waits_for_a_reviewer(svc: DecisionService, run_id: UUID) -> None:
    summary = svc.run_summary(run_id)
    pending = set(summary["pending_groups"])
    assert pending and [p["group_id"] for p in summary["pending"]] == summary["pending_groups"]
    groups = {str(g["group_id"]): g for g in summary["groups"]}
    refined = {str(g["escalated_from"]) for g in groups.values() if g["escalated_from"]}
    for gid, g in groups.items():
        waiting = g["outcome"] in ("proposed", "escalated") or g["anchor_won"]
        expected = (
            g["task_id"] == "term.fits"
            and waiting
            and g["outcome"] not in ("human", "rejected")
            and not (gid in refined and _refinement_open(groups, gid))
        )
        assert (gid in pending) is expected, g
    assert any(groups[gid]["anchor_won"] for gid in pending)
    assert all(groups[gid]["task_id"] == "term.fits" for gid in pending)


def _refinement_open(groups: dict[str, dict[str, Any]], parent: str) -> bool:
    return any(
        str(g["escalated_from"]) == parent
        and g["outcome"] in ("auto", "proposed", "escalated", "human")
        for g in groups.values()
    )


def test_candidates_come_from_the_sidecar_best_first_with_the_anchor(
    svc: DecisionService, run_id: UUID
) -> None:
    g = _proposed_group(svc, run_id)
    cands = svc.candidates_for_group(UUID(str(g["group_id"])))
    p_fits = [c["p_fit"] for c in cands]
    assert p_fits == sorted(p_fits, reverse=True)
    anchors = [c for c in cands if c["is_anchor"]]
    assert len(anchors) == 1 and anchors[0]["option_key"] == ANCHOR_KEY
    assert anchors[0]["p_fit"] == pytest.approx(0.5)  # σ(s_c = 0) at zero shot
    assert len(cands) == g["n_candidates"] + 1
    top = cands[0]
    assert not top["is_anchor"] and top["link_id"] and top["write_status"] == "proposed"
    assert all(c["link_id"] is None for c in cands[1:] if not c["is_anchor"])
    assert all(c["level"] == "zero_shot" and c["method"] == "fake" for c in cands)
    assert all(c["label"] and c["iri"] for c in cands if not c["is_anchor"])
    ranks = [c["rank"] for c in cands]
    assert ranks == list(range(1, len(cands) + 1))


def test_ontology_groups_offer_only_the_ontologies_in_play(
    svc: DecisionService, run_id: UUID
) -> None:
    for g in _groups(svc, run_id, task_id="column.ontology_fits"):
        cands = svc.candidates_for_group(UUID(str(g["group_id"])))
        keys = {c["option_key"] for c in cands}
        allowed = set(g["search_json"]["allowed"])
        assert keys == allowed | {ANCHOR_KEY}
        assert all(c["task_id"] == "column.ontology_fits" for c in cands)


def test_explain_a_run(svc: DecisionService, run_id: UUID) -> None:
    out = svc.explain(owner="alice", run_id=run_id, limit=5)
    json.dumps(out)
    assert out["run"]["run_id"] == str(run_id) and out["run"]["owner"] == "alice"
    assert len(out["decisions"]) == 5 < out["n_decisions"]
    assert [d["seq"] for d in out["decisions"]] == [1, 2, 3, 4, 5]
    assert len(out["groups"]) <= 5 and all(len(g["top"]) <= 3 for g in out["groups"])
    assert out["pending_groups"] == svc.run_summary(run_id)["pending_groups"]
    full = svc.explain(owner="alice", run_id=run_id, limit=10_000)
    assert len(full["links"]) == len(svc.store.links(run_id))
    for g in full["groups"]:
        if g["task_id"] == "term.fits" and g["n_candidates"]:
            assert g["top"] and g["top"][0]["p_fit"] == max(t["p_fit"] for t in g["top"])
    with pytest.raises(ValueError, match="run_id or an irods_path"):
        svc.explain(owner="alice")
    with pytest.raises(ValueError, match="limit"):
        svc.explain(owner="alice", run_id=run_id, limit=0)


# -- links built by a pick ----------------------------------------------------------------------


def test_a_winner_pick_accepts_its_link(svc: DecisionService, run_id: UUID) -> None:
    g = _proposed_group(svc, run_id)
    gid = UUID(str(g["group_id"]))
    top = svc.candidates_for_group(gid)[0]
    out = svc.record_human_pick(
        gid, "carol", via="cli", owner="alice", option_key=top["option_key"]
    )
    assert out["accepted_link_id"] == top["link_id"] and out["outcome"] == "human"
    links = {str(k["link_id"]): k for k in svc.store.links(run_id)}
    assert links[top["link_id"]]["write_status"] == "accepted"
    assert links[top["link_id"]]["accepted_by"] == "human"
    assert svc.accepted_link_ids(run_id) == [top["link_id"]]
    assert str(gid) not in svc.run_summary(run_id)["pending_groups"]


def test_a_non_winner_pick_builds_an_accepted_link(svc: DecisionService, run_id: UUID) -> None:
    g = _proposed_group(svc, run_id)
    gid = UUID(str(g["group_id"]))
    cands = svc.candidates_for_group(gid)
    winner = cands[0]
    other = next(c for c in cands[1:] if not c["is_anchor"])
    before = {str(k["link_id"]) for k in svc.store.links(run_id)}
    out = svc.record_human_pick(
        gid, "carol", via="cli", owner="alice", option_key=other["option_key"]
    )
    links = {str(k["link_id"]): k for k in svc.store.links(run_id)}
    new = links[out["accepted_link_id"]]
    assert out["accepted_link_id"] not in before
    assert (new["write_status"], new["accepted_by"]) == ("accepted", "human")
    assert new["term_curie"] == other["option_key"] and new["source"] == "mesa-clm:feedback:cli"
    old = links[winner["link_id"]]
    assert new["value_kind"] == old["value_kind"] and new["column_name"] == old["column_name"]
    assert old["write_status"] == "proposed"  # resolved by the group's outcome, not a status
    term = Candidate(
        label=other["label"],
        curie=other["option_key"],
        iri=other["iri"],
        ontology_id=g["ontology_id"],
    )
    assert (new["attribute"], new["unit"]) == (build_avu(term, "x")["attribute"], term.curie)
    assert svc.store.group(gid)["outcome"] == "human"  # type: ignore[index]


def test_an_anchor_won_group_pick_builds_a_link(svc: DecisionService, run_id: UUID) -> None:
    [g, *_] = [g for g in _groups(svc, run_id, task_id="term.fits") if g["anchor_won"]]
    gid = UUID(str(g["group_id"]))
    cands = svc.candidates_for_group(gid)
    assert cands[0]["is_anchor"]  # the anchor won: it is first
    pick = cands[1]
    out = svc.record_human_pick(
        gid, "carol", via="elicitation", owner="alice", option_key=pick["option_key"]
    )
    [link] = [k for k in svc.store.links(run_id) if str(k["link_id"]) == out["accepted_link_id"]]
    assert link["term_curie"] == pick["option_key"] and link["accepted_by"] == "human"
    assert link["value_kind"] == VALUE_KIND_LABEL or g["scope"] == "site"


def test_a_top_value_kind_reuses_the_winners_value(svc: DecisionService, run_id: UUID) -> None:
    """Defect (c): mesa-anyjev compared the kind with a literal that is not a value kind, so a
    pick next to a most-frequent-value AVU silently got the term label."""
    g = next(g for g in _groups(svc, run_id, task_id="term.fits") if g["anchor_won"])
    gid = UUID(str(g["group_id"]))
    cands = [c for c in svc.candidates_for_group(gid) if not c["is_anchor"]]
    first, second = cands[0], cands[1]
    term = Candidate(
        label=first["label"],
        curie=first["option_key"],
        iri=first["iri"] or "",
        ontology_id=g["ontology_id"],
    )
    avu = build_avu(term, "SRER")
    svc.store.insert_links(
        [
            AvuLinkRow(
                run_id=run_id,
                group_id=gid,
                decision_id=UUID(first["decision_id"]),
                attribute=avu["attribute"],
                value="SRER",
                unit=avu["unit"],
                term_curie=term.curie,
                ontology_id=term.ontology_id,
                column_name=g["column_name"] or "",
                site_code=g["site_code"] or "",
                value_kind=VALUE_KIND_TOP,
            )
        ]
    )
    out = svc.record_human_pick(
        gid, "carol", via="cli", owner="alice", option_key=second["option_key"]
    )
    [link] = [k for k in svc.store.links(run_id) if str(k["link_id"]) == out["accepted_link_id"]]
    assert link["value_kind"] == VALUE_KIND_TOP
    assert link["value"] == "SRER" != second["label"]


# -- the second opinion (D22) -------------------------------------------------------------------


def test_a_disagreeing_second_opinion_escalates(tmp_path: Path) -> None:
    claude = FakeClaude(lambda options: ANCHOR_KEY)
    svc = service_with(tmp_path, claude)
    run = svc.annotate(card(SERVICE_CARD), "agent-x", owner="alice", second_opinion=True)
    assert claude.calls > 0
    escalated = [p for p in run.proposals if p.outcome == "escalated"]
    assert escalated and all("claude prefers __none__" in p.rationale for p in escalated)
    decisions = {str(d["decision_id"]): d for d in svc.store.decisions(run.run_id)}
    second = [d for d in decisions.values() if d["method"] == "claude:structured_output"]
    assert len(second) == claude.calls
    for d in second:
        assert d["level"] == "none" and d["probs"] is None and d["outcome"] != "auto"
        assert decisions[str(d["parent_decision_id"])]["task_id"] == "term.fits"
    groups = {str(g["group_id"]): g for g in svc.store.groups(run.run_id)}
    for p in escalated:
        assert groups[str(p.group_id)]["outcome"] == "escalated"
    links = {str(k["link_id"]): k for k in svc.store.links(run.run_id)}
    assert all(links[str(p.link_id)]["write_status"] == "proposed" for p in escalated)
    pending = set(svc.run_summary(run.run_id)["pending_groups"])
    assert {str(p.group_id) for p in escalated} <= pending


def test_an_agreeing_second_opinion_is_noted(tmp_path: Path) -> None:
    claude = FakeClaude(lambda options: options[0])  # the proposal is offered first
    svc = service_with(tmp_path, claude)
    run = svc.annotate(card(SERVICE_CARD), "agent-x", owner="alice", second_opinion=True)
    asked = [p for p in run.proposals if "claude" in p.rationale]
    assert asked and all(p.outcome == "proposed" and "claude agrees" in p.rationale for p in asked)


def service_with(tmp_path: Path, claude: FakeClaude) -> DecisionService:
    from mesa_clm.provenance.store import DuckDBStore

    return service(DuckDBStore(tmp_path / "prov.duckdb"), claude_client=claude)


def test_from_config_builds_the_collaborators(tmp_path: Path) -> None:
    """``DecisionService.from_config``: the configured planner, OLS replayed from the configured
    fixtures (never EBI in ``replay``), the policy file and profile, the store from the DSN."""
    from mesa_clm.ols import RecordingOLS
    from tests.fakes.pipeline import OLS_DIR, config, fake_provider

    cfg = config(
        OLS__FIXTURES_DIR=str(OLS_DIR),
        PROVENANCE__DSN=f"duckdb:///{tmp_path / 'prov.duckdb'}",
        POLICY__MAX_WAIT_S="0.5",
    )
    svc = DecisionService.from_config(cfg, fake_provider(), bench_cards=())
    assert svc.planner.name == "static" and svc.policy.profile.name == "dev"
    assert isinstance(svc.ols.client, RecordingOLS) and svc.ols.client.mode == "replay"
    assert svc.max_wait_s == 0.5
    run = svc.annotate(card(SERVICE_CARD), "agent-x", owner="alice")
    assert svc.check_owner(run.run_id, "alice")["status"] == "decided"
    svc.close()
