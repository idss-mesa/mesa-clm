"""Where a pick comes from decides what it may teach (DESIGN D1, D21, D30; defect (g)).

* ``via='tool'`` (a plain ``mesa_clm_feedback`` call): ``agent_pick`` rows at weight 0,
  ``fold_eligible=false``; a link it accepts is ``accepted_by='agent'``.
* ``via='cli'`` / ``'elicitation'``: the pick as ``curator`` 1.0, every other offered candidate
  as ``curator_implicit`` 0.7 (a negative); ``bench_card`` on a bench card.
* An explicit "none of these" (no option, the anchor, or ``reject``): an anchor-positive row plus
  a curator negative per candidate; the group is ``rejected`` and its links stay ``proposed``.
* ``decline``: no label, no link or group change; only the override row.

Every row carries the D1 identity ``(task_key, target_sha256, option_key, label_source)`` of the
group decision's *own* task (mesa-anyjev hard-coded ``term.fits``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from mesa_clm.identity import target_sha256
from mesa_clm.learn.labels import product_code_of
from mesa_clm.pipeline import AnnotationRun
from mesa_clm.provenance.labels import LabelRow
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.service import DecisionService
from mesa_clm.states import state_sha256
from mesa_clm.tasks import TASKS
from tests.fakes.pipeline import SERVICE_CARD, annotated_template, copy_store, service


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, AnnotationRun]:
    return annotated_template(tmp_path_factory.mktemp("template"))


@pytest.fixture
def store_file(template: tuple[Path, AnnotationRun], tmp_path: Path) -> Any:
    return copy_store(template[0], tmp_path)


@pytest.fixture
def svc(store_file: Any) -> DecisionService:
    return service(store_file, bench_cards=())


@pytest.fixture
def run_id(template: tuple[Path, AnnotationRun]) -> UUID:
    return template[1].run_id


def _group(svc: DecisionService, run_id: UUID, task_id: str = "term.fits") -> UUID:
    """A group of ``task_id`` with at least three candidates, proposed for ``term.fits`` (an
    ontology group of any outcome: Q3 keeps its top two whatever the outcome, D28)."""
    for g in svc.store.groups(run_id):
        proposed = g["outcome"] == "proposed" or task_id != "term.fits"
        if g["task_id"] == task_id and proposed and g["n_candidates"] >= 3:
            return UUID(str(g["group_id"]))
    raise AssertionError(f"no {task_id} group with three candidates")


def _labels(svc: DecisionService, task_id: str = "term.fits") -> list[dict[str, Any]]:
    return svc.store.labels_for(task_id)


def _decision(svc: DecisionService, run_id: UUID, gid: UUID) -> dict[str, Any]:
    g = svc.store.group(gid)
    assert g is not None
    [d] = [
        d
        for d in svc.store.decisions(run_id)
        if str(d["decision_id"]) == str(g["winner_decision_id"])
    ]
    return d


def _check_identity(rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    task_id = decision["task_id"]
    for r in rows:
        assert r["task_id"] == task_id and r["task_key"] == TASKS[task_id].key
        assert r["target_sha256"] == decision["target_sha256"]
        assert r["target_sha256"] == target_sha256(task_id, r["state_json"])
        assert r["state_sha256"] == decision["state_sha256"] == state_sha256(r["state_json"])
        assert r["card"] == SERVICE_CARD and r["product_code"] == product_code_of(SERVICE_CARD)
        assert r["leak_group"] == r["product_code"] and r["actor"] == "carol"
        assert r["label"] == TASKS[task_id].options[r["label_index"]]


def test_a_tool_pick_is_an_agent_pick_at_weight_zero(svc: DecisionService, run_id: UUID) -> None:
    gid = _group(svc, run_id)
    cands = svc.candidates_for_group(gid)
    pick = cands[1] if not cands[1]["is_anchor"] else cands[2]
    out = svc.record_human_pick(
        gid, "carol", via="tool", owner="alice", option_key=pick["option_key"]
    )
    assert out["label_source"] == "agent_pick" and out["outcome"] == "human"
    rows = _labels(svc)
    n_cands = sum(not c["is_anchor"] for c in cands)
    assert len(rows) == out["labels_written"] == n_cands
    for r in rows:
        assert r["label_source"] == "agent_pick" and r["weight"] == 0.0
        assert r["fold_eligible"] is False and r["bench_card"] is False
        assert r["label"] == ("Yes" if r["option_key"] == pick["option_key"] else "No")
    _check_identity(rows, _decision(svc, run_id, gid))
    [ov] = svc.store.overrides(run_id)
    assert (ov["via"], ov["action"], ov["label_source"]) == ("tool", "pick", "agent_pick")
    assert ov["chosen_option_key"] == pick["option_key"] and ov["labels_written"] == n_cands
    assert len(ov["offered"]) == len(cands)
    [link] = [k for k in svc.store.links(run_id) if str(k["link_id"]) == out["accepted_link_id"]]
    assert (link["write_status"], link["accepted_by"]) == ("accepted", "agent")
    assert link["source"] == "mesa-clm:feedback:tool"


@pytest.mark.parametrize("via", ["cli", "elicitation"])
def test_a_curator_pick_and_its_implicit_negatives(
    svc: DecisionService, run_id: UUID, via: str
) -> None:
    gid = _group(svc, run_id)
    cands = svc.candidates_for_group(gid)
    pick = cands[0]
    out = svc.record_human_pick(
        gid,
        "carol",
        via=via,  # type: ignore[arg-type]
        owner="alice",
        option_key=pick["option_key"],
        elicitation_key=f"clm_term_choice:{gid}" if via == "elicitation" else None,
    )
    assert out["label_source"] == "curator" and out["outcome"] == "human"
    rows = {r["option_key"]: r for r in _labels(svc)}
    assert ANCHOR_KEY not in rows
    assert set(rows) == {c["option_key"] for c in cands if not c["is_anchor"]}
    chosen = rows.pop(pick["option_key"])
    assert (chosen["label_source"], chosen["weight"], chosen["label"]) == ("curator", 1.0, "Yes")
    for r in rows.values():
        assert (r["label_source"], r["weight"], r["label"]) == ("curator_implicit", 0.7, "No")
    assert all(r["fold_eligible"] and not r["bench_card"] for r in _labels(svc))
    _check_identity(_labels(svc), _decision(svc, run_id, gid))
    [ov] = svc.store.overrides(run_id)
    assert ov["via"] == via and ov["label_source"] == "curator"
    assert ov["elicitation_key"] == (f"clm_term_choice:{gid}" if via == "elicitation" else None)
    [link] = [k for k in svc.store.links(run_id) if str(k["link_id"]) == out["accepted_link_id"]]
    assert link["accepted_by"] == "human"


@pytest.mark.parametrize(
    ("action", "option_key", "stored"),
    [("pick", None, "none"), ("pick", ANCHOR_KEY, "none"), ("reject", None, "reject")],
)
def test_an_explicit_none_is_an_anchor_positive_plus_negatives(
    svc: DecisionService, run_id: UUID, action: str, option_key: str | None, stored: str
) -> None:
    gid = _group(svc, run_id)
    cands = svc.candidates_for_group(gid)
    links_before = {str(k["link_id"]): k["write_status"] for k in svc.store.links(run_id)}
    out = svc.record_human_pick(
        gid,
        "carol",
        via="cli",
        owner="alice",
        option_key=option_key,
        action=action,  # type: ignore[arg-type]
    )
    assert out["outcome"] == "rejected" and out["accepted_link_id"] is None
    rows = {r["option_key"]: r for r in _labels(svc)}
    anchor = rows.pop(ANCHOR_KEY)
    assert (anchor["label_source"], anchor["weight"], anchor["label"]) == ("curator", 1.0, "Yes")
    assert set(rows) == {c["option_key"] for c in cands if not c["is_anchor"]}
    for r in rows.values():
        assert (r["label_source"], r["weight"], r["label"]) == ("curator", 1.0, "No")
    _check_identity(_labels(svc), _decision(svc, run_id, gid))
    assert svc.store.group(gid)["outcome"] == "rejected"  # type: ignore[index]
    # No write status says "rejected": the links keep theirs, the group and override say it.
    assert {str(k["link_id"]): k["write_status"] for k in svc.store.links(run_id)} == links_before
    [ov] = svc.store.overrides(run_id)
    assert ov["action"] == stored and ov["chosen_option_key"] == ANCHOR_KEY
    assert str(gid) not in svc.run_summary(run_id)["pending_groups"]


def test_a_tool_none_is_still_only_an_agent_pick(svc: DecisionService, run_id: UUID) -> None:
    gid = _group(svc, run_id)
    out = svc.record_human_pick(gid, "carol", via="tool", owner="alice", option_key=None)
    rows = _labels(svc)
    assert out["label_source"] == "agent_pick" and rows
    assert all(r["label_source"] == "agent_pick" and r["weight"] == 0.0 for r in rows)
    assert any(r["option_key"] == ANCHOR_KEY and r["label"] == "Yes" for r in rows)


def test_decline_writes_nothing_but_its_override(svc: DecisionService, run_id: UUID) -> None:
    gid = _group(svc, run_id)
    links = svc.store.links(run_id)
    group = svc.store.group(gid)
    for via in ("cli", "elicitation", "tool"):
        out = svc.record_human_pick(gid, "carol", via=via, owner="alice", action="decline")  # type: ignore[arg-type]
        assert out["outcome"] == "declined" and out["labels_written"] == 0
        assert out["label_source"] is None and out["accepted_link_id"] is None
    assert all(_labels(svc, t) == [] for t in ("term.fits", "column.ontology_fits"))
    assert svc.store.links(run_id) == links and svc.store.group(gid) == group
    overrides = svc.store.overrides(run_id)
    assert [o["action"] for o in overrides] == ["decline"] * 3
    assert all(o["labels_written"] == 0 and o["label_source"] is None for o in overrides)
    assert str(gid) in svc.run_summary(run_id)["pending_groups"]


def test_a_pick_labels_the_groups_own_task(svc: DecisionService, run_id: UUID) -> None:
    """A pick on a ``column.ontology_fits`` group labels that task (option_key = registry id),
    not ``term.fits``, and builds no AVU link."""
    gid = _group(svc, run_id, "column.ontology_fits")
    cands = svc.candidates_for_group(gid)
    pick = next(c for c in cands if not c["is_anchor"])
    out = svc.record_human_pick(
        gid, "carol", via="cli", owner="alice", option_key=pick["option_key"]
    )
    assert out["accepted_link_id"] is None and out["outcome"] == "human"
    assert _labels(svc, "term.fits") == []
    rows = _labels(svc, "column.ontology_fits")
    assert {r["option_key"] for r in rows} == {c["option_key"] for c in cands if not c["is_anchor"]}
    _check_identity(rows, _decision(svc, run_id, gid))


def test_curator_rows_on_a_bench_card_are_tagged(store_file: Any, run_id: UUID) -> None:
    svc = service(store_file)  # bench cards from the store's silver labels (D30)
    gid = _group(svc, run_id)
    d = _decision(svc, run_id, gid)
    assert not svc.is_bench_card(SERVICE_CARD)
    svc.store.insert_labels(
        [
            LabelRow(
                task_id="term.fits",
                task_key=TASKS["term.fits"].key,
                target_sha256=d["target_sha256"],
                option_key="NCBITaxon:8782",
                label_source="consensus_all",
                label="Yes",
                label_index=0,
                weight=0.8,
                state_sha256=d["state_sha256"],
                state_json=d["state_json"],
                card=SERVICE_CARD,
            )
        ]
    )
    assert svc.is_bench_card(SERVICE_CARD)
    svc.record_human_pick(gid, "carol", via="cli", owner="alice", option_key=None)
    mine = [r for r in _labels(svc) if r["label_source"] == "curator"]
    assert mine and all(r["bench_card"] for r in mine)


def test_bad_feedback_is_refused_before_anything_is_written(
    svc: DecisionService, run_id: UUID
) -> None:
    gid = _group(svc, run_id)
    with pytest.raises(ValueError, match="not among the offered"):
        svc.record_human_pick(gid, "carol", via="cli", owner="alice", option_key="PATO:0000001x")
    with pytest.raises(ValueError, match="via"):
        svc.record_human_pick(gid, "carol", via="email", owner="alice")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action"):
        svc.record_human_pick(gid, "carol", via="cli", owner="alice", action="accept")  # type: ignore[arg-type]
    assert svc.store.overrides(run_id) == [] and _labels(svc) == []


def test_a_repeated_pick_writes_no_new_labels(svc: DecisionService, run_id: UUID) -> None:
    gid = _group(svc, run_id)
    key = svc.candidates_for_group(gid)[0]["option_key"]
    first = svc.record_human_pick(gid, "carol", via="cli", owner="alice", option_key=key)
    again = svc.record_human_pick(gid, "carol", via="cli", owner="alice", option_key=key)
    assert first["labels_written"] > 0 and again["labels_written"] == 0
    assert first["accepted_link_id"] == again["accepted_link_id"]
    assert len(svc.store.overrides(run_id)) == 2
