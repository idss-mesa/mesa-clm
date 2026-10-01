"""Runs carry an owner, and explain, feedback, apply and MRTR resume check it (DESIGN D21; plan
§7.1): another identity gets :class:`~mesa_clm.service.NotOwner` and nothing is written or
shown. ``owner`` is the authenticated identity of the request, never the free-text ``actor``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mesa_clm.pipeline import AnnotationRun
from mesa_clm.service import DecisionService, NotOwner
from tests.fakes.pipeline import annotated_template, copy_store, service


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, AnnotationRun]:
    return annotated_template(tmp_path_factory.mktemp("template"))


@pytest.fixture
def svc(template: tuple[Path, AnnotationRun], tmp_path: Path) -> DecisionService:
    return service(copy_store(template[0], tmp_path), bench_cards=())


@pytest.fixture
def run_id(template: tuple[Path, AnnotationRun]) -> UUID:
    return template[1].run_id


def _a_group(svc: DecisionService, run_id: UUID) -> UUID:
    return UUID(svc.run_summary(run_id)["pending_groups"][0])


def test_the_run_stores_the_owner_not_the_actor(svc: DecisionService, run_id: UUID) -> None:
    row = svc.check_owner(run_id, "alice")
    assert row["owner"] == "alice" and "agent-x" not in row.values()


def test_check_owner(svc: DecisionService, run_id: UUID) -> None:
    for other in ("bob", "", "Alice"):
        with pytest.raises(NotOwner) as exc:
            svc.check_owner(run_id, other)
        assert "alice" not in str(exc.value)
    with pytest.raises(KeyError):
        svc.check_owner(uuid4(), "alice")


def test_explain_refuses_another_owner(svc: DecisionService, run_id: UUID) -> None:
    with pytest.raises(NotOwner):
        svc.explain(owner="bob", run_id=run_id)
    with pytest.raises(NotOwner):
        svc.run_summary(run_id, owner="bob")
    assert svc.explain(owner="alice", run_id=run_id)["run"]["owner"] == "alice"


def test_explain_by_path_shows_only_the_owners_runs(svc: DecisionService, run_id: UUID) -> None:
    path = "/iplant/home/alice/data/brd_countdata.csv"
    [link, *_] = svc.store.links(run_id)
    svc.store.set_link_status([UUID(str(link["link_id"]))], "proposed", irods_path=path)
    mine = svc.explain(owner="alice", irods_path=path)
    assert [r["run"]["run_id"] for r in mine["runs"]] == [str(run_id)]
    assert svc.explain(owner="bob", irods_path=path) == {"irods_path": path, "runs": []}


def test_feedback_refuses_another_owner_and_writes_nothing(
    svc: DecisionService, run_id: UUID
) -> None:
    gid = _a_group(svc, run_id)
    before_links = svc.store.links(run_id)
    before_group = svc.store.group(gid)
    for via in ("tool", "cli", "elicitation"):
        with pytest.raises(NotOwner):
            svc.record_human_pick(gid, "mallory", via=via, owner="mallory", option_key=None)  # type: ignore[arg-type]
    assert svc.store.overrides(run_id) == []
    assert svc.store.labels_for("term.fits") == []
    assert svc.store.links(run_id) == before_links
    assert svc.store.group(gid) == before_group


def test_feedback_on_an_unknown_group(svc: DecisionService) -> None:
    with pytest.raises(KeyError):
        svc.record_human_pick(uuid4(), "alice", via="cli", owner="alice")
    with pytest.raises(KeyError):
        svc.candidates_for_group(uuid4())
