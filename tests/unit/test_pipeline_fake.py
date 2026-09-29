"""The whole decide phase on every fixture card, hermetically (plan §4.2, §4.6, §8 M1-B).

``FakeProvider`` answers the CLM questions (two fake heads, so both a replaced D24 child and the
keep rule's dedup occur), ``RecordingOLS`` replays ``tests/fixtures/ols`` (the fixture closure
covers every search the pipeline can make, so a ``ReplayMiss`` here is a closure bug, never
something to record live), the shipped policy has every ``auto: null`` and the run is committed
to a DuckDB sidecar, whose CHECK constraints every row must pass. Asserted: every proposal has a
decision chain back to Q1; nothing autos; anchor-won groups abstain and stay pending; the keep
rule dedups exact triples and caps by ``p_fit``; the value kind is decided for the proposal's own
column (defect (b)); every row carries its identity and the D5 fingerprints; contexts come from
the state builders only; the degraded ``ols_rank`` path proposes and never autos (D28).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from mesa_clm import __version__
from mesa_clm import framings as fr
from mesa_clm.avu import VALUE_KIND_COLUMN, build_avu, triple
from mesa_clm.cards import is_identifier, load_card
from mesa_clm.config import config_sha256
from mesa_clm.identity import target_sha256
from mesa_clm.ols import OLSLayer, RecordingOLS, ReplayMiss
from mesa_clm.pipeline import MAX_CHILDREN, AnnotationRun, Annotator
from mesa_clm.planner.static_planner import StaticPlanner
from mesa_clm.policy import Policy
from mesa_clm.provenance.models import DecisionRow
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers import FakeProvider
from mesa_clm.registry import ANCHOR_KEY, allowed_for_aspect
from mesa_clm.states import card_header, state_sha256
from mesa_clm.tasks import RANK_FIT_TASKS, TASKS
from tests.fakes.pipeline import (
    CARD_PATHS,
    FAKE_MODEL,
    FAKE_SEED,
    RAW_MODEL,
    RAW_SEED,
    SERVICE_CARD,
    AspectPlanner,
    HintPlanner,
    annotator,
    card,
    config,
    down_provider,
    fake_provider,
    service,
)

HEADS = [(FAKE_MODEL, FAKE_SEED), (RAW_MODEL, RAW_SEED)]
# The state views a decision may carry (D23): column_state, target_state (column, site,
# dataset) and value_kind_state; never a raw card.
VIEWS = {
    frozenset({"card", "column"}),
    frozenset({"card", "scope", "aspect", "column"}),
    frozenset({"card", "scope", "aspect", "site"}),
    frozenset({"card", "scope", "aspect"}),
    frozenset({"card", "column", "aspect", "term"}),
}


@dataclass
class Batch:
    """Every fixture card annotated once under one fake head, committed to one store."""

    model: str
    seed: int
    store: DuckDBStore
    provider: FakeProvider
    runs: dict[str, AnnotationRun]

    def rows(self, run: AnnotationRun) -> Rows:
        return Rows(self.store, run.run_id)


class Rows:
    """One run's sidecar rows, keyed for the assertions."""

    def __init__(self, store: DuckDBStore, run_id: UUID) -> None:
        self.run = store.run(run_id) or {}
        self.decisions = {str(d["decision_id"]): d for d in store.decisions(run_id)}
        self.groups = {str(g["group_id"]): g for g in store.groups(run_id)}
        self.links = {str(link["link_id"]): link for link in store.links(run_id)}
        self.options: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for o in store.options(run_id):
            self.options[str(o["decision_id"])].append(o)
        self.calls = store.clm_calls(run_id)

    def option_keys(self, decision_id: Any) -> list[str]:
        opts = sorted(self.options[str(decision_id)], key=lambda o: o["option_index"])
        return [str(o["option_key"]) for o in opts]

    def p_fit(self, decision_id: Any, key: str) -> float:
        [opt] = [o for o in self.options[str(decision_id)] if o["option_key"] == key]
        return float(opt["p_fit"])

    def of_column(self, name: str, task_id: str) -> list[dict[str, Any]]:
        return [
            d
            for d in self.decisions.values()
            if d["column_name"] == name and d["task_id"] == task_id
        ]


@pytest.fixture(scope="module", params=HEADS, ids=[f"{m}-seed{s}" for m, s in HEADS])
def batch(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Batch:
    model, seed = request.param
    store = DuckDBStore(tmp_path_factory.mktemp("prov") / "prov.duckdb")
    provider = fake_provider(seed, model)
    ann = annotator(store, provider=provider)
    runs = {p.stem: ann.annotate(load_card(p)) for p in CARD_PATHS}
    return Batch(model, seed, store, provider, runs)


# -- the run ----------------------------------------------------------------------------------


def test_every_card_commits_a_decided_run(batch: Batch) -> None:
    assert set(batch.runs) == {p.stem for p in CARD_PATHS}
    cfg = config()
    for name, run in batch.runs.items():
        rows = batch.rows(run)
        r = rows.run
        assert r["status"] == "decided" and r["owner"] == "alice" and r["card_name"] == name
        assert (
            r["card_sha256"]
            == run.card.sha256
            == load_card(CARD_PATHS[0].parent / f"{name}.md").sha256
        )
        assert r["framings_lock_sha"] == fr.lock_sha() == run.fingerprint["framings_lock_sha"]
        assert r["config_sha256"] == config_sha256(cfg)
        assert r["mesa_clm_version"] == __version__ and r["policy_profile"] == "dev"
        assert r["planner"] == "static" and r["plan_json"]["ontologies"]
        assert r["provider"] == "fake" and r["clm_model"] == batch.model and r["tier"] == "auto"
        assert r["encoder_model"] == "fake-ngram" and len(r["clm_commit"]) == 40
        for key, value in batch.provider.fingerprint.as_dict().items():
            assert r[key] == value == run.fingerprint[key]
        assert len(rows.decisions) == run.n_decisions == r["n_decisions"] > 0
        assert len(rows.calls) == run.n_calls == r["n_clm_calls"] > 0
        assert all(c["status"] == "ok" and c["model"] == batch.model for c in rows.calls)
        assert r["n_encoder_tokens"] == run.input_tokens >= 0
        assert r["degraded"] is False and run.degraded is False
        assert sum(run.outcomes.values()) == run.n_decisions
        json.dumps(run.to_dict())  # the tool output is JSON as it stands


def test_rows_read_back_satisfy_the_row_models(batch: Batch) -> None:
    """The commit passed the DuckDB CHECKs; what comes back also re-validates as a row."""
    for run in batch.runs.values():
        rows = batch.rows(run)
        for d in rows.decisions.values():
            DecisionRow.model_validate(d)
            if d["method"] == "ols_rank":  # pragma: no cover - no ols_rank in a healthy run
                assert d["rank"] is not None and d["outcome"] in ("proposed", "abstain")


def test_every_proposal_has_a_decision_chain(batch: Batch) -> None:
    n_column = 0
    for run in batch.runs.values():
        rows = batch.rows(run)
        assert len(rows.links) == len(run.proposals)
        for p in run.proposals:
            link = rows.links[str(p.link_id)]
            assert link["write_status"] == "proposed" and link["accepted_by"] is None
            assert (link["attribute"], link["value"], link["unit"]) == triple(p.avu)
            assert link["term_curie"] == p.candidate.curie and link["value_kind"] == p.value_kind
            assert link["column_name"] == p.column_name and link["site_code"] == p.site_code
            d = rows.decisions[str(link["decision_id"])]
            g = rows.groups[str(link["group_id"])]
            assert str(d["group_id"]) == str(g["group_id"]) == str(p.group_id)
            assert d["task_id"] == g["task_id"] == "term.fits"
            assert str(g["winner_decision_id"]) == str(d["decision_id"])
            assert p.candidate.curie in rows.option_keys(d["decision_id"])
            assert g["outcome"] in ("proposed", "escalated") and d["outcome"] == "proposed"
            if g["escalated_from"]:  # a D24 child: its rank hangs under the parent group's
                parent = rows.groups[str(g["escalated_from"])]
                assert str(d["parent_decision_id"]) == str(parent["winner_decision_id"])
            if p.value_kind_decision_id is not None:
                vk = rows.decisions[str(p.value_kind_decision_id)]
                assert str(vk["parent_decision_id"]) == str(d["decision_id"])
            if p.scope != "column":
                continue
            n_column += 1
            [q1] = rows.of_column(p.column_name, "column.annotate")
            assert q1["answer"] == "Yes" and q1["method"] == "fake"
            [q2] = rows.of_column(p.column_name, "column.aspect")
            assert q2["outcome"] != "rejected"
            q3 = rows.of_column(p.column_name, "column.ontology_fits")
            if p.ontology_id == "uo":
                assert any(d["method"] == "rule" and d["answer"] == "uo" for d in q3)
            else:
                kept = {
                    o
                    for d in q3
                    if d["group_id"]
                    for o in rows.groups[str(d["group_id"])]["search_json"]["kept"]
                }
                assert p.ontology_id in kept
    assert n_column > 0


def test_nothing_autos_while_every_threshold_is_null(batch: Batch) -> None:
    for run in batch.runs.values():
        rows = batch.rows(run)
        assert all(d["outcome"] != "auto" for d in rows.decisions.values())
        assert all(d["threshold_auto"] is None for d in rows.decisions.values())
        assert all(g["outcome"] != "auto" for g in rows.groups.values())
        assert all(link["write_status"] == "proposed" for link in rows.links.values())
        assert all(p.outcome == "proposed" for p in run.proposals)


def test_every_row_carries_its_identity_and_fingerprint(batch: Batch) -> None:
    fp = batch.provider.fingerprint
    for run in batch.runs.values():
        rows = batch.rows(run)
        for d in rows.decisions.values():
            assert (d["encoder_fp"], d["clm_model_fp"], d["schema_sha256"]) == (
                fp.encoder_fp,
                fp.clm_model_fp,
                fp.schema_sha256,
            )
            assert d["task_key"] == TASKS[d["task_id"]].key
            framing = fr.by_question_key(d["question_key"])
            assert (framing.task_id, framing.id) == (d["task_id"], d["framing_id"])
            assert framing is fr.active_framing(d["task_id"])
            assert d["state_sha256"] == state_sha256(d["state_json"])
            assert d["target_sha256"] == target_sha256(d["task_id"], d["state_json"])
            assert len(d["context_sha256"]) == 64 and d["k"] == len(d["options"])
            assert rows.option_keys(d["decision_id"]) == d["options"]
            if d["task_id"] in RANK_FIT_TASKS:
                assert d["anchor_index"] == d["options"].index(ANCHOR_KEY)
            else:
                assert d["anchor_index"] is None
            if d["method"] == "fake":
                assert d["level"] == "zero_shot" and d["calibration"] == "uncalibrated"
                assert d["probs"] is not None and d["raw_probs"] is not None
            else:
                assert d["method"] == "rule" and d["probs"] is None and d["level"] == "none"
        for g in rows.groups.values():
            assert g["task_key"] == TASKS[g["task_id"]].key


def test_contexts_come_from_the_state_builders_only(batch: Batch) -> None:
    for run in batch.runs.values():
        raw = (CARD_PATHS[0].parent / f"{run.card.name}.md").read_text(encoding="utf-8")
        header = card_header(run.card)
        for d in batch.rows(run).decisions.values():
            state = d["state_json"]
            assert frozenset(state) in VIEWS
            assert state["card"] == header
            text = json.dumps(state, ensure_ascii=False)
            assert "## Columns" not in text and raw not in text


def test_identifiers_get_rule_rows_and_nothing_else(batch: Batch) -> None:
    for run in batch.runs.values():
        rows = batch.rows(run)
        for col in run.card.columns:
            if not is_identifier(col):
                continue
            mine = [d for d in rows.decisions.values() if d["column_name"] == col.name]
            assert len(mine) == 1
            [d] = mine
            assert (d["task_id"], d["method"], d["answer"], d["outcome"], d["reason"]) == (
                "column.annotate",
                "rule",
                "No",
                "rule",
                "is_identifier",
            )
            assert d["context_tokens"] == 0


def test_q1_and_q2_share_one_request_per_column(batch: Batch) -> None:
    for run in batch.runs.values():
        rows = batch.rows(run)
        q1 = [d for d in rows.decisions.values() if d["task_id"] == "column.annotate"]
        asked = [d for d in q1 if d["method"] == "fake"]
        q2 = [d for d in rows.decisions.values() if d["task_id"] == "column.aspect"]
        assert len(q2) == len(asked)  # every asked column gets its aspect in the same call
        # The requests that carried Q1 and Q2 together have two questions.
        assert sum(1 for c in rows.calls if c["n_questions"] >= 2) >= len(asked)
        for d in q2:
            if d["outcome"] == "rejected":
                assert d["reason"] == "column_not_annotated"
                [a] = rows.of_column(d["column_name"], "column.annotate")
                assert a["answer"] == "No"


def test_q3_ranks_the_registry_masked_by_aspect_and_keeps_two(batch: Batch) -> None:
    n = 0
    for run in batch.runs.values():
        rows = batch.rows(run)
        for d in rows.decisions.values():
            if d["task_id"] != "column.ontology_fits":
                continue
            aspect = d["state_json"]["aspect"]
            if d["method"] == "rule":
                assert (aspect, d["answer"], d["reason"]) == ("unit", "uo", "unit_aspect")
                continue
            n += 1
            allowed = allowed_for_aspect(aspect)
            opts = sorted(rows.options[str(d["decision_id"])], key=lambda o: o["option_index"])
            assert [o["option_key"] for o in opts[:-1]] == [e.id for e in fr.ONTOLOGY_REGISTRY]
            for o in opts:
                expected = o["option_key"] not in allowed and o["option_key"] != ANCHOR_KEY
                assert o["masked"] is expected
                assert (o["prob"] == 0.0) if expected else (o["prob"] > 0.0)
            group = rows.groups[str(d["group_id"])]
            kept = group["search_json"]["kept"]
            in_play = sorted(
                (o for o in opts if not o["masked"] and o["option_key"] != ANCHOR_KEY),
                key=lambda o: -o["p_fit"],
            )
            assert kept == [o["option_key"] for o in in_play[:2]]  # whatever the outcome (D28)
            assert group["anchor_won"] == (d["answer"] == ANCHOR_KEY)
    assert n > 0


def test_anchor_won_groups_abstain_and_stay_pending(batch: Batch) -> None:
    svc = service(batch.store, provider=batch.provider)
    n_term = 0
    for run in batch.runs.values():
        rows = batch.rows(run)
        pending = set(svc.run_summary(run.run_id)["pending_groups"])
        abstained = {a["group_id"]: a["reason"] for a in run.abstained}
        for gid, g in rows.groups.items():
            if not g["anchor_won"]:
                continue
            d = rows.decisions[str(g["winner_decision_id"])]
            assert d["answer"] == ANCHOR_KEY and d["outcome"] == "abstain"
            assert d["reason"] == "anchor_won" and g["outcome"] == "abstain"
            assert not [link for link in rows.links.values() if str(link["group_id"]) == gid]
            if g["task_id"] == "term.fits":
                n_term += 1
                assert gid in pending and abstained[gid] == "anchor_won"
    assert n_term > 0


def test_the_keep_rule_dedups_triples_and_caps(batch: Batch) -> None:
    dropped = 0
    for run in batch.runs.values():
        rows = batch.rows(run)
        triples = [(k["attribute"], k["value"], k["unit"]) for k in rows.links.values()]
        assert len(triples) == len(set(triples)) <= 25
        for a in run.abstained:
            if a["reason"] in ("duplicate", "over_cap"):
                dropped += 1
                assert rows.groups[a["group_id"]]["outcome"] == "rejected"
        p_fits = [p.p_fit for p in run.proposals]
        assert p_fits == sorted(p_fits, key=lambda p: -(p or 0.0))  # kept best first
    assert dropped > 0, "both fake heads produce at least one duplicate triple"


def test_the_cap_keeps_the_best_by_p_fit(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, cfg=config(POLICY__MAX_AVUS="2")).annotate(
        card("DP1.10022.001.bet_expertTaxonomistIDProcessed")
    )
    assert len(run.proposals) == 2 == len(store.links(run.run_id))
    over = [a for a in run.abstained if a["reason"] == "over_cap"]
    assert over
    groups = {str(g["group_id"]): g for g in store.groups(run.run_id)}
    worst_kept = min(p.p_fit or 0.0 for p in run.proposals)
    for a in over:
        assert groups[a["group_id"]]["outcome"] == "rejected"
        assert (groups[a["group_id"]]["top_p_fit"] or 0.0) <= worst_kept + 1e-12


def test_value_kind_is_decided_for_the_proposals_own_column(batch: Batch) -> None:
    """Defect (b): mesa-anyjev recorded every value-kind decision under the last column of an
    earlier loop. Each one must name, and be asked over, its own proposal's column."""
    several = False
    n = 0
    for run in batch.runs.values():
        rows = batch.rows(run)
        vk = [d for d in rows.decisions.values() if d["task_id"] == "avu.value_kind"]
        for d in vk:
            n += 1
            assert d["scope"] == "avu"
            assert d["column_name"] == d["state_json"]["column"]["name"]
            parent = rows.decisions[str(d["parent_decision_id"])]
            assert parent["column_name"] == d["column_name"]
            assert rows.groups[str(d["group_id"])]["column_name"] == d["column_name"]
            assert d["state_json"]["term"]["curie"] in rows.option_keys(parent["decision_id"])
        several |= len({d["column_name"] for d in vk}) > 1
        for p in run.proposals:
            if p.value_kind == VALUE_KIND_COLUMN:
                assert p.avu["value"] == p.column_name
    assert n > 0 and several


def test_specificity_ranks_parent_and_children_once_and_only_proposes(batch: Batch) -> None:
    replaced = 0
    n = 0
    for run in batch.runs.values():
        rows = batch.rows(run)
        for gid, g in rows.groups.items():
            if not g["escalated_from"]:
                continue
            n += 1
            parent_group = rows.groups[str(g["escalated_from"])]
            parent = rows.decisions[str(parent_group["winner_decision_id"])]
            d = rows.decisions[str(g["winner_decision_id"])]
            assert str(d["parent_decision_id"]) == str(parent["decision_id"])
            keys = rows.option_keys(d["decision_id"])
            assert keys[0] == parent["answer"] and keys[-1] == ANCHOR_KEY
            assert 3 <= len(keys) <= MAX_CHILDREN + 2
            assert g["search_json"]["specificity_of"] == parent["answer"]
            assert d["outcome"] != "auto" and g["outcome"] in ("proposed", "rejected")
            # s_c is set-independent (D2): the parent scores the same in both ranks.
            assert rows.p_fit(d["decision_id"], keys[0]) == pytest.approx(
                rows.p_fit(parent["decision_id"], keys[0]), abs=1e-9
            )
            links = [k for k in rows.links.values() if str(k["group_id"]) == gid]
            if g["outcome"] == "proposed":
                replaced += 1
                [link] = links
                child = link["term_curie"]
                assert child != keys[0] and d["answer"] == child
                assert (
                    rows.p_fit(d["decision_id"], child)
                    >= rows.p_fit(d["decision_id"], keys[0]) + 0.10 - 1e-12
                )
            else:
                assert not links
    assert n > 0
    if batch.model == RAW_MODEL:
        assert replaced > 0, "the raw head refines at least one winner to a child (D24)"


def test_eval_result_has_the_neon_avu_eval_shape(batch: Batch) -> None:
    for name, run in batch.runs.items():
        res = run.to_eval_result()
        assert res["card"] == name and res["parse_ok"] is True and res["family"] == "mesa-clm"
        assert res["run_id"] == str(run.run_id) and res["n_prompts"] == run.n_calls
        assert len(res["avus"]) == len(run.proposals)
        for avu, p in zip(res["avus"], run.proposals, strict=True):
            assert set(avu) >= {"attribute", "value", "unit", "ontology_id", "curie", "iri"}
            assert p.avu == build_avu(p.candidate, p.avu["value"])
            assert avu["unit"] == avu["curie"] == p.candidate.curie
            assert avu["attribute"].startswith(p.candidate.ontology_id + ".")
        json.dumps(res)


def test_two_runs_on_one_card_agree() -> None:
    ann = annotator(None)
    a = ann.annotate(card(SERVICE_CARD))
    b = ann.annotate(card(SERVICE_CARD))
    assert [triple(p.avu) for p in a.proposals] == [triple(p.avu) for p in b.proposals]
    assert a.outcomes == b.outcomes and a.buffer is not None and not a.buffer.committed


def test_the_prod_profile_does_not_auto_either(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, cfg=config(POLICY__PROFILE="prod")).annotate(card(SERVICE_CARD))
    assert run.proposals and all(p.outcome == "proposed" for p in run.proposals)
    assert store.run(run.run_id)["policy_profile"] == "prod"  # type: ignore[index]


# -- degraded mode (D28) ----------------------------------------------------------------------


def _assert_ols_rank_rows(rows: Rows) -> None:
    for d in rows.decisions.values():
        assert d["outcome"] != "auto"
        if d["method"] == "ols_rank":
            assert d["probs"] is None and d["level"] == "none" and d["calibration"] == "none"
            assert d["rank"] is not None and d["outcome"] in ("proposed", "abstain")
            assert d["answer"] != ANCHOR_KEY
            DecisionRow.model_validate(d)


def test_clm_down_degrades_to_ols_rank_proposals(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    ann = annotator(store, provider=down_provider(), planner=AspectPlanner())
    for name in (SERVICE_CARD, "DP1.10022.001.bet_sorting"):
        run = ann.annotate(card(name))
        rows = Rows(store, run.run_id)
        assert run.degraded and rows.run["degraded"] is True and rows.run["status"] == "decided"
        assert run.proposals, "D28: a run without CLM still proposes"
        assert any(p.scope == "column" for p in run.proposals)
        for p in run.proposals:
            assert (p.method, p.level, p.outcome, p.p_fit) == ("ols_rank", "none", "proposed", None)
        assert all(link["write_status"] == "proposed" for link in rows.links.values())
        _assert_ols_rank_rows(rows)
        unavailable = [d for d in rows.decisions.values() if d["method"] == "unavailable"]
        assert unavailable
        for d in unavailable:
            assert d["outcome"] == "decider_unavailable" and d["reason"] == "decider_unavailable"
        # Every ols_rank rank of a group hangs under the CLM record that could not answer.
        for d in rows.decisions.values():
            if d["method"] == "ols_rank":
                parent = rows.decisions[str(d["parent_decision_id"])]
                assert parent["method"] == "unavailable" and parent["group_id"] == d["group_id"]
        assert rows.calls and all(c["status"] == "unavailable" for c in rows.calls)
        assert run.n_calls == len(rows.calls) and run.input_tokens == 0


def test_the_ols_rank_tier_skips_clm_for_every_rank_fit_task(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, tier="ols_rank").annotate(card(SERVICE_CARD))
    rows = Rows(store, run.run_id)
    assert rows.run["tier"] == "ols_rank" and run.degraded
    assert run.proposals and all(p.method == "ols_rank" for p in run.proposals)
    methods = defaultdict(set)
    for d in rows.decisions.values():
        methods[d["task_id"]].add(d["method"])
    assert methods["term.fits"] == {"ols_rank"}
    assert methods["column.ontology_fits"] <= {"ols_rank", "rule"}
    assert methods["column.annotate"] == {"fake", "rule"}  # choices still go to CLM
    assert not any(g["escalated_from"] for g in rows.groups.values())  # no D24 without p_fit
    _assert_ols_rank_rows(rows)


def test_ols_rank_per_task(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, ols_rank_tasks={"term.fits"}).annotate(card(SERVICE_CARD))
    rows = Rows(store, run.run_id)
    by_task = defaultdict(set)
    for d in rows.decisions.values():
        by_task[d["task_id"]].add(d["method"])
    assert by_task["term.fits"] == {"ols_rank"} and "fake" in by_task["column.ontology_fits"]


def test_bad_tiers_and_owners_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        annotator(None, tier="L2")
    with pytest.raises(ValueError, match="candidate groups"):
        annotator(None, ols_rank_tasks={"column.aspect"})
    with pytest.raises(ValueError, match="owner"):
        annotator(None, owner=" ")


def test_a_failed_run_is_committed_as_failed(tmp_path: Path) -> None:
    """An OLS replay miss (an empty fixture directory) fails the run; what was decided before it
    is committed with status ``failed`` and the error reaches the caller."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    cfg = config()
    empty = tmp_path / "no-fixtures"
    empty.mkdir()
    ann = Annotator(
        provider=fake_provider(),
        planner=StaticPlanner(),
        ols=OLSLayer(RecordingOLS(None, empty, "replay")),
        policy=Policy.from_config(cfg.policy),
        cfg=cfg,
        owner="alice",
        store=store,
    )
    with pytest.raises(ReplayMiss):
        ann.annotate(card(SERVICE_CARD))
    [r] = store.runs()
    assert r["status"] == "failed" and r["owner"] == "alice"
    assert store.decisions(UUID(str(r["run_id"])))


# -- the paths a fake head does not take on its own -----------------------------------------------


def test_q1_leaving_a_column_out_makes_its_aspect_moot(tmp_path: Path) -> None:
    """Seed 0 answers "No" to every column: the aspect asked in the same request is recorded
    ``rejected`` (``column_not_annotated``) and no column group is searched."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, provider=fake_provider(0, FAKE_MODEL)).annotate(card(SERVICE_CARD))
    rows = Rows(store, run.run_id)
    q2 = [d for d in rows.decisions.values() if d["task_id"] == "column.aspect"]
    assert q2 and all(
        (d["outcome"], d["reason"]) == ("rejected", "column_not_annotated") for d in q2
    )
    assert all(g["scope"] != "column" for g in rows.groups.values())
    assert all(p.scope in ("site", "dataset") for p in run.proposals)


def test_the_unit_aspect_searches_uo_by_rule(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    planner = HintPlanner(lambda col: {"aspect": "unit"} if col.unit else {})
    run = annotator(store, planner=planner).annotate(card(SERVICE_CARD))
    rows = Rows(store, run.run_id)
    rules = [
        d
        for d in rows.decisions.values()
        if d["task_id"] == "column.ontology_fits" and d["method"] == "rule"
    ]
    assert any(d["column_name"] == "observerDistance" for d in rules)
    for d in rules:
        assert (d["answer"], d["reason"], d["state_json"]["aspect"]) == (
            "uo",
            "unit_aspect",
            "unit",
        )
        assert d["outcome"] == "rule" and d["group_id"] is None
    [uo] = [
        g
        for g in rows.groups.values()
        if g["ontology_id"] == "uo" and g["column_name"] == "observerDistance"
    ]
    assert uo["search_json"]["table"] is True  # "meter" is a UNIT_TABLE hit: no OLS search
    assert [c["curie"] for c in uo["search_json"]["candidates"]] == ["UO:0000008"]
    for p in run.proposals:
        if p.ontology_id == "uo":
            assert p.value_kind == "the term label" and p.avu["unit"] == p.candidate.curie


def test_a_planner_can_leave_a_column_out(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    planner = HintPlanner(lambda col: {"annotate": False} if col.name == "detectionMethod" else {})
    run = annotator(store, planner=planner).annotate(card(SERVICE_CARD))
    mine = [
        d
        for d in Rows(store, run.run_id).decisions.values()
        if d["column_name"] == "detectionMethod"
    ]
    [d] = mine
    assert (d["method"], d["model"], d["answer"], d["reason"]) == (
        "rule",
        "planner",
        "No",
        "planner_annotate_false",
    )


def test_clm_down_without_any_aspect_still_proposes(tmp_path: Path) -> None:
    """No CLM and no planner aspects: every column is reported ``no_aspect``, and the sites and
    the dataset taxon still get ``ols_rank`` proposals (D28: always something)."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, provider=down_provider()).annotate(card(SERVICE_CARD))
    live = [c for c in run.card.columns if not is_identifier(c)]
    no_aspect = [a for a in run.abstained if a["reason"] == "no_aspect"]
    assert {a["column_name"] for a in no_aspect} == {c.name for c in live}
    assert all(a["group_id"] is None for a in no_aspect)
    assert run.proposals and {p.scope for p in run.proposals} <= {"site", "dataset"}
    assert all(p.method == "ols_rank" for p in run.proposals)
    _assert_ols_rank_rows(Rows(store, run.run_id))
