"""DESIGN A6: the closed choices (Q1, Q2, Q7) by deterministic rules, CLM's answers audit-only.

Hermetic: the fake provider (never evidence), OLS replayed from ``tests/fixtures/ols`` (a
``ReplayMiss`` would be a closure bug), DuckDB sidecars under ``tmp_path``. Shown here: (a) by
default no CLM answer drives Q1, Q2 or Q7 (runs whose closed-choice answers, or whose Q3 answers,
are skewed in opposite directions decide the same things); (b) the rule records and the
audit-only records are stored with their markers, and Q1 annotates every non-identifier column
whatever the planner says; (c) the hint, lookup and fallback paths choose aspects as A6 says: the
hint and the lookup's top two uncapped, the lookup M0's over the packaged frozen table (equal to
M0's per-fold lookup on the registered snapshot) with the annotated card held out, the fallback
in the order of the lookup's training prior, capped at two, never ``unit`` for a column without
a unit, and needing no CLM answer; (d) ``closed_choice: clm`` is the M2 behaviour; (e) every
fixture card annotates end to end in both modes; and explain, review and feedback never offer an
audit-only record.

The frozen-table tests read the registered snapshot's ``column.aspect`` labels: the items of M0's
``neon_aspect`` task, which M0's lookup cell already read (DESIGN A6).
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args
from uuid import UUID

import duckdb
import pytest
from pydantic import ValidationError

from mesa_clm.avu import VALUE_KIND_LABEL, VALUE_KIND_TOP
from mesa_clm.bench import registered as reg
from mesa_clm.bench.baselines import Lookup, evaluate_lookup, task_keys
from mesa_clm.bench.run import load_inputs
from mesa_clm.cards import DatasetCard, is_identifier, load_card
from mesa_clm.cli import EXIT_OK, main
from mesa_clm.clm.fake import FakeClm
from mesa_clm.clm.http import ChoiceAnswer, Question, SystemOneResponse
from mesa_clm.closed_choice import (
    ANNOTATE_REASON,
    ASPECT_MODELS,
    ASPECT_REASONS,
    AUDIT_ONLY_REASON,
    CLOSED_CHOICE_MODES,
    CLOSED_CHOICE_TASKS,
    FALLBACK_CAP,
    LOOKUP_TOP,
    TABLE_PATH,
    TABLE_SHA256,
    VALUE_KIND_REASON,
    AspectLookup,
    AspectTable,
    AspectTableError,
    candidate_aspects,
    fallback_aspects,
    freeze_table,
    is_audit_only,
    packaged_table,
    render_table,
)
from mesa_clm.config import ClosedChoice, config_sha256, load_config
from mesa_clm.pipeline import AnnotationRun
from mesa_clm.planner.base import PlanResult
from mesa_clm.planner.static_planner import StaticPlanner
from mesa_clm.provenance.models import DecisionRow
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers.tiered import FakeClmClient, FakeProvider
from mesa_clm.registry import ASPECT_OPTIONS, ASPECTS, allowed_for_aspect
from tests.fakes.pipeline import (
    CARD_PATHS,
    FAKE_MODEL,
    FAKE_SEED,
    MODES,
    OLS_DIR,
    SERVICE_CARD,
    HintPlanner,
    annotator,
    card,
    config,
    down_provider,
    service,
    shipped_config,
)

ROOT = Path(__file__).resolve().parents[2]
CARD_FILE = ROOT / "tests" / "fixtures" / "cards" / f"{SERVICE_CARD}.md"
_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")
# Up to three aspects for a column: the planner's hint plus the lookup's top two.
MAX_CHOSEN = 1 + LOOKUP_TOP


# -- helpers ------------------------------------------------------------------------------------


class Rows:
    """One run's sidecar rows."""

    def __init__(self, store: DuckDBStore, run: AnnotationRun) -> None:
        self.run = store.run(run.run_id)
        self.decisions = sorted(store.decisions(run.run_id), key=lambda d: int(d["seq"]))
        self.by_id = {str(d["decision_id"]): d for d in self.decisions}
        self.groups = {str(g["group_id"]): g for g in store.groups(run.run_id)}
        self.links = store.links(run.run_id)
        self.options: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for o in store.options(run.run_id):
            self.options[str(o["decision_id"])].append(o)

    def of(self, task_id: str, column: str | None = None) -> list[dict[str, Any]]:
        return [
            d
            for d in self.decisions
            if d["task_id"] == task_id and (column is None or d["column_name"] == column)
        ]

    def aspects(self) -> dict[str, list[tuple[str, str, str]]]:
        """Every column's aspect rule records: ``(aspect, reason, model)`` in record order."""
        out: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for d in self.of("column.aspect"):
            if d["method"] == "rule":
                aspect = ASPECTS[ASPECT_OPTIONS.index(d["answer"])]
                out[d["column_name"]].append((aspect, d["reason"], d["model"]))
        return dict(out)

    def asked_q3(self) -> list[tuple[str, str]]:
        """``(column, aspect)`` of every Q3 question put to CLM, in order."""
        return [
            (d["column_name"], d["state_json"]["aspect"])
            for d in self.of("column.ontology_fits")
            if d["method"] not in ("rule", "ols_rank")
        ]

    def a6(self, column: str) -> list[dict[str, Any]]:
        """``search_json.a6`` of the column's Q3 groups."""
        return [
            g["search_json"]["a6"]
            for g in self.groups.values()
            if g["task_id"] == "column.ontology_fits" and g["column_name"] == column
        ]


def live_columns(dataset: DatasetCard) -> list[str]:
    """What A6's Q1 annotates: every non-identifier column."""
    return [c.name for c in dataset.columns if not is_identifier(c)]


def option(aspect: str) -> str:
    return ASPECT_OPTIONS[ASPECTS.index(aspect)]


def table_of(*items: tuple[str, str, str]) -> AspectTable:
    """A test table of ``(card, column, aspect)`` items."""
    return AspectTable.from_items(items, labels_sha256="t" * 64, min_weight=0.6)


def oracle_top(table: AspectTable, held_out: str, column: str) -> list[str]:
    """The lookup's top two, recomputed from the raw items: the most rows among the other
    cards, a tie to the label of the alphabetically first card carrying it."""
    rows = sorted(
        (i for i in table.items if i.card != held_out and i.column == column),
        key=lambda i: i.card,
    )
    return [a for a, _ in Counter(i.aspect for i in rows).most_common(LOOKUP_TOP)]


def oracle_order(table: AspectTable, held_out: str) -> list[str]:
    """The fallback's order recomputed from the raw items: the other cards' labels by rows (a
    tie to the label first seen in card order), then the rest in registry order."""
    rows = sorted((i for i in table.items if i.card != held_out), key=lambda i: i.card)
    counted = [a for a, _ in Counter(i.aspect for i in rows).most_common()]
    return counted + [a for a in ASPECTS if a not in counted]


class Skewed:
    """A fake clm-serve whose answers to the questions of the tasks in ``favour`` put 0.97 on the
    option ``favour`` names (a wire key); every other question is the fake's own answer."""

    def __init__(self, favour: Mapping[str, str]) -> None:
        self.inner = FakeClmClient(FakeClm(seed=FAKE_SEED), model=FAKE_MODEL)
        self.model = FAKE_MODEL
        self.favour = dict(favour)

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        response = self.inner.system_one(state, questions, model=model, temperature=temperature)
        answers = dict(response.answers)
        for qid, answer in response.answers.items():
            key = self.favour.get(qid.split("#")[0])
            if key is None or not isinstance(answer, ChoiceAnswer):
                continue
            keys = list(answer.probabilities)
            assert key in keys, (key, keys)
            rest = 0.03 / (len(keys) - 1)
            answers[qid] = ChoiceAnswer(
                choice=key,
                confidence=0.97 - rest,
                probabilities={k: 0.97 if k == key else rest for k in keys},
            )
        return response.model_copy(update={"answers": answers})


def skewed(favour: Mapping[str, str]) -> FakeProvider:
    return FakeProvider(client=Skewed(favour), model=FAKE_MODEL, seed=FAKE_SEED)


# Closed-choice answers pulled two opposite ways (wire keys of Q1, Q2 and Q7).
SKEW_NO = {"column.annotate": "No", "column.aspect": "other", "avu.value_kind": "column_name"}
SKEW_YES = {"column.annotate": "Yes", "column.aspect": "taxon", "avu.value_kind": "top_value"}


def decided(store: DuckDBStore, run: AnnotationRun) -> dict[str, Any]:
    """Everything a run decided, without its audit-only records and without ids."""
    rows = Rows(store, run)
    seq = {str(d["decision_id"]): int(d["seq"]) for d in rows.decisions}
    winner = {gid: seq.get(str(g["winner_decision_id"])) for gid, g in rows.groups.items()}
    fields = (
        "seq", "task_id", "method", "model", "answer", "outcome", "reason", "column_name",
        "site_code", "probs", "p_fit", "rank", "state_sha256",
    )  # fmt: skip
    return {
        "decisions": [
            {k: d[k] for k in fields}
            | {
                "group": winner.get(str(d["group_id"])) if d["group_id"] else None,
                "parent": seq.get(str(d["parent_decision_id"])),
            }
            for d in rows.decisions
            if not is_audit_only(d)
        ],
        "groups": sorted(
            json.dumps(
                {k: g[k] for k in ("task_id", "aspect", "ontology_id", "outcome", "column_name")}
                | {"search": g["search_json"], "winner": winner[gid]},
                sort_keys=True,
                default=str,
            )
            for gid, g in rows.groups.items()
        ),
        "links": sorted(
            (link["attribute"], link["value"], link["unit"], link["value_kind"])
            for link in rows.links
        ),
        "proposals": [
            (p.candidate.curie, p.column_name, p.aspect, p.value_kind, p.outcome)
            for p in run.proposals
        ],
    }


# -- the configuration --------------------------------------------------------------------------


def test_the_shipped_closed_choice_is_rules() -> None:
    assert get_args(ClosedChoice) == CLOSED_CHOICE_MODES == MODES == ("rules", "clm")
    assert load_config(env={}).decider.closed_choice == "rules"
    clm = load_config(env={"MESA_CLM_DECIDER__CLOSED_CHOICE": "clm"})
    assert clm.decider.closed_choice == "clm"
    assert config_sha256(clm) != config_sha256(load_config(env={}))
    with pytest.raises(ValidationError):
        load_config(env={"MESA_CLM_DECIDER__CLOSED_CHOICE": "llm"})
    with pytest.raises(ValueError, match="closed_choice"):
        annotator(None, closed_choice="llm")
    assert annotator(None).closed_choice == "rules"
    assert annotator(None).aspect_table is packaged_table()
    assert annotator(None, closed_choice="clm").closed_choice == "clm"
    assert annotator(None, closed_choice="clm").aspect_table is None  # clm reads no table
    assert annotator(None, cfg=config(DECIDER__CLOSED_CHOICE="clm")).closed_choice == "clm"


# -- the rules as functions ---------------------------------------------------------------------


def test_candidate_aspects_hint_then_the_lookups_top_two_uncapped() -> None:
    # The hint, then the lookup's top two: three aspects (the decision caps the fallback only).
    assert candidate_aspects("taxon", ["method", "unit"]) == [
        ("taxon", "hint"),
        ("method", "lookup"),
        ("unit", "lookup"),
    ]
    assert candidate_aspects("unit", ["unit", "measurement"]) == [
        ("unit", "hint"),
        ("measurement", "lookup"),
    ]
    # The lookup gives its top two; a third label never counts, other is excluded after.
    assert candidate_aspects(None, ["method", "unit", "location"]) == [
        ("method", "lookup"),
        ("unit", "lookup"),
    ]
    assert candidate_aspects(None, ["other", "method", "data_type"]) == [("method", "lookup")]
    assert candidate_aspects("other", ["other"]) == []
    assert candidate_aspects(None, []) == []


def test_fallback_aspects_follow_the_order_capped_at_two() -> None:
    every = frozenset({"envo", "ncbitaxon", "pato", "uo", "obi", "iao", "gaz", "ro"})
    order = ["other", "unit", "method", "taxon", "measurement"]
    assert FALLBACK_CAP == 2
    # other is never an aspect; unit only for a column with a unit.
    assert fallback_aspects(order, in_play=every, has_unit=False) == ["method", "taxon"]
    assert fallback_aspects(order, in_play=every, has_unit=True) == ["unit", "method"]
    # An aspect with no ontology in play is skipped (method: obi, iao, bco, genepio).
    assert fallback_aspects(order, in_play={"ncbitaxon", "pato"}, has_unit=False) == [
        "taxon",
        "measurement",
    ]
    assert fallback_aspects(order, in_play={"ro"}, has_unit=True) == []
    assert fallback_aspects(list(ASPECTS), in_play=every, has_unit=False) == [
        "taxon",
        "environment",
    ]


# -- the frozen table ---------------------------------------------------------------------------


def test_the_packaged_table_is_m0s_neon_aspect_lookup(tmp_path: Path) -> None:
    """The packaged table is M0's ``neon_aspect`` items on the registered snapshot, frozen: it
    rebuilds byte for byte from the snapshot, its sha256 is the pinned one, and for every
    leave-one-card-out fold the lookup A6 builds from it is M0's lookup (the same counts in the
    same order, the same prior), giving M0's prediction on every seen key."""
    text = TABLE_PATH.read_text(encoding="utf-8")
    assert freeze_table() == text
    table = packaged_table()
    assert table.sha256 == TABLE_SHA256 == AspectTable.load(TABLE_PATH, sha256=None).sha256
    r = reg.current()
    assert (table.labels_sha256, table.labels_content_sha256) == (
        r.labels_sha256,
        r.labels_content_sha256,
    )
    assert table.snapshot == r.snapshot and table.min_weight == 0.6
    counts = r.counts["neon_aspect"]
    assert len(table.items) == counts.n == 60
    assert dict(Counter(ASPECTS.index(i.aspect) for i in table.items)) == dict(counts.class_counts)
    assert dict(Counter(i.card for i in table.items)) == dict(counts.per_card)
    task = load_inputs(None, tmp_path).tasks["neon_aspect"]
    keys = task_keys(task)
    n_seen_right = n_unseen = 0
    for fold in task.leave_one_card_out():
        m0 = Lookup.fit(task, fold.train, keys)
        a6 = AspectLookup.for_card(table, fold.held_out)
        assert a6.n_items == len(fold.train)
        assert {k: list(v.items()) for k, v in a6.lookup.table.items()} == {
            k: list(v.items()) for k, v in m0.table.items()
        }
        assert list(a6.lookup.prior.items()) == list(m0.prior.items())
        assert [a for a, _ in a6.prior_order()][: len(m0.prior)] == [
            ASPECTS[label] for label, _ in m0.prior.most_common()
        ]
        for i in fold.test:
            state, label = task.items[i]
            if not m0.seen(keys[i]):
                n_unseen += 1
                assert a6.ranked(state) == []  # unseen: no aspect, the fallback decides
                continue
            assert a6.top(state)[0] == ASPECTS[m0.predict(keys[i])]
            n_seen_right += a6.top(state)[0] == ASPECTS[label]
    ev = evaluate_lookup(task)
    assert n_unseen == int(ev.novel.sum()) == 28  # baselines.json: novel_key.n
    # M0's lookup_acc 0.55 (33 of 60) = the seen keys it gets right + the unseen keys its
    # majority gets right; A6's top-1 on the seen keys is exactly M0's.
    unseen_right = sum(int(ev.pred[j] == ev.labels[j]) for j in range(ev.n) if bool(ev.novel[j]))
    assert n_seen_right + unseen_right == round(ev.acc() * 60) == 33


def test_the_table_is_pinned_and_checked(tmp_path: Path) -> None:
    raw = TABLE_PATH.read_bytes()
    copy = tmp_path / "aspect_lookup.json"
    copy.write_bytes(raw.replace(b'"aspect": "taxon"', b'"aspect": "method"', 1))
    with pytest.raises(AspectTableError, match="is not the pinned"):
        AspectTable.load(copy)
    doc = json.loads(raw)
    assert json.loads(render_table(doc)) == doc and render_table(doc).encode() == raw
    for broken, match in (
        ({**doc, "format": "x"}, "format"),
        ({**doc, "task_key": "0" * 16}, "task_key"),
        ({**doc, "n_items": 59}, "n_items"),
        (
            {**doc, "items": [{"card": "c", "column": "x", "aspect": "biome"}], "n_items": 1},
            "biome",
        ),
    ):
        copy.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(AspectTableError, match=match):
            AspectTable.load(copy, sha256=None)
    copy.write_text("{", encoding="utf-8")
    with pytest.raises(AspectTableError, match="not JSON"):
        AspectTable.load(copy, sha256=None)


# -- (a) no CLM answer drives Q1, Q2 or Q7 ---------------------------------------------------------


def test_no_clm_answer_drives_q1_q2_or_q7(tmp_path: Path) -> None:
    """Rules mode: CLM's closed-choice answers pulled to "No"/"other"/"the column name" and to
    "Yes"/"taxon"/"the most frequent data value" decide exactly the same records, groups, links
    and proposals; only the audit-only records differ, and they carry the skew."""
    runs: dict[str, tuple[DuckDBStore, AnnotationRun]] = {}
    for name, favour in (("no", SKEW_NO), ("yes", SKEW_YES)):
        store = DuckDBStore(tmp_path / f"{name}.duckdb")
        runs[name] = (store, annotator(store, provider=skewed(favour)).annotate(card(SERVICE_CARD)))
    assert decided(*runs["no"]) == decided(*runs["yes"])
    answers = {
        name: {(d["task_id"], d["answer"]) for d in Rows(*runs[name]).decisions if is_audit_only(d)}
        for name in runs
    }
    assert answers["no"] == {
        ("column.annotate", "No"),
        ("column.aspect", option("other")),
        ("avu.value_kind", "the column name"),
    }
    assert answers["yes"] == {
        ("column.annotate", "Yes"),
        ("column.aspect", option("taxon")),
        ("avu.value_kind", VALUE_KIND_TOP),
    }
    store, run = runs["no"]
    rows = Rows(store, run)
    # CLM said No to every column; every non-identifier column is annotated all the same.
    annotated = {d["column_name"] for d in rows.of("column.annotate") if d["answer"] == "Yes"}
    assert annotated == set(live_columns(run.card))
    assert any(p.scope == "column" for p in run.proposals)
    assert all(p.value_kind == VALUE_KIND_LABEL for p in run.proposals if p.scope == "column")
    assert "other" not in {p.aspect for p in run.proposals}


def test_no_q3_answer_chooses_an_aspect(tmp_path: Path) -> None:
    """Q3 (F9) pulled to NCBITaxon in one run and to PATO in the other: every column keeps the
    same aspects from the same sources, and Q3 is asked the same questions (once per chosen
    aspect but unit). The skew changes only what Q3 decides: which ontologies are kept."""
    rows: dict[str, Rows] = {}
    for favour in ("ncbitaxon", "pato"):
        store = DuckDBStore(tmp_path / f"{favour}.duckdb")
        run = annotator(store, provider=skewed({"column.ontology_fits": favour})).annotate(
            card(SERVICE_CARD)
        )
        rows[favour] = Rows(store, run)
    taxon, pato = rows["ncbitaxon"], rows["pato"]
    assert taxon.aspects() == pato.aspects()
    assert set(taxon.aspects()) == set(live_columns(card(SERVICE_CARD)))
    assert taxon.asked_q3() == pato.asked_q3()
    for column, chosen in taxon.aspects().items():
        assert len(chosen) <= MAX_CHOSEN
        asked = [a for c, a in taxon.asked_q3() if c == column]
        assert asked == [a for a, _, _ in chosen if a != "unit" and allowed_for_aspect(a)]

    def kept(r: Rows) -> dict[tuple[str, str], list[str]]:
        return {
            (g["column_name"], g["aspect"]): g["search_json"]["kept"]
            for g in r.groups.values()
            if g["task_id"] == "column.ontology_fits"
        }

    assert kept(taxon) != kept(pato)  # the skew reached Q3; it chose no aspect


def test_the_same_skews_drive_the_m2_behaviour(tmp_path: Path) -> None:
    """Under ``closed_choice: clm`` the skews decide (the contrast that makes the test above
    meaningful): "No" leaves every column out, "Yes"/"taxon"/"top_value" searches taxon only and
    builds the most-frequent-value kind."""
    cfg = config(DECIDER__CLOSED_CHOICE="clm")
    store = DuckDBStore(tmp_path / "no.duckdb")
    run = annotator(store, provider=skewed(SKEW_NO), cfg=cfg).annotate(card(SERVICE_CARD))
    assert not any(p.scope == "column" for p in run.proposals)
    assert not any(g["scope"] == "column" for g in store.groups(run.run_id))
    store = DuckDBStore(tmp_path / "yes.duckdb")
    run = annotator(store, provider=skewed(SKEW_YES), cfg=cfg).annotate(card(SERVICE_CARD))
    column = [p for p in run.proposals if p.scope == "column"]
    assert column and {p.aspect for p in column} == {"taxon"}
    assert {p.value_kind for p in column} == {VALUE_KIND_TOP}
    assert run.n_audit_only == 0


# -- (b) the rule records, the audit-only records, Q1 ------------------------------------------------


def test_rule_and_audit_records_carry_their_markers(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    audit = [d for d in rows.decisions if is_audit_only(d)]
    assert len(audit) == run.n_audit_only == run.to_dict()["n_audit_only"] > 0
    assert run.closed_choice == run.to_dict()["closed_choice"] == "rules"
    for d in audit:
        assert d["task_id"] in CLOSED_CHOICE_TASKS and d["method"] == "fake"
        assert (d["outcome"], d["reason"], d["level"]) == (
            "abstain",
            AUDIT_ONLY_REASON,
            "zero_shot",
        )
        assert d["probs"] is not None and d["raw_probs"] is not None  # kept as served
    assert run.outcomes["abstain"] >= run.n_audit_only
    for d in rows.decisions:
        DecisionRow.model_validate(d)  # the commit passed the CHECKs; the rows re-validate
        if d["method"] == "rule":
            assert (d["level"], d["calibration"], d["outcome"]) == ("none", "none", "rule")
            assert d["probs"] is None and d["raw_probs"] is None and d["context_tokens"] == 0
    for col in run.card.columns:
        q1 = rows.of("column.annotate", col.name)
        [rule] = [d for d in q1 if d["method"] == "rule"]
        if is_identifier(col):
            assert (rule["answer"], rule["reason"]) == ("No", "is_identifier") and q1 == [rule]
            assert not rows.of("column.aspect", col.name)
            continue
        assert (rule["answer"], rule["reason"], rule["model"]) == ("Yes", ANNOTATE_REASON, "rule")
        [clm1] = [d for d in q1 if d["method"] == "fake"]
        assert is_audit_only(clm1)
        q2 = rows.of("column.aspect", col.name)
        [clm2] = [d for d in q2 if d["method"] == "fake"]
        assert is_audit_only(clm2) and clm2["state_sha256"] == clm1["state_sha256"]
        rules = [d for d in q2 if d["method"] == "rule"]
        assert 1 <= len(rules) <= MAX_CHOSEN  # the fixture card: no column ends without one
        assert {d["reason"] for d in rules} <= set(ASPECT_REASONS.values())
    q7 = rows.of("avu.value_kind")
    rules = [d for d in q7 if d["method"] == "rule"]
    audits = [d for d in q7 if is_audit_only(d)]
    assert rules and len(rules) == len(audits) == len(q7) // 2
    for r in rules:
        assert (r["answer"], r["reason"], r["outcome"]) == (
            VALUE_KIND_LABEL,
            VALUE_KIND_REASON,
            "rule",
        )
        twins = [
            a
            for a in audits
            if (a["parent_decision_id"], a["group_id"], a["state_sha256"])
            == (r["parent_decision_id"], r["group_id"], r["state_sha256"])
        ]
        assert len(twins) == 1  # the CLM question of the same proposal, kept for audit
        assert rows.by_id[str(r["parent_decision_id"])]["task_id"] == "term.fits"
    for p in run.proposals:
        if p.value_kind_decision_id is not None:
            assert rows.by_id[str(p.value_kind_decision_id)]["reason"] == VALUE_KIND_REASON
            assert p.value_kind == VALUE_KIND_LABEL
    # Links come from term.fits decisions only: no rule or audit record of a closed choice.
    assert rows.links
    assert {rows.by_id[str(link["decision_id"])]["task_id"] for link in rows.links} == {"term.fits"}


class IdentifierPlanner(StaticPlanner):
    """The static plan with ``annotate=True`` on the identifier column ``siteID``."""

    name = "identifier-static"

    def plan(self, dataset: DatasetCard) -> PlanResult:
        result = super().plan(dataset)
        columns = dict(result.plan.columns)
        columns["siteID"] = columns["siteID"].model_copy(update={"annotate": True})
        plan = result.plan.model_copy(update={"columns": columns})
        return PlanResult(plan=plan, planner=self.name)


@pytest.mark.parametrize("mode", MODES)
def test_an_identifier_stays_out_whatever_the_planner_says(tmp_path: Path, mode: str) -> None:
    """A planner's ``annotate=True`` on an identifier only ever sent it to CLM. Under the rules
    it stays out even when CLM answers Yes: a ``No`` rule, CLM's answers audit-only. Under
    ``clm`` CLM is asked and decides, as in M2 (here it answers No: the OLS fixture closure,
    enumerated under the static planner, holds no identifier column's searches)."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    cfg = config(DECIDER__CLOSED_CHOICE=mode)
    provider = skewed(SKEW_YES if mode == "rules" else SKEW_NO)
    run = annotator(store, provider=provider, planner=IdentifierPlanner(), cfg=cfg).annotate(
        card(SERVICE_CARD)
    )
    rows = Rows(store, run)
    assert is_identifier(run.card.column("siteID"))
    q1 = rows.of("column.annotate", "siteID")
    q2 = rows.of("column.aspect", "siteID")
    [asked] = [d for d in q1 if d["method"] == "fake"]
    if mode == "rules":
        [rule] = [d for d in q1 if d["method"] == "rule"]
        assert (rule["answer"], rule["reason"]) == ("No", "is_identifier")
        assert asked["answer"] == "Yes" and is_audit_only(asked)
        assert len(q2) == 1 and is_audit_only(q2[0])  # no aspect rule: not annotated
        assert not rows.of("column.ontology_fits", "siteID")
        assert not any(g["column_name"] == "siteID" for g in rows.groups.values())
    else:
        assert q1 == [asked] and asked["answer"] == "No" and not is_audit_only(asked)
        [aspect] = q2
        assert (aspect["outcome"], aspect["reason"]) == ("rejected", "column_not_annotated")


def test_a_planners_exclusion_keeps_no_column_out(tmp_path: Path) -> None:
    """ "Annotate every non-identifier column": a planner's ``annotate=False`` decides nothing
    under the rules. The column is annotated (a ``Yes`` rule) and gets its aspects; the plan
    keeps the planner's opinion (``runs.plan_json``); CLM is not asked about the column, as M2
    never asked about an excluded column, so it has no audit record."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    excluded = {"detectionMethod", "taxonID", "clusterSize"}
    planner = HintPlanner(lambda col: {"annotate": False} if col.name in excluded else {})
    run = annotator(store, planner=planner).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    assert not [d for d in rows.decisions if d["reason"] == "planner_annotate_false"]
    for name in excluded:
        assert rows.run is not None and rows.run["plan_json"]["columns"][name]["annotate"] is False
        q1 = rows.of("column.annotate", name)
        assert [(d["method"], d["answer"], d["reason"]) for d in q1] == [
            ("rule", "Yes", ANNOTATE_REASON)
        ]
        assert not [d for d in rows.of("column.aspect", name) if d["method"] == "fake"]
        assert rows.aspects()[name]
    yes = {d["column_name"] for d in rows.of("column.annotate") if d["answer"] == "Yes"}
    assert yes == set(live_columns(run.card))
    searched = {g["column_name"] for g in rows.groups.values() if g["task_id"] == "term.fits"}
    assert excluded <= searched  # each reached S and Q4


# -- (c) the hint, lookup and fallback paths ---------------------------------------------------------


def test_the_packaged_lookup_decides_on_a_host_without_labels(tmp_path: Path) -> None:
    """A sidecar that holds no label (the serving host's) changes nothing: the lookup reads the
    packaged table, the annotated card's own items held out. Each column takes the top two of
    its name on the other cards; a name no other card carries goes to the fallback, in the
    order of the other cards' prior. The run row names the snapshot the table came from."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    assert not store.labels_for("column.aspect")
    run = annotator(store).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    table = packaged_table()
    assert rows.run is not None and rows.run["labels_sha256"] == reg.current().labels_sha256
    own = {i.column for i in table.items if i.card == SERVICE_CARD}
    order = oracle_order(table, SERVICE_CARD)
    n_lookup = n_fallback = 0
    for column in live_columns(run.card):
        col = run.card.column(column)
        chosen = rows.aspects()[column]
        top = oracle_top(table, SERVICE_CARD, column)
        looked_up = [a for a in top if a != "other"]
        if looked_up:
            n_lookup += 1
            assert chosen == [
                (a, ASPECT_REASONS["lookup"], ASPECT_MODELS["lookup"]) for a in looked_up
            ]
        else:
            n_fallback += 1
            want = fallback_aspects(order, in_play=allowed_ontologies(), has_unit=bool(col.unit))
            assert chosen == [
                (a, ASPECT_REASONS["fallback"], ASPECT_MODELS["fallback"]) for a in want
            ]
        for a6 in rows.a6(column):
            lookup = a6["lookup"]
            assert (lookup["held_out"], lookup["table_sha256"]) == (SERVICE_CARD, TABLE_SHA256)
            assert lookup["n_items"] == 60 - sum(1 for i in table.items if i.card == SERVICE_CARD)
            assert lookup["key"] == ["column.aspect", "column", column, ""]
            assert lookup["top"] == top and lookup["seen"] is bool(top)
            assert lookup["labels_sha256"] == reg.current().labels_sha256
            assert a6["aspects"] == [a for a, _, _ in chosen]
            if a6["aspect_source"] == "fallback":
                assert a6["fallback"]["kept"] == a6["aspects"]
                assert [a for a, _ in a6["fallback"]["order"]] == order
            else:
                assert a6["fallback"] is None
    assert n_lookup >= 3 and n_fallback >= 3
    # The card's own labels never count: a column only it carries goes to the fallback.
    assert own and any(not oracle_top(table, SERVICE_CARD, c) for c in own)


def allowed_ontologies() -> frozenset[str]:
    """The static planner's ontologies in play: the whole registry."""
    return frozenset(o for a in ASPECTS for o in allowed_for_aspect(a))


# A test table: other cards' labels for three of the service card's columns, plus a taxon row on
# the service card itself (held out). observerDistance: unit (zz_b), measurement (zz_c), location
# (zz_d), one row each, so card order breaks the ties; detectionMethod: other 2, method 1,
# data_type 1; clusterSize: unit 2, measurement 1. The prior over the other cards: unit 3,
# other 2, measurement 2 (other seen first), location, method, data_type 1 each.
TEST_TABLE = table_of(
    ("zz_b", "observerDistance", "unit"),
    ("zz_b", "detectionMethod", "other"),
    ("zz_b", "clusterSize", "measurement"),
    ("zz_c", "observerDistance", "measurement"),
    ("zz_c", "detectionMethod", "other"),
    ("zz_c", "clusterSize", "unit"),
    ("zz_d", "observerDistance", "location"),
    ("zz_d", "detectionMethod", "method"),
    ("zz_d", "clusterSize", "unit"),
    ("zz_e", "detectionMethod", "data_type"),
    (SERVICE_CARD, "observerDistance", "taxon"),
    (SERVICE_CARD, "vernacularName", "taxon"),
)


def test_the_hint_comes_first_and_the_lookup_adds_its_top_two(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    hints = {"clusterSize": "measurement", "observerDistance": "taxon", "vernacularName": "other"}
    planner = HintPlanner(lambda col: {"aspect": hints[col.name]} if col.name in hints else {})
    run = annotator(store, planner=planner, aspect_table=TEST_TABLE).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    hint = (ASPECT_REASONS["hint"], ASPECT_MODELS["hint"])
    look = (ASPECT_REASONS["lookup"], ASPECT_MODELS["lookup"])
    fall = (ASPECT_REASONS["fallback"], ASPECT_MODELS["fallback"])
    chosen = rows.aspects()
    # clusterSize: the hint (measurement), then the lookup's unit; its measurement is a duplicate.
    assert chosen["clusterSize"] == [("measurement", *hint), ("unit", *look)]
    # observerDistance: the hint, then both of the lookup's top two; no cap drops one, and the
    # card's own taxon row is held out.
    assert chosen["observerDistance"] == [
        ("taxon", *hint),
        ("unit", *look),
        ("measurement", *look),
    ]
    # detectionMethod: the lookup's top two are other and method; other is excluded.
    assert chosen["detectionMethod"] == [("method", *look)]
    # vernacularName: a hint of other is no aspect, and only its own card carries the name: the
    # fallback, in the prior's order (unit 3: no unit here; other; measurement; location).
    assert chosen["vernacularName"] == [("measurement", *fall), ("location", *fall)]
    # boutNumber has a unit: the fallback starts with unit (uo by rule, as M2's unit aspect).
    assert chosen["boutNumber"] == [("unit", *fall), ("measurement", *fall)]
    [uo] = [d for d in rows.of("column.ontology_fits", "boutNumber") if d["method"] == "rule"]
    assert (uo["answer"], uo["reason"]) == ("uo", "unit_aspect")
    for a6 in rows.a6("observerDistance"):
        assert a6["sources"] == ["hint", "lookup", "lookup"]
        assert a6["lookup"]["counts"] == {"unit": 1, "measurement": 1, "location": 1}
        assert a6["lookup"]["laplace"] == {
            a: pytest.approx(2 / 11) for a in ("unit", "measurement", "location")
        }
        assert a6["lookup"]["n_items"] == 10 and a6["lookup"]["top"] == ["unit", "measurement"]
    # Q3 is asked once per chosen aspect but unit: three questions at most for one column.
    assert [a for c, a in rows.asked_q3() if c == "observerDistance"] == ["taxon", "measurement"]
    assert rows.run is not None and rows.run["labels_sha256"] == "t" * 64


def test_an_unseen_name_takes_the_fallback_never_unit_without_a_unit(tmp_path: Path) -> None:
    """A table whose prior puts unit first: a column without a unit is never given unit (S would
    search UO for the word "unit"); a column with a unit is. No column is left without an
    aspect, and the fallback asks Q3 at most twice per column."""
    table = table_of(
        ("zz_b", "elsewhere", "unit"),
        ("zz_c", "elsewhere", "unit"),
        ("zz_d", "elsewhere", "unit"),
        ("zz_b", "other1", "other"),
        ("zz_b", "other2", "data_type"),
        ("zz_c", "other3", "location"),
    )
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, aspect_table=table).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    for name in live_columns(run.card):
        chosen = [
            a for a, reason, _ in rows.aspects()[name] if reason == ASPECT_REASONS["fallback"]
        ]
        if run.card.column(name).unit:
            assert chosen == ["unit", "data_type"], name
        else:
            assert chosen == ["data_type", "location"], name
        assert len([a for c, a in rows.asked_q3() if c == name]) <= FALLBACK_CAP
    assert not [a for a in run.abstained if a["reason"] == "no_aspect"]


def test_without_any_ontology_in_play_a_column_is_no_aspect(tmp_path: Path) -> None:
    """The fallback skips aspects with no ontology in play; with only RO in play (it serves
    ``other`` alone) an unseen column has no aspect and is reported ``no_aspect``."""

    class RoOnly(StaticPlanner):
        name = "ro-only"

        def plan(self, dataset: DatasetCard) -> PlanResult:
            result = super().plan(dataset)
            return PlanResult(
                plan=result.plan.model_copy(update={"ontologies": ["ro"]}), planner=self.name
            )

    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, planner=RoOnly(), aspect_table=table_of()).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    assert {a["column_name"] for a in run.abstained if a["reason"] == "no_aspect"} == set(
        live_columns(run.card)
    )
    assert not rows.aspects() and not rows.of("column.ontology_fits")


@pytest.mark.parametrize(
    "kw",
    [
        {"tier": "ols_rank"},
        {"ols_rank_tasks": {"column.ontology_fits", "term.fits"}},
        {"provider": "down"},
    ],
    ids=["tier-ols_rank", "ols_rank_tasks", "clm-down"],
)
def test_the_fallback_needs_no_clm(tmp_path: Path, kw: dict[str, Any]) -> None:
    """Nothing about a column's aspects depends on CLM: with ``column.ontology_fits`` decided by
    ``ols_rank`` (by tier or by task) or with clm-serve down, every column keeps exactly the
    aspects of an ordinary run and its Q3 groups are ``ols_rank``; none is ``no_aspect``."""
    base = DuckDBStore(tmp_path / "base.duckdb")
    expected = Rows(base, annotator(base).annotate(card(SERVICE_CARD))).aspects()
    store = DuckDBStore(tmp_path / "prov.duckdb")
    if kw.get("provider") == "down":
        kw = {"provider": down_provider()}
    run = annotator(store, **kw).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    assert rows.aspects() == expected
    assert set(expected) == set(live_columns(run.card))
    assert not [a for a in run.abstained if a["reason"] == "no_aspect"]
    groups = [g for g in rows.groups.values() if g["task_id"] == "column.ontology_fits"]
    assert groups and {g["method"] for g in groups} == {"ols_rank"}
    assert "column" in {p.scope for p in run.proposals}


class RestrictedPlanner(HintPlanner):
    """The static plan with only PATO, UO and ENVO in play, plus column hints."""

    name = "restricted-static"

    def plan(self, dataset: DatasetCard) -> PlanResult:
        result = super().plan(dataset)
        plan = result.plan.model_copy(update={"ontologies": ["pato", "uo", "envo"]})
        return PlanResult(plan=plan, planner=self.name)


def test_the_plan_bounds_the_fallback_and_its_ontology_is_appended(tmp_path: Path) -> None:
    """The ontologies the plan puts in play bound the fallback (an aspect with none in play is
    skipped) but not a hinted aspect (with none in play it is kept, with nothing to search, as in
    M2); the planner's ontology is appended under the column's first aspect, as in M2."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    hints = {
        "clusterSize": {"aspect": "measurement", "ontology": "uo"},
        "observerDistance": {"aspect": "taxon"},
    }
    planner = RestrictedPlanner(lambda col: hints.get(col.name, {}))
    run = annotator(store, planner=planner, aspect_table=table_of()).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    in_play = {"pato", "uo", "envo"}
    # An empty table has no prior: registry order, skipping taxon and method (none in play).
    for name in set(live_columns(run.card)) - set(hints):
        want = ["environment", "measurement"]
        assert [a for a, _, _ in rows.aspects()[name]] == want
        assert all(allowed_for_aspect(a) & in_play for a in want)
    # observerDistance: taxon has no ontology in play: no Q3, no group, the aspect stands.
    assert not rows.of("column.ontology_fits", "observerDistance")
    assert rows.aspects()["observerDistance"] == [
        ("taxon", ASPECT_REASONS["hint"], ASPECT_MODELS["hint"])
    ]
    assert "observerDistance" not in {a["column_name"] for a in run.abstained}
    # clusterSize: measurement keeps PATO and ENVO at most; UO comes from the planner's hint.
    [q3] = rows.of("column.ontology_fits", "clusterSize")
    assert set(rows.groups[str(q3["group_id"])]["search_json"]["kept"]) <= {"pato", "envo"}
    terms = {
        (g["ontology_id"], g["aspect"])
        for g in rows.groups.values()
        if g["task_id"] == "term.fits" and g["column_name"] == "clusterSize"
    }
    assert ("uo", "measurement") in terms


def test_a_bare_duckdb_file_without_the_schema(tmp_path: Path) -> None:
    """An existing DuckDB file without the ``mesa_clm`` schema: the rules read no sidecar, so a
    run decides and commits (the commit creates the schema), as under ``clm``."""
    for mode in MODES:
        path = tmp_path / f"{mode}.duckdb"
        duckdb.connect(str(path)).close()
        store = DuckDBStore(path)
        run = annotator(store, cfg=config(DECIDER__CLOSED_CHOICE=mode)).annotate(card(SERVICE_CARD))
        row = store.run(run.run_id)
        assert row is not None and row["status"] == "decided"


# -- (d) closed_choice: clm is the M2 behaviour -------------------------------------------------------


def test_clm_mode_is_the_m2_behaviour(tmp_path: Path) -> None:
    """No A6 record or marker; Q1's CLM answer decides annotation, Q2's top-1 (proposed) or
    top-2 the aspects (``other`` dropped), Q7's answer the value kind when proposed; the run row
    names no labels snapshot. The M2 pipeline assertions of ``tests/unit/test_pipeline_fake.py``
    also run in this mode."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    run = annotator(store, cfg=config(DECIDER__CLOSED_CHOICE="clm")).annotate(card(SERVICE_CARD))
    rows = Rows(store, run)
    assert run.closed_choice == "clm" and run.n_audit_only == 0
    assert rows.run is not None and rows.run["labels_sha256"] is None
    assert not any(
        is_audit_only(d) or str(d["reason"] or "").startswith("a6_") for d in rows.decisions
    )
    assert not any("a6" in g["search_json"] for g in rows.groups.values())
    n_annotated = 0
    for name in live_columns(run.card):
        [q1] = rows.of("column.annotate", name)
        [q2] = rows.of("column.aspect", name)
        assert q1["method"] == q2["method"] == "fake"
        q3_aspects = {d["state_json"]["aspect"] for d in rows.of("column.ontology_fits", name)}
        if q1["answer"] != "Yes":
            assert (q2["outcome"], q2["reason"]) == ("rejected", "column_not_annotated")
            assert not q3_aspects
            continue
        n_annotated += 1
        order = sorted(range(len(ASPECTS)), key=lambda i: -q2["probs"][i])
        n = 1 if q2["outcome"] in ("auto", "proposed") else 2
        assert q3_aspects == {ASPECTS[i] for i in order[:n]} - {"other"}
    assert n_annotated > 0
    for p in run.proposals:
        if p.value_kind_decision_id is not None:
            vk = rows.by_id[str(p.value_kind_decision_id)]
            assert vk["method"] == "fake"
            want = vk["answer"] if vk["outcome"] in ("auto", "proposed") else VALUE_KIND_LABEL
            assert p.value_kind == want


# -- (e) every fixture card, both modes, the shipped decider ----------------------------------------


def _q3_per_column(rows: Rows) -> Counter[str]:
    return Counter(c for c, _ in rows.asked_q3())


@pytest.mark.parametrize("mode", MODES)
def test_every_fixture_card_annotates_under_the_shipped_decider(tmp_path: Path, mode: str) -> None:
    """All seven fixture cards end to end under the shipped decider (K1's ``ols_rank`` for
    term.fits) in both modes, OLS replayed (a ``ReplayMiss`` fails the test). Under ``rules``
    every non-identifier column is annotated and keeps one to three aspects (none is
    ``no_aspect``), Q3 is asked at most three times for a column (twice for a fallback
    column), as M2 asked at most three, and the run names the table's snapshot."""
    store = DuckDBStore(tmp_path / "prov.duckdb")
    ann = annotator(store, cfg=shipped_config(DECIDER__CLOSED_CHOICE=mode))
    for path in CARD_PATHS:
        run = ann.annotate(load_card(path))
        rows = Rows(store, run)
        assert rows.run is not None and rows.run["status"] == "decided" and not run.degraded
        assert run.proposals and run.closed_choice == mode
        yes = {d["column_name"] for d in rows.of("column.annotate") if d["answer"] == "Yes"}
        if mode == "clm":
            assert run.n_audit_only == 0 and rows.run["labels_sha256"] is None
            continue
        assert yes == set(live_columns(run.card))
        assert run.n_audit_only == sum(is_audit_only(d) for d in rows.decisions) > 0
        assert rows.run["labels_sha256"] == reg.current().labels_sha256
        assert not [a for a in run.abstained if a["reason"] == "no_aspect"]
        chosen = rows.aspects()
        assert set(chosen) == yes
        per_column = _q3_per_column(rows)
        for name, aspects in chosen.items():
            assert 1 <= len(aspects) <= MAX_CHOSEN
            sources = {reason for _, reason, _ in aspects}
            assert len(sources) == 1 or ASPECT_REASONS["fallback"] not in sources
            cap = FALLBACK_CAP if sources == {ASPECT_REASONS["fallback"]} else MAX_CHOSEN
            assert per_column[name] <= min(cap, len(aspects))


# -- explain, review, feedback ----------------------------------------------------------------------


def test_explain_review_and_feedback_never_offer_an_audit_record(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    svc = service(store)
    run = svc.annotate(card(SERVICE_CARD), "agent-x", owner="alice")
    rows = Rows(store, run)
    audit = {str(d["decision_id"]) for d in rows.decisions if is_audit_only(d)}
    assert audit
    summary = svc.run_summary(run.run_id, owner="alice")
    assert summary["pending_groups"]
    for g in summary["groups"]:
        assert str(g["winner_decision_id"]) not in audit
    for gid in summary["pending_groups"]:
        cands = svc.candidates_for_group(UUID(gid))
        assert cands and not {c["decision_id"] for c in cands} & audit
        assert {c["task_id"] for c in cands} == {"term.fits"}
    out = svc.explain(owner="alice", run_id=run.run_id, limit=10_000)
    shown = [d for d in out["decisions"] if d["reason"] == AUDIT_ONLY_REASON]
    assert len(shown) == len(audit)
    assert all(d["outcome"] == "abstain" and d["task_id"] in CLOSED_CHOICE_TASKS for d in shown)
    # A pick labels the group's deciding record only: never a closed-choice task.
    gid = UUID(summary["pending_groups"][0])
    pick = next(c for c in svc.candidates_for_group(gid) if not c["is_anchor"])
    result = svc.record_human_pick(
        gid, "alice", via="tool", owner="alice", option_key=pick["option_key"]
    )
    assert result["labels_written"] > 0
    assert store.labels_for("term.fits")
    for task_id in CLOSED_CHOICE_TASKS:
        assert not store.labels_for(task_id)


# -- the command line ---------------------------------------------------------------------------


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


def _annotate(capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[dict[str, Any], str]:
    code = main(
        ["annotate", "--card", str(CARD_FILE), "--provider", "fake", "--fake-seed", str(FAKE_SEED),
         "--out", "-", *extra]
    )  # fmt: skip
    assert code == EXIT_OK
    out = capsys.readouterr()
    return dict(json.loads(out.out)), out.err


def test_cli_closed_choice(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, summary = _annotate(capsys)
    assert data["closed_choice"] == "rules" and data["n_audit_only"] > 0
    assert "closed choices: rules (Q1, Q2, Q7 by rule, DESIGN A6" in summary
    old, summary = _annotate(capsys, "--closed-choice", "clm")
    assert old["closed_choice"] == "clm" and old["n_audit_only"] == 0
    assert "closed choices: clm (the M2 behaviour" in summary
    evaluated, _ = _annotate(capsys, "--eval-result")
    assert evaluated["closed_choice"] == "rules" and evaluated["n_audit_only"] > 0
    monkeypatch.setenv("MESA_CLM_DECIDER__CLOSED_CHOICE", "clm")
    configured, _ = _annotate(capsys)
    assert configured["closed_choice"] == "clm"
    flagged, _ = _annotate(capsys, "--closed-choice", "rules")
    assert flagged["closed_choice"] == "rules"
    with pytest.raises(SystemExit) as exc:
        main(["annotate", "--card", str(CARD_FILE), "--provider", "fake", "--closed-choice", "llm"])
    assert exc.value.code == 2
