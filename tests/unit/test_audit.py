"""Curator audits (``mesa_clm.audit``; plan §4.7, §8 M4; the M4 brief R7): sampling refuses a
bench card, draws by default only decisions that carry an ``artifact_version`` (``--all-tiers``
restores the whole pool) with the would-be-auto decile computed per task over that pool,
stratifies by outcome in equal shares from a seeded draw spanning enough cards, the sample file
holds ids only and its sampling mode, the review shows what a curator needs to judge an item
(target, description, dtype, unit, aspect, ontology, OLS queries, the tier marked plainly), a
verdict mints curator labels outside every fold, the ``audits`` rows follow the pass rule (and
get none for decisions that apply no artifact), the CLI's ``audit sample|review|record`` run end
to end with the review at a terminal only. The runs are fake-provider runs over synthetic
(non-bench) copies of the fixture card, with or without a synthetic promoted ``term.fits`` probe
(``tests/fakes/m4.synthetic_probe``, never evidence); OLS replays."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from mesa_clm import audit
from mesa_clm.audit import (
    CHECKLIST,
    STRATA,
    TOP_DECILE,
    AuditError,
    AuditItem,
    AuditSample,
    audit_rows,
    build_sample,
    candidates,
    context_lines,
    labels_for_verdict,
    load_sample,
    proposed_precision,
    stratified_sample,
    summarize,
    tier_note,
    write_sample,
)
from mesa_clm.cards import parse_card
from mesa_clm.cli import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, audit_review_interactive, main
from mesa_clm.policy import Policy
from mesa_clm.provenance.models import AuditRow
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers.tiered import (
    ArtifactBundle,
    FakeProvider,
    Promotion,
    ServedProbe,
    fake_fingerprint,
)
from tests.conftest import CARD_TEXT, OLS_DIR
from tests.fakes.m4 import synthetic_probe
from tests.fakes.pipeline import FAKE_SEED, annotator, config, service, shipped_config

_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")


def _synthetic_card(i: int) -> Any:
    """The fixture card under a product and table no bench card has (D30)."""
    text = CARD_TEXT.replace("DP1.10003.001 / brd_countdata", f"DP9.{i:05d}.001 / synth_{i}")
    text = text.replace("Product: DP1.10003.001", f"Product: DP9.{i:05d}.001")
    return parse_card(text)


def _runs(store: DuckDBStore, n_cards: int = 3) -> list[dict[str, Any]]:
    ann = annotator(store, cfg=config())
    out = []
    for i in range(n_cards):
        run = ann.annotate(_synthetic_card(i))
        row = store.run(run.run_id)
        assert row is not None
        out.append(row)
    return out


def _promoted_provider(version: str = "v1") -> FakeProvider:
    """The fake provider with a synthetic ``term.fits`` probe promoted (as ``CURRENT.json``
    gives the live provider after ``learn promote --tier probe``, DESIGN A7)."""
    fp = fake_fingerprint(seed=FAKE_SEED)
    probe = synthetic_probe("lowdim.v1", task_id="term.fits", fp=fp)
    bundle = ArtifactBundle(
        version=version,
        encoder_fp=fp.encoder_fp,
        clm_model_fp=fp.clm_model_fp,
        probes={probe.question_key: ServedProbe(probe, version=version)},
        promoted={
            "term.fits": Promotion(tier="probe", question_key=probe.question_key, version=version)
        },
    )
    return FakeProvider(seed=FAKE_SEED, artifacts=bundle)


def _promoted_runs(store: DuckDBStore, n_cards: int = 3) -> list[dict[str, Any]]:
    """Synthetic cards annotated under the shipped defaults with the promoted probe: the
    ``term.fits`` decisions carry ``artifact_version``, the zero-shot Q3 decisions none."""
    out = []
    for i in range(n_cards):
        ann = annotator(store, provider=_promoted_provider(), cfg=shipped_config())
        run = ann.annotate(_synthetic_card(i))
        row = store.run(run.run_id)
        assert row is not None
        out.append(row)
    return out


def _bench(card: str) -> bool:
    return card.startswith("DP1.")


@pytest.fixture
def store(tmp_path: Path) -> DuckDBStore:
    s = DuckDBStore(tmp_path / "prov.duckdb")
    s.ensure_schema()
    return s


def test_sampling_refuses_bench_cards_and_stratifies(store: DuckDBStore, tmp_path: Path) -> None:
    from tests.fakes.pipeline import card as fixture_card

    svc = service(store)
    runs = _runs(store)
    bench_run = annotator(store).annotate(fixture_card("DP1.10003.001.brd_countdata"))
    policy = Policy.from_config(config().policy)
    with pytest.raises(AuditError, match="bench card"):
        candidates(
            store,
            [*runs, store.run(bench_run.run_id)],
            policy=policy,
            results_root=tmp_path,
            is_bench_card=svc.is_bench_card,
        )
    # Fake zero_shot runs apply no artifact: the default (artifact-only) pool is empty and
    # build_sample names --all-tiers; --all-tiers (artifact_only=False) is the whole pool.
    empty, _ = candidates(
        store, runs, policy=policy, results_root=tmp_path, is_bench_card=svc.is_bench_card
    )
    assert empty == []
    with pytest.raises(AuditError, match=r"no artifact-backed decision.*--all-tiers"):
        build_sample(empty, runs=[], thresholds={}, n=3, min_cards=1)
    pool, thresholds = candidates(
        store,
        runs,
        policy=policy,
        results_root=tmp_path,
        is_bench_card=svc.is_bench_card,
        artifact_only=False,
    )
    assert pool and all(i.card.startswith("DP9.") for i in pool)
    assert all(i.artifact_version is None for i in pool)
    assert thresholds == {t: None for t in thresholds}  # the shipped policy cites nothing
    present = {i.stratum for i in pool}
    assert "proposed" in present and "anchor_abstain" in present and "would_be_auto" in present
    # No rule/planner/unavailable decision, and ols_rank proposals are proposed (K1's).
    assert {i.method for i in pool} <= {"fake", "ols_rank"}
    assert all(i.stat is not None for i in pool if i.stratum == "would_be_auto")
    # Equal shares, seeded, spanning the cards; the remainder goes to the first strata.
    sample = stratified_sample(pool, n=7, min_cards=2)
    counts = {s: sum(1 for i in sample if i.stratum == s) for s in STRATA}
    assert counts == {"would_be_auto": 3, "proposed": 2, "anchor_abstain": 2}
    assert sample == stratified_sample(pool, n=7, min_cards=2)
    assert len({i.card for i in sample}) >= 2
    assert sample != stratified_sample(pool, n=7, min_cards=2, seed=1)
    assert len({i.card for i in stratified_sample(pool, n=30, min_cards=3)}) == 3
    # the draw is interleaved by card: a small sample already spans every card the pool has
    small = stratified_sample(pool, n=3, min_cards=3)
    assert len({i.card for i in small}) == 3
    with pytest.raises(AuditError, match="fewer than --min-cards"):
        stratified_sample(pool, n=3, min_cards=4)
    with pytest.raises(AuditError, match="no eligible"):
        stratified_sample([], n=3, min_cards=1)
    # A stratum short of its share passes the rest on.
    few = [i for i in pool if i.stratum != "anchor_abstain"]
    big = stratified_sample(few, n=min(len(few), 9), min_cards=1)
    assert len(big) == min(len(few), 9) and not any(i.stratum == "anchor_abstain" for i in big)
    # The file: ids only, the checklist, the strata, the sampling mode.
    built = build_sample(
        pool,
        runs=[str(r["run_id"]) for r in runs],
        thresholds=thresholds,
        n=9,
        min_cards=3,
        artifact_only=False,
    )
    path = write_sample(tmp_path / "sample.json", built)
    assert oct(os.stat(path).st_mode)[-3:] == "600"
    data = json.loads(path.read_text())
    assert data["format"] == audit.FORMAT and data["checklist"] == list(CHECKLIST)
    assert data["artifact_only"] is False
    assert set(data["items"][0]) == set(AuditItem.model_fields)
    assert not any("observerDistance" in json.dumps(i) for i in data["items"])  # no card content
    assert load_sample(path) == built
    # A sample file written before the mode existed reads as artifact-only (the format is kept).
    del data["artifact_only"]
    (tmp_path / "old.json").write_text(json.dumps(data))
    assert load_sample(tmp_path / "old.json").artifact_only is True
    with pytest.raises(AuditError, match="not an audit sample"):
        load_sample(tmp_path / "prov.duckdb")


class _Rows:
    """A store of one run's decisions, for the pool rules alone."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def decisions(self, run_id: Any) -> list[dict[str, Any]]:
        return [dict(r, run_id=str(run_id)) for r in self.rows]


def _decision(i: int, task_id: str, p_fit: float, version: str | None) -> dict[str, Any]:
    return {
        "decision_id": f"d{i:03d}",
        "group_id": None,
        "task_id": task_id,
        "task_key": "k",
        "shape": "rank_fit",
        "method": "clm",
        "level": "probe" if version else "zero_shot",
        "artifact_version": version,
        "p_fit": p_fit,
        "outcome": "proposed",
        "reason": None,
    }


def test_the_would_be_auto_decile_is_per_task_over_the_artifact_backed_pool(
    tmp_path: Path,
) -> None:
    """Unpromoted zero_shot ``column.ontology_fits`` decisions (p_fit 1.000 everywhere, no
    artifact) never set or join the would-be-auto stratum by default; with ``--all-tiers``
    each task's decile is its own."""
    probe = [_decision(i, "term.fits", 0.50 + i * 0.01, "v1") for i in range(20)]  # 0.50..0.69
    saturated = [_decision(100 + i, "column.ontology_fits", 1.0, None) for i in range(20)]
    policy = Policy.from_config(config().policy)
    runs = [{"run_id": "r1", "card_name": "DP9.00001.001 / synth_1"}]
    pool, thresholds = candidates(
        _Rows(probe + saturated),
        runs,
        policy=policy,
        results_root=tmp_path,
        is_bench_card=lambda c: False,
    )
    assert thresholds == {"term.fits": None}
    assert {i.task_id for i in pool} == {"term.fits"} and all(
        i.artifact_version == "v1" for i in pool
    )
    cut = float(np.quantile([d["p_fit"] for d in probe], TOP_DECILE))
    auto = {i.decision_id for i in pool if i.stratum == "would_be_auto"}
    assert auto == {d["decision_id"] for d in probe if d["p_fit"] >= cut} and len(auto) == 2
    assert all(i.stat is not None and i.stat < cut for i in pool if i.stratum == "proposed")
    everything, thresholds_all = candidates(
        _Rows(probe + saturated),
        runs,
        policy=policy,
        results_root=tmp_path,
        is_bench_card=lambda c: False,
        artifact_only=False,
    )
    assert thresholds_all == {"term.fits": None, "column.ontology_fits": None}
    by_task = {t: [i for i in everything if i.task_id == t] for t in thresholds_all}
    # The term.fits decile is unchanged by the saturated task; the saturated task is all
    # would-be-auto at its own (meaningless) decile of 1.0, which is why it is report-only.
    assert {i.decision_id for i in by_task["term.fits"] if i.stratum == "would_be_auto"} == auto
    assert all(i.stratum == "would_be_auto" for i in by_task["column.ontology_fits"])
    assert all(i.artifact_version is None for i in by_task["column.ontology_fits"])


def test_default_sample_is_the_promoted_artifacts_decisions(
    store: DuckDBStore, tmp_path: Path
) -> None:
    svc = service(store)
    runs = _promoted_runs(store)
    policy = Policy.from_config(shipped_config().policy)
    pool, thresholds = candidates(
        store, runs, policy=policy, results_root=tmp_path, is_bench_card=svc.is_bench_card
    )
    assert pool and thresholds == {"term.fits": None}
    assert {(i.task_id, i.level, i.artifact_version) for i in pool} == {
        ("term.fits", "probe", "v1")
    }
    assert {i.stratum for i in pool} == set(STRATA)
    everything, _ = candidates(
        store,
        runs,
        policy=policy,
        results_root=tmp_path,
        is_bench_card=svc.is_bench_card,
        artifact_only=False,
    )
    assert {i.decision_id for i in pool} < {i.decision_id for i in everything}
    rest = [i for i in everything if i.decision_id not in {p.decision_id for p in pool}]
    assert rest and all(i.artifact_version is None and i.level == "zero_shot" for i in rest)
    assert {i.task_id for i in rest} == {"column.ontology_fits"}
    built = build_sample(
        pool, runs=[str(r["run_id"]) for r in runs], thresholds=thresholds, n=9, min_cards=3
    )
    assert built.artifact_only is True and all(i.artifact_version == "v1" for i in built.items)
    # Every item of an artifact-only sample can feed an audits row once reviewed.
    for item in built.items:
        if item.stratum == "would_be_auto":
            item.verdict = "correct"
    s = summarize(built)
    assert s["artifact_only"] is True and s["reviewed_without_artifact"] == 0
    assert set(s["would_be_auto"]) <= {f"{i.task_key}@v1" for i in built.items}


def test_review_shows_what_a_curator_needs(store: DuckDBStore, tmp_path: Path) -> None:
    svc = service(store)
    runs = _promoted_runs(store, 1)
    run = runs[0]
    rows = store.decisions(run["run_id"])
    # A term.fits item on a column with a description and a unit, decided by the probe.
    term = next(
        d
        for d in rows
        if d["task_id"] == "term.fits"
        and d["scope"] == "column"
        and d["state_json"]["column"]["unit"]
        and d["group_id"]
        and not (store.group(d["group_id"]) or {}).get("search_json", {}).get("specificity_of")
    )
    group = store.group(term["group_id"])
    assert group is not None
    col = term["state_json"]["column"]
    text = "\n".join(context_lines(term, group))
    assert f"target: column {col['name']}  (scope column)" in text
    assert f"description: {col['description']}" in text
    assert f"dtype: {col['dtype']}  unit: {col['unit']}" in text
    assert f"aspect: {group['aspect']}  ontology: {group['ontology_id']}" in text
    queries = group["search_json"]["queries"]
    assert queries and f"OLS queries: {' | '.join(queries)}" in text
    assert "tier: probe/fake v1 (platt)" in text and tier_note(term) == "probe/fake v1 (platt)"
    # A column.ontology_fits item: the aspect, that the candidates are ontologies, the tier.
    onto = next(d for d in rows if d["task_id"] == "column.ontology_fits")
    ogroup = store.group(onto["group_id"])
    assert ogroup is not None
    otext = "\n".join(context_lines(onto, ogroup))
    assert f"target: column {onto['column_name']}  (scope column)" in otext
    assert f"description: {onto['state_json']['column']['description']}" in otext
    assert f"aspect: {ogroup['aspect']}  (the candidates are ontologies" in otext
    assert "ontology:" not in otext and "OLS queries" not in otext
    assert "tier: zero_shot/fake (uncalibrated: p_fit is σ(s_c), saturates near 1)" in otext
    # A site group names the site; a D24 refinement says what it refines; a rule has no tier.
    site = next(d for d in rows if d["scope"] == "site" and d["group_id"])
    stext = "\n".join(context_lines(site, store.group(site["group_id"])))
    assert f"target: site {site['site_code']}  (scope site)" in stext and "habitat:" in stext
    refinement = next(
        (
            d
            for d in rows
            if d["group_id"]
            and (store.group(d["group_id"]) or {}).get("search_json", {}).get("specificity_of")
        ),
        None,
    )
    if refinement is not None:
        rtext = "\n".join(context_lines(refinement, store.group(refinement["group_id"])))
        assert "refines:" in rtext and "(D24" in rtext
    rule = next(d for d in rows if d["method"] == "rule")
    assert "tier: none/rule" in "\n".join(context_lines(rule))
    assert tier_note({"level": "zero_shot", "method": "clm", "shape": "choice"}).startswith(
        "zero_shot/clm (uncalibrated: confidence"
    )
    assert tier_note({"method": "ols_rank", "level": "none"}).startswith("ols_rank (degraded")
    # The interactive walk prints the context before the candidates for both kinds of item.
    policy = Policy.from_config(shipped_config().policy)
    pool, thresholds = candidates(
        store,
        runs,
        policy=policy,
        results_root=tmp_path,
        is_bench_card=svc.is_bench_card,
        artifact_only=False,
    )
    sample = build_sample(
        pool, runs=[str(run["run_id"])], thresholds=thresholds, n=len(pool), min_cards=1,
        artifact_only=False,
    )  # fmt: skip
    said: list[str] = []
    done = audit_review_interactive(
        svc, store, sample, owner="alice", actor="alice",
        ask=lambda prompt: "s", say=said.append, save=lambda s: None,
    )  # fmt: skip
    assert done == {"correct": 0, "incorrect": 0, "skipped": len(pool)}
    out = "\n".join(said)
    assert f"description: {col['description']}" in out and f"unit: {col['unit']}" in out
    assert f"OLS queries: {' | '.join(queries)}" in out
    assert "(the candidates are ontologies" in out and "tier: probe/fake v1 (platt)" in out
    assert "saturates near 1" in out
    # Order within an item: header, context, answer, candidates, in that order.
    head = next(i for i, line in enumerate(said) if "term.fits on" in line)
    block = said[head : head + 12]
    tier_at = next(i for i, line in enumerate(block) if line.startswith("  tier:"))
    answer_at = next(i for i, line in enumerate(block) if line.startswith("  answer:"))
    assert 0 < tier_at < answer_at


def test_verdicts_mint_curator_labels_outside_every_fold(store: DuckDBStore) -> None:
    runs = _runs(store, 1)
    run = runs[0]
    rows = store.decisions(run["run_id"])
    rank = next(r for r in rows if r["shape"] == "rank_fit" and r["answer"] not in ("", "__none__"))
    choice = next(r for r in rows if r["shape"] == "choice" and r["answer_index"] >= 0)
    anchor = next(r for r in rows if r["answer"] == "__none__")
    yes = labels_for_verdict(rank, run, "correct", actor="alice", audit_id="A")
    no = labels_for_verdict(rank, run, "incorrect", actor="alice", audit_id="A")
    assert len(yes) == len(no) == 1
    assert (yes[0].label, yes[0].label_index, no[0].label, no[0].label_index) == ("Yes", 0, "No", 1)
    assert yes[0].option_key == rank["answer"] and yes[0].task_id == rank["task_id"]
    assert yes[0].label_source == "curator" and yes[0].weight == 1.0
    assert not yes[0].fold_eligible and not yes[0].bench_card and yes[0].origin == "audit:A"
    assert yes[0].card == run["card_name"] and yes[0].product_code.startswith("DP9.")
    ok = labels_for_verdict(choice, run, "correct", actor="alice", audit_id="A")
    assert len(ok) == 1 and ok[0].label == choice["answer"] and ok[0].option_key == ""
    assert labels_for_verdict(choice, run, "incorrect", actor="alice", audit_id="A") == []
    assert labels_for_verdict(anchor, run, "correct", actor="alice", audit_id="A") == []
    assert store.insert_labels(yes) == 1


def _item(stratum: str, verdict: str | None, version: str | None, card: str, i: int) -> AuditItem:
    return AuditItem(
        decision_id=f"d{i}",
        run_id="r",
        task_id="term.fits",
        task_key="0ccc8d141ffd30ff",
        artifact_version=version,
        level="probe" if version else "zero_shot",
        method="clm",
        outcome="proposed",
        stratum=stratum,  # type: ignore[arg-type]
        card=card,
        stat=0.9,
        verdict=verdict,  # type: ignore[arg-type]
    )


def test_audit_rows_follow_the_pass_rule_and_skip_unversioned_decisions() -> None:
    items = [
        *(_item("would_be_auto", "correct", "v1", f"c{i % 4}", i) for i in range(58)),
        *(_item("would_be_auto", "incorrect", "v1", "c0", 100 + i) for i in range(2)),
        _item("would_be_auto", "incorrect", None, "c9", 200),
        _item("would_be_auto", None, "v1", "c9", 201),
        *(_item("proposed", "correct", None, "c1", 300 + i) for i in range(8)),
        _item("proposed", "incorrect", None, "c1", 310),
        _item("anchor_abstain", "correct", None, "c2", 400),
    ]
    sample = AuditSample(
        audit_id="A", created_at="now", n=len(items), min_cards=3, seed=0, runs=["r"],
        cards=["c0", "c1", "c2", "c3", "c9"], strata={}, items=items,
    )  # fmt: skip
    rows, skipped = audit_rows(sample, reviewer="alice", risk_of=lambda t: 0.05)
    assert len(rows) == 1 and list(skipped) == ["0ccc8d141ffd30ff@none"]
    row = rows[0]
    assert isinstance(row, AuditRow) and (row.n, row.n_cards, row.n_errors) == (60, 4, 2)
    assert row.cards == ["c0", "c1", "c2", "c3"] and row.reviewer == "alice" and row.risk == 0.05
    assert row.cp95_upper == pytest.approx(0.1012, abs=1e-3) and row.passed is False  # > 0.10
    tight, _ = audit_rows(sample, reviewer="alice", risk_of=lambda t: 0.06)
    assert tight[0].passed is True  # 0.101 <= 0.12
    s = summarize(sample)
    assert s["reviewed"] == len(items) - 1
    assert s["artifact_only"] is True  # the field's default; the file records the mode drawn
    assert s["reviewed_without_artifact"] == 1 + 9 + 1  # d200, the proposed, the abstain
    assert s["strata"]["proposed"] == proposed_precision(
        [i for i in items if i.stratum == "proposed"]
    )
    assert s["strata"]["proposed"]["n"] == 9 and s["strata"]["proposed"][
        "precision"
    ] == pytest.approx(8 / 9)
    assert 0 < s["strata"]["proposed"]["lower"] < 8 / 9 < s["strata"]["proposed"]["upper"] <= 1
    assert proposed_precision([]) == {
        "n": 0,
        "n_errors": 0,
        "precision": None,
        "lower": None,
        "upper": None,
    }
    # Fewer than 50 would-be-auto decisions never pass, whatever the errors.
    small = AuditSample(**{**sample.model_dump(), "items": items[:10]})
    assert audit_rows(small, reviewer="a", risk_of=lambda t: 0.5)[0][0].passed is False


def test_cli_sample_review_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(OLS_DIR))
    monkeypatch.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    monkeypatch.setenv("MESA_CLM_DECIDER__OLS_RANK_TASKS", "[]")
    db = tmp_path / "prov.duckdb"
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{db}")
    monkeypatch.setenv("USER", "alice")
    runs = []
    for i in range(3):
        card = tmp_path / f"synth_{i}.md"
        text = CARD_TEXT.replace("DP1.10003.001 / brd_countdata", f"DP9.{i:05d}.001 / synth_{i}")
        card.write_text(text.replace("Product: DP1.10003.001", f"Product: DP9.{i:05d}.001"))
        out = tmp_path / f"run_{i}.json"
        assert (
            main(
                [
                    "annotate",
                    "--card",
                    str(card),
                    "--provider",
                    "fake",
                    "--fake-seed",
                    str(FAKE_SEED),
                    "--out",
                    str(out),
                ]
            )
            == EXIT_OK
        )
        runs.append(json.loads(out.read_text())["run_id"])
    capsys.readouterr()  # the annotate summaries
    sample_path = tmp_path / "audit.json"
    draw = [
        "audit",
        "sample",
        "--n",
        "9",
        "--min-cards",
        "3",
        "--runs",
        ",".join(r[:8] for r in runs),
        "--out",
        str(sample_path),
    ]
    # Fake zero_shot runs apply no artifact: the default draw refuses and names --all-tiers.
    assert main(draw) == EXIT_FAIL
    assert "--all-tiers" in capsys.readouterr().err and not sample_path.exists()
    assert main([*draw, "--all-tiers"]) == EXIT_OK
    summary = json.loads(capsys.readouterr().out)
    assert summary["n"] == 9 and len(summary["cards"]) == 3 and set(summary["runs"]) == set(runs)
    assert summary["artifact_only"] is False and load_sample(sample_path).artifact_only is False
    # review: a terminal is required.
    monkeypatch.setattr("mesa_clm.cli.stdin_is_terminal", lambda: False)
    assert main(["audit", "review", "--file", str(sample_path)]) == EXIT_CONFIG
    assert "needs a terminal" in capsys.readouterr().err
    assert main(["audit", "record", "--file", str(sample_path)]) == EXIT_FAIL  # nothing reviewed
    # The interactive walk with scripted answers (what the terminal would type).
    from mesa_clm.cli import _open_store, _reader_service
    from mesa_clm.config import load_config

    cfg = load_config()
    store = _open_store(cfg)
    svc = _reader_service(cfg, store)
    sample = load_sample(sample_path)
    answers = iter(["c", "i", "x", "s", "c", "q"])
    said: list[str] = []
    saved: list[int] = []
    done = audit_review_interactive(
        svc, store, sample, owner="alice", actor="alice",
        ask=lambda prompt: next(answers), say=said.append,
        save=lambda s: saved.append(len(s.reviewed)),
    )  # fmt: skip
    assert done == {"correct": 2, "incorrect": 1, "skipped": 1} and saved == [1, 2, 3]
    assert any("is not an answer" in line for line in said) and sample.reviewer == "alice"
    reviewed = sample.reviewed
    assert len(reviewed) == 3 and all(i.reviewed_at for i in reviewed)
    minted = [r for r in store.labels_for("term.fits") if r["origin"] == f"audit:{sample.audit_id}"]
    assert all(not r["fold_eligible"] and r["label_source"] == "curator" for r in minted)
    assert sum(i.labels_written for i in reviewed) == len(minted) + len(
        [
            r
            for r in store.labels_for("column.ontology_fits")
            if r["origin"] == f"audit:{sample.audit_id}"
        ]
    ) + len(
        [r for r in store.labels_for("column.aspect") if r["origin"] == f"audit:{sample.audit_id}"]
    )
    write_sample(sample_path, sample)
    capsys.readouterr()
    assert main(["audit", "record", "--file", str(sample_path)]) == EXIT_OK
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["audit_id"] == sample.audit_id and recorded["reviewer"] == "alice"
    assert recorded["rows"] == []  # fake zero_shot decisions apply no artifact: no audits row
    assert recorded["summary"]["reviewed"] == 3
    assert recorded["artifact_only"] is False
    assert recorded["summary"]["reviewed_without_artifact"] == 3
    assert recorded["report_only"].startswith("3 of 3 reviewed item(s) apply no artifact")
    assert (
        "--all-tiers" in recorded["report_only"] and "feed no audits row" in recorded["report_only"]
    )
    assert store.audits() == []
    # A sample of another owner's runs is refused.
    assert (
        main(
            [
                "--actor",
                "bob",
                "audit",
                "sample",
                "--n",
                "3",
                "--runs",
                runs[0],
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
        == EXIT_FAIL
    )
    assert (
        main(["audit", "sample", "--n", "3", "--since", "nope", "--out", str(tmp_path / "x.json")])
        == EXIT_CONFIG
    )
    assert (
        main(
            [
                "audit",
                "sample",
                "--n",
                "3",
                "--since",
                "2020-01-01",
                "--min-cards",
                "1",
                "--all-tiers",
                "--out",
                str(tmp_path / "y.json"),
            ]
        )
        == EXIT_OK
    )
    assert str(UUID(json.loads(capsys.readouterr().out)["audit_id"]))
