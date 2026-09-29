"""No-model baselines (plan §5.4, X2): the lookup key, the deterministic lookup and
``lookup_prob``, the novel-key and leave-one-product-out controls, the baselines cell and
results file, and the M0 acceptance reproduction: fixture ingestion -> lookup 0.772 / 0.800,
LOPO 0.723 / 0.721, novel 159 (45 Yes / 114 No) / 99 (33 / 66), majority 0.717 / 0.667, and
every number of the committed ``bench/results/2026-09-29/baselines.json``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm.bench.baselines import (
    BASELINE_FRAMING,
    BASELINE_TIER,
    LAPLACE_ALPHA,
    Lookup,
    baselines_cell,
    evaluate_lookup,
    guard_skipped_folds,
    lookup_key,
    run_baselines,
    task_keys,
)
from mesa_clm.bench.results import (
    FORMAT,
    BenchResults,
    cell_key,
    cite,
    labels_content_sha256,
    load_results,
    markdown_table,
    results_path,
    write_results,
)
from mesa_clm.bench.tasks.base import Task
from mesa_clm.bench.tasks.neon import tasks_from_store
from mesa_clm.learn.labels import TermResolver, ingest_neon_eval, snapshot
from mesa_clm.ols import RecordingOLS
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.tasks import TASKS as SPECS

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
OLS_DIR = FIXTURES / "ols"
NEON_EVAL_ROOT = FIXTURES / "neon-avu-eval"
COMMITTED = REPO / "bench" / "results" / "2026-09-29" / "baselines.json"

TERM = SPECS["term.fits"]


def _state(card: str, column: str | None, curie: str, *, scope: str = "column") -> dict[str, Any]:
    st: dict[str, Any] = {"card": {"dataset": card}, "scope": scope, "aspect": "measurement"}
    if column is not None:
        st["column"] = {"name": column}
    st["candidate"] = {"curie": curie}
    return st


def _toy() -> Task:
    """Three cards sharing key (column a, X:1): A says Yes, B says No, C says Yes; a key seen
    once (column b) and a key only C has (column z)."""
    rows = [
        ("A", "a", "X:1", 0),
        ("A", "b", "X:2", 1),
        ("B", "a", "X:1", 1),
        ("B", "b", "X:2", 1),
        ("C", "a", "X:1", 0),
        ("C", "z", "X:9", 1),
        ("C", "b", "X:2", 0),
    ]
    return Task(
        "toy",
        TERM,
        [(_state(c, col, cur), lbl) for c, col, cur, lbl in rows],
        "l",
        "s",
        cards=[r[0] for r in rows],
        products=["P0", "P0", "P1", "P1", "P0", "P0", "P0"],
        option_keys=[r[2] for r in rows],
    )


# -- keys ------------------------------------------------------------------------------------------------


def test_lookup_key_is_task_scope_target_option() -> None:
    assert lookup_key("term.fits", _state("D", "siteID", "ENVO:1"), "ENVO:1") == (
        "term.fits",
        "column",
        "siteID",
        "ENVO:1",
    )
    assert lookup_key("term.fits", _state("D", None, "PCO:1", scope="dataset"), "PCO:1") == (
        "term.fits",
        "dataset",
        "",
        "PCO:1",
    )
    site = {"card": {"dataset": "D"}, "scope": "site", "site": {"code": "HARV"}}
    assert lookup_key("term.fits", site, "ENVO:2") == ("term.fits", "site", "HARV", "ENVO:2")
    # no scope in the state: the task's scope (column.ontology_fits states carry none)
    ont = {
        "card": {"dataset": "D"},
        "column": {"name": "c"},
        "aspect": "unit",
        "ontology": {"id": "uo"},
    }
    assert lookup_key("column.ontology_fits", ont, "uo") == (
        "column.ontology_fits",
        "column",
        "c",
        "uo",
    )
    avu = {"card": {"dataset": "D"}, "column": {"name": "c"}, "term": {"curie": "X:1"}}
    assert lookup_key("avu.value_kind", avu) == ("avu.value_kind", "avu", "c", "")
    with pytest.raises(KeyError):
        lookup_key("term.fits", {"card": {"dataset": "D"}, "scope": "column"}, "X")
    with pytest.raises(ValueError, match="unknown scope"):
        lookup_key("term.fits", {"card": {"dataset": "D"}, "scope": "galaxy"}, "X")
    toy = _toy()
    assert task_keys(toy)[0] == ("term.fits", "column", "a", "X:1") and len(task_keys(toy)) == 7


# -- Lookup ------------------------------------------------------------------------------------------------


def test_lookup_majority_ties_and_laplace() -> None:
    toy = _toy()
    keys = task_keys(toy)
    # hold out C: training A (Yes) and B (No) tie on (a, X:1) -> alphabetically first card, A -> Yes
    lk = Lookup.fit(toy, [0, 1, 2, 3], keys)
    assert lk.n_train == 4 and lk.majority == 1 and lk.prior == {0: 1, 1: 3}
    assert lk.seen(keys[4]) and lk.tied(keys[4]) and lk.conflicting(keys[4])
    assert lk.predict(keys[4]) == 0
    assert not lk.seen(keys[5]) and lk.predict(keys[5]) == 1  # unseen -> training majority
    np.testing.assert_allclose(lk.prob(keys[4]), [(1 + 1) / (2 + 2), (1 + 1) / (2 + 2)])
    np.testing.assert_allclose(
        lk.prob(keys[1]), [(0 + 1) / (2 + 2), (2 + 1) / (2 + 2)]
    )  # (b, X:2): No, No
    np.testing.assert_allclose(lk.prob(keys[5]), [0.25, 0.75])  # unseen -> empirical prior
    np.testing.assert_allclose(lk.prob(keys[4], alpha=0.5), [0.5, 0.5])
    # hold out A: B (No) and C (Yes) tie on both keys -> B first -> No, whatever the item order
    lk2 = Lookup.fit(toy, [6, 5, 4, 3, 2], keys)
    assert lk2.predict(keys[0]) == 1 and lk2.tied(keys[1]) and lk2.predict(keys[1]) == 1
    assert lk2.majority == 1 and lk2.prior == {0: 2, 1: 3}
    # a card-free task still works (one card '')
    bare = Task("b", TERM, toy.items[:2], "l", "s", option_keys=toy.option_keys[:2])
    assert Lookup.fit(bare, [0, 1]).predict(task_keys(bare)[0]) == 0
    empty = Lookup(2)
    assert empty.majority == 0 and empty.predict(("term.fits", "column", "x", "y")) == 0
    np.testing.assert_allclose(empty.prior_probs, [0.5, 0.5])
    assert LAPLACE_ALPHA == 1.0


def test_evaluate_lookup_pools_in_item_order_and_flags_novel_keys() -> None:
    toy = _toy()
    ev = evaluate_lookup(toy)
    assert ev.n == 7 and ev.n_folds == 3 and ev.folds == ["A", "B", "C"]
    assert ev.idx.tolist() == list(range(7)) and ev.labels.tolist() == toy.labels
    assert ev.cards == toy.cards and ev.fold == toy.cards
    # A held out: (a,X:1) B No / C Yes -> tie -> B -> No (wrong); (b,X:2) B No / C Yes -> tie -> B -> No (right)
    # B held out: (a,X:1) A Yes / C Yes -> Yes (wrong); (b,X:2) A No / C Yes -> tie -> A -> No (right)
    # C held out: (a,X:1) tie -> A Yes (right); (z,X:9) novel -> majority No (right); (b,X:2) No,No -> No (wrong)
    assert ev.pred.tolist() == [1, 1, 0, 1, 0, 1, 1]
    assert ev.novel.tolist() == [False, False, False, False, False, True, False]
    # the training majority is No except when B is held out (A + C: 3 Yes / 2 No)
    assert ev.majority_pred.tolist() == [1, 1, 0, 0, 1, 1, 1]
    assert ev.acc() == pytest.approx(4 / 7) and ev.majority_acc() == pytest.approx(2 / 7)
    assert ev.probs.shape == (7, 2) and np.allclose(ev.probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(ev.probs[5], [1 / 4, 3 / 4])  # C's fold prior: 1 Yes / 4 items
    nov = ev.subset(ev.novel)
    assert nov.n == 1 and nov.cards == ["C"] and nov.label_list == [1]
    lopo = evaluate_lookup(toy, toy.leave_one_product_out())
    assert lopo.folds == ["P0", "P1"] and lopo.n == 7
    assert evaluate_lookup(toy, []).n == 0


# -- reproduction -----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ingested(tmp_path_factory: pytest.TempPathFactory) -> tuple[LabelStore, str]:
    tmp = tmp_path_factory.mktemp("labels")
    store = LabelStore(tmp / "labels.duckdb")
    report = ingest_neon_eval(
        store, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay"))
    )
    assert report.inserted == 934
    return store, snapshot(store, tmp / "snapshot.parquet")


@pytest.fixture(scope="module")
def fresh(ingested: tuple[LabelStore, str]) -> BenchResults:
    store, sha = ingested
    return run_baselines(store, labels_sha256=sha, date="2026-09-29")


def test_lookup_controls_reproduce_the_planning_numbers(fresh: BenchResults) -> None:
    """Plan §5.4 / RESEARCH.md: LOCO 0.772 / 0.800, LOPO 0.723 / 0.721, novel-key 159 (45/114)
    and 99 (33/66), majority 0.717 / 0.667 on those subsets. Recomputed, not copied."""
    term = fresh.cells[cell_key("neon_term_fits", BASELINE_TIER, BASELINE_FRAMING)]
    ont = fresh.cells[cell_key("neon_ontology_fits", BASELINE_TIER, BASELINE_FRAMING)]
    assert term.baselines is not None and ont.baselines is not None
    assert (
        term.counts.n == 285
        and term.counts.n_neg == 199
        and term.counts.class_counts == {0: 86, 1: 199}
    )
    assert (
        ont.counts.n == 190
        and ont.counts.n_neg == 114
        and ont.counts.class_counts == {0: 76, 1: 114}
    )
    assert (
        term.baselines.lookup_acc == pytest.approx(220 / 285)
        and round(term.baselines.lookup_acc, 3) == 0.772
    )
    assert (
        ont.baselines.lookup_acc == pytest.approx(152 / 190)
        and round(ont.baselines.lookup_acc, 3) == 0.800
    )
    assert (
        term.baselines.lopo.acc == pytest.approx(206 / 285)
        and round(term.baselines.lopo.acc or 0, 3) == 0.723
    )
    assert (
        ont.baselines.lopo.acc == pytest.approx(137 / 190)
        and round(ont.baselines.lopo.acc or 0, 3) == 0.721
    )
    assert term.baselines.lopo.n_folds == 2 and ont.baselines.lopo.n_folds == 2
    assert term.baselines.novel_key.n == 159 and term.baselines.novel_key.class_counts == {
        0: 45,
        1: 114,
    }
    assert ont.baselines.novel_key.n == 99 and ont.baselines.novel_key.class_counts == {
        0: 33,
        1: 66,
    }
    assert term.baselines.novel_key.majority_acc == pytest.approx(114 / 159)
    assert ont.baselines.novel_key.majority_acc == pytest.approx(66 / 99)
    assert round(term.baselines.novel_key.majority_acc or 0, 3) == 0.717
    assert round(ont.baselines.novel_key.majority_acc or 0, 3) == 0.667
    assert term.baselines.majority_acc == pytest.approx(
        199 / 285
    ) and ont.baselines.majority_acc == pytest.approx(114 / 190)
    # lookup_prob differs from the deterministic lookup exactly where a key is tied (Laplace argmax -> Yes)
    assert term.baselines.lookup_prob_acc == pytest.approx(215 / 285)
    assert ont.baselines.lookup_prob_acc == pytest.approx(153 / 190)
    assert term.diagnostics["n_novel"] == 159 and term.diagnostics["n_tied_keys"] == 6
    assert (
        term.diagnostics["n_conflicting_keys"] == 30 and ont.diagnostics["n_conflicting_keys"] == 8
    )
    # on novel keys lookup_prob is the fold prior: its NLL is a prior cross-entropy, AUROC near 0.5
    assert term.baselines.novel_key.nll is not None and 0.55 < term.baselines.novel_key.nll < 0.65
    assert term.baselines.novel_key.auroc is not None and 0.3 < term.baselines.novel_key.auroc < 0.6
    assert (
        term.baselines.novel_key.auroc_ci is not None
        and term.baselines.novel_key.auroc_ci.n_clusters == 7
    )
    assert term.baselines.beats_lookup_novel is None


def test_baselines_cell_shape(fresh: BenchResults) -> None:
    assert fresh.format == FORMAT and fresh.date == "2026-09-29" and fresh.name == "baselines"
    assert set(fresh.cells) == {
        f"{t}.baseline.lookup_prob"
        for t in (
            "neon_term_fits",
            "neon_ontology_fits",
            "neon_annotate",
            "neon_aspect",
            "neon_value_kind",
        )
    }
    for key, cell in fresh.cells.items():
        assert key == cell.key == cell_key(cell.task, "baseline", "lookup_prob")
        assert cell.tier == "baseline" and cell.framing == "lookup_prob"
        assert cell.question_key is None and cell.fingerprint is None and cell.feature_spec is None
        assert cell.labels_sha256 == fresh.labels_sha256 and len(cell.labels_sha256) == 64
        assert cell.labels_content_sha256 == fresh.labels_content_sha256
        assert cell.loco and cell.pre_registered and not cell.exploratory and not cell.servable
        assert cell.selection == "none" and not cell.teacher and not cell.teacher_in_test
        assert cell.masked is True and cell.n_folds == 7 and cell.skipped_folds == {}
        assert cell.metrics is not None and cell.baselines is not None
        assert cell.counts.n_nonmodal == cell.counts.n - max(cell.counts.class_counts.values())
        assert set(cell.metrics.threshold_cp) == {"0.05", "0.10"}
        assert 0.0 <= cell.metrics.ece <= 1.0 and 0.0 <= cell.metrics.cov_at_5 <= 1.0
        assert cell.baselines.lookup_key.startswith("(task_id, scope, target")
        if cell.task_id in ("term.fits", "column.ontology_fits", "column.annotate"):
            assert cell.metrics.auroc is not None and cell.metrics.auroc_ci is not None
            assert cell.metrics.auroc_ci.lower <= cell.metrics.auroc <= cell.metrics.auroc_ci.upper  # type: ignore[operator]
            assert cell.metrics.auroc_ci.B == 2000 and cell.metrics.auroc_ci.seed == 0
        else:
            assert (
                cell.metrics.auroc is None
                and cell.metrics.auroc_ci is None
                and cell.counts.n_neg is None
            )
    annotate = fresh.cells["neon_annotate.baseline.lookup_prob"]
    assert len(annotate.guard_skipped_folds) == 6  # only one fold passes 30/5 (plan §4.2 Q1)
    assert fresh.cells["neon_term_fits.baseline.lookup_prob"].guard_skipped_folds == {}
    assert cell.mask == ("aspect" if cell.task_id == "column.ontology_fits" else None)


def _walk(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


def test_committed_baselines_file_is_reproduced_from_the_fixtures(fresh: BenchResults) -> None:
    """``bench/results/2026-09-29/baselines.json`` was produced by this code from the fixture
    ingestion; re-deriving it gives the same rows (``labels_content_sha256``) and the same
    numbers. ``labels_sha256`` and ``environment`` legitimately differ per run (D30)."""
    assert COMMITTED.exists(), "bench/results/2026-09-29/baselines.json is committed with M0"
    committed = load_results(COMMITTED)
    assert committed.labels_content_sha256 == fresh.labels_content_sha256
    assert committed.mesa_clm == fresh.mesa_clm and committed.date == fresh.date
    assert set(committed.cells) == set(fresh.cells)
    for key, cell in committed.cells.items():
        a = cell.model_dump(by_alias=True, exclude={"labels_sha256"})
        b = fresh.cells[key].model_dump(by_alias=True, exclude={"labels_sha256"})
        _walk(a, b, key)


def test_write_load_round_trip_and_markdown(fresh: BenchResults, tmp_path: Path) -> None:
    path = write_results(fresh, tmp_path)
    assert (
        path
        == results_path(tmp_path, "2026-09-29", "baselines")
        == tmp_path / "2026-09-29" / "baselines.json"
    )
    assert path.with_suffix(".md").exists()
    back = load_results(path)
    assert back == fresh
    text = path.read_text(encoding="utf-8")
    assert '"cov@5%"' in text and "cov_at_5" not in text and text.endswith("}\n")
    md = markdown_table(fresh)
    assert md == path.with_suffix(".md").read_text(encoding="utf-8")
    assert "| neon_term_fits.baseline.lookup_prob | 285 | 199 |" in md and "0.772" in md
    assert "bench/results/2026-09-29/baselines.json" in md
    assert cite(
        "bench/results/2026-09-29/baselines.json", "neon_term_fits", "baseline", "lookup_prob"
    ) == ("bench/results/2026-09-29/baselines.json#neon_term_fits.baseline.lookup_prob")


def test_labels_content_hash_is_ingestion_independent(
    ingested: tuple[LabelStore, str], tmp_path: Path
) -> None:
    store, sha = ingested
    other = LabelStore(tmp_path / "other.duckdb")
    ingest_neon_eval(other, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay")))
    assert labels_content_sha256(other) == labels_content_sha256(store)
    assert snapshot(other, tmp_path / "s.parquet") != sha  # while D30's snapshot hash differs
    assert labels_content_sha256(LabelStore(tmp_path / "none.duckdb")) == labels_content_sha256(
        LabelStore(tmp_path / "none2.duckdb")
    )


def test_baselines_cell_on_a_toy_task_and_guards(ingested: tuple[LabelStore, str]) -> None:
    toy = _toy()
    toy.meta.update({"min_weight": 0.5, "masked": True, "mask": None, "label_sources": {"x": 7}})
    cell = baselines_cell(toy, labels_sha256="0" * 64, B=50, seed=1)
    assert cell.counts.n == 7 and cell.n_folds == 3 and cell.baselines is not None
    assert cell.baselines.lookup_acc == pytest.approx(4 / 7) and cell.baselines.novel_key.n == 1
    assert cell.baselines.lopo.n_folds == 2 and cell.metrics is not None
    assert (
        cell.metrics.auroc_ci is not None
        and cell.metrics.auroc_ci.B == 50
        and cell.metrics.auroc_ci.seed == 1
    )
    assert cell.baselines.novel_key.auroc_ci is None  # one card in the novel subset: no cluster CI
    assert set(guard_skipped_folds(toy)) == {"A", "B", "C"}
    single = Task(
        "s", TERM, toy.items[:2], "l", "s", cards=["A", "A"], option_keys=toy.option_keys[:2]
    )
    single.meta.update({"min_weight": 0.5})
    lone = baselines_cell(single, labels_sha256="0" * 64, B=10)
    assert (
        lone.baselines is not None
        and lone.baselines.lopo.n_folds == 0
        and lone.baselines.lopo.acc is None
    )
    assert lone.metrics is not None and lone.metrics.auroc_ci is None
    store, _ = ingested
    tasks = tasks_from_store(store)
    assert math.isclose(
        baselines_cell(tasks["neon_term_fits"], labels_sha256="0" * 64, B=20).baselines.lookup_acc,  # type: ignore[union-attr]
        220 / 285,
    )
