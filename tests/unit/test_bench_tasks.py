"""Bench tasks (mesa-anyjev ``bench/tasks``): the ``Task`` container and its folds, the 30/5
fold guards, the registry, and the neon loaders reading ``min_weight`` from the policy only
(DESIGN D9) with ``meta["masked"]`` set. Hermetic: recorded OLS fixtures, the committed
neon-avu-eval copy, DuckDB under tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mesa_clm.bench.tasks import (
    ALL_TIERS,
    MIN_HELDOUT_PER_CLASS,
    MIN_TRAIN_PER_CLASS,
    TASKS,
    Fold,
    Task,
    fold_guard,
    get_task,
    register,
)
from mesa_clm.bench.tasks.neon import (
    LICENSE,
    MASKS,
    NEON_TASKS,
    SOURCE,
    neon_task,
    tasks_from_store,
)
from mesa_clm.learn.labels import TermResolver, ingest_neon_eval, labelled_targets
from mesa_clm.ols import RecordingOLS
from mesa_clm.policy_defaults import DEFAULTS_PATH, min_weight_for
from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.tasks import TASKS as SPECS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OLS_DIR = FIXTURES / "ols"
NEON_EVAL_ROOT = FIXTURES / "neon-avu-eval"

TERM = SPECS["term.fits"]
ASPECT = SPECS["column.aspect"]


def _state(card: str, column: str, curie: str) -> dict[str, object]:
    return {
        "card": {"dataset": card},
        "scope": "column",
        "column": {"name": column},
        "aspect": "measurement",
        "candidate": {"curie": curie},
    }


def _toy(n_per_card: dict[str, int], *, positives: int = 1) -> Task:
    """A two-class toy task with ``n_per_card`` items per card, the first ``positives`` of each
    card labelled Yes; products alternate between two codes."""
    items = []
    cards: list[str] = []
    products: list[str] = []
    opts: list[str] = []
    for ci, (card, n) in enumerate(sorted(n_per_card.items())):
        for i in range(n):
            items.append((_state(card, f"col{i}", f"X:{i}"), 0 if i < positives else 1))
            cards.append(card)
            products.append(f"P{ci % 2}")
            opts.append(f"X:{i}")
    return Task("toy", TERM, items, "lic", "src", cards=cards, products=products, option_keys=opts)


# -- Task -------------------------------------------------------------------------------------------


def test_task_validates_parallel_lists_and_labels() -> None:
    items = [(_state("c", "a", "X:1"), 0), (_state("c", "b", "X:2"), 1)]
    with pytest.raises(ValueError, match="cards needs one entry per item"):
        Task("t", TERM, items, "l", "s", cards=["c"])
    with pytest.raises(ValueError, match="not an option"):
        Task("t", TERM, [(_state("c", "a", "X:1"), 2)], "l", "s")
    t = Task("t", TERM, items, "l", "s")
    assert t.task_id == "term.fits" and t.k == 2 and t.binary and t.tiers_supported == ALL_TIERS
    assert t.labels == [0, 1] and t.states[0]["column"] == {"name": "a"}
    assert t.class_counts() == {0: 1, 1: 1} and t.n_neg() == 1 and t.n_nonmodal() == 1
    assert t.subset([1]) == [items[1]]
    choice = Task(
        "a",
        ASPECT,
        [(_state("c", "a", ""), 3), (_state("c", "b", ""), 3), (_state("c", "d", ""), 1)],
        "l",
        "s",
    )
    assert not choice.binary and choice.n_neg() is None and choice.n_nonmodal() == 1
    assert choice.class_counts() == {1: 1, 3: 2}


def test_split_is_reproducible_and_disjoint() -> None:
    t = _toy({"A": 10, "B": 10})
    test1, calib1 = t.split(5, 7, seed=3)
    test2, calib2 = t.split(5, 7, seed=3)
    assert test1 == test2 and calib1 == calib2 and len(test1) == 5 and len(calib1) == 7
    assert not {id(x) for x in test1} & {id(x) for x in calib1}
    assert t.split(5, 7, seed=4)[0] != test1


def test_leave_one_card_out_yields_sorted_index_folds() -> None:
    t = _toy({"B": 3, "A": 2, "C": 4})
    folds = list(t.leave_one_card_out())
    assert [f.held_out for f in folds] == ["A", "B", "C"]
    assert all(isinstance(f, Fold) for f in folds)
    for f in folds:
        assert sorted(f.test + f.train) == list(range(len(t.items)))
        assert {t.cards[i] for i in f.test} == {f.held_out}
        assert f.held_out not in {t.cards[i] for i in f.train}
        card, test, train = f  # anyjev-style unpacking still works
        assert card == f.held_out and test == f.test and train == f.train
    assert t.class_counts(folds[2].test) == {0: 1, 1: 3}


def test_leave_one_product_out_and_missing_groups() -> None:
    t = _toy({"A": 2, "B": 2, "C": 2})  # products P0 (A, C) and P1 (B)
    folds = list(t.leave_one_product_out())
    assert [f.held_out for f in folds] == ["P0", "P1"]
    assert {t.cards[i] for i in folds[0].test} == {"A", "C"}
    bare = Task("t", TERM, [(_state("c", "a", "X:1"), 0)], "l", "s")
    with pytest.raises(ValueError, match="leave_one_card_out needs one group per item"):
        list(bare.leave_one_card_out())
    with pytest.raises(ValueError, match="leave_one_product_out"):
        list(bare.leave_one_product_out())


def test_fold_guard_applies_to_two_class_tasks_only() -> None:
    assert (MIN_TRAIN_PER_CLASS, MIN_HELDOUT_PER_CLASS) == (30, 5)
    big = _toy({"A": 40, "B": 40, "C": 40}, positives=20)
    for fold in big.leave_one_card_out():
        assert fold_guard(big, fold) is None
    small = _toy({"A": 40, "B": 40, "C": 6}, positives=20)  # C: 6 Yes, no No
    reasons = {f.held_out: fold_guard(small, f) for f in small.leave_one_card_out()}
    assert reasons["C"] == "insufficient_heldout_per_class {0: 6}"
    assert reasons["A"] == "insufficient_train_per_class {0: 26, 1: 20}"  # B 20/20 + C 6/0
    assert reasons["A"] is not None and reasons["A"].startswith("insufficient_train_per_class")
    few = _toy({"A": 40, "B": 40, "C": 40}, positives=2)
    assert fold_guard(few, next(iter(few.leave_one_card_out()))).startswith(
        "insufficient_train_per_class"
    )  # type: ignore[union-attr]
    ok_train = _toy({"A": 40, "B": 40, "C": 40}, positives=20)
    fold = next(f for f in ok_train.leave_one_card_out() if f.held_out == "A")
    thin = Fold("A", fold.test[:4] + fold.test[20:24], fold.train)  # 4 Yes / 4 No held out
    assert fold_guard(ok_train, thin) == "insufficient_heldout_per_class {0: 4, 1: 4}"
    assert fold_guard(ok_train, thin, min_heldout=4) is None
    choice = Task(
        "a",
        ASPECT,
        [(_state("c", str(i), ""), i % 6) for i in range(12)],
        "l",
        "s",
        cards=["c"] * 6 + ["d"] * 6,
    )
    assert all(fold_guard(choice, f) is None for f in choice.leave_one_card_out())


def test_register_and_get_task() -> None:
    @register("unit_toy")
    def _load() -> Task:
        return _toy({"A": 2})

    try:
        assert get_task("unit_toy").name == "toy"
        with pytest.raises(KeyError, match="unknown task"):
            get_task("no_such_task")
    finally:
        TASKS.pop("unit_toy", None)


# -- neon tasks ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> LabelStore:
    st = LabelStore(tmp_path_factory.mktemp("labels") / "labels.duckdb")
    report = ingest_neon_eval(
        st, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay"))
    )
    assert report.inserted == 934
    return st


def test_neon_tasks_read_min_weight_from_the_policy_only(store: LabelStore) -> None:
    tasks = tasks_from_store(store)
    assert (
        set(tasks)
        == set(NEON_TASKS)
        == {
            "neon_term_fits",
            "neon_ontology_fits",
            "neon_annotate",
            "neon_aspect",
            "neon_value_kind",
        }
    )
    for name, task in tasks.items():
        task_id = NEON_TASKS[name]
        assert task.task_id == task_id and task.spec is SPECS[task_id]
        assert task.meta["min_weight"] == min_weight_for(task_id)
        assert task.meta["task_key"] == SPECS[task_id].key
        ls = labelled_targets(store, task_id, min_weight=min_weight_for(task_id))
        assert task.class_counts() == ls.class_counts() == task.meta["class_counts"]
        assert (
            task.cards == ls.cards
            and task.weights == ls.weights
            and task.option_keys == ls.option_key
        )
        assert task.products == ls.leak_group and set(task.products) == {
            "DP1.10003.001",
            "DP1.10022.001",
        }
        assert len(set(task.cards)) == 7 and task.license == LICENSE and task.source == SOURCE
        assert task.meta["masked"] is True and task.meta["mask"] == MASKS[task_id]
        assert task.meta["excluded"] == {} and sum(task.meta["label_sources"].values()) == len(
            task.items
        )
        assert str(task.meta["class_counts"]) in task.notes
    assert tasks["neon_term_fits"].class_counts() == {0: 86, 1: 199}
    assert tasks["neon_ontology_fits"].class_counts() == {0: 76, 1: 114}
    assert tasks["neon_annotate"].class_counts() == {0: 63, 1: 35}
    assert tasks["neon_aspect"].meta["min_weight"] == 0.6
    assert tasks["neon_ontology_fits"].meta["mask"] == "aspect"
    assert tasks["neon_term_fits"].meta["label_sources"] == {
        "consensus_all": 16,
        "consensus_majority": 70,
        "consensus_negative": 199,
    }


def test_neon_task_honours_another_policy_file(store: LabelStore, tmp_path: Path) -> None:
    """D9: change the one source and the bench follows; at 0.6 term.fits loses every No."""
    raw = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    raw["tasks"]["term.fits"]["min_weight"] = 0.6
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    strict = neon_task(store, "neon_term_fits", "term.fits", policy_path=path)
    assert strict.meta["min_weight"] == 0.6 and strict.class_counts() == {0: 86}
    assert tasks_from_store(store, policy_path=path)["neon_term_fits"].class_counts() == {0: 86}


def test_neon_task_drops_teacher_and_bench_card_rows(tmp_path: Path) -> None:
    """D19 / D30: rows that may not enter a fold never become items; the drop is counted."""
    st = LabelStore(tmp_path / "labels.duckdb")
    key = SPECS["term.fits"].key
    base = {
        "task_id": "term.fits",
        "task_key": key,
        "label": "Yes",
        "label_index": 0,
        "state_sha256": "b" * 64,
        "state_json": _state("DP1.x.t", "a", "X:1"),
        "card": "DP1.x.t",
        "product_code": "DP1.x",
        "leak_group": "DP1.x",
    }
    rows = [
        LabelRow(
            **base,
            target_sha256="1" * 64,
            option_key="X:1",
            label_source="consensus_all",
            weight=0.8,
        ),  # type: ignore[arg-type]
        LabelRow(
            **base,
            target_sha256="2" * 64,
            option_key="X:2",
            label_source="teacher",
            weight=0.5,
            fold_eligible=False,
        ),  # type: ignore[arg-type]
        LabelRow(
            **base,
            target_sha256="3" * 64,
            option_key="X:3",
            label_source="curator",
            weight=1.0,
            bench_card=True,
        ),  # type: ignore[arg-type]
        LabelRow(
            **base,
            target_sha256="4" * 64,
            option_key="X:4",
            label_source="agent_pick",
            weight=0.0,
            fold_eligible=False,
        ),  # type: ignore[arg-type]
    ]
    assert st.insert_labels(rows) == 4
    task = neon_task(st, "neon_term_fits", "term.fits")
    assert len(task.items) == 1 and task.option_keys == ["X:1"]
    assert task.meta["excluded"] == {
        "bench_card": 1,
        "not_fold_eligible": 1,
    }  # agent_pick is below 0.5
    assert task.meta["label_sources"] == {"consensus_all": 1}
    assert "neon_annotate" not in tasks_from_store(st)  # empty tasks are dropped


# -- a curator answer never changes a pre-registered item (D30) ----------------------------------


def _curator_flip(store: LabelStore, *, tagged: bool, card: str, source: str = "curator") -> int:
    """A curator answer at weight 1.0 on an existing ``consensus_negative`` identity of
    ``card``, labelled Yes (a flip), tagged ``bench_card`` or not as the sidecar recorded it."""
    negatives = [
        r
        for r in store.labels_for("term.fits", sources=["consensus_negative"])
        if r["card"] == card
    ]
    row = negatives[0]
    return store.insert_labels(
        [
            LabelRow(
                task_id="term.fits",
                task_key=row["task_key"],
                target_sha256=row["target_sha256"],
                option_key=row["option_key"],
                label_source=source,  # type: ignore[arg-type]
                label="Yes",
                label_index=0,
                weight=1.0,
                state_sha256=row["state_sha256"],
                state_json=row["state_json"],
                card=card,
                product_code=row["product_code"],
                leak_group=row["leak_group"],
                bench_card=tagged,
                origin="override:test",
                actor="curator",
            )
        ]
    )


def _items(task: Task) -> list[tuple[str, int]]:
    return sorted(
        (key, label) for key, (_, label) in zip(task.option_keys, task.items, strict=True)
    )


@pytest.mark.parametrize("tagged", [True, False])
def test_a_curator_answer_on_a_bench_card_leaves_the_bench_items_unchanged(
    tmp_path: Path, tagged: bool
) -> None:
    """The finding the fold filter answers: a tagged 1.0 curator row used to win its identity
    in ``labelled_targets`` and then be dropped as ``bench_card``, removing the silver item
    (285 -> 284 term.fits items); an untagged one (recorded while the sidecar held no silver
    labels, so tagging failed open) replaced the silver label in the test fold. Both now leave
    the pre-registered items and labels exactly as they were, and the row is reported."""
    st = LabelStore(tmp_path / "labels.duckdb")
    assert (
        ingest_neon_eval(
            st, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay"))
        ).inserted
        == 934
    )
    before = neon_task(st, "neon_term_fits", "term.fits")
    assert len(before.items) == 285 and before.class_counts() == {0: 86, 1: 199}
    card = "DP1.10022.001.bet_expertTaxonomistIDProcessed"
    assert _curator_flip(st, tagged=tagged, card=card) == 1
    assert _curator_flip(st, tagged=tagged, card=card, source="curator_implicit") == 1
    after = neon_task(st, "neon_term_fits", "term.fits")
    assert _items(after) == _items(before)
    assert after.class_counts() == {0: 86, 1: 199} and after.cards == before.cards
    assert after.meta["excluded"] == {"bench_card": 2}
    assert after.meta["label_sources"] == before.meta["label_sources"]
    # Without the fold filter the curator rows still compete (a fitter excludes them itself).
    competing = labelled_targets(st, "term.fits", min_weight=0.5)
    assert {"curator", "curator_implicit"} & set(competing.sources)


def test_curator_rows_on_other_cards_still_compete(tmp_path: Path) -> None:
    """A curator answer on a card that is not a bench card is a fold-eligible label: the fold
    filter drops bench-card and not-fold-eligible rows only."""
    from mesa_clm.learn.labels import BENCH_CARDS, fold_exclusion, is_bench_card

    assert len(BENCH_CARDS) == 7 and is_bench_card("") and is_bench_card("x", ["x"])
    assert not is_bench_card("DP1.00004.001.BP_30min")
    row = {"fold_eligible": True, "bench_card": False, "label_source": "curator"}
    assert fold_exclusion({**row, "card": "DP1.00004.001.BP_30min"}) is None
    assert fold_exclusion({**row, "card": "DP1.10003.001.brd_countdata"}) == "bench_card"
    assert fold_exclusion({**row, "card": ""}) == "bench_card"
    assert fold_exclusion({**row, "fold_eligible": False, "card": "c"}) == "not_fold_eligible"
    silver = {**row, "label_source": "consensus_all", "card": "DP1.10003.001.brd_countdata"}
    assert fold_exclusion(silver) is None


def test_the_bench_cards_are_the_snapshots_and_the_fixtures() -> None:
    """The fixed list equals the cards of the frozen snapshot and the committed eval copy."""
    import duckdb

    from mesa_clm.learn.labels import BENCH_CARDS

    snapshot = Path(__file__).resolve().parents[2] / "bench" / "snapshots" / "2026-09-29.parquet"
    con = duckdb.connect()
    try:
        cards = {
            str(r[0])
            for r in con.execute(
                "SELECT DISTINCT card FROM read_parquet(?)", [str(snapshot)]
            ).fetchall()
        }
    finally:
        con.close()
    assert cards == set(BENCH_CARDS)
    assert {p.stem for p in (NEON_EVAL_ROOT / "cards").glob("*.md")} == set(BENCH_CARDS)


def test_the_live_smoke_card_is_not_a_bench_card() -> None:
    """Live runs before the M2 cells exist use non-bench cards only (DESIGN, "G1 freeze"); the
    engine test's card must stay outside the fixed list."""
    from mesa_clm.learn.labels import is_bench_card

    srer = Path(__file__).resolve().parents[1] / "fixtures" / "cards-srer"
    cards = sorted(p.stem for p in srer.glob("*.md"))
    assert cards == ["DP1.00004.001.BP_30min"]
    assert not any(is_bench_card(c) for c in cards)
