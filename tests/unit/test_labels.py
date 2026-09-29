"""Labels: the ``mesa_clm.labels`` row and store (DESIGN D11), the neon-avu-eval ingestion
(plan §5.1 counts, the 285/303 reconciliation), the D9 class counts at the policy
``min_weight``, snapshots (D30) and stats. Ported from mesa-anyjev
``tests/test_labels_and_bench.py`` and ``tests/test_provenance_duckdb.py`` (the labels half).
Hermetic: recorded OLS fixtures, the committed neon-avu-eval copy, DuckDB under tmp_path."""

from __future__ import annotations

import fcntl
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pytest
from pydantic import ValidationError

from mesa_clm.learn.labels import (
    INGESTED_TASKS,
    NOT_FOLD_ELIGIBLE,
    WEIGHTS,
    IngestReport,
    TermResolver,
    check_eval_root,
    ingest_neon_eval,
    labelled_targets,
    product_code_of,
    snapshot,
    stats,
)
from mesa_clm.ols import RecordingOLS
from mesa_clm.policy_defaults import min_weight_for
from mesa_clm.provenance.labels import (
    IDENTITY_COLUMNS,
    LABEL_SOURCES,
    LABELS_DDL,
    TABLE,
    LabelRow,
    LabelStore,
    lock_path_for,
)
from mesa_clm.tasks import TASKS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OLS_DIR = FIXTURES / "ols"
NEON_EVAL_ROOT = FIXTURES / "neon-avu-eval"

TERM_KEY = TASKS["term.fits"].key

# The five GAZ CURIEs EBI OLS returns with ``isRoot: true`` (dropped by ``to_candidates``), and
# how many (card, CURIE) pairs of validated.json name them: 303 pairs - 18 = 285 term.fits rows.
GAZ_MISSING = {
    "GAZ:00002518": 6,  # State of Arizona
    "GAZ:00002537": 6,  # Commonwealth of Massachusetts
    "GAZ:00162932": 3,  # Harvard Forest
    "GAZ:00000448": 2,  # geographic location
    "GAZ:00057784": 1,  # Lechuguilla Desert
}


def _row(**over: Any) -> LabelRow:
    base: dict[str, Any] = {
        "task_id": "term.fits",
        "task_key": TERM_KEY,
        "target_sha256": "a" * 64,
        "option_key": "PATO:0000040",
        "label_source": "curator",
        "label": "Yes",
        "label_index": 0,
        "weight": 1.0,
        "state_sha256": "b" * 64,
        "state_json": {"card": {"dataset": "DP1.x"}},
        "card": "DP1.x",
    }
    base.update(over)
    return LabelRow(**base)


# -- LabelRow ---------------------------------------------------------------------------------------


def test_label_row_defaults_and_validation() -> None:
    row = _row()
    assert row.option_key == "PATO:0000040" and row.fold_eligible and not row.bench_card
    assert row.product_code == "" and row.leak_group == "" and row.origin == "" and row.actor == ""
    assert row.created_at.tzinfo is not None
    assert list(row.model_dump()) == [
        "label_id",
        "task_id",
        "task_key",
        "target_sha256",
        "option_key",
        "label_source",
        "label",
        "label_index",
        "weight",
        "state_sha256",
        "state_json",
        "card",
        "product_code",
        "leak_group",
        "fold_eligible",
        "bench_card",
        "origin",
        "actor",
        "created_at",
    ]
    for bad in (
        {"task_key": "xyz"},
        {"target_sha256": "a" * 63},
        {"state_sha256": "G" * 64},
        {"weight": 1.5},
        {"label_index": -1},
        {"label": ""},
        {"task_id": " "},
        {"label_source": "hosted_jev"},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            _row(**bad)


def test_ddl_is_one_constant_with_sentinels_and_the_identity_unique() -> None:
    ddl = "\n".join(LABELS_DDL)
    assert LABELS_DDL[0] == "CREATE SCHEMA IF NOT EXISTS mesa_clm"
    assert TABLE == "mesa_clm.labels" and f"CREATE TABLE IF NOT EXISTS {TABLE}" in ddl
    for sentinel in ("option_key", "card", "product_code", "leak_group", "origin", "actor"):
        assert f"{sentinel} TEXT NOT NULL DEFAULT ''" in ddl, sentinel
    assert "NULLS NOT DISTINCT" not in ddl  # DuckDB 1.5.5 rejects it (D11)
    assert "UNIQUE (task_key, target_sha256, option_key, label_source)" in ddl
    assert IDENTITY_COLUMNS == ("task_key", "target_sha256", "option_key", "label_source")
    assert all(f"'{s}'" in ddl for s in LABEL_SOURCES)
    assert set(LABEL_SOURCES) == set(WEIGHTS)


# -- LabelStore -------------------------------------------------------------------------------------


def test_store_round_trip_ignores_duplicates_and_enforces_checks(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.duckdb")
    assert store.count() == 0 and store.labels_for("term.fits") == [] and store.columns() == []
    a = _row()
    dup = _row(label_id=uuid4(), label="No", label_index=1)  # same identity, first wins
    other_source = _row(label_source="consensus_negative", weight=0.5, label="No", label_index=1)
    assert store.insert_labels([a, dup, other_source]) == 2
    assert store.insert_labels([a]) == 0 and store.insert_labels([]) == 0
    rows = store.labels_for("term.fits")
    assert [r["label_source"] for r in rows] == ["consensus_negative", "curator"]
    assert rows[1]["label"] == "Yes" and rows[1]["state_json"] == {"card": {"dataset": "DP1.x"}}
    assert rows[1]["weight"] == 1.0 and isinstance(rows[1]["fold_eligible"], bool)
    assert store.labels_for("term.fits", min_weight=0.6) == [rows[1]]
    assert store.labels_for("term.fits", exclude_cards=["DP1.x"]) == []
    assert store.labels_for("term.fits", sources=["curator"]) == [rows[1]]
    assert store.labels_for("term.fits", sources=[]) == []
    assert store.labels_for("column.annotate") == []  # another task key
    with pytest.raises(KeyError):
        store.labels_for("no.such.task")
    assert store.count() == 2 and store.task_ids() == ["term.fits"]
    counts = store.counts("term.fits")
    assert [(c["label_source"], c["n"]) for c in counts] == [
        ("consensus_negative", 1),
        ("curator", 1),
    ]
    names = [name for name, _ in store.columns()]
    assert names == list(LabelRow.model_fields)
    nullable = [name for name, is_null in store.columns() if is_null]
    assert nullable == []  # every column NOT NULL: sentinels, not NULLs (D11)
    # CHECK constraints hold in DuckDB 1.5.x; bypass pydantic to prove the DDL enforces them.
    with store.connect() as con, pytest.raises(duckdb.ConstraintException):
        con.execute(
            f"INSERT INTO {TABLE} (label_id, task_id, task_key, target_sha256, label_source, label, "  # noqa: S608
            "label_index, weight, state_sha256, state_json, created_at) VALUES "
            "('x', 'term.fits', ?, ?, 'hosted_jev', 'Yes', 0, 1.0, ?, '{}', now())",
            [TERM_KEY, "a" * 64, "b" * 64],
        )


def test_store_holds_no_connection_between_operations(tmp_path: Path) -> None:
    """D11: a DuckDB file has one writer; another process must be able to open it between
    our operations."""
    path = tmp_path / "labels.duckdb"
    store = LabelStore(path)
    store.insert_labels([_row()])
    con = duckdb.connect(str(path))  # would raise "Could not set lock" if we still held it
    assert con.execute(f"SELECT count(*) FROM {TABLE}").fetchone() == (1,)  # noqa: S608
    con.close()
    # The D11 lock, <dir>/locks/provenance.lock, is the default: the M1 sidecar store over the
    # same file derives the same path, so the two stores serialise on one flock.
    assert store.lock_path == tmp_path / "locks" / "provenance.lock" and store.lock_path.exists()
    assert store.lock_path == lock_path_for(path) == lock_path_for(str(path))
    assert lock_path_for("~/.mesa/clm/provenance.duckdb") == (
        Path("~/.mesa/clm/locks/provenance.lock").expanduser()
    )
    custom = LabelStore(path, lock_path=tmp_path / "other.lock")
    assert custom.count() == 1 and (tmp_path / "other.lock").exists()


def test_store_operations_wait_for_the_flock(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.duckdb")
    store.ensure_schema()
    done = threading.Event()
    result: list[int] = []

    def writer() -> None:
        result.append(store.insert_labels([_row()]))
        done.set()

    with store.lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        t = threading.Thread(target=writer)
        t.start()
        time.sleep(0.2)
        assert not done.is_set() and result == []  # blocked behind our lock
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    assert done.wait(10) and result == [1]
    t.join()


def test_store_write_is_one_transaction(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.duckdb")
    good = _row()
    bad = _row(target_sha256="c" * 64)
    bad.__dict__["weight"] = 2.0  # past pydantic: only the CHECK sees it
    with pytest.raises(duckdb.ConstraintException):
        store.insert_labels([good, bad])
    assert store.count() == 0  # rolled back together


# -- ingestion ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ingested(tmp_path_factory: pytest.TempPathFactory) -> tuple[LabelStore, IngestReport]:
    store = LabelStore(tmp_path_factory.mktemp("labels") / "labels.duckdb")
    report = ingest_neon_eval(
        store, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay"))
    )
    return store, report


def test_weights_and_fold_eligibility() -> None:
    assert WEIGHTS == {
        "curator": 1.0,
        "curator_implicit": 0.7,
        "agent_pick": 0.0,
        "consensus_all": 0.8,
        "consensus_majority": 0.6,
        "consensus_negative": 0.5,
        "teacher": 0.5,
        "teacher_implicit": 0.3,
        "gold": 1.0,
    }
    assert {"agent_pick", "teacher", "teacher_implicit"} == NOT_FOLD_ELIGIBLE
    assert set(INGESTED_TASKS) == {
        "term.fits",
        "column.annotate",
        "column.aspect",
        "column.ontology_fits",
        "avu.value_kind",
    }


def test_ingest_counts_match_plan_5_1(ingested: tuple[LabelStore, IngestReport]) -> None:
    store, report = ingested
    per = report.per_task
    assert per["term.fits"] == {
        "consensus_all": 16,
        "consensus_majority": 70,
        "consensus_negative": 199,
    }
    assert per["column.ontology_fits"] == {"consensus_majority": 76, "consensus_negative": 114}
    assert per["column.annotate"] == {"consensus_majority": 63, "consensus_negative": 35}
    assert per["column.aspect"] == {"consensus_majority": 60}
    # value_kind rows are gathered per valid column-linked AVU (450) and collapse onto 301
    # distinct (column, aspect, term) identities before the insert; the first AVU wins (as
    # mesa-anyjev's INSERT OR IGNORE did), so the report counts what the store holds: the 82
    # class-3 rows are the 0.5 negatives (D9), 149 rows collapsed, 28 of them naming another
    # value kind than the winner.
    assert per["avu.value_kind"] == {"consensus_majority": 219, "consensus_negative": 82}
    assert report.inserted == 934 and report.skipped_existing == 0
    assert sum(sum(c.values()) for c in per.values()) == 934
    assert set(per) == set(INGESTED_TASKS)  # no column.ontology, no avu.keep (D25)
    assert report.source_ref.startswith("neon-avu-eval/results/validated.json@")
    assert report.skipped == {"collapsed_identity": 149} and report.conflicts == 28
    assert report.summary()["skipped"] == {"collapsed_identity": 149}
    st = stats(store)
    assert st["n_labels"] == 934
    assert st["tasks"]["term.fits"]["labels"] == {"No": 199, "Yes": 86}
    assert st["tasks"]["term.fits"]["weights"] == {"0.5": 199, "0.6": 70, "0.8": 16}
    assert st["tasks"]["avu.value_kind"]["sources"] == {
        "consensus_majority": 219,
        "consensus_negative": 82,
    }
    assert st["tasks"]["avu.value_kind"]["labels"]["the most frequent data value"] == 82
    assert st["tasks"]["column.ontology_fits"]["n"] == 190
    assert store.task_ids() == sorted(INGESTED_TASKS)


def test_285_of_303_pairs_the_gap_is_gaz_root_terms(
    ingested: tuple[LabelStore, IngestReport],
) -> None:
    """RESEARCH.md counts 303 unique valid (card, CURIE) pairs; labels.duckdb has 285 term.fits
    rows. The 18 missing pairs are GAZ terms OLS flags ``isRoot`` (no GAZ hierarchy on EBI)."""
    _, report = ingested
    assert report.terms_resolved == 285
    assert len(report.terms_missing) == 18 and report.terms_resolved + 18 == 303
    assert {c: report.terms_missing.count(c) for c in set(report.terms_missing)} == GAZ_MISSING
    assert all(c.startswith("GAZ:") for c in report.terms_missing)
    assert report.summary()["terms_missing_curies"] == sorted(GAZ_MISSING)


def test_ingest_is_idempotent(ingested: tuple[LabelStore, IngestReport]) -> None:
    store, first = ingested
    again = ingest_neon_eval(
        store, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay"))
    )
    assert again.inserted == 0 and again.skipped_existing == 934
    assert again.per_task == first.per_task and again.skipped == first.skipped
    assert store.count() == 934


def test_check_eval_root_names_what_is_missing(tmp_path: Path) -> None:
    """The rule `labels ingest-neon-eval` and the doctor share: ``results/validated.json`` and
    ``cards/`` must exist, else ``FileNotFoundError`` (a usage error at the CLI, never a
    traceback)."""
    assert check_eval_root(NEON_EVAL_ROOT) == NEON_EVAL_ROOT
    with pytest.raises(FileNotFoundError, match=r"results/validated\.json, cards/"):
        check_eval_root(tmp_path / "nope")
    (tmp_path / "cards").mkdir()
    with pytest.raises(FileNotFoundError, match=r"missing results/validated.json\)"):
        check_eval_root(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "validated.json").write_text("[]", encoding="utf-8")
    assert check_eval_root(str(tmp_path)) == tmp_path
    with pytest.raises(FileNotFoundError):
        ingest_neon_eval(
            LabelStore(tmp_path / "x.duckdb"),
            tmp_path / "nope",
            TermResolver(RecordingOLS(None, OLS_DIR, "replay")),
        )
    assert not (tmp_path / "x.duckdb").exists()


def test_policy_min_weight_counts(ingested: tuple[LabelStore, IngestReport]) -> None:
    """D9: each bench task's class counts at the policy min_weight (the highest weight per
    target winning, as mesa-anyjev's ``labelled_states`` did) are the published ones."""
    store, _ = ingested
    expected = {
        "term.fits": {0: 86, 1: 199},
        "column.ontology_fits": {0: 76, 1: 114},
        "column.annotate": {0: 63, 1: 35},
        "column.aspect": {0: 12, 1: 10, 2: 18, 3: 11, 4: 6, 5: 3},
        # 301 rows collapse to 278 targets: 23 targets carry both a 0.6 majority row and a
        # 0.5 class-3 row and the majority wins, exactly as in mesa-anyjev (278 / 59).
        "avu.value_kind": {0: 133, 1: 62, 2: 24, 3: 59},
    }
    for task_id, counts in expected.items():
        ls = labelled_targets(store, task_id, min_weight=min_weight_for(task_id))
        assert ls.class_counts() == counts, task_id
        assert len(set(ls.cards)) == 7
        assert len(ls) == len(set(zip(ls.target_sha256, ls.option_key, strict=True)))
        assert all(ls.fold_eligible) and not any(ls.bench_card)
        assert set(ls.leak_group) == {"DP1.10003.001", "DP1.10022.001"}
    # The A2 trap min_weight 0.5 avoids: at 0.6 the negatives vanish.
    assert set(labelled_targets(store, "term.fits", min_weight=0.6).labels) == {0}
    assert labelled_targets(store, "column.annotate", min_weight=0.6).class_counts() == {0: 63}
    assert labelled_targets(store, "column.ontology_fits", min_weight=0.6).class_counts() == {0: 76}
    assert 3 not in labelled_targets(store, "avu.value_kind", min_weight=0.6).class_counts()


def test_labelled_targets_filters_and_identity(ingested: tuple[LabelStore, IngestReport]) -> None:
    store, _ = ingested
    full = labelled_targets(store, "term.fits", min_weight=0.5)
    held_out = "DP1.10003.001.brd_countdata"
    loco = labelled_targets(store, "term.fits", min_weight=0.5, exclude_cards=[held_out])
    assert held_out not in loco.cards and len(loco) == len(full) - full.cards.count(held_out)
    positives = labelled_targets(
        store, "term.fits", sources=["consensus_all", "consensus_majority"]
    )
    assert positives.class_counts() == {0: 86}
    assert labelled_targets(store, "term.fits", sources=[]).labels == []
    assert full.option_key == [s["candidate"]["curie"] for s in full.states]
    assert full.target_sha256 == sorted(full.target_sha256)
    assert all(s["n_candidates"] == 0 for s in full.states)
    ont = labelled_targets(store, "column.ontology_fits")
    assert set(ont.option_key) <= {
        "envo",
        "ncbitaxon",
        "pato",
        "uo",
        "obi",
        "iao",
        "pco",
        "bco",
        "gaz",
        "ro",
        "taxrank",
        "genepio",
    }
    assert all(k == "" for k in labelled_targets(store, "column.aspect").option_key)


def test_rows_carry_product_code_leak_group_and_origin(
    ingested: tuple[LabelStore, IngestReport],
) -> None:
    store, report = ingested
    for row in store.labels_for("column.annotate"):
        assert (
            row["product_code"]
            == product_code_of(row["card"])
            in {"DP1.10003.001", "DP1.10022.001"}
        )
        assert row["leak_group"] == row["product_code"]
        assert row["origin"] == report.source_ref and row["actor"] == "ingest-neon-eval"
        assert row["label"] in ("Yes", "No") and row["option_key"] == ""
        assert row["task_key"] == TASKS["column.annotate"].key
    assert product_code_of("DP1.10003.001.brd_countdata") == "DP1.10003.001"
    assert product_code_of("weird") == "weird"


def test_exclude_models_changes_the_consensus(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "minus.duckdb")
    report = ingest_neon_eval(
        store,
        NEON_EVAL_ROOT,
        TermResolver(RecordingOLS(None, OLS_DIR, "replay")),
        exclude_models=["claude-opus-5-5"],
    )
    assert report.source_ref.endswith(" minus:claude-opus-5-5")
    tf = report.per_task["term.fits"]
    assert sum(tf.values()) < 285 and tf["consensus_all"] != 16
    assert report.terms_resolved + len(report.terms_missing) < 303


def test_snapshot_is_deterministic_and_hashed(
    ingested: tuple[LabelStore, IngestReport], tmp_path: Path
) -> None:
    store, _ = ingested
    sha1 = snapshot(store, tmp_path / "snap" / "a.parquet")
    sha2 = snapshot(store, tmp_path / "snap" / "b.parquet")
    assert sha1 == sha2 and len(sha1) == 64
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT task_key, target_sha256, option_key, label_source FROM '{tmp_path / 'snap' / 'a.parquet'}'"  # noqa: S608
    ).fetchall()
    con.close()
    assert len(rows) == 934 and rows == sorted(rows)
    # Another ingestion has other label_ids and timestamps: another labels_sha256 (D30).
    other = LabelStore(tmp_path / "other.duckdb")
    ingest_neon_eval(other, NEON_EVAL_ROOT, TermResolver(RecordingOLS(None, OLS_DIR, "replay")))
    assert snapshot(other, tmp_path / "snap" / "c.parquet") != sha1


def test_term_resolver_caches_and_handles_failures(tmp_path: Path) -> None:
    class Flaky:
        calls = 0

        def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None:
            self.calls += 1
            if "boom" in iri:
                raise RuntimeError("down")
            if "root" in iri:
                return {"curie": "X:root", "iri": iri, "label": "r", "isRoot": True}
            return {"curie": "PATO:1", "iri": iri, "label": "x", "description": "d"}

    client = Flaky()
    r = TermResolver(client)  # type: ignore[arg-type]
    assert r.resolve("X:boom") is None and r.resolve("X:root") is None
    cand = r.resolve("PATO:1")
    assert cand is not None and cand.curie == "PATO:1" and cand.ontology_id == "pato"
    assert r.resolve("PATO:1") is cand and client.calls == 3
