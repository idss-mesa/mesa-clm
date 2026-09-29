"""``test_label_identity`` (plan §9): labels are identified by ``(task_key, target_sha256,
option_key, label_source)`` (DESIGN D1). The same finding recorded on states that differ only in
``n_candidates`` is one row; an anchor pick needs no candidate state; ``import_anyjev`` derives
the identity from mesa-anyjev ``state_json`` rows and keeps the highest weight per pair. The
anyjev sidecar here is built in the test with mesa-anyjev's own DDL, never read from the
sibling checkout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pytest

from mesa_clm.cards import DatasetCard
from mesa_clm.identity import identity
from mesa_clm.learn.labels import WEIGHTS, import_anyjev, labelled_targets
from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.states import candidate_state, state_sha256, target_state, value_kind_state
from mesa_clm.tasks import TASKS

CAND = {
    "label": "distance",
    "curie": "PATO:0000040",
    "ontology_id": "pato",
    "description": "d",
    "synonyms": [],
    "has_children": False,
}
OTHER = {**CAND, "label": "length", "curie": "PATO:0000122"}

# mesa-anyjev ``provenance/store.py`` DUCKDB_DDL, labels table only (6159281).
ANYJEV_DDL = (
    "CREATE SCHEMA IF NOT EXISTS mesa_anyjev",
    """CREATE TABLE IF NOT EXISTS mesa_anyjev.labels (
        label_id TEXT PRIMARY KEY, question_id TEXT NOT NULL, question_key TEXT NOT NULL,
        state_sha256 TEXT NOT NULL, state_json JSON NOT NULL, label_index INTEGER NOT NULL,
        label_source TEXT NOT NULL, weight REAL NOT NULL, source_ref TEXT, card TEXT,
        decision_id TEXT, batch_id TEXT, consumed_in_bundle TEXT, ts TIMESTAMPTZ NOT NULL,
        UNIQUE (question_key, state_sha256, label_source))""",
)


def _label(
    task_id: str, state: dict[str, Any], label_index: int, source: str, **over: Any
) -> LabelRow:
    t = TASKS[task_id]
    ident = identity(task_id, state, over.pop("option", None))
    return LabelRow(
        task_id=task_id,
        task_key=ident.task_key,
        target_sha256=ident.target_sha256,
        option_key=ident.option_key,
        label_source=source,  # type: ignore[arg-type]
        label=t.options[label_index],
        label_index=label_index,
        weight=WEIGHTS[source],
        state_sha256=state_sha256(state),
        state_json=state,
        card=str(state["card"]["dataset"]),
        **over,
    )


def test_same_pair_different_group_sizes_is_one_label(card: DatasetCard, tmp_path: Path) -> None:
    col = card.column("observerDistance")
    small = candidate_state(card, "column", col, "measurement", CAND, 3)
    large = candidate_state(card, "column", col, "measurement", CAND, 12)
    assert state_sha256(small) != state_sha256(large)
    store = LabelStore(tmp_path / "labels.duckdb")
    assert store.insert_labels([_label("term.fits", small, 0, "curator")]) == 1
    assert store.insert_labels([_label("term.fits", large, 0, "curator")]) == 0
    # Another source on the same pair is another row; labelled_targets keeps the heaviest.
    assert store.insert_labels([_label("term.fits", large, 1, "consensus_negative")]) == 1
    ls = labelled_targets(store, "term.fits")
    assert len(ls) == 1 and ls.labels == [0] and ls.weights == [1.0] and ls.sources == ["curator"]
    assert ls.option_key == ["PATO:0000040"]
    only_silver = labelled_targets(store, "term.fits", sources=["consensus_negative"])
    assert only_silver.labels == [1] and only_silver.target_sha256 == ls.target_sha256


def test_anchor_pick_needs_no_candidate_state(card: DatasetCard, tmp_path: Path) -> None:
    """D21: an explicit "none of these" is an anchor-positive row plus per-candidate negatives."""
    col = card.column("observerDistance")
    view = target_state(card, "column", "measurement", column=col)
    rows = [
        _label("term.fits", view, 0, "curator", option=ANCHOR_KEY),
        _label(
            "term.fits", candidate_state(card, "column", col, "measurement", CAND, 2), 1, "curator"
        ),
        _label(
            "term.fits", candidate_state(card, "column", col, "measurement", OTHER, 2), 1, "curator"
        ),
    ]
    assert len({r.target_sha256 for r in rows}) == 1
    assert [r.option_key for r in rows] == [ANCHOR_KEY, "PATO:0000040", "PATO:0000122"]
    store = LabelStore(tmp_path / "labels.duckdb")
    assert store.insert_labels(rows) == 3
    ls = labelled_targets(store, "term.fits")
    # Sorted by option_key: CURIEs (upper-case letters) sort before the anchor's underscores.
    assert ls.option_key == ["PATO:0000040", "PATO:0000122", ANCHOR_KEY] and ls.labels == [1, 1, 0]


def test_value_kind_targets_differ_per_term(card: DatasetCard, tmp_path: Path) -> None:
    col = card.column("observerDistance")
    a = _label(
        "avu.value_kind", value_kind_state(card, col, CAND, "measurement"), 0, "consensus_majority"
    )
    b = _label(
        "avu.value_kind", value_kind_state(card, col, OTHER, "measurement"), 3, "consensus_negative"
    )
    assert a.target_sha256 != b.target_sha256 and a.option_key == b.option_key == ""
    store = LabelStore(tmp_path / "labels.duckdb")
    assert store.insert_labels([a, b]) == 2


def _anyjev_sidecar(path: Path, rows: list[dict[str, Any]]) -> None:
    con = duckdb.connect(str(path))
    for stmt in ANYJEV_DDL:
        con.execute(stmt)
    for r in rows:
        con.execute(
            "INSERT INTO mesa_anyjev.labels (label_id, question_id, question_key, state_sha256, "
            "state_json, label_index, label_source, weight, source_ref, card, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(uuid4()),
                r["question_id"],
                r["question_key"],
                state_sha256(r["state"]),
                json.dumps(r["state"], sort_keys=True, separators=(",", ":")),
                r["label_index"],
                r["label_source"],
                r["weight"],
                "neon-avu-eval/results/validated.json@deadbeef0000",
                r["state"]["card"]["dataset"],
                r["ts"],
            ],
        )
    con.close()


def test_import_anyjev_derives_identity_and_keeps_the_highest_weight(
    card: DatasetCard, tmp_path: Path
) -> None:
    col = card.column("observerDistance")
    small = candidate_state(card, "column", col, "measurement", CAND, 3)
    large = candidate_state(card, "column", col, "measurement", CAND, 12)
    other = candidate_state(card, "column", col, "measurement", OTHER, 12)
    vk = value_kind_state(card, col, CAND, "measurement")
    keep = {
        "card": small["card"],
        "avu": {},
        "term": {},
        "scope": "column",
        "target": "x",
        "siblings": [],
    }
    tk = TASKS["term.fits"].key
    rows = [
        # the same pair at two group sizes and two sources: one identity per source, 1.0 wins
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": small,
            "label_index": 1,
            "label_source": "consensus_negative",
            "weight": 0.5,
            "ts": "2026-01-01 00:00:00+00",
        },
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": large,
            "label_index": 0,
            "label_source": "curator",
            "weight": 1.0,
            "ts": "2026-01-02 00:00:00+00",
        },
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": small,
            "label_index": 0,
            "label_source": "curator",
            "weight": 1.0,
            "ts": "2026-01-03 00:00:00+00",
        },
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": other,
            "label_index": 1,
            "label_source": "curator_implicit",
            "weight": 0.7,
            "ts": "2026-01-02 00:00:00+00",
        },
        # the same source, the same identity, a different label: the later one wins, counted
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": candidate_state(card, "column", col, "measurement", CAND, 5),
            "label_index": 1,
            "label_source": "consensus_negative",
            "weight": 0.5,
            "ts": "2026-01-04 00:00:00+00",
        },
        {
            "question_id": "avu.value_kind",
            "question_key": TASKS["avu.value_kind"].key,
            "state": vk,
            "label_index": 2,
            "label_source": "consensus_majority",
            "weight": 0.6,
            "ts": "2026-01-01 00:00:00+00",
        },
        # skipped: inactive task, rotated key, a source mesa-clm does not know
        {
            "question_id": "avu.keep",
            "question_key": TASKS["avu.keep"].key,
            "state": keep,
            "label_index": 0,
            "label_source": "consensus_all",
            "weight": 0.8,
            "ts": "2026-01-01 00:00:00+00",
        },
        {
            "question_id": "term.fits",
            "question_key": "0" * 16,
            "state": small,
            "label_index": 0,
            "label_source": "curator",
            "weight": 1.0,
            "ts": "2026-01-01 00:00:00+00",
        },
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": other,
            "label_index": 0,
            "label_source": "hosted_jev",
            "weight": 0.5,
            "ts": "2026-01-01 00:00:00+00",
        },
    ]
    src = tmp_path / "anyjev.duckdb"
    _anyjev_sidecar(src, rows)
    store = LabelStore(tmp_path / "labels.duckdb")
    report = import_anyjev(f"duckdb:///{src}", store)
    assert report.inserted == 4 and report.skipped_existing == 0
    assert report.skipped == {
        "inactive_task": 1,
        "key_mismatch": 1,
        "unknown_source": 1,
        "collapsed_identity": 2,
    }
    assert report.conflicts == 0
    assert report.per_task == {
        "term.fits": {"consensus_negative": 1, "curator": 1, "curator_implicit": 1},
        "avu.value_kind": {"consensus_majority": 1},
    }
    assert report.source_ref.startswith("anyjev-import:anyjev.duckdb@")
    ls = labelled_targets(store, "term.fits")
    assert ls.option_key == ["PATO:0000040", "PATO:0000122"]
    assert ls.labels == [0, 1] and ls.weights == [1.0, 0.7]
    assert ls.cards == [card.name, card.name] and ls.leak_group == [
        "DP1.10003.001",
        "DP1.10003.001",
    ]
    picked = next(r for r in store.labels_for("term.fits") if r["label_source"] == "curator")
    assert picked["state_json"]["n_candidates"] == 3  # the later row of equal weight won
    assert picked["origin"].endswith("neon-avu-eval/results/validated.json@deadbeef0000")
    assert picked["actor"] == "import-anyjev" and picked["fold_eligible"] is True
    assert import_anyjev(str(src), store).inserted == 0  # idempotent
    vk_rows = labelled_targets(store, "avu.value_kind")
    assert vk_rows.labels == [2] and vk_rows.states == [vk]
    with pytest.raises(FileNotFoundError):
        import_anyjev(tmp_path / "missing.duckdb", store)


def test_import_anyjev_counts_conflicts(card: DatasetCard, tmp_path: Path) -> None:
    col = card.column("observerDistance")
    tk = TASKS["term.fits"].key
    rows = [
        {
            "question_id": "term.fits",
            "question_key": tk,
            "state": candidate_state(card, "column", col, "measurement", CAND, n),
            "label_index": i,
            "label_source": "consensus_negative",
            "weight": 0.5,
            "ts": f"2026-01-0{n} 00:00:00+00",
        }
        for n, i in ((1, 1), (2, 0))
    ]
    src = tmp_path / "anyjev.duckdb"
    _anyjev_sidecar(src, rows)
    store = LabelStore(tmp_path / "labels.duckdb")
    report = import_anyjev(src, store)
    assert report.inserted == 1 and report.conflicts == 1
    assert labelled_targets(store, "term.fits").labels == [0]  # the later row won
