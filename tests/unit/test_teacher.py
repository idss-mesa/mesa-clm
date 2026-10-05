"""The teacher ingest (``mesa_clm.learn.teacher``; DESIGN D19; the M4 analysis plan R3) on a
**synthetic** corpus and synthetic table cards: every drop reason, the target resolution to
every card of the product that carries the column, Yes-only rows for both tasks, the
aspect-allowed rule for ``column.ontology_fits``, the row flags (``fold_eligible=False``,
``leak_group``, ``origin``), the refusal of a ``mesa-clm`` model, the corpus pin, the teacher
rows read back as :class:`~mesa_clm.learn.labels.TeacherRows`, and the silver-minus-Opus
survivor rule. No silver label and no model output is involved. One label-free check on the
recorded ``tests/fixtures/ols-teacher`` closure runs only when the real corpus is present."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from mesa_clm.bench import registered as reg
from mesa_clm.learn import teacher as T
from mesa_clm.learn.labels import (
    WEIGHTS,
    LabelledSet,
    TeacherRows,
    TermResolver,
    surviving_identities,
    teacher_rows,
)
from mesa_clm.ols import RecordingOLS, iri_for
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.registry import prefix_of
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
REAL_NEON_ROOT = Path(os.environ.get("MESA_CLM_NEON__ROOT", "~/neon-ducklake")).expanduser()

CARD = """# Dataset: {product} / {table}
Product: {product} — A product. Its description
Source: NEON (National Ecological Observatory Network) Data API.
Sites: SRER (Santa Rita Experimental Range NEON, AZ, domain D14 Desert Southwest; Desert shrubland).
Rows: 10; months covered: 2024-05 to 2024-05 (1 distinct months).

## Columns (name | NEON description | type | unit | profile)
{columns}
"""


def _card(root: Path, product: str, table: str, columns: list[str]) -> None:
    lines = "\n".join(
        f"- {c} | the {c} column | string | - | 3 distinct; top: a (1)" for c in columns
    )
    (root / T.CARDS_DIR).mkdir(parents=True, exist_ok=True)
    (root / T.CARDS_DIR / f"{product}.{table}.md").write_text(
        CARD.format(product=product, table=table, columns=lines), encoding="utf-8"
    )


def _item(curie: str, aspect: str, column: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "status": "accepted",
        "curie": curie,
        "label": curie.lower(),
        "ontologyId": prefix_of(curie).lower(),
        "aspect": aspect,
        "column": column,
        **extra,
    }


def _validated(
    root: Path, product: str, items: list[dict[str, Any]], model: str = "claude-opus-5-5"
) -> Path:
    (root / T.CORPUS_DIR).mkdir(parents=True, exist_ok=True)
    path = root / T.CORPUS_DIR / f"{product}.validated.json"
    path.write_text(
        json.dumps(
            {
                "productCode": product,
                "model": model,
                "replicates": [
                    {"rep": 1, "proposal": f"sites/X/curation/proposals/{product}.rep1.json"}
                ],
                "accepted": items,
                "proposed": [],
                "rejected": [],
            }
        ),
        encoding="utf-8",
    )
    return path


class _Resolver:
    """A TermResolver stand-in: every CURIE but ``missing`` resolves to a term."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None:
        self.calls.append(iri)
        if "missing" in iri:
            return None
        curie = iri.rsplit("/", 1)[-1].replace("_", ":", 1)
        return {
            "curie": curie,
            "iri": iri,
            "label": f"label of {curie}",
            "description": f"description of {curie} " + "x" * 400,
            "ontologyId": ontology_id,
            "isRoot": False,
            "hasChildren": False,
            "synonyms": [],
        }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "neon"
    _card(root, "DP1.00001.001", "t1", ["windSpeedMean", "siteID"])
    _card(root, "DP1.00001.001", "t2", ["windSpeedMean", "other"])
    _card(root, "DP1.00002.001", "t1", ["temp"])
    _validated(
        root,
        "DP1.00001.001",
        [
            _item(
                "ENVO:01001362", "measured_property", "windSpeedMean"
            ),  # 2 cards, envo allows measurement
            _item("OBI:0000001", "method", "siteID"),  # 1 card; obi allows method
            _item("PATO:0000001", "measured_property", "nowhere"),  # unresolved column
            _item("STATO:0000001", "measured_property", "windSpeedMean"),  # out of registry
            _item("ENVO:0000002", "process", "windSpeedMean"),  # unmapped aspect
            _item("ENVO:0000003", "data_type", None),  # no column
            _item(
                "ENVO:0000004", "data_type", "windSpeedMean"
            ),  # envo not allowed for data_type: term.fits only
            _item("IAO:0000missing", "data_type", "windSpeedMean"),  # term missing
            _item("ENVO:0000005", "measured_property", "siteID", ontologyId="obi"),  # mismatch
        ],
    )
    _validated(root, "DP1.00002.001", [_item("PATO:0000002", "measured_property", "temp")])
    _validated(
        root, "DP1.00003.001", [_item("PATO:0000003", "measured_property", "temp")]
    )  # no card
    return root


def test_ingest_counts_rows_flags_and_resolution(corpus: Path, tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.duckdb")
    resolver = TermResolver(_Resolver())  # type: ignore[arg-type]
    report = T.ingest_teacher(store, corpus, resolver, actor="t")
    s = report.summary()
    assert s["files"] == 3 and s["items"] == 11 and s["proposal_files"] == 0
    assert s["implicit_rows"] == 0  # no replicate proposal file exists: nothing synthesized
    assert s["dropped"] == {
        "no_column": 1,
        "ontology_mismatch": 1,
        "out_of_registry": 1,
        "term_missing": 1,
        "unmapped_aspect": 1,
        "unresolved_column": 2,  # "nowhere" and the product without a card
    }
    # envo for data_type: the term.fits row is written, the ontology_fits row skipped (not a drop)
    assert s["ontology_rows_skipped"] == 1 and "ontology_not_aspect_allowed" not in s["dropped"]
    assert s["terms_missing"] == ["IAO:0000missing"]
    assert s["products_without_card"] == ["DP1.00003.001"]
    assert s["corpus_sha256"] == reg.teacher_corpus_sha256(corpus / T.CORPUS_DIR)
    # windSpeedMean -> 2 cards (ENVO:01001362 ×2, ENVO:0000004 ×2), siteID -> 1 (OBI), temp -> 1
    assert s["per_task"]["term.fits"] == {"teacher": 6}
    # ontology_fits only when the ontology is aspect-allowed: envo/measurement ×2, obi/method, pato
    assert s["per_task"]["column.ontology_fits"] == {"teacher": 4}
    assert s["inserted"] == 10 and s["skipped_existing"] == 0
    rows = store.labels_for("term.fits")
    assert len(rows) == 6 and all(r["label"] == "Yes" and r["label_index"] == 0 for r in rows)
    for r in rows:
        assert r["label_source"] == "teacher" and r["weight"] == WEIGHTS["teacher"]
        assert r["fold_eligible"] is False and r["bench_card"] is False
        assert r["leak_group"] == r["product_code"] == r["card"].rsplit(".", 1)[0]
        assert (
            r["origin"].startswith(T.TEACHER_ORIGIN)
            and len(r["origin"]) == len(T.TEACHER_ORIGIN) + 12
        )
        assert r["state_json"]["candidate"]["description"].startswith("description of")
        assert len(r["state_json"]["candidate"]["description"]) == 300  # cut as the framings render
        assert r["state_json"]["scope"] == "column"
    cards = sorted(r["card"] for r in rows if r["option_key"] == "ENVO:01001362")
    assert cards == ["DP1.00001.001.t1", "DP1.00001.001.t2"]
    onto = store.labels_for("column.ontology_fits")
    assert {r["option_key"] for r in onto} == {"envo", "obi", "pato"}
    assert all(r["state_json"]["aspect"] in ("measurement", "method") for r in onto)
    # idempotent: a second ingest inserts nothing
    again = T.ingest_teacher(store, corpus, resolver)
    assert again.inserted == 0 and again.skipped_existing == 10
    # the rows read back for training only
    tr = teacher_rows(store, "term.fits")
    assert (
        len(tr) == 6
        and set(tr.sources) == {"teacher"}
        and set(tr.leak_group)
        == {
            "DP1.00001.001",
            "DP1.00002.001",
        }
    )
    assert tr.training_indices("DP1.00001.001") == [
        i for i, g in enumerate(tr.leak_group) if g != "DP1.00001.001"
    ]
    assert tr.reweighted({"teacher": 0.3}).weights == [0.3] * 6
    task = tr.as_task()
    assert task.meta["teacher"] and task.products == tr.leak_group and len(task.items) == 6
    with pytest.raises(RuntimeError, match="no feature builder"):
        tr.matrix("lowdim.v1")
    assert teacher_rows(store, "column.aspect").states == []


def test_refusals_and_the_corpus_pin(corpus: Path, tmp_path: Path) -> None:
    folder = corpus / T.CORPUS_DIR
    pairs = sorted(
        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in folder.glob("*.validated.json")
    )
    assert T.corpus_sha256(folder) == hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
    files = T.load_corpus(corpus)
    assert [f.product_code for f in files] == ["DP1.00001.001", "DP1.00002.001", "DP1.00003.001"]
    assert T.teacher_curies(files) == [
        "ENVO:0000003",
        "ENVO:0000004",
        "ENVO:0000005",
        "ENVO:01001362",
        "IAO:0000missing",
        "OBI:0000001",
        "PATO:0000001",
        "PATO:0000002",
        "PATO:0000003",
    ]
    _validated(corpus, "DP1.00009.001", [], model="mesa-clm-0.1")
    with pytest.raises(T.TeacherError, match="mesa-clm"):
        T.load_corpus(corpus)
    (corpus / T.CORPUS_DIR / "DP1.00009.001.validated.json").unlink()
    # a replicate proposal file of a mesa-clm model is refused too (D19)
    prop = corpus / "sites/X/curation/proposals/DP1.00002.001.rep1.json"
    prop.parent.mkdir(parents=True)
    prop.write_text(json.dumps({"model": "mesa-clm"}))
    with pytest.raises(T.TeacherError, match="mesa-clm"):
        T.load_corpus(corpus)
    prop.write_text(json.dumps({"model": "claude-opus-5-5"}))
    assert T.load_corpus(corpus)[1].replicate_models == ("claude-opus-5-5",)
    with pytest.raises(T.TeacherError, match="no such corpus"):
        T.load_corpus(tmp_path / "nowhere")
    with pytest.raises(T.TeacherError, match="no such cards"):
        T.ingest_teacher(
            LabelStore(tmp_path / "x.duckdb"),
            corpus,
            TermResolver(_Resolver()),
            cards_dir=tmp_path / "no",
        )  # type: ignore[arg-type]


def test_teacher_rows_validation() -> None:
    with pytest.raises(ValueError, match="one entry per row"):
        TeacherRows("term.fits", [{}], [0], [], [], [], [], [], [])
    with pytest.raises(ValueError, match="not teacher sources"):
        TeacherRows("term.fits", [{}], [0], [0.5], ["curator"], ["c"], ["p"], ["t"], ["o"])


def test_surviving_identities_is_same_identity_and_same_label() -> None:
    registered = LabelledSet(
        states=[{}] * 4,
        labels=[0, 1, 0, 1],
        weights=[0.8, 0.5, 0.6, 0.5],
        cards=["c"] * 4,
        target_sha256=["t1", "t2", "t3", "t4"],
        option_key=["a", "b", "c", "d"],
    )
    rebuilt = [("t1", "a", 0), ("t2", "b", 0), ("t4", "d", 1)]  # t2 flipped, t3 gone
    assert surviving_identities(registered, rebuilt) == [0, 3]
    assert surviving_identities(registered, []) == []


@pytest.mark.skipif(
    not (REAL_NEON_ROOT / T.CORPUS_DIR).is_dir(), reason="the neon-ducklake corpus is not here"
)
def test_the_recorded_teacher_fixtures_cover_the_corpus_curies() -> None:
    """Label-free: every in-registry, aspect-mapped CURIE of the real corpus has a recorded
    ``get_term`` fixture under ``tests/fixtures/ols-teacher`` (the ingest replays them)."""
    replay = RecordingOLS(None, ROOT / "tests" / "fixtures" / "ols-teacher", "replay")
    files = T.load_corpus(REAL_NEON_ROOT)
    curies = T.teacher_curies(files)
    assert len(files) == 55 and len(curies) == 110
    missing = [
        c
        for c in curies
        if not replay.fixture_path(
            "get_term", {"ontology_id": prefix_of(c).lower(), "iri": iri_for(c)}
        ).exists()
    ]
    assert missing == []
    # the pin is the integrator's (bench.registered.REGISTERED_M4.teacher_corpus_sha256, filled
    # after the ingest); here only the formula and the closure are checked
    assert len(T.corpus_sha256(REAL_NEON_ROOT / T.CORPUS_DIR)) == 64
    assert all(TASKS[t].active for t in T.TEACHER_TASKS)
