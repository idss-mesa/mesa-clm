"""X2 baselines (``mesa_clm.bench.x2``; PR "X2 baselines"; ``design/m2-analysis-plan.md``
§10), on synthetic data: the PR #13 replica is scikit-learn's ``StandardScaler +
LogisticRegression(C=1, max_iter=2000)`` fitted per fold on the training rows only, unweighted,
bit for bit; the AnyJev L2 dump joins the bench items by D1 identity (unmatched items on either
side counted, never imputed; duplicates and fold mismatches refused); ``run_x2`` keeps the M0
control. One label-free check on the committed files: every identity of the AnyJev dump's
states is an identity of the frozen snapshot (state_json and identity columns only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from mesa_clm.bench import registered as reg
from mesa_clm.bench.baselines import baselines_cell
from mesa_clm.bench.cells import CellError, TextIndex, item_identities
from mesa_clm.bench.tasks.base import Task
from mesa_clm.bench.x2 import (
    ANYJEV_FORMAT,
    REPLICA_TASKS,
    SPEC_VARIANTS,
    anyjev_cell,
    join_anyjev,
    load_anyjev,
    replica_cell,
    replica_pipeline,
    run_x2,
)
from mesa_clm.identity import identity, target_sha256
from mesa_clm.learn.features import ROLE_SIDE, X2_SPECS, FeatureStore, ManifestRow, text_sha256
from mesa_clm.providers.tiered import fake_fingerprint
from mesa_clm.states import state_sha256
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
DUMP = ROOT / "bench" / "baselines" / "anyjev_l2_2026-09-29.json"
FP = fake_fingerprint().encoder_fp
DIM = 32
CARDS = tuple(f"DP0.0000{1 + i // 4}.001.card{i}" for i in range(7))
FINGERPRINT = {
    "encoder_fp": FP,
    "clm_model_fp": "0f0f0f0f0f0f",
    "schema_sha256": "5" * 64,
    "serving_lock_sha": "6" * 64,
}
COMMON: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64, "B": 200}


def _row(fid: str, target: str, option: str, text: str) -> ManifestRow:
    return ManifestRow(
        task_id="term.fits",
        framing_id=fid,
        target_sha256=target,
        option_key=option,
        role="context",
        side=ROLE_SIDE["context"],
        text_sha256=text_sha256(text),
        text=text,
    )


@dataclass
class World:
    task: Task
    rows: list[ManifestRow]
    store: FeatureStore

    @property
    def index(self) -> TextIndex:
        return TextIndex(self.rows)


def x2_world(root: Path, *, n_targets: int = 12, n_cands: int = 3, seed: int = 0) -> World:
    """term.fits pairs whose joint-state vectors carry a planted linear signal for Yes."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(DIM)
    direction /= np.linalg.norm(direction)
    rows: list[ManifestRow] = []
    texts: list[str] = []
    vecs: list[np.ndarray] = []
    items: list[tuple[dict[str, Any], int]] = []
    cards: list[str] = []
    keys: list[str] = []
    for ci, card in enumerate(CARDS):
        for t in range(n_targets):
            base = {
                "card": {"dataset": card},
                "scope": "column",
                "aspect": "measurement",
                "column": {"name": f"col{t}"},
            }
            target = target_sha256("term.fits", base)
            for c in range(n_cands):
                key = f"X:{ci}{t:03d}{c}"
                label = int(rng.random() >= 0.4)  # 0 = Yes
                for spec in X2_SPECS:
                    text = f"{spec} joint {key}"
                    vec = rng.standard_normal(DIM) + (1.2 if label == 0 else -1.2) * direction
                    rows.append(_row(spec, target, key, text))
                    texts.append(text)
                    vecs.append(vec)
                items.append(({**base, "candidate": {"curie": key}}, label))
                cards.append(card)
                keys.append(key)
    store = FeatureStore(root / "features", FP, dim=DIM)
    store.add(texts, np.stack(vecs), [3] * len(texts))
    yes = sum(1 for _, y in items if y == 0)
    task = Task(
        "neon_term_fits",
        TASKS["term.fits"],
        items,
        "synthetic",
        "synthetic",
        cards=cards,
        weights=[0.6 if y == 0 else 0.5 for _, y in items],
        products=[c.rsplit(".", 1)[0] for c in cards],
        option_keys=keys,
        meta={
            "min_weight": 0.5,
            "masked": True,
            "mask": None,
            "label_sources": {"consensus_majority": yes, "consensus_negative": len(items) - yes},
            "excluded": {},
        },
    )
    return World(task, rows, store)


# -- the PR #13 replica ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", X2_SPECS)
def test_replica_equals_scikit_learn_fold_by_fold(tmp_path: Path, spec: str) -> None:
    sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    linear = pytest.importorskip("sklearn.linear_model")
    w = x2_world(tmp_path)
    cell = replica_cell(w.task, w.index, w.store, spec, fingerprint=FINGERPRINT, **COMMON)
    assert cell.key == f"neon_term_fits.baseline.pr13@{SPEC_VARIANTS[spec]}"
    assert cell.tier == "baseline" and cell.feature_spec == spec and not cell.servable
    assert cell.fingerprint == {k: v for k, v in FINGERPRINT.items() if k != "clm_model_fp"}
    assert cell.selection == "none" and cell.pre_registered and not cell.exploratory
    assert cell.n_folds == 7 and cell.skipped_folds == {}
    assert cell.diagnostics["weighted"] is False and cell.diagnostics["sklearn"]
    got = {(i.target_sha256, i.option_key): i.probs for i in cell.items or []}
    ids = item_identities(w.task)
    texts = [w.index.text("term.fits", spec, t, o, "context") for t, o in ids]
    x = np.asarray(w.store.get(texts), dtype=np.float64)
    y = np.asarray(w.task.labels)
    for fold in w.task.leave_one_card_out():
        ref = sklearn_pipeline.make_pipeline(
            preprocessing.StandardScaler(), linear.LogisticRegression(C=1.0, max_iter=2000)
        ).fit(x[fold.train], y[fold.train])
        p_yes = ref.predict_proba(x[fold.test])[:, list(ref.classes_).index(0)]
        for i, p in zip(fold.test, p_yes, strict=True):
            assert got[ids[i]][0] == p  # bit for bit: the same recipe on the same rows
            assert got[ids[i]][1] == pytest.approx(1.0 - p, abs=1e-15)
        info = cell.diagnostics["per_fold"][fold.held_out]
        assert info["n_iter"] == int(ref[-1].n_iter_[0]) and info["converged"]
    assert cell.metrics is not None and cell.metrics.auroc is not None
    assert cell.metrics.auroc > 0.8  # the planted signal is there


def test_replica_is_unweighted_and_fits_on_training_rows_only(tmp_path: Path) -> None:
    w = x2_world(tmp_path)
    base = replica_cell(w.task, w.index, w.store, "joint4096@S1", fingerprint=None, **COMMON)
    heavy = Task(
        w.task.name, w.task.spec, w.task.items, "s", "s", cards=w.task.cards,
        weights=[5.0 if i % 2 else 0.1 for i in range(len(w.task.items))],
        products=w.task.products, option_keys=w.task.option_keys, meta=w.task.meta,
    )  # fmt: skip
    again = replica_cell(heavy, w.index, w.store, "joint4096@S1", fingerprint=None, **COMMON)
    assert [i.probs for i in base.items or []] == [i.probs for i in again.items or []]
    # flipping one card's labels leaves that card's held-out predictions untouched
    card = CARDS[3]
    flipped = Task(
        w.task.name, w.task.spec,
        [(st, 1 - y) if w.task.cards[i] == card else (st, y) for i, (st, y) in enumerate(w.task.items)],
        "s", "s", cards=w.task.cards, products=w.task.products,
        option_keys=w.task.option_keys, meta=w.task.meta,
    )  # fmt: skip
    other = replica_cell(flipped, w.index, w.store, "joint4096@S1", fingerprint=None, **COMMON)
    a = {(i.target_sha256, i.option_key): i.probs for i in base.items or [] if i.card == card}
    b = {(i.target_sha256, i.option_key): i.probs for i in other.items or [] if i.card == card}
    assert a == b and a


def test_replica_guards_and_refusals(tmp_path: Path) -> None:
    w = x2_world(tmp_path)
    one_class = Task(
        w.task.name, w.task.spec,
        [(st, 1) if w.task.cards[i] == CARDS[0] else (st, y) for i, (st, y) in enumerate(w.task.items)],
        "s", "s", cards=w.task.cards, products=w.task.products,
        option_keys=w.task.option_keys, meta=w.task.meta,
    )  # fmt: skip
    cell = replica_cell(one_class, w.index, w.store, "joint4096@S1ns", fingerprint=None, **COMMON)
    assert cell.skipped_folds[CARDS[0]].startswith("insufficient_heldout_per_class")
    assert cell.n_folds == 6
    with pytest.raises(CellError, match="unknown joint spec"):
        replica_cell(w.task, w.index, w.store, "F7", fingerprint=None, **COMMON)
    aspect = Task("neon_aspect", TASKS["column.aspect"], [], "s", "s")
    with pytest.raises(CellError, match="covers"):
        replica_cell(aspect, w.index, w.store, "joint4096@S1", fingerprint=None, **COMMON)
    assert REPLICA_TASKS == ("term.fits", "column.ontology_fits")
    assert type(replica_pipeline()[-1]).__name__ == "LogisticRegression"
    assert replica_pipeline()[-1].C == 1.0 and replica_pipeline()[-1].max_iter == 2000


# -- AnyJev L2 ------------------------------------------------------------------------------------------


def _dump_item(task: Task, i: int, p_yes: float) -> dict[str, Any]:
    state, label = task.items[i]
    return {
        "card": task.cards[i],
        "fold": task.cards[i],
        "state_sha256": state_sha256(state),
        "target": {"dataset": task.cards[i]},
        "label": label,
        "label_text": TASKS["term.fits"].options[label],
        "p_yes": p_yes,
        "probs": [p_yes, 1 - p_yes],
        "answer_index": 0 if p_yes >= 0.5 else 1,
        "level": "L2",
        "prompt_sha256": "p",
        "diagnostics": {},
        "state_json": state,
    }


def _write_dump(path: Path, items: list[dict[str, Any]], **over: Any) -> Path:
    payload = {
        "format": ANYJEV_FORMAT,
        "mesa_anyjev": "0.1.0",
        "questions_lock_sha": "q",
        "labels_file_sha256": "l",
        "level": "L2",
        "tasks": {"neon_term_fits": {"question_key": "0ccc8d141ffd30ff", "items": items}},
        **over,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_anyjev_join_and_cell(tmp_path: Path) -> None:
    w = x2_world(tmp_path, n_targets=6)
    n = len(w.task.items)
    rng = np.random.default_rng(1)
    p = rng.random(n)
    items = [_dump_item(w.task, i, float(p[i])) for i in range(n) if i != 5]  # item 5 unmatched
    items[0]["label"] = 1 - items[0]["label"]  # a label disagreement: counted, never used
    items[1]["state_sha256"] = "0" * 64  # a re-rendered state: counted
    extra = dict(_dump_item(w.task, 0, 0.5))
    extra["state_json"] = {**extra["state_json"], "candidate": {"curie": "X:none"}}
    items.append(extra)  # matches no bench item
    dump = load_anyjev(_write_dump(tmp_path / "dump.json", items))
    assert set(dump.items) == {"term.fits"} and dump.question_keys == {
        "term.fits": "0ccc8d141ffd30ff"
    }
    assert dump.meta["format"] == ANYJEV_FORMAT and len(dump.sha256) == 64
    join = join_anyjev(w.task, dump.items["term.fits"])
    rep = join.report
    assert rep["matched"] == n - 1 and rep["unmatched_bench"] == 1 and rep["unmatched_dump"] == 1
    assert rep["label_differs"] == 1 and rep["state_sha256_differs"] == 1 and not rep["full_set"]
    ids = item_identities(w.task)
    assert rep["unmatched_bench_identities"] == [f"{ids[5][0][:12]}/{ids[5][1]}"]
    assert 5 not in join.idx and join.idx == sorted(join.idx)
    cell = anyjev_cell(w.task, join, dump, **COMMON)
    assert cell.key == "neon_term_fits.baseline.anyjev_l2" and cell.counts.n == n - 1
    assert cell.fingerprint is None and cell.question_key is None and not cell.servable
    assert cell.n_folds == 7 and cell.skipped_folds == {}
    assert cell.diagnostics["anyjev"]["question_key"] == "0ccc8d141ffd30ff"
    got = {(i.target_sha256, i.option_key): (i.probs, i.label) for i in cell.items or []}
    for i in join.idx:
        probs, label = got[ids[i]]
        assert probs == [p[i], 1 - p[i]] and label == w.task.items[i][1]  # the snapshot's label


def test_anyjev_refusals(tmp_path: Path) -> None:
    w = x2_world(tmp_path, n_targets=2)
    good = [_dump_item(w.task, i, 0.5) for i in range(len(w.task.items))]
    dup = load_anyjev(_write_dump(tmp_path / "dup.json", [*good, good[0]]))
    with pytest.raises(CellError, match="twice"):
        join_anyjev(w.task, dup.items["term.fits"])
    moved = [dict(it) for it in good]
    moved[0]["fold"] = CARDS[1]
    with pytest.raises(CellError, match="folds are not the same"):
        join_anyjev(w.task, load_anyjev(_write_dump(tmp_path / "m.json", moved)).items["term.fits"])
    with pytest.raises(CellError, match="format"):
        load_anyjev(_write_dump(tmp_path / "f.json", good, format="other/1"))
    bare = [dict(it) for it in good]
    del bare[0]["state_json"]
    with pytest.raises(CellError, match="no state_json"):
        load_anyjev(_write_dump(tmp_path / "s.json", bare))
    nop = [dict(it) for it in good]
    nop[0]["p_yes"] = None
    with pytest.raises(CellError, match="probability"):
        load_anyjev(_write_dump(tmp_path / "p.json", nop))
    with pytest.raises(CellError, match="unknown AnyJev task"):
        load_anyjev(_write_dump(tmp_path / "t.json", good, tasks={"neon_keep": {"items": []}}))
    ont = Task("neon_ontology_fits", TASKS["column.ontology_fits"], [], "s", "s")
    with pytest.raises(CellError, match="task_key"):
        join_anyjev(ont, load_anyjev(_write_dump(tmp_path / "k.json", good)).items["term.fits"])


def _stand_in(monkeypatch: pytest.MonkeyPatch, dump_sha256: str = "d" * 64) -> None:
    """A registration for the synthetic labels of ``COMMON``, the synthetic ``FINGERPRINT`` and
    a synthetic AnyJev dump (never the real snapshot's or the real dump's)."""
    monkeypatch.setattr(
        reg,
        "REGISTERED",
        reg.Registration(
            snapshot="synthetic.parquet",
            labels_sha256=COMMON["labels_sha256"],
            labels_content_sha256=COMMON["labels_content_sha256"],
            published="synthetic",
            B=COMMON["B"],
            anyjev_sha256=dump_sha256,
            fingerprints={"clm-latest": FINGERPRINT},
        ),
    )


def test_run_x2_keeps_the_m0_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = x2_world(tmp_path, n_targets=12)
    items = [_dump_item(w.task, i, 0.3) for i in range(len(w.task.items))]
    dump = load_anyjev(_write_dump(tmp_path / "dump.json", items))
    _stand_in(monkeypatch, dump.sha256)  # this synthetic dump is the registered one here
    results = run_x2(
        {"neon_term_fits": w.task},
        index=w.index,
        store=w.store,
        fingerprint=FINGERPRINT,
        anyjev=dump,
        date="2026-10-01",
        registered=True,
        **COMMON,
    )
    assert set(results.cells) == {
        "neon_term_fits.baseline.lookup_prob",
        "neon_term_fits.baseline.pr13@S1",
        "neon_term_fits.baseline.pr13@S1ns",
        "neon_term_fits.baseline.anyjev_l2",
    }
    assert results.name == "x2" and results.environment["sklearn"]
    m0 = baselines_cell(w.task, labels_sha256="a" * 64, labels_content_sha256="b" * 64, B=200)
    assert results.cells["neon_term_fits.baseline.lookup_prob"] == m0
    assert results.cells["neon_term_fits.baseline.anyjev_l2"].diagnostics["join"]["full_set"]
    only_m0 = run_x2({"neon_term_fits": w.task}, date="2026-10-01", registered=True, **COMMON)
    assert set(only_m0.cells) == {"neon_term_fits.baseline.lookup_prob"}
    assert all(c.pre_registered and not c.exploratory for c in results.cells.values())


def test_an_x2_run_off_the_registration_is_unregistered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§13.2: a run its caller marks unregistered, and one on other labels or with another B or
    seed whatever its caller says, writes every cell ``pre_registered: false`` and
    ``exploratory: true`` with its deviations in the notes."""
    w = x2_world(tmp_path, n_targets=12)
    tasks = {"neon_term_fits": w.task}
    told = run_x2(tasks, date="d", registered=False, deviations=["by hand"], **COMMON)
    assert told.notes[0].startswith("NOT the pre-registered run") and "by hand" in told.notes[0]
    assert all(c.exploratory and not c.pre_registered for c in told.cells.values())
    _stand_in(monkeypatch)
    for change, text in (
        ({"B": 100}, "B 100 is not 200"),
        ({"seed": 1}, "seed 1 is not 0"),
        ({"labels_sha256": "c" * 64}, "labels_sha256 cccccccccccc"),
        ({"labels_content_sha256": "c" * 64}, "labels_content_sha256 cccccccccccc"),
    ):
        off = run_x2(tasks, date="d", registered=True, **{**COMMON, **change})
        assert text in off.notes[0], (change, off.notes[0])
        assert all(c.exploratory and not c.pre_registered for c in off.cells.values())
    with pytest.raises(CellError, match="cannot be the registered one"):
        run_x2(tasks, date="d", registered=True, deviations=["x"], **COMMON)


# -- the committed files, label-free ---------------------------------------------------------------------


def test_the_committed_dump_covers_the_snapshot_identities() -> None:
    """The join key works on the real files: every AnyJev item's state derives a D1 identity
    of the frozen snapshot, one item per identity, all of them. Only ``state_json`` is read from
    the dump and only the identity columns from the snapshot (no label, weight or score)."""
    data = json.loads(DUMP.read_text(encoding="utf-8"))
    names = {"neon_term_fits": "term.fits", "neon_ontology_fits": "column.ontology_fits"}
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT DISTINCT task_id, target_sha256, option_key FROM read_parquet(?) "
            "WHERE task_id IN ('term.fits', 'column.ontology_fits')",
            [str(SNAPSHOT)],
        ).fetchall()
    finally:
        con.close()
    snapshot = {(str(t), str(ts), str(o)) for t, ts, o in rows}
    for name, task_id in names.items():
        derived = []
        for it in data["tasks"][name]["items"]:
            ident = identity(task_id, it["state_json"])
            assert ident.task_key == TASKS[task_id].key
            derived.append((task_id, ident.target_sha256, ident.option_key))
        assert len(derived) == len(set(derived))  # one item per identity
        assert set(derived) == {k for k in snapshot if k[0] == task_id}
    assert sum(1 for k in snapshot if k[0] == "term.fits") == 285
    assert sum(1 for k in snapshot if k[0] == "column.ontology_fits") == 190


# -- the second review round: interleaved cards, the dump's registration, refusals --------------


def test_the_replica_on_interleaved_cards_is_scikit_learn_by_identity(tmp_path: Path) -> None:
    """Review finding: the replica pools fold by fold; in the bench's identity order (cards
    interleaved) every item's prediction is still scikit-learn's for its own fold."""
    sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    linear = pytest.importorskip("sklearn.linear_model")
    w = x2_world(tmp_path)
    ids0 = item_identities(w.task)
    order = sorted(range(len(ids0)), key=lambda i: ids0[i])
    t = w.task
    task = Task(
        t.name, t.spec, [t.items[i] for i in order], "s", "s",
        cards=[t.cards[i] for i in order], weights=[t.weights[i] for i in order],
        products=[t.products[i] for i in order], option_keys=[t.option_keys[i] for i in order],
        meta=t.meta,
    )  # fmt: skip
    assert sum(1 for a, b in zip(task.cards, task.cards[1:], strict=False) if a != b) > 50
    cell = replica_cell(task, w.index, w.store, "joint4096@S1", fingerprint=None, **COMMON)
    ids = item_identities(task)
    x = np.asarray(
        w.store.get([w.index.text("term.fits", "joint4096@S1", a, o, "context") for a, o in ids]),
        dtype=np.float64,
    )
    y = np.asarray(task.labels)
    got = {(i.target_sha256, i.option_key): (i.label, i.probs) for i in cell.items or []}
    for fold in task.leave_one_card_out():
        ref = sklearn_pipeline.make_pipeline(
            preprocessing.StandardScaler(), linear.LogisticRegression(C=1.0, max_iter=2000)
        ).fit(x[fold.train], y[fold.train])
        p_yes = ref.predict_proba(x[fold.test])[:, list(ref.classes_).index(0)]
        for i, p in zip(fold.test, p_yes, strict=True):
            label, probs = got[ids[i]]
            assert label == y[i] and probs[0] == p


def test_x2_registers_only_the_registered_dump_and_the_whole_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§13.1-§13.2 in the producer (review finding: a library call with another dump wrote
    pre-registered AnyJev cells): another dump, no dump, no replica, another encoder or serving
    lock in the fingerprint are deviations, whatever the caller says."""
    w = x2_world(tmp_path, n_targets=12)
    tasks = {"neon_term_fits": w.task}
    items = [_dump_item(w.task, i, 0.3) for i in range(len(w.task.items))]
    dump = load_anyjev(_write_dump(tmp_path / "dump.json", items))
    other = load_anyjev(_write_dump(tmp_path / "other.json", items, level="L1"))
    assert other.sha256 != dump.sha256
    _stand_in(monkeypatch, dump.sha256)
    full: dict[str, Any] = {"index": w.index, "store": w.store, "fingerprint": FINGERPRINT}
    ok = run_x2(tasks, date="d", registered=True, anyjev=dump, **full, **COMMON)
    assert all(c.pre_registered and not c.exploratory for c in ok.cells.values())
    moved = {**FINGERPRINT, "serving_lock_sha": "7" * 64}
    for kw, text in (
        ({"anyjev": other, **full}, f"AnyJev L2 dump sha256 {other.sha256[:12]} is not"),
        (full, "no AnyJev L2 dump"),
        ({"anyjev": dump}, "no PR #13 replica"),
        ({"anyjev": dump, **full, "fingerprint": moved}, "serving_lock_sha 777777777777"),
        ({"anyjev": dump, **full, "fingerprint": None}, "clm-latest: no fingerprint"),
        ({"anyjev": dump, **full, "specs": ["joint4096@S1"]}, "specs ['joint4096@S1']"),
    ):
        off = run_x2(tasks, date="d", registered=True, **kw, **COMMON)
        assert text in off.notes[0], (text, off.notes[0])
        assert all(c.exploratory and not c.pre_registered for c in off.cells.values())
    # the clm_model_fp of the bundle is not compared: the replica reads no head
    headless = {**FINGERPRINT, "clm_model_fp": "f" * 12}
    same = run_x2(tasks, date="d", registered=True, anyjev=dump, **{**full, "fingerprint": headless},
                  **COMMON)  # fmt: skip
    assert all(c.pre_registered for c in same.cells.values())


def test_anyjev_probabilities_and_folds_without_a_prediction(tmp_path: Path) -> None:
    """A dump probability outside [0, 1] is refused; a card with no AnyJev prediction is
    skipped (``no_anyjev_prediction``), never an empty evaluated fold (review finding)."""
    w = x2_world(tmp_path, n_targets=4)
    good = [_dump_item(w.task, i, 0.4) for i in range(len(w.task.items))]
    above = [dict(it) for it in good]
    above[0]["p_yes"] = 1.5
    with pytest.raises(CellError, match=r"probability in \[0, 1\]"):
        load_anyjev(_write_dump(tmp_path / "above.json", above))
    below = [dict(it) for it in good]
    below[1]["p_yes"] = -0.01
    with pytest.raises(CellError, match="probability"):
        load_anyjev(_write_dump(tmp_path / "below.json", below))
    gone = [it for i, it in enumerate(good) if w.task.cards[i] != CARDS[2]]
    dump = load_anyjev(_write_dump(tmp_path / "gone.json", gone))
    cell = anyjev_cell(w.task, join_anyjev(w.task, dump.items["term.fits"]), dump, **COMMON)
    assert cell.skipped_folds == {CARDS[2]: "no_anyjev_prediction"} and cell.n_folds == 6
    assert cell.counts.n == len(gone) and CARDS[2] not in {i.card for i in cell.items or []}
