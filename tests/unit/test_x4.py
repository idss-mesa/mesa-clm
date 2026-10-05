"""X4 (``mesa_clm.bench.x4``; plan §5.6 X4; the M4 analysis plan R3) on planted feature sources
(``test_probe``): the three arms run on identical outer folds, the teacher rows change the
on-arm fits and never reach a test fold, the silver-minus-Opus cells score the same
predictions on the surviving subset, the decision record applies rule R on the novel-key NLL
for both scorings with A = on and B = off (``delta`` = NLL_off − NLL_on, recomputed directly)
and a planted world where the teacher rows help gives ``keep == True``, ``teacher: true`` on
the on-arms, and the registration (every pin UNPINNED makes the run unregistered, naming each;
a stand-in with the pins makes it registered; the corpus hash is compared only when a corpus
was hashed). No silver label meets a model output here."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mesa_clm import framings
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench import x4 as X4
from mesa_clm.bench.cells import CellError
from mesa_clm.bench.tasks.base import Task
from mesa_clm.learn.labels import TeacherRows
from mesa_clm.learn.probe import Grid
from mesa_clm.tasks import TASKS
from tests.unit.test_cells import FINGERPRINTS, ZERO
from tests.unit.test_probe import FakeSource, planted_source, rank_task

B = 200
IDENTITY: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64}
GRID = Grid(("lowdim.v1", "pair4096.v1"), ("logreg",), {"logreg": ({"lambda": 1.0},)})
TEACHER_HASHES = ("1" * 64, "2" * 64)
MINUS_HASHES = ("3" * 64, "4" * 64)


def _teacher(
    task: Task,
    s: np.ndarray,
    n: int = 60,
    seed: int = 11,
    *,
    mu: float = 2.5,
    sd: float = 1.0,
    signal: str = "lowdim.v1",
) -> TeacherRows:
    """Teacher rows of two products (one of them a bench product): Yes rows whose planted
    ``signal`` column is high (``N(mu, sd)``), with a planted feature source attached."""
    rng = np.random.default_rng(seed)
    products = [task.products[0], "DP9.99999.001"]
    states, cards, groups, keys = [], [], [], []
    for i in range(n):
        p = products[i % 2]
        states.append(
            {
                "card": {"dataset": f"{p}.teach"},
                "scope": "column",
                "aspect": "measurement",
                "column": {"name": f"tc{i}"},
                "candidate": {"curie": f"T:{i}"},
            }
        )
        cards.append(f"{p}.teach")
        groups.append(p)
        keys.append(f"T:{i}")
    scores = rng.normal(mu, sd, n)  # Yes rows: high planted score
    rows = TeacherRows(
        task_id="term.fits",
        states=states,
        labels=[0] * n,
        weights=[0.5] * n,
        sources=["teacher"] * n,
        cards=cards,
        leak_group=groups,
        target_sha256=[f"{i:064x}" for i in range(n)],
        option_key=keys,
    )
    src = planted_source(scores, seed=seed + 1, signal=signal)
    return rows.with_builder(src)


PIN_FIELDS = (
    "teacher_corpus_sha256",
    "neon_ducklake_commit",
    "teacher_snapshot",
    "teacher_labels_sha256",
    "teacher_labels_content_sha256",
    "minus_opus_snapshot",
    "minus_opus_labels_sha256",
    "minus_opus_labels_content_sha256",
)


def _minus(task: Task, drop: set[int], flip: set[int]) -> Task:
    """The task rebuilt without Opus: some identities gone, some labels flipped."""
    items = [
        (st, (1 - y) if i in flip else y) for i, (st, y) in enumerate(task.items) if i not in drop
    ]
    keep = [i for i in range(len(task.items)) if i not in drop]
    return Task(
        task.name, task.spec, items, "s", "s", cards=[task.cards[i] for i in keep],
        products=[task.products[i] for i in keep], option_keys=[task.option_keys[i] for i in keep],
        meta=dict(task.meta),
    )  # fmt: skip


def _standin(monkeypatch: pytest.MonkeyPatch, *, pins: bool) -> None:
    monkeypatch.setattr(
        reg,
        "REGISTERED",
        reg.Registration(
            snapshot="s",
            labels_sha256=IDENTITY["labels_sha256"],
            labels_content_sha256=IDENTITY["labels_content_sha256"],
            published="p",
            B=B,
            framings_lock_sha=framings.lock_sha(),
            fingerprints={m: dict(fp) for m, fp in FINGERPRINTS.items()},
        ),
    )
    grid = {
        **dict(reg.X3_GRID),
        "specs": {"rank_fit": GRID.specs, "choice": ("choice.state.v1", "choice.raw.v1")},
        "fitters": GRID.fitters,
        "hypers": {"logreg": (1.0,)},
    }
    # pins=False passes every pin field explicitly as UNPINNED (the real registration is
    # filled, so the "UNPINNED -> unregistered" path needs the stand-in to say so).
    extra = (
        {
            "teacher_labels_sha256": TEACHER_HASHES[0],
            "teacher_labels_content_sha256": TEACHER_HASHES[1],
            "minus_opus_labels_sha256": MINUS_HASHES[0],
            "minus_opus_labels_content_sha256": MINUS_HASHES[1],
            "teacher_snapshot": "t.parquet",
            "minus_opus_snapshot": "m.parquet",
            "teacher_corpus_sha256": "c" * 64,
            "neon_ducklake_commit": "d" * 40,
        }
        if pins
        else dict.fromkeys(PIN_FIELDS, reg.UNPINNED)
    )
    monkeypatch.setattr(
        reg,
        "REGISTERED_M4",
        reg.RegistrationM4(framings_lock_sha=framings.lock_sha(), grid=grid, **extra),
    )


def test_the_arms_share_folds_and_teacher_rows_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _standin(monkeypatch, pins=True)
    task, s = rank_task(40, seed=5)
    fb = planted_source(s, signal="lowdim.v1")
    teacher = _teacher(task, s)
    minus = _minus(task, drop={0, 1, 2}, flip={3, 4})
    cells, decision = X4.task_x4_cells(
        task,
        fb,
        teacher,
        FINGERPRINTS,
        registered=True,
        minus=minus,
        grid=GRID,
        B=B,
        clock=ZERO,
        **IDENTITY,
    )
    variants = ["teacher-off", "teacher-0.5", "teacher-0.3"]
    assert set(cells) == {
        f"neon_term_fits.probe.F7@{v}{sfx}" for v in variants for sfx in ("", "-minus-opus")
    }
    off = cells["neon_term_fits.probe.F7@teacher-off"]
    on = cells["neon_term_fits.probe.F7@teacher-0.5"]
    low = cells["neon_term_fits.probe.F7@teacher-0.3"]
    # identical outer folds: the same held-out items, pooled in the same order
    assert [(i.target_sha256, i.option_key, i.card) for i in off.items or []] == [
        (i.target_sha256, i.option_key, i.card) for i in on.items or []
    ]
    assert off.n_folds == on.n_folds == low.n_folds == 7
    # teacher rows: never pooled, counted per fold, the held-out product's rows dropped
    assert not off.teacher and on.teacher and low.teacher
    assert all(not c.teacher_in_test for c in cells.values())
    assert (
        on.diagnostics["teacher_rows_available"] == 60
        and on.diagnostics["teacher_rows_weighted"] == 60
    )
    assert off.diagnostics["teacher_rows_weighted"] == 0 and off.diagnostics["teacher"] is False
    for card, info in on.diagnostics["per_fold"].items():
        product = card.rsplit(".", 1)[0]
        assert info["n_teacher"] == sum(1 for g in teacher.leak_group if g != product)
    assert on.diagnostics["teacher_weights"] == {"teacher": 0.5, "teacher_implicit": 0.3}
    assert low.diagnostics["teacher_weights"] == {"teacher": 0.3, "teacher_implicit": 0.1}
    assert on.counts.n == len(task.items) and on.label_sources == off.label_sources
    # the teacher rows change the fit (D20): the on-arm's predictions differ from the off-arm's
    assert any(
        abs(a.probs[0] - b.probs[0]) > 1e-6
        for a, b in zip(off.items or [], on.items or [], strict=True)
    )
    assert any(
        abs(a.probs[0] - b.probs[0]) > 1e-9
        for a, b in zip(on.items or [], low.items or [], strict=True)
    )
    # every cell is a variant, pre-registered, exploratory, never citable; identity = the off-arm choice
    for c in cells.values():
        assert (
            c.tier == "probe"
            and c.selection == "nested"
            and c.pre_registered
            and c.exploratory
            and c.variant
        )
        assert c.feature_spec == off.feature_spec == "lowdim.v1" and c.model == off.model
        assert c.diagnostics["teacher_decision"] == decision
    # silver-minus-Opus: the same predictions on the surviving items (not dropped, not flipped)
    survivors = X4.minus_opus_subset(task, minus)
    assert survivors == [i for i in range(len(task.items)) if i not in {0, 1, 2, 3, 4}]
    m = cells["neon_term_fits.probe.F7@teacher-0.5-minus-opus"]
    assert m.counts.n == len(survivors) and m.diagnostics["scoring"] == "silver-minus-opus"
    by_id = {(i.target_sha256, i.option_key): i.probs for i in on.items or []}
    assert all(by_id[(i.target_sha256, i.option_key)] == i.probs for i in m.items or [])
    assert m.diagnostics["subset_n"] == len(survivors)
    # the decision record: rule R on novel-key NLL for both scorings, the (0.5, 0.3) arm decides
    assert decision["decision_arm"] == "teacher-0.5" and set(decision["arms"]) == {
        "teacher-0.5",
        "teacher-0.3",
    }
    for arm in decision["arms"].values():
        for scoring in X4.SCORINGS:
            rec = arm[scoring]
            assert (
                rec["metric"] == "nll" and "lower_bound" in rec and rec["n_novel"] == rec["n"]
            )  # every key novel
    assert decision["keep"] == all(decision["keep_detail"].values())
    assert set(decision["keep_detail"]) == set(X4.SCORINGS)
    assert decision["keep"] == (
        decision["arms"]["teacher-0.5"]["silver"]["passed"]
        and decision["arms"]["teacher-0.5"]["silver-minus-opus"]["passed"]
    )


def _nll(cell: Any, keep: set[tuple[str, str]] | None = None) -> float:
    items = [i for i in cell.items or [] if keep is None or (i.target_sha256, i.option_key) in keep]
    return float(stats.metric_value("nll", [i.probs for i in items], [i.label for i in items]))


def test_the_decision_direction_and_a_world_where_the_teacher_helps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A = on, B = off: ``delta`` is NLL_off − NLL_on over the novel items (every key is novel
    here), positive when the teacher rows help. A planted world with few silver items per card,
    a 12-d spec and 200 informative teacher rows: on ≻ off on both scorings, ``keep`` true."""
    _standin(monkeypatch, pins=True)
    task, s = rank_task(18, a=3.0, b=0.0, seed=5)
    grid = Grid(("pair4096.v1",), ("logreg",), {"logreg": ({"lambda": 0.01},)})
    fb = planted_source(s, signal="pair4096.v1")
    teacher = _teacher(task, s, n=200, mu=1.5, sd=0.7, signal="pair4096.v1")
    minus = _minus(task, drop={0, 1, 2}, flip={3, 4})
    cells, decision = X4.task_x4_cells(
        task, fb, teacher, FINGERPRINTS, registered=True, minus=minus, grid=grid, B=B,
        clock=ZERO, **IDENTITY,
    )  # fmt: skip
    off = cells["neon_term_fits.probe.F7@teacher-off"]
    on = cells["neon_term_fits.probe.F7@teacher-0.5"]
    rec = decision["arms"]["teacher-0.5"]["silver"]
    assert rec["n_novel"] == rec["n"] == len(task.items)
    assert rec["delta"] == pytest.approx(_nll(off) - _nll(on))
    assert rec["delta"] > 0 and rec["lower_bound"] > 0 and rec["passed"]
    assert rec["sign"]["wins"] >= rec["sign"]["needed"] == 6
    # silver-minus-Opus: the same direction over the surviving items (dropped and flipped gone)
    on_m = cells["neon_term_fits.probe.F7@teacher-0.5-minus-opus"]
    off_m = cells["neon_term_fits.probe.F7@teacher-off-minus-opus"]
    rec_m = decision["arms"]["teacher-0.5"]["silver-minus-opus"]
    survivors = {(i.target_sha256, i.option_key) for i in on_m.items or []}
    assert rec_m["n"] == len(survivors) == len(task.items) - 5
    assert rec_m["delta"] == pytest.approx(_nll(off_m) - _nll(on_m))
    assert rec_m["delta"] == pytest.approx(_nll(off, survivors) - _nll(on, survivors))
    assert rec_m["passed"]
    assert decision["keep"] is True and decision["keep_detail"] == {
        "silver": True,
        "silver-minus-opus": True,
    }
    assert on.diagnostics["teacher_decision"]["keep"] is True


def test_without_minus_opus_or_teacher_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _standin(monkeypatch, pins=True)
    task, s = rank_task(30, seed=8)
    fb = planted_source(s)
    cells, decision = X4.task_x4_cells(
        task,
        fb,
        None,
        FINGERPRINTS,
        registered=True,
        minus=None,
        grid=GRID,
        B=B,
        clock=ZERO,
        **IDENTITY,
    )
    assert set(cells) == {f"neon_term_fits.probe.F7@teacher-{v}" for v in ("off", "0.5", "0.3")}
    off, on = (
        cells["neon_term_fits.probe.F7@teacher-off"],
        cells["neon_term_fits.probe.F7@teacher-0.5"],
    )
    assert [i.probs for i in off.items or []] == [
        i.probs for i in on.items or []
    ]  # no rows: the same fit
    assert on.teacher and on.diagnostics["teacher_rows_weighted"] == 0
    assert decision["arms"]["teacher-0.5"]["silver-minus-opus"] == {
        "reason": "no silver-minus-Opus subset"
    }
    assert decision["keep"] is False
    with pytest.raises(CellError, match="rank_fit"):
        choice = Task(
            "neon_aspect",
            TASKS["column.aspect"],
            [({"card": {"dataset": "c"}, "column": {"name": "x"}}, 0)],
            "s",
            "s",
            cards=["c"],
            meta={"min_weight": 0.6},
        )
        X4.task_x4_cells(
            choice, fb, None, FINGERPRINTS, registered=True, grid=GRID, B=B, **IDENTITY
        )


def _as_ontology_fits(task: Task) -> Task:
    """The same items as a ``column.ontology_fits`` task (the registration runs both)."""
    items = []
    keys = []
    for (st, y), key in zip(task.items, task.option_keys, strict=True):
        oid = f"ont{key.split(':')[1]}"
        items.append(
            (
                {k: v for k, v in st.items() if k not in ("candidate", "scope")}
                | {"ontology": {"id": oid}},
                y,
            )
        )
        keys.append(oid)
    return Task("neon_ontology_fits", TASKS["column.ontology_fits"], items, "s", "s", cards=task.cards,
                weights=task.weights, products=task.products, option_keys=keys, meta=dict(task.meta))  # fmt: skip


def test_run_x4_applies_the_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    task, s = rank_task(30, seed=9)
    onto = _as_ontology_fits(task)
    fb = planted_source(s)
    teacher = _teacher(task, s, n=20)
    minus = _minus(task, drop={0}, flip=set())

    class _Index:
        pass

    def builder(t: Task, *args: Any, **kwargs: Any) -> Any:
        return fb if t in (task, onto) else teacher.builder

    monkeypatch.setattr(X4, "feature_builder", builder)
    tasks = {task.name: task, onto.name: onto}
    common: dict[str, Any] = {
        "teacher": {"term.fits": teacher},
        "minus_opus": {task.name: minus},
        "date": "2026-10-05",
        "grid": GRID,
        "B": B,
        "clock": ZERO,
        **IDENTITY,
    }
    scorers = {"clm-latest": object(), "clm-raw": object()}
    # pins in place: registered
    _standin(monkeypatch, pins=True)
    res = X4.run_x4(
        tasks,
        _Index(),
        None,
        scorers,
        FINGERPRINTS,
        registered=True,
        teacher_hashes=TEACHER_HASHES,
        minus_opus_hashes=MINUS_HASHES,
        **common,
    )  # type: ignore[arg-type]
    assert all(c.pre_registered for c in res.cells.values()), res.notes[0]
    assert (
        len(res.cells) == 6 + 3
    )  # term.fits: 3 arms x 2 scorings; ontology_fits: no minus-Opus task
    assert res.name == "x4" and any(": keep=" in n for n in res.notes)
    # the corpus hash is compared only when a corpus was hashed: the pin's hash is no deviation,
    # another hash is named, and no hash (the corpus may have moved) checks nothing
    res_c = X4.run_x4(
        tasks,
        _Index(),
        None,
        scorers,
        FINGERPRINTS,
        registered=True,
        teacher_hashes=TEACHER_HASHES,
        minus_opus_hashes=MINUS_HASHES,
        teacher_corpus_sha256="c" * 64,
        **common,
    )  # type: ignore[arg-type]
    assert all(c.pre_registered for c in res_c.cells.values())
    res_d = X4.run_x4(
        tasks,
        _Index(),
        None,
        scorers,
        FINGERPRINTS,
        registered=True,
        teacher_hashes=TEACHER_HASHES,
        minus_opus_hashes=MINUS_HASHES,
        teacher_corpus_sha256="9" * 64,
        **common,
    )  # type: ignore[arg-type]
    assert "teacher corpus sha256 999999999999 is not the pinned cccccccccccc" in res_d.notes[0]
    # every pin UNPINNED (the stand-in says so explicitly): unregistered, each pin named
    _standin(monkeypatch, pins=False)
    assert all(getattr(reg.current_m4(), f) == reg.UNPINNED for f in PIN_FIELDS)
    res2 = X4.run_x4(
        tasks,
        _Index(),
        None,
        scorers,
        FINGERPRINTS,
        registered=True,
        teacher_hashes=TEACHER_HASHES,
        minus_opus_hashes=MINUS_HASHES,
        teacher_corpus_sha256="c" * 64,
        **common,
    )  # type: ignore[arg-type]
    assert all(not c.pre_registered and c.exploratory for c in res2.cells.values())
    assert (
        "teacher snapshot is not pinned" in res2.notes[0]
        and "minus_opus snapshot is not pinned" in res2.notes[0]
        and "teacher corpus is not pinned" in res2.notes[0]
    )
    # other teacher bytes than the pin, or no teacher snapshot at all
    _standin(monkeypatch, pins=True)
    res3 = X4.run_x4(
        tasks,
        _Index(),
        None,
        scorers,
        FINGERPRINTS,
        registered=True,
        teacher_hashes=("9" * 64, "2" * 64),
        minus_opus_hashes=None,
        **common,
    )  # type: ignore[arg-type]
    assert "teacher labels_sha256 999999999999 is not 111111111111" in res3.notes[0]
    assert "no silver-minus-Opus snapshot" in res3.notes[0]
    assert reg.pinned_snapshot_deviations("teacher", None, None)
    assert X4.arm_variant(None) == "teacher-off" and X4.arm_variant((0.5, 0.3)) == "teacher-0.5"
    assert X4.arm_weights((0.3, 0.1)) == {"teacher": 0.3, "teacher_implicit": 0.1}
    assert isinstance(fb, FakeSource)
