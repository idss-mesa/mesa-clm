"""The probe tier (``mesa_clm.learn.probe``; plan §5.3 "probe", §5.4, §5.5; DESIGN D18, D19,
D20, D27; the M4 analysis plan "probe"), on synthetic worlds only: the feature-store worlds of
``test_cells`` (planted vectors, generated labels) for the :class:`FeatureBuilder`, and fake
feature sources with a planted signal for the nested procedure. No silver label meets a model
output here.

Checked: every spec matrix is its formula over the vectors ``score_arm`` reads, and the
builder's zero-shot logits equal the tier cells' scores; the nested selection recovers the
planted spec in every fold, ties go to the first configuration in grid order, the inner
criterion is the pooled OOF NLL; the label weights change the fit; teacher rows are never in a
test fold, are dropped for the held-out card's product (outer and inner), and change the fit;
two runs are byte-identical; the 30/5 guards, the 40 floor, the inner floors and the 100 OOF
calibrator floor skip folds with the stated reasons and the report-only diagnostics remain;
an unfittable configuration is recorded and never chosen; ``ProbeArtifact.predict`` equals the
bench's held-out probabilities; the pooled result builds a ``probe`` cell through
``assemble_cell``; the full-data artifact; JSON round trips; refusals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings
from mesa_clm.bench.cells import CellError, assemble_cell, score_arm
from mesa_clm.bench.tasks.base import Task
from mesa_clm.clm.headproj import l2
from mesa_clm.learn import calibrate
from mesa_clm.learn import probe as P
from mesa_clm.learn.calibrate import sigmoid, softmax
from mesa_clm.learn.linear import GRIDS, prepare
from mesa_clm.learn.offline import LATEST_MODEL, RAW_MODEL, store_vectors
from mesa_clm.learn.probe import (
    DEFAULT_GRID,
    SPECS,
    FeatureBuilder,
    Grid,
    ProbeArtifact,
    ProbeConfig,
    ProbeError,
    TeacherRows,
    full_probe,
    inner_select,
    nested_probe,
)
from mesa_clm.tasks import TASKS
from tests.unit.test_cells import (
    CARDS,
    DIM,
    FINGERPRINTS,
    LATEST_FP,
    ZERO,
    World,
    _row,
    _scorers,
    _unit,
    choice_world,
    rank_world,
)

IDENTITY: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64}
FPS = {RAW_MODEL: FINGERPRINTS[RAW_MODEL]["clm_model_fp"], LATEST_MODEL: LATEST_FP}
RANK_SPECS = tuple(s.id for s in SPECS if s.shape == "rank_fit")


# -- fake worlds for the nested procedure ------------------------------------------------------------


class FakeSource:
    """A :class:`~mesa_clm.learn.probe.FeatureSource` of planted matrices (item order)."""

    def __init__(self, mats: dict[str, np.ndarray], *, shape: str = "rank_fit", k: int = 2) -> None:
        self.mats = mats
        self._shape = shape
        self._k = k
        self.n = next(iter(mats.values())).shape[0]
        self.question_key = "q" * 16
        self.framing_id = "F7"
        self.calls: list[str] = []

    @property
    def shape(self) -> Any:
        return self._shape

    @property
    def k(self) -> int:
        return self._k

    def matrix(self, spec_id: str) -> np.ndarray:
        self.calls.append(spec_id)
        return self.mats[spec_id]


def rank_task(
    n_per_card: int = 45, *, a: float = 1.8, b: float = -0.5, seed: int = 0, weights: bool = True
) -> tuple[Task, np.ndarray]:
    """A term.fits task of ``7 × n_per_card`` labelled pairs with a planted score ``s`` per
    item (Yes ~ σ(a·s + b)); every lookup key is novel."""
    rng = np.random.default_rng(seed)
    states: list[dict[str, Any]] = []
    cards: list[str] = []
    options: list[str] = []
    scores: list[float] = []
    for ci, card in enumerate(CARDS):
        for t in range(n_per_card):
            states.append(
                {
                    "card": {"dataset": card},
                    "scope": "column",
                    "aspect": "measurement",
                    "column": {"name": f"col{t}"},
                    "candidate": {"curie": f"X:{ci}{t}"},
                }
            )
            cards.append(card)
            options.append(f"X:{ci}{t}")
            scores.append(float(rng.normal(0.0, 2.0)))
    s = np.asarray(scores)
    y = np.where(rng.random(len(s)) < sigmoid(a * s + b), 0, 1)
    task = Task(
        "neon_term_fits",
        TASKS["term.fits"],
        [(st, int(lbl)) for st, lbl in zip(states, y, strict=True)],
        "synthetic",
        "synthetic",
        cards=cards,
        weights=[0.6 if lbl == 0 else 0.5 for lbl in y] if weights else [],
        products=[c.rsplit(".", 1)[0] for c in cards],
        option_keys=options,
        meta={
            "min_weight": 0.5,
            "masked": True,
            "mask": None,
            "label_sources": {
                "consensus_majority": int((y == 0).sum()),
                "consensus_negative": int((y == 1).sum()),
            },
            "excluded": {},
        },
    )
    return task, s


def planted_source(s: np.ndarray, seed: int = 1, *, signal: str = "lowdim.v1") -> FakeSource:
    """Every rank_fit spec is noise except ``signal``, whose first column is the planted score."""
    rng = np.random.default_rng(seed)
    n = len(s)
    dims = {
        "lowdim.v1": 2,
        "pair512.v1": 9,
        "pair4096.v1": 12,
        "joint4096@S1": 10,
        "joint4096@S1ns": 10,
    }
    mats = {spec: rng.normal(size=(n, d)) for spec, d in dims.items()}
    mats[signal][:, 0] = s
    return FakeSource(mats)


def _relabel(task: Task, labels: list[int], weights: list[float] | None = None) -> Task:
    return Task(
        task.name,
        task.spec,
        [(st, int(y)) for (st, _), y in zip(task.items, labels, strict=True)],
        "s",
        "s",
        cards=task.cards,
        weights=list(weights) if weights is not None else task.weights,
        products=task.products,
        option_keys=task.option_keys,
        meta=dict(task.meta),
    )


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


# -- the feature builder ------------------------------------------------------------------------------


def _with_joint_rows(w: World, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Add the X2 joint context rows (one per labelled pair) with random unit vectors."""
    ids = [(w.task.option_keys[i], st) for i, (st, _) in enumerate(w.task.items)]
    texts: list[str] = []
    vecs: list[np.ndarray] = []
    joint: dict[str, list[np.ndarray]] = {s: [] for s in ("joint4096@S1", "joint4096@S1ns")}
    from mesa_clm.identity import target_sha256

    for key, st in ids:
        target = target_sha256("term.fits", {k: v for k, v in st.items() if k != "candidate"})
        for spec in joint:
            text = f"{spec} joint {key}"
            v = _unit(rng)
            texts.append(text)
            vecs.append(v)
            joint[spec].append(v)
            w.rows.append(_row("term.fits", spec, target, key, "context", text))
    w.store.add(texts, np.stack(vecs), [3] * len(texts))
    return {s: l2(np.stack(v)).astype(np.float64) for s, v in joint.items()}


def test_rank_fit_matrices_are_the_spec_formulas_over_score_arms_vectors(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=4)
    joint = _with_joint_rows(w, np.random.default_rng(11))
    scorers = _scorers(w.store)
    fb = FeatureBuilder(w.task, w.index, w.store, scorers, "F7", clock=ZERO)
    assert fb.shape == "rank_fit" and fb.k == 2 and fb.n == len(w.task.items)
    assert fb.question_key == framings.framing("term.fits", "F7").question_key
    assert fb.framing_id == "F7" and fb.task_id == "term.fits"
    sc = {m: score_arm(w.task, w.index, w.store, scorers[m], "F7", clock=ZERO) for m in scorers}
    for m, arm in sc.items():
        np.testing.assert_array_equal(fb.zero_shot_logits[m], arm.x)
    ids = fb.ids
    ctx = [w.index.text("term.fits", "F7", t, "", "context") for t, _ in ids]
    cand = [w.index.text("term.fits", "F7", t, o, "candidate") for t, o in ids]
    zs = store_vectors(scorers[LATEST_MODEL], w.store, "state", ctx).astype(np.float64)
    zc = store_vectors(scorers[LATEST_MODEL], w.store, "action", cand).astype(np.float64)
    xs = store_vectors(scorers[RAW_MODEL], w.store, "state", ctx).astype(np.float64)
    xc = store_vectors(scorers[RAW_MODEL], w.store, "action", cand).astype(np.float64)
    s_latest, s_raw = sc[LATEST_MODEL].x, sc[RAW_MODEL].x
    want = {
        "lowdim.v1": np.stack([s_latest, s_raw], axis=1),
        "pair512.v1": np.hstack([zs * zc, np.abs(zs - zc), s_latest[:, None]]),
        "pair4096.v1": np.hstack([xs * xc, s_raw[:, None]]),
        "joint4096@S1": joint["joint4096@S1"],
        "joint4096@S1ns": joint["joint4096@S1ns"],
    }
    for spec, x in want.items():
        got = fb.matrix(spec)
        assert got.dtype == np.float64 and got.shape == x.shape, spec
        np.testing.assert_array_equal(got, x)
        assert fb.matrix(spec) is got  # cached
        diag = fb.spec_diagnostics(spec)
        assert diag["model"] == P.SPEC_BY_ID[spec].model and diag["dim"] == x.shape[1]
        assert diag["calls"] > 0 and diag["input_tokens"] > 0 and -1 <= diag["mean_state_cos"] <= 1
    assert (
        fb.matrix("pair512.v1").shape[1] == 2 * 16 + 1
        and fb.matrix("pair4096.v1").shape[1] == DIM + 1
    )
    assert fb.arm_diagnostics(RAW_MODEL) == {
        **sc[RAW_MODEL].diagnostics,
        "ms_per_decision": 0.0,
    }
    with pytest.raises(ProbeError, match="is a choice spec"):
        fb.matrix("choice.state.v1")
    with pytest.raises(ProbeError, match="unknown spec"):
        fb.matrix("nope")


def test_choice_matrices_are_the_state_vector_and_the_option_logits(tmp_path: Path) -> None:
    w = choice_world(tmp_path, "avu.value_kind", n_targets=3)
    scorers = _scorers(w.store)
    fb = FeatureBuilder(w.task, w.index, w.store, scorers, "F7", clock=ZERO)
    assert fb.shape == "choice" and fb.k == 4
    ctx = [w.index.text("avu.value_kind", "F7", t, "", "context") for t, _ in fb.ids]
    for spec, model in (("choice.state.v1", LATEST_MODEL), ("choice.raw.v1", RAW_MODEL)):
        arm = score_arm(w.task, w.index, w.store, scorers[model], "F7", clock=ZERO)
        states = store_vectors(scorers[model], w.store, "state", ctx).astype(np.float64)
        np.testing.assert_array_equal(fb.matrix(spec), np.hstack([states, arm.x]))
        np.testing.assert_array_equal(fb.zero_shot_logits[model], arm.x)
    assert fb.matrix("choice.state.v1").shape == (21, 16 + 4)
    assert fb.matrix("choice.raw.v1").shape == (21, DIM + 4)
    with pytest.raises(ProbeError, match="is a rank_fit spec"):
        fb.matrix("lowdim.v1")


def test_builder_refusals(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=2)
    scorers = _scorers(w.store)
    with pytest.raises(CellError, match="control framing"):
        FeatureBuilder(w.task, w.index, w.store, scorers, "F1")
    with pytest.raises(CellError, match="no items"):
        FeatureBuilder(
            Task("neon_term_fits", TASKS["term.fits"], [], "s", "s"),
            w.index,
            w.store,
            scorers,
            "F7",
        )
    with pytest.raises(CellError, match="no context text"):
        FeatureBuilder(w.task, w.index, w.store, scorers, "F9")
    fb = FeatureBuilder(w.task, w.index, w.store, {RAW_MODEL: scorers[RAW_MODEL]}, "F7")
    with pytest.raises(CellError, match="no scorer for model"):
        fb.matrix("lowdim.v1")
    assert fb.matrix("pair4096.v1").shape[1] == DIM + 1
    with pytest.raises(CellError, match="no context text"):
        fb.matrix("joint4096@S1")  # the index has no joint rows


def test_spec_formulas_and_the_registered_dimensions() -> None:
    zs, zc = np.array([[1.0, 2.0]]), np.array([[3.0, -1.0]])
    np.testing.assert_array_equal(
        P.rank_fit_features("pair512.v1", zs=zs, zc=zc, s_latest=[0.5]),
        [[3.0, -2.0, 2.0, 3.0, 0.5]],
    )
    np.testing.assert_array_equal(
        P.rank_fit_features("lowdim.v1", s_latest=[1.0], s_raw=[2.0]), [[1.0, 2.0]]
    )
    np.testing.assert_array_equal(
        P.rank_fit_features("pair4096.v1", xs=zs, xc=zc, s_raw=[7.0]), [[3.0, -2.0, 7.0]]
    )
    np.testing.assert_array_equal(P.rank_fit_features("joint4096@S1ns", joint=zs), zs)
    np.testing.assert_array_equal(
        P.choice_features("choice.raw.v1", xs=zs, logits=[[1.0, 2.0, 3.0]]),
        [[1.0, 2.0, 1.0, 2.0, 3.0]],
    )
    with pytest.raises(ProbeError, match="needs s_raw"):
        P.rank_fit_features("lowdim.v1", s_latest=[1.0])
    with pytest.raises(ProbeError, match="needs zs"):
        P.choice_features("choice.state.v1", xs=zs, logits=[[1.0, 2.0]])
    with pytest.raises(ProbeError, match="aligned"):
        P.choice_features("choice.state.v1", zs=zs, logits=[[1.0], [2.0]])
    dims = {s.id: s.dim(8) for s in SPECS}
    assert dims == {
        "lowdim.v1": 2,
        "pair512.v1": 1025,
        "pair4096.v1": 4097,
        "joint4096@S1": 4096,
        "joint4096@S1ns": 4096,
        "choice.state.v1": 520,
        "choice.raw.v1": 4104,
    }
    assert {s.id: s.model for s in SPECS} == {
        "lowdim.v1": LATEST_MODEL,
        "pair512.v1": LATEST_MODEL,
        "pair4096.v1": RAW_MODEL,
        "joint4096@S1": RAW_MODEL,
        "joint4096@S1ns": RAW_MODEL,
        "choice.state.v1": LATEST_MODEL,
        "choice.raw.v1": RAW_MODEL,
    }


# -- the grid -----------------------------------------------------------------------------------------------


def test_grid_order_restriction_and_refusals() -> None:
    g = DEFAULT_GRID["rank_fit"]
    assert (
        g.specs == RANK_SPECS and g.fitters == ("logreg", "lda", "ridge") and g.shape == "rank_fit"
    )
    configs = g.configurations()
    assert len(configs) == 5 * (4 + 3 + 3)
    assert [c.spec for c in configs[:10]] == ["lowdim.v1"] * 10
    assert [(c.fitter, c.hyper) for c in configs[:10]] == [
        *[("logreg", h) for h in GRIDS["logreg"]],
        *[("lda", h) for h in GRIDS["lda"]],
        *[("ridge", h) for h in GRIDS["ridge"]],
    ]
    assert g.restrict(LATEST_MODEL).specs == ("lowdim.v1", "pair512.v1")
    assert g.restrict(RAW_MODEL).specs == ("pair4096.v1", "joint4096@S1", "joint4096@S1ns")
    assert DEFAULT_GRID["choice"].specs == ("choice.state.v1", "choice.raw.v1")
    assert DEFAULT_GRID["choice"].restrict(RAW_MODEL).specs == ("choice.raw.v1",)
    assert g.as_dict()["hypers"]["lda"] == [{"gamma": 0.1}, {"gamma": 0.5}, {"gamma": 0.9}]
    with pytest.raises(ProbeError, match="share one shape"):
        Grid(("lowdim.v1", "choice.raw.v1"), ("logreg",), GRIDS)
    with pytest.raises(ProbeError, match="unknown fitter"):
        Grid(("lowdim.v1",), ("svm",), GRIDS)
    with pytest.raises(ProbeError, match="no hyperparameter grid"):
        Grid(("lowdim.v1",), ("logreg",), {"logreg": ()})
    with pytest.raises(ProbeError, match="unknown spec"):
        Grid(("nope",), ("logreg",), GRIDS)
    with pytest.raises(ProbeError, match="no spec of"):
        DEFAULT_GRID["choice"].restrict("other")
    c = ProbeConfig.of({"spec": "lowdim.v1", "fitter": "lda", "hyper": {"gamma": 0.5}, "x": 1})
    assert c == ProbeConfig("lowdim.v1", "lda", {"gamma": 0.5}) and c.model == LATEST_MODEL


# -- the nested procedure -----------------------------------------------------------------------------------


def test_the_nested_selection_recovers_the_planted_spec_in_every_fold() -> None:
    task, s = rank_task()
    fb = planted_source(s)
    res = nested_probe(task, fb, grid=DEFAULT_GRID["rank_fit"], clm_model_fps=FPS)
    assert res.n_folds == 7 and res.skipped_folds == {} and len(res.pooled) == len(task.items)
    assert set(res.fold_choices) == set(CARDS) and res.probs.shape == (len(task.items), 2)
    for card, choice in res.fold_choices.items():
        assert choice["decision"] == "configuration" and choice["evaluated"] is True, card
        assert choice["spec"] == "lowdim.v1" and choice["model"] == LATEST_MODEL
        assert choice["clm_model_fp"] == LATEST_FP and choice["fitter"] in (
            "logreg",
            "lda",
            "ridge",
        )
        assert choice["inner_n"] == len(task.items) - 45 and choice["inner_skips"] == {}
        assert choice["inner_folds"] == 6 and choice["n_teacher"] == 0
        cal = choice["calibrator"]
        assert (
            cal["kind"] == "platt"
            and set(cal["params"]) == {"a", "b"}
            and cal["n"] == choice["inner_n"]
        )
        assert cal["converged"] is True and cal["inverted"] is False and cal["params"]["a"] > 0.5
        grid = res.diagnostics["per_fold"][card]["inner_grid"]
        assert len(grid) == 50 and all(g["error"] is None for g in grid)
        nlls = {g["spec"]: min(x["nll"] for x in grid if x["spec"] == g["spec"]) for g in grid}
        assert nlls["lowdim.v1"] == choice["inner_nll"] == min(nlls.values())
        assert all(nlls[spec] > nlls["lowdim.v1"] + 0.2 for spec in nlls if spec != "lowdim.v1")
        # The winner is the configuration with the lowest inner NLL (strictly, in grid order).
        best = min(grid, key=lambda g: g["nll"])
        assert (best["spec"], best["fitter"], best["hyper"]) == (
            choice["spec"],
            choice["fitter"],
            choice["hyper"],
        )
    d = res.diagnostics
    assert d["calibration"] == "platt" and d["selection"] == "nested" and d["teacher"] is False
    assert d["teacher_in_test"] is False and d["floors"] == {"probe": 40, "calibrator_oof": 100}
    assert d["chosen_specs"] == ["lowdim.v1"] and d["fitters"]["logreg"]["max_iter"] == 100
    assert (
        d["uncalibrated_pooled"]["n"] == len(task.items) and d["uncalibrated_pooled"]["nll"] < 0.5
    )
    for info in d["per_fold"].values():
        assert set(info["uncalibrated_heldout"]) == {"nll", "ece", "acc", "brier", "in_the_large"}
        assert info["linear"]["converged"] is True and info["linear"]["d"] == 2
    # Pooled, never averaged: every held-out item's probability came from its own fold's fit.
    cell = assemble_cell(
        task,
        res.pooled,
        res.probs,
        res.folds,
        tier="probe",
        framing="F7",
        selection="nested",
        pre_registered=True,
        exploratory=False,
        servable=True,
        model=LATEST_MODEL,
        feature_spec="lowdim.v1",
        question_key=fb.question_key,
        fingerprint=FINGERPRINTS[LATEST_MODEL],
        skipped_folds=res.skipped_folds,
        fold_choices=res.fold_choices,
        diagnostics=res.diagnostics,
        B=100,
        **IDENTITY,
    )
    assert cell.tier == "probe" and cell.n_folds == 7 and cell.counts.n == len(task.items)
    assert cell.metrics is not None and cell.metrics.nll < 0.4 and cell.metrics.acc > 0.8
    assert cell.baselines is not None and cell.baselines.beats_lookup_novel is True
    assert cell.fold_choices[CARDS[0]]["spec"] == "lowdim.v1" and cell.teacher_in_test is False
    json.dumps(cell.model_dump(by_alias=True, mode="json"), allow_nan=False)


def test_ties_go_to_the_first_configuration_in_grid_order() -> None:
    task, s = rank_task(30)
    fb = planted_source(s)
    fb.mats["pair512.v1"] = fb.mats["lowdim.v1"].copy()  # identical matrices: identical fits
    grid = Grid(("pair512.v1", "lowdim.v1"), ("lda", "logreg"), GRIDS)  # declared: pair512 first
    res = nested_probe(task, fb, grid=grid)
    for card, choice in res.fold_choices.items():
        inner = res.diagnostics["per_fold"][card]["inner_grid"]
        by = {(g["spec"], g["fitter"], _dump(g["hyper"])): g["nll"] for g in inner}
        for (spec, fitter, hyper), nll in by.items():
            if spec == "pair512.v1":
                assert by[("lowdim.v1", fitter, hyper)] == nll  # exact ties
        assert choice["spec"] == "pair512.v1", card
    reversed_grid = Grid(("lowdim.v1", "pair512.v1"), ("lda", "logreg"), GRIDS)
    again = nested_probe(task, fb, grid=reversed_grid)
    assert all(c["spec"] == "lowdim.v1" for c in again.fold_choices.values())
    for card in CARDS:  # the same fit, so the same predictions, whichever name won
        np.testing.assert_array_equal(
            res.probs[
                [res.pooled.index(i) for i in range(len(task.items)) if task.cards[i] == card]
            ],
            again.probs[
                [again.pooled.index(i) for i in range(len(task.items)) if task.cards[i] == card]
            ],
        )


def test_the_label_weights_change_the_fit() -> None:
    task, s = rank_task(30, seed=3)
    fb = planted_source(s)
    grid = Grid(("lowdim.v1",), ("logreg",), {"logreg": ({"lambda": 0.1},)})
    base = nested_probe(task, fb, grid=grid)
    heavy = nested_probe(
        _relabel(task, task.labels, [3.0 if y == 0 else 0.5 for y in task.labels]), fb, grid=grid
    )
    plain = nested_probe(_relabel(task, task.labels, []), fb, grid=grid)
    assert base.probs.shape == heavy.probs.shape and not np.array_equal(base.probs, heavy.probs)
    assert not np.array_equal(base.probs, plain.probs)

    def shift(res: Any, card: str) -> float:
        cal = res.fold_choices[card]["calibrator"]["params"]
        return float(cal["a"] * res.fits[card].linear.intercept[0] + cal["b"])

    for card in CARDS:  # heavier Yes weights raise the calibrated intercept a·b_probe + b
        assert shift(heavy, card) > shift(base, card) + 1.0
        assert heavy.fits[card].linear.intercept[0] > base.fits[card].linear.intercept[0] + 1.0
    assert heavy.probs[:, 0].mean() > base.probs[:, 0].mean() + 0.1
    # The inner criterion is unweighted: the inner NLL of a fixed fit does not read the weights.
    x = fb.matrix("lowdim.v1")
    sel = inner_select(task, fb, range(len(task.items)), grid=grid)
    oof = softmax(sel.oof_scores)
    y = np.asarray(task.labels)[sel.oof_idx]
    assert sel.inner_nll == pytest.approx(float(np.mean(-np.log(oof[np.arange(len(y)), y]))))
    assert x.shape[1] == 2 and sel.inner_folds == 7


def test_two_runs_are_byte_identical() -> None:
    task, s = rank_task(30, seed=4)
    fb = planted_source(s)
    one = nested_probe(task, fb, grid=DEFAULT_GRID["rank_fit"], clm_model_fps=FPS)
    two = nested_probe(task, fb, grid=DEFAULT_GRID["rank_fit"], clm_model_fps=FPS)
    assert _dump(one.fold_choices) == _dump(two.fold_choices)
    assert _dump(one.diagnostics) == _dump(two.diagnostics)
    assert one.pooled == two.pooled and np.array_equal(one.probs, two.probs)
    kw = {"fingerprint": FINGERPRINTS[LATEST_MODEL], **IDENTITY}
    for card in CARDS:
        a = one.artifact(card, task, fb, **kw)
        b = two.artifact(card, task, fb, **kw)
        assert _dump(a.model_dump(mode="json")) == _dump(b.model_dump(mode="json"))


def test_predict_equals_the_benchs_held_out_probabilities() -> None:
    task, s = rank_task(30, seed=5)
    fb = planted_source(s)
    res = nested_probe(task, fb, grid=DEFAULT_GRID["rank_fit"])
    for fold in res.folds:
        art = res.artifact(
            fold.held_out, task, fb, fingerprint=FINGERPRINTS[LATEST_MODEL], **IDENTITY
        )
        assert (
            art.format == "mesa-clm/probe/1" and art.spec == res.fold_choices[fold.held_out]["spec"]
        )
        assert art.question_key == fb.question_key and art.task_id == "term.fits" and art.k == 2
        assert art.n_train == len(fold.train) and art.selection["fold"] == fold.held_out
        pred = art.predict(fb.matrix(art.spec)[fold.test])
        rows = [res.pooled.index(i) for i in fold.test]
        np.testing.assert_array_equal(pred, res.probs[rows])
        back = P.load_probe(json.loads(json.dumps(art.model_dump(mode="json"))))
        assert back == art
        np.testing.assert_array_equal(back.predict(fb.matrix(art.spec)[fold.test]), pred)
        # calibrated = the OOF Platt applied to the logit difference of the refit's scores
        cal = calibrate.load_calibrator(res.diagnostics["per_fold"][fold.held_out]["calibrator"])
        assert isinstance(cal, calibrate.PlattCalibrator) and cal.feature == "logit_difference"
        np.testing.assert_array_equal(cal.probs(art.scores(fb.matrix(art.spec)[fold.test])), pred)
    with pytest.raises(ValueError):
        ProbeArtifact.model_validate({**art.model_dump(mode="json"), "extra": 1})


def test_the_oof_calibrator_and_the_winners_oof_scores_are_what_the_rule_says() -> None:
    """The OOF calibrator rule (P.4): every evaluated outer fold's calibrator is byte for byte
    ``calibrate.fit_calibrator("choice", inner_select(task, fb, fold.train).oof_scores,
    labels[oof_idx], weights[oof_idx])``, its ``oof_idx`` ⊆ ``train`` and disjoint from
    ``test``; and the winner's OOF scores are the per-inner-fold refits' scores bit for bit."""
    task, s = rank_task(30, seed=15)
    fb = planted_source(s)
    grid = Grid(("lowdim.v1", "pair512.v1"), ("logreg", "lda"), GRIDS)
    res = nested_probe(task, fb, grid=grid)
    assert res.n_folds == 7
    labels = np.asarray(task.labels, dtype=np.int64)
    weights = np.asarray(task.weights, dtype=np.float64)
    cards = np.asarray(task.cards, dtype=object)
    for fold in res.folds:
        sel = inner_select(task, fb, fold.train, grid=grid)
        oof = set(sel.oof_idx.tolist())
        assert oof <= set(fold.train) and not oof & set(fold.test) and len(oof) == len(fold.train)
        assert sel.winner is not None and sel.winner == res.fits[fold.held_out].config
        assert sel.oof_scores is not None
        # k = 2: calibrator_input is the scores themselves (Platt on their difference)
        cal = calibrate.fit_calibrator(
            "choice", sel.oof_scores, labels[sel.oof_idx], weights[sel.oof_idx]
        )
        got = res.fits[fold.held_out].calibrator
        assert _dump(cal.model_dump(mode="json")) == _dump(got.model_dump(mode="json"))
        assert _dump(P._calibrator_record(cal)) == _dump(
            res.fold_choices[fold.held_out]["calibrator"]
        )
        # the winner's OOF scores equal the per-inner-fold refits bit for bit
        x = fb.matrix(sel.winner.spec)
        train = np.asarray(sorted(fold.train), dtype=np.int64)
        pos = {int(i): j for j, i in enumerate(sel.oof_idx.tolist())}
        seen = 0
        for inner in sorted({str(c) for c in cards[train]}):
            on = cards[train] == inner
            tr, te = train[~on], train[on]
            model = P.fit_design(
                P.prepare(x[tr], weights[tr]),
                sel.winner.fitter,  # type: ignore[arg-type]
                labels[tr],
                k=2,
                hyper=sel.winner.hyper,
            )
            rows = [pos[int(i)] for i in te.tolist()]
            np.testing.assert_array_equal(model.scores(x[te]), sel.oof_scores[rows])
            seen += len(rows)
        assert seen == len(sel.oof_idx) == sel.inner_n


def test_a_held_out_card_never_sees_its_own_labels() -> None:
    task, s = rank_task(30, seed=6)
    fb = planted_source(s)
    grid = Grid(("lowdim.v1", "pair512.v1"), ("logreg", "lda"), GRIDS)
    base = nested_probe(task, fb, grid=grid)
    for held in (CARDS[1], CARDS[5]):
        flipped = _relabel(
            task,
            [1 - y if c == held else y for y, c in zip(task.labels, task.cards, strict=True)],
            [0.6 if c == held else w for w, c in zip(task.weights, task.cards, strict=True)],
        )
        other = nested_probe(flipped, fb, grid=grid)
        mine = [i for i, c in enumerate(task.cards) if c == held]
        a = base.probs[[base.pooled.index(i) for i in mine]]
        b = other.probs[[other.pooled.index(i) for i in mine]]
        np.testing.assert_array_equal(a, b)
        assert _dump(base.fold_choices[held]) == _dump(other.fold_choices[held])
        rest = [i for i, c in enumerate(task.cards) if c != held]
        assert not np.array_equal(
            base.probs[[base.pooled.index(i) for i in rest]],
            other.probs[[other.pooled.index(i) for i in rest]],
        )


# -- teacher rows -------------------------------------------------------------------------------------------


def _teacher(s_source: FakeSource, rng: np.random.Generator) -> TeacherRows:
    """20 teacher rows: 12 of product DP0.00001.001 (cards 0-3), 8 of DP0.00002.001 (cards 4-6),
    all Yes at weight 0.5, with a planted signal in ``lowdim.v1``."""
    m = 20
    mats = {spec: rng.normal(size=(m, x.shape[1])) for spec, x in s_source.mats.items()}
    mats["lowdim.v1"][:, 0] = rng.normal(2.0, 1.0, m)
    groups = tuple(["DP0.00001.001"] * 12 + ["DP0.00002.001"] * 8)
    return TeacherRows(FakeSource(mats), np.zeros(m, dtype=np.int64), np.full(m, 0.5), groups)


def test_teacher_rows_train_only_and_are_dropped_for_their_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, s = rank_task(30, seed=7)
    fb = planted_source(s)
    rng = np.random.default_rng(8)
    teacher = _teacher(fb, rng)
    assert teacher.n == 20 and teacher.without("DP0.00001.001").n == 8
    assert teacher.without("DP0.00002.001").n == 12 and teacher.without("other").n == 20
    assert teacher.without("DP0.00001.001").without("DP0.00002.001").n == 0
    grid = Grid(("lowdim.v1",), ("logreg",), {"logreg": ({"lambda": 0.1},)})
    sizes: list[int] = []
    real = prepare

    def spy(x: Any, w: Any = None) -> Any:
        sizes.append(int(np.asarray(x).shape[0]))
        return real(x, w)

    monkeypatch.setattr(P, "prepare", spy)
    res = nested_probe(task, fb, grid=grid, teacher=teacher)
    assert res.n_folds == 7 and res.diagnostics["teacher"] is True
    assert res.diagnostics["teacher_in_test"] is False and res.diagnostics["teacher_rows"] == 20
    # Never in a test index: the pooled items are exactly the task's items.
    assert sorted(res.pooled) == list(range(len(task.items)))
    assert all(set(f.test) <= set(range(len(task.items))) for f in res.folds)
    # Outer: the held-out card's product is dropped from its fold's training set.
    n = len(task.items)
    products = {c: c.rsplit(".", 1)[0] for c in CARDS}
    expected: list[int] = []
    for card in CARDS:
        outer_t = {"DP0.00001.001": 8, "DP0.00002.001": 12}[products[card]]
        assert res.fold_choices[card]["n_teacher"] == outer_t
        assert res.fold_choices[card]["n_train"] == n - 30
        for inner in sorted(c for c in CARDS if c != card):
            # the inner training set: 5 cards + teacher rows of neither product held out
            same = products[inner] == products[card]
            expected.append(n - 60 + (outer_t if same else 0))
        expected.append(n - 30 + outer_t)  # the refit on all six training cards
    assert sizes == expected
    monkeypatch.setattr(P, "prepare", real)
    without = nested_probe(task, fb, grid=grid)
    assert not np.array_equal(res.probs, without.probs)  # the teacher rows change the fit
    # A full-data fit uses every teacher row; its inner LOCO drops each inner product.
    sizes.clear()
    monkeypatch.setattr(P, "prepare", spy)
    art, report = full_probe(
        task, fb, grid=grid, fingerprints=FINGERPRINTS, teacher=teacher, **IDENTITY
    )
    assert sizes[-1] == n + 20 and art.n_train == n + 20 and report["n_teacher"] == 20
    assert sizes[:-1] == [n - 30 + (8 if products[c] == "DP0.00001.001" else 12) for c in CARDS]
    cell = assemble_cell(
        task, res.pooled, res.probs, res.folds, tier="probe", framing="F7", selection="nested",
        pre_registered=True, exploratory=False, servable=True, skipped_folds=res.skipped_folds,
        fold_choices=res.fold_choices, diagnostics=res.diagnostics, B=50, **IDENTITY,
    )  # fmt: skip
    assert cell.teacher_in_test is False and cell.counts.n == n


def test_teacher_row_refusals() -> None:
    task, s = rank_task(30)
    fb = planted_source(s)
    src = FakeSource({k: v[:5] for k, v in fb.mats.items()})
    with pytest.raises(ProbeError, match="leak_group"):
        TeacherRows(src, np.zeros(5, dtype=np.int64), np.ones(5), ("p", "", "p", "p", "p"))
    with pytest.raises(ProbeError, match="one label, weight and leak_group"):
        TeacherRows(src, np.zeros(4, dtype=np.int64), np.ones(5), ("p",) * 5)
    with pytest.raises(ProbeError, match="index the task's options"):
        TeacherRows(src, np.full(5, 2, dtype=np.int64), np.ones(5), ("p",) * 5)
    rows = TeacherRows(src, np.zeros(5, dtype=np.int64), np.ones(5), ("p",) * 5)
    no_products = Task(
        task.name, task.spec, task.items, "s", "s", cards=task.cards, weights=task.weights,
        option_keys=task.option_keys, meta=task.meta,
    )  # fmt: skip
    with pytest.raises(ProbeError, match="products"):
        nested_probe(no_products, fb, grid=DEFAULT_GRID["rank_fit"], teacher=rows)
    with pytest.raises(ProbeError, match="teacher items need their leak_group"):
        TeacherRows.from_task(no_products, fb)
    t = TeacherRows.from_task(task, fb, weights=[0.3] * len(task.items))
    assert t.n == len(task.items) and float(t.w[0]) == 0.3 and t.leak_groups[0] == "DP0.00001.001"


# -- guards, floors and unfittable configurations -------------------------------------------------------


def test_guards_and_floors_skip_folds_with_the_stated_reasons() -> None:
    def few_no(n: int, seed: int) -> tuple[Task, np.ndarray]:
        task, s = rank_task(n, seed=seed)
        y = list(task.labels)
        for i, c in enumerate(task.cards):
            if c == CARDS[0]:
                y[i] = 0
        y[:3] = [1, 1, 1]  # card0 holds out three No: below the 5 guard
        return _relabel(task, y), s

    task, s = few_no(30, 9)
    res = nested_probe(task, planted_source(s), grid=DEFAULT_GRID["rank_fit"])
    assert res.skipped_folds[CARDS[0]].startswith("insufficient_heldout_per_class")
    assert res.n_folds == 6 and res.fold_choices[CARDS[0]]["evaluated"] is False
    assert res.fold_choices[CARDS[0]]["decision"] == res.skipped_folds[CARDS[0]]
    assert set(res.pooled) == {i for i, c in enumerate(task.cards) if c != CARDS[0]}
    assert "inner_grid" not in res.diagnostics["per_fold"][CARDS[0]]
    # The inner 30/5 guard: on every other outer fold the inner fold that holds card0 out is
    # skipped for the same reason, so the inner OOF pool is the training set minus card0.
    n_train = len(task.items) - 30
    for card, choice_ in res.fold_choices.items():
        if card == CARDS[0]:
            continue
        assert choice_["evaluated"] and choice_["n_train"] == n_train, card
        assert choice_["inner_skips"][CARDS[0]].startswith("insufficient_heldout_per_class")
        assert set(choice_["inner_skips"]) == {CARDS[0]} and choice_["inner_folds"] == 5
        assert choice_["inner_n"] == n_train - 30
    # A closed choice has no 30/5 guard: the floors alone decide (7 cards × n targets).
    k4 = {"shape": "choice", "k": 4}
    rng = np.random.default_rng(10)

    def choice(n_targets: int, seed: int = 0) -> tuple[Task, FakeSource]:
        states = []
        cards = []
        logits = []
        for card in CARDS:
            for t in range(n_targets):
                states.append({"card": {"dataset": card}, "column": {"name": f"c{t}"}})
                cards.append(card)
                logits.append(rng.normal(size=4) * 3)
        x = np.stack(logits)
        p = softmax(x, 3.0)
        y = np.minimum((rng.random(len(x))[:, None] > np.cumsum(p, axis=1)).sum(axis=1), 3)
        task = Task(
            "neon_value_kind", TASKS["avu.value_kind"],
            [(st, int(lbl)) for st, lbl in zip(states, y, strict=True)], "s", "s", cards=cards,
            weights=[0.6] * len(states), products=[c.rsplit(".", 1)[0] for c in cards],
            option_keys=[""] * len(states),
            meta={"min_weight": 0.5, "masked": True, "mask": None, "label_sources": {}, "excluded": {}},
        )  # fmt: skip
        mats = {
            "choice.state.v1": np.hstack([rng.normal(size=(len(x), 6)), x]),
            "choice.raw.v1": rng.normal(size=(len(x), 9)),
        }
        return task, FakeSource(mats, **k4)

    grid = Grid(("choice.state.v1",), ("logreg", "ridge"), GRIDS)
    task, fb = choice(6)  # 42 items: 36 training items < 40
    res = nested_probe(task, fb, grid=grid)
    assert res.skipped_folds == {c: "below_floor 36 < 40 training items (probe)" for c in CARDS}
    assert res.n_folds == 0 and res.probs.shape == (0, 4) and res.pooled == []
    task, fb = choice(7)  # 49 items: the outer 42 pass, every inner 35 < 40
    res = nested_probe(task, fb, grid=grid)
    assert res.skipped_folds == dict.fromkeys(CARDS, "inner_no_passing_fold")
    for card, choice_ in res.fold_choices.items():
        assert choice_["evaluated"] is False and choice_["spec"] is None
        assert choice_["inner_skips"] == {
            c: "below_floor 35 < 40 training items (probe)" for c in CARDS if c != card
        }
        assert all(
            g["error"] == "no passing inner fold"
            for g in res.diagnostics["per_fold"][card]["inner_grid"]
        )
    task, fb = choice(8)  # 56 items: inner folds pass on 40, but 48 OOF items < 100
    res = nested_probe(task, fb, grid=grid)
    assert res.skipped_folds == {
        c: "below_floor 48 < 100 OOF items (probe calibrator)" for c in CARDS
    }
    assert res.n_folds == 0
    for card, choice_ in res.fold_choices.items():
        assert choice_["decision"] == "configuration" and choice_["evaluated"] is False
        assert choice_["spec"] == "choice.state.v1" and choice_["inner_n"] == 48
        assert choice_["skip_reason"] == res.skipped_folds[card] and "calibrator" not in choice_
        info = res.diagnostics["per_fold"][card]
        assert info["uncalibrated_heldout"]["nll"] > 0 and info["linear"]["n"] == 48
    assert res.diagnostics["uncalibrated_pooled"]["n"] == 56
    assert res.diagnostics["uncalibrated_pooled"]["cards"] == sorted(CARDS)
    assert res.diagnostics["calibration"] == "temperature"
    task, fb = choice(15)  # 105 items: 90 OOF items < 100 still; 20 → 140 items pass
    assert nested_probe(task, fb, grid=grid).n_folds == 0
    task, fb = choice(20)
    res = nested_probe(task, fb, grid=grid, calibration_floor=100)
    assert res.n_folds == 7 and res.skipped_folds == {}
    for card, choice_ in res.fold_choices.items():
        cal = choice_["calibrator"]
        assert cal["kind"] == "temperature" and set(cal["params"]) == {"temperature"}
        assert cal["n"] == 120 and "at_bound" in cal and cal["inverted"] is False
        art = res.artifact(card, task, fb, fingerprint=FINGERPRINTS[LATEST_MODEL], **IDENTITY)
        fold = next(f for f in res.folds if f.held_out == card)
        pred = art.predict(fb.matrix(art.spec)[fold.test])
        np.testing.assert_array_equal(pred, res.probs[[res.pooled.index(i) for i in fold.test]])
        t = calibrate.load_calibrator(res.diagnostics["per_fold"][card]["calibrator"])
        assert isinstance(t, calibrate.TemperatureCalibrator)
        np.testing.assert_allclose(
            pred, softmax(art.scores(fb.matrix(art.spec)[fold.test]), t.temperature)
        )
    res2 = nested_probe(task, fb, grid=grid, calibration_floor=200)
    assert res2.skipped_folds == {
        c: "below_floor 120 < 200 OOF items (probe calibrator)" for c in CARDS
    }
    res3 = nested_probe(task, fb, grid=grid, probe_floor=130)
    assert res3.skipped_folds == {c: "below_floor 120 < 130 training items (probe)" for c in CARDS}


def test_an_unfittable_configuration_is_recorded_and_never_chosen() -> None:
    """A K = 4 choice whose class 3 lives on one card: LDA cannot be fitted on the inner folds
    that hold that card out (no training weight for the class), is recorded ``error`` and not
    selectable there; logreg and ridge still are."""
    rng = np.random.default_rng(12)
    n_targets = 20
    states, cards, y = [], [], []
    for card in CARDS:
        for t in range(n_targets):
            states.append({"card": {"dataset": card}, "column": {"name": f"c{t}"}})
            cards.append(card)
            y.append(3 if card == CARDS[2] and t < 5 else int(rng.integers(0, 3)))
    task = Task(
        "neon_value_kind", TASKS["avu.value_kind"], [(st, lbl) for st, lbl in zip(states, y, strict=True)],
        "s", "s", cards=cards, weights=[0.6] * len(states), products=[c.rsplit(".", 1)[0] for c in cards],
        option_keys=[""] * len(states),
        meta={"min_weight": 0.5, "masked": True, "mask": None, "label_sources": {}, "excluded": {}},
    )  # fmt: skip
    x = rng.normal(size=(len(y), 8))
    x[:, 0] = np.asarray(y) + rng.normal(0, 0.3, len(y))
    fb = FakeSource({"choice.state.v1": x}, shape="choice", k=4)
    res = nested_probe(task, fb, grid=DEFAULT_GRID["choice"].restrict(LATEST_MODEL))
    assert res.n_folds == 7
    for card, choice in res.fold_choices.items():
        grid = res.diagnostics["per_fold"][card]["inner_grid"]
        lda = [g for g in grid if g["fitter"] == "lda"]
        if card == CARDS[2]:  # class 3 is held out: no inner training set has it
            assert all(g["error"] and "class 3 has no training weight" in g["error"] for g in lda)
        else:
            assert all(g["error"] and f"{CARDS[2]}: lda: class 3" in g["error"] for g in lda)
        assert choice["fitter"] in ("logreg", "ridge")
        assert all(g["nll"] is None and g["n"] == 0 for g in lda)
    sel = inner_select(task, fb, range(len(y)), grid=Grid(("choice.state.v1",), ("lda",), GRIDS))
    assert sel.winner is None and sel.skip_reason == "inner_no_fittable_configuration"
    assert sel.inner_n == len(y) and all(f.error for f in sel.inner_grid)
    only_lda = nested_probe(task, fb, grid=Grid(("choice.state.v1",), ("lda",), GRIDS))
    assert only_lda.skipped_folds == dict.fromkeys(CARDS, "inner_no_fittable_configuration")
    with pytest.raises(ProbeError, match="inner_no_fittable_configuration"):
        full_probe(
            task,
            fb,
            grid=Grid(("choice.state.v1",), ("lda",), GRIDS),
            fingerprints=FINGERPRINTS,
            **IDENTITY,
        )


# -- the full-data artifact and the fixed configuration ------------------------------------------------------


def test_full_probe_and_the_fixed_configuration() -> None:
    task, s = rank_task(30, seed=13)
    fb = planted_source(s)
    grid = DEFAULT_GRID["rank_fit"]
    art, report = full_probe(task, fb, grid=grid, fingerprints=FINGERPRINTS, **IDENTITY)
    assert (
        art.spec == "lowdim.v1"
        and art.model == LATEST_MODEL
        and art.fingerprint == FINGERPRINTS[LATEST_MODEL]
    )
    assert art.n_train == len(task.items) and art.labels_sha256 == "a" * 64
    assert art.selection["selection"] == "full" and art.selection["cards"] == sorted(CARDS)
    assert art.selection["inner_n"] == len(task.items) and art.selection["inner_folds"] == 7
    assert art.selection["calibrator"]["kind"] == "platt" and "inner_grid" not in art.selection
    assert len(report["inner_grid"]) == 50 and report["n_train"] == len(task.items)
    assert report["spec"] == art.spec and report["calibrator"]["kind"] == "platt"
    assert report["inner_nll"] == min(g["nll"] for g in report["inner_grid"])
    config = ProbeConfig.of(report)
    assert (
        config.spec == art.spec
        and config.fitter == art.linear.kind
        and config.hyper == art.linear.hyper
    )
    back = P.load_probe(json.loads(json.dumps(art.model_dump(mode="json"))))
    assert back == art
    pred = art.predict(fb.matrix(art.spec))
    assert pred.shape == (len(task.items), 2) and np.allclose(pred.sum(axis=1), 1.0)
    assert float(np.mean((pred[:, 0] > 0.5) == (np.asarray(task.labels) == 0))) > 0.8
    fixed = nested_probe(task, fb, grid=grid, fixed=config, clm_model_fps=FPS)
    assert fixed.n_folds == 7 and fixed.diagnostics["selection"] == "fixed"
    assert fixed.diagnostics["fixed"] == config.as_dict()
    for card, choice in fixed.fold_choices.items():
        assert (choice["spec"], choice["fitter"], choice["hyper"]) == (
            config.spec,
            config.fitter,
            config.hyper,
        )
        assert len(fixed.diagnostics["per_fold"][card]["inner_grid"]) == 1
    nested = nested_probe(task, fb, grid=grid, clm_model_fps=FPS)
    for card in CARDS:  # a fold that chose the fixed configuration itself predicts identically
        if (nested.fold_choices[card]["fitter"], nested.fold_choices[card]["hyper"]) == (
            config.fitter,
            config.hyper,
        ):
            rows = [i for i, c in enumerate(task.cards) if c == card]
            np.testing.assert_array_equal(
                nested.probs[[nested.pooled.index(i) for i in rows]],
                fixed.probs[[fixed.pooled.index(i) for i in rows]],
            )
    small, s_small = rank_task(4)
    with pytest.raises(ProbeError, match="below_floor 28 < 40 training items"):
        full_probe(small, planted_source(s_small), grid=grid, fingerprints=FINGERPRINTS, **IDENTITY)
    one_class = _relabel(task, [0] * len(task.items))
    with pytest.raises(ProbeError, match="insufficient_train_per_class"):
        full_probe(one_class, fb, grid=grid, fingerprints=FINGERPRINTS, **IDENTITY)
    with pytest.raises(ProbeError, match="no fingerprint"):
        full_probe(task, fb, grid=grid, fingerprints={}, **IDENTITY)
    task20, s20 = rank_task(12, seed=14)  # 84 items: the 7-card OOF pool is below 100
    with pytest.raises(ProbeError, match=r"below_floor \d+ < 100 OOF items \(probe calibrator\)"):
        full_probe(task20, planted_source(s20), grid=grid, fingerprints=FINGERPRINTS, **IDENTITY)


def test_procedure_refusals() -> None:
    task, s = rank_task(30)
    fb = planted_source(s)
    with pytest.raises(ProbeError, match="the grid is choice"):
        nested_probe(task, fb, grid=DEFAULT_GRID["choice"])
    with pytest.raises(ProbeError, match="feature source is choice"):
        nested_probe(
            task,
            FakeSource({"choice.state.v1": np.zeros((len(s), 3))}, shape="choice"),
            grid=DEFAULT_GRID["rank_fit"],
        )
    with pytest.raises(ProbeError, match="does not match the task's items"):
        nested_probe(task, planted_source(s[:-1]), grid=DEFAULT_GRID["rank_fit"])
    with pytest.raises(ProbeError, match="is not a rank_fit spec"):
        inner_select(
            task,
            fb,
            range(len(s)),
            grid=DEFAULT_GRID["rank_fit"],
            fixed=ProbeConfig("choice.raw.v1", "lda", {"gamma": 0.5}),
        )
    with pytest.raises(ProbeError, match="empty training set"):
        inner_select(task, fb, [], grid=DEFAULT_GRID["rank_fit"])
    with pytest.raises(ProbeError, match=r"scores must be \[n, 2\]"):
        P.calibrator_input(np.zeros((3, 4)), 2)
    with pytest.raises(ProbeError, match="not of this task's shape"):
        nested_probe(
            task, fb, grid=DEFAULT_GRID["rank_fit"],
            teacher=TeacherRows(
                FakeSource({"choice.state.v1": np.zeros((2, 3))}, shape="choice", k=4),
                np.zeros(2, dtype=np.int64), np.ones(2), ("p", "p"),
            ),
        )  # fmt: skip
    assert P.spec_shape(task) == "rank_fit"
    assert (
        P.floor_reason(39) == "below_floor 39 < 40 training items (probe)"
        and P.floor_reason(40) is None
    )
    np.testing.assert_allclose(
        P.calibrator_input(np.array([[1.0, 2.0, 3.0]]), 3),
        np.log(softmax(np.array([[1.0, 2.0, 3.0]]))),
    )
    np.testing.assert_array_equal(P.calibrator_input(np.array([[1.0, 2.0]]), 2), [[1.0, 2.0]])
