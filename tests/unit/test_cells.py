"""zero_shot and calibrated LOCO cells (``mesa_clm.bench.cells``; plan §5.3, §5.4, §8 M2;
``design/m2-analysis-plan.md`` §6, §9), on synthetic worlds only: a feature store of planted
vectors whose scores ``s_c = 100·(zs·zc − zs·za)`` (or option logits) are known, and labels drawn
from a planted Platt or temperature model. No silver label meets a model score here.

Checked: the arm scores equal the offline scorer's; zero shot is ``σ(s_c)`` / the option softmax;
the calibrated tier recovers the planted Platt ``(a, b)`` and temperature ``T``; a held-out
card's predictions never depend on its own labels; the 30/5 guards and the 100-item floor skip
folds as stated; the nested cell follows X1's per-fold arms (and its skipped folds), the
``@full`` cell uses A1 everywhere; ``beats_lookup_novel`` passes with real signal and fails
without, and is not applicable at K > 2; every frozen cell field is present; refusals; the
results file round-trips and is deterministic."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings
from mesa_clm.bench import cells as C
from mesa_clm.bench import stats
from mesa_clm.bench.cells import (
    CALIBRATED_FLOOR,
    Arm,
    ArmScores,
    CellError,
    FoldChoice,
    TextIndex,
    X1Selection,
    assemble_cell,
    item_identities,
    mean_pairwise_cosine,
    run_tier_cells,
    score_arm,
    task_cells,
    tier_cell,
)
from mesa_clm.bench.results import BenchCell, load_results, markdown_table, write_results
from mesa_clm.bench.tasks.base import Fold, Task
from mesa_clm.clm.headproj import l2, random_head
from mesa_clm.identity import target_sha256
from mesa_clm.learn import calibrate
from mesa_clm.learn.features import (
    ROLE_SIDE,
    FeatureMissing,
    FeatureStore,
    ManifestRow,
    text_sha256,
)
from mesa_clm.learn.offline import LATEST_MODEL, RAW_MODEL, OfflineScorer, store_vectors
from mesa_clm.providers.tiered import fake_fingerprint
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import TASKS

FP = fake_fingerprint().encoder_fp
DIM = 64
B = 200
CARDS = tuple(f"DP0.0000{1 + i // 4}.001.card{i}" for i in range(7))
LATEST_FP = "a1b2c3d4e5f6"
FINGERPRINTS = {
    RAW_MODEL: {
        "encoder_fp": FP,
        "clm_model_fp": "0f0f0f0f0f0f",
        "schema_sha256": "5" * 64,
        "serving_lock_sha": "6" * 64,
    },
    LATEST_MODEL: {
        "encoder_fp": FP,
        "clm_model_fp": LATEST_FP,
        "schema_sha256": "5" * 64,
        "serving_lock_sha": "6" * 64,
    },
}
COMMON: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64, "B": B}
TC: dict[str, Any] = {**COMMON, "registered": True}
ZERO: Callable[[], float] = lambda: 0.0  # noqa: E731 - a deterministic clock


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def _orth(rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    v = rng.standard_normal(DIM)
    v -= (v @ base) * base
    return v / np.linalg.norm(v)


def _row(task_id: str, fid: str, target: str, option: str, role: str, text: str) -> ManifestRow:
    return ManifestRow(
        task_id=task_id,
        framing_id=fid,
        target_sha256=target,
        option_key=option,
        role=role,  # type: ignore[arg-type]
        side=ROLE_SIDE[role],
        text_sha256=text_sha256(text),
        text=text,
    )


@dataclass
class World:
    task: Task
    rows: list[ManifestRow]
    store: FeatureStore
    planted: dict[str, np.ndarray]  # framing -> s_c [n] or raw logits [n, K]

    @property
    def index(self) -> TextIndex:
        return TextIndex(self.rows)


def _meta(labels: Sequence[int], binary: bool) -> dict[str, Any]:
    yes = sum(1 for y in labels if y == 0)
    sources = (
        {"consensus_majority": yes, "consensus_negative": len(labels) - yes}
        if binary
        else {"consensus_majority": len(labels)}
    )
    return {
        "min_weight": 0.5,
        "masked": True,
        "mask": None,
        "label_sources": sources,
        "excluded": {},
    }


def _store(root: Path, texts: list[str], vecs: list[np.ndarray]) -> FeatureStore:
    store = FeatureStore(root / "features", FP, dim=DIM)
    distinct: dict[str, np.ndarray] = {}
    for t, v in zip(texts, vecs, strict=True):
        distinct.setdefault(t, v)
    store.add(list(distinct), np.stack(list(distinct.values())), [len(t.split()) for t in distinct])
    return store


def rank_world(
    root: Path,
    *,
    framing_ids: Sequence[str] = ("F7",),
    n_targets: int = 12,
    n_cands: int = 3,
    a: float = 1.8,
    b: float = -0.5,
    seed: int = 0,
    labels: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    task_id: str = "term.fits",
    name: str = "neon_term_fits",
) -> World:
    """Every (target, candidate) gets a planted ``s`` per framing; Yes ~ σ(a·s_F7 + b). Every
    lookup key is novel (each card has its own CURIEs), so ``lookup_prob`` is the prior."""
    rng = np.random.default_rng(seed)
    anchors = {fid: _unit(rng) for fid in framing_ids}
    texts: list[str] = []
    vecs: list[np.ndarray] = []
    rows: list[ManifestRow] = []
    states: list[dict[str, Any]] = []
    cards: list[str] = []
    options: list[str] = []
    planted: dict[str, list[float]] = {fid: [] for fid in framing_ids}
    for fid in framing_ids:
        texts.append(f"{fid} anchor")
        vecs.append(anchors[fid])
    for ci, card in enumerate(CARDS):
        for t in range(n_targets):
            base: dict[str, Any] = {
                "card": {"dataset": card},
                "scope": "column",
                "aspect": "measurement",
                "column": {"name": f"col{t}"},
            }
            if task_id == "column.ontology_fits":
                base.pop("scope")
            target = target_sha256(task_id, base)
            zs: dict[str, np.ndarray] = {}
            for fid in framing_ids:
                zs[fid] = 0.2 * anchors[fid] + math.sqrt(0.96) * _orth(rng, anchors[fid])
                ctx = f"{fid} context {card} col{t}"
                texts.append(ctx)
                vecs.append(zs[fid])
                rows.append(_row(task_id, fid, target, "", "context", ctx))
                rows.append(_row(task_id, fid, target, ANCHOR_KEY, "anchor", f"{fid} anchor"))
            for c in range(n_cands):
                key = f"X:{ci}{t:03d}{c}" if task_id == "term.fits" else f"ont{ci}{t}{c}"
                state = dict(base)
                if task_id == "term.fits":
                    state["candidate"] = {"curie": key}
                else:
                    state["ontology"] = {"id": key}
                for fid in framing_ids:
                    s = float(rng.normal(0.0, 2.0))
                    alpha = 0.2 + s / 100.0
                    zc = alpha * zs[fid] + math.sqrt(1 - alpha**2) * _orth(rng, zs[fid])
                    cand = f"{fid} candidate {key}"
                    texts.append(cand)
                    vecs.append(zc)
                    rows.append(_row(task_id, fid, target, key, "candidate", cand))
                    planted[fid].append(s)
                states.append(state)
                cards.append(card)
                options.append(key)
    s_f7 = np.asarray(planted[framing_ids[0]])
    y = (
        labels(s_f7, rng)
        if labels is not None
        else np.where(rng.random(len(s_f7)) < calibrate.sigmoid(a * s_f7 + b), 0, 1)
    )
    task = Task(
        name,
        TASKS[task_id],
        [(st, int(lbl)) for st, lbl in zip(states, y, strict=True)],
        "synthetic",
        "synthetic",
        cards=cards,
        weights=[0.6 if lbl == 0 else 0.5 for lbl in y],
        products=[c.rsplit(".", 1)[0] for c in cards],
        option_keys=options,
        meta=_meta([int(v) for v in y], True),
    )
    return World(
        task,
        rows,
        _store(root, texts, vecs),
        {fid: np.asarray(v) for fid, v in planted.items()},
    )


def choice_world(
    root: Path,
    task_id: str,
    *,
    n_targets: int = 20,
    seed: int = 0,
    temperature: float = 3.0,
    a: float = 0.3,
    b: float = 0.2,
    name: str | None = None,
    novel: bool = False,
) -> World:
    """Closed-choice targets with random context and option vectors; labels drawn from
    ``softmax(raw logits / temperature)`` (K > 2) or ``σ(a·d + b)`` on the logit difference.
    Column names repeat across cards (lookup keys seen in training) unless ``novel``."""
    rng = np.random.default_rng(seed)
    f = framings.framing(task_id, "F7")
    wire = list(f.closed_options or {})
    opt_vec = {k: _unit(rng) for k in wire}
    opt_text = {k: f"option {task_id} {k}" for k in wire}
    texts = list(opt_text.values())
    vecs = [opt_vec[k] for k in wire]
    rows: list[ManifestRow] = []
    states: list[dict[str, Any]] = []
    cards: list[str] = []
    logits: list[np.ndarray] = []
    for card in CARDS:
        for t in range(n_targets):
            col = f"{card[-1]}col{t}" if novel else f"col{t}"
            state: dict[str, Any] = {"card": {"dataset": card}, "column": {"name": col}}
            if task_id == "avu.value_kind":
                state.update(aspect="measurement", term={"curie": f"X:{t}", "label": "l"})
            target = target_sha256(task_id, state)
            zs = _unit(rng)
            ctx = f"context {task_id} {card} {col}"
            texts.append(ctx)
            vecs.append(zs)
            rows.append(_row(task_id, "F7", target, "", "context", ctx))
            rows.extend(_row(task_id, "F7", target, k, "option", opt_text[k]) for k in wire)
            logits.append(100.0 * np.array([zs @ opt_vec[k] for k in wire]))
            states.append(state)
            cards.append(card)
    x = np.stack(logits)
    if len(wire) == 2:
        y = np.where(rng.random(len(x)) < calibrate.sigmoid(a * (x[:, 0] - x[:, 1]) + b), 0, 1)
    else:
        p = calibrate.softmax(x, temperature)
        y = np.minimum(
            (rng.random(len(x))[:, None] > np.cumsum(p, axis=1)).sum(axis=1), len(wire) - 1
        )
    bench = (
        name
        or {
            "column.annotate": "neon_annotate",
            "column.aspect": "neon_aspect",
            "avu.value_kind": "neon_value_kind",
        }[task_id]
    )
    task = Task(
        bench,
        TASKS[task_id],
        [(st, int(lbl)) for st, lbl in zip(states, y, strict=True)],
        "synthetic",
        "synthetic",
        cards=cards,
        weights=[0.6] * len(states),
        products=[c.rsplit(".", 1)[0] for c in cards],
        option_keys=[""] * len(states),
        meta=_meta([int(v) for v in y], len(wire) == 2),
    )
    return World(task, rows, _store(root, texts, vecs), {"F7": x})


def _scorers(store: FeatureStore) -> dict[str, OfflineScorer]:
    head = random_head(3, width=32, projection_dim=16, hidden_size=DIM)
    return {
        RAW_MODEL: OfflineScorer(RAW_MODEL),
        LATEST_MODEL: OfflineScorer(LATEST_MODEL, head, clm_model_fp=LATEST_FP),
    }


def _all(arm: Arm) -> X1Selection:
    return X1Selection.of(arm, {card: arm for card in CARDS})


def _items(cell: BenchCell) -> dict[tuple[str, str], list[float]]:
    assert cell.items is not None
    return {(i.target_sha256, i.option_key): i.probs for i in cell.items}


# -- scores ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [RAW_MODEL, LATEST_MODEL])
def test_rank_fit_scores_equal_the_offline_scorer(tmp_path: Path, model: str) -> None:
    w = rank_world(tmp_path, n_targets=3)
    scorer = _scorers(w.store)[model]
    sc = score_arm(w.task, w.index, w.store, scorer, "F7", clock=ZERO)
    assert sc.arm == Arm("F7", model) and sc.shape == "rank_fit" and sc.x.shape == (63,)
    assert sc.question_key == framings.framing("term.fits", "F7").question_key
    ids = item_identities(w.task)
    groups: dict[str, list[int]] = {}
    for i, (t, _) in enumerate(ids):
        groups.setdefault(t, []).append(i)
    for target, members in groups.items():
        ctx = w.index.text("term.fits", "F7", target, "", "context")
        cands = {
            ids[i][1]: w.index.text("term.fits", "F7", target, ids[i][1], "candidate")
            for i in members
        }
        anchor = w.index.text("term.fits", "F7", target, ANCHOR_KEY, "anchor")
        group = scorer.rank_fit_texts(w.store, ctx, cands, anchor)
        for i in members:
            assert abs(sc.x[i] - group.s_c[ids[i][1]]) <= 1e-9
    if model == RAW_MODEL:  # the planted scores, up to the store's float32 rounding
        np.testing.assert_allclose(sc.x, w.planted["F7"], atol=5e-5)
    diag = sc.diagnostics
    assert diag["n_items"] == 63 and diag["n_targets"] == diag["calls"] == 21
    assert diag["timing_source"] == "offline_replay" and diag["ms_per_decision"] == 0.0
    assert diag["input_tokens"] > 0 and -1.0 <= diag["mean_state_cos"] <= 1.0
    if model == RAW_MODEL:  # random contexts: far from collapsed (a random head collapses them)
        assert 0.0 < diag["mean_state_cos"] < 0.2


@pytest.mark.parametrize("model", [RAW_MODEL, LATEST_MODEL])
def test_choice_scores_are_the_option_logits(tmp_path: Path, model: str) -> None:
    w = choice_world(tmp_path, "column.aspect", n_targets=2)
    scorer = _scorers(w.store)[model]
    sc = score_arm(w.task, w.index, w.store, scorer, "F7", clock=ZERO)
    assert sc.shape == "choice" and sc.x.shape == (14, 8)
    ids = item_identities(w.task)
    for i, (target, _) in enumerate(ids):
        ctx = w.index.text("column.aspect", "F7", target, "", "context")
        opts = [t for _, t in w.index.options("column.aspect", "F7", target)]
        zs = store_vectors(scorer, w.store, "state", [ctx])
        zo = store_vectors(scorer, w.store, "action", opts)
        np.testing.assert_allclose(
            sc.x[i], scorer.scale * (zo.astype(float) @ zs[0].astype(float)), atol=1e-9
        )
    if model == RAW_MODEL:
        np.testing.assert_allclose(sc.x, w.planted["F7"], atol=5e-5)
    np.testing.assert_allclose(sc.zero_shot(range(14)), calibrate.softmax(sc.x))


def test_mean_pairwise_cosine() -> None:
    z = l2(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    want = (0.0 + 2 * (1 / math.sqrt(2))) / 3
    assert mean_pairwise_cosine(z) == pytest.approx(want)
    assert mean_pairwise_cosine(z[:1]) is None


# -- tiers ----------------------------------------------------------------------------------------------


def test_zero_shot_cell_is_sigma_of_s_c_over_every_fold(tmp_path: Path) -> None:
    w = rank_world(tmp_path)
    arm = Arm("F7", RAW_MODEL)
    cells = task_cells(
        w.task,
        w.index,
        w.store,
        _scorers(w.store),
        FINGERPRINTS,
        selection=_all(arm),
        tiers=("zero_shot",),
        clock=ZERO,
        **TC,
    )
    assert set(cells) == {"neon_term_fits.zero_shot.F7", "neon_term_fits.zero_shot.F7@full"}
    nested, full = cells["neon_term_fits.zero_shot.F7"], cells["neon_term_fits.zero_shot.F7@full"]
    for cell in (nested, full):
        assert cell.n_folds == 7 and cell.skipped_folds == {}
        assert cell.counts.n == len(w.task.items)
        assert cell.diagnostics["calibration"] == "uncalibrated"
        probs = np.array([p for p in _items(cell).values()])
        assert np.all(probs.sum(axis=1) == pytest.approx(1.0))
    ids = item_identities(w.task)
    got = _items(nested)
    for i, key in enumerate(ids):
        assert got[key][0] == pytest.approx(
            float(calibrate.sigmoid(w.planted["F7"][i : i + 1])[0]), abs=2e-5
        )
    assert _items(full) == got
    assert nested.selection == "nested" and not nested.exploratory and nested.pre_registered
    assert full.selection == "full" and full.exploratory and full.pre_registered
    assert full.variant == "full" and full.key == "neon_term_fits.zero_shot.F7@full"
    assert nested.fingerprint == FINGERPRINTS[RAW_MODEL] and nested.model == RAW_MODEL
    raw = nested.diagnostics["auroc_raw_score"]
    assert raw["score"] == "s_c" and raw["point"] == pytest.approx(nested.metrics.auroc, abs=1e-12)  # type: ignore[union-attr]
    assert nested.diagnostics["n_saturated"] == 0


def test_calibrated_cell_recovers_the_planted_platt(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=40, a=1.8, b=-0.5, seed=2)
    arm = Arm("F7", RAW_MODEL)
    cells = task_cells(
        w.task,
        w.index,
        w.store,
        _scorers(w.store),
        FINGERPRINTS,
        selection=_all(arm),
        clock=ZERO,
        **TC,
    )
    cal = cells["neon_term_fits.calibrated.F7"]
    zs = cells["neon_term_fits.zero_shot.F7"]
    assert cal.n_folds == 7 and cal.diagnostics["calibration"] == "platt"
    assert cal.diagnostics["inverted_folds"] == []
    for card, info in cal.diagnostics["per_fold"].items():
        fit = calibrate.load_calibrator(info["calibrator"])
        assert isinstance(fit, calibrate.PlattCalibrator) and fit.feature == "s_c", card
        assert fit.a == pytest.approx(1.8, abs=0.35) and fit.b == pytest.approx(-0.5, abs=0.35)
        assert info["n_train"] == len(w.task.items) - info["n"] and not info["inverted"]
    assert cal.metrics is not None and zs.metrics is not None
    assert cal.metrics.nll < zs.metrics.nll  # σ(s) is the wrong slope; Platt fixes it
    assert cal.diagnostics["per_fold"][CARDS[0]]["arm"] == "F7@clm-raw"


def test_a_held_out_card_never_sees_its_own_labels(tmp_path: Path) -> None:
    """Flipping card c's labels changes no prediction on card c, at either tier (LOCO); the
    other cards' calibrated predictions move (c trained them)."""
    w = rank_world(tmp_path, n_targets=12, seed=4)
    arm = Arm("F7", RAW_MODEL)
    scorers = _scorers(w.store)
    base = task_cells(
        w.task, w.index, w.store, scorers, FINGERPRINTS, selection=_all(arm), clock=ZERO, **TC
    )
    ids = item_identities(w.task)
    for card in CARDS:
        flipped_items = [
            (st, 1 - lbl) if w.task.cards[i] == card else (st, lbl)
            for i, (st, lbl) in enumerate(w.task.items)
        ]
        flipped = Task(
            w.task.name,
            w.task.spec,
            flipped_items,
            "s",
            "s",
            cards=w.task.cards,
            weights=w.task.weights,
            products=w.task.products,
            option_keys=w.task.option_keys,
            meta=dict(w.task.meta),
        )
        other = task_cells(
            flipped,
            w.index,
            w.store,
            scorers,
            FINGERPRINTS,
            selection=_all(arm),
            clock=ZERO,
            **TC,
        )
        for tier in ("zero_shot", "calibrated"):
            before = _items(base[f"neon_term_fits.{tier}.F7"])
            after = _items(other[f"neon_term_fits.{tier}.F7"])
            mine = [k for i, k in enumerate(ids) if w.task.cards[i] == card]
            rest = [k for i, k in enumerate(ids) if w.task.cards[i] != card]
            assert all(before[k] == after[k] for k in mine), (tier, card)
            if tier == "calibrated":
                assert any(before[k] != after[k] for k in rest), card
            else:
                assert all(before[k] == after[k] for k in rest)


def test_guards_and_the_calibrated_floor(tmp_path: Path) -> None:
    def few_no(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = np.where(rng.random(len(s)) < calibrate.sigmoid(1.5 * s), 0, 1)
        y[:36] = 0  # card0 holds the first 36 items: no "No" held out
        y[:3] = 1
        return y

    w = rank_world(tmp_path / "a", labels=few_no)
    arm = Arm("F7", RAW_MODEL)
    cells = task_cells(
        w.task,
        w.index,
        w.store,
        _scorers(w.store),
        FINGERPRINTS,
        selection=_all(arm),
        clock=ZERO,
        **TC,
    )
    cal, zs = cells["neon_term_fits.calibrated.F7"], cells["neon_term_fits.zero_shot.F7"]
    assert cal.skipped_folds[CARDS[0]].startswith("insufficient_heldout_per_class")
    assert cal.n_folds == 6 and cal.counts.n == len(w.task.items) - 36
    assert cal.fold_choices[CARDS[0]]["evaluated"] is False
    assert zs.n_folds == 7 and zs.skipped_folds == {}  # zero shot fits nothing (item 1.5)
    assert set(zs.guard_skipped_folds) == {CARDS[0]}
    # Below the floor: 7 cards x 3 targets x 3 candidates = 63 items, every fold trains on 54.
    small = rank_world(tmp_path / "b", n_targets=3, a=2.0, b=0.0)
    cells = task_cells(
        small.task,
        small.index,
        small.store,
        _scorers(small.store),
        FINGERPRINTS,
        selection=_all(arm),
        clock=ZERO,
        **TC,
    )
    cal = cells["neon_term_fits.calibrated.F7"]
    assert cal.n_folds == 0 and cal.metrics is None and cal.baselines is None
    assert cal.counts.n == 0 and cal.items == []
    assert (
        all(
            r.startswith("insufficient_train_per_class")
            or r.startswith(f"below_floor 54 < {CALIBRATED_FLOOR}")
            for r in cal.skipped_folds.values()
        )
        and len(cal.skipped_folds) == 7
    )
    assert cells["neon_term_fits.zero_shot.F7"].metrics is not None
    # A closed choice has no 30/5 guard; the floor alone skips: 7 x 5 targets, 30 per fold.
    few = choice_world(tmp_path / "c", "avu.value_kind", n_targets=5)
    cells = task_cells(
        few.task, few.index, few.store, _scorers(few.store), FINGERPRINTS, clock=ZERO, **TC
    )
    cal = cells["neon_value_kind.calibrated.F7"]
    assert cal.skipped_folds == {
        c: f"below_floor 30 < {CALIBRATED_FLOOR} training items (calibrated)" for c in CARDS
    }
    assert cal.metrics is None and cells["neon_value_kind.zero_shot.F7"].n_folds == 7


def test_the_nested_cell_follows_x1s_fold_arms(tmp_path: Path) -> None:
    w = rank_world(tmp_path, framing_ids=("F7", "F9"), n_targets=12, seed=6)
    f7, f7_latest = Arm("F7", RAW_MODEL), Arm("F7", LATEST_MODEL)
    choices: dict[str, Any] = {card: f7 for card in CARDS[:4]}
    choices[CARDS[4]] = {"framing": "F9", "model": RAW_MODEL, "inner_nll": 0.41}
    choices[CARDS[5]] = f7_latest
    x1 = X1Selection.of(f7, choices, fold_skips={CARDS[6]: "inner_k1"}, note="x1.json")
    scorers = _scorers(w.store)
    cells = task_cells(
        w.task, w.index, w.store, scorers, FINGERPRINTS, selection=x1, clock=ZERO, **TC
    )
    nested = cells["neon_term_fits.zero_shot.F7"]
    assert nested.selection == "nested" and nested.n_folds == 6
    assert nested.skipped_folds == {CARDS[6]: "inner_k1"}
    fc = nested.fold_choices
    assert fc[CARDS[6]] == {
        "decision": "inner_k1",
        "framing": None,
        "model": None,
        "evaluated": False,
    }
    assert fc[CARDS[4]]["framing"] == "F9" and fc[CARDS[4]]["inner_nll"] == 0.41
    assert fc[CARDS[4]]["question_key"] == framings.framing("term.fits", "F9").question_key
    assert fc[CARDS[5]]["clm_model_fp"] == LATEST_FP and fc[CARDS[0]]["decision"] == "arm"
    assert nested.diagnostics["fold_agreement_with_a1"] == {"agree": 4, "folds": 7}
    assert nested.question_key == framings.framing("term.fits", "F7").question_key
    assert nested.notes == "x1.json"
    ids = item_identities(w.task)
    got = _items(nested)
    sc_latest = score_arm(w.task, w.index, w.store, scorers[LATEST_MODEL], "F7")
    for i, key in enumerate(ids):
        card = w.task.cards[i]
        if card == CARDS[6]:
            assert key not in got
            continue
        if card == CARDS[4]:
            want = w.planted["F9"][i]
        elif card == CARDS[5]:
            want = sc_latest.x[i]
        else:
            want = w.planted["F7"][i]
        assert got[key][0] == pytest.approx(float(calibrate.sigmoid(np.array([want]))[0]), abs=2e-5)
    full = cells["neon_term_fits.zero_shot.F7@full"]
    assert full.n_folds == 7 and full.fold_choices == {}
    assert all(i["arm"] == "F7@clm-raw" for i in full.diagnostics["per_fold"].values())
    calibrated = cells["neon_term_fits.calibrated.F7"]
    assert calibrated.diagnostics["per_fold"][CARDS[4]]["arm"] == "F9@clm-raw"


def test_k1_and_a_missing_selection(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=3)
    k1 = X1Selection.of(None, note="K1: no arm qualified")
    cells = task_cells(
        w.task,
        w.index,
        w.store,
        _scorers(w.store),
        FINGERPRINTS,
        selection=k1,
        clock=ZERO,
        **TC,
    )
    assert set(cells) == {"neon_term_fits.zero_shot.F7", "neon_term_fits.calibrated.F7"}
    for cell in cells.values():
        assert cell.selection == "none" and cell.exploratory and not cell.pre_registered
        assert cell.model == LATEST_MODEL and cell.notes.startswith("K1: no arm qualified; no A1")
    with pytest.raises(CellError, match="X1's selection"):
        task_cells(w.task, w.index, w.store, _scorers(w.store), FINGERPRINTS, **TC)
    with pytest.raises(CellError, match="both an arm and a skip"):
        X1Selection.of(
            Arm("F7", RAW_MODEL), {CARDS[0]: Arm("F7", RAW_MODEL)}, {CARDS[0]: "inner_k1"}
        )


def test_closed_choice_cells_temperature_and_variants(tmp_path: Path) -> None:
    w = choice_world(tmp_path, "column.aspect", n_targets=40, temperature=3.0, seed=1, novel=True)
    cells = task_cells(w.task, w.index, w.store, _scorers(w.store), FINGERPRINTS, clock=ZERO, **TC)
    assert set(cells) == {
        "neon_aspect.zero_shot.F7",
        "neon_aspect.calibrated.F7",
        "neon_aspect.zero_shot.F7@clm-raw",
        "neon_aspect.calibrated.F7@clm-raw",
    }
    prod = cells["neon_aspect.calibrated.F7"]
    raw = cells["neon_aspect.calibrated.F7@clm-raw"]
    assert prod.model == LATEST_MODEL and prod.selection == "none" and prod.pre_registered
    assert not prod.exploratory and prod.fold_choices == {} and prod.servable
    assert raw.variant == RAW_MODEL and not raw.pre_registered and raw.exploratory
    assert raw.diagnostics["calibration"] == "temperature"
    for info in raw.diagnostics["per_fold"].values():
        fit = calibrate.load_calibrator(info["calibrator"])
        assert isinstance(fit, calibrate.TemperatureCalibrator)
        assert fit.temperature == pytest.approx(3.0, rel=0.25)
    assert raw.counts.n_neg is None and raw.metrics is not None and raw.metrics.auroc is None
    assert raw.baselines is not None and raw.baselines.beats_lookup_novel is False
    assert raw.baselines.beats_detail is not None
    detail = raw.baselines.beats_detail
    assert detail["reason"] == "auroc_not_applicable_k8" and detail["auroc_condition"] is None
    assert detail["n_novel"] == len(w.task.items) and "nll_rule_r" in detail  # NLL still reported
    seen = choice_world(tmp_path / "seen", "column.aspect", n_targets=4)
    cells_seen = task_cells(
        seen.task,
        seen.index,
        seen.store,
        _scorers(seen.store),
        FINGERPRINTS,
        tiers=("zero_shot",),
        clock=ZERO,
        **TC,
    )
    base_seen = cells_seen["neon_aspect.zero_shot.F7"].baselines
    assert base_seen is not None and base_seen.beats_detail == {
        "rule": C.BEATS_RULE,
        "n_novel": 0,
        "reason": "no_novel_items",
    }
    zs = cells["neon_aspect.zero_shot.F7@clm-raw"]
    np.testing.assert_allclose(
        np.array([i.probs for i in zs.items or []]),
        calibrate.softmax(w.planted["F7"]),
        atol=1e-4,
    )


def test_annotate_platt_on_the_logit_difference(tmp_path: Path) -> None:
    w = choice_world(tmp_path, "column.annotate", n_targets=60, a=0.3, b=0.2, seed=5)
    cells = task_cells(w.task, w.index, w.store, _scorers(w.store), FINGERPRINTS, clock=ZERO, **TC)
    cal = cells["neon_annotate.calibrated.F7@clm-raw"]
    assert cal.diagnostics["calibration"] == "platt"
    for info in cal.diagnostics["per_fold"].values():
        fit = calibrate.load_calibrator(info["calibrator"])
        assert isinstance(fit, calibrate.PlattCalibrator) and fit.feature == "logit_difference"
        assert fit.a == pytest.approx(0.3, abs=0.12) and fit.b == pytest.approx(0.2, abs=0.4)
    zs = cells["neon_annotate.zero_shot.F7@clm-raw"]
    assert zs.diagnostics["auroc_raw_score"]["score"] == "logit_difference"
    assert zs.counts.n_neg is not None and zs.metrics is not None and zs.metrics.auroc is not None


def test_value_kind_reports_the_pre_rule(tmp_path: Path) -> None:
    w = choice_world(tmp_path, "avu.value_kind", n_targets=4)
    cells = task_cells(
        w.task,
        w.index,
        w.store,
        _scorers(w.store),
        FINGERPRINTS,
        tiers=("zero_shot",),
        clock=ZERO,
        **TC,
    )
    cell = cells["neon_value_kind.zero_shot.F7"]
    assert cell.diagnostics["n_pre_rule"] == 0  # aspect "measurement": serving asks CLM
    unit = Task(
        w.task.name,
        w.task.spec,
        [({**st, "aspect": "unit"}, lbl) for st, lbl in w.task.items[:5]] + w.task.items[5:],
        "s",
        "s",
        cards=w.task.cards,
        weights=w.task.weights,
        products=w.task.products,
        option_keys=w.task.option_keys,
        meta=w.task.meta,
    )
    assert C._pre_rule_count(unit) == 5


# -- beats_lookup_novel --------------------------------------------------------------------------------


def test_beats_lookup_novel_with_signal_and_not_without(tmp_path: Path) -> None:
    arm = Arm("F7", RAW_MODEL)
    strong = rank_world(tmp_path / "s", n_targets=12, a=3.0, b=0.0, seed=8)
    cells = task_cells(
        strong.task,
        strong.index,
        strong.store,
        _scorers(strong.store),
        FINGERPRINTS,
        selection=_all(arm),
        clock=ZERO,
        **TC,
    )
    cal = cells["neon_term_fits.calibrated.F7"]
    assert cal.baselines is not None and cal.baselines.beats_lookup_novel is True
    detail = cal.baselines.beats_detail or {}
    assert detail["reason"] == "passed" and detail["auroc_condition"] is True
    assert detail["n_novel"] == len(strong.task.items) and detail["nll_rule_r"]["passed"] is True
    assert cal.baselines.novel_key.n == len(strong.task.items)  # the cell's own predictions
    lookup = cal.baselines.novel_key_lookup
    assert lookup is not None and cal.baselines.novel_key.nll is not None and lookup.nll is not None
    assert cal.baselines.novel_key.nll < lookup.nll
    assert lookup.auroc == pytest.approx(0.5, abs=0.15)  # the training prior on every item
    noise = rank_world(
        tmp_path / "n",
        n_targets=12,
        seed=9,
        labels=lambda s, rng: np.where(rng.random(len(s)) < 0.4, 0, 1),
    )
    cells = task_cells(
        noise.task,
        noise.index,
        noise.store,
        _scorers(noise.store),
        FINGERPRINTS,
        selection=_all(arm),
        clock=ZERO,
        **TC,
    )
    for tier in ("zero_shot", "calibrated"):
        base = cells[f"neon_term_fits.{tier}.F7"].baselines
        assert base is not None and base.beats_lookup_novel is False


# -- the schema --------------------------------------------------------------------------------------------

FROZEN_IDENTITY = (
    "question_key",
    "fingerprint",
    "labels_sha256",
    "feature_spec",
    "label_sources",
    "teacher",
    "teacher_in_test",
    "masked",
    "loco",
    "selection",
    "pre_registered",
    "exploratory",
    "servable",
    "n_folds",
    "skipped_folds",
    "fold_choices",
    "labels_content_sha256",
)
FROZEN_COUNTS = ("n", "n_neg", "n_nonmodal", "class_counts")
FROZEN_METRICS = (
    "acc",
    "macro_f1",
    "brier",
    "nll",
    "ece",
    "cov@5%",
    "cov@10%",
    "aurc",
    "auroc",
    "auroc_ci",
    "threshold_cp",
)
FROZEN_BASELINES = (
    "majority_acc",
    "lookup_acc",
    "lookup_nll",
    "novel_key",
    "lopo",
    "beats_lookup_novel",
)
FROZEN_DIAGNOSTICS = ("mean_state_cos", "calls", "input_tokens", "ms_per_decision")


def test_every_frozen_cell_field_is_present(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=12)
    arm = Arm("F7", RAW_MODEL)
    cells = task_cells(
        w.task,
        w.index,
        w.store,
        _scorers(w.store),
        FINGERPRINTS,
        selection=_all(arm),
        clock=ZERO,
        **TC,
    )
    for key, cell in cells.items():
        data = json.loads(json.dumps(cell.model_dump(by_alias=True, mode="json"), allow_nan=False))
        for name in FROZEN_IDENTITY:
            assert name in data, (key, name)
        assert data["fingerprint"] == FINGERPRINTS[RAW_MODEL] and len(data["question_key"]) == 16
        assert (
            data["loco"] is True and data["teacher"] is False and data["teacher_in_test"] is False
        )
        assert data["masked"] is True and data["servable"] is True and data["n_folds"] == 7
        for name in FROZEN_COUNTS:
            assert data["counts"][name] is not None, (key, name)
        for name in FROZEN_METRICS:
            assert data["metrics"][name] is not None, (key, name)
        assert set(data["metrics"]["threshold_cp"]) == {"0.05", "0.10"}
        assert data["metrics"]["auroc_ci"]["B"] == B and data["metrics"]["auroc_ci"]["seed"] == 0
        for name in FROZEN_BASELINES:
            assert data["baselines"][name] is not None, (key, name)
        for name in ("n", "acc", "auroc", "nll"):
            assert name in data["baselines"]["novel_key"]
        assert {"acc", "nll"} <= set(data["baselines"]["lopo"]) and data["baselines"]["lopo"][
            "n_folds"
        ] == 2
        for name in FROZEN_DIAGNOSTICS:
            assert name in data["diagnostics"], (key, name)
        assert len(data["items"]) == data["counts"]["n"]
        assert {"target_sha256", "option_key", "card", "label", "probs", "novel"} == set(
            data["items"][0]
        )


# -- refusals ----------------------------------------------------------------------------------------------


def test_refusals(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=3)
    scorers = _scorers(w.store)
    with pytest.raises(CellError, match="control framing"):
        score_arm(w.task, w.index, w.store, scorers[RAW_MODEL], "F1")
    empty = Task("neon_term_fits", TASKS["term.fits"], [], "s", "s")
    with pytest.raises(CellError, match="no items"):
        score_arm(empty, w.index, w.store, scorers[RAW_MODEL], "F7")
    with pytest.raises(CellError, match="no context text"):
        score_arm(w.task, w.index, w.store, scorers[RAW_MODEL], "F9")
    gone = rank_world(tmp_path / "gone", n_targets=3)
    store = FeatureStore(tmp_path / "empty" / "features", FP, dim=DIM)
    with pytest.raises(FeatureMissing):
        score_arm(gone.task, gone.index, store, scorers[RAW_MODEL], "F7")
    with pytest.raises(CellError, match="clm_model_fp"):
        score_arm(
            w.task,
            w.index,
            w.store,
            OfflineScorer(LATEST_MODEL, random_head(1, hidden_size=DIM)),
            "F7",
        )
    sc = score_arm(w.task, w.index, w.store, scorers[RAW_MODEL], "F7")
    with pytest.raises(CellError, match="was not scored"):
        tier_cell(
            w.task,
            {sc.arm: sc},
            "zero_shot",
            arm=Arm("F9", RAW_MODEL),
            fingerprint=None,
            selection="none",
            pre_registered=True,
            exploratory=False,
            **COMMON,
        )
    folds = list(w.task.leave_one_card_out())
    n = len(w.task.items)
    good = calibrate.zero_shot_probs("rank_fit", sc.x)
    kw: dict[str, Any] = {
        "tier": "zero_shot",
        "framing": "F7",
        "selection": "none",
        "pre_registered": True,
        "exploratory": False,
        "servable": False,
        **COMMON,
    }
    with pytest.raises(CellError, match="not probability distributions"):
        assemble_cell(w.task, range(n), good * 2, folds, **kw)
    with pytest.raises(CellError, match="are not the pooled items"):
        assemble_cell(w.task, range(n - 1), good[:-1], folds, **kw)
    with pytest.raises(CellError, match="pooled twice"):
        assemble_cell(w.task, [0, 0], good[:2], [Fold("x", [0, 0], [])], **kw)
    teacher = Task(
        w.task.name,
        w.task.spec,
        w.task.items,
        "s",
        "s",
        cards=w.task.cards,
        option_keys=w.task.option_keys,
        meta={**w.task.meta, "label_sources": {"teacher": 3}},
    )
    with pytest.raises(CellError, match="teacher"):
        assemble_cell(teacher, range(n), good, folds, **kw)
    with pytest.raises(CellError, match="no fingerprint"):
        task_cells(
            w.task, w.index, w.store, scorers, {}, selection=_all(Arm("F7", RAW_MODEL)), **TC
        )


def test_text_index(tmp_path: Path) -> None:
    a = _row("term.fits", "F7", "t" * 64, "", "context", "one")
    same = _row("term.fits", "F7", "t" * 64, "", "context", "one")
    other = _row("term.fits", "F7", "t" * 64, "", "context", "two")
    assert len(TextIndex([a, same])) == 1
    with pytest.raises(CellError, match="disagree"):
        TextIndex([a, other])
    idx = TextIndex([a])
    assert idx.text("term.fits", "F7", "t" * 64, "", "context") == "one"
    with pytest.raises(CellError, match="no options"):
        idx.options("term.fits", "F7", "t" * 64)
    m1 = type("M", (), {"labels_sha256": "x", "rows": (a,)})()
    m2 = type("M", (), {"labels_sha256": "y", "rows": (a,)})()
    with pytest.raises(CellError, match="different snapshots"):
        TextIndex.from_manifests(m1, m2)  # type: ignore[arg-type]
    assert len(TextIndex.from_manifests(m1, m1)) == 1  # type: ignore[arg-type]


# -- the results file ------------------------------------------------------------------------------------


def test_results_file_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    term = rank_world(tmp_path / "t", n_targets=12)
    aspect = choice_world(tmp_path / "a", "column.aspect", n_targets=4)
    store = term.store
    # one store holds both worlds' texts, as the real store holds every task's
    store.add(
        aspect.store.embedded_texts(),
        aspect.store.get(aspect.store.embedded_texts()),
        aspect.store.token_counts(aspect.store.embedded_texts()),
    )
    index = TextIndex([*term.rows, *aspect.rows])
    arm = Arm("F7", RAW_MODEL)

    def run() -> Any:
        return run_tier_cells(
            {"neon_term_fits": term.task, "neon_aspect": aspect.task},
            index,
            store,
            _scorers(store),
            FINGERPRINTS,
            selections={"term.fits": _all(arm)},
            date="2026-10-01",
            clock=ZERO,
            **TC,
        )

    results = run()
    assert results.name == "tiers" and results.labels_sha256 == "a" * 64
    assert len(results.cells) == 8 and all(k == c.key for k, c in results.cells.items())
    path = write_results(results, tmp_path / "out")
    back = load_results(path)
    assert back == results
    md = markdown_table(results)
    assert "| neon_term_fits.calibrated.F7 |" in md and "| neon_aspect.zero_shot.F7@clm-raw |" in md
    assert "auroc [one-sided 95% bounds; 90% interval]" in md
    again = run()
    a = json.dumps(
        results.model_dump(by_alias=True, mode="json", exclude={"environment"}), sort_keys=True
    )
    b = json.dumps(
        again.model_dump(by_alias=True, mode="json", exclude={"environment"}), sort_keys=True
    )
    assert a == b


def test_m0_cells_still_load() -> None:
    committed = load_results(
        Path(__file__).resolve().parents[2] / "bench/results/2026-09-29/baselines.json"
    )
    for key, cell in committed.cells.items():
        assert (
            cell.key == key and cell.variant is None and cell.model is None and cell.items is None
        )
        assert cell.baselines is not None and cell.baselines.novel_key_lookup is None


def test_fold_choice_and_arm_parsing() -> None:
    assert Arm.of({"framing": "F9", "model": RAW_MODEL}) == Arm("F9", RAW_MODEL)
    assert Arm("F9", RAW_MODEL).name == "F9@clm-raw"
    fc = FoldChoice.of({"framing": "F4", "model": LATEST_MODEL, "nll": 0.3})
    assert fc.arm == Arm("F4", LATEST_MODEL) and fc.record == {"nll": 0.3}
    assert FoldChoice.of(fc) is fc and FoldChoice.of(Arm("F7", RAW_MODEL)).record == {}
    sel = X1Selection.of({"framing": "F7", "model": RAW_MODEL}, {"c": Arm("F7", RAW_MODEL)})
    assert sel.a1 == Arm("F7", RAW_MODEL) and sel.fold_choices["c"].arm == sel.a1
    assert isinstance(ArmScores, type)


# -- review findings: bookkeeping, exactness, leakage, the gate's two halves -------------------------


def _relabel(task: Task, labels: Sequence[int], weights: Sequence[float] | None = None) -> Task:
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


def _subset(task: Task, keep: Sequence[int]) -> Task:
    return Task(
        task.name,
        task.spec,
        [task.items[i] for i in keep],
        "s",
        "s",
        cards=[task.cards[i] for i in keep],
        weights=[task.weights[i] for i in keep],
        products=[task.products[i] for i in keep],
        option_keys=[task.option_keys[i] for i in keep],
        meta=dict(task.meta),
    )


def test_a_guarded_fold_without_an_arm_is_accounted_for_in_both_tiers(tmp_path: Path) -> None:
    """A fold that the guards would skip and whose inner selection chose no arm is listed in
    both tiers' ``skipped_folds`` (with the inner outcome), never lost (§9.5)."""

    def few_no(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = np.where(rng.random(len(s)) < calibrate.sigmoid(1.5 * s), 0, 1)
        y[:36] = 0
        y[:3] = 1  # card0: 3 No held out < 5
        return y

    w = rank_world(tmp_path, labels=few_no)
    arm = Arm("F7", RAW_MODEL)
    sel = X1Selection.of(arm, {c: arm for c in CARDS[1:]}, fold_skips={CARDS[0]: "inner_k1"})
    cells = task_cells(
        w.task, w.index, w.store, _scorers(w.store), FINGERPRINTS, selection=sel, clock=ZERO, **TC
    )
    for tier in ("zero_shot", "calibrated"):
        cell = cells[f"neon_term_fits.{tier}.F7"]
        assert cell.skipped_folds[CARDS[0]] == "inner_k1"
        assert cell.n_folds + len(cell.skipped_folds) == len(CARDS)
        assert set(cell.fold_choices) == set(CARDS)
    with pytest.raises(CellError, match="both an arm and a skip"):
        X1Selection.of(arm, {CARDS[0]: arm}, fold_skips={CARDS[0]: "inner_k1"})


def test_each_fold_is_scored_exactly_by_its_own_arm(tmp_path: Path) -> None:
    """Folds choosing different arms: every held-out card's zero-shot probability is σ(s) of its
    own fold's arm and its calibrated one that fold's stored Platt applied to the same s,
    exactly; a fold choosing F7@clm-raw does not agree with an A1 of F7@clm-latest."""
    w = rank_world(tmp_path, framing_ids=("F7", "F9"), n_targets=12, seed=6)
    scorers = _scorers(w.store)
    latest, raw, f9 = Arm("F7", LATEST_MODEL), Arm("F7", RAW_MODEL), Arm("F9", RAW_MODEL)
    arms = {CARDS[0]: raw, CARDS[1]: f9, **{c: latest for c in CARDS[2:]}}
    sel = X1Selection.of(latest, arms)
    cells = task_cells(
        w.task, w.index, w.store, scorers, FINGERPRINTS, selection=sel, clock=ZERO, **TC
    )
    scores = {
        a: score_arm(w.task, w.index, w.store, scorers[a.model], a.framing) for a in arms.values()
    }
    ids = item_identities(w.task)
    zero = _items(cells["neon_term_fits.zero_shot.F7"])
    cal_cell = cells["neon_term_fits.calibrated.F7"]
    cal = _items(cal_cell)
    for fold in w.task.leave_one_card_out():
        a = arms[fold.held_out]
        x = scores[a].x[np.asarray(fold.test)]
        want_zero = calibrate.zero_shot_probs("rank_fit", x)
        fit = calibrate.load_calibrator(
            cal_cell.diagnostics["per_fold"][fold.held_out]["calibrator"]
        )
        want_cal = fit.probs(x)
        for k, i in enumerate(fold.test):
            assert zero[ids[i]] == [float(v) for v in want_zero[k]]
            assert cal[ids[i]] == [float(v) for v in want_cal[k]]
    agreement = cells["neon_term_fits.zero_shot.F7"].diagnostics["fold_agreement_with_a1"]
    assert agreement == {"agree": 5, "folds": 7}  # same framing on another model disagrees


def test_x1s_selection_then_the_tier_cells_never_read_the_held_out_labels(tmp_path: Path) -> None:
    """End to end (D27): X1 chooses each fold's arm on the six training cards
    (``framing.nested_selection``) and the tier cells score the held-out card with it; flipping
    every label (and weight) of card h changes no prediction on card h, at either tier."""
    from mesa_clm.bench import framing as x1
    from mesa_clm.learn.features import Manifest

    w = rank_world(tmp_path, framing_ids=("F7", "F9"), n_targets=12, a=2.5, b=-0.3, seed=12)
    scorers = _scorers(w.store)
    manifest = Manifest(
        snapshot="synthetic",
        labels_sha256="a" * 64,
        tasks=("term.fits",),
        framings=("F7", "F9"),
        rows=tuple(w.rows),
    )

    def cells_of(task: Task) -> dict[str, BenchCell]:
        items = x1.score_items(task, manifest, w.store, scorers, framings=("F7", "F9"))
        record, _ = x1.evaluate_task(
            task, items, labels_sha256="a" * 64, labels_content_sha256="b" * 64,
            registered=False, B=B,
        )  # fmt: skip
        sel = X1Selection.of(**x1.selection(record))
        if sel.a1 is None:  # keep the test about the nested path whatever the full run says
            sel = X1Selection.of(
                {"framing": "F7", "model": RAW_MODEL}, sel.fold_choices, sel.fold_skips
            )
        return task_cells(
            task, w.index, w.store, scorers, FINGERPRINTS, selection=sel, clock=ZERO, **TC
        )

    base = cells_of(w.task)
    nested_keys = [k for k, c in base.items() if c.selection == "nested"]
    assert len(nested_keys) == 2
    ids = item_identities(w.task)
    for held in (CARDS[2], CARDS[5]):
        flipped = _relabel(
            w.task,
            [1 - y if c == held else y for y, c in zip(w.task.labels, w.task.cards, strict=True)],
            [0.6 if c == held else wt for wt, c in zip(w.task.weights, w.task.cards, strict=True)],
        )
        other = cells_of(flipped)
        mine = [ids[i] for i, c in enumerate(w.task.cards) if c == held]
        for key in nested_keys:
            before, after = _items(base[key]), _items(other[key])
            assert base[key].fold_choices[held] == other[key].fold_choices[held]
            assert all(before.get(k) == after.get(k) for k in mine), (key, held)


def test_the_label_weights_change_the_calibrated_fit(tmp_path: Path) -> None:
    """D20 at cell level: the calibrator is fitted with the label weights; other weights give
    another fit, and the cell reports the unweighted fit next to it (§6.6)."""
    w = rank_world(tmp_path, n_targets=12, seed=3)
    arm = Arm("F7", RAW_MODEL)
    heavy = _relabel(w.task, w.task.labels, [3.0 if y == 0 else 0.5 for y in w.task.labels])
    fits = []
    sensitivities = []
    for task in (w.task, heavy):
        cells = task_cells(
            task, w.index, w.store, _scorers(w.store), FINGERPRINTS, selection=_all(arm),
            tiers=("calibrated",), clock=ZERO, **TC,
        )  # fmt: skip
        cell = cells["neon_term_fits.calibrated.F7"]
        fits.append(cell.diagnostics["per_fold"][CARDS[0]]["calibrator"])
        sens = cell.diagnostics["sensitivity_unweighted"]
        assert sens["n"] == cell.counts.n and set(sens["weighted"]) == {
            "nll", "ece", "acc", "brier", "in_the_large",
        }  # fmt: skip
        assert cell.diagnostics["fitters"]["platt"]["targets"] == "hard"
        assert cell.diagnostics["fitters"]["temperature"]["bounds"] == [1e-4, 1e4]
        sensitivities.append(sens)
    assert fits[1]["b"] > fits[0]["b"] + 0.5  # heavier Yes weights raise the intercept
    assert fits[1]["weight_sum"] != fits[0]["weight_sum"]
    # §6.6: the unweighted fit ignores the weights (the same for both), the weighted one leans
    # towards Yes when the Yes items weigh more
    assert sensitivities[0]["unweighted"] == sensitivities[1]["unweighted"]
    heavy_sens = sensitivities[1]
    assert heavy_sens["weighted"]["in_the_large"][0] > heavy_sens["unweighted"]["in_the_large"][0]
    assert heavy_sens["weighted"] != heavy_sens["unweighted"]


def test_the_calibrated_floor_at_its_boundary(tmp_path: Path) -> None:
    """A calibrator is fitted on 100 training items, not on 99 (§6.3): a K > 2 task has no
    30/5 guard, so the floor alone decides."""
    w = choice_world(tmp_path, "avu.value_kind", n_targets=20)
    per = {c: [i for i, cc in enumerate(w.task.cards) if cc == c] for c in CARDS}
    for total, evaluated in ((100, True), (99, False)):
        sizes = [17, 17, 17, 17, 16, total - 84]  # six training cards, then the held-out one
        keep = [i for c, n in zip(CARDS[:6], sizes, strict=True) for i in per[c][:n]]
        task = _subset(w.task, keep + per[CARDS[6]])
        cells = task_cells(
            task, w.index, w.store, _scorers(w.store), FINGERPRINTS, tiers=("calibrated",),
            clock=ZERO, **TC,
        )  # fmt: skip
        cell = cells["neon_value_kind.calibrated.F7"]
        assert (CARDS[6] not in cell.skipped_folds) is evaluated, cell.skipped_folds.get(CARDS[6])
        if not evaluated:
            assert (
                cell.skipped_folds[CARDS[6]] == "below_floor 99 < 100 training items (calibrated)"
            )


def _novel_gate(
    tmp_path: Path, probs_of: Callable[[np.ndarray, list[str]], np.ndarray]
) -> dict[str, Any]:
    """``beats_lookup_novel`` on a two-class task whose lookup keys are all novel and whose
    cards' Yes rates run from 0.2 to 0.8, for the cell predictions ``probs_of(labels, cards)``."""
    from mesa_clm.bench.baselines import evaluate_lookup, novel_key_block

    w = rank_world(tmp_path, n_targets=12, seed=21)
    labels = []
    for k, card in enumerate(CARDS):
        idx = [i for i, c in enumerate(w.task.cards) if c == card]
        n_yes = round((0.2 + 0.1 * k) * len(idx))
        labels.extend((0 if j < n_yes else 1) for j in range(len(idx)))
    task = _relabel(w.task, labels)
    ev = evaluate_lookup(task)
    assert ev.novel.all()
    probs = probs_of(ev.labels, ev.cards)
    mine = novel_key_block(dataclasses.replace(ev, probs=probs), binary=True, B=B, seed=0)
    beats, detail = C.beats_lookup_novel(task, ev, probs, mine, B=B)
    detail["beats"] = beats
    # the rule R record is the statistics' own at the registered seed 0 (review finding)
    want = stats.rule_r("nll", probs, ev.probs, ev.labels, ev.cards, B=B, seed=0, alpha=0.05)
    assert detail["nll_rule_r"] == C._json_safe(want.as_dict())
    return detail


def test_beats_lookup_novel_needs_both_conditions(tmp_path: Path) -> None:
    """PR "Lookup as a probability model": a cell that ranks well but whose NLL does not beat
    ``lookup_prob`` fails, and so does one that beats it on NLL with an AUROC bound at 0.5."""

    def overconfident(labels: np.ndarray, cards: list[str]) -> np.ndarray:
        rng = np.random.default_rng(3)
        right = rng.random(len(labels)) < 0.85
        yes = (labels == 0) == right
        p = np.where(yes, 0.999, 0.001)
        return np.column_stack([p, 1 - p])

    ranked = _novel_gate(tmp_path / "a", overconfident)
    assert ranked["beats"] is False and ranked["auroc_condition"] is True
    assert ranked["reason"].startswith("nll_") and ranked["nll_rule_r"]["passed"] is False

    def flat(labels: np.ndarray, cards: list[str]) -> np.ndarray:
        rate = float(np.mean(labels == 0))  # the pooled rate everywhere: no ranking at all
        return np.column_stack([np.full(len(labels), rate), np.full(len(labels), 1 - rate)])

    calibrated = _novel_gate(tmp_path / "b", flat)
    assert calibrated["beats"] is False and calibrated["nll_rule_r"]["passed"] is True
    assert calibrated["auroc_condition"] is False
    assert calibrated["reason"] == "auroc_lower_bound_not_above_0.5"


def test_an_unregistered_run_stamps_every_cell_exploratory(tmp_path: Path) -> None:
    w = rank_world(tmp_path, n_targets=12)
    arm = Arm("F7", RAW_MODEL)
    cells = task_cells(
        w.task, w.index, w.store, _scorers(w.store), FINGERPRINTS, selection=_all(arm),
        clock=ZERO, **{**TC, "registered": False},
    )  # fmt: skip
    assert cells and all(c.exploratory and not c.pre_registered for c in cells.values())
    assert all("not the pre-registered run" in c.notes for c in cells.values())
    with pytest.raises(CellError, match="cannot be the registered one"):
        run_tier_cells(
            {"neon_term_fits": w.task}, w.index, w.store, _scorers(w.store), FINGERPRINTS,
            selections={"term.fits": _all(arm)}, date="d", deviations=["tiers"], **TC,
        )  # fmt: skip


def test_a_tier_run_off_the_registration_is_unregistered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§13.2: other labels or another B or seed make the tier run unregistered whatever its
    caller says (every cell ``pre_registered: false``, ``exploratory: true``, the deviation in the
    notes); with the registration's labels and settings it stays registered, and its tasks must
    have their published counts (§1.3)."""
    from mesa_clm.bench import registered as reg

    w = rank_world(tmp_path, n_targets=12)
    arm = Arm("F7", RAW_MODEL)
    kw: dict[str, Any] = {"selections": {"term.fits": _all(arm)}, "date": "d", "clock": ZERO}
    tasks = {"neon_term_fits": w.task}
    off = run_tier_cells(tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **TC)
    assert "labels_sha256 aaaaaaaaaaaa" in off.notes[0] and "B 200 is not 2000" in off.notes[0]
    assert all(c.exploratory and not c.pre_registered for c in off.cells.values())
    stand_in = reg.Registration(
        snapshot="s", labels_sha256="a" * 64, labels_content_sha256="b" * 64,
        published="synthetic", B=B, fingerprints=FINGERPRINTS,
    )  # fmt: skip
    monkeypatch.setattr(reg, "REGISTERED", stand_in)
    on = run_tier_cells(tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **TC)
    nested = on.cells["neon_term_fits.zero_shot.F7"]
    assert nested.pre_registered and not nested.exploratory
    # a subset of the tiers is not the registered run (the producer says so itself)
    zero = run_tier_cells(
        tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **TC,
        tiers=("zero_shot",),
    )  # fmt: skip
    assert "tiers ['zero_shot'] is not ['zero_shot', 'calibrated']" in zero.notes[0]
    assert not zero.cells["neon_term_fits.zero_shot.F7"].pre_registered
    # nor is another model fingerprint, or another framings lock (§13.1)
    moved = {**FINGERPRINTS, RAW_MODEL: {**FINGERPRINTS[RAW_MODEL], "encoder_fp": "e" * 12}}
    other = run_tier_cells(tasks, w.index, w.store, _scorers(w.store), moved, **kw, **TC)
    assert "clm-raw fingerprint: encoder_fp eeeeeeeeeeee" in other.notes[0]
    assert not other.cells["neon_term_fits.zero_shot.F7"].pre_registered
    monkeypatch.setattr(C.framings, "lock_sha", lambda: "0" * 64)
    rotated = run_tier_cells(tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **TC)
    assert "framings lock_sha 000000000000" in rotated.notes[0]
    monkeypatch.undo()
    monkeypatch.setattr(reg, "REGISTERED", stand_in)
    seeded = run_tier_cells(
        tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **{**TC, "seed": 3}
    )
    assert "seed 3 is not 0" in seeded.notes[0]
    assert not seeded.cells["neon_term_fits.zero_shot.F7"].pre_registered
    per_card = {c: w.task.cards.count(c) for c in CARDS}
    counts = reg.TaskCounts(len(w.task.items), w.task.class_counts(), per_card, 0.5)
    monkeypatch.setattr(reg, "REGISTERED", dataclasses.replace(stand_in, counts={
        "neon_term_fits": counts}))  # fmt: skip
    run_tier_cells(tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **TC)
    wrong = dataclasses.replace(counts, n=counts.n + 1)
    monkeypatch.setattr(reg, "REGISTERED", dataclasses.replace(stand_in, counts={
        "neon_term_fits": wrong}))  # fmt: skip
    with pytest.raises(reg.RegistrationError, match="not the published"):
        run_tier_cells(tasks, w.index, w.store, _scorers(w.store), FINGERPRINTS, **kw, **TC)


# -- the second review round: interleaved cards, report-only values, X1's trace pointer ----------


def _identity_order(task: Task) -> Task:
    """``task`` with its items in the bench's order, sorted by D1 identity (target_sha256,
    option_key), as ``learn.labels.labelled_targets`` returns them: the cards interleave."""
    ids = item_identities(task)
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    return Task(
        task.name, task.spec, [task.items[i] for i in order], "s", "s",
        cards=[task.cards[i] for i in order], weights=[task.weights[i] for i in order],
        products=[task.products[i] for i in order],
        option_keys=[task.option_keys[i] for i in order], meta=dict(task.meta),
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["term.fits", "column.annotate", "column.aspect"])
def test_cells_on_interleaved_cards_match_an_independent_computation(
    tmp_path: Path, kind: str
) -> None:
    """Review finding: every synthetic world listed its items card by card, where the cells'
    realignment of fold-pooled predictions does nothing. In the bench's identity order (cards
    interleaved) each item's zero-shot and calibrated prediction, the pooled NLL, the §6.6
    sensitivity block and the raw-score AUROC equal a computation keyed by identity."""
    if kind == "term.fits":
        w = rank_world(tmp_path, n_targets=12, seed=41, a=2.0, b=-0.4)
    else:
        w = choice_world(tmp_path, kind, n_targets=40, seed=5, novel=True)
    task = _identity_order(w.task)
    assert sum(1 for a, b in zip(task.cards, task.cards[1:], strict=False) if a != b) > 50
    arm = Arm("F7", RAW_MODEL)
    sel = _all(arm) if kind == "term.fits" else None
    cells = task_cells(
        task, w.index, w.store, _scorers(w.store), FINGERPRINTS, selection=sel, clock=ZERO, **TC
    )
    sc = score_arm(task, w.index, w.store, _scorers(w.store)[RAW_MODEL], "F7")
    labels = np.asarray(task.labels)
    weights = np.asarray(task.weights)
    ids = item_identities(task)
    suffix = "" if kind == "term.fits" else "@clm-raw"
    for tier in ("zero_shot", "calibrated"):
        cell = cells[f"{task.name}.{tier}.F7{suffix}"]
        want: dict[tuple[str, str], np.ndarray] = {}
        plain: dict[tuple[str, str], np.ndarray] = {}
        for fold in task.leave_one_card_out():
            te, tr = np.asarray(fold.test), np.asarray(fold.train)
            if tier == "zero_shot":
                probs = calibrate.zero_shot_probs(sc.shape, sc.x[te])
                unw = probs
            else:
                fit = calibrate.fit_calibrator(sc.shape, sc.x[tr], labels[tr], weights[tr])
                probs = fit.probs(sc.x[te])
                unw = calibrate.fit_calibrator(sc.shape, sc.x[tr], labels[tr], None).probs(sc.x[te])
            for k, i in enumerate(te):
                want[ids[i]], plain[ids[i]] = probs[k], unw[k]
        got = {(it.target_sha256, it.option_key): it for it in cell.items or []}
        assert set(got) == set(want), tier
        index = {key: i for i, key in enumerate(ids)}
        for key, it in got.items():
            assert it.label == labels[index[key]] and it.card == task.cards[index[key]]
            assert it.probs == pytest.approx(want[key].tolist(), abs=1e-12), (tier, key)
        keys = sorted(want)
        p = np.stack([want[k] for k in keys])
        y = np.asarray([labels[index[k]] for k in keys])
        nll = float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))
        assert cell.metrics is not None and cell.metrics.nll == pytest.approx(nll, abs=1e-12)
        if tier == "calibrated":
            sens = cell.diagnostics["sensitivity_unweighted"]
            u = np.stack([plain[k] for k in keys])
            assert sens["weighted"] == C.calibration_summary(p, y)
            assert sens["unweighted"] == C.calibration_summary(u, y)
        elif task.binary:
            raw = sc.x if sc.shape == "rank_fit" else calibrate.logit_difference(sc.x)
            score = np.asarray([raw[index[k]] for k in keys])
            cards = [task.cards[index[k]] for k in keys]
            ci = stats.auroc_ci(score, y, cards, B=B, seed=0)
            got_raw = cell.diagnostics["auroc_raw_score"]
            assert got_raw["point"] == pytest.approx(stats.auroc(score, y), abs=1e-15)
            assert (got_raw["lower"], got_raw["upper"]) == pytest.approx((ci.lower, ci.upper))


def test_n_saturated_counts_both_rounded_ends(tmp_path: Path) -> None:
    """§3.5, §9.10 (value, review finding): ``n_saturated`` counts the zero-shot p(Yes) that
    float64 rounds to exactly 1.0 (s above about 37) or to exactly 0.0 (below about −745)."""
    w = rank_world(tmp_path, n_targets=3)
    sc = score_arm(w.task, w.index, w.store, _scorers(w.store)[RAW_MODEL], "F7")
    x = sc.x.copy()
    x[:5] = [40.0, 41.0, 37.5, -800.0, -760.0]
    crafted = dataclasses.replace(sc, x=x)
    cell = tier_cell(
        w.task, {crafted.arm: crafted}, "zero_shot", arm=crafted.arm, fingerprint=None,
        selection="none", pre_registered=False, exploratory=True, **COMMON,
    )  # fmt: skip
    p = calibrate.sigmoid(x)
    assert int(np.sum(p == 1.0)) == 3 and int(np.sum(p == 0.0)) == 2
    assert cell.diagnostics["n_saturated"] == 5


def test_annotates_raw_score_auroc_is_of_the_logit_difference_yes_minus_no(tmp_path: Path) -> None:
    """§9.10 (value, review finding): at K = 2 the zero-shot raw-score AUROC is the AUROC of
    ``logit_Yes − logit_No`` for Yes, computed here from the option logits."""
    w = choice_world(tmp_path, "column.annotate", n_targets=60, a=0.3, b=0.2, seed=5)
    cells = task_cells(w.task, w.index, w.store, _scorers(w.store), FINGERPRINTS, clock=ZERO, **TC)
    zs = cells["neon_annotate.zero_shot.F7@clm-raw"]
    sc = score_arm(w.task, w.index, w.store, _scorers(w.store)[RAW_MODEL], "F7")
    d = sc.x[:, 0] - sc.x[:, 1]
    want = stats.auroc(d, w.task.labels)
    assert want > 0.6 and zs.diagnostics["auroc_raw_score"]["point"] == pytest.approx(want)


def test_a_tier_cell_keeps_x1s_trace_pointer_and_refuses_a_colliding_record(
    tmp_path: Path,
) -> None:
    """§9.5 (review finding: the pointer was overwritten by ``decision: "arm"``): X1's record
    reaches each evaluated fold's ``fold_choices`` entry unchanged, its ``x1_trace`` the JSON
    pointer of the inner trace; a selector record that uses a key the entry sets itself is
    refused, never overwritten."""
    from mesa_clm.bench import framing as x1
    from mesa_clm.learn.features import Manifest

    w = rank_world(tmp_path, framing_ids=("F7", "F9"), n_targets=12, a=2.5, b=-0.3, seed=12)
    scorers = _scorers(w.store)
    manifest = Manifest(
        snapshot="synthetic", labels_sha256="a" * 64, tasks=("term.fits",),
        framings=("F7", "F9"), rows=tuple(w.rows),
    )  # fmt: skip
    items = x1.score_items(w.task, manifest, w.store, scorers, framings=("F7", "F9"))
    record, _ = x1.evaluate_task(
        w.task, items, labels_sha256="a" * 64, labels_content_sha256="b" * 64,
        registered=False, B=B,
    )  # fmt: skip
    raw = x1.selection(record)
    sel = X1Selection.of(**raw)
    a1 = sel.a1 or Arm("F7", RAW_MODEL)
    sel = X1Selection.of(a1, sel.fold_choices, sel.fold_skips)
    cells = task_cells(
        w.task, w.index, w.store, scorers, FINGERPRINTS, selection=sel, clock=ZERO, **TC
    )
    nested = cells[f"neon_term_fits.zero_shot.{a1.framing}"]
    assert raw["fold_choices"]
    for card, given in raw["fold_choices"].items():
        entry = nested.fold_choices[card]
        assert entry["x1_trace"] == given["x1_trace"]
        assert entry["x1_trace"] == f"#/x1/tasks/neon_term_fits/nested/folds/{card}/decision"
        assert (entry["decision"], entry["outcome"]) == ("arm", "choice")
    bad = {c: {"framing": "F7", "model": RAW_MODEL, "decision": "#/x"} for c in CARDS}
    with pytest.raises(CellError, match="own key"):
        task_cells(
            w.task, w.index, w.store, scorers, FINGERPRINTS,
            selection=X1Selection.of(Arm("F7", RAW_MODEL), bad), clock=ZERO, **TC,
        )  # fmt: skip
