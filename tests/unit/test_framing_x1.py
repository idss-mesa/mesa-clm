"""X1 framing A/B (``mesa_clm.bench.framing``; ``design/m2-analysis-plan.md`` §1–§8, §12–§13) on
synthetic data only: generated cards, labels with planted signal and vectors, temporary feature
and label stores. No test reads the bench snapshot's labels or a model output on a bench item
(the M2 pre-commitment rule): the plan's rules are checked on numbers built to make each rule
fire.

Covered: weighted Platt (§6.4); the 30/5 guards and the floor of 100 training items per fit, at
its boundary (§6.2, §6.3); the within-card shuffle's seeded derangements, its mean-AUROC rule R
against brute force, its null calibration (where the drafts' expectation-of-scores comparator was
biased against every arm) and its killing a state-free signal (§5); rule (1)'s thresholds, rule
(2)'s literal cacheability reading and F7/F9 tie-break, rule (3)'s model choice, rule (4)'s D3
recommendation that never adopts, K1 and the undecidable outcome (§7); nesting with K1 folds and
``fold_choices`` (§8); the registration (a non-registered configuration stamps every cell
exploratory and is refused by ``decide_from_json``); the replay's consistency checks; scores
from a feature store equal to the offline scorer's, the candidate-only probe, the label content
check (§3, §7.11, §1.1); the items file's exact round trip, ``decide_from_json``'s replay and
recompute, the refusal to overwrite (§12); determinism.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings as fr
from mesa_clm.bench import cells as tier_cells
from mesa_clm.bench import framing as x1
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.results import BenchResults, snapshot_content_sha256
from mesa_clm.bench.tasks.base import Task
from mesa_clm.clm.fingerprint import ClmModelSpec, clm_model_fp, load_serving_lock
from mesa_clm.clm.headproj import HeadProjector, l2, random_head
from mesa_clm.identity import target_sha256
from mesa_clm.learn import calibrate
from mesa_clm.learn.features import FeatureStore, Manifest, ManifestRow, text_sha256
from mesa_clm.learn.labels import snapshot
from mesa_clm.learn.offline import OfflineScorer, pairwise_s_c, store_vectors
from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.providers.tiered import fake_fingerprint
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.serving import CLM_COMMIT
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "serving" / "serving.lock.json"
LATEST, RAW = "clm-latest", "clm-raw"
CARDS = tuple(f"DP0.0000{k}.001.synthetic{k}" for k in range(7))
FP = fake_fingerprint().encoder_fp
MODEL_FP = clm_model_fp(ClmModelSpec(head_name=LATEST, head_sha256="", clm_commit=CLM_COMMIT))
B = 200  # small resamples keep the suite fast; the registered run uses 2000
LAX: dict[str, Any] = {"require_registered": False}
# the synthetic runs' model fingerprints (a stand-in registration names them)
FPS: dict[str, dict[str, str]] = {m: {"clm_model_fp": m[:12].ljust(12, "0")} for m in (LATEST, RAW)}


# -- synthetic tasks and item tables ---------------------------------------------------------------


def _state(card: str, column: str, curie: str) -> dict[str, Any]:
    return {
        "card": {"dataset": card},
        "scope": "column",
        "column": {"name": column},
        "aspect": "measurement",
        "candidate": {"curie": curie},
    }


def _task(
    per_card: int = 40,
    targets_per_card: int = 8,
    *,
    cards: tuple[str, ...] = CARDS,
    yes_every: int = 3,
    yes_override: dict[str, int] | None = None,
    name: str = "neon_term_fits",
    task_id: str = "term.fits",
) -> Task:
    """A two-class task: ``per_card`` items per card over ``targets_per_card`` columns, every
    ``yes_every``-th item Yes (deterministic), weights 0.7 for Yes and 0.5 for No (a weight that
    follows the label, as the silver's sources do)."""
    items: list[tuple[dict[str, Any], int]] = []
    card_of: list[str] = []
    weights: list[float] = []
    products: list[str] = []
    options: list[str] = []
    for k, card in enumerate(cards):
        n_yes = (yes_override or {}).get(card)
        for j in range(per_card):
            curie = f"X:{k}{j:03d}"
            yes = j < n_yes if n_yes is not None else j % yes_every == 0
            items.append((_state(card, f"col{j % targets_per_card}", curie), 0 if yes else 1))
            card_of.append(card)
            weights.append(0.7 if yes else 0.5)
            products.append(card.rsplit(".", 1)[0])
            options.append(curie)
    task = Task(
        name,
        TASKS[task_id],
        items,
        "synthetic",
        "synthetic",
        cards=card_of,
        weights=weights,
        products=products,
        option_keys=options,
        meta={"min_weight": 0.5, "label_sources": {"synthetic": len(items)}, "masked": True},
    )
    task.meta["class_counts"] = task.class_counts()
    return task


def _layout(task: Task) -> tuple[np.ndarray, np.ndarray, int]:
    """Each item's column among its card's sorted targets, the card's target count, the widest
    card (what :class:`TaskItems` derives)."""
    targets = [target_sha256(task.task_id, s) for s in task.states]
    per: dict[str, list[str]] = {}
    for t, c in zip(targets, task.cards, strict=True):
        per.setdefault(c, [])
        if t not in per[c]:
            per[c].append(t)
    order = {c: sorted(ts) for c, ts in per.items()}
    slot = np.asarray([order[c].index(t) for t, c in zip(targets, task.cards, strict=True)])
    size = np.asarray([len(order[c]) for c in task.cards])
    return slot, size, int(size.max())


def _planted(
    task: Task,
    d: float | dict[str, float],
    *,
    shuffle_d: float | dict[str, float] = 0.0,
    seed: int,
    mc_d: float = 0.0,
) -> x1.ArmScores:
    """Scores with a planted per-card separation ``d`` (Yes items shifted by d, unit noise); the
    item's scores under the other contexts of its card carry ``shuffle_d`` (a state-free signal
    when it equals ``d``); the mean-context score carries ``mc_d``."""
    rng = np.random.default_rng(seed)
    yes = np.asarray(task.labels) == 0

    def shift(v: float | dict[str, float]) -> np.ndarray:
        if isinstance(v, dict):
            return np.asarray([v.get(c, 0.0) for c in task.cards], dtype=float)
        return np.full(len(yes), float(v))

    s = shift(d) * yes + rng.normal(size=len(yes))
    slot, size, width = _layout(task)
    ctx = np.full((len(yes), width), np.nan)
    other = shift(shuffle_d) * yes
    for i in range(len(yes)):
        ctx[i, : size[i]] = other[i] + rng.normal(size=size[i])
        ctx[i, slot[i]] = s[i]
    mean_context = mc_d * yes + rng.normal(size=len(yes))
    return x1.ArmScores(s=s, contexts=ctx, mean_context=mean_context)


def _control(task: Task, d: float, *, seed: int) -> x1.ArmScores:
    rng = np.random.default_rng(seed)
    yes = np.asarray(task.labels) == 0
    return x1.ArmScores(s=d * yes + rng.normal(size=len(yes)))


def _items(
    task: Task,
    scores: dict[str, x1.ArmScores],
    tokens: dict[str, int] | None = None,
) -> x1.TaskItems:
    n = len(task.items)
    items = x1.TaskItems(
        task=task.name,
        task_id=task.task_id,
        target=[target_sha256(task.task_id, s) for s in task.states],
        option=list(task.option_keys),
        card=list(task.cards),
        label=np.asarray(task.labels),
        weight=np.asarray(task.weights),
        scores=scores,
        scale={LATEST: 100.0, RAW: 100.0},
    )
    for framing, ctx in (tokens or {"F1": 300, "F4": 260, "F7": 250, "F9": 60}).items():
        items.tokens[framing] = x1.TokenCounts(
            context=np.full(n, ctx), candidate=np.full(n, 20), anchor=np.full(n, 12)
        )
    return items


def _registration(
    labels_sha256: str = "a" * 64, content: str = "b" * 64, *, b: int = B
) -> reg.Registration:
    """A stand-in registration for synthetic labels (never the real snapshot's) and the
    synthetic runs' fingerprints. The framings lock is the checkout's: the real registration
    keeps G1's, which DESIGN A1 rotated after the registered run, and the producers compare the
    lock they score under with the registration's."""
    return reg.Registration(
        snapshot="synthetic.parquet",
        labels_sha256=labels_sha256,
        labels_content_sha256=content,
        published="synthetic",
        B=b,
        framings_lock_sha=fr.lock_sha(),
        fingerprints=FPS,
    )


# -- a fake Evidence for the decision rules (§7) -------------------------------------------------


def _rule(passed: bool, metric: str = "nll") -> x1.RuleRSummary:
    per_card = {c: (0.1 if passed else (0.1 if i < 3 else -0.1)) for i, c in enumerate(CARDS)}
    return x1.RuleRSummary(
        metric=metric,
        passed=passed,
        reason="passed" if passed else "sign_test_failed",
        delta=0.1 if passed else 0.0,
        lower_bound=0.01,
        upper_bound=0.2,
        n_valid=200,
        n_clusters=7,
        m_c=7,
        wins=7 if passed else 3,
        needed=6,
        per_card=per_card,
    )


class FakeEvidence:
    """Arms as ``{arm: (auroc, lower, shuffle_passed, nll)}``; ``wins`` the ordered pairs that
    pass rule R on NLL; ``tokens`` per framing."""

    def __init__(
        self,
        arms: dict[str, tuple[float, float, bool, float | None]],
        wins: set[tuple[str, str]] | None = None,
        tokens: dict[str, float] | None = None,
    ) -> None:
        self._arms = arms
        self._wins = wins or set()
        self._tokens = tokens or {"F1": 300.0, "F4": 260.0, "F7": 250.0, "F9": 60.0}
        self.asked: list[tuple[str, str]] = []

    def arms(self) -> list[str]:
        return list(self._arms)

    def arm(self, arm: str) -> x1.ArmEvidence:
        auroc, lower, shuffled, nll = self._arms[arm]
        framing, model = x1.split_arm(arm)
        return x1.ArmEvidence(
            arm=arm,
            framing=framing,
            model=model,
            auroc=auroc,
            auroc_lower=lower,
            shuffle=None if framing == "F1" else _rule(shuffled, "auroc"),
            nll=nll,
            n_loco=280,
        )

    def tokens_per_target(self, framing: str) -> float | None:
        return self._tokens.get(framing)

    def compare_nll(self, a: str, b: str) -> x1.RuleRSummary:
        self.asked.append((a, b))
        return _rule((a, b) in self._wins)


GOOD = (0.8, 0.7, True)  # qualifies when paired with an NLL
BAD = (0.55, 0.45, True)


def _arms(**nll: float | None) -> dict[str, tuple[float, float, bool, float | None]]:
    """``F7_latest=0.4`` -> a qualifying arm F7@clm-latest with NLL 0.4."""
    out: dict[str, tuple[float, float, bool, float | None]] = {}
    for key, value in nll.items():
        framing, model = key.split("_")
        out[x1.arm_id(framing, f"clm-{model}")] = (*GOOD, value)
    return out


# -- Platt (§6.4) ----------------------------------------------------------------------------------


def test_x1_fits_the_code_bases_one_platt() -> None:
    """§6.4: X1's LOCO-Platt is :func:`mesa_clm.learn.calibrate.fit_platt` with the hard targets,
    so X1's NLL and the calibrated cells' come from one calibrator."""
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 3.0, 20000)
    pos = rng.random(len(s)) < calibrate.sigmoid(0.8 * s - 0.5)
    fit = x1.platt_fit(s, pos, np.ones(len(s)))
    shared = calibrate.fit_platt(s, pos, np.ones(len(s)), targets="hard")
    assert calibrate.PLATT_TARGETS == "hard"
    assert fit == shared and fit.feature == "s_c" and fit.targets == "hard"
    assert fit.converged and not fit.inverted
    assert abs(fit.a - 0.8) < 0.05 and abs(fit.b + 0.5) < 0.05
    # D20: the label weight is a sample weight; up-weighting the positives raises the intercept
    heavier = x1.platt_fit(s, pos, np.where(pos, 2.0, 1.0))
    assert heavier.b > fit.b + 0.3
    assert abs(x1.platt_fit(s, pos, None).b - fit.b) < 1e-9  # unweighted = unit weights
    assert x1.platt_fit(-s, pos, np.ones(len(s))).inverted  # a < 0 is flagged, not refused
    assert x1.platt_fit(s, pos, np.ones(len(s))) == fit  # deterministic, bit for bit
    sep = x1.platt_fit([-2.0, -1.0, 1.0, 2.0], [False, False, True, True], [1.0] * 4)
    assert np.isfinite(sep.a) and sep.a > 0  # a separable fold still has a finite fit
    with pytest.raises(ValueError):
        x1.platt_fit([], [], [])
    with pytest.raises(ValueError):
        x1.platt_fit([1.0, np.nan], [True, False], [1.0, 1.0])


# -- folds, guards, floor (§6.1-§6.3) -------------------------------------------------------------


def test_guards_skip_a_fold_for_every_arm_and_nesting_ignores_the_outer_guard() -> None:
    thin = CARDS[2]
    task = _task(yes_override={thin: 3})  # 3 Yes held out < 5: the fold is guarded
    items = _items(
        task,
        {
            "F7@clm-latest": _planted(task, 2.0, seed=1),
            "F9@clm-latest": _planted(task, 1.0, seed=2),
        },
    )
    ev = x1.ItemEvidence(items, B=B)
    assert list(ev.plan.skipped) == [thin]
    assert ev.plan.skipped[thin].startswith("insufficient_heldout_per_class")
    assert thin not in {items.card[i] for i in ev.plan.pool}
    assert ev.arm("F7@clm-latest").n_loco == ev.arm("F9@clm-latest").n_loco == len(ev.plan.pool)
    assert set(ev.loco("F7@clm-latest").fits) == set(CARDS) - {thin}
    # nested: the guarded card's fold still chooses an arm on its 6 training cards; the outer
    # guard is the tier cells' (§8.4), so X1 records only inner outcomes as skips
    nested = x1.nested_selection(items, B=B)
    assert nested.fold_choices[thin].outcome == "choice"
    assert nested.skipped_folds == {}


def test_the_floor_applies_to_every_fit_at_its_boundary() -> None:
    """§6.3: no calibrator is fitted on fewer than 100 items: a fold training on 99 is skipped
    (``below_floor``), one training on exactly 100 runs."""
    small = _task(per_card=14)  # 98 items, every fold trains on 84
    sitems = _items(small, {"F7@clm-latest": _planted(small, 3.0, seed=3)})
    sev = x1.ItemEvidence(sitems, B=B)
    assert len(sev.plan.pool) == 0 and len(sev.plan.skipped) == 7
    assert set(sev.plan.skipped.values()) == {
        f"below_floor 84 < {x1.CALIBRATION_FLOOR} training items (calibrated)"
    }
    assert sev.arm("F7@clm-latest").nll is None
    # 6 cards of 20 items train each fold of a 7th card: 120 items. Hold out a 20-item card from
    # a set whose other cards hold exactly 100 or 99 items.
    for n_last, expect in ((100 - 5 * 18, True), (99 - 5 * 18, False)):
        sizes = [18] * 5 + [n_last, 40]
        per_card = {c: n for c, n in zip(CARDS, sizes, strict=True)}
        task = _task(per_card=max(sizes), yes_every=2)
        keep = [
            i
            for i, c in enumerate(task.cards)
            if sum(1 for j in range(i) if task.cards[j] == c) < per_card[c]
        ]
        sub = Task(
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
        items = _items(sub, {"F7@clm-latest": _planted(sub, 2.0, seed=4)})
        plan = x1.fold_plan(items)
        last = CARDS[6]  # held out, it leaves the other six cards: 5*18 + n_last items
        n_train = len(sub.items) - per_card[last]
        assert n_train == 90 + n_last
        assert (last not in plan.skipped) is expect, plan.skipped.get(last)
        if not expect:
            assert plan.skipped[last] == "below_floor 99 < 100 training items (calibrated)"


# -- the shuffle (§5) ------------------------------------------------------------------------------


def test_shuffle_draws_are_seeded_uniform_derangements() -> None:
    draws = x1.shuffle_draws({"b": 3, "a": 5, "solo": 1}, k=4000, seed=0)
    assert list(draws) == ["a", "b", "solo"] and draws["solo"] is None
    again = x1.shuffle_draws({"solo": 1, "a": 5, "b": 3}, k=4000, seed=0)
    assert all(
        (draws[c] is None and again[c] is None) or np.array_equal(draws[c], again[c])  # type: ignore[arg-type]
        for c in draws
    )
    for card, m in (("a", 5), ("b", 3)):
        perm = draws[card]
        assert perm is not None and perm.shape == (4000, m)
        assert not np.any(perm == np.arange(m))  # derangements: no target keeps its context
        assert all(sorted(row) == list(range(m)) for row in perm[:50])
    # m = 3 has exactly two derangements, drawn about equally often
    rows = {tuple(r) for r in draws["b"].tolist()}  # type: ignore[union-attr]
    assert rows == {(1, 2, 0), (2, 0, 1)}
    share = float(np.mean(draws["b"][:, 0] == 1))  # type: ignore[index]
    assert abs(share - 0.5) < 0.03
    with pytest.raises(ValueError):
        x1.derangements(1, 2, np.random.default_rng(0))


def test_shuffled_scores_come_from_other_targets_of_the_same_card() -> None:
    task = _task(per_card=12, targets_per_card=4)
    items = _items(task, {"F7@clm-latest": _planted(task, 1.0, seed=5)})
    ctx = items.scores["F7@clm-latest"].contexts
    assert ctx is not None
    shuffled = items.shuffled("F7@clm-latest", 50, 0)
    idx = items.shuffle_index(50, 0)
    assert shuffled.shape == (50, items.n)
    assert not np.any(idx == items.slot[None, :])  # never the item's own context
    for i in (0, 13, 77):
        assert set(shuffled[:, i].tolist()) <= set(ctx[i, : items.card_size[i]].tolist())
    # every item of one target takes the same target's context in a draw: a state shuffle
    same = [i for i in range(items.n) if items.target[i] == items.target[0]]
    assert len({int(idx[3, i]) for i in same}) == 1
    assert items.n_unshuffled == 0


def test_context_scores_and_the_own_column() -> None:
    rng = np.random.default_rng(5)
    zt = l2(rng.normal(size=(4, 16)))
    owner = np.asarray([0, 0, 1, 2, 3, 3, 1])
    cards = ["c", "c", "c", "d", "d", "d", "c"]
    zc = l2(rng.normal(size=(7, 16)))
    za = np.broadcast_to(l2(rng.normal(size=(1, 16)))[0], (7, 16))
    s = pairwise_s_c(100.0, zt[owner], zc, za)
    ctx = x1.context_scores(zt, owner, cards, zc, za, 100.0)
    assert ctx.shape == (7, 2)  # cards c and d each have two targets
    rows = {"c": [0, 1], "d": [2, 3]}
    for i in range(7):
        for j, t in enumerate(rows[cards[i]]):
            want = pairwise_s_c(100.0, zt[t][None, :], zc[i][None, :], za[i][None, :])[0]
            assert ctx[i, j] == pytest.approx(want, rel=1e-13, abs=1e-12)
        # the own column is s (the same arithmetic; SIMD sums may differ in the last bit)
        assert ctx[i, rows[cards[i]].index(int(owner[i]))] == pytest.approx(
            s[i], rel=1e-13, abs=1e-12
        )


def test_the_mean_auroc_rule_r_is_the_brute_force_mean_over_draws() -> None:
    rng = np.random.default_rng(7)
    n = 140
    labels = (rng.random(n) < 0.35).astype(int)
    cards = [CARDS[i % 7] for i in range(n)]
    s = rng.normal(size=n) + (labels == 0) * 0.9
    shuffles = rng.normal(size=(30, n)) + (labels == 0) * 0.3
    r = stats.rule_r_auroc_mean(s, shuffles, labels, cards, B=300)
    brute = stats.auroc(s, labels) - np.mean([stats.auroc(row, labels) for row in shuffles])
    assert abs(r.delta - brute) < 1e-12
    w = stats.bootstrap_weights(cards, B=300, seed=0)
    for b in (0, 17, 299):
        idx = np.repeat(np.arange(n), w[b].astype(int))
        want = stats.auroc(s[idx], labels[idx]) - np.mean(
            [stats.auroc(row[idx], labels[idx]) for row in shuffles]
        )
        got = stats.kernel_auroc(
            stats.pair_kernel(s, labels == 0) - stats.pair_kernel(shuffles, labels == 0),
            labels == 0,
            w[b : b + 1],
        )[0]
        assert abs(got - want) < 1e-12
    for card, d in r.sign.per_card.items():
        on = np.asarray(cards) == card
        want = stats.auroc(s[on], labels[on]) - np.mean(
            [stats.auroc(row[on], labels[on]) for row in shuffles]
        )
        assert abs(d - want) < 1e-12
    ci = stats.mean_auroc_ci(shuffles, labels, cards, B=300)
    assert abs(ci.point - np.mean([stats.auroc(row, labels) for row in shuffles])) < 1e-12
    one = stats.rule_r_auroc_mean(s, shuffles[:1], labels, cards, B=300)
    plain = stats.rule_r_auroc(s, shuffles[0], labels, cards, B=300)
    assert abs(one.delta - plain.delta) < 1e-12 and abs(one.lower_bound - plain.lower_bound) < 1e-12
    assert one.reason == plain.reason


def _null_world(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Labels depend on candidate quality only; each context adds label-irrelevant noise to the
    score (a candidate-context interaction). Returns labels, s, cards and the contexts."""
    labels, s, cards, ctx_rows = [], [], [], []
    for c in range(7):
        ctx = rng.normal(size=(8, 16))
        for t in range(8):
            for _ in range(5):
                q = rng.normal()
                labels.append(0 if rng.random() < 1 / (1 + math.exp(-1.5 * q)) else 1)
                w = rng.normal(size=16)
                row = q + 0.8 * (ctx @ w) / 4
                ctx_rows.append(row)
                s.append(row[t])
                cards.append(f"c{c}")
    return np.asarray(labels), np.asarray(s), cards, np.asarray(ctx_rows)


def test_the_shuffle_comparator_is_null_calibrated() -> None:
    """§5.5: without target-specific signal, Δ(real − mean over derangements) is about 0 on
    average; the drafts' comparator (the AUROC of the item's expected score over the other
    contexts) sat about 0.05 below 0, against every arm."""
    rng = np.random.default_rng(123)
    mine, drafts = [], []
    for _ in range(24):
        labels, s, cards, ctx = _null_world(rng)
        slot = np.tile(np.repeat(np.arange(8), 5), 7)
        draws = x1.shuffle_draws(dict.fromkeys(sorted(set(cards)), 8), k=200, seed=0)
        card_arr = np.asarray(cards)
        idx = np.empty((200, len(s)), dtype=int)
        for card, perm in draws.items():
            on = card_arr == card
            idx[:, on] = perm[:, slot[on]]  # type: ignore[index]
        shuffled = ctx[np.arange(len(s))[None, :], idx]
        mine.append(stats.rule_r_auroc_mean(s, shuffled, labels, cards, B=20).delta)
        others = (ctx.sum(axis=1) - s) / 7
        drafts.append(stats.auroc(s, labels) - stats.auroc(others, labels))
    assert abs(float(np.mean(mine))) < 0.015
    assert float(np.mean(drafts)) < -0.03


def _vector_arm(state_dependent: bool, seed: int) -> tuple[list[str], np.ndarray, x1.TaskItems]:
    """Vectors on 7 cards x 6 targets x 6 candidates (2 Yes each). Every context shares a
    direction ``u`` plus its own part ``r``; Yes candidates align with their own target's ``r``
    (state signal) or with ``u`` (a candidate-only signal every context would reward)."""
    rng = np.random.default_rng(seed)
    d = 64
    u = l2(rng.normal(size=(1, d)))[0]
    anchor = l2(rng.normal(size=(1, d)))[0]
    targets, owner, cards, labels, cands, names = [], [], [], [], [], []
    for card in CARDS:
        for t in range(6):
            r = rng.normal(size=d) / np.sqrt(d)
            targets.append(l2((0.7 * u + r)[None, :])[0])
            for j in range(6):
                yes = j < 2
                base = (r if state_dependent else u) if yes else rng.normal(size=d) / np.sqrt(d)
                cands.append(l2((base + 0.6 * rng.normal(size=d) / np.sqrt(d))[None, :])[0])
                owner.append(len(targets) - 1)
                cards.append(card)
                labels.append(0 if yes else 1)
                names.append(f"{card}/t{t}")
    zt, zc = np.asarray(targets), np.asarray(cands)
    za = np.broadcast_to(anchor, zc.shape)
    o = np.asarray(owner)
    s = pairwise_s_c(100.0, zt[o], zc, za)
    ctx = x1.context_scores(zt, o, cards, zc, za, 100.0)
    shas = [hashlib.sha256(n.encode()).hexdigest() for n in names]
    # TaskItems orders a card's targets by sha256; keep the columns in that order
    order = {c: sorted({shas[i] for i in range(len(shas)) if cards[i] == c}) for c in CARDS}
    rows = {c: sorted({int(o[i]) for i in range(len(o)) if cards[i] == c}) for c in CARDS}
    col_of = {c: {rows[c][k]: k for k in range(len(rows[c]))} for c in CARDS}
    by_sha = {shas[i]: int(o[i]) for i in range(len(o))}
    ctx2 = np.stack(
        [[ctx[i, col_of[cards[i]][by_sha[t]]] for t in order[cards[i]]] for i in range(len(o))]
    )
    items = x1.TaskItems(
        task="neon_term_fits",
        task_id="term.fits",
        target=shas,
        option=[f"X:{i}" for i in range(len(shas))],
        card=cards,
        label=np.asarray(labels),
        weight=np.ones(len(shas)),
        scores={"F7@clm-raw": x1.ArmScores(s=s, contexts=ctx2)},
    )
    return cards, np.asarray(labels), items


def test_the_shuffle_kills_a_state_free_signal() -> None:
    for state_dependent in (True, False):
        cards, labels, items = _vector_arm(state_dependent, seed=11)
        s = items.scores["F7@clm-raw"].s
        auroc = stats.auroc_ci(s, labels, cards, B=500)
        rule = stats.rule_r_auroc_mean(s, items.shuffled("F7@clm-raw"), labels, cards, B=500)
        assert auroc.point >= 0.9 and auroc.lower > 0.5  # both arms rank well on their own
        assert rule.passed is state_dependent, (state_dependent, rule.reason, rule.delta)


def test_item_evidence_qualifies_state_signal_and_rejects_a_state_free_one() -> None:
    task = _task()
    items = _items(
        task,
        {
            "F1@clm-latest": _control(task, 1.0, seed=9),
            "F4@clm-latest": _planted(task, 0.0, seed=10),  # no signal
            "F7@clm-latest": _planted(task, 2.5, seed=11),  # state signal, shuffle has none
            "F9@clm-latest": _planted(task, 2.5, shuffle_d=2.5, seed=12),  # state-free
        },
    )
    d = x1.decide(x1.ItemEvidence(items, B=B))
    assert d.arms["F7@clm-latest"].qualified is True
    assert d.arms["F9@clm-latest"].failed == ["shuffle_rule_r"]
    assert (d.arms["F9@clm-latest"].evidence.auroc or 0) > 0.8  # high AUROC, yet not qualified
    assert "auroc_floor" in d.arms["F4@clm-latest"].failed
    assert d.arms["F1@clm-latest"].qualified is None
    assert d.outcome == "choice" and d.choice is not None and d.choice.arm == "F7@clm-latest"
    assert d.step4 is not None and not d.step4.d3_amendment_recommended


# -- the decision rules (§7) ----------------------------------------------------------------------


def test_rule_1_thresholds_are_strict_where_the_text_is() -> None:
    cases = {
        "F4@clm-latest": (0.70, 0.50, True, 0.5),  # LB exactly 0.5: fails
        "F7@clm-latest": (0.60, 0.51, True, 0.5),  # AUROC exactly 0.60: passes
        "F9@clm-latest": (0.5999, 0.51, True, 0.5),  # below the floor
        "F4@clm-raw": (0.9, 0.8, False, 0.5),  # shuffle rule R failed
    }
    d = x1.decide(FakeEvidence(cases))
    verdicts = {a: (v.qualified, v.failed) for a, v in d.arms.items()}
    assert verdicts == {
        "F4@clm-latest": (False, ["auroc_lower_bound"]),
        "F7@clm-latest": (True, []),
        "F9@clm-latest": (False, ["auroc_floor"]),
        "F4@clm-raw": (False, ["shuffle_rule_r"]),
    }


def test_k1_when_nothing_qualifies_and_undecidable_without_nll() -> None:
    k1 = x1.decide(FakeEvidence({"F7@clm-latest": (*BAD, 0.5), "F1@clm-latest": (*GOOD, 0.3)}))
    assert k1.outcome == "K1" and k1.choice is None and k1.comparisons == []
    assert k1.step4 is None and "K1" in k1.notes[0]  # rule (4) needs an eligible arm (§7.9)
    und = x1.decide(FakeEvidence(_arms(F7_latest=None)))
    assert und.outcome == "undecidable" and und.choice is None


@pytest.mark.parametrize(
    ("nll", "wins", "tokens", "expected"),
    [
        # F4 wins on NLL but is not ≻ F7 (it is ≻ F9): take the more cacheable arm it does not beat
        (
            {"F4": 0.30, "F7": 0.35, "F9": 0.36},
            {("F4@clm-latest", "F9@clm-latest")},
            None,
            "F7",
        ),
        # F4 ≻ every more cacheable arm: F4 stands
        (
            {"F4": 0.30, "F7": 0.35, "F9": 0.36},
            {("F4@clm-latest", "F7@clm-latest"), ("F4@clm-latest", "F9@clm-latest")},
            None,
            "F4",
        ),
        # F4 beats neither; F7/F9 tie (no rule R win): fewer tokens per target
        ({"F4": 0.30, "F7": 0.35, "F9": 0.36}, set(), {"F7": 250.0, "F9": 60.0}, "F9"),
        ({"F4": 0.30, "F7": 0.35, "F9": 0.36}, set(), {"F7": 60.0, "F9": 250.0}, "F7"),
        # ... and F7 on equal tokens
        ({"F4": 0.30, "F7": 0.35, "F9": 0.36}, set(), {"F7": 90.0, "F9": 90.0}, "F7"),
        # F4 beats neither, the lower-NLL of F7/F9 is ≻ the other: it is chosen over the tokens
        (
            {"F4": 0.30, "F7": 0.35, "F9": 0.33},
            {("F9@clm-latest", "F7@clm-latest")},
            {"F7": 60.0, "F9": 250.0},
            "F9",
        ),
        # F9 wins on NLL: no arm is more cacheable, F9 stands (literal reading, §7.5)
        ({"F4": 0.40, "F7": 0.35, "F9": 0.30}, set(), {"F7": 60.0, "F9": 250.0}, "F9"),
        # F7 wins on NLL and is not ≻ F9: F7 still stands; tokens never override the winner
        ({"F4": 0.40, "F7": 0.30, "F9": 0.31}, set(), {"F7": 250.0, "F9": 60.0}, "F7"),
    ],
)
def test_rule_2_cacheability_and_the_f7_f9_tie(
    nll: dict[str, float],
    wins: set[tuple[str, str]],
    tokens: dict[str, float] | None,
    expected: str,
) -> None:
    arms = _arms(**{f"{f}_latest": v for f, v in nll.items()})
    ev = FakeEvidence(arms, wins, tokens)
    d = x1.decide(ev)
    assert d.outcome == "choice" and d.step2 is not None and d.choice is not None
    assert d.step2.model == "clm-latest" and d.choice.framing == expected, d.step2.reason
    assert d.step2.ranked[0] == x1.arm_id(min(nll, key=nll.__getitem__), "clm-latest")
    if min(nll, key=nll.__getitem__) in x1.MORE_CACHEABLE:
        # the winner stands without a comparison among the framings (§7.5)
        assert not [p for p in ev.asked if p[0].startswith(("F4", "F7", "F9"))]


def test_rule_2_with_one_qualified_arm_and_its_comparison_order() -> None:
    only = x1.decide(FakeEvidence(_arms(F4_latest=0.3)))
    assert only.choice is not None and only.choice.framing == "F4"
    assert only.step2 is not None and "only qualified arm" in only.step2.reason
    ev = FakeEvidence(_arms(F4_latest=0.3, F7_latest=0.4, F9_latest=0.5))
    x1.decide(ev)
    # F4 against the more cacheable arms in F7, F9 order, then the F7/F9 resolution (lower NLL
    # first), then rule (3) has nothing to compare (no clm-raw arm)
    assert ev.asked == [
        ("F4@clm-latest", "F7@clm-latest"),
        ("F4@clm-latest", "F9@clm-latest"),
        ("F7@clm-latest", "F9@clm-latest"),
    ]


def test_rule_3_model_choice() -> None:
    both = _arms(F7_latest=0.40, F7_raw=0.30)
    raw_wins = x1.decide(FakeEvidence(both, {("F7@clm-raw", "F7@clm-latest")}))
    assert raw_wins.choice is not None and raw_wins.choice.arm == "F7@clm-raw"
    assert raw_wins.step3 is not None and raw_wins.step3.compared
    latest_kept = x1.decide(FakeEvidence(both))
    assert latest_kept.choice is not None and latest_kept.choice.model == "clm-latest"
    # clm-raw "better" but not qualified: never chosen, not even compared
    unqualified = {**_arms(F7_latest=0.40), "F7@clm-raw": (*BAD, 0.10)}
    ev = FakeEvidence(unqualified, {("F7@clm-raw", "F7@clm-latest")})
    d = x1.decide(ev)
    assert d.choice is not None and d.choice.arm == "F7@clm-latest"
    assert ("F7@clm-raw", "F7@clm-latest") not in ev.asked
    # only clm-raw qualifies: rule (2) ranks clm-raw's arms and the model is clm-raw
    only_raw = x1.decide(
        FakeEvidence({"F7@clm-latest": (*BAD, 0.2), **_arms(F9_raw=0.4, F4_raw=0.3)})
    )
    assert only_raw.step2 is not None and only_raw.step2.model == "clm-raw"
    assert only_raw.step3 is not None and not only_raw.step3.compared
    assert only_raw.choice is not None and only_raw.choice.model == "clm-raw"
    # the framing is ranked on clm-latest even when clm-raw has a better other framing (§7.4)
    mixed = x1.decide(FakeEvidence(_arms(F4_latest=0.4, F9_raw=0.1), {}))
    assert mixed.choice is not None and mixed.choice.arm == "F4@clm-latest"


def test_rule_4_recommends_an_amendment_and_never_adopts_f1() -> None:
    arms = {**_arms(F7_latest=0.4, F9_raw=0.45), "F1@clm-latest": (0.9, 0.85, True, 0.2)}
    wins = {("F1@clm-latest", "F7@clm-latest"), ("F1@clm-latest", "F9@clm-raw")}
    d = x1.decide(FakeEvidence(arms, wins))
    assert d.step4 is not None and d.step4.d3_amendment_recommended
    assert d.step4.beats_all == {"F1@clm-latest": True}
    assert d.choice is not None and d.choice.framing == "F7"  # D3 stands; nothing is adopted
    assert d.arms["F1@clm-latest"].qualified is None
    assert any("reopening D3" in n for n in d.notes)
    partial = x1.decide(FakeEvidence(arms, {("F1@clm-latest", "F7@clm-latest")}))
    assert partial.step4 is not None and not partial.step4.d3_amendment_recommended


def test_the_decision_replays_from_its_own_trace_bit_for_bit() -> None:
    arms = {**_arms(F4_latest=0.3, F7_latest=0.35, F9_latest=0.36, F7_raw=0.2)}
    arms["F1@clm-raw"] = (0.7, 0.6, True, 0.25)
    d = x1.decide(FakeEvidence(arms, {("F4@clm-latest", "F9@clm-latest")}))
    assert x1.decide(x1.StoredEvidence(d)) == d
    stored = x1.Decision.model_validate(json.loads(json.dumps(d.model_dump(mode="json"))))
    assert x1.decide(x1.StoredEvidence(stored)) == d
    # a flipped comparison in the trace no longer replays to the same decision
    tampered = stored.model_copy(deep=True)
    tampered.comparisons[0].rule_r.passed = not tampered.comparisons[0].rule_r.passed
    assert x1.decide(x1.StoredEvidence(tampered)) != tampered


def test_a_stored_rule_r_must_follow_from_its_numbers() -> None:
    good = _rule(True)
    x1.check_rule(good, "ok")
    x1.check_rule(_rule(False), "ok")
    # flag and reason changed together, numbers left: caught
    lie = good.model_copy(update={"passed": False, "reason": "sign_test_failed"})
    with pytest.raises(x1.X1ReproError, match="its numbers say"):
        x1.check_rule(lie, "lie")
    with pytest.raises(x1.X1ReproError, match="do not follow"):
        x1.check_rule(good.model_copy(update={"needed": 5}), "needed")
    neg = good.model_copy(update={"lower_bound": -0.01})
    with pytest.raises(x1.X1ReproError):
        x1.check_rule(neg, "lower")
    x1.check_rule(
        neg.model_copy(update={"passed": False, "reason": "bootstrap_lower_bound_not_positive"}),
        "ok",
    )
    few = x1.RuleRSummary.model_validate(
        {**good.model_dump(), "per_card": {"a": 0.1, "b": 0.1, "c": 0.1}, "m_c": 3, "wins": 3,
         "needed": 3, "passed": False, "reason": "insufficient_clusters"}
    )  # fmt: skip
    x1.check_rule(few, "few")
    x1.check_rule(x1.RuleRSummary.unavailable("nll", "no_pooled_items"), "none")


# -- nesting (§8) ----------------------------------------------------------------------------------


def test_nested_selection_records_fold_choices_and_k1_folds() -> None:
    """Signal on 5 cards, a shuffle that beats the real score on 2: the full run fails the card
    sign test (5 of 7 < 6) and is K1; a fold holding out one of the 2 keeps 5 of 6 winning cards
    (needed 5) and chooses F7; a fold holding out a signal card has 4 of 6 and is K1 (§8.5)."""
    task = _task()
    anti = set(CARDS[5:])
    d = {c: (0.0 if c in anti else 3.0) for c in CARDS}
    sd = {c: (1.0 if c in anti else 0.0) for c in CARDS}
    items = _items(task, {"F7@clm-latest": _planted(task, d, shuffle_d=sd, seed=21)})
    full = x1.decide(x1.ItemEvidence(items, B=B))
    shuffle = full.arms["F7@clm-latest"].evidence.shuffle
    assert full.outcome == "K1" and shuffle is not None and shuffle.reason == "sign_test_failed"
    nested = x1.nested_selection(items, B=B)
    outcomes = {c: fc.outcome for c, fc in nested.fold_choices.items()}
    assert outcomes == {c: ("choice" if c in anti else "K1") for c in CARDS}
    assert {c: fc.arm for c, fc in nested.fold_choices.items() if c in anti} == dict.fromkeys(
        anti, "F7@clm-latest"
    )
    assert nested.skipped_folds == {c: "inner_k1" for c in CARDS if c not in anti}
    # the inner decision used only its 6 training cards: m_c counts those
    inner = nested.folds[CARDS[0]].decision.arms["F7@clm-latest"].evidence.shuffle
    assert inner is not None and inner.m_c == 6 and CARDS[0] not in inner.per_card
    # no A1 (full-data K1): x1.json holds the arm cells only; the tier run gets the selection
    run = x1.run_x1_items(
        {task.name: task},
        {task.name: items},
        date="d",
        labels_sha256="a" * 64,
        labels_content_sha256="b" * 64,
        framings=["F7"],
        models=["clm-latest"],
        B=B,
    )
    assert set(run.results.cells) == {"neon_term_fits.calibrated.F7@clm-latest"}
    assert x1.selection(run.results.x1.tasks["neon_term_fits"]) == {
        "a1": None,
        "fold_choices": {
            c: {
                "framing": "F7",
                "model": "clm-latest",
                "outcome": "choice",
                "x1_trace": f"#/x1/tasks/neon_term_fits/nested/folds/{c}/decision",
            }
            for c in sorted(anti)
        },
        "fold_skips": {c: "inner_k1" for c in CARDS if c not in anti},
        "note": f"X1 neon_term_fits: K1 (full run, {x1.PLAN})",
    }


def test_a_folds_inner_decision_never_reads_its_held_out_cards_labels() -> None:
    """D27: the choice of fold h is made on the 6 training cards only, so flipping every label
    of card h (and its weights) leaves fold h's inner decision unchanged, bit for bit."""
    task = _task()
    scores = {
        "F4@clm-latest": _planted(task, 1.5, seed=71),
        "F7@clm-latest": _planted(task, 2.0, seed=72),
        "F1@clm-latest": _control(task, 1.0, seed=73),
    }
    items = _items(task, scores)
    held = CARDS[3]
    flipped = _items(task, scores)
    on = flipped.card_arr == held
    flipped.label = np.where(on, 1 - flipped.label, flipped.label)
    flipped.weight = np.where(on, 0.6, flipped.weight)
    flipped.__post_init__()
    train = np.flatnonzero(~on)
    a = x1.decide(x1.ItemEvidence(items, train, B=B))
    b = x1.decide(x1.ItemEvidence(flipped, train, B=B))
    assert a == b and a.outcome == "choice"
    assert (
        x1.nested_selection(items, B=B).folds[held] == x1.nested_selection(flipped, B=B).folds[held]
    )


def _clean_items_for(task: Task) -> x1.TaskItems:
    """Every arm of the grid on ``task``: F7@clm-latest carries a strong state signal (and a
    weaker one at the mean context), F9@clm-latest a state-free one the shuffle kills."""
    items = _items(
        task,
        {
            "F1@clm-latest": _control(task, 0.5, seed=31),
            "F4@clm-latest": _planted(task, 1.0, seed=32),
            "F7@clm-latest": _planted(task, 3.0, seed=33, mc_d=0.5),
            "F9@clm-latest": _planted(task, 2.0, shuffle_d=2.0, seed=34),
            "F1@clm-raw": _control(task, 0.3, seed=35),
            "F4@clm-raw": _planted(task, 0.5, seed=36),
            "F7@clm-raw": _planted(task, 1.5, seed=37),
            "F9@clm-raw": _planted(task, 0.0, seed=38),
        },
    )
    items.ms_per_item = {a: 0.25 for a in items.scores}
    items.input_tokens = {"F1": 9000, "F4": 5000, "F7": 4000, "F9": 2000}
    return items


def _clean_items() -> tuple[Task, x1.TaskItems]:
    task = _task()
    return task, _clean_items_for(task)


def _clean_run(*, nested: bool = True, full: bool = True, b: int = B) -> x1.X1Run:
    task, items = _clean_items()
    return x1.run_x1_items(
        {task.name: task},
        {task.name: items},
        date="2026-10-01",
        labels_sha256="a" * 64,
        labels_content_sha256="b" * 64,
        fingerprints=FPS,
        latency=(
            {"term.fits": {"F7@clm-latest": [30.0, 10.0, 20.0]}},
            {
                "file": "t.json",
                "first": {"term.fits": {"F7@clm-latest": [True, False, True]}},
                "new_text": {"term.fits": {"F7@clm-latest": [True, False, False]}},
            },
        ),
        nested=nested,
        full=full,
        B=b,
    )


def test_a_clean_signal_is_chosen_in_every_fold_and_the_arm_cells_say_so() -> None:
    run = _clean_run()
    rec = run.results.x1.tasks["neon_term_fits"]
    assert rec.a1 is not None and rec.a1.arm == "F7@clm-latest"
    assert rec.nested is not None
    assert {c.arm for c in rec.nested.fold_choices.values()} == {"F7@clm-latest"}
    cells = run.results.cells
    assert len(cells) == 8 and all("@" in k for k in cells)
    assert all(c.exploratory and c.selection == "full" for c in cells.values())
    assert not cells["neon_term_fits.calibrated.F1@clm-latest"].servable  # D3
    arm_cell = cells["neon_term_fits.calibrated.F7@clm-latest"]
    assert arm_cell.framing == "F7" and arm_cell.variant == arm_cell.model == "clm-latest"
    assert arm_cell.items is not None and len(arm_cell.items) == 280
    assert arm_cell.fingerprint == {"clm_model_fp": "clm-latest00"}
    assert arm_cell.question_key == x1.fr.framing("term.fits", "F7").question_key
    # a synthetic run is not the registered one: every cell says so
    assert not run.results.x1.registered and not arm_cell.pre_registered
    assert any(d.startswith("B 200") for d in run.results.x1.deviations)
    assert run.results.notes[0].startswith("NOT the pre-registered run")
    sel = x1.selection(rec)
    assert sel["a1"] == {"framing": "F7", "model": "clm-latest"} and sel["fold_skips"] == {}
    x1sel = tier_cells.X1Selection.of(**sel)  # the tier cells take X1's outcome as is
    assert x1sel.a1 == tier_cells.Arm("F7", "clm-latest") and len(x1sel.fold_choices) == 7
    arm = rec.arms["F7@clm-latest"]
    assert arm.qualified and arm.mean_context is not None and arm.shuffle is not None
    assert arm.shuffle.k == x1.SHUFFLE_K and arm.shuffle.n_unshuffled == 0
    diag = arm.diagnostics
    assert diag["latency"]["p50"] == 20.0 and diag["calls"] == len(
        set(run.items["neon_term_fits"].target)
    )
    # §11.3: the targets this model was asked first for, and second, and the new-text share
    assert diag["latency"]["first"] == {"n": 2, "p50": 25.0, "p95": 29.5}
    assert diag["latency"]["second"] == {"n": 1, "p50": 10.0, "p95": 10.0}
    assert diag["latency"]["new_text_share"] == pytest.approx(1 / 3)
    assert diag["ms_per_decision"] == 0.25 and diag["input_tokens"] == 4000
    assert diag["timing_source"] == "offline_replay" and diag["calls_per_target"] == 1.0
    assert rec.arms["F1@clm-latest"].diagnostics["calls"] == 280
    assert diag["input_tokens_per_target"] == 250 + 12 + 20 * 5  # context, anchor, 5 candidates
    # §6.6 sensitivity: the weights (Yes 0.7, No 0.5) lean the weighted fit towards Yes
    sens = arm.sensitivity
    assert sens is not None and sens.n == 280
    assert (sens.weighted.mean_p_minus_rate or 0) > (sens.unweighted.mean_p_minus_rate or 0)
    assert abs(sens.unweighted.mean_p_minus_rate or 1) < 0.02
    assert run.results.x1.latency == {"file": "t.json"}


def test_a_registered_configuration_stamps_its_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """§13: with the registered data and configuration the cells are ``pre_registered``; a
    single model, a subset of the framings, another seed or only part of the run is not."""
    monkeypatch.setattr(reg, "REGISTERED", _registration())
    task, items = _clean_items()
    with pytest.raises(x1.X1DataError, match="no X1 task"):
        x1.run_x1_items({}, {}, date="d", labels_sha256="a" * 64, labels_content_sha256="b" * 64)
    # both X1 tasks are part of the registered configuration: one task alone is a deviation
    one = x1.run_x1_items(
        {task.name: task}, {task.name: items}, date="d", labels_sha256="a" * 64,
        labels_content_sha256="b" * 64, fingerprints=FPS, B=B,
    )  # fmt: skip
    assert one.results.x1.deviations == [
        "tasks ['term.fits'] is not ['term.fits', 'column.ontology_fits']"
    ]
    cfg = one.results.x1.config.model_copy(update={"tasks": list(x1.TASK_IDS)})
    assert x1.config_deviations(cfg) == []
    for change, text in (
        ({"models": ["clm-latest"]}, "models"),
        ({"framings": ["F7"]}, "framings"),
        ({"seed": 7}, "seed 7"),
        ({"nested": False}, "nested False"),
        ({"shuffle_k": 20}, "shuffle_k 20"),
    ):
        assert any(d.startswith(text) for d in x1.config_deviations(cfg.model_copy(update=change)))
    assert reg.data_deviations("a" * 64, "c" * 64) == [
        "labels_content_sha256 cccccccccccc is not bbbbbbbbbbbb"
    ]


def test_decide_from_json_requires_the_registered_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _clean_run()
    path = x1.write_x1(run, tmp_path)
    with pytest.raises(x1.X1ReproError, match="not the pre-registered run"):
        x1.decide_from_json(path, recompute=False)
    with pytest.raises(x1.X1ReproError, match="not the pre-registered run"):
        x1.selections_from_json(path)
    assert x1.decide_from_json(path, recompute=False, **LAX)
    # a file claiming registration its configuration does not have is caught
    raw = json.loads(path.read_text())
    raw["x1"]["registered"] = True
    raw["x1"]["deviations"] = []
    path.write_text(json.dumps(raw))
    with pytest.raises(x1.X1ReproError, match="registered status"):
        x1.decide_from_json(path, recompute=False, **LAX)


# -- the files and decide_from_json (§12) ----------------------------------------------------------


def test_items_file_round_trips_float64_exactly(tmp_path: Path) -> None:
    task = _task()
    rng = np.random.default_rng(41)
    s = rng.normal(size=280) * 10.0 ** rng.integers(-300, 300, size=280)
    s[:3] = [0.1 + 0.2, 1e-300, -2.5e299]
    planted = _planted(task, 1.0, seed=42)
    ctx = planted.contexts
    assert ctx is not None
    slot, _, _ = _layout(task)
    ctx = ctx * 10.0 ** rng.integers(-200, 200, size=ctx.shape)
    ctx[np.arange(280), slot] = s
    items = _items(
        task,
        {
            "F7@clm-latest": x1.ArmScores(s=s, contexts=ctx, mean_context=-s),
            "F1@clm-raw": x1.ArmScores(s=s * 7.0),
        },
    )
    items.p_loco["F7@clm-latest"] = np.where(np.arange(280) % 9 == 0, np.nan, 1 / 3.0)
    items.p_probe = np.where(np.arange(280) % 4 == 0, np.nan, 2 / 3.0)
    path = tmp_path / "x1_items.parquet"
    assert x1.write_items({task.name: items}, path) == 560
    back = x1.read_items(path)[task.name]
    for arm in ("F7@clm-latest", "F1@clm-raw"):
        a, b = items.scores[arm], back.scores[arm]
        assert np.array_equal(a.s, b.s)
        for what in ("contexts", "mean_context"):
            x, y = getattr(a, what), getattr(b, what)
            assert (x is None and y is None) or np.array_equal(x, y, equal_nan=True)
    assert np.array_equal(
        back.p_loco["F7@clm-latest"], items.p_loco["F7@clm-latest"], equal_nan=True
    )
    assert back.p_probe is not None and np.array_equal(back.p_probe, items.p_probe, equal_nan=True)
    assert np.array_equal(back.weight, items.weight) and np.array_equal(back.label, items.label)
    assert back.target == items.target and back.option == items.option and back.card == items.card
    assert np.array_equal(back.tokens["F7"].context, items.tokens["F7"].context)
    # the same tables give the same bytes (the sha256 x1.json records is stable)
    again = tmp_path / "again.parquet"
    x1.write_items({task.name: items}, again)
    assert again.read_bytes() == path.read_bytes()
    # a context table whose own column is not s is refused
    bad = ctx.copy()
    bad[0, slot[0]] += 1.0
    with pytest.raises(x1.X1DataError, match="own context column"):
        _items(task, {"F7@clm-latest": x1.ArmScores(s=s, contexts=bad)})


def test_decide_from_json_reproduces_the_decision_and_detects_tampering(tmp_path: Path) -> None:
    run = _clean_run()
    path = x1.write_x1(run, tmp_path)
    assert path == tmp_path / "2026-10-01" / "x1.json"
    md = (tmp_path / "2026-10-01" / "x1.md").read_text()
    assert md.startswith("# X1 framing A/B")
    # §4.2: the intervals are one-sided 95% bounds, a 90% interval, and say so
    assert md.count("[one-sided 95% bounds; 90% interval]") == 2 and "95% cluster" not in md
    assert "beats_lookup_novel (report-only)" in md
    decisions = x1.decide_from_json(path, **LAX)
    rec = x1.load_x1(path).x1.tasks["neon_term_fits"]
    assert decisions == {"neon_term_fits": rec.decision}
    assert rec.decision is not None and rec.decision.choice is not None
    assert rec.decision.choice.arm == "F7@clm-latest"
    cells = x1.x1_cells(path)
    assert isinstance(cells, BenchResults)
    assert "neon_term_fits.calibrated.F7@clm-latest" in cells.cells
    raw = json.loads(path.read_text())

    def rewrite(edit: Any) -> None:
        data = json.loads(json.dumps(raw))
        edit(data)
        path.write_text(json.dumps(data))

    # (a) replay: a stored comparison flipped no longer follows from its numbers
    def flip(data: dict[str, Any]) -> None:
        comp = data["x1"]["tasks"]["neon_term_fits"]["decision"]["comparisons"]
        assert comp, "the clean run consults rule R comparisons"
        r = comp[0]["rule_r"]
        r["passed"] = not r["passed"]
        r["reason"] = "passed" if r["passed"] else "sign_test_failed"

    rewrite(flip)
    with pytest.raises(x1.X1ReproError, match="its numbers say"):
        x1.decide_from_json(path, recompute=False, **LAX)

    # a cell that disagrees with the trace is caught by the replay alone
    def requalify(data: dict[str, Any]) -> None:
        data["cells"]["neon_term_fits.calibrated.F9@clm-latest"]["diagnostics"]["qualified"] = True

    rewrite(requalify)
    with pytest.raises(x1.X1ReproError, match="rule \\(1\\) verdict"):
        x1.decide_from_json(path, recompute=False, **LAX)

    def refold(data: dict[str, Any]) -> None:
        folds = data["x1"]["tasks"]["neon_term_fits"]["nested"]["fold_choices"]
        folds[CARDS[0]]["model"] = "clm-raw"

    rewrite(refold)
    with pytest.raises(x1.X1ReproError, match="fold_choices"):
        x1.decide_from_json(path, recompute=False, **LAX)

    # the arm record's NLL (x1.md's column) must be the trace's: the replay alone sees it
    def record_only(data: dict[str, Any]) -> None:
        data["x1"]["tasks"]["neon_term_fits"]["arms"]["F7@clm-latest"]["loco"]["nll"] += 1e-6

    rewrite(record_only)
    with pytest.raises(x1.X1ReproError, match="LOCO-Platt NLL is not its trace's"):
        x1.decide_from_json(path, recompute=False, **LAX)

    # (b) recompute: a number changed consistently in the trace and the record (the replay
    # finds them in agreement) is caught against the items file
    def nudge(data: dict[str, Any]) -> None:
        task = data["x1"]["tasks"]["neon_term_fits"]
        task["decision"]["arms"]["F7@clm-latest"]["evidence"]["nll"] += 1e-6
        task["arms"]["F7@clm-latest"]["loco"]["nll"] += 1e-6
        task["arms"]["F7@clm-latest"]["sensitivity"]["weighted"]["nll"] += 1e-6

    rewrite(nudge)
    x1.decide_from_json(path, recompute=False, **LAX)  # the replay alone cannot see it
    with pytest.raises(x1.X1ReproError, match="recomputed from the items"):
        x1.decide_from_json(path, **LAX)
    # the items file must be the recorded one
    path.write_text(json.dumps(raw))
    items_path = path.with_name("x1_items.parquet")
    items_path.write_bytes(items_path.read_bytes() + b"\0")
    with pytest.raises(x1.X1ReproError, match="sha256"):
        x1.decide_from_json(path, **LAX)
    items_path.unlink()
    with pytest.raises(x1.X1ReproError, match="missing"):
        x1.decide_from_json(path, **LAX)


def test_decide_from_json_refuses_other_constants(tmp_path: Path) -> None:
    run = _clean_run(nested=False)
    path = x1.write_x1(run, tmp_path)
    raw = json.loads(path.read_text())
    raw["x1"]["config"]["auroc_floor"] = 0.55
    path.write_text(json.dumps(raw))
    with pytest.raises(x1.X1ReproError, match="other constants"):
        x1.decide_from_json(path, recompute=False, **LAX)
    raw["x1"]["config"]["auroc_floor"] = x1.AUROC_FLOOR
    raw["x1"]["config"]["fitters"]["platt"]["armijo"] = 0.5
    path.write_text(json.dumps(raw))
    with pytest.raises(x1.X1ReproError, match="other constants"):
        x1.decide_from_json(path, recompute=False, **LAX)


def test_partial_runs_replay_what_they_ran(tmp_path: Path) -> None:
    full = x1.write_x1(_clean_run(nested=False), tmp_path / "full")
    nested = x1.write_x1(_clean_run(full=False), tmp_path / "nested")
    assert x1.decide_from_json(full, **LAX)["neon_term_fits"] is not None
    assert x1.decide_from_json(nested, **LAX) == {"neon_term_fits": None}
    assert x1.load_x1(nested).cells == {}
    with pytest.raises(x1.X1ReproError, match="no full run or no nesting"):
        x1.selection(x1.load_x1(nested).x1.tasks["neon_term_fits"])
    with pytest.raises(x1.X1DataError, match="nothing to run"):
        _clean_run(nested=False, full=False)


def test_write_x1_refuses_to_overwrite(tmp_path: Path) -> None:
    run = _clean_run(nested=False)
    path = x1.write_x1(run, tmp_path)
    with pytest.raises(x1.X1Exists, match="one run per results file"):
        x1.write_x1(run, tmp_path)
    assert x1.write_x1(run, tmp_path, force=True) == path


def test_the_run_is_deterministic() -> None:
    one = _clean_run().results.model_dump(mode="json")
    two = _clean_run().results.model_dump(mode="json")
    one.pop("environment")
    two.pop("environment")
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


# -- scores from a feature store (§3) --------------------------------------------------------------


def _vec(text: str, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
    return rng.normal(size=dim)


def _texts(task: Task) -> tuple[list[ManifestRow], dict[str, str]]:
    """A synthetic manifest for ``task``: per framing a context per target (F1: per pair), the
    shared candidate and anchor texts (F1: two noul texts)."""
    rows: list[ManifestRow] = []
    ctx_of: dict[str, str] = {}

    def row(f: str, t: str, o: str, role: str, text: str) -> ManifestRow:
        side = "state" if role == "context" else "action"
        return ManifestRow(
            task_id=task.task_id,
            framing_id=f,
            target_sha256=t,
            option_key=o,
            role=role,  # type: ignore[arg-type]
            side=side,  # type: ignore[arg-type]
            text_sha256=text_sha256(text),
            text=text,
        )

    targets = [target_sha256(task.task_id, s) for s in task.states]
    for f in ("F4", "F7", "F9"):
        for t in sorted(set(targets)):
            ctx_of[f"{f}|{t}"] = f"{f} context of {t[:16]}"
            rows.append(row(f, t, "", "context", ctx_of[f"{f}|{t}"]))
            rows.append(row(f, t, ANCHOR_KEY, "anchor", "None of these terms fits."))
        for t, o in zip(targets, task.option_keys, strict=True):
            rows.append(row(f, t, o, "candidate", f"candidate {o}: a term"))
    for t, o in zip(targets, task.option_keys, strict=True):
        rows.append(row("F1", t, o, "context", f"F1 state {t[:16]} {o}"))
        rows.append(row("F1", t, o, "noul_true", "Yes. This is true: it fits."))
        rows.append(row("F1", t, o, "noul_false", "No. This is false: it fits."))
    return rows, ctx_of


def _store(tmp_path: Path, rows: list[ManifestRow], *, drop: str | None = None) -> FeatureStore:
    texts = sorted({r.text for r in rows} - ({drop} if drop else set()))
    store = FeatureStore(tmp_path / "features", FP, dim=16)
    store.add(texts, np.stack([_vec(t) for t in texts]), [len(t.split()) for t in texts])
    return store


def _scorers() -> dict[str, OfflineScorer]:
    h = random_head(3, width=8, depth=3, projection_dim=8, hidden_size=16)
    head = HeadProjector(h.cfg, h.state, h.action, h.logit_scale, "")
    return {
        LATEST: OfflineScorer(LATEST, head, clm_model_fp=MODEL_FP),
        RAW: OfflineScorer(RAW),
    }


def _snapshot_of(task: Task, tmp_path: Path) -> tuple[Path, str, str]:
    """A labels snapshot of the synthetic task's own rows (its generated labels): the file,
    its labels_sha256 and its label content hash, as ``labels snapshot`` writes them."""
    store = LabelStore(tmp_path / "labels.duckdb")
    store.insert_labels(
        [
            LabelRow(
                task_id=task.task_id,
                task_key=task.spec.key,
                target_sha256=target_sha256(task.task_id, state),
                option_key=opt,
                label_source="consensus_majority" if label == 0 else "consensus_negative",
                label="Yes" if label == 0 else "No",
                label_index=label,
                weight=w,
                state_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                state_json=state,
                card=card,
            )
            for (state, label), opt, card, w in zip(
                task.items, task.option_keys, task.cards, task.weights, strict=True
            )
        ]
    )
    path = tmp_path / "snapshot.parquet"
    sha = snapshot(store, path)
    return path, sha, snapshot_content_sha256(path)


def test_scores_from_the_store_equal_the_offline_scorer(tmp_path: Path) -> None:
    task = _task(per_card=18, targets_per_card=6)
    rows, ctx_of = _texts(task)
    snap, sha, content = _snapshot_of(task, tmp_path)
    manifest = Manifest(
        snapshot=str(snap),
        labels_sha256=sha,
        tasks=("term.fits",),
        framings=("F1", "F4", "F7", "F9"),
        rows=tuple(rows),
    )
    store = _store(tmp_path, rows)
    scorers = _scorers()
    ticks = iter(float(i) for i in range(1000))
    items = x1.score_items(task, manifest, store, scorers, clock=lambda: next(ticks))
    assert items.arms() == list(x1.ARM_ORDER)
    targets = items.target
    for m, sc in scorers.items():
        for f in ("F4", "F7", "F9"):
            got = items.scores[x1.arm_id(f, m)]
            for i in (0, 7, 50, 125):
                ctx = ctx_of[f"{f}|{targets[i]}"]
                cand = {items.option[i]: f"candidate {items.option[i]}: a term"}
                r = sc.rank_fit_texts(store, ctx, cand, "None of these terms fits.")
                assert abs(got.s[i] - r.s_c[items.option[i]]) < 1e-9
            # every column of item 0's context row: its candidate under each target of its card
            card = items.card[0]
            for j, t in enumerate(items.card_targets[card]):
                cand0 = {items.option[0]: f"candidate {items.option[0]}: a term"}
                want = sc.rank_fit_texts(
                    store, ctx_of[f"{f}|{t}"], cand0, "None of these terms fits."
                ).s_c[items.option[0]]
                assert got.contexts is not None and abs(got.contexts[0, j] - want) < 1e-9
        f1 = items.scores[x1.arm_id("F1", m)]
        for i in (3, 99):
            r1 = sc.noul_texts(
                store,
                f"F1 state {targets[i][:16]} {items.option[i]}",
                "No. This is false: it fits.",
                "Yes. This is true: it fits.",
            )
            assert abs(f1.s[i] - r1.s) < 1e-9 and f1.contexts is None
    assert items.scale == {LATEST: scorers[LATEST].scale, RAW: 100.0}
    ctx0 = ctx_of[f"F7|{targets[0]}"]
    assert items.tokens["F7"].context[0] == len(ctx0.split())
    assert set(items.collapse) == {"F1", "F4", "F7", "F9"}
    assert items.collapse["F7"]["n"] == len(set(targets))
    contexts = sorted({ctx_of[f"F7|{t}"] for t in targets})
    u = l2(store.get(contexts)).astype(np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    n = len(u)
    brute = np.mean([u[i] @ u[j] for i in range(n) for j in range(n) if i < j])
    assert abs(float(items.collapse["F7"]["raw"] or 0.0) - brute) < 1e-9
    # the projected side is the head's state projections of the same contexts (§11.1)
    proj = store_vectors(scorers[LATEST], store, "state", contexts).astype(np.float64)
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    brute_proj = np.mean([proj[i] @ proj[j] for i in range(n) for j in range(n) if i < j])
    assert abs(float(items.collapse["F7"]["projected"] or 0.0) - brute_proj) < 1e-9
    assert abs(brute_proj - brute) > 1e-3  # the two sides differ here
    texts_f7 = {r.text for r in rows if r.framing_id == "F7"}
    assert items.input_tokens["F7"] == sum(len(t.split()) for t in texts_f7)
    assert all(v == 1000.0 / items.n for v in items.ms_per_item.values())  # one tick per arm
    assert items.candidate_vectors is not None and items.candidate_vectors.shape == (items.n, 16)
    run = x1.run_x1(
        {task.name: task},
        manifest,
        store,
        scorers,
        date="2026-10-01",
        labels_sha256=sha,
        labels_content_sha256=content,
        nested=False,
        B=B,
    )
    rec = run.results.x1.tasks[task.name]
    assert set(rec.arms) == set(x1.ARM_ORDER) and run.results.x1.store["encoder_fp"] == FP
    assert run.results.x1.manifest["rows"] == len(rows)
    assert run.results.x1.snapshot.labels_content_sha256 == content
    probe = rec.candidate_probe  # the learned candidate-only control (§7.11), report-only
    assert probe is not None and probe.n_folds == 7 and probe.n == items.n
    assert set(probe.folds) == set(CARDS) and probe.auroc is not None
    p = run.items[task.name].p_probe
    assert p is not None and np.all((p > 0) & (p < 1))
    # its record scores p_probe for Yes (index 0) with the run's bootstrap settings
    assert probe.auroc.point == stats.auroc(p, items.label) and probe.auroc.B == B
    # the recompute rebuilds the probe's folds and statistics from the items file
    path = x1.write_x1(run, tmp_path / "out")
    x1.decide_from_json(path, **LAX)
    pristine = path.read_text()
    for edit, message in (
        (lambda r: r["x1"]["tasks"][task.name]["candidate_probe"].__setitem__("nll", 0.1),
         "probe's statistics"),
        (lambda r: r["x1"]["tasks"][task.name]["candidate_probe"]["auroc"].__setitem__(
            "point", 0.5), "probe's statistics"),
        (lambda r: r["x1"]["tasks"][task.name]["candidate_probe"]["skipped"].__setitem__(
            CARDS[0], "x"), "probe's folds"),
    ):  # fmt: skip
        path.write_text(pristine)
        _tamper(path, edit)
        x1.decide_from_json(path, recompute=False, **LAX)
        with pytest.raises(x1.X1ReproError, match=message):
            x1.decide_from_json(path, **LAX)


def test_run_x1_checks_the_labels_it_is_given(tmp_path: Path) -> None:
    task = _task(per_card=18, targets_per_card=6)
    rows, _ = _texts(task)
    snap, sha, content = _snapshot_of(task, tmp_path)
    full = Manifest(
        snapshot=str(snap),
        labels_sha256=sha,
        tasks=("term.fits",),
        framings=x1.FRAMING_IDS,
        rows=tuple(rows),
    )
    store = _store(tmp_path / "a", rows)
    kw: dict[str, Any] = {"date": "d", "B": B, "nested": False}
    with pytest.raises(x1.X1DataError, match="snapshot"):
        x1.run_x1({task.name: task}, full, store, _scorers(), labels_sha256="d" * 64,
                  labels_content_sha256=content, **kw)  # fmt: skip
    with pytest.raises(x1.X1DataError, match="labels_content_sha256"):
        x1.run_x1({task.name: task}, full, store, _scorers(), labels_sha256=sha,
                  labels_content_sha256="e" * 64, **kw)  # fmt: skip
    gone = full.model_copy(update={"snapshot": str(tmp_path / "gone.parquet")})
    with pytest.raises(x1.X1DataError, match="not readable"):
        x1.run_x1({task.name: task}, gone, store, _scorers(), labels_sha256=sha,
                  labels_content_sha256=content, **kw)  # fmt: skip
    heavy = Task(task.name, task.spec, task.items, "s", "s", cards=task.cards,
                 weights=task.weights, products=task.products, option_keys=task.option_keys,
                 meta={**task.meta, "min_weight": 0.6})  # fmt: skip
    with pytest.raises(x1.X1DataError, match="shipped policy"):
        x1.run_x1({task.name: heavy}, full, store, _scorers(), labels_sha256=sha,
                  labels_content_sha256=content, **kw)  # fmt: skip


def test_score_items_fails_loudly_on_a_missing_text_or_vector(tmp_path: Path) -> None:
    task = _task(per_card=18, targets_per_card=6)
    rows, _ = _texts(task)
    full = Manifest(
        snapshot="s",
        labels_sha256="c" * 64,
        tasks=("term.fits",),
        framings=x1.FRAMING_IDS,
        rows=tuple(rows),
    )
    short = full.model_copy(update={"rows": tuple(r for r in rows if r.role != "anchor")})
    store = _store(tmp_path / "a", rows)
    with pytest.raises(x1.X1DataError, match="manifest text"):
        x1.score_items(task, short, store, _scorers())
    gappy = _store(tmp_path / "b", rows, drop="candidate X:0000: a term")
    with pytest.raises(x1.X1DataError, match=r"token count|no vector"):
        x1.score_items(task, full, gappy, _scorers())


# -- diagnostics, inputs, stats additions ----------------------------------------------------------


def test_latency_and_lock_helpers(tmp_path: Path) -> None:
    assert x1.latency_summary([30.0, 10.0, 20.0]) == {"n": 3, "p50": 20.0, "p95": 29.0}
    assert x1.latency_summary([])["p50"] is None
    task = _task(per_card=18, targets_per_card=6)
    rows, _ = _texts(task)
    man = Manifest(
        snapshot="s",
        labels_sha256="c" * 64,
        tasks=("term.fits",),
        framings=x1.FRAMING_IDS,
        rows=tuple(rows),
    )
    drawn = x1.latency_targets(man, "term.fits", "F7", n=5)
    assert drawn == x1.latency_targets(man, "term.fits", "F7", n=5) and len(drawn) == 5
    assert drawn == x1.latency_targets(man, "term.fits", "F7", n=50)[:5]
    lock = load_serving_lock(LOCK)
    fps = x1.fingerprints_for_lock(lock)
    assert set(fps) == {LATEST, RAW} and fps[LATEST]["encoder_fp"] == lock.encoder_fp
    assert fps[LATEST]["clm_model_fp"] != fps[RAW]["clm_model_fp"]
    with pytest.raises(x1.X1DataError, match="head"):
        x1.scorers_for_lock(lock, random_head(0))
    h = random_head(0)
    pinned = HeadProjector(h.cfg, h.state, h.action, h.logit_scale, lock.head.sha256)
    scorers = x1.scorers_for_lock(lock, pinned)
    assert scorers[LATEST].clm_model_fp == fps[LATEST]["clm_model_fp"] and scorers[RAW].raw
    good = tmp_path / "lat.json"
    good.write_text(
        json.dumps({"format": x1.LATENCY_FORMAT, "ms": {"term.fits": {"F7@clm-raw": [1, 2.5]}}})
    )
    ms, info = x1.load_latency(good)
    assert (
        ms == {"term.fits": {"F7@clm-raw": [1.0, 2.5]}}
        and info["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest()
    )
    for bad in (
        {"format": "other"},
        {"format": x1.LATENCY_FORMAT, "ms": {"avu.keep": {}}},
        {"format": x1.LATENCY_FORMAT, "ms": {"term.fits": {"F2@clm-raw": [1]}}},
        {"format": x1.LATENCY_FORMAT, "ms": {"term.fits": {"F7@clm-raw": [-1]}}},
    ):
        good.write_text(json.dumps(bad))
        with pytest.raises(x1.X1DataError):
            x1.load_latency(good)


def test_x1_tasks_are_the_two_rank_fit_tasks_with_the_fold_filter(tmp_path: Path) -> None:
    st = LabelStore(tmp_path / "labels.duckdb")
    rows = []
    for j, (label, source, weight) in enumerate(
        [("Yes", "consensus_all", 0.8), ("No", "consensus_negative", 0.5)]
    ):
        state = _state("DP0.00009.001.offbench", "c", f"X:{j}")
        rows.append(
            LabelRow(
                task_id="term.fits",
                task_key=TASKS["term.fits"].key,
                target_sha256=target_sha256("term.fits", state),
                option_key=f"X:{j}",
                label_source=source,  # type: ignore[arg-type]
                label=label,
                label_index=0 if label == "Yes" else 1,
                weight=weight,
                state_sha256=str(j) * 64,
                state_json=state,
                card="DP0.00009.001.offbench",
                product_code="DP0.00009.001",
                leak_group="DP0.00009.001",
            )
        )
    st.insert_labels(rows)
    tasks = x1.x1_tasks(st)
    assert list(tasks) == ["neon_term_fits", "neon_ontology_fits"]
    assert tasks["neon_term_fits"].option_keys == ["X:0", "X:1"]
    assert tasks["neon_ontology_fits"].items == []


def test_auroc_on_scores_matches_probabilities_without_saturation_ties() -> None:
    rng = np.random.default_rng(61)
    labels = (rng.random(140) < 0.4).astype(int)
    cards = [CARDS[i % 7] for i in range(140)]
    s = rng.normal(size=140) + (labels == 0) * 1.5
    p = np.column_stack(
        [calibrate.sigmoid(s / 10), 1 - calibrate.sigmoid(s / 10)]
    )  # monotone, no saturation
    a = stats.auroc_ci(s, labels, cards, B=300)
    b = stats.metric_ci("auroc", p, labels, cards, B=300)
    assert (a.point, a.lower, a.upper, a.n_valid) == (b.point, b.lower, b.upper, b.n_valid)
    s2 = s + rng.normal(size=140)
    p2 = np.column_stack([calibrate.sigmoid(s2 / 10), 1 - calibrate.sigmoid(s2 / 10)])
    r1 = stats.rule_r_auroc(s, s2, labels, cards, B=300)
    r2 = stats.rule_r("auroc", p, p2, labels, cards, B=300)
    assert (r1.delta, r1.lower_bound, r1.reason) == (r2.delta, r2.lower_bound, r2.reason)
    # at scale 100 the sigmoid saturates into ties the score does not have (§3.5)
    big = 100.0 * s
    tied = np.column_stack([calibrate.sigmoid(big), 1 - calibrate.sigmoid(big)])
    assert stats.auroc(big, labels) == a.point
    assert stats.metric_value("auroc", tied, labels) != a.point


def test_derangement_brute_force_mean_converges_to_the_old_expectation() -> None:
    """The draws sample derangements uniformly: with many draws an item's mean shuffled score
    approaches its mean over every other context of its card (the drafts' expectation)."""
    rng = np.random.default_rng(5)
    zt = l2(rng.normal(size=(4, 16)))
    owner = np.asarray([0, 0, 1, 2, 3, 3, 1])
    zc = l2(rng.normal(size=(7, 16)))
    za = np.broadcast_to(l2(rng.normal(size=(1, 16)))[0], (7, 16))
    ctx = x1.context_scores(zt, owner, ["c"] * 7, zc, za, 100.0)
    perms = [p for p in itertools.permutations(range(4)) if all(p[i] != i for i in range(4))]
    exact = np.mean([[ctx[i, p[owner[i]]] for i in range(7)] for p in perms], axis=0)
    draws = x1.shuffle_draws({"c": 4}, k=20000, seed=0)["c"]
    assert draws is not None
    sampled = np.mean(ctx[np.arange(7)[None, :], draws[:, owner]], axis=0)
    assert np.allclose(sampled, exact, atol=0.15 * np.abs(ctx).max())


# -- review findings: the registration, the replay's checks, the K draws, the learned probe ------


RUN_PARAMETER_CHANGES: dict[str, Any] = {
    "B": 50,
    "seed": 7,
    "alpha": 0.1,
    "tasks": ["term.fits"],
    "framings": ["F1", "F4", "F7"],
    "models": ["clm-latest"],
    "full": False,
    "nested": False,
    "shuffle_k": 20,
    "shuffle_seed": 3,
}


def test_the_registered_configuration_is_the_pre_registered_one() -> None:
    """§13.1 against the real registration (no stand-in): both tasks, F1/F4/F7/F9, both models,
    the full run and the nesting, B = 2000, seed 0, alpha 0.05, 200 shuffles with seed 0."""
    cfg = x1.registered_config()
    assert (cfg.B, cfg.seed, cfg.alpha) == (2000, 0, 0.05)
    assert cfg.tasks == ["term.fits", "column.ontology_fits"]
    assert cfg.framings == ["F1", "F4", "F7", "F9"] and cfg.models == ["clm-latest", "clm-raw"]
    assert cfg.full and cfg.nested and (cfg.shuffle_k, cfg.shuffle_seed) == (200, 0)
    assert x1.config_deviations(cfg) == []


def test_every_run_parameter_departing_from_the_registration_is_a_deviation() -> None:
    """§13.2: each run parameter, changed alone, is one deviation (so the run is written
    unregistered); a new run parameter must be added here."""
    assert set(x1._RUN_PARAMETERS) == set(RUN_PARAMETER_CHANGES)
    want = x1.registered_config()
    for name, value in RUN_PARAMETER_CHANGES.items():
        got = x1.config_deviations(want.model_copy(update={name: value}))
        assert got == [f"{name} {value!r} is not {getattr(want, name)!r}"], name


def test_any_deviation_writes_every_cell_unregistered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registered labels, one other setting (alpha): every arm cell ``pre_registered: false``,
    ``exploratory: true``, the deviation recorded, and ``decide_from_json`` refuses the file."""
    monkeypatch.setattr(reg, "REGISTERED", _registration())
    task, items = _clean_items()
    run = x1.run_x1_items(
        {task.name: task}, {task.name: items}, date="d", labels_sha256="a" * 64,
        labels_content_sha256="b" * 64, alpha=0.1, B=B, nested=False,
    )  # fmt: skip
    block = run.results.x1
    assert "alpha 0.1 is not 0.05" in block.deviations and not block.registered
    assert run.results.cells and all(
        c.exploratory and not c.pre_registered for c in run.results.cells.values()
    )


def test_registered_labels_must_give_the_published_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§1.3 in the producer: labels that claim to be the registered snapshot but give other
    items are refused before any statistic; other labels are simply not registered."""
    task, items = _clean_items()
    per_card = {c: 40 for c in CARDS}
    published = reg.TaskCounts(280, task.class_counts(), per_card, 0.5)
    monkeypatch.setattr(
        reg,
        "REGISTERED",
        reg.Registration(snapshot="s", labels_sha256="a" * 64, labels_content_sha256="b" * 64,
                         published="synthetic", counts={"neon_term_fits": published}, B=B),
    )  # fmt: skip
    kw: dict[str, Any] = {"date": "d", "B": B, "nested": False, "labels_content_sha256": "b" * 64}
    x1.run_x1_items({task.name: task}, {task.name: items}, labels_sha256="a" * 64, **kw)
    other = reg.TaskCounts(281, task.class_counts(), per_card, 0.5)
    monkeypatch.setattr(
        reg, "REGISTERED", replace(reg.REGISTERED, counts={"neon_term_fits": other})
    )
    with pytest.raises(reg.RegistrationError, match="not the published"):
        x1.run_x1_items({task.name: task}, {task.name: items}, labels_sha256="a" * 64, **kw)
    run = x1.run_x1_items({task.name: task}, {task.name: items}, labels_sha256="c" * 64, **kw)
    assert not run.results.x1.registered  # not the registered labels: nothing to compare


def _tamper(path: Path, edit: Any) -> None:
    raw = json.loads(path.read_text())
    edit(raw)
    path.write_text(json.dumps(raw))


def test_the_replay_checks_every_cell_record_and_fold_against_the_traces(tmp_path: Path) -> None:
    """§12.6 (a), from the JSON alone: ``a1``, every arm cell's identity fields, the shape of
    each task record, ``fold_choices`` and ``skipped_folds`` against the nested traces, every
    nested trace replayed, and no cell without an arm record. Each edit alone is refused."""
    run = _clean_run()
    path = x1.write_x1(run, tmp_path)
    pristine = path.read_text()
    x1.decide_from_json(path, recompute=False, **LAX)
    task = "neon_term_fits"
    cell = f"{task}.calibrated.F7@clm-latest"

    def at(*keys: str) -> Any:
        def edit(raw: dict[str, Any]) -> dict[str, Any]:
            out = raw
            for k in keys:
                out = out[k]
            return out

        return edit

    def set_field(where: Any, key: str, value: Any) -> Any:
        return lambda raw: where(raw).__setitem__(key, value)

    def nested_trace(raw: dict[str, Any]) -> None:
        folds = raw["x1"]["tasks"][task]["nested"]["folds"]
        evidence = folds[CARDS[0]]["decision"]["arms"]["F9@clm-latest"]["evidence"]
        evidence["auroc_lower"] = 0.0  # its stored verdict no longer follows

    def extra_fold_choice(raw: dict[str, Any]) -> None:
        choices = raw["x1"]["tasks"][task]["nested"]["fold_choices"]
        choices["DP0.99999.001.extra"] = dict(choices[CARDS[0]])

    def extra_cell(raw: dict[str, Any]) -> None:
        raw["cells"][f"{task}.calibrated.F7@clm-extra"] = raw["cells"][cell]

    def no_decision(raw: dict[str, Any]) -> None:
        raw["x1"]["tasks"][task]["decision"] = None
        raw["x1"]["tasks"][task]["a1"] = None

    cells = at("cells", cell)
    rec = at("x1", "tasks", task)
    nested = at("x1", "tasks", task, "nested")
    edits: dict[str, tuple[Any, str]] = {
        "a1": (
            set_field(rec, "a1", {"framing": "F9", "model": "clm-latest", "arm": "F9@clm-latest"}),
            "a1 is not the full-run choice",
        ),
        "question_key": (set_field(cells, "question_key", "0" * 16), "identity fields"),
        "labels_sha256": (set_field(cells, "labels_sha256", "c" * 64), "identity fields"),
        "labels_content_sha256": (
            set_field(cells, "labels_content_sha256", "c" * 64),
            "identity fields",
        ),
        "fingerprint": (set_field(cells, "fingerprint", {"clm_model_fp": "x"}), "identity fields"),
        "pre_registered": (set_field(cells, "pre_registered", True), "identity fields"),
        "exploratory": (set_field(cells, "exploratory", False), "identity fields"),
        "selection": (set_field(cells, "selection", "nested"), "identity fields"),
        "skipped_folds": (
            set_field(nested, "skipped_folds", {CARDS[0]: "inner_k1"}),
            "skipped_folds",
        ),
        "fold_choices": (extra_fold_choice, "one per outer fold"),
        "nested_trace": (nested_trace, "does not follow from its statistics"),
        "extra_cell": (extra_cell, "belongs to no arm record"),
        "record_shape": (no_decision, "not what the configuration ran"),
    }
    for edit, message in edits.values():
        path.write_text(pristine)
        _tamper(path, edit)
        with pytest.raises(x1.X1ReproError, match=message):
            x1.decide_from_json(path, recompute=False, **LAX)
    path.write_text(pristine)
    assert x1.decide_from_json(path, **LAX)


def test_the_arm_shuffle_is_the_mean_over_all_registered_draws() -> None:
    """§5.3-§5.4: an arm's shuffled AUROC and its Δ use all K = 200 seeded draws, exactly (to
    rounding) the brute-force mean of the K AUROCs; a nested fold uses the same draws restricted
    to its training cards."""
    _, items = _clean_items()
    arm = "F7@clm-latest"
    shuffled = items.shuffled(arm, x1.SHUFFLE_K, x1.SHUFFLE_SEED)
    assert shuffled.shape == (x1.SHUFFLE_K, items.n) and x1.SHUFFLE_K == 200
    s, labels = items.scores[arm].s, items.label
    ev = x1.ItemEvidence(items, B=50)
    sh = ev.shuffle(arm)
    assert sh is not None
    mean = float(np.mean([stats.auroc(row, labels) for row in shuffled]))
    assert abs(sh[0].point - mean) < 1e-12
    assert abs(sh[1].delta - (stats.auroc(s, labels) - mean)) < 1e-12
    train = np.flatnonzero(items.card_arr != CARDS[0])
    inner = x1.ItemEvidence(items, train, B=50).shuffle(arm)
    assert inner is not None
    sub = float(np.mean([stats.auroc(row[train], labels[train]) for row in shuffled]))
    assert abs(inner[1].delta - (stats.auroc(s[train], labels[train]) - sub)) < 1e-12


def test_the_candidate_probe_is_the_pr13_recipe_on_the_training_cards() -> None:
    """§7.11: the learned candidate-only probe is X2's PR #13 recipe (StandardScaler then
    LogisticRegression(C=1, max_iter=2000)), unweighted, fitted per fold on the six training
    cards only: equal to scikit-learn fold by fold, blind to the label weights and to the
    held-out card's labels; a guarded fold is skipped."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    thin = CARDS[2]
    task = _task(yes_override={thin: 3})
    scores = {"F7@clm-latest": _planted(task, 2.0, seed=80)}
    items = _items(task, scores)
    rng = np.random.default_rng(81)
    x = rng.normal(size=(items.n, 12)) + (items.label == 0)[:, None] * 0.4
    probe = x1.candidate_probe(items, x)
    assert set(probe.skipped) == {thin} and probe.skipped[thin].startswith("insufficient")
    run, skipped = x1.probe_folds(items)
    assert skipped == probe.skipped and [f.held_out for f in run] == sorted(set(CARDS) - {thin})
    # the record scores the probe's p(Yes) for Yes (index 0), not its complement (§7.11)
    record = x1._probe_record(items, probe, B=B)
    pooled = ~np.isnan(probe.probs)
    want = stats.auroc(probe.probs[pooled], items.label[pooled])
    assert record.auroc is not None and record.auroc.point == want and want > 0.6
    assert record.n == int(pooled.sum()) and record.n_folds == 6
    for card in CARDS:
        test = items.card_items(card)
        if card == thin:
            assert np.all(np.isnan(probe.probs[test]))
            continue
        train = np.flatnonzero(items.card_arr != card)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(x[train], items.label[train])
        want = model.predict_proba(x[test])[:, list(model.classes_).index(0)]
        assert np.array_equal(probe.probs[test], want), card
        assert probe.folds[card]["n_train"] == len(train)
    heavier = _items(task, scores)
    heavier.weight = np.where(heavier.label == 0, 3.0, 0.2)
    heavier.__post_init__()
    assert np.array_equal(x1.candidate_probe(heavier, x).probs, probe.probs, equal_nan=True)
    held = CARDS[4]
    flipped = _items(task, scores)
    on = flipped.card_arr == held
    flipped.label = np.where(on, 1 - flipped.label, flipped.label)
    flipped.__post_init__()
    again = x1.candidate_probe(flipped, x)
    assert np.array_equal(again.probs[on], probe.probs[on])


# -- the second review round: identity, grid, replay scope, pinned settings, inner folds ---------


def _two_tasks() -> tuple[dict[str, Task], dict[str, x1.TaskItems]]:
    """Both X1 tasks (synthetic), every arm of the grid."""
    term = _task()
    onto = _task(name="neon_ontology_fits", task_id="column.ontology_fits")
    return {t.name: t for t in (term, onto)}, {t.name: _clean_items_for(t) for t in (term, onto)}


REG_KW: dict[str, Any] = {
    "date": "2026-10-01",
    "labels_sha256": "a" * 64,
    "labels_content_sha256": "b" * 64,
    "B": B,
}


def test_the_registration_covers_the_framings_and_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§13.1-§13.2: registered only with the registered framings lock and each model's
    registered fingerprint; any other (or none) is a deviation, and ``decide_from_json``
    re-derives the status from what ``x1.json`` records (§12.6 (0))."""
    monkeypatch.setattr(reg, "REGISTERED", _registration())
    tasks, items = _two_tasks()
    run = x1.run_x1_items(tasks, items, fingerprints=FPS, **REG_KW)
    assert run.results.x1.registered and run.results.x1.deviations == []
    assert all(c.pre_registered for c in run.results.cells.values())
    path = x1.write_x1(run, tmp_path / "ok")
    assert set(x1.decide_from_json(path)) == set(tasks)  # the registered run replays and recomputes
    other = {**FPS, RAW: {"clm_model_fp": "f" * 12}}
    assert x1.run_x1_items(tasks, items, fingerprints=other, **REG_KW).results.x1.deviations == [
        "clm-raw fingerprint: clm_model_fp ffffffffffff (registered clm-raw00000)"
    ]
    none = x1.run_x1_items(tasks, items, **REG_KW).results.x1
    assert not none.registered and [d.split(":")[0] for d in none.deviations] == [LATEST, RAW]
    with monkeypatch.context() as mp:
        mp.setattr(x1.fr, "lock_sha", lambda: "0" * 64)
        rotated = x1.run_x1_items(tasks, items, fingerprints=FPS, **REG_KW).results.x1
    assert rotated.deviations == [f"framings lock_sha 000000000000 is not {fr.lock_sha()[:12]}"]
    assert rotated.framings_lock_sha == "0" * 64
    pristine = path.read_text()
    for edit in (
        lambda raw: raw["x1"].__setitem__("framings_lock_sha", "0" * 64),
        lambda raw: raw["x1"]["models"][RAW]["fingerprint"].__setitem__("clm_model_fp", "f" * 12),
    ):
        path.write_text(pristine)
        _tamper(path, edit)
        with pytest.raises(x1.X1ReproError, match="registered status"):
            x1.decide_from_json(path, recompute=False, **LAX)


def test_the_registered_stamp_needs_every_configured_task_and_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§13.2 (review finding): the configuration names the grid it ran. An item table that does
    not score exactly the configured arms is refused; a registered file without one of its
    tasks, or whose trace (full run or a nested fold) lacks a configured arm, is refused."""
    monkeypatch.setattr(reg, "REGISTERED", _registration())
    tasks, items = _two_tasks()
    path = x1.write_x1(x1.run_x1_items(tasks, items, fingerprints=FPS, **REG_KW), tmp_path)
    pristine = path.read_text()

    def drop_task(raw: dict[str, Any]) -> None:
        del raw["x1"]["tasks"]["neon_ontology_fits"]
        raw["cells"] = {k: v for k, v in raw["cells"].items() if not k.startswith("neon_onto")}

    def drop_arm(raw: dict[str, Any]) -> None:
        rec = raw["x1"]["tasks"]["neon_term_fits"]
        del rec["decision"]["arms"]["F1@clm-raw"]
        del rec["arms"]["F1@clm-raw"]
        del raw["cells"]["neon_term_fits.calibrated.F1@clm-raw"]

    def drop_inner_arm(raw: dict[str, Any]) -> None:
        folds = raw["x1"]["tasks"]["neon_term_fits"]["nested"]["folds"]
        del folds[CARDS[2]]["decision"]["arms"]["F1@clm-raw"]

    for edit, message in (
        (drop_task, "tasks recorded"),
        (drop_arm, "not the configured"),
        (drop_inner_arm, "not the configured"),
    ):
        path.write_text(pristine)
        _tamper(path, edit)
        for recompute in (False, True):
            with pytest.raises(x1.X1ReproError, match=message):
                x1.decide_from_json(path, recompute=recompute)
    # the producer refuses a table that is not the configured grid ...
    partial = items["neon_term_fits"]
    partial.scores = {a: s for a, s in partial.scores.items() if a.startswith("F7@")}
    partial.__post_init__()
    with pytest.raises(x1.X1DataError, match="not the configured arms"):
        x1.run_x1_items(tasks, items, fingerprints=FPS, **REG_KW)
    # ... runs it under the configuration it does score, unregistered ...
    f7 = x1.run_x1_items(tasks, items | {"neon_ontology_fits": _restrict(items)}, framings=["F7"],
                         fingerprints=FPS, **REG_KW).results.x1  # fmt: skip
    assert not f7.registered and f7.deviations[0].startswith("framings ['F7']")
    # ... and evaluate_task never stamps a partial grid or a partial run registered
    term, full_items = _clean_items()
    for table, kw in ((partial, {}), (full_items, {"nested": False})):
        with pytest.raises(x1.X1DataError, match="every arm of"):
            x1.evaluate_task(
                term, table, labels_sha256="a" * 64, labels_content_sha256="b" * 64,
                registered=True, B=B, **kw,
            )  # fmt: skip
    with pytest.raises(x1.X1DataError, match="not a grid"):
        x1.run_x1_items(tasks, items, framings=["F7", "F2"], fingerprints=FPS, **REG_KW)


def _restrict(items: dict[str, x1.TaskItems]) -> x1.TaskItems:
    onto = items["neon_ontology_fits"]
    onto.scores = {a: s for a, s in onto.scores.items() if a.startswith("F7@")}
    onto.__post_init__()
    return onto


def test_the_item_table_must_be_the_tasks_item_for_item() -> None:
    """``evaluate_task`` refuses an item table whose weights, targets or options are not the
    task's (review finding: only labels and cards were compared)."""
    task, items = _clean_items()

    def variant(**over: Any) -> Task:
        fields = {
            "cards": task.cards, "weights": task.weights, "products": task.products,
            "option_keys": task.option_keys, "meta": dict(task.meta),
        } | over  # fmt: skip
        return Task(task.name, task.spec, over.pop("items", task.items), "s", "s", **fields)

    kw: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64}
    heavier = variant(weights=[w + 0.1 for w in task.weights])
    with pytest.raises(x1.X1DataError, match="weights"):
        x1.evaluate_task(heavier, items, registered=False, B=B, **kw)
    renamed = variant(option_keys=["X:other", *task.option_keys[1:]])
    with pytest.raises(x1.X1DataError, match="targets or options"):
        x1.evaluate_task(renamed, items, registered=False, B=B, **kw)
    moved = list(task.items)
    moved[0] = ({**moved[0][0], "column": {"name": "colX"}}, moved[0][1])
    with pytest.raises(x1.X1DataError, match="targets or options"):
        x1.evaluate_task(
            Task(task.name, task.spec, moved, "s", "s", cards=task.cards, weights=task.weights,
                 products=task.products, option_keys=task.option_keys, meta=dict(task.meta)),
            items, registered=False, B=B, **kw,
        )  # fmt: skip
    with pytest.raises(x1.X1DataError, match="unique"):
        x1.TaskItems(
            task=task.name, task_id=task.task_id, target=[items.target[0]] * 2,
            option=[items.option[0]] * 2, card=items.card[:2], label=items.label[:2],
            weight=items.weight[:2],
        )  # fmt: skip


def test_the_replay_holds_every_reported_number_to_its_trace_and_its_items(
    tmp_path: Path,
) -> None:
    """§12.6 (a) (review finding: x1.md printed numbers the replay never checked): the arm
    records' AUROC, shuffle verdict and LOCO-Platt NLL are the trace's, the cells' diagnostics
    the records', each cell's counts, metrics and novel-key block its own items', and every
    stored verdict and interval used the run's bootstrap settings."""
    path = x1.write_x1(_clean_run(), tmp_path)
    pristine = path.read_text()
    task, cell = "neon_term_fits", "neon_term_fits.calibrated.F7@clm-latest"

    def arms(raw: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = raw["x1"]["tasks"][task]["arms"]
        return out

    def flip_shuffle(raw: dict[str, Any]) -> None:
        r = arms(raw)["F9@clm-latest"]["shuffle"]["rule_r"]
        r["passed"], r["reason"] = True, "passed"

    def bump(where: Any, key: str, by: float = 0.01) -> Any:
        return lambda raw: where(raw).__setitem__(key, where(raw)[key] + by)

    edits: dict[str, tuple[Any, str]] = {
        "auroc": (bump(lambda r: arms(r)["F7@clm-latest"]["auroc"], "lower"), "AUROC is not"),
        "shuffle verdict": (flip_shuffle, "shuffle rule R is not"),
        "shuffle auroc": (
            bump(lambda r: arms(r)["F7@clm-latest"]["shuffle"]["auroc"], "point"),
            "diagnostics are not the arm record's",
        ),
        "loco nll": (bump(lambda r: arms(r)["F7@clm-latest"]["loco"], "nll"), "LOCO-Platt NLL"),
        "inverted": (
            lambda r: arms(r)["F7@clm-latest"]["loco"].__setitem__("inverted", [CARDS[0]]),
            "inverted folds",
        ),
        "cell metrics": (bump(lambda r: r["cells"][cell]["metrics"], "nll"), "metrics are not"),
        "cell item": (
            lambda r: r["cells"][cell]["items"][0].__setitem__("probs", [0.999, 0.001]),
            "metrics are not",
        ),
        "cell counts": (
            bump(lambda r: r["cells"][cell]["counts"], "n_neg", 1),
            "counts are not its items",
        ),
        "novel block": (
            bump(lambda r: r["cells"][cell]["baselines"]["novel_key"], "nll"),
            "novel-key block",
        ),
        "cell latency": (
            lambda r: r["cells"][cell]["diagnostics"]["latency"].__setitem__("p50", 1.0),
            "diagnostics are not the arm record's",
        ),
        "calls": (
            lambda r: arms(r)["F7@clm-latest"]["diagnostics"].__setitem__("calls", 1),
            "diagnostics are not the arm record's",
        ),
        "rule seed": (
            lambda r: r["x1"]["tasks"][task]["decision"]["comparisons"][0]["rule_r"].__setitem__(
                "seed", 1
            ),
            "its bootstrap used",
        ),
        "ci seed": (
            lambda r: arms(r)["F7@clm-latest"]["auroc"].__setitem__("seed", 1),
            "bootstrap is not the run's",
        ),
    }
    for name, (edit, message) in edits.items():
        path.write_text(pristine)
        _tamper(path, edit)
        with pytest.raises(x1.X1ReproError, match=message):
            x1.decide_from_json(path, recompute=False, **LAX)
        assert name
    path.write_text(pristine)
    x1.decide_from_json(path, **LAX)


def _rewrite_items(path: Path, update: str) -> None:
    """Apply an SQL ``update`` to ``x1.json``'s items file and record its new sha256 there (so
    only the recompute can catch the edit)."""
    import duckdb

    items = path.with_name("x1_items.parquet")
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE items AS SELECT * FROM read_parquet(?)", [str(items)])
        con.execute(update)
        con.execute(
            "COPY (SELECT * FROM items ORDER BY task, framing, model, item) "  # noqa: S608
            f"TO '{items}' (FORMAT PARQUET)"
        )
    finally:
        con.close()
    _tamper(
        path,
        lambda raw: raw["x1"]["items_file"].__setitem__(
            "sha256", hashlib.sha256(items.read_bytes()).hexdigest()
        ),
    )


def test_the_recompute_rebuilds_the_records_and_the_items_file(tmp_path: Path) -> None:
    """§12.6 (b): from the items file, the arm records' statistics (the sensitivity and
    mean-context records included), the ``p_loco`` column, the cells' per-item predictions,
    the shuffle draws and the cross-arm identity of every item are rebuilt; an edit the replay
    cannot see is refused by the recompute."""
    path = x1.write_x1(_clean_run(), tmp_path)
    pristine, items_bytes = path.read_text(), path.with_name("x1_items.parquet").read_bytes()
    rec = ["x1", "tasks", "neon_term_fits"]

    def at(raw: dict[str, Any], *keys: str) -> Any:
        out: Any = raw
        for k in [*rec, *keys]:
            out = out[k]
        return out

    def foreign_item(raw: dict[str, Any]) -> None:
        cell = raw["cells"]["neon_term_fits.calibrated.F7@clm-latest"]
        cell["items"][0]["target_sha256"] = "f" * 64

    def sensitivity(raw: dict[str, Any]) -> None:
        at(raw, "arms", "F7@clm-latest", "sensitivity", "unweighted")["nll"] = 0.5

    def mean_context(raw: dict[str, Any]) -> None:
        at(raw, "arms", "F7@clm-latest", "mean_context")["nll"] = 0.5

    record_edits: dict[str, tuple[Any, str]] = {
        "sensitivity": (sensitivity, "recomputed from the items, the arm record differs"),
        "mean context": (mean_context, "recomputed from the items, the arm record differs"),
        "draws": (lambda r: at(r, "shuffle").__setitem__("index_sha256", "0" * 64), "draws"),
        "summary": (
            lambda r: at(r).__setitem__("class_counts", {"0": 1, "1": 279}),
            "task summary is not the items file's",
        ),
        "cell item": (foreign_item, "not an item of the items file"),
    }
    for name, (edit, message) in record_edits.items():
        path.write_text(pristine)
        _tamper(path, edit)
        x1.decide_from_json(path, recompute=False, **LAX)  # the replay cannot see it
        with pytest.raises(x1.X1ReproError, match=message):
            x1.decide_from_json(path, **LAX)
        assert name
    for update, message in (
        (
            "UPDATE items SET p_loco = p_loco + 1e-6 WHERE model = 'clm-latest' AND framing = "
            "'F7' AND p_loco IS NOT NULL AND item = 5",
            "p_loco is not the recomputed",
        ),
        (
            "UPDATE items SET label_index = 1 - label_index WHERE model = 'clm-raw' AND "
            "framing = 'F4' AND item = 3",
            "item columns differ",
        ),
    ):
        path.write_text(pristine)
        path.with_name("x1_items.parquet").write_bytes(items_bytes)
        _rewrite_items(path, update)
        x1.decide_from_json(path, recompute=False, **LAX)
        with pytest.raises(x1.X1ReproError, match=message):
            x1.decide_from_json(path, **LAX)
    # a cell item whose prediction is not the recomputed LOCO-Platt (its metrics aside)
    path.write_text(pristine)
    path.with_name("x1_items.parquet").write_bytes(items_bytes)
    res = x1.load_x1(path)
    tables = x1.read_items(path.with_name("x1_items.parquet"))
    items = tables["neon_term_fits"]
    ev = x1.ItemEvidence(items, B=B)
    st = x1._arm_stats(items, ev, "F7@clm-latest")
    cell = res.cells["neon_term_fits.calibrated.F7@clm-latest"]
    x1._check_cell_items(cell, items, st.p_full, x1.REPRO_TOLERANCE, "ok")
    assert cell.items is not None
    moved = cell.items[0].model_copy(update={"probs": [cell.items[0].probs[0] + 1e-6,
                                                        cell.items[0].probs[1] - 1e-6]})  # fmt: skip
    for bad, message in (
        (cell.model_copy(update={"items": [moved, *cell.items[1:]]}), "not the recomputed one"),
        (cell.model_copy(update={"items": cell.items[1:]}), "not the pooled items"),
    ):
        with pytest.raises(x1.X1ReproError, match=message):
            x1._check_cell_items(bad, items, st.p_full, x1.REPRO_TOLERANCE, "bad")


def test_a_registered_file_holds_the_published_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1.3 in the replay (review finding): a file on the registered labels must hold the
    published items per task, n, class counts and items per card."""
    tasks, items = _two_tasks()
    counts = {}
    for name, t in tasks.items():
        counts[name] = reg.TaskCounts(
            len(t.items), t.class_counts(), {c: t.cards.count(c) for c in CARDS}, 0.5
        )
    stand_in = replace(_registration(), counts=counts)
    monkeypatch.setattr(reg, "REGISTERED", stand_in)
    path = x1.write_x1(x1.run_x1_items(tasks, items, fingerprints=FPS, **REG_KW), tmp_path)
    x1.decide_from_json(path, recompute=False)
    wrong = dict(counts)
    wrong["neon_term_fits"] = replace(counts["neon_term_fits"], n=counts["neon_term_fits"].n + 1)
    monkeypatch.setattr(reg, "REGISTERED", replace(stand_in, counts=wrong))
    with pytest.raises(x1.X1ReproError, match="not the published"):
        x1.decide_from_json(path, recompute=False)


def test_the_bootstraps_and_draws_use_the_registered_settings_at_every_call_site() -> None:
    """§13.3 at the call sites (review finding: mutating a seed, B, alpha or K inside the
    nesting or ItemEvidence went unnoticed): each nested fold is ``decide`` on an ItemEvidence
    built here with the literal registered settings, and ItemEvidence's intervals, shuffle
    rule R and NLL comparisons are the statistics' own functions at seed 0."""
    _, items = _clean_items()
    nested = x1.nested_selection(items, B=B)
    for card in CARDS:
        train = np.flatnonzero(items.card_arr != card)
        want = x1.decide(
            x1.ItemEvidence(items, train, B=B, seed=0, alpha=0.05, shuffle_k=200, shuffle_seed=0)
        )
        assert nested.folds[card].decision == want, card
    ev = x1.ItemEvidence(items, B=B)
    labels, cards = items.label, [str(c) for c in items.card]
    weights = stats.bootstrap_weights(cards, B=B, seed=0)
    for arm in ("F7@clm-latest", "F9@clm-raw"):
        s = items.scores[arm].s
        assert ev.auroc(arm) == stats.auroc_ci(s, labels, cards, B=B, seed=0, alpha=0.05)
        assert ev.auroc(arm) == stats.auroc_ci(s, labels, cards, B=B, alpha=0.05, weights=weights)
        shuffled = items.shuffled(arm, 200, 0)
        sh = ev.shuffle(arm)
        assert sh is not None
        assert sh[0] == stats.mean_auroc_ci(shuffled, labels, cards, B=B, seed=0, alpha=0.05)
        assert sh[1] == stats.rule_r_auroc_mean(s, shuffled, labels, cards, B=B, seed=0, alpha=0.05)
    pool = ev.plan.pool
    pool_cards = [str(c) for c in items.card_arr[pool]]
    for a, b in (("F7@clm-latest", "F7@clm-raw"), ("F1@clm-latest", "F7@clm-latest")):
        want_r = stats.rule_r(
            "nll", ev.loco(a).matrix(), ev.loco(b).matrix(), items.label[pool], pool_cards,
            B=B, seed=0, alpha=0.05, weights=stats.bootstrap_weights(pool_cards, B=B, seed=0),
        )  # fmt: skip
        got = ev.compare_nll(a, b)
        assert got == x1.RuleRSummary.of(want_r) and (got.B, got.seed, got.alpha) == (B, 0, 0.05)


def test_the_shuffle_draws_are_seed_0_rejection_sampled_derangements() -> None:
    """§5.2 against an independent implementation (one ``default_rng(0)`` stream, cards sorted,
    ``rng.permutation(m)`` until no fixed point) and golden draws of numpy's PCG64 at seed 0."""
    sizes = {"b": 3, "a": 5, "c": 4, "solo": 1}
    draws = x1.shuffle_draws(sizes)
    rng = np.random.default_rng(0)
    for card in sorted(sizes):
        m = sizes[card]
        if m < 2:
            assert draws[card] is None
            continue
        want = []
        for _ in range(x1.SHUFFLE_K):
            while True:
                perm = rng.permutation(m)
                if not np.any(perm == np.arange(m)):
                    break
            want.append(perm)
        got = draws[card]
        assert got is not None and np.array_equal(got, np.asarray(want)), card
    a, b, c = draws["a"], draws["b"], draws["c"]
    assert a is not None and b is not None and c is not None
    assert a[:3].tolist() == [[2, 4, 3, 0, 1], [4, 0, 3, 2, 1], [2, 4, 3, 0, 1]]
    assert b[:3].tolist() == [[1, 2, 0], [1, 2, 0], [2, 0, 1]]
    assert c[:3].tolist() == [[2, 3, 1, 0], [1, 0, 3, 2], [3, 2, 0, 1]]
    whole = np.ascontiguousarray(np.concatenate([a, b, c], axis=1), dtype="<i8")
    assert hashlib.sha256(whole.tobytes()).hexdigest() == (
        "f90e83b4c7e9853d80ede0d569baa57befe061c20a0c537ac061499f01367ff4"
    )


def _sized(sizes: list[int], yes: list[int] | None = None) -> Task:
    """A task with ``sizes[k]`` items on card k (every other item Yes, or the first ``yes[k]``)."""
    per_card = dict(zip(CARDS, sizes, strict=True))
    override = None if yes is None else dict(zip(CARDS, yes, strict=True))
    task = _task(per_card=max(sizes), yes_every=2, yes_override=override)
    keep = [
        i
        for i, c in enumerate(task.cards)
        if sum(1 for j in range(i) if task.cards[j] == c) < per_card[c]
    ]
    return Task(
        task.name, task.spec, [task.items[i] for i in keep], "s", "s",
        cards=[task.cards[i] for i in keep], weights=[task.weights[i] for i in keep],
        products=[task.products[i] for i in keep],
        option_keys=[task.option_keys[i] for i in keep], meta=dict(task.meta),
    )  # fmt: skip


def test_inside_a_nested_fold_the_floor_and_the_guards_count_its_training_cards_only() -> None:
    """§6.3, §8.2 (review finding: only outer folds were tested): holding out card 6, the inner
    fold of card 4 trains on 99 items (skipped below the floor) and that of card 5 on exactly 100
    (runs), whatever the outer card adds; and an inner fold whose 30th Yes would come from the
    outer card is guarded, while the full run's fold of the same card is not."""
    task = _sized([20, 20, 20, 20, 20, 19, 40])
    items = _items(task, {"F7@clm-latest": _planted(task, 2.0, seed=91)})
    train = np.flatnonzero(items.card_arr != CARDS[6])
    inner = x1.fold_plan(items, train)
    assert inner.skipped[CARDS[4]] == "below_floor 99 < 100 training items (calibrated)"
    assert CARDS[5] not in inner.skipped
    assert CARDS[4] not in x1.fold_plan(items).skipped  # the full run trains it on 139
    nested = x1.nested_selection(items, B=B)
    evidence = nested.folds[CARDS[6]].decision.arms["F7@clm-latest"].evidence
    assert evidence.n_loco == len(inner.pool) and evidence.n_loco < len(train)
    guarded = _sized([40] * 7, yes=[5, 6, 6, 6, 6, 5, 6])
    gitems = _items(guarded, {"F7@clm-latest": _planted(guarded, 2.0, seed=92)})
    gtrain = np.flatnonzero(gitems.card_arr != CARDS[6])
    assert (
        x1.fold_plan(gitems, gtrain)
        .skipped[CARDS[0]]
        .startswith("insufficient_train_per_class {0: 29")
    )
    assert CARDS[0] not in x1.fold_plan(gitems).skipped  # 35 Yes train it in the full run


def test_inner_tokens_per_target_and_an_inner_fold_that_pools_nothing() -> None:
    """§7.7 inside a nested fold: tokens per target over its six training cards' targets only
    (cards here differ in context tokens); §8.5: an inner LOCO that pools nothing (every inner
    fold below the floor) makes the fold ``undecidable``, skipped as ``inner_undecidable``."""
    task = _task()
    items = _items(task, {"F7@clm-latest": _planted(task, 3.0, seed=93)})
    k = np.asarray([CARDS.index(c) for c in items.card])
    items.tokens["F7"] = x1.TokenCounts(
        context=200 + 50 * k, candidate=20 + 3 * k, anchor=np.full(items.n, 12)
    )
    nested = x1.nested_selection(items, B=B)
    for card in CARDS:
        on = items.card_arr != card
        per: dict[str, int] = {}
        for i in np.flatnonzero(on):
            t = items.target[i]
            if t not in per:
                per[t] = int(items.tokens["F7"].context[i] + items.tokens["F7"].anchor[i])
            per[t] += int(items.tokens["F7"].candidate[i])
        got = nested.folds[card].decision.tokens_per_target["F7"]
        assert got == pytest.approx(float(np.mean(list(per.values()))), rel=1e-12), card
    small = _task(per_card=17)  # 119 items: full-run folds train on 102, inner ones on 85
    sitems = _items(small, {"F7@clm-latest": _planted(small, 3.0, seed=94)})
    assert x1.fold_plan(sitems).skipped == {}
    snested = x1.nested_selection(sitems, B=B)
    assert {c: f.outcome for c, f in snested.fold_choices.items()} == dict.fromkeys(
        CARDS, "undecidable"
    )
    assert snested.skipped_folds == dict.fromkeys(CARDS, "inner_undecidable")
    inner = snested.folds[CARDS[0]].decision
    assert inner.arms["F7@clm-latest"].qualified and inner.choice is None


def test_the_report_only_controls_have_their_values() -> None:
    """§7.11, §11.1 values (review finding: only their presence was tested): the mean-context
    score is ``scale·(z̄·zc − z̄·za)`` with z̄ the L2-normalised mean context; the arm's
    mean-context record is that score's AUROC and LOCO-Platt NLL (not s's)."""
    rng = np.random.default_rng(3)
    zt = l2(rng.normal(size=(5, 16))).astype(np.float64)
    zc = l2(rng.normal(size=(9, 16))).astype(np.float64)
    za = l2(rng.normal(size=(9, 16))).astype(np.float64)
    zbar = zt.mean(axis=0) / np.linalg.norm(zt.mean(axis=0))
    want = 100.0 * (zc @ zbar - za @ zbar)
    assert np.allclose(x1.mean_context_scores(zt, zc, za, 100.0), want, rtol=1e-13, atol=1e-12)
    run = _clean_run(nested=False)
    rec = run.results.x1.tasks["neon_term_fits"]
    items = run.items["neon_term_fits"]
    mc = items.scores["F7@clm-latest"].mean_context
    assert mc is not None
    record = rec.arms["F7@clm-latest"].mean_context
    assert record is not None
    assert record.auroc.point == stats.auroc(mc, items.label)
    assert record.auroc.point != stats.auroc(items.scores["F7@clm-latest"].s, items.label)
    p = np.full(items.n, np.nan)
    for card in CARDS:
        test, train = items.card_arr == card, items.card_arr != card
        fit = calibrate.fit_platt(mc[train], items.positive[train], items.weight[train])
        p[test] = fit.p_yes(mc[test])
    nll = float(np.mean(-np.log(np.where(items.label == 0, p, 1.0 - p))))
    assert record.nll == pytest.approx(nll, rel=1e-12)


def test_the_mean_auroc_rule_r_counts_the_cards_rule_r_counts_and_halves_ties() -> None:
    """§5.4 (ii) at its edges (review finding): a card counts with 10 items and both classes,
    not with 9, and not with one class; tied (integer) scores count one half in the pair kernel,
    as in ``stats.auroc``. Against brute force, and for one draw against ``card_sign_test``."""
    rng = np.random.default_rng(17)
    plan = {"c10": (10, 4), "c9": (9, 4), "one": (14, 0), "d1": (12, 5), "d2": (16, 6),
            "d3": (11, 3), "d4": (20, 8)}  # fmt: skip
    cards, labels = [], []
    for card, (n, n_yes) in plan.items():
        cards += [card] * n
        labels += [0] * n_yes + [1] * (n - n_yes)
    y = np.asarray(labels)
    s = rng.integers(0, 4, size=len(y)) + (y == 0)  # heavy ties
    shuffles = rng.integers(0, 4, size=(25, len(y))).astype(float)
    r = stats.rule_r_auroc_mean(s, shuffles, y, cards, B=100)
    assert set(r.sign.per_card) == {"c10", "d1", "d2", "d3", "d4"}
    on = {c: np.asarray(cards) == c for c in plan}
    for card, d in r.sign.per_card.items():
        m = on[card]
        want = stats.auroc(s[m], y[m]) - np.mean([stats.auroc(row[m], y[m]) for row in shuffles])
        assert abs(d - want) < 1e-12, card
    assert r.sign.m_c == 5 and r.sign.needed == 4
    brute = stats.auroc(s, y) - np.mean([stats.auroc(row, y) for row in shuffles])
    assert abs(r.delta - brute) < 1e-12
    kernel = stats.pair_kernel(s, y == 0)
    assert float(kernel.mean()) == pytest.approx(stats.auroc(s, y), abs=1e-15)
    assert np.isin(kernel, (0.0, 0.5, 1.0)).all() and (kernel == 0.5).any()  # ties weigh ½
    one = stats.rule_r_auroc_mean(s, shuffles[:1], y, cards, B=100)
    plain = stats.card_sign_test(
        "auroc", stats.score_matrix(s), stats.score_matrix(shuffles[0]), y, cards
    )
    assert one.sign.per_card == pytest.approx(plain.per_card, abs=1e-12)
    assert (one.sign.m_c, one.sign.wins, one.sign.needed) == (plain.m_c, plain.wins, plain.needed)
