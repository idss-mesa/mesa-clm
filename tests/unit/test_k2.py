"""K2 (``mesa_clm.bench.k2``; plan §8; the M4 analysis plan R2) on synthetic results files: the
cells are built by ``assemble_cell`` from a planted term.fits task (generated labels, planted
probabilities), never from a committed file. Checked: the candidates and the best tier by
pooled NLL; every condition with its numbers (``beats_lookup_novel``, non-inferiority to AnyJev
on the full item set with the card-sign condition read literally as rule R's sign test — the
margin-shifted variant is reported in integers, never gated — ECE and its cluster upper bound);
the verdicts a, b and c, a tier within −0.02 on every card that beats AnyJev on fewer than 80%
of them (not a), the boundaries (a lower bound exactly −margin fails; ECE ≤ 0.08 with an upper
bound > 0.12 is b), an exploratory or unregistered nested cell ineligible, a cell without a task
count failing closed, a closed choice K2(c) by construction, a tier that does not pool the full
item set, a missing AnyJev cell; the clm-raw clause; the registration (other labels, an
unregistered x3); the files and the ``bench table`` rows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm.bench import k2 as K
from mesa_clm.bench import registered as reg
from mesa_clm.bench import run as m2run
from mesa_clm.bench import stats
from mesa_clm.bench.cells import CellError, assemble_cell
from mesa_clm.bench.results import BenchCell, BenchResults, environment
from mesa_clm.bench.tasks.base import Fold, Task
from mesa_clm.learn.calibrate import sigmoid
from mesa_clm.tasks import TASKS
from tests.unit.test_probe import rank_task

B = 200
IDENTITY: dict[str, Any] = {"labels_sha256": "a" * 64, "labels_content_sha256": "b" * 64}
FP = {
    "encoder_fp": "c" * 12,
    "clm_model_fp": "d" * 12,
    "schema_sha256": "5" * 64,
    "serving_lock_sha": "6" * 64,
}


def _cell(
    task: Task,
    p_yes: np.ndarray,
    *,
    tier: str,
    framing: str,
    selection: str = "nested",
    variant: str | None = None,
    servable: bool = True,
    folds: list[Fold] | None = None,
    idx: list[int] | None = None,
    exploratory: bool = False,
    pre_registered: bool = True,
    task_counts: bool = True,
) -> BenchCell:
    folds = folds if folds is not None else list(task.leave_one_card_out())
    idx = idx if idx is not None else sorted(i for f in folds for i in f.test)
    p = np.clip(p_yes[idx], 1e-6, 1 - 1e-6)
    diagnostics = (
        {"task_counts": {"n": len(task.items), "class_counts": task.class_counts()}}
        if task_counts
        else {}
    )
    return assemble_cell(
        task,
        idx,
        np.stack([p, 1 - p], axis=1),
        folds,
        tier=tier,  # type: ignore[arg-type]
        framing=framing,
        variant=variant,
        model="clm-latest" if servable else None,
        question_key="q" * 16,
        fingerprint=FP if servable else None,
        feature_spec="lowdim.v1" if tier == "probe" else None,
        selection=selection,  # type: ignore[arg-type]
        pre_registered=pre_registered,
        exploratory=exploratory,
        servable=servable,
        diagnostics=diagnostics,
        B=B,
        **IDENTITY,
    )


def _results(name: str, cells: list[BenchCell]) -> BenchResults:
    return BenchResults(
        date="2026-10-05",
        name=name,
        mesa_clm="0",
        labels_sha256=IDENTITY["labels_sha256"],
        labels_content_sha256=IDENTITY["labels_content_sha256"],
        environment=environment(),
        cells={c.key: c for c in cells},
    )


@pytest.fixture(scope="module")
def world() -> dict[str, Any]:
    task, s = rank_task(40, seed=3)
    rng = np.random.default_rng(7)
    truth = sigmoid(1.8 * s - 0.5)
    good = np.clip(truth + rng.normal(0, 0.03, len(s)), 0.01, 0.99)  # near the truth: wins
    anyjev = np.clip(truth + rng.normal(0, 0.25, len(s)), 0.01, 0.99)  # noisier
    noise = rng.random(len(s))  # no signal
    return {"task": task, "good": good, "anyjev": anyjev, "noise": noise}


def _inputs() -> dict[str, dict[str, str]]:
    return {
        k: {"path": f"bench/results/2026-10-05/{k}.json", "sha256": "0" * 64}
        for k in ("tiers", "x2", "x3")
    }


def _standin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reg,
        "REGISTERED",
        reg.Registration(
            snapshot="s",
            labels_sha256=IDENTITY["labels_sha256"],
            labels_content_sha256=IDENTITY["labels_content_sha256"],
            published="p",
            B=B,
        ),
    )


def test_verdict_a_with_a_probe_that_wins(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _standin(monkeypatch)
    task = world["task"]
    tiers = _results("tiers", [_cell(task, world["noise"], tier="calibrated", framing="F7")])
    x2 = _results(
        "x2",
        [
            _cell(
                task,
                world["anyjev"],
                tier="baseline",
                framing="anyjev_l2",
                selection="none",
                servable=False,
            )
        ],
    )
    x3 = _results(
        "x3",
        [
            _cell(task, world["good"], tier="probe", framing="F7"),
            _cell(task, world["good"], tier="probe", framing="F7", variant="latest"),
            _cell(task, world["good"], tier="probe", framing="F7", variant="raw"),
        ],
    )
    res = K.evaluate_k2(
        tiers=tiers, x2=x2, x3=x3, date="2026-10-05", inputs=_inputs(), registered=True, B=B
    )
    assert res.registered and res.deviations == []
    t = res.tasks["neon_term_fits"]
    assert [c.tier for c in t.candidates] == ["calibrated", "probe"]
    assert all(c.eligible for c in t.candidates)
    assert t.best == "probe" and t.best_cite.endswith("x3.json#neon_term_fits.probe.F7")
    assert t.probe_vs_calibrated_nll is not None and t.promotion_probe_over_calibrated is True
    c = t.conditions
    assert c["beats_lookup_novel"].passed and c["beats_lookup_novel"].reason == "passed"
    ni = c["non_inferior_acc"]
    assert ni.passed and ni.numbers["join"]["n_paired"] == len(task.items) == ni.numbers["n_task"]
    assert ni.numbers["margin"] == 0.02 and ni.numbers["lower_bound"] > -0.02
    assert ni.numbers["acc_tier"] > ni.numbers["acc_anyjev"]
    cs = c["card_sign"]
    assert cs.passed and cs.numbers["m_c"] == 7 and cs.numbers["needed"] == 6
    assert set(cs.numbers["per_card"]) == set(task.cards) and "margin" not in cs.numbers
    # the gate is rule R's sign test verbatim: a card wins when Δacc > 0, strictly
    pa, pb, y, cl, _ = K.paired_items(
        x3.cells["neon_term_fits.probe.F7"], x2.cells["neon_term_fits.baseline.anyjev_l2"]
    )
    literal = stats.card_sign_test("acc", pa, pb, y, cl)
    assert literal.as_dict() == {k: cs.numbers[k] for k in literal.as_dict()}
    assert cs.numbers["wins"] == sum(1 for d in cs.numbers["per_card"].values() if d > 0) >= 6
    # the margin-shifted variant is reported in integers, never gated
    rep = cs.numbers["card_sign_margin_report"]
    assert rep["margin"] == 0.02 and rep["m_c"] == 7 and "REPORT-ONLY" in rep["what"]
    for card, row in rep["per_card"].items():
        ca, cb, n = row["correct_tier"], row["correct_anyjev"], row["n"]
        assert row["wins_within_margin"] == (50 * (ca - cb) > -n)
        assert cs.numbers["per_card"][card] == pytest.approx((ca - cb) / n)
    assert rep["wins"] >= cs.numbers["wins"] and rep["passed"] and rep["reason"] == "passed"
    e = c["ece"]
    assert e.passed and e.numbers["ece"] <= 0.08 and e.numbers["ece_upper"] <= 0.12
    assert t.verdict == "a" and t.reason.startswith("auto-eligible")
    # the clm-raw clause: identical @latest and @raw cells -> latest not ≻ raw, raw ≽ latest
    h = t.head_adds_nothing
    assert h is not None and h.evaluated and h.head_adds_nothing is True
    assert h.latest_not_better_nll["passed"] is False and h.raw_noninferior_acc["passed"] is True
    assert h.n_paired == len(task.items)
    # constants recorded; markdown names the verdict
    assert res.constants["acc_margin"] == 0.02 and res.constants["ece_upper_max"] == 0.12
    assert "| neon_term_fits | probe |" in K.markdown_k2(res) and "**a**" in K.markdown_k2(res)


def test_verdicts_b_and_c_and_the_full_set_condition(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _standin(monkeypatch)
    task = world["task"]
    anyjev = _cell(
        task,
        world["anyjev"],
        tier="baseline",
        framing="anyjev_l2",
        selection="none",
        servable=False,
    )
    x2 = _results("x2", [anyjev])
    # (b): the probe beats the lookup on novel keys but is miscalibrated (compressed towards 0.5)
    compressed = 0.5 + 0.2 * (world["good"] - 0.5)
    x3 = _results("x3", [_cell(task, compressed, tier="probe", framing="F7")])
    res = K.evaluate_k2(
        tiers=_results("tiers", []), x2=x2, x3=x3, date="d", inputs=_inputs(), registered=True, B=B
    )
    t = res.tasks["neon_term_fits"]
    assert t.best == "probe" and [c.tier for c in t.candidates] == ["probe"]
    assert t.conditions["beats_lookup_novel"].passed and not t.conditions["ece"].passed
    assert t.verdict == "b" and "ece" in t.reason
    assert t.head_adds_nothing is not None and not t.head_adds_nothing.evaluated
    # (c): no signal -> beats_lookup_novel false
    x3c = _results("x3", [_cell(task, world["noise"], tier="probe", framing="F7")])
    res_c = K.evaluate_k2(
        tiers=_results("tiers", []), x2=x2, x3=x3c, date="d", inputs=_inputs(), registered=True, B=B
    )
    tc = res_c.tasks["neon_term_fits"]
    assert (
        tc.verdict == "c" and tc.best == "probe" and not tc.conditions["beats_lookup_novel"].passed
    )
    # a tier that skipped a fold is not on the full item set: K2(a) fails on that condition
    folds = [f for f in task.leave_one_card_out() if f.held_out != task.cards[0]]
    partial = _cell(task, world["good"], tier="probe", framing="F7", folds=folds)
    res_p = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=x2,
        x3=_results("x3", [partial]),
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    tp = res_p.tasks["neon_term_fits"]
    assert tp.conditions["non_inferior_acc"].reason == "tier_not_on_full_item_set"
    assert tp.conditions["card_sign"].reason == "tier_not_on_full_item_set"
    assert tp.verdict == "b"
    # no AnyJev cell (the closed choices): the paired conditions fail by construction
    res_n = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=_results("x2", []),
        x3=x3,
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    assert res_n.tasks["neon_term_fits"].conditions["non_inferior_acc"].reason == "no_anyjev_cell"
    # no candidate at all: c with the reason
    k1 = _cell(
        task, world["good"], tier="calibrated", framing="F7", selection="none", exploratory=True
    )
    res_k = K.evaluate_k2(
        tiers=_results("tiers", [k1]),
        x2=x2,
        x3=_results("x3", []),
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    tk = res_k.tasks["neon_term_fits"]
    assert tk.best is None and tk.verdict == "c" and "selection 'none'" in tk.best_reason
    assert tk.candidates[0].eligible is False


def _anyjev_cell(world: dict[str, Any], p: np.ndarray | None = None) -> BenchCell:
    return _cell(
        world["task"],
        world["anyjev"] if p is None else p,
        tier="baseline",
        framing="anyjev_l2",
        selection="none",
        servable=False,
    )


def _k2(
    world: dict[str, Any], probe: BenchCell, anyjev: BenchCell | None, **kw: Any
) -> K.TaskVerdict:
    res = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=_results("x2", [] if anyjev is None else [anyjev]),
        x3=_results("x3", [probe]),
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
        **kw,
    )
    return res.tasks["neon_term_fits"]


def test_within_the_margin_on_every_card_but_not_beating_anyjev_is_not_a(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal card-sign gate: AnyJev identical to the tier on four cards (Δacc = 0 there,
    within −0.02 but not a win) and noisier on three: the tier wins on at most 3 < 6 cards, so
    K2(a) fails on card_sign while the margin-shifted report passes; verdict b."""
    _standin(monkeypatch)
    task = world["task"]
    first_four = set(sorted(set(task.cards))[:4])
    same = np.asarray([c in first_four for c in task.cards])
    anyjev_p = np.where(same, world["good"], world["anyjev"])
    t = _k2(
        world, _cell(task, world["good"], tier="probe", framing="F7"), _anyjev_cell(world, anyjev_p)
    )
    cs = t.conditions["card_sign"]
    assert not cs.passed and cs.reason == "sign_test_failed"
    assert cs.numbers["m_c"] == 7 and cs.numbers["needed"] == 6 and cs.numbers["wins"] <= 3
    assert all(cs.numbers["per_card"][c] == 0.0 for c in first_four)
    rep = cs.numbers["card_sign_margin_report"]
    assert rep["passed"] and all(rep["per_card"][c]["wins_within_margin"] for c in first_four)
    assert t.conditions["non_inferior_acc"].passed and t.conditions["ece"].passed
    assert t.conditions["beats_lookup_novel"].passed
    assert t.verdict == "b" and "card_sign" in t.reason


def test_boundaries_eligibility_and_the_unknown_task_count(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _standin(monkeypatch)
    task = world["task"]
    good = _cell(task, world["good"], tier="probe", framing="F7")
    anyjev = _anyjev_cell(world)
    pa, pb, y, cl, _ = K.paired_items(good, anyjev)
    # stats.non_inferior: a lower bound exactly −margin fails (strict), just above it passes.
    real_pb = stats.paired_bootstrap
    monkeypatch.setattr(
        stats, "paired_bootstrap", lambda *a, **k: replace(real_pb(*a, **k), lower=-0.02)
    )
    assert stats.non_inferior("acc", pa, pb, y, cl, 0.02, B=50).passed is False
    monkeypatch.setattr(
        stats, "paired_bootstrap", lambda *a, **k: replace(real_pb(*a, **k), lower=-0.02 + 1e-9)
    )
    assert stats.non_inferior("acc", pa, pb, y, cl, 0.02, B=50).passed is True
    monkeypatch.setattr(stats, "paired_bootstrap", real_pb)
    # ECE ≤ 0.08 but a cluster upper bound > 0.12: b, not a.
    real_ci = stats.metric_ci

    def high_upper(metric: Any, *a: Any, **k: Any) -> Any:
        ci = real_ci(metric, *a, **k)
        return replace(ci, upper=0.2) if metric == "ece" else ci

    monkeypatch.setattr(stats, "metric_ci", high_upper)
    t = _k2(world, good, anyjev)
    e = t.conditions["ece"]
    assert e.numbers["ece"] <= 0.08 and e.numbers["ece_upper"] == 0.2 and not e.passed
    assert e.reason == "ece_upper_bound_above_max" and t.verdict == "b"
    monkeypatch.setattr(stats, "metric_ci", real_ci)
    assert _k2(world, good, anyjev).verdict == "a"
    # An exploratory nested cell, or one not pre_registered, is ineligible with the reason.
    for over, why in (
        ({"exploratory": True}, "exploratory"),
        ({"pre_registered": False}, "not pre_registered"),
    ):
        t = _k2(world, _cell(task, world["good"], tier="probe", framing="F7", **over), anyjev)
        assert t.candidates[0].eligible is False and t.candidates[0].reason == why
        assert t.best is None and t.verdict == "c" and why in t.best_reason
    # A cell without diagnostics.task_counts and no registered count fails closed.
    t = _k2(
        world, _cell(task, world["good"], tier="probe", framing="F7", task_counts=False), anyjev
    )
    assert t.conditions["non_inferior_acc"].reason == "task_count_unknown"
    assert t.conditions["card_sign"].reason == "task_count_unknown"
    assert t.conditions["non_inferior_acc"].numbers["n_task"] is None and t.verdict == "b"


def test_a_closed_choice_is_k2c_by_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    _standin(monkeypatch)
    rng = np.random.default_rng(0)
    cards = [f"DP0.0000{1 + i // 4}.001.card{i}" for i in range(7)]
    items: list[tuple[dict[str, Any], int]] = []
    card_of: list[str] = []
    for ci, card in enumerate(cards):
        for t in range(20):
            items.append(
                (
                    {"card": {"dataset": card}, "column": {"name": f"c{ci}{t}"}},
                    int(rng.integers(0, 8)),
                )
            )
            card_of.append(card)
    task = Task(
        "neon_aspect", TASKS["column.aspect"], items, "s", "s", cards=card_of,
        products=[c.rsplit(".", 1)[0] for c in cards for _ in range(20)], option_keys=[""] * len(items),
        meta={"min_weight": 0.6, "masked": True, "mask": None,
              "label_sources": {"consensus_majority": len(items)}, "excluded": {}},
    )  # fmt: skip
    probs = rng.dirichlet(np.ones(8), len(items))
    folds = list(task.leave_one_card_out())
    cell = assemble_cell(
        task, list(range(len(items))), probs, folds, tier="probe", framing="F7", model="clm-latest",
        question_key="q" * 16, fingerprint=FP, feature_spec="choice.state.v1", selection="nested",
        pre_registered=True, exploratory=False, servable=True, B=B, **IDENTITY,
    )  # fmt: skip
    res = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=_results("x2", []),
        x3=_results("x3", [cell]),
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    t = res.tasks["neon_aspect"]
    assert t.shape == "choice" and t.verdict == "c" and t.head_adds_nothing is None
    assert t.conditions["beats_lookup_novel"].reason == "auroc_not_applicable_k8"


def test_registration_pairing_and_files(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _standin(monkeypatch)
    task = world["task"]
    x3 = _results("x3", [_cell(task, world["good"], tier="probe", framing="F7")])
    x2 = _results(
        "x2",
        [
            _cell(
                task,
                world["anyjev"],
                tier="baseline",
                framing="anyjev_l2",
                selection="none",
                servable=False,
            )
        ],
    )
    # other labels than the registration's, other B: unregistered with the deviations listed
    res = K.evaluate_k2(
        tiers=_results("tiers", []), x2=x2, x3=x3, date="d", inputs=_inputs(), registered=True, B=50
    )
    assert not res.registered and any("B 50" in d for d in res.deviations)
    foreign = x2.model_copy(update={"labels_sha256": "f" * 64})
    res2 = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=foreign,
        x3=x3,
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    assert any("x2.json labels_sha256" in d for d in res2.deviations)
    unreg = x3.model_copy(
        update={
            "cells": {
                k: c.model_copy(update={"pre_registered": False}) for k, c in x3.cells.items()
            }
        }
    )
    res3 = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=x2,
        x3=unreg,
        date="d",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    assert "x3.json is not the pre-registered run" in res3.deviations
    # the pairing refuses a label disagreement
    a = x3.cells["neon_term_fits.probe.F7"]
    bad = a.model_copy(
        update={"items": [it.model_copy(update={"label": 1 - it.label}) for it in a.items or []]}
    )
    with pytest.raises(CellError, match="disagree on the label"):
        K.paired_items(a, bad)
    _, _, _, _, report = K.paired_items(a, x2.cells["neon_term_fits.baseline.anyjev_l2"])
    assert report == {
        "n_a": len(task.items),
        "n_b": len(task.items),
        "n_paired": len(task.items),
        "unmatched_a": 0,
        "unmatched_b": 0,
    }
    # files: one run per file, round trip, the bench table lists the verdicts
    ok = K.evaluate_k2(
        tiers=_results("tiers", []),
        x2=x2,
        x3=x3,
        date="2026-10-05",
        inputs=_inputs(),
        registered=True,
        B=B,
    )
    path = K.write_k2(ok, tmp_path)
    assert path == tmp_path / "2026-10-05" / "k2.json" and path.with_suffix(".md").is_file()
    assert K.load_k2(path) == ok
    with pytest.raises(FileExistsError):
        K.write_k2(ok, tmp_path)
    data = json.loads(path.read_text())
    assert data["format"] == K.FORMAT and data["tasks"]["neon_term_fits"]["verdict"] == "a"
    table = m2run.table([path], root=tmp_path)
    assert "2026-10-05/k2.json#neon_term_fits | probe |" in table and "**a**" in table
    assert "no cells" not in table
    # card_sign_margin_report: insufficient clusters below 4 counting cards; integer rule
    rep = K.card_sign_margin_report(
        np.array([[0.9, 0.1]] * 12),
        np.array([[0.2, 0.8]] * 12),
        np.zeros(12, dtype=int),
        ["c1"] * 12,
        0.02,
    )
    assert rep["reason"] == "insufficient_clusters" and not rep["passed"] and rep["m_c"] == 1
    assert rep["per_card"] == {
        "c1": {"n": 12, "correct_tier": 12, "correct_anyjev": 0, "wins_within_margin": True}
    }
    # exactly one wrong in fifty is Δacc = −0.02: not within the margin (strict), 1 in 51 is
    one_wrong = np.array([[0.9, 0.1]] * 49 + [[0.1, 0.9]])
    perfect = np.array([[0.9, 0.1]] * 50)
    row = K.card_sign_margin_report(one_wrong, perfect, np.zeros(50, dtype=int), ["c"] * 50)
    assert row["per_card"]["c"]["wins_within_margin"] is False
    row = K.card_sign_margin_report(
        np.vstack([one_wrong, [[0.9, 0.1]]]), np.vstack([perfect, [[0.9, 0.1]]]),
        np.zeros(51, dtype=int), ["c"] * 51,
    )  # fmt: skip
    assert row["per_card"]["c"]["wins_within_margin"] is True
