"""The minimum-detectable-effect simulation (plan §5.4 ``bench mde``): the binormal model,
populations from label counts only, rule R's power along an AUROC grid, deterministic seeds,
the JSON file, and the committed ``bench/results/2026-09-29/mde.json``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mesa_clm.bench import stats
from mesa_clm.bench.mde import (
    DEFAULT_GRID,
    DEFAULT_N_SIMS,
    FORMAT,
    METRICS,
    POWER_TARGET,
    MdeResults,
    Population,
    load_mde,
    mde_curves,
    norm_cdf,
    norm_ppf,
    populations_for,
    power_at,
    run_mde,
    separation_for_auroc,
    simulate_arms,
    write_mde,
)
from mesa_clm.bench.tasks.base import Task
from mesa_clm.bench.tasks.neon import tasks_from_store
from mesa_clm.learn.labels import TermResolver, ingest_neon_eval
from mesa_clm.ols import RecordingOLS
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.tasks import TASKS as SPECS

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
COMMITTED = REPO / "bench" / "results" / "2026-09-29" / "mde.json"


def _toy(per_card: dict[str, tuple[int, int]]) -> Task:
    items = []
    cards: list[str] = []
    opts: list[str] = []
    for card, (pos, neg) in sorted(per_card.items()):
        for i in range(pos + neg):
            state = {
                "card": {"dataset": card},
                "scope": "column",
                "column": {"name": f"c{i}"},
                "candidate": {"curie": f"X:{i}"},
            }
            items.append((state, 0 if i < pos else 1))
            cards.append(card)
            opts.append(f"X:{i}")
    return Task("toy", SPECS["term.fits"], items, "l", "s", cards=cards, option_keys=opts)


def test_normal_helpers_and_separation() -> None:
    assert norm_cdf(0.0) == 0.5 and norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-9)
    for q in (0.01, 0.3, 0.5, 0.8, 0.999):
        assert norm_cdf(norm_ppf(q)) == pytest.approx(q, abs=1e-10)
    with pytest.raises(ValueError):
        norm_ppf(1.0)
    assert separation_for_auroc(0.5) == 0.0
    for a in (0.55, 0.7, 0.9):
        d = separation_for_auroc(a)
        assert norm_cdf(d / math.sqrt(2)) == pytest.approx(a, abs=1e-9)
    with pytest.raises(ValueError):
        separation_for_auroc(0.4)


def test_population_from_task_counts_labels_only() -> None:
    toy = _toy({"B": (3, 9), "A": (2, 10), "C": (0, 4)})
    pop = Population.from_task(toy)
    assert pop.cards == ("A", "B", "C") and pop.n_pos == (2, 3, 0) and pop.n_neg == (10, 9, 4)
    assert (
        pop.n == 28
        and pop.base_rate == pytest.approx(5 / 28)
        and pop.name == "full"
        and pop.task == "toy"
    )
    labels = pop.labels()
    assert labels.tolist() == [0, 0] + [1] * 10 + [0] * 3 + [1] * 9 + [1] * 4
    assert pop.clusters() == ["A"] * 12 + ["B"] * 12 + ["C"] * 4
    assert pop.per_card()["C"] == {"n_pos": 0, "n_neg": 4} and pop.cards_counting() == 2
    sub = Population.from_task(toy, [0, 1, 2, 12, 13, 14, 15], name="novel_key")
    assert (
        sub.cards == ("A", "B")
        and sub.n_pos == (2, 3)
        and sub.n_neg == (1, 1)
        and sub.name == "novel_key"
    )
    choice = Task("a", SPECS["column.aspect"], [(toy.items[0][0], 1)], "l", "s", cards=["A"])
    with pytest.raises(ValueError, match="two-class"):
        Population.from_task(choice)
    with pytest.raises(ValueError, match="one card per item"):
        Population.from_task(Task("t", SPECS["term.fits"], toy.items[:2], "l", "s"))
    empty = Population("e", "t", (), (), ())
    assert empty.n == 0 and math.isnan(empty.base_rate) and empty.labels().shape == (0,)


def test_simulate_arms_is_calibrated_and_separates() -> None:
    labels = np.repeat([0, 1], [400, 1600])
    rng = np.random.default_rng(0)
    a0, b0 = simulate_arms(labels, 0.0, 0.2, rng)
    assert a0.shape == b0.shape == (2000, 2)
    np.testing.assert_allclose(a0.sum(axis=1), 1.0)
    np.testing.assert_allclose(a0[:, 0], 0.2)  # no separation: the prior everywhere
    np.testing.assert_allclose(b0, np.tile([0.2, 0.8], (2000, 1)))
    d = separation_for_auroc(0.8)
    a1, _ = simulate_arms(labels, d, 0.2, np.random.default_rng(1))
    assert stats.auroc(a1[:, 0], labels) == pytest.approx(0.8, abs=0.03)
    assert float(a1[:, 0].mean()) == pytest.approx(0.2, abs=0.02)  # calibrated on average
    assert stats.metric_value("nll", a1, labels) < stats.metric_value("nll", b0, labels)
    assert stats.metric_value("ece", a1, labels) < 0.05


def test_power_rises_with_the_effect_and_is_deterministic() -> None:
    pop = Population.from_task(_toy({f"c{i}": (12, 28) for i in range(7)}))
    weak = power_at(pop, 0.55, n_sims=12, B=200)
    strong = power_at(pop, 0.9, n_sims=12, B=200)
    assert set(weak) == set(METRICS) == {"auroc", "nll"}
    for m in METRICS:
        assert weak[m].power <= strong[m].power and strong[m].power == 1.0
        assert sum(weak[m].reasons.values()) == weak[m].n_sims == 12 and weak[m].passed <= 12
        assert weak[m].separation == pytest.approx(separation_for_auroc(0.55))
    assert strong["auroc"].delta_mean == pytest.approx(0.4, abs=0.03)
    assert strong["nll"].delta_mean is not None and strong["nll"].delta_mean > 0.2
    again = power_at(pop, 0.9, n_sims=12, B=200)
    assert again == strong
    other = power_at(pop, 0.9, n_sims=12, B=200, seed=1)
    assert other["auroc"].delta_mean != strong["auroc"].delta_mean
    w = stats.bootstrap_weights(pop.clusters(), B=200, seed=0)
    assert power_at(pop, 0.9, n_sims=12, B=200, weights=w) == strong


def test_mde_curves_read_off_the_grid() -> None:
    pop = Population.from_task(_toy({f"c{i}": (12, 28) for i in range(7)}))
    curves = mde_curves(pop, grid=(0.5, 0.95), n_sims=8, B=100)
    assert [c.metric for c in curves] == ["auroc", "nll"]
    for c in curves:
        assert c.population == "full" and c.task == "toy" and c.n == 280 and c.n_cards == 7
        assert c.n_pos == 84 and c.n_neg == 196 and c.base_rate == pytest.approx(0.3)
        assert c.cards_counting == 7 and [p.auroc for p in c.grid] == [0.5, 0.95]
        assert c.grid[0].power < c.power_target <= c.grid[1].power
        assert c.mde_auroc == 0.95 and c.not_applicable is None
    assert curves[0].mde_delta == pytest.approx(0.45)
    assert curves[1].mde_delta == curves[1].grid[1].delta_mean
    nothing = mde_curves(pop, grid=(0.5,), n_sims=8, B=100)
    assert all(c.mde_auroc is None and c.mde_delta is None for c in nothing)
    one_class = Population.from_task(_toy({"a": (0, 20), "b": (0, 20)}))
    na = mde_curves(one_class, grid=(0.6,), n_sims=2, B=10)
    assert all(c.grid == [] and c.not_applicable for c in na)


@pytest.fixture(scope="module")
def tasks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Task]:
    store = LabelStore(tmp_path_factory.mktemp("labels") / "labels.duckdb")
    ingest_neon_eval(
        store,
        FIXTURES / "neon-avu-eval",
        TermResolver(RecordingOLS(None, FIXTURES / "ols", "replay")),
    )
    return tasks_from_store(store)


def test_populations_for_uses_the_novel_key_subset(tasks: dict[str, Task]) -> None:
    full, novel = populations_for(tasks["neon_term_fits"])
    assert (
        full.name == "full"
        and full.n == 285
        and sum(full.n_pos) == 86
        and full.cards_counting() == 7
    )
    assert (
        novel.name == "novel_key"
        and novel.n == 159
        and sum(novel.n_pos) == 45
        and sum(novel.n_neg) == 114
    )
    _, novel_ont = populations_for(tasks["neon_ontology_fits"])
    assert novel_ont.n == 99 and sum(novel_ont.n_pos) == 33 and novel_ont.cards_counting() == 4


def test_run_mde_and_file_round_trip(tasks: dict[str, Task], tmp_path: Path) -> None:
    res = run_mde(tasks, labels_sha256="0" * 64, date="2026-01-01", grid=(0.6, 0.9), n_sims=3, B=50)
    assert (
        res.format == FORMAT
        and res.n_sims == 3
        and res.B == 50
        and res.seed == 0
        and res.grid == [0.6, 0.9]
    )
    assert "rule R" in res.rule and res.labels_sha256 == "0" * 64
    binary = [c for c in res.curves if c.not_applicable is None]
    na = [c for c in res.curves if c.not_applicable]
    assert (
        len(binary) == 3 * 2 * 2 and len(na) == 2
    )  # (term, ontology, annotate) x (full, novel) x 2 metrics
    assert {c.task for c in na} == {"neon_aspect", "neon_value_kind"} and all(
        "classes" in (c.not_applicable or "") for c in na
    )
    assert {(c.task, c.population) for c in binary} >= {
        ("neon_term_fits", "novel_key"),
        ("neon_annotate", "full"),
    }
    path = write_mde(res, tmp_path)
    assert path == tmp_path / "2026-01-01" / "mde.json"
    assert load_mde(path) == res
    assert MdeResults.model_validate(res.model_dump()) == res


def test_committed_mde_file_matches_the_fixture_populations(tasks: dict[str, Task]) -> None:
    """The committed MDE was simulated from these label counts (nothing else): its populations
    equal the fixture tasks', it used the pre-registered settings, and it names an MDE."""
    assert COMMITTED.exists(), "bench/results/2026-09-29/mde.json is committed with M0"
    res = load_mde(COMMITTED)
    assert res.format == FORMAT and res.date == "2026-09-29" and res.name == "mde"
    assert res.n_sims == DEFAULT_N_SIMS == 200 and res.B == 2000 and res.seed == 0
    assert res.grid == list(DEFAULT_GRID) and res.grid[0] == 0.55 and res.grid[-1] == 0.85
    by = {(c.task, c.population, c.metric): c for c in res.curves}
    for name in ("neon_term_fits", "neon_ontology_fits", "neon_annotate"):
        for pop in populations_for(tasks[name]):
            for metric in METRICS:
                c = by[(name, pop.name, metric)]
                assert c.n == pop.n and c.n_pos == sum(pop.n_pos) and c.n_neg == sum(pop.n_neg)
                assert c.per_card == pop.per_card() and c.cards_counting == pop.cards_counting()
                assert c.power_target == POWER_TARGET and len(c.grid) == len(res.grid)
                powers = [p.power for p in c.grid]
                assert all(0.0 <= p <= 1.0 for p in powers) and powers[-1] >= powers[0]
                assert all(sum(p.reasons.values()) == 200 for p in c.grid)
    term_full = by[("neon_term_fits", "full", "auroc")]
    assert term_full.mde_auroc is not None and 0.55 <= term_full.mde_auroc <= 0.75
    assert term_full.mde_delta == pytest.approx(term_full.mde_auroc - 0.5)
    term_nll = by[("neon_term_fits", "full", "nll")]
    assert term_nll.mde_auroc is not None and term_nll.mde_auroc >= term_full.mde_auroc
    assert term_nll.mde_delta is not None and term_nll.mde_delta > 0
    # the novel-key gate is harder than the full set (fewer items, fewer counting cards)
    novel = by[("neon_term_fits", "novel_key", "auroc")]
    assert novel.mde_auroc is not None and novel.mde_auroc >= term_full.mde_auroc
    assert {c.task for c in res.curves if c.not_applicable} == {"neon_aspect", "neon_value_kind"}
