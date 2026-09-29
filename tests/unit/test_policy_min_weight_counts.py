"""DESIGN D9: ``policy_defaults.yaml`` is the one source of ``min_weight``, the bench reads it
too, and each bench task's class counts at the policy weight equal the ones in the published
cell (``bench/results/2026-09-29/baselines.json``). mesa-anyjev kept 0.6 in the policy and 0.5
in the bench, so its cells were fitted on rows the policy would never have used."""

from __future__ import annotations

from pathlib import Path

import pytest

from mesa_clm.bench.results import load_results
from mesa_clm.bench.tasks.neon import NEON_TASKS, tasks_from_store
from mesa_clm.learn.labels import TermResolver, ingest_neon_eval, labelled_targets
from mesa_clm.ols import RecordingOLS
from mesa_clm.policy_defaults import load_policy_defaults, min_weight_for
from mesa_clm.provenance.labels import LabelStore
from mesa_clm.tasks import TASKS

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
PUBLISHED = REPO / "bench" / "results" / "2026-09-29" / "baselines.json"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> LabelStore:
    st = LabelStore(tmp_path_factory.mktemp("labels") / "labels.duckdb")
    report = ingest_neon_eval(
        st, FIXTURES / "neon-avu-eval", TermResolver(RecordingOLS(None, FIXTURES / "ols", "replay"))
    )
    assert report.inserted == 934
    return st


def test_policy_min_weight_counts(store: LabelStore) -> None:
    """Each bench task's class counts at the policy ``min_weight`` equal the published cell's."""
    assert PUBLISHED.exists(), "the M0 baselines cell must be committed"
    published = load_results(PUBLISHED)
    policy = load_policy_defaults()
    tasks = tasks_from_store(store)
    assert set(tasks) == set(NEON_TASKS)
    for name, task_id in NEON_TASKS.items():
        cell = published.cells[f"{name}.baseline.lookup_prob"]
        min_weight = min_weight_for(task_id)
        assert cell.task_id == task_id and cell.task_key == TASKS[task_id].key
        assert cell.min_weight == min_weight == policy.min_weight(task_id), task_id
        counts = labelled_targets(store, task_id, min_weight=min_weight).class_counts()
        assert counts == cell.counts.class_counts, task_id
        assert tasks[name].class_counts() == cell.counts.class_counts, name
        assert tasks[name].meta["min_weight"] == min_weight
        assert cell.counts.n == sum(counts.values())
        if TASKS[task_id].kind == "noul":
            assert cell.counts.n_neg == counts.get(1, 0)
        else:
            assert cell.counts.n_neg is None
        assert cell.counts.n_nonmodal == cell.counts.n - max(counts.values())


def test_published_cells_carry_the_published_counts() -> None:
    """The counts the plan and DESIGN D9 quote, read from the committed cell."""
    published = load_results(PUBLISHED)
    counts = {k.split(".")[0]: c.counts.class_counts for k, c in published.cells.items()}
    assert counts["neon_term_fits"] == {0: 86, 1: 199}
    assert counts["neon_ontology_fits"] == {0: 76, 1: 114}
    assert counts["neon_annotate"] == {0: 63, 1: 35}
    assert counts["neon_aspect"] == {0: 12, 1: 10, 2: 18, 3: 11, 4: 6, 5: 3}
    assert counts["neon_value_kind"] == {0: 133, 1: 62, 2: 24, 3: 59}
    weights = {k.split(".")[0]: c.min_weight for k, c in published.cells.items()}
    assert weights == {
        "neon_term_fits": 0.5,
        "neon_ontology_fits": 0.5,
        "neon_annotate": 0.5,
        "neon_aspect": 0.6,
        "neon_value_kind": 0.5,
    }


def test_the_a2_trap_at_0_6_is_what_d9_avoids(store: LabelStore) -> None:
    """At 0.6 the consensus negatives vanish: no No for term.fits, annotate and ontology_fits,
    no class 3 for value_kind (DESIGN D9 rationale)."""
    assert set(labelled_targets(store, "term.fits", min_weight=0.6).labels) == {0}
    assert labelled_targets(store, "column.annotate", min_weight=0.6).class_counts() == {0: 63}
    assert labelled_targets(store, "column.ontology_fits", min_weight=0.6).class_counts() == {0: 76}
    assert 3 not in labelled_targets(store, "avu.value_kind", min_weight=0.6).class_counts()
