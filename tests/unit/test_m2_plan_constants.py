"""``design/m2-analysis-plan.md`` against the code: Appendix B ("every constant") is parsed row by
row and each value compared with the constant the code applies, every name in its "where" column
must resolve, and §1.3's published counts must be the registration's. The plan was committed
before the first run on real labels, so a change to either side fails here instead of letting
the code drift from what was registered. Reads the committed plan only; no data, no model."""

from __future__ import annotations

import importlib
import inspect
import math
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from mesa_clm.bench import _metrics, cells, framing, registered, stats, x2
from mesa_clm.bench import metrics as tie_metrics
from mesa_clm.bench.tasks import base as task_base
from mesa_clm.clm import headproj
from mesa_clm.clm.headproj import HeadProjector, random_head
from mesa_clm.learn import calibrate, offline

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "design" / "m2-analysis-plan.md"
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])")
# Formulas in a value cell whose digits are notation, not constants.
NOTATION = ("log(1/T)", 'sha256("task\\|framing\\|target")')


def _cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def appendix_b() -> dict[str, tuple[str, str]]:
    """``{constant: (value, where)}`` of the plan's Appendix B table, in order."""
    text = PLAN.read_text(encoding="utf-8")
    head = "## Appendix B: every constant"
    assert text.count(head) == 1, "the plan must have one Appendix B"
    rows: dict[str, tuple[str, str]] = {}
    for line in text.split(head, 1)[1].splitlines():
        if line.startswith("## "):
            break
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        name, value, where = _cells(line)
        if name == "constant":
            continue
        assert name not in rows, f"Appendix B lists {name!r} twice"
        rows[name] = (value, where)
    return rows


def numbers(value: str) -> list[float]:
    for fragment in NOTATION:
        value = value.replace(fragment, "")
    return [float(n) for n in NUMBER.findall(value)]


def ticked(value: str) -> list[str]:
    return re.findall(r"`([^`]+)`", value)


def _head_scale(logit_scale: float) -> float:
    h = random_head(0)
    return HeadProjector(h.cfg, h.state, h.action, logit_scale, "").scale


def _default(func: Callable[..., Any], name: str) -> Any:
    return inspect.signature(func).parameters[name].default


R = registered.REGISTERED
PLATT = calibrate.FITTER_CONSTANTS["platt"]
TEMPERATURE = calibrate.FITTER_CONSTANTS["temperature"]
assert isinstance(PLATT, dict) and isinstance(TEMPERATURE, dict)

# constant -> the numbers its value cell must list, in order (strings are compared below)
EXPECTED: dict[str, list[float]] = {
    "snapshot": [],
    "labels_sha256": [],
    "labels_content_sha256": [],
    "AnyJev L2 dump sha256": [],
    "framings lock sha": [],
    "serving lock sha": [],
    "model fingerprints": [],
    "bootstrap B, seed, α": [stats.DEFAULT_B, stats.DEFAULT_SEED, stats.DEFAULT_ALPHA],
    "rule R sign test": [stats.SIGN_MIN_ITEMS, stats.SIGN_FRACTION, stats.MIN_CLUSTERS],
    "rule (1)": [framing.AUROC_LOWER_GATE, framing.AUROC_FLOOR],
    "shuffle": [framing.SHUFFLE_K, framing.SHUFFLE_SEED],
    "guards": [task_base.MIN_TRAIN_PER_CLASS, task_base.MIN_HELDOUT_PER_CLASS],
    "floors": [framing.CALIBRATION_FLOOR, framing.PROBE_FLOOR],
    "Platt": [
        calibrate.PLATT_RIDGE,
        calibrate.PLATT_MAX_ITER,
        calibrate.PLATT_GTOL,
        calibrate.PLATT_ARMIJO,
        calibrate.PLATT_MAX_HALVINGS,
        calibrate.PLATT_QUADRATIC,
        *PLATT["start"],
    ],
    "temperature": [
        *calibrate.TEMPERATURE_BOUNDS,
        calibrate.TEMPERATURE_TOL,
        calibrate.TEMPERATURE_MAX_ITER,
    ],
    "scales": [headproj.MAX_SCALE, headproj.RAW_SCALE, framing.TEMPERATURE],
    "PR #13 replica, probe": [x2.REPLICA_C, x2.REPLICA_MAX_ITER],
    "ECE": [_default(tie_metrics.ece, "n_bins")],
    "threshold_cp": [
        stats.CP_MIN_N,
        round(100 * (1 - _default(stats.threshold_cp, "alpha"))),
        *stats.THRESHOLD_RISKS,
    ],
    "NLL clip": [1e-12],
    "recompute tolerance": [framing.REPRO_TOLERANCE, 1],
    "latency draw": [framing.LATENCY_TARGETS],
}


def test_appendix_b_lists_exactly_the_registered_constants() -> None:
    assert list(appendix_b()) == list(EXPECTED)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_appendix_b_value_is_the_codes(name: str) -> None:
    value, _ = appendix_b()[name]
    got = numbers(value) if EXPECTED[name] else []
    assert len(got) == len(EXPECTED[name]), (name, value, got)
    for g, want in zip(got, EXPECTED[name], strict=True):
        assert math.isclose(g, float(want), rel_tol=0, abs_tol=0), (name, value, g, want)


def test_the_appendix_b_strings_and_recipes_are_the_codes() -> None:
    rows = appendix_b()
    assert ticked(rows["snapshot"][0]) == [R.snapshot]
    assert ticked(rows["labels_sha256"][0]) == [R.labels_sha256]
    assert ticked(rows["labels_content_sha256"][0]) == [R.labels_content_sha256]
    assert ticked(rows["AnyJev L2 dump sha256"][0]) == [R.anyjev_sha256]
    # the framings and the models of G1 (§1.4-§1.5, §13.1)
    assert ticked(rows["framings lock sha"][0]) == [R.framings_lock_sha]
    assert ticked(rows["serving lock sha"][0]) == [R.serving_lock_sha]
    latest, raw = R.fingerprints["clm-latest"], R.fingerprints["clm-raw"]
    assert ticked(rows["model fingerprints"][0]) == [
        latest["encoder_fp"],
        latest["clm_model_fp"],
        raw["clm_model_fp"],
        latest["schema_sha256"],
    ]
    for fp in (latest, raw):
        assert (fp["encoder_fp"], fp["schema_sha256"], fp["serving_lock_sha"]) == (
            latest["encoder_fp"],
            latest["schema_sha256"],
            R.serving_lock_sha,
        )
    assert sorted(R.fingerprints) == sorted(framing.MODELS)
    assert "clm-latest first at even positions" in rows["latency draw"][0]
    # the registration's bootstrap is the statistics' default (one set of numbers)
    assert (R.B, R.seed, R.alpha) == (stats.DEFAULT_B, stats.DEFAULT_SEED, stats.DEFAULT_ALPHA)
    assert "one-sided" in rows["bootstrap B, seed, α"][0]
    assert "default_rng" in rows["bootstrap B, seed, α"][0]
    assert "rejection sampling" in rows["shuffle"][0]
    # the floors are one pair of numbers for X1 and the tier cells
    assert (framing.CALIBRATION_FLOOR, framing.PROBE_FLOOR) == (
        cells.CALIBRATED_FLOOR,
        cells.PROBE_FLOOR,
    )
    platt = rows["Platt"][0]
    assert platt.startswith(f"{calibrate.PLATT_TARGETS} targets")
    assert PLATT == {
        "implementation": "mesa_clm.learn.calibrate.fit_platt",
        "targets": calibrate.PLATT_TARGETS,
        "ridge": calibrate.PLATT_RIDGE,
        "max_iter": calibrate.PLATT_MAX_ITER,
        "gtol": calibrate.PLATT_GTOL,
        "armijo": calibrate.PLATT_ARMIJO,
        "max_halvings": calibrate.PLATT_MAX_HALVINGS,
        "quadratic": calibrate.PLATT_QUADRATIC,
        "start": [0.0, 0.0],
        "weights": "label weights as sample weights (D20)",
    }
    assert TEMPERATURE["bounds"] == list(calibrate.TEMPERATURE_BOUNDS)
    assert (TEMPERATURE["tol"], TEMPERATURE["max_iter"]) == (
        calibrate.TEMPERATURE_TOL,
        calibrate.TEMPERATURE_MAX_ITER,
    )
    assert "log(1/T)" in rows["temperature"][0]
    # every fitter applies its module constants by default
    assert _default(calibrate.fit_platt, "ridge") == calibrate.PLATT_RIDGE
    assert _default(calibrate.fit_platt, "max_iter") == calibrate.PLATT_MAX_ITER
    assert _default(calibrate.fit_platt, "gtol") == calibrate.PLATT_GTOL
    assert _default(calibrate.fit_temperature, "bounds") == calibrate.TEMPERATURE_BOUNDS
    # scales: the head's clamp, clm-raw's fixed scale, temperature 1
    assert _head_scale(math.log(1000.0)) == headproj.MAX_SCALE
    assert _head_scale(math.log(50.0)) == pytest.approx(50.0)
    assert offline.OfflineScorer(offline.RAW_MODEL).scale == headproj.RAW_SCALE
    # the PR #13 replica (and the probe, which uses it): exactly this recipe
    pipe = x2.replica_pipeline()
    assert [name for name, _ in pipe.steps] == ["standardscaler", "logisticregression"]
    lr = pipe[-1]
    assert (lr.C, lr.max_iter) == (x2.REPLICA_C, x2.REPLICA_MAX_ITER)
    assert (
        "StandardScaler + LogisticRegression(C=1, max_iter=2000), unweighted"
        in (rows["PR #13 replica, probe"][0])
    )
    assert framing.PROBE_RECIPE.startswith(x2.REPLICA_RECIPE.split(", unweighted")[0])
    # threshold_cp's risks and minimum; the NLL clip of the vendored metrics
    assert _default(stats.thresholds_cp, "risks") == stats.THRESHOLD_RISKS
    assert _default(_metrics.ece, "n_bins") == _default(tie_metrics.ece, "n_bins")
    clipped = _metrics.nll(np.array([[0.0, 1.0]]), [0])
    assert clipped == pytest.approx(-math.log(1e-12))
    assert _default(framing.decide_from_json, "tolerance") == framing.REPRO_TOLERANCE
    assert _default(framing.latency_targets, "n") == framing.LATENCY_TARGETS


MODULES: dict[str, str] = {
    "bench.registered": "mesa_clm.bench.registered",
    "stats": "mesa_clm.bench.stats",
    "framing": "mesa_clm.bench.framing",
    "bench.tasks.base": "mesa_clm.bench.tasks.base",
    "cells": "mesa_clm.bench.cells",
    "calibrate": "mesa_clm.learn.calibrate",
    "offline": "mesa_clm.learn.offline",
    "x2": "mesa_clm.bench.x2",
    "bench.metrics": "mesa_clm.bench.metrics",
}


def test_every_name_in_the_where_column_exists() -> None:
    """``framing.SHUFFLE_K`` names an attribute of ``mesa_clm.bench.framing``; a bare name is in
    the module named before it in the same cell; ``same`` repeats the row above; ``X_*`` needs
    at least one attribute with that prefix."""
    module: ModuleType | None = None
    resolved = 0
    for name, (_, where) in appendix_b().items():
        if where == "same":
            assert module is not None, name
            continue
        for token in ticked(where):
            if token in MODULES:
                module = importlib.import_module(MODULES[token])
                resolved += 1
                continue
            prefix, _, attr = token.rpartition(".")
            if prefix:
                assert prefix in MODULES, (name, token)
                module = importlib.import_module(MODULES[prefix])
            assert module is not None, (name, token)
            if attr.endswith("*"):
                assert any(a.startswith(attr[:-1]) for a in dir(module)), (name, token)
            else:
                assert hasattr(module, attr), (name, token)
            resolved += 1
    assert resolved >= 20


def test_section_1_3_counts_are_the_registrations() -> None:
    """§1.3 quotes the published counts the verbs check before any statistic; they are the
    registration's (which ``test_bench_registered`` checks against ``baselines.json``)."""
    text = PLAN.read_text(encoding="utf-8")
    section = text.split("1.3 **Counts, before any statistic.**", 1)[1].split("\n\n", 1)[0]
    flat = " ".join(section.split())
    c = R.counts

    def per_card(name: str) -> str:
        return ", ".join(str(v) for _, v in sorted(c[name].per_card.items()))

    term, ont = c["neon_term_fits"], c["neon_ontology_fits"]
    ann, asp, vk = c["neon_annotate"], c["neon_aspect"], c["neon_value_kind"]
    for phrase in (
        f"term.fits {term.n} items ({term.class_counts[0]} Yes / {term.class_counts[1]} No), "
        f"per card {per_card('neon_term_fits')}",
        f"column.ontology_fits {ont.n} ({ont.class_counts[0]} / {ont.class_counts[1]}), "
        f"per card {per_card('neon_ontology_fits')}",
        f"column.annotate {ann.n} ({ann.class_counts[0]} / {ann.class_counts[1]})",
        f"column.aspect {asp.n} (classes 0–5: "
        + ", ".join(str(asp.class_counts[k]) for k in range(6))
        + ")",
        f"avu.value_kind {vk.n} (" + ", ".join(str(vk.class_counts[k]) for k in range(4)) + ")",
    ):
        assert phrase in flat, phrase
    cards = flat.split("cards in sorted order: ", 1)[1].split(")", 1)[0].split(", ")
    assert cards == [k.rsplit(".", 1)[1] for k in sorted(term.per_card)]
    assert {t.min_weight for t in c.values()} == {0.5, 0.6} and asp.min_weight == 0.6
