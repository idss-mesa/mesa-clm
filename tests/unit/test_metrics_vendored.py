"""The vendored AnyJev metrics are byte-identical to upstream (THIRD_PARTY.md,
``vendored.sha256``) and work. Ported from mesa-anyjev ``tests/test_metrics_vendored.py``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mesa_clm.bench import _metrics

UPSTREAM_SHA256 = "df0af62c248f27cc806c90266ae62c28b47df8dd8aa9645de815dedc50d7bbbe"
REPO = Path(_metrics.__file__).resolve().parents[3]


def test_sha256_matches_upstream() -> None:
    assert hashlib.sha256(Path(_metrics.__file__).read_bytes()).hexdigest() == UPSTREAM_SHA256


def test_sha256_is_recorded_in_third_party_md() -> None:
    third_party = REPO / "THIRD_PARTY.md"
    if not third_party.exists():
        pytest.skip("THIRD_PARTY.md is written by another M0 task")
    assert UPSTREAM_SHA256 in third_party.read_text(encoding="utf-8")


def test_sha256_is_pinned_in_vendored_sha256() -> None:
    pinned = REPO / "vendored.sha256"
    if not pinned.exists():
        pytest.skip("vendored.sha256 is written by another M0 task")
    lines = [ln.split() for ln in pinned.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [UPSTREAM_SHA256, "src/mesa_clm/bench/_metrics.py"] in lines


def test_summarize_runs() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]] * 5)
    labels = [0, 1, 0, 1] * 5
    out = _metrics.summarize(probs, labels)
    assert out["acc"] == 1.0
    assert 0.0 <= out["ece"] <= 1.0
    assert set(out) == {"n", "acc", "macro_f1", "brier", "nll", "ece", "cov@5%", "aurc"}
    assert _metrics.coverage_at_risk(probs, labels, target=0.10) == 1.0
    flipped = _metrics.summarize(probs, labels, probs_flipped=probs[:, ::-1])
    assert flipped["flip"] == 1.0
