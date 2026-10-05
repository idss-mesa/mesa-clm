"""Synthetic M4 fixtures for the serving tests (plan §5.5, §4.7; the M4 brief R4, R6): a probe
artifact fitted on random features under the fake stack's fingerprint, a bench cell with every
frozen field and planted per-item predictions, a results file under a temporary results root,
a snapshot file that exists only to be hashed, and a ``k2.json`` shaped like
:mod:`mesa_clm.bench.k2`'s. No real label, vector or results file is read anywhere here."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mesa_clm import framings as fr
from mesa_clm.bench.results import (
    BenchCell,
    BenchResults,
    CellBaselines,
    CellCounts,
    CellItem,
    CellMetrics,
    ClusterCI,
    Lopo,
    NovelKey,
    environment,
    write_results,
)
from mesa_clm.bench.tasks.neon import MASKS
from mesa_clm.clm.fingerprint import Fingerprint
from mesa_clm.learn import calibrate
from mesa_clm.learn.linear import fit_linear
from mesa_clm.learn.probe import ProbeArtifact, calibrator_input
from mesa_clm.providers.tiered import SPEC_MODELS, fake_fingerprint, spec_dim
from mesa_clm.tasks import TASKS

FP = fake_fingerprint()
LABELS_SHA = "a" * 64
CONTENT_SHA = "b" * 64
CARDS = [f"DP9.{i:05d}.001.card_{i}" for i in range(7)]
BENCH_TASK = {
    "term.fits": "neon_term_fits",
    "column.ontology_fits": "neon_ontology_fits",
    "column.annotate": "neon_annotate",
    "column.aspect": "neon_aspect",
    "avu.value_kind": "neon_value_kind",
}


def synthetic_probe(
    spec: str = "lowdim.v1",
    *,
    task_id: str = "term.fits",
    fp: Fingerprint = FP,
    seed: int = 0,
    n: int = 80,
    labels_sha256: str = LABELS_SHA,
    labels_content_sha256: str = CONTENT_SHA,
) -> ProbeArtifact:
    """A :class:`ProbeArtifact` of ``spec`` for ``task_id``'s active framing: a logistic
    regression fitted on random features with planted labels and its calibrator (Platt on the
    logit difference at K = 2, temperature above), under ``fp``."""
    framing = fr.active_framing(task_id)
    k = TASKS[task_id].k
    rng = np.random.default_rng(seed)
    d = spec_dim(spec, k=k)
    x = rng.standard_normal((n, d))
    if k == 2:
        y = (x[:, 0] + 0.5 * rng.standard_normal(n) < 0).astype(np.int64)
    else:
        y = np.argmax(x[:, :k] + 0.5 * rng.standard_normal((n, k)), axis=1).astype(np.int64)
    linear = fit_linear("logreg", x, y, None, k=k, hyper={"lambda": 1.0})
    cal = calibrate.fit_calibrator("choice", calibrator_input(linear.scores(x), k), y, None)
    return ProbeArtifact(
        question_key=framing.question_key,
        task_id=task_id,
        framing_id=framing.id,
        spec=spec,
        model=SPEC_MODELS[spec],
        fingerprint=fp.as_dict(),
        linear=linear,
        calibrator=cal,
        selection={"inner_nll": 0.5, "inner_n": n, "inner_skips": {}},
        labels_sha256=labels_sha256,
        labels_content_sha256=labels_content_sha256,
        n_train=n,
    )


def _ci(point: float | None) -> ClusterCI | None:
    if point is None:
        return None
    return ClusterCI(
        point=point, lower=point - 0.05, upper=point + 0.05, B=2000, seed=0, alpha=0.05,
        n_clusters=7, n_valid=2000,
    )  # fmt: skip


def _novel(n: int, binary: bool) -> NovelKey:
    return NovelKey(
        n=n,
        class_counts={0: n // 2, 1: n - n // 2},
        majority_acc=0.6,
        acc=0.7,
        auroc=0.8 if binary else None,
        auroc_ci=_ci(0.8) if binary else None,
        nll=0.5,
        brier=0.2,
    )


def planted_items(
    *,
    k: int = 2,
    cards: Sequence[str] = CARDS,
    per_card: int = 12,
    p_correct: float = 0.8,
    seed: int = 0,
) -> list[CellItem]:
    """``per_card`` items per card with the label's probability ``p_correct`` (the rest spread
    evenly), labels alternating so every card has both classes; ``option_key`` distinct."""
    rng = np.random.default_rng(seed)
    items: list[CellItem] = []
    for c, card in enumerate(cards):
        for i in range(per_card):
            label = (i + c) % k
            probs = [(1.0 - p_correct) / (k - 1)] * k
            probs[label] = p_correct
            jitter = rng.uniform(-0.02, 0.02)
            probs = [
                max(p + (jitter if j == label else -jitter / (k - 1)), 1e-3)
                for j, p in enumerate(probs)
            ]
            total = sum(probs)
            items.append(
                CellItem(
                    target_sha256=hashlib.sha256(f"{card}/{i}".encode()).hexdigest(),
                    option_key=f"X:{c:02d}{i:03d}" if k == 2 else "",
                    card=card,
                    label=label,
                    probs=[p / total for p in probs],
                    novel=(i % 3 == 0),
                )
            )
    return items


def fold_choices(
    *,
    cards: Sequence[str] = CARDS,
    tier: str = "calibrated",
    framing: str = "F7",
    model: str = "clm-latest",
    spec: str | None = None,
    agree: int | None = None,
) -> dict[str, Any]:
    """Seven outer folds choosing ``(framing, model[, spec])``; ``agree`` folds agree, the rest
    chose another arm/spec."""
    out: dict[str, Any] = {}
    n_agree = len(cards) if agree is None else agree
    for i, card in enumerate(cards):
        ok = i < n_agree
        entry: dict[str, Any] = {"decision": "arm", "evaluated": True}
        if tier == "probe":
            entry.update(
                spec=spec if ok else "pair4096.v1",
                fitter="logreg",
                hyper={"lambda": 1.0},
                model=model if ok else "clm-raw",
                clm_model_fp="0" * 12,
                inner_nll=0.5,
            )
        else:
            entry.update(
                framing=framing if ok else "F4",
                model=model if ok else "clm-raw",
                question_key="0" * 16,
                clm_model_fp="0" * 12,
            )
        out[card] = entry
    return out


def cell(
    task_id: str = "term.fits",
    tier: str = "calibrated",
    *,
    fp: Fingerprint | Mapping[str, str] = FP,
    labels_sha256: str = LABELS_SHA,
    labels_content_sha256: str | None = CONTENT_SHA,
    model: str = "clm-latest",
    spec: str | None = None,
    items: Sequence[CellItem] | None = None,
    threshold_cp: Mapping[str, float | None] | None = None,
    choices: Mapping[str, Any] | None = None,
    n_folds: int = 7,
    **over: Any,
) -> BenchCell:
    """A nested, pre-registered, servable LOCO cell of ``task_id``'s active framing with every
    frozen field, planted items and ``over`` applied last."""
    task = TASKS[task_id]
    framing = fr.active_framing(task_id)
    k = task.k
    rows = list(items) if items is not None else planted_items(k=k)
    n = len(rows)
    binary = k == 2
    fingerprint = fp.as_dict() if isinstance(fp, Fingerprint) else dict(fp)
    fields: dict[str, Any] = {
        "task": BENCH_TASK[task_id],
        "task_id": task_id,
        "task_key": task.key,
        "tier": tier,
        "framing": framing.id,
        "question_key": framing.question_key,
        "fingerprint": fingerprint,
        "labels_sha256": labels_sha256,
        "labels_content_sha256": labels_content_sha256,
        "feature_spec": spec,
        "label_sources": {"consensus_majority": n},
        "min_weight": 0.5,
        "teacher": False,
        "teacher_in_test": False,
        # As ``bench.cells.assemble_cell`` records them from ``bench.tasks.neon`` (every cell
        # ``masked: true``; ``mask`` the task's option restriction, "aspect" or None).
        "masked": True,
        "mask": MASKS.get(task_id),
        "loco": True,
        "selection": "nested",
        "pre_registered": True,
        "exploratory": False,
        "servable": True,
        "n_folds": n_folds,
        "skipped_folds": {},
        "guard_skipped_folds": {},
        "fold_choices": dict(choices)
        if choices is not None
        else fold_choices(tier=tier, framing=framing.id, model=model, spec=spec),
        "counts": CellCounts(
            n=n,
            n_neg=sum(1 for r in rows if r.label == 1) if binary else None,
            n_nonmodal=n - max(sum(1 for r in rows if r.label == c) for c in range(k)),
            class_counts={c: sum(1 for r in rows if r.label == c) for c in range(k)},
        ),
        "metrics": CellMetrics(
            acc=0.8,
            macro_f1=0.75,
            brier=0.2,
            nll=0.5,
            ece=0.05,
            **{"cov@5%": 0.3, "cov@10%": 0.5},
            aurc=0.1,
            auroc=0.85 if binary else None,
            auroc_ci=_ci(0.85) if binary else None,
            threshold_cp=dict(threshold_cp)
            if threshold_cp is not None
            else {"0.05": 0.9, "0.10": 0.8},
        ),
        "baselines": CellBaselines(
            lookup_key="(task, scope, target, option_key)",
            lookup_rule="most common label among the training cards",
            majority_acc=0.6,
            lookup_acc=0.7,
            lookup_nll=0.6,
            lookup_prob_acc=0.7,
            novel_key=_novel(n // 3, binary),
            lopo=Lopo(n_folds=2, n=n, acc=0.7, nll=0.6, novel_n=n // 3, majority_acc=0.6),
            beats_lookup_novel=True,
            novel_key_lookup=_novel(n // 3, binary),
            beats_detail={"reason": "passed"},
        ),
        "diagnostics": {},
        "notes": "synthetic",
        "model": model,
        "variant": None,
        "items": rows,
    }
    fields.update(over)
    return BenchCell.model_validate(fields)


def results_file(
    root: Path, date: str, name: str, cells: Sequence[BenchCell], *, labels_sha256: str = LABELS_SHA
) -> str:
    """Write ``<root>/bench/results/<date>/<name>.json`` and return its cite path."""
    results = BenchResults(
        date=date,
        name=name,
        mesa_clm=environment()["mesa_clm"],
        labels_sha256=labels_sha256,
        labels_content_sha256=CONTENT_SHA,
        environment=environment(),
        cells={c.key: c for c in cells},
    )
    write_results(results, root / "bench" / "results", force=True)
    return f"bench/results/{date}/{name}.json"


def snapshot_file(root: Path, content: bytes = b"synthetic parquet bytes") -> tuple[Path, str]:
    """A file under ``<root>/bench/snapshots/`` that exists only to be hashed (D30)."""
    directory = root / "bench" / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "2026-09-29.parquet"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def k2_json(verdicts: Mapping[str, tuple[str, str]]) -> dict[str, Any]:
    """``k2.json`` as :mod:`mesa_clm.bench.k2` writes it: ``tasks[task_id]`` with ``best`` and
    ``verdict`` (``verdicts``: task_id -> (best tier, verdict))."""
    return {
        "format": "mesa-clm/bench-results/1",
        "name": "k2",
        "tasks": {
            task_id: {"task_id": task_id, "best": best, "verdict": verdict}
            for task_id, (best, verdict) in verdicts.items()
        },
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path
