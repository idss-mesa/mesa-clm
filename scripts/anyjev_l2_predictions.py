#!/usr/bin/env python3
"""Dump mesa-anyjev's per-item leave-one-card-out L2 predictions for the paired comparison
(mesa-clm plan §5.6 X2, K2(a); DESIGN D8).

Run with **mesa-anyjev's own interpreter on the GPU host**, never mesa-clm's (mesa-clm imports
nothing from ``anyjev``/``mesa_anyjev``, DESIGN U1), in a window where the vLLM pooling
container is stopped (Qwen3-8B bf16 needs the GPU to itself, plan M1-A). With ``MESA_CLM`` the
mesa-clm checkout::

    cd <mesa-anyjev checkout>
    .venv/bin/python $MESA_CLM/scripts/anyjev_l2_predictions.py \\
        --provenance duckdb:///$PWD/.local/labels.duckdb \\
        --out $MESA_CLM/bench/results/<date>/anyjev_l2_predictions.json

It reproduces ``mesa_anyjev.bench.run._fit_eval_loco`` for ``neon_term_fits`` and
``neon_ontology_fits`` at level ``L2`` step for step: the tasks come from ``tasks_from_store``
(labels at the bench ``min_weight`` 0.5, highest weight per state), each card is held out in
sorted order, the 30/5 fold guards are applied, a fresh fitting Decider (``fit_decider``:
``adaptive_shifts=False``) gets ``fit_on(..., "L2")`` (``Decider.fit_head`` with
``listing="auto"``, ``n_folds=5``, ``seed=0``, the block and head kind chosen by out-of-fold NLL)
on the training states, the held-out states are decided at L2 and turned into records with
``provider.record_from``. Where the bench only pools the records into metrics, this script keeps
every record: card, ``state_sha256``, the target identity (scope, column or site, aspect, the
candidate CURIE or registry id), the label, ``p_yes`` (``probs[0]``, AnyJev's ``p_true``) and the
diagnostics, plus the pooled and per-fold cells so the dump can be checked against the committed
``bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json`` (term.fits 0.765 / ontology_fits 0.837).

The whole ``state_json`` is stored per item unless ``--no-states`` is given, so mesa-clm can
derive ``target_sha256`` / ``option_key`` (its D1 identity) at import time instead of this
script re-implementing them. Nothing is written to the mesa-anyjev store (opened read-only).

Feasibility (M0 read of ``bench/run.py`` and ``learn/fit.py``): per-item predictions are
obtainable because ``_fit_eval_loco`` builds the ``DecisionRecord`` list per fold before
pooling; this script is that loop with the records kept. Cost: the committed L2 cells report
391.8 s for term.fits (285 items, 7 folds, 1374.7 ms per decision including the seven
``fit_head`` feature passes) and a comparable ~260 s for ontology_fits (190 items), so expect
about 12–15 minutes plus model load on the GB10 with the GPU free; the fits are deterministic
(``seed=0``) given the same weights and hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

try:
    from mesa_anyjev import __version__ as mesa_anyjev_version
    from mesa_anyjev.backends.factory import make_backend
    from mesa_anyjev.bench.run import (
        MIN_HELDOUT_PER_CLASS,
        MIN_TRAIN_PER_CLASS,
        _cell,
        environment,
    )
    from mesa_anyjev.bench.tasks.base import Task
    from mesa_anyjev.bench.tasks.neon import tasks_from_store
    from mesa_anyjev.config import load_config
    from mesa_anyjev.learn.fit import fit_decider, fit_on
    from mesa_anyjev.provenance.store import open_store
    from mesa_anyjev.providers.anyjev_provider import AnyJevProvider
    from mesa_anyjev.providers.base import DecisionRecord, LevelUnavailable
    from mesa_anyjev.questions import lock_sha
except ImportError:  # pragma: no cover - operator error
    sys.exit(
        "anyjev_l2_predictions.py must run under mesa-anyjev's interpreter, e.g.\n"
        "  <mesa-anyjev checkout>/.venv/bin/python scripts/anyjev_l2_predictions.py "
        "--provenance duckdb:///<mesa-anyjev checkout>/.local/labels.duckdb --out ..."
    )

FORMAT = "mesa-clm/anyjev-l2-predictions/1"
TASKS = ("neon_term_fits", "neon_ontology_fits")
LEVEL = "L2"


def _target(state: dict[str, Any]) -> dict[str, Any]:
    """The compact target identity mesa-clm's ``identity.target_key`` reads from a state."""
    out: dict[str, Any] = {
        "dataset": state["card"]["dataset"],
        "scope": state.get("scope") or "column",
        "aspect": state.get("aspect"),
    }
    if "column" in state:
        out["target"] = state["column"]["name"]
    elif "site" in state:
        out["target"] = state["site"]["code"]
    else:
        out["target"] = out["dataset"]
    cand = state.get("candidate")
    ont = state.get("ontology")
    if isinstance(cand, dict):
        out["option_key"] = cand.get("curie")
    elif isinstance(ont, dict):
        out["option_key"] = str(ont.get("id", "")).lower()
    else:
        out["option_key"] = ""
    return out


def _item(rec: DecisionRecord, label: int, card: str, fold: str, *, states: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "card": card,
        "fold": fold,
        "state_sha256": rec.state_sha256,
        "target": _target(rec.state),
        "label": int(label),
        "label_text": rec.options[label],
        "p_yes": float(rec.probs[0]) if rec.probs else None,
        "probs": [float(p) for p in rec.probs] if rec.probs else None,
        "answer_index": rec.answer_index,
        "level": rec.level,
        "prompt_sha256": rec.prompt_sha256,
        "diagnostics": dict(rec.diagnostics),
    }
    if states:
        item["state_json"] = rec.state
    return item


def loco_predictions(provider: AnyJevProvider, task: Task, *, states: bool) -> dict[str, Any]:
    """``_fit_eval_loco`` with the records kept (mesa-anyjev ``bench/run.py``)."""
    q = task.question
    positive = 0 if q.kind == "noul" else None
    items: list[dict[str, Any]] = []
    pooled: list[DecisionRecord] = []
    pooled_labels: list[int] = []
    folds: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    timings: dict[str, float] = {}
    for card, test, train in task.leave_one_card_out():
        tr_counts = Task("f", q, list(train), "", "").class_counts()
        te_counts = Task("f", q, list(test), "", "").class_counts()
        if positive is not None and (
            min(tr_counts.values(), default=0) < MIN_TRAIN_PER_CLASS or len(tr_counts) < 2
        ):
            skipped[card] = f"insufficient_train_per_class {tr_counts}"
            continue
        if positive is not None and (
            min(te_counts.values(), default=0) < MIN_HELDOUT_PER_CLASS or len(te_counts) < 2
        ):
            skipped[card] = f"insufficient_heldout_per_class {te_counts}"
            continue
        started = time.monotonic()
        dec = fit_decider(provider.backend, provider.cfg)
        with provider.lock:
            try:
                artifact = fit_on(dec, q, [s for s, _ in train], [lbl for _, lbl in train], LEVEL)
            except (ValueError, LevelUnavailable) as exc:
                skipped[card] = f"fit failed: {exc}"
                continue
            decisions = dec.decide_batch([s for s, _ in test], q, level=LEVEL)
        recs = [provider.record_from(q, s, d) for (s, _), d in zip(test, decisions, strict=True)]
        labels = [lbl for _, lbl in test]
        cell = _cell(recs, labels, positive=positive)
        cell["head"] = {
            k: artifact.get(k) for k in ("layer_abs", "method", "kind", "temperature", "n_calib")
        }
        folds[card] = cell
        timings[card] = round(time.monotonic() - started, 1)
        items.extend(
            _item(r, lbl, card, card, states=states) for r, lbl in zip(recs, labels, strict=True)
        )
        pooled.extend(recs)
        pooled_labels.extend(labels)
        print(f"  {card}: n={len(test)} acc={cell['acc']:.3f} {timings[card]}s", file=sys.stderr)
    summary: dict[str, Any] = (
        _cell(pooled, pooled_labels, positive=positive) if pooled else {"not_applicable": True}
    )
    summary.update({"loco": True, "n_folds": len(folds), "skipped_folds": skipped, "level": LEVEL})
    return {
        "task": task.name,
        "question_id": task.meta.get("question_id"),
        "question_key": q.key,
        "n_items": len(task.items),
        "class_counts": task.class_counts(),
        "min_weight": task.meta.get("min_weight"),
        "cell": summary,
        "folds": folds,
        "seconds_per_fold": timings,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--provenance", required=True, help="mesa-anyjev labels DSN (opened read-only)")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--config", help="mesa-anyjev YAML config (backend defaults to hf/Qwen3-8B)")
    p.add_argument(
        "--backend", default="hf", choices=["hf", "fake"], help="fake is for a dry run only"
    )
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--no-states", action="store_true", help="omit state_json per item")
    args = p.parse_args(argv)

    cfg = load_config(args.config, flag_overrides={"backend": {"kind": args.backend}})
    store = open_store(args.provenance, read_only=True)
    try:
        tasks = tasks_from_store(store)
    finally:
        store.close()
    wanted = args.tasks.split(",")
    missing = [t for t in wanted if t not in tasks]
    if missing:
        print(f"unknown or empty tasks: {missing}; available: {sorted(tasks)}", file=sys.stderr)
        return 1
    backend = make_backend(cfg.backend)
    provider = AnyJevProvider(
        backend, cfg.decider
    )  # no promoted bundle: every fold fits its own head
    if not provider.capabilities.hidden_states:
        print(
            f"backend {provider.model} exposes no hidden states; L2 is unsupported (D11)",
            file=sys.stderr,
        )
        return 1
    labels_path = Path(args.provenance.removeprefix("duckdb:///")).expanduser()
    labels_sha = (
        hashlib.sha256(labels_path.read_bytes()).hexdigest() if labels_path.exists() else None
    )
    payload: dict[str, Any] = {
        "format": FORMAT,
        "mesa_anyjev": mesa_anyjev_version,
        "questions_lock_sha": lock_sha(),
        "environment": environment(provider, cfg.backend.kind),
        "labels_dsn": args.provenance,
        "labels_file_sha256": labels_sha,
        "level": LEVEL,
        "host": platform.node(),
        "tasks": {},
    }
    for name in wanted:
        print(f"== {name}: {len(tasks[name].items)} items", file=sys.stderr)
        payload["tasks"][name] = loco_predictions(provider, tasks[name], states=not args.no_states)
        cell = payload["tasks"][name]["cell"]
        print(
            f"   pooled acc={cell.get('acc')} ece={cell.get('ece')} nll={cell.get('nll')} "
            f"n={cell.get('n')} folds={cell.get('n_folds')}",
            file=sys.stderr,
        )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
