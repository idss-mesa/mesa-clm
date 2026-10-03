#!/usr/bin/env python3
"""X1's p50 latency (``design/m2-analysis-plan.md`` §11.3): a label-free live timing run.

For each X1 task (``term.fits``, ``column.ontology_fits``) and framing (F1, F4, F7, F9), the
first ``--n`` (20) targets of the manifest in ``sha256("task|framing|target")`` order
(:func:`mesa_clm.bench.framing.latency_targets`) are each asked once under each model
(``clm-latest``, ``clm-raw``) through clm-serve's ``/v1/systemone``, sequentially from one client,
the two models back to back: the target at an even position of the draw is asked under
``clm-latest`` first, one at an odd position under ``clm-raw`` first, so each model has as many
first asks as second ones (``first``). F4/F7/F9 send one Choice over the target's labelled
candidates plus the anchor (the request ``features build`` embedded, ``framings.build_request``),
F1 one noul per labelled candidate, summed per target. The client's wall time per target is kept;
**every answer is discarded**, never kept, written or printed. No label is read: the requests are
built from the snapshot's identity and state columns (``features._snapshot_pairs``) and checked
against the manifest's texts.

What "cold" means (§11.3): clm-serve's embedder cache is keyed by text and shared by both models,
the units are not restarted and nothing is cleared. A target's second ask (under the other model)
finds its texts cached; the anchor is the same text for every target of a task and framing, and a
candidate's text is the same under F4, F7 and F9, so they can be cached by an earlier ask in the
run. ``new_text`` records, per target and model, whether its requests carried a text this run had
not sent before (the share that may have needed the encoder), next to ``first``.

    uv run python scripts/x1_latency.py --out bench/results/<date>/x1_latency.json

Run it once, after the plan and the code are pushed and before ``bench framing`` (plan §14.3),
and pass its file to ``bench framing --latency``. Keys are read through the configuration inside
this process only and are never printed, logged or written; the output is checked for both
before it is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from mesa_clm import framings, render
from mesa_clm.bench import framing as x1
from mesa_clm.bench import registered as reg
from mesa_clm.learn import features as feat
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import TASKS

Request = tuple[Any, dict[str, Any]]


class SystemOne(Protocol):
    def system_one(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> Any: ...


def requests_by_target(
    snapshot: Path, manifest: feat.Manifest, task_id: str, framing_id: str
) -> dict[str, list[Request]]:
    """Every request each labelled target of ``task_id`` costs under ``framing_id``, built as
    ``features.manifest`` builds them and checked to render exactly the manifest's texts."""
    pairs = feat._snapshot_pairs(snapshot, [task_id])[task_id]
    f = framings.framing(task_id, framing_id)
    texts = {
        (r.target_sha256, r.option_key, r.role): r.text
        for r in manifest.rows
        if r.task_id == task_id and r.framing_id == framing_id
    }
    by_target: dict[str, list[Any]] = {}
    for p in pairs:
        by_target.setdefault(p.target_sha256, []).append(p)
    out: dict[str, list[Request]] = {}
    for target, group in by_target.items():
        if f.control:
            requests: list[Request] = []
            for p in group:
                state = framings.build_context(f, p.state)
                questions = {p.option_key: framings.noul_question(f)}
                stext, keys, ntexts = render.build_pairs(state, questions)[p.option_key]
                noul = dict(zip(keys, ntexts, strict=True))
                if (
                    stext != texts.get((target, p.option_key, "context"))
                    or noul["true"] != texts.get((target, p.option_key, "noul_true"))
                    or noul["false"] != texts.get((target, p.option_key, "noul_false"))
                ):
                    raise SystemExit(f"{task_id}/{framing_id}/{target[:12]}: request != manifest")
                requests.append((state, questions))
            out[target] = requests
            continue
        scope = TASKS[task_id].scope
        states = [{**p.state, "scope": p.state.get("scope") or scope} for p in group]
        cands = [feat._candidate(task_id, p) for p in group]
        state, questions = framings.build_request(f, states[0], cands)
        stext, keys, ctexts = render.build_pairs(state, questions)[task_id]
        if (
            stext != texts.get((target, "", "context"))
            or keys[-1] != ANCHOR_KEY
            or ctexts[-1] != texts.get((target, ANCHOR_KEY, "anchor"))
            or any(
                t != texts.get((target, k, "candidate"))
                for k, t in zip(keys[:-1], ctexts[:-1], strict=True)
            )
        ):
            raise SystemExit(f"{task_id}/{framing_id}/{target[:12]}: request != manifest")
        out[target] = [(state, questions)]
    return out


def time_target(
    client: SystemOne, requests: Sequence[Request], model: str, clock: Callable[[], float]
) -> float:
    """Wall ms to ask every request of one target, one after another; the answers are
    discarded."""
    started = clock()
    for state, questions in requests:
        client.system_one(state, questions, model=model)  # answer discarded (label-free run)
    return 1000.0 * (clock() - started)


def target_texts(manifest: feat.Manifest, task_id: str, framing_id: str) -> dict[str, set[str]]:
    """Every text one target's requests carry under ``framing_id`` (its context, anchor and
    candidates; F1: each pair's context and noul texts), from the manifest
    (:func:`requests_by_target` checks the requests render exactly these)."""
    out: dict[str, set[str]] = {}
    for r in manifest.rows:
        if r.task_id == task_id and r.framing_id == framing_id:
            out.setdefault(r.target_sha256, set()).add(r.text)
    return out


Timing = dict[str, dict[str, dict[str, list[Any]]]]


def model_order(position: int) -> tuple[str, ...]:
    """The models in the order the target at ``position`` of the draw asks them: ``clm-latest``
    first at an even position, ``clm-raw`` first at an odd one (§11.3)."""
    return tuple(x1.MODELS) if position % 2 == 0 else tuple(reversed(x1.MODELS))


def measure(
    client: SystemOne,
    snapshot: Path,
    manifest: feat.Manifest,
    *,
    n: int,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """``{"ms": {task_id: {arm: [ms per target]}}, "first": {…: [bool]}, "new_text": {…:
    [bool]}}`` over the drawn targets of every X1 arm, both models of a target back to back in
    :func:`model_order`. ``new_text`` is computed before the ask from the texts this run has
    sent so far (never from an answer)."""
    out: Timing = {"ms": {}, "first": {}, "new_text": {}}
    sent: set[str] = set()
    for task_id in feat.X1_TASKS:
        for what in out.values():
            what[task_id] = {}
        for framing_id in feat.X1_FRAMINGS:
            requests = requests_by_target(snapshot, manifest, task_id, framing_id)
            texts = target_texts(manifest, task_id, framing_id)
            drawn = x1.latency_targets(manifest, task_id, framing_id, n)
            arms = {m: x1.arm_id(framing_id, m) for m in x1.MODELS}
            for what in out.values():
                what[task_id].update({arm: [] for arm in arms.values()})
            for position, target in enumerate(drawn):
                for k, model in enumerate(model_order(position)):
                    arm = arms[model]
                    out["new_text"][task_id][arm].append(bool(texts[target] - sent))
                    sent |= texts[target]
                    out["first"][task_id][arm].append(k == 0)
                    out["ms"][task_id][arm].append(
                        time_target(client, requests[target], model, clock)
                    )
    return out


def main(argv: list[str] | None = None) -> int:
    from mesa_clm.clm.fingerprint import clm_model_fp, load_serving_lock
    from mesa_clm.clm.http import ClmHttpClient
    from mesa_clm.config import load_config
    from mesa_clm.providers.live import live_lock_path

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--snapshot", type=Path, default=reg.current().snapshot_path())
    ap.add_argument("--n", type=int, default=x1.LATENCY_TARGETS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if reg.file_sha256(args.snapshot) != reg.current().labels_sha256:
        raise SystemExit(f"{args.snapshot}: not the registered snapshot")
    cfg = load_config(None)
    lock = load_serving_lock(live_lock_path())
    manifest = feat.manifest(args.snapshot, feat.X1_TASKS, feat.X1_FRAMINGS)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    t0 = time.monotonic()
    clm = ClmHttpClient.from_config(cfg.clm)
    secrets = [cfg.clm.resolved_api_key() or "", cfg.encoder.resolved_api_key() or ""]
    try:
        timing = measure(clm, args.snapshot, manifest, n=args.n)
    finally:
        clm.close()
    ms = timing["ms"]
    payload = {
        "format": x1.LATENCY_FORMAT,
        "started_at": started,
        "seconds": round(time.monotonic() - t0, 1),
        "command": f"uv run python scripts/x1_latency.py --out {args.out}",
        "what": "design/m2-analysis-plan.md §11.3: wall ms per target through clm-serve "
        "/v1/systemone, one client, sequential; label-free, every answer discarded",
        "labels_sha256": manifest.labels_sha256,
        "draw": "per (task, framing) the manifest's targets in sha256('task|framing|target') "
        f"order, the first {args.n}, each asked under both models back to back",
        "order": "a target at an even position of the draw is asked under clm-latest first, one "
        "at an odd position under clm-raw first ('first' per target and arm)",
        "warm": "clm-serve's embedder cache is keyed by text and shared by both models: a "
        "target's second ask finds its texts cached, the anchor and the F4/F7/F9 candidate texts "
        "can be cached by an earlier ask; 'new_text' per target and arm says whether its "
        "requests carried a text this run had not sent before",
        "serving": {"lock_sha": lock.lock_sha, "encoder_fp": lock.encoder_fp},
        "clm_model_fp": {m: clm_model_fp(lock.model_spec(m)) for m in x1.MODELS},
        "ms": ms,
        "first": timing["first"],
        "new_text": timing["new_text"],
        "summary": {
            t: {
                a: x1.latency_summary(v, timing["first"][t][a], timing["new_text"][t][a])
                for a, v in arms.items()
            }
            for t, arms in ms.items()
        },
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    if any(s and s in text for s in secrets):
        raise SystemExit("refusing to write: a key value appears in the output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
