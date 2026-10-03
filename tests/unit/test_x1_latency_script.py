"""``scripts/x1_latency.py`` (``design/m2-analysis-plan.md`` §11.3), without a server: the
requests are built from the committed snapshot's identity and state columns (no label column is
read) and checked against the manifest's texts, each drawn target is asked once under each model
through a stub that answers nothing (the two models back to back, alternating which goes first),
and only times and two label-free flags per ask are kept. No model is involved."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mesa_clm.bench import framing as x1
from mesa_clm.bench import registered as reg
from mesa_clm.learn import features as feat
from mesa_clm.learn.features import Manifest, ManifestRow, text_sha256

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"


def _script() -> ModuleType:
    path = ROOT / "scripts" / "x1_latency.py"
    spec = importlib.util.spec_from_file_location("x1_latency", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ask(model: str | None, state: Any, questions: Mapping[str, Any]) -> tuple[Any, ...]:
    return (model, json.dumps(state, sort_keys=True, default=str), sorted(questions))


class Stub:
    """Records every ask; answers nothing (the run discards every answer anyway)."""

    def __init__(self) -> None:
        self.asked: list[tuple[Any, ...]] = []

    def system_one(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> Any:
        self.asked.append(_ask(model, state, questions))
        return None

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return feat.manifest(SNAPSHOT, feat.X1_TASKS, feat.X1_FRAMINGS)


def test_the_timing_run_asks_each_drawn_target_once_per_model_and_keeps_only_times(
    tmp_path: Path, manifest: Manifest
) -> None:
    """Exactly the drawn targets' requests, each once per model, the models back to back with
    ``clm-latest`` first at even positions and ``clm-raw`` first at odd ones (§11.3); the
    ``new_text`` flag of each ask says whether its texts were new to the run so far."""
    script = _script()
    stub = Stub()
    ticks = itertools.count()
    n = 3
    timing = script.measure(stub, SNAPSHOT, manifest, n=n, clock=lambda: float(next(ticks)))
    assert set(timing) == {"ms", "first", "new_text"}
    expected: list[tuple[Any, ...]] = []
    sent: set[str] = set()
    for task_id in feat.X1_TASKS:
        assert set(timing["ms"][task_id]) == set(x1.ARM_ORDER)
        for framing_id in feat.X1_FRAMINGS:
            requests = script.requests_by_target(SNAPSHOT, manifest, task_id, framing_id)
            drawn = x1.latency_targets(manifest, task_id, framing_id, n)
            assert len(drawn) == n
            for position, target in enumerate(drawn):
                order = (
                    ["clm-latest", "clm-raw"] if position % 2 == 0 else ["clm-raw", "clm-latest"]
                )
                texts = {
                    r.text
                    for r in manifest.rows
                    if (r.task_id, r.framing_id, r.target_sha256) == (task_id, framing_id, target)
                }
                for k, model in enumerate(order):
                    arm = x1.arm_id(framing_id, model)
                    assert timing["first"][task_id][arm][position] is (k == 0)
                    assert timing["new_text"][task_id][arm][position] is bool(texts - sent)
                    sent |= texts
                    expected += [_ask(model, s, q) for s, q in requests[target]]
            for model in ("clm-latest", "clm-raw"):
                arm = x1.arm_id(framing_id, model)
                # one tick per target: the stub answers instantly
                assert timing["ms"][task_id][arm] == [1000.0] * n
            if framing_id != "F1":
                assert all(len(requests[t]) == 1 for t in requests)  # one Choice per target
    assert stub.asked == expected  # every request exactly once per model, in this order
    # what the run writes is what bench framing --latency reads, flags included
    out = tmp_path / "lat.json"
    out.write_text(json.dumps({"format": x1.LATENCY_FORMAT, **timing}), encoding="utf-8")
    loaded, info = x1.load_latency(out)
    assert loaded == timing["ms"] and info["format"] == x1.LATENCY_FORMAT
    assert info["first"] == timing["first"] and info["new_text"] == timing["new_text"]


def test_the_draw_is_in_sha256_order_of_task_framing_target() -> None:
    """§11.3, Appendix B: ``latency_targets`` takes a (task, framing)'s distinct targets in
    ``sha256("task|framing|target")`` order (a golden list on a synthetic manifest)."""

    def row(target: str, framing: str = "F7") -> ManifestRow:
        text = f"{framing} context {target}"
        return ManifestRow(
            task_id="term.fits", framing_id=framing, target_sha256=target, option_key="",
            role="context", side="state", text_sha256=text_sha256(text), text=text,
        )  # fmt: skip

    targets = [f"t{k}" for k in range(10)]
    man = Manifest(
        snapshot="s", labels_sha256="c" * 64, tasks=("term.fits",), framings=("F4", "F7"),
        rows=tuple([row(t) for t in targets] + [row("t0", "F4")]),
    )  # fmt: skip
    drawn = x1.latency_targets(man, "term.fits", "F7", n=10)
    assert drawn == ["t5", "t2", "t8", "t7", "t4", "t6", "t3", "t1", "t9", "t0"]
    assert drawn == sorted(
        targets, key=lambda t: hashlib.sha256(f"term.fits|F7|{t}".encode()).hexdigest()
    )
    assert x1.latency_targets(man, "term.fits", "F7", n=4) == drawn[:4]
    assert x1.latency_targets(man, "term.fits", "F4", n=4) == ["t0"]


def test_a_request_that_does_not_render_the_manifest_is_refused(manifest: Manifest) -> None:
    """The requests must render exactly the manifest's texts, for a Choice (F7) and for F1's
    nouls; one altered text stops the run before anything is asked."""
    script = _script()
    for framing_id, role in (("F7", "context"), ("F7", "candidate"), ("F1", "noul_true")):
        victim = next(
            r
            for r in manifest.rows
            if (r.task_id, r.framing_id, r.role) == ("term.fits", framing_id, role)
        )
        text = victim.text + " (edited)"
        edited = victim.model_copy(update={"text": text, "text_sha256": text_sha256(text)})
        rows = tuple(edited if r is victim else r for r in manifest.rows)
        with pytest.raises(SystemExit, match="request != manifest"):
            script.requests_by_target(
                SNAPSHOT, manifest.model_copy(update={"rows": rows}), "term.fits", framing_id
            )


def test_the_run_refuses_to_write_a_key_and_records_its_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` writes nothing when a key value appears in its output, and otherwise records
    the labels and the serving lock its file must match in ``bench framing`` (§11.3). The
    client is a stub; the keys are dummies."""
    from mesa_clm import config
    from mesa_clm.clm import http

    script = _script()

    def cfg_with(key: str | None) -> Any:
        return SimpleNamespace(
            clm=SimpleNamespace(resolved_api_key=lambda: key),
            encoder=SimpleNamespace(resolved_api_key=lambda: None),
        )

    class Client(Stub):
        @classmethod
        def from_config(cls, cfg: Any) -> Client:
            return cls()

    monkeypatch.setattr(http, "ClmHttpClient", Client)
    out = tmp_path / "lat.json"
    monkeypatch.setattr(config, "load_config", lambda path=None: cfg_with("x1-latency/1"))
    with pytest.raises(SystemExit, match="a key value appears in the output"):
        script.main(["--n", "1", "--out", str(out)])
    assert not out.exists()
    monkeypatch.setattr(config, "load_config", lambda path=None: cfg_with("dummy-key-0000"))
    assert script.main(["--n", "1", "--out", str(out)]) == 0
    assert "dummy-key-0000" not in out.read_text()
    from mesa_clm.clm.fingerprint import load_serving_lock

    lock = load_serving_lock(ROOT / "serving" / "serving.lock.json")
    ms, info = x1.load_latency(
        out, labels_sha256=reg.current().labels_sha256, lock_sha=lock.lock_sha
    )
    assert set(ms) == set(feat.X1_TASKS) and info["lock_sha"] == lock.lock_sha
    assert set(info["first"]) == set(feat.X1_TASKS)
    with pytest.raises(x1.X1DataError, match="timing run's labels_sha256"):
        x1.load_latency(out, labels_sha256="0" * 64)
    with pytest.raises(x1.X1DataError, match="timing run's lock_sha"):
        x1.load_latency(out, lock_sha="1" * 64)
    with pytest.raises(SystemExit, match="not the registered snapshot"):
        script.main(["--snapshot", str(out), "--out", str(tmp_path / "x.json")])


def test_the_loader_refuses_flags_that_are_not_one_per_target(tmp_path: Path) -> None:
    out = tmp_path / "lat.json"
    arm = "F7@clm-raw"
    base: dict[str, Any] = {"format": x1.LATENCY_FORMAT, "ms": {"term.fits": {arm: [1.0, 2.0]}}}
    for flag, value in (("first", [True]), ("new_text", [1, 0])):
        out.write_text(json.dumps({**base, flag: {"term.fits": {arm: value}}}))
        with pytest.raises(x1.X1DataError, match=f"{flag} is not one flag per target"):
            x1.load_latency(out)
    out.write_text(json.dumps({**base, "first": {"term.fits": {arm: [True, False]}}}))
    _, info = x1.load_latency(out)
    assert info["first"] == {"term.fits": {arm: [True, False]}} and "new_text" not in info
    summary = x1.latency_summary([1.0, 2.0], [True, False], [True, True])
    assert summary["first"]["p50"] == 1.0 and summary["second"]["p50"] == 2.0
    assert summary["new_text_share"] == 1.0
