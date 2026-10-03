"""The M2 bench verbs of the console script (``bench framing``, ``bench framing --decide --from``,
``bench run``, ``bench x2``, ``bench table``; ``design/m2-analysis-plan.md`` §12, §14) on a
**synthetic** world only, as the M2 pre-commitment rule requires: the bench targets' states are
read from the committed snapshot's identity and state columns (no label column is read), every
label is generated here (the rank_fit labels from the fake encoder's own scores, so X1 has a
state-dependent signal to find), the vectors come from the deterministic fake encoder (hashed
n-grams, never evidence), the head is a random one exported under the lock's sha, and the
registration (:mod:`mesa_clm.bench.registered`) is stood in by the synthetic snapshot's hashes
and counts. No silver label and no model output on a bench item is involved.

Checked: the registered run writes ``x1.json``, ``x1_items.parquet`` and ``x1.md`` and replays;
a partial run is written unregistered and ``--decide --from`` refuses it; a results file is
never overwritten without ``--force``; another snapshot, a store with other labels, other counts
and a tampered or foreign ``x1.json`` are refused with exit 1; usage errors exit 2; ``bench run``
writes the tier cells of every task from X1's replayed outcome; ``bench x2`` the baselines with
the pinned AnyJev dump; ``bench table`` names every cell's file; ``--help`` of every verb."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from mesa_clm import framings, serving
from mesa_clm.bench import framing as x1
from mesa_clm.bench import registered as reg
from mesa_clm.bench import run as m2run
from mesa_clm.bench import x2 as x2_module
from mesa_clm.bench.baselines import run_baselines
from mesa_clm.bench.results import load_results, snapshot_content_sha256, write_results
from mesa_clm.bench.tasks.neon import tasks_from_store
from mesa_clm.cli import EXIT_CONFIG, EXIT_FAIL, EXIT_OK, REGISTERED_SNAPSHOT, main
from mesa_clm.clm.fake import FakeEncoder
from mesa_clm.clm.fingerprint import load_serving_lock
from mesa_clm.clm.headproj import HeadProjector, l2, random_head
from mesa_clm.learn import features as feat
from mesa_clm.learn.labels import NEON_EVAL_ORIGIN, snapshot
from mesa_clm.learn.offline import pairwise_s_c
from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.providers import live
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
REAL_SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
TASK_IDS = (
    "term.fits",
    "column.ontology_fits",
    "column.annotate",
    "column.aspect",
    "avu.value_kind",
)
DATE = "2026-10-02"
_ENV_PREFIXES = ("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)


def _identity_rows() -> list[dict[str, Any]]:
    """One row per D1 identity of the five tasks, identity and state columns only."""
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT task_id, task_key, target_sha256, option_key, state_sha256, state_json, card, "
            "product_code, leak_group FROM read_parquet(?) "
            "QUALIFY row_number() OVER (PARTITION BY task_id, target_sha256, option_key "
            "ORDER BY state_sha256) = 1 ORDER BY task_id, target_sha256, option_key",
            [str(REAL_SNAPSHOT)],
        ).fetchall()
    finally:
        con.close()
    names = (
        "task_id",
        "task_key",
        "target",
        "option",
        "state_sha",
        "state",
        "card",
        "product",
        "leak",
    )
    out = [dict(zip(names, r, strict=True)) for r in rows]
    for r in out:
        r["state"] = json.loads(r["state"]) if isinstance(r["state"], str) else dict(r["state"])
    return out


def _write_snapshot(
    rows: list[dict[str, Any]], labels: dict[tuple[str, str, str], int], path: Path
) -> str:
    store = LabelStore(path.with_suffix(".duckdb"))
    label_rows = []
    for r in rows:
        k = labels[(r["task_id"], r["target"], r["option"])]
        spec = TASKS[r["task_id"]]
        binary = spec.kind == "noul"
        source = "consensus_majority" if (not binary or k == 0) else "consensus_negative"
        label_rows.append(
            LabelRow(
                task_id=r["task_id"],
                task_key=r["task_key"],
                target_sha256=r["target"],
                option_key=r["option"],
                label_source=source,  # type: ignore[arg-type]
                label=spec.options[k],
                label_index=k,
                weight=0.6 if source == "consensus_majority" else 0.5,
                state_sha256=r["state_sha"],
                state_json=r["state"],
                card=r["card"],
                product_code=r["product"],
                leak_group=r["leak"],
                origin=NEON_EVAL_ORIGIN + "synthetic000",
                actor="synthetic",
            )
        )
    store.insert_labels(label_rows)
    return snapshot(store, path)


def _rank_labels(
    rows: list[dict[str, Any]], scores: dict[tuple[str, str, str], float]
) -> dict[tuple[str, str, str], int]:
    """Yes for the top 35% of a rank_fit task's fake F7 scores (plus a little hashed noise), a
    hashed class for a closed choice: generated, never the silver."""
    out: dict[tuple[str, str, str], int] = {}
    for task_id in TASK_IDS:
        mine = [r for r in rows if r["task_id"] == task_id]
        k = TASKS[task_id].k
        if task_id in ("term.fits", "column.ontology_fits"):
            noisy = {
                (task_id, r["target"], r["option"]): scores[(task_id, r["target"], r["option"])]
                + 0.3 * ((_h(r["target"], r["option"]) % 1000) / 1000 - 0.5)
                for r in mine
            }
            cut = np.quantile(list(noisy.values()), 0.65)
            out.update({key: 0 if v > cut else 1 for key, v in noisy.items()})
        else:
            for r in mine:
                out[(task_id, r["target"], r["option"])] = (
                    _h("cls", r["target"]) % 3 if k > 2 else _h("cls", r["target"]) % 2
                ) % k
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("m2world")
    rows = _identity_rows()
    draft = root / "draft.parquet"
    _write_snapshot(rows, {(r["task_id"], r["target"], r["option"]): 1 for r in rows}, draft)
    manifests = [
        feat.manifest(draft, feat.X1_TASKS, "F1,F4,F7,F9,X2"),
        feat.manifest(draft, feat.CHOICE_TASKS, "F7"),
    ]
    texts = sorted({t for m in manifests for t in m.texts()})
    lock = load_serving_lock(ROOT / "serving" / "serving.lock.json")
    store = feat.FeatureStore.for_lock(root / "features", lock)
    vecs, _ = FakeEncoder().embed(texts)
    store.add(texts, vecs, [max(1, len(t.split())) for t in texts])
    # the fake F7 raw scores the rank_fit labels are drawn from (one store read per role)
    index = {
        (r.task_id, r.framing_id, r.target_sha256, r.option_key, r.role): r.text
        for r in manifests[0].rows
    }
    ranked = [r for r in rows if r["task_id"] in feat.X1_TASKS]
    keys = [(r["task_id"], "F7", r["target"]) for r in ranked]
    ctx = [index[(*k, "", "context")] for k in keys]
    cand = [index[(*k, r["option"], "candidate")] for k, r in zip(keys, ranked, strict=True)]
    anchor = [index[(*k, ANCHOR_KEY, "anchor")] for k in keys]
    s = pairwise_s_c(100.0, l2(store.get(ctx)), l2(store.get(cand)), l2(store.get(anchor)))
    scores = {
        (r["task_id"], r["target"], r["option"]): float(v) for r, v in zip(ranked, s, strict=True)
    }
    synthetic = root / "synthetic.parquet"
    sha = _write_snapshot(rows, _rank_labels(rows, scores), synthetic)
    content = snapshot_content_sha256(synthetic)
    tasks = tasks_from_store(LabelStore(synthetic.with_suffix(".duckdb")))
    counts = {}
    for name, task in tasks.items():
        per: dict[str, int] = {}
        for c in task.cards:
            per[c] = per.get(c, 0) + 1
        counts[name] = reg.TaskCounts(
            len(task.items), task.class_counts(), per, float(task.meta["min_weight"])
        )
    dump_items: dict[str, list[dict[str, Any]]] = {}
    for name, bench in (
        ("neon_term_fits", "neon_term_fits"),
        ("neon_ontology_fits", "neon_ontology_fits"),
    ):
        task = tasks[bench]
        dump_items[name] = [
            {
                "card": task.cards[i],
                "fold": task.cards[i],
                "state_sha256": "0" * 64,
                "label": label,
                "p_yes": (_h("p", str(i), name) % 1000) / 1000,
                "state_json": state,
            }
            for i, (state, label) in enumerate(task.items)
        ]
    dump = root / "anyjev.json"
    dump.write_text(
        json.dumps(
            {
                "format": "mesa-clm/anyjev-l2-predictions/1",
                "tasks": {k: {"question_key": "q", "items": v} for k, v in dump_items.items()},
            }
        ),
        encoding="utf-8",
    )
    # what `bench baselines` would have published for these synthetic labels (M0's file)
    published = run_baselines(
        LabelStore(synthetic.with_suffix(".duckdb")), labels_sha256=sha, date="2026-09-29", B=200
    )
    published_path = write_results(published, root / "published")
    registration = reg.Registration(
        snapshot=str(synthetic),
        labels_sha256=sha,
        labels_content_sha256=content,
        published=str(published_path),
        counts=counts,
        anyjev_dump=str(dump),
        anyjev_sha256=hashlib.sha256(dump.read_bytes()).hexdigest(),
        B=200,
        # the checkout's framings lock: the real registration keeps G1's, which DESIGN A1
        # rotated after the registered run
        framings_lock_sha=framings.lock_sha(),
    )
    home = root / "serving-home"
    h = random_head(3, hidden_size=4096)
    npz = home / serving.HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    npz.parent.mkdir(parents=True)
    HeadProjector(h.cfg, h.state, h.action, h.logit_scale, lock.head.sha256).to_npz(npz)
    return {"root": root, "registration": registration, "snapshot": synthetic, "home": home}


@pytest.fixture
def m2(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The synthetic world in force for one test; returns the results root."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_FEATURES__DIR", str(world["root"] / "features"))
    monkeypatch.setattr(serving, "DEFAULT_HOME", str(world["home"]))
    monkeypatch.setattr(reg, "REGISTERED", world["registration"])
    # the PR #13 recipe on 4096-d fake vectors converges slowly; the plumbing under test does not
    # depend on it (tests/unit/test_x2.py checks the recipe itself against scikit-learn)
    monkeypatch.setattr(x2_module, "replica_pipeline", _quick_pipeline)
    assert live.live_lock_path() == ROOT / "serving" / "serving.lock.json"
    return tmp_path / "results"


def _quick_pipeline() -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=25))


def _framing(out: Path, *extra: str) -> int:
    return main(["bench", "framing", "--date", DATE, "--out-dir", str(out), *extra])


def test_parser_defaults_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert reg.REGISTERED.snapshot == REGISTERED_SNAPSHOT  # the parser's default is the pin
    for verb in ("framing", "run", "x2", "table"):
        with pytest.raises(SystemExit) as exc:
            main(["bench", verb, "--help"])
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--decide" in out and "--from" in out and "--loco" in out and "--framing-from" in out
    assert "--B" not in out.split("bench framing")[-1]  # no bootstrap knobs on the M2 verbs


def test_framing_runs_registered_and_replays(m2: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _framing(m2, "--decide") == EXIT_OK
    out = capsys.readouterr()
    path = m2 / DATE / "x1.json"
    assert out.out.splitlines()[:2] == [str(path), "registered run"]
    assert "no --latency file" in out.err and "replayed and recomputed" in out.out
    for name in ("x1.json", "x1_items.parquet", "x1.md"):
        assert (m2 / DATE / name).is_file()
    res = x1.load_x1(path)
    assert res.x1.registered and res.x1.deviations == []
    assert all(c.pre_registered and c.exploratory for c in res.cells.values())
    assert len(res.cells) == 16  # 4 framings x 2 models x 2 tasks
    assert res.labels_sha256 == reg.REGISTERED.labels_sha256
    assert res.x1.config.B == 200 and res.x1.config.shuffle_k == x1.SHUFFLE_K
    # the replay from the JSON alone
    assert main(["bench", "framing", "--decide", "--from", str(path)]) == EXIT_OK
    replay = capsys.readouterr().out
    assert "the pre-registered run" in replay and "neon_term_fits:" in replay
    # one run per results file
    assert _framing(m2) == EXIT_FAIL
    assert "pass --force" in capsys.readouterr().err
    assert _framing(m2, "--force", "--tasks", "term.fits") == EXIT_OK
    assert "NOT the registered run" in capsys.readouterr().out
    assert main(["bench", "framing", "--decide", "--from", str(path)]) == EXIT_FAIL
    assert "not the pre-registered run" in capsys.readouterr().err


def test_partial_and_refused_framing_runs(
    m2: Path, world: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _framing(m2, "--full") == EXIT_OK
    assert "NOT the registered run (nested False" in capsys.readouterr().out
    assert main(["bench", "framing", "--decide", "--from", str(m2 / DATE / "x1.json")]) == EXIT_FAIL
    capsys.readouterr()
    # a run that is not the registered one still replays and recomputes as what it is
    assert _framing(m2, "--force", "--tasks", "term.fits", "--decide") == EXIT_OK
    out = capsys.readouterr().out
    assert "NOT the registered run (tasks ['term.fits']" in out and "replayed and recomputed" in out
    # usage: --from without --decide, an unknown task, a missing file
    assert _framing(m2, "--from", "x.json") == EXIT_CONFIG
    assert _framing(m2, "--tasks", "column.aspect") == EXIT_CONFIG
    assert (
        main(["bench", "framing", "--decide", "--from", str(tmp_path / "no.json")]) == EXIT_CONFIG
    )
    capsys.readouterr()
    # another snapshot (one more label row) is refused before anything runs
    other = tmp_path / "other.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet(?) UNION ALL (SELECT * REPLACE ('extra' AS label_id, "  # noqa: S608
            f"'teacher' AS label_source) FROM read_parquet(?) LIMIT 1)) TO '{other}' (FORMAT PARQUET)",
            [str(world["snapshot"]), str(world["snapshot"])],
        )
    finally:
        con.close()
    assert _framing(tmp_path / "o", "--snapshot", str(other)) == EXIT_FAIL
    assert "not the pre-registered snapshot" in capsys.readouterr().err
    # a snapshot that does not exist is refused and never written (§1.1: no write fallback)
    missing = tmp_path / "snapshots" / f"{DATE}.parquet"
    assert _framing(tmp_path / "m", "--snapshot", str(missing)) == EXIT_FAIL
    assert "no such labels snapshot" in capsys.readouterr().err and not missing.exists()
    # the published counts are checked before any statistic
    reg_now = reg.current()
    counts = dict(reg_now.counts)
    first = next(iter(counts))
    counts[first] = replace(counts[first], n=counts[first].n + 1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(reg, "REGISTERED", replace(reg_now, counts=counts))
        assert _framing(tmp_path / "c", "--force") == EXIT_FAIL
    assert "not the published" in capsys.readouterr().err


def test_run_x2_and_table(m2: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "run", "--date", DATE, "--out-dir", str(m2), "--loco"]) == EXIT_CONFIG
    assert "no X1 results file" in capsys.readouterr().err
    assert _framing(m2) == EXIT_OK
    capsys.readouterr()
    run = ["bench", "run", "--date", DATE, "--out-dir", str(m2)]
    assert main(run) == EXIT_CONFIG
    assert "--loco" in capsys.readouterr().err
    assert main([*run, "--loco", "--tiers", "zero_shot,probe"]) == EXIT_CONFIG
    assert "probe lands in M4" in capsys.readouterr().err
    assert main([*run, "--loco"]) == EXIT_OK
    out = capsys.readouterr().out
    tiers = load_results(m2 / DATE / "tiers.json")
    assert out.splitlines()[0] == str(m2 / DATE / "tiers.json")
    tasks = {c.task for c in tiers.cells.values()}
    assert tasks == {
        "neon_term_fits",
        "neon_ontology_fits",
        "neon_annotate",
        "neon_aspect",
        "neon_value_kind",
    }
    sel = x1.selections_from_json(m2 / DATE / "x1.json", recompute=False)
    for task_id, name in (
        ("term.fits", "neon_term_fits"),
        ("column.ontology_fits", "neon_ontology_fits"),
    ):
        a1 = sel[task_id]["a1"]
        key = f"{name}.calibrated.{a1['framing'] if a1 else 'F7'}"
        cell = tiers.cells[key]
        if a1 is None:  # K1 or undecidable: audit-only default arm
            assert cell.selection == "none" and not cell.pre_registered and cell.exploratory
        else:
            assert cell.selection == "nested" and cell.pre_registered and not cell.exploratory
            assert set(cell.fold_choices) == set(
                x1.load_x1(m2 / DATE / "x1.json").x1.tasks[name].cards
            )
            assert cell.model == a1["model"]
    assert tiers.cells["neon_aspect.calibrated.F7"].pre_registered
    # §9.5, §12.3: each evaluated fold keeps X1's pointer to its inner trace, and the note names
    # the x1.json it came from by sha256
    x1_raw = json.loads((m2 / DATE / "x1.json").read_text())
    sha = hashlib.sha256((m2 / DATE / "x1.json").read_bytes()).hexdigest()
    assert any(f"(sha256 {sha})" in note for note in tiers.notes)
    pointers = 0
    for cell in tiers.cells.values():
        if cell.selection != "nested":
            continue
        for card, entry in cell.fold_choices.items():
            if entry["decision"] != "arm":
                continue
            node: Any = x1_raw
            for part in entry["x1_trace"].removeprefix("#/").split("/"):
                node = node[part]
            assert node["choice"]["framing"] == entry["framing"], card
            assert node["choice"]["model"] == entry["model"], card
            pointers += 1
    assert pointers > 0
    assert main([*run, "--loco"]) == EXIT_FAIL  # tiers.json exists
    capsys.readouterr()
    assert main([*run, "--loco", "--force", "--tiers", "zero_shot"]) == EXIT_OK
    capsys.readouterr()
    partial = load_results(m2 / DATE / "tiers.json")
    assert all(not c.pre_registered and c.exploratory for c in partial.cells.values())
    # a tampered x1.json is replayed and refused; a foreign one is refused by identity
    x1_path = m2 / DATE / "x1.json"
    raw = json.loads(x1_path.read_text())
    bad = json.loads(json.dumps(raw))
    fold = next(iter(bad["x1"]["tasks"]["neon_term_fits"]["nested"]["fold_choices"].values()))
    fold["outcome"] = "K1" if fold["outcome"] != "K1" else "choice"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(bad))
    assert main([*run, "--loco", "--force", "--framing-from", str(tampered)]) == EXIT_FAIL
    assert "fold_choices" in capsys.readouterr().err
    foreign = json.loads(json.dumps(raw))
    foreign["x1"]["framings_lock_sha"] = "0" * 64
    other = tmp_path / "foreign.json"
    other.write_text(json.dumps(foreign))
    assert main([*run, "--loco", "--force", "--framing-from", str(other)]) == EXIT_FAIL
    assert "framings lock" in capsys.readouterr().err
    # §12.7, §14.6: the tier run recomputes X1 from its items file before it takes the outcome;
    # an items file the replay cannot see edited (its new sha256 recorded) is refused
    copy = tmp_path / "copy"
    copy.mkdir()
    copied = copy / "x1.json"
    items_copy = copy / "x1_items.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE items AS SELECT * FROM read_parquet(?)",
            [str(m2 / DATE / "x1_items.parquet")],
        )
        con.execute("UPDATE items SET p_loco = p_loco + 1e-6 WHERE p_loco IS NOT NULL AND item = 0")
        con.execute(
            "COPY (SELECT * FROM items ORDER BY task, framing, model, item) "  # noqa: S608
            f"TO '{items_copy}' (FORMAT PARQUET)"
        )
    finally:
        con.close()
    edited = json.loads(json.dumps(raw))
    edited["x1"]["items_file"]["sha256"] = hashlib.sha256(items_copy.read_bytes()).hexdigest()
    copied.write_text(json.dumps(edited))
    assert main([*run, "--loco", "--force", "--framing-from", str(copied)]) == EXIT_FAIL
    assert "p_loco is not the recomputed" in capsys.readouterr().err
    # X2 and the table; another AnyJev dump is refused by the verb before anything runs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(reg, "REGISTERED", replace(reg.current(), anyjev_sha256="e" * 64))
        assert main(["bench", "x2", "--date", DATE, "--out-dir", str(m2)]) == EXIT_FAIL
    assert "not the registered AnyJev L2 dump" in capsys.readouterr().err
    assert not (m2 / DATE / "x2.json").exists()
    assert main(["bench", "x2", "--date", DATE, "--out-dir", str(m2)]) == EXIT_OK
    x2 = load_results(m2 / DATE / "x2.json")
    assert {"neon_term_fits.baseline.pr13@S1", "neon_term_fits.baseline.anyjev_l2"} <= set(x2.cells)
    assert all(c.pre_registered for c in x2.cells.values())
    # §10.1: the recomputed no-model controls are the published M0 cells, field for field
    published = reg.current().published_path()
    assert x2.notes[-1].startswith("M0 controls recomputed: all 5 equal to"), x2.notes[-1]
    raw = json.loads(published.read_text())
    raw["cells"]["neon_aspect.baseline.lookup_prob"]["counts"]["n"] += 1
    edited = tmp_path / "published.json"
    edited.write_text(json.dumps(raw))
    note = m2run.m0_comparison(x2, edited)
    assert "4 equal" in note and "NOT equal: neon_aspect.baseline.lookup_prob (counts)" in note
    assert "no published file" in m2run.m0_comparison(x2, tmp_path / "none.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    assert "could not be read" in m2run.m0_comparison(x2, broken)  # report-only, never fails
    capsys.readouterr()
    assert main(["bench", "table", "--date", DATE, "--out-dir", str(m2)]) == EXIT_OK
    table = capsys.readouterr().out
    assert f"{m2 / DATE / 'x1.json'}#neon_term_fits.calibrated.F7@clm-latest" in table
    assert "auroc [one-sided 95% bounds; 90% interval]" in table and "95% cluster" not in table
    assert "tiers.json#neon_aspect.zero_shot.F7" in table and "x2.json#" in table
    out = tmp_path / "table.md"
    assert (
        main(["bench", "table", "--results", str(m2 / DATE / "x2.json"), "--out", str(out)])
        == EXIT_OK
    )
    assert out.read_text().startswith("# bench table")
    assert (
        main(["bench", "table", "--results", str(m2 / DATE / "x2.json"), "--out", str(out)])
        == EXIT_FAIL
    )
    assert main(["bench", "table", "--results", str(tmp_path / "none.json")]) == EXIT_CONFIG


def test_the_scratch_store_must_hold_the_snapshots_rows(
    m2: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1.1: the bench tasks come from a scratch store holding exactly the snapshot's rows; a
    store that does not (one row lost) is refused before any task is built."""
    from mesa_clm.provenance.labels import TABLE

    inputs = m2run.load_inputs(None, tmp_path / "ok")
    assert inputs.labels_sha256 == reg.current().labels_sha256
    assert set(inputs.tasks) == set(reg.current().counts)
    real = m2run.snapshot_store

    def lossy(snapshot: Any, path: Any) -> Any:
        store = real(snapshot, path)
        with store.connect() as con:
            con.execute(
                f"DELETE FROM {TABLE} WHERE label_id = (SELECT min(label_id) FROM {TABLE})"  # noqa: S608
            )
        return store

    monkeypatch.setattr(m2run, "snapshot_store", lossy)
    with pytest.raises(reg.RegistrationError, match="scratch store"):
        m2run.load_inputs(None, tmp_path / "lossy")


def test_the_tier_run_takes_only_an_x1_file_about_its_own_inputs(tmp_path: Path) -> None:
    """§12.7: ``bench run`` refuses an ``x1.json`` whose labels, label content, framings lock or
    either model's fingerprint is not the tier run's own (each checked alone)."""
    from types import SimpleNamespace

    from tests.unit.test_framing_x1 import LATEST, RAW, _clean_run

    path = x1.write_x1(_clean_run(nested=False), tmp_path)
    fps = {m: {"clm_model_fp": m[:12].ljust(12, "0")} for m in (LATEST, RAW)}
    inputs = m2run.Inputs(Path("s"), "a" * 64, "b" * 64)
    m2run._check_x1_identity(path, inputs, SimpleNamespace(fingerprints=fps))  # type: ignore[arg-type]
    cases: list[tuple[Any, Any, str]] = [
        (replace(inputs, labels_sha256="c" * 64), fps, "labels_sha256"),
        (replace(inputs, labels_content_sha256="c" * 64), fps, "labels_content_sha256"),
        (inputs, {**fps, RAW: {"clm_model_fp": "other"}}, "clm-raw fingerprint"),
        (inputs, {**fps, LATEST: {"clm_model_fp": "other"}}, "clm-latest fingerprint"),
    ]
    for given, prints, what in cases:
        with pytest.raises(m2run.BenchRunError, match=what):
            m2run._check_x1_identity(path, given, SimpleNamespace(fingerprints=prints))  # type: ignore[arg-type]
    raw = json.loads(path.read_text())
    raw["x1"]["framings_lock_sha"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(m2run.BenchRunError, match="framings lock"):
        m2run._check_x1_identity(path, inputs, SimpleNamespace(fingerprints=fps))  # type: ignore[arg-type]


def test_the_verbs_refuse_a_store_of_another_recipe_and_a_foreign_timing_file(
    m2: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refusals before any statistic (review finding): the timing file must be about the
    registered labels and this serving lock (§11.3), and the feature store must carry the
    serving lock's vector recipe, checked by ``serving`` itself before it returns."""
    from mesa_clm.config import load_config

    lock = load_serving_lock(ROOT / "serving" / "serving.lock.json")
    timing = tmp_path / "lat.json"
    for labels, lock_sha, what in (
        ("0" * 64, lock.lock_sha, "labels_sha256"),
        (reg.current().labels_sha256, "1" * 64, "lock_sha"),
    ):
        timing.write_text(
            json.dumps(
                {"format": x1.LATENCY_FORMAT, "labels_sha256": labels,
                 "serving": {"lock_sha": lock_sha}, "ms": {}}
            )
        )  # fmt: skip
        assert _framing(tmp_path / "o", "--latency", str(timing)) == EXIT_FAIL
        assert f"timing run's {what}" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
    assert m2run.serving(load_config(None)).store.encoder_fp == lock.encoder_fp
    other = tmp_path / "features"
    feat.FeatureStore(other, lock.encoder_fp, vector_recipe={"image": "another"}).ensure()
    monkeypatch.setenv("MESA_CLM_FEATURES__DIR", str(other))
    with pytest.raises(feat.FeatureStoreError, match="vector recipe"):
        m2run.serving(load_config(None))
