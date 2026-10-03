"""The closed-choice manifest (``mesa_clm.learn.features``; plan §4.2 Q1, Q2, Q7, §8 M2;
``design/m2-analysis-plan.md`` §2.2, §9.1) and the store's token counts.

From the committed snapshot, label-free: ``column.annotate``, ``column.aspect`` and
``avu.value_kind`` under their one framing F7 give one context per labelled target, exactly the
text the builders make from the fixture card (``column_state`` / ``value_kind_state``, no
instructions), and one option row per closed option with the text CLM's ``build_pairs`` renders
from the framing; annotate and aspect share the column's context; flipping every label changes
nothing; a closed choice under any other framing is refused, as before, and the default
manifest is still the X1/X2 one."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from mesa_clm import framings, render
from mesa_clm.cards import load_card
from mesa_clm.clm.fake import FakeEncoder
from mesa_clm.learn import features as feat
from mesa_clm.learn.features import (
    CHOICE_FRAMINGS,
    CHOICE_TASKS,
    MANIFEST_TASKS,
    FeatureMissing,
    FeatureStore,
    ManifestError,
    manifest,
)
from mesa_clm.providers.tiered import fake_fingerprint
from mesa_clm.registry import ANNOTATE_OPTIONS, ASPECT_OPTIONS, ASPECTS, VALUE_KINDS
from mesa_clm.states import column_state, value_kind_state

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
CARDS = ROOT / "tests" / "fixtures" / "cards"
K = {"column.annotate": 2, "column.aspect": 8, "avu.value_kind": 4}


@pytest.fixture(scope="module")
def choice() -> feat.Manifest:
    return manifest(SNAPSHOT, CHOICE_TASKS, "F7")


def _targets(task_id: str) -> int:
    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT count(DISTINCT target_sha256) FROM read_parquet(?) WHERE task_id = ?",
            [str(SNAPSHOT), task_id],
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    return int(row[0])


def _states() -> dict[tuple[str, str], dict[str, Any]]:
    """One stored state per (task, target) of the closed choices (no label column read)."""
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT task_id, target_sha256, state_json FROM read_parquet(?) "
            "WHERE task_id IN ('column.annotate', 'column.aspect', 'avu.value_kind') "
            "ORDER BY task_id, target_sha256, state_sha256",
            [str(SNAPSHOT)],
        ).fetchall()
    finally:
        con.close()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for task_id, target, state in rows:
        out.setdefault((str(task_id), str(target)), json.loads(state))
    return out


def test_constants_and_the_default_manifest_is_unchanged() -> None:
    assert CHOICE_TASKS == ("column.annotate", "column.aspect", "avu.value_kind")
    assert CHOICE_FRAMINGS == ("F7",)
    assert (*feat.X1_TASKS, *CHOICE_TASKS) == MANIFEST_TASKS
    assert feat.ROLE_SIDE["option"] == "action" and "option" in feat.ROLES
    default = manifest(SNAPSHOT, framing_ids="F7")
    assert default.tasks == feat.X1_TASKS
    assert {r.task_id for r in default.rows} == set(feat.X1_TASKS)


def test_choice_manifest_shape_counts_and_determinism(choice: feat.Manifest) -> None:
    assert choice.tasks == CHOICE_TASKS and choice.framings == ("F7",)
    assert choice.labels_sha256 == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert choice.context_conflicts == 0
    summary = choice.summary()
    for task_id in CHOICE_TASKS:
        n = _targets(task_id)
        assert summary["counts"][task_id] == {"F7": {"context": n, "option": n * K[task_id]}}
    # the option texts are the same for every target: 2 + 8 + 4 distinct action texts
    assert summary["texts_by_side"]["action"] == 14
    assert manifest(SNAPSHOT, CHOICE_TASKS, "F7").rows == choice.rows
    for r in choice.rows:
        assert r.side == ("state" if r.role == "context" else "action")
        assert r.role in ("context", "option") and r.framing_id == "F7"
        assert (r.option_key == "") == (r.role == "context")
    for task_id in CHOICE_TASKS:
        targets = [r.target_sha256 for r in choice.rows if r.task_id == task_id]
        assert targets == sorted(targets)


def test_option_rows_are_the_framings_closed_options(choice: feat.Manifest) -> None:
    want: dict[str, list[tuple[str, str]]] = {}
    for task_id in CHOICE_TASKS:
        f = framings.framing(task_id, "F7")
        keys, texts = render.candidates(framings.build_question(f))
        want[task_id] = list(zip(keys, texts, strict=True))
    assert [k for k, _ in want["column.annotate"]] == list(ANNOTATE_OPTIONS)
    assert [t for _, t in want["column.annotate"]] == list(ANNOTATE_OPTIONS.values())
    assert [k for k, _ in want["column.aspect"]] == list(ASPECTS)
    assert [t for _, t in want["column.aspect"]] == list(ASPECT_OPTIONS)
    assert [k for k, _ in want["avu.value_kind"]] == list(framings.VALUE_KIND_KEYS)
    assert [t for _, t in want["avu.value_kind"]] == list(VALUE_KINDS)
    per_target: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for r in choice.rows:
        if r.role == "option":
            per_target.setdefault((r.task_id, r.target_sha256), []).append((r.option_key, r.text))
    assert per_target
    for (task_id, _), opts in per_target.items():
        assert opts == want[task_id]  # every target, in wire (= option) order


def test_every_context_is_the_one_the_builders_make(choice: feat.Manifest) -> None:
    """The column text Q1/Q2 send and the value-kind text Q7 sends, rebuilt from the fixture
    card (the snapshot keeps sorted-key JSON; the manifest restores builder order)."""
    states = _states()
    checked = 0
    for r in choice.rows:
        if r.role != "context":
            continue
        state = states[(r.task_id, r.target_sha256)]
        card = load_card(CARDS / f"{state['card']['dataset']}.md")
        col = card.column(state["column"]["name"])
        if r.task_id == "avu.value_kind":
            built = value_kind_state(card, col, state["term"], state["aspect"])
        else:
            built = column_state(card, col)
        f = framings.framing(r.task_id, "F7")
        assert f.instructions is None
        assert r.text == framings.context_text(f, built) == render.state_text(built, None)
        checked += 1
    assert checked == sum(_targets(t) for t in CHOICE_TASKS)


def test_annotate_and_aspect_share_the_column_text(choice: feat.Manifest) -> None:
    """Q1 and Q2 ask one request per column (plan §4.2): the aspect contexts are annotate's."""
    ctx = {
        t: {r.text for r in choice.rows if r.task_id == t and r.role == "context"}
        for t in CHOICE_TASKS
    }
    assert ctx["column.aspect"] <= ctx["column.annotate"]
    assert not ctx["avu.value_kind"] & ctx["column.annotate"]


def test_choice_manifest_is_label_free(tmp_path: Path, choice: feat.Manifest) -> None:
    from tests.unit.test_features import LABEL_FREE_REPLACE

    flipped = tmp_path / "flipped.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY ({LABEL_FREE_REPLACE}) TO '{flipped}' (FORMAT PARQUET)",
            [str(SNAPSHOT)],
        )
    finally:
        con.close()
    other = manifest(flipped, CHOICE_TASKS, "F7")
    assert other.rows == choice.rows and other.labels_sha256 != choice.labels_sha256


def test_choice_tasks_take_f7_only() -> None:
    for framing_ids in ("F1", "F4", "F9", "X2", "F7,F9"):
        with pytest.raises(ManifestError, match=r"unknown task 'column\.aspect' for framing"):
            manifest(SNAPSHOT, "column.aspect", framing_ids)
    with pytest.raises(ManifestError, match="unknown task"):
        manifest(SNAPSHOT, "column.aspect")  # the default framings are X1/X2's
    with pytest.raises(ManifestError, match=r"unknown task 'column\.ontology'"):
        manifest(SNAPSHOT, "column.ontology", "F7")
    mixed = manifest(SNAPSHOT, "term.fits,avu.value_kind", "F7")
    assert {r.task_id for r in mixed.rows} == {"term.fits", "avu.value_kind"}
    assert {r.role for r in mixed.rows if r.task_id == "term.fits"} == {
        "context",
        "anchor",
        "candidate",
    }


def test_token_counts_read_the_stored_counts(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features", fake_fingerprint().encoder_fp)
    texts = ["alpha beta", "gamma delta epsilon", "zeta"]
    vecs, _ = FakeEncoder().embed(texts)
    store.add(texts, vecs, [11, 22, 33])
    store.add_truncated(["far too long"], [5000])
    assert store.token_counts([texts[2], texts[0], texts[2]]) == [33, 11, 33]
    assert store.token_counts(["far too long"]) == [5000]  # truncated texts keep their count
    assert store.token_counts([]) == []
    with pytest.raises(FeatureMissing, match="token count"):
        store.token_counts(["never seen"])
    assert np.array_equal(store.get(texts[:1]), store.get(texts[:1]))


def test_a_request_whose_keys_are_not_the_closed_options_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest takes a closed choice's option texts from the request serving sends; a
    request whose keys are not the framing's closed options, in wire order, stops it (review
    finding: the check was untested). Label-free (the snapshot's identity and state columns)."""
    real = render.build_pairs

    def reordered(state: Any, questions: Any) -> Any:
        return {
            qid: (text, list(reversed(keys)), list(reversed(texts)))
            for qid, (text, keys, texts) in real(state, questions).items()
        }

    monkeypatch.setattr(feat.render, "build_pairs", reordered)
    for task_id in CHOICE_TASKS:
        with pytest.raises(ManifestError, match="the request keys are not the closed options"):
            manifest(SNAPSHOT, [task_id], "F7")
