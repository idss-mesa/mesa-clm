"""The frozen tasks (``mesa_clm.tasks``): every ``task_key`` equals the mesa-anyjev lock key
(DESIGN D1), texts and options are verbatim, and ``task_key`` reproduces AnyJev's
``Question.key`` formula. Ported from mesa-anyjev ``tests/test_questions_lock.py``; the lock
copy is the committed fixture ``tests/fixtures/anyjev_questions.lock.json``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from mesa_clm.registry import ASPECT_OPTIONS, ONTOLOGY_OPTIONS, VALUE_KINDS
from mesa_clm.tasks import (
    ACTIVE_TASKS,
    MAX_OPTIONS,
    NOUL_OPTIONS,
    RANK_FIT_TASKS,
    TASKS,
    TWINS,
    Task,
    TaskError,
    by_key,
    task,
    task_key,
    task_keys,
)

LOCK_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "anyjev_questions.lock.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["questions"]


def test_every_task_key_equals_the_anyjev_lock() -> None:
    assert set(TASKS) == set(LOCK)
    for task_id, t in TASKS.items():
        assert t.key == LOCK[task_id]["key"], task_id
        assert task_keys()[task_id] == LOCK[task_id]["key"]


def test_texts_kinds_and_options_are_verbatim() -> None:
    for task_id, t in TASKS.items():
        assert t.kind == LOCK[task_id]["kind"], task_id
        assert t.text == LOCK[task_id]["text"], task_id
        assert list(t.options) == LOCK[task_id]["options"], task_id


def test_twins_match_the_lock() -> None:
    assert {qid: spec["twin"] for qid, spec in LOCK.items() if spec["twin"]} == TWINS
    for wide, twin in TWINS.items():
        assert TASKS[wide].kind == "choice" and TASKS[twin].kind == "noul"
        assert TASKS[wide].k > 8, "only choices past the gateway limit have a twin"


def test_known_keys() -> None:
    assert TASKS["term.fits"].key == "0ccc8d141ffd30ff"
    assert TASKS["column.annotate"].key == "8b8c7f14be35925d"
    assert TASKS["column.aspect"].key == "982e2baa2c1c7762"
    assert TASKS["column.ontology_fits"].key == "f45eb7fe2fbefdb5"
    assert TASKS["avu.value_kind"].key == "18a3a5159c3a8492"
    keys = [t.key for t in TASKS.values()]
    assert len(keys) == len(set(keys)) == 17


def test_active_set_and_rank_fit_tasks() -> None:
    assert ACTIVE_TASKS == (
        "column.annotate",
        "column.aspect",
        "column.ontology",
        "column.ontology_fits",
        "term.fits",
        "avu.value_kind",
    )
    assert task_keys(active_only=True).keys() == set(ACTIVE_TASKS)
    assert set(ACTIVE_TASKS) >= RANK_FIT_TASKS
    assert all(TASKS[t].kind == "noul" for t in RANK_FIT_TASKS)
    for inactive in ("avu.keep", "term.fits.chooser", "dataset.ontology_applies"):
        assert not TASKS[inactive].active


def test_scopes_follow_the_anyjev_specs() -> None:
    assert TASKS["avu.value_kind"].scope == "avu" and TASKS["avu.keep"].scope == "avu"
    assert TASKS["term.fits"].scope == "column"
    assert TASKS["dataset.ontology_applies"].scope == "dataset"
    assert all(t.scope == "dataset" for i, t in TASKS.items() if i.startswith("datacite."))


def test_registry_options_are_the_task_options() -> None:
    assert TASKS["column.ontology"].options == ONTOLOGY_OPTIONS and TASKS["column.ontology"].k == 12
    assert TASKS["column.aspect"].options == ASPECT_OPTIONS
    assert TASKS["avu.value_kind"].options == VALUE_KINDS
    assert all(t.options == NOUL_OPTIONS for t in TASKS.values() if t.kind == "noul")


def test_task_key_reproduces_the_anyjev_formula() -> None:
    """sha256 of json.dumps({...}, sort_keys=True, ensure_ascii=False)[:16], floats in scale."""
    payload = json.dumps(
        {
            "kind": "noul",
            "text": "Ça va?",
            "options": ["Yes", "No"],
            "scale": [0.0, 1.0],
            "centers": None,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert task_key("noul", "Ça va?", ("Yes", "No")) == expected
    assert "é" not in payload.encode("ascii", "backslashreplace").decode() or "\\u" not in payload
    # Integers in scale hash like AnyJev's floats; an empty centers tuple hashes as None.
    assert task_key("noul", "t", ("Yes", "No"), (0, 1)) == task_key("noul", "t", ("Yes", "No"))
    assert task_key("choice", "t", ("a", "b"), centers=()) == task_key("choice", "t", ("a", "b"))
    assert task_key("score", "t", ("lo", "hi"), (0.0, 1.0), (0.0, 1.0)) != task_key(
        "score", "t", ("lo", "hi")
    )


def test_key_excludes_id_but_includes_wording() -> None:
    same = Task("renamed", "noul", TASKS["term.fits"].text, NOUL_OPTIONS, "column", False)
    assert same.key == TASKS["term.fits"].key
    reworded = Task(
        "term.fits", "noul", TASKS["term.fits"].text + " Really?", NOUL_OPTIONS, "column", True
    )
    assert reworded.key != TASKS["term.fits"].key


def test_task_validation_mirrors_anyjev_constructors() -> None:
    with pytest.raises(TaskError, match="noul task has the options"):
        Task("x", "noul", "t", ("Yes", "Nope"), "column", False)
    with pytest.raises(TaskError, match="at least 2"):
        Task("x", "choice", "t", ("only",), "column", False)
    with pytest.raises(TaskError, match="unique"):
        Task("x", "choice", "t", ("a", "a"), "column", False)
    with pytest.raises(TaskError, match=f"at most {MAX_OPTIONS}"):
        Task("x", "choice", "t", tuple(str(i) for i in range(27)), "column", False)
    with pytest.raises(TaskError, match="2 to 10"):
        Task("x", "score", "t", ("one",), "column", False)
    with pytest.raises(TaskError, match="unknown kind"):
        Task("x", "rank", "t", ("a", "b"), "column", False)  # type: ignore[arg-type]
    with pytest.raises(TaskError, match="empty text"):
        Task("x", "noul", "  ", NOUL_OPTIONS, "column", False)


def test_tasks_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TASKS["term.fits"].text = "changed"  # type: ignore[misc]


def test_lookups() -> None:
    assert task("term.fits") is TASKS["term.fits"]
    assert by_key("0ccc8d141ffd30ff").id == "term.fits"
    with pytest.raises(KeyError):
        by_key("0" * 16)
    with pytest.raises(KeyError):
        task("no.such.task")
    assert TASKS["column.aspect"].index_of(ASPECT_OPTIONS[2]) == 2
    assert TASKS["term.fits"].index_of("No") == 1
    with pytest.raises(TaskError, match="not one of its options"):
        TASKS["term.fits"].index_of("Maybe")
