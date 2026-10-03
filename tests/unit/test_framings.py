"""Framings (``mesa_clm.framings``, DESIGN D1–D3, D23; plan §4.1–4.4, §5.6 X1): the registry,
the ``question_key`` formula and its rotation semantics (a template edit rotates, an encoder
change cannot), context projection, the wire questions, the F1 control's anyjev-shaped states
and the ``framings.lock.json`` drift detection (pattern: mesa-anyjev ``test_questions_lock.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import framings as fr
from mesa_clm import render
from mesa_clm.cards import DatasetCard
from mesa_clm.registry import (
    ANCHOR_KEY,
    ANCHORS,
    ANNOTATE_OPTIONS,
    ASPECT_OPTIONS,
    ASPECTS,
    ONTOLOGY_OPTIONS,
    ONTOLOGY_REGISTRY,
    VALUE_KINDS,
    entry,
)
from mesa_clm.states import (
    MAX_DESCRIPTION,
    candidate_state,
    column_state,
    ontology_state,
    state_sha256,
    target_state,
    value_kind_state,
)
from mesa_clm.tasks import NOUL_OPTIONS, TASKS

FIT_TASKS = ("term.fits", "column.ontology_fits")
CLOSED_TASKS = ("column.annotate", "column.aspect", "avu.value_kind")

CANDS = [
    fr.FramingCandidate(
        "PATO:0000040", "distance", "x" * 500, "pato", ("a", "b", "c", "d", "e", "f"), True
    ),
    fr.FramingCandidate("UO:0000008", "meter", "a length unit", "uo"),
    fr.FramingCandidate("ENVO:00000001", "bare label"),
]


def _target(card: DatasetCard) -> dict[str, object]:
    return target_state(card, "column", "measurement", column=card.column("observerDistance"))


# -- registry ------------------------------------------------------------------------------------


def test_registry_covers_the_five_clm_tasks_with_the_x1_arms() -> None:
    assert set(fr.FRAMINGS) == set(FIT_TASKS) | set(CLOSED_TASKS)
    for task_id in FIT_TASKS:
        assert list(fr.FRAMINGS[task_id]) == ["F1", "F4", "F7", "F9"]
        assert [f.id for f in fr.framings_for(task_id)] == ["F1", "F4", "F7", "F9"]
        assert all(f.shape == "rank_fit" for f in fr.framings_for(task_id))
    for task_id in CLOSED_TASKS:
        assert list(fr.FRAMINGS[task_id]) == ["F7"]
        assert fr.framings_for(task_id)[0].shape == "choice"
    assert (
        "column.ontology" not in fr.FRAMINGS
    )  # asked as ontology_fits (Q3), never as a wide choice
    with pytest.raises(fr.FramingError, match="no framings"):
        fr.framings_for("column.ontology")


# DESIGN A1 (2026-10-03): X1's registered run chose F9 on clm-latest for column.ontology_fits
# (bench/results/2026-10-03/x1.json#/x1/tasks/neon_ontology_fits/a1); term.fits was K1 and keeps
# F7 for its audit records; the closed choices have F7 only. Before A1 every task was at F7.
A1_ACTIVE = {
    "column.annotate": "F7",
    "column.aspect": "F7",
    "column.ontology_fits": "F9",
    "term.fits": "F7",
    "avu.value_kind": "F7",
}


def test_the_active_framings_are_a1s() -> None:
    assert fr.ACTIVE == A1_ACTIVE
    for task_id, arms in fr.FRAMINGS.items():
        active = fr.active_framing(task_id)
        assert active is arms[A1_ACTIVE[task_id]] and active.active and not active.control
        assert [fid for fid, f in arms.items() if f.active] == [A1_ACTIVE[task_id]]
    with pytest.raises(fr.FramingError, match="no active framing"):
        fr.active_framing("avu.keep")


def test_the_a1_rotation_moved_no_question_key() -> None:
    """``active`` is not part of a ``question_key`` (D1): A1 rotated the lock and no key. The
    lock of G1 (every task at F7), which M2's registration pins, is today's framings under the
    G1 map; the checkout's lock differs from it in column.ontology_fits' active framing only."""
    g1 = {task_id: "F7" for task_id in fr.FRAMINGS}
    assert fr.lock_sha(g1) == "b432d32a7536c8f455098ae4a23139f6badbae7bc181a3a64853df3e80921eca"
    assert fr.lock_sha() != fr.lock_sha(g1) and fr.lock_sha(fr.ACTIVE) == fr.lock_sha()
    before, after = fr.lock_payload(g1)["tasks"], fr.lock_payload()["tasks"]
    for task_id in fr.FRAMINGS:
        keys = {fid: e["question_key"] for fid, e in after[task_id]["framings"].items()}
        assert keys == {fid: e["question_key"] for fid, e in before[task_id]["framings"].items()}
        if task_id != "column.ontology_fits":
            assert after[task_id] == before[task_id]
    onto_before, onto_after = before["column.ontology_fits"], after["column.ontology_fits"]
    assert (onto_before["active_framing"], onto_after["active_framing"]) == ("F7", "F9")
    assert onto_after["framings"]["F9"]["question_key"] == "c95785008b523fd0"
    assert onto_after["framings"]["F9"]["active"] and not onto_after["framings"]["F7"]["active"]
    with pytest.raises(fr.FramingError, match="cannot be the active framing"):
        fr.lock_payload({**g1, "term.fits": "F1"})  # the control is never active
    with pytest.raises(fr.FramingError, match="an active framing for each"):
        fr.lock_payload({"term.fits": "F7"})


def test_fit_arms_follow_the_pre_registration() -> None:
    for task_id in FIT_TASKS:
        f1, f4, f7, f9 = fr.framings_for(task_id)
        anchor = ANCHORS["term" if task_id == "term.fits" else "ontology"]
        # F1: the mesa-anyjev shape (noul per candidate over its state, task sentence as question).
        assert f1.control and not f1.active
        assert f1.context_view == (
            "candidate_state" if task_id == "term.fits" else "ontology_state"
        )
        assert f1.instructions == TASKS[task_id].text
        assert f1.anchor_key is None and f1.anchor_text is None
        assert f1.option_keys == NOUL_OPTIONS and f1.label_map == {"true": "Yes", "false": "No"}
        # F4: Choice + anchor + a task sentence; F7: state alone; F9: the query template.
        assert f4.instructions and f4.instructions.endswith("?") and f4.context_template is None
        assert f7.instructions is None and f7.context_template is None
        assert f9.instructions is None and f9.context_template is not None
        for f in (f4, f7, f9):
            assert f.context_view == "target_state"
            assert f.anchor_key == ANCHOR_KEY and f.anchor_text == anchor
            assert f.closed_options is None and f.option_keys is None and f.label_map is None
            assert f.candidate_template == "{label}: {description}"
        assert all(
            f.mask_rule == ("aspect" if task_id == "column.ontology_fits" else None)
            for f in (f1, f4, f7, f9)
        )
    assert fr.framing("term.fits", "F4").instructions == (
        "Which term is the right concept for this target, not merely related?"
    )
    assert fr.framing("column.ontology_fits", "F9").context_template == (
        "NEON dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:"
    )
    assert fr.framing("term.fits", "F9").context_template == {
        "column": "NEON dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:",
        "site": "NEON dataset {title}. Site {code}: {site_name}, {habitat}. {aspect} ontology term:",
        "dataset": "NEON dataset {title}: {product_description}. {aspect} ontology term:",
    }


def test_closed_framings_carry_the_anyjev_options() -> None:
    annotate = fr.active_framing("column.annotate")
    assert annotate.context_view == "column_state"
    assert annotate.closed_options == dict(ANNOTATE_OPTIONS)
    assert annotate.option_keys == NOUL_OPTIONS and annotate.label_map == {"Yes": "Yes", "No": "No"}
    aspect = fr.active_framing("column.aspect")
    assert aspect.context_view == "column_state"
    assert tuple(aspect.closed_options or {}) == ASPECTS
    assert tuple((aspect.closed_options or {}).values()) == ASPECT_OPTIONS
    assert aspect.option_keys == ASPECT_OPTIONS and aspect.label_map == dict(
        zip(ASPECTS, ASPECT_OPTIONS, strict=True)
    )
    kind = fr.active_framing("avu.value_kind")
    assert kind.context_view == "value_kind_state"
    assert (
        tuple(kind.closed_options or {})
        == fr.VALUE_KIND_KEYS
        == ("term_label", "site_code", "column_name", "top_value")
    )
    assert kind.option_keys == VALUE_KINDS and set((kind.label_map or {}).values()) == set(
        VALUE_KINDS
    )
    for f in (annotate, aspect, kind):
        assert f.anchor_key is None and f.anchor_text is None and f.instructions is None
        assert f.context_template is None and f.mask_rule is None and not f.control
        # Every wire key maps to a task option, so a CLM answer becomes a label_index in one step.
        for key, label in (f.label_map or {}).items():
            assert f.task.index_of(label) == list(f.closed_options or {}).index(key)


def test_framings_are_frozen_hashable_and_looked_up() -> None:
    f7 = fr.framing("term.fits", "F7")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f7.instructions = "x"  # type: ignore[misc]
    assert len({f for arms in fr.FRAMINGS.values() for f in arms.values()}) == 11
    assert f7.task is TASKS["term.fits"] and f7.task_key == "0ccc8d141ffd30ff"
    assert fr.by_question_key(f7.question_key) is f7
    with pytest.raises(fr.FramingError, match="no framing has"):
        fr.by_question_key("0" * 16)
    with pytest.raises(fr.FramingError, match="no framing 'F2'"):
        fr.framing("term.fits", "F2")


# -- question_key --------------------------------------------------------------------------------


def test_question_key_formula_and_fields() -> None:
    f = fr.framing("term.fits", "F7")
    payload = fr.question_payload(f)
    assert tuple(payload) == fr.QUESTION_KEY_FIELDS
    assert (
        payload["task_key"] == "0ccc8d141ffd30ff"
        and payload["schema_sha256"] == render.schema_sha256()
    )
    assert payload["schema_sha256"].startswith("52cec58a") and payload["schema_sha256"].endswith(
        "7335"
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert fr.question_key(f) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert f.question_key == fr.question_key(f) and len(f.question_key) == 16
    assert f.question_key != f.task_key


def test_question_keys_are_unique_and_differ_from_task_keys() -> None:
    keys = [f.question_key for arms in fr.FRAMINGS.values() for f in arms.values()]
    assert len(keys) == len(set(keys)) == 11
    assert not set(keys) & {t.key for t in TASKS.values()}


def test_encoder_or_head_identity_never_enters_the_key() -> None:
    """The key takes only the framing: no fingerprint, model or route is an input (D5 owns them),
    and flipping ``active``/``control`` (bookkeeping) leaves it unchanged."""
    assert list(inspect.signature(fr.question_key).parameters) == ["framing"]
    forbidden = ("encoder", "head", "model", "revision", "dtype", "route", "fingerprint", "fp")
    assert not [k for k in fr.QUESTION_KEY_FIELDS if any(w in k for w in forbidden)]
    f7 = fr.framing("term.fits", "F7")
    assert dataclasses.replace(f7, active=False).question_key == f7.question_key
    # anchor_key is implied by anchor_text (a rank_fit always uses "__none__"): not in the payload.
    assert "anchor_key" not in fr.QUESTION_KEY_FIELDS


def test_template_instruction_anchor_and_option_edits_rotate_the_key() -> None:
    f7 = fr.framing("term.fits", "F7")
    f9 = fr.framing("term.fits", "F9")
    assert isinstance(f9.context_template, dict)
    f9_templates = dict(f9.context_template)
    variants = [
        dataclasses.replace(f7, instructions="Which term fits?"),
        dataclasses.replace(f7, candidate_template="{label} — {description}"),
        dataclasses.replace(f7, anchor_text=ANCHORS["term"] + " Really."),
        dataclasses.replace(f7, render_version="2"),
        dataclasses.replace(f7, context_view="column_state"),
        dataclasses.replace(f7, id="F8"),
        dataclasses.replace(
            f9, context_template={**f9_templates, "column": "{name}: {description}"}
        ),
    ]
    keys = {f7.question_key, *[v.question_key for v in variants]}
    assert len(keys) == len(variants) + 1
    aspect = fr.active_framing("column.aspect")
    # Option texts enter the key (the closed options are the candidate texts CLM embeds).
    edited = dataclasses.replace(
        aspect,
        closed_options={k: v + "." for k, v in (aspect.closed_options or {}).items()},
    )
    assert edited.question_key != aspect.question_key
    # So does their order: the payload keeps closed_options as ordered [key, text] pairs.
    payload = fr.question_payload(aspect)
    assert payload["closed_options"] == [[k, v] for k, v in (aspect.closed_options or {}).items()]
    assert payload["closed_options"] != sorted(payload["closed_options"])


def test_schema_change_rotates_every_key(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {f.question_key for arms in fr.FRAMINGS.values() for f in arms.values()}
    monkeypatch.setattr(render, "schema_sha256", lambda: "0" * 64)
    after = {f.question_key for arms in fr.FRAMINGS.values() for f in arms.values()}
    assert not before & after
    monkeypatch.undo()
    assert {f.question_key for arms in fr.FRAMINGS.values() for f in arms.values()} == before


# -- validation ----------------------------------------------------------------------------------


def _anchored(fid: str, task_id: str, view: fr.ContextView, **kw: Any) -> fr.Framing:
    """A rank_fit framing with the term anchor plus the field under test."""
    return fr.Framing(
        fid, task_id, "rank_fit", view, anchor_key=ANCHOR_KEY, anchor_text=ANCHORS["term"], **kw
    )


def test_framing_validation() -> None:
    kw = {"anchor_key": ANCHOR_KEY, "anchor_text": ANCHORS["term"]}
    with pytest.raises(fr.FramingError, match="unknown task"):
        _anchored("F7", "no.such", "target_state")
    with pytest.raises(fr.FramingError, match="unknown shape"):
        fr.Framing("F7", "term.fits", "score", "target_state", **kw)  # type: ignore[arg-type]
    with pytest.raises(fr.FramingError, match="unknown context view"):
        _anchored("F7", "term.fits", "avu_state")  # type: ignore[arg-type]
    with pytest.raises(fr.FramingError, match="anchor"):
        fr.Framing("F7", "term.fits", "rank_fit", "target_state")
    with pytest.raises(fr.FramingError, match="control framings only"):
        _anchored("F7", "term.fits", "candidate_state")
    with pytest.raises(fr.FramingError, match="open candidates"):
        _anchored("F7", "term.fits", "target_state", label_map={"a": "Yes"})
    with pytest.raises(fr.FramingError, match="no anchor"):
        fr.Framing(
            "F7", "column.aspect", "choice", "column_state", anchor_key=ANCHOR_KEY, anchor_text="x"
        )
    with pytest.raises(fr.FramingError, match="needs closed_options"):
        fr.Framing("F7", "column.aspect", "choice", "column_state")
    with pytest.raises(fr.FramingError, match="task's options"):
        fr.Framing(
            "F7",
            "column.aspect",
            "choice",
            "column_state",
            closed_options={"a": "x", "b": "y"},
            option_keys=("a", "b"),
            label_map={"a": "a", "b": "b"},
        )
    with pytest.raises(fr.FramingError, match="label_map"):
        fr.Framing(
            "F7",
            "column.annotate",
            "choice",
            "column_state",
            closed_options=dict(ANNOTATE_OPTIONS),
            option_keys=NOUL_OPTIONS,
            label_map={"Yes": "No", "No": "Yes"},
        )
    with pytest.raises(fr.FramingError, match="unknown template field"):
        _anchored("F9", "term.fits", "target_state", context_template="{nope}")
    with pytest.raises(fr.FramingError, match="unknown template field 'code'"):
        _anchored("F9", "term.fits", "target_state", context_template={"column": "{code}"})
    with pytest.raises(fr.FramingError, match="unknown target kind"):
        _anchored("F9", "term.fits", "target_state", context_template={"row": "{name}"})
    with pytest.raises(fr.FramingError, match="empty template map"):
        _anchored("F9", "term.fits", "target_state", context_template={})
    with pytest.raises(fr.FramingError, match="bad candidate template"):
        _anchored("F7", "term.fits", "target_state", candidate_template="{curie}")
    with pytest.raises(fr.FramingError, match="aspect mask"):
        _anchored("F7", "term.fits", "target_state", mask_rule="aspect")
    with pytest.raises(fr.FramingError, match="unknown mask rule"):
        _anchored("F7", "column.ontology_fits", "target_state", mask_rule="unit")
    with pytest.raises(fr.FramingError, match="non-empty sentence"):
        _anchored("F4", "term.fits", "target_state", instructions="  ")
    with pytest.raises(fr.FramingError, match="needs an id"):
        _anchored(" ", "term.fits", "target_state")
    with pytest.raises(fr.FramingError, match="task sentence"):
        fr.Framing(
            "F1",
            "term.fits",
            "rank_fit",
            "candidate_state",
            option_keys=NOUL_OPTIONS,
            label_map=fr.NOUL_LABEL_MAP,
            control=True,
        )
    with pytest.raises(fr.FramingError, match="label map"):
        fr.Framing(
            "F1",
            "term.fits",
            "rank_fit",
            "candidate_state",
            instructions="Q?",
            option_keys=NOUL_OPTIONS,
            label_map={"true": "No", "false": "Yes"},
            control=True,
        )
    with pytest.raises(fr.FramingError, match="no anchor option"):
        fr.Framing(
            "F1",
            "term.fits",
            "rank_fit",
            "candidate_state",
            instructions="Q?",
            anchor_key=ANCHOR_KEY,
            anchor_text="x",
            option_keys=NOUL_OPTIONS,
            label_map=fr.NOUL_LABEL_MAP,
            control=True,
        )
    with pytest.raises(fr.FramingError, match="control is a rank_fit"):
        fr.Framing(
            "F1",
            "column.annotate",
            "choice",
            "candidate_state",
            instructions="Q?",
            option_keys=NOUL_OPTIONS,
            label_map=fr.NOUL_LABEL_MAP,
            control=True,
        )


# -- contexts ------------------------------------------------------------------------------------


def test_build_context_sends_the_view_dict_for_state_only_framings(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st = _target(card)
    ctx = fr.build_context(fr.active_framing("term.fits"), st)
    assert ctx == st and list(ctx) == ["card", "scope", "aspect", "column"]
    # A stored anyjev candidate_state projects onto the same target_state (key order too).
    stored = candidate_state(card, "column", col, "measurement", {"label": "d", "curie": "X:1"}, 9)
    assert fr.build_context(fr.active_framing("term.fits"), stored) == st
    assert fr.build_context(fr.framing("column.ontology_fits", "F7"), stored) == st
    # column.ontology_fits' active framing is F9 since DESIGN A1: its template, not the dict.
    f9 = fr.build_context(fr.active_framing("column.ontology_fits"), stored)
    assert isinstance(f9, str) and f9.startswith("NEON dataset ")
    assert f9.endswith("measurement ontology term:")
    ann = fr.build_context(fr.active_framing("column.annotate"), stored)
    assert ann == column_state(card, col) and list(ann) == ["card", "column"]
    vk = value_kind_state(card, col, {"label": "distance", "curie": "PATO:0000040"}, "measurement")
    assert fr.build_context(fr.active_framing("avu.value_kind"), {**vk, "extra": 1}) == vk


def test_build_context_rejects_a_state_without_the_view(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    with pytest.raises(fr.FramingError, match="lacks \\['scope', 'aspect'\\]"):
        fr.build_context(fr.active_framing("term.fits"), column_state(card, col))
    with pytest.raises(fr.FramingError, match="lacks \\['term'\\]"):
        fr.build_context(
            fr.active_framing("avu.value_kind"), {**column_state(card, col), "aspect": "x"}
        )
    with pytest.raises(fr.FramingError, match="both"):
        fr.build_context(fr.active_framing("term.fits"), {**_target(card), "site": {"code": "X"}})
    with pytest.raises(fr.FramingError, match="unknown context view"):
        fr.project("avu_state", _target(card))


def test_f9_template_renders_column_site_and_dataset_targets(card: DatasetCard) -> None:
    f9 = fr.framing("term.fits", "F9")
    assert fr.build_context(f9, _target(card)) == (
        "NEON dataset Breeding landbird point counts. Column observerDistance: Radial distance "
        "between the observer and the individual(s) being observed (meter). measurement ontology term:"
    )
    # A unitless column shows its dtype in the parenthesis instead of an empty "()".
    assert fr.build_context(
        f9, target_state(card, "column", "other", column=card.column("siteID"))
    ) == (
        "NEON dataset Breeding landbird point counts. Column siteID: NEON site code (string). other ontology term:"
    )
    assert fr.build_context(f9, target_state(card, "site", "environment", site=card.sites[1])) == (
        "NEON dataset Breeding landbird point counts. Site SRER: Santa Rita Experimental Range NEON, "
        "semi-arid desert grassland/shrubland. environment ontology term:"
    )
    assert fr.build_context(f9, target_state(card, "dataset", "taxon")) == (
        "NEON dataset Breeding landbird point counts: Count, distance from observer, and taxonomic "
        "identification of breeding landbirds observed during point counts. taxon ontology term:"
    )
    # The ontology_fits template is column-only: a site target has no template.
    with pytest.raises(fr.FramingError, match="not in this view"):
        fr.build_context(
            fr.framing("column.ontology_fits", "F9"),
            target_state(card, "site", "environment", site=card.sites[0]),
        )
    only_column = dataclasses.replace(f9, context_template={"column": "{name}"})
    with pytest.raises(fr.FramingError, match="no template for a site target"):
        fr.build_context(only_column, target_state(card, "site", "environment", site=card.sites[0]))


def test_template_fields_namespace(card: DatasetCard) -> None:
    fields = fr.template_fields(_target(card))
    assert fields["dataset"] == card.name and fields["title"] == card.product_title
    assert (
        fields["sites"] == "HARV, SRER"
        and fields["scope"] == "column"
        and fields["aspect"] == "measurement"
    )
    assert (
        fields["name"] == "observerDistance"
        and fields["unit"] == "meter"
        and fields["dtype"] == "real"
    )
    assert "code" not in fields and "term_label" not in fields
    site = fr.template_fields(target_state(card, "site", "environment", site=card.sites[0]))
    assert site["code"] == "HARV" and site["domain"] == "D01" and "name" not in site
    vk = fr.template_fields(
        value_kind_state(card, card.columns[0], {"label": "L", "curie": "C:1"}, "x")
    )
    assert vk["term_label"] == "L" and vk["term_curie"] == "C:1"
    with pytest.raises(fr.FramingError, match="no card header"):
        fr.template_fields({"scope": "column"})


def test_context_text_is_the_state_head_input(card: DatasetCard) -> None:
    st = _target(card)
    f7, f4 = fr.framing("term.fits", "F7"), fr.framing("term.fits", "F4")
    assert fr.context_text(f7, st) == render.to_text(st)
    assert fr.context_text(f4, st) == render.to_text(st) + "\n\n" + str(f4.instructions)
    assert fr.context_text(f7, st).endswith("column:\n" + render.to_text(st["column"], 2))


# -- questions and requests ----------------------------------------------------------------------


def test_build_question_rank_fit_keys_candidates_and_anchor_last() -> None:
    f7 = fr.framing("term.fits", "F7")
    q = fr.build_question(f7, CANDS)
    assert list(q) == ["type", "instructions", "criteria"]  # CLM client's Choice.to_dict order
    assert q["type"] == "choice" and q["instructions"] is None
    assert list(q["criteria"]) == ["PATO:0000040", "UO:0000008", "ENVO:00000001", ANCHOR_KEY]
    assert q["criteria"]["PATO:0000040"] == "distance: " + "x" * MAX_DESCRIPTION
    assert q["criteria"]["UO:0000008"] == "meter: a length unit"
    assert q["criteria"]["ENVO:00000001"] == "bare label"  # empty description: label alone
    assert q["criteria"][ANCHOR_KEY] == ANCHORS["term"]
    keys, texts = render.candidates(q)
    assert keys == list(q["criteria"]) and texts == list(q["criteria"].values())
    assert (
        fr.build_question(fr.framing("term.fits", "F4"), CANDS)["instructions"]
        == fr.framing("term.fits", "F4").instructions
    )


def test_build_question_rejects_bad_candidate_sets() -> None:
    f7 = fr.framing("term.fits", "F7")
    with pytest.raises(fr.FramingError, match="at least one"):
        fr.build_question(f7, [])
    with pytest.raises(fr.FramingError, match="at least one"):
        fr.build_question(f7, None)
    with pytest.raises(fr.FramingError, match="duplicate"):
        fr.build_question(f7, [CANDS[0], CANDS[0]])
    with pytest.raises(fr.FramingError, match="anchor key"):
        fr.build_question(f7, [fr.FramingCandidate(ANCHOR_KEY, "none")])
    with pytest.raises(fr.FramingError, match="no key"):
        fr.build_question(f7, [fr.FramingCandidate("", "none")])
    with pytest.raises(fr.FramingError, match="use build_requests"):
        fr.build_question(fr.framing("term.fits", "F1"), CANDS)
    with pytest.raises(fr.FramingError, match="takes no candidates"):
        fr.build_question(fr.active_framing("column.aspect"), CANDS)
    with pytest.raises(fr.FramingError, match="only a control"):
        fr.noul_question(f7)


def test_build_question_closed_choices() -> None:
    q = fr.build_question(fr.active_framing("column.annotate"))
    assert q == {"type": "choice", "instructions": None, "criteria": dict(ANNOTATE_OPTIONS)}
    q = fr.build_question(fr.active_framing("column.aspect"))
    assert list(q["criteria"]) == list(ASPECTS) and list(q["criteria"].values()) == list(
        ASPECT_OPTIONS
    )
    q = fr.build_question(fr.active_framing("avu.value_kind"))
    assert list(q["criteria"].values()) == list(VALUE_KINDS)
    assert ANCHOR_KEY not in q["criteria"]


def test_build_request_returns_one_request_per_group(card: DatasetCard) -> None:
    st = _target(card)
    f7 = fr.active_framing("term.fits")
    ctx, qs = fr.build_request(f7, st, CANDS)
    assert ctx == st and list(qs) == ["term.fits"]
    pairs = render.build_pairs(ctx, qs)
    state_text, keys, texts = pairs["term.fits"]
    assert state_text == render.to_text(st) and keys[-1] == ANCHOR_KEY and len(texts) == 4
    assert fr.build_requests(f7, st, CANDS) == [(ctx, qs)]
    ann = fr.active_framing("column.annotate")
    ctx2, qs2 = fr.build_request(ann, st)
    assert ctx2 == column_state(card, card.column("observerDistance")) and list(qs2) == [
        "column.annotate"
    ]
    with pytest.raises(fr.FramingError, match="use build_requests"):
        fr.build_request(fr.framing("term.fits", "F1"), st, CANDS)


def test_f1_control_asks_one_noul_per_candidate_over_the_anyjev_state(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st = _target(card)
    f1 = fr.framing("term.fits", "F1")
    reqs = fr.build_requests(f1, st, CANDS)
    assert len(reqs) == 3
    for (ctx, qs), c in zip(reqs, CANDS, strict=True):
        assert isinstance(ctx, dict)
        assert list(ctx) == ["card", "scope", "column", "aspect", "candidate", "n_candidates"]
        expected = candidate_state(
            card,
            "column",
            col,
            "measurement",
            {
                "label": c.label,
                "curie": c.key,
                "ontology_id": c.ontology_id,
                "description": c.description,
                "synonyms": list(c.synonyms),
                "has_children": c.has_children,
            },
            3,
        )
        assert ctx == expected and state_sha256(ctx) == state_sha256(
            expected
        )  # joins imported labels
        assert qs == {c.key: {"type": "noul", "instructions": TASKS["term.fits"].text}}
        state_text, keys, texts = render.build_pairs(ctx, qs)[c.key]
        assert state_text.endswith("\n\n" + TASKS["term.fits"].text) and keys == ["false", "true"]
        assert texts[1].startswith("true: Yes. This is true: ")
    # A site target keeps the anyjev order too (card, scope, site, aspect, candidate, n_candidates).
    site_st = target_state(card, "site", "environment", site=card.sites[0])
    ((ctx, _),) = fr.build_requests(f1, site_st, CANDS[:1])
    assert list(ctx) == ["card", "scope", "site", "aspect", "candidate", "n_candidates"]
    assert ctx == candidate_state(
        card,
        "site",
        card.sites[0],
        "environment",
        {
            "label": "distance",
            "curie": "PATO:0000040",
            "ontology_id": "pato",
            "description": "x" * 500,
            "synonyms": ["a", "b", "c", "d", "e", "f"],
            "has_children": True,
        },
        1,
    )
    with pytest.raises(fr.FramingError, match="at least one"):
        fr.build_requests(f1, st, [])
    with pytest.raises(fr.FramingError, match="unique"):
        fr.build_requests(f1, st, [CANDS[0], CANDS[0]])
    with pytest.raises(fr.FramingError, match="not a control"):
        fr.control_state(fr.framing("term.fits", "F7"), st, CANDS[0], 1)


def test_f1_control_for_ontology_fits_is_the_anyjev_ontology_state(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st = _target(card)
    f1 = fr.framing("column.ontology_fits", "F1")
    cands = fr.ontology_candidates(["pato", "obi"])
    reqs = fr.build_requests(f1, st, cands)
    assert [next(iter(qs)) for _, qs in reqs] == ["pato", "obi"]
    for (ctx, qs), c in zip(reqs, cands, strict=True):
        e = entry(c.key)
        expected = ontology_state(card, col, "measurement", e.id, e.option_text, sorted(e.aspects))
        assert ctx == expected and list(ctx) == list(expected)
        assert qs[c.key] == {"type": "noul", "instructions": TASKS["column.ontology_fits"].text}
    with pytest.raises(fr.FramingError, match="not a registry ontology"):
        fr.build_requests(f1, st, [fr.FramingCandidate("go", "GO")])


def test_ontology_candidates_reproduce_the_registry_option_texts() -> None:
    cands = fr.ontology_candidates()
    assert [c.key for c in cands] == [e.id for e in ONTOLOGY_REGISTRY]
    # F7 and F9 (the A1 framing) share the candidate template: the same option texts.
    for f in (fr.framing("column.ontology_fits", "F7"), fr.active_framing("column.ontology_fits")):
        assert [fr.candidate_text(f, c) for c in cands] == list(ONTOLOGY_OPTIONS)
        q = fr.build_question(f, cands)
        assert list(q["criteria"]) == [*[e.id for e in ONTOLOGY_REGISTRY], ANCHOR_KEY]
        assert q["criteria"][ANCHOR_KEY] == ANCHORS["ontology"]
    assert [c.key for c in fr.ontology_candidates(["UO", "envo"])] == ["envo", "uo"]
    with pytest.raises(fr.FramingError, match="unknown registry"):
        fr.ontology_candidates(["go"])
    with pytest.raises(fr.FramingError, match="does not start with"):
        fr.FramingCandidate.from_registry(
            dataclasses.replace(ONTOLOGY_REGISTRY[0], option_text="x")
        )


def test_framing_candidate_from_term_matches_the_ols_state_shape() -> None:
    term = {
        "label": 42,
        "curie": "ENVO:1",
        "description": None,
        "synonyms": None,
        "has_children": 1,
    }
    c = fr.FramingCandidate.from_term(term)
    assert c == fr.FramingCandidate("ENVO:1", "42", "None", "", (), True)
    assert fr.candidate_text(fr.active_framing("term.fits"), c) == "42: None"


# -- lock ----------------------------------------------------------------------------------------


def test_lock_matches_code() -> None:
    assert fr.LOCK_PATH.name == "framings.lock.json" and fr.LOCK_PATH.exists()
    assert fr.lock_drift() == []
    locked = fr.read_lock()
    assert fr.lock_sha() == locked["lock_sha"]
    assert locked["schema_sha256"] == render.schema_sha256()


def test_lock_payload_shape() -> None:
    payload = fr.lock_payload()
    assert list(payload) == ["schema_sha256", "tasks", "lock_sha"]
    assert set(payload["tasks"]) == set(fr.FRAMINGS)
    for task_id, entry_ in payload["tasks"].items():
        assert list(entry_) == ["task_key", "shape", "active_framing", "framings"]
        assert entry_["task_key"] == TASKS[task_id].key
        assert entry_["active_framing"] == A1_ACTIVE[task_id]
        for fid, f in entry_["framings"].items():
            framing_ = fr.framing(task_id, fid)
            assert f["question_key"] == framing_.question_key
            assert f["shape"] == entry_["shape"]
            assert set(f) == {
                "question_key",
                *(
                    x.name
                    for x in dataclasses.fields(fr.Framing)
                    if x.name not in ("id", "task_id")
                ),
            }
    assert payload["tasks"]["term.fits"]["framings"]["F1"]["control"] is True
    assert payload["tasks"]["term.fits"]["framings"]["F7"]["control"] is False
    json.dumps(payload)  # JSON-safe (tuples became lists)


def test_lock_drift_detection(tmp_path: Path) -> None:
    path = tmp_path / "framings.lock.json"
    assert fr.lock_drift(path) == [
        "framings.lock.json is missing; run `mesa-clm framings --update-lock`"
    ]
    sha = fr.write_lock(path)
    assert sha == fr.lock_sha() and fr.lock_drift(path) == []
    assert path.read_text(encoding="utf-8").endswith("}\n")

    def tampered(mutate: Callable[[dict[str, Any]], None]) -> list[str]:
        fr.write_lock(path)  # each scenario starts from a clean lock
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        data["lock_sha"] = fr._sha({k: v for k, v in data.items() if k != "lock_sha"})
        path.write_text(json.dumps(data), encoding="utf-8")
        return fr.lock_drift(path)

    def rotate(d: dict[str, Any]) -> None:
        f = d["tasks"]["term.fits"]["framings"]["F7"]
        f["question_key"], f["instructions"] = "deadbeefdeadbeef", "Edited?"

    problems = tampered(rotate)
    assert len(problems) == 1
    assert problems[0].startswith("term.fits/F7: question_key deadbeefdeadbeef -> ")
    assert "changed: instructions, question_key" in problems[0] and "labels do not" in problems[0]

    def drop_task(d: dict[str, Any]) -> None:
        del d["tasks"]["avu.value_kind"]

    assert tampered(drop_task) == ["avu.value_kind: new task, not in the lock"]

    def extra(d: dict[str, Any]) -> None:
        t = d["tasks"]
        t["avu.keep"] = {"task_key": "x", "shape": "choice", "active_framing": "F7", "framings": {}}
        t["term.fits"]["framings"]["F2"] = dict(t["term.fits"]["framings"]["F7"])
        del t["term.fits"]["framings"]["F9"]
        t["term.fits"]["active_framing"] = "F4"
        t["term.fits"]["task_key"] = "0" * 16
        t["column.aspect"]["shape"] = "rank_fit"
        t["column.annotate"]["framings"]["F7"]["active"] = False

    problems = tampered(extra)
    assert "avu.keep: in the lock but no longer framed" in problems
    assert "term.fits/F2: in the lock but no longer defined" in problems
    assert "term.fits/F9: new framing, not in the lock" in problems
    assert any(p.startswith("term.fits: active framing F4 -> F7") for p in problems)
    assert any(
        p.startswith("term.fits: task_key 0000000000000000 -> 0ccc8d141ffd30ff") for p in problems
    )
    assert "column.aspect: shape rank_fit -> choice" in problems
    assert "column.annotate/F7: control/active flags changed" in problems

    def schema(d: dict[str, object]) -> None:
        d["schema_sha256"] = "f" * 64

    assert [p for p in tampered(schema) if p.startswith("schema_sha256 ff")]
    # A hand edit without recomputing lock_sha is caught as tampering.
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lock_sha"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    assert any("lock_sha does not match" in p for p in fr.lock_drift(path))
