"""DESIGN D23 over the seven fixture cards: every production context view ends with the target
(the column's name and description, the site code, or the aspect of a dataset-scope target;
for ``avu.value_kind`` the term the AVU is built from), never contains the raw card text, and
stays under the client token guard (chars/2 < 4096 − 16). The F1 control arms end with the
candidate by design (the mesa-anyjev shape) and are checked for exactly that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mesa_clm import framings as fr
from mesa_clm import render
from mesa_clm.cards import DatasetCard, load_card
from mesa_clm.registry import ASPECTS
from mesa_clm.states import column_state, target_state, value_kind_state

MAX_LEN = 4096
GUARD = 16
# The longest plausible candidate for the control arms and the value-kind term: worst-case tails.
CAND = fr.FramingCandidate(
    "PATO:0000040", "distance", "d" * 400, "pato", ("s1", "s2", "s3", "s4", "s5", "s6"), True
)
TERM = {"label": "distance", "curie": "PATO:0000040"}
PRODUCTION = [f for arms in fr.FRAMINGS.values() for f in arms.values() if not f.control]
CONTROLS = [f for arms in fr.FRAMINGS.values() for f in arms.values() if f.control]


def _text(ctx: fr.Context) -> str:
    return ctx if isinstance(ctx, str) else render.to_text(ctx)


def _last_block(text: str) -> str:
    """The last top-level field of a ``to_text`` rendering (top-level fields are separated by a
    blank line; nested ones only by newlines)."""
    return text.rsplit("\n\n", 1)[-1]


def _assert_no_raw_card(text: str, card_text: str) -> None:
    assert card_text not in text
    assert "# Dataset:" not in text and "## Columns" not in text and "Source:" not in text
    for line in card_text.splitlines():
        if line.startswith("- ") and "|" in line:
            assert line not in text  # a raw column row never reaches the encoder


def _assert_tail_is(text: str, target: str, fixed_tail: str | None) -> None:
    """``target`` is the last content in ``text``: nothing but the framing's fixed template tail
    (or nothing at all) follows it."""
    assert target in text, target
    after = text[text.rfind(target) + len(target) :]
    if fixed_tail is None:
        assert after == "", after
    else:
        assert after == fixed_tail, after


def _assert_guard(framing: fr.Framing, state: dict[str, object]) -> None:
    text = fr.context_text(framing, state)
    assert len(text) / 2.0 < MAX_LEN - GUARD, (framing.id, len(text))


@pytest.fixture(scope="module")
def cards(fixture_cards: list[Path]) -> list[tuple[DatasetCard, str]]:
    assert len(fixture_cards) == 7
    return [(load_card(p), p.read_text(encoding="utf-8")) for p in fixture_cards]


@pytest.fixture(scope="module")
def fixture_cards() -> list[Path]:
    return sorted((Path(__file__).resolve().parents[1] / "fixtures" / "cards").glob("*.md"))


def test_every_production_view_ends_with_the_column(cards: list[tuple[DatasetCard, str]]) -> None:
    for card, raw in cards:
        for col in card.columns:
            for aspect in ("measurement", "taxon"):
                target = target_state(card, "column", aspect, column=col)
                column_block = render.to_text(target["column"], 2)  # the nested rendering
                for f in PRODUCTION:
                    if f.context_view == "value_kind_state":
                        state = value_kind_state(card, col, TERM, aspect)
                    elif f.context_view == "column_state":
                        state = column_state(card, col)
                    else:
                        state = target
                    ctx = fr.build_context(f, state)
                    text = _text(ctx)
                    _assert_no_raw_card(text, raw)
                    _assert_guard(f, state)
                    if f.context_view == "value_kind_state":
                        # The target of avu.value_kind is (column, term): the term closes the
                        # view and the column is the block before the aspect.
                        assert (
                            _last_block(text) == "term:\n  label: distance\n  curie: PATO:0000040"
                        )
                        blocks = text.split("\n\n")
                        assert blocks[-3] == "column:\n" + column_block
                        assert blocks[-2] == f"aspect: {aspect}"
                    elif isinstance(ctx, dict):
                        assert _last_block(text) == "column:\n" + column_block
                        assert (
                            f"  name: {col.name}\n  description: {col.description}"
                            in _last_block(text)
                        )
                    else:
                        unit = col.unit or col.dtype
                        _assert_tail_is(
                            text,
                            f"Column {col.name}: {col.description}",
                            f" ({unit}). {aspect} ontology term:",
                        )
                        assert text.startswith(f"NEON dataset {card.product_title}. ")
                        # No other column's name follows the target (the card header is not in a
                        # template context at all).
                        assert not any(
                            f"Column {other.name}:" in text
                            for other in card.columns
                            if other is not col
                        )


def test_site_and_dataset_targets_end_with_the_target(cards: list[tuple[DatasetCard, str]]) -> None:
    fits = [f for f in fr.framings_for("term.fits") if not f.control]
    for card, raw in cards:
        assert card.sites, card.name
        for site in card.sites:
            state = target_state(card, "site", "environment", site=site)
            for f in fits:
                ctx = fr.build_context(f, state)
                text = _text(ctx)
                _assert_no_raw_card(text, raw)
                _assert_guard(f, state)
                if isinstance(ctx, dict):
                    assert _last_block(text) == "site:\n" + render.to_text(state["site"], 2)
                    assert _last_block(text).startswith(f"site:\n  code: {site.code}\n")
                else:
                    _assert_tail_is(
                        text,
                        f"Site {site.code}: {site.name}, {site.habitat}",
                        ". environment ontology term:",
                    )
        state = target_state(card, "dataset", "taxon")
        for f in fits:
            ctx = fr.build_context(f, state)
            text = _text(ctx)
            _assert_no_raw_card(text, raw)
            _assert_guard(f, state)
            if isinstance(ctx, dict):
                # A dataset-scope target is the card itself: the view ends with the aspect.
                assert _last_block(text) == "aspect: taxon"
                assert text.startswith(f"card:\n  dataset: {card.name}\n")
            else:
                _assert_tail_is(
                    text,
                    f"{card.product_title}: {card.product_description}",
                    ". taxon ontology term:",
                )


def test_control_arms_end_with_the_candidate_not_the_target(
    cards: list[tuple[DatasetCard, str]],
) -> None:
    """The F1 control keeps the mesa-anyjev layout on purpose: its context ends with the
    candidate (and ``n_candidates``), which is exactly the tail-weighting X1 measures against."""
    for card, raw in cards:
        col = card.columns[-1]
        state = target_state(card, "column", "measurement", column=col)
        for f in CONTROLS:
            cands = (
                fr.ontology_candidates(["pato"]) if f.task_id == "column.ontology_fits" else [CAND]
            )
            for ctx, questions in fr.build_requests(f, state, cands):
                text = _text(ctx)
                _assert_no_raw_card(text, raw)
                state_text = render.state_text(ctx, f.instructions)
                assert len(state_text) / 2.0 < MAX_LEN - GUARD
                assert state_text.endswith("\n\n" + str(f.instructions))
                assert list(questions) == [cands[0].key]
                if f.task_id == "term.fits":
                    assert _last_block(text) == "n_candidates: 1"
                    assert text.split("\n\n")[-2].startswith("candidate:\n  label: distance\n")
                else:
                    assert _last_block(text).startswith("ontology:\n  id: pato\n")


def test_token_guard_headroom_on_the_fixture_cards(cards: list[tuple[DatasetCard, str]]) -> None:
    """chars/2 is the conservative token estimate (cards measure 2.4–2.7 chars/token, RESEARCH.md);
    the largest fixture context must stay well under ``max_len − 16`` with the longest arm."""
    longest = 0
    for card, _ in cards:
        for col in card.columns:
            for aspect in ASPECTS:
                state = target_state(card, "column", aspect, column=col)
                for f in fr.framings_for("term.fits"):
                    if f.control:
                        ctx = fr.control_state(f, state, CAND, 12)
                        text = render.state_text(fr.build_context(f, ctx), f.instructions)
                    else:
                        text = fr.context_text(f, state)
                    longest = max(longest, len(text))
    assert longest / 2.0 < MAX_LEN - GUARD
    assert longest > 1000  # the check is not vacuous: full card headers are in the contexts
