"""Masking is idempotent and the zero-shot invariant accounts for masked records (plan §4.2 Q3,
§4.6; the bug the M1 pipeline hit).

``column.ontology_fits`` is masked twice: the provider applies the framing's ``mask_rule``
(the aspect) after scoring, and the pipeline then masks by the ontologies in play. When the
second mask removed everything the first one kept, ``apply_mask`` left ``probs`` at the first
mask's renormalisation, and ``_check_mask`` rejected the result ("a zero_shot record's probs
are CLM's raw_probs, untransformed") because it compared against ``raw_probs`` untransformed.
The invariant now reads the support off ``probs``: at zero shot ``probs`` is ``raw_probs``
restricted to its own support and renormalised, and that support contains every option the flags
keep. Tampering is still caught.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest
from pydantic import ValidationError

from mesa_clm.cards import DatasetCard
from mesa_clm.framings import active_framing, ontology_candidates
from mesa_clm.policy import Policy, masked, masked_for_aspect
from mesa_clm.providers import DecisionRecord, FakeProvider, apply_mask
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.states import target_state

FRAMING = active_framing("column.ontology_fits")


@pytest.fixture
def onto(card: DatasetCard) -> DecisionRecord:
    """The repro: an ontology rank the provider already masked by the ``measurement`` aspect."""
    col = card.column("observerDistance")
    state = target_state(card, "column", "measurement", column=col)
    [rec] = FakeProvider().decide(FRAMING, [state], [ontology_candidates()])
    assert rec.masked is not None and any(rec.masked), "the provider masks by aspect"
    assert rec.level == "zero_shot" and rec.probs != rec.raw_probs
    return rec


def _dump(rec: DecisionRecord) -> dict[str, Any]:
    return rec.model_dump()


def test_masking_an_already_masked_record_to_nothing_abstains(onto: DecisionRecord) -> None:
    """The exact repro from the pipeline: every option out on top of the provider's mask."""
    empty = apply_mask(onto, [False] * onto.k)
    assert empty.answer_index == -1 and empty.answer == ""
    assert empty.reason == "mask_empty" and empty.diagnostics["mask_empty"] is True
    assert empty.masked == [i != empty.anchor_index for i in range(empty.k)]
    # ``probs`` is left as the provider's renormalisation, never pushed onto the anchor alone.
    assert empty.probs == onto.probs
    assert Policy.load().verdict(empty).reason == "mask_empty"


def test_apply_mask_is_idempotent(onto: DecisionRecord) -> None:
    keeps = [
        [True] * onto.k,
        [False] * onto.k,
        [o in ("pato", "envo", ANCHOR_KEY) for o in onto.options],
        [o == "obi" for o in onto.options],
        [o in ("ncbitaxon", "gaz") for o in onto.options],  # only already-masked ids
    ]
    for keep in keeps:
        once = apply_mask(onto, keep)
        twice = apply_mask(once, keep)
        assert _dump(twice) == _dump(once)
        assert _dump(apply_mask(twice, keep)) == _dump(once)


def test_masks_compose_like_their_conjunction(onto: DecisionRecord) -> None:
    """Masking by A then B gives B-then-A and the one mask A∧B (same flags, same answer, the
    same probabilities to 1e-12), whether or not the result is empty."""
    ids = [o for o in onto.options if o != ANCHOR_KEY]
    subsets = [frozenset(c) for n in (0, 1, 3, 6) for c in itertools.combinations(ids, n)][:40]
    for a, b in itertools.product(subsets[::7], subsets[::5]):
        ka = [o in a for o in onto.options]
        kb = [o in b for o in onto.options]
        ab = apply_mask(apply_mask(onto, ka), kb)
        ba = apply_mask(apply_mask(onto, kb), ka)
        both = apply_mask(onto, [x and y for x, y in zip(ka, kb, strict=True)])
        for other in (ba, both):
            assert ab.masked == other.masked and ab.answer_index == other.answer_index
            if ab.answer_index >= 0:
                assert ab.probs is not None and other.probs is not None
                assert all(
                    math.isclose(p, q, abs_tol=1e-12)
                    for p, q in zip(ab.probs, other.probs, strict=True)
                )


def test_policy_masks_by_the_ontologies_in_play_after_the_provider(
    onto: DecisionRecord,
) -> None:
    """``masked_for_aspect`` on the provider's record: in-play ids that the aspect allows stay,
    an in-play set disjoint from the aspect's empties the record (no exception)."""
    kept = masked_for_aspect(onto, "measurement", frozenset({"pato"}))
    assert kept.answer == "pato" or kept.anchor_won
    assert [o for o, m in zip(kept.options, kept.masked or [], strict=True) if not m] == [
        "pato",
        ANCHOR_KEY,
    ]
    empty = masked_for_aspect(onto, "measurement", frozenset({"ncbitaxon"}))
    assert empty.answer_index == -1 and empty.reason == "mask_empty"
    assert _dump(masked_for_aspect(empty, "measurement", frozenset({"ncbitaxon"}))) == _dump(empty)
    assert _dump(masked(empty, [True] * empty.k)) == _dump(empty)


def test_the_invariant_still_rejects_transformed_probs(onto: DecisionRecord) -> None:
    probs = list(onto.probs or [])
    kept = [i for i, m in enumerate(onto.masked or []) if not m]
    assert len(kept) >= 3
    # Swap two kept probabilities: still a distribution, no longer CLM's.
    swapped = list(probs)
    i, j = kept[0], kept[1]
    swapped[i], swapped[j] = swapped[j], swapped[i]
    if not math.isclose(swapped[i], probs[i]):
        with pytest.raises(ValidationError, match="untransformed"):
            onto.revise(**_consistent(onto, swapped))
    # Zero an option the flags keep and renormalise: a silent, unflagged mask.
    dropped = list(probs)
    dropped[kept[0]] = 0.0
    total = math.fsum(dropped)
    dropped = [p / total for p in dropped]
    with pytest.raises(ValidationError, match="no mask removed"):
        onto.revise(**_consistent(onto, dropped))


def test_a_fully_masked_record_must_still_carry_a_renormalised_distribution(
    onto: DecisionRecord,
) -> None:
    empty = apply_mask(onto, [False] * onto.k)
    probs = list(empty.probs or [])
    nonzero = [i for i, p in enumerate(probs) if p > 0.0]
    bent = list(probs)
    bent[nonzero[0]] += 0.01
    bent[nonzero[1]] -= 0.01
    with pytest.raises(ValidationError, match="untransformed"):
        empty.revise(probs=bent, **_stats(bent))
    # Pushing the whole mass onto the anchor alone is not the renormalisation either (it would
    # fabricate a one-hot distribution) and is refused.
    one_hot = [1.0 if i == empty.anchor_index else 0.0 for i in range(empty.k)]
    with pytest.raises(ValidationError, match="one option"):
        empty.revise(probs=one_hot, **_stats(one_hot))


def _stats(probs: list[float]) -> dict[str, float]:
    ordered = sorted(probs, reverse=True)
    return {"confidence": ordered[0], "margin": ordered[0] - ordered[1]}


def _consistent(rec: DecisionRecord, probs: list[float]) -> dict[str, Any]:
    """``probs`` with the answer, confidence and margin that follow from it (so only the
    zero-shot rule can object)."""
    flags = rec.masked or [False] * rec.k
    idx = max((i for i in range(rec.k) if not flags[i]), key=lambda i: probs[i])
    return {"probs": probs, "answer_index": idx, "answer": rec.options[idx], **_stats(probs)}
