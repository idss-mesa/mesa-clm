"""Label identity (``mesa_clm.identity``, DESIGN D1): the target key is independent of
``n_candidates`` and of the card header's volatile fields, the option key names the candidate
for the rank_fit tasks and is empty for closed choices, and every anyjev-shaped state built by
``mesa_clm.states`` maps to an identity."""

from __future__ import annotations

import hashlib
import json

import pytest

from mesa_clm.cards import DatasetCard
from mesa_clm.identity import (
    IdentityError,
    LabelIdentity,
    identity,
    option_key,
    target_key,
    target_sha256,
)
from mesa_clm.registry import ANCHOR_KEY, ONTOLOGY_REGISTRY
from mesa_clm.states import (
    candidate_state,
    column_state,
    ontology_state,
    target_state,
    target_state_from,
    value_kind_state,
)
from mesa_clm.tasks import TASKS

CAND = {
    "label": "distance",
    "curie": "PATO:0000040",
    "ontology_id": "pato",
    "description": "d",
    "synonyms": [],
    "has_children": True,
}


def test_column_scope_target_key(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st = column_state(card, col)
    assert target_key("column.annotate", st) == {
        "dataset": "DP1.10003.001.brd_countdata",
        "scope": "column",
        "target": "observerDistance",
    }
    assert target_key("column.aspect", st) == target_key("column.annotate", st)
    assert len(target_sha256("column.annotate", st)) == 64


def test_candidate_state_target_key_per_scope(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st_col = candidate_state(card, "column", col, "measurement", CAND, 12)
    assert target_key("term.fits", st_col) == {
        "dataset": card.name,
        "scope": "column",
        "target": "observerDistance",
        "aspect": "measurement",
    }
    st_site = candidate_state(card, "site", card.sites[1], "environment", CAND, 3)
    assert target_key("term.fits", st_site) == {
        "dataset": card.name,
        "scope": "site",
        "target": "SRER",
        "aspect": "environment",
    }
    st_ds = candidate_state(card, "dataset", None, "taxon", CAND, 5)
    assert target_key("term.fits", st_ds) == {
        "dataset": card.name,
        "scope": "dataset",
        "target": card.name,
        "aspect": "taxon",
    }


def test_target_is_independent_of_n_candidates_and_candidate(card: DatasetCard) -> None:
    """D1 rationale: one (target, candidate) pair must not hash differently per group size."""
    col = card.column("observerDistance")
    a = candidate_state(card, "column", col, "measurement", CAND, 3)
    b = candidate_state(card, "column", col, "measurement", CAND, 12)
    other = candidate_state(card, "column", col, "measurement", {**CAND, "curie": "PATO:1"}, 12)
    assert a != b
    assert target_sha256("term.fits", a) == target_sha256("term.fits", b)
    assert target_sha256("term.fits", a) == target_sha256("term.fits", other)
    assert identity("term.fits", a) == identity("term.fits", b)
    assert identity("term.fits", a) != identity("term.fits", other)


def test_target_ignores_volatile_card_header_fields(card: DatasetCard) -> None:
    """Row counts, months and profiles change between card builds; the label does not."""
    col = card.column("observerDistance")
    st = column_state(card, col)
    rebuilt = card.model_copy(update={"rows": card.rows + 1, "months_to": "2030-01"})
    st2 = column_state(rebuilt, col.model_copy(update={"profile": "changed"}))
    assert st != st2
    assert target_sha256("column.annotate", st) == target_sha256("column.annotate", st2)


def test_target_state_view_shares_the_identity(card: DatasetCard) -> None:
    """The mesa-clm context view and the mesa-anyjev candidate state name the same target."""
    col = card.column("observerDistance")
    anyjev = candidate_state(card, "column", col, "measurement", CAND, 7)
    view = target_state(card, "column", "measurement", column=col)
    assert target_key("term.fits", view) == target_key("term.fits", anyjev)
    assert target_key("term.fits", target_state_from(anyjev)) == target_key("term.fits", anyjev)
    site_view = target_state(card, "site", "environment", site=card.sites[0])
    assert target_key("term.fits", site_view)["target"] == "HARV"


def test_ontology_fits_identity(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    e = ONTOLOGY_REGISTRY[2]  # pato
    st = ontology_state(card, col, "measurement", e.id, e.option_text, sorted(e.aspects))
    assert target_key("column.ontology_fits", st) == {
        "dataset": card.name,
        "scope": "column",
        "target": "observerDistance",
        "aspect": "measurement",
    }
    assert option_key("column.ontology_fits", st) == "pato"
    assert identity("column.ontology_fits", st) == LabelIdentity(
        TASKS["column.ontology_fits"].key, target_sha256("column.ontology_fits", st), "pato"
    )


def test_value_kind_identity_carries_the_term(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st = value_kind_state(card, col, CAND, "measurement")
    assert target_key("avu.value_kind", st) == {
        "dataset": card.name,
        "scope": "avu",
        "target": "observerDistance",
        "aspect": "measurement",
        "term": "PATO:0000040",
    }
    assert option_key("avu.value_kind", st) == ""
    other = value_kind_state(card, col, {**CAND, "curie": "PATO:1"}, "measurement")
    assert target_sha256("avu.value_kind", st) != target_sha256("avu.value_kind", other)


def test_option_keys(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    st = candidate_state(card, "column", col, "measurement", CAND, 1)
    assert option_key("term.fits", st) == "PATO:0000040"
    assert option_key("term.fits", st, "ignored") == "PATO:0000040"
    assert option_key("column.annotate", column_state(card, col)) == ""
    assert option_key("column.aspect", column_state(card, col), "anything") == ""
    view = target_state(card, "column", "measurement", column=col)
    assert option_key("term.fits", view, ANCHOR_KEY) == ANCHOR_KEY
    assert option_key("term.fits", view, "ENVO:1") == "ENVO:1"
    assert option_key("column.ontology_fits", view, "envo") == "envo"
    with pytest.raises(IdentityError, match="names no candidate"):
        option_key("term.fits", view)


def test_target_sha256_is_the_canonical_json_hash(card: DatasetCard) -> None:
    st = column_state(card, card.column("siteID"))
    payload = json.dumps(
        target_key("column.annotate", st), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert target_sha256("column.annotate", st) == hashlib.sha256(payload.encode()).hexdigest()


def test_errors(card: DatasetCard) -> None:
    col = card.column("siteID")
    with pytest.raises(IdentityError, match="unknown task"):
        target_key("no.such", column_state(card, col))
    with pytest.raises(IdentityError, match="inactive"):
        target_key("avu.keep", column_state(card, col))
    with pytest.raises(IdentityError, match="no card header"):
        target_key("column.annotate", {"column": {"name": "x"}})
    with pytest.raises(IdentityError, match="needs a column"):
        target_key("column.annotate", {"card": {"dataset": "d"}})
    with pytest.raises(IdentityError, match="needs a site"):
        target_key("term.fits", {"card": {"dataset": "d"}, "scope": "site"})
    with pytest.raises(IdentityError, match="both a column and a site"):
        target_key("term.fits", {"card": {"dataset": "d"}, "column": {"name": "c"}, "site": {}})
    with pytest.raises(IdentityError, match="unknown scope"):
        target_key("term.fits", {"card": {"dataset": "d"}, "scope": "galaxy"})
    with pytest.raises(IdentityError, match="needs a term"):
        target_key("avu.value_kind", {"card": {"dataset": "d"}, "column": {"name": "c"}})
