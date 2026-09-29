"""State builders (``mesa_clm.states``): ported from mesa-anyjev ``tests/test_states.py`` plus
the mesa-clm ``target_state`` view (DESIGN D23: the target comes last)."""

from __future__ import annotations

import hashlib
import json

import pytest

from mesa_clm.cards import DatasetCard
from mesa_clm.states import (
    MAX_DESCRIPTION,
    MAX_PROFILE,
    MAX_SIBLINGS,
    avu_state,
    candidate_state,
    card_header,
    chooser_state,
    column_state,
    datacite_state,
    dataset_ontology_state,
    ontology_state,
    state_sha256,
    target_state,
    target_state_from,
    value_kind_state,
)

CAND = {
    "label": "distance",
    "curie": "PATO:0000040",
    "ontology_id": "pato",
    "description": "x" * 500,
    "synonyms": ["a", "b", "c", "d", "e", "f"],
    "has_children": True,
}


def _json_safe(obj: object) -> None:
    json.dumps(obj)  # raises on numpy, datetime, sets


def test_states_are_json_and_share_the_card_header(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    states = [
        column_state(card, col),
        candidate_state(card, "column", col, "measurement", CAND, 7),
        candidate_state(card, "site", card.sites[0], "environment", CAND, 3),
        value_kind_state(card, col, CAND, "measurement"),
        avu_state(
            card,
            {"attribute": "pato.distance", "value": "distance", "unit": "PATO:0000040"},
            CAND,
            "column",
            col.name,
            [],
        ),
        chooser_state("pato", "distance", CAND, 8),
    ]
    for s in states:
        _json_safe(s)
        assert len(state_sha256(s)) == 64
    header = card_header(card)
    for s in states[:5]:
        assert s["card"] == header
    assert len(states[1]["candidate"]["description"]) == MAX_DESCRIPTION == 300
    assert len(states[1]["candidate"]["synonyms"]) == 5
    assert "site" in states[2] and "column" not in states[2]


def test_state_hash_is_key_order_independent(card: DatasetCard) -> None:
    a = {"card": card_header(card), "x": 1}
    b = {"x": 1, "card": card_header(card)}
    assert state_sha256(a) == state_sha256(b)


def test_state_sha256_formula_keeps_non_ascii() -> None:
    state = {"b": "é", "a": [1, {"z": None, "y": True}]}
    payload = '{"a":[1,{"y":true,"z":null}],"b":"é"}'
    assert state_sha256(state) == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_card_header_shape_and_order(card: DatasetCard) -> None:
    header = card_header(card)
    assert list(header) == [
        "dataset",
        "product_title",
        "product_description",
        "sites",
        "rows",
        "months",
        "columns",
    ]
    assert header["dataset"] == card.name
    assert header["months"] == "2020-04 to 2024-06"
    assert header["columns"] == [c.name for c in card.columns]
    assert header["sites"][0] == {
        "code": "HARV",
        "name": "Harvard Forest & Quabbin Watershed NEON",
        "domain": "D01",
        "habitat": "temperate deciduous/mixed forest",
    }
    assert "state" not in header["sites"][0]  # site state and domain name never enter a state


def test_column_profile_is_capped(card: DatasetCard) -> None:
    col = card.column("detectionMethod").model_copy(update={"profile": "p" * 1000})
    state = column_state(card, col)
    assert list(state) == ["card", "column"]
    assert list(state["column"]) == ["name", "description", "dtype", "unit", "profile"]
    assert len(state["column"]["profile"]) == MAX_PROFILE == 300


def test_candidate_state_coerces_and_orders_keys(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    sparse = {
        "label": 42,
        "curie": "ENVO:1",
        "description": None,
        "synonyms": None,
        "has_children": 1,
    }
    state = candidate_state(card, "column", col, "measurement", sparse, "7")  # type: ignore[arg-type]
    assert list(state) == ["card", "scope", "column", "aspect", "candidate", "n_candidates"]
    assert state["candidate"] == {
        "label": "42",
        "curie": "ENVO:1",
        "ontology_id": "",
        "description": "None",
        "synonyms": [],
        "has_children": True,
    }
    assert state["n_candidates"] == 7
    dataset = candidate_state(card, "dataset", None, "taxon", sparse, 0)
    assert list(dataset) == ["card", "scope", "aspect", "candidate", "n_candidates"]
    site = candidate_state(card, "site", card.sites[1], "environment", sparse, 2)
    assert site["site"] == {
        "code": "SRER",
        "name": "Santa Rita Experimental Range NEON",
        "domain": "D14",
        "habitat": "semi-arid desert grassland/shrubland",
    }


def test_ontology_state_sorts_aspects(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    state = ontology_state(card, col, "measurement", "obi", "OBI: ...", ["method", "data_type"])
    assert list(state) == ["card", "column", "aspect", "ontology"]
    assert state["ontology"] == {
        "id": "obi",
        "description": "OBI: ...",
        "aspects": ["data_type", "method"],
    }


def test_avu_state_caps_siblings_and_drops_extra_keys(card: DatasetCard) -> None:
    sib = [{"attribute": f"a{i}", "value": "v", "unit": "u", "extra": "x"} for i in range(40)]
    avu = {"attribute": "pato.distance", "value": "distance", "unit": "PATO:0000040", "k": "v"}
    state = avu_state(card, avu, CAND, "dataset", None, sib)
    assert list(state) == ["card", "avu", "term", "scope", "target", "siblings"]
    assert state["avu"] == {
        "attribute": "pato.distance",
        "value": "distance",
        "unit": "PATO:0000040",
    }
    assert state["target"] == ""
    assert len(state["siblings"]) == MAX_SIBLINGS == 25
    assert state["siblings"][0] == {"attribute": "a0", "value": "v", "unit": "u"}
    assert len(state["term"]["description"]) == MAX_DESCRIPTION


def test_dataset_ontology_and_datacite_states(card: DatasetCard) -> None:
    d = dataset_ontology_state(card, "ENVO: environments")
    assert list(d) == ["card", "ontology"] and d["ontology"] == "ENVO: environments"
    plain = datacite_state(None, "DateType", 2024)  # type: ignore[arg-type]
    assert plain == {"vocabulary": "DateType", "text": "2024"}
    full = datacite_state(card, "ResourceTypeGeneral", "t" * 5000, value="Dataset")
    assert list(full) == ["vocabulary", "text", "card", "value"]
    assert len(full["text"]) == MAX_DESCRIPTION * 4 == 1200
    assert full["card"] == card_header(card)


def test_chooser_state_carries_only_the_picker_fields() -> None:
    state = chooser_state("pato", "distance", CAND, 8)
    assert list(state) == ["ontology_id", "value", "candidate", "n_candidates"]
    assert list(state["candidate"]) == ["label", "curie", "description"]
    assert "card" not in state


# -- target_state (mesa-clm, DESIGN D23) -----------------------------------------------------


def test_target_state_column_comes_last(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    state = target_state(card, "column", "measurement", column=col)
    assert list(state) == ["card", "scope", "aspect", "column"]
    assert state["card"] == card_header(card)
    assert state["column"] == column_state(card, col)["column"]
    assert state["column"] == candidate_state(card, "column", col, "measurement", CAND, 1)["column"]
    _json_safe(state)
    assert len(state_sha256(state)) == 64


def test_target_state_site_comes_last(card: DatasetCard) -> None:
    site = card.sites[1]
    state = target_state(card, "site", "environment", site=site)
    assert list(state) == ["card", "scope", "aspect", "site"]
    assert state["site"] == card_header(card)["sites"][1]
    assert state["site"] == candidate_state(card, "site", site, "environment", CAND, 1)["site"]


def test_target_state_dataset_scope_ends_with_aspect(card: DatasetCard) -> None:
    state = target_state(card, "dataset", "taxon")
    assert list(state) == ["card", "scope", "aspect"]
    assert state == {"card": card_header(card), "scope": "dataset", "aspect": "taxon"}


def test_target_state_rejects_two_targets(card: DatasetCard) -> None:
    with pytest.raises(ValueError, match="not both"):
        target_state(card, "column", "taxon", column=card.columns[0], site=card.sites[0])


def test_target_state_is_shared_by_every_candidate_group(card: DatasetCard) -> None:
    col = card.column("observerDistance")
    target = target_state(card, "column", "measurement", column=col)
    shas = {
        state_sha256(target_state_from(candidate_state(card, "column", col, "measurement", c, n)))
        for c in (CAND, {"label": "other"})
        for n in (1, 5, 12)
    }
    assert shas == {state_sha256(target)}  # the candidate and n_candidates never enter it
    assert state_sha256(target) != state_sha256(
        candidate_state(card, "column", col, "measurement", CAND, 1)
    )


def test_target_state_from_projects_every_scope(card: DatasetCard) -> None:
    col, site = card.column("scientificName"), card.sites[0]
    for scope, target, kwargs in (
        ("column", col, {"column": col}),
        ("site", site, {"site": site}),
        ("dataset", None, {}),
    ):
        stored = candidate_state(card, scope, target, "taxon", CAND, 3)
        derived = target_state_from(stored)
        direct = target_state(card, scope, "taxon", **kwargs)
        assert derived == direct and list(derived) == list(direct)
    # Sorted-key JSON (as the sidecar stores it) projects to the same builder order.
    round_tripped = json.loads(
        json.dumps(candidate_state(card, "column", col, "taxon", CAND, 3), sort_keys=True)
    )
    assert list(target_state_from(round_tripped)) == ["card", "scope", "aspect", "column"]
    with pytest.raises(KeyError):
        target_state_from(column_state(card, col))  # no scope/aspect: not a target view
    with pytest.raises(ValueError, match="both"):
        target_state_from({"card": {}, "scope": "x", "aspect": "y", "column": {}, "site": {}})
