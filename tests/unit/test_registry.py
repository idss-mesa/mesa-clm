"""Frozen vocabularies (``mesa_clm.registry``): ported from mesa-anyjev ``tests/test_registry.py``
plus the mesa-anyjev lock cross-check, the mesa-mcp DataCite parity and the mesa-clm additions
(``ANCHORS``, ``ANNOTATE_OPTIONS``, ``NEON_ASPECT_MAP``)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mesa_clm.registry import (
    ANCHOR_KEY,
    ANCHORS,
    ANNOTATE_OPTIONS,
    ASPECT_OPTIONS,
    ASPECT_TO_NEON,
    ASPECTS,
    DATACITE_CONTRIBUTOR_TYPES,
    DATACITE_DATE_TYPES,
    DATACITE_DESCRIPTION_TYPES,
    DATACITE_RELATION_TYPES,
    DATACITE_VOCABULARIES,
    NEON_ASPECT_MAP,
    ONTOLOGY_OPTIONS,
    ONTOLOGY_REGISTRY,
    VALUE_KINDS,
    allowed_for_aspect,
    entry,
    mask_for_aspect,
    prefix_of,
)

LOCK_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "anyjev_questions.lock.json"

# neon-ducklake ``site/validate.py:108`` ``ASPECTS`` (fd7a0d5) and ``curation/prompt_product.md``.
NEON_ASPECTS = frozenset(
    {
        "organism_group",
        "measured_property",
        "environmental_material",
        "method",
        "data_type",
        "process",
    }
)


def test_registry_rule_d7() -> None:
    ids = [e.id for e in ONTOLOGY_REGISTRY]
    assert ids == [
        "envo",
        "ncbitaxon",
        "pato",
        "uo",
        "obi",
        "iao",
        "pco",
        "bco",
        "gaz",
        "ro",
        "taxrank",
        "genepio",
    ]
    assert len(ONTOLOGY_OPTIONS) == len(set(ONTOLOGY_OPTIONS)) == 12
    assert all(
        opt.split(":")[0] == e.curie_prefix
        for opt, e in zip(ONTOLOGY_OPTIONS, ONTOLOGY_REGISTRY, strict=True)
    )
    assert all(e.aspects <= set(ASPECTS) for e in ONTOLOGY_REGISTRY)


def test_aspects_and_value_kinds_frozen() -> None:
    assert len(ASPECTS) == len(ASPECT_OPTIONS) == 8
    assert all(opt.startswith(a + ":") for a, opt in zip(ASPECTS, ASPECT_OPTIONS, strict=True))
    assert len(VALUE_KINDS) == 4
    assert VALUE_KINDS[0] == "the term label"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ONTOLOGY_REGISTRY[0].id = "x"  # type: ignore[misc]


def test_prefix_of_canonical_case() -> None:
    assert prefix_of("ncbitaxon:8782") == "NCBITaxon"
    assert prefix_of("envo:01000178") == "ENVO"
    assert prefix_of("GO:0007601") == "GO"
    assert prefix_of("nocolon") == ""


def test_entry_is_case_insensitive() -> None:
    assert entry("ENVO").id == "envo" and entry("NCBITaxon").curie_prefix == "NCBITaxon"
    with pytest.raises(KeyError):
        entry("go")


def test_masks() -> None:
    assert allowed_for_aspect("unit") == {"uo"}
    assert allowed_for_aspect("other") == {e.id for e in ONTOLOGY_REGISTRY}
    assert allowed_for_aspect("no_such_aspect") == frozenset()
    m = mask_for_aspect("taxon", frozenset({"ncbitaxon", "envo"}))
    assert m.dtype == np.bool_ and m.shape == (12,)
    assert m.sum() == 1 and m[1]
    assert mask_for_aspect("environment").sum() == 2


def test_vocabularies_match_the_anyjev_lock() -> None:
    """Character-for-character parity with the mesa-anyjev question options (D1: same task_key)."""
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["questions"]
    assert lock["column.annotate"]["options"] == list(ANNOTATE_OPTIONS)
    assert lock["column.aspect"]["options"] == list(ASPECT_OPTIONS)
    assert lock["column.ontology"]["options"] == list(ONTOLOGY_OPTIONS)
    assert lock["avu.value_kind"]["options"] == list(VALUE_KINDS)
    assert lock["datacite.contributor_type"]["options"] == list(DATACITE_CONTRIBUTOR_TYPES)
    assert lock["datacite.relation_type"]["options"] == list(DATACITE_RELATION_TYPES)
    assert lock["datacite.date_type"]["options"] == list(DATACITE_DATE_TYPES)
    assert lock["datacite.description_type"]["options"] == list(DATACITE_DESCRIPTION_TYPES)
    assert lock["term.fits"]["key"] == "0ccc8d141ffd30ff"
    assert lock["column.annotate"]["key"] == "8b8c7f14be35925d"


def test_datacite_vocabularies_match_mesa_mcp() -> None:
    schema = pytest.importorskip("mesa_mcp.datacite.schema")
    for name, members in DATACITE_VOCABULARIES.items():
        assert tuple(m.value for m in getattr(schema, name)) == members, name
    assert len(DATACITE_VOCABULARIES["ResourceTypeGeneral"]) == 28


# -- mesa-clm additions -----------------------------------------------------------------------


def test_anchors() -> None:
    assert ANCHOR_KEY == "__none__"
    assert ANCHORS == {
        "term": "None of these terms is the right concept for this target.",
        "ontology": "None of these ontologies has a suitable term for this target.",
    }
    for text in ANCHORS.values():
        assert text.endswith(".") and text not in ONTOLOGY_OPTIONS


def test_annotate_options_keep_the_anyjev_labels() -> None:
    assert list(ANNOTATE_OPTIONS) == ["Yes", "No"]  # option index 0 = Yes, as in the lock
    yes, no = ANNOTATE_OPTIONS["Yes"], ANNOTATE_OPTIONS["No"]
    assert yes != no and yes.endswith(".") and no.endswith(".")
    assert "ontology annotation" in yes and "identifier" in no
    assert not {yes, no} & set(ANNOTATE_OPTIONS)  # sentences, not labels


def test_neon_aspect_map() -> None:
    assert set(NEON_ASPECT_MAP) == NEON_ASPECTS
    assert NEON_ASPECT_MAP == {
        "organism_group": "taxon",
        "measured_property": "measurement",
        "environmental_material": "environment",
        "method": "method",
        "data_type": "data_type",
        "process": None,
    }
    for ours in NEON_ASPECT_MAP.values():
        assert ours is None or ours in ASPECTS
    assert ASPECT_TO_NEON == {
        "taxon": "organism_group",
        "measurement": "measured_property",
        "environment": "environmental_material",
        "method": "method",
        "data_type": "data_type",
    }
    assert not {"unit", "location", "other"} & set(ASPECT_TO_NEON)
    for neon, ours in NEON_ASPECT_MAP.items():
        if ours is not None:
            assert ASPECT_TO_NEON[ours] == neon
