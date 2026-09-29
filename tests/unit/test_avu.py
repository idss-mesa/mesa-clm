"""AVU construction and value rules (``mesa_clm.avu``): ``test_value_rules`` ported from
mesa-anyjev ``tests/test_avu_parity.py`` plus the defect (c) ``VALUE_KIND_*`` constants."""

from __future__ import annotations

import pytest

from mesa_clm.avu import (
    VALUE_KIND_COLUMN,
    VALUE_KIND_LABEL,
    VALUE_KIND_SITE,
    VALUE_KIND_TOP,
    build_avu,
    pre_rule_value_kind,
    top_profile_value,
    triple,
    value_for,
)
from mesa_clm.ols import Candidate, iri_for
from mesa_clm.registry import VALUE_KINDS


def _cand(label: str, curie: str, ontology_id: str) -> Candidate:
    return Candidate(label=label, curie=curie, iri=iri_for(curie), ontology_id=ontology_id)


def test_value_kind_constants_are_the_registry_strings() -> None:
    """Defect (c): mesa-anyjev ``service.py:405`` compared against ``"the top profile value"``,
    which ``VALUE_KINDS`` never contained."""
    assert VALUE_KIND_TOP == "the most frequent data value"
    assert VALUE_KIND_TOP == VALUE_KINDS[3] == VALUE_KINDS[VALUE_KINDS.index(VALUE_KIND_TOP)]
    assert (VALUE_KIND_LABEL, VALUE_KIND_SITE, VALUE_KIND_COLUMN, VALUE_KIND_TOP) == VALUE_KINDS
    assert "the top profile value" not in VALUE_KINDS
    with pytest.raises(ValueError, match="unknown value kind 'the top profile value'"):
        value_for(
            "the top profile value", term_label="x", column_name=None, site_code=None, top_value="v"
        )


def test_value_rules() -> None:
    assert pre_rule_value_kind("environment", "site") == "the site code"
    assert pre_rule_value_kind("unit", "column") == "the term label"
    assert pre_rule_value_kind("measurement", "column") is None
    assert (
        value_for(
            "the site code", term_label="x", column_name="c", site_code="HARV", top_value=None
        )
        == "HARV"
    )
    assert (
        value_for(
            "the column name", term_label="x", column_name="c", site_code=None, top_value=None
        )
        == "c"
    )
    assert (
        value_for(
            "the most frequent data value",
            term_label="x",
            column_name=None,
            site_code=None,
            top_value="singing",
        )
        == "singing"
    )
    assert (
        value_for("the site code", term_label="x", column_name=None, site_code=None, top_value=None)
        == "x"
    )
    assert top_profile_value("2 distinct; top: SRER (9416), HARV (6068)") == "SRER"
    assert top_profile_value("numeric, n=3") is None


def test_pre_rules_use_the_constants() -> None:
    assert pre_rule_value_kind("taxon", "site") == VALUE_KIND_SITE
    assert pre_rule_value_kind("location", "dataset") == VALUE_KIND_SITE
    assert pre_rule_value_kind("environment", "column") is None
    assert pre_rule_value_kind("location", "column") is None
    assert pre_rule_value_kind("unit", "dataset") == VALUE_KIND_LABEL
    assert pre_rule_value_kind("taxon", "dataset") == VALUE_KIND_LABEL
    assert pre_rule_value_kind("taxon", "column") is None
    for kind in (
        pre_rule_value_kind(a, s)
        for a in ("unit", "taxon", "environment")
        for s in ("site", "dataset", "column")
    ):
        assert kind is None or kind in VALUE_KINDS


def test_value_for_falls_back_to_the_term_label() -> None:
    common = {"term_label": "Aves", "column_name": "taxonID", "site_code": "HARV", "top_value": "x"}
    assert value_for(VALUE_KIND_LABEL, **common) == "Aves"
    assert value_for(VALUE_KIND_SITE, **common) == "HARV"
    assert value_for(VALUE_KIND_COLUMN, **common) == "taxonID"
    assert value_for(VALUE_KIND_TOP, **common) == "x"
    empty = {"term_label": "Aves", "column_name": None, "site_code": "", "top_value": None}
    for kind in VALUE_KINDS:
        assert value_for(kind, **empty) == "Aves"
    with pytest.raises(ValueError, match="unknown value kind"):
        value_for("the profile", **common)


def test_build_avu_shape() -> None:
    avu = build_avu(_cand("degree Celsius", "UO:0000027", "uo"), "degree Celsius")
    assert avu == {
        "attribute": "uo.degree_celsius",
        "value": "degree Celsius",
        "unit": "UO:0000027",
    }
    assert triple(avu) == ("uo.degree_celsius", "degree Celsius", "UO:0000027")
    assert set(avu) == {"attribute", "value", "unit"}
    # The ontology prefix is lower-cased and the label snake-cased like mesa-mcp does.
    avu = build_avu(_cand("pH value", "PATO:0001428", "PATO"), "  7.2 ")
    assert avu == {"attribute": "pato.ph_value", "value": "7.2", "unit": "PATO:0001428"}


def test_build_avu_errors() -> None:
    with pytest.raises(ValueError, match="does not snake-case"):
        build_avu(_cand("???", "PATO:0000001", "pato"), "x")
    with pytest.raises(ValueError, match="could not build an AVU"):
        build_avu(_cand("distance", "PATO:0000040", "pato"), "   ")


def test_top_profile_value_edges() -> None:
    assert top_profile_value("10 distinct; top: singing (7644), calling (5600)") == "singing"
    assert top_profile_value("898 distinct; top: 2024-05-03T12:19Z (36)") == "2024-05-03T12:19Z"
    assert top_profile_value(
        "179 distinct; top: Campylorhynchus brunneicapillus (1098), Z (1)"
    ) == ("Campylorhynchus brunneicapillus")
    assert top_profile_value("(identifier; not profiled)") is None
    assert top_profile_value("all blank") is None
    assert top_profile_value("1 distinct; top: ") is None
    assert top_profile_value("top: HARV") == "HARV"
