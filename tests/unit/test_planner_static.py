"""The plan contract and the static planner, ported from mesa-anyjev ``tests/test_planner_static.py``
plus the fixture cards, the helper tables and :func:`make_planner`'s static default."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mesa_clm.cards import ColumnInfo, DatasetCard, SiteInfo, is_identifier, load_card
from mesa_clm.config import Config
from mesa_clm.planner import Planner, make_planner
from mesa_clm.planner.base import MAX_QUERIES, ColumnHint, Plan, PlanResult
from mesa_clm.planner.gateway_planner import _coerce, parse_json
from mesa_clm.planner.static_planner import (
    HABITAT_QUERIES,
    PRODUCT_TAXA,
    StaticPlanner,
    habitat_queries,
    queries_for_column,
    split_camel,
    taxon_queries,
)
from mesa_clm.registry import ONTOLOGY_REGISTRY

# -- ported from mesa-anyjev ---------------------------------------------------------------------


def test_static_plan(card: DatasetCard) -> None:
    result = StaticPlanner().plan(card)
    plan = result.plan
    assert result.planner == "static" and not result.fallback
    assert len(plan.ontologies) == 12
    assert plan.columns["uid"].annotate is False
    assert plan.columns["observerDistance"].annotate is None
    assert plan.columns["observerDistance"].queries[0] == "observer distance"
    assert "meter" in plan.columns["observerDistance"].queries
    assert plan.sites["HARV"].environment_queries[0].startswith("temperate deciduous")
    assert plan.sites["SRER"].environment_queries[:2] == ["desert biome", "xeric shrubland biome"]
    assert plan.taxon_queries == ["Aves"]


def test_helpers() -> None:
    assert split_camel("scientificName") == "scientific name"
    assert split_camel("taxonID") == "taxon id"


def test_plan_schema_drops_unknown_enums() -> None:
    plan = Plan.model_validate(
        {
            "ontologies": ["ENVO", "bogus", "envo"],
            "columns": {
                "x": {"aspect": "nope", "ontology": "NOPE", "queries": ["a", "b", "c", "d"]}
            },
        }
    )
    assert plan.ontologies == ["envo"]
    assert plan.columns["x"].aspect is None and plan.columns["x"].ontology is None
    assert plan.columns["x"].queries == ["a", "b", "c"]


def test_parse_json_and_coerce() -> None:
    raw = (
        '<think>hmm</think>```json\n{"ontologies": ["envo"], '
        '"columns": [{"name": "c", "queries": "q"}], "sites": 3}\n```'
    )
    parsed = parse_json(raw)
    assert parsed is not None
    plan = Plan.model_validate(_coerce(parsed))
    assert plan.columns["c"].queries == ["q"] and plan.sites == {}
    assert parse_json("no json here") is None


# -- mesa-clm additions --------------------------------------------------------------------------


def test_static_plan_is_a_proposal_over_the_whole_registry(card: DatasetCard) -> None:
    """Every registry ontology in play, in registry order; every column gets a hint; the model
    field is None (no LLM ran) and the notes say so."""
    result = StaticPlanner().plan(card)
    assert result.plan.ontologies == [e.id for e in ONTOLOGY_REGISTRY]
    assert set(result.plan.columns) == {c.name for c in card.columns}
    assert result.model is None and result.prompt_sha256 is None and result.raw_text is None
    assert result.usage == {}
    assert result.plan.notes.startswith("static rules")
    # A static hint never carries an aspect or an ontology: the decider chooses those.
    assert all(h.aspect is None and h.ontology is None for h in result.plan.columns.values())


def test_static_plan_over_fixture_cards(fixture_cards: list[Path]) -> None:
    assert len(fixture_cards) == 7
    planner = StaticPlanner()
    for path in fixture_cards:
        card = load_card(path)
        plan = planner.plan(card).plan
        assert len(plan.ontologies) == 12
        for col in card.columns:
            hint = plan.columns[col.name]
            assert hint.annotate is (False if is_identifier(col) else None)
            assert 1 <= len(hint.queries) <= MAX_QUERIES
            assert all(q and q == q.strip() for q in hint.queries)
        assert set(plan.sites) == {s.code for s in card.sites}
        if card.product_code == "DP1.10022.001":
            assert plan.taxon_queries == ["Carabidae", "Coleoptera"]
        else:
            assert plan.taxon_queries == ["Aves"]


def _col(name: str, description: str = "", unit: str = "", dtype: str = "string") -> ColumnInfo:
    return ColumnInfo(name=name, description=description, dtype=dtype, unit=unit, profile="")


def test_queries_for_column_rules() -> None:
    # name, first description clause (80 chars max), unit - capped at three.
    col = _col(
        "observerDistance",
        "Radial distance between the observer and the individual(s) being observed; in meters.",
        unit="meter",
    )
    assert queries_for_column(col) == [
        "observer distance",
        "Radial distance between the observer and the individual",
        "meter",
    ]
    # A description that repeats the name (case-insensitively) is not a second query.
    assert queries_for_column(_col("siteID", "Site ID")) == ["site id"]
    # Underscores split like camel case; an empty description adds nothing.
    assert queries_for_column(_col("plot_type", "")) == ["plot type"]
    long = "x" * 200
    assert queries_for_column(_col("a", long)) == ["a", "x" * 80]
    assert queries_for_column(_col("", "", unit="percent")) == ["percent"]


def _site(habitat: str) -> SiteInfo:
    return SiteInfo(code="X", name="x", state="s", domain="D00", domain_name="d", habitat=habitat)


def test_habitat_queries_table_and_fallback() -> None:
    assert habitat_queries(_site("temperate deciduous/mixed forest")) == [
        "temperate deciduous forest biome",
        "temperate broadleaf forest biome",
        "temperate mixed forest biome",
    ]  # capped at three (the 'mixed forest' key's second query is dropped)
    assert habitat_queries(_site("Arctic Tundra")) == ["tundra biome"]
    assert habitat_queries(_site("kelp forest")) == ["kelp forest biome"]  # no keyword: fallback
    assert set(HABITAT_QUERIES) >= {"deciduous", "desert", "grassland", "tundra", "coniferous"}


def test_taxon_queries_dedup_and_cap(card: DatasetCard) -> None:
    assert taxon_queries(card) == ["Aves"]  # 'landbird' and 'bird' both match; deduplicated
    everything = card.model_copy(
        update={"product_title": "beetle mosquito tick plant fish", "product_description": ""}
    )
    assert taxon_queries(everything) == ["Carabidae", "Coleoptera", "Culicidae"]  # capped at 3
    none = card.model_copy(update={"product_title": "Soil chemistry", "product_description": ""})
    assert taxon_queries(none) == []
    assert PRODUCT_TAXA["small mammal"] == ["Mammalia", "Rodentia"]


def test_column_hint_validators() -> None:
    hint = ColumnHint(aspect="measurement", ontology="PATO", queries=["  a ", "", "b", " "])
    assert (hint.aspect, hint.ontology, hint.queries) == ("measurement", "pato", ["a", "b"])
    assert ColumnHint(aspect="Measurement").aspect is None  # aspects are case-sensitive ids
    with pytest.raises(ValidationError):
        ColumnHint(curie="ENVO:00000446")  # type: ignore[call-arg]  # a plan never names a term


def test_plan_result_is_closed() -> None:
    with pytest.raises(ValidationError):
        PlanResult(plan=Plan(), planner="static", extra_field=1)  # type: ignore[call-arg]
    result = PlanResult(plan=Plan(), planner="static")
    assert result.model is None and result.fallback is False and result.usage == {}


def test_make_planner_static_default() -> None:
    cfg = Config()
    assert cfg.planner.kind == "static"
    planner: Planner = make_planner(cfg)
    assert isinstance(planner, StaticPlanner)
    assert isinstance(make_planner(cfg, "static"), StaticPlanner)
    assert (planner.name, planner.model) == ("static", None)
