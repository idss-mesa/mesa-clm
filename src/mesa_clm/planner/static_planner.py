"""The rule-based planner: the fallback (and the hermetic default for tests and the bench).

Every registry ontology is in play; column queries come from the camel-split name, the first
clause of the NEON description and the unit; site queries from keyword tables over the habitat
text; taxon queries from keywords in the product title and description. Deterministic, offline,
and the plan every LLM planner falls back to (DESIGN D22).

Ported from mesa-anyjev ``planner/static_planner.py`` (``6159281``); the pipeline reuses
:func:`queries_for_column` on top of any planner's hints.
"""

from __future__ import annotations

import re

from mesa_clm.cards import ColumnInfo, DatasetCard, SiteInfo, is_identifier
from mesa_clm.planner.base import ColumnHint, Plan, PlanResult, SiteHint
from mesa_clm.registry import ONTOLOGY_REGISTRY

# camelCase boundaries: "observerDistance" -> ["observer", "Distance"]; "taxonID" -> ["taxon", "ID"].
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Habitat keyword (matched against the lower-cased card habitat text) -> ENVO biome queries.
HABITAT_QUERIES: dict[str, list[str]] = {
    "deciduous": ["temperate deciduous forest biome", "temperate broadleaf forest biome"],
    "mixed forest": ["temperate mixed forest biome", "mixed forest biome"],
    "desert": ["desert biome", "xeric shrubland biome"],
    "shrubland": ["shrubland biome"],
    "grassland": ["grassland biome", "desert grassland"],
    "wetland": ["wetland biome"],
    "tundra": ["tundra biome"],
    "coniferous": ["temperate coniferous forest biome"],
}

# Product keyword (matched against the lower-cased title + description) -> NCBITaxon queries.
PRODUCT_TAXA: dict[str, list[str]] = {
    "landbird": ["Aves"],
    "bird": ["Aves"],
    "beetle": ["Carabidae", "Coleoptera"],
    "mosquito": ["Culicidae"],
    "tick": ["Ixodida"],
    "small mammal": ["Mammalia", "Rodentia"],
    "plant": ["Viridiplantae"],
    "fish": ["Actinopterygii"],
}


def split_camel(name: str) -> str:
    """``"scientificName"`` -> ``"scientific name"``; ``"taxonID"`` -> ``"taxon id"``."""
    return " ".join(w.lower() for w in _CAMEL.split(name) if w)


def queries_for_column(col: ColumnInfo) -> list[str]:
    """Up to three OLS queries for a column: its split name, the first clause of its description
    (unless it repeats the name; 80 characters at most) and its unit."""
    out: list[str] = []
    name_q = split_camel(col.name).replace("_", " ").strip()
    if name_q:
        out.append(name_q)
    first = re.split(r"[;.(]", col.description)[0].strip()
    if first and first.lower() != name_q:
        out.append(first[:80])
    if col.unit:
        out.append(col.unit)
    return out[:3]


def habitat_queries(site: SiteInfo) -> list[str]:
    """Biome queries for a site from :data:`HABITAT_QUERIES`; ``"<habitat> biome"`` when no
    keyword matches."""
    text = site.habitat.lower()
    out: list[str] = []
    for key, qs in HABITAT_QUERIES.items():
        if key in text:
            out.extend(q for q in qs if q not in out)
    return out[:3] or [f"{text} biome"]


def taxon_queries(card: DatasetCard) -> list[str]:
    """Up to three NCBITaxon queries from :data:`PRODUCT_TAXA` keywords in the product text."""
    title = (card.product_title + " " + card.product_description).lower()
    out: list[str] = []
    for key, taxa in PRODUCT_TAXA.items():
        if key in title:
            out.extend(t for t in taxa if t not in out)
    return out[:3]


class StaticPlanner:
    """The rule-based planner; ``fallback`` is always ``False`` because there is nothing to
    fall back from."""

    name: str = "static"
    model: str | None = None

    def plan(self, card: DatasetCard) -> PlanResult:
        columns = {
            col.name: ColumnHint(
                annotate=False if is_identifier(col) else None, queries=queries_for_column(col)
            )
            for col in card.columns
        }
        sites = {s.code: SiteHint(environment_queries=habitat_queries(s)) for s in card.sites}
        plan = Plan(
            ontologies=[e.id for e in ONTOLOGY_REGISTRY],
            columns=columns,
            sites=sites,
            taxon_queries=taxon_queries(card),
            notes="static rules: every registry ontology in play; queries from column names, descriptions and units",
        )
        return PlanResult(plan=plan, planner=self.name, model=None, fallback=False)
