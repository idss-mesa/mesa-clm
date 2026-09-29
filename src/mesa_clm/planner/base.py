"""The plan contract: what a planner may propose and how a plan is reported.

A :class:`Plan` is a proposal, never an answer (DESIGN D22): every field is either a free-text
search query or a value drawn from a closed vocabulary (:data:`mesa_clm.registry.ASPECTS`,
:data:`mesa_clm.registry.ONTOLOGY_REGISTRY`), and the validators *drop* anything outside those
vocabularies instead of failing, so a planner can never smuggle an identifier into a decision.
The pipeline reads the plan as hints (which ontologies are in play, which columns to skip, extra
OLS queries); the decider still asks every question itself.

Ported from mesa-anyjev ``planner/base.py`` (``6159281``) with the imports renamed.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mesa_clm.cards import DatasetCard
from mesa_clm.registry import ASPECTS, ONTOLOGY_REGISTRY

_ONTOLOGY_IDS = {e.id for e in ONTOLOGY_REGISTRY}
# At most this many OLS queries per column hint; the static rules produce the same cap.
MAX_QUERIES = 3


class ColumnHint(BaseModel):
    """What a planner proposes for one column: whether to annotate it at all, an aspect, an
    ontology and up to :data:`MAX_QUERIES` short OLS queries. ``None`` means "no opinion"."""

    model_config = ConfigDict(extra="forbid")

    annotate: bool | None = None
    aspect: str | None = None
    ontology: str | None = None
    queries: list[str] = Field(default_factory=list)

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, v: str | None) -> str | None:
        return v if v is None or v in ASPECTS else None

    @field_validator("ontology")
    @classmethod
    def _ontology(cls, v: str | None) -> str | None:
        return v.lower() if v and v.lower() in _ONTOLOGY_IDS else None

    @field_validator("queries")
    @classmethod
    def _queries(cls, v: list[str]) -> list[str]:
        return [q.strip() for q in v if q and q.strip()][:MAX_QUERIES]


class SiteHint(BaseModel):
    """Biome / habitat queries for one site (ENVO, aspect ``environment``)."""

    model_config = ConfigDict(extra="forbid")

    environment_queries: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """What the reasoning model proposes; closed enums keep it a plan, not an answer."""

    model_config = ConfigDict(extra="forbid")

    ontologies: list[str] = Field(default_factory=list)
    columns: dict[str, ColumnHint] = Field(default_factory=dict)
    sites: dict[str, SiteHint] = Field(default_factory=dict)
    taxon_queries: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("ontologies")
    @classmethod
    def _ontologies(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for o in v:
            o = o.lower()
            if o in _ONTOLOGY_IDS and o not in seen:
                seen.append(o)
        return seen


class PlanResult(BaseModel):
    """A plan plus its provenance: which planner and model produced it, the prompt hash, the raw
    model text, token usage, and whether the static rules had to stand in (``fallback``)."""

    model_config = ConfigDict(extra="forbid")

    plan: Plan
    planner: str
    model: str | None = None
    prompt_sha256: str | None = None
    raw_text: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    fallback: bool = False


class Planner(Protocol):
    """The planner role: ``plan(card)`` never raises for a model failure; it falls back to the
    static rules and says so through ``PlanResult.fallback``."""

    name: str
    model: str | None

    def plan(self, card: DatasetCard) -> PlanResult: ...
