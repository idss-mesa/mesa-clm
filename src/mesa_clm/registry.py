"""Frozen vocabularies that make questions learnable (DESIGN D1; mesa-anyjev D1, D7).

Every string in the mesa-anyjev block below is part of a task's option list and therefore of
its ``task_key`` (DESIGN D1: ``task_key`` hashes the original mesa-anyjev texts, so it equals
the mesa-anyjev lock key and labels carry over). The block is ported character for character
from mesa-anyjev ``registry.py`` (``6159281``); ``tests/unit/test_registry.py`` checks it
against the committed mesa-anyjev ``questions.lock.json``. Editing one rotates that key:
update ``framings.lock.json``, say how labels migrate, and expect a refit.

The mesa-clm additions at the end (``ANCHORS``, ``ANNOTATE_OPTIONS``) are *framing* texts: they
enter ``question_key`` but never ``task_key``, so rewording them rotates decisions, features
and artifacts, never labels. ``NEON_ASPECT_MAP`` translates neon-ducklake curation aspects.

Ontology registry rule (mesa-anyjev D7): the ten ontologies allowed by neon-avu-eval
``prompt.md`` plus every CURIE prefix with at least five valid AVUs in
``results/validated.json`` (TAXRANK 7, GENEPIO 5; GO has 3 and is out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

Aspect = str


@dataclass(frozen=True)
class OntologyEntry:
    """One registry ontology: its OLS id, canonical CURIE prefix, option text and the aspects
    it may serve."""

    id: str
    curie_prefix: str
    option_text: str
    aspects: frozenset[Aspect]


ASPECTS: Final[tuple[str, ...]] = (
    "taxon",
    "environment",
    "method",
    "measurement",
    "unit",
    "data_type",
    "location",
    "other",
)

ASPECT_OPTIONS: Final[tuple[str, ...]] = (
    "taxon: the organisms observed, sampled or identified",
    "environment: the biome, habitat or environmental material at a site",
    "method: the sampling protocol, device, assay or processing step",
    "measurement: a measured quantity, quality or attribute of a thing",
    "unit: the unit of measure of a measured quantity",
    "data_type: the kind of record, identifier, dataset or information artifact",
    "location: a named place, region, plot or geographic feature",
    "other: none of the above or not an annotatable column",
)

ONTOLOGY_REGISTRY: Final[tuple[OntologyEntry, ...]] = (
    OntologyEntry(
        "envo",
        "ENVO",
        "ENVO: environments, biomes, habitats and environmental materials",
        frozenset({"environment", "location", "measurement"}),
    ),
    OntologyEntry(
        "ncbitaxon",
        "NCBITaxon",
        "NCBITaxon: organisms and taxa at every rank",
        frozenset({"taxon"}),
    ),
    OntologyEntry(
        "pato",
        "PATO",
        "PATO: qualities and attributes of things (size, sex, age, colour)",
        frozenset({"measurement"}),
    ),
    OntologyEntry(
        "uo",
        "UO",
        "UO: units of measurement (meter, degree Celsius, percent)",
        frozenset({"unit"}),
    ),
    OntologyEntry(
        "obi",
        "OBI",
        "OBI: investigations, assays, protocols, devices and specimens",
        frozenset({"method", "measurement", "data_type"}),
    ),
    OntologyEntry(
        "iao",
        "IAO",
        "IAO: information artifacts, data items, identifiers and documents",
        frozenset({"data_type", "method"}),
    ),
    OntologyEntry(
        "pco",
        "PCO",
        "PCO: populations, communities and their collective properties",
        frozenset({"taxon", "measurement"}),
    ),
    OntologyEntry(
        "bco",
        "BCO",
        "BCO: biological collections, sampling and identification processes",
        frozenset({"method"}),
    ),
    OntologyEntry(
        "gaz",
        "GAZ",
        "GAZ: named geographic places and administrative regions",
        frozenset({"location", "environment"}),
    ),
    OntologyEntry(
        "ro",
        "RO",
        "RO: relations between entities (part of, located in, participates in)",
        frozenset({"other"}),
    ),
    OntologyEntry(
        "taxrank",
        "TAXRANK",
        "TAXRANK: taxonomic ranks (species, genus, family, order)",
        frozenset({"taxon", "data_type"}),
    ),
    OntologyEntry(
        "genepio",
        "GENEPIO",
        "GENEPIO: sample collection, identification and metadata fields",
        frozenset({"method", "data_type"}),
    ),
)

ONTOLOGY_OPTIONS: Final[tuple[str, ...]] = tuple(e.option_text for e in ONTOLOGY_REGISTRY)

VALUE_KINDS: Final[tuple[str, ...]] = (
    "the term label",
    "the site code",
    "the column name",
    "the most frequent data value",
)

_PREFIX_CANON: Final[dict[str, str]] = {
    e.curie_prefix.lower(): e.curie_prefix for e in ONTOLOGY_REGISTRY
}
_BY_ID: Final[dict[str, OntologyEntry]] = {e.id: e for e in ONTOLOGY_REGISTRY}


def entry(ontology_id: str) -> OntologyEntry:
    """The registry entry for an ontology id (case-insensitive); ``KeyError`` when unknown."""
    return _BY_ID[ontology_id.lower()]


def prefix_of(curie: str) -> str:
    """The canonical CURIE prefix (``NCBITaxon`` keeps its case, others are upper-cased), or ``''``."""
    if ":" not in curie:
        return ""
    raw = curie.split(":", 1)[0]
    return _PREFIX_CANON.get(raw.lower(), raw.upper())


def allowed_for_aspect(aspect: Aspect) -> frozenset[str]:
    """Ontology ids that may serve an aspect; ``other`` allows everything."""
    if aspect == "other":
        return frozenset(_BY_ID)
    return frozenset(e.id for e in ONTOLOGY_REGISTRY if aspect in e.aspects)


def mask_for_aspect(aspect: Aspect, in_play: frozenset[str] | None = None) -> npt.NDArray[np.bool_]:
    """Boolean mask over ``ONTOLOGY_OPTIONS`` in registry order (used after the decision,
    the 2048 pattern: mask, renormalise, log the masked mass)."""
    allowed = allowed_for_aspect(aspect)
    if in_play is not None:
        allowed = allowed & in_play
    return np.array([e.id in allowed for e in ONTOLOGY_REGISTRY], dtype=bool)


# -- DataCite controlled vocabularies (DataCite Metadata Schema 4.x), frozen here so the question
# option lists never move; ``tests/unit/test_registry.py`` asserts parity with
# ``mesa_mcp.datacite.schema``. The DataCite questions are deferred to M8 in mesa-clm; the lists
# are kept verbatim so their mesa-anyjev task keys stay reproducible.
DATACITE_RESOURCE_TYPES: Final[tuple[str, ...]] = (
    "Audiovisual",
    "Book",
    "BookChapter",
    "Collection",
    "ComputationalNotebook",
    "ConferencePaper",
    "ConferenceProceeding",
    "DataPaper",
    "Dataset",
    "Dissertation",
    "Event",
    "Image",
    "InteractiveResource",
    "Journal",
    "JournalArticle",
    "Model",
    "OutputManagementPlan",
    "PeerReview",
    "PhysicalObject",
    "Preprint",
    "Report",
    "Service",
    "Software",
    "Sound",
    "Standard",
    "Text",
    "Workflow",
    "Other",
)
DATACITE_CONTRIBUTOR_TYPES: Final[tuple[str, ...]] = (
    "ContactPerson",
    "DataCollector",
    "DataCurator",
    "DataManager",
    "Distributor",
    "Editor",
    "HostingInstitution",
    "Producer",
    "ProjectLeader",
    "ProjectManager",
    "ProjectMember",
    "RegistrationAgency",
    "RegistrationAuthority",
    "RelatedPerson",
    "Researcher",
    "ResearchGroup",
    "RightsHolder",
    "Sponsor",
    "Supervisor",
    "WorkPackageLeader",
    "Other",
)
DATACITE_RELATION_TYPES: Final[tuple[str, ...]] = (
    "IsCitedBy",
    "Cites",
    "IsSupplementTo",
    "IsSupplementedBy",
    "IsContinuedBy",
    "Continues",
    "IsDescribedBy",
    "Describes",
    "IsPartOf",
    "HasPart",
    "IsReferencedBy",
    "References",
    "IsDocumentedBy",
    "Documents",
    "IsCompiledBy",
    "Compiles",
    "IsVariantFormOf",
    "IsDerivedFrom",
    "IsSourceOf",
    "IsVersionOf",
    "HasVersion",
    "IsNewVersionOf",
    "IsObsoletedBy",
)
DATACITE_DATE_TYPES: Final[tuple[str, ...]] = (
    "Accepted",
    "Available",
    "Copyrighted",
    "Collected",
    "Created",
    "Issued",
    "Submitted",
    "Updated",
    "Valid",
    "Withdrawn",
)
DATACITE_DESCRIPTION_TYPES: Final[tuple[str, ...]] = (
    "Abstract",
    "Methods",
    "SeriesInformation",
    "TableOfContents",
    "TechnicalInfo",
    "Other",
)
DATACITE_VOCABULARIES: Final[dict[str, tuple[str, ...]]] = {
    "ResourceTypeGeneral": DATACITE_RESOURCE_TYPES,
    "ContributorType": DATACITE_CONTRIBUTOR_TYPES,
    "RelationType": DATACITE_RELATION_TYPES,
    "DateType": DATACITE_DATE_TYPES,
    "DescriptionType": DATACITE_DESCRIPTION_TYPES,
}


# -- mesa-clm additions (framing texts: part of question_key, never of task_key) ---------------

# The fixed abstain anchor appended to every rank_fit Choice as ``"__none__"`` (DESIGN D2,
# plan §4.3). ``s_c = ln p_c - ln p_anchor`` is independent of the rest of the candidate set,
# and an anchor win is an abstain. X1 also tests a longer variant, which would be a new framing.
ANCHOR_KEY: Final[str] = "__none__"
ANCHORS: Final[dict[str, str]] = {
    "term": "None of these terms is the right concept for this target.",
    "ontology": "None of these ontologies has a suitable term for this target.",
}

# ``column.annotate`` as a CLM Choice (plan §4.2 Q1). Keys are the mesa-anyjev noul labels, in
# order, so ``task_key`` and imported labels are unchanged (option index 0 = "Yes"); values are
# the candidate sentences CLM scores against the column context.
ANNOTATE_OPTIONS: Final[dict[str, str]] = {
    "Yes": (
        "This column records a scientific concept, such as an organism, a measured property, "
        "an environment or a method, that deserves an ontology annotation."
    ),
    "No": (
        "This column is an identifier, a bookkeeping field, a timestamp, a free-text remark or "
        "otherwise needs no ontology annotation."
    ),
}

# neon-ducklake curation aspects (``site/validate.py`` ``ASPECTS`` and
# ``curation/prompt_product.md``, neon-ducklake ``fd7a0d5``) -> the mesa-anyjev ``ASPECTS`` above
# (plan §7.4). ``process`` has no counterpart and maps to ``None``: adding it to ``ASPECTS``
# would rotate ``column.aspect``'s task_key, so it stays a reported gap.
NEON_ASPECT_MAP: Final[dict[str, str | None]] = {
    "organism_group": "taxon",
    "measured_property": "measurement",
    "environmental_material": "environment",
    "method": "method",
    "data_type": "data_type",
    "process": None,
}

# The inverse, for writing neon proposals: mesa-clm aspect -> neon aspect. ``unit``,
# ``location`` and ``other`` have no neon aspect (neon forbids unit and place terms).
ASPECT_TO_NEON: Final[dict[str, str]] = {
    ours: neon for neon, ours in NEON_ASPECT_MAP.items() if ours is not None
}
