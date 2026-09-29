"""Deterministic candidate generation over mesa-mcp's OLS client.

Everything a model would otherwise do by tool calls is code here: search, prefix filtering
(``search_terms(ontology_id=...)`` also returns imported terms, GO/CL/PR/UBERON under pato),
dropping obsolete or root terms, dedup by CURIE, capping. ``RecordingOLS`` records and replays
responses so tests and the bench never touch the network.

Ported from mesa-anyjev ``ols.py`` (``6159281``; DESIGN U1). Changes: defect (a), the
``unit_candidate`` lookup order (exact key, then the longest whole-word suffix, never a
dict-order ``endswith``), compound units (``X per Y``, ``square X``, ``cubic X``, ``X squared``)
never resolving to their trailing base unit, and the wrong CURIEs in ``UNIT_TABLE`` (kilometer
was UO:0000009, which is kilogram); every table CURIE is checked against a recorded ``get_term``
fixture in ``tests/unit/test_unit_lookup.py``. Replay misses raise ``ReplayMiss`` naming the
method and arguments and are listed in ``RecordingOLS.missed``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final, Protocol

import requests
from mesa_mcp.ols.client import OLSAPIError
from pydantic import BaseModel, ConfigDict, Field

from mesa_clm.config import FixtureMode  # one definition: OLSConfig.fixtures uses it too
from mesa_clm.registry import prefix_of

BIOME_IRI = "http://purl.obolibrary.org/obo/ENVO_00000428"

# Normalised NEON unit string -> UO label and CURIE. Keys are what ``normalize_unit`` yields
# (lower case, camelCase split into words), so ``kilometersPerHour`` reaches the alias below.
# A compound unit is listed only when UO has that exact term (the plural spelling is NEON's);
# every CURIE here has a recorded ``get_term`` fixture whose label must equal ``label``
# (``test_unit_lookup.py``); add a fixture before adding a row.
UNIT_TABLE: dict[str, dict[str, str]] = {
    "meter": {"label": "meter", "curie": "UO:0000008"},
    "centimeter": {"label": "centimeter", "curie": "UO:0000015"},
    "millimeter": {"label": "millimeter", "curie": "UO:0000016"},
    "kilometer": {"label": "kilometer", "curie": "UO:0010066"},
    "gram": {"label": "gram", "curie": "UO:0000021"},
    "kilogram": {"label": "kilogram", "curie": "UO:0000009"},
    "milligram": {"label": "milligram", "curie": "UO:0000022"},
    "second": {"label": "second", "curie": "UO:0000010"},
    "minute": {"label": "minute", "curie": "UO:0000031"},
    "hour": {"label": "hour", "curie": "UO:0000032"},
    "day": {"label": "day", "curie": "UO:0000033"},
    "celsius": {"label": "degree Celsius", "curie": "UO:0000027"},
    "degree celsius": {"label": "degree Celsius", "curie": "UO:0000027"},
    "percent": {"label": "percent", "curie": "UO:0000187"},
    "kilometer per hour": {"label": "kilometer per hour", "curie": "UO:0010008"},
    "kilometers per hour": {"label": "kilometer per hour", "curie": "UO:0010008"},
    "meter per second": {"label": "meter per second", "curie": "UO:0000094"},
    "meters per second": {"label": "meter per second", "curie": "UO:0000094"},
    "watt per square meter": {"label": "watt per square meter", "curie": "UO:0000155"},
    "watts per square meter": {"label": "watt per square meter", "curie": "UO:0000155"},
    "milligram per kilogram": {"label": "milligram per kilogram", "curie": "UO:0000308"},
    "milligrams per kilogram": {"label": "milligram per kilogram", "curie": "UO:0000308"},
    "square meter": {"label": "square meter", "curie": "UO:0000080"},
    "square meters": {"label": "square meter", "curie": "UO:0000080"},
    "cubic meter": {"label": "cubic meter", "curie": "UO:0000096"},
    "cubic meters": {"label": "cubic meter", "curie": "UO:0000096"},
    "number": {"label": "count unit", "curie": "UO:0000189"},
    "count": {"label": "count unit", "curie": "UO:0000189"},
    "degree": {"label": "degree", "curie": "UO:0000185"},
    "milliliter": {"label": "milliliter", "curie": "UO:0000098"},
    "liter": {"label": "liter", "curie": "UO:0000099"},
}

_CAMEL_BOUNDARY: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS: Final = re.compile(r"[_\-]+")
_SPACES: Final = re.compile(r"\s+")
# A normalised unit with one of these words is a compound unit (``meters per second``, ``square
# meter``, ``celsius squared``): it names a different quantity than its trailing base unit, so
# without an exact row it must miss and reach the OLS search, never resolve to ``second``.
_COMPOUND_WORDS: Final[frozenset[str]] = frozenset({"per", "square", "squared", "cubic", "cubed"})


def normalize_unit(unit_str: str) -> str:
    """The ``UNIT_TABLE`` key form of a card unit: camelCase split into words
    (``kilometersPerHour`` -> ``kilometers per hour``), ``_``/``-`` as spaces, single spaces,
    lower case."""
    words = _CAMEL_BOUNDARY.sub(" ", unit_str.strip())
    words = _SEPARATORS.sub(" ", words)
    return _SPACES.sub(" ", words).strip().lower()


def lookup_unit(unit_str: str) -> dict[str, str] | None:
    """The ``UNIT_TABLE`` row for a unit string, or ``None`` (defect (a)).

    Exact key first; then the longest table key that is a whole-word suffix of the normalised
    string, provided the words before it are a plain modifier (``decimalDegree`` -> ``degree``,
    ``nominalDay`` -> ``day``, ``meanMetersPerSecond`` -> ``meter per second``; never
    ``micrometer`` -> ``meter``, which is what a substring ``endswith`` gave); finally the
    mesa-anyjev ``celsius`` containment rule. A compound unit without its own row
    (``micromolesPerSquareMeterPerSecond``, ``millimolesPerLiter``, ``cubicCentimeter``,
    ``celsiusSquared``: a ``per``, ``square``, ``squared``, ``cubic`` or ``cubed`` word before
    the matched unit) misses instead, because the pipeline takes a table hit as the *only*
    ``uo`` candidate and the trailing base unit (``second``, ``liter``) would be the wrong
    CURIE. A miss lets the pipeline fall back to an OLS search on the unit string.
    """
    key = normalize_unit(unit_str)
    if not key:
        return None
    exact = UNIT_TABLE.get(key)
    if exact is not None:
        return exact
    suffixes = [name for name in UNIT_TABLE if key.endswith(" " + name)]
    if suffixes:
        longest = max(suffixes, key=len)
        modifier = key[: -(len(longest) + 1)].split(" ")
        # A compound modifier names another quantity; a shorter suffix would only leave more
        # of it in front, so the lookup misses rather than return the base unit.
        return None if _COMPOUND_WORDS.intersection(modifier) else UNIT_TABLE[longest]
    if "celsius" in key and not _COMPOUND_WORDS.intersection(key.split(" ")):
        return UNIT_TABLE["celsius"]
    return None


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    curie: str
    iri: str
    ontology_id: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    has_children: bool = False
    rank: int = 0
    query: str = ""

    def as_state(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "curie": self.curie,
            "ontology_id": self.ontology_id,
            "description": self.description[:300],
            "synonyms": self.synonyms[:5],
            "has_children": self.has_children,
        }


def iri_for(curie: str) -> str:
    prefix, local = curie.split(":", 1)
    return f"http://purl.obolibrary.org/obo/{prefix}_{local}"


class OLSLike(Protocol):
    def search_terms(
        self, query: str, ontology_id: str | None = None, size: int = 15
    ) -> list[dict[str, Any]]: ...
    def search_term_descendants(
        self, query: str, ontology_id: str, parent_iri: str, size: int = 20
    ) -> list[dict[str, Any]]: ...
    def get_term_children(
        self, ontology_id: str, iri: str, size: int = 50
    ) -> list[dict[str, Any]]: ...
    def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None: ...


# What mesa-mcp's client raises: ``OLSAPIError`` from every v2 call, ``requests.HTTPError``
# from ``search_term_descendants`` (``raise_for_status`` on the v1 search endpoint).
OLS_ERRORS: Final[tuple[type[Exception], ...]] = (
    OLSAPIError,
    requests.RequestException,
    RuntimeError,
)


def fixture_key(method: str, args: dict[str, Any]) -> str:
    """The fixture file stem: ``sha256(json{"method", **args})`` with sorted keys, exactly as
    mesa-anyjev computed it, so its recorded fixtures replay here unchanged."""
    return hashlib.sha256(
        json.dumps({"method": method, **args}, sort_keys=True).encode("utf-8")
    ).hexdigest()


def describe_call(method: str, args: dict[str, Any]) -> str:
    """``method(arg=value, ...)`` with sorted, repr'd arguments: the name a miss is listed by."""
    return f"{method}({', '.join(f'{k}={args[k]!r}' for k in sorted(args))})"


class ReplayMiss(FileNotFoundError):
    """A ``replay`` lookup with no recorded fixture. Carries ``method``, ``call_args``
    (``args`` is ``BaseException``'s tuple), ``key`` and ``path`` so a closure test can list
    every miss by name."""

    def __init__(self, method: str, args: dict[str, Any], path: Path) -> None:
        self.method = method
        self.call_args = dict(args)
        self.key = path.stem
        self.path = path
        super().__init__(
            f"no OLS fixture for {describe_call(method, args)} "
            f"(expected {path}; record it with fixtures=record)"
        )


class RecordingOLS:
    """Record/replay wrapper: ``<dir>/<sha256(method|args)>.json``. In ``replay`` a miss is a
    ``ReplayMiss`` (the fixture set is the contract for hermetic tests); in ``record`` and
    ``auto`` a miss calls the real client and records the response (``auto`` is for engine
    runs, whose decisions differ from the fake backend's and therefore search pairs never seen
    before). Every miss, replayed or recorded, is appended to ``missed``."""

    def __init__(self, inner: OLSLike | None, fixture_dir: str | Path, mode: FixtureMode) -> None:
        self.inner = inner
        self.dir = Path(fixture_dir)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self.missed: list[dict[str, Any]] = []
        if mode in ("record", "auto"):
            self.dir.mkdir(parents=True, exist_ok=True)

    def fixture_path(self, method: str, args: dict[str, Any]) -> Path:
        return self.dir / f"{fixture_key(method, args)}.json"

    def _call(self, method: str, args: dict[str, Any]) -> Any:
        path = self.fixture_path(method, args)
        if self.mode != "off" and path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.mode != "off":
            self.missed.append(
                {"method": method, "args": dict(args), "key": path.stem, "mode": self.mode}
            )
        if self.mode == "replay":
            raise ReplayMiss(method, args, path)
        if self.inner is None:
            raise RuntimeError(
                f"RecordingOLS has no inner client and no fixture for {describe_call(method, args)}"
            )
        self.misses += 1
        response = getattr(self.inner, method)(**args)
        if self.mode in ("record", "auto"):
            path.write_text(
                json.dumps(
                    {"method": method, "args": args, "response": response},
                    indent=1,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return response

    def search_terms(
        self, query: str, ontology_id: str | None = None, size: int = 15
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._call(
            "search_terms", {"query": query, "ontology_id": ontology_id, "size": size}
        )
        return out

    def search_term_descendants(
        self, query: str, ontology_id: str, parent_iri: str, size: int = 20
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._call(
            "search_term_descendants",
            {"query": query, "ontology_id": ontology_id, "parent_iri": parent_iri, "size": size},
        )
        return out

    def get_term_children(self, ontology_id: str, iri: str, size: int = 50) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._call(
            "get_term_children", {"ontology_id": ontology_id, "iri": iri, "size": size}
        )
        return out

    def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None:
        out: dict[str, Any] | None = self._call(
            "get_term", {"ontology_id": ontology_id, "iri": iri}
        )
        return out


def to_candidates(hits: list[dict[str, Any]], ontology_id: str, query: str) -> list[Candidate]:
    """Prefix filter (the CURIE must belong to the ontology searched), drop obsolete, root and
    incomplete hits, dedup by CURIE keeping the first rank."""
    want = ontology_id.lower()
    out: list[Candidate] = []
    seen: set[str] = set()
    for rank, hit in enumerate(hits):
        curie = str(hit.get("curie") or "")
        iri = str(hit.get("iri") or "")
        label = str(hit.get("label") or "")
        if not curie or not iri or not label:
            continue
        if prefix_of(curie).lower() != want:
            continue
        if hit.get("isRoot") or label.lower().startswith("obsolete"):
            continue
        if curie in seen:
            continue
        seen.add(curie)
        out.append(
            Candidate(
                label=label,
                curie=curie,
                iri=iri,
                ontology_id=want,
                description=str(hit.get("description") or ""),
                synonyms=[str(s) for s in (hit.get("synonyms") or [])][:5],
                has_children=bool(hit.get("hasChildren", False)),
                rank=rank,
                query=query,
            )
        )
    return out


class OLSLayer:
    def __init__(self, client: OLSLike, *, max_candidates: int = 12, search_size: int = 20) -> None:
        self.client = client
        self.max_candidates = max_candidates
        self.search_size = search_size
        self.calls = 0

    def search_candidates(
        self, queries: list[str], ontology_id: str
    ) -> tuple[list[Candidate], dict[str, Any]]:
        """Candidates for an ontology over up to three queries, with the search log for the
        decision group (queries, raw hit counts, CURIEs filtered out)."""
        merged: dict[str, Candidate] = {}
        log: dict[str, Any] = {
            "ontology_id": ontology_id,
            "queries": [],
            "raw_hits": {},
            "filtered_out": [],
        }
        for query in [q for q in queries if q][:3]:
            self.calls += 1
            try:
                hits = self.client.search_terms(
                    query=query, ontology_id=ontology_id, size=self.search_size
                )
            except OLS_ERRORS as exc:
                log.setdefault("errors", []).append(f"{query}: {exc}")
                continue
            log["queries"].append(query)
            log["raw_hits"][query] = len(hits)
            kept = to_candidates(hits, ontology_id, query)
            log["filtered_out"].extend(
                sorted({str(h.get("curie")) for h in hits} - {c.curie for c in kept})
            )
            for c in kept:
                merged.setdefault(c.curie, c)
        cands = sorted(merged.values(), key=lambda c: (c.rank, c.curie))[: self.max_candidates]
        log["n_candidates"] = len(cands)
        return cands, log

    def biome_candidates(self, queries: list[str]) -> tuple[list[Candidate], dict[str, Any]]:
        """The closed ENVO biome set under ENVO:00000428 for a site's habitat queries; parents
        are kept so specificity is the model's decision."""
        merged: dict[str, Candidate] = {}
        log: dict[str, Any] = {
            "ontology_id": "envo",
            "parent": BIOME_IRI,
            "queries": [],
            "raw_hits": {},
            "filtered_out": [],
        }
        for query in [q for q in queries if q][:3]:
            self.calls += 1
            try:
                hits = self.client.search_term_descendants(
                    query=query, ontology_id="envo", parent_iri=BIOME_IRI, size=self.search_size
                )
            except OLS_ERRORS as exc:  # the v1 search raises HTTPError, not OLSAPIError
                log.setdefault("errors", []).append(f"{query}: {exc}")
                continue
            log["queries"].append(query)
            log["raw_hits"][query] = len(hits)
            for c in to_candidates(hits, "envo", query):
                merged.setdefault(c.curie, c)
        cands = sorted(merged.values(), key=lambda c: (c.rank, c.curie))[: self.max_candidates]
        log["n_candidates"] = len(cands)
        return cands, log

    def children(self, ontology_id: str, iri: str, size: int = 10) -> list[Candidate]:
        self.calls += 1
        hits = self.client.get_term_children(ontology_id=ontology_id, iri=iri, size=size)
        return to_candidates(hits, ontology_id, f"children:{iri}")

    def unit_candidate(self, unit_str: str) -> Candidate | None:
        """The ``UNIT_TABLE`` candidate for a card unit (``lookup_unit``), or ``None`` so the
        caller falls back to an OLS search. Pure table lookup: no OLS call, ``calls`` unchanged."""
        spec = lookup_unit(unit_str)
        if spec is None:
            return None
        return Candidate(
            label=spec["label"],
            curie=spec["curie"],
            iri=iri_for(spec["curie"]),
            ontology_id="uo",
            query=unit_str,
        )
