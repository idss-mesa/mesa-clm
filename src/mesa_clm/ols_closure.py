"""The OLS fixture closure: every OLS call the annotate pipeline can make on a set of cards.

The hermetic pipeline test (plan §8, ``test_pipeline_fake``) runs all seven fixture cards
through ``OLSLayer`` over a ``RecordingOLS`` in ``replay`` mode. With a fake CLM the
aspect and ontology choices of Q2/Q3 are not predictable, so the fixture set must cover every
search any decision path could make, not only the ones a particular run took. This module
enumerates that closure from the cards alone, in the exact ``(method, kwargs)`` form
``RecordingOLS`` hashes (``mesa_clm.ols.fixture_key``), so

- ``scripts/record_ols_closure.py`` records the missing calls once against the live EBI OLS4
  (plan M0: "every column × aspect-allowed ontology × static queries, children of every
  candidate, biome queries"), and
- ``tests/unit/test_ols_fixture_closure.py`` re-enumerates the calls from the committed cards
  and asserts that every fixture file exists (acceptance: 0 missing).

What is mirrored, step by step, from mesa-anyjev ``pipeline.py`` (``6159281``; the mesa-clm
port keeps the OLS call shapes, plan §4.2):

Q1  ``column.annotate``: an identifier column (``cards.is_identifier``) is a ``rule`` row and
    never searched unless the planner says ``annotate=True``; a planner ``annotate=False`` is
    a rule too. Every other column may be annotated.
Q2  ``column.aspect``: top-1 or top-2 of ``ASPECTS`` plus the planner's aspect, ``other``
    dropped. The closure takes every aspect (``aspects``, default ``ASPECTS``; ``other``
    contributes only ``ro``, the one ontology that serves nothing else).
Q3  ``column.ontology_fits``: per aspect the ``allowed_for_aspect(aspect) & in_play``
    ontologies (``in_play`` = the plan's ontologies, else the whole registry); ``unit`` is
    ``uo`` by rule; the planner's ontology is appended. The closure takes the union.
S   ``search_candidates(queries[:3], ont)``: one ``search_terms(query, ontology_id, size=20)``
    per non-empty query, ``queries`` = planner hint queries then ``queries_for_column``. For
    ``uo`` the pipeline uses ``unit_candidate(col.unit)`` (a table lookup, no call) and falls
    back to ``search_terms(col.unit or "unit", "uo", 20)`` when the table misses.
Q4b Specificity: ``children(ont, winner.iri)`` = ``get_term_children(ontology_id, iri, size=10)``
    for the winning candidate when ``has_children``. Any of the (at most ``max_candidates``)
    candidates can win, so the closure takes the children of every candidate with children
    (``children_for_all=True`` takes all of them). Children are derived from the recorded
    search responses through ``OLSLayer`` itself (same merge, sort and cap), so this level needs
    an ``OLSLike`` (a replaying ``RecordingOLS``) and is skipped when none is given.
Q5  Site environment: ``biome_candidates(queries)`` = ``search_term_descendants(query, "envo",
    BIOME_IRI, 20)`` per query, ``queries`` = planner site hint else ``habitat_queries``.
Q6  Dataset taxon: ``search_candidates(taxon_q[:3], "ncbitaxon")`` when the plan or
    ``taxon_queries`` yields any and ``ncbitaxon`` is in play.

``get_term`` is never called by the pipeline (only by label ingestion and the ``UNIT_TABLE``
test), so it is not part of this closure. ``ThrottledOLS`` is the polite, retrying client the
recording script wraps around mesa-mcp's ``OLSClient``; ``RecordingOLS`` itself does not persist
errors, so a call that still fails after the retries is reported, never recorded.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests
from mesa_mcp.ols.client import OLSAPIError

from mesa_clm.cards import DatasetCard, is_identifier
from mesa_clm.ols import (
    BIOME_IRI,
    OLS_ERRORS,
    OLSLayer,
    OLSLike,
    describe_call,
    fixture_key,
    lookup_unit,
)
from mesa_clm.planner.base import Planner
from mesa_clm.planner.static_planner import (
    StaticPlanner,
    habitat_queries,
    queries_for_column,
    taxon_queries,
)
from mesa_clm.registry import ASPECTS, ONTOLOGY_REGISTRY, allowed_for_aspect

#: ``(method, kwargs)`` exactly as ``RecordingOLS`` passes them to ``fixture_key``.
Call = tuple[str, dict[str, Any]]

# ``OLSLayer`` defaults the pipeline runs with (``search_size``, ``children(size=)``) and the
# policy default for ``max_candidates`` (mesa-anyjev ``config.py:102``, plan §4.2 "≤12").
SEARCH_SIZE = 20
CHILDREN_SIZE = 10
DEFAULT_MAX_CANDIDATES = 12

# The one query the pipeline sends for a unit column whose unit is blank or not in
# ``UNIT_TABLE`` (``pipeline.py``: ``search_candidates([col.unit or "unit"], "uo")``).
UNIT_FALLBACK_QUERY = "unit"

Scope = Literal["column", "site", "dataset"]
SearchMethod = Literal["search_terms", "search_term_descendants"]


@dataclass(frozen=True)
class SearchGroup:
    """One candidate group the pipeline searches: what becomes a ``decision_groups`` row
    (``scope``, ``target`` = column name / site code / ``None``, ``ontology_id``) plus the
    queries ``OLSLayer`` sends for it and the client method they go through."""

    card: str
    scope: Scope
    target: str | None
    ontology_id: str
    queries: tuple[str, ...]
    method: SearchMethod = "search_terms"

    def calls(self, search_size: int = SEARCH_SIZE) -> list[Call]:
        """The search calls ``OLSLayer.search_candidates`` / ``biome_candidates`` make for this
        group: one per non-empty query, at most three, in query order."""
        out: list[Call] = []
        for query in [q for q in self.queries if q][:3]:
            if self.method == "search_term_descendants":
                args: dict[str, Any] = {
                    "query": query,
                    "ontology_id": self.ontology_id,
                    "parent_iri": BIOME_IRI,
                    "size": search_size,
                }
            else:
                args = {"query": query, "ontology_id": self.ontology_id, "size": search_size}
            out.append((self.method, args))
        return out


def column_ontologies(
    aspects: Iterable[str], in_play: frozenset[str], hint_ontology: str | None = None
) -> list[str]:
    """The ontologies Q3 can hand to S for one column over the given aspects, in registry
    order: the union of ``allowed_for_aspect(aspect) & in_play`` (``uo`` by rule for ``unit``,
    so it enters whenever ``unit`` is among the aspects) plus the planner's ontology when it is
    in play."""
    chosen: set[str] = set()
    for aspect in aspects:
        if aspect == "unit":
            chosen.add("uo")
            continue
        chosen |= allowed_for_aspect(aspect) & in_play
    if hint_ontology and hint_ontology in in_play:
        chosen.add(hint_ontology)
    return [e.id for e in ONTOLOGY_REGISTRY if e.id in chosen]


def enumerate_groups(
    cards: Sequence[DatasetCard],
    planner: Planner | None = None,
    *,
    aspects: Sequence[str] = ASPECTS,
) -> list[SearchGroup]:
    """Every candidate group the pipeline can search on ``cards`` under ``planner`` (default
    ``StaticPlanner``), in pipeline order per card: columns (S), sites (Q5), taxon (Q6).
    Duplicates across cards are kept here (the group is per card); ``enumerate_calls`` dedups
    the calls."""
    planner = planner or StaticPlanner()
    groups: list[SearchGroup] = []
    for card in cards:
        plan = planner.plan(card).plan
        in_play = (
            frozenset(plan.ontologies)
            if plan.ontologies
            else frozenset(e.id for e in ONTOLOGY_REGISTRY)
        )
        # Q1: which columns may reach the model at all.
        for col in card.columns:
            hint = plan.columns.get(col.name)
            if is_identifier(col) and not (hint and hint.annotate is True):
                continue
            if hint and hint.annotate is False:
                continue
            # S: the same query list for every ontology of the column.
            queries = list(hint.queries) if hint and hint.queries else []
            for q in queries_for_column(col):
                if q not in queries:
                    queries.append(q)
            for ont in column_ontologies(aspects, in_play, hint.ontology if hint else None):
                if ont == "uo":
                    if col.unit and lookup_unit(col.unit) is not None:
                        continue  # table hit: no OLS call
                    groups.append(
                        SearchGroup(
                            card.name, "column", col.name, "uo", (col.unit or UNIT_FALLBACK_QUERY,)
                        )
                    )
                    continue
                groups.append(SearchGroup(card.name, "column", col.name, ont, tuple(queries[:3])))
        # Q5: one ENVO biome group per site.
        for site in card.sites:
            hint_s = plan.sites.get(site.code)
            site_q = (
                list(hint_s.environment_queries) if hint_s and hint_s.environment_queries else []
            ) or habitat_queries(site)
            groups.append(
                SearchGroup(
                    card.name, "site", site.code, "envo", tuple(site_q), "search_term_descendants"
                )
            )
        # Q6: the dataset taxon group.
        taxon_q = list(plan.taxon_queries) or taxon_queries(card)
        if taxon_q and "ncbitaxon" in in_play:
            groups.append(SearchGroup(card.name, "dataset", None, "ncbitaxon", tuple(taxon_q[:3])))
    return groups


def unique_calls(calls: Iterable[Call]) -> list[Call]:
    """``calls`` without repeats, first occurrence kept, identity = ``fixture_key``."""
    seen: set[str] = set()
    out: list[Call] = []
    for method, args in calls:
        key = fixture_key(method, args)
        if key not in seen:
            seen.add(key)
            out.append((method, dict(args)))
    return out


def search_calls(groups: Iterable[SearchGroup], *, search_size: int = SEARCH_SIZE) -> list[Call]:
    """The unique search-level calls (``search_terms``, ``search_term_descendants``) of
    ``groups``, in first-occurrence order."""
    return unique_calls(c for g in groups for c in g.calls(search_size))


def children_calls(
    groups: Iterable[SearchGroup],
    ols: OLSLike,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    search_size: int = SEARCH_SIZE,
    children_size: int = CHILDREN_SIZE,
    children_for_all: bool = False,
) -> list[Call]:
    """The unique ``get_term_children`` calls Q4b can make: for every group, the candidates
    ``OLSLayer`` (over ``ols``, with the pipeline's ``max_candidates``) returns for its queries,
    and for each one with ``has_children`` (or each one, with ``children_for_all``) the call
    ``get_term_children(ontology_id, iri, size=children_size)``.

    ``ols`` is normally a replaying ``RecordingOLS`` (a missing search fixture raises
    ``ReplayMiss``) or, while recording, one in ``auto`` mode over a live client."""
    layer = OLSLayer(ols, max_candidates=max_candidates, search_size=search_size)
    calls: list[Call] = []
    for group in groups:
        queries = list(group.queries)
        if group.method == "search_term_descendants":
            cands, _ = layer.biome_candidates(queries)
        else:
            cands, _ = layer.search_candidates(queries, group.ontology_id)
        for cand in cands:
            if children_for_all or cand.has_children:
                calls.append(
                    (
                        "get_term_children",
                        {"ontology_id": group.ontology_id, "iri": cand.iri, "size": children_size},
                    )
                )
    return unique_calls(calls)


def enumerate_calls(
    cards: Sequence[DatasetCard],
    planner: Planner | None = None,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ols: OLSLike | None = None,
    aspects: Sequence[str] = ASPECTS,
    search_size: int = SEARCH_SIZE,
    children_size: int = CHILDREN_SIZE,
    children_for_all: bool = False,
) -> list[Call]:
    """Every OLS call the pipeline can make on ``cards``, as unique ``(method, kwargs)`` pairs
    in the form ``RecordingOLS`` hashes: the search level first (pipeline order, first
    occurrence), then, when ``ols`` is given, the ``get_term_children`` level derived from the
    search responses ``ols`` serves (see ``children_calls``). Without ``ols`` only the search
    level is returned."""
    groups = enumerate_groups(cards, planner, aspects=aspects)
    calls = search_calls(groups, search_size=search_size)
    if ols is not None:
        calls += children_calls(
            groups,
            ols,
            max_candidates=max_candidates,
            search_size=search_size,
            children_size=children_size,
            children_for_all=children_for_all,
        )
    return unique_calls(calls)


def fixture_path(fixture_dir: str | Path, call: Call) -> Path:
    """Where ``RecordingOLS`` keeps (or would keep) the fixture for ``call``."""
    method, args = call
    return Path(fixture_dir) / f"{fixture_key(method, args)}.json"


def missing_fixtures(calls: Iterable[Call], fixture_dir: str | Path) -> list[Call]:
    """The calls of ``calls`` with no fixture file under ``fixture_dir``, in order."""
    return [c for c in calls if not fixture_path(fixture_dir, c).exists()]


def count_by_method(calls: Iterable[Call]) -> dict[str, int]:
    """``{method: n}`` in method order of first occurrence."""
    out: dict[str, int] = {}
    for method, _ in calls:
        out[method] = out.get(method, 0) + 1
    return out


def describe_calls(calls: Iterable[Call]) -> list[str]:
    """``describe_call`` of each call: the names a closure report or a failing test lists."""
    return [describe_call(m, a) for m, a in calls]


# -- recording support --------------------------------------------------------------------------


def status_of(exc: BaseException) -> int | None:
    """The HTTP status an OLS error carries, if any: ``OLSAPIError.status_code`` or the
    response status of a ``requests.HTTPError`` (what ``search_term_descendants`` raises)."""
    if isinstance(exc, OLSAPIError):
        code = exc.status_code  # untyped (mesa_mcp is ignore_missing_imports)
        return None if code is None else int(code)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return int(exc.response.status_code)
    return None


def retryable(exc: BaseException) -> bool:
    """Whether a failed OLS call is worth another attempt: connection errors, timeouts, 429
    and 5xx are; every other 4xx (a malformed query, an unknown term) is final."""
    if not isinstance(exc, OLS_ERRORS):
        return False
    status = status_of(exc)
    if status is None:
        return True
    return status == 429 or status >= 500


class ThrottledOLS:
    """An ``OLSLike`` over another that spaces requests at most ``rate`` per second and retries
    ``retryable`` failures with exponential backoff (``backoff * 2**attempt`` seconds, at most
    ``attempts`` tries). A call that still fails is appended to ``failures`` (method, args,
    status, error text) and re-raised, so ``RecordingOLS`` records nothing for it and
    ``OLSLayer`` skips the query as it would live. ``sleep`` and ``clock`` are injectable for
    tests."""

    def __init__(
        self,
        inner: OLSLike,
        *,
        rate: float = 4.0,
        attempts: int = 5,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.inner = inner
        self.interval = 1.0 / rate
        self.attempts = attempts
        self.backoff = backoff
        self._sleep = sleep
        self._clock = clock
        self._last = -float("inf")
        self.requests = 0
        self.retries = 0
        self.failures: list[dict[str, Any]] = []

    def _wait(self) -> None:
        now = self._clock()
        due = self._last + self.interval
        if due > now:
            self._sleep(due - now)
            now = due
        self._last = now

    def _call(self, method: str, args: dict[str, Any]) -> Any:
        for attempt in range(self.attempts):
            self._wait()
            self.requests += 1
            try:
                return getattr(self.inner, method)(**args)
            except OLS_ERRORS as exc:
                last = attempt == self.attempts - 1
                if last or not retryable(exc):
                    self.failures.append(
                        {
                            "method": method,
                            "args": dict(args),
                            "status": status_of(exc),
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempts": attempt + 1,
                        }
                    )
                    raise
                self.retries += 1
                self._sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

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
