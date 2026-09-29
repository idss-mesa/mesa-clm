"""The OLS fixture closure (``mesa_clm.ols_closure``): the enumeration mirrors the pipeline's
OLS calls step by step, and every call it yields for the committed cards has a recorded fixture
under ``tests/fixtures/ols`` (plan M0 acceptance: ``test_ols_fixture_closure`` finds 0 missing).
Hermetic: the closure is re-enumerated from the committed cards and replayed fixtures only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests
from mesa_mcp.ols.client import OLSAPIError

from mesa_clm.cards import DatasetCard, load_card
from mesa_clm.ols import BIOME_IRI, RecordingOLS, ReplayMiss, fixture_key, iri_for
from mesa_clm.ols_closure import (
    CHILDREN_SIZE,
    DEFAULT_MAX_CANDIDATES,
    SEARCH_SIZE,
    UNIT_FALLBACK_QUERY,
    Call,
    SearchGroup,
    ThrottledOLS,
    children_calls,
    column_ontologies,
    count_by_method,
    describe_calls,
    enumerate_calls,
    enumerate_groups,
    fixture_path,
    missing_fixtures,
    retryable,
    search_calls,
    status_of,
    unique_calls,
)
from mesa_clm.planner.base import ColumnHint, Plan, PlanResult, SiteHint
from mesa_clm.registry import ASPECTS, ONTOLOGY_REGISTRY, allowed_for_aspect

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CARDS_DIR = FIXTURES / "cards"
OLS_DIR = FIXTURES / "ols"
REGISTRY_IDS = [e.id for e in ONTOLOGY_REGISTRY]


@pytest.fixture(scope="module")
def cards() -> list[DatasetCard]:
    return [load_card(p) for p in sorted(CARDS_DIR.glob("*.md"))]


@pytest.fixture
def replay() -> RecordingOLS:
    return RecordingOLS(None, OLS_DIR, "replay")


def _hit(curie: str, label: str, *, has_children: bool = False) -> dict[str, Any]:
    return {
        "curie": curie,
        "iri": iri_for(curie),
        "label": label,
        "isRoot": False,
        "hasChildren": has_children,
        "synonyms": [],
        "description": "",
        "ontologyId": curie.split(":")[0].lower(),
    }


class _FakeOLS:
    """An ``OLSLike`` whose search responses are scripted per query; records its calls."""

    def __init__(self, by_query: dict[str, Any]) -> None:
        self.by_query = by_query
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _answer(self, method: str, args: dict[str, Any]) -> Any:
        self.calls.append((method, args))
        out = self.by_query.get(args.get("query", ""), [])
        if isinstance(out, Exception):
            raise out
        return out

    def search_terms(
        self, query: str, ontology_id: str | None = None, size: int = 15
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._answer(
            "search_terms", {"query": query, "ontology_id": ontology_id, "size": size}
        )
        return out

    def search_term_descendants(
        self, query: str, ontology_id: str, parent_iri: str, size: int = 20
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._answer(
            "search_term_descendants",
            {"query": query, "ontology_id": ontology_id, "parent_iri": parent_iri, "size": size},
        )
        return out

    def get_term_children(self, ontology_id: str, iri: str, size: int = 50) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._answer(
            "get_term_children", {"ontology_id": ontology_id, "iri": iri, "size": size}
        )
        return out

    def get_term(self, ontology_id: str, iri: str) -> dict[str, Any] | None:
        out: dict[str, Any] | None = self._answer(
            "get_term", {"ontology_id": ontology_id, "iri": iri}
        )
        return out


# -- the acceptance test ------------------------------------------------------------------------


def test_ols_fixture_closure(cards: list[DatasetCard], replay: RecordingOLS) -> None:
    """Every call the pipeline can make on the seven cards (search level from the cards,
    children level from the replayed search responses) has a fixture: 0 missing, listed by
    name when not."""
    searches = enumerate_calls(cards)
    missing = missing_fixtures(searches, OLS_DIR)
    assert not missing, f"{len(missing)} search fixtures missing: {describe_calls(missing)[:10]}"
    closure = enumerate_calls(cards, ols=replay, max_candidates=DEFAULT_MAX_CANDIDATES)
    assert replay.missed == []
    missing = missing_fixtures(closure, OLS_DIR)
    assert not missing, f"{len(missing)} fixtures missing: {describe_calls(missing)[:10]}"
    counts = count_by_method(closure)
    assert set(counts) == {"search_terms", "search_term_descendants", "get_term_children"}
    assert counts["search_terms"] >= 1300 and counts["get_term_children"] >= 200
    assert len(closure) == len({fixture_key(m, a) for m, a in closure})


def test_closure_replays_through_recording_ols(
    cards: list[DatasetCard], replay: RecordingOLS
) -> None:
    """The enumerated ``(method, kwargs)`` are exactly what ``RecordingOLS`` hashes: calling each
    one on the replaying recorder is a hit, never a miss."""
    closure = enumerate_calls(cards, ols=replay)
    checker = RecordingOLS(None, OLS_DIR, "replay")
    for method, args in closure:
        assert checker.fixture_path(method, args) == fixture_path(OLS_DIR, (method, args))
        getattr(checker, method)(**args)
    assert checker.hits == len(closure) and checker.missed == []


# -- enumeration mirrors the pipeline ------------------------------------------------------------


def test_column_ontologies_union_over_aspects() -> None:
    every = frozenset(REGISTRY_IDS)
    assert column_ontologies(["unit"], every) == ["uo"]
    assert column_ontologies(["taxon"], every) == ["ncbitaxon", "pco", "taxrank"]
    assert column_ontologies(["taxon", "unit"], every) == ["ncbitaxon", "uo", "pco", "taxrank"]
    # ``other`` allows everything (registry order); a plan restricts the union to what is in play.
    assert column_ontologies(["other"], every) == REGISTRY_IDS
    assert column_ontologies(["taxon"], frozenset({"pco", "envo"})) == ["pco"]
    # A planner ontology joins only when in play; ``unit`` is ``uo`` by rule, not by mask.
    assert column_ontologies(["method"], frozenset({"bco"}), "envo") == ["bco"]
    assert column_ontologies(["method"], frozenset({"bco", "envo"}), "envo") == ["envo", "bco"]
    without_other = [a for a in ASPECTS if a != "other"]
    assert set(column_ontologies(without_other, every)) == set(REGISTRY_IDS) - {"ro"}
    for aspect in without_other:
        if aspect != "unit":
            assert set(column_ontologies([aspect], every)) == allowed_for_aspect(aspect)


def test_groups_for_the_conftest_card(card: DatasetCard) -> None:
    groups = enumerate_groups([card])
    by_target: dict[str | None, list[SearchGroup]] = {}
    for g in groups:
        by_target.setdefault(g.target, []).append(g)
    # Identifier columns (uid, siteID, startDate, identificationHistoryID) are rule rows:
    # never searched.
    assert {g.target for g in groups if g.scope == "column"} == {
        "scientificName",
        "observerDistance",
        "detectionMethod",
    }
    # Every registry ontology but uo goes through search_candidates with the static queries...
    dist = {g.ontology_id: g for g in by_target["observerDistance"]}
    assert set(dist) == set(REGISTRY_IDS) - {"uo"}
    assert dist["pato"].queries == (
        "observer distance",
        "Radial distance between the observer and the individual",  # clause cut at "("
        "meter",
    )
    assert all(g.method == "search_terms" and g.scope == "column" for g in dist.values())
    # ...and uo is the UNIT_TABLE lookup: "meter" hits, so no uo group for observerDistance,
    # while a column without a unit falls back to one search for "unit".
    name_uo = [g for g in by_target["scientificName"] if g.ontology_id == "uo"]
    assert len(name_uo) == 1 and name_uo[0].queries == (UNIT_FALLBACK_QUERY,)
    # Q5: one biome group per site through the descendant search under ENVO:00000428.
    harv, srer = by_target["HARV"][0], by_target["SRER"][0]
    assert harv.scope == "site" and harv.method == "search_term_descendants"
    assert harv.queries == (
        "temperate deciduous forest biome",
        "temperate broadleaf forest biome",
        "temperate mixed forest biome",
    )
    assert srer.queries == ("desert biome", "xeric shrubland biome", "shrubland biome")
    assert harv.calls()[0] == (
        "search_term_descendants",
        {
            "query": "temperate deciduous forest biome",
            "ontology_id": "envo",
            "parent_iri": BIOME_IRI,
            "size": SEARCH_SIZE,
        },
    )
    # Q6: the dataset taxon group from PRODUCT_TAXA ("landbird" -> Aves).
    (taxon,) = by_target[None]
    assert (taxon.scope, taxon.ontology_id, taxon.queries) == ("dataset", "ncbitaxon", ("Aves",))
    # Pipeline order: columns, then sites, then the taxon group.
    scopes = [g.scope for g in groups]
    assert scopes == sorted(scopes, key=["column", "site", "dataset"].index)


def test_unit_column_with_unknown_unit_searches_uo(card: DatasetCard) -> None:
    col = card.column("observerDistance").model_copy(update={"unit": "furlong"})
    odd = card.model_copy(update={"columns": [col], "sites": []})
    groups = enumerate_groups([odd])
    uo = [g for g in groups if g.ontology_id == "uo"]
    assert len(uo) == 1 and uo[0].queries == ("furlong",)
    assert uo[0].calls() == [
        ("search_terms", {"query": "furlong", "ontology_id": "uo", "size": SEARCH_SIZE})
    ]


def test_search_calls_follow_search_candidates(card: DatasetCard) -> None:
    """One ``search_terms(query, ontology_id, size=20)`` per non-empty query, at most three;
    duplicates (the same query for the same ontology from two columns or cards) collapse."""
    g = SearchGroup("c", "column", "x", "pato", ("a", "", "b", "c", "d"))
    assert g.calls() == [
        ("search_terms", {"query": "a", "ontology_id": "pato", "size": 20}),
        ("search_terms", {"query": "b", "ontology_id": "pato", "size": 20}),
        ("search_terms", {"query": "c", "ontology_id": "pato", "size": 20}),
    ]
    twice = search_calls([g, g, SearchGroup("d", "column", "y", "pato", ("b",))])
    assert len(twice) == 3
    calls = enumerate_calls([card])
    assert count_by_method(calls) == {"search_terms": len(calls) - 6, "search_term_descendants": 6}
    assert all(a["size"] == SEARCH_SIZE for _, a in calls)
    assert ("search_terms", {"query": "Aves", "ontology_id": "ncbitaxon", "size": 20}) in calls
    assert ("search_terms", {"query": "unit", "ontology_id": "uo", "size": 20}) in calls
    assert not any(a.get("ontology_id") == "uo" and a["query"] == "meter" for _, a in calls)


def test_planner_hints_shape_the_closure(card: DatasetCard) -> None:
    """A plan narrows ``in_play``, can force an identifier column in or a live column out,
    prepends its queries, appends its ontology, and replaces the site and taxon queries."""

    class _Planner:
        name = "scripted"
        model: str | None = None

        def plan(self, c: DatasetCard) -> PlanResult:
            plan = Plan(
                ontologies=["pato", "envo"],
                columns={
                    "uid": ColumnHint(annotate=True, queries=["identifier"]),
                    "scientificName": ColumnHint(annotate=False),
                    "observerDistance": ColumnHint(queries=["radial distance"], ontology="obi"),
                },
                sites={"HARV": SiteHint(environment_queries=["forest biome"])},
                taxon_queries=["Passeriformes"],
            )
            return PlanResult(plan=plan, planner=self.name)

    groups = enumerate_groups([card], _Planner())
    targets = {g.target for g in groups if g.scope == "column"}
    assert "uid" in targets and "scientificName" not in targets
    dist = {g.ontology_id: g for g in groups if g.target == "observerDistance"}
    # in_play {pato, envo}: obi is the hint but not in play, uo is by rule (table hit -> no group).
    assert set(dist) == {"pato", "envo"}
    assert (
        dist["pato"].queries[0] == "radial distance"
        and dist["pato"].queries[1] == "observer distance"
    )
    uid = {g.ontology_id: g for g in groups if g.target == "uid"}
    assert uid["envo"].queries == ("identifier", "uid", "Unique ID within NEON database")
    harv = next(g for g in groups if g.target == "HARV")
    assert harv.queries == ("forest biome",)
    srer = next(g for g in groups if g.target == "SRER")
    assert srer.queries[0] == "desert biome"  # no hint: static habitat queries
    # ncbitaxon is not in play: no taxon group at all.
    assert not any(g.scope == "dataset" for g in groups)


def test_children_calls_follow_specificity() -> None:
    """Children are asked for each candidate ``OLSLayer`` keeps (merged over the queries, prefix
    filtered, capped at ``max_candidates``) that has children; ``children_for_all`` takes every
    candidate; the size is the pipeline's 10."""
    parents = [_hit(f"PATO:{i:07d}", f"q{i}", has_children=(i % 2 == 0)) for i in range(15)]
    fake = _FakeOLS(
        {
            "a": [*parents[:8], _hit("GO:0000001", "imported", has_children=True)],
            "b": parents[6:],
            "biome": [_hit("ENVO:01000174", "forest biome", has_children=True)],
        }
    )
    groups = [
        SearchGroup("c", "column", "x", "pato", ("a", "b")),
        SearchGroup("c", "site", "HARV", "envo", ("biome",), "search_term_descendants"),
    ]
    calls = children_calls(groups, fake, max_candidates=12)
    # 15 unique PATO candidates merged by (rank, curie): 0,1,2,8,3,9,4,10,5,11,6,12,7,13,14,
    # capped at 12; the even ids among them have children -> 7, plus the biome.
    assert count_by_method(calls) == {"get_term_children": 8}
    assert calls[0] == (
        "get_term_children",
        {"ontology_id": "pato", "iri": iri_for("PATO:0000000"), "size": CHILDREN_SIZE},
    )
    assert calls[-1][1] == {"ontology_id": "envo", "iri": iri_for("ENVO:01000174"), "size": 10}
    assert not any("GO_" in a["iri"] for _, a in calls)
    assert len(children_calls(groups, fake, max_candidates=12, children_for_all=True)) == 13
    assert len(children_calls(groups, fake, max_candidates=4)) == 4  # 0, 2, 8 and the biome
    # The search level went through the same OLSLike; the biome group used the descendant search.
    methods = {m for m, _ in fake.calls}
    assert methods == {"search_terms", "search_term_descendants"}


def test_children_level_needs_search_fixtures(card: DatasetCard, tmp_path: Path) -> None:
    """Without ``ols`` only the search level comes back; a replaying recorder over an empty
    directory raises the first ``ReplayMiss`` by name."""
    searches = enumerate_calls([card])
    assert all(m != "get_term_children" for m, _ in searches)
    empty = RecordingOLS(None, tmp_path, "replay")
    with pytest.raises(ReplayMiss) as info:
        enumerate_calls([card], ols=empty)
    assert info.value.method == "search_terms"
    assert missing_fixtures(searches, tmp_path) == searches


def test_unique_calls_and_missing_fixtures(tmp_path: Path) -> None:
    a: Call = ("search_terms", {"query": "x", "ontology_id": "uo", "size": 20})
    a2: Call = ("search_terms", {"size": 20, "ontology_id": "uo", "query": "x"})  # same key
    b: Call = ("get_term_children", {"ontology_id": "uo", "iri": "i", "size": 10})
    assert unique_calls([a, a2, b, a]) == [a, b]
    assert fixture_path(tmp_path, a) == tmp_path / f"{fixture_key(*a)}.json"
    fixture_path(tmp_path, a).write_text("{}", encoding="utf-8")
    assert missing_fixtures([a, b], tmp_path) == [b]
    assert describe_calls([b]) == ["get_term_children(iri='i', ontology_id='uo', size=10)"]
    assert count_by_method([a, b, a2]) == {"search_terms": 2, "get_term_children": 1}


# -- ThrottledOLS ----------------------------------------------------------------------------------


def _http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status}", response=resp)


def test_status_and_retryable() -> None:
    assert status_of(OLSAPIError("x", status_code=503)) == 503
    assert status_of(OLSAPIError("timed out")) is None
    assert status_of(_http_error(404)) == 404
    assert status_of(RuntimeError("x")) is None
    assert retryable(OLSAPIError("timed out"))  # no status: connection error / timeout
    assert retryable(OLSAPIError("x", status_code=429))
    assert retryable(OLSAPIError("x", status_code=502))
    assert retryable(_http_error(500)) and retryable(requests.ConnectionError("x"))
    assert not retryable(OLSAPIError("x", status_code=404))
    assert not retryable(_http_error(400))
    assert not retryable(ValueError("not an OLS error"))


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(round(s, 6))
        self.now += s

    def __call__(self) -> float:
        return self.now


def test_throttled_spaces_requests_and_records_fixtures(tmp_path: Path) -> None:
    fake = _FakeOLS({"a": [_hit("UO:0000001", "a")], "b": []})
    clock = _Clock()
    polite = ThrottledOLS(fake, rate=4.0, sleep=clock.sleep, clock=clock)
    rec = RecordingOLS(polite, tmp_path, "auto")
    assert rec.search_terms("a", "uo", 20) == [_hit("UO:0000001", "a")]
    assert rec.search_terms("b", "uo", 20) == []
    assert rec.search_terms("a", "uo", 20) == [_hit("UO:0000001", "a")]  # replayed, no request
    assert polite.requests == 2 and polite.retries == 0 and polite.failures == []
    assert clock.sleeps == [0.25]  # the second request waited for the 4/s interval
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_throttled_retries_with_backoff_then_gives_up() -> None:
    attempts = {"n": 0}

    class _Flaky(_FakeOLS):
        def _answer(self, method: str, args: dict[str, Any]) -> Any:
            self.calls.append((method, args))
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OLSAPIError("OLS API error (503): busy", status_code=503)
            return [{"curie": "UO:0000001", "iri": iri_for("UO:0000001"), "label": "a"}]

    clock = _Clock()
    polite = ThrottledOLS(
        _Flaky({}), rate=4.0, attempts=5, backoff=1.0, sleep=clock.sleep, clock=clock
    )
    out = polite.search_terms("a", "uo", 20)
    assert out[0]["label"] == "a" and polite.requests == 3 and polite.retries == 2
    assert clock.sleeps == [1.0, 2.0]  # backoff 1, 2 (the rate wait is absorbed by the backoff)
    assert polite.failures == []

    clock = _Clock()
    always = _FakeOLS({"x": OLSAPIError("OLS API error (500): down", status_code=500)})
    polite = ThrottledOLS(always, rate=4.0, attempts=3, backoff=1.0, sleep=clock.sleep, clock=clock)
    with pytest.raises(OLSAPIError):
        polite.search_terms("x", "uo", 20)
    assert polite.requests == 3 and polite.retries == 2
    assert polite.failures == [
        {
            "method": "search_terms",
            "args": {"query": "x", "ontology_id": "uo", "size": 20},
            "status": 500,
            "error": "OLSAPIError: OLS API error (500): down",
            "attempts": 3,
        }
    ]


def test_throttled_does_not_retry_final_errors() -> None:
    clock = _Clock()
    fake = _FakeOLS({"x": _http_error(400)})
    polite = ThrottledOLS(fake, rate=4.0, attempts=5, sleep=clock.sleep, clock=clock)
    with pytest.raises(requests.HTTPError):
        polite.search_term_descendants("x", "envo", BIOME_IRI, 20)
    assert polite.requests == 1 and polite.retries == 0 and clock.sleeps == []
    assert polite.failures[0]["status"] == 400 and polite.failures[0]["attempts"] == 1
    # A non-OLS error is neither retried nor listed: it is a bug, not a network condition.
    boom = _FakeOLS({"x": ValueError("bug")})
    polite = ThrottledOLS(boom, rate=4.0, sleep=clock.sleep, clock=clock)
    with pytest.raises(ValueError, match="bug"):
        polite.search_terms("x", "uo", 20)
    assert polite.failures == []


def test_throttled_rejects_bad_settings() -> None:
    fake = _FakeOLS({})
    with pytest.raises(ValueError, match="rate"):
        ThrottledOLS(fake, rate=0)
    with pytest.raises(ValueError, match="attempts"):
        ThrottledOLS(fake, attempts=0)
    # get_term passes through the same throttle (used by the UNIT_TABLE recording).
    clock = _Clock()
    polite = ThrottledOLS(_FakeOLS({"": None}), sleep=clock.sleep, clock=clock)
    assert polite.get_term("uo", "iri") is None and polite.requests == 1
