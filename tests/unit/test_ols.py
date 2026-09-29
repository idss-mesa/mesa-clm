"""OLS candidate generation (``mesa_clm.ols``): the behaviour ported from mesa-anyjev ``ols.py``
plus the mesa-clm changes (``ReplayMiss``, ``RecordingOLS.missed``, ``OLS_ERRORS``), driven by
the recorded fixtures under ``tests/fixtures/ols`` and never the network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
from mesa_mcp.ols.client import OLSAPIError
from pydantic import ValidationError

from mesa_clm.ols import (
    BIOME_IRI,
    Candidate,
    OLSLayer,
    RecordingOLS,
    ReplayMiss,
    describe_call,
    fixture_key,
    iri_for,
    to_candidates,
)

OLS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ols"
FIXTURE_FILES = sorted(OLS_DIR.glob("*.json"))

# mesa-anyjev recorded 777 responses (6159281); mesa-clm adds the UNIT_TABLE get_term closure.
N_ANYJEV_FIXTURES = 777


def _hit(
    curie: str,
    label: str,
    *,
    iri: str | None = None,
    is_root: bool = False,
    has_children: bool = False,
    synonyms: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    return {
        "curie": curie,
        "iri": iri if iri is not None else iri_for(curie),
        "label": label,
        "isRoot": is_root,
        "hasChildren": has_children,
        "synonyms": synonyms or [],
        "description": description,
        "ontologyId": curie.split(":")[0].lower(),
    }


class _FakeClient:
    """An ``OLSLike`` whose responses (or exceptions) are scripted per method."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _answer(self, method: str, args: dict[str, Any]) -> Any:
        self.calls.append((method, args))
        out = self.responses[method]
        if callable(out):
            out = out(**args)
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


@pytest.fixture
def replay() -> RecordingOLS:
    return RecordingOLS(None, OLS_DIR, "replay")


# -- fixture layout ---------------------------------------------------------------------------


def test_fixture_files_are_keyed_exactly_as_anyjev() -> None:
    """Every recorded file sits at ``sha256(json{"method", **args})``: the mesa-anyjev layout,
    so its 777 fixtures replay here unchanged."""
    assert len(FIXTURE_FILES) >= N_ANYJEV_FIXTURES
    for path in FIXTURE_FILES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert set(doc) == {"method", "args", "response"}, path.name
        assert fixture_key(doc["method"], doc["args"]) == path.stem, path.name
        assert doc["method"] in {
            "search_terms",
            "search_term_descendants",
            "get_term_children",
            "get_term",
        }


def test_fixture_key_is_order_independent() -> None:
    a = fixture_key("get_term", {"ontology_id": "uo", "iri": "x"})
    b = fixture_key("get_term", {"iri": "x", "ontology_id": "uo"})
    assert a == b and len(a) == 64
    assert fixture_key("get_term", {"ontology_id": "uo", "iri": "y"}) != a


def test_describe_call_sorts_and_reprs() -> None:
    assert describe_call("search_terms", {"size": 20, "query": "a b", "ontology_id": None}) == (
        "search_terms(ontology_id=None, query='a b', size=20)"
    )


# -- Candidate / iri_for / to_candidates --------------------------------------------------------


def test_iri_for() -> None:
    assert iri_for("UO:0000008") == "http://purl.obolibrary.org/obo/UO_0000008"
    assert iri_for("NCBITaxon:8782") == "http://purl.obolibrary.org/obo/NCBITaxon_8782"
    assert iri_for("ENVO:01000385").endswith("/ENVO_01000385")


def test_candidate_is_frozen_and_as_state_truncates() -> None:
    cand = Candidate(
        label="distance",
        curie="PATO:0000040",
        iri=iri_for("PATO:0000040"),
        ontology_id="pato",
        description="x" * 500,
        synonyms=["a", "b", "c", "d", "e", "f"],
        has_children=True,
        rank=3,
        query="distance",
    )
    state = cand.as_state()
    assert set(state) == {
        "label",
        "curie",
        "ontology_id",
        "description",
        "synonyms",
        "has_children",
    }
    assert len(state["description"]) == 300
    assert state["synonyms"] == ["a", "b", "c", "d", "e"]
    with pytest.raises(ValidationError):
        cand.label = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Candidate(label="x", curie="PATO:1", iri="i", ontology_id="pato", extra=1)  # type: ignore[call-arg]


def test_to_candidates_filters_dedups_and_ranks() -> None:
    hits = [
        _hit("GO:0008150", "biological_process"),  # imported term under pato: wrong prefix
        _hit("PATO:0000001", "quality", is_root=True),  # root
        _hit("PATO:0000040", "distance", has_children=True, synonyms=list("abcdefg")),
        {"curie": "PATO:0000041", "iri": "", "label": "no iri"},  # incomplete
        {"curie": "", "iri": iri_for("PATO:0000042"), "label": "no curie"},
        {"curie": "PATO:0000043", "iri": iri_for("PATO:0000043"), "label": ""},
        _hit("PATO:0000044", "obsolete distance"),  # obsolete
        _hit("PATO:0000040", "distance again"),  # duplicate CURIE: first rank kept
        _hit("pato:0000045", "lower-case prefix", description="d"),  # prefix_of canonicalises
    ]
    cands = to_candidates(hits, "PATO", "distance")
    assert [c.curie for c in cands] == ["PATO:0000040", "pato:0000045"]
    first = cands[0]
    assert first.rank == 2 and first.query == "distance" and first.ontology_id == "pato"
    assert first.has_children is True and first.synonyms == list("abcde")
    assert cands[1].rank == 8 and cands[1].description == "d"


def test_to_candidates_on_recorded_hits() -> None:
    """The genepio ``genus`` search returns obsolete and foreign (NCBITaxon, DOID, MONDO) hits;
    the uo ``kilometer`` search returns each hit twice."""
    replay = RecordingOLS(None, OLS_DIR, "replay")
    genus = to_candidates(replay.search_terms("genus", "genepio", 20), "genepio", "genus")
    assert genus and all(c.curie.startswith("GENEPIO:") for c in genus)
    assert not any(c.label.lower().startswith("obsolete") for c in genus)
    km_hits = replay.search_terms("kilometer", "uo", 20)
    assert len(km_hits) > len({h["curie"] for h in km_hits})
    km = to_candidates(km_hits, "uo", "kilometer")
    assert [c.curie for c in km] == ["UO:0010066", "UO:0010008"]
    assert km[0].label == "kilometer" and km[0].rank == 0


# -- RecordingOLS -------------------------------------------------------------------------------


def test_replay_hit(replay: RecordingOLS) -> None:
    term = replay.get_term("uo", iri_for("UO:0000008"))
    assert term is not None and term["label"] == "meter"
    assert (replay.hits, replay.misses, replay.missed) == (1, 0, [])


def test_replay_miss_is_named(replay: RecordingOLS) -> None:
    args = {"query": "no such recorded query", "ontology_id": "uo", "size": 20}
    with pytest.raises(ReplayMiss) as info:
        replay.search_terms(**args)
    exc = info.value
    assert isinstance(exc, FileNotFoundError)
    assert exc.method == "search_terms" and exc.call_args == args
    assert exc.key == fixture_key("search_terms", args) and exc.path == OLS_DIR / f"{exc.key}.json"
    assert "search_terms(" in str(exc) and "query='no such recorded query'" in str(exc)
    assert "fixtures=record" in str(exc)
    assert replay.missed == [
        {"method": "search_terms", "args": args, "key": exc.key, "mode": "replay"}
    ]
    assert (replay.hits, replay.misses) == (0, 0)


def test_replay_lists_every_miss_by_name(replay: RecordingOLS) -> None:
    for q in ("zzz one", "zzz two"):
        with pytest.raises(ReplayMiss):
            replay.search_terms(q, "envo", 20)
    with pytest.raises(ReplayMiss):
        replay.get_term_children("envo", iri_for("ENVO:99999999"), 10)
    names = [describe_call(m["method"], m["args"]) for m in replay.missed]
    assert names == [
        "search_terms(ontology_id='envo', query='zzz one', size=20)",
        "search_terms(ontology_id='envo', query='zzz two', size=20)",
        "get_term_children(iri='http://purl.obolibrary.org/obo/ENVO_99999999', "
        "ontology_id='envo', size=10)",
    ]


def test_record_then_replay_roundtrip(tmp_path: Path) -> None:
    term = {"curie": "UO:0000008", "iri": iri_for("UO:0000008"), "label": "meter", "synonyms": []}
    inner = _FakeClient({"get_term": term})
    fixture_dir = tmp_path / "ols"
    rec = RecordingOLS(inner, fixture_dir, "record")
    assert fixture_dir.is_dir()  # record/auto create the directory

    assert rec.get_term("uo", iri_for("UO:0000008")) == term
    assert (rec.hits, rec.misses) == (0, 1) and len(inner.calls) == 1
    assert [m["method"] for m in rec.missed] == ["get_term"] and rec.missed[0]["mode"] == "record"

    args = {"ontology_id": "uo", "iri": iri_for("UO:0000008")}
    path = fixture_dir / f"{fixture_key('get_term', args)}.json"
    assert path == rec.fixture_path("get_term", args)
    # Byte format of mesa-anyjev's recorder: sorted keys, indent 1, UTF-8 kept.
    expected = json.dumps(
        {"method": "get_term", "args": args, "response": term},
        indent=1,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert path.read_text(encoding="utf-8") == expected

    # A second call in record mode is a hit: the inner client is not consulted again.
    assert rec.get_term("uo", iri_for("UO:0000008")) == term
    assert (rec.hits, rec.misses) == (1, 1) and len(inner.calls) == 1

    replay = RecordingOLS(None, fixture_dir, "replay")
    assert replay.get_term("uo", iri_for("UO:0000008")) == term
    assert replay.hits == 1


def test_auto_mode_records_misses_and_replays_hits(tmp_path: Path) -> None:
    inner = _FakeClient({"search_terms": [_hit("UO:0000008", "meter")]})
    rec = RecordingOLS(inner, tmp_path, "auto")
    assert rec.search_terms("meter", "uo", 20) == inner.responses["search_terms"]
    assert rec.search_terms("meter", "uo", 20) == inner.responses["search_terms"]
    assert (rec.hits, rec.misses, len(inner.calls)) == (1, 1, 1)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_off_mode_never_touches_disk(tmp_path: Path) -> None:
    inner = _FakeClient({"get_term": None})
    rec = RecordingOLS(inner, tmp_path / "never-created", "off")
    assert rec.get_term("uo", "x") is None
    assert rec.get_term("uo", "x") is None
    assert len(inner.calls) == 2
    assert (rec.hits, rec.misses, rec.missed) == (0, 2, [])
    assert not (tmp_path / "never-created").exists()


def test_no_inner_and_no_fixture_raises(tmp_path: Path) -> None:
    rec = RecordingOLS(None, tmp_path, "auto")
    with pytest.raises(
        RuntimeError, match=r"no inner client.*get_term\(iri='x', ontology_id='uo'\)"
    ):
        rec.get_term("uo", "x")
    assert rec.missed and rec.missed[0]["mode"] == "auto"


# -- OLSLayer -----------------------------------------------------------------------------------


def test_search_candidates_replayed(replay: RecordingOLS) -> None:
    """``ncbitaxon`` ``genus`` returns TAXRANK hits too: they are filtered out and logged."""
    layer = OLSLayer(replay, max_candidates=12)
    cands, log = layer.search_candidates(["genus"], "ncbitaxon")
    assert cands and len(cands) <= 12
    assert all(c.curie.startswith("NCBITaxon:") and c.ontology_id == "ncbitaxon" for c in cands)
    assert all(c.query == "genus" for c in cands)
    assert [(c.rank, c.curie) for c in cands] == sorted((c.rank, c.curie) for c in cands)
    assert log["ontology_id"] == "ncbitaxon" and log["queries"] == ["genus"]
    assert log["raw_hits"] == {"genus": 20}
    assert any(f.startswith("TAXRANK:") for f in log["filtered_out"])
    assert log["filtered_out"] == sorted(log["filtered_out"])
    assert log["n_candidates"] == len(cands)
    assert "errors" not in log
    assert layer.calls == 1 and replay.hits == 1


def test_search_candidates_merges_dedups_caps_and_skips(replay: RecordingOLS) -> None:
    layer = OLSLayer(replay, max_candidates=3)
    # Blank queries are skipped and only three are searched: the fourth has no fixture and
    # would raise ReplayMiss if it were called.
    cands, log = layer.search_candidates(
        ["", "genus", "class", "family", "never recorded query"], "ncbitaxon"
    )
    assert log["queries"] == ["genus", "class", "family"]
    assert len(cands) == 3 and log["n_candidates"] == 3
    assert len({c.curie for c in cands}) == 3
    assert layer.calls == 3 and replay.missed == []
    # The cap keeps the best (rank, curie) pairs of the merged set.
    full, _ = OLSLayer(replay, max_candidates=100).search_candidates(
        ["genus", "class", "family"], "ncbitaxon"
    )
    assert cands == full[:3]
    assert len(full) == len({c.curie for c in full})


def test_search_candidates_logs_errors_and_continues() -> None:
    def scripted(query: str, ontology_id: str | None, size: int) -> Any:
        return {
            "boom": OLSAPIError("OLS API error (503): down", status_code=503),
            "net": requests.ConnectionError("refused"),
            "rt": RuntimeError("no client"),
            "ok": [_hit("UO:0000008", "meter")],
        }[query]

    client = _FakeClient({"search_terms": scripted})
    layer = OLSLayer(client, max_candidates=12, search_size=7)
    cands, log = layer.search_candidates(["boom", "net", "ok", "rt"], "uo")
    assert [c.curie for c in cands] == ["UO:0000008"]
    assert log["queries"] == ["ok"] and log["raw_hits"] == {"ok": 1}
    assert log["errors"] == ["boom: OLS API error (503): down", "net: refused"]
    assert log["n_candidates"] == 1 and layer.calls == 3
    assert all(args["size"] == 7 for _, args in client.calls)


def test_biome_candidates_replayed(replay: RecordingOLS) -> None:
    layer = OLSLayer(replay, max_candidates=12)
    cands, log = layer.biome_candidates(["shrubland", "semi-arid desert grassland", "grassland"])
    assert cands and all(c.curie.startswith("ENVO:") and c.ontology_id == "envo" for c in cands)
    assert log["ontology_id"] == "envo" and log["parent"] == BIOME_IRI
    assert log["queries"] == ["shrubland", "semi-arid desert grassland", "grassland"]
    assert log["raw_hits"] == {"shrubland": 12, "semi-arid desert grassland": 0, "grassland": 12}
    assert log["filtered_out"] == [] and log["n_candidates"] == len(cands) == 12
    assert layer.calls == 3 and replay.hits == 3
    assert {c.query for c in cands} == {"shrubland", "grassland"}


def test_biome_candidates_logs_http_errors() -> None:
    client = _FakeClient(
        {"search_term_descendants": requests.HTTPError("500 Server Error: v1 search")}
    )
    cands, log = OLSLayer(client).biome_candidates(["desert"])
    assert cands == [] and log["queries"] == [] and log["n_candidates"] == 0
    assert log["errors"] == ["desert: 500 Server Error: v1 search"]
    assert client.calls[0][1]["parent_iri"] == BIOME_IRI and client.calls[0][1]["size"] == 20


def test_children_replayed(replay: RecordingOLS) -> None:
    layer = OLSLayer(replay)
    iri = iri_for("PATO:0000040")
    kids = layer.children("pato", iri)
    assert [c.curie for c in kids] == [
        "PATO:0000374",
        "PATO:0000375",
        "PATO:0002207",
        "PATO:0045014",
    ]
    assert all(c.query == f"children:{iri}" and c.ontology_id == "pato" for c in kids)
    assert layer.calls == 1


def test_unit_candidate_is_offline(replay: RecordingOLS) -> None:
    layer = OLSLayer(replay)
    cand = layer.unit_candidate("meter")
    assert cand is not None and cand.curie == "UO:0000008" and cand.ontology_id == "uo"
    assert layer.unit_candidate("furlong") is None
    assert layer.calls == 0 and (replay.hits, replay.misses, replay.missed) == (0, 0, [])
