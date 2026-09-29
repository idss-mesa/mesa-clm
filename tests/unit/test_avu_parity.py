"""``build_avu`` equals mesa-mcp's ``mesa_avu_from_term`` byte for byte (mesa-anyjev amendment
B1): on the fixed cases, and on every recorded ``get_term`` fixture of a registry ontology with
the tool resolving the label itself through the replayed OLS client. Hermetic: no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from mesa_mcp.ols.client import OLSClient, _label_to_snake
from mesa_mcp.ols.tools.avu_from_term import AvuFromTermInput, handle_avu_from_term

from mesa_clm.avu import build_avu
from mesa_clm.ols import Candidate, RecordingOLS, iri_for, to_candidates
from mesa_clm.registry import ONTOLOGY_REGISTRY

OLS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ols"
REGISTRY_IDS = {e.id for e in ONTOLOGY_REGISTRY}


def _tool(args: AvuFromTermInput, client: RecordingOLS | None = None) -> dict[str, str]:
    # ``handle_avu_from_term`` types ``client`` as the concrete OLSClient; it only calls
    # ``get_term``, which RecordingOLS provides.
    out = asyncio.run(handle_avu_from_term(args, client=cast(OLSClient, client)))
    avu: dict[str, str] = out["avu"]
    return avu


def _recorded_terms() -> list[tuple[str, dict[str, Any]]]:
    """(ontology_id, term) for every recorded ``get_term`` of a registry ontology."""
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(OLS_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["method"] != "get_term" or doc["response"] is None:
            continue
        ontology_id = doc["args"]["ontology_id"]
        if ontology_id in REGISTRY_IDS:
            out.append((ontology_id, doc["response"]))
    return out


def test_parity_with_the_tool() -> None:
    cases = [
        ("envo", "temperate deciduous broadleaf forest", "ENVO:01000385", "HARV"),
        ("ncbitaxon", "Aves", "NCBITaxon:8782", "Aves"),
        ("uo", "degree Celsius", "UO:0000027", "degree Celsius"),
        ("uo", "kilometer", "UO:0010066", "kilometer"),
        ("pato", "has number of", "PATO:0001555", "clusterSize"),
    ]
    for ont, label, curie, value in cases:
        cand = Candidate(label=label, curie=curie, iri=iri_for(curie), ontology_id=ont)
        ours = build_avu(cand, value)
        theirs = _tool(
            AvuFromTermInput(ontology_id=ont, value=value, iri=cand.iri, curie=curie, label=label)
        )
        assert ours == theirs, (ours, theirs)
        assert ours["unit"] == curie and ours["attribute"].startswith(ont + ".")


def test_parity_over_every_recorded_get_term() -> None:
    """The tool is given only ``ontology_id``, ``iri`` and ``curie`` and looks the label up
    through the replayed fixtures, exactly as an agent call without ``label`` would."""
    terms = _recorded_terms()
    assert len(terms) >= 100
    replay = RecordingOLS(None, OLS_DIR, "replay")
    n_checked = 0
    for ontology_id, term in terms:
        if not _label_to_snake(term["label"]):
            continue
        # The candidate as the pipeline would hold it (filters permitting) or straight from
        # the recorded term: build_avu only reads label, curie and ontology_id.
        cands = to_candidates([term], ontology_id, "")
        cand = (
            cands[0]
            if cands
            else Candidate(
                label=term["label"], curie=term["curie"], iri=term["iri"], ontology_id=ontology_id
            )
        )
        for value in (cand.label, "HARV", "some free-form value"):
            ours = build_avu(cand, value)
            theirs = _tool(
                AvuFromTermInput(
                    ontology_id=ontology_id, value=value, iri=cand.iri, curie=cand.curie
                ),
                client=replay,
            )
            assert ours == theirs, (cand.curie, value, ours, theirs)
            assert ours["unit"] == cand.curie
            assert ours["attribute"] == f"{ontology_id}.{_label_to_snake(cand.label)}"
        n_checked += 1
    assert n_checked >= 100
    assert replay.missed == [] and replay.misses == 0 and replay.hits == 3 * n_checked


def test_parity_through_replayed_unit_lookup() -> None:
    """Defect (a) end to end: the fixed ``unit_candidate`` rows build the same AVU the tool
    builds after resolving the CURIE's label from the recorded ``get_term``."""
    from mesa_clm.ols import UNIT_TABLE, OLSLayer

    replay = RecordingOLS(None, OLS_DIR, "replay")
    layer = OLSLayer(replay)
    for name in UNIT_TABLE:
        cand = layer.unit_candidate(name)
        assert cand is not None
        ours = build_avu(cand, cand.label)
        theirs = _tool(
            AvuFromTermInput(ontology_id="uo", value=cand.label, iri=cand.iri, curie=cand.curie),
            client=replay,
        )
        assert ours == theirs, (name, ours, theirs)
    assert replay.missed == []
