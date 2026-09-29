"""Wire parity between ``ClmHttpClient`` and CLM's own client (``tests/_vendor/clm_client.py``,
the oracle; DESIGN D4) plus the committed goldens in ``tests/fixtures/clm_golden/``.

Both clients run against the same ``httpx.MockTransport`` handler (the vendored ``requests``
client through the adapter in ``tests/fakes/clm_transport.py``); the bodies they put on the wire
are compared parsed *and* in key order, and both parse the same responses to the same values.
The goldens record, per case, the vendored client's request body, the exact bytes
``ClmHttpClient`` must send (``body_json``), the fake server's response and what the vendored
client parsed from it. ``MESA_CLM_WRITE_GOLDENS=1`` rewrites them through the oracle; the
default run asserts the oracle still produces the committed bodies and that our client is
byte-identical to them.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from mesa_clm.cards import DatasetCard
from mesa_clm.clm.http import (
    Choice,
    ChoiceAnswer,
    ClmHttpClient,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    question_to_dict,
)
from mesa_clm.registry import ANCHOR_KEY, ANCHORS, ANNOTATE_OPTIONS, ONTOLOGY_REGISTRY
from mesa_clm.states import target_state
from tests._vendor import clm_client as vendored
from tests.fakes.clm_transport import CLM_URL, FakeClmServer, wire_json

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "clm_golden"
API_KEY = "golden-key-not-a-secret"
PROB_TOL = 1e-9

TERM_CANDIDATES = [
    (
        "PATO:0000040",
        "distance: A 1-D extent quality which is equal to the distance between two points.",
    ),
    (
        "PATO:0000122",
        "length: A 1-D extent quality which is equal to the distance between two points.",
    ),
    (
        "UO:0000008",
        "meter: A length unit which is equal to the length of the path travelled by light.",
    ),
]
UNIT_ANSWERS = ["meter", "degree Celsius", "percent"]


@dataclass(frozen=True)
class Case:
    """One golden: the vendored-shaped inputs and the same inputs as our dataclasses."""

    name: str
    kind: str  # "systemone" | "rank"
    state: Any
    vendored_questions: dict[str, Any]
    our_questions: dict[str, Any]
    model: str | None = None
    temperature: float | None = None
    question: str | None = None  # rank only
    answers: tuple[str, ...] = ()  # rank only

    @property
    def path(self) -> str:
        return "/v1/systemone" if self.kind == "systemone" else "/v1/rank"


def golden_cases(card: DatasetCard) -> list[Case]:
    col = card.column("observerDistance")
    term_state = target_state(card, "column", "measurement", column=col)
    fit_criteria: dict[str, Any] = dict(TERM_CANDIDATES)
    fit_criteria[ANCHOR_KEY] = ANCHORS["term"]
    name_state = target_state(card, "column", "taxon", column=card.column("scientificName"))
    ont_criteria: dict[str, Any] = {e.id: e.option_text for e in ONTOLOGY_REGISTRY[:4]}
    ont_criteria[ANCHOR_KEY] = ANCHORS["ontology"]
    prose_state = (
        "Column airTemperature: mean air temperature over 30 minutes, in °C (unit: degree Celsius)."
    )
    return [
        Case(
            "term_fits_observerDistance",
            "systemone",
            term_state,
            {"fit": vendored.Choice(criteria=fit_criteria)},
            {"fit": Choice(criteria=fit_criteria)},
        ),
        Case(
            "ontology_fits_and_annotate_scientificName",
            "systemone",
            name_state,
            {
                "ontology": vendored.Choice(criteria=ont_criteria, instructions=None),
                "annotate": vendored.Choice(criteria=dict(ANNOTATE_OPTIONS)),
            },
            {
                "ontology": Choice(criteria=ont_criteria),
                "annotate": Choice(criteria=ANNOTATE_OPTIONS),
            },
            model="clm-raw",
            temperature=1.0,
        ),
        Case(
            "control_noul_score_unicode",
            "systemone",
            prose_state,
            {
                "n1": vendored.Noul(instructions="Is this column a measured quantity?"),
                "n2": vendored.Noul(
                    instructions="Is the unit a temperature unit?",
                    criteria={
                        "true": "A temperature unit such as °C or K.",
                        "false": "Not a temperature unit.",
                    },
                ),
                "n3": vendored.Noul(instructions="Empty criteria are dropped", criteria={}),
                "s": vendored.Score(
                    criteria=["Calm", "Frustrated", "Very angry"], instructions="How hot?"
                ),
                "wire": {
                    "type": "choice",
                    "instructions": "raw dict",
                    "criteria": {"a": "alpha", "b": "beta"},
                },
            },
            {
                "n1": Noul(instructions="Is this column a measured quantity?"),
                "n2": Noul(
                    instructions="Is the unit a temperature unit?",
                    criteria={
                        "true": "A temperature unit such as °C or K.",
                        "false": "Not a temperature unit.",
                    },
                ),
                "n3": Noul(instructions="Empty criteria are dropped", criteria={}),
                "s": Score(criteria=("Calm", "Frustrated", "Very angry"), instructions="How hot?"),
                "wire": {
                    "type": "choice",
                    "instructions": "raw dict",
                    "criteria": {"a": "alpha", "b": "beta"},
                },
            },
            temperature=2.0,
        ),
        Case(
            "rank_units_no_question",
            "rank",
            term_state,
            {},
            {},
            question=None,
            answers=tuple(UNIT_ANSWERS),
        ),
    ]


def _answer_dict(a: Any) -> dict[str, Any]:
    """A vendored dataclass answer or one of ours, as the wire dict (probabilities included)."""
    if isinstance(a, vendored.NoulAnswer | NoulAnswer):
        return {"type": "noul", "noul": a.noul, "probabilities": a.probabilities}
    if isinstance(a, vendored.ChoiceAnswer | ChoiceAnswer):
        return {
            "type": "choice",
            "choice": a.choice,
            "confidence": a.confidence,
            "probabilities": dict(a.probabilities),
        }
    assert isinstance(a, vendored.ScoreAnswer | ScoreAnswer)
    return {
        "type": "score",
        "score": a.score,
        "confidence": a.confidence,
        "probabilities": dict(a.probabilities),
        "legend": dict(a.legend),
    }


def _parsed(resp: Any) -> dict[str, Any]:
    """The vendored ``SystemOneResponse`` or ours, flattened for comparison."""
    return {
        "model": resp.model,
        "answers": {k: _answer_dict(a) for k, a in resp.answers.items()},
        "usage": {
            "billing_units": resp.usage.billing_units,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
        "latency_ms": resp.latency_ms,
    }


def _run_vendored(server: FakeClmServer, case: Case) -> tuple[dict[str, Any], Any]:
    client = vendored.CLMClient(CLM_URL, API_KEY)
    client._s = server.requests_session()
    client._s.headers["Authorization"] = f"Bearer {API_KEY}"
    before = len(server.requests)
    if case.kind == "systemone":
        out: Any = client.system_one(
            case.state, case.vendored_questions, model=case.model, temperature=case.temperature
        )
    else:
        out = client.rank(
            case.state,
            case.question,
            list(case.answers),
            model=case.model,
            temperature=case.temperature,
        )
    assert len(server.requests) == before + 1
    return json.loads(server.requests[-1].content), out


def _run_ours(transport: httpx.BaseTransport, case: Case) -> tuple[bytes, Any]:
    with ClmHttpClient(CLM_URL, API_KEY, transport=transport, sleep=lambda _s: None) as client:
        recorder: list[httpx.Request] = []
        client.endpoint._client.event_hooks["request"] = [recorder.append]
        if case.kind == "systemone":
            out: Any = client.system_one(
                case.state, case.our_questions, model=case.model, temperature=case.temperature
            )
        else:
            out = client.rank(
                case.state,
                case.question,
                case.answers,
                model=case.model,
                temperature=case.temperature,
            )
    assert len(recorder) == 1
    return recorder[0].content, out


def _assert_close(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and not isinstance(a, bool):
        assert isinstance(b, int | float) and math.isclose(a, b, rel_tol=0, abs_tol=PROB_TOL), (
            f"{path}: {a} vs {b}"
        )
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture
def server() -> FakeClmServer:
    return FakeClmServer(clm_api_key=API_KEY, latency_ms=12.5)


def _cases(card: DatasetCard) -> Iterator[Case]:
    yield from golden_cases(card)


# -- the same transport, both clients ------------------------------------------------------------


def test_bodies_identical_parsed_and_in_key_order(server: FakeClmServer, card: DatasetCard) -> None:
    for case in _cases(card):
        vendored_body, vendored_out = _run_vendored(server, case)
        server.clm.reset_cache()  # both clients see a cold server: usage.input_tokens agrees
        our_bytes, our_out = _run_ours(server.transport(), case)
        our_body = json.loads(our_bytes)
        assert our_body == vendored_body, case.name
        # json.loads keeps key order; re-dumping without sorting compares the order too.
        assert wire_json(our_body) == wire_json(vendored_body) == our_bytes.decode("utf-8"), (
            case.name
        )
        assert server.requests[-1].headers["Authorization"] == f"Bearer {API_KEY}"
        assert server.requests[-2].headers["Authorization"] == f"Bearer {API_KEY}"
        if case.kind == "systemone":
            assert isinstance(our_out, SystemOneResponse)
            _assert_close(_parsed(our_out), _parsed(vendored_out), case.name)
        else:
            assert [r.model_dump() for r in our_out] == vendored_out, case.name


def test_question_dataclasses_to_dict_match_vendored() -> None:
    pairs = [
        (Noul(), vendored.Noul()),
        (Noul(instructions="q"), vendored.Noul(instructions="q")),
        (Noul(instructions="q", criteria={}), vendored.Noul(instructions="q", criteria={})),
        (
            Noul(instructions="q", criteria={"true": "t"}),
            vendored.Noul(instructions="q", criteria={"true": "t"}),
        ),
        (Choice(criteria={"a": "A"}), vendored.Choice(criteria={"a": "A"})),
        (
            Choice(criteria={"a": "A"}, instructions="i"),
            vendored.Choice(criteria={"a": "A"}, instructions="i"),
        ),
        (Choice({"a": None}, "i"), vendored.Choice({"a": None}, "i")),
        (Score(criteria=["x", "y"]), vendored.Score(criteria=["x", "y"])),
        (Score(("x", "y"), "i"), vendored.Score(["x", "y"], "i")),
    ]
    for ours, theirs in pairs:
        assert ours.to_dict() == theirs.to_dict()
        assert wire_json(ours.to_dict()) == wire_json(theirs.to_dict())
        assert question_to_dict(ours) == vendored.question_to_dict(theirs)
    raw = {"type": "choice", "instructions": None, "criteria": {"k": "v"}}
    assert question_to_dict(raw) == vendored.question_to_dict(raw) == raw
    assert question_to_dict(raw) is not raw  # ours copies; equality is what the wire needs


def test_vendored_oracle_is_the_pinned_file() -> None:
    import hashlib

    sha = hashlib.sha256(Path(vendored.__file__).read_bytes()).hexdigest()
    assert sha == "45f565c1b30964dedc8f66ed4d4448ad43233dc4a8eb62558d539d87d2485738"


# -- goldens -------------------------------------------------------------------------------------


def _golden(case: Case) -> Path:
    return GOLDEN_DIR / f"{case.name}.json"


def _record(server: FakeClmServer, case: Case) -> dict[str, Any]:
    server.clm.reset_cache()
    body, out = _run_vendored(server, case)
    # Replay the same request on a cold cache to capture exactly what the vendored client parsed.
    server.clm.reset_cache()
    replay = server(server.requests[-1])
    return {
        "kind": case.kind,
        "path": case.path,
        "body": body,
        "body_json": wire_json(body),
        "response": json.loads(replay.content),
        "latency_ms": replay.headers["X-CLM-Latency-Ms"],
        "parsed": _parsed(out) if case.kind == "systemone" else out,
    }


@pytest.mark.skipif(
    not os.environ.get("MESA_CLM_WRITE_GOLDENS"), reason="set MESA_CLM_WRITE_GOLDENS=1 to rewrite"
)
def test_write_goldens(server: FakeClmServer, card: DatasetCard) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for case in _cases(card):
        _golden(case).write_text(
            json.dumps(_record(server, case), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def test_goldens_exist_for_every_case(card: DatasetCard) -> None:
    names = {c.name for c in _cases(card)}
    assert {p.stem for p in GOLDEN_DIR.glob("*.json")} == names
    assert len(names) >= 3


def test_vendored_oracle_still_produces_the_goldens(
    server: FakeClmServer, card: DatasetCard
) -> None:
    """The committed request bodies are what the vendored client sends today; the responses
    match the fake up to float noise (the goldens are not a lock on BLAS summation order)."""
    for case in _cases(card):
        golden = json.loads(_golden(case).read_text(encoding="utf-8"))
        server.clm.reset_cache()
        body, out = _run_vendored(server, case)
        assert body == golden["body"], case.name
        assert wire_json(body) == golden["body_json"], case.name
        assert golden["path"] == case.path
        parsed = _parsed(out) if case.kind == "systemone" else out
        _assert_close(parsed, golden["parsed"], case.name)


def test_clm_http_client_is_byte_identical_to_the_goldens(card: DatasetCard) -> None:
    """Our request bytes equal the golden ``body_json`` and, fed the golden response, our
    parser agrees with the vendored client's recorded parse exactly."""
    for case in _cases(card):
        golden = json.loads(_golden(case).read_text(encoding="utf-8"))
        seen: list[httpx.Request] = []

        def replay(
            request: httpx.Request,
            golden: dict[str, Any] = golden,
            seen: list[httpx.Request] = seen,
        ) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200, json=golden["response"], headers={"X-CLM-Latency-Ms": golden["latency_ms"]}
            )

        our_bytes, out = _run_ours(httpx.MockTransport(replay), case)
        assert our_bytes.decode("utf-8") == golden["body_json"], case.name
        assert seen[0].url.path == golden["path"]
        if case.kind == "systemone":
            assert _parsed(out) == golden["parsed"], case.name
            assert out.latency_ms == float(golden["latency_ms"])
        else:
            assert [r.model_dump() for r in out] == golden["parsed"], case.name


def test_golden_bodies_carry_the_documented_shapes(card: DatasetCard) -> None:
    """The wire details RESEARCH.md records: Choice criteria are description-only, the anchor
    key rides inside the criteria, temperature is absent unless given, rank sends a null
    question and no instructions key."""
    goldens = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in GOLDEN_DIR.glob("*.json")}
    fit = goldens["term_fits_observerDistance"]["body"]
    assert list(fit) == ["state", "model", "questions"] and fit["model"] == "clm-latest"
    q = fit["questions"]["fit"]
    assert q["type"] == "choice" and q["instructions"] is None
    assert list(q["criteria"])[-1] == ANCHOR_KEY and q["criteria"][ANCHOR_KEY] == ANCHORS["term"]
    assert list(fit["state"]) == ["card", "scope", "aspect", "column"]  # target last (D23)
    two = goldens["ontology_fits_and_annotate_scientificName"]["body"]
    assert list(two) == ["state", "model", "questions", "temperature"] and two["model"] == "clm-raw"
    assert set(two["questions"]) == {"ontology", "annotate"}
    ctl = goldens["control_noul_score_unicode"]["body"]
    assert "criteria" not in ctl["questions"]["n1"] and "criteria" not in ctl["questions"]["n3"]
    assert ctl["questions"]["n2"]["criteria"]["true"].startswith("A temperature unit")
    assert (
        "°C" in goldens["control_noul_score_unicode"]["body_json"]
    )  # ensure_ascii=False on the wire
    rank = goldens["rank_units_no_question"]["body"]
    assert list(rank) == ["context", "question", "answers", "model"] and rank["question"] is None
    assert rank["answers"] == UNIT_ANSWERS
