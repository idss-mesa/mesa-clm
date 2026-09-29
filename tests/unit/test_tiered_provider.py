"""The tiered provider (``providers.tiered``; DESIGN D2, D5, D6, D7, D23, D28; plan §4.1-4.3,
§5.3): records built from the framing registry's wire questions, one request per shared
context, the ``clm_calls`` callback, closed choices, the aspect mask, the zero_shot and
calibrated tiers (Platt, temperature, the artifact bundle and its K4 refusal), probe/head
refusals, the token guard, failure handling, the fake provider in and out of process, and the
degraded ``ols_rank`` method."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from mesa_clm import framings as fr
from mesa_clm import render
from mesa_clm.cards import DatasetCard
from mesa_clm.clm.encoder import EncoderClient, chars_estimate
from mesa_clm.clm.fake import FakeClm
from mesa_clm.clm.fingerprint import EncoderSpec, FingerprintMismatch, encoder_fp
from mesa_clm.clm.http import ClmError, ClmHttpClient, Question, SystemOneResponse, question_to_dict
from mesa_clm.identity import target_sha256
from mesa_clm.net import CircuitBreaker
from mesa_clm.providers import (
    CALL_STATUSES,
    ArtifactBundle,
    ArtifactError,
    CalibratorError,
    ClaudeStructuredProvider,
    ClmCall,
    DecisionProvider,
    DecisionRecord,
    DecisionRequest,
    FakeClmClient,
    FakeProvider,
    OlsRankProvider,
    PlattCalibrator,
    TemperatureCalibrator,
    TieredProvider,
    TierUnavailable,
    fake_fingerprint,
    load_calibrator,
    ols_rank_record,
    sha256_text,
    sigmoid,
)
from mesa_clm.registry import ANCHOR_KEY, ANCHORS, ANNOTATE_OPTIONS, ASPECT_OPTIONS, VALUE_KINDS
from mesa_clm.states import column_state, state_sha256, target_state, value_kind_state
from tests.fakes.clm_transport import CLM_URL, ENCODER_URL, FakeClmServer

FP = fake_fingerprint()
TERM = fr.active_framing("term.fits")
ONTOLOGY = fr.active_framing("column.ontology_fits")
ANNOTATE = fr.active_framing("column.annotate")
ASPECT = fr.active_framing("column.aspect")
VALUE_KIND = fr.active_framing("avu.value_kind")
CANDS = [
    fr.FramingCandidate("PATO:0000040", "distance", "A 1-D extent quality between two points."),
    fr.FramingCandidate("PATO:0000014", "color", "A composite chromatic quality."),
    fr.FramingCandidate("UO:0000008", "meter", "A length unit equal to the SI base unit."),
]
TERM_JSON = {"label": "distance", "curie": "PATO:0000040"}


def _col(card: DatasetCard, name: str = "observerDistance") -> Any:
    return next(c for c in card.columns if c.name == name)


@pytest.fixture
def target(card: DatasetCard) -> dict[str, Any]:
    return target_state(card, "column", "measurement", column=_col(card))


@pytest.fixture
def server() -> FakeClmServer:
    return FakeClmServer()


def _client(server: FakeClmServer, **kw: Any) -> ClmHttpClient:
    kw.setdefault("sleep", lambda _s: None)
    return ClmHttpClient(CLM_URL, kw.pop("api_key", None), transport=server.transport(), **kw)


class StubClient:
    """A scripted clm-serve: each question's distribution is its keys' ``weights`` (default 1)
    normalised, or whatever ``answer(qid, question)`` returns."""

    model = "clm-latest"

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        answer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        latency_ms: float | None = 3.0,
    ) -> None:
        self.weights = dict(weights or {})
        self.answer = answer
        self.latency_ms = latency_ms
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        qs = {k: question_to_dict(q) for k, q in questions.items()}
        self.calls.append((state, qs))
        answers: dict[str, Any] = {}
        for qid, q in qs.items():
            if self.answer is not None:
                answers[qid] = self.answer(qid, q)
                continue
            keys = list(q["criteria"])
            w = [self.weights.get(k, 1.0) for k in keys]
            answers[qid] = render.answer_from_probs(q, keys, [x / sum(w) for x in w])
        return SystemOneResponse.model_validate(
            {
                "model": model or self.model,
                "answers": answers,
                "usage": {"billing_units": len(qs), "input_tokens": 7},
                "latency_ms": self.latency_ms,
            }
        )


def _stub_provider(client: StubClient, **kw: Any) -> TieredProvider:
    return TieredProvider(client, None, FP, method="clm", **kw)


def _bundle(**calibrators: Any) -> ArtifactBundle:
    return ArtifactBundle(
        version="v3",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        calibrators=calibrators,
    )


# -- records and requests ---------------------------------------------------------------------------


def test_zero_shot_rank_fit_record_and_its_request(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    calls: list[ClmCall] = []
    provider = FakeProvider(client=_client(server), on_call=calls.append)
    [rec] = provider.decide(TERM, [target], [CANDS])
    # The wire body is the framing registry's, temperature left to the server (plan §4.1).
    assert server.bodies == [
        {
            "state": fr.build_context(TERM, target),
            "model": "clm-latest",
            "questions": {"term.fits": fr.build_question(TERM, CANDS)},
        }
    ]
    assert rec.options == [c.key for c in CANDS] + [ANCHOR_KEY]
    assert rec.option_texts == [fr.candidate_text(TERM, c) for c in CANDS] + [ANCHORS["term"]]
    assert (rec.task_id, rec.task_key, rec.kind, rec.shape) == (
        "term.fits",
        TERM.task_key,
        "noul",
        "rank_fit",
    )
    assert (rec.question_key, rec.framing_id) == (TERM.question_key, "F7")
    assert (rec.provider, rec.method, rec.model, rec.served_model) == (
        "fake",
        "fake",
        "clm-latest",
        "clm-latest",
    )
    assert (rec.encoder_fp, rec.clm_model_fp, rec.schema_sha256) == (
        FP.encoder_fp,
        FP.clm_model_fp,
        FP.schema_sha256,
    )
    assert rec.state == target and rec.state_sha256 == state_sha256(target)
    assert rec.target_sha256 == target_sha256("term.fits", target)
    text = fr.context_text(TERM, target)
    assert rec.context_sha256 == sha256_text(text)
    assert rec.context_tokens == chars_estimate(text) and rec.diagnostics["token_source"] == "chars"
    assert rec.level == "zero_shot" and rec.calibration == "uncalibrated"
    assert rec.probs == rec.raw_probs and rec.confidence == max(rec.probs or [])
    assert rec.clm_confidence == pytest.approx(render.clm_confidence(rec.raw_probs or []))
    assert rec.p_fit == [sigmoid(s) for s in rec.s_c or []]
    assert rec.latency_ms == 12.5  # the server's X-CLM-Latency-Ms
    [call] = calls
    assert (call.endpoint, call.model, call.n_questions, call.n_candidates) == (
        "/v1/systemone",
        "clm-latest",
        1,
        len(CANDS) + 1,
    )
    assert call.status == "ok" and call.billing_units == 1 and call.latency_ms == 12.5
    assert call.input_tokens is not None and call.input_tokens > 0 and call.error is None
    assert provider.n_calls == 1 and provider.input_tokens == call.input_tokens


def test_questions_sharing_a_context_go_in_one_request(
    server: FakeClmServer, card: DatasetCard, target: dict[str, Any]
) -> None:
    calls: list[ClmCall] = []
    provider = FakeProvider(client=_client(server), on_call=calls.append)
    col_a = column_state(card, _col(card))
    col_b = column_state(card, _col(card, "scientificName"))
    recs = provider.decide_many(
        [
            DecisionRequest(ANNOTATE, col_a),
            DecisionRequest(TERM, target, CANDS),
            DecisionRequest(ASPECT, col_a),
            DecisionRequest(ASPECT, col_b),
            DecisionRequest(TERM, target, CANDS[:2]),
        ]
    )
    assert [r.task_id for r in recs] == [
        "column.annotate",
        "term.fits",
        "column.aspect",
        "column.aspect",
        "term.fits",
    ]
    assert len(server.requests) == 3  # col_a (2 questions), target (2 groups), col_b
    questions = [list(b["questions"]) for b in server.bodies]
    assert ["column.annotate", "column.aspect"] in questions
    assert ["term.fits", "term.fits#1"] in questions
    assert ["column.aspect"] in questions
    assert sorted(c.n_questions for c in calls) == [1, 2, 2]
    assert recs[1].diagnostics["qid"] == "term.fits" and recs[4].diagnostics["qid"] == "term.fits#1"
    assert recs[4].options == [c.key for c in CANDS[:2]] + [ANCHOR_KEY]


def test_closed_choice_records(server: FakeClmServer, card: DatasetCard) -> None:
    provider = FakeProvider(client=_client(server))
    col = _col(card)
    [aspect] = provider.decide(ASPECT, [column_state(card, col)])
    assert aspect.options == list(ASPECT_OPTIONS) and aspect.option_texts == list(ASPECT_OPTIONS)
    assert aspect.anchor_index is None and aspect.s_c is None and aspect.p_fit is None
    assert aspect.kind == "choice" and aspect.answer in ASPECT_OPTIONS
    [annotate] = provider.decide(ANNOTATE, [column_state(card, col)])
    assert annotate.options == ["Yes", "No"] and annotate.kind == "noul"
    assert annotate.option_texts == list(ANNOTATE_OPTIONS.values())
    [vk] = provider.decide(VALUE_KIND, [value_kind_state(card, col, TERM_JSON, "measurement")])
    assert vk.options == list(VALUE_KINDS) and vk.answer in VALUE_KINDS
    # The wire keys are the framing's, the record's options the mesa-anyjev labels.
    assert list(server.bodies[-1]["questions"]["avu.value_kind"]["criteria"]) == list(
        fr.VALUE_KIND_KEYS
    )


def test_anchor_wins_or_loses_as_clm_says(target: dict[str, Any]) -> None:
    [none] = _stub_provider(StubClient({ANCHOR_KEY: 5.0})).decide(TERM, [target], [CANDS])
    assert none.anchor_won and none.answer == ANCHOR_KEY and none.level == "zero_shot"
    assert none.answer_s_c == 0.0 and none.answer_p_fit == 0.5
    [pick] = _stub_provider(StubClient({"UO:0000008": 5.0})).decide(TERM, [target], [CANDS])
    assert pick.answer == "UO:0000008" and not pick.anchor_won
    assert pick.answer_s_c == pytest.approx(math.log(5.0))
    assert pick.answer_p_fit == pytest.approx(5.0 / 6.0)


# -- the aspect mask (Q3) ---------------------------------------------------------------------------


def test_ontology_fits_is_masked_by_the_state_aspect(card: DatasetCard) -> None:
    unit = target_state(card, "column", "unit", column=_col(card))
    client = StubClient({"pato": 8.0, "uo": 2.0})
    [rec] = _stub_provider(client).decide(ONTOLOGY, [unit], [fr.ontology_candidates()])
    assert rec.masked == [o not in ("uo", ANCHOR_KEY) for o in rec.options]
    assert rec.answer == "uo"
    raw = rec.raw_probs or []
    kept = raw[rec.options.index("uo")] + raw[-1]
    assert rec.diagnostics["masked_mass"] == pytest.approx(1.0 - kept)
    assert rec.probs is not None and rec.probs[rec.options.index("pato")] == 0.0
    assert rec.probs[rec.options.index("uo")] == pytest.approx(2.0 / 3.0)
    # Nothing allowed for the aspect among the candidates: the record abstains.
    only = fr.ontology_candidates(["pato", "ncbitaxon"])
    [empty] = _stub_provider(StubClient()).decide(ONTOLOGY, [unit], [only])
    assert empty.answer_index == -1 and empty.reason == "mask_empty"


# -- tiers ------------------------------------------------------------------------------------------


def test_tier_resolution_without_artifacts(target: dict[str, Any]) -> None:
    provider = FakeProvider()
    assert provider.resolve_tier(TERM.question_key) == "zero_shot"
    assert provider.supports_tier("term.fits", "zero_shot") and provider.supports_tier(
        "column.aspect", "auto"
    )
    for tier in ("calibrated", "probe", "head", "none"):
        assert not provider.supports_tier("term.fits", tier)
    assert not provider.supports_tier("column.ontology", "zero_shot")  # never asked
    with pytest.raises(TierUnavailable, match="no calibrator for question_key"):
        provider.decide(TERM, [target], [CANDS], tier="calibrated")
    with pytest.raises(TierUnavailable, match=r"probe tier .* lands in M4"):
        provider.decide(TERM, [target], [CANDS], tier="probe")
    with pytest.raises(TierUnavailable, match=r"head tier .* lands in M7"):
        provider.decide(TERM, [target], [CANDS], tier="head")
    with pytest.raises(TierUnavailable, match="OlsRankProvider"):
        provider.decide(TERM, [target], [CANDS], tier="none")
    with pytest.raises(ValueError, match="unknown tier"):
        provider.decide(TERM, [target], [CANDS], tier="L2")


def test_calibrated_rank_fit_applies_platt(target: dict[str, Any]) -> None:
    bundle = _bundle(**{TERM.question_key: PlattCalibrator(a=2.0, b=-1.0, n=285)})
    provider = _stub_provider(StubClient({"PATO:0000040": 3.0}), artifacts=bundle)
    assert provider.resolve_tier(TERM.question_key) == "calibrated"
    assert provider.supports_tier("term.fits", "calibrated")
    assert not provider.supports_tier("column.aspect", "calibrated")
    [rec] = provider.decide(TERM, [target], [CANDS])
    assert (rec.level, rec.calibration, rec.artifact_version) == ("calibrated", "platt", "v3")
    assert rec.artifact is not None and rec.artifact.question_key == TERM.question_key
    assert rec.s_c is not None and rec.p_fit is not None and rec.probs is not None
    assert rec.p_fit == [sigmoid(2.0 * s - 1.0) for s in rec.s_c]
    a = rec.anchor_index
    assert a is not None
    for i in range(len(CANDS)):
        # Each candidate's odds against the anchor under probs are exactly its p_fit.
        assert rec.probs[i] / (rec.probs[i] + rec.probs[a]) == pytest.approx(rec.p_fit[i])
    assert "inverted" not in rec.diagnostics
    [explicit] = provider.decide(TERM, [target], [CANDS], tier="calibrated")
    assert explicit == rec.revise(latency_ms=explicit.latency_ms)
    [zs] = provider.decide(TERM, [target], [CANDS], tier="zero_shot")
    assert zs.level == "zero_shot" and zs.artifact is None and zs.s_c == rec.s_c


def test_calibrated_choice_and_temperature(card: DatasetCard, target: dict[str, Any]) -> None:
    col = column_state(card, _col(card))
    bundle = _bundle(
        **{
            ASPECT.question_key: TemperatureCalibrator(T=2.0),
            ANNOTATE.question_key: PlattCalibrator(a=-1.5, b=0.25),
            TERM.question_key: TemperatureCalibrator(T=4.0),
        }
    )
    provider = _stub_provider(StubClient({"taxon": 4.0, "Yes": 3.0}), artifacts=bundle)
    [aspect] = provider.decide(ASPECT, [col])
    raw = aspect.raw_probs or []
    expected = render.softmax([math.log(p) / 2.0 for p in raw])
    assert aspect.calibration == "temperature" and aspect.probs == pytest.approx(expected)
    [annotate] = provider.decide(ANNOTATE, [col])
    r = annotate.raw_probs or []
    p0 = sigmoid(-1.5 * (math.log(r[0]) - math.log(r[1])) + 0.25)
    assert annotate.calibration == "platt" and annotate.probs == pytest.approx([p0, 1 - p0])
    assert annotate.diagnostics["inverted"] is True  # a < 0 is flagged (#15)
    [term] = provider.decide(TERM, [target], [CANDS])
    assert term.p_fit == pytest.approx([sigmoid(s / 4.0) for s in term.s_c or []])


def test_platt_on_a_wide_choice_is_refused_before_any_request(card: DatasetCard) -> None:
    client = StubClient()
    provider = _stub_provider(
        client, artifacts=_bundle(**{ASPECT.question_key: PlattCalibrator(a=1, b=0)})
    )
    with pytest.raises(CalibratorError, match="K=2"):
        provider.decide(ASPECT, [column_state(card, _col(card))])
    assert client.calls == []
    with pytest.raises(CalibratorError, match="K=2"):
        PlattCalibrator(a=1.0, b=0.0).choice([0.2, 0.3, 0.5])


def test_bundle_fitted_under_another_stack_is_refused() -> None:
    other = EncoderSpec(model="Qwen/Qwen3-8B", revision="r", route="transformers")
    bundle = ArtifactBundle(
        version="v1", encoder_fp=encoder_fp(other), clm_model_fp=FP.clm_model_fp
    )
    with pytest.raises(FingerprintMismatch, match="K4"):
        FakeProvider(artifacts=bundle)


def test_bundle_and_calibrator_io(tmp_path: Path) -> None:
    bundle = _bundle(**{TERM.question_key: PlattCalibrator(a=1.25, b=-0.5)})
    path = tmp_path / "calibrators.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    loaded = ArtifactBundle.load(path)
    assert loaded == bundle and loaded.calibrator_for(TERM.question_key) == PlattCalibrator(
        a=1.25, b=-0.5
    )
    assert loaded.calibrator_for("0" * 16) is None
    assert load_calibrator({"kind": "temperature", "T": 1.5}) == TemperatureCalibrator(T=1.5)
    with pytest.raises(ArtifactError, match="not found"):
        ArtifactBundle.load(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        ArtifactBundle.load(tmp_path / "bad.json")
    data = json.loads(bundle.model_dump_json())
    data["calibrators"] = {"term.fits": {"kind": "platt", "a": 1, "b": 0}}
    (tmp_path / "key.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="validation error"):
        ArtifactBundle.load(tmp_path / "key.json")
    with pytest.raises(ValueError):
        load_calibrator({"kind": "temperature", "T": 0.0})
    with pytest.raises(ValueError):
        load_calibrator({"kind": "platt", "a": math.inf, "b": 0.0})
    with pytest.raises(ValueError):
        load_calibrator({"kind": "isotonic"})


# -- the token guard (D23) --------------------------------------------------------------------------


def test_a_context_over_the_window_is_never_sent(
    server: FakeClmServer, card: DatasetCard, target: dict[str, Any]
) -> None:
    calls: list[ClmCall] = []
    provider = FakeProvider(client=_client(server), max_len=64, on_call=calls.append)
    short = column_state(card, _col(card))
    recs = provider.decide(TERM, [target], [CANDS])
    [rec] = recs
    assert rec.truncated and rec.method == "unavailable" and rec.level == "none"
    assert rec.answer_index == -1 and rec.probs is None and rec.reason == "truncated"
    assert rec.anchor_index == len(CANDS) and rec.context_tokens > 64 - 16
    assert server.requests == [] and calls == []
    assert FakeProvider(max_len=10_000).decide(ASPECT, [short])[0].truncated is False
    with pytest.raises(ValueError, match="guard margin"):
        FakeProvider(max_len=16)


def test_the_encoder_counts_tokens_and_failures_fall_back(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    encoder = EncoderClient(
        ENCODER_URL, None, transport=server.transport(), retries=1, sleep=lambda _s: None
    )
    provider = FakeProvider(client=_client(server), encoder=encoder)
    assert provider.max_len == encoder.max_len
    [rec] = provider.decide(TERM, [target], [CANDS])
    text = fr.context_text(TERM, target)
    assert rec.context_tokens == len(text.split()) and rec.diagnostics["token_source"] == "server"
    # Counted once per text: the second decision needs no /tokenize.
    before = server.paths.count("/tokenize")
    provider.decide(TERM, [target], [CANDS[:1]])
    assert server.paths.count("/tokenize") == before
    other = FakeProvider(
        client=_client(server),
        encoder=EncoderClient(
            ENCODER_URL, None, transport=server.transport(), retries=1, sleep=lambda _s: None
        ),
    )
    server.fail_next = [500]
    [fallback] = other.decide(TERM, [target], [CANDS])
    assert fallback.diagnostics["token_source"] == "chars"
    assert fallback.context_tokens == chars_estimate(text) and fallback.answer_index >= 0


def test_the_token_cache_is_bounded(
    target: dict[str, Any], card: DatasetCard, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mesa_clm.providers.tiered as tiered

    monkeypatch.setattr(tiered, "_TOKEN_CACHE_MAX", 1)
    provider = FakeProvider()
    provider.decide(TERM, [target], [CANDS])
    provider.decide(ASPECT, [column_state(card, _col(card))])
    assert len(provider._tokens) == 1  # cleared wholesale, then refilled


# -- failures -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (503, "error"),
        (httpx.ConnectError("connection refused"), "unavailable"),
        (httpx.ReadTimeout("slow"), "timeout"),
    ],
)
def test_a_failed_request_gives_unavailable_records(
    server: FakeClmServer, target: dict[str, Any], failure: Any, status: str
) -> None:
    calls: list[ClmCall] = []
    provider = FakeProvider(client=_client(server, retries=2), on_call=calls.append)
    server.fail_next = [failure, failure]
    recs = provider.decide(TERM, [target, target], [CANDS, CANDS[:1]])
    assert len(server.requests) == 2  # one request (two attempts) for both groups
    for rec in recs:
        assert rec.method == "unavailable" and rec.level == "none" and rec.probs is None
        assert rec.answer_index == -1 and rec.reason == "decider_unavailable"
        assert not rec.truncated and "ClmError" in str(rec.diagnostics["error"])
    [call] = calls
    assert call.status == status and call.n_questions == 2 and call.error is not None
    assert provider.n_calls == 0


def test_refused_and_breaker_open_requests(server: FakeClmServer, target: dict[str, Any]) -> None:
    keyed = FakeClmServer(clm_api_key="right")
    calls: list[ClmCall] = []
    [rec] = FakeProvider(client=_client(keyed, api_key="wrong"), on_call=calls.append).decide(
        TERM, [target], [CANDS]
    )
    assert rec.reason == "decider_unavailable" and "401" in str(rec.diagnostics["error"])
    assert calls[-1].status == "error"
    breaker = CircuitBreaker(failures=1, open_for=60.0)
    breaker.failure()
    provider = FakeProvider(client=_client(server, breaker=breaker), on_call=calls.append)
    [rec] = provider.decide(TERM, [target], [CANDS])
    assert rec.reason == "decider_unavailable" and "BreakerOpenError" in str(
        rec.diagnostics["error"]
    )
    assert calls[-1].status == "unavailable" and server.requests == []
    [unknown] = FakeProvider(model="no-such-head").decide(TERM, [target], [CANDS])
    assert unknown.reason == "decider_unavailable" and "422" in str(unknown.diagnostics["error"])


def _answer(probs: dict[str, float], choice: str | None = None) -> dict[str, Any]:
    top = choice or max(probs, key=probs.__getitem__)
    return {"type": "choice", "choice": top, "confidence": 0.1, "probabilities": probs}


def test_malformed_and_underflowing_answers_abstain(target: dict[str, Any]) -> None:
    keys = [c.key for c in CANDS] + [ANCHOR_KEY]

    def missing(qid: str, q: dict[str, Any]) -> dict[str, Any]:
        return _answer({k: 1.0 / (len(keys) - 1) for k in keys[1:]})

    def underflow(qid: str, q: dict[str, Any]) -> dict[str, Any]:
        return _answer({k: (0.0 if i == 0 else 1.0 / (len(keys) - 1)) for i, k in enumerate(keys)})

    def unnormalised(qid: str, q: dict[str, Any]) -> dict[str, Any]:
        return _answer({k: 0.5 for k in keys})

    def noul(qid: str, q: dict[str, Any]) -> dict[str, Any]:
        return {"type": "noul", "noul": 0.4}

    def disagrees(qid: str, q: dict[str, Any]) -> dict[str, Any]:
        return _answer(
            {k: 1.0 / len(keys) + (0.01 if i == 0 else -0.01 / 3) for i, k in enumerate(keys)},
            choice=ANCHOR_KEY,
        )

    expect = {
        missing: "malformed_answer",
        underflow: "numeric_underflow",
        unnormalised: "malformed_answer",
        noul: "malformed_answer",
    }
    for fn, reason in expect.items():
        [rec] = _stub_provider(StubClient(answer=fn)).decide(TERM, [target], [CANDS])
        assert rec.method == "unavailable" and rec.reason == reason and rec.answer_index == -1
    [rec] = _stub_provider(StubClient(answer=disagrees)).decide(TERM, [target], [CANDS])
    assert rec.answer == keys[0] and rec.diagnostics["clm_choice"] == ANCHOR_KEY


def test_bad_batches_are_programming_errors(card: DatasetCard, target: dict[str, Any]) -> None:
    provider = FakeProvider()
    col = column_state(card, _col(card))
    with pytest.raises(ValueError, match="one candidate group per state"):
        provider.decide(TERM, [target, target], [CANDS])
    with pytest.raises(ValueError, match="one candidate group per state"):
        provider.decide(TERM, [target])
    with pytest.raises(ValueError, match="takes no candidates"):
        provider.decide(ASPECT, [col], [CANDS])
    with pytest.raises(fr.FramingError, match="noul control"):
        provider.decide(fr.framing("term.fits", "F1"), [target], [CANDS])
    with pytest.raises(fr.FramingError, match="at least one candidate"):
        provider.decide(TERM, [target], [[]])
    assert provider.decide(TERM, [], []) == []


# -- the fake in and out of process ------------------------------------------------------------------


def test_fake_provider_in_process_matches_the_wire(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    [wire] = FakeProvider(client=_client(server)).decide(TERM, [target], [CANDS])
    [local] = FakeProvider().decide(TERM, [target], [CANDS])
    assert local.raw_probs == wire.raw_probs and local.s_c == wire.s_c
    assert local.latency_ms is not None and local.latency_ms >= 0.0  # wall time, no header
    client = FakeClmClient(FakeClm(seed=0), latency_ms=1.5)
    assert (
        client.system_one(
            fr.build_context(TERM, target), {"q": fr.build_question(TERM, CANDS)}
        ).latency_ms
        == 1.5
    )
    with pytest.raises(ClmError, match="422"):
        client.system_one("s", {"q": {"type": "essay"}})
    seeded = FakeProvider(seed=7).decide(TERM, [target], [CANDS])[0]
    assert seeded.s_c != local.s_c and seeded.clm_model_fp == local.clm_model_fp


def test_fake_fingerprint() -> None:
    assert fake_fingerprint() == FP
    assert fake_fingerprint(seed=1).encoder_fp != FP.encoder_fp
    assert fake_fingerprint("clm-raw").clm_model_fp != FP.clm_model_fp
    real = EncoderSpec(model="Qwen/Qwen3-8B", revision="b968826d", route="vllm")
    assert encoder_fp(real) != FP.encoder_fp


# -- ols_rank (D28) ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedCandidate:
    key: str
    label: str
    description: str
    rank: int


def test_ols_rank_records(card: DatasetCard, target: dict[str, Any]) -> None:
    provider = OlsRankProvider(FP)
    [top] = provider.decide(TERM, [target], [CANDS])
    assert (top.method, top.level, top.calibration, top.provider) == (
        "ols_rank",
        "none",
        "none",
        "ols_rank",
    )
    assert top.answer == CANDS[0].key and top.rank == 1 and top.probs is None
    assert top.context_tokens == 0 and top.diagnostics["context_sent"] is False
    assert top.context_sha256 == sha256_text(fr.context_text(TERM, target))
    second = ols_rank_record(TERM, target, CANDS, FP, pick=1)
    assert second.answer == CANDS[1].key and second.rank == 2
    ranked = [RankedCandidate("PATO:0000040", "distance", "", 3)]
    assert ols_rank_record(TERM, target, ranked, FP).rank == 3
    assert provider.resolve_tier(TERM.question_key) == "none"
    assert provider.supports_tier("term.fits", "none") and provider.supports_tier(
        "column.ontology_fits", "auto"
    )
    assert not provider.supports_tier("column.aspect", "none")
    assert not provider.supports_tier("term.fits", "zero_shot")
    assert not provider.supports_tier("nope", "none")
    with pytest.raises(TierUnavailable, match="level 'none' only"):
        provider.decide(TERM, [target], [CANDS], tier="calibrated")
    with pytest.raises(fr.FramingError, match="rank_fit"):
        ols_rank_record(ASPECT, column_state(card, _col(card)), CANDS, FP)
    with pytest.raises(fr.FramingError, match="at least one candidate"):
        ols_rank_record(TERM, target, [], FP)
    with pytest.raises(ValueError, match="out of range"):
        ols_rank_record(TERM, target, CANDS, FP, pick=3)


# -- contracts shared with other modules -------------------------------------------------------------


def test_every_provider_satisfies_the_protocol() -> None:
    from mesa_clm.config import ClaudeConfig

    for p in (
        FakeProvider(),
        TieredProvider(StubClient(), None, FP),
        OlsRankProvider(FP),
        ClaudeStructuredProvider(ClaudeConfig(), FP, client=object()),
    ):
        assert isinstance(p, DecisionProvider), type(p).__name__


def test_call_statuses_match_the_sidecar() -> None:
    from mesa_clm.provenance.models import CALL_STATUSES as SIDECAR

    assert CALL_STATUSES == SIDECAR


def test_records_are_decision_records(target: dict[str, Any]) -> None:
    recs = FakeProvider().decide(TERM, [target, target], [CANDS, CANDS[1:]])
    assert all(isinstance(r, DecisionRecord) for r in recs) and len(recs) == 2
