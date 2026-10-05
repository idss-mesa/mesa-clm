"""The probe tier of the tiered provider (``providers.tiered``; plan §4.1, §5.3, §5.5; the M4
brief R4) over the fake stack: a promoted synthetic ``ProbeArtifact`` is served locally (the
fake encoder embeds, the fake head projects), its records are honest (``level='probe'``, the
calibrator's kind, ``feature_spec``, ``artifact_version`` and the artifact reference set,
``raw_probs`` the zero-shot parity distribution clm-serve would answer, ``s_c`` its log-ratios,
``probs`` the mirror of ``PlattCalibrator.rank_fit`` over ``logit(p_fit)``), closed choices work
the same way, an unpromoted probe is ``TierUnavailable``, K4 refuses another stack, the encoder
down gives ``unavailable`` records and a rejected key refuses the run, the vector cache reads the
store and never writes it, the spec features are the bench's formulas, and a pipeline run over a
promoted probe stores ``feature_spec`` rows. Synthetic only: random features, the fake transport."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings as fr
from mesa_clm.cards import DatasetCard
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.fake import FakeClm
from mesa_clm.clm.fingerprint import FingerprintMismatch
from mesa_clm.clm.headproj import l2, random_head
from mesa_clm.clm.http import ClmError, ClmHttpClient
from mesa_clm.learn import probe as learn_probe
from mesa_clm.providers import (
    ArtifactBundle,
    ArtifactError,
    DeciderRefused,
    DecisionRecord,
    FakeProvider,
    PlattCalibrator,
    TierUnavailable,
)
from mesa_clm.providers.tiered import (
    CHOICE_SPECS,
    PROBE_LOGIT_CLIP,
    RANK_FIT_SPECS,
    SPEC_MODELS,
    SPEC_NEEDS_HEAD,
    Promotion,
    ServedProbe,
    VectorCache,
    choice_features,
    fake_fingerprint,
    joint_texts,
    probe_rank_probs,
    rank_fit_features,
    spec_dim,
    spec_features,
)
from mesa_clm.states import column_state, target_state
from tests.fakes.clm_transport import CLM_URL, ENCODER_URL, FakeClmServer
from tests.fakes.m4 import FP, synthetic_probe
from tests.unit.test_policy import CANDS

TERM = fr.active_framing("term.fits")
ASPECT = fr.active_framing("column.aspect")
ONTOLOGY = fr.active_framing("column.ontology_fits")


def _col(card: DatasetCard, name: str = "observerDistance") -> Any:
    return next(c for c in card.columns if c.name == name)


def _target(card: DatasetCard) -> dict[str, Any]:
    return target_state(card, "column", "measurement", column=_col(card))


def _bundle(*probes: Any, promote: bool = True, version: str = "v3") -> ArtifactBundle:
    served = {p.question_key: ServedProbe(p, version=version) for p in probes}
    promoted = (
        {
            p.task_id: Promotion(tier="probe", question_key=p.question_key, version=version)
            for p in probes
        }
        if promote
        else {}
    )
    return ArtifactBundle(
        version=version,
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        probes=served,
        promoted=promoted,
    )


# -- rank_fit --------------------------------------------------------------------------------------


def test_rank_fit_probe_records_are_honest_and_mirror_platt(card: DatasetCard) -> None:
    probe = synthetic_probe("lowdim.v1")
    bundle = _bundle(probe)
    provider = FakeProvider(artifacts=bundle)
    assert provider.resolve_tier(TERM.question_key) == "probe"
    assert provider.supports_tier("term.fits", "probe") and provider.supports_tier(
        "term.fits", "auto"
    )
    assert not provider.supports_tier("term.fits", "calibrated")
    calls: list[Any] = []
    provider.on_call = calls.append
    [rec] = provider.decide(TERM, [_target(card)], [CANDS])
    assert isinstance(rec, DecisionRecord)
    assert (rec.method, rec.level, rec.calibration) == ("fake", "probe", "platt")
    assert rec.feature_spec == "lowdim.v1" and rec.artifact_version == "v3"
    assert rec.artifact is not None and rec.artifact.question_key == TERM.question_key
    assert (rec.artifact.encoder_fp, rec.artifact.clm_model_fp) == (FP.encoder_fp, FP.clm_model_fp)
    assert rec.served_model is None and rec.clm_confidence is None
    assert rec.diagnostics["probe"] is True and rec.diagnostics["spec"] == "lowdim.v1"
    assert rec.diagnostics["feature_dim"] == 2 and rec.diagnostics["n_embedded"] == 5
    # raw_probs is the zero-shot parity: what the fake clm-serve answers for the same question.
    [zs] = provider.decide(TERM, [_target(card)], [CANDS], tier="zero_shot")
    assert rec.raw_probs is not None and zs.raw_probs is not None
    assert max(abs(a - b) for a, b in zip(rec.raw_probs, zs.raw_probs, strict=True)) <= 1e-9
    assert rec.s_c is not None and zs.s_c is not None
    assert max(abs(a - b) for a, b in zip(rec.s_c, zs.s_c, strict=True)) <= 1e-9
    # p_fit per option is the probe's calibrated Yes probability; probs the Platt mirror.
    a = rec.anchor_index
    assert a is not None and rec.p_fit is not None and rec.probs is not None
    assert len(rec.p_fit) == len(CANDS) + 1 and all(0.0 <= p <= 1.0 for p in rec.p_fit)
    assert rec.probs == pytest.approx(probe_rank_probs(rec.p_fit, a))
    for i in range(len(CANDS)):
        assert rec.probs[i] / (rec.probs[i] + rec.probs[a]) == pytest.approx(rec.p_fit[i])
    assert rec.confidence == pytest.approx(max(rec.probs))
    # The encoder call is reported once (the LRU answers the next decision); the zero_shot
    # decision above went to clm-serve.
    embeds = [c for c in calls if c.endpoint == "/v1/embeddings"]
    assert len(embeds) == 1 and embeds[0].status == "ok"
    assert embeds[0].n_candidates == len(CANDS) + 1
    assert [c.endpoint for c in calls if c not in embeds] == ["/v1/systemone"]
    [again] = provider.decide(TERM, [_target(card)], [CANDS], tier="probe")
    assert again.p_fit == rec.p_fit and again.diagnostics["n_embedded"] == 0
    assert len([c for c in calls if c.endpoint == "/v1/embeddings"]) == 1
    assert provider.input_tokens == sum(int(c.input_tokens or 0) for c in calls)
    # The anchor's p_fit is the probe's value for a candidate scoring exactly like the anchor:
    # the features of the anchor's own vector on the candidate side (s_latest = s_raw = 0).
    expected_anchor = probe.predict(np.zeros((1, 2)))[0, 0]
    assert rec.p_fit[a] == pytest.approx(expected_anchor)


def test_probe_rank_probs_mirror_and_clip() -> None:
    cal = PlattCalibrator(a=2.0, b=-1.0)
    s_c = [1.5, -0.3, 0.0]
    p_fit, probs = cal.rank_fit(s_c, 2)
    # Platt's probs are softmax(a·s + b, 0 for the anchor) = softmax(logit(p_fit), 0).
    assert probe_rank_probs(p_fit, 2) == pytest.approx(probs)
    clipped = probe_rank_probs([1.0, 0.0, 0.5], 2)
    assert all(math.isfinite(p) for p in clipped) and clipped[0] > clipped[1]
    assert PROBE_LOGIT_CLIP == 1e-12


def test_ontology_fits_probe_is_masked_after_scoring(card: DatasetCard) -> None:
    probe = synthetic_probe("pair512.v1", task_id="column.ontology_fits", seed=2)
    provider = FakeProvider(artifacts=_bundle(probe))
    [rec] = provider.decide(ONTOLOGY, [_target(card)], [fr.ontology_candidates()])
    assert rec.level == "probe" and rec.masked is not None and any(rec.masked)
    assert rec.probs is not None and all(
        p == 0.0 for p, m in zip(rec.probs, rec.masked, strict=True) if m
    )
    assert rec.diagnostics["feature_dim"] == 2 * 512 + 1


# -- closed choices ---------------------------------------------------------------------------


def test_choice_probe_records(card: DatasetCard) -> None:
    probe = synthetic_probe("choice.state.v1", task_id="column.aspect", seed=1)
    provider = FakeProvider(artifacts=_bundle(probe))
    [rec] = provider.decide(ASPECT, [column_state(card, _col(card))])
    assert (rec.level, rec.calibration, rec.feature_spec) == (
        "probe",
        "temperature",
        "choice.state.v1",
    )
    assert rec.s_c is None and rec.p_fit is None and rec.anchor_index is None
    assert (
        rec.probs is not None and len(rec.probs) == 8 and math.fsum(rec.probs) == pytest.approx(1.0)
    )
    assert (
        rec.confidence == pytest.approx(max(rec.probs))
        and rec.answer == rec.options[rec.answer_index]
    )
    assert rec.diagnostics["feature_dim"] == 512 + 8
    [zs] = provider.decide(ASPECT, [column_state(card, _col(card))], tier="zero_shot")
    assert rec.raw_probs == pytest.approx(zs.raw_probs or [])
    # A probe of the wrong K for the framing is refused before any work.
    wrong = synthetic_probe("choice.state.v1", task_id="avu.value_kind", seed=1).model_copy(
        update={
            "question_key": ASPECT.question_key,
            "task_id": "column.aspect",
            "framing_id": ASPECT.id,
        }
    )
    with pytest.raises(ArtifactError, match="K=4"):
        FakeProvider(artifacts=_bundle(wrong)).decide(ASPECT, [column_state(card, _col(card))])


# -- promotion, refusals -------------------------------------------------------------------------


def test_an_unpromoted_probe_is_not_served(card: DatasetCard) -> None:
    probe = synthetic_probe("lowdim.v1")
    provider = FakeProvider(artifacts=_bundle(probe, promote=False))
    assert provider.resolve_tier(TERM.question_key) == "zero_shot"
    assert not provider.supports_tier("term.fits", "probe")
    [rec] = provider.decide(TERM, [_target(card)], [CANDS])
    assert rec.level == "zero_shot" and rec.feature_spec is None
    with pytest.raises(TierUnavailable, match=r"no probe for question_key .* is promoted"):
        provider.decide(TERM, [_target(card)], [CANDS], tier="probe")
    # Under a promotion table an unpromoted calibrator is not served either.
    bundle = ArtifactBundle(
        version="v3",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        calibrators={ASPECT.question_key: PlattCalibrator(a=1.0, b=0.0)},
        probes={probe.question_key: ServedProbe(probe, version="v3")},
        promoted={
            "term.fits": Promotion(tier="probe", question_key=probe.question_key, version="v3")
        },
    )
    provider = FakeProvider(artifacts=bundle)
    assert provider.resolve_tier(ASPECT.question_key) == "zero_shot"
    with pytest.raises(TierUnavailable, match="no calibrator"):
        provider.decide(ASPECT, [column_state(card, _col(card))], tier="calibrated")
    # A promotion naming a probe the bundle does not hold is refused at construction.
    with pytest.raises(ValueError, match="is not loaded"):
        ArtifactBundle(
            version="v3",
            encoder_fp=FP.encoder_fp,
            clm_model_fp=FP.clm_model_fp,
            promoted={
                "term.fits": Promotion(tier="probe", question_key=TERM.question_key, version="v3")
            },
        )


def test_k4_refusals() -> None:
    probe = synthetic_probe("lowdim.v1")
    other = fake_fingerprint(seed=7)
    # The bundle refuses a probe fitted under another stack ...
    with pytest.raises(ValueError, match="K4"):
        ArtifactBundle(
            version="v1",
            encoder_fp=other.encoder_fp,
            clm_model_fp=other.clm_model_fp,
            probes={probe.question_key: ServedProbe(probe, version="v1")},
        )
    # ... the provider a bundle fitted under another stack ...
    with pytest.raises(FingerprintMismatch, match="K4"):
        FakeProvider(artifacts=_bundle(probe), fingerprint=other)
    # ... a promoted probe of another served model than the provider's ...
    raw_fp = fake_fingerprint("clm-raw")
    raw = synthetic_probe("pair4096.v1", fp=raw_fp)
    raw_bundle = ArtifactBundle(
        version="v1",
        encoder_fp=raw_fp.encoder_fp,
        clm_model_fp=raw_fp.clm_model_fp,
        probes={raw.question_key: ServedProbe(raw, version="v1")},
        promoted={
            "term.fits": Promotion(tier="probe", question_key=raw.question_key, version="v1")
        },
    )
    FakeProvider(artifacts=raw_bundle, model="clm-raw")  # served by the clm-raw provider
    with pytest.raises(ArtifactError, match="belongs to clm-raw"):
        FakeProvider(artifacts=raw_bundle, model="clm-latest", fingerprint=raw_fp)
    # ... and a promoted probe without its head export or an encoder that embeds.
    with pytest.raises(ArtifactError, match="needs the served head"):
        FakeProvider(client=_StubClient(), artifacts=_bundle(probe), embed=FakeClm().encoder.embed)
    with pytest.raises(ArtifactError, match="no encoder embeds"):
        FakeProvider(client=_StubClient(), artifacts=_bundle(probe), head=random_head(0))
    # A probe whose spec belongs to another model than it claims, or an unknown spec.
    with pytest.raises(ArtifactError, match="belongs to"):
        ServedProbe(probe.model_copy(update={"model": "clm-raw"}), version="v1")
    with pytest.raises(ArtifactError, match="unknown feature spec"):
        ServedProbe(probe.model_copy(update={"spec": "nope.v9"}), version="v1")


class _StubClient:
    model = "clm-latest"

    def system_one(
        self, state: Any, questions: Any, *, model: Any = None, temperature: Any = None
    ) -> Any:
        raise AssertionError("the probe path never asks clm-serve")


def test_the_probe_path_never_asks_clm_serve_and_handles_the_encoder_down(
    card: DatasetCard,
) -> None:
    probe = synthetic_probe("lowdim.v1")
    clm = FakeClm(seed=0)
    down = {"error": None}

    def embed(texts: Sequence[str]) -> Any:
        if down["error"] is not None:
            raise down["error"]
        return clm.encoder.embed(texts)

    provider = FakeProvider(
        client=_StubClient(), artifacts=_bundle(probe), embed=embed, head=clm.heads["clm-latest"]
    )
    calls: list[Any] = []
    provider.on_call = calls.append
    [rec] = provider.decide(TERM, [_target(card)], [CANDS])
    assert rec.level == "probe"
    down["error"] = ClmError(0, "ConnectError: refused")
    [rec2] = provider.decide(
        TERM, [_target(card)], [[*CANDS, fr.FramingCandidate("UO:0000001", "unit", "x")]]
    )
    assert rec2.method == "unavailable" and rec2.reason == "decider_unavailable"
    assert calls[-1].status == "unavailable" and calls[-1].endpoint == "/v1/embeddings"
    down["error"] = ClmError(401, "invalid API key")
    with pytest.raises(DeciderRefused, match="rejected"):
        provider.decide(TERM, [_target(card)], [[fr.FramingCandidate("UO:0000002", "u2", "y")]])
    assert provider.decide(TERM, [_target(card)], [CANDS])[0].level == "probe"  # cached vectors


def test_through_the_wire_the_encoder_embeds_and_clm_serve_is_idle(card: DatasetCard) -> None:
    server = FakeClmServer()
    client = ClmHttpClient(CLM_URL, None, transport=server.transport(), sleep=lambda _s: None)
    encoder = EncoderClient(ENCODER_URL, None, transport=server.transport(), sleep=lambda _s: None)
    provider = FakeProvider(
        server.clm, client=client, encoder=encoder, artifacts=_bundle(synthetic_probe("lowdim.v1"))
    )
    [rec] = provider.decide(TERM, [_target(card)], [CANDS])
    assert rec.level == "probe" and rec.diagnostics["token_source"] == "server"
    assert "/v1/embeddings" in server.paths and "/v1/systemone" not in server.paths
    # The same question at zero shot goes to clm-serve and answers the probe's raw_probs.
    [zs] = provider.decide(TERM, [_target(card)], [CANDS], tier="zero_shot")
    assert "/v1/systemone" in server.paths
    assert (
        max(abs(a - b) for a, b in zip(rec.raw_probs or [], zs.raw_probs or [], strict=True))
        <= 1e-6
    )


# -- the vector cache ----------------------------------------------------------------------------


class _Store:
    """A read-only stand-in for the feature store: ``known`` texts have vectors."""

    def __init__(self, known: dict[str, np.ndarray]) -> None:
        self.known = known
        self.reads = 0
        self.writes = 0

    def exists(self) -> bool:
        return True

    def missing(self, texts: Sequence[str]) -> list[str]:
        return [t for t in dict.fromkeys(texts) if t not in self.known]

    def get(self, texts: Sequence[str]) -> np.ndarray:
        self.reads += len(texts)
        return np.stack([self.known[t] for t in texts])

    def add(self, *a: Any, **k: Any) -> None:
        self.writes += 1


def test_vector_cache_reads_the_store_then_the_encoder_and_is_bounded() -> None:
    rng = np.random.default_rng(0)
    store = _Store({"known": l2(rng.standard_normal((1, 8)))[0]})
    embedded: list[list[str]] = []

    def embed(texts: Sequence[str]) -> tuple[np.ndarray, int]:
        embedded.append(list(texts))
        return l2(rng.standard_normal((len(texts), 8))), 3 * len(texts)

    cache = VectorCache(embed, store=store, max_entries=3)
    got = cache.vectors(["known", "new", "known", "other"])
    assert got.rows.shape == (4, 8) and got.embedded == 2 and got.tokens == 6
    assert np.array_equal(got.rows[0], got.rows[2]) and embedded == [["new", "other"]]
    assert store.reads == 1 and store.writes == 0 and cache.store_hits == 1
    again = cache.vectors(["new", "known"])
    assert again.embedded == 0 and again.tokens == 0 and len(embedded) == 1
    assert np.array_equal(again.rows[0], got.rows[1])
    cache.vectors(["a", "b"])  # evicts the oldest beyond max_entries
    assert len(cache) == 3 and cache.embed_calls == 2


# -- the spec features are the bench's formulas ----------------------------------------------------


def test_spec_features_equal_the_bench_formulas() -> None:
    rng = np.random.default_rng(1)
    head = random_head(0, projection_dim=512)
    xs = l2(rng.standard_normal((1, 4096)))[0]
    xo = l2(rng.standard_normal((4, 4096)))
    xj = l2(rng.standard_normal((4, 4096)))
    a = 3
    zs = head.project_states(xs[None, :])[0].astype(np.float64)
    zo = head.project_actions(xo).astype(np.float64)
    s_raw = 100.0 * (
        xo.astype(np.float64) @ xs.astype(np.float64)
        - float(xo[a].astype(np.float64) @ xs.astype(np.float64))
    )
    s_latest = head.scale * (zo @ zs - float(zo[a] @ zs))
    expected = {
        "lowdim.v1": learn_probe.rank_fit_features("lowdim.v1", s_latest=s_latest, s_raw=s_raw),
        "pair512.v1": learn_probe.rank_fit_features(
            "pair512.v1", zs=np.repeat(zs[None, :], 4, 0), zc=zo, s_latest=s_latest
        ),
        "pair4096.v1": learn_probe.rank_fit_features(
            "pair4096.v1", xs=np.repeat(xs[None, :].astype(np.float64), 4, 0), xc=xo, s_raw=s_raw
        ),
        "joint4096@S1": learn_probe.rank_fit_features("joint4096@S1", joint=xj),
    }
    for spec, want in expected.items():
        got = spec_features(spec, xs=xs, xo=xo, anchor_index=a, head=head, xj=xj)
        assert got.shape == (4, spec_dim(spec)) and np.allclose(got, want, atol=1e-12), spec
        assert (
            got[a, -1] == pytest.approx(0.0)
            if spec in ("lowdim.v1", "pair512.v1", "pair4096.v1")
            else True
        )
    logits_latest = (head.scale * (zo @ zs))[None, :]
    assert np.allclose(
        choice_features("choice.state.v1", xs=xs, xo=xo, head=head),
        learn_probe.choice_features("choice.state.v1", zs=zs[None, :], logits=logits_latest),
    )
    assert np.allclose(
        choice_features("choice.raw.v1", xs=xs, xo=xo),
        learn_probe.choice_features(
            "choice.raw.v1", xs=xs[None, :], logits=(100.0 * (xo.astype(np.float64) @ xs))[None, :]
        ),
    )
    assert spec_features("choice.raw.v1", xs=xs, xo=xo, anchor_index=None).shape == (1, 4096 + 4)
    with pytest.raises(ValueError, match="needs the served head"):
        rank_fit_features("lowdim.v1", xs=xs, xc=xo, xa=xo[a])
    with pytest.raises(ValueError, match="joint context"):
        rank_fit_features("joint4096@S1ns", xs=xs, xc=xo, xa=xo[a])
    # The spec table agrees with the learn core's declaration.
    assert (
        {s.id for s in learn_probe.SPECS}
        == set(SPEC_MODELS)
        == set(RANK_FIT_SPECS) | set(CHOICE_SPECS)
    )
    for s in learn_probe.SPECS:
        assert SPEC_MODELS[s.id] == s.model and s.dim(8 if s.shape == "choice" else 2) == spec_dim(
            s.id, k=8 if s.shape == "choice" else 2
        )
        assert (s.id in SPEC_NEEDS_HEAD) == (s.model == "clm-latest")


def test_joint_texts_render_the_control_state(card: DatasetCard) -> None:
    texts = joint_texts("joint4096@S1", TERM, _target(card), CANDS)
    plain = joint_texts("joint4096@S1ns", TERM, _target(card), CANDS)
    control = fr.framing("term.fits", "F1")
    assert len(texts) == len(plain) == len(CANDS) and control.instructions
    assert all(t.endswith(control.instructions) for t in texts)
    assert all(not p.endswith(control.instructions) for p in plain)
    assert all("3" in t for t in texts)  # n_candidates is the group's size


# -- through the pipeline ---------------------------------------------------------------------


def test_a_pipeline_run_over_a_promoted_probe_stores_feature_spec(tmp_path: Path) -> None:
    from mesa_clm.provenance.store import DuckDBStore
    from tests.fakes.pipeline import FAKE_SEED, annotator
    from tests.fakes.pipeline import card as load

    probe = synthetic_probe("lowdim.v1", fp=fake_fingerprint(seed=FAKE_SEED))
    bundle = ArtifactBundle(
        version="v1",
        encoder_fp=probe.fingerprint["encoder_fp"],
        clm_model_fp=probe.fingerprint["clm_model_fp"],
        probes={probe.question_key: ServedProbe(probe, version="v1")},
        promoted={
            "term.fits": Promotion(tier="probe", question_key=probe.question_key, version="v1")
        },
    )
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    ann = annotator(store, provider=FakeProvider(seed=FAKE_SEED, artifacts=bundle))
    run = ann.annotate(load("DP1.10003.001.brd_countdata"))
    rows = store.decisions(run.run_id)
    probes = [r for r in rows if r["level"] == "probe"]
    assert probes and all(r["task_id"] == "term.fits" for r in probes)
    assert {r["feature_spec"] for r in probes} == {"lowdim.v1"}
    assert {r["artifact_version"] for r in probes} == {"v1"} and {
        r["calibration"] for r in probes
    } == {"platt"}
    assert all(r["feature_spec"] is None for r in rows if r["level"] != "probe")
    assert store.run(run.run_id)["artifacts_version"] == "v1"  # type: ignore[index]
    assert any(c["endpoint"] == "/v1/embeddings" for c in store.clm_calls(run.run_id))


def test_serving_features_equal_the_feature_builder_on_stored_vectors(tmp_path: Path) -> None:
    """Term for term: the provider's per-request features built from the store's vectors equal
    ``FeatureBuilder.matrix`` rows on the same vectors, every non-joint rank_fit spec."""
    from mesa_clm.bench.cells import item_identities
    from mesa_clm.learn.probe import FeatureBuilder
    from mesa_clm.registry import ANCHOR_KEY
    from tests.unit.test_cells import _scorers, rank_world

    world = rank_world(tmp_path / "world", n_targets=6)
    scorers = _scorers(world.store)
    fb = FeatureBuilder(world.task, world.index, world.store, scorers, "F7")
    head = scorers["clm-latest"].projector
    assert head is not None
    ids = item_identities(world.task)
    for spec in ("lowdim.v1", "pair512.v1", "pair4096.v1"):
        matrix = fb.matrix(spec)
        for i, (target, option) in enumerate(ids):
            texts = [
                world.index.text("term.fits", "F7", target, "", "context"),
                world.index.text("term.fits", "F7", target, option, "candidate"),
                world.index.text("term.fits", "F7", target, ANCHOR_KEY, "anchor"),
            ]
            xs, xc, xa = world.store.get(texts)
            row = rank_fit_features(spec, xs=xs, xc=xc[None, :], xa=xa, head=head)[0]
            assert np.allclose(row, matrix[i], atol=1e-9), (spec, i)


def test_a_probe_refusing_its_features_abstains(card: DatasetCard) -> None:
    """A dimension mismatch (the artifact's ``predict`` raising a ValueError subclass) is a
    malformed answer, never an uncaught exception."""
    probe = synthetic_probe("pair512.v1")  # 1025 features
    wrong = ServedProbe(probe.model_copy(update={"spec": "lowdim.v1"}), version="v1")
    bundle = ArtifactBundle(
        version="v1",
        encoder_fp=FP.encoder_fp,
        clm_model_fp=FP.clm_model_fp,
        probes={wrong.question_key: wrong},
        promoted={
            "term.fits": Promotion(tier="probe", question_key=wrong.question_key, version="v1")
        },
    )
    [rec] = FakeProvider(artifacts=bundle).decide(TERM, [_target(card)], [CANDS])
    assert rec.method == "unavailable" and rec.reason == "malformed_answer"
    with pytest.raises(ArtifactError, match="refused the features"):
        wrong.predict(np.zeros((1, 2)))
