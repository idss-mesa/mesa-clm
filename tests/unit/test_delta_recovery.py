"""s_c recovery is set-invariant (DESIGN D2, D24; plan §4.1, M1 track B acceptance).

CLM's probabilities are a softmax over the offered set, so a candidate's probability moves when
other candidates join or leave. The fixed anchor fixes that: ``s_c = ln p_c - ln p_anchor =
(scale/T)(cos_c - cos_anchor)`` does not depend on the rest of the set. These tests ask the
fake engine (:class:`mesa_clm.clm.fake.FakeClm`) through the wire (``ClmHttpClient`` over
``tests/fakes/clm_transport.py``) with random subsets and orders of a 12-candidate group and
check every candidate's ``s_c`` (and ``p_fit``, zero-shot and calibrated) moves by at most 1e-9,
while its raw probability visibly does; that ``s_c`` is exactly the engine's scaled cosine gap;
and that packing, caching and masking leave it alone.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from mesa_clm import framings as fr
from mesa_clm.cards import DatasetCard
from mesa_clm.clm.http import ClmHttpClient
from mesa_clm.providers import (
    ArtifactBundle,
    DecisionRecord,
    FakeProvider,
    PlattCalibrator,
    fake_fingerprint,
)
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.states import target_state
from tests.fakes.clm_transport import CLM_URL, FakeClmServer

TOL = 1e-9
TERM = fr.active_framing("term.fits")
ONTOLOGY = fr.active_framing("column.ontology_fits")
CANDS = [
    fr.FramingCandidate("PATO:0000040", "distance", "A 1-D extent quality equal to the gap."),
    fr.FramingCandidate("PATO:0000122", "length", "A 1-D extent quality along the longest axis."),
    fr.FramingCandidate("PATO:0000014", "color", "A composite chromatic quality."),
    fr.FramingCandidate("PATO:0000125", "mass", "A physical quality of a thing's matter."),
    fr.FramingCandidate("UO:0000008", "meter", "A length unit equal to the SI base unit."),
    fr.FramingCandidate("UO:0010066", "kilometer", "A length unit equal to 1000 meters."),
    fr.FramingCandidate("ENVO:00000428", "biome", "An environmental system."),
    fr.FramingCandidate("NCBITaxon:8782", "Aves", ""),
    fr.FramingCandidate("OBI:0000070", "assay", "A planned process to produce information."),
    fr.FramingCandidate("IAO:0000027", "data item", "An information content entity."),
    fr.FramingCandidate("PATO:0001595", "speed", "A physical quality of the rate of motion."),
    fr.FramingCandidate("PATO:0000146", "temperature", "A physical quality of thermal energy."),
]


@pytest.fixture
def target(card: DatasetCard) -> dict[str, Any]:
    col = next(c for c in card.columns if c.name == "observerDistance")
    return target_state(card, "column", "measurement", column=col)


@pytest.fixture
def server() -> FakeClmServer:
    return FakeClmServer()


def _provider(server: FakeClmServer, **kw: Any) -> FakeProvider:
    return FakeProvider(client=ClmHttpClient(CLM_URL, None, transport=server.transport()), **kw)


def _by_key(rec: DecisionRecord) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    assert rec.s_c is not None and rec.p_fit is not None and rec.raw_probs is not None
    return (
        dict(zip(rec.options, rec.s_c, strict=True)),
        dict(zip(rec.options, rec.p_fit, strict=True)),
        dict(zip(rec.options, rec.raw_probs, strict=True)),
    )


def _subsets(rng: random.Random) -> list[list[fr.FramingCandidate]]:
    out = [list(reversed(CANDS))]
    for n in range(1, len(CANDS)):
        for _ in range(3):
            out.append(rng.sample(CANDS, n))
    return out


def test_s_c_does_not_move_when_the_set_changes(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    provider = _provider(server)
    [full] = provider.decide(TERM, [target], [CANDS])
    s_full, p_full, raw_full = _by_key(full)
    assert s_full[ANCHOR_KEY] == 0.0
    raw_moved = 0.0
    for subset in _subsets(random.Random(0)):
        [rec] = provider.decide(TERM, [target], [subset])
        s_sub, p_sub, raw_sub = _by_key(rec)
        assert rec.options[-1] == ANCHOR_KEY and s_sub[ANCHOR_KEY] == 0.0
        for c in subset:
            assert abs(s_sub[c.key] - s_full[c.key]) <= TOL, c.key
            assert abs(p_sub[c.key] - p_full[c.key]) <= TOL, c.key
            raw_moved = max(raw_moved, abs(raw_sub[c.key] - raw_full[c.key]))
    # The served probabilities are set-relative; only the anchor-relative score is not.
    assert raw_moved > 1e-3
    assert len(server.requests) == 1 + len(_subsets(random.Random(0)))


def test_calibrated_p_fit_is_set_invariant_too(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    fp = fake_fingerprint()
    bundle = ArtifactBundle(
        version="v1",
        encoder_fp=fp.encoder_fp,
        clm_model_fp=fp.clm_model_fp,
        calibrators={TERM.question_key: PlattCalibrator(a=4.0, b=0.3)},
    )
    provider = _provider(server, artifacts=bundle)
    [full] = provider.decide(TERM, [target], [CANDS])
    assert full.level == "calibrated"
    s_full, p_full, _ = _by_key(full)
    for subset in _subsets(random.Random(1))[:12]:
        [rec] = provider.decide(TERM, [target], [subset])
        s_sub, p_sub, _ = _by_key(rec)
        for c in subset:
            assert abs(s_sub[c.key] - s_full[c.key]) <= TOL
            assert abs(p_sub[c.key] - p_full[c.key]) <= TOL
            assert p_sub[c.key] == pytest.approx(
                1.0 / (1.0 + math.exp(-(4.0 * s_sub[c.key] + 0.3)))
            )


def test_s_c_is_the_engines_scaled_cosine_gap(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    """``s_c = (scale/T)(cos_c - cos_anchor)`` with the fake head's own projections (T = 1)."""
    [rec] = _provider(server).decide(TERM, [target], [CANDS])
    head = server.clm.heads["clm-latest"]
    enc = server.clm.encoder
    xs, _ = enc.embed([fr.context_text(TERM, target)])
    xa, _ = enc.embed(rec.option_texts)
    zs = head.project_states(xs)[0].astype(np.float64)
    za = head.project_actions(xa).astype(np.float64)
    cos = za @ zs
    a = rec.anchor_index
    assert a is not None and rec.s_c is not None
    expected = head.scale * (cos - cos[a])
    np.testing.assert_allclose(rec.s_c, expected, rtol=0, atol=TOL)


def test_s_c_is_the_log_ratio_of_the_wire_answer(
    server: FakeClmServer, target: dict[str, Any]
) -> None:
    [rec] = _provider(server).decide(TERM, [target], [CANDS[:5]])
    body = server.bodies[-1]
    assert body["state"] == fr.build_context(TERM, target)
    assert list(body["questions"]) == ["term.fits"]
    probs = server.clm.answer(body["state"], body["questions"])["answers"]["term.fits"][
        "probabilities"
    ]
    assert rec.s_c is not None
    for key, s in zip(rec.options, rec.s_c, strict=True):
        assert s == pytest.approx(math.log(probs[key]) - math.log(probs[ANCHOR_KEY]), abs=1e-12)


def test_packing_and_caching_do_not_move_s_c(target: dict[str, Any]) -> None:
    """Two groups over one context go out as one request; a cold server and a warm one agree."""
    warm = FakeClmServer()
    provider = _provider(warm)
    [alone] = provider.decide(TERM, [target], [CANDS[:6]])
    packed = provider.decide(TERM, [target, target], [CANDS[:6], CANDS[4:]])
    assert len(warm.requests) == 2  # the second call carried both groups
    assert list(warm.bodies[-1]["questions"]) == ["term.fits", "term.fits#1"]
    cold = FakeClmServer()
    [fresh] = _provider(cold).decide(TERM, [target], [CANDS[4:]])
    s_alone, _, _ = _by_key(alone)
    s_packed0, _, _ = _by_key(packed[0])
    s_packed1, _, _ = _by_key(packed[1])
    s_fresh, _, _ = _by_key(fresh)
    for key in s_alone:
        assert abs(s_alone[key] - s_packed0[key]) <= TOL
    for key in s_fresh:
        assert abs(s_fresh[key] - s_packed1[key]) <= TOL
        if key in s_alone:
            assert abs(s_fresh[key] - s_alone[key]) <= TOL


def test_the_aspect_mask_moves_probs_not_s_c(server: FakeClmServer, target: dict[str, Any]) -> None:
    """column.ontology_fits under ``measurement``: the full registry is masked after scoring, the
    allowed subset needs no mask, and both give every allowed ontology the same s_c."""
    provider = _provider(server)
    [masked] = provider.decide(ONTOLOGY, [target], [fr.ontology_candidates()])
    allowed = [c for c in fr.ontology_candidates() if c.key in ("envo", "pato", "obi", "pco")]
    [subset] = provider.decide(ONTOLOGY, [target], [allowed])
    assert masked.masked is not None and any(masked.masked)
    assert subset.masked == [False] * subset.k
    assert float(masked.diagnostics["masked_mass"]) > 0.0
    assert subset.diagnostics["masked_mass"] == 0.0
    s_m, p_m, _ = _by_key(masked)
    s_s, p_s, _ = _by_key(subset)
    for key in s_s:
        assert abs(s_m[key] - s_s[key]) <= TOL and abs(p_m[key] - p_s[key]) <= TOL
    # After the mask the in-play options and the anchor carry the same relative mass as the
    # subset's own distribution (renormalising over the same kept set).
    assert masked.probs is not None and subset.probs is not None
    for i, key in enumerate(masked.options):
        if key in subset.options:
            assert masked.probs[i] == pytest.approx(subset.probs[subset.options.index(key)])
