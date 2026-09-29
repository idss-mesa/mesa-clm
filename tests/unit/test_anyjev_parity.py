"""100% parity with mesa-anyjev's cards, states and registry (DESIGN D1, D23; M0 acceptance).

``tests/fixtures/anyjev_parity.json`` was recorded from mesa-anyjev by
``scripts/gen_anyjev_parity.py`` (run under mesa-anyjev's own interpreter; its docstring
documents the encoding). Every recorded state is rebuilt here with the mesa-clm builders from
the same inputs and compared as a dict, as ``state_sha256`` and as the order-sensitive
``ordered_sha256`` (key order at every depth). Nothing here reads the sibling checkout or the
network: CI does not have mesa-anyjev.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import cards as m_cards
from mesa_clm import registry as m_registry
from mesa_clm import states as m_states

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CARDS_DIR = FIXTURES / "cards"
PARITY_PATH = FIXTURES / "anyjev_parity.json"
LOCK_PATH = FIXTURES / "anyjev_questions.lock.json"

PARITY: dict[str, Any] = json.loads(PARITY_PATH.read_text(encoding="utf-8"))

# The mesa-anyjev task keys mesa-clm keeps (plan D1, §4.2); the lock copy must carry them.
KNOWN_TASK_KEYS = {
    "column.annotate": "8b8c7f14be35925d",
    "column.aspect": "982e2baa2c1c7762",
    "column.ontology_fits": "f45eb7fe2fbefdb5",
    "term.fits": "0ccc8d141ffd30ff",
    "avu.value_kind": "18a3a5159c3a8492",
}

# mesa-clm views added on top of the mesa-anyjev builders (never recorded by the generator).
CLM_ONLY = {"state_sha256", "target_state", "target_state_from"}


def _ordered_sha256(state: dict[str, Any]) -> str:
    payload = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@cache
def _card(name: str) -> m_cards.DatasetCard:
    return m_cards.load_card(CARDS_DIR / PARITY["cards"][name]["file"])


def _resolve(value: Any, card: m_cards.DatasetCard | None) -> Any:
    """The generator's kwargs encoding, resolved against mesa-clm objects."""
    if isinstance(value, dict) and len(value) == 1:
        ((tag, ref),) = value.items()
        if tag == "$card":
            return _card(ref)
        if tag == "$column":
            assert card is not None
            return card.column(ref)
        if tag == "$site":
            assert card is not None
            return next(s for s in card.sites if s.code == ref)
        if tag == "$synthetic":
            return copy.deepcopy(PARITY["synthetic"][ref])
    return value


def _expected_state(record: dict[str, Any]) -> dict[str, Any]:
    state = dict(record["state"])
    if isinstance(state.get("card"), dict) and "$card_header" in state["card"]:
        state["card"] = PARITY["cards"][state["card"]["$card_header"]]["card_header"]
    return state


def _rebuild(record: dict[str, Any]) -> dict[str, Any]:
    card = _card(record["card"]) if record["card"] else None
    kwargs = {k: _resolve(v, card) for k, v in record["kwargs"].items()}
    builder = getattr(m_states, record["builder"])
    result: dict[str, Any] = builder(**kwargs)
    return result


# -- provenance and coverage ------------------------------------------------------------------


def test_fixture_provenance_and_lock_copy() -> None:
    assert PARITY["format"] == "mesa-clm/anyjev-parity/1"
    commit = PARITY["anyjev"]["commit"]
    assert len(commit) == 40 and int(commit, 16) >= 0
    assert PARITY["anyjev"]["dirty"] is False, "fixture recorded from a dirty mesa-anyjev tree"
    lock_bytes = LOCK_PATH.read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == PARITY["questions_lock"]["sha256"]
    lock = json.loads(lock_bytes)
    assert lock["lock_sha"] == PARITY["questions_lock"]["lock_sha"]
    keys = {q: v["key"] for q, v in lock["questions"].items()}
    assert keys == PARITY["questions_lock"]["keys"]
    for question, key in KNOWN_TASK_KEYS.items():
        assert keys[question] == key, question


def test_every_anyjev_builder_is_recorded_and_ported() -> None:
    ported = {
        name
        for name, fn in vars(m_states).items()
        if inspect.isfunction(fn)
        and fn.__module__ == m_states.__name__
        and not name.startswith("_")
    }
    assert ported - CLM_ONLY == set(PARITY["builders"])
    assert set(PARITY["counts"]) == set(PARITY["builders"])
    assert all(n > 0 for n in PARITY["counts"].values())
    assert len(PARITY["records"]) == sum(PARITY["counts"].values())
    assert len(PARITY["cards"]) == 7
    # Every builder keeps its mesa-anyjev signature, so recorded kwargs resolve by name.
    for record in PARITY["records"]:
        params = inspect.signature(getattr(m_states, record["builder"])).parameters
        assert set(record["kwargs"]) <= set(params), record["id"]


def test_constants_match() -> None:
    c = PARITY["constants"]
    assert c["MAX_PROFILE"] == m_states.MAX_PROFILE
    assert c["MAX_DESCRIPTION"] == m_states.MAX_DESCRIPTION
    assert c["MAX_SIBLINGS"] == m_states.MAX_SIBLINGS
    assert list(m_cards.IDENTIFIER_SUFFIXES) == c["IDENTIFIER_SUFFIXES"]


# -- cards ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARITY["cards"]))
def test_card_parses_identically(name: str) -> None:
    entry = PARITY["cards"][name]
    text = (CARDS_DIR / entry["file"]).read_text(encoding="utf-8")
    card = m_cards.parse_card(text)
    assert card.name == name == entry["name"]
    assert card.sha256 == entry["sha256"] == m_cards.card_sha256(text)
    assert card.model_dump(mode="json") == entry["parsed"]
    assert m_states.card_header(card) == entry["card_header"]
    assert [c.name for c in card.columns if m_cards.is_identifier(c)] == entry["identifiers"]


@pytest.mark.parametrize("key", sorted(PARITY["parse_cases"]))
def test_grammar_edge_cases(key: str) -> None:
    case = PARITY["parse_cases"][key]
    if "error" in case:
        with pytest.raises(ValueError, match="Dataset") as info:
            m_cards.parse_card(case["text"])
        assert str(info.value) == case["error"]["message"]
    else:
        assert m_cards.parse_card(case["text"]).model_dump(mode="json") == case["parsed"]


def test_is_identifier_cases() -> None:
    for case in PARITY["is_identifier_cases"]:
        col = m_cards.ColumnInfo(**case["column"])
        assert m_cards.is_identifier(col) is case["result"], case["column"]["name"]


# -- registry ---------------------------------------------------------------------------------


def test_registry_vocabularies_match() -> None:
    r = PARITY["registry"]
    assert list(m_registry.ASPECTS) == r["ASPECTS"]
    assert list(m_registry.ASPECT_OPTIONS) == r["ASPECT_OPTIONS"]
    assert list(m_registry.ONTOLOGY_OPTIONS) == r["ONTOLOGY_OPTIONS"]
    assert list(m_registry.VALUE_KINDS) == r["VALUE_KINDS"]
    assert [
        {
            "id": e.id,
            "curie_prefix": e.curie_prefix,
            "option_text": e.option_text,
            "aspects": sorted(e.aspects),
        }
        for e in m_registry.ONTOLOGY_REGISTRY
    ] == r["ONTOLOGY_REGISTRY"]
    assert {k: list(v) for k, v in m_registry.DATACITE_VOCABULARIES.items()} == r[
        "DATACITE_VOCABULARIES"
    ]


def test_registry_helpers_match() -> None:
    r = PARITY["registry"]
    for aspect, ids in r["allowed_for_aspect"].items():
        assert sorted(m_registry.allowed_for_aspect(aspect)) == ids, aspect
    for aspect, mask in r["mask_for_aspect"].items():
        assert m_registry.mask_for_aspect(aspect).tolist() == mask, aspect
    in_play = frozenset(r["mask_for_aspect_in_play"]["in_play"])
    for aspect, mask in r["mask_for_aspect_in_play"]["masks"].items():
        assert m_registry.mask_for_aspect(aspect, in_play).tolist() == mask, aspect
    for curie, prefix in r["prefix_of"].items():
        assert m_registry.prefix_of(curie) == prefix, curie
    for ontology_id, resolved in r["entry"].items():
        assert m_registry.entry(ontology_id).id == resolved


# -- states -----------------------------------------------------------------------------------


@pytest.mark.parametrize("builder", PARITY["builders"])
def test_states_rebuild_identically(builder: str) -> None:
    records = [r for r in PARITY["records"] if r["builder"] == builder]
    assert records, builder
    failures: list[str] = []
    for record in records:
        state = _rebuild(record)
        expected = _expected_state(record)
        if state != expected:
            failures.append(f"{record['id']}: dict differs")
        elif m_states.state_sha256(state) != record["state_sha256"]:
            failures.append(f"{record['id']}: state_sha256 differs")
        elif _ordered_sha256(state) != record["ordered_sha256"]:
            failures.append(f"{record['id']}: key order differs: {list(state)}")
    assert not failures, f"{len(failures)}/{len(records)} {builder} mismatches: {failures[:5]}"


def test_ordered_sha_is_order_sensitive() -> None:
    # The builders' key order is not alphabetical, so the two hashes must differ for at least
    # one record; otherwise the ordered check above would prove nothing.
    assert any(r["ordered_sha256"] != r["state_sha256"] for r in PARITY["records"])
    cand = next(r for r in PARITY["records"] if r["builder"] == "candidate_state")
    assert cand["ordered_sha256"] != cand["state_sha256"]


def test_state_sha256_formula() -> None:
    # Independent restatement of the mesa-anyjev formula against a recorded value.
    record = next(r for r in PARITY["records"] if r["builder"] == "chooser_state")
    payload = json.dumps(record["state"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == record["state_sha256"]
    assert m_states.state_sha256(record["state"]) == record["state_sha256"]


def test_target_state_derives_from_every_recorded_candidate_state() -> None:
    """``target_sha256`` (D1) for imported mesa-anyjev labels comes from ``target_state_from``
    over the stored state and must equal the mesa-clm ``target_state`` of the same target."""
    seen: set[str] = set()
    for record in (r for r in PARITY["records"] if r["builder"] == "candidate_state"):
        card = _card(record["card"])
        kw = record["kwargs"]
        target = _resolve(kw["target"], card)
        direct = m_states.target_state(
            card,
            kw["scope"],
            kw["aspect"],
            column=target if isinstance(target, m_cards.ColumnInfo) else None,
            site=target if isinstance(target, m_cards.SiteInfo) else None,
        )
        derived = m_states.target_state_from(_expected_state(record))
        assert derived == direct
        assert list(derived) == list(direct)
        if target is None:
            last = "aspect"
        elif isinstance(target, m_cards.ColumnInfo):
            last = "column"
        else:
            last = "site"
        assert list(direct)[-1] == last, record["id"]
        seen.add(m_states.state_sha256(direct))
    # Several candidate groups (n_candidates) collapse onto far fewer targets.
    assert 0 < len(seen) < PARITY["counts"]["candidate_state"]
