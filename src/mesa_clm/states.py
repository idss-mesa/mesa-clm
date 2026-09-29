"""JSON-serialisable state builders (plain dicts of str/int/list only: no numpy, no datetime,
so the sidecar can store and hash them).

The builders, their key order and ``state_sha256`` are byte-identical to mesa-anyjev
``states.py`` (``6159281``; DESIGN D1, D23): labels and states recorded by mesa-anyjev stay
joinable, and ``tests/unit/test_anyjev_parity.py`` rebuilds every state recorded from
mesa-anyjev on the seven fixture cards and compares dicts, key order and hashes. Never edit an
existing builder; add a new view instead.

The shared card header comes first in every state so the rendered prefix is identical across
the questions asked of one card (vLLM prefix caching keys on it). CLM pools the encoder's
*last* token, which weights the tail of the text, so the context views mesa-clm renders are
additive builders that **end with the target** (``target_state``; DESIGN D23).
``state_sha256`` is identity only (sorted keys); rendering uses the builder-ordered dict.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mesa_clm.cards import ColumnInfo, DatasetCard, SiteInfo

MAX_PROFILE = 300
MAX_DESCRIPTION = 300
MAX_SIBLINGS = 25


def state_sha256(state: dict[str, Any]) -> str:
    """sha256 over the canonical JSON (sorted keys, compact separators, UTF-8, non-ASCII kept)."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def card_header(card: DatasetCard) -> dict[str, Any]:
    """The shared prefix of every card state: dataset, product, sites, rows, months and the
    column *names* only (never the raw card)."""
    return {
        "dataset": card.name,
        "product_title": card.product_title,
        "product_description": card.product_description,
        "sites": [
            {"code": s.code, "name": s.name, "domain": s.domain, "habitat": s.habitat}
            for s in card.sites
        ],
        "rows": card.rows,
        "months": f"{card.months_from} to {card.months_to}",
        "columns": [c.name for c in card.columns],
    }


def _column(col: ColumnInfo) -> dict[str, Any]:
    return {
        "name": col.name,
        "description": col.description,
        "dtype": col.dtype,
        "unit": col.unit,
        "profile": col.profile[:MAX_PROFILE],
    }


def column_state(card: DatasetCard, col: ColumnInfo) -> dict[str, Any]:
    """``column.annotate`` / ``column.aspect``: the card header and one column."""
    return {"card": card_header(card), "column": _column(col)}


def ontology_state(
    card: DatasetCard,
    col: ColumnInfo,
    aspect: str,
    ontology_id: str,
    option_text: str,
    aspects: list[str],
) -> dict[str, Any]:
    """``column.ontology_fits`` (mesa-anyjev noul form): one registry entry for one column."""
    return {
        "card": card_header(card),
        "column": _column(col),
        "aspect": aspect,
        "ontology": {"id": ontology_id, "description": option_text, "aspects": sorted(aspects)},
    }


def candidate_state(
    card: DatasetCard,
    scope: str,
    target: ColumnInfo | SiteInfo | None,
    aspect: str,
    candidate: dict[str, Any],
    n_candidates: int,
) -> dict[str, Any]:
    """``term.fits`` (mesa-anyjev noul form): one OLS candidate for one target. Kept for
    label import and parity; ``n_candidates`` makes one (target, candidate) pair hash
    differently per group size, which is why labels key on ``target_sha256`` (DESIGN D1)."""
    state: dict[str, Any] = {"card": card_header(card), "scope": scope}
    if isinstance(target, ColumnInfo):
        state["column"] = _column(target)
    elif isinstance(target, SiteInfo):
        state["site"] = {
            "code": target.code,
            "name": target.name,
            "domain": target.domain,
            "habitat": target.habitat,
        }
    state["aspect"] = aspect
    state["candidate"] = {
        "label": str(candidate.get("label", "")),
        "curie": str(candidate.get("curie", "")),
        "ontology_id": str(candidate.get("ontology_id", "")),
        "description": str(candidate.get("description", ""))[:MAX_DESCRIPTION],
        "synonyms": [str(s) for s in list(candidate.get("synonyms") or [])[:5]],
        "has_children": bool(candidate.get("has_children", False)),
    }
    state["n_candidates"] = int(n_candidates)
    return state


def chooser_state(
    ontology_id: str, value: str, candidate: dict[str, Any], n_candidates: int
) -> dict[str, Any]:
    """The state for ``term.fits.chooser``: only what mesa-mcp's picker carries."""
    return {
        "ontology_id": ontology_id,
        "value": value,
        "candidate": {
            "label": str(candidate.get("label", "")),
            "curie": str(candidate.get("curie", "")),
            "description": str(candidate.get("description", ""))[:MAX_DESCRIPTION],
        },
        "n_candidates": int(n_candidates),
    }


def value_kind_state(
    card: DatasetCard, col: ColumnInfo, term: dict[str, Any], aspect: str
) -> dict[str, Any]:
    """``avu.value_kind``: which value an AVU built from ``term`` for ``col`` should carry."""
    return {
        "card": card_header(card),
        "column": _column(col),
        "aspect": aspect,
        "term": {"label": str(term.get("label", "")), "curie": str(term.get("curie", ""))},
    }


def avu_state(
    card: DatasetCard,
    avu: dict[str, str],
    term: dict[str, Any],
    scope: str,
    target_name: str | None,
    siblings: list[dict[str, str]],
) -> dict[str, Any]:
    """``avu.keep`` (a rule in mesa-clm, DESIGN D25; kept for label import and parity)."""
    return {
        "card": card_header(card),
        "avu": {"attribute": avu["attribute"], "value": avu["value"], "unit": avu["unit"]},
        "term": {
            "label": str(term.get("label", "")),
            "description": str(term.get("description", ""))[:MAX_DESCRIPTION],
        },
        "scope": scope,
        "target": target_name or "",
        "siblings": [
            {"attribute": s["attribute"], "value": s["value"], "unit": s["unit"]}
            for s in siblings[:MAX_SIBLINGS]
        ],
    }


def dataset_ontology_state(card: DatasetCard, ontology_option: str) -> dict[str, Any]:
    """``dataset.ontology_applies``: the card header and one registry entry (planner audit)."""
    return {"card": card_header(card), "ontology": ontology_option}


def datacite_state(
    card: DatasetCard | None, vocabulary: str, text: str, value: str | None = None
) -> dict[str, Any]:
    """DataCite questions: the (optional) card header, the vocabulary name, the text being
    classified (a description, a contributor line, a related identifier, a date) and, for the
    yes/no twins, the candidate value."""
    state: dict[str, Any] = {"vocabulary": vocabulary, "text": str(text)[: MAX_DESCRIPTION * 4]}
    if card is not None:
        state["card"] = card_header(card)
    if value is not None:
        state["value"] = value
    return state


# -- mesa-clm additive views (DESIGN D23) -------------------------------------------------------


def _site(site: SiteInfo) -> dict[str, Any]:
    # The same four keys, in the same order, as the site entries of ``card_header``.
    return {"code": site.code, "name": site.name, "domain": site.domain, "habitat": site.habitat}


def target_state(
    card: DatasetCard,
    scope: str,
    aspect: str,
    *,
    column: ColumnInfo | None = None,
    site: SiteInfo | None = None,
) -> dict[str, Any]:
    """The target-last context of a (card, target) pair: ``{card, scope, aspect, column|site}``.

    Unlike ``candidate_state`` it carries no candidate and no group size: candidates are the
    options of the CLM Choice, not part of the state, so every candidate group for the same
    target shares one context (and one cached encoder vector). The target comes last because
    CLM pools the last token. The ``column`` value is the same dict as ``column_state``'s and
    the ``site`` value the same dict as a ``card_header`` site entry, so ``target_sha256`` can
    be derived from mesa-anyjev states. A dataset-scope target (neither ``column`` nor
    ``site``) is the card itself and the view ends with the aspect.
    """
    if column is not None and site is not None:
        raise ValueError("target_state takes a column or a site, not both")
    state: dict[str, Any] = {"card": card_header(card), "scope": scope, "aspect": aspect}
    if column is not None:
        state["column"] = _column(column)
    elif site is not None:
        state["site"] = _site(site)
    return state


def target_state_from(state: dict[str, Any]) -> dict[str, Any]:
    """Project a stored ``candidate_state``-shaped dict (``card``, ``scope``, ``aspect`` and
    optionally ``column`` or ``site``) onto the ``target_state`` it was asked about, key order
    included, so mesa-anyjev ``state_json`` rows map to a ``target_sha256`` without the card.
    Raises ``KeyError`` when ``card``, ``scope`` or ``aspect`` is missing."""
    out: dict[str, Any] = {
        "card": state["card"],
        "scope": state["scope"],
        "aspect": state["aspect"],
    }
    if "column" in state and "site" in state:
        raise ValueError("state has both a column and a site target")
    if "column" in state:
        out["column"] = state["column"]
    elif "site" in state:
        out["site"] = state["site"]
    return out
