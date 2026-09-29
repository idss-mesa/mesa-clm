"""Canonical AVU construction and value rules (mesa-anyjev DESIGN D12; amendment B1).

The AVU shape is mesa-mcp's contract: ``attribute = '<ontology>.<snake_case_label>'``,
``unit = CURIE``. ``mesa_mcp.ols.tools.avu_from_term.handle_avu_from_term`` is ``async``, and
with curie+iri+label supplied it never touches OLS, so ``build_avu`` calls the two sync pieces
it wraps; ``tests/unit/test_avu_parity.py`` asserts byte-for-byte parity with the tool.

Ported from mesa-anyjev ``avu.py`` (``6159281``; DESIGN U1). Change: defect (c). mesa-anyjev
``service.py:405`` compared a value kind against the literal ``"the top profile value"``, which
is not in ``registry.VALUE_KINDS``, so a curator pick of that kind silently got the term label.
The four kinds are named constants here (``VALUE_KIND_TOP`` is the most-frequent-value kind)
and every comparison in this package goes through them.
"""

from __future__ import annotations

from typing import Final

from mesa_mcp.ols.client import _label_to_snake
from mesa_mcp.ols.transform import ontology_annotations_to_avus

from mesa_clm.ols import Candidate
from mesa_clm.registry import VALUE_KINDS

Avu = dict[str, str]

# The exact ``registry.VALUE_KINDS`` strings (frozen: they are ``avu.value_kind``'s options and
# therefore part of its task_key). ``VALUE_KIND_TOP`` is "the most frequent data value", the
# kind mesa-anyjev's service compared against the wrong literal (defect (c)).
VALUE_KIND_LABEL: Final[str] = "the term label"
VALUE_KIND_SITE: Final[str] = "the site code"
VALUE_KIND_COLUMN: Final[str] = "the column name"
VALUE_KIND_TOP: Final[str] = "the most frequent data value"

if {VALUE_KIND_LABEL, VALUE_KIND_SITE, VALUE_KIND_COLUMN, VALUE_KIND_TOP} != set(VALUE_KINDS):
    raise RuntimeError("mesa_clm.avu value-kind constants drifted from registry.VALUE_KINDS")


def build_avu(candidate: Candidate, value: str) -> Avu:
    key = _label_to_snake(candidate.label)
    if not key:
        raise ValueError(f"label {candidate.label!r} does not snake-case to a key")
    avus = ontology_annotations_to_avus(
        candidate.ontology_id, [{"key": key, "value": value, "curie": candidate.curie}]
    )
    if not avus:
        raise ValueError(f"could not build an AVU for {candidate.curie} with value {value!r}")
    avu = avus[0]
    return {
        "attribute": str(avu["attribute"]),
        "value": str(avu["value"]),
        "unit": str(avu["unit"]),
    }


def pre_rule_value_kind(aspect: str, scope: str) -> str | None:
    """Deterministic pre-rules before the value-kind question is asked (D12)."""
    if scope == "site" or (aspect in ("environment", "location") and scope != "column"):
        return VALUE_KIND_SITE
    if aspect == "unit":
        return VALUE_KIND_LABEL
    if scope == "dataset":
        return VALUE_KIND_LABEL
    return None


def value_for(
    kind: str,
    *,
    term_label: str,
    column_name: str | None,
    site_code: str | None,
    top_value: str | None,
) -> str:
    if kind not in VALUE_KINDS:
        raise ValueError(f"unknown value kind {kind!r}")
    if kind == VALUE_KIND_SITE and site_code:
        return site_code
    if kind == VALUE_KIND_COLUMN and column_name:
        return column_name
    if kind == VALUE_KIND_TOP and top_value:
        return top_value
    return term_label


def top_profile_value(profile: str) -> str | None:
    """The most frequent value from a card profile like ``'2 distinct; top: SRER (9416), HARV (6068)'``."""
    marker = "top:"
    if marker not in profile:
        return None
    first = profile.split(marker, 1)[1].split(",", 1)[0].strip()
    return first.rsplit(" (", 1)[0].strip() or None


def triple(avu: Avu) -> tuple[str, str, str]:
    return (avu["attribute"], avu["value"], avu["unit"])
