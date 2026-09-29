"""Label identity: what a label is *about*, independent of how the question was framed
(DESIGN D1).

mesa-anyjev keyed a label on ``(question_key, state_sha256, label_source)``. That identity
breaks in mesa-clm for two reasons the plan names: ``candidate_state`` carries
``n_candidates``, so the same (target, candidate) pair hashes differently per group size, and
an anchor pick ("none of these") has no candidate state at all. mesa-clm therefore identifies a
label by ``(task_key, target_sha256, option_key, label_source)``:

* ``task_key`` — the frozen task (:mod:`mesa_clm.tasks`), equal to the mesa-anyjev lock key;
* ``target_sha256`` — sha256 of :func:`target_key`, a *compact* identity of the thing the task
  is asked about: the dataset (card name), the scope, the target within the card (column name,
  site code or the dataset itself), the aspect when the state carries one and, for an AVU-scope
  task, the term the AVU is built from. It never contains the rendered card header, so editing a
  card's row count, months or profiles does not orphan its labels, and it never contains
  ``n_candidates`` or the candidate's OLS record;
* ``option_key`` — ``''`` for closed-choice tasks (the option is the label itself), the candidate
  CURIE for ``term.fits``, the registry ontology id for ``column.ontology_fits`` and
  ``registry.ANCHOR_KEY`` for an explicit "none of these" (an anchor-positive row, D21).

``state_sha256`` stays on every row for parity with mesa-anyjev and for rendering the exact
state a label was recorded on; it is not part of the identity.

The functions read the anyjev-shaped state dicts built by :mod:`mesa_clm.states`
(``column_state``, ``ontology_state``, ``candidate_state``, ``value_kind_state`` and the
additive ``target_state`` view), so a mesa-anyjev ``state_json`` row maps to the same identity
as a fresh mesa-clm decision on the same target.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import RANK_FIT_TASKS, TASKS, Task


class IdentityError(ValueError):
    """A state that does not carry what the task's identity needs."""


class LabelIdentity(NamedTuple):
    """The three identity parts a label row stores (``label_source`` is the fourth key)."""

    task_key: str
    target_sha256: str
    option_key: str


def _active_task(task_id: str) -> Task:
    try:
        t = TASKS[task_id]
    except KeyError as exc:
        raise IdentityError(f"unknown task {task_id!r}") from exc
    if not t.active:
        raise IdentityError(f"{task_id}: no target identity is defined for an inactive task")
    return t


def _dataset(state: dict[str, Any]) -> str:
    card = state.get("card")
    if not isinstance(card, dict) or not card.get("dataset"):
        raise IdentityError("state has no card header with a dataset name")
    return str(card["dataset"])


def target_key(task_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """The compact target identity of ``state`` under ``task_id``, as an ordered dict.

    Keys, always in this order: ``dataset`` (the card name), ``scope`` (the state's ``scope``
    when it has one, else the task's scope), ``target`` (the column name for a column target,
    the site code for a site target, the dataset name for a dataset target), then ``aspect``
    only when the state carries one and, for the AVU-scope ``avu.value_kind``, ``term`` (the
    CURIE of the term the AVU is built from). Raises :class:`IdentityError` for an inactive task
    or a state missing the parts its task needs.
    """
    t = _active_task(task_id)
    dataset = _dataset(state)
    scope = str(state.get("scope") or t.scope)
    if "column" in state and "site" in state:
        raise IdentityError("state has both a column and a site target")
    out: dict[str, Any] = {"dataset": dataset, "scope": scope}
    if scope in ("column", "avu"):
        column = state.get("column")
        if not isinstance(column, dict) or not column.get("name"):
            raise IdentityError(f"{task_id}: a {scope}-scope state needs a column with a name")
        out["target"] = str(column["name"])
    elif scope == "site":
        site = state.get("site")
        if not isinstance(site, dict) or not site.get("code"):
            raise IdentityError(f"{task_id}: a site-scope state needs a site with a code")
        out["target"] = str(site["code"])
    elif scope == "dataset":
        out["target"] = dataset
    else:
        raise IdentityError(f"{task_id}: unknown scope {scope!r}")
    if state.get("aspect") is not None:
        out["aspect"] = str(state["aspect"])
    if scope == "avu":
        term = state.get("term")
        if not isinstance(term, dict) or not term.get("curie"):
            raise IdentityError(f"{task_id}: an avu-scope state needs a term with a curie")
        out["term"] = str(term["curie"])
    return out


def target_sha256(task_id: str, state: dict[str, Any]) -> str:
    """Full 64-hex sha256 of the canonical JSON of :func:`target_key` (sorted keys, compact
    separators, UTF-8, non-ASCII kept), like ``states.state_sha256``."""
    payload = json.dumps(
        target_key(task_id, state), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def option_key(task_id: str, state: dict[str, Any], label: str | None = None) -> str:
    """The option a label on ``state`` refers to.

    Closed-choice tasks return ``''``: the option is the label. For the rank_fit tasks the
    option comes from the state (``candidate.curie`` for ``term.fits``, ``ontology.id`` for
    ``column.ontology_fits``); when the state carries no candidate (an anchor row has none),
    ``label`` names the option explicitly (a CURIE, an ontology id or ``ANCHOR_KEY``) and is
    required. ``label`` is ignored when the state carries the candidate.
    """
    t = _active_task(task_id)
    if t.id not in RANK_FIT_TASKS:
        return ""
    if t.id == "term.fits":
        cand = state.get("candidate")
        if isinstance(cand, dict) and cand.get("curie"):
            return str(cand["curie"])
    elif t.id == "column.ontology_fits":
        ont = state.get("ontology")
        if isinstance(ont, dict) and ont.get("id"):
            return str(ont["id"]).lower()
    if label:
        return label if label == ANCHOR_KEY else str(label)
    raise IdentityError(
        f"{task_id}: the state names no candidate; pass the option (a CURIE, an ontology id "
        f"or {ANCHOR_KEY!r}) as label"
    )


def identity(task_id: str, state: dict[str, Any], label: str | None = None) -> LabelIdentity:
    """``(task_key, target_sha256, option_key)`` of a label on ``state`` in one call."""
    return LabelIdentity(
        task_key=_active_task(task_id).key,
        target_sha256=target_sha256(task_id, state),
        option_key=option_key(task_id, state, label),
    )
