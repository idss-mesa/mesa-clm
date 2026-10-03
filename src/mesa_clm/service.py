"""``DecisionService``: one provider, one planner, one OLS layer, one policy and one store behind
a lock, shared by the CLI and the ``mesa_clm_*`` MCP tools (DESIGN D10, D11, D21, D22, D26).

Ported from mesa-anyjev ``service.py`` (``6159281``; DESIGN U1). Changes against the source:

* **Owners (D21).** Every run stores its ``owner``; :meth:`DecisionService.explain`,
  :meth:`DecisionService.record_human_pick` and :meth:`DecisionService.check_owner` (the apply
  and MRTR-resume paths call it) refuse another identity with :class:`NotOwner`. ``owner`` is
  the authenticated identity of the request, never the free-text ``actor``.
* **Where a pick comes from (D21, defect (g)).** Curator labels come only from an MRTR
  elicitation answer or the interactive CLI (``via='elicitation'|'cli'``): the pick at weight
  1.0 (``curator``), the other offered candidates as implicit negatives at 0.7
  (``curator_implicit``), an explicit "none of these" (a pick of nothing, of the anchor, or a
  ``reject``) as an anchor-positive row plus a negative per candidate at 1.0. A plain tool call
  (``via='tool'``) writes the same rows as ``agent_pick`` at weight 0, ``fold_eligible=false``,
  and accepts links ``accepted_by='agent'``. A ``decline`` writes no label and changes no link
  or group; only its override row is kept, so the audit trail shows the question was seen.
* **Label identity (D1).** Rows are keyed ``(task_key, target_sha256, option_key,
  label_source)`` from the group decision's *own* task and state (mesa-anyjev hard-coded
  ``question_id='term.fits'``): a pick on a ``column.ontology_fits`` group labels that task.
  Curator rows on a bench card carry ``bench_card=true`` (D30). Bench-card membership is fixed
  and fails closed: by default the seven neon-avu-eval cards
  (:data:`mesa_clm.learn.labels.BENCH_CARDS`) plus any card with silver consensus labels in the
  store, and a run whose card name is unknown; it never depends on the sidecar already holding
  the silver labels.
* **Links.** The chosen candidate's link becomes ``accepted`` (``accepted_by`` ``human`` for
  elicitation/cli, ``agent`` for a tool); a candidate without a link (a non-winner, or any
  candidate of an anchor-won group) gets a new accepted link built like the group's winner
  (value kind, target), with defect (c) fixed through ``avu.VALUE_KIND_TOP``. The sidecar has no
  ``rejected`` write status: a rejected group keeps its links ``proposed`` (an accepted one goes
  back to ``proposed``) and the group's outcome ``rejected`` plus the override row say that it
  was rejected. After a pick the group is ``human`` and its other links go back to (or stay)
  ``proposed``, so a group has at most one accepted link: **apply must write, from a group whose
  outcome is ``human`` or ``rejected``, only its accepted links.**
* **One answer per group (DESIGN A2).** A curator's answer (``via`` ``elicitation`` or ``cli``)
  settles a group: a different second answer is refused with :class:`AlreadyAnswered` (M1 has no
  amend flow, and the D1 ``INSERT OR IGNORE`` would silently keep the first answer's rows), the
  same answer again is idempotent (no new label, the same link). An agent's answer
  (``via='tool'``) does not settle a group for a curator: the group stays pending, a curator's
  answer supersedes it (the agent's accepted link goes back to ``proposed``; its ``agent_pick``
  rows stay, weight 0), and a different second *agent* answer is refused.
* **Reads from the sidecar only (D26).** :meth:`candidates_for_group` rebuilds the offered set
  (every in-play option of the group's deciding record, the anchor included, best ``p_fit``
  first) from ``decision_options``; a resumed elicitation never trusts client state.

The decider lock serialises annotate calls (the provider's token cache and counters are not
thread-safe); a caller that cannot take it within ``policy.max_wait_s`` gets
:class:`DeciderBusy`, which the tools turn into ``decider_busy``. Serving never fits (D15):
labels wait for ``learn fit``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal
from uuid import UUID

from mesa_clm.avu import VALUE_KIND_LABEL, VALUE_KIND_TOP, build_avu, pre_rule_value_kind, value_for
from mesa_clm.cards import DatasetCard
from mesa_clm.config import Config, PlannerKind, expand_path
from mesa_clm.identity import identity
from mesa_clm.learn.labels import (
    CONSENSUS_SOURCES,
    INGESTED_TASKS,
    NOT_FOLD_ELIGIBLE,
    WEIGHTS,
    is_bench_card,
    product_code_of,
)
from mesa_clm.ols import Candidate, OLSLayer, RecordingOLS
from mesa_clm.pipeline import AnnotationRun, Annotator
from mesa_clm.planner import Planner, make_planner
from mesa_clm.policy import Policy
from mesa_clm.provenance.models import AvuLinkRow, HumanOverrideRow, LabelRow, OverrideAction
from mesa_clm.provenance.store import ProvenanceStore, open_store
from mesa_clm.providers.base import DecisionProvider
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.tasks import RANK_FIT_TASKS, TASKS
from mesa_clm.vocab import VIAS, AcceptedBy, LabelSource, Via

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIONS",
    "CONSENSUS_SOURCES",
    "HUMAN_VIAS",
    "AlreadyAnswered",
    "Collaborators",
    "DeciderBusy",
    "DecisionService",
    "NotOwner",
    "PickAction",
    "build_collaborators",
]

PickAction = Literal["pick", "reject", "decline"]
ACTIONS: Final[tuple[str, ...]] = ("pick", "reject", "decline")
# Settled group outcomes; such a group is no longer pending (unless only an agent answered it).
# The pipeline sets "rejected" itself (the Q8 keep rule's drops, a D24 refinement that did not
# replace its parent), so the outcome never says *who* settled a group: only the override rows do
# (_RunView.answered_by).
_RESOLVED: Final[frozenset[str]] = frozenset({"human", "rejected"})
# Where a curator's answer comes from (D21, DESIGN A2); ``tool`` is an agent's.
HUMAN_VIAS: Final[frozenset[str]] = frozenset({"elicitation", "cli"})
# Group outcomes that wait for a reviewer (plus any anchor-won group).
_WAITING: Final[frozenset[str]] = frozenset({"proposed", "escalated"})
# Outcomes of a specificity group whose child replaced its parent (D24): the parent's group is
# then asked through the refinement, which offers the parent too.
_REFINED: Final[frozenset[str]] = frozenset({"auto", "proposed", "escalated", "human"})
_YES, _NO = 0, 1  # label_index of the noul labels ("Yes", "No") of the rank_fit tasks


class DeciderBusy(RuntimeError):
    """The decider lock could not be taken within ``policy.max_wait_s``."""


class NotOwner(PermissionError):
    """The run belongs to another owner (D21); the message never names that owner."""


class AlreadyAnswered(ValueError):
    """The group already carries an answer this one may not replace (module docstring)."""


# -- collaborators ----------------------------------------------------------------------------


@dataclass
class Collaborators:
    """What a :class:`DecisionService` needs besides the configuration."""

    provider: DecisionProvider
    planner: Planner
    ols: OLSLayer
    policy: Policy
    store: ProvenanceStore


def build_collaborators(
    cfg: Config,
    provider: DecisionProvider,
    *,
    planner_kind: PlannerKind | None = None,
    store: ProvenanceStore | None = None,
    ols: OLSLayer | None = None,
) -> Collaborators:
    """The planner (``make_planner``), the OLS layer (``RecordingOLS`` per ``cfg.ols.fixtures``;
    ``replay`` never reaches EBI, every other mode wraps mesa-mcp's ``OLSClient``), the policy
    (``Policy.from_config``) and the store (``open_store(cfg.provenance.dsn)``) around the given
    ``provider``. Building the real CLM provider (clm-serve + encoder + the live fingerprint from
    ``serving.lock.json``) is the caller's job; tests and ``annotate --provider fake`` pass a
    :class:`~mesa_clm.providers.tiered.FakeProvider`."""
    planner = make_planner(cfg, planner_kind)
    if ols is None:
        inner: Any = None
        if cfg.ols.fixtures != "replay":
            from mesa_mcp.ols.client import OLSClient

            inner = OLSClient(cfg.ols.base_url)
        client: Any = (
            RecordingOLS(inner, expand_path(cfg.ols.fixtures_dir), cfg.ols.fixtures)
            if cfg.ols.fixtures != "off"
            else inner
        )
        ols = OLSLayer(client, max_candidates=cfg.policy.max_candidates)
    policy = Policy.from_config(cfg.policy)
    if store is None:
        store = open_store(cfg.provenance.dsn)
    return Collaborators(provider, planner, ols, policy, store)


# -- the run view -----------------------------------------------------------------------------


class _RunView:
    """One run's sidecar rows, read once: the run, its decisions by id, their options, its
    groups and links."""

    def __init__(self, store: ProvenanceStore, run_id: UUID) -> None:
        run = store.run(run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        self.run = run
        self.decisions = {str(d["decision_id"]): d for d in store.decisions(run_id)}
        self.options: dict[str, list[dict[str, Any]]] = {}
        for o in store.options(run_id):
            self.options.setdefault(str(o["decision_id"]), []).append(o)
        self.groups = store.groups(run_id)
        self.links = store.links(run_id)
        self.overrides = store.overrides(run_id)

    def answers(self, group_id: UUID | str) -> list[dict[str, Any]]:
        """The group's override rows that answered it (every action but ``decline``), oldest
        first."""
        return [
            o
            for o in self.overrides
            if str(o.get("group_id")) == str(group_id) and o.get("action") != "decline"
        ]

    def answered_by(self, group_id: UUID | str) -> Literal["human", "agent"] | None:
        """``human`` when a curator answered the group (an override row ``via`` elicitation or
        cli), ``agent`` when only plain tool calls did, else ``None``. The override rows decide
        (run exports and imports carry them): a ``rejected`` outcome without one is the
        pipeline's own (the Q8 keep rule, a refinement that did not replace its parent), which
        nobody answered, so ``feedback`` can still answer it. Only a ``human`` outcome without a
        row counts as a curator's (only :meth:`DecisionService.record_human_pick` sets it)."""
        vias = {str(o.get("via")) for o in self.answers(group_id)}
        if vias & HUMAN_VIAS:
            return "human"
        if vias:
            return "agent"
        if self.group(group_id).get("outcome") == "human":
            return "human"
        return None

    def group(self, group_id: UUID | str) -> dict[str, Any]:
        for g in self.groups:
            if str(g["group_id"]) == str(group_id):
                return g
        raise KeyError(f"group {group_id} not found")

    def links_of(self, group_id: UUID | str) -> list[dict[str, Any]]:
        return [link for link in self.links if str(link.get("group_id")) == str(group_id)]

    def deciding(self, group: dict[str, Any]) -> dict[str, Any] | None:
        """The record that decided the group: its ``winner_decision_id`` (the CLM rank, or the
        ``ols_rank`` fallback)."""
        winner = group.get("winner_decision_id")
        return self.decisions.get(str(winner)) if winner else None

    def candidates(self, group_id: UUID | str) -> list[dict[str, Any]]:
        """The offered set of a group (module docstring), best ``p_fit`` first (by OLS rank for
        an ``ols_rank`` record), the anchor included; masked options are not offered."""
        group = self.group(group_id)
        decision = self.deciding(group)
        if decision is None:
            return []
        search = group.get("search_json") or {}
        meta = {str(c.get("curie")): c for c in search.get("candidates") or []}
        links = {
            str(link.get("term_curie")): link
            for link in self.links_of(group_id)
            if link.get("term_curie")
        }
        out: list[dict[str, Any]] = []
        for o in self.options.get(str(decision["decision_id"]), []):
            if o.get("masked"):
                continue
            key = str(o["option_key"])
            c = meta.get(key, {})
            link = links.get(key)
            out.append(
                {
                    "decision_id": str(decision["decision_id"]),
                    "option_key": key,
                    "option_index": int(o["option_index"]),
                    "option_text": o.get("option_text"),
                    "is_anchor": key == ANCHOR_KEY,
                    "label": c.get("label") or (None if key == ANCHOR_KEY else key),
                    "iri": c.get("iri"),
                    "ontology_id": c.get("ontology_id") or group.get("ontology_id"),
                    "p_fit": o.get("p_fit"),
                    "s_c": o.get("s_c"),
                    "prob": o.get("prob"),
                    "rank": o.get("rank"),
                    "level": decision.get("level"),
                    "calibration": decision.get("calibration"),
                    "method": decision.get("method"),
                    "task_id": decision.get("task_id"),
                    "link_id": str(link["link_id"]) if link else None,
                    "write_status": link.get("write_status") if link else None,
                }
            )

        def order(c: dict[str, Any]) -> tuple[int, float, int]:
            p = c["p_fit"]
            if p is not None:
                return (0, -float(p), c["option_index"])
            rank = c["rank"]
            return (1, float(rank) if rank is not None else float("inf"), c["option_index"])

        out.sort(key=order)
        return out

    def pending(self) -> list[dict[str, Any]]:
        """Groups waiting for a reviewer: ``term.fits`` groups that are proposed, escalated or
        anchor-won, or that only an agent answered, and that neither a curator nor the pipeline
        settled (a ``rejected`` outcome without an override row is the keep rule's or a
        refinement's), minus a group a specificity rank refined (its refinement group, which
        offers the parent too, is asked instead). Each carries ``agent_answered``. ``feedback``
        may still answer any group of the run, pending or not."""
        superseded = {
            str(g["escalated_from"])
            for g in self.groups
            if g.get("escalated_from") and g["outcome"] in _REFINED
        }
        out: list[dict[str, Any]] = []
        for g in self.groups:
            if g["task_id"] != "term.fits" or str(g["group_id"]) in superseded:
                continue
            by = self.answered_by(g["group_id"])
            if by == "human" or (by is None and g["outcome"] in _RESOLVED):
                continue  # a curator's answer, or settled by the pipeline itself
            if by == "agent" or g["outcome"] in _WAITING or g.get("anchor_won"):
                out.append({**_slim_group(g), "agent_answered": by == "agent"})
        return out


def _slim_group(g: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": str(g["group_id"]),
        "task_id": g["task_id"],
        "scope": g["scope"],
        "column_name": g.get("column_name") or "",
        "site_code": g.get("site_code") or "",
        "aspect": g.get("aspect"),
        "ontology_id": g.get("ontology_id"),
        "outcome": g["outcome"],
        "anchor_won": bool(g.get("anchor_won")),
        "top_p_fit": g.get("top_p_fit"),
        "group_margin": g.get("group_margin"),
        "n_candidates": g.get("n_candidates"),
        "escalated_from": str(g["escalated_from"]) if g.get("escalated_from") else None,
    }


_SLIM_DECISION: Final[tuple[str, ...]] = (
    "seq",
    "task_id",
    "question_key",
    "framing_id",
    "scope",
    "column_name",
    "site_code",
    "method",
    "level",
    "calibration",
    "answer",
    "confidence",
    "p_fit",
    "s_c",
    "rank",
    "outcome",
    "reason",
)


def _slim_decision(d: dict[str, Any]) -> dict[str, Any]:
    out = {k: d.get(k) for k in _SLIM_DECISION}
    for key in ("decision_id", "group_id", "parent_decision_id"):
        out[key] = str(d[key]) if d.get(key) else None
    return out


def _slim_link(link: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "attribute",
        "value",
        "unit",
        "term_curie",
        "term_label",
        "ontology_id",
        "aspect",
        "column_name",
        "site_code",
        "value_kind",
        "write_status",
        "accepted_by",
        "source",
        "irods_path",
        "snapshot_id",
    )
    out = {k: link.get(k) for k in keep}
    for key in ("link_id", "run_id", "group_id", "decision_id"):
        out[key] = str(link[key]) if link.get(key) else None
    return out


def _json_run(run: dict[str, Any]) -> dict[str, Any]:
    """A run row with ids and timestamps as strings (tool output is JSON)."""
    return {
        k: (str(v) if isinstance(v, UUID) or hasattr(v, "isoformat") else v) for k, v in run.items()
    }


def _answer_key(action: str, chosen: str | None) -> tuple[str, str] | None:
    """What an override answered, comparable across ``pick``/``none``/``reject``: ``("pick",
    key)`` or ``("none", "")`` (``reject`` and an explicit none are one answer); ``None`` for a
    ``decline``."""
    if action == "decline":
        return None
    if action in ("none", "reject") or chosen in (None, ANCHOR_KEY):
        return ("none", "")
    return ("pick", str(chosen))


@dataclass
class _Answer:
    """A checked answer, ready to record (:meth:`DecisionService._prepare_answer`)."""

    run_id: UUID
    run: dict[str, Any]
    view: _RunView
    group: dict[str, Any]
    decision: dict[str, Any]
    cands: list[dict[str, Any]]
    by_key: dict[str, dict[str, Any]]
    chosen: str | None
    explicit_none: bool
    override_action: OverrideAction
    human: bool


# -- the service ------------------------------------------------------------------------------


class DecisionService:
    """The collaborators behind a lock (module docstring). Build it with
    :meth:`from_config` or pass the collaborators; ``bench_cards`` names the cards whose curator
    labels are tagged ``bench_card`` (default ``None``: the fixed
    :data:`~mesa_clm.learn.labels.BENCH_CARDS` plus the cards with silver consensus labels in
    the store, D30; a caller that passes a list owns it, which only tests do, and the bench drops
    curator rows on the fixed bench cards whatever their tag); ``ols_rank_tasks`` sends those
    rank_fit tasks to the degraded method (D28; default ``cfg.decider.ols_rank_tasks``,
    ``term.fits`` under K1, DESIGN A1; an empty collection asks CLM for every task, an audit
    run); ``claude_client`` is handed to the second-opinion provider (a fake in tests)."""

    def __init__(
        self,
        cfg: Config,
        *,
        provider: DecisionProvider,
        planner: Planner,
        ols: OLSLayer,
        policy: Policy,
        store: ProvenanceStore,
        bench_cards: Collection[str] | None = None,
        ols_rank_tasks: Collection[str] | None = None,
        claude_client: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.planner = planner
        self.ols = ols
        self.policy = policy
        self.store = store
        self.lock = threading.Lock()
        self.max_wait_s = float(cfg.policy.max_wait_s)
        self.bench_cards = frozenset(bench_cards) if bench_cards is not None else None
        self.ols_rank_tasks: frozenset[str] | None = (
            frozenset(ols_rank_tasks) if ols_rank_tasks is not None else None
        )
        self._claude_client = claude_client
        self._second_opinion: DecisionProvider | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        provider: DecisionProvider,
        *,
        planner_kind: PlannerKind | None = None,
        store: ProvenanceStore | None = None,
        ols: OLSLayer | None = None,
        **kw: Any,
    ) -> DecisionService:
        """:func:`build_collaborators` around ``provider``, then the service."""
        c = build_collaborators(cfg, provider, planner_kind=planner_kind, store=store, ols=ols)
        return cls(
            cfg,
            provider=c.provider,
            planner=c.planner,
            ols=c.ols,
            policy=c.policy,
            store=c.store,
            **kw,
        )

    # -- lock -----------------------------------------------------------------------------------

    def _acquire(self) -> None:
        if not self.lock.acquire(timeout=self.max_wait_s):
            raise DeciderBusy(f"decider busy for more than {self.max_wait_s}s")

    # -- decide phase -----------------------------------------------------------------------------

    def second_opinion_provider(self, enabled: bool | None = None) -> DecisionProvider | None:
        """The Claude second-opinion provider when ``enabled`` (default
        ``cfg.claude.second_opinion``); built lazily over the live fingerprint and cached.
        Credentials resolve through the SDK, never here."""
        on = self.cfg.claude.second_opinion if enabled is None else enabled
        if not on:
            return None
        if self._second_opinion is None:
            from mesa_clm.providers.claude_provider import ClaudeStructuredProvider

            self._second_opinion = ClaudeStructuredProvider(
                self.cfg.claude, self.provider.fingerprint, client=self._claude_client
            )
        return self._second_opinion

    def annotate(
        self,
        card: DatasetCard,
        actor: str,
        *,
        owner: str,
        second_opinion: bool | None = None,
        tier: str | None = None,
    ) -> AnnotationRun:
        """Decide ``card`` for ``owner`` (the identity the run belongs to) at the request of
        ``actor``; ``tier`` overrides ``cfg.decider.tier`` (``ols_rank`` for the degraded
        method); ``second_opinion`` defaults to ``cfg.claude.second_opinion``.
        :class:`DeciderBusy` when the lock stays taken for ``policy.max_wait_s``."""
        self._acquire()
        try:
            return Annotator(
                provider=self.provider,
                planner=self.planner,
                ols=self.ols,
                policy=self.policy,
                cfg=self.cfg,
                owner=owner,
                actor=actor,
                store=self.store,
                second_opinion=self.second_opinion_provider(second_opinion),
                tier=tier,
                ols_rank_tasks=self.ols_rank_tasks,
            ).annotate(card)
        finally:
            self.lock.release()

    # -- owners -----------------------------------------------------------------------------------

    def check_owner(self, run_id: UUID, owner: str) -> dict[str, Any]:
        """The run row when ``owner`` owns it; :class:`NotOwner` otherwise, ``KeyError`` for an
        unknown run. Apply and MRTR resume call this before touching a run (D21)."""
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"run {run_id} not found")
        if not owner or run.get("owner") != owner:
            raise NotOwner(f"run {run_id} belongs to another owner")
        return run

    def _run_of_group(self, group_id: UUID) -> UUID:
        group = self.store.group(group_id)
        if group is None:
            raise KeyError(f"group {group_id} not found")
        return UUID(str(group["run_id"]))

    # -- reads ------------------------------------------------------------------------------------

    def run_summary(self, run_id: UUID, *, owner: str | None = None) -> dict[str, Any]:
        """The run row, its links and groups, and the groups waiting for a reviewer
        (``pending_groups``: ids; ``pending``: their summaries). ``owner`` checks ownership."""
        if owner is not None:
            self.check_owner(run_id, owner)
        view = _RunView(self.store, run_id)
        pending = view.pending()
        return {
            "run": view.run,
            "links": view.links,
            "groups": view.groups,
            "pending_groups": [p["group_id"] for p in pending],
            "pending": pending,
        }

    def candidates_for_group(self, group_id: UUID) -> list[dict[str, Any]]:
        """The offered candidates of a group with their sidecar numbers, best ``p_fit`` first,
        the anchor (``__none__``) included."""
        view = _RunView(self.store, self._run_of_group(group_id))
        return view.candidates(group_id)

    def explain(
        self,
        *,
        owner: str,
        run_id: UUID | None = None,
        irods_path: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Read-only account of a run (or of the owner's runs that proposed AVUs for an iRODS
        path): the run, slim decisions, the top three options of each group, the links and the
        pending groups. Only the owner's runs are shown (D21): a ``run_id`` of another owner is
        :class:`NotOwner`; runs of other owners at ``irods_path`` are left out."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if run_id is not None:
            self.check_owner(run_id, owner)
            return self._explain_run(run_id, limit)
        if irods_path:
            rows = self.store.decisions_for_path(irods_path, limit=limit)
            runs: list[dict[str, Any]] = []
            for rid in dict.fromkeys(str(r["run_id"]) for r in rows):
                run = self.store.run(UUID(rid))
                if run is not None and run.get("owner") == owner:
                    runs.append(self._explain_run(UUID(rid), limit))
            return {"irods_path": irods_path, "runs": runs}
        raise ValueError("explain needs a run_id or an irods_path")

    def _explain_run(self, run_id: UUID, limit: int) -> dict[str, Any]:
        view = _RunView(self.store, run_id)
        decisions = sorted(view.decisions.values(), key=lambda d: int(d["seq"]))
        groups = []
        for g in view.groups[:limit]:
            slim = _slim_group(g)
            slim["top"] = [
                {
                    k: c[k]
                    for k in ("option_key", "label", "p_fit", "s_c", "rank", "is_anchor", "link_id")
                }
                for c in view.candidates(g["group_id"])[:3]
            ]
            groups.append(slim)
        return {
            "run": _json_run(view.run),
            "decisions": [_slim_decision(d) for d in decisions[:limit]],
            "n_decisions": len(decisions),
            "groups": groups,
            "links": [_slim_link(link) for link in view.links[:limit]],
            "pending_groups": [p["group_id"] for p in view.pending()],
        }

    def accepted_link_ids(self, run_id: UUID) -> list[str]:
        return [
            str(link["link_id"])
            for link in self.store.links(run_id)
            if link["write_status"] == "accepted"
        ]

    # -- human (and agent) feedback ------------------------------------------------------------

    def is_bench_card(self, card_name: str) -> bool:
        """Whether ``card_name`` is a bench card (D30), failing closed: a blank (unknown) name
        always is; otherwise one named in ``bench_cards`` when the caller gave them, else one
        of the fixed :data:`~mesa_clm.learn.labels.BENCH_CARDS` or a card carrying silver
        consensus labels in the store."""
        if not (card_name or "").strip():
            return True
        if self.bench_cards is not None:
            return card_name in self.bench_cards
        if is_bench_card(card_name):
            return True
        for task_id in INGESTED_TASKS:
            rows = self.store.labels_for(task_id, sources=CONSENSUS_SOURCES)
            if any(r.get("card") == card_name for r in rows):
                return True
        return False

    def _prepare_answer(
        self,
        group_id: UUID,
        *,
        via: Via,
        owner: str,
        option_key: str | None,
        action: PickAction,
    ) -> _Answer:
        """Every check :meth:`record_human_pick` makes before it writes (module docstring)."""
        if via not in VIAS:
            raise ValueError(f"unknown via {via!r}; expected one of {VIAS}")
        if action not in ACTIONS:
            raise ValueError(f"unknown feedback action {action!r}; expected one of {ACTIONS}")
        run_id = self._run_of_group(group_id)
        run = self.check_owner(run_id, owner)
        view = _RunView(self.store, run_id)
        group = view.group(group_id)
        if group["task_id"] not in RANK_FIT_TASKS:  # pragma: no cover - groups are rank_fits
            raise ValueError(f"group {group_id} is not a candidate group")
        decision = view.deciding(group)
        cands = view.candidates(group_id)
        if decision is None or not cands:
            raise KeyError(f"group {group_id} has no candidates to pick from")
        by_key = {c["option_key"]: c for c in cands}
        if option_key is not None and option_key not in by_key:
            raise ValueError("the chosen option is not among the offered candidates")
        chosen = None if option_key == ANCHOR_KEY else option_key
        if action == "reject":
            chosen = None
        explicit_none = action in ("pick", "reject") and chosen is None
        override_action: OverrideAction = (
            "decline"
            if action == "decline"
            else "reject"
            if action == "reject"
            else ("none" if explicit_none else "pick")
        )
        human = via in HUMAN_VIAS
        answer = _answer_key(override_action, chosen)
        by = view.answered_by(group_id)
        prior = {
            _answer_key(str(o["action"]), o.get("chosen_option_key"))
            for o in view.answers(group_id)
            if (str(o.get("via")) in HUMAN_VIAS) == (by == "human")
        }
        if by == "human" and (answer is None or answer not in prior):
            raise AlreadyAnswered(
                f"group {group_id} already has a curator answer (outcome {group['outcome']}); "
                "a curator answer is final in M1 (DESIGN A2)"
            )
        if by == "agent" and not human and answer is not None and answer not in prior:
            raise AlreadyAnswered(
                f"group {group_id} already has an agent answer; only a curator answer (at an "
                "interactive terminal or through MRTR elicitation) replaces it (DESIGN A2)"
            )
        return _Answer(
            run_id=run_id,
            run=run,
            view=view,
            group=group,
            decision=decision,
            cands=cands,
            by_key=by_key,
            chosen=chosen,
            explicit_none=explicit_none,
            override_action=override_action,
            human=human,
        )

    def check_answer(
        self,
        group_id: UUID,
        *,
        via: Via,
        owner: str,
        option_key: str | None = None,
        action: PickAction = "pick",
    ) -> None:
        """Raise what :meth:`record_human_pick` would raise for this answer, writing nothing
        (the CLI checks a whole batch of answers before it records any)."""
        self._prepare_answer(group_id, via=via, owner=owner, option_key=option_key, action=action)

    def record_human_pick(
        self,
        group_id: UUID,
        actor: str,
        *,
        via: Via,
        owner: str,
        option_key: str | None = None,
        action: PickAction = "pick",
        elicitation_key: str | None = None,
    ) -> dict[str, Any]:
        """Record what a reviewer (or an agent) did with a group (module docstring, D21).

        ``action='pick'`` with ``option_key`` one of the offered candidates accepts it; a pick
        of ``None`` or of the anchor is an explicit "none of these", as is ``'reject'``;
        ``'decline'`` answers nothing. ``via`` says where the answer came from: ``elicitation``
        (MRTR) and ``cli`` (an interactive terminal, DESIGN A2) mint curator labels, ``tool``
        only ``agent_pick``. ``owner`` must own the run (:class:`NotOwner`); a group a curator
        already answered differently, or an agent answered differently when this is an agent
        too, is :class:`AlreadyAnswered`. Returns ``override_id``, ``labels_written``,
        ``label_source``, ``accepted_link_id``, ``outcome`` (``human``, ``rejected`` or
        ``declined``), ``action``, ``group_id`` and ``run_id``."""
        a = self._prepare_answer(
            group_id, via=via, owner=owner, option_key=option_key, action=action
        )
        view, decision, cands, chosen = a.view, a.decision, a.cands, a.chosen
        pick_source: LabelSource = "curator" if a.human else "agent_pick"
        implicit_source: LabelSource = "curator_implicit" if a.human else "agent_pick"
        labels: list[LabelRow] = []
        if action != "decline":
            labels = self._labels(
                a.run,
                group_id,
                decision,
                cands,
                chosen=chosen,
                pick_source=pick_source,
                implicit_source=implicit_source,
                actor=actor,
            )
        n_labels = self.store.insert_labels(labels) if labels else 0
        accepted_link: str | None = None
        outcome: str
        by: AcceptedBy = "human" if a.human else "agent"
        accepted_before = [
            str(link["link_id"])
            for link in view.links_of(group_id)
            if link["write_status"] == "accepted"
        ]
        if action == "decline":
            outcome = "declined"
        elif chosen is None:
            outcome = "rejected"
            if accepted_before:
                self.store.set_link_status([UUID(i) for i in accepted_before], "proposed")
            self.store.update_group(group_id, outcome="rejected")
        else:
            outcome = "human"
            accepted_link = self._accept(view, a.group, decision, a.by_key[chosen], by, via)
            # One accepted link per group: an earlier agent (or policy) acceptance of another
            # candidate goes back to proposed.
            stale = [i for i in accepted_before if i != accepted_link]
            if stale:
                self.store.set_link_status([UUID(i) for i in stale], "proposed")
            self.store.update_group(group_id, outcome="human")
        override = HumanOverrideRow(
            run_id=a.run_id,
            group_id=group_id,
            decision_id=UUID(str(decision["decision_id"])),
            link_id=UUID(accepted_link) if accepted_link else None,
            actor=actor,
            via=via,
            action=a.override_action,
            chosen_decision_id=UUID(str(decision["decision_id"])) if chosen else None,
            chosen_option_key=chosen
            if chosen is not None
            else (ANCHOR_KEY if a.explicit_none else None),
            elicitation_key=elicitation_key,
            label_source=pick_source if action != "decline" else None,
            labels_written=n_labels,
            offered=[
                {
                    "option_key": c["option_key"],
                    "rank": c["rank"],
                    "p_fit": c["p_fit"],
                    "level": c["level"],
                    "method": c["method"],
                    "is_anchor": c["is_anchor"],
                }
                for c in cands
            ],
        )
        self.store.insert_override(override)
        return {
            "override_id": str(override.override_id),
            "group_id": str(group_id),
            "run_id": str(a.run_id),
            "action": a.override_action,
            "labels_written": n_labels,
            "label_source": override.label_source,
            "accepted_link_id": accepted_link,
            "outcome": outcome,
            "via": via,
        }

    def _labels(
        self,
        run: dict[str, Any],
        group_id: UUID,
        decision: dict[str, Any],
        cands: Sequence[dict[str, Any]],
        *,
        chosen: str | None,
        pick_source: LabelSource,
        implicit_source: LabelSource,
        actor: str,
    ) -> list[LabelRow]:
        """The label rows of a pick (D1, D21): the group decision's own task and target state,
        one row per offered candidate, plus an anchor-positive row for an explicit none."""
        task_id = str(decision["task_id"])
        task = TASKS[task_id]
        state: dict[str, Any] = dict(decision["state_json"] or {})
        card = str(run.get("card_name") or "")
        product = product_code_of(card)
        agent = pick_source == "agent_pick"
        bench = False if agent else self.is_bench_card(card)

        def row(option: str, label_index: int, source: LabelSource) -> LabelRow:
            ident = identity(task_id, state, option)
            return LabelRow(
                task_id=task_id,
                task_key=ident.task_key,
                target_sha256=ident.target_sha256,
                option_key=ident.option_key,
                label_source=source,
                label=task.options[label_index],
                label_index=label_index,
                weight=WEIGHTS[source],
                state_sha256=str(decision["state_sha256"]),
                state_json=state,
                card=card,
                product_code=product,
                leak_group=product,
                fold_eligible=source not in NOT_FOLD_ELIGIBLE,
                bench_card=bench,
                origin=f"override:{group_id}",
                actor=actor,
            )

        rows: list[LabelRow] = []
        if chosen is None:
            rows.append(row(ANCHOR_KEY, _YES, pick_source))
        for c in cands:
            key = c["option_key"]
            if c["is_anchor"]:
                continue
            if chosen is None:
                rows.append(row(key, _NO, pick_source))
            elif key == chosen:
                rows.append(row(key, _YES, pick_source))
            else:
                rows.append(row(key, _NO, implicit_source))
        return rows

    def _accept(
        self,
        view: _RunView,
        group: dict[str, Any],
        decision: dict[str, Any],
        cand: dict[str, Any],
        by: AcceptedBy,
        via: Via,
    ) -> str | None:
        """Accept the chosen candidate's link, or build one (a non-winner, an anchor-won group).
        ``None`` for a group that builds no AVU (``column.ontology_fits``)."""
        if group["task_id"] != "term.fits":
            return None
        if cand["link_id"]:
            self.store.set_link_status([UUID(cand["link_id"])], "accepted", accepted_by=by)
            return str(cand["link_id"])
        links = view.links_of(group["group_id"])
        shape = links[0] if links else None
        term = Candidate(
            label=str(cand["label"] or cand["option_key"]),
            curie=str(cand["option_key"]),
            iri=str(cand.get("iri") or ""),
            ontology_id=str(cand.get("ontology_id") or group.get("ontology_id") or ""),
        )
        column_name = str(group.get("column_name") or "")
        site_code = str(group.get("site_code") or "")
        aspect = str(group.get("aspect") or "")
        kind = (
            str(shape.get("value_kind") or VALUE_KIND_LABEL)
            if shape is not None
            else pre_rule_value_kind(aspect, str(group["scope"])) or VALUE_KIND_LABEL
        )
        value = value_for(
            kind,
            term_label=term.label,
            column_name=column_name or None,
            site_code=site_code or None,
            # defect (c): the most-frequent-value kind reuses the winner's value
            top_value=str(shape["value"]) if shape is not None and kind == VALUE_KIND_TOP else None,
        )
        avu = build_avu(term, value)
        for link in view.links:  # the same triple on the same target: accept that link
            same = (
                link["attribute"] == avu["attribute"]
                and link["value"] == avu["value"]
                and link["unit"] == avu["unit"]
                and (link.get("column_name") or "") == column_name
                and (link.get("site_code") or "") == site_code
            )
            if same:
                self.store.set_link_status([UUID(str(link["link_id"]))], "accepted", accepted_by=by)
                return str(link["link_id"])
        row = AvuLinkRow(
            run_id=UUID(str(group["run_id"])),
            group_id=UUID(str(group["group_id"])),
            decision_id=UUID(str(decision["decision_id"])),
            attribute=avu["attribute"],
            value=avu["value"],
            unit=avu["unit"],
            term_curie=term.curie,
            term_iri=term.iri or None,
            term_label=term.label,
            ontology_id=term.ontology_id,
            aspect=aspect or None,
            column_name=column_name,
            site_code=site_code,
            value_kind=kind,
            source=f"mesa-clm:feedback:{via}",
            write_status="accepted",
            accepted_by=by,
        )
        self.store.insert_links([row])
        return str(row.link_id)

    def close(self) -> None:
        self.store.close()
