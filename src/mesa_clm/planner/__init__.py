"""Planners: the reasoning role. A planner proposes (ontologies, columns, queries); it never
answers a question and never writes (DESIGN D22: "LLM reasons, CLM selects").

:func:`make_planner` picks the planner the configuration (or a ``--planner`` flag) names; the
gateway and Claude planners are imported only when chosen, so the static default pulls in
neither httpx client construction nor the optional ``anthropic`` SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_clm.planner.base import ColumnHint, Plan, Planner, PlanResult, SiteHint
from mesa_clm.planner.static_planner import StaticPlanner

if TYPE_CHECKING:
    from mesa_clm.config import Config, PlannerKind

__all__ = [
    "ColumnHint",
    "Plan",
    "PlanResult",
    "Planner",
    "SiteHint",
    "StaticPlanner",
    "make_planner",
]


def make_planner(cfg: Config, kind: PlannerKind | None = None) -> Planner:
    """The planner ``kind`` names (default ``cfg.planner.kind``): ``static``, ``gateway`` (the
    CARC LiteLLM gateway, key from the configured secrets source, remote hosts only with
    ``cfg.clm.allow_remote`` and https) or ``claude`` (the ``claude`` extra)."""
    chosen = kind or cfg.planner.kind
    if chosen == "gateway":
        from mesa_clm.planner.gateway_planner import GatewayPlanner

        return GatewayPlanner.from_config(cfg.planner, allow_remote=cfg.clm.allow_remote)
    if chosen == "claude":
        from mesa_clm.planner.claude_planner import ClaudePlanner

        return ClaudePlanner(cfg.planner)
    return StaticPlanner()
