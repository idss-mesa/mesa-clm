"""Claude as the planner (DESIGN D22; the ``claude`` extra).

Structured outputs through ``client.messages.parse(output_format=Plan)`` with adaptive
thinking and a configurable effort; no forced ``tool_choice`` (rejected on the newest
models) and no prefill. A refusal, a timeout, a validation failure, a missing SDK or an SDK
error falls back to the static planner and marks the result ``fallback=True``. Credentials
resolve through the SDK itself (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile);
nothing is read from this package's configuration, and nothing about them is logged.

``anthropic`` is imported lazily on the first call, so the core installs and imports without
the extra; a missing SDK is one more reason to fall back, not an import error at startup.
Ported from mesa-anyjev ``planner/claude_planner.py`` (``6159281``).
"""

from __future__ import annotations

import logging
from typing import Any

from mesa_clm.cards import DatasetCard
from mesa_clm.config import PlannerConfig
from mesa_clm.planner.base import Plan, PlanResult
from mesa_clm.planner.gateway_planner import SYSTEM, plan_prompt, prompt_sha256
from mesa_clm.planner.static_planner import StaticPlanner

logger = logging.getLogger(__name__)

# Structured output plus adaptive thinking: generous, and the SDK streams internally if needed.
MAX_TOKENS = 16000


class ClaudePlanner:
    """``messages.parse`` into :class:`Plan`; static rules on refusal, error or no parse."""

    name: str = "claude"
    model: str | None

    def __init__(self, cfg: PlannerConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.claude_model
        # A pre-built (or fake) client skips the SDK import entirely.
        self._client = client
        self._static = StaticPlanner()

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # the `claude` extra

            self._client = anthropic.Anthropic(timeout=self.cfg.timeout)
        return self._client

    def plan(self, card: DatasetCard) -> PlanResult:
        prompt = plan_prompt(card)
        sha = prompt_sha256(prompt)
        usage: dict[str, Any] = {}
        raw: str | None = None
        try:
            response = self._get_client().messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.claude_effort},
                messages=[{"role": "user", "content": prompt}],
                output_format=Plan,
            )
            u = getattr(response, "usage", None)
            usage = {
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
            }
            if getattr(response, "stop_reason", None) == "refusal":
                logger.warning("claude planner: refusal; falling back to static rules")
            else:
                parsed = getattr(response, "parsed_output", None)
                if isinstance(parsed, Plan):
                    return PlanResult(
                        plan=parsed,
                        planner=self.name,
                        model=self.model,
                        prompt_sha256=sha,
                        usage=usage,
                    )
                raw = "".join(
                    getattr(b, "text", "") for b in getattr(response, "content", []) or []
                )
                logger.warning("claude planner: no parsed plan; falling back to static rules")
        except Exception as exc:  # SDK, network, import and parse errors alike
            logger.warning(
                "claude planner failed (%s: %s); falling back to static rules",
                type(exc).__name__,
                exc,
            )
        fallback = self._static.plan(card)
        return fallback.model_copy(
            update={
                "planner": self.name,
                "model": self.model,
                "prompt_sha256": sha,
                "raw_text": raw,
                "usage": usage,
                "fallback": True,
            }
        )
