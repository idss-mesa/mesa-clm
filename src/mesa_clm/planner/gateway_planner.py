"""The reasoning model on the CARC LiteLLM gateway (``carc-tools``) as a planner (DESIGN D22).

A port of neon-avu-eval's streamed chat loop, via mesa-anyjev ``planner/gateway_planner.py``:
streaming keeps bytes flowing so the gateway's read timeout fires only on real stalls;
``chat_template_kwargs.enable_thinking=false`` where the model honours it; JSON-only
instruction, :func:`parse_json` strips ``<think>`` blocks and fences; up to two nudges; any
failure falls back to :class:`~mesa_clm.planner.static_planner.StaticPlanner` and marks the
result ``fallback=True`` (the run row records it as ``planner_fallback``).

The base URL goes through :func:`mesa_clm.net.assert_loopback` (loopback only unless
``allow_remote`` *and* https), the client is built with ``trust_env=False`` so no proxy variable
can redirect the bearer key, and the key itself is only ever placed in the ``Authorization``
header, never logged. Before every keyed request over the real network the planner asks who
holds the loopback port (:func:`mesa_clm.net.assert_listener_owner`, DESIGN A5), and its
transport asks again for each new connection once it is made and before a byte is sent on it
(:class:`mesa_clm.net.OwnerCheckedTransport`): the default gateway port is the
``carc-litellm-tunnel`` user unit's, free for any local account to take whenever the tunnel is
down, and another account's socket there gets no key (the plan falls back to the static rules).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import httpx
from pydantic import ValidationError

from mesa_clm.cards import DatasetCard
from mesa_clm.config import PlannerConfig
from mesa_clm.net import OwnerCheckedTransport, assert_listener_owner, assert_loopback
from mesa_clm.planner.base import Plan, PlanResult
from mesa_clm.planner.static_planner import StaticPlanner
from mesa_clm.registry import ASPECT_OPTIONS, ONTOLOGY_REGISTRY

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a careful scientific metadata curator planning how to annotate a dataset with OBO "
    "Foundry ontology terms. You only PLAN: which ontologies apply, which columns deserve an "
    "annotation and with which aspect and ontology, and short search queries. You never invent "
    "identifiers. Reply with ONLY a JSON object, no prose, no code fence."
)

# Hard-coded with the prompt: one planner call must not run away on a chatty model.
MAX_TOKENS = 4096
# One answer plus two nudges ("Reply now with ONLY the JSON object ...").
MAX_ATTEMPTS = 3
NUDGE = "Reply now with ONLY the JSON object described, nothing else."


def parse_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model reply, with ``<think>`` blocks and code fences removed;
    ``None`` when there is none (or it is not an object)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def plan_prompt(card: DatasetCard) -> str:
    """The user prompt: the card (columns with a 160-character profile, sites, rows), the
    ontology registry, the aspects and the exact JSON shape wanted."""
    registry = "\n".join(f"- {e.id}: {e.option_text}" for e in ONTOLOGY_REGISTRY)
    aspects = "\n".join(f"- {a}" for a in ASPECT_OPTIONS)
    columns = "\n".join(
        f"- {c.name} | {c.description} | {c.dtype} | {c.unit or '-'} | {c.profile[:160]}"
        for c in card.columns
    )
    sites = "; ".join(f"{s.code} ({s.name}, {s.domain}; {s.habitat})" for s in card.sites)
    return (
        f"Dataset: {card.name}\nProduct: {card.product_title}. {card.product_description}\n"
        f"Sites: {sites}\nRows: {card.rows}; months {card.months_from} to {card.months_to}\n\n"
        f"Columns (name | description | type | unit | profile):\n{columns}\n\n"
        f"Ontology registry (use only these ids):\n{registry}\n\nAspects:\n{aspects}\n\n"
        "Return JSON of this exact shape:\n"
        '{"ontologies": ["envo", ...], "columns": {"<column name>": {"annotate": true|false|null, '
        '"aspect": "<aspect id or null>", "ontology": "<registry id or null>", "queries": ["<=3 short OLS queries"]}}, '
        '"sites": {"<SITE>": {"environment_queries": ["biome query", ...]}}, "taxon_queries": ["taxon name", ...], '
        '"notes": "one sentence"}\n'
        "Mark record identifiers, internal codes and timestamps annotate=false. Keep queries short (1-4 words)."
    )


def prompt_sha256(prompt: str) -> str:
    """The hash stored on the run: over the system prompt and the user prompt together."""
    return hashlib.sha256((SYSTEM + prompt).encode("utf-8")).hexdigest()


class GatewayPlanner:
    """Streamed ``/v1/chat/completions`` on the gateway; static rules on any failure."""

    name: str = "gateway"
    model: str | None

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str = "carc-tools",
        timeout: float = 600.0,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        think: bool = False,
        allow_remote: bool = False,
    ) -> None:
        # Raises EndpointError before any request: a misconfigured URL is loud, not a fallback.
        self.base_url = assert_loopback(base_url, allow_remote=allow_remote, what="planner gateway")
        self.model = model
        self.timeout = timeout
        self.think = think
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # Over the real network (no injected client or transport) a keyed request first asks
        # who holds a loopback port, and each new connection is asked again before a byte is
        # sent on it: the key never goes to another account's socket (DESIGN A5).
        self._check_owner = client is None and transport is None and bool(api_key)
        if self._check_owner:
            transport = OwnerCheckedTransport("planner gateway")
        # `client` replaces the whole client (its headers included); `transport` keeps ours and
        # swaps only the wire (httpx.MockTransport in tests).
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            trust_env=False,
            transport=transport,
        )
        self._static = StaticPlanner()

    @classmethod
    def from_config(
        cls,
        cfg: PlannerConfig,
        *,
        allow_remote: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> GatewayPlanner:
        """A planner from the ``planner`` section: the key resolves through the configured
        secrets source (``resolved_gateway_api_key``), never through ``str(SecretStr)``."""
        return cls(
            cfg.gateway_base_url,
            cfg.resolved_gateway_api_key(),
            cfg.gateway_model,
            cfg.timeout,
            transport=transport,
            allow_remote=allow_remote,
        )

    def _chat(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
        """One streamed chat completion: the concatenated content deltas and the final usage
        object (``stream_options.include_usage``)."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": self.think},
        }
        content: list[str] = []
        usage: dict[str, Any] = {}
        if self._check_owner:
            # ListenerOwnerError: refused before a byte is sent; plan() falls back to the rules.
            assert_listener_owner(self.base_url, what="planner gateway")
        with self._client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        content.append(delta)
        return "".join(content), usage

    def plan(self, card: DatasetCard) -> PlanResult:
        prompt = plan_prompt(card)
        sha = prompt_sha256(prompt)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        usage_total: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        raw = ""
        try:
            for nudge in range(MAX_ATTEMPTS):
                raw, usage = self._chat(messages, max_tokens=MAX_TOKENS)
                for k in ("prompt_tokens", "completion_tokens"):
                    usage_total[k] += int(usage.get(k, 0))
                usage_total["llm_calls"] += 1
                parsed = parse_json(raw)
                if parsed is not None:
                    try:
                        plan = Plan.model_validate(_coerce(parsed))
                    except ValidationError as exc:
                        logger.warning(
                            "gateway planner: plan rejected (%s); nudging", exc.error_count()
                        )
                        parsed = None
                    else:
                        return PlanResult(
                            plan=plan,
                            planner=self.name,
                            model=self.model,
                            prompt_sha256=sha,
                            raw_text=raw,
                            usage=usage_total,
                        )
                if nudge < MAX_ATTEMPTS - 1:
                    messages += [
                        {"role": "assistant", "content": raw or "(no answer)"},
                        {"role": "user", "content": NUDGE},
                    ]
        except Exception as exc:  # a planner failure must never fail the run
            # httpx errors, the HTTP >= 400 RuntimeError, malformed SSE chunks (JSONDecodeError,
            # a non-object chunk): the plan is only a hint, so the static rules take over. The
            # message names the exception, never the request (whose header carries the key).
            logger.warning(
                "gateway planner failed (%s: %s); falling back to static rules",
                type(exc).__name__,
                exc,
            )
        fallback = self._static.plan(card)
        return fallback.model_copy(
            update={
                "planner": self.name,
                "model": self.model,
                "prompt_sha256": sha,
                "raw_text": raw or None,
                "usage": usage_total,
                "fallback": True,
            }
        )


def _coerce(parsed: dict[str, Any]) -> dict[str, Any]:
    """Tolerate the small shape deviations local models make (a list of column objects, etc.)."""
    cols = parsed.get("columns")
    if isinstance(cols, list):
        parsed["columns"] = {
            str(c.get("name", i)): {k: v for k, v in c.items() if k != "name"}
            for i, c in enumerate(cols)
            if isinstance(c, dict)
        }
    for key in ("columns", "sites"):
        if not isinstance(parsed.get(key), dict):
            parsed[key] = {}
    for key in ("ontologies", "taxon_queries"):
        if not isinstance(parsed.get(key), list):
            parsed[key] = []
    if not isinstance(parsed.get("notes"), str):
        parsed["notes"] = ""
    for name, hint in list(parsed["columns"].items()):
        if isinstance(hint, dict):
            hint.setdefault("queries", [])
            if not isinstance(hint["queries"], list):
                hint["queries"] = [str(hint["queries"])]
            parsed["columns"][name] = {
                k: hint.get(k) for k in ("annotate", "aspect", "ontology", "queries")
            }
        else:
            parsed["columns"].pop(name)
    for code, hint in list(parsed["sites"].items()):
        if isinstance(hint, dict):
            eq = hint.get("environment_queries", [])
            parsed["sites"][code] = {
                "environment_queries": eq if isinstance(eq, list) else [str(eq)]
            }
        else:
            parsed["sites"].pop(code)
    return parsed
