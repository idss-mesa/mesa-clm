"""Claude as a *recorded second opinion* (DESIGN D6, D22; the ``claude`` extra).

The Messages API exposes no scores CLM's tiers could calibrate, so Claude is never a decider:
it answers the same question over the same context CLM saw (``framings.context_text``, which
ends with the target and never contains the raw card) through structured outputs
(``client.messages.parse(output_format=...)`` into a model whose only field is a ``Literal`` of
the option keys, adaptive thinking, no forced ``tool_choice``). For a rank_fit the options are
the group's candidates plus ``__none__``, so Claude can say "none of these" as CLM's anchor
does; for a closed choice they are the task's option labels.

Every record is ``provider='claude'``, ``method='claude:structured_output'``, ``level='none'``,
``calibration='none'``, ``probs=None``: invariants 1 and 2 of plan §4.6 keep it from ever
reaching ``auto`` ("LLM reasons, CLM selects"). What it can do is agree or disagree with a
proposed winner; the pipeline records it on the group and escalates a disagreement to a human.
Its ``question_key`` is derived from the framing's and this module's prompt, so a Claude answer
is never mistaken for a CLM decision of the framing's key in features, bench cells or artifacts.

``anthropic`` is imported lazily on the first call; credentials resolve through the SDK
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile), nothing about them is read or logged
here. A refusal, an unparsable answer, an SDK error or a missing SDK gives an abstaining record
(``answer_index=-1``, ``diagnostics['reason']``), never an exception. Ported from mesa-anyjev
``providers/claude_provider.py`` (``6159281``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from mesa_clm.clm.fingerprint import Fingerprint, canonical_json, sha256_hex
from mesa_clm.config import ClaudeConfig
from mesa_clm.framings import FRAMINGS, CandidateLike, Framing, context_text
from mesa_clm.providers.base import (
    DecisionRecord,
    Scalar,
    TierUnavailable,
    base_fields,
    check_batch,
    framing_options,
    sha256_text,
)
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.vocab import Level, Method

logger = logging.getLogger(__name__)

__all__ = [
    "METHOD",
    "PROMPT_VERSION",
    "PROVIDER",
    "SYSTEM",
    "ClaudeStructuredProvider",
    "answer_model",
    "claude_question_key",
    "render_prompt",
]

PROVIDER: Final[str] = "claude"
METHOD: Final[Method] = "claude:structured_output"
# Bump when SYSTEM or render_prompt changes: it enters claude_question_key.
PROMPT_VERSION: Final[str] = "1"
# Adaptive thinking shares max_tokens with the answer; below this a short answer can be cut off.
MIN_MAX_TOKENS: Final[int] = 1024
_ERROR_CHARS: Final[int] = 160
SYSTEM: Final[str] = (
    "You are a second-opinion decision function for ontology-grounded metadata annotation. "
    "You are given a context that describes a dataset target and a question with a closed "
    "set of options, each with an id. Pick exactly one option id from the list. Do not "
    "invent options and do not explain."
)


def _question_text(framing: Framing) -> str:
    """The question: the task's own wording for a closed choice; for a rank_fit, its per-
    candidate yes/no wording turned into a pick among the candidates or the anchor."""
    text = framing.task.text
    if framing.shape == "rank_fit":
        return (
            f"For each option, ask: {text} Pick the option for which the answer is most "
            f"clearly Yes; pick {ANCHOR_KEY} if the answer is No for every other option."
        )
    return text


def render_prompt(
    framing: Framing, state: Mapping[str, Any], options: Sequence[str], texts: Sequence[str]
) -> str:
    """The user turn: the context CLM's state head saw, the question, the options as
    ``- <id>: <text>``, and the answer instruction."""
    lines = [
        "Context:",
        context_text(framing, state),
        "",
        f"Question: {_question_text(framing)}",
        "",
        "Options:",
    ]
    lines += [f"- {key}: {text}" for key, text in zip(options, texts, strict=True)]
    lines += ["", "Answer with the id of exactly one option."]
    return "\n".join(lines)


def answer_model(options: Sequence[str]) -> type[BaseModel]:
    """``{"answer": <one of options>}`` as a Pydantic model for ``messages.parse``."""
    literal = Literal[tuple(options)]  # type: ignore[valid-type]
    model: type[BaseModel] = create_model(
        "Answer",
        __config__=ConfigDict(extra="forbid"),
        answer=(literal, Field(description="the id of the chosen option, verbatim")),
    )
    return model


def claude_question_key(framing: Framing) -> str:
    """The ``question_key`` of Claude's answers to ``framing``: the framing's key, the method,
    the system prompt and the prompt version, hashed like ``framings.question_key``."""
    payload = {
        "task_key": framing.task_key,
        "base_question_key": framing.question_key,
        "method": METHOD,
        "system": SYSTEM,
        "prompt_version": PROMPT_VERSION,
    }
    return sha256_hex(canonical_json(payload))[:16]


class ClaudeStructuredProvider:
    """Structured-output second opinion (module docstring). ``fingerprint`` is the run's live D5
    bundle, stamped on every record so the second opinion joins the decisions it was asked
    about; ``client`` is an ``anthropic.Anthropic`` (or a fake with ``messages.parse``)."""

    name: str = PROVIDER

    def __init__(
        self, cfg: ClaudeConfig, fingerprint: Fingerprint, client: Any | None = None
    ) -> None:
        self.cfg = cfg
        self.model: str = cfg.model
        self.served_model: str | None = None
        self.fingerprint = fingerprint
        self.requests = 0
        # A pre-built (or fake) client skips the SDK import entirely.
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # the `claude` extra

            self._client = anthropic.Anthropic()
        return self._client

    def resolve_tier(self, question_key: str) -> Level:
        return "none"

    def supports_tier(self, task_id: str, tier: str) -> bool:
        return task_id in FRAMINGS and tier in ("auto", "none")

    def decide(
        self,
        framing: Framing,
        states: Sequence[Mapping[str, Any]],
        candidates_per_state: Sequence[Sequence[CandidateLike]] | None = None,
        *,
        tier: str | None = None,
    ) -> list[DecisionRecord]:
        if tier not in (None, "auto", "none"):
            raise TierUnavailable(
                f"Claude is a second opinion at level 'none' and has no tier {tier!r} (D22)"
            )
        groups = check_batch(framing, states, candidates_per_state)
        return [self._decide_one(framing, s, g) for s, g in zip(states, groups, strict=True)]

    def _decide_one(
        self,
        framing: Framing,
        state: Mapping[str, Any],
        candidates: Sequence[CandidateLike] | None,
    ) -> DecisionRecord:
        opts = framing_options(framing, candidates)
        prompt = render_prompt(framing, state, opts.options, opts.option_texts)
        diag: dict[str, Scalar] = {"base_question_key": framing.question_key}
        answer_index = -1
        input_tokens = 0
        served: str | None = None
        try:
            self.requests += 1
            response = self._get_client().messages.parse(
                model=self.model,
                max_tokens=max(MIN_MAX_TOKENS, int(self.cfg.max_tokens)),
                system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.effort},
                messages=[{"role": "user", "content": prompt}],
                output_format=answer_model(opts.options),
            )
            usage = getattr(response, "usage", None)
            input_tokens = _int_or_zero(getattr(usage, "input_tokens", None))
            diag["output_tokens"] = _int_or_zero(getattr(usage, "output_tokens", None))
            model_name = getattr(response, "model", None)
            served = model_name if isinstance(model_name, str) and model_name else None
            stop = getattr(response, "stop_reason", None)
            diag["stop_reason"] = str(stop) if stop is not None else None
            if stop == "refusal":
                diag["reason"] = "refusal"
            else:
                parsed = getattr(response, "parsed_output", None)
                text = str(getattr(parsed, "answer", "") or "")
                if text in opts.options:
                    answer_index = opts.options.index(text)
                else:
                    diag["reason"] = "unparsed"
                    diag["unparsed"] = text[:100]
        except Exception as exc:  # SDK, network, import and parse errors alike
            logger.warning("claude second opinion failed (%s)", type(exc).__name__)
            diag["reason"] = "error"
            diag["error"] = f"{type(exc).__name__}: {str(exc)[:_ERROR_CHARS]}"
        if served is not None:
            self.served_model = served
        f = base_fields(
            framing,
            state,
            self.fingerprint,
            provider=PROVIDER,
            method=METHOD,
            model=self.model,
            context_sha256=sha256_text(SYSTEM + "\n\n" + prompt),
            context_tokens=input_tokens,
            question_key=claude_question_key(framing),
        )
        f.update(
            options=opts.options,
            option_texts=opts.option_texts,
            anchor_index=opts.anchor_index,
            served_model=served,
            level="none",
            calibration="none",
            answer_index=answer_index,
            answer=opts.options[answer_index] if answer_index >= 0 else "",
            diagnostics=diag,
        )
        return DecisionRecord.model_validate(f)


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
