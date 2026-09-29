"""The torch-free client for clm-serve (plan §3 ``clm/http.py``, §4.1, §6.4; DESIGN D16).

:class:`ClmHttpClient` speaks CLM's System One wire format over ``httpx``: ``POST /v1/systemone``
(every question answered against one state), ``POST /v1/rank``, ``GET /v1/models`` and
``GET /health``. Its request bodies are byte-for-byte what CLM's own ``client.py`` sends
(``tests/_vendor/clm_client.py`` is the oracle; ``tests/unit/test_wire_parity.py`` diffs them),
so a golden recorded through the vendored client replays against this one. The question
dataclasses :class:`Noul`, :class:`Choice` and :class:`Score` have the same ``to_dict()`` output
as the vendored ones; answers come back as typed Pydantic models instead of bare dicts.

Transport rules (plan §6.4): the base URL passes :func:`mesa_clm.net.assert_loopback` (loopback,
or ``allow_remote`` *and* https), the ``httpx.Client`` runs with ``trust_env=False`` so no proxy
variable can redirect a request, a :class:`~mesa_clm.net.CircuitBreaker` stops a client from
hammering a server that keeps failing, and transient failures (transport errors and the statuses
in :data:`RETRY_STATUSES`) are retried with exponential backoff. Every other non-2xx status is a
:class:`ClmError` ``(status, message)`` mirroring the vendored ``CLMError``: ``message`` is the
server's ``detail`` when the body is JSON, else the first 300 characters of the text, with the
bearer key redacted should a server ever echo it. Status ``0`` means no HTTP response at all
(unreachable, timed out): the provider maps it to ``decider_unavailable``.

Nothing here logs a header or a body; log lines carry the redacted base URL, the path, the
status and the elapsed time only.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mesa_clm.config import ClmConfig
from mesa_clm.net import CircuitBreaker, assert_loopback, redact_url

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "clm-latest"
LATENCY_HEADER: Final[str] = "X-CLM-Latency-Ms"
# 429 too many requests, 5xx gateway/server trouble, 529 (Anthropic-style overloaded).
RETRY_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504, 529})
DEFAULT_RETRIES: Final[int] = 3
DEFAULT_BACKOFF_S: Final[float] = 0.5
MAX_RETRY_AFTER_S: Final[float] = 30.0
HEALTH_TIMEOUT_S: Final[float] = 5.0
_DETAIL_CHARS: Final[int] = 300
_REDACTED: Final[str] = "<redacted>"


class ClmError(RuntimeError):
    """A request clm-serve (or the encoder) refused or could not be completed.

    ``status`` is the HTTP status, or ``0`` when no response arrived (transport error, timeout,
    breaker open is *not* this: :class:`mesa_clm.net.BreakerOpenError` is raised before any
    request). ``message`` never carries a secret.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


# -- questions (same to_dict() output as the vendored client) -----------------------------------


@dataclass(frozen=True)
class Noul:
    """A yes/no question; the answer is the probability that it is true (control arm only, D3)."""

    instructions: Any = None
    criteria: Mapping[str, Any] | None = None  # optional {"true": ..., "false": ...} descriptions

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            d["criteria"] = dict(self.criteria)
        return d


@dataclass(frozen=True)
class Choice:
    """Pick one option; the answer is a distribution over ``criteria`` keys (the rank_fit shape
    with ``registry.ANCHOR_KEY`` among the keys, or a closed choice)."""

    criteria: Mapping[str, Any]
    instructions: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """Rate on an ordered rubric. Never used by mesa-clm decisions (D3); kept for wire parity."""

    criteria: Sequence[Any]
    instructions: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score | Mapping[str, Any]


def question_to_dict(q: Question) -> dict[str, Any]:
    """The wire dict of a question object, or a copy of an already-wire-shaped mapping."""
    if isinstance(q, Noul | Choice | Score):
        return q.to_dict()
    return dict(q)


# -- answers ------------------------------------------------------------------------------------


class _Wire(BaseModel):
    """A response model: unknown fields are refused, because the response contract is pinned to
    CLM ``bb42c6c5`` and a new field is drift the doctor must see (K4), not something to skip."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NoulAnswer(_Wire):
    type: Literal["noul"] = "noul"
    noul: float

    @property
    def probabilities(self) -> dict[str, float]:
        return {"false": 1.0 - self.noul, "true": self.noul}


class ChoiceAnswer(_Wire):
    """``confidence`` is CLM's own field (``p_top - mean(p_rest)``); records store it only as
    ``clm_confidence`` and compute ``confidence = max(probs)`` locally (D7)."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(_Wire):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    probabilities: dict[str, float]
    legend: dict[str, Any] = Field(default_factory=dict)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(_Wire):
    """``billing_units = len(questions)``; ``input_tokens`` counts encoder tokens on cache misses
    only (RESEARCH.md), so it is a lower bound on the tokens a request cost."""

    billing_units: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _null_units_are_zero(cls, data: Any) -> Any:
        # The vendored client reads ``int(u.get("billing_units", 0) or 0)``.
        if isinstance(data, dict) and data.get("billing_units") is None:
            data = {**data, "billing_units": 0}
        return data


class SystemOneResponse(_Wire):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    # From the X-CLM-Latency-Ms header; None when the server did not send it.
    latency_ms: float | None = None


class RankedCandidate(_Wire):
    rank: int
    candidate: str
    prob: float


# -- the endpoint: loopback gate, breaker, retries ----------------------------------------------


class HttpEndpoint:
    """One base URL with the transport rules of the module docstring applied.

    ``request`` returns the 2xx :class:`httpx.Response` or raises :class:`ClmError`;
    :class:`~mesa_clm.net.BreakerOpenError` propagates from the breaker. ``sleep`` and
    ``transport`` are injectable so tests run without waiting or a network.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        what: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        allow_remote: bool = False,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF_S,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1 (the first attempt counts)")
        # Raises EndpointError before any request: a misconfigured URL is loud, not a fallback.
        self.base_url = assert_loopback(base_url, allow_remote=allow_remote, what=what)
        self.what = what
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.breaker = breaker or CircuitBreaker(name=what)
        self._sleep = sleep
        self._api_key = api_key or None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            trust_env=False,
            transport=transport,
        )
        self.calls = 0  # successful (2xx) requests, for runs.n_clm_calls

    @property
    def shown(self) -> str:
        """The base URL as it may appear in a log or an error (no userinfo, query or fragment)."""
        return redact_url(self.base_url)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def redact(self, text: str) -> str:
        """``text`` with the bearer key replaced, should a server ever echo it back."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, _REDACTED)
        return text

    def detail(self, response: httpx.Response) -> str:
        """The server's error text: JSON ``detail`` (FastAPI) or ``error`` (vLLM), else the body."""
        text: str
        try:
            j = response.json()
        except ValueError:
            text = response.text
        else:
            if isinstance(j, dict) and "detail" in j:
                text = str(j["detail"])
            elif isinstance(j, dict) and "error" in j:
                err = j["error"]
                text = str(err.get("message", err)) if isinstance(err, dict) else str(err)
            else:
                text = response.text
        return self.redact(text[:_DETAIL_CHARS])

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        timeout: float | None = None,
        retry: bool = True,
    ) -> httpx.Response:
        self.breaker.check()
        attempts = self.retries if retry else 1
        started = time.monotonic()
        for attempt in range(attempts):
            last_try = attempt + 1 >= attempts
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json,
                    timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
                )
            except httpx.TransportError as exc:
                self.breaker.failure()
                reason = f"{type(exc).__name__}: {self.redact(str(exc))}"[:_DETAIL_CHARS]
                logger.debug(
                    "%s %s %s: no response (%s)", self.what, method, path, type(exc).__name__
                )
                if last_try:
                    raise ClmError(
                        0,
                        f"{self.what} at {self.shown} unreachable after {attempts} attempt(s): {reason}",
                    ) from exc
                self._sleep(self._delay(attempt, None))
                continue
            status = response.status_code
            if status in RETRY_STATUSES:
                self.breaker.failure()
                logger.debug("%s %s %s -> %d (retryable)", self.what, method, path, status)
                if last_try:
                    raise ClmError(
                        status,
                        f"{self.what} at {self.shown} still failing after {attempts} attempt(s): "
                        f"{self.detail(response)}",
                    )
                self._sleep(self._delay(attempt, response))
                continue
            if not 200 <= status < 300:
                # The request, not the endpoint, is wrong (401 bad key, 422 malformed): no breaker
                # failure, no retry, the server's own explanation.
                logger.debug("%s %s %s -> %d", self.what, method, path, status)
                raise ClmError(status, self.detail(response))
            self.breaker.success()
            self.calls += 1
            logger.debug(
                "%s %s %s -> %d in %.0f ms",
                self.what,
                method,
                path,
                status,
                (time.monotonic() - started) * 1000,
            )
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        delay = self.backoff * (2.0**attempt)
        if response is not None:
            hint = response.headers.get("Retry-After")
            if hint is not None:
                # An HTTP-date Retry-After is ignored; the backoff stands.
                with contextlib.suppress(ValueError):
                    delay = max(delay, min(float(hint), MAX_RETRY_AFTER_S))
        return delay

    def get_json(self, path: str, *, timeout: float | None = None, retry: bool = True) -> Any:
        return self._json(self.request("GET", path, timeout=timeout, retry=retry))

    def post_json(
        self, path: str, body: Any, *, timeout: float | None = None
    ) -> tuple[Any, httpx.Response]:
        response = self.request("POST", path, json=body, timeout=timeout)
        return self._json(response), response

    def _json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            raise ClmError(
                response.status_code,
                f"{self.what} returned a non-JSON body: "
                f"{self.redact(response.text[:_DETAIL_CHARS])}",
            ) from None


# -- the CLM client -----------------------------------------------------------------------------


class ClmHttpClient:
    """CLM's System One API over :class:`HttpEndpoint` (module docstring).

    ``model`` is the default served head (``clm-latest`` | ``clm-raw`` | a promoted head's unique
    name, D15); a call may override it. Requests always use the server's default temperature
    unless one is passed: calibration is client-side (plan §4.1).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
        allow_remote: bool = False,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF_S,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.endpoint = HttpEndpoint(
            base_url,
            api_key,
            what="CLM server",
            timeout=timeout,
            transport=transport,
            allow_remote=allow_remote,
            retries=retries,
            backoff=backoff,
            breaker=breaker,
            sleep=sleep,
        )

    @classmethod
    def from_config(
        cls, cfg: ClmConfig, *, transport: httpx.BaseTransport | None = None
    ) -> ClmHttpClient:
        """A client from the ``clm`` section: the key resolves through the configured secrets
        source (``resolved_api_key``), never through ``str(SecretStr)``."""
        return cls(
            cfg.base_url,
            cfg.resolved_api_key(),
            model=cfg.model,
            timeout=cfg.timeout,
            transport=transport,
            allow_remote=cfg.allow_remote,
        )

    @property
    def base_url(self) -> str:
        return self.endpoint.base_url

    @property
    def breaker(self) -> CircuitBreaker:
        return self.endpoint.breaker

    @property
    def calls(self) -> int:
        """Successful requests so far (``runs.n_clm_calls``)."""
        return self.endpoint.calls

    def close(self) -> None:
        self.endpoint.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        """One request: every question answered against one state (questions sharing a context
        go in one request, plan §4.1). ``temperature`` (server default 1.0) flattens (>1) or
        sharpens (<1) the distributions; mesa-clm leaves it unset."""
        body: dict[str, Any] = {
            "state": state,
            "model": model or self.model,
            "questions": {k: question_to_dict(q) for k, q in questions.items()},
        }
        if temperature is not None:
            body["temperature"] = temperature
        j, response = self.endpoint.post_json("/v1/systemone", body)
        latency = response.headers.get(LATENCY_HEADER)
        payload = dict(j) if isinstance(j, dict) else {"answers": j}
        payload["latency_ms"] = _float_or_none(latency)
        try:
            return SystemOneResponse.model_validate(payload)
        except ValidationError as exc:
            raise ClmError(
                response.status_code,
                f"malformed /v1/systemone response: {exc.error_count()} error(s): {exc}",
            ) from None

    def rank(
        self,
        context: Any,
        question: str | None,
        answers: Sequence[str],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> list[RankedCandidate]:
        """Rank ``answers`` for ``context`` (+ ``question``, which may be None); best first."""
        body: dict[str, Any] = {
            "context": context,
            "question": question,
            "answers": list(answers),
            "model": model or self.model,
        }
        if temperature is not None:
            body["temperature"] = temperature
        j, response = self.endpoint.post_json("/v1/rank", body)
        try:
            ranked = j["ranked"]
            return [RankedCandidate.model_validate(item) for item in ranked]
        except (KeyError, TypeError, ValidationError) as exc:
            raise ClmError(response.status_code, f"malformed /v1/rank response: {exc}") from None

    def models(self) -> list[dict[str, Any]]:
        """The served models (``clm-latest``, ``clm-raw`` and every promoted head)."""
        j = self.endpoint.get_json("/v1/models")
        try:
            models = j["models"]
        except (KeyError, TypeError) as exc:
            raise ClmError(200, f"malformed /v1/models response: {exc}") from None
        if not isinstance(models, list):
            raise ClmError(200, "malformed /v1/models response: 'models' is not a list")
        return [dict(m) for m in models]

    def health(self) -> bool:
        """``GET /health`` ``ok`` with a 5 s timeout and no retry; any failure is ``False``."""
        try:
            j = self.endpoint.get_json("/health", timeout=HEALTH_TIMEOUT_S, retry=False)
        except Exception:  # health is a probe; the reason belongs to the doctor
            return False
        return bool(isinstance(j, dict) and j.get("ok"))


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
