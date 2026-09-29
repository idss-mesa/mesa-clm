"""Where mesa-clm may send a request: loopback by default, a remote host only over https.

:func:`assert_loopback` is the gate every HTTP client in the core passes its base URL through:
the CLM server, the encoder and the planner gateway (DESIGN D16; plan §6.4). Loopback hosts
(``127.0.0.1``, ``localhost``, ``::1``) are allowed over http or https. Any other host is refused
unless the caller passes ``allow_remote=True`` (``MESA_CLM_CLM__ALLOW_REMOTE=1``) *and* the URL
is ``https://``: plain http to a remote host would carry the bearer key in clear, so that case
goes through an ``ssh -L`` forward to loopback instead. :class:`CircuitBreaker` keeps a client
from hammering an endpoint that keeps failing; both are ported from mesa-anyjev's gateway backend
(``backends/gateway.py``), the breaker made thread-safe because clients call it from worker
threads.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_SCHEMES = frozenset({"http", "https"})


class EndpointError(RuntimeError):
    """A base URL mesa-clm refuses to call, or an endpoint the breaker has taken offline."""


class BreakerOpenError(EndpointError):
    """The circuit breaker is open: recent calls failed and the cool-down has not elapsed."""


def redact_url(url: str) -> str:
    """``url`` reduced to scheme, host, port and path: no userinfo, query or fragment, so it can
    go into a log line or an error message."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "<unparseable URL>"
    if ":" in host:  # an IPv6 literal goes back into brackets
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` (as ``urlsplit(...).hostname`` gives it: lower-case, no brackets) is one
    of the loopback names mesa-clm trusts."""
    return host.lower() in LOOPBACK_HOSTS


def assert_loopback(base_url: str, *, allow_remote: bool = False, what: str = "endpoint") -> str:
    """Return ``base_url`` without its trailing slash if mesa-clm may call it, else raise.

    Refused: credentials in the URL (use the ``*_API_KEY`` settings), a scheme other than
    http(s), a missing host, a query or fragment, a path ending in ``/v1`` (every client appends
    its own ``/v1/...`` paths), and any non-loopback host unless ``allow_remote`` *and* https.
    ``what`` names the endpoint in messages (``"CLM server"``, ``"encoder"``, ``"planner
    gateway"``).
    """
    url = base_url.strip()
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        raise EndpointError(f"{what}: {redact_url(url)!r} is not a valid URL") from None
    shown = redact_url(url)
    if parts.username is not None or parts.password is not None:
        raise EndpointError(
            f"{what}: the URL for {shown!r} carries credentials; put the key in the *_API_KEY "
            "(or *_API_KEY_FILE) setting instead"
        )
    if parts.scheme not in ALLOWED_SCHEMES:
        raise EndpointError(f"{what}: {shown!r} must use http:// or https://")
    if not host:
        raise EndpointError(f"{what}: {shown!r} has no host")
    if parts.query or parts.fragment:
        raise EndpointError(f"{what}: a base URL takes no query or fragment, got {shown!r}")
    if parts.path.rstrip("/").endswith("/v1"):
        raise EndpointError(
            f"{what}: the base URL must not end in /v1 (the client appends /v1/...), got {shown!r}"
        )
    if not is_loopback_host(host):
        if not allow_remote:
            raise EndpointError(
                f"{what} must be reached over loopback (127.0.0.1, localhost, ::1), got {shown!r}; "
                "for a remote host set MESA_CLM_CLM__ALLOW_REMOTE=1 and use https://, or forward "
                "the port to loopback with ssh -L"
            )
        if parts.scheme != "https":
            raise EndpointError(
                f"{what}: a remote host needs https://, got {shown!r} (plain http would send the "
                "bearer key in clear; use an ssh -L forward to loopback instead)"
            )
    return url.rstrip("/")


@dataclass
class CircuitBreaker:
    """Open after ``failures`` consecutive failures; stay open for ``open_for`` seconds.

    ``check()`` raises :class:`BreakerOpenError` while open and closes the breaker (count reset)
    once the cool-down has elapsed; ``success()`` resets the count; ``failure()`` counts one more.
    ``clock`` is injectable for tests. Safe to share between threads.
    """

    failures: int = 3
    open_for: float = 60.0
    name: str = "endpoint"
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    _count: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.failures < 1:
            raise ValueError("CircuitBreaker.failures must be at least 1")
        if self.open_for < 0:
            raise ValueError("CircuitBreaker.open_for must not be negative")

    def check(self) -> None:
        """Raise :class:`BreakerOpenError` while open; close once ``open_for`` has elapsed."""
        with self._lock:
            if self._opened_at is None:
                return
            elapsed = self.clock() - self._opened_at
            if elapsed < self.open_for:
                raise BreakerOpenError(
                    f"{self.name} circuit breaker is open for another "
                    f"{self.open_for - elapsed:.0f}s after {self.failures} consecutive failures"
                )
            self._opened_at = None
            self._count = 0

    def success(self) -> None:
        with self._lock:
            self._count = 0

    def failure(self) -> None:
        with self._lock:
            self._count += 1
            if self._count >= self.failures:
                self._opened_at = self.clock()

    def reset(self) -> None:
        """Close the breaker and forget the failures (the doctor after a fix)."""
        with self._lock:
            self._count = 0
            self._opened_at = None

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and self.clock() - self._opened_at < self.open_for
