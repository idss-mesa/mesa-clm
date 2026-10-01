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

Loopback is shared by every account on a multi-user host, and a port nobody holds can be taken
by anyone: the serving units are not enabled at boot, so 127.0.0.1:8090 and :8700 are usually
free. :func:`assert_listener_owner` is what a keyed client asks before every request
(:class:`mesa_clm.clm.http.HttpEndpoint`): it reads the listening sockets of this network
namespace from ``/proc/net/tcp`` and ``tcp6`` (:func:`proc_tcp_listeners`) and refuses
(:class:`ListenerOwnerError`) when the socket a connection to the URL's loopback address and port
would reach belongs to another account than this process's (root excepted: it can read the key
files anyway), so the bearer key never goes to a port another account squats (DESIGN A5).
"""

from __future__ import annotations

import ipaddress
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_SCHEMES = frozenset({"http", "https"})
# /proc/net/tcp{,6} ``st`` column: 0A is TCP_LISTEN.
TCP_LISTEN = "0A"
# Where the listening sockets are read (tests point it at a fixture directory).
PROC_NET = Path("/proc/net")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class EndpointError(RuntimeError):
    """A base URL mesa-clm refuses to call, or an endpoint the breaker has taken offline."""


class BreakerOpenError(EndpointError):
    """The circuit breaker is open: recent calls failed and the cool-down has not elapsed."""


class ListenerOwnerError(EndpointError):
    """The loopback port of a keyed endpoint is held by another account's socket: the key is
    not sent (:func:`assert_listener_owner`)."""


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


# -- who holds a loopback port (DESIGN A5) ---------------------------------------------------------


def proc_address(hex_addr: str) -> IPAddress:
    """A ``/proc/net/tcp{,6}`` address: little-endian 32-bit words, as the kernel prints them."""
    raw = bytes.fromhex(hex_addr)
    if len(raw) == 4:
        return ipaddress.IPv4Address(raw[::-1])
    words = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4))
    return ipaddress.IPv6Address(words)


def proc_tcp_listeners(text: str) -> list[tuple[IPAddress, int, int]]:
    """``(address, port, uid)`` of every listening socket in one ``/proc/net/tcp`` or ``tcp6``
    file (the uid is the socket's owner as this process's user namespace sees it)."""
    out: list[tuple[IPAddress, int, int]] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8 or parts[3] != TCP_LISTEN:
            continue
        host, _, port = parts[1].partition(":")
        try:
            out.append((proc_address(host), int(port, 16), int(parts[7])))
        except ValueError:
            continue
    return out


def _v4(addr: IPAddress) -> IPAddress:
    """An IPv4-mapped IPv6 address as its IPv4 address; anything else unchanged."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _wildcard_for(addr: IPAddress, target: IPAddress) -> bool:
    """Whether a socket listening on the wildcard ``addr`` accepts a connection to ``target``:
    ``0.0.0.0`` (or its v4-mapped form) IPv4 only, ``::`` IPv6 and, unless it is v6-only (which
    ``/proc`` does not show), IPv4 as well."""
    if not _v4(addr).is_unspecified:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is None:
        return True
    return _v4(addr).version == target.version


def listener_owners(
    host: str, port: int, *, proc_net: Path | None = None
) -> list[tuple[str, int]] | None:
    """``(address:port, uid)`` of the listening sockets in this network namespace that a
    connection to the loopback ``host`` and ``port`` would reach: a socket bound to that exact
    address if there is one (the kernel prefers it), else a wildcard one (``0.0.0.0``, or ``::``,
    which accepts IPv4 too unless it is v6-only, which ``/proc`` does not show). ``localhost``
    is checked as both ``127.0.0.1`` and ``::1``. ``None`` when ``/proc/net/tcp`` cannot be read
    (not Linux); an empty list when nothing listens there."""
    base = proc_net if proc_net is not None else PROC_NET
    listeners: list[tuple[IPAddress, int, int]] = []
    readable = False
    for name in ("tcp", "tcp6"):
        try:
            text = (base / name).read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        readable = True
        listeners.extend(proc_tcp_listeners(text))
    if not readable:
        return None
    names = ("127.0.0.1", "::1") if host.lower() == "localhost" else (host,)
    out: list[tuple[str, int]] = []
    for name in names:
        target = _v4(ipaddress.ip_address(name))
        on_port = [(addr, uid) for addr, p, uid in listeners if p == port]
        exact = [(a, uid) for a, uid in on_port if _v4(a) == target]
        wildcard = [(a, uid) for a, uid in on_port if _wildcard_for(a, target)]
        for addr, uid in exact or wildcard:
            shown = f"[{addr}]" if addr.version == 6 else str(addr)
            out.append((f"{shown}:{port}", uid))
    return out


def assert_listener_owner(base_url: str, *, what: str, proc_net: Path | None = None) -> None:
    """Raise :class:`ListenerOwnerError` when the listening socket a request to ``base_url``
    would reach is owned by another account than this process's (root excepted), so a bearer
    key is never sent to a port another local account holds (module docstring). A URL that is
    not loopback, an unreadable ``/proc`` and a port nobody listens on pass (the request then
    fails on its own)."""
    try:
        parts = urlsplit(base_url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return
    if not is_loopback_host(host):
        return
    owners = listener_owners(host, port, proc_net=proc_net)
    if not owners:
        return
    me = os.getuid()
    foreign = [(where, uid) for where, uid in owners if uid not in (me, 0)]
    if foreign:
        where, uid = foreign[0]
        raise ListenerOwnerError(
            f"{what}: {where} is held by a socket of uid {uid}, not of this account (uid {me}); "
            "the key is not sent there. Another local account may hold the port while the "
            "serving units are down: start the units (docs/deploy/serving.md) or find the "
            "listener with `ss -ltnpe` (DESIGN A5)"
        )


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
