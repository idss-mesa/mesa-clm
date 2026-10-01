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
free. A keyed client therefore asks twice (DESIGN A5). :func:`assert_listener_owner` is what it
asks before every request (:class:`mesa_clm.clm.http.HttpEndpoint`): it reads the listening
sockets of this network namespace from ``/proc/net/tcp`` and ``tcp6`` (:func:`proc_tcp_listeners`)
and refuses (:class:`ListenerOwnerError`) when the socket a connection to the URL's loopback
address and port would reach belongs to another account than this process's (root excepted: it
can read the key files anyway). That read and the connect are two steps, and another account can
bind the port between them, so the connection itself is checked as well:
:class:`OwnerCheckedTransport` checks every new connection to a loopback address once it is made
and before a byte is sent on it (:func:`assert_connection_owner`), asking the kernel's socket
diagnostics (``sock_diag(7)``, what ``ss -e`` reads; no privilege needed) which account holds the
server end of that very connection. The bearer key never goes to a port another account squats.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

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


# -- who holds the server end of a connection (DESIGN A5) ----------------------------------------
#
# The /proc read above and the connect that follows are two steps; another account can bind the
# port in between (check-then-connect). So every new connection to a loopback address is checked
# again after it is made and before anything is sent on it: one exact sock_diag(7) lookup of the
# server end of that connection, the same request serving/encoder_proxy.py makes for the client
# end. The server end of a connection not yet accepted already carries its listener's owner on
# this host's kernel (6.17: the listener's uid, inode 0); a socket still in the handshake or in
# TIME_WAIT is reported with uid 0 and no inode, as some kernels also report a socket no process
# has accepted yet, so such an answer is asked again until it names an owner.

NETLINK_SOCK_DIAG = 4
SOCK_DIAG_BY_FAMILY = 20
NLM_F_REQUEST = 0x1
NLMSG_ERROR = 0x2
INET_DIAG_NOCOOKIE = 0xFFFFFFFF
# include/net/tcp_states.h: the states the kernel reports with uid 0 and no inode.
TCP_SYN_RECV = 3
TCP_TIME_WAIT = 6
_NLMSG = struct.Struct("=IHHII")  # nlmsghdr: length, type, flags, sequence, port id
_DIAG_REQ = struct.Struct("=BBBBI")  # inet_diag_req_v2 head: family, protocol, ext, pad, states
_DIAG_PORTS = struct.Struct("!HH")  # inet_diag_sockid: source and destination port
_DIAG_TAIL = struct.Struct("=III")  # interface, cookie (two words)
_DIAG_MSG_LEN = 72  # inet_diag_msg: family, state, timer, retrans, the 48-byte sockid, five u32
_DIAG_UID_INODE = struct.Struct("=II")  # idiag_uid, idiag_inode at offset 64
# How long an undecided answer is asked again (the clients' connect timeout).
CONNECTION_OWNER_WAIT_S = 5.0
# httpcore's trace event once a TCP connection is made (before TLS and before any request byte).
CONNECT_TCP_COMPLETE = "connection.connect_tcp.complete"

Trace = Callable[[str, dict[str, Any]], Any]


class OwnerUnknown(OSError):
    """The kernel's answer names no socket of the connection (gone, or another socket)."""


@dataclass(frozen=True)
class SocketOwner:
    """What the kernel's socket diagnostics report for one TCP socket."""

    state: int
    uid: int
    inode: int

    @property
    def decisive(self) -> bool:
        """Whether ``uid`` names the owner: a socket in the handshake or in TIME_WAIT, and on
        some kernels one not yet accepted, is reported as uid 0 with no inode."""
        return self.state not in (TCP_SYN_RECV, TCP_TIME_WAIT) and (
            self.uid != 0 or self.inode != 0
        )


def _diag_address(host: str) -> IPAddress:
    """``host`` (a socket address, maybe with a zone) as an address, IPv4-mapped as IPv4."""
    return _v4(ipaddress.ip_address(host.split("%", 1)[0]))


def sock_diag_request(own: tuple[str, int], remote: tuple[str, int], seq: int = 1) -> bytes:
    """The ``SOCK_DIAG_BY_FAMILY`` request for the TCP socket whose own address is ``own`` and
    whose remote address is ``remote``."""
    src, dst = _diag_address(own[0]), _diag_address(remote[0])
    if src.version != dst.version:
        raise OwnerUnknown(f"mixed address families: {own[0]} and {remote[0]}")
    family = socket.AF_INET if src.version == 4 else socket.AF_INET6
    sockid = (
        _DIAG_PORTS.pack(own[1], remote[1])
        + src.packed.ljust(16, b"\0")
        + dst.packed.ljust(16, b"\0")
        + _DIAG_TAIL.pack(0, INET_DIAG_NOCOOKIE, INET_DIAG_NOCOOKIE)
    )
    body = _DIAG_REQ.pack(family, socket.IPPROTO_TCP, 0, 0, 0xFFFFFFFF) + sockid
    return _NLMSG.pack(_NLMSG.size + len(body), SOCK_DIAG_BY_FAMILY, NLM_F_REQUEST, seq, 0) + body


def _same_address(raw: bytes, family: int, addr: IPAddress) -> bool:
    """Whether a sockid address field (16 bytes) of a socket of ``family`` is ``addr``: an IPv6
    socket on an IPv4 connection (a dual-stack listener's) reports the IPv4-mapped form."""
    if family == socket.AF_INET:
        return addr.version == 4 and raw[:4] == addr.packed
    if addr.version == 4:
        return raw == b"\0" * 10 + b"\xff\xff" + addr.packed
    return raw == addr.packed


def sock_diag_owner(reply: bytes, own: tuple[str, int], remote: tuple[str, int]) -> SocketOwner:
    """The kernel's answer to :func:`sock_diag_request`. Raises :class:`OwnerUnknown` for an
    error reply (``ENOENT``: no such socket) and for an answer about another socket (with no
    connection of that address pair the lookup falls back to a listener)."""
    if len(reply) < _NLMSG.size:
        raise OwnerUnknown("short sock_diag reply")
    _, kind, _, _, _ = _NLMSG.unpack_from(reply)
    if kind == NLMSG_ERROR:
        (err,) = struct.unpack_from("=i", reply, _NLMSG.size) if len(reply) >= 20 else (0,)
        raise OwnerUnknown(-err, f"sock_diag: {os.strerror(-err) if err else 'error'}")
    msg = reply[_NLMSG.size :]
    if kind != SOCK_DIAG_BY_FAMILY or len(msg) < _DIAG_MSG_LEN:
        raise OwnerUnknown(f"unexpected sock_diag reply (type {kind}, {len(msg)} bytes)")
    family = msg[0]
    if (
        _DIAG_PORTS.unpack_from(msg, 4) != (own[1], remote[1])
        or not _same_address(msg[8:24], family, _diag_address(own[0]))
        or not _same_address(msg[24:40], family, _diag_address(remote[0]))
    ):
        raise OwnerUnknown("sock_diag answered about another socket")
    uid, inode = _DIAG_UID_INODE.unpack_from(msg, 64)
    return SocketOwner(state=msg[1], uid=int(uid), inode=int(inode))


def tcp_socket_owner(own: tuple[str, int], remote: tuple[str, int]) -> SocketOwner:
    """What the kernel reports for the TCP socket whose own address is ``own`` and whose remote
    address is ``remote`` (one ``sock_diag`` request; the kernel answers while it is sent).
    Raises :class:`OwnerUnknown`, or another ``OSError`` when netlink cannot be used."""
    request = sock_diag_request(own, remote)
    with socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, NETLINK_SOCK_DIAG) as nl:
        nl.settimeout(1.0)
        nl.sendto(request, (0, 0))
        try:
            reply = nl.recv(1 << 16)
        except TimeoutError:
            raise OwnerUnknown("no sock_diag reply") from None
    return sock_diag_owner(reply, own, remote)


def connection_owner(sock: socket.socket) -> SocketOwner:
    """What the kernel reports for the server end of the connected TCP socket ``sock``: the
    socket whose own address is ``sock``'s peer and whose remote address is ``sock``'s own."""
    own, peer = sock.getsockname(), sock.getpeername()
    return tcp_socket_owner((str(peer[0]), int(peer[1])), (str(own[0]), int(own[1])))


def assert_connection_owner(
    sock: socket.socket,
    *,
    what: str,
    wait: float = CONNECTION_OWNER_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Raise :class:`ListenerOwnerError` unless the server end of the connected TCP socket
    ``sock`` belongs to this account (root excepted, as in :func:`assert_listener_owner`): the
    check of the connection a key would travel on, made after the connect and before anything is
    sent (module docstring). A peer that is not a loopback address passes, and so does a host
    without netlink (not Linux, as an unreadable ``/proc`` does there). An answer that names no
    owner is asked again for ``wait`` seconds and then refused: nothing is sent unchecked."""
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    try:
        peer = sock.getpeername()
    except OSError:
        raise ListenerOwnerError(
            f"{what}: the connection closed before its owner was checked; the key is not sent"
        ) from None
    host, port = str(peer[0]), int(peer[1])
    if not _diag_address(host).is_loopback or not hasattr(socket, "AF_NETLINK"):
        return
    where = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    me = os.getuid()
    deadline = time.monotonic() + wait
    pause = 0.0005
    while True:
        try:
            answer = connection_owner(sock)
        except OwnerUnknown as exc:
            reason = str(exc)
        except OSError as exc:  # netlink refused (a sandbox): nothing can be checked
            raise ListenerOwnerError(
                f"{what}: cannot ask the kernel which account holds the server end of the "
                f"connection to {where} ({exc}); the key is not sent"
            ) from None
        else:
            if answer.decisive:
                if answer.uid in (me, 0):
                    return
                raise ListenerOwnerError(
                    f"{what}: the connection to {where} reached a socket of uid {answer.uid}, not "
                    f"of this account (uid {me}); the key is not sent there. Another local "
                    "account took the port: start the units (docs/deploy/serving.md) or find the "
                    "listener with `ss -ltnpe` (DESIGN A5)"
                )
            reason = f"state {answer.state}, no owner reported yet"
        if time.monotonic() >= deadline:
            raise ListenerOwnerError(
                f"{what}: the kernel named no owner for the server end of the connection to "
                f"{where} within {wait:.0f} s ({reason}); the key is not sent"
            )
        sleep(pause)
        pause = min(pause * 2, 0.05)


def owner_trace(what: str, chained: Trace | None = None) -> Trace:
    """An ``httpcore`` trace callback (the ``trace`` request extension) that runs
    :func:`assert_connection_owner` on each new TCP connection as soon as it is made, before TLS
    and before the request is written, and closes the connection when it refuses. ``chained``
    is a caller's own trace callback, called first."""

    def trace(event: str, info: dict[str, Any]) -> None:
        if chained is not None:
            chained(event, info)
        if event != CONNECT_TCP_COMPLETE:
            return
        stream = info.get("return_value")
        try:
            sock = stream.get_extra_info("socket") if stream is not None else None
            if not isinstance(sock, socket.socket):
                raise ListenerOwnerError(
                    f"{what}: the new connection has no socket to check; the key is not sent"
                )
            assert_connection_owner(sock, what=what)
        except BaseException:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
            raise

    return trace


class OwnerCheckedTransport(httpx.BaseTransport):
    """An ``httpx`` transport for a keyed client: every new connection to a loopback address is
    checked by :func:`assert_connection_owner` once it is made and before a byte is sent on it
    (:func:`owner_trace`), so the key never travels on a connection another account's socket
    answered, however the port changed hands after :func:`assert_listener_owner` read ``/proc``.
    A refusal is a :class:`ListenerOwnerError` raised from the request. ``inner`` defaults to
    ``httpx.HTTPTransport(trust_env=False)``, what ``httpx.Client(trust_env=False)`` builds."""

    def __init__(self, what: str, inner: httpx.BaseTransport | None = None) -> None:
        self.what = what
        self._inner = inner if inner is not None else httpx.HTTPTransport(trust_env=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        chained = request.extensions.get("trace")
        request.extensions = {**request.extensions, "trace": owner_trace(self.what, chained)}
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


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
