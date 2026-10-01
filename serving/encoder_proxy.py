#!/usr/bin/env python3
"""mesa-clm-encoder-proxy: the encoder's loopback endpoint, 127.0.0.1:8090 to its unix socket.

``mesa-clm-encoder-proxy.socket`` owns 127.0.0.1:8090 and starts this proxy on the first
connection (systemd hands over the listening socket as file descriptor 3); the proxy relays every
connection to the encoder container's API socket, ``~/.mesa/clm/run/encoder.sock`` (DESIGN A5).
It replaces ``systemd-socket-proxyd``, which connected to that path for each connection and so
followed whatever the path named. The container writes that directory (it creates the socket
there), so code running in the container could swap ``encoder.sock`` for a symlink to any unix
socket this account can reach (the session bus, the user manager), and every local account
connecting to 127.0.0.1:8090 would then talk to it with this account's credentials. Here, for
every connection:

* the client must be this account: loopback is reachable from every local account, so the proxy
  asks the kernel's socket diagnostics (``sock_diag(7)``, what ``ss -e`` reads; no privilege
  needed) who owns the client's end of the connection, one exact lookup of its address and port,
  and closes a connection of any other account, or one whose owner it cannot tell (the client
  already gone), at once: it takes no slot, reaches no encoder and holds nothing
  (``SO_PEERCRED`` answers only on unix sockets);
* the directory is opened with ``O_PATH | O_DIRECTORY | O_NOFOLLOW`` and must belong to this
  account with no group or other bits, then ``encoder.sock`` is opened relative to it with
  ``O_PATH | O_NOFOLLOW`` and must be a socket of this account: a symlink, a FIFO, a directory or
  another account's socket is refused and the client's connection closed;
* the proxy connects through ``/proc/self/fd/<fd>``, the inode it has just checked, so nothing
  can be swapped in between.

It also bounds what this account's own clients can hold: at most ``--connections-max`` relayed
connections at a time (each takes two file descriptors; the unit raises ``LimitNOFILE`` above
that), a connection over the limit is closed at once, and a connection on which neither side
sends a byte for ``--idle-timeout`` seconds is closed (longer than any client's request timeout:
a request that is still being answered is never cut). ``systemd-socket-proxyd`` used six
descriptors per connection under the user manager's soft limit of 1,024, so about 170 idle
connections from any account blocked the endpoint. A refusal is logged at most once per
:data:`LOG_EVERY_S` seconds for each reason, with the number of refusals since the last line, so
connecting in a loop cannot write the journal once per connection.

Standard library only; the unit runs it under the serve venv's Python in isolated mode
(``python -I``) in a systemd sandbox (no new privileges, a system-call allow list, unix and
netlink sockets only; the unit says why no namespace). It never reads a key: the bytes it
relays carry the bearer header the encoder's guard checks (DESIGN A4).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import logging
import os
import socket
import stat
import struct
import sys
import time
from collections.abc import Sequence
from pathlib import Path

LISTEN_FDS_START = 3  # sd_listen_fds(3)
CHUNK = 1 << 16
DEFAULT_CONNECTIONS_MAX = 4096
DEFAULT_IDLE_TIMEOUT_S = 900.0
# A unix socket whose backlog is full refuses a non-blocking connect with EAGAIN; retry a while.
CONNECT_TRIES = 50
CONNECT_PAUSE_S = 0.02
# At most one log line per reason in this many seconds (the refusals in between are counted).
LOG_EVERY_S = 10.0

# sock_diag(7): one SOCK_DIAG_BY_FAMILY request for the exact address pair of a TCP socket.
NETLINK_SOCK_DIAG = 4
SOCK_DIAG_BY_FAMILY = 20
NLM_F_REQUEST = 0x1
NLMSG_ERROR = 0x2
INET_DIAG_NOCOOKIE = 0xFFFFFFFF
_NLMSG = struct.Struct("=IHHII")  # nlmsghdr: length, type, flags, sequence, port id
_REQ = struct.Struct("=BBBBI")  # inet_diag_req_v2 head: family, protocol, ext, pad, states
_PORTS = struct.Struct("!HH")  # inet_diag_sockid: source and destination port, network order
_SOCKID_TAIL = struct.Struct("=III")  # interface, cookie (two words)
_MSG_LEN = 72  # inet_diag_msg: family, state, timer, retrans, the 48-byte sockid, five u32
_MSG_UID_INODE = struct.Struct("=II")  # idiag_uid, idiag_inode at offset 64 of inet_diag_msg
# tcp_info(7) of a listening socket: tcpi_state at 0 is TCP_LISTEN (10) and tcpi_sacked, at 28,
# the backlog the socket was given (what `ss -l` prints as Send-Q).
TCP_INFO = getattr(socket, "TCP_INFO", 11)
TCP_LISTEN = 10
_TCPI_SACKED = struct.Struct("=I")
_TCPI_SACKED_AT = 28
SOMAXCONN = Path("/proc/sys/net/core/somaxconn")
DEFAULT_BACKLOG = 4096

log = logging.getLogger("mesa-clm-encoder-proxy")


class TargetRefused(OSError):
    """The socket entry is not a socket of this account in an owner-only directory."""


class PeerUnknown(OSError):
    """The kernel cannot say which account owns the client's end of a connection."""


def open_target(path: Path, uid: int | None = None) -> int:
    """An ``O_PATH`` file descriptor of the socket ``path``, never through a symlink in its last
    two components (the module docstring's checks); the caller closes it."""
    me = os.getuid() if uid is None else uid
    dfd = os.open(path.parent, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        dst = os.fstat(dfd)
        if dst.st_uid != me or stat.S_IMODE(dst.st_mode) & 0o077:
            raise TargetRefused(f"{path.parent}: not a directory of uid {me} with mode 0700")
        fd = os.open(path.name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dfd)
    finally:
        os.close(dfd)
    st = os.fstat(fd)
    if not stat.S_ISSOCK(st.st_mode) or st.st_uid != me:
        os.close(fd)
        kind = "a symlink" if stat.S_ISLNK(st.st_mode) else f"not a socket of uid {me}"
        raise TargetRefused(f"{path}: {kind}")
    return fd


async def connect_target(path: Path, uid: int | None = None) -> socket.socket:
    """A connected, non-blocking unix stream socket to the checked inode of ``path``."""
    fd = open_target(path, uid)
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setblocking(False)
        try:
            for _ in range(CONNECT_TRIES):
                try:
                    sock.connect(f"/proc/self/fd/{fd}")
                    return sock
                except BlockingIOError:  # the encoder's backlog is full: wait, then retry
                    await asyncio.sleep(CONNECT_PAUSE_S)
            raise TimeoutError(f"{path}: the encoder's accept queue stayed full")
        except BaseException:
            sock.close()
            raise
    finally:
        os.close(fd)


# -- whose connection is it -------------------------------------------------------------------


def _address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """``host`` as an address, an IPv4-mapped IPv6 address as its IPv4 one (the client's own
    socket is IPv4 then)."""
    addr = ipaddress.ip_address(host.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def diag_request(local: tuple[str, int], peer: tuple[str, int], seq: int = 1) -> bytes:
    """The ``SOCK_DIAG_BY_FAMILY`` request for the client's socket of a connection: the socket
    whose own address is ``peer`` (the client, as ``getpeername()`` gives it) and whose remote
    address is ``local`` (the proxy's end, ``getsockname()``)."""
    src, dst = _address(peer[0]), _address(local[0])
    if src.version != dst.version:
        raise PeerUnknown(f"mixed address families: {peer[0]} and {local[0]}")
    family = socket.AF_INET if src.version == 4 else socket.AF_INET6
    sockid = (
        _PORTS.pack(peer[1], local[1])
        + src.packed.ljust(16, b"\0")
        + dst.packed.ljust(16, b"\0")
        + _SOCKID_TAIL.pack(0, INET_DIAG_NOCOOKIE, INET_DIAG_NOCOOKIE)
    )
    body = _REQ.pack(family, socket.IPPROTO_TCP, 0, 0, 0xFFFFFFFF) + sockid
    return _NLMSG.pack(_NLMSG.size + len(body), SOCK_DIAG_BY_FAMILY, NLM_F_REQUEST, seq, 0) + body


def _same_address(
    raw: bytes, family: int, addr: ipaddress.IPv4Address | ipaddress.IPv6Address
) -> bool:
    """Whether a sockid address field (16 bytes) of a socket of ``family`` is ``addr``: an IPv6
    socket connected over IPv4 (a dual-stack client) reports the IPv4-mapped form."""
    if family == socket.AF_INET:
        return addr.version == 4 and raw[:4] == addr.packed
    if addr.version == 4:
        return raw == b"\0" * 10 + b"\xff\xff" + addr.packed
    return raw == addr.packed


def diag_uid(reply: bytes, local: tuple[str, int], peer: tuple[str, int]) -> int:
    """The owner's uid in the kernel's reply to :func:`diag_request`. Raises
    :class:`PeerUnknown` for an error reply (``ENOENT``: no such socket), a reply about another
    socket, or a socket with no inode (closed: an orphan or ``TIME_WAIT``, whose uid the kernel
    does not keep)."""
    if len(reply) < _NLMSG.size:
        raise PeerUnknown("short sock_diag reply")
    _, kind, _, _, _ = _NLMSG.unpack_from(reply)
    if kind == NLMSG_ERROR:
        (err,) = struct.unpack_from("=i", reply, _NLMSG.size) if len(reply) >= 20 else (0,)
        raise PeerUnknown(-err, f"sock_diag: {os.strerror(-err) if err else 'error'}")
    msg = reply[_NLMSG.size :]
    if kind != SOCK_DIAG_BY_FAMILY or len(msg) < _MSG_LEN:
        raise PeerUnknown(f"unexpected sock_diag reply (type {kind}, {len(msg)} bytes)")
    family = msg[0]
    if (
        _PORTS.unpack_from(msg, 4) != (peer[1], local[1])
        or not _same_address(msg[8:24], family, _address(peer[0]))
        or not _same_address(msg[24:40], family, _address(local[0]))
    ):
        raise PeerUnknown("sock_diag answered about another socket")
    uid, inode = _MSG_UID_INODE.unpack_from(msg, 64)
    if inode == 0:
        raise PeerUnknown("the client's socket is closed")
    return int(uid)


def peer_uid(local: tuple[str, int], peer: tuple[str, int]) -> int:
    """The uid of the account that owns the client's socket of a loopback TCP connection
    (:func:`diag_request`, :func:`diag_uid`). The kernel answers while the request is sent, so
    the reply is read without waiting. Raises :class:`PeerUnknown` (an ``OSError``) when it
    cannot tell."""
    request = diag_request(local, peer)
    with socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, NETLINK_SOCK_DIAG) as nl:
        nl.setblocking(False)
        nl.sendto(request, (0, 0))
        try:
            reply = nl.recv(1 << 16)
        except BlockingIOError:
            raise PeerUnknown("no sock_diag reply") from None
    return diag_uid(reply, local, peer)


def client_uid(sock: socket.socket) -> int:
    """The uid owning the client's end of the accepted connection ``sock``."""
    local, peer = sock.getsockname(), sock.getpeername()
    return peer_uid((str(local[0]), int(local[1])), (str(peer[0]), int(peer[1])))


# -- the relay --------------------------------------------------------------------------------


class _Activity:
    """When a byte last crossed a connection, in either direction."""

    def __init__(self) -> None:
        self.last = time.monotonic()


async def _pump(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    idle: float,
    activity: _Activity,
) -> None:
    """Copy ``reader`` to ``writer`` until end of file, then half-close ``writer``. Raises
    :class:`TimeoutError` once no byte has crossed the connection, either way, for ``idle``
    seconds (a cancelled ``read`` leaves its data in the reader's buffer)."""
    while True:
        remaining = idle - (time.monotonic() - activity.last)
        if remaining <= 0:
            raise TimeoutError(f"no traffic for {idle:.0f} s")
        try:
            data = await asyncio.wait_for(reader.read(CHUNK), remaining)
        except TimeoutError:
            continue  # the other direction may have been busy meanwhile
        if not data:
            break
        activity.last = time.monotonic()
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


class _Throttled:
    """One kind of refusal: logged at most once per ``every`` seconds, the refusals in between
    counted and named in the next line."""

    def __init__(self, level: int, every: float = LOG_EVERY_S) -> None:
        self.level = level
        self.every = every
        self.last: float | None = None
        self.quiet = 0

    def __call__(self, msg: str, *args: object) -> None:
        now = time.monotonic()
        if self.last is not None and now - self.last < self.every:
            self.quiet += 1
            return
        self.last = now
        if self.quiet:
            msg += " (%d more since the last such line)"
            args = (*args, self.quiet)
            self.quiet = 0
        log.log(self.level, msg, *args)


class Proxy:
    """The relay: one instance per process, one :meth:`handle` per client connection."""

    def __init__(
        self,
        target: Path,
        *,
        connections_max: int,
        idle_timeout: float,
        uid: int | None = None,
    ) -> None:
        if connections_max < 1:
            raise ValueError("connections_max must be at least 1")
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self.target = target
        self.connections_max = connections_max
        self.idle_timeout = idle_timeout
        self.uid = os.getuid() if uid is None else uid
        self.active = 0
        self.refused = 0  # over the connection limit
        self.foreign = 0  # another account's, or an owner the kernel could not name
        self._log_full = _Throttled(logging.WARNING)
        self._log_foreign = _Throttled(logging.WARNING)
        self._log_target = _Throttled(logging.WARNING)
        self._log_down = _Throttled(logging.INFO)

    def _owner_refused(self, writer: asyncio.StreamWriter) -> bool:
        """Whether the connection is not this account's (logged, throttled)."""
        sock = writer.get_extra_info("socket")
        try:
            if sock is None:
                raise PeerUnknown("the transport has no socket")
            uid = client_uid(sock)
        except (OSError, ValueError) as exc:
            self.foreign += 1
            self._log_foreign("closing a connection whose owner is unknown: %s", exc)
            return True
        if uid != self.uid:
            self.foreign += 1
            self._log_foreign(
                "closing a connection of another account (uid %d; only uid %d is relayed)",
                uid,
                self.uid,
            )
            return True
        return False

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._owner_refused(writer):
            await _close(writer)
            return
        if self.active >= self.connections_max:
            self.refused += 1
            self._log_full(
                "connection limit %d reached; closing new connections (%d so far)",
                self.connections_max,
                self.refused,
            )
            await _close(writer)
            return
        self.active += 1
        try:
            await self._relay(reader, writer)
        finally:
            self.active -= 1

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            sock = await connect_target(self.target)
        except TargetRefused as exc:
            self._log_target("refused: %s", exc)
            await _close(writer)
            return
        except OSError as exc:  # no socket yet (the model is loading), or the encoder is down
            self._log_down("cannot reach %s: %s", self.target, exc.strerror or exc)
            await _close(writer)
            return
        up_reader, up_writer = await asyncio.open_unix_connection(sock=sock, limit=CHUNK)
        activity = _Activity()
        tasks = [
            asyncio.ensure_future(_pump(reader, up_writer, self.idle_timeout, activity)),
            asyncio.ensure_future(_pump(up_reader, writer, self.idle_timeout, activity)),
        ]
        try:
            # Both directions to their end of file, or the first error (a reset, the idle
            # timeout) closes both.
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for w in (writer, up_writer):
                w.close()
            await asyncio.gather(
                *(w.wait_closed() for w in (writer, up_writer)), return_exceptions=True
            )


def listen_socket() -> socket.socket:
    """The listening socket systemd passed (``LISTEN_PID``/``LISTEN_FDS``, exactly one)."""
    if os.environ.get("LISTEN_PID") != str(os.getpid()):
        raise SystemExit("mesa-clm-encoder-proxy: not socket-activated (no LISTEN_PID for us)")
    if os.environ.get("LISTEN_FDS") != "1":
        raise SystemExit("mesa-clm-encoder-proxy: expected exactly one listening socket")
    sock = socket.socket(fileno=LISTEN_FDS_START)
    sock.setblocking(False)
    return sock


def listen_backlog(sock: socket.socket | None = None, proc: Path = SOMAXCONN) -> int:
    """The backlog ``sock`` listens with (a listening TCP socket's ``tcp_info`` names it: the
    socket unit's ``Backlog=``, by default the kernel's ``net.core.somaxconn``); without one, the
    kernel's ``somaxconn`` from ``proc``, else 4096."""
    if sock is not None:
        with contextlib.suppress(OSError):
            info = sock.getsockopt(socket.IPPROTO_TCP, TCP_INFO, 104)
            if len(info) >= _TCPI_SACKED_AT + 4 and info[0] == TCP_LISTEN:
                (backlog,) = _TCPI_SACKED.unpack_from(info, _TCPI_SACKED_AT)
                if backlog > 0:
                    return int(backlog)
    try:
        return int(proc.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return DEFAULT_BACKLOG


async def serve(listener: socket.socket, proxy: Proxy) -> None:
    # asyncio calls listen() again on the socket it is given (backlog 100 by default; Python's
    # socket.SOMAXCONN is a compile-time 128): keep the socket unit's backlog instead.
    server = await asyncio.start_server(
        proxy.handle, sock=listener, limit=CHUNK, backlog=listen_backlog(listener)
    )
    async with server:
        await server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mesa-clm-encoder-proxy",
        description="Relay 127.0.0.1:8090 (socket-activated) to the encoder's unix socket.",
    )
    parser.add_argument(
        "socket", type=Path, help="the encoder's API socket (~/.mesa/clm/run/encoder.sock)"
    )
    parser.add_argument("--connections-max", type=int, default=DEFAULT_CONNECTIONS_MAX)
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT_S)
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    proxy = Proxy(args.socket, connections_max=args.connections_max, idle_timeout=args.idle_timeout)
    listener = listen_socket()
    for name in ("LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"):
        os.environ.pop(name, None)
    log.info(
        "relaying %s to %s for uid %d only (at most %d connections, idle timeout %.0f s, "
        "backlog %d)",
        listener.getsockname(),
        args.socket,
        proxy.uid,
        args.connections_max,
        args.idle_timeout,
        listen_backlog(listener),
    )
    asyncio.run(serve(listener, proxy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
