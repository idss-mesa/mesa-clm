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

* the directory is opened with ``O_PATH | O_DIRECTORY | O_NOFOLLOW`` and must belong to this
  account with no group or other bits, then ``encoder.sock`` is opened relative to it with
  ``O_PATH | O_NOFOLLOW`` and must be a socket of this account: a symlink, a FIFO, a directory or
  another account's socket is refused and the client's connection closed;
* the proxy connects through ``/proc/self/fd/<fd>``, the inode it has just checked, so nothing
  can be swapped in between.

It also bounds what one local account can hold: at most ``--connections-max`` relayed
connections at a time (each takes two file descriptors; the unit raises ``LimitNOFILE`` above
that), a connection over the limit is closed at once, and a connection on which neither side
sends a byte for ``--idle-timeout`` seconds is closed (longer than any client's request timeout:
a request that is still being answered is never cut). ``systemd-socket-proxyd`` used six
descriptors per connection under the user manager's soft limit of 1,024, so about 170 idle
connections from any account blocked the endpoint.

Standard library only; the unit runs it under the serve venv's Python in isolated mode
(``python -I``). It never reads a key: the bytes it relays carry the bearer header the encoder's
guard checks (DESIGN A4).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import socket
import stat
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

log = logging.getLogger("mesa-clm-encoder-proxy")


class TargetRefused(OSError):
    """The socket entry is not a socket of this account in an owner-only directory."""


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


class Proxy:
    """The relay: one instance per process, one :meth:`handle` per client connection."""

    def __init__(self, target: Path, *, connections_max: int, idle_timeout: float) -> None:
        if connections_max < 1:
            raise ValueError("connections_max must be at least 1")
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self.target = target
        self.connections_max = connections_max
        self.idle_timeout = idle_timeout
        self.active = 0
        self.refused = 0
        self._last_full_log = 0.0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.active >= self.connections_max:
            self.refused += 1
            now = time.monotonic()
            if now - self._last_full_log >= 10.0:
                self._last_full_log = now
                log.warning(
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
            log.warning("refused: %s", exc)
            await _close(writer)
            return
        except OSError as exc:  # no socket yet (the model is loading), or the encoder is down
            log.info("cannot reach %s: %s", self.target, exc.strerror or exc)
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


def listen_backlog(proc: Path = Path("/proc/sys/net/core/somaxconn")) -> int:
    """The kernel's ``net.core.somaxconn``: what the socket unit's default ``Backlog=`` comes to
    (4096 when it cannot be read)."""
    try:
        return int(proc.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return 4096


async def serve(listener: socket.socket, proxy: Proxy) -> None:
    # asyncio calls listen() again on the socket it is given (backlog 100 by default; Python's
    # socket.SOMAXCONN is a compile-time 128): keep the socket unit's backlog instead.
    server = await asyncio.start_server(
        proxy.handle, sock=listener, limit=CHUNK, backlog=listen_backlog()
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
        "relaying %s to %s (at most %d connections, idle timeout %.0f s)",
        listener.getsockname(),
        args.socket,
        args.connections_max,
        args.idle_timeout,
    )
    asyncio.run(serve(listener, proxy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
