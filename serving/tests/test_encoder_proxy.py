"""``serving/encoder_proxy.py``: the encoder's loopback endpoint (DESIGN A5). It relays a TCP
connection to the encoder's unix socket only when that entry is a socket of this account in an
owner-only directory, never through a symlink (the container writes that directory), and bounds
the connections one account can hold. Everything runs on loopback and in ``tmp_path``."""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
import os
import socket
import stat
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import encoder_proxy as ep
import pytest

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """The stand-in encoder: answers ``enc:`` and then echoes until end of file, then closes."""
    prefix = b"enc:"
    while data := await reader.read(65536):
        writer.write(prefix + data)
        prefix = b""
        await writer.drain()
    writer.close()


async def _other(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Another unix socket of the same account (the session bus, the user manager...)."""
    writer.write(b"OTHER SOCKET")
    await writer.drain()
    writer.close()


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    run.chmod(0o700)
    return run


async def _unix_server(path: Path, handler: Handler) -> asyncio.Server:
    return await asyncio.start_unix_server(handler, path=str(path))


@pytest.fixture
async def encoder(tmp_path: Path) -> AsyncIterator[Path]:
    """``<tmp>/run/encoder.sock`` served by :func:`_echo`."""
    sock = _run_dir(tmp_path) / "encoder.sock"
    server = await _unix_server(sock, _echo)
    try:
        yield sock
    finally:
        server.close()
        await server.wait_closed()


async def _start_proxy(target: Path, **kw: float) -> tuple[ep.Proxy, asyncio.Task[None], int]:
    """The proxy on an ephemeral loopback port, as the socket unit would hand it over."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    proxy = ep.Proxy(
        target,
        connections_max=int(kw.get("connections_max", 8)),
        idle_timeout=float(kw.get("idle_timeout", 30.0)),
    )
    task = asyncio.ensure_future(ep.serve(listener, proxy))
    await asyncio.sleep(0)
    return proxy, task, port


async def _stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _exchange(port: int, payload: bytes) -> bytes:
    """Send ``payload``, half-close, read everything back; a connection the proxy closed
    unread (a reset) gave back nothing."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        return await asyncio.wait_for(reader.read(), 5)
    except (ConnectionResetError, BrokenPipeError):
        return b""
    finally:
        writer.close()


# -- the target checks ------------------------------------------------------------------------


async def test_a_socket_of_this_account_is_reached(encoder: Path) -> None:
    fd = ep.open_target(encoder)
    try:
        assert stat.S_ISSOCK(os.fstat(fd).st_mode)
    finally:
        os.close(fd)
    sock = await ep.connect_target(encoder)
    reader, writer = await asyncio.open_unix_connection(sock=sock)
    writer.write(b"ping")
    writer.write_eof()
    assert await reader.read() == b"enc:ping"
    writer.close()


async def test_a_symlink_to_another_socket_is_refused(tmp_path: Path, encoder: Path) -> None:
    """The DESIGN A5 deputy: code in the container replaces encoder.sock with a symlink to a
    socket of the account (here a stand-in); systemd-socket-proxyd followed it."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    bus = elsewhere / "bus"
    server = await _unix_server(bus, _other)
    try:
        encoder.unlink()
        encoder.symlink_to(bus)
        with pytest.raises(ep.TargetRefused, match="a symlink"):
            ep.open_target(encoder)
        with pytest.raises(ep.TargetRefused):
            await ep.connect_target(encoder)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("kind", ["file", "fifo", "directory"])
def test_anything_but_a_socket_is_refused(tmp_path: Path, kind: str) -> None:
    entry = _run_dir(tmp_path) / "encoder.sock"
    if kind == "file":
        entry.write_text("x")
    elif kind == "fifo":
        os.mkfifo(entry)
    else:
        entry.mkdir()
    with pytest.raises(ep.TargetRefused, match="not a socket"):
        ep.open_target(entry)


async def test_another_owner_or_a_loose_directory_is_refused(encoder: Path) -> None:
    with pytest.raises(ep.TargetRefused, match="uid"):
        ep.open_target(encoder, uid=os.getuid() + 1)
    encoder.parent.chmod(0o755)
    try:
        with pytest.raises(ep.TargetRefused, match="mode 0700"):
            ep.open_target(encoder)
    finally:
        encoder.parent.chmod(0o700)
    link = encoder.parent.parent / "run-link"
    link.symlink_to(encoder.parent)
    with pytest.raises(OSError):  # O_NOFOLLOW on the directory as well
        ep.open_target(link / "encoder.sock")


def test_a_missing_socket_is_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ep.open_target(_run_dir(tmp_path) / "encoder.sock")


# -- the relay --------------------------------------------------------------------------------


async def test_the_proxy_relays_both_ways_and_half_closes(encoder: Path) -> None:
    proxy, task, port = await _start_proxy(encoder)
    try:
        assert await _exchange(port, b"GET /health") == b"enc:GET /health"
        big = os.urandom(1 << 20)  # many chunks each way
        assert await _exchange(port, big) == b"enc:" + big
        await asyncio.sleep(0.05)
        assert proxy.active == 0
    finally:
        await _stop(task)


async def test_the_proxy_closes_a_connection_to_a_swapped_socket(
    tmp_path: Path, encoder: Path
) -> None:
    other = tmp_path / "other.sock"
    server = await _unix_server(other, _other)
    _, task, port = await _start_proxy(encoder)
    try:
        encoder.unlink()
        encoder.symlink_to(other)
        assert await _exchange(port, b"POST /org.freedesktop.systemd1") == b""
    finally:
        await _stop(task)
        server.close()
        await server.wait_closed()


async def test_no_socket_yet_closes_the_connection(tmp_path: Path) -> None:
    target = _run_dir(tmp_path) / "encoder.sock"
    _, task, port = await _start_proxy(target)
    try:
        assert await _exchange(port, b"GET /health") == b""
    finally:
        await _stop(task)


async def test_the_connection_limit(encoder: Path) -> None:
    proxy, task, port = await _start_proxy(encoder, connections_max=1)
    try:
        held_reader, held_writer = await asyncio.open_connection("127.0.0.1", port)
        held_writer.write(b"a")
        await held_writer.drain()
        assert await asyncio.wait_for(held_reader.read(5), 5) == b"enc:a"
        assert proxy.active == 1
        assert await _exchange(port, b"b") == b""  # over the limit: closed at once
        assert proxy.refused == 1
        held_writer.close()
        await asyncio.sleep(0.05)
        assert await _exchange(port, b"c") == b"enc:c"
    finally:
        await _stop(task)


async def test_an_idle_connection_is_closed(encoder: Path) -> None:
    proxy, task, port = await _start_proxy(encoder, idle_timeout=0.2)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"x")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(5), 5) == b"enc:x"
        assert await asyncio.wait_for(reader.read(), 5) == b""  # closed after 0.2 s of silence
        writer.close()
        await asyncio.sleep(0.05)
        assert proxy.active == 0
    finally:
        await _stop(task)


def test_the_proxy_refuses_bad_limits_and_a_missing_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="connections_max"):
        ep.Proxy(Path("/x"), connections_max=0, idle_timeout=1.0)
    with pytest.raises(ValueError, match="idle_timeout"):
        ep.Proxy(Path("/x"), connections_max=1, idle_timeout=0)
    monkeypatch.delenv("LISTEN_PID", raising=False)
    with pytest.raises(SystemExit, match="not socket-activated"):
        ep.listen_socket()
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", "2")
    with pytest.raises(SystemExit, match="exactly one"):
        ep.listen_socket()


def test_the_listen_backlog_is_the_kernels(tmp_path: Path) -> None:
    """asyncio re-listens on the socket it is given; the proxy keeps the socket unit's backlog
    (systemd's default Backlog= comes to net.core.somaxconn) rather than asyncio's 100."""
    knob = tmp_path / "somaxconn"
    knob.write_text("4096\n", encoding="ascii")
    assert ep.listen_backlog(knob) == 4096
    assert ep.listen_backlog(tmp_path / "missing") == 4096
    knob.write_text("x", encoding="ascii")
    assert ep.listen_backlog(knob) == 4096
