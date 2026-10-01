"""``serving/encoder_proxy.py``: the encoder's loopback endpoint (DESIGN A5). It relays a TCP
connection only when the client's socket belongs to this account (the kernel's socket
diagnostics name its owner), and only to the encoder's unix socket when that entry is a socket
of this account in an owner-only directory, never through a symlink (the container writes that
directory); it bounds the connections this account can hold and throttles its refusal logs.
Everything runs on loopback and in ``tmp_path``."""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
import logging
import os
import socket
import stat
import struct
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import encoder_proxy as ep
import pytest

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
ME = os.getuid()


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


async def _late(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """An encoder that reads the whole request to end of file and answers after a pause."""
    data = await reader.read()
    await asyncio.sleep(0.2)
    writer.write(b"late:" + data)
    await writer.drain()
    writer.close()


async def _stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """After the first byte, ten bytes 0.1 s apart (a slow answer), then closes."""
    await reader.read(1)
    for _ in range(10):
        writer.write(b".")
        await writer.drain()
        await asyncio.sleep(0.1)
    writer.close()


class _Counting:
    """A stand-in encoder that counts the connections it is given and echoes."""

    def __init__(self) -> None:
        self.connections = 0

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        await _echo(reader, writer)


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


async def _start_proxy(
    target: Path, *, backlog: int = 16, uid: int | None = None, **kw: float
) -> tuple[ep.Proxy, asyncio.Task[None], int]:
    """The proxy on an ephemeral loopback port, as the socket unit would hand it over."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(backlog)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    proxy = ep.Proxy(
        target,
        connections_max=int(kw.get("connections_max", 8)),
        idle_timeout=float(kw.get("idle_timeout", 30.0)),
        uid=uid,
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


async def test_a_swap_after_the_check_does_not_redirect(
    tmp_path: Path, encoder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proxy connects through ``/proc/self/fd/<fd>``, the inode it checked: the container
    swapping the entry for a symlink between the check and the connect changes nothing (a
    connect by path, systemd-socket-proxyd's, would reach the other socket)."""
    other = tmp_path / "other.sock"
    other_server = await _unix_server(other, _other)
    checked = ep.open_target

    def racing(path: Path, uid: int | None = None) -> int:
        fd = checked(path, uid)
        path.rename(path.with_name("moved.sock"))  # the checked inode lives on elsewhere
        path.symlink_to(other)  # and the entry now names another socket of the account
        return fd

    monkeypatch.setattr(ep, "open_target", racing)
    try:
        sock = await ep.connect_target(encoder)
        assert encoder.is_symlink()
        reader, writer = await asyncio.open_unix_connection(sock=sock)
        writer.write(b"ping")
        writer.write_eof()
        assert await asyncio.wait_for(reader.read(), 5) == b"enc:ping"
        writer.close()
    finally:
        other_server.close()
        await other_server.wait_closed()


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
    with pytest.raises(ep.TargetRefused, match="not a directory of uid"):
        ep.open_target(encoder, uid=ME + 1)
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


async def test_a_socket_of_another_account_is_refused(
    encoder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The socket's own owner is checked, apart from the directory's (another account cannot
    be made to own a file here, so ``fstat`` reports it)."""
    real = os.fstat

    def foreign_socket(fd: int) -> os.stat_result:
        st = real(fd)
        if not stat.S_ISSOCK(st.st_mode):
            return st
        fields = list(st)
        fields[stat.ST_UID] = st.st_uid + 1
        return os.stat_result(fields)

    monkeypatch.setattr("encoder_proxy.os.fstat", foreign_socket)
    with pytest.raises(ep.TargetRefused, match=f"not a socket of uid {ME}"):
        ep.open_target(encoder)


def test_a_missing_socket_is_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ep.open_target(_run_dir(tmp_path) / "encoder.sock")


# -- whose connection ---------------------------------------------------------------------------


def test_the_kernel_names_the_owner_of_a_client_socket() -> None:
    """``sock_diag`` answers for an accepted loopback connection: this account's client, an
    IPv6 one and a dual-stack client on IPv4 are this uid; a client that has closed has no
    owner any more."""
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        port = srv.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        dual = socket.socket(socket.AF_INET6)
        dual.connect(("::ffff:127.0.0.1", port))
        dual_conn, _ = srv.accept()
        try:
            assert ep.client_uid(conn) == ME
            assert ep.client_uid(dual_conn) == ME
            client.close()
            time.sleep(0.05)
            with pytest.raises(ep.PeerUnknown, match="closed"):
                ep.client_uid(conn)
        finally:
            for s in (client, conn, dual, dual_conn):
                s.close()
    try:
        srv6 = socket.socket(socket.AF_INET6)
        srv6.bind(("::1", 0))
    except OSError:  # no IPv6 loopback here
        return
    with srv6:
        srv6.listen(1)
        with socket.create_connection(("::1", srv6.getsockname()[1])):
            conn6, _ = srv6.accept()
            with conn6:
                assert ep.client_uid(conn6) == ME


def _reply(kind: int, body: bytes) -> bytes:
    return ep._NLMSG.pack(16 + len(body), kind, 0, 1, 0) + body


def _msg(sport: int, dport: int, uid: int, inode: int, src: str = "127.0.0.1") -> bytes:
    head = bytes([socket.AF_INET, 1, 0, 0]) + struct.pack("!HH", sport, dport)
    addrs = socket.inet_aton(src).ljust(16, b"\0") + socket.inet_aton("127.0.0.1").ljust(16, b"\0")
    return head + addrs + struct.pack("=III", 0, 0, 0) + struct.pack("=IIIII", 0, 0, 0, uid, inode)


def test_a_sock_diag_reply_is_read_strictly() -> None:
    local, peer = ("127.0.0.1", 8090), ("127.0.0.1", 50000)
    request = ep.diag_request(local, peer)
    assert len(request) == 72 and struct.unpack_from("=I", request)[0] == 72
    assert struct.unpack_from("!HH", request, 24) == (50000, 8090)  # the client's own socket
    assert (
        ep.diag_uid(_reply(ep.SOCK_DIAG_BY_FAMILY, _msg(50000, 8090, 4242, 7)), local, peer) == 4242
    )
    with pytest.raises(ep.PeerUnknown, match="closed"):  # TIME_WAIT, an orphan: no inode, uid 0
        ep.diag_uid(_reply(ep.SOCK_DIAG_BY_FAMILY, _msg(50000, 8090, 0, 0)), local, peer)
    with pytest.raises(ep.PeerUnknown, match="another socket"):
        ep.diag_uid(_reply(ep.SOCK_DIAG_BY_FAMILY, _msg(50001, 8090, ME, 7)), local, peer)
    with pytest.raises(ep.PeerUnknown, match="another socket"):
        ep.diag_uid(
            _reply(ep.SOCK_DIAG_BY_FAMILY, _msg(50000, 8090, ME, 7, "127.0.0.2")), local, peer
        )
    with pytest.raises(ep.PeerUnknown, match="No such file"):
        ep.diag_uid(_reply(ep.NLMSG_ERROR, struct.pack("=i", -2) + request), local, peer)
    with pytest.raises(ep.PeerUnknown, match="unexpected"):
        ep.diag_uid(_reply(ep.SOCK_DIAG_BY_FAMILY, b"\0" * 8), local, peer)
    with pytest.raises(ep.PeerUnknown, match="short"):
        ep.diag_uid(b"\0" * 4, local, peer)


async def test_another_accounts_connection_is_closed_unrelayed(tmp_path: Path) -> None:
    """127.0.0.1:8090 is reachable from every local account: a connection whose client socket
    is not this account's is closed at once, never reaches the encoder and holds no slot (a
    proxy that relays only uid ``ME + 1`` stands in for another account's client)."""
    counting = _Counting()
    sock = _run_dir(tmp_path) / "encoder.sock"
    server = await _unix_server(sock, counting)
    proxy, task, port = await _start_proxy(sock, uid=ME + 1, connections_max=1)
    try:
        for _ in range(3):
            assert await _exchange(port, b"GET /health") == b""
        await asyncio.sleep(0.05)
        assert counting.connections == 0
        assert proxy.foreign == 3 and proxy.refused == 0 and proxy.active == 0
    finally:
        await _stop(task)
        server.close()
        await server.wait_closed()


async def test_an_owner_the_kernel_cannot_name_is_refused(
    encoder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unknown(sock: socket.socket) -> int:
        raise ep.PeerUnknown("sock_diag: No such file or directory")

    monkeypatch.setattr(ep, "client_uid", unknown)
    proxy, task, port = await _start_proxy(encoder)
    try:
        assert await _exchange(port, b"GET /health") == b""
        assert proxy.foreign == 1 and proxy.active == 0
    finally:
        await _stop(task)


# -- the relay --------------------------------------------------------------------------------


async def test_the_proxy_relays_both_ways_and_half_closes(encoder: Path) -> None:
    proxy, task, port = await _start_proxy(encoder)
    try:
        assert await _exchange(port, b"GET /health") == b"enc:GET /health"
        big = os.urandom(1 << 20)  # many chunks each way
        assert await _exchange(port, big) == b"enc:" + big
        await asyncio.sleep(0.05)
        assert proxy.active == 0 and proxy.foreign == 0
    finally:
        await _stop(task)


async def test_a_half_closed_client_still_gets_a_late_answer(tmp_path: Path) -> None:
    """The client's end of file ends one direction only: the answer that comes after it still
    reaches the client."""
    sock = _run_dir(tmp_path) / "encoder.sock"
    server = await _unix_server(sock, _late)
    _, task, port = await _start_proxy(sock)
    try:
        assert await _exchange(port, b"GET /health") == b"late:GET /health"
    finally:
        await _stop(task)
        server.close()
        await server.wait_closed()


async def test_traffic_one_way_keeps_the_connection(tmp_path: Path) -> None:
    """The idle timer counts bytes in either direction: a slow answer streamed for longer than
    the timeout, while the client sends nothing more, is not cut."""
    sock = _run_dir(tmp_path) / "encoder.sock"
    server = await _unix_server(sock, _stream)
    _, task, port = await _start_proxy(sock, idle_timeout=0.3)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"x")  # then silent, without a half-close (an HTTP client waiting)
        await writer.drain()
        got = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        assert got == b"." * 10
    finally:
        await _stop(task)
        server.close()
        await server.wait_closed()


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


async def test_refusals_are_logged_once_per_interval(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Connecting in a loop while the encoder is down (or from another account) must not write
    one journal line per connection: one line per reason and interval, and the next line counts
    what was left out."""
    caplog.set_level(logging.INFO, logger=ep.log.name)
    target = _run_dir(tmp_path) / "encoder.sock"  # no socket: the model is loading
    proxy, task, port = await _start_proxy(target)
    foreign, foreign_task, foreign_port = await _start_proxy(target, uid=ME + 1)
    try:
        for _ in range(5):
            assert await _exchange(port, b"GET /health") == b""
            assert await _exchange(foreign_port, b"GET /health") == b""
        down = [r.getMessage() for r in caplog.records if "cannot reach" in r.getMessage()]
        other = [r.getMessage() for r in caplog.records if "another account" in r.getMessage()]
        assert len(down) == 1 and len(other) == 1, (down, other)
        assert foreign.foreign == 5
        proxy._log_down.last = (proxy._log_down.last or 0.0) - 2 * ep.LOG_EVERY_S
        assert await _exchange(port, b"GET /health") == b""
        down = [r.getMessage() for r in caplog.records if "cannot reach" in r.getMessage()]
        assert len(down) == 2 and down[1].endswith("(4 more since the last such line)")
    finally:
        await _stop(task)
        await _stop(foreign_task)


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


def _backlog_of(sock: socket.socket) -> int:
    """The backlog a listening TCP socket has (``tcp_info``; what ``ss -l`` shows as Send-Q)."""
    info = sock.getsockopt(socket.IPPROTO_TCP, ep.TCP_INFO, 104)
    assert info[0] == ep.TCP_LISTEN
    return int(struct.unpack_from("=I", info, 28)[0])


def test_the_listen_backlog_is_the_sockets_then_the_kernels(tmp_path: Path) -> None:
    """The socket unit's own backlog first (asyncio re-listens on the socket it is given); a
    socket that does not say falls back to net.core.somaxconn, then to 4096."""
    knob = tmp_path / "somaxconn"
    knob.write_text("512\n", encoding="ascii")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(37)
        assert ep.listen_backlog(listener, knob) == 37
    with socket.socket(socket.AF_UNIX) as unix:
        assert ep.listen_backlog(unix, knob) == 512
    assert ep.listen_backlog(None, knob) == 512
    assert ep.listen_backlog(None, tmp_path / "missing") == 4096
    knob.write_text("x", encoding="ascii")
    assert ep.listen_backlog(None, knob) == 4096


async def test_serve_keeps_the_socket_units_backlog(encoder: Path) -> None:
    """Without ``backlog=`` asyncio's re-listen sets 100 (the recorded proxy of serving_m1d
    had that); the proxy keeps what the socket was given (here 37)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(37)
    listener.setblocking(False)
    proxy = ep.Proxy(encoder, connections_max=8, idle_timeout=30.0)
    task = asyncio.ensure_future(ep.serve(listener, proxy))
    try:
        await asyncio.sleep(0.05)
        assert _backlog_of(listener) == 37
    finally:
        await _stop(task)
        listener.close()
