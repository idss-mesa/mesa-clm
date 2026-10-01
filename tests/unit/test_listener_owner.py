"""A keyed client never sends its key to a loopback port another account holds (DESIGN A5).

The serving units are not enabled at boot, so 127.0.0.1:8090 and :8700 are usually free, and
any local account can bind them. ``net.assert_listener_owner`` reads the listening sockets of
this network namespace from ``/proc/net/tcp`` and ``tcp6`` (here fixture files) and refuses the
port when the socket a connection would reach belongs to another account; ``HttpEndpoint`` asks
before every keyed request over the real network, the pre-flight turns a refusal into exit 2 and
a refusal mid-run fails the run. Nothing here reaches a listener (one test connects to a closed
loopback port)."""

from __future__ import annotations

import ipaddress
import os
import socket
import struct
import threading
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import net
from mesa_clm.clm.http import HttpEndpoint
from mesa_clm.config import load_config
from mesa_clm.net import (
    EndpointError,
    ListenerOwnerError,
    assert_listener_owner,
    listener_owners,
    proc_address,
    proc_tcp_listeners,
)
from mesa_clm.providers import live
from mesa_clm.providers.base import DeciderRefused

ME = os.getuid()
OTHER = ME + 4242
HEADER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
    "timeout inode\n"
)


def _hex(address: str) -> str:
    """``address`` as ``/proc/net/tcp{,6}`` prints it (little-endian 32-bit words)."""
    packed = ipaddress.ip_address(address).packed
    return b"".join(packed[i : i + 4][::-1] for i in range(0, len(packed), 4)).hex().upper()


def _row(n: int, address: str, port: int, uid: int, state: str = "0A") -> str:
    remote = "0" * len(_hex(address))
    return (
        f"   {n}: {_hex(address)}:{port:04X} {remote}:0000 {state} 00000000:00000000 "
        f"00:00000000 00000000 {uid:>5}        0 {1000 + n} 1 0000000000000000 100 0 0 10 0\n"
    )


def _proc(tmp_path: Path, *listeners: tuple[str, int, int], state: str = "0A") -> Path:
    """A ``/proc/net`` with ``tcp`` and ``tcp6`` holding ``(address, port, uid)`` listeners."""
    base = tmp_path / "proc-net"
    base.mkdir(exist_ok=True)
    v4 = [x for x in listeners if ipaddress.ip_address(x[0]).version == 4]
    v6 = [x for x in listeners if ipaddress.ip_address(x[0]).version == 6]
    (base / "tcp").write_text(
        HEADER + "".join(_row(i, *x, state=state) for i, x in enumerate(v4)), encoding="ascii"
    )
    (base / "tcp6").write_text(
        HEADER + "".join(_row(i, *x, state=state) for i, x in enumerate(v6)), encoding="ascii"
    )
    return base


def test_proc_rows_are_parsed_with_their_owner(tmp_path: Path) -> None:
    base = _proc(tmp_path, ("127.0.0.1", 8090, ME), ("::1", 8700, OTHER), ("0.0.0.0", 22, 0))  # noqa: S104
    v4 = proc_tcp_listeners((base / "tcp").read_text(encoding="ascii"))
    v6 = proc_tcp_listeners((base / "tcp6").read_text(encoding="ascii"))
    assert v4 == [
        (ipaddress.IPv4Address("127.0.0.1"), 8090, ME),
        (ipaddress.IPv4Address("0.0.0.0"), 22, 0),  # noqa: S104
    ]
    assert v6 == [(ipaddress.IPv6Address("::1"), 8700, OTHER)]
    assert proc_address(_hex("::ffff:127.0.0.1")) == ipaddress.IPv6Address("::ffff:127.0.0.1")
    # Established connections and short rows are not listeners.
    assert proc_tcp_listeners(HEADER + _row(0, "127.0.0.1", 8090, OTHER, state="01")) == []
    assert proc_tcp_listeners(HEADER + "   0: 0100007F:1F9A 00000000:0000 0A 0\n") == []


@pytest.mark.parametrize(
    ("listeners", "host", "refused"),
    [
        ([("127.0.0.1", 8090, OTHER)], "127.0.0.1", True),
        ([("127.0.0.1", 8090, ME)], "127.0.0.1", False),
        ([("127.0.0.1", 8090, 0)], "127.0.0.1", False),  # root can read the key files anyway
        # The kernel prefers the exact address: an own exact socket wins over a foreign wildcard.
        ([("127.0.0.1", 8090, ME), ("0.0.0.0", 8090, OTHER)], "127.0.0.1", False),  # noqa: S104
        ([("0.0.0.0", 8090, OTHER)], "127.0.0.1", True),  # noqa: S104
        ([("::", 8090, OTHER)], "127.0.0.1", True),  # dual-stack unless v6-only
        ([("::ffff:127.0.0.1", 8090, OTHER)], "127.0.0.1", True),
        ([("0.0.0.0", 8090, OTHER)], "::1", False),  # noqa: S104 - IPv4 only
        ([("::", 8090, OTHER)], "::1", True),
        ([("::1", 8090, OTHER)], "localhost", True),
        ([("127.0.0.2", 8090, OTHER)], "127.0.0.1", False),
        ([("127.0.0.1", 8700, OTHER)], "127.0.0.1", False),  # another port
        ([], "127.0.0.1", False),
    ],
)
def test_who_holds_the_port(
    tmp_path: Path, listeners: list[tuple[str, int, int]], host: str, refused: bool
) -> None:
    base = _proc(tmp_path, *listeners)
    shown = f"[{host}]" if ":" in host else host
    url = f"http://{shown}:8090"
    if refused:
        with pytest.raises(ListenerOwnerError, match=f"uid {OTHER}, not of this account"):
            assert_listener_owner(url, what="encoder", proc_net=base)
    else:
        assert_listener_owner(url, what="encoder", proc_net=base)


def test_what_is_not_checked(tmp_path: Path) -> None:
    base = _proc(tmp_path, ("127.0.0.1", 443, OTHER), ("127.0.0.1", 80, OTHER))
    assert listener_owners("127.0.0.1", 443, proc_net=base) == [("127.0.0.1:443", OTHER)]
    # A remote URL (allow_remote and https) is not a loopback port.
    assert_listener_owner("https://clm.example.org", what="CLM server", proc_net=base)
    # Not Linux, or /proc unreadable: nothing to read, nothing refused.
    assert listener_owners("127.0.0.1", 8090, proc_net=tmp_path / "nowhere") is None
    assert_listener_owner("http://127.0.0.1:8090", what="encoder", proc_net=tmp_path / "x")
    # The URL's default port is the one checked.
    with pytest.raises(ListenerOwnerError):
        assert_listener_owner("http://127.0.0.1", what="encoder", proc_net=base)
    assert issubclass(ListenerOwnerError, EndpointError)


def test_the_keyed_endpoint_refuses_before_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path, ("127.0.0.1", 8090, OTHER)))
    keyed = HttpEndpoint("http://127.0.0.1:8090", "k" * 24, what="encoder", timeout=1.0)
    with pytest.raises(ListenerOwnerError, match="the key is not sent") as exc:
        keyed.request("GET", "/v1/models")
    assert "k" * 24 not in str(exc.value)
    assert keyed.calls == 0 and keyed.breaker.consecutive_failures == 0
    keyed.close()
    # Without a key, or through an injected transport (the tests' fakes), nothing is checked.
    assert not HttpEndpoint("http://127.0.0.1:8090", None, what="e", timeout=1.0)._check_owner
    import httpx

    mocked = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    assert not HttpEndpoint(
        "http://127.0.0.1:8090", "k" * 24, what="e", timeout=1.0, transport=mocked
    )._check_owner


def test_the_owner_is_checked_before_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart window DESIGN A5 names (the port is free for a moment on every restart): a
    retry asks again who holds the port. Here nothing listens at the first attempt and another
    account takes the port during the backoff, so the second attempt is refused unsent."""
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    def squat(_delay: float) -> None:
        _proc(tmp_path, ("127.0.0.1", port, OTHER))

    keyed = HttpEndpoint(
        f"http://127.0.0.1:{port}", "k" * 24, what="encoder", timeout=1.0, retries=2, sleep=squat
    )
    try:
        with pytest.raises(ListenerOwnerError, match=f"127.0.0.1:{port} is held by a socket"):
            keyed.request("GET", "/v1/models")
        assert keyed.calls == 0 and keyed.breaker.consecutive_failures == 1
    finally:
        keyed.close()


def test_the_serving_probes_raw_clients_check_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``scripts/serving_probes.py`` (the runbook's probe tool) also sends both keys through raw
    ``httpx`` clients: the auth matrix, ``/tokenize``, ``/metrics``, ``/version`` and the bounded
    embeddings requests. Each keyed request asks first (the review of 70dbefe); a request
    without a key is not checked."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "serving_probes.py"
    spec = importlib.util.spec_from_file_location("serving_probes_owner_check", path)
    assert spec is not None and spec.loader is not None
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path, ("127.0.0.1", port, OTHER)))
    http = probes.Http()
    try:
        with pytest.raises(ListenerOwnerError, match="serving probe"):
            http.status("GET", f"http://127.0.0.1:{port}/v1/models", "k" * 24)
        # No key: not asked; nothing really listens there, so the probe records -1.
        assert http.status("GET", f"http://127.0.0.1:{port}/health") == -1
    finally:
        http.client.close()
    monkeypatch.setattr(probes, "ENC_URL", f"http://127.0.0.1:{port}")
    with pytest.raises(ListenerOwnerError):
        probes._bounded("k" * 24, {"input": ["x"]}, 1.0)


def _keyed_env() -> dict[str, str]:
    return {
        "MESA_CLM_CLM__API_KEY": "clm-key-0123456789abcdef",
        "MESA_CLM_ENCODER__API_KEY": "enc-key-0123456789abcdef",
    }


def test_the_preflight_refuses_a_squatted_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clm-serve's /health is the first keyed request of an annotate run: another account's
    socket on :8700 is a pre-flight error (exit 2, never degraded), not "unreachable"."""
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path, ("127.0.0.1", 8700, OTHER)))
    cfg = load_config(env=_keyed_env())
    stack = live.clm_provider(cfg)
    try:
        with pytest.raises(live.PreflightError, match="8700 is held by a socket of uid"):
            live.preflight(stack, cfg)
    finally:
        stack.close()
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path, ("127.0.0.1", 8090, OTHER)))
    stack = live.clm_provider(cfg)
    try:
        with pytest.raises(live.PreflightError, match="8090 is held"):
            live._models("encoder", stack.encoder.models, stack.encoder.endpoint.shown)
    finally:
        stack.close()


def test_annotate_exits_2_on_a_squatted_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mesa_clm.cli import EXIT_CONFIG, main

    card = Path(__file__).resolve().parents[1] / "fixtures" / "cards-srer"
    for name in list(os.environ):
        if name.startswith(("MESA_CLM_", "CLM_", "MESA_LLM_", "MESA_HOME")):
            monkeypatch.delenv(name, raising=False)
    for name, value in _keyed_env().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MESA_CLM_PROVENANCE__DSN", f"duckdb:///{tmp_path / 'prov.duckdb'}")
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path, ("0.0.0.0", 8700, OTHER)))  # noqa: S104
    argv = ["annotate", "--card", str(card / "DP1.00004.001.BP_30min.md"), "--tier", "zero_shot"]
    assert main(argv) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "the key is not sent" in err and "0123456789abcdef" not in err
    assert not (tmp_path / "prov.duckdb").exists()


class _SquattedClient:
    model = "clm-latest"

    def system_one(self, *args: Any, **kw: Any) -> Any:
        raise ListenerOwnerError("CLM server: 127.0.0.1:8700 is held by a socket of uid 4242")


def test_a_squatted_port_mid_run_refuses_the_run(card: Any) -> None:
    """Between two requests another account takes clm-serve's port: the run fails (recorded
    ``failed``) rather than degrading every group to ``ols_rank``."""
    from mesa_clm import framings as fr
    from mesa_clm.providers.tiered import FakeProvider
    from mesa_clm.states import target_state

    column = next(c for c in card.columns if c.name == "observerDistance")
    state = target_state(card, "column", "measurement", column=column)
    cands = [fr.FramingCandidate("PATO:0000040", "distance", "A 1-D extent quality.")]
    provider = FakeProvider(client=_SquattedClient())
    with pytest.raises(DeciderRefused, match="held by a socket") as exc:
        provider.decide(fr.active_framing("term.fits"), [state], [cands])
    assert exc.value.status == 0


# -- the connection itself: checked after the connect, before a byte is sent (DESIGN A5) ----------
#
# The /proc read and the connect are two steps, and another account can bind the port in between
# (the convergence round after 9681eed captured a dummy key on 16 of 400 attempts against a
# toggling listener). Each new connection of a keyed client is therefore checked once it exists:
# the kernel's socket diagnostics name the owner of its server end. These tests run a real
# loopback listener of this account; "another account" is the kernel's answer replaced
# (``net.connection_owner``), since a test cannot own a socket as another uid.

KEY = "dummy-key-0123456789abcdef"


class _Recorder:
    """A loopback server of this account for one connection: it records every byte it gets and
    answers ``200 {}`` once a request's headers are in."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self.listener = socket.socket(family, socket.SOCK_STREAM)
        self.listener.bind((host, 0))
        self.listener.listen(8)
        self.port = int(self.listener.getsockname()[1])
        self.url = f"http://[{host}]:{self.port}" if ":" in host else f"http://{host}:{self.port}"
        self.received = bytearray()
        self.accepted = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        self.listener.settimeout(5.0)
        try:
            conn, _ = self.listener.accept()
        except OSError:
            return
        self.accepted.set()
        with conn:
            conn.settimeout(5.0)
            while True:
                try:
                    chunk = conn.recv(1 << 16)
                except OSError:
                    break
                if not chunk:
                    break
                self.received += chunk
                if b"\r\n\r\n" in self.received:
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        b"Content-Length: 2\r\nConnection: close\r\n\r\n{}"
                    )
                    break

    def close(self) -> bytes:
        """Wait for the connection to end; the bytes the server received."""
        self.thread.join(10.0)
        self.listener.close()
        return bytes(self.received)


def _foreign_server_end(monkeypatch: pytest.MonkeyPatch, asked: list[int] | None = None) -> None:
    """Make the kernel's answer name another account as the server end's owner."""

    def owner(sock: socket.socket) -> net.SocketOwner:
        if asked is not None:
            asked.append(int(sock.getpeername()[1]))
        return net.SocketOwner(state=1, uid=OTHER, inode=0)

    monkeypatch.setattr(net, "connection_owner", owner)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_the_kernel_names_the_server_end_of_a_connection(host: str) -> None:
    """The real kernel's answer about a connection to a listener of this account. Before the
    server accepts, the serving host's kernel (6.17) already names the listener's owner (no
    inode yet), while older kernels report uid 0 with no inode, an answer the check asks again
    about; once accepted, this account's uid with the inode."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.bind((host, 0))
    except OSError:
        pytest.skip(f"no {host} on this host")
    with listener, socket.socket(family, socket.SOCK_STREAM) as client:
        listener.listen(1)
        client.connect(listener.getsockname()[:2])
        before = net.connection_owner(client)
        assert before.state == 1 and before.inode == 0
        assert before.uid == ME if before.decisive else before.uid == 0
        server, _ = listener.accept()
        with server:
            after = net.connection_owner(client)
            assert after.uid == ME and after.inode != 0 and after.decisive
            net.assert_connection_owner(client, what="encoder")


def test_a_dual_stack_listener_is_named_too() -> None:
    """A listener on ``::`` takes an IPv4 connection with an IPv6 socket, which the kernel
    reports in the IPv4-mapped form."""
    try:
        listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        listener.bind(("::", 0))
    except OSError:
        pytest.skip("no dual-stack IPv6 on this host")
    with listener, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        listener.listen(1)
        client.connect(("127.0.0.1", int(listener.getsockname()[1])))
        server, _ = listener.accept()
        with server:
            assert net.connection_owner(client).uid == ME
            net.assert_connection_owner(client, what="encoder")


def _reply(
    own: tuple[str, int], remote: tuple[str, int], *, state: int, uid: int, inode: int
) -> bytes:
    """A ``SOCK_DIAG_BY_FAMILY`` answer about the IPv4 socket ``own`` -> ``remote``."""
    msg = bytearray(72)
    msg[0], msg[1] = socket.AF_INET, state
    msg[4:8] = struct.pack("!HH", own[1], remote[1])
    msg[8:12] = ipaddress.ip_address(own[0]).packed
    msg[24:28] = ipaddress.ip_address(remote[0]).packed
    msg[64:72] = struct.pack("=II", uid, inode)
    return struct.pack("=IHHII", 16 + 72, net.SOCK_DIAG_BY_FAMILY, 0, 1, 0) + bytes(msg)


def test_the_kernels_answer_is_read_strictly() -> None:
    own, remote = ("127.0.0.1", 8090), ("127.0.0.1", 40000)
    request = net.sock_diag_request(own, remote)
    assert struct.unpack_from("!HH", request, 16 + 8) == (8090, 40000)
    assert net.sock_diag_owner(_reply(own, remote, state=1, uid=ME, inode=0), own, remote) == (
        net.SocketOwner(state=1, uid=ME, inode=0)
    )
    # No such socket (ENOENT), an answer about another socket, a short reply: no owner.
    enoent = struct.pack("=IHHII", 36, net.NLMSG_ERROR, 0, 1, 0) + struct.pack("=i", -2)
    with pytest.raises(net.OwnerUnknown, match="sock_diag"):
        net.sock_diag_owner(enoent, own, remote)
    other = _reply(own, ("127.0.0.1", 40001), state=1, uid=ME, inode=7)
    with pytest.raises(net.OwnerUnknown, match="another socket"):
        net.sock_diag_owner(other, own, remote)
    with pytest.raises(net.OwnerUnknown, match="short"):
        net.sock_diag_owner(b"\0" * 8, own, remote)
    done = struct.pack("=IHHII", 20, 3, 0, 1, 0) + b"\0" * 4  # NLMSG_DONE: not an answer
    with pytest.raises(net.OwnerUnknown, match="unexpected sock_diag reply"):
        net.sock_diag_owner(done, own, remote)
    with pytest.raises(net.OwnerUnknown, match="mixed"):
        net.sock_diag_request(("127.0.0.1", 1), ("::1", 2))
    # The handshake and TIME_WAIT, and uid 0 without an inode, name nobody; root with an inode
    # (an accepted socket) does.
    assert not net.SocketOwner(state=net.TCP_SYN_RECV, uid=0, inode=0).decisive
    assert not net.SocketOwner(state=net.TCP_TIME_WAIT, uid=0, inode=0).decisive
    assert not net.SocketOwner(state=1, uid=0, inode=0).decisive
    assert net.SocketOwner(state=1, uid=0, inode=9).decisive
    assert net.SocketOwner(state=1, uid=OTHER, inode=0).decisive


def test_another_accounts_server_end_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with (
        socket.create_server(("127.0.0.1", 0)) as listener,
        socket.create_connection(listener.getsockname()[:2]) as client,
    ):
        _foreign_server_end(monkeypatch)
        with pytest.raises(ListenerOwnerError, match=f"uid {OTHER}, not of this account"):
            net.assert_connection_owner(client, what="encoder")
        # An answer that names nobody is asked again and then refused: nothing unchecked.
        answers = iter([net.SocketOwner(net.TCP_SYN_RECV, 0, 0), net.SocketOwner(1, ME, 0)])
        monkeypatch.setattr(net, "connection_owner", lambda s: next(answers))
        net.assert_connection_owner(client, what="encoder", sleep=lambda s: None)
        monkeypatch.setattr(net, "connection_owner", lambda s: net.SocketOwner(1, 0, 0))
        with pytest.raises(ListenerOwnerError, match="named no owner"):
            net.assert_connection_owner(client, what="encoder", wait=0.0)
        monkeypatch.setattr(net, "connection_owner", lambda s: net.SocketOwner(1, 0, 5))
        net.assert_connection_owner(client, what="encoder")  # root's accepted socket

        def gone(sock: socket.socket) -> net.SocketOwner:
            raise net.OwnerUnknown("sock_diag answered about another socket")

        monkeypatch.setattr(net, "connection_owner", gone)
        with pytest.raises(ListenerOwnerError, match="another socket"):
            net.assert_connection_owner(client, what="encoder", wait=0.0)

        def no_netlink(sock: socket.socket) -> net.SocketOwner:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(net, "connection_owner", no_netlink)
        with pytest.raises(ListenerOwnerError, match="cannot ask the kernel"):
            net.assert_connection_owner(client, what="encoder")
    # A connection that is not to a loopback address is not asked about.
    monkeypatch.setattr(net, "connection_owner", gone)

    class Remote:
        family = socket.AF_INET

        def getpeername(self) -> tuple[str, int]:
            return ("192.0.2.7", 443)

    net.assert_connection_owner(Remote(), what="CLM server")  # type: ignore[arg-type]
    # A unix socket is not asked about; a TCP socket that is not connected is refused.
    left, right = socket.socketpair()
    with left, right:
        net.assert_connection_owner(left, what="encoder")
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as unconnected,
        pytest.raises(ListenerOwnerError, match="closed before its owner was checked"),
    ):
        net.assert_connection_owner(unconnected, what="encoder")


def test_the_trace_checks_only_a_new_connection_and_closes_it_on_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    trace = net.owner_trace("encoder", lambda event, info: seen.append(event))
    trace("http11.send_request_headers.started", {})  # not a connect: nothing asked

    class Stream:
        closed = False

        def __init__(self, sock: object) -> None:
            self.sock = sock

        def get_extra_info(self, name: str) -> object:
            return self.sock if name == "socket" else None

        def close(self) -> None:
            self.closed = True

    no_socket = Stream(None)
    with pytest.raises(ListenerOwnerError, match="no socket to check"):
        trace(net.CONNECT_TCP_COMPLETE, {"return_value": no_socket})
    assert no_socket.closed
    with pytest.raises(ListenerOwnerError, match="no socket to check"):
        trace(net.CONNECT_TCP_COMPLETE, {"return_value": None})
    with (
        socket.create_server(("127.0.0.1", 0)) as listener,
        socket.create_connection(listener.getsockname()[:2]) as client,
        listener.accept()[0],
    ):
        own = Stream(client)
        trace(net.CONNECT_TCP_COMPLETE, {"return_value": own})
        assert not own.closed
        _foreign_server_end(monkeypatch)
        foreign = Stream(client)
        with pytest.raises(ListenerOwnerError):
            trace(net.CONNECT_TCP_COMPLETE, {"return_value": foreign})
        assert foreign.closed
    assert seen == ["http11.send_request_headers.started", *[net.CONNECT_TCP_COMPLETE] * 4]


def test_a_port_taken_after_the_check_gets_no_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check-then-connect race: ``/proc`` shows the port as this account's (or nobody's)
    when the endpoint asks, and another account's socket answers the connect. The connection
    is refused before the request line, let alone the key, is written."""
    server = _Recorder()
    monkeypatch.setattr(net, "PROC_NET", _proc(tmp_path, ("127.0.0.1", server.port, ME)))
    asked: list[int] = []
    _foreign_server_end(monkeypatch, asked)
    keyed = HttpEndpoint(server.url, KEY, what="encoder", timeout=5.0, retries=1)
    try:
        with pytest.raises(ListenerOwnerError, match="not of this account") as exc:
            keyed.request("GET", "/v1/models")
    finally:
        keyed.close()
    assert server.close() == b"" and asked == [server.port]
    assert KEY not in str(exc.value) and keyed.calls == 0


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_this_accounts_connection_carries_the_request(host: str) -> None:
    """The same endpoint against a listener of this account, the kernel's real answer: the
    request (with its key) goes through the checked connection."""
    try:
        server = _Recorder(host)
    except OSError:
        pytest.skip(f"no {host} on this host")
    keyed = HttpEndpoint(server.url, KEY, what="encoder", timeout=5.0, retries=1)
    try:
        assert isinstance(keyed._client._transport, net.OwnerCheckedTransport)
        assert keyed.get_json("/v1/models") == {}
    finally:
        keyed.close()
    received = server.close()
    assert received.startswith(b"GET /v1/models ") and f"Bearer {KEY}".encode() in received


def test_the_planner_and_the_probes_check_each_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other keyed clients: the planner gateway client falls back to the static rules, and
    the probes' raw client refuses a connection another account answered, keyed or not (a
    connection is reused across requests)."""
    import importlib.util

    from mesa_clm.cards import load_card
    from mesa_clm.planner.gateway_planner import GatewayPlanner

    _foreign_server_end(monkeypatch)
    server = _Recorder()
    planner = GatewayPlanner(server.url, KEY, timeout=5.0)
    assert planner._check_owner
    assert isinstance(planner._client._transport, net.OwnerCheckedTransport)
    card = load_card(
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "cards-srer"
        / "DP1.00004.001.BP_30min.md"
    )
    try:
        assert planner.plan(card).fallback
    finally:
        planner._client.close()
    assert server.close() == b""

    path = Path(__file__).resolve().parents[2] / "scripts" / "serving_probes.py"
    spec = importlib.util.spec_from_file_location("serving_probes_connection_check", path)
    assert spec is not None and spec.loader is not None
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    server = _Recorder()
    http = probes.Http()
    try:
        assert isinstance(http.client._transport, net.OwnerCheckedTransport)
        with pytest.raises(ListenerOwnerError):
            http.status("GET", f"{server.url}/health")
    finally:
        http.client.close()
    assert server.close() == b""
