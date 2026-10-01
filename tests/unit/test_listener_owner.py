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
