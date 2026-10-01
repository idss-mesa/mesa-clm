#!/usr/bin/env python3
"""Record the encoder endpoint of the running units as a results file (DESIGN A5, the revision
before the M1 merge): who owns the two loopback ports, how the units are wired, what the proxy
runs with, and whether the endpoint still answers a keyed request while many idle connections
are held open::

    uv run python scripts/endpoint_checks.py --out bench/results/<date>/serving_m1d.json
    uv run python scripts/endpoint_checks.py --foreign --out bench/results/<date>/serving_m1e.json

Read-only apart from the idle connections it opens and closes (``--idle``, default 1,000, each
an unauthenticated TCP connection that sends nothing): ``systemctl --user show``, ``ss -ltne``,
``/proc/<proxy pid>/limits``, ``status`` and ``fd``, ``systemd-analyze --user security`` of the
proxy unit, the run directory, the clients' port-owner check
(:func:`mesa_clm.net.assert_listener_owner`), ``verify_serving_lock(require_live=True)``, and one
keyed ``GET /v1/models`` through :class:`~mesa_clm.clm.encoder.EncoderClient` (the key read from
its file inside this process). ``--foreign`` adds what another local account gets on
127.0.0.1:8090: throwaway containers of the lock's pinned image on the host network, as uid 0
and as uid 65534, each send an unauthenticated ``GET /health`` (no key exists in them), and
the bytes they got back are recorded next to the same request from this account, with the
proxy's journal lines about them. The home directory is written as ``~``; before anything is
written the text is checked for both keys and the run aborts if either appears.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mesa_clm import serving
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.config import load_config
from mesa_clm.health import parse_listener_details
from mesa_clm.net import ListenerOwnerError, assert_listener_owner

UNIT_PROPS = {
    serving.ENCODER_UNIT: [
        "Requires",
        "After",
        "Wants",
        "StartLimitBurst",
        "StartLimitIntervalUSec",
    ],
    serving.PROXY_SOCKET: ["Listen", "PartOf", "ActiveState"],
    serving.PROXY_SERVICE: [
        "ExecStart",
        "LimitNOFILE",
        "LimitNOFILESoft",
        "MainPID",
        "Type",
        # The sandbox (the review of 70dbefe).
        "NoNewPrivileges",
        "PrivateUsers",
        "ProtectSystem",
        "ProtectHome",
        "PrivateTmp",
        "ProtectProc",
        "ProcSubset",
        "RestrictAddressFamilies",
        "RestrictNamespaces",
        "MemoryDenyWriteExecute",
        "SystemCallArchitectures",
    ],
    serving.SERVE_UNIT: ["Requires", "After", "ActiveState"],
}
FOREIGN_UIDS = (0, 65534)
HEALTH_REQUEST = "GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"


def _run(argv: list[str], timeout: float = 30) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout).stdout


def _show(unit: str, props: list[str]) -> dict[str, str]:
    out = _run(["systemctl", "--user", "show", unit, *(f"--property={p}" for p in props)])
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _listeners() -> dict[str, Any]:
    text = _run(["ss", "-ltne"])
    return {
        str(port): [x._asdict() for x in parse_listener_details(text, port)]
        for port in (serving.ENCODER_PORT, serving.CLM_PORT)
    }


def _run_dir() -> dict[str, Any]:
    run = serving.serving_home() / serving.RUN_DIR
    st = os.lstat(run)
    sock = run / serving.SOCKET_NAME
    sst = os.lstat(sock)
    return {
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "owner_is_this_account": st.st_uid == os.getuid(),
        "entries": sorted(os.listdir(run)),
        "encoder_sock_is_socket": stat.S_ISSOCK(sst.st_mode),
        "encoder_sock_owner_is_this_account": sst.st_uid == os.getuid(),
    }


def _proxy(pid: int) -> dict[str, Any]:
    limits = Path(f"/proc/{pid}/limits").read_text(encoding="ascii")
    nofile = next(line for line in limits.splitlines() if line.startswith("Max open files"))
    status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    fields = {k.strip(): v.strip() for k, _, v in (ln.partition(":") for ln in status.splitlines())}
    return {
        "max_open_files": nofile.split()[3:5],
        "fds": len(os.listdir(f"/proc/{pid}/fd")),
        "no_new_privs": fields.get("NoNewPrivs"),
        "seccomp_mode": fields.get("Seccomp"),
        "uid_map": Path(f"/proc/{pid}/uid_map").read_text(encoding="ascii").split(),
    }


def _exposure() -> str:
    """The last line of ``systemd-analyze --user security`` for the proxy unit (its score)."""
    out = _run(["systemd-analyze", "--user", "security", serving.PROXY_SERVICE, "--no-pager"])
    lines = [ln for ln in out.splitlines() if "Overall exposure level" in ln]
    return " ".join(lines[-1].split(":", 1)[1].split()[:2]) if lines else "unavailable"


def _idle_then_keyed(enc: EncoderClient, n_idle: int, pid: int) -> dict[str, Any]:
    """Hold ``n_idle`` unauthenticated connections that send nothing, then ask with the key."""
    held: list[socket.socket] = []
    try:
        for _ in range(n_idle):
            held.append(socket.create_connection(("127.0.0.1", serving.ENCODER_PORT), timeout=10))
        time.sleep(1.0)
        during = _proxy(pid)
        t0 = time.monotonic()
        models = enc.models()
        ms = (time.monotonic() - t0) * 1000.0
    finally:
        for s in held:
            s.close()
    time.sleep(1.0)
    return {
        "idle_connections_held": len(held),
        "proxy_while_held": during,
        "keyed_get_v1_models": {"answered": [m.get("id") for m in models], "ms": round(ms, 1)},
        "proxy_after_release": _proxy(pid),
    }


def _health_status_line(data: bytes) -> str:
    return data.split(b"\r\n", 1)[0].decode("ascii", "replace") if data else ""


def _own_health() -> dict[str, Any]:
    """This account's unauthenticated ``GET /health`` on :8090, raw."""
    with socket.create_connection(("127.0.0.1", serving.ENCODER_PORT), timeout=10) as s:
        s.sendall(HEALTH_REQUEST.encode("ascii"))
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    return {"bytes_back": len(data), "status_line": _health_status_line(data)}


def _foreign(image: str, since: str) -> dict[str, Any]:
    """Another local account's view of :8090: a throwaway container of ``image`` on the host
    network per uid of :data:`FOREIGN_UIDS` (no key, no mount, no GPU) sends the same
    unauthenticated request and reports how many bytes came back."""
    script = (
        "exec 3<>/dev/tcp/127.0.0.1/" + str(serving.ENCODER_PORT) + " || { echo connect-failed; "
        "exit 0; }; printf " + shlex.quote(HEALTH_REQUEST.replace("\r\n", "\\r\\n")) + " >&3; "
        "timeout 10 cat <&3 2>/dev/null | wc -c"
    )
    out: dict[str, Any] = {"image": image, "network": "host"}
    for uid in FOREIGN_UIDS:
        cmd = (
            f"docker run --rm --network host --user {uid}:{uid} --entrypoint bash {image} -c "
            + shlex.quote(script)
        )
        got = _run(["sg", "docker", "-c", cmd], timeout=120).strip().splitlines()
        out[f"uid_{uid}"] = {"bytes_back": got[-1] if got else "no output"}
    time.sleep(1.0)
    journal = _run(
        [
            "journalctl",
            "--user",
            "-u",
            serving.PROXY_SERVICE,
            "--since",
            since,
            "-o",
            "cat",
            "--no-pager",
        ]
    )
    out["proxy_journal_since_start"] = [
        ln for ln in journal.splitlines() if "closing a connection" in ln or "relaying" in ln
    ]
    out["own_account_same_request"] = _own_health()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--idle", type=int, default=1000)
    ap.add_argument(
        "--foreign",
        action="store_true",
        help="also connect from uid 0 and 65534 (throwaway containers of the lock's image)",
    )
    args = ap.parse_args(argv)
    cfg = load_config()
    started = dt.datetime.now(tz=dt.UTC)
    since = started.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    units = {unit: _show(unit, props) for unit, props in UNIT_PROPS.items()}
    owners: dict[str, str] = {}
    for name, url in (("encoder", cfg.encoder.url), ("clm-serve", cfg.clm.base_url)):
        try:
            assert_listener_owner(url, what=name)
            owners[name] = "this account's socket (the key may be sent)"
        except ListenerOwnerError as exc:
            owners[name] = f"refused: {exc}"
    pid = int(units[serving.PROXY_SERVICE].get("MainPID") or 0)
    enc = EncoderClient.from_config(cfg.encoder, allow_remote=cfg.clm.allow_remote)
    try:
        idle = _idle_then_keyed(enc, args.idle, pid) if pid else {"skipped": "no proxy process"}
    finally:
        enc.close()
    foreign: dict[str, Any] | None = None
    if args.foreign:
        lock = json.loads((serving.serving_home() / "serving.lock.json").read_text("utf-8"))
        foreign = _foreign(f"{lock['image']['ref']}@{lock['image']['digest']}", since)
    flags = " --foreign" if args.foreign else ""
    payload = {
        "format": "mesa-clm/endpoint-checks/1",
        "command": f"uv run python scripts/endpoint_checks.py --out {args.out} --idle {args.idle}"
        + flags,
        "started": started.isoformat(timespec="seconds"),
        "units": units,
        "enabled": _run(["systemctl", "--user", "is-enabled", *serving.UNIT_NAMES]).split(),
        "listeners": _listeners(),
        "client_port_owner_check": owners,
        "run_directory": _run_dir(),
        "proxy_security_exposure": _exposure(),
        "lock_problems_require_live": serving.verify_serving_lock(require_live=True),
        "idle_connections": idle,
    }
    if foreign is not None:
        payload["other_accounts"] = foreign
    text = json.dumps(payload, indent=1, ensure_ascii=False).replace(str(Path.home()), "~") + "\n"
    keys = [k for k in (cfg.clm.resolved_api_key(), cfg.encoder.resolved_api_key()) if k]
    if any(k in text for k in keys):  # never written, never printed
        print("endpoint_checks: a key appeared in the record; nothing written", file=sys.stderr)
        return 2
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
