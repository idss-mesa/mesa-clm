#!/usr/bin/env python3
"""Record the encoder endpoint of the running units as a results file (DESIGN A5, the revision
before the M1 merge): who owns the two loopback ports, how the units are wired, what the proxy
runs with, and whether the endpoint still answers a keyed request while many idle connections
are held open::

    uv run python scripts/endpoint_checks.py --out bench/results/<date>/serving_m1d.json

Read-only apart from the idle connections it opens and closes (``--idle``, default 1,000, each
an unauthenticated TCP connection that sends nothing): ``systemctl --user show``, ``ss -ltne``,
``/proc/<proxy pid>/limits`` and ``fd``, the run directory, the clients' port-owner check
(:func:`mesa_clm.net.assert_listener_owner`), ``verify_serving_lock(require_live=True)``, and one
keyed ``GET /v1/models`` through :class:`~mesa_clm.clm.encoder.EncoderClient` (the key read from
its file inside this process). The home directory is written as ``~``; before anything is
written the text is checked for both keys and the run aborts if either appears.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
    serving.PROXY_SERVICE: ["ExecStart", "LimitNOFILE", "LimitNOFILESoft", "MainPID", "Type"],
    serving.SERVE_UNIT: ["Requires", "After", "ActiveState"],
}


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30).stdout


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
    return {"max_open_files": nofile.split()[3:5], "fds": len(os.listdir(f"/proc/{pid}/fd"))}


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--idle", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = load_config()
    started = dt.datetime.now(tz=dt.UTC)
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
    payload = {
        "format": "mesa-clm/endpoint-checks/1",
        "command": f"uv run python scripts/endpoint_checks.py --out {args.out} --idle {args.idle}",
        "started": started.isoformat(timespec="seconds"),
        "units": units,
        "enabled": _run(["systemctl", "--user", "is-enabled", *serving.UNIT_NAMES]).split(),
        "listeners": _listeners(),
        "client_port_owner_check": owners,
        "run_directory": _run_dir(),
        "lock_problems_require_live": serving.verify_serving_lock(require_live=True),
        "idle_connections": idle,
    }
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
