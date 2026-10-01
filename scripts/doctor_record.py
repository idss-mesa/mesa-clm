#!/usr/bin/env python3
"""Record ``mesa-clm doctor --serve`` on the serving host as a results file (plan §6.8, §8 M1-A
"doctor green").

Runs the doctor in process with every check (``quick=False``, ``serve=True``: vendored hashes,
the serving lock with ``require_live``, the 401 matrix, the encoder goldens, the long-input probe,
the systemone parity and the drift probes) against the configuration this host resolves (keys
from the configured files or the default ``~/.mesa/clm/secrets/{clm,encoder}.key``), then writes
the report as JSON::

    uv run python scripts/doctor_record.py --out bench/results/<date>/doctor_serve.json

The home directory is written as ``~`` and the host name as its first label (``sparky-1``), so
the record carries no account path or network name; before anything is written the text is
checked for both bearer keys (read inside this process only) and the run aborts if either
appears. Exit 0 when the report is ``ok`` (warnings allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import sys
import time
from pathlib import Path

from mesa_clm import __version__
from mesa_clm.config import load_config
from mesa_clm.health import doctor


def _sanitise(text: str) -> str:
    text = text.replace(str(Path.home()), "~")
    for host in sorted({socket.gethostname(), socket.getfqdn()}, key=len, reverse=True):
        short = host.split(".", 1)[0]
        if host and host != short:
            text = text.replace(host, short)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="the results JSON to write")
    args = ap.parse_args(argv)
    cfg = load_config()
    started = dt.datetime.now(tz=dt.UTC)
    t0 = time.monotonic()
    rep = doctor(cfg, quick=False, serve=True)
    payload = {
        "format": "mesa-clm/doctor-record/1",
        "command": "doctor(cfg, quick=False, serve=True) == `mesa-clm doctor --serve`",
        "mesa_clm": __version__,
        "host": socket.gethostname().split(".", 1)[0],
        "started": started.isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - t0, 1),
        **rep.as_dict(),
    }
    text = _sanitise(json.dumps(payload, indent=1, ensure_ascii=False)) + "\n"
    keys = [k for k in (cfg.clm.resolved_api_key(), cfg.encoder.resolved_api_key()) if k]
    if any(k in text for k in keys):  # never written, never printed
        print("doctor_record: a key appeared in the report; nothing written", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    counts = rep.counts()
    print(f"wrote {args.out}: ok={rep.ok} ({counts['ok']} ok, {counts['warn']} warn, "
          f"{counts['fail']} fail) in {payload['seconds']} s")  # fmt: skip
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
