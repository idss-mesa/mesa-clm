# The encoder endpoint before and after the review of 70dbefe, sparky-1, 2026-10-01

Source: `serving_m1e.json` (this directory), two records written by `uv run python
scripts/endpoint_checks.py --foreign` (systemd's view of the units, `ss -ltne`, the proxy's
`/proc/<pid>/limits`, `status` and descriptors, `systemd-analyze --user security` of the proxy
unit, the run directory, the clients' port-owner check, `verify_serving_lock(require_live=True)`,
one keyed `GET /v1/models` while 1,000 idle connections are held, and one unauthenticated
`GET /health` each from uid 0 and uid 65534, in throwaway containers of the lock's pinned image
on the host network, and from this account; home directory as `~`; checked for both keys before
writing), one before and one after the restart. `encoder_fp` c3b3d5e1a283 and `lock_sha`
`dd33f9fe…` are unchanged; the doctor's record of the restarted units is `doctor_serve_m1e.json`.

| | Before (`before_restart`, 70dbefe) | After (`after_restart`) |
|---|---|---|
| `GET /health` from uid 0 / uid 65534 | 92 bytes back each (the encoder's `200 OK`) | nothing back: closed at once, one journal line for both (throttled) |
| The same request from this account | `200 OK` | `200 OK` |
| Proxy process | `NoNewPrivs` 0, `Seccomp` 0 | `NoNewPrivs` 1, `Seccomp` 2 (filter) |
| `systemd-analyze --user security` | 9.8 UNSAFE | 5.9 MEDIUM |
| Doctor `serving units` (new) | fail: the unit file is not the checkout's rendering, the proxy is not sandboxed | ok (`doctor_serve_m1e.json`) |
| 1,000 idle connections of this account, proxy descriptors | 2,007 | 2,009 |
| Keyed `GET /v1/models` while held | answered, 8.5 ms | answered, 7.7 ms |
| Listeners, run directory, lock problems, enabled at boot | 8090 and 8700 sockets of uid 1000, `encoder.sock` alone in a 0700 directory, none, none | the same |

**The restart, and an outage it caused.** `systemctl --user reset-failed` (the six units), then
`systemctl --user restart mesa-clm-encoder.service mesa-clm-serve.service` at 21:55:23Z. The
first sandbox of the proxy unit had the mount-namespace options (`ProtectSystem=`,
`ProtectHome=`, `PrivateTmp=`, `ProtectProc=`, `ProcSubset=` and others), validated beforehand
against a socket created on the host. In a rootless user manager those options imply
`PrivateUsers=`, and Ubuntu's AppArmor confines a process in an unprivileged user namespace
(`unprivileged_userns`): it denied every `connect()` of the proxy to the socket the container
had bound ("Failed name lookup - disconnected path", `run/mesa-clm/encoder.sock`, error -13; 284
kernel audit lines), so from 21:57:24Z, when the encoder served again, until 22:07:04Z :8090
closed every connection ("cannot reach …: Permission denied", 69 throttled proxy lines) and
clm-serve's wait for the encoder timed out once (22:05:25Z; systemd restarted it). The namespace
options were removed from the unit (what remains needs no namespace: `NoNewPrivileges=`, a
system-call allow list, `AF_UNIX` and `AF_NETLINK` only, no W+X memory, no realtime, no
set-uid, no new namespaces, a private keyring, `UMask=0077`), the unit reinstalled, and only the
proxy restarted (`systemctl --user restart mesa-clm-encoder-proxy.service`, 22:07:04Z); clm-serve
came up the same second. The encoder was not restarted again.

The proxy process recorded after the restart runs the copy installed at 21:54Z, which differs
from the committed `serving/encoder_proxy.py` (installed afterwards) only in comments and the
wording of the other-account log line; the committed copy takes effect at the next start of the
units. Not recorded live: the proxy's refusal of a symlinked `encoder.sock` and of a swap after
its check (`serving/tests/test_encoder_proxy.py`; DESIGN A5).
