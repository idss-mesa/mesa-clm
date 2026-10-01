# The encoder endpoint before and after the DESIGN A5 revision, sparky-1, 2026-10-01

Source: `serving_m1d.json` (this directory), two records written by `uv run python
scripts/endpoint_checks.py` (systemd's view of the units, `ss -ltne`, the proxy's
`/proc/<pid>/limits` and descriptors, the run directory, the clients' port-owner check,
`verify_serving_lock(require_live=True)`, and one keyed `GET /v1/models` while idle connections
are held open; home directory as `~`; checked for both keys before writing), one before and one
after the single restart of the units at 20:25:21Z (`systemctl --user reset-failed
mesa-clm-encoder.service`, then `systemctl --user restart mesa-clm-encoder.service
mesa-clm-serve.service`, back at 20:27:25Z). `encoder_fp` c3b3d5e1a283 and `lock_sha`
`dd33f9fe…` are unchanged; the doctor's record of the restarted units is
`doctor_serve_m1d.json`.

| | Before (`before_restart`) | After (`after_restart`) |
|---|---|---|
| Encoder unit | `Wants=mesa-clm-encoder-proxy.socket mesa-clm-headroom.timer`, no `Requires=`/`After=` on the socket | `Requires=` and `After=` `mesa-clm-encoder-proxy.socket`, `Wants=mesa-clm-headroom.timer` |
| Proxy | `systemd-socket-proxyd`, `Type=notify`, `RLIMIT_NOFILE` 1,024 soft | `mesa-clm-encoder-proxy --connections-max 4096` under the serve venv's Python (`-I`), `LimitNOFILE=16384` |
| Listeners | 8090: uid 1000, `user@1000.service/init.scope`; 8700: uid 1000, `mesa-clm-serve.service` | the same |
| Run directory | 0700, `encoder.sock` only, a socket of this account | the same |
| Idle connections held, proxy descriptors | 100 → 609 (six per connection) | 1,000 → 2,009 (two per connection) |
| Keyed `GET /v1/models` while held | answered, 6.0 ms | answered, 9.1 ms |
| Lock problems (`require_live`) | none | none |
| Enabled at boot | none (`disabled`/`static`) | none |

At the user manager's soft limit of 1,024, `systemd-socket-proxyd`'s six descriptors per
connection run out at about 170 idle connections, which any local account can hold (the finding
of the pre-merge review; vLLM's server never closes a connection that sends nothing); the new
proxy holds 4,096 under its limit and closes a connection after 900 s without traffic. The proxy
process recorded here started before the listen-backlog fix of `serving/encoder_proxy.py`, so
127.0.0.1:8090 has asyncio's backlog of 100 instead of the socket unit's 4,096 until the units
next start; the installed copy has the fix. Not recorded live, because it would mean tampering
with the running container's directory: the proxy's refusal of a symlinked `encoder.sock`, which
`serving/tests/test_encoder_proxy.py` covers and which a scratch setup on this host showed for
both proxies (`systemd-socket-proxyd` reached another socket through the symlink,
`mesa-clm-encoder-proxy` reset the connection; DESIGN A5).
