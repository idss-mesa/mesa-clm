# `mesa-clm doctor --serve` after the review of 70dbefe, sparky-1, 2026-10-01

Source: `doctor_serve_m1e.json` (this directory), written by `uv run python
scripts/doctor_record.py --out bench/results/2026-10-01/doctor_serve_m1e.json` (the doctor in
process with every check, `quick=False`, `serve=True`; home directory as `~`, host name as
`sparky-1`; checked for both bearer keys before writing) at 22:08:20Z, 29.5 s, after the restart
of the units for the review of 70dbefe (21:55:23Z, and the proxy alone at 22:07:04Z:
`serving_m1e.md`; `encoder_fp` c3b3d5e1a283 and `lock_sha` `dd33f9fe…` unchanged). The previous
record is `doctor_serve_m1d.md`; the endpoint itself, before and after the restart, is
`serving_m1e.json`.

**Verdict: ok** (36 ok, 1 warning, 0 failures).

| Check | Result |
|---|---|
| Serving lock | `dd33f9fedbae`: 11 checks ok, 0 skipped (the running container against the recipe) |
| Binds | loopback only, sockets of uid 1000; the encoder's 127.0.0.1:8090 held by the user manager's `mesa-clm-encoder-proxy.socket` |
| Encoder socket | `~/.mesa/clm/run/encoder.sock`: a socket of uid 1000, alone in a 0700 directory |
| Serving units (new) | the six units systemd loaded are the checkout's rendering for `~/.mesa/clm`; the running proxy has the rendered command line and its sandbox (`NoNewPrivs`, seccomp). Run alone before the restart it failed (`serving_m1e.json#/before_restart/doctor_serving_units`) |
| Encoder network | the container's namespace has `lo` only; 7 listening sockets, all inside it |
| 401 matrix | encoder: 18 routes answer 401 without a key (an unknown path too); clm-serve: 3 routes |
| Golden `/v1/systemone` | the non-bench SRER card's `staPresMean` (DESIGN, "G1 freeze", item 7): `clm-latest` answered `UO:0000110` (8,895 ms, the first request after the restart), never set against a label |
| Encoder goldens | 20/20 bitwise one text per request; one batch against one by one min cosine 1 − 3.6e-15 |
| Long input | an 8,741-token text completed, 4,095 tokens charged; cosine with its last 4,095 token ids 1.0000000 |
| Systemone parity | the golden question against the local route: `clm-latest` max \|Δp\| 2.2e-9, `clm-raw` 1.0e-7 (gate 1e-4) |
| Drift (warn) | quickstart urgency 0.8466 (\|Δ\| 0.0100116) against issue #15's 0.8366 ± 0.01: the warning DESIGN A3 records for this `encoder_fp` |
| Host | MemAvailable 84.9 GiB; no CARC vLLM backend active |
