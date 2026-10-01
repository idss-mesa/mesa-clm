# `mesa-clm doctor --serve` on the DESIGN A5 recipe, sparky-1, 2026-10-01

Source: `doctor_serve_m1c.json` (this directory), written by `uv run python
scripts/doctor_record.py --out bench/results/2026-10-01/doctor_serve_m1c.json` (the doctor in
process with every check, `quick=False`, `serve=True`; home directory as `~`, host name as
`sparky-1`; checked for both bearer keys before writing) at 18:34:41Z, 4.7 s, after the one
restart of the units for DESIGN A5 (`encoder_fp` c3b3d5e1a283, `lock_sha` `dd33f9fe…`), the
feature store's rebuild in format 2 and the `chmod` of `~/.mesa/clm/locks`. The A3/A4 recipe's
record is `doctor_serve.md`.

**Verdict: ok** (34 ok, 1 warning, 0 failures).

| Check | Result |
|---|---|
| Pins | vendored files 4/4, framings lock in sync (`b432d32a7536`), `schema.py` `52cec58afbf4` = vendored entry = serving pin, mesa-mcp `c74f3aa` and mesa-ducklake `7bc143f` pinned |
| Feature store (new) | `~/.mesa/clm/features/c3b3d5e1a283/features.duckdb`: `mesa-clm-features/2`, 1,563 vectors, 0 truncated texts, vector recipe `ed486dcd74a3` = the live lock's, fp16 round trip 0.9999999 |
| Permissions | **ok**: mesa-clm's state is owner-only (`~/.mesa/clm/locks` 0700, `provenance.lock` 0600 since the operator `chmod` at 18:17Z) |
| Serving lock | `dd33f9fedbae`: 11 checks ok, 0 skipped (the running container against the recipe's arguments, environment, mounts and network) |
| Binds | 127.0.0.1:8090 and 127.0.0.1:8700 only |
| Encoder network (new) | the container's namespace has `lo` only; 7 listening sockets, all inside it |
| Headroom timer (new) | `mesa-clm-headroom.timer` active (it starts and stops with the encoder unit) |
| 401 matrix | encoder: 18 routes (vLLM's 17 besides `/health`, and an unknown path) answer 401 without a key; clm-serve: `/v1/models`, `/v1/systemone`, `/v1/rank` answer 401; `/health` 200 on both |
| Models | encoder `qwen3-8b` with the lock's model, window and route (`vllm`); clm-serve `clm-latest`, `clm-raw` |
| Encoder goldens | `encoder_golden_c3b3d5e1a283.npz`: 20/20 bitwise one text per request; as one batch against one by one min cosine 1 − 3.6e-15 |
| Long input | an 8,741-token text completed, 4,095 tokens charged; cosine with its last 4,095 token ids 1.0000000 |
| Systemone parity | the golden question against the local route: `clm-latest` max \|Δp\| 1.7e-8, `clm-raw` 1.1e-5 (gate 1e-4) |
| Drift (warn) | quickstart urgency 0.8466 (\|Δ\| 0.0100116) against issue #15's 0.8366 ± 0.01; billing and frustration within; tides 0.9932 within 0.993 ± 0.005: the warning DESIGN A3 records for this `encoder_fp` |
| Host | MemAvailable 85.1 GiB; no CARC vLLM backend active |
