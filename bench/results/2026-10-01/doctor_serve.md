# `mesa-clm doctor --serve` on the final recipe, sparky-1, 2026-10-01

Source: `doctor_serve.json` (this directory), written by `uv run python
scripts/doctor_record.py --out bench/results/2026-10-01/doctor_serve.json` (the doctor in
process with every check: `quick=False`, `serve=True`; home directory as `~`, host name as
`sparky-1`; checked for both bearer keys before writing) at 16:17:16Z, 4.6 s, against
`mesa-clm-encoder.service` and `mesa-clm-serve.service` on the recipe of DESIGN A3/A4
(`encoder_fp` c3b3d5e1a283, `lock_sha` `59625b8e…`), with no key configured: the serving pair's
keys came from the default files `~/.mesa/clm/secrets/{clm,encoder}.key`.

**Verdict: ok** (30 ok, 2 warnings, 0 failures): plan §8's M1-A gate "doctor green".

| Check | Result |
|---|---|
| Pins | vendored files 4/4, framings lock in sync (`b432d32a7536`), `schema.py` `52cec58afbf4` = vendored entry = serving pin, mesa-mcp `c74f3aa` and mesa-ducklake `7bc143f` pinned |
| Keys | `clm.key`, `encoder.key` used as the defaults (0600, not read by the doctor) |
| Serving lock | the checkout's lock, `59625b8efedb`: 11 checks ok, 0 skipped (installed copy, clone, applied patches, schema in clone and venv, head, bearer guard, image, running container) |
| Binds | 127.0.0.1:8090 and 127.0.0.1:8700 only |
| 401 matrix | encoder: 18 routes (vLLM's 17 besides `/health`, and an unknown path) answer 401 without a key; clm-serve: `/v1/models`, `/v1/systemone`, `/v1/rank` answer 401; `/health` 200 on both |
| Models | encoder `qwen3-8b` with the lock's model, window and route (`vllm`); clm-serve `clm-latest`, `clm-raw` |
| Encoder goldens | `encoder_golden_c3b3d5e1a283.npz`: **20/20 bitwise** one text per request; as one batch against one by one min cosine **1 − 3.6e-15** |
| Long input | an **8,741-token** text completed, **4,095** tokens charged (the A4 cap); cosine with the vector of its last 4,095 token ids **1.0000000** (gate 0.9999) |
| Systemone parity | the golden question on the local route against clm-serve: `clm-latest` max \|Δp\| **1.7e-8**, `clm-raw` **1.1e-5** (gate 1e-4) |
| Drift (warn) | quickstart urgency 0.8466 (\|Δ\| 0.0100116), billing 0.9903 (\|Δ\| 0.0014636), frustration 2.0000 (\|Δ\| 0.0000210) against issue #15 ± 0.01; tides top answer "The Moon's gravitational pull." 0.9932 against the model card's 0.993 ± 0.005 |
| Permissions (warn) | `~/.mesa/clm/locks` 0775 and `locks/provenance.lock` 0664, created before mesa-clm set modes explicitly; the fix the doctor prints is `chmod 700 ~/.mesa/clm/locks; chmod 600 ~/.mesa/clm/locks/provenance.lock` (the next sidecar operation also tightens both) |
| Host | MemAvailable 77.9 GiB; no CARC vLLM backend active |

The urgency warning is the one A3 records: the batch-invariant kernels move the quickstart's
noul answer by about 0.01 at CLM's scale 100, and `serving_m1b.json#/drift` holds the same
values for this `encoder_fp`, so the warning is the recipe, not upstream drift (DESIGN
implementation notes, "Upstream drift stays a warning"). The golden `/v1/systemone` call reports
1 ms and 0 input tokens because earlier doctor runs that day had left its texts in clm-serve's
caches.
