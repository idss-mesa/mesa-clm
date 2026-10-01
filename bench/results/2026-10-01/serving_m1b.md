# M1-B serving probes on the final recipe, sparky-1, 2026-10-01

Source: `serving_m1b.json` (this directory), written by `uv run python scripts/serving_probes.py
--mem-pre-start 107.46 --mem-encoder-only 80.03` (started 2026-10-01T15:21:31Z, 244.6 s) against
`mesa-clm-encoder.service` and `mesa-clm-serve.service` on the final recipe: vLLM 0.27.1, image
`vllm/vllm-openai@sha256:0a51ea5b…d967`, Qwen/Qwen3-8B `b968826d` bf16, `--runner pooling`,
`--max-model-len 4096`, `--max-num-seqs 8`, prefix caching off, **`VLLM_BATCH_INVARIANT=1`**,
**`--middleware vllm_auth.require_api_key`**, **`--disable-fastapi-docs`**, usage stats off;
clients truncating at **4,095** tokens (DESIGN A3, A4). `encoder_fp` **c3b3d5e1a283** (lock and
recomputed agree; was 852efc921a8a), `lock_sha` `59625b8e…e3cb`. The M1-A record of the previous
recipe is `bench/results/2026-09-29/serving_m1.md`.

| Check | Result | Gate | Verdict |
|---|---|---|---|
| `verify_serving_lock(require_live=True)` | no problems: lock, patch files, installed lock, clone, applied series, schema (clone, venv), head, **auth module** (0700/0600, sha256), image, **encoder container** (arguments, environment, read-only guard mount) | none | pass |
| Units | both `active`, both `disabled` (not enabled at boot) | not enabled | pass |
| Binds (`ss -Hltn`) | 127.0.0.1:8090, 127.0.0.1:8700 only; docker publishes 8090/tcp on 127.0.0.1 | loopback only | pass |
| Encoder route table | 19 routes from vLLM's own `build_app` for the locked arguments in a throwaway, network-less copy of the image (`serving/vllm_routes.py`): 18 `APIRoute` + the `/metrics` mount, **no WebSocket route**, no `/docs`/`/openapi.json`/`/redoc`; the guard is the outermost middleware; in-container check 18/18 guarded requests 401 | 401 off `/health` | pass |
| Encoder auth, live (:8090) | all 17 routes other than `/health` (`/load`, `/metrics`, `/ping` ×2, `/v1/models`, `/version`, `/detokenize`, `/invocations`, `/pooling`, `/rerank`, `/score`, `/tokenize`, `/v1/embeddings`, `/v1/rerank`, `/v1/score`, `/v2/embed`, `/v2/rerank`) → **401** with no key, a wrong key and clm-serve's key; with the key 200 (`/v2/embed` 400 on the probe body: it reached the handler); `/health` 200 | 401 | pass |
| Unknown paths | `/nonexistent`, `/metrics/anything`, `/docs`, `/openapi.json`, `/redoc`, `/health/` → 401 without a key | 401 | pass |
| Docker-bridge address of the container (not recorded) | reachable from any local account, as M1-A's `ss` view could not show: `/health` 200, `/version` and `/pooling` 401 without a key, `/v1/models` 200 with it | key everywhere but `/health` | pass (the guard is the control) |
| clm-serve (:8700) | `/v1/models`, `/v1/systemone`, `/v1/rank` 401 with no, wrong and the encoder's key, 200 with the right key; `/health` 200; **`/docs`, `/openapi.json`, `/redoc` 200 without a key** (schema only) | 401 on `/v1` | pass; finding |
| Models | encoder `qwen3-8b` (max_model_len 4096); clm-serve `clm-latest`, `clm-raw`, arena on CPU 536.9 MB | listed | pass |
| ~5,000-token text (5,005 tokens) through `EncoderClient` | completes in **0.98 s**, 4,095 tokens charged; cos with the embedding of the last 4,095 token ids **1.0000000** (to 9 decimals); cos with the first 4,095 ids 0.8127 | ≥ 0.9999 | pass |
| Same text as a clm-serve `/v1/systemone` state (cold, never cached) | completes in **1.08 s** (server 1,082.6 ms), 4,141 input tokens (the 4,095-token state and the candidates); probabilities vs the local route max abs diff 9.2e-10 | completes | pass |
| Exactly 4,096 token ids (last section) | still **times out** at 30.0 s; 5 s later the engine shows 0 running and 0 waiting requests (aborted on disconnect); a 4,095-id request then answers 200 in 1.03 s | clients never send it | recorded |
| Golden determinism (20 texts) | one text per request twice: **20/20 bitwise**; the 20 as one batch vs one by one: min cos 1.0 (16/20 bitwise); batch vs batch repeat 1.0 (18/20 bitwise); against the M1-A goldens (852efc921a8a): min cos 0.9998646 | ≥ 0.9999 | pass; written to `.local/serving/encoder_golden_c3b3d5e1a283.npz` |
| clm-serve `/v1/systemone` vs the local route, 50 pairs (30 term.fits F7, 10 F1 noul, 10 ontology_fits F7) | clm-latest max abs prob diff **3.9e-6** (50/50 top); clm-raw **6.6e-5** (50/50) | ≤ 1e-4 | **pass** (M1-A: 0.0434 / 0.0591, fail) |
| Local route against itself | clm-latest 4.7e-8, clm-raw 3.9e-7 | — | recorded |
| Head projection alone (numpy `HeadProjector` vs CLM torch `HeadPair`, 50 questions / 159 candidates) | max abs projection diff 5.7e-7, prob diff 4.6e-6; `weights_only=True` load ok | ≤ 1e-5 | pass |
| Quickstart drift (README at bb42c6c5, verbatim) | urgency **0.8466** (unrounded \|Δ\| **0.0100116**), billing 0.9903 (\|Δ\| 0.0015), frustration 2.0000 | #15: 0.8366 / 0.9888 / 2.0000 ± 0.01 (warn) | urgency **outside by 1.2e-5**; the rest within |
| Tides rank (model card) | "The Moon's gravitational pull." **0.9932** | 0.993 ± 0.005 | pass |
| Latency, rank-fit Choice (observerDistance target_state, 371 tokens; 12 PATO candidates + `__none__`), 30 repeats | warm p50 **1.6 ms** / p95 2.0; cold p50 **116.0 ms** / p95 119.5 | — | recorded (M1-A 110.1 cold) |
| Latency, `/v1/embeddings` batch of 32 contexts (14,004 tokens), 30 repeats | p50 **3,273.5 ms** / p95 3,285.0 | ≤ 4× M1-A's 2,820 | recorded (1.16×) |
| Footprint | MemAvailable 107.46 GiB stopped → 80.03 encoder only → 78.31 now: Δ both units **29.15 GiB**, encoder 27.43, clm-serve 1.72; `nvidia-smi` VLLM::EngineCore **25,153 MiB** (24.56 GiB); `docker stats` 3.09 GiB; clm-serve VmRSS 1.19 GiB (1,250,852 kB; VmHWM 1.20 GiB, unit MemoryPeak 1.05 GB) | vLLM ≤ 24.3 GiB | **above by 0.26 GiB** by `nvidia-smi`; vLLM's own accounting is its 0.20 share |
| Encoder log | model load 14.11 GiB in 82.1 s; KV cache 9.87 GiB; free on startup 104.87/121.69 GiB; `seq_pooling_type='LAST'`, `use_activation=True`; `enable_prefix_caching=False` | — | recorded |

Reading the footprint: vLLM sizes its KV cache at start-up so that weights, non-torch memory,
peak activation and KV cache fill its 0.20 share (the log's "Desired GPU memory utilization is
(0.2, 24.34 GiB)", recorded in `encoder_logs.json`); this start chose 9.87 GiB of KV cache where M1-A's measured start chose 9.49
GiB, and `nvidia-smi` also counts the process's CUDA context, which vLLM's accounting leaves out.
Pinning `--kv-cache-memory` would fix the total; it is listed as an open item, not changed here.
**Resolved by DESIGN A5** (`serving_m1c.md`): with `--kv-cache-memory-bytes 4831838208` (8
sequences of the whole window) the encoder measures 19,609 MiB by `nvidia-smi` and 21.45 GiB by
the MemAvailable difference, both under the 24.3 GiB gate; the same restart took the container off
the docker network (no bridge address), and per-card latencies are `annotate_latency.md`.

Reading the drift row: the quickstart's urgency is a noul answer at CLM's scale 100, so a
kernel-level change of the vectors (the batch-invariant matmul moves vectors by up to 6.3e-4 in
cosine, `batch_invariance.json#/secondary`) moves it by about 0.01, the size of the tolerance; the
same probe gave 0.8449 in M1-A and 0.8366 today under the old kernels
(`batch_invariance.json#/arms/B0/drift`). The probe is the doctor's warn-level upstream-drift
check (plan §6.8); DESIGN A3 records how it is read.
