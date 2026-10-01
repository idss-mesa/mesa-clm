# M1-A serving probes, sparky-1, 2026-09-29

Source: `serving_m1.json` (this directory), written by `uv run python scripts/serving_probes.py`
(started 2026-09-29T19:06:06Z, 303.5 s) against `mesa-clm-encoder.service` (vLLM 0.27.1,
image `vllm/vllm-openai@sha256:0a51ea5b…d967`, Qwen/Qwen3-8B `b968826d` bf16, `--runner pooling`,
`--max-model-len 4096`, `--gpu-memory-utilization 0.20`, prefix caching off) and
`mesa-clm-serve.service` (patched clm-serve, heads on CPU). `encoder_fp` 852efc921a8a (lock and
recomputed agree); image digest matches the lock; both units `disabled` (not enabled at boot).

| Check | Result | Gate | Verdict |
|---|---|---|---|
| Binds (`ss -Hltn`) | 127.0.0.1:8090, 127.0.0.1:8700 only; docker publishes 8090/tcp on 127.0.0.1 | loopback only | pass |
| `GET/POST /v1/models`, `/v1/embeddings` (:8090), `/v1/models`, `/v1/systemone`, `/v1/rank` (:8700) | 401 with no key, a wrong key and the other unit's key; 200 with the right key | 401 | pass |
| Open by design | `/health` (both), `/metrics`, `/version`, `/tokenize` (:8090) 200 without a key | loopback | as expected |
| `/tokenize` under `--runner pooling` | works (200), **not guarded** by the key | — | recorded |
| Other :8090 routes without a key | **`POST /pooling`, `/invocations`, `/score`, `/rerank`, `/detokenize` → 200** (embeddings and scores served unauthenticated); `/v1/score`, `/v2/embed` → 401 | not in plan §6.8 | **finding** |
| Models | encoder `qwen3-8b` (root Qwen/Qwen3-8B, max_model_len 4096); clm-serve `clm-latest`, `clm-raw` | listed | pass |
| Quickstart drift (README at bb42c6c5, verbatim) | urgency **0.8449**, department billing **0.9878**, frustration **2.0000** | #15: 0.8366 / 0.9888 / 2.0000 ± 0.01 (README prints 0.41022 / 0.93878 / 1.98386) | pass vs #15 |
| Tides rank (model card) | "The Moon's gravitational pull." **0.9939** | 0.993 ± 0.005 (README 0.997) | pass |
| Left truncation, ~5,005-token text | cos(left-truncated, last 4095 token ids) **1.0000000**; right (default) truncation = first 4095 ids (cos 1.0), cos(right, left) 0.8107 | ≥ 0.9999 | pass at 4095 |
| **Input of exactly 4096 tokens** (= max-model-len) | **never completes** (2/2 token-id requests and the 4096 left-truncated text request timed out at 30/60 s; 2048, 2049, 4095 tokens answer in 0.44–0.91 s; no truncation → 400 in 0.02 s) | — | **finding** |
| Golden vectors (20 texts, `.local/serving/encoder_golden.npz`) | one text per request: bitwise identical on repeat (20/20); batch-of-20 vs one-by-one min cos 0.9999507; batch vs batch repeat min cos **0.9998659** | ≥ 0.9999 | pass only one text per request |
| Head projection alone (same vectors, numpy `HeadProjector` vs CLM torch `HeadPair`, 50 questions / 159 candidates) | max abs projection diff **6.3e-7**, max abs prob diff 3.8e-6; `weights_only=True` load ok | ≤ 1e-5 | pass |
| clm-serve `/v1/systemone` vs local route, 50 pairs (30 term.fits F7, 10 F1 noul, 10 ontology_fits F7) | clm-latest max abs prob diff **0.0434** (mean 0.0056, top choice 49/50); clm-raw **0.0591** (49/50) | ≤ 1e-4 | **fail** |
| Same local route run twice (encoder noise floor) | clm-latest max 0.0248, clm-raw max 0.0363 | — | explains the fail |
| Footprint | MemAvailable 78.15 GiB now vs 107.8 pre-start (Δ 29.65 GiB) and 80.3 encoder-only (encoder Δ 27.5 GiB); `nvidia-smi` VLLM::EngineCore **24,799 MiB**; `docker stats` 4.79 GiB; clm-serve VmRSS **1.20 GiB** (VmHWM 1.20 GiB, unit MemoryPeak 1.06 GB) | vLLM ≤ 24.3 GiB | GPU share within; host Δ above |
| Encoder log | model load 14.11 GiB in 66.0 s; KV cache 9.49 GiB (69,072 tokens); free on startup 105.2/121.69 GiB; `seq_pooling_type='LAST'`, `use_activation=True`; `enable_prefix_caching=False` | — | recorded |
| Latency, rank-fit Choice (observerDistance target_state, 371 tokens; 12 PATO candidates + `__none__`), 30 repeats | warm (same state) p50 **1.0 ms** / p95 **1.6 ms**; cold (new state each time, candidates cached) p50 **110.1 ms** / p95 **111.6 ms** (server header 108.6 / 110.1) | — | recorded |
| Latency, `/v1/embeddings` batch of 32 contexts (14,004 tokens), 30 repeats | p50 **2,820 ms** / p95 **2,841 ms** | — | recorded |

Reading the parity failure: the head is exact (6.3e-7), but the vLLM encoder is not
batch-invariant. A text embedded in different batch compositions moves by up to ~1.3e-4 in
cosine, and CLM's scale of 100 turns that into probability differences of a few hundredths.
clm-serve and the local route batch their texts differently, so the ≤ 1e-4 systemone gate cannot
hold end to end on this configuration. It holds on identical vectors. The doctor's golden check
needs one text per request, which is deterministic here.
