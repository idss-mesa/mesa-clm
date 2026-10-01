# Serving probes on the DESIGN A5 recipe, sparky-1, 2026-10-01

Source: `serving_m1c.json` (this directory), written by `uv run python scripts/serving_probes.py
--mem-pre-start 107.46 --mem-encoder-only 86.01 --out bench/results/2026-10-01/serving_m1c.json
--golden-out .local/serving/m1c/encoder_golden_c3b3d5e1a283.npz --pairs-out
.local/serving/m1c/systemone_pairs.json` (started 18:30:36Z, 216.3 s); the start-up log lines
are `encoder_logs.json`, the container's network namespace before and after the restart
`encoder_netns.json` (bridge addresses masked). The recipe is A3/A4's
plus DESIGN A5: the encoder container runs with **`--network none`**, vLLM serves on the unix
socket `/run/mesa-clm/encoder.sock` (`~/.mesa/clm/run`, 0700), `mesa-clm-encoder-proxy.socket`
owns 127.0.0.1:8090 and `systemd-socket-proxyd` forwards to the socket; the engine's rendezvous
is on loopback (`VLLM_HOST_IP=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo`); the KV cache is pinned to
**`--kv-cache-memory-bytes 4831838208`** (8 sequences × 4,096 tokens × 144 KiB); the encoder
unit pulls in the endpoint and the headroom timer and starts at most three times an hour.
`encoder_fp` c3b3d5e1a283 (unchanged), `lock_sha` **`dd33f9fe…5237`** (was `59625b8e…`).

The units were restarted once for this (18:17:11Z stop, 18:18:00Z encoder start, 18:20:17Z
clm-serve start), installed from `deploy/systemd/` and not enabled at boot (encoder, serve and
timer `disabled`; the proxy socket and service have no install section, `static`).

| Check | Result | Gate | Verdict |
|---|---|---|---|
| `verify_serving_lock(require_live=True)` | no problems (11 checks; the container's arguments, environment, mounts and network mode against the lock) | none | pass |
| Encoder start log | `distributed_init_method=tcp://127.0.0.1:55793`; "reserved 4.5 GiB memory for KV Cache as specified by kv_cache_memory_bytes"; GPU KV cache **32,768 tokens**, maximum concurrency 8.00x at 4,096 tokens; "Starting vLLM server on **unix:/run/mesa-clm/encoder.sock**"; model load 14.11 GiB in 68.3 s (`encoder_logs.json`) | — | recorded |
| Container network namespace | before (A3/A4 recipe, `bridge`): interfaces eth0, lo; listening `0.0.0.0:8090`, six engine sockets on the bridge address and `[::]:53581`, flagged; after (`none`): interface **lo only**; six Gloo sockets on 127.0.0.1 and the rendezvous store on `[::]:55793` inside the loopback-only namespace; no problem | no listener reachable from the host | **pass** (the finding is closed) |
| Docker-bridge address | none: the container has no network and no IP address | — | pass |
| Binds (`ss -Hltn`) | 127.0.0.1:8090 (the socket unit) and 127.0.0.1:8700 only; `docker port` empty (nothing published) | loopback only | pass |
| Encoder route table and 401 matrix | 19 routes (vLLM's `build_app` for the locked arguments, in a throwaway network-less copy of the image), the guard outermost; on the live encoder all 17 routes besides `/health` and every unknown path answer **401** without a key, with a wrong key and with clm-serve's key; with the key 200 (`/v2/embed` 400 on the probe body); `/health` 200 | 401 off `/health` | pass |
| clm-serve (:8700) | `/v1/models`, `/v1/systemone`, `/v1/rank` 401 without, with a wrong and with the encoder's key; 200 with its key | 401 | pass |
| Encoder goldens | one text per request twice: 20/20 bitwise; batch against one by one 16/20 bitwise, min cos 1.0; against the **existing c3b3d5e1a283 reference** (`encoder_golden_c3b3d5e1a283.npz`, sha256 `b86cdd93…`): **20/20 bitwise** | bitwise | pass: the A5 recipe moves no vector |
| ~5,000-token text (5,005 tokens) | completes in 0.97 s, 4,095 tokens charged, cos 1.0000000 with its last 4,095 token ids; as a clm-serve state 0.98 s | completes, ≥ 0.9999 | pass |
| clm-serve `/v1/systemone` vs the local route, 50 pairs | `clm-latest` max \|Δp\| **3.9e-6**, `clm-raw` **6.6e-5** (50/50 top) | ≤ 1e-4 | pass |
| Footprint | MemAvailable 107.46 GiB with both units stopped, **86.01** with the encoder alone (Δ **21.45 GiB**; A3/A4 recipe 27.43), 84.79 with both (Δ 22.67; clm-serve 1.22); `nvidia-smi` VLLM::EngineCore **19,609 MiB** (19.15 GiB; was 25,153 MiB); `docker stats` 3.10 GiB; clm-serve VmRSS 1.18 GiB (1,235,712 kB) | vLLM ≤ 24.3 GiB | **pass by either measure** |
| Latency | rank-fit Choice (371-token state, 12 PATO candidates + anchor, `clm-latest`), 30 repeats: warm p50 1.5 ms / p95 1.7; cold p50 **118.9 ms** / p95 121.4; `/v1/embeddings` batch of 32 (14,004 tokens) p50 3,213.8 ms / p95 3,469.0 | — | recorded |
| Window boundary | exactly 4,096 token ids still times out at 30.1 s; 0 running and 0 waiting requests 5 s later; 4,095 ids then 200 in 1.02 s | clients never send it | recorded |
| Drift (warn) | quickstart urgency 0.8466 (\|Δ\| 0.0100116), billing 0.9903, frustration 2.0000; tides 0.9932 | issue #15 ± 0.01 (warn) | unchanged from `serving_m1b.json` |

Reading:

- The engine's rendezvous store and its six collective sockets have no key; on the docker bridge
  they were reachable from every local account (the finding). With no network namespace to reach
  them through they are unreachable, and the API itself is a unix socket in an owner-only
  directory behind the loopback socket unit. The bearer guard still answers 401 on every route
  but `/health`, now behind two boundaries instead of one.
- Pinning the KV cache to what `--max-num-seqs 8` at `--max-model-len 4096` needs (4.5 GiB, the
  figure plan §6.5's budget assumed) brings the encoder from about 25 GiB (vLLM fills its 0.20
  share, with a KV cache of 9.38-9.87 GiB that varies between starts, `encoder_logs.json`) to 19.2
  GiB by `nvidia-smi` and 21.45 GiB by the plan's own measure (MemAvailable before and after),
  both under the 24.3 GiB gate. vLLM's start-up check still requires 0.20 of the GPU free.
- Vectors are unchanged: the goldens and the 1,563 feature-store vectors are bitwise equal to the
  A3/A4 recipe's (`features_build_m1c.md`), so `encoder_fp` stays c3b3d5e1a283; only
  `serving_lock_sha` rotates (DESIGN A5).
