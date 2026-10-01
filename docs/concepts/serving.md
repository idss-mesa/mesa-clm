---
title: "Serving"
description: "The mesa-clm serving stack of milestone M1: a batch-invariant vLLM pooling container serving Qwen3-8B behind a bearer guard and a patched clm-serve on loopback with keys, the carried CLM patches, fingerprints and the serving lock, what the doctor probes, the GPU budget, and the in-process fallback encoder."
type: Guide
tags:
  - concepts
  - serving
  - vllm
  - clm
  - deploy
generated:
  by: "claude/fable-5.1"
  at: "2026-10-01T18:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md)"
    author: "team:idss-mesa"
  - id: clm
    resource: "https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
    title: "Contrastive-LM/CLM at bb42c6c5 (server.py, embedder.py, heads.py)"
    author: "org:Contrastive-LM"
status: draft
stale_after: "2026-12-31T00:00:00Z"
---

# Serving

Milestone M1 (track A) brings the stack up on the serving host (sparky-1); the runbook is
[Serving](../deploy/serving.md). The core never imports torch: it talks to two loopback HTTP
services (DESIGN D16), and the serve-venv side lives under `serving/` and `deploy/` in the
repository and never imports `mesa_clm`. `mesa-clm annotate --provider clm` uses both
([Install](../getting-started/install.md)).

## Topology

| Process | Port | What | Unit |
|---|---|---|---|
| Encoder | `127.0.0.1:8090` (a socket unit) | `vllm/vllm-openai` v0.27.1 (pinned by digest) running `Qwen/Qwen3-8B` at the pinned revision with `--runner pooling --dtype bfloat16 --max-model-len 4096 --max-num-seqs 8 --no-enable-prefix-caching --gpu-memory-utilization 0.20 --kv-cache-memory-bytes 4831838208 --enforce-eager`, vLLM's batch-invariant kernels (`VLLM_BATCH_INVARIANT=1`, DESIGN A3), the bearer guard `--middleware vllm_auth.require_api_key` mounted read-only and `--disable-fastapi-docs` (DESIGN A4), usage reporting off, offline HF cache mounted read-only, `VLLM_API_KEY` from an env file; the container has **no network** and serves on the unix socket `~/.mesa/clm/run/encoder.sock`, which `mesa-clm-encoder-proxy.socket` exposes on 127.0.0.1:8090 through `mesa-clm-encoder-proxy` (`serving/encoder_proxy.py`: a socket of this account only, never through a symlink, at most 4,096 connections; DESIGN A5) | `mesa-clm-encoder.service` (user unit, `sg docker`), which requires the proxy socket (a port it cannot bind stops the encoder and clm-serve from starting) and starts the headroom timer |
| clm-serve | `127.0.0.1:8700` | CLM `bb42c6c5` with patches 0001–0006 in its own venv (`~/.mesa/clm/serve/.venv`, uv CPython 3.11, torch CPU, **no vllm**), `--device cpu --action-cache 512MiB --max-tokens 4095 --no-download --no-ui`, the pinned head `CLM_v0.1-8B.pt` and promoted heads from `heads/served/` | `mesa-clm-serve.service` (`Requires=` the encoder) |

Both bind loopback only and require bearer keys; the encoder answers nothing but `/health`
without its key, unknown paths included (DESIGN A4: vLLM's own key check covers four path
prefixes, and in M1-A `/pooling`, `/invocations`, `/score`, `/rerank`, `/detokenize`,
`/tokenize`, `/metrics` and `/version` answered without a key,
`bench/results/2026-09-29/serving_m1.json`). Under DESIGN A4 the container sat on the docker
bridge, where its port, and seven sockets of the engine process that no key covers (the
rendezvous store and six collective sockets), were reachable from any local account
(`bench/results/2026-10-01/serving_m1b.json`, `encoder_netns.json`); since DESIGN A5 the
container has no network at all, the API is a unix socket in an owner-only directory, and the key
is the control on the one loopback port that reaches it (`serving_m1c.json`). clm-serve's
`/health` (which lists the head names)
and its FastAPI schema pages answer without a key. Every client truncates its inputs to 4,095
tokens from the left (`max_len - 1`): vLLM 0.27.1 never completes an input of exactly 4,096
tokens (DESIGN A4).

`mesa-clm serve keys --init` writes the raw 0600 key files
`~/.mesa/clm/secrets/{clm,encoder}.key` (missing ones only; idempotent) and derives the units'
env files (`clm.env` with `CLM_API_KEY`, `CLM_EMB_API_KEY` and `CLM_EMB_CACHE_SIZE`;
`encoder.env` with `VLLM_API_KEY`); `--rotate` replaces both keys and prints the
`systemctl --user restart mesa-clm-encoder.service mesa-clm-serve.service` command to run, since
the running units keep the old keys until restarted. Neither ever prints a key; the core reads
those two files by default (or `MESA_CLM_CLM__API_KEY_FILE` and
`MESA_CLM_ENCODER__API_KEY_FILE` when set). `mesa-clm serve units` prints the rendered user units
(the default rendering is committed under `deploy/systemd/`); installing and enabling them is a
user action. The core refuses a non-loopback URL unless `MESA_CLM_CLM__ALLOW_REMOTE=1` **and**
the scheme is https; DE VMs reach the stack through an `ssh -L` forward or a GPU VM of their own
(0.1.0 annotate runs on the serving host). A timer runs the headroom check every 5 minutes
(`MemAvailable` at least 8 GiB; the encoder needs 32 GiB to start); the spool recorder's timer
arrives with the history backends in M3. Heads on CPU (DESIGN D17): the two heads are 18.9M
parameters, and the encoder gets the whole 0.20 GPU share (about 24.3 GiB on the 121.7 GiB
unified-memory GB10). Without a pin vLLM sized its KV cache at each start to fill the share
(25,153 MiB by `nvidia-smi`, `serving_m1b.json`); DESIGN A5 pins it to the 4.5 GiB that 8
sequences of 4,096 tokens need, and the encoder now measures 19,609 MiB by `nvidia-smi` and
21.45 GiB by the `MemAvailable` difference (`bench/results/2026-10-01/serving_m1c.json`). The
encoder is started at most three times an hour, manual starts included, so a recipe that fails
after its model load does not reload it on the shared GPU every few minutes. A loopback port is
shared by every account on the host and free while the units are down, so every mesa-clm
client that sends a key first checks that the socket holding the port belongs to this account
and refuses another's (DESIGN A5).

## Patches carried

None touches the vendored `schema.py` or `client.py`; their sha256 are pinned in
`serving/serving.lock.json` (RESEARCH.md, issues and pull requests).

| Patch | Upstream | Change |
|---|---|---|
| 0001 | PR #6 | `truncation_side: "left"` on every embedding request (vLLM ≥0.18 honours it; pooling truncates from the right by default and would drop the tail) |
| 0002 | PR #11 | loopback default; refuse an exposed unkeyed server |
| 0003 | PR #10 | `torch.load(..., weights_only=True)` |
| 0004 | PR #23 | vector-arena slot leak on concurrent misses |
| 0005 | local | Embedder reads `CLM_EMB_API_KEY` and `CLM_EMB_CACHE_SIZE` (the upstream host LRU is 200,000 vectors, about 3.3 GB) |
| 0006 | local | constant-time API-key comparison |

## Fingerprints and the lock

`encoder_fp = sha256{model, revision, dtype, pooling: "LAST", normalize, max_len,
truncation_side: "left", prefix_caching, route[, batch_invariance]}[:12]`, `clm_model_fp`,
`schema_sha256` and `serving_lock_sha` are stamped on every decision, feature row, artifact and
bench cell and must match exactly (DESIGN D5): CLM heads are encoder-locked, the announced
CLM-35B may change the encoder, prefix caching is off until golden-vector parity with it on has
been recorded, and a vector's batch was found to move it (DESIGN A3: `batch_invariance` is
`kernels` for the vLLM route, `encoder_fp` **c3b3d5e1a283**, and `serial` for the fallback at
batch 1, b171b1ba4536; it is hashed only when not `none`). `serving/serving.lock.json` pins the
CLM commit, the patches and their sha256, the head (repository, revision, file, size, sha256),
the encoder recipe, the image digest and the container `recipe` (the whole `vllm serve` argument
list, the non-secret environment, the bearer guard with its sha256 and read-only mount, the
4,095-token cap); it is self-verifying (`lock_sha` over its content, `encoder_fp` over its
encoder block). The live provider stamps every record with the fingerprint of the lock the host
runs (the installed copy `~/.mesa/clm/serving.lock.json`, else the checkout's) and refuses to
start on a lock that does not verify, whose `schema_sha256` is not the vendored `schema.py`, or,
when both exist, an installed copy that differs from the checkout's (re-run the bootstrap). Its
pre-flight then checks the keys, the served models and the encoder's model, window and route
(`owned_by`: `vllm` for the image, `mesa-clm-fallback` for the fallback) against the lock. An
artifact or bench cell under another fingerprint is refused until the task is re-benched (K4).

## What the doctor checks

`mesa-clm serve lock --check` (and the doctor's `serving lock` line, whenever `~/.mesa/clm`
exists) compares the lock with the host: the patch files in the checkout, the installed lock
copy, the serve clone (commit, applied patch series, `schema.py`), the serve venv's `clm.schema`,
the head file (size and sha256), the installed bearer guard (owner-only, its sha256), the pinned
image and the running encoder container (image, arguments, the locked environment values read
through a `docker inspect` template that never asks for the key, the guard's read-only mount, the
network mode and the socket directory; through `sg docker` when the session lacks the docker
group). Absent pieces are skipped unless `--require-live`; present and different is a failure.
The same container check runs before `features build` stores a vector and before an annotate run:
the encoder's `/v1/models` reads the same for every container of the image, whatever its kernels
or flags, so a container that departs from the lock is refused there too, and a check docker
could not answer is reported as "not verified".

`mesa-clm doctor --serve` (the light half automatic when both units are active) adds the live
probes. Every URL first passes the clients' loopback rule and is printed redacted. Then:
the **binds** of both ports (`ss -ltne`: loopback only, sockets of this account, and while the
socket unit is active the encoder's port in the user manager's cgroup); the **encoder socket**
(`~/.mesa/clm/run` holds nothing but a socket of this account, never a symlink); the **encoder
network**: the container's own namespace read from `/proc/<pid>/net`, which `ss` cannot see,
where a listener on a non-loopback address next to an interface besides `lo` fails (DESIGN A5); the headroom timer active (a
warning otherwise); `/health` 200 on both without a key; the **401
matrix**: without a key, every route vLLM registers on the encoder (17 besides `/health`) and an
unknown path, and clm-serve's `/v1/models`, `/v1/systemone` and `/v1/rank`, must answer 401;
the authenticated model lists (`qwen3-8b` with the lock's model, window and route on the
encoder, `clm-latest` and `clm-raw` on clm-serve), so a key that differs from the unit's shows
up as a 401; and one golden `/v1/systemone` question whose answer must cover every key with
probabilities above 0 summing to 1. Under `--serve` (or serve mode without `--quick`) the slow
probes follow: the **encoder goldens** (`encoder_golden_<encoder_fp>.npz`: each text one per
request bitwise equal to the reference, the texts as one batch within 1 − 1e-6 of one by one),
the **long-input probe** (a text over the window completes, truncated to 4,095 tokens, with
cosine ≥ 0.9999 against the vector of its last 4,095 token ids), the **systemone parity** (the
golden question recomputed on the local route, encoder plus the pinned head's numpy export,
within 1e-4 of clm-serve for `clm-latest` and `clm-raw`) and the **upstream drift** probes, a
warning only (the CLM README quickstart against issue #15 with the unrounded differences, the
tides rank against the model card). `host`, `gpu_budget` and `permissions` report
`MemAvailable`, any active CARC vLLM backend and group or other bits on mesa-clm's files, and
`feature store` the store of the live `encoder_fp` (its format and the vector recipe it was
built under against the live lock's). On the DESIGN A5 recipe the doctor is green with one
warning, the quickstart urgency of DESIGN A3 (`bench/results/2026-10-01/doctor_serve_m1c.json`).

## Fallback encoder

`serving/fallback_serve.py` loads `Qwen/Qwen3-8B` bf16 with transformers (left truncation,
`last_hidden_state` at the last token, L2-normalised; one sequence per forward pass, so no
padding), runs an in-process `Engine(embedder=CudaEncoder)` on :8700 and a second server
exposing `/v1/embeddings`, `/tokenize`, `/v1/models` and `/health` on :8090 behind the same rule
as the container (the key on every route but `/health`). `route: transformers` and
`batch_invariance: serial` enter `encoder_fp`. Plan §6.6's sequential parity run (minimum
cosine ≥0.999, mean ≥0.9999 against the vLLM route) failed at batch 8 in M1-A (min 0.998975,
`bench/results/2026-09-29/fallback_parity.json`) and passes its two cosine gates at batch 1 on
the A3/A4 recipe (min 0.999260, mean 0.999919, `bench/results/2026-10-01/fallback_parity.json`;
the vectors are unchanged under A5); the third condition, probe agreement ≥99%, has nothing to
measure before M4, and the routes' `/v1/systemone` answers still differ by a few hundredths, so
nothing may be shared between the routes yet. Artifacts stay keyed by `encoder_fp`, which differs
between them. The server refuses `--batch` other than 1, which its fingerprint does not name. The fallback never runs
alongside the vLLM container. If the GPU share is unavailable (the co-tenant plan for this host
sums to 0.83), the headroom check fails, the doctor reports `gpu_budget`, both units stop and
the tools return `decider_unavailable`; the bench is unaffected because features are cached.
