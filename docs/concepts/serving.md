---
title: "Serving"
description: "The mesa-clm serving stack planned for milestone M1: a vLLM pooling container serving Qwen3-8B and a patched clm-serve on loopback with keys, the carried CLM patches, fingerprints and the serving lock, the GPU budget, and the in-process fallback encoder."
type: Guide
tags:
  - concepts
  - serving
  - vllm
  - clm
  - deploy
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T00:00:00Z"
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

Planned for milestone M1 (track A); nothing here is running yet. The core never imports torch:
it talks to two loopback HTTP services (DESIGN D16), and the serve-venv side lives under
`serving/` and `deploy/` in the repository and never imports `mesa_clm`.

## Topology

| Process | Port | What | Unit |
|---|---|---|---|
| Encoder | `127.0.0.1:8090` | `vllm/vllm-openai` v0.27.1 (pinned by digest) running `Qwen/Qwen3-8B` at the pinned revision with `--runner pooling --dtype bfloat16 --max-model-len 4096 --max-num-seqs 8 --no-enable-prefix-caching --gpu-memory-utilization 0.20 --enforce-eager`, offline HF cache mounted read-only, `VLLM_API_KEY` from an env file | `mesa-clm-encoder.service` (user unit, `sg docker`) |
| clm-serve | `127.0.0.1:8700` | CLM `bb42c6c5` with patches 0001–0006 in its own venv (`~/.mesa/clm/serve/.venv`, uv CPython 3.11, torch CPU, **no vllm**), `--device cpu --action-cache 512MiB --max-tokens 4096 --no-download --no-ui`, the pinned head `CLM_v0.1-8B.pt` and promoted heads from `heads/served/` | `mesa-clm-serve.service` (`Requires=` the encoder) |

Both bind loopback only and require bearer keys; `mesa-clm serve keys --init|--rotate` writes
raw 0600 key files and the derived env files and restarts both units. The core refuses a
non-loopback URL unless `MESA_CLM_CLM__ALLOW_REMOTE=1` **and** the scheme is https; DE VMs
reach the stack through an `ssh -L` forward or a GPU VM of their own (0.1.0 annotate runs on
the serving host). Timers run the spool recorder every 10 minutes and a GPU headroom check every
5 minutes. Heads on CPU (DESIGN D17): the two heads are 18.9M parameters, and the encoder gets
the whole 0.20 GPU share (about 24.3 GiB on the 121.7 GiB unified-memory GB10: weights about
16 GiB plus 4.5 GiB of KV cache).

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
truncation_side: "left", prefix_caching, route}[:12]`, `clm_model_fp`, `schema_sha256` and
`serving_lock_sha` are stamped on every decision, feature row, artifact and bench cell and must
match exactly (DESIGN D5): CLM heads are encoder-locked, the announced CLM-35B may change the
encoder, and prefix caching is off until golden-vector parity with it on has been recorded.
`serving/serving.lock.json` pins the CLM commit, the patches and their sha256, the head
(repository, revision, file, size, sha256), the encoder recipe and the image digest.
`mesa-clm doctor --serve` checks the lock against `docker inspect`, the head file, the serve
venv's schema sha and the patch shas; that unauthenticated `/v1` requests get 401 on both ports
(while `/health`, `/tokenize` and `/metrics` are open by design on loopback); a golden cosine
and a 5,000-token tail probe on the encoder; golden `/v1/systemone` answers; and upstream drift
against the numbers reported in issue #15 and the model card. Any mismatch fails the doctor and
the provider refuses until the task is re-benched (K4).

## Fallback encoder

`serving/fallback_serve.py` loads `Qwen/Qwen3-8B` bf16 with transformers (right padding, left
truncation, `last_hidden_state` at the last non-pad token, L2-normalised), runs an in-process
`Engine(embedder=CudaEncoder)` on :8700 and a second server exposing `/v1/embeddings`,
`/tokenize`, `/v1/models` and `/health` on :8090 with the same keys. `route: transformers`
enters `encoder_fp`, and artifacts are shared across routes only after a sequential parity run
(minimum cosine ≥0.999, mean ≥0.9999, probe agreement ≥99%). The fallback never runs alongside
the vLLM container. If the GPU share is unavailable (the co-tenant plan for this host sums to
0.83), the headroom check fails, the doctor reports `gpu_budget`, both units stop and the tools
return `decider_unavailable`; the bench is unaffected because features are cached.
