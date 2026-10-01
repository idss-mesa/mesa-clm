# Fallback encoder parity (sequential), sparky-1, 2026-09-29

Source: `fallback_parity.json` (this directory). Sequence: `uv run python
scripts/fallback_parity.py dump` with both units up (220 texts: the 20 goldens and the first 200
distinct state-only contexts of the collapse spike, through `EncoderClient` in batches of 32, plus
one text per request; clm-serve answers for 20 systemone pairs, `clm-latest` and `clm-raw`), then
`systemctl --user stop mesa-clm-serve.service mesa-clm-encoder.service`, which took MemAvailable
from 78.7 to 107.96 GiB with no GPU process left, then
`HF_HUB_OFFLINE=1 ~/.mesa/clm/serve/.venv/bin/python scripts/fallback_parity.py compare`:
`serving/cuda_encoder.py` `CudaEncoder` (bf16, right padding, left truncation, max_len 4096,
batch 8, Triton overrides deregistered, torch 2.14.0+cu130, transformers 5.17.0, offline) and
CLM's `Engine(embedder=CudaEncoder, device="cpu", action_cache="512MiB")`.

| Route | encoder_fp |
|---|---|
| vllm (reference) | 852efc921a8a |
| transformers (fallback) | 82f4ff33b628 |

| Gate (pre-registered, plan §6.6) | Value | Verdict |
|---|---|---|
| min cosine ≥ 0.999 | **0.998975** (worst: a ~500-token state-only context; 52 of 220 texts below 0.9999) | **FAIL** |
| mean cosine ≥ 0.9999 | **0.999907** | pass |
| overall | | **FAIL** |

Supporting numbers (not gates):

- Against the one-text-per-request vLLM reference: min 0.998975, mean 0.999907. Goldens alone
  (short texts): min 0.999868, mean 0.999961.
- The vLLM reference against itself: batched vs one-text-per-request min 0.999895.
- Diagnostic re-embed in the same process at **batch 1** (no right padding): min **0.999401**,
  mean 0.999913 against the batched reference. The batch-8 run is reproducible: two runs gave
  the identical min 0.9989754535. Right padding inside bf16 batches of 8 accounts for most of
  the gap. Changing the fallback recipe is a decision for its owner, not a re-tuned threshold.
- Systemone through the fallback vs the vLLM route (20 pairs): clm-latest max abs prob diff
  **0.0328** (mean 0.0083, top choice 20/20); clm-raw **0.0717** (mean 0.0152, top choice 19/20).
- Timing: model load 88.3 s; 220 texts (84,660 tokens) in 25.95 s (118 ms per text); 20 pairs ×
  2 models through the engine in 9.05 s.
- Memory: MemAvailable 107.63 → low-water 89.49 GiB (peak Δ 18.15 GiB);
  `torch.cuda.max_memory_allocated` 14.6 GiB (reserved 14.9); process max RSS 4.89 GiB. The
  process exited after the run and MemAvailable returned to 112.8 GiB.

Until a parity run passes, nothing produced on the transformers route may be shared with the
vLLM route (DESIGN D5, D16).
