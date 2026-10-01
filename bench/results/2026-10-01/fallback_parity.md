# Fallback encoder parity (sequential, batch 1), sparky-1, 2026-10-01

Source: `fallback_parity.json` (this directory). The re-run the M1-A failure called for
(`bench/results/2026-09-29/fallback_parity.md`: min cosine 0.998975 < 0.999 at batch 8), after
the final vLLM recipe was chosen (`batch_invariance.json`, DESIGN A3) and with the fallback's
recipe changed, not its threshold: **batch 1**, one sequence per forward pass with no padding.

Sequence (`fallback_parity.json#/commands`; MemAvailable from `.local/serving/m1b/gpu_window_m1b.log`):
`uv run python scripts/fallback_parity.py dump` with both units up on the final recipe (220
texts: the 20 goldens and the first 200 distinct state-only contexts of the collapse spike,
**one text per request** as the reference, batches of 32 beside it; clm-serve's answers for the
first 20 systemone pairs, `clm-latest` and `clm-raw`), then `systemctl --user stop
mesa-clm-serve.service mesa-clm-encoder.service` (MemAvailable 78.71 → 107.36 GiB, no GPU
process left), then `HF_HUB_OFFLINE=1 ~/.mesa/clm/serve/.venv/bin/python
scripts/fallback_parity.py compare --batch 1 --diagnostic-batch 8`: `serving/cuda_encoder.py`
`CudaEncoder` (bf16, left truncation, max_len 4096, Triton overrides deregistered, torch
2.14.0+cu130, transformers 5.17.0, offline) and CLM's `Engine(embedder=CudaEncoder, device="cpu",
action_cache="512MiB")`, then both units started again.

| Route | Recipe | encoder_fp |
|---|---|---|
| vllm (reference) | `VLLM_BATCH_INVARIANT=1`, `batch_invariance: kernels` | c3b3d5e1a283 |
| transformers (fallback) | batch 1, `batch_invariance: serial` | b171b1ba4536 |

| Gate (pre-registered, plan §6.6) | Value | Verdict |
|---|---|---|
| min cosine ≥ 0.999 | **0.999260** (worst: the same ~500-token state-only context as in M1-A; 42 of 220 texts below 0.9999) | **pass** |
| mean cosine ≥ 0.9999 | **0.999919** | **pass** |
| overall | | **PASS** |

The plan's third parity condition, probe agreement ≥ 99%, has nothing to measure yet: no probe
exists before M4.

Supporting numbers (not gates):

- Against the batched vLLM reference: min 0.999260, mean 0.999919 (identical to the
  one-text-per-request reference: under the adopted kernels the vLLM vectors do not depend on the
  batch, batched vs single min cosine 1 − 3.8e-15). Goldens alone: min 0.999847, mean 0.999958.
- Diagnostic re-embed in the same process at **batch 8** (the M1-A recipe): min **0.998663**, mean
  0.999908 against the reference: it would still fail, so the recipe change is what passes.
- Systemone through the fallback vs the vLLM route (20 pairs): clm-latest max abs prob diff
  **0.0569** (mean 0.0144, top choice 20/20); clm-raw **0.0744** (mean 0.0204, top choice 19/20).
  Parity of encoders within 1e-3 in cosine still moves probabilities by a few hundredths at CLM's
  scale 100, which is why `route` and `batch_invariance` enter `encoder_fp` (DESIGN D5, D16,
  A3).
- Timing: model load 104.8 s; 220 texts (84,660 tokens) in 27.61 s (125.5 ms per text, against
  118 ms at batch 8 in M1-A); 20 pairs × 2 models through the engine in 12.18 s.
- Memory: MemAvailable 107.02 → low-water 89.24 GiB (peak Δ 17.78 GiB);
  `torch.cuda.max_memory_allocated` 14.6 GiB (reserved 14.96); process max RSS 4.89 GiB.

This run passes plan §6.6's two cosine gates (min ≥ 0.999, mean ≥ 0.9999). Its third gate, probe
agreement ≥ 99%, is unmeasured (no probe exists before M4), and the routes' `/v1/systemone`
answers still differ by 0.0569 (`clm-latest`) and 0.0744 (`clm-raw`) in probability, so D16's
precondition for sharing anything between the routes is **not** met yet and nothing may be shared
across them: artifacts stay keyed by `encoder_fp`, which differs between the routes (D5 requires
an exact match), and how a cross-route artifact would be admitted is not decided here.
