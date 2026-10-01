# Batch invariance of the vLLM encoder (pre-registered), sparky-1, 2026-10-01

Source: `batch_invariance.json` (this directory), written by `uv run python
scripts/batch_invariance.py decide` from the two arm runs `scripts/batch_invariance.py measure
--arm B0|B1` (`.local/serving/m1b/batch_invariance_{B0,B1}.json`, folded into the JSON under
`arms`). Both arms ran the guarded recipe (bearer guard on every route but `/health`, clients
truncating at 4,095 tokens, DESIGN A4) on `mesa-clm-encoder.service` (vLLM 0.27.1, image
`sha256:0a51ea5b…d967`, Qwen/Qwen3-8B `b968826d` bf16, `--max-num-seqs 8`, prefix caching off)
and `mesa-clm-serve.service`, each arm after a restart of both units.

Pre-registration (`batch_invariance.json#/preregistration`): the rule below was written into
`scripts/batch_invariance.py` before the first arm ran, but the script was neither committed nor
hashed into the arm records then (its file is dated 08:51:48 MDT, after both arms, 08:39-08:50;
the arm files carry no pre-registration block, which the verdict step copied from the script), so
nothing tamper-evident shows the rule predates the arms. The margins make the verdict insensitive
to where the new thresholds were set: B0 misses the cosine criterion by a factor of about 100 and
the ≤ 1e-4 systemone gate (fixed in plan §5.6 and §8 since 2026-09-28) by 377-591 times; B1 meets
the cosine criterion by more than eight orders of magnitude, that gate at 3.9e-6 and 6.6e-5, and
the slowdown criterion at 1.16× of 4× (DESIGN A3). The rule: 120 texts (the 20 goldens of `.local/serving/encoder_golden.npz`, the first 100
state-only contexts of `.local/serving/collapse_contexts.json`) and the 50 systemone pairs of
`.local/serving/systemone_pairs.json`. Patterns (a) one text per request, (b) batches of 32,
(c) 8 concurrent single-text requests, (d) repeat of (a). **Adopt B1 iff** min cosine across
(a)–(d) ≥ 1 − 1e-6 **and** (e) clm-serve vs the local route ≤ 1e-4 for both models **and** (f)
slowdown ≤ 4× against M1-A's p50s (2,820 ms for a 32-text batch, 110.1 ms for a cold rank-fit
request; `bench/results/2026-09-29/serving_m1.json`); else B2 (serial) on the same cosine and
(e) criteria; else keep the recipe and re-scope the gate.

| Arm | Recipe | (a)–(d) min cosine | max abs element diff | (e) clm-latest | (e) clm-raw | (f) 32-text p50 / cold p50 | slowdown | Verdict |
|---|---|---|---|---|---|---|---|---|
| B0 | current kernels | 0.99989461 (1 − 1.05e-4) | 6.0e-3 | 0.0377 (49/50 top) | 0.0591 (50/50) | 2,885.2 / 113.4 ms | 1.03× | reference (fails the cosine and (e) criteria) |
| **B1** | `VLLM_BATCH_INVARIANT=1` | **1 − 3.3e-15** | **6.0e-8** | **3.9e-6** (50/50) | **6.6e-5** (50/50) | **3,266.6 / 116.0 ms** | **1.16×** | **ADOPT** |

B2 (serial: `--max-num-seqs 1`) was not run: the rule stops at B1.

Per pattern pair (`arms.*.pairwise`): under B0, (a) and (d) are bitwise identical (120/120 rows)
and every pair that involves batching or concurrency moves vectors (93/120 rows bitwise equal
for (a) vs (b)); under B1 every pair agrees to 1 − 3.3e-15, with 104–120 of 120 rows bitwise
equal and the rest one float32 ulp apart (6.0e-8). The local route against itself (same request
pattern, run twice) shows why the end-to-end gate failed before: B0 0.0377 (clm-latest) and
0.0297 (clm-raw) even with identical batches, B1 8.8e-8 and 2.9e-7.

Evidence that the mode applies on GB10 (sm_121): the container ran with `VLLM_BATCH_INVARIANT=1`
(`arms.B1.container.env`), and the attention selector dropped FLASHINFER, which lacks
batch-invariance support (`arms.B1.container.log_lines`: `['FLASH_ATTN', 'TRITON_ATTN',
'FLEX_ATTENTION']`, against B0's list with FLASHINFER). The B1 start's encoder log (`journalctl
--user -u mesa-clm-encoder.service --since "2026-10-01 08:43:58"`) also shows torch's
`preferred_blas_library` warning, which `enable_batch_invariant_mode` triggers, and the JIT
compilation of the Triton `matmul_kernel_persistent` on the first requests: the linear layers run
the persistent batch-invariant matmul (the image's `batch_invariant.py` takes the non-SM80 branch
on this GPU, disabling cuBLAS split-k, and overrides the unquantized linear layers, RMSNorm,
softmax and `bmm`; FlashAttention runs with `num_splits=1`).

What adopting B1 changes (`secondary.B1_vs_B0_one_text_per_request`): the same texts, one per
request, move by min cosine 0.99937 (mean 0.99993, max abs element 0.0157) from the M1-A recipe,
so every vector differs and `encoder_fp` rotates (DESIGN A3).

Secondary, not part of the rule (`secondary.drift_g_not_gated`): (g) the tides rank stays within
0.993 ± 0.005 on both arms (0.9934, 0.9932). The CLM README quickstart against issue #15's values
(± 0.01): B0 urgency 0.8366, billing 0.9888, frustration 2.0000 (all within); B1 billing 0.9903
and frustration 2.0000 within, **urgency 0.8466 outside** (unrounded 0.8466116, |Δ| = 0.0100116
against 0.8366; re-measured three times with the same answer, `serving_m1b.json` records it with
the probe now comparing unrounded values). The quickstart is a warn-level upstream-drift probe of
the doctor (plan §6.8); see DESIGN A3 for how it is read.
