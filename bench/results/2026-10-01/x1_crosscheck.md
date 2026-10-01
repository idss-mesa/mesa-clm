# X1's cross-check from the float32 feature store, sparky-1, 2026-10-01

Source: `x1_crosscheck.json` (this directory), written by `uv run python scripts/x1_crosscheck.py
--out bench/results/2026-10-01/x1_crosscheck.json` (started 18:26:09Z, 85.7 s) against
`mesa-clm-encoder.service` and `mesa-clm-serve.service` on the recipe of DESIGN A5 (`encoder_fp`
c3b3d5e1a283, `lock_sha` `dd33f9fe…`, vector recipe `ed486dcd…`) and the feature store rebuilt in
format 2 (`features_build_m1c.md`). Diagnostics, not a bench cell: no label is read.

Plan §5.6 scores X1 offline from the cache and pre-registers a "200-pair cross-check vs clm-serve
≤1e-4 in probs". The format-1 store kept float16 vectors only and missed it by more than ten times
(`features_build.json#/rerun/crosscheck`: 1.31e-3 for `clm-latest`, 3.59e-3 for `clm-raw`). The
store now keeps the float32 vectors (DESIGN, implementation note "The feature store"). The draw
is the earlier one: per (task, framing) stratum of the F4/F7/F9 groups a share of 200
proportional to its size (47, 47, 47 term.fits and 20, 20, 19 column.ontology_fits groups), the
groups ordered by `sha256("task|framing|target_sha256")`; 649 options, 307 distinct texts. Per
group and model:

- **store**: `OfflineScorer.rank_fit_texts` over the store (opened with the live lock's vector
  recipe, which it checks), X1's route;
- **served**: clm-serve's `/v1/systemone` answer to the same request;
- **rank**: clm-serve's `/v1/rank` over the same context, instructions and candidate texts
  (plan §9's "s_c vs /v1/rank");
- **fp16**: the store's vectors cast to float16 before scoring, what format 1 held.

| Model | Pair | max \|Δp\| | mean \|Δp\| | p95 \|Δp\| | top choice | max \|Δs_c\| | ≤ 1e-4 |
|---|---|---|---|---|---|---|---|
| clm-latest | store vs served | **5.2e-6** | 7.2e-7 | 2.7e-6 | 200/200 | 3.5e-5 | **yes** |
| clm-latest | store vs rank | 5.2e-6 | 7.2e-7 | 2.7e-6 | 200/200 | 3.5e-5 | yes |
| clm-latest | rank vs served | 0 | 0 | 0 | 200/200 | 0 | yes |
| clm-latest | fp16 vs served | 1.31e-3 | 1.23e-4 | 4.35e-4 | 199/200 | 6.3e-3 | no |
| clm-raw | store vs served | **7.47e-5** | 1.48e-5 | 4.0e-5 | 200/200 | 3.2e-4 | **yes** |
| clm-raw | store vs rank | 7.47e-5 | 1.48e-5 | 4.0e-5 | 200/200 | 3.2e-4 | yes |
| clm-raw | rank vs served | 0 | 0 | 0 | 200/200 | 0 | yes |
| clm-raw | fp16 vs served | 3.59e-3 | 3.56e-4 | 1.03e-3 | 200/200 | 1.34e-2 | no |

The F1 noul supplement (50 pairs, 30 term.fits and 20 ontology_fits, the same draw rule) gives
store vs served 5.5e-6 (`clm-latest`, 50/50) and 7.0e-5 (`clm-raw`). The 307 texts re-embedded
one per request are **bitwise equal** to the stored float32 vectors (307/307), and `/v1/rank`
returns exactly `/v1/systemone`'s probabilities (both run the same engine path).

Reading:

- From float32 vectors the offline route meets the pre-registered 1e-4 for both models, so X1
  scores from the store as planned; the gate is not loosened.
- `clm-raw` passes with less room (7.47e-5 of 1e-4) than `clm-latest` (5.2e-6). Its scores are
  raw 4096-d cosines at scale 100, and clm-serve accumulates them in float32 while the offline
  scorer uses float64 (`OfflineScorer._cosines`): the residual is CLM's float32 accumulation, not
  the vectors (fresh and stored vectors are bitwise equal), and the head's 512-d projection
  shrinks it for `clm-latest`. A recipe change that moves this margin is a re-measurement, not a
  gate change.
- The fp16 rows reproduce the format-1 failure to the digit: the float16 copy, not serving, made
  the difference.

Served: clm-serve answered the 200 `clm-latest` `/v1/systemone` requests at p50 138.8 ms / p95
261.5 ms (61,535 encoder tokens on its cache misses) and the `clm-raw` ones from its cache (p50
0.3 ms). The script is committed (`scripts/x1_crosscheck.py`); `tests/engine/test_offline_parity.py`
runs a 12-group version of the same check against the live stack (`MESA_CLM_ENGINE=1`).
