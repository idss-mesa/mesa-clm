# Annotate latency per card, non-bench SRER cards, sparky-1, 2026-10-01

Source: `annotate_latency.json` (this directory), written by `uv run python
scripts/annotate_latency.py --out bench/results/2026-10-01/annotate_latency.json` (started
18:28:13Z) through the live provider on the recipe of DESIGN A5 (`encoder_fp` c3b3d5e1a283,
`lock_sha` `dd33f9fe…`; the keyed pre-flight passed and the container check found the pinned
image and recipe), at tier `zero_shot` (uncalibrated, never `auto`), into a scratch sidecar
(`.local/latency/`). Diagnostics for plan §8 M1-A's "p50/p95 per card" and the M3 latency budget;
outcomes are not recorded.

Cards: **non-bench only** (DESIGN, "G1 freeze": no live run looks at a bench card before the M2
cells exist): plan §9's smoke card `DP1.00004.001.BP_30min`, then the first nine other cards of
`~/neon-ducklake/sites/SRER/cards/anyjev/` in `sha256(name)` order, the bench products
DP1.10003.001 and DP1.10022.001 skipped. Per card:

- **cold**: the card's first run after the units' restart (18:20Z), OLS live from EMBL-EBI
  (recorded, 5 calls for all ten cards), clm-serve's caches empty except for what earlier cards
  share (the closed options, the anchors);
- **warm**: 10 further runs with OLS replayed from that recording (no network) and clm-serve's
  caches warm.

`seconds` is the run's own clock (planning, OLS, CLM calls, deciding); the wall clock around the
call adds the sidecar commit (one transaction under the flock). "CLM ms" sums the client-side
latency of the run's calls.

| Card | decisions | CLM calls | encoder tokens (cold) | cold s | cold CLM ms | warm p50 / p95 ms | warm wall p50 / p95 ms |
|---|--:|--:|--:|--:|--:|--:|--:|
| DP1.00004.001.BP_30min | 39 | 17 | 7,687 | 4.33 | 2,545 | 16.3 / 24.9 | 133 / 141 |
| DP1.10047.001.spc_particlesize | 63 | 26 | 13,247 | 4.01 | 3,903 | 32.3 / 41.1 | 170 / 177 |
| DP1.10107.001.mms_metagenomeSequencing | 52 | 18 | 7,089 | 2.40 | 2,310 | 31.2 / 36.4 | 179 / 187 |
| DP1.10109.001.mga_soilGroupAbundances | 42 | 15 | 5,578 | 1.90 | 1,828 | 17.6 / 21.2 | 163 / 173 |
| DP1.10038.001.mos_BOLDtaxonomy | 37 | 16 | 4,231 | 3.51 | 1,984 | 20.5 / 22.1 | 163 / 170 |
| DP1.00022.001.SRPP_1min | 33 | 14 | 4,926 | 1.83 | 1,762 | 13.0 / 17.6 | 165 / 176 |
| DP1.10043.001.mos_expertTaxonomistIDProcessed | 67 | 26 | 9,992 | 3.49 | 3,377 | 33.0 / 47.2 | 204 / 215 |
| DP1.10067.001.bbc_percore | 62 | 23 | 10,010 | 3.19 | 3,071 | 41.5 / 45.7 | 217 / 228 |
| DP1.00094.001.SWS_30_minute | 49 | 22 | 11,910 | 3.64 | 3,534 | 28.2 / 40.8 | 206 / 219 |
| DP1.00046.001.THRPRE_1min | 23 | 9 | 2,780 | 1.10 | 1,062 | 12.6 / 15.0 | 180 / 184 |

Across the ten cards: cold p50 **3.34 s**, p95 **4.18 s**, max 4.33 s; the warm per-card p50s
have a p50 of 24.4 ms and a maximum of 41.5 ms, the warm per-card p95s a maximum of 47.2 ms; no
call failed and no run degraded.

Reading:

- A cold run is dominated by the encoder: every new context costs one embedding on clm-serve's
  cache miss (about 120 ms for a few hundred tokens, `serving_m1c.json#/latency`), and the CLM
  calls sum to 1.1-3.9 s of the 1.1-4.3 s; the rest is the live OLS calls and planning.
- Warm, the decide phase takes tens of milliseconds and the sidecar commit about 115-180 ms more
  (wall p50 minus run p50 per card).
- How many questions a card costs depends on how many columns `column.annotate` sends on to the
  ontology and term steps (outcomes are not recorded here; the annotate smoke saw zero shot answer
  No to every column of its two cards, `annotate_smoke.md`). A tier that sends more columns on
  asks more questions, so the M3 budget is to be re-measured on the tier M3 runs.
- Only one cold run per card exists: a second cold run needs another restart of the units. The
  cold figures are therefore one sample per card over ten cards, not a per-card distribution.
