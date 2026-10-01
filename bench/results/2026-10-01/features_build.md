# Feature store build on the final recipe, sparky-1, 2026-10-01

Source: `features_build.json` (this directory), assembled from the outputs of the commands below
(home directory written as `~`). Diagnostics, not citable bench cells: no score was computed for
X1 or X2, which are M2. Everything ran against `mesa-clm-encoder.service` and
`mesa-clm-serve.service` on the final recipe of DESIGN A3/A4 (`encoder_fp` **c3b3d5e1a283**,
`batch_invariance: kernels`, `lock_sha` `59625b8e…`, the installed lock byte-identical to the
checkout's), in two sessions:

- **first build**, 16:12:25Z to 16:15:40Z, the JSON's top-level fields (unchanged since that run);
- **re-run**, 16:49:29Z to 17:00:41Z, under `rerun`: an idempotent build, a fresh rebuild into a
  scratch store, projection, export, stats and a cross-check against clm-serve. Neither unit was
  restarted or reconfigured; both stayed active and not enabled at boot.

## First build (16:12Z)

    uv run mesa-clm features build --snapshot bench/snapshots/2026-09-29.parquet
    uv run mesa-clm features project
    uv run mesa-clm features project --model clm-raw
    uv run mesa-clm features export-npz --out ~/.mesa/clm/features/c3b3d5e1a283/textcache
    uv run mesa-clm features stats --json

The store is `~/.mesa/clm/features/c3b3d5e1a283/features.duckdb` (directory 0700, database and
lock 0600). Nothing exists under the M1-A fingerprint 852efc921a8a: no vector was ever stored
under the old recipe.

| Item | Value |
|---|---|
| Snapshot | `bench/snapshots/2026-09-29.parquet`, `labels_sha256` `aafd18f8…752c` (read label-free) |
| Tasks, framings | term.fits, column.ontology_fits; F1, F4, F7, F9, `joint4096@S1`, `joint4096@S1ns` |
| Manifest | 5,018 rows, **1,563 distinct texts** (1,400 state side, 163 action side) |
| Embedded | 1,563 new vectors, 0 already stored, **0 truncated**, one text per request (batch 1) |
| Encoder tokens | 620,336 in all; per text max 685, p50 450, p95 588.8 |
| Time | 195.2 s for the build (3:15.5 wall clock with the CLI start), max RSS 165 MiB |
| fp16 round trip | min cosine **0.9999999** (gate ≥ 0.9999, plan §5.2) |
| Projections | `clm-latest` (`clm_model_fp` 78be8c462b2e, head `b2b4a8c9…`): 1,563 state and 1,563 action in 2.81 s wall clock (numpy on the CPU, the CLI included); `clm-raw` has nothing to project |
| TextCache export | `choice_Qwen_Qwen3-8B_4096.npz`: keys `sha1(text)` `<U40`, vecs float16 [1563, 4096], 13,054,678 bytes, sha256 `c6628095…7293`, mode 0600, 0.29 s |
| Store size | 31,207,424 bytes |

The row and text counts equal what `features.manifest` gives for this snapshot without an
encoder (`manifest(...).summary()`), and the per-text maximum (685 tokens) equals the term.fits
maximum of RESEARCH.md's token table (the F1 context is that table's rendering); the truncated
count is 0 because every X1/X2 text is under `max_len - 16` = 4,080 tokens (D23). The export is
what `finetune.py --embed-cache ~/.mesa/clm/features/c3b3d5e1a283/textcache --embed-model
Qwen/Qwen3-8B --max-len 4096` reads (plan §5.3, M7).

## Re-run (16:49Z to 17:00Z)

| Step | Command | Result |
|---|---|---|
| Doctor | `uv run mesa-clm doctor --serve` | exit 0; 30 ok, 2 warn, 0 fail (the warnings below) |
| Build, existing store | `uv run mesa-clm features build --snapshot bench/snapshots/2026-09-29.parquet` | same manifest (5,018 rows, 1,563 texts); already stored 1,563, embedded 0, 0 truncated, 0 encoder tokens; 0.30 s wall clock |
| Build, scratch store | `MESA_CLM_FEATURES__DIR=.local/features-rebuild uv run mesa-clm features build --snapshot …` | embedded 1,563, 0 truncated, 620,336 encoder tokens, batch 1, **195.9 s** (3:16.24 wall clock), max RSS 166 MiB |
| Rebuild vs store | (read both stores) | **1,563 / 1,563 vectors bitwise equal**, token counts, truncation flags and fp16 cosines equal; float16 digest `e5610655…7c0e` for both; only `created_at` differs |
| Project | `uv run mesa-clm features project --model clm-latest` | `clm_model_fp` 78be8c462b2e: state 1,563, action 1,563 (all already cached; 0.55 s); `--model clm-raw`: nothing to project |
| Export | `uv run mesa-clm features export-npz --out .local/features-npz/` | 1,563 rows, 13,054,678 bytes, sha256 `c6628095…7293`, mode 0600 (directory 0700), keys and float16 rows bitwise equal to the store; **byte-identical to the first export**; fp16 gate 0.9999999 ≥ 0.9999 |
| Stats | `uv run mesa-clm features stats [--json]` | one store, live: 1,563 texts, 1,563 vectors, 0 truncated, 620,336 tokens (max 685), fp16 min cosine 0.9999999, 31,207,424 bytes |

The scratch rebuild ran on the final code (`src/mesa_clm/learn/features.py` changed at
16:44:03Z, after the first build) and reproduced the store bit for bit 38 minutes later: under the
A3 kernels, one text per request, the store is a function of the snapshot and the recipe. The
doctor's two warnings, verbatim:

    [WARN] permissions: ~/.mesa/clm/locks 0775, ~/.mesa/clm/locks/provenance.lock 0664 (owner-only state; fix: chmod 700 ~/.mesa/clm/locks; chmod 600 ~/.mesa/clm/locks/provenance.lock)
    [WARN] clm drift: quickstart urgency 0.8466 (|Δ| 0.0100116), billing 0.9903 (|Δ| 0.0014636), frustration 2.0000 (|Δ| 0.0000210) against issue #15 ± 0.01; tides top "The Moon's gravitational pull." 0.9932 against the model card 0.993 ± 0.005 (warn only; DESIGN A3)

## Texts per task, framing and role

Each cell is manifest rows / distinct texts / tokens of those distinct texts (exact counts from
the store). A text used in several cells is counted in each: the F1 contexts are the
`joint4096@S1` contexts, and F4, F7 and F9 share their candidates and anchors.

| Task | Framing | context | candidate | anchor | noul_true + noul_false |
|---|---|---|---|---|---|
| term.fits | F1 | 285 / 285 / 145,663 | - | - | 570 / 2 / 74 |
| term.fits | F4 | 143 / 143 / 61,846 | 285 / 146 / 3,883 | 143 / 1 / 12 | - |
| term.fits | F7 | 143 / 143 / 59,808 | 285 / 146 / 3,883 | 143 / 1 / 12 | - |
| term.fits | F9 | 143 / 101 / 3,585 | 285 / 146 / 3,883 | 143 / 1 / 12 | - |
| term.fits | joint4096@S1 | 285 / 285 / 145,663 | - | - | - |
| term.fits | joint4096@S1ns | 285 / 285 / 137,113 | - | - | - |
| column.ontology_fits | F1 | 190 / 190 / 92,165 | - | - | 380 / 2 / 54 |
| column.ontology_fits | F4 | 60 / 60 / 26,702 | 190 / 11 / 134 | 60 / 1 / 13 | - |
| column.ontology_fits | F7 | 60 / 60 / 25,857 | 190 / 11 / 134 | 60 / 1 / 13 | - |
| column.ontology_fits | F9 | 60 / 45 / 1,616 | 190 / 11 / 134 | 60 / 1 / 13 | - |
| column.ontology_fits | joint4096@S1 | 190 / 190 / 92,165 | - | - | - |
| column.ontology_fits | joint4096@S1ns | 190 / 190 / 88,365 | - | - | - |

Per framing, both tasks: F1 479 texts / 237,956 tokens, F4 362 / 92,590, F7 304 / 64,710, F9
261 / 7,686, `joint4096@S1` 475 / 237,828, `joint4096@S1ns` 475 / 225,478. F9's template gives
some targets the same context (101 distinct of 143 term.fits targets, 45 of 60 ontology_fits
targets). No target rendered two different contexts (`context_conflicts` 0).

## Cross-check against clm-serve (plan §5.6 X1)

Plan §5.6 scores X1 offline from the cache and asks for a "200-pair cross-check vs clm-serve
≤1e-4 in probs". Two hundred (context, candidates + anchor) groups of the F4, F7 and F9 arms were
drawn from the manifest: per (task, framing) a share proportional to its group count (47, 47 and
47 term.fits groups of 143 each; 20, 20 and 19 ontology_fits groups of 60 each), the groups
ordered by `sha256("task|framing|target_sha256")` and the first ones taken; 649 options, 307
distinct texts. Each group was scored three ways for `clm-latest` and `clm-raw`:

- **store**: `mesa_clm.learn.offline.OfflineScorer.rank_fit_texts` over the feature store (float16
  vectors; the cached `clm-latest` projections), the route X1 would use;
- **local**: the same scorer over float32 vectors of the same texts fetched again from the encoder,
  one text per request (rounded as the store rounds them, all 307 are bitwise equal to the store's);
- **served**: clm-serve's `/v1/systemone` answer to the same request, built with
  `framings.build_request` exactly as the manifest builds it (the script checks that the request
  renders the group's manifest texts).

| Model | Pair | max \|Δp\| | mean \|Δp\| | p95 \|Δp\| | top choice | max \|Δs_c\| | ≤ 1e-4 |
|---|---|---|---|---|---|---|---|
| clm-latest | store vs served | 1.31e-3 | 1.23e-4 | 4.35e-4 | 199/200 | 6.3e-3 | **no** |
| clm-latest | local vs served | 5.2e-6 | 6.9e-7 | 2.7e-6 | 200/200 | 3.5e-5 | yes |
| clm-latest | store vs local | 1.31e-3 | 1.23e-4 | 4.33e-4 | 199/200 | 6.3e-3 | no |
| clm-raw | store vs served | 3.59e-3 | 3.56e-4 | 1.03e-3 | 200/200 | 1.34e-2 | **no** |
| clm-raw | local vs served | 7.5e-5 | 1.5e-5 | 4.0e-5 | 200/200 | 3.2e-4 | yes |
| clm-raw | store vs local | 3.60e-3 | 3.55e-4 | 9.97e-4 | 200/200 | 1.34e-2 | no |

From the float16 store the offline route **fails** the pre-registered ≤ 1e-4, and store vs local
is as large as store vs served: the gap is the float16 rounding of the stored vectors, not
serving. From float32 vectors it passes with room (5.2e-6 and 7.5e-5, the size of the
end-to-end parity the serving record reports for A3). The one flipped top choice is a
one-candidate `clm-latest` group (`term.fits/F7/d990d135bddc`) where the store gives the anchor
0.50039 and clm-serve gives the candidate 0.50018. A supplementary 50-pair F1 noul check (30
term.fits, 20 ontology_fits pairs, the same draw rule) gives store vs served 5.0e-4
(`clm-latest`, 50/50) and 4.1e-4 (`clm-raw`, 49/50: one pair at p 0.50007 against 0.49972).

What the float16 copy loses (`rerun.crosscheck.cause`, the 307 texts re-embedded once more):

| Offline vectors | clm-latest max \|Δp\| vs served | clm-raw max \|Δp\| vs served |
|---|---|---|
| float16 (the store) | 1.31e-3 | 3.59e-3 |
| float16, renormalised before the head | 1.26e-3 | - |
| float16 with dimensions 2202, 2284, 3169 from float32 | 1.37e-3 | 8.0e-4 |
| float32 | 5.2e-6 | 7.5e-5 |

Qwen3-8B's last-token vectors carry three large dimensions (2202, 2284 and 3169 hold every
text's largest component; median |value| 0.27 to 0.47 against a median component of 0.0038;
2284 and 3169 carry 27% and 22% of a typical pairwise cosine), where the float16 spacing is
2.4e-4. Restoring those three dimensions removes most of `clm-raw`'s gap but none of
`clm-latest`'s, and renormalising does not help either: only float32 vectors reproduce
clm-serve. The plan §5.2 gate (round-trip cosine ≥ 0.9999, met at 0.9999999) bounds the angle
between each vector and its float16 copy; it does not bound scores at CLM's scale 100.

Timing: the 307 texts (62,099 tokens) took 30.3 s one per request; clm-serve answered the 200
`clm-latest` requests in 26.95 s (p50 131.95 ms, 60,786 encoder tokens on its cache misses) and
the 200 `clm-raw` requests from its cache in 0.29 s. The scripts and the per-group rows are not
committed (`.local/` is git-ignored): `.local/live-final/crosscheck.py`, `crosscheck.json`,
`fp16_cause.py`, `fp16_cause.json`.

## Reading

- The store is complete and reproducible: every X1/X2 text of the snapshot has a vector, none is
  truncated, and two independent builds agree bit for bit.
- The TextCache export passes its own gate and is byte-identical across exports.
- The X1 cross-check as pre-registered cannot pass while X1 scores from the float16 store; it
  passes from float32 vectors. The fix (keep float32 vectors for offline scoring, float16 only for
  the export) or a change of the check's scope belongs in the record before G1, the M1 merge.
- **Resolved before G1** (the check's scope unchanged): the store keeps float32 vectors (format 2,
  DESIGN implementation note "The feature store"); this format-1 store was moved aside and the
  store rebuilt on the DESIGN A5 recipe (`features_build_m1c.md`: every float16 cast bitwise equal
  to this store's vectors, the export byte-identical), and from it the 200-group cross-check
  passes at 5.2e-6 (`clm-latest`) and 7.47e-5 (`clm-raw`) (`x1_crosscheck.md`). The scripts are
  committed as `scripts/x1_crosscheck.py`.
