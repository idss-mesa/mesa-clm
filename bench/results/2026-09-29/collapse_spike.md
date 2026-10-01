# Collapse spike (label-free), sparky-1, 2026-09-29

Source: `collapse_spike.json` (this directory), written by `uv run python scripts/collapse_spike.py`.
Snapshot `bench/snapshots/2026-09-29.parquet` (labels_sha256 `aafd18f8…752c`; 285 term.fits and
190 column.ontology_fits rows; no label was read). Encoder: vLLM route, `encoder_fp`
852efc921a8a; projection: the released CLM-v0.1-8B state head (`b2b4a8c9…`). 925 distinct texts,
392,978 prompt tokens, 79.5 s.

Mean pairwise cosine between *different* contexts (distinct texts, i < j):

| task | variant | n | raw 4096-d | projected 512-d | top PC share (raw) |
|---|---|--:|--:|--:|--:|
| term.fits | state only (F7) | 143 | **0.730** | **0.675** | 0.588 |
| term.fits | suffixed (question appended) | 143 | **0.993** | **0.969** | 0.295 |
| term.fits | F9 query template | 101 | 0.945 | 0.749 | 0.256 |
| term.fits | F1 control (anyjev state + question) | 285 | 0.989 | 0.949 | 0.163 |
| column.ontology_fits | state only (F7) | 60 | **0.854** | **0.740** | 0.678 |
| column.ontology_fits | suffixed | 60 | **0.995** | **0.974** | 0.146 |
| column.ontology_fits | F9 query template | 45 | 0.939 | 0.727 | 0.319 |
| column.ontology_fits | F1 control | 190 | 0.989 | 0.946 | 0.178 |

Issue #15 (its own data): 0.452 state only vs 0.958 suffixed (raw), 0.947 suffixed after the
head. The direction reproduces on mesa-clm's contexts. Appending the task question pushes
different targets to ≥ 0.993 raw / ≥ 0.969 projected, so the shared suffix dominates last-token
pooling. The state-only contexts stay much further apart. Within-card means are only 0.01–0.03
above across-card means in every cell. The F1 control's token lengths (term.fits p50/p95/max
502/628/685; ontology_fits 470/587/630) equal RESEARCH.md's measured table, which cross-checks
the rendering.
