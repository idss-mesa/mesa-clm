# Collapse spike (label-free) on the pipeline's own rendering, sparky-1, 2026-10-01

Sources (this directory), both written by `scripts/collapse_spike.py` on the final recipe
(`encoder_fp` c3b3d5e1a283, `batch_invariance: kernels`) with the released CLM-v0.1-8B state head
(`b2b4a8c9…`) for the projection, from the 285 term.fits and 190 column.ontology_fits rows of
`bench/snapshots/2026-09-29.parquet` (`labels_sha256` `aafd18f8…752c`; no label read), 925
distinct texts each:

- `collapse_spike.json`: `uv run python scripts/collapse_spike.py`. The stored `state_json` is
  put back in the builders' key order (`learn.features.builder_ordered`) before it is projected
  and rendered, so the four variants are the texts the pipeline sends: the `state_only` texts
  equal the manifest's F7 contexts, `f9_query` its F9 contexts and `control_f1` its F1 contexts,
  all 285 and 190 rows (checked against `features.manifest` on 2026-10-01). 393,133 prompt
  tokens, 92.9 s of embedding.
- `collapse_spike_sorted_keys.json`: the same with `--sorted-keys`, the rendering of the first
  run (`bench/results/2026-09-29/collapse_spike.json`, on the M1-A recipe 852efc921a8a), which
  read the snapshot's canonical, sorted-key `state_json` directly: its nested objects, the card
  header first, came out in an order the pipeline never sends (the card header began with
  `columns:`). None of its state-only texts equals a pipeline context.

Mean pairwise cosine between *different* contexts (distinct texts, i < j):

| task | variant | n | raw, pipeline rendering | projected, pipeline | raw, sorted keys (10-01) | raw, sorted keys (09-29, M1-A recipe) |
|---|---|--:|--:|--:|--:|--:|
| term.fits | state only (F7) | 143 | **0.831** | **0.677** | 0.730 | 0.730 |
| term.fits | suffixed (question appended) | 143 | **0.993** | **0.965** | 0.993 | 0.993 |
| term.fits | F9 query template | 101 | 0.945 | 0.749 | 0.945 | 0.945 |
| term.fits | F1 control | 285 | 0.992 | 0.959 | 0.989 | 0.989 |
| column.ontology_fits | state only (F7) | 60 | **0.949** | **0.767** | 0.854 | 0.854 |
| column.ontology_fits | suffixed | 60 | **0.995** | **0.971** | 0.995 | 0.995 |
| column.ontology_fits | F9 query template | 45 | 0.939 | 0.727 | 0.939 | 0.939 |
| column.ontology_fits | F1 control | 190 | 0.991 | 0.946 | 0.989 | 0.989 |

Reading it:

- **The recipe does not move these numbers; the rendering does.** On the same sorted-key texts
  the two encoder recipes give mean pairwise cosines within about 5e-4 of each other (term.fits
  state only 0.7297 both times, ontology_fits 0.8541 against 0.8536, term.fits F9 0.9448 against
  0.945; `bench/results/2026-09-29/collapse_spike.json` and `collapse_spike_sorted_keys.json`), as
  the A3 recipe change predicts (vectors move by at most about 6e-4 in cosine). The builder-ordered contexts, the ones the pipeline sends, are closer to
  one another in the raw space than the sorted-key ones (term.fits 0.831 against 0.730,
  ontology_fits 0.949 against 0.854), across cards as much as within one (term.fits within-card
  0.832, across-card 0.831; the sorted-key run 0.744 and 0.727). After the head the two
  renderings are much closer (0.677 against 0.675; 0.767 against 0.740). Why the key order moves
  the raw vectors this much is not measured here.
- **Issue #15's direction holds on the pipeline's texts.** Appending the task question pushes
  different targets to ≥ 0.993 raw and ≥ 0.965 projected; the state-only contexts stay further
  apart, most clearly after the head (0.677 term.fits, 0.767 ontology_fits). Issue #15's own
  data: 0.452 state only and 0.958 suffixed (raw), 0.947 suffixed after the head.
- The F9 template is the same text in both renderings (a short string over a few fields), so
  its 2026-10-01 numbers are identical in the two runs.

Report-only (plan §5.6 X1: the collapse diagnostic decides nothing); X1 measures it again per
framing inside its pre-registered cells.
