# bench 2026-10-03 · tiers · mesa-clm 0.1.0.dev0

Every number below names `bench/results/2026-10-03/tiers.json` (labels_sha256 `aafd18f8cea7…`). Silver labels are four-model agreement, not truth.

| cell | n | n_neg | acc | nll | ece | auroc [one-sided 95% bounds; 90% interval] | majority | lookup | novel n | novel auroc | novel nll | lopo acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neon_annotate.zero_shot.F7 | 98 | 35 | 0.357 | 3.971 | 0.620 | 0.498 [0.401, 0.588] | 0.643 | 0.796 | 39 | 0.303 | 4.778 | 0.724 |
| neon_annotate.calibrated.F7 | 0 | 0 |  |  |  |  |  |  |  |  |  |  |
| neon_annotate.zero_shot.F7@clm-raw | 98 | 35 | 0.347 | 1.070 | 0.386 | 0.392 [0.313, 0.424] | 0.643 | 0.796 | 39 | 0.328 | 1.165 | 0.724 |
| neon_annotate.calibrated.F7@clm-raw | 0 | 0 |  |  |  |  |  |  |  |  |  |  |
| neon_aspect.zero_shot.F7 | 60 |  | 0.050 | 5.918 | 0.705 |  | 0.300 | 0.550 | 28 |  | 7.011 | 0.350 |
| neon_aspect.calibrated.F7 | 0 |  |  |  |  |  |  |  |  |  |  |  |
| neon_aspect.zero_shot.F7@clm-raw | 60 |  | 0.200 | 2.251 | 0.253 |  | 0.300 | 0.550 | 28 |  | 2.396 | 0.350 |
| neon_aspect.calibrated.F7@clm-raw | 0 |  |  |  |  |  |  |  |  |  |  |  |
| neon_ontology_fits.zero_shot.F9 | 190 | 114 | 0.489 | 2.624 | 0.464 | 0.751 [0.697, 0.805] | 0.600 | 0.800 | 99 | 0.725 | 2.846 | 0.721 |
| neon_ontology_fits.zero_shot.F9@full | 190 | 114 | 0.489 | 2.624 | 0.464 | 0.751 [0.697, 0.805] | 0.600 | 0.800 | 99 | 0.725 | 2.846 | 0.721 |
| neon_ontology_fits.calibrated.F9 | 190 | 114 | 0.642 | 0.602 | 0.123 | 0.732 [0.673, 0.802] | 0.600 | 0.800 | 99 | 0.709 | 0.594 | 0.721 |
| neon_ontology_fits.calibrated.F9@full | 190 | 114 | 0.642 | 0.602 | 0.123 | 0.732 [0.673, 0.802] | 0.600 | 0.800 | 99 | 0.709 | 0.594 | 0.721 |
| neon_term_fits.zero_shot.F7 | 285 | 199 | 0.495 | 2.047 | 0.374 | 0.511 [0.482, 0.544] | 0.698 | 0.772 | 159 | 0.539 | 2.117 | 0.723 |
| neon_term_fits.calibrated.F7 | 285 | 199 | 0.698 | 0.621 | 0.105 | 0.457 [0.435, 0.491] | 0.698 | 0.772 | 159 | 0.442 | 0.610 | 0.723 |
| neon_value_kind.zero_shot.F7 | 278 |  | 0.324 | 1.987 | 0.274 |  | 0.478 | 0.680 | 80 |  | 1.949 | 0.543 |
| neon_value_kind.calibrated.F7 | 278 |  | 0.324 | 1.399 | 0.129 |  | 0.478 | 0.680 | 80 |  | 1.403 | 0.543 |
| neon_value_kind.zero_shot.F7@clm-raw | 278 |  | 0.392 | 1.406 | 0.117 |  | 0.478 | 0.680 | 80 |  | 1.565 | 0.543 |
| neon_value_kind.calibrated.F7@clm-raw | 278 |  | 0.392 | 1.343 | 0.136 |  | 0.478 | 0.680 | 80 |  | 1.433 | 0.543 |

zero_shot and calibrated cells for every task (plan §8 M2, X3), scored offline from the feature store; design/m2-analysis-plan.md §9 says how every field is computed.

A rank_fit cell without @ is nested (fold_choices from X1's inner CV; the only citable-form rank_fit cells of M2); @full uses A1 in every fold and is exploratory (D27). Without an A1 (X1's K1 or undecidable outcome) the default arm F7@clm-latest is reported for audit only. A closed choice is F7 under clm-latest; @clm-raw is reported, not pre-registered.

Silver labels are four-model agreement, not truth. ms_per_decision is the offline replay time, not serving latency.

X1's outcome: bench/results/2026-10-03/x1.json (sha256 bbef6cc9dbf094abba8e0fcdc404e00f09da391ce43e97b4dab8440d2b3455b8), replayed and recomputed before use.
