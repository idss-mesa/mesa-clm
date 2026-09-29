# bench 2026-09-29 · baselines · mesa-clm 0.1.0.dev0

Every number below names `bench/results/2026-09-29/baselines.json` (labels_sha256 `aafd18f8cea7…`). Silver labels are four-model agreement, not truth.

| cell | n | n_neg | acc | nll | ece | auroc [95% cluster] | majority | lookup | novel n | novel auroc | novel nll | lopo acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neon_annotate.baseline.lookup_prob | 98 | 35 | 0.796 | 0.506 | 0.145 | 0.829 [0.784, 0.918] | 0.643 | 0.796 | 39 | 0.458 | 0.672 | 0.724 |
| neon_aspect.baseline.lookup_prob | 60 |  | 0.583 | 1.610 | 0.411 |  | 0.300 | 0.550 | 28 |  | 1.993 | 0.350 |
| neon_ontology_fits.baseline.lookup_prob | 190 | 114 | 0.805 | 0.522 | 0.170 | 0.818 [0.772, 0.870] | 0.600 | 0.800 | 99 | 0.396 | 0.653 | 0.721 |
| neon_term_fits.baseline.lookup_prob | 285 | 199 | 0.754 | 0.555 | 0.094 | 0.622 [0.591, 0.663] | 0.698 | 0.772 | 159 | 0.424 | 0.599 | 0.723 |
| neon_value_kind.baseline.lookup_prob | 278 |  | 0.676 | 0.967 | 0.128 |  | 0.478 | 0.680 | 80 |  | 1.289 | 0.543 |

No-model controls (plan §5.4, X2). `lookup_acc` copies the most common training label of (task, scope, target, option_key) from the other cards; `lookup_prob` is its Laplace(α=1) frequency with the training prior on unseen keys. Novel keys are the held-out items no training card had a key for; `lopo` holds out a whole NEON product.

Every cell is a leave-one-card-out pool over all 7 cards (a lookup needs no fit guard); `guard_skipped_folds` lists the folds a fitted tier would skip under the 30/5 guards.
