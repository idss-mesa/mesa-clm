# bench table

Silver labels are four-model agreement, not truth. A cell is citable only if it is pre-registered, nested, not exploratory and passes the rest of the citation test (DESIGN, 'Citation test').

| file#cell | n | acc | nll | ece | auroc [one-sided 95% bounds; 90% interval] | beats_lookup_novel | selection | pre_registered | exploratory |
|---|---|---|---|---|---|---|---|---|---|
| bench/results/2026-10-03/tiers.json#neon_annotate.calibrated.F7 | 0 |  |  |  |  |  | none | True | False |
| bench/results/2026-10-03/tiers.json#neon_annotate.calibrated.F7@clm-raw | 0 |  |  |  |  |  | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_annotate.zero_shot.F7 | 98 | 0.357 | 3.971 | 0.620 | 0.498 [0.401, 0.588] | False | none | True | False |
| bench/results/2026-10-03/tiers.json#neon_annotate.zero_shot.F7@clm-raw | 98 | 0.347 | 1.070 | 0.386 | 0.392 [0.313, 0.424] | False | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_aspect.calibrated.F7 | 0 |  |  |  |  |  | none | True | False |
| bench/results/2026-10-03/tiers.json#neon_aspect.calibrated.F7@clm-raw | 0 |  |  |  |  |  | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_aspect.zero_shot.F7 | 60 | 0.050 | 5.918 | 0.705 |  | False | none | True | False |
| bench/results/2026-10-03/tiers.json#neon_aspect.zero_shot.F7@clm-raw | 60 | 0.200 | 2.251 | 0.253 |  | False | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_ontology_fits.calibrated.F9 | 190 | 0.642 | 0.602 | 0.123 | 0.732 [0.673, 0.802] | False | nested | True | False |
| bench/results/2026-10-03/tiers.json#neon_ontology_fits.calibrated.F9@full | 190 | 0.642 | 0.602 | 0.123 | 0.732 [0.673, 0.802] | False | full | True | True |
| bench/results/2026-10-03/tiers.json#neon_ontology_fits.zero_shot.F9 | 190 | 0.489 | 2.624 | 0.464 | 0.751 [0.697, 0.805] | False | nested | True | False |
| bench/results/2026-10-03/tiers.json#neon_ontology_fits.zero_shot.F9@full | 190 | 0.489 | 2.624 | 0.464 | 0.751 [0.697, 0.805] | False | full | True | True |
| bench/results/2026-10-03/tiers.json#neon_term_fits.calibrated.F7 | 285 | 0.698 | 0.621 | 0.105 | 0.457 [0.435, 0.491] | False | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_term_fits.zero_shot.F7 | 285 | 0.495 | 2.047 | 0.374 | 0.511 [0.482, 0.544] | False | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_value_kind.calibrated.F7 | 278 | 0.324 | 1.399 | 0.129 |  | False | none | True | False |
| bench/results/2026-10-03/tiers.json#neon_value_kind.calibrated.F7@clm-raw | 278 | 0.392 | 1.343 | 0.136 |  | False | none | False | True |
| bench/results/2026-10-03/tiers.json#neon_value_kind.zero_shot.F7 | 278 | 0.324 | 1.987 | 0.274 |  | False | none | True | False |
| bench/results/2026-10-03/tiers.json#neon_value_kind.zero_shot.F7@clm-raw | 278 | 0.392 | 1.406 | 0.117 |  | False | none | False | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F1@clm-latest | 190 | 0.626 | 0.682 | 0.137 | 0.566 [0.516, 0.641] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F1@clm-raw | 190 | 0.584 | 0.685 | 0.118 | 0.441 [0.408, 0.496] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F4@clm-latest | 190 | 0.616 | 0.666 | 0.070 | 0.586 [0.506, 0.663] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F4@clm-raw | 190 | 0.542 | 0.693 | 0.157 | 0.439 [0.374, 0.531] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F7@clm-latest | 190 | 0.679 | 0.633 | 0.103 | 0.679 [0.594, 0.757] | True | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F7@clm-raw | 190 | 0.563 | 0.688 | 0.105 | 0.498 [0.473, 0.575] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F9@clm-latest | 190 | 0.642 | 0.602 | 0.123 | 0.732 [0.673, 0.802] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_ontology_fits.calibrated.F9@clm-raw | 190 | 0.721 | 0.618 | 0.119 | 0.736 [0.689, 0.804] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F1@clm-latest | 285 | 0.660 | 0.604 | 0.081 | 0.618 [0.575, 0.664] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F1@clm-raw | 285 | 0.698 | 0.618 | 0.080 | 0.556 [0.524, 0.585] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F4@clm-latest | 285 | 0.698 | 0.628 | 0.127 | 0.385 [0.347, 0.444] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F4@clm-raw | 285 | 0.695 | 0.623 | 0.095 | 0.441 [0.416, 0.462] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F7@clm-latest | 285 | 0.698 | 0.621 | 0.105 | 0.457 [0.435, 0.491] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F7@clm-raw | 285 | 0.698 | 0.624 | 0.091 | 0.428 [0.367, 0.496] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F9@clm-latest | 285 | 0.698 | 0.619 | 0.130 | 0.535 [0.499, 0.575] | False | full | True | True |
| bench/results/2026-10-03/x1.json#neon_term_fits.calibrated.F9@clm-raw | 285 | 0.698 | 0.619 | 0.099 | 0.526 [0.486, 0.581] | False | full | True | True |
| bench/results/2026-10-03/x2.json#neon_annotate.baseline.lookup_prob | 98 | 0.796 | 0.506 | 0.131 | 0.829 [0.784, 0.918] | None | none | True | False |
| bench/results/2026-10-03/x2.json#neon_aspect.baseline.lookup_prob | 60 | 0.583 | 1.610 | 0.401 |  | None | none | True | False |
| bench/results/2026-10-03/x2.json#neon_ontology_fits.baseline.anyjev_l2 | 190 | 0.837 | 0.375 | 0.078 | 0.908 [0.884, 0.940] | True | none | True | False |
| bench/results/2026-10-03/x2.json#neon_ontology_fits.baseline.lookup_prob | 190 | 0.805 | 0.522 | 0.152 | 0.818 [0.772, 0.870] | None | none | True | False |
| bench/results/2026-10-03/x2.json#neon_ontology_fits.baseline.pr13@S1 | 190 | 0.821 | 0.791 | 0.131 | 0.884 [0.842, 0.927] | False | none | True | False |
| bench/results/2026-10-03/x2.json#neon_ontology_fits.baseline.pr13@S1ns | 190 | 0.821 | 0.568 | 0.137 | 0.905 [0.869, 0.941] | False | none | True | False |
| bench/results/2026-10-03/x2.json#neon_term_fits.baseline.anyjev_l2 | 285 | 0.765 | 0.499 | 0.058 | 0.782 [0.743, 0.833] | True | none | True | False |
| bench/results/2026-10-03/x2.json#neon_term_fits.baseline.lookup_prob | 285 | 0.754 | 0.555 | 0.070 | 0.622 [0.591, 0.663] | None | none | True | False |
| bench/results/2026-10-03/x2.json#neon_term_fits.baseline.pr13@S1 | 285 | 0.723 | 1.460 | 0.215 | 0.709 [0.643, 0.771] | False | none | True | False |
| bench/results/2026-10-03/x2.json#neon_term_fits.baseline.pr13@S1ns | 285 | 0.747 | 1.019 | 0.184 | 0.786 [0.738, 0.839] | False | none | True | False |
| bench/results/2026-10-03/x2.json#neon_value_kind.baseline.lookup_prob | 278 | 0.676 | 0.967 | 0.142 |  | None | none | True | False |

Not tables of cells: bench/results/2026-10-03/x1_latency.json (mesa-clm/x1-latency/1: no cells).
