# bench table

Silver labels are four-model agreement, not truth. A cell is citable only if it is pre-registered, nested, not exploratory and passes the rest of the citation test (DESIGN, 'Citation test').

| file#cell | n | acc | nll | ece | auroc [one-sided 95% bounds; 90% interval] | beats_lookup_novel | selection | pre_registered | exploratory |
|---|---|---|---|---|---|---|---|---|---|
| bench/results/2026-10-04/x3.json#neon_annotate.probe.F7 | 0 |  |  |  |  |  | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_annotate.probe.F7@full | 0 |  |  |  |  |  | full | True | True |
| bench/results/2026-10-04/x3.json#neon_annotate.probe.F7@latest | 0 |  |  |  |  |  | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_annotate.probe.F7@raw | 0 |  |  |  |  |  | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_aspect.probe.F7 | 0 |  |  |  |  |  | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_aspect.probe.F7@full | 0 |  |  |  |  |  | full | True | True |
| bench/results/2026-10-04/x3.json#neon_aspect.probe.F7@latest | 0 |  |  |  |  |  | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_aspect.probe.F7@raw | 0 |  |  |  |  |  | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_ontology_fits.probe.F9 | 190 | 0.821 | 0.404 | 0.061 | 0.892 [0.855, 0.933] | False | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_ontology_fits.probe.F9@full | 190 | 0.868 | 0.340 | 0.076 | 0.926 [0.892, 0.960] | False | full | True | True |
| bench/results/2026-10-04/x3.json#neon_ontology_fits.probe.F9@latest | 190 | 0.858 | 0.358 | 0.049 | 0.916 [0.884, 0.953] | False | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_ontology_fits.probe.F9@raw | 190 | 0.811 | 0.424 | 0.102 | 0.882 [0.848, 0.919] | True | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_term_fits.probe.F7 | 285 | 0.754 | 0.485 | 0.079 | 0.804 [0.749, 0.864] | True | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_term_fits.probe.F7@full | 285 | 0.768 | 0.469 | 0.077 | 0.817 [0.774, 0.866] | True | full | True | True |
| bench/results/2026-10-04/x3.json#neon_term_fits.probe.F7@latest | 285 | 0.768 | 0.469 | 0.077 | 0.817 [0.774, 0.866] | True | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_term_fits.probe.F7@raw | 285 | 0.786 | 0.509 | 0.071 | 0.770 [0.700, 0.843] | False | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_value_kind.probe.F7 | 278 | 0.655 | 0.859 | 0.109 |  | False | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_value_kind.probe.F7@full | 278 | 0.658 | 0.832 | 0.116 |  | False | full | True | True |
| bench/results/2026-10-04/x3.json#neon_value_kind.probe.F7@latest | 278 | 0.658 | 0.832 | 0.116 |  | False | nested | True | False |
| bench/results/2026-10-04/x3.json#neon_value_kind.probe.F7@raw | 278 | 0.637 | 0.841 | 0.112 |  | False | nested | True | False |
| bench/results/2026-10-04/x4.json#neon_ontology_fits.probe.F9@teacher-0.3 | 190 | 0.768 | 0.458 | 0.093 | 0.860 [0.818, 0.914] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_ontology_fits.probe.F9@teacher-0.3-minus-opus | 116 | 0.819 | 0.383 | 0.086 | 0.914 [0.867, 0.975] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_ontology_fits.probe.F9@teacher-0.5 | 190 | 0.768 | 0.457 | 0.110 | 0.866 [0.821, 0.922] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_ontology_fits.probe.F9@teacher-0.5-minus-opus | 116 | 0.819 | 0.381 | 0.109 | 0.912 [0.864, 0.971] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_ontology_fits.probe.F9@teacher-off | 190 | 0.821 | 0.405 | 0.050 | 0.892 [0.855, 0.933] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_ontology_fits.probe.F9@teacher-off-minus-opus | 116 | 0.871 | 0.317 | 0.122 | 0.938 [0.898, 0.984] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_term_fits.probe.F7@teacher-0.3 | 285 | 0.761 | 0.494 | 0.116 | 0.778 [0.733, 0.838] | True | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_term_fits.probe.F7@teacher-0.3-minus-opus | 187 | 0.759 | 0.484 | 0.137 | 0.772 [0.719, 0.844] | True | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_term_fits.probe.F7@teacher-0.5 | 285 | 0.765 | 0.508 | 0.086 | 0.763 [0.728, 0.813] | True | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_term_fits.probe.F7@teacher-0.5-minus-opus | 187 | 0.770 | 0.489 | 0.108 | 0.763 [0.715, 0.831] | False | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_term_fits.probe.F7@teacher-off | 285 | 0.761 | 0.507 | 0.073 | 0.776 [0.706, 0.853] | True | nested | True | True |
| bench/results/2026-10-04/x4.json#neon_term_fits.probe.F7@teacher-off-minus-opus | 187 | 0.754 | 0.517 | 0.090 | 0.755 [0.671, 0.854] | False | nested | True | True |

| file#task | best tier | best cite | verdict | reason | head_adds_nothing |
|---|---|---|---|---|---|
| bench/results/2026-10-04/k2.json#neon_annotate | - | - | **c** | killed: no eligible candidate: calibrated (selection 'none' is not 'nested'); probe (no pooled items) |  |
| bench/results/2026-10-04/k2.json#neon_aspect | - | - | **c** | killed: no eligible candidate: calibrated (selection 'none' is not 'nested'); probe (no pooled items) |  |
| bench/results/2026-10-04/k2.json#neon_ontology_fits | probe | bench/results/2026-10-04/x3.json#neon_ontology_fits.probe.F9 | **c** | killed: beats_lookup_novel failed (nll_sign_test_failed) | False |
| bench/results/2026-10-04/k2.json#neon_term_fits | probe | bench/results/2026-10-04/x3.json#neon_term_fits.probe.F7 | **b** | proposer-only: beats_lookup_novel passed; failed non_inferior_acc, card_sign, ece | False |
| bench/results/2026-10-04/k2.json#neon_value_kind | probe | bench/results/2026-10-04/x3.json#neon_value_kind.probe.F7 | **c** | killed: beats_lookup_novel failed (auroc_not_applicable_k4) |  |
