# M2 analysis plan (pre-run): X1, the zero_shot and calibrated tiers, X2

*Written 2026-10-01 on `feat/m2-evidence` (from `main` at `97f7088`, the G1 merge commit), before
any M2 statistic was computed on real labels. This file and the code it describes are committed
and pushed **before the first run on the snapshot's labels**, so the commit is tamper-evident
evidence that every choice below was made before a result existed. Until then the code ran only
on synthetic data (generated labels with planted signal, fake feature stores, the hermetic
fixtures); nothing combined a model score with a silver label (DESIGN, implementation note "M2
pre-run disclosure").*

The code: `src/mesa_clm/bench/framing.py` (X1), `bench/cells.py` (the tier cells), `bench/x2.py`
(X2), `bench/registered.py` (the registered inputs), `bench/run.py` and `cli.py` (the verbs),
`bench/stats.py` (rule R, the mean-AUROC rule R of §5), `learn/calibrate.py` (Platt, temperature),
`learn/features.py` (the manifest, the closed-choice texts), `learn/offline.py` (offline scoring),
`bench/results.py` (the cell schema), `scripts/x1_latency.py` (the timing run). Symbols are in
`code font`; "PR" is DESIGN.md "Pre-registration (G1)".

This is a reading of the frozen text, not an amendment. It merges the two drafts written in
parallel (X1; the tiers and X2) as revised after three independent reviews (faithfulness,
statistics, integrity). The reviews changed, before any run: the shuffle comparator (§5), the
reading of X1's rule (2) (§7.5–§7.6), one floor reading for every fit (§6.3), one owner of the
citable cells (§8.6), the candidate-only probe (§7.11), the temperature bounds (§6.5), the
registration of the run configuration and of the snapshot (§1, §13), the replay's own checks
(§12.6) and the run protocol (§14). A finishing pass, still before any run, made every producer
apply the registration itself (§1.3, §13.2), added X2's comparison of the recomputed controls with
M0's (§10.1) and a test that holds Appendix B to the code (§0.2). A second review round (faithfulness
and code), still before any run, added to the registration the framings lock, each model's
fingerprint and X2's dump as the producers see them and the grid each run evaluated (§13), widened
the replay and the recompute to every number the reports print (§12.6), kept X1's trace pointer in
the tier cells (§9.5), named three readings it found unstated (§4.2, §4.3, §9.9) and the failure path
of a fit that does not converge (§6.4, §6.5, §10.2), and changed the timing run's order (§11.3). If
this plan and the frozen text disagree, the frozen text wins and the disagreement is a bug in this
plan, fixed only by amendment (§15).

## 0. How the frozen text is read

0.1 Each frozen rule is read as one deterministic algorithm. Where the text admits more than one
reading, the reading chosen is named with every alternative found and why it was rejected. The
choice is the **more conservative** one: the one less likely to adopt an arm, a model or a change
of production decision on weak evidence. Where an alternative is not a reading of the words at all
(it computes something the text does not name), that is said instead.

0.2 Every constant, seed and threshold is fixed here and listed in Appendix B; the run records
them (`x1.json#/x1/config`, every calibrated cell's `diagnostics.fitters`), and
`tests/unit/test_m2_plan_constants.py` parses Appendix B and §1.3 and fails when a value there is
not the one the code applies (or a name in the "where" column does not exist).

0.3 "Conservative" never overrides the text: a reading that the words exclude is not available
because it would be safer.

## 1. Inputs

1.1 **Labels.** The registered snapshot `bench/snapshots/2026-09-29.parquet`, `labels_sha256`
`aafd18f8cea7992c9a8e0d12ded5a60ae2534a6677e911c28c2b05deaa1b752c` (the sha256 of its bytes) and
`labels_content_sha256` `5c60a8a69cf71d90df776cc331ba1347a8f7f3f074e75dca6ee6f95efb987b2b` (D30;
both are `bench/results/2026-09-29/baselines.json`'s, `mesa_clm.bench.registered.REGISTERED`). It
is the only label source of every M2 verb: `bench framing`, `bench run` and `bench x2` refuse any
other file (by both hashes) and never write a snapshot (the M0 verbs' `bench/snapshots/<date>`
fallback is not used). They load the snapshot's rows into a scratch label store
(`bench.run.snapshot_store`), so the bench tasks are the snapshot's by construction and the
per-host sidecar is never read; the scratch store's content hash must equal the registered one.
Rejected: reading a sidecar whose content equals the snapshot (`bench baselines`' route), which
depends on what a host's sidecar holds on the day of the run.

1.2 **Tasks and policy.** The five bench tasks of `bench/tasks/neon.py`: `neon_term_fits`
(`term.fits`), `neon_ontology_fits` (`column.ontology_fits`; "ontology_fits" in the frozen X1
grid), `neon_annotate` (`column.annotate`), `neon_aspect` (`column.aspect`), `neon_value_kind`
(`avu.value_kind`), at the **shipped** policy's `min_weight` (D9: 0.5; 0.6 for `column.aspect`),
no override (the configured `policy.policy_path` is not read by the M2 verbs).

1.3 **Counts, before any statistic.** Every task's item count, class counts, items per card and
`min_weight` must equal the published ones (`baselines.json`, cells `<task>.baseline.lookup_prob`:
`counts.n`, `counts.class_counts`, `diagnostics.per_fold_n`, `min_weight`); otherwise the verb
refuses (`registered.check_tasks`), and so do X1, the tier cells and X2 themselves for any task
whose labels claim the registered snapshot (`registered.check_task`). term.fits 285 items (86 Yes / 199 No), per card 37, 40, 41,
48, 45, 33, 41 (cards in sorted order: brd_countdata, brd_perpoint, bet_archivepooling,
bet_expertTaxonomistIDProcessed, bet_fielddata, bet_parataxonomistID, bet_sorting);
column.ontology_fits 190 (76 / 114), per card 21, 27, 17, 44, 28, 21, 32; column.annotate 98
(63 / 35); column.aspect 60 (classes 0–5: 12, 10, 18, 11, 6, 3); avu.value_kind 278 (133, 62, 24,
59). These are counts: no label value beyond the published class counts is read.

1.4 **Framings.** F1, F4, F7, F9 exactly as `framings.lock.json` pins them at G1, lock sha
`b432d32a7536c8f455098ae4a23139f6badbae7bc181a3a64853df3e80921eca` (each cell and `x1.json` record
the `question_key`s and the lock sha; a run under another lock is not registered, §13.2). The F1
sentence is each task's own frozen question (the frozen "`Q_TERM_FITS`" names term.fits';
column.ontology_fits' locked F1 asks that task's sentence). Anchors: the two registry anchors only
(rule (6) withdrawn, §7.10). The closed choices have one framing, F7 (`framings.FRAMINGS`).

1.5 **Models.** `clm-latest` through the lock's pinned head (its numpy export, whose `source_sha256`
must be the lock's head sha, else refused: K4, `framing.scorers_for_lock`) and `clm-raw`, under the
serving lock of G1 (`serving/serving.lock.json`, lock sha
`dd33f9fedbaee0129b09a21311227118461bed9326c942c2af7d0bd9b31b5237`). Each model's D5 fingerprint is
registered (`bench.registered.FINGERPRINTS`): `encoder_fp` c3b3d5e1a283, `clm_model_fp` 78be8c462b2e
(`clm-latest`) and 9f44b0301ee3 (`clm-raw`), the vendored schema's sha256 and that serving lock's
sha; a run whose cells carry another is not registered (§13.2). An **arm** is a (framing, model)
pair, written `F7@clm-latest`: 8 per X1 task, 16 in the grid.

1.6 **Vectors.** The feature store of `encoder_fp` c3b3d5e1a283, opened through
`FeatureStore.for_lock` (it refuses another encoder, format or vector recipe): float32 vectors
(format 2), projections cached for `clm_model_fp` 78be8c462b2e, the store the 200-group
cross-check passed on (`bench/results/2026-10-01/x1_crosscheck.json`). M2 reads it and never
embeds. An item without a manifest text, a vector or a token count, or a truncated text, stops the
verb with an error naming it before any statistic; nothing is imputed or dropped (label-free
coverage before this commit: all 911 identities of the five tasks have every text they need, 1,953
texts, 0 truncated).

1.7 **X2's AnyJev L2 dump.** `bench/baselines/anyjev_l2_2026-09-29.json`, sha256
`557c53b7284749cc42f5a9ff1056f93721e891ff40b80713572e8457207c6483`, refused otherwise.

## 2. Item sets

2.1 The items of a task are its bench task's: one per D1 identity `(task_key, target_sha256,
option_key)` from `learn.labels.labelled_targets(fold_only=True)` at the policy `min_weight`, the
D30 fold filter applied **before** the per-identity highest-weight selection, then the defensive
`_eligible` filter. Label index 0 is Yes, the positive class of a two-class task.

2.2 X1 scores the **same items** under every framing and model. Each item's texts come from
`learn.features.manifest` (label-free; builder key order restored), keyed by (task, framing,
target, option, role): F4/F7/F9 the target's `context` and `anchor` and the item's `candidate`; F1
the pair's `context`, `noul_true`, `noul_false`. An item is never dropped from one framing or model
only.

2.3 Clusters are the items' cards: 7 per task.

2.4 **Weights.** D20's label weights are sample weights in every fit (Platt, temperature, never the
X2 replica, §10.2). Every metric, Δ, bootstrap and sign test is an **unweighted** mean over
held-out items, as the M0 cells and the MDE simulation are, and rule R is a statement about
held-out items. Stated in advance: here a weight is a function of the label source (plan §5.1:
`consensus_all` 0.8 and `consensus_majority` 0.6 are the Yes items, `consensus_negative` 0.5 the
No items), so a weighted fit's intercept leans towards Yes by about ln(w̄_Yes / w̄_No) in log-odds:
from the published label sources, term.fits w̄_Yes = (16·0.8 + 70·0.6)/86 = 0.637, a lean of
+0.24; column.ontology_fits 0.6 against 0.5, +0.18 (column.annotate the same, but it has no
calibrated cell, §6.3; avu.value_kind's class 3, 59 items at 0.5, against 0.6 for the rest moves
only its one temperature). The calibrated cells' mean p(Yes) therefore sits above the Yes rate
and their NLL and ECE are higher than an unweighted fit's (in the statistics review's synthetic
check at term.fits-like weights: mean p − rate +0.039, NLL 0.4865 vs 0.4819, ECE 0.0237 vs 0.0185).
Every arm and tier shares it; §6.6 reports the unweighted fit next to every calibrated cell so a
later ECE gate (K2(a): ECE ≤ 0.08) can be read with it. Rejected: weighting the metrics (the
registered metrics and the MDE are unweighted).

## 3. Scores

3.1 **F4/F7/F9** on model m: `s = scale_m·(zs·zc − zs·za)/T`, `T = 1` (plan §4.1). `clm-latest`: zs,
zc, za are the L2-normalised head projections of the stored float32 vectors (state head for the
context, action head for candidate and anchor; `offline.store_vectors`), `scale =
min(exp(logit_scale), 100)` of the head (the pinned head's 100.811, clamped to 100). `clm-raw`: the
L2-normalised raw 4096-d vectors, `scale = 100`. Cosines in float64 (`offline.pairwise_s_c`). The
frozen "100·" is therefore both models' scale; the scale used is recorded per model.

3.2 **F1** on model m: `s = scale_m·(zs·z_true − zs·z_false)`, the noul logit difference;
`σ(s)` is CLM's `p_true`.

3.3 **Closed choices** (F7): logits `scale·zs·z_k` over the task's K options in option order
(`OfflineScorer.logits`).

3.4 **Zero-shot probabilities**: `p = σ(s)` (D7); a closed choice's softmax over its logits at
temperature 1 (`render.answer_from_logits`' arithmetic).

3.5 Every score is float64 from float32 vectors. AUROC is computed on `s`, not on `σ(s)`: they are
rank-equivalent in exact arithmetic, but σ rounds to exactly 1.0 above s ≈ 37 in float64 (a
cosine difference of 0.37 at scale 100), which would add ties the score does not have.

## 4. Pooled AUROC with its cluster CI (rule (1), first two conditions)

4.1 "Pooled AUROC" of an arm is one AUROC of `s` over all the task's items (Mann-Whitney, ties
one half, positive class Yes; `stats.auroc`): no folds and no fit, since `s` has no fitted
parameter. Rejected: the AUROC of the pooled LOCO-Platt probabilities. Rule (1) qualifies an
arm's ranking before anything is fitted, the third condition compares it with the fit-free shuffle
(§5), which needs the same raw scores, and the calibrated AUROC mixes seven fitted slopes into the
ranking (a fold with `a < 0` inverts its card's ranking only there). That number is reported (the
arm cells' `metrics.auroc`), never used by a rule.

4.2 "Cluster-LB": the card-cluster bootstrap (`stats.auroc_ci`): B = 2000 replicates, seed 0
(`numpy.random.default_rng`), each drawing the m distinct cards (sorted) with replacement and
keeping every item of a drawn card once per draw; the lower bound is the 0.05 quantile (numpy
linear interpolation) of the replicates where both classes are present (`n_valid` recorded): a
one-sided 95% bound. Rejected: the 0.025 quantile, the lower end of a two-sided 95% interval,
which would be more conservative for rule (1) and for `beats_lookup_novel`: the frozen rule R
names "the one-sided 95% lower bound", the "cluster-LB" of rule (1) and of the lookup gate is the
same bootstrap's bound, and M0's pre-registered cells already carry it (`auroc_ci` with `alpha`
0.05, `bench/results/2026-09-29/baselines.json`). Every reported interval `[lower, upper]` is the
pair of one-sided 95% bounds (the 0.05 and 0.95 quantiles, `stats.BootstrapCI`), a **90%**
two-sided interval: `x1.md`, the tier and X2 tables and `bench table` label it "one-sided 95%
bounds; 90% interval" (a label, no number changes).

4.3 "AUROC cluster-LB > 0.5" is strict; "AUROC ≥ 0.60" compares the point estimate of 4.1,
inclusive. "Cluster-LB > 0.5" is the bootstrap's lower bound alone, a CI condition, not rule R: the
frozen text writes rule R as "≻" (rule (1)'s third condition, the lookup gate's NLL half), says
"Effect floors (AUROC ≥0.60, ECE ≤0.08) are always paired with a CI condition", which is what the
bound is next to rule (1)'s floor, and keeps the lookup gate as "cluster-LB > 0.5". Not adopted: the
reading of the frozen minimum-detectable-effect section, which simulated its "AUROC gate", on the
full population and on the novel keys, as rule R against the constant prior, the card sign test (≥
10 items with both classes, ⌈0.8·m_c⌉ wins, m_c ≥ 4) included. Read as the gate of rule (1)'s first
condition (full population) and of the lookup gate's AUROC half (novel keys), it adds a condition
the definitions do not name, and §0.3 forbids a stricter reading the words exclude. Stated in
advance: M0's "AUROC gate: MDE" column (`mde.json`) therefore describes a stricter gate than the one
run here, so the AUROC condition as run can pass at a smaller effect than that column says; for
example `column.annotate`'s novel-key AUROC bound can pass although its NLL half cannot
(`insufficient_clusters`, one card with ≥ 10 novel items), so `beats_lookup_novel` stays false for
it as stated in advance. The same reading applies to `beats_lookup_novel`'s AUROC half (§9.9).

## 5. The within-card state shuffle (rule (1), third condition)

5.1 **A shuffle** permutes the states (contexts) of one card's targets among themselves: for card
k with distinct targets `T_k` (sorted by `target_sha256`), a derangement π of `T_k` gives every
item of target t its own candidate and anchor scored against the context of π(t):
`s_π(i) = scale·(z_π(t)·z_c − z_π(t)·z_a)`. The score of every item under every context of its
card is computed once (`framing.context_scores`, the arithmetic of §3.1; the item's own column is
its `s`) and stored (`x1_items.parquet#s_contexts`), so the shuffles are recomputed from the
items file alone.

5.2 **The draws.** K = 200 derangements per card, uniform, by rejection sampling
(`rng.permutation(m)` until no target keeps its own context), from one
`numpy.random.default_rng(0)` stream, cards in sorted order, then k in order
(`framing.shuffle_draws`). Every arm of a task uses the same draws (common random numbers), and a
nested fold uses the task's draws restricted to its training cards (a draw never crosses cards).

5.3 **The statistic.** ΔAUROC(real − shuffle) = AUROC(s) − (1/K)·Σ_k AUROC(s_πk) over the items
the procedure sees; the reported shuffled AUROC is the mean over the K draws, with its cluster CI
(`stats.mean_auroc_ci`).

5.4 **"ΔAUROC ≻ 0" is rule R** with A the real score and B the K shuffles
(`stats.rule_r_auroc_mean`): (i) in each of the B = 2000 card-cluster replicates (seed 0) Δ_b =
AUROC_b(s) − mean_k AUROC_b(s_πk) is computed on the **same** resampled cards, and the 0.05
quantile must be > 0; (ii) on each card with ≥ 10 items and both classes, Δ_card = AUROC_card(s) −
mean_k AUROC_card(s_πk), and A must win (Δ_card > 0, strict) on ≥ ⌈0.8·m_c⌉ of the m_c such cards;
m_c < 4 is `insufficient_clusters`, which fails. The mean over K draws is computed exactly through
the pairwise Mann-Whitney kernel averaged over the draws (the AUROC of any weighted item set is a
weighted mean of one kernel matrix), equal to the brute-force mean of the K AUROCs to ~1e-12
(`tests/unit/test_framing_x1.py`).

5.5 **Why this reading.** The frozen words are "ΔAUROC(real − shuffle)" with the control
"within-card state shuffle": the AUROC of a shuffled scoring. Its value under the null (the
context carries no target-specific information) is the expectation over uniform derangements of
that AUROC, which the K = 200 draws estimate (a single draw's AUROC varies by about 0.02 on
synthetic data; the mean of 200 by about 0.0015). Rejected:
- **the drafts' reading**, the AUROC of each item's expected score over the other contexts of its
  card, AUROC(E_π[s_π]): it is not the AUROC of any shuffle. Averaging removes the label-irrelevant
  noise one context puts into a score while the real score keeps it, so without any target signal
  Δ is negative on average (−0.053 vs −0.001 for one random derangement over 60 synthetic null
  worlds in the statistics review; `test_the_shuffle_comparator_is_null_calibrated` checks the
  same) and rule R loses most of its power (pass rate 0.60 vs 0.96 at a target-specific card share
  of 0.3, 0.12 vs 0.72 at 0.5). It was called conservative by comparison with one random context;
  measured against the null it is biased against every arm, which the words do not ask for;
- **one seeded derangement**: unbiased, but Δ then carries the draw's own variance, which costs
  power (pass rate 0.13 vs 0.57 at card share 0.5 in the review);
- **a permutation rather than a derangement**: its fixed points score items with their own
  context and dilute Δ towards 0;
- **every derangement**: infeasible at up to 25 targets per card; 200 draws suffice (above).

5.6 Contexts come only from the item's own card: the card header is kept and the target-specific
part (column or site, aspect) moves. A card with a single target among the items would have no
other context; its items keep `s` (counted in `n_unshuffled`, diluting Δ towards 0). On the
snapshot every card has at least 5 targets per task (5–13 for column.ontology_fits, 14–25 for
term.fits; a label-free count of the identity columns).

5.7 F1 has no shuffle: its context is the mesa-anyjev per-candidate state, which embeds the
candidate, so a shuffled F1 context is a text the manifest never embedded; and rule (1) qualifies
F4/F7/F9 only.

## 6. Folds, guards, floors, calibrators and metrics

6.1 **Folds.** Leave-one-card-out over the cards of the item set (`for card in sorted(cards)`):
training items = the other cards', held out = the card's. Held-out predictions are pooled, never
fold-averaged.

6.2 **Guards** ("guards 30 train / 5 held-out per class"): a fold of a **fitted** tier is skipped
when a class has fewer than 30 training or 5 held-out items (`bench.tasks.fold_guard`). The guards
depend on labels and cards only, so every arm of a task pools the same items. Reading: (i) the
guards gate fitting, so a zero_shot cell, which fits nothing, pools every fold and records in
`guard_skipped_folds` what a fitted tier would skip; (ii) they apply to the two-class tasks (term.fits,
column.ontology_fits, column.annotate), as in mesa-anyjev's `_fit_eval_loco` and in the M0 control
cells, which pooled all seven folds of every task and only listed `guard_skipped_folds`
(`baselines.json`). Rejected, the literal per-class guard for every tier and every K: avu.value_kind's
class 2 has 24 items in all (< 30), so its calibrated cell would have no fold, and column.annotate's
zero_shot cell would keep 1 of 7 folds (6 are guard-skipped). Neither reading changes a citable M2
cell (no closed-choice cell can be cited in M2, §9.5), and both agree on term.fits and
column.ontology_fits, where no fold is guarded.

6.3 **Floor** ("floors 100 (calibrated) / 40 (probe/head)"): **no calibrator is ever fitted on
fewer than 100 labelled items**: the training items of every fold (X1's outer and inner LOCO-Platt
folds, every tier cell's fold, the 6 training cards of a nested fold); a fold below it is skipped as
`below_floor n < 100 training items (calibrated)`. The probe floor of 40 applies the same way to the
X2 replica and X1's candidate-only probe (both on two-class tasks, where a fold that passes the
30-per-class training guard already trains on at least 60 items, so that floor never binds). mesa-anyjev's `MIN_LABELS_L1 = 100` is a task-level
minimum checked once on the whole labelled set before its folds (`learn/fit.py`); the per-fit
reading is stricter (every training set is smaller than the task) and is one reading for X1 and
the tier cells alike. Rejected: the task-level check alone (the X1 draft's set-level reading left
X1's inner 5-card training sets unchecked). On the snapshot the choice changes nothing: the
smallest training sets (from the published per-card counts) are term.fits 237 (outer) and 192
(inner, 5 cards), column.ontology_fits 146 and 114, avu.value_kind 231; column.annotate (98 items)
and column.aspect (60) have no calibrated fold under either reading.

6.4 **Platt** (`calibrate.fit_platt`, the code base's one Platt, used by X1's LOCO-Platt, every
calibrated rank_fit cell, the K = 2 closed choice on the logit difference, and M4's
`calibrators.json`): it minimises the weighted mean NLL of the labels, Σ_i w_i·[−y_i·log σ(a·x_i + b)
− (1 − y_i)·log(1 − σ(a·x_i + b))] / Σ_i w_i + (λ/2)(a² + b²), λ = 1e-6, by Newton's method from
(0, 0) with Armijo step halving (constant 1e-4, at most 60 halvings; inside the quadratic region,
a predicted decrease below 1e-12 of the loss, the full step is taken), at most 100 iterations,
stopping when every gradient component is below 1e-10. The targets are the labels themselves
(`calibrate.PLATT_TARGETS = "hard"`, frozen in this commit). The ridge only keeps a separable fold
finite; `a < 0` is flagged `inverted` and kept, never refused or corrected (plan §5.3). Rejected:
Platt's (1999) smoothed targets (N₊+1)/(N₊+2) and 1/(N₋+2) (the X1 draft; scikit-learn's sigmoid
calibration): they are a regulariser for small calibration sets of SVM margins, every fit here has at
least 100 items (§6.3), they would be computed from unweighted class counts while the loss is
D20-weighted (a combination with no published definition), and they move every probability away
from 0 and 1 by design, an offset the bench's NLL and ECE would then measure as miscalibration. With
the hard targets the calibrator is the maximum (weighted) likelihood fit of exactly the loss rule R
compares. The choice is made before any result and is identical for every arm and tier, so it cannot
favour one. **A fit that does not converge** (Newton reaches 100 iterations, or the line search
finds no decrease, before every gradient component is below 1e-10) is used as it is: its
`converged: false` is recorded in the fold's calibrator (X1's `loco.folds`, a tier cell's
`diagnostics.per_fold`), and the fold is never refitted or skipped.

6.5 **Temperature** (closed choices with K > 2): T minimises the weighted mean NLL of
softmax(logits/T) over **T ∈ [1e-4, 1e4]**: the NLL is convex in β = 1/T, so the root of its
derivative is bisected in log β to an interval below 1e-12 (at most 200 halvings; no random start);
a T on a bound is used as it is and reported (`at_bound`, in the fold's calibrator record), never
refitted or skipped. The drafts' bounds [0.01, 100] were widened so that a compressed logit scale
(options whose cosines differ by about 1e-4 at scale 100, a collapsed head) is fitted rather than
clamped.

6.6 **Report-only sensitivity** (§2.4): every calibrated cell (`diagnostics.sensitivity_unweighted`)
and every X1 arm (`x1.json#/x1/tasks/<task>/arms/<arm>/sensitivity`) also reports the same folds
with the calibrator fitted **without** the label weights: NLL, ECE, accuracy, Brier and the
calibration in the large (mean predicted probability minus frequency, per option; an X1 arm, which
is two-class, reports it for Yes, `mean_p_minus_rate`). No rule reads it.

6.7 **Metrics**, unweighted over the pooled held-out items (`bench.baselines.pooled_metrics`):
accuracy (argmax of [p, 1 − p], p = 0.5 counting as Yes), macro-F1, Brier, NLL (vendored, p
clipped at 1e-12), ECE (15 equal-mass bins), cov@5%, cov@10% and AURC (tie-invariant, D33), the
AUROC of the index-0 probability with its cluster CI (two-class tasks; `None` for K > 2),
`threshold_cp{0.05, 0.10}` (the smallest statistic t with ≥ 30 items at or above it and a one-sided
95% Clopper–Pearson error bound ≤ risk; `p_fit` for rank_fit, `max(probs)` for a closed choice). For
rule R each arm keeps its per-item [p, 1 − p]; an item's NLL is −log max(p_label, 1e-12).

## 7. X1's decision rule (`framing.decide`, per task)

7.1 Arms: §1.5. F1 arms are controls: never qualified, never chosen.

7.2 **Rule (1).** An F4/F7/F9 arm qualifies iff its AUROC lower bound > 0.5 (§4.2), its AUROC ≥ 0.60
(§4.1) and ΔAUROC(real − shuffle) passes rule R (§5.4). The **eligible** arms are the qualified ones
of both models.

7.3 **Rule (5)** is checked first: no eligible arm on either model → outcome `K1` (the task's
zero_shot/calibrated tiers audit-only, `ols_rank` proposals, M4 probe-first, as K1 says); rules
(2)–(4) do not run.

7.4 **Rule (2): which arms are ranked.** The qualified arms of `clm-latest`, the model rule (3)
defaults to; when `clm-latest` has none, those of `clm-raw`. Rule (2) chooses the framing and rule
(3) the model for it. Rejected: ranking both models' arms together ("≻ a more cacheable arm" would
then compare across models, and rule (3) could be asked of a framing whose `clm-latest` arm did not
qualify); choosing a framing per model and comparing the two choices (rule (3)'s comparison would
mix a change of framing with the change of model).

7.5 **Rule (2): the winner and cacheability (the literal reading).** W = the ranked arm with the
lowest pooled LOCO-Platt NLL (an exact tie in NLL falls to the cost order of §7.7). F7 and F9
(state-only contexts, the same for every question about a target) are more cacheable than F4 (a task
sentence appended). **If W is F7 or F9, W is the framing**: no arm is more cacheable than it, so
"if not ≻ a more cacheable arm, take the more cacheable" cannot apply. If W is F4, the candidates
are the more cacheable qualified arms of the same model that W is not ≻ on NLL (rule R, W as A):
none → F4; one → it; F7 and F9 both → §7.6.

7.6 **"F7/F9 tie"** arises only when the cacheability step leaves both F7 and F9 (W = F4 and W is
≻ neither): let V be the one with the lower NLL (exact equality: §7.7); V is chosen if V ≻ the other
on NLL (rule R), otherwise they tie and the one with fewer encoder tokens per target is chosen, F7 on
equal counts. "Tie" means "neither is shown better under rule R", the rule's own notion of a
difference. Rejected: **the X1 draft's reading**, under which the tie-break also applied when W
itself was F7 or F9 (W kept only if ≻ the other, else fewer tokens). With F9's context at 34.7 and
F7's at 418.2 encoder tokens per target on term.fits (35.1 and 430.9 on column.ontology_fits; a
label-free count of the manifest's contexts in the store's token column), it chooses F9 in every
comparison rule R does not separate, whichever has the lower NLL, and so moves production off the
default F7 (`framings.ACTIVE`, plan §4.2 "F7 default until A1") with no NLL evidence for F9; the words
give the NLL winner the choice and the tie-break to two equally cacheable candidates. Also rejected:
a "tie" in cacheability alone (F7 and F9 are always equally cacheable), under which the token count
could choose an arm rule R shows worse than the other.

7.7 **Tokens per target** (`_tokens_per_target`): the mean over the targets the procedure sees (the
full run: all; a nested fold: its 6 training cards') of the encoder tokens of the texts one target's
request embeds: its context and the anchor once and each labelled candidate (F1: each item's context
and its two noul texts); the store's exact counts. The cost order for exact NLL ties: F7/F9 before
F4, fewer tokens, then F7.

7.8 **Rule (3).** f* = the framing of §7.5–§7.6, chosen on model m0. If m0 is `clm-latest`: the model
is `clm-raw` iff (f*, `clm-raw`) qualified and (f*, `clm-raw`) ≻ (f*, `clm-latest`) on NLL (rule R);
otherwise `clm-latest`. If m0 is `clm-raw` (`clm-latest` has no qualified arm), the model is
`clm-raw`. An arm that did not qualify is never chosen.

7.9 **Rule (4).** For each model m: does F1@m ≻ e on NLL hold for every eligible arm e of both models?
If it does for either model, the decision records `d3_amendment_recommended: true` with every
comparison; nothing is adopted: the choice of §7.4–§7.8 stands and only an amendment can reopen D3
("never automatically"). Not evaluated without an eligible arm (an empty "all eligible" is not
evidence). Requiring F1 to beat the eligible arms of both models is the strictest reading.

7.10 **Rule (6)** is withdrawn (DESIGN, "The X1 anchor variant"): no anchor variant is scored,
compared or applied.

7.11 **Controls ("candidate-only probe"), report-only.**
- **The candidate-only probe** (`framing.candidate_probe`, `x1.json#/x1/tasks/<task>/candidate_probe`):
  a learned model on the candidate alone, X2's pre-registered PR #13 recipe (`StandardScaler` then
  `LogisticRegression(C=1, max_iter=2000)`, scikit-learn defaults otherwise, unweighted, §10.2) on the
  raw 4096-d vectors of the items' candidate texts (shared by F4/F7/F9, so one probe per task),
  leave-one-card-out with the 30/5 guards and the probe floor of 40 training items per fold (a fold
  whose fit stops at `max_iter` is used as it is, flagged in its record: `n_iter`, `converged`,
  `warnings`). It
  measures how much label signal a learned model extracts from candidate identity alone (for example
  a per-ontology base rate in column.ontology_fits): AUROC with cluster CI, NLL, accuracy, ECE,
  Brier. Rejected: the drafts' zero-parameter score as "the probe": in this code base a probe is a
  learned model (plan §5.3), and the recipe was pre-registered with X2.
- **The mean-context score** (kept from the drafts, renamed): the arm's own score with the context
  replaced by one fixed vector, the L2-normalised mean of the task's contexts on the arm's side,
  `scale·(z̄·z_c − z̄·z_a)`; its AUROC with CI and LOCO-Platt metrics per F4/F7/F9 arm. It isolates the
  candidate side of the arm's own geometry; like the drafts' shuffle it averages context noise away,
  so it reads optimistic against the real score. z̄ uses texts only.
- Neither is read by rules (1)–(5).

7.12 **Undecidable.** If eligible arms exist but LOCO-Platt pooled no item (every fold guarded or
below the floor), rules (2)–(4) cannot rank: outcome `undecidable`, no choice, no amendment (F7 stays
the production default). Not expected (no term/ontology fold is guarded, and every training set is
above the floor, §6.3).

7.13 **Rule R on NLL** (used by §7.5, §7.6, §7.8, §7.9): A and B are two arms' per-item LOCO-Platt
probabilities on the same pooled items. (i) The paired card-cluster bootstrap of Δ = NLL(B) − NLL(A)
(B = 2000, seed 0) has a lower bound > 0; (ii) A's NLL is below B's on ≥ ⌈0.8·m_c⌉ of the m_c cards
with ≥ 10 pooled items; m_c < 4 → `insufficient_clusters`, not ≻ (`stats.rule_r("nll", …)`). One
cluster resample is shared by all comparisons on one item set (identical to drawing it per call: the
seed and the sorted clusters fix it).

## 8. Nesting and X1's outcome

8.1 **Full run**: §4–§7 on the 7 cards. Its decision is the production choice (A1 per task). Its
sixteen arm cells (§12.1) are `exploratory: true` (D27: a full-data selection).

8.2 **Nested run**: for each outer card h (sorted), §4–§7 are re-run on the items of the other 6
cards only (`framing.nested_selection`): AUROC CIs and the shuffle's rule R over those 6 clusters
(B = 2000, seed 0, m_c from those cards), inner LOCO-Platt leaving out one of the 6 at a time with
the guards and the floor per inner fold, NLL rule R over the inner pool, tokens per target over the 6
cards' targets. The shuffle draws are the task's, within cards (§5.2). Nothing reads card h.

8.3 `fold_choices[h]` = the inner outcome and, for a choice, {framing, model}; the inner decision
trace is kept in `x1.json`.

8.4 **Outer guards are the tier cells'** (§9.4): X1 records only the inner outcome per fold. A
calibrated tier fold that the guards or the floor skip keeps its X1 choice with `evaluated: false`;
zero_shot fits nothing and pools it.

8.5 **A fold whose inner outcome is K1 or undecidable chooses no arm**: recorded in `fold_choices`
and `skipped_folds` (`inner_k1`, `inner_undecidable`), and the tier cells skip it with that reason
in both tiers, whatever the guards say. Reason: the procedure abstains on that fold (under K1
production would ask `ols_rank` and have no probability); predicting with a default arm would report
an arm the procedure rejected; the skip depends only on the 6 training cards. The citation test
counts such a fold as disagreeing with A1, and `n_folds` drops.

8.6 **One producer of the citable cells.** X1 chooses; it writes no `<task>.<tier>.<A1>` cell. The
tier run (§9) is the only producer of the citable-form rank_fit cells, taking X1's outcome as data
through `framing.selections_from_json`, which replays the file first (§12.7). Rejected: x1.json as
their producer (the X1 draft): the drafts produced the same keys twice, numerically identical with an
A1 but with different `fold_choices` and diagnostics schemas and different cells under K1; the tier
cells carry every frozen field (`mean_state_cos`, `calls`, `input_tokens`, `ms_per_decision`
included, §9.10).

8.7 **Agreement with A1**: a fold agrees when it chose the same framing **and** model (the stricter
reading: a cell's fingerprint names the model). The citation test needs ≥ 5/7; a fold without an arm
disagrees (`diagnostics.fold_agreement_with_a1 = {agree, folds}`).

8.8 **No A1** (the full run's outcome is K1 or undecidable): there is no nested cell. As K1 says
("its zero_shot/calibrated tiers become audit-only"), the task's tier cells are reported for audit
only, for the production default arm F7@clm-latest in every fold: `selection: "none"`,
`pre_registered: false`, `exploratory: true`, the outcome in `notes`; the per-fold choices stay in
`x1.json`. Rejected: the X1 draft's `<task>.<tier>.none` cells pooling only the folds that did
choose (an arm mixture production would never serve, under a key that names no framing).

## 9. The zero_shot and calibrated cells of every task (`bench/cells.py`, `tiers.json`)

9.1 **Scores**: §3, offline from the store for the cell's arm (`cells.score_arm`), the texts matched
to each item's D1 identity through the manifests (`cells.TextIndex`); `s_c` does not depend on the
other candidates of a group (D2), so a candidate scored alone equals what serving computes for it in
any candidate set, the full registry request that the aspect mask then masks included (Q3).

9.2 **rank_fit** (term.fits, column.ontology_fits): zero_shot `p_fit = σ(s_c)` (D7), calibration
`uncalibrated` (D6); calibrated: in every evaluated fold a weighted Platt (§6.4) on the fold's training
items (the six other cards), held-out `p_fit = σ(a·s_c + b)`, calibration `platt`.

9.3 **Closed choices** (column.annotate K = 2 `ANNOTATE_OPTIONS`; column.aspect K = 8
`ASPECT_OPTIONS`; avu.value_kind K = 4 `VALUE_KINDS`, wire keys `framings.VALUE_KIND_KEYS`), F7: zero
shot is CLM's softmax at temperature 1 (`uncalibrated`); calibrated K = 2 a weighted Platt on the
logit difference `d = logit_Yes − logit_No` (`platt`; zero shot is a = 1, b = 0), K > 2 weighted
temperature (§6.5, `temperature`).

9.4 **Guards and floors**: §6.2–§6.3. Stated in advance: column.annotate and column.aspect never
have 100 training items in a fold, so their calibrated cells carry no metrics and list every fold
`below_floor`; every term.fits and column.ontology_fits fold passes both (PR, LOCO: "all 14
term/ontology folds pass at 0.5").

9.5 **Arms and selection.** The cell code never chooses an arm.
- term.fits and column.ontology_fits: the **nested cell** `<task>.<tier>.<A1 framing>` scores (and at
  calibrated fits) outer fold c under fold c's arm. `fold_choices` lists every outer fold: `decision:
  "arm"` with the fold's framing, model, `question_key`, `clm_model_fp`, `evaluated` and X1's record
  as `framing.selection` gives it (`outcome`, the inner outcome, and `x1_trace`, the JSON pointer of
  the inner trace in `x1.json`), or the inner skip (§8.5). A record key that the entry sets itself
  is refused (`cells.FOLD_CHOICE_KEYS`), never overwritten. `selection:
  "nested"`, `pre_registered: true`, `exploratory: false` in the registered run; the cell's identity
  (`question_key`, `fingerprint`, `model`) is A1's, the arm production serves.
- The **full-data cell** `<task>.<tier>.<A1 framing>@full`: A1 in every fold, `selection: "full"`,
  `exploratory: true` (D27); the citation pattern admits no `@`.
- No A1: §8.8.
- column.annotate, column.aspect, avu.value_kind: F7 with `clm-latest` (the production default) is
  `<task>.<tier>.F7`, `selection: "none"`, `pre_registered: true`, `exploratory: false`,
  `fold_choices: {}` (nothing is chosen); `clm-raw` is reported as `<task>.<tier>.F7@clm-raw`,
  `pre_registered: false`, `exploratory: true` (the frozen text names no model comparison for these
  tasks). No closed-choice cell can be cited in M2: the citation test requires `selection:
  "nested"`, K > 2 has no AUROC (so `beats_lookup_novel` is false, PR MDE table "not applicable"), and
  column.annotate is below the floor with `insufficient_clusters` on novel keys. Whether a fixed arm
  may count as nested is left to a later amendment.
- The amendment A1 that records X1's choice and rotates the framings lock is the outcome X1 was
  registered to produce, not a change to these cells' pre-registration; it does not make them
  exploratory.

9.6 **Identity fields** (PR "Cell fields"): `task`, `task_id`, `task_key`, `tier`, `framing`,
`variant` (`None`, `full` or a non-production model; the results key is
`<task>.<tier>.<framing>[@<variant>]`), `model`, `question_key` (`framings.framing(task,
framing).question_key`), `fingerprint` (`ServingLock.fingerprint(model).as_dict()`: `encoder_fp`,
`clm_model_fp`, `schema_sha256`, `serving_lock_sha`), `labels_sha256`, `labels_content_sha256`,
`feature_spec` (`None`), `label_sources`, `min_weight`, `masked` (true: the item set carries every
option restriction serving applies; column.ontology_fits' items exist only for aspect-allowed
ontologies, `mask: "aspect"`), `teacher: false`, `teacher_in_test: false` (checked: no pooled item
has a teacher source), `loco: true`, `selection`, `pre_registered`, `exploratory`, `servable` (a
non-control framing under `clm-latest` or `clm-raw`; a zero_shot cell is servable and never auto,
D6; X2 cells are not), `n_folds`, `skipped_folds`, `guard_skipped_folds`, `fold_choices`.

9.7 **Counts** over the pooled items: `n`, `n_neg` (two-class: items not Yes; `None` for K > 2),
`n_nonmodal`, `class_counts`; the task totals in `diagnostics.task_counts`.

9.8 **Metrics**: §6.7. A cell with no evaluated fold has `metrics: null`.

9.9 **Baselines** over the cell's own evaluated folds and items (so comparisons are paired):
`majority_acc`, `lookup_acc`, `lookup_nll`, `lookup_prob_acc` (`evaluate_lookup`); for the novel
keys (held-out items whose lookup key no training card of their fold carries) `novel_key` is **this
cell's** predictions on them and `novel_key_lookup` is `lookup_prob`'s on the same items. `lopo`
(the leave-one-product-out control) is the exception: a task-level, unpaired control over every item
of the task with its product folds, whatever folds the cell evaluated, as in M0's cells (another
split, so it cannot be paired with a card fold). `beats_lookup_novel` := `novel_key.auroc_ci.lower >
0.5` **and** this cell ≻ `lookup_prob` on NLL over the novel-key items (rule R); false, with the
reason in `beats_detail`, when the AUROC is undefined (K > 2, or one class among the novel items),
when there is no novel item, or when fewer than two cards carry one. `beats_detail` keeps the AUROC
bound, the full rule R record and the reason. The AUROC half is the bound alone, not rule R (§4.3:
the minimum-detectable-effect section's rule-R reading, card sign test included, is named there and
not adopted; M0's AUROC-gate MDE describes that stricter gate). Novel-key accuracy against the
majority is reported, never gated.

9.10 **Diagnostics, units defined once for every M2 cell** (X1's arm cells use the same), all
report-only and all computed for the cell's **arm on every item of the task**, whichever folds the
cell pools (a nested cell's are A1's, the arm production serves; each fold's own arm is in
`per_fold`): `mean_state_cos`, the mean pairwise cosine between the distinct context vectors of the
task's items under that arm, on the model's side (projected for `clm-latest`, raw for `clm-raw`; the
collapse diagnostic); `calls`, the total `/v1/systemone` requests serving would make for the task's
items (one per target for a Choice framing; F1 one per labelled candidate; column.annotate and
column.aspect share one request per column in serving, counted here per task); `input_tokens`, the
total encoder tokens of the distinct texts those items need (contexts, candidates or options, the
anchor; a cold cache), from the store's exact counts; `ms_per_decision`, the wall time of the
offline replay per item (`timing_source: offline_replay`; not serving latency, which is §11.3's and
`annotate_latency.json`'s). Each calibrated cell adds per fold `n`, `n_train` and the fitted
calibrator, `inverted_folds`, `fitters` (every fitter constant, Appendix B) and §6.6's sensitivity; a
two-class zero_shot cell adds `auroc_raw_score` (the AUROC of `s_c`, or of the logit difference at
K = 2, with its CI) and `n_saturated` (§3.5).

9.11 **Per-item predictions**: every cell carries `items` (target, option, card, label, probs,
novel), so every paired comparison (rule R between cells, X3, K2(a) against AnyJev) can be
recomputed from the results file alone.

9.12 **avu.value_kind and the pre-rule**: serving answers some value kinds by rule before CLM
(`avu.pre_rule_value_kind`); the cell measures the CLM tier on all 278 pre-registered items, the
items the M0 control pools, and reports in `diagnostics.n_pre_rule` how many the pre-rule would
route away (not removed).

## 10. X2 baselines (`bench/x2.py`, `x2.json`)

10.1 **No-model controls** (majority, `lookup_prob`, novel-key, LOPO): the M0 cell
`<task>.baseline.lookup_prob` for every task, recomputed by the M0 code from the same snapshot with
the same B and seed, so it should equal the published cell of
`bench/results/2026-09-29/baselines.json` field for field. `bench x2` compares each with the
published cell and records the outcome in the file's notes (`run.m0_comparison`): report-only, a
difference is recorded, never repaired or refused.

10.2 **PR #13 replica**, as frozen: `StandardScaler` then `LogisticRegression(C=1, max_iter=2000)`
(scikit-learn from the `bench` extra; defaults otherwise: L2, lbfgs, `tol=1e-4`, unpenalised
intercept), per task leave-one-card-out on the store's **float32** L2-normalised 4096-d vectors of
`joint4096@S1` (the mesa-anyjev per-candidate state with the task question appended, the same text as
F1's context) and `joint4096@S1ns` (without the question), one row per labelled pair (DESIGN "What the
manifest embeds"). CLM's `TextCache`, which PR #13 read, holds float16 casts of these vectors; the
replica keeps float32, the inputs every M2 cell reads (a float16 cast moves synthetic predictions by up
to 4e-4). The scaler and the model see the fold's training rows only; the held-out probability is
`predict_proba` of Yes. **Unweighted**: the frozen X2 text names PR #13's recipe and no weights, PR #13
fitted unweighted, and D20's weights govern mesa-clm's own fitters. Tasks: term.fits and
column.ontology_fits (the joint specs are rank_fit probe specs, plan §5.3). Folds: the 30/5 guards and
the probe floor of 40. Cells `<task>.baseline.pr13@S1` and `@S1ns`: tier `baseline`, `feature_spec`
the spec, `fingerprint` without `clm_model_fp` (no head), `servable: false`, `selection: "none"`; the
scikit-learn version and each fold's `n_iter_` in diagnostics. A fold whose fit stops at `max_iter`
(a `ConvergenceWarning`) is used as it is and flagged in `diagnostics.per_fold` (`n_iter`,
`converged`, `warnings`), never refitted or skipped. X2 gates nothing.

10.3 **AnyJev L2** (§1.7): per-item held-out predictions joined to the bench items by the D1 identity
derived from each dump item's `state_json` (not `state_sha256`, which differs with `n_candidates`);
a duplicate identity or a matched item held out on another card is refused; an unmatched bench item
is left out (counted and listed, never imputed) and an unmatched dump item dropped (counted); the
cell's labels are the snapshot's; `<task>.baseline.anyjev_l2`, `fingerprint: null`, `question_key:
null`. Label-free before this commit: the identities derived from the dump's 285 and 190 states are
exactly the snapshot's (`tests/unit/test_x2.py`), so the join is the full set and K2(a) (M4) is paired
on it.

## 11. Report-only diagnostics

11.1 **Collapse** (`cells.mean_pairwise_cosine`): per task and framing, the mean pairwise cosine of the
distinct contexts of the items (F4/F7/F9 one per target, F1 one per pair), raw (L2-normalised 4096-d)
and projected (`clm-latest` state head), `(|Σu|² − Σ|u_i|²)/(n(n − 1))` in float64.

11.2 **Calls and tokens per target** (X1, `diagnostics.calls_per_target`,
`input_tokens_per_target`, `context_tokens_per_target`): §7.7's counts, next to the totals of §9.10.

11.3 **p50 latency** (the frozen X1 metric) comes from a separate, **label-free live timing run**
made once before the X1 run (§14.3): `scripts/x1_latency.py` takes, per task and framing, the first
N = 20 targets of the manifest in `sha256("task|framing|target")` order (`framing.latency_targets`)
and asks each once under each model through clm-serve's `/v1/systemone` (F4/F7/F9 one Choice per
target; F1 one noul per labelled candidate, summed per target), sequentially from one client, the
two models of a target back to back: `clm-latest` first at an even position of the draw, `clm-raw`
first at an odd one, so each model has 10 first asks and 10 second ones per (task, framing). It
keeps only the wall ms per target and two label-free flags per ask; every answer is discarded.
**What is warm.** clm-serve's embedder cache is keyed by text and shared by both models (RESEARCH,
"Embedder, caches, server"), the units are not restarted and nothing is cleared: a target's second
ask finds its texts cached; the anchor text is the same for every target of a task and framing, and a
candidate's text is the same under F4, F7 and F9, so an earlier ask in the run may have cached them;
texts cached before the run are unknown. The flags say so per ask: `first` (this model was asked
first for the target) and `new_text` (its requests carried a text this run had not sent before,
computed from the manifest before the ask). The drafts' order (every target under `clm-latest`, then
again under `clm-raw`) measured `clm-raw` warm throughout (rejected). Its file
(`mesa-clm/x1-latency/1`, with `ms`, `first`, `new_text`) is committed; `bench framing --latency
FILE` refuses a file about other labels or another serving lock (`labels_sha256`, `serving.lock_sha`)
and records its path, sha256 and identity in `x1.json#/x1/latency` and each arm's `{n, p50, p95}`
with the same for its first and its second asks and its `new_text_share` in its diagnostics.

## 12. Output and reproducibility

12.1 **`bench/results/<date>/x1.json`** (`mesa-clm/bench-x1/1`): the `BenchResults` fields with the
sixteen arm cells `<task>.calibrated.<framing>@<model>` (the arm's LOCO-Platt, §6; `selection:
"full"`, `exploratory: true`; `variant` and `model` the model; never citable: the cite pattern admits
no `@`; X1's statistics in `diagnostics`), and the `x1` block: the configuration (§13), `registered`
and `deviations`, the snapshot and both label hashes, the framings' `question_key`s and lock sha, the
models' scales and fingerprints, the store's identity, the latency file, and per task the item
summary, the shuffle draws (K, seed, the index sha256), the candidate-only probe, every arm's record,
the full decision trace, A1, and the nested folds (inner traces, `fold_choices`, `skipped_folds`).
`framing.x1_cells` reads the cells back as a `BenchResults`.

12.2 **`x1_items.parquet`**, one row per (task, framing, model, item): task, task_id, framing, model,
item, target_sha256, option_key, card, label_index, weight, scale, `s`, `s_contexts` (the item's
score under every target context of its card, §5.1), `s_mean_context`, tokens (context, candidate,
anchor), `p_loco` (the full run's held-out LOCO-Platt probability) and `p_probe` (the candidate-only
probe's); written by DuckDB `COPY`, float64 exact (the same tables give the same bytes).

12.3 **`tiers.json`** (`BenchResults`): §9's cells, stamped with both label hashes and the environment;
a note names the `x1.json` the selections came from, with its sha256, and each nested cell's
`fold_choices` points into it (`x1_trace`, §9.5).

12.4 **`x2.json`**: §10's cells.

12.5 Each JSON has a `.md` beside it; every number names its JSON. `bench table` prints (or writes
with `--out`) one markdown table over the results files of a date, every row naming `file#cell`.

12.6 **`bench framing --decide --from x1.json`** (`framing.decide_from_json`), reading neither the
feature store nor a label store:
- (0) **the configuration**: its constants are this code's; the file's `registered` and
  `deviations` follow from its configuration, its label hashes, the framings lock sha and each
  model's fingerprint it records (`framing.x1_deviations`, §13.2); a run that is not the registered
  one is refused; the file holds exactly the configured tasks, every trace (the full run and each
  nested fold) exactly the configured arms, and, on the registered labels, each task the published
  `n`, class counts and items per card (§1.3);
- (a) **replay**, from the JSON: every stored rule R verdict follows from its own numbers (`needed` =
  ⌈0.8·m_c⌉, `m_c` and `wins` from its cards, the reason from m_c ≥ 4, the lower bound > 0 and the
  wins) and used the run's B, seed and α (recorded on every verdict and interval); rules (1)–(5)
  re-run on the statistics of every trace give the identical trace bit for bit; `a1` is the full
  run's choice; each task record holds exactly the parts its configuration ran; `fold_choices` (one
  per outer fold) and `skipped_folds` are the traces'; every arm record's AUROC and lower bound,
  shuffle verdict and LOCO-Platt NLL are its trace's, its folds and skips the task's fold plan, its
  calls and tokens per target the trace's; every arm cell's identity fields, rule (1) verdict and
  diagnostics are its record's and the run's, and its counts, metrics and novel-key block (the
  cell's own predictions on its novel items) follow from its own per-item predictions; no cell lacks
  an arm record;
- (b) **recompute**, from the items file alone (sha256 checked, every item identical across arms):
  the task summaries, the fold plans, the decisions, every arm record's statistics (AUROC and mean
  shuffled AUROC with their CIs, the shuffle's rule R, the LOCO-Platt fits and NLL, the mean-context
  record, §6.6's sensitivity), the `p_loco` column and each arm cell's per-item predictions, the
  candidate-only probe's folds and statistics from its `p_probe` column, the nesting and the shuffle
  draws are rebuilt and must be identical, every number within 1e-9.

Any mismatch is an error (exit 1). **Not replayed** (report-only, stated so in `x1.md`): the arm
cells' lookup controls (`majority_acc`, `lookup_*`, `novel_key_lookup`, `lopo`) and
`beats_lookup_novel`, which need the items' states; the probe's `p_probe` values and its solvers'
state, which need the candidate vectors; the label-free diagnostics of the store and the clock
(`collapse`, `mean_state_cos`, `input_tokens`, `ms_per_decision`) and the timing run's latencies
(its file is named by sha256).

12.7 **`bench run`** takes X1's outcome only through `framing.selections_from_json` (12.6 (0)–(b):
the replay **and** the recompute from the items file, refusing an unregistered file) and only from
an `x1.json` whose `labels_sha256`, `labels_content_sha256`, framings lock sha and both models'
fingerprints are the tier run's own.

12.8 No results file is overwritten unless `--force` is given (one run per file, §14.9).

## 13. Registration, determinism and constants

13.1 **The registered configuration** (`bench.registered.REGISTERED`). Every M2 run: the registered
snapshot (both hashes, §1.1); the framings of the lock of G1 (lock sha `b432d32a7536…`, §1.4); the
models with their registered fingerprints under the serving lock of G1 (lock sha `dd33f9fedbae…`;
`encoder_fp` c3b3d5e1a283, `clm_model_fp` 78be8c462b2e for `clm-latest` and 9f44b0301ee3 for
`clm-raw`, §1.5); B = 2000, seed 0. X1: both tasks; framings F1, F4, F7, F9; models `clm-latest`,
`clm-raw`, every one of the 16 arms scored; the full run and the nesting in one invocation; α = 0.05;
K = 200 shuffle draws with seed 0. The tier cells: `zero_shot` and `calibrated` for the five tasks,
X1's outcome from a registered `x1.json` with this run's identity (§12.7). X2: the five tasks' M0
controls, the PR #13 replica on both joint specs (the feature store of `encoder_fp` c3b3d5e1a283
under that serving lock; the replica reads no head, so `clm_model_fp` is not part of it) and the
registered AnyJev dump (sha256 `557c53b72847…`, §1.7). The verbs expose no option for B, the seed,
α, K, the models or the framings; the M2 verbs take B, the seed and α from the registration.

13.2 **Anything else is not the registered run**: a subset of tasks (`--tasks`), the full run or the
nesting alone (`--full`, `--nested`), a subset of tiers, another snapshot or other label content,
another framings lock, a model fingerprint other than the registered one (or none), another or no
AnyJev dump, X2 without the replica. Such a run is written with every cell `pre_registered: false`
and `exploratory: true` and its deviations listed in the file (`x1.json#/x1/deviations`, the results
notes); `bench framing --decide --from` and `bench run` refuse an unregistered `x1.json`. The verbs
refuse another snapshot, other label content or another AnyJev dump outright (§1.1, §1.7); the
producers apply the registration themselves, so a library call on other labels, with another B, seed
or α, under another framings lock or model fingerprint, with another dump, on a subset of the
registered tasks or tiers, or with any other X1 run parameter, is written unregistered whatever its
caller says (`framing.x1_deviations`: `config_deviations`, `registered.data_deviations` and
`registered.identity_deviations`, for X1; `registered.run_deviations`, `identity_deviations`,
`anyjev_deviations` and `task_set_deviations` for the tier cells and X2). X1 also refuses an item
table that does not score exactly the configured framings × models (`framing.configured_arms`), so
the configuration a file records is the grid it evaluated, and `framing.evaluate_task` never stamps
a partial grid or a partial run registered.

13.3 **Determinism.** Every bootstrap (CIs and rule R, full and inner) uses B = 2000 and seed 0
(`numpy.random.default_rng`) on the sorted cluster set; the shuffle draws use seed 0 (§5.2); Platt
starts from a fixed point and temperature bisects; no other randomness exists (the PR #13 replica's
lbfgs and the probe are deterministic). Items are in (target_sha256, option_key) order; float64
throughout.

13.4 **Strict inequalities are strict**: a card's Δ of exactly 0 is not a win; a lower bound of exactly
0 (or 0.5 for the AUROC) does not pass.

13.5 On another platform a recomputed number can differ in its last bits (BLAS and SIMD summation
order), hence §12.6's 1e-9; a discrete outcome that flips under such a difference makes
`decide_from_json` fail rather than silently disagree.

## 14. Run protocol

14.1 **Before the commit** nothing combines a model output with a real label: the code is exercised
on synthetic data only (fake stores, generated labels, the hermetic fixtures), and no M2 verb (`bench
framing`, `bench run`, `bench x2`) is run on the registered snapshot, not even as a dry run.

14.2 Commit and push this plan, the code and DESIGN's M2 notes (the pre-run disclosure) together.

14.3 The label-free timing run, once: `uv run python scripts/x1_latency.py --out
bench/results/<date>/x1_latency.json` (answers discarded; disclosed in DESIGN's M2 notes).

14.4 X1, once: `uv run mesa-clm bench framing --date <date> --latency
bench/results/<date>/x1_latency.json --decide` (both tasks, the full run and the nesting; it writes
`x1.json`, `x1_items.parquet`, `x1.md` and replays and recomputes the decision).

14.5 `uv run mesa-clm bench framing --decide --from bench/results/<date>/x1.json`.

14.6 The tier cells, once, and only after 14.5 has passed: `uv run mesa-clm bench run --tiers
zero_shot,calibrated --loco --date <date>` (X1's outcome from `bench/results/<date>/x1.json`, which
the verb replays and recomputes again before use, §12.7).

14.7 X2, once: `uv run mesa-clm bench x2 --date <date>`.

14.8 `uv run mesa-clm bench table --date <date> --out bench/results/<date>/table.md`.

14.9 Every output is committed as produced, in one commit, before any of it is written into prose. A
verb that fails is re-run only after its error is recorded; `--force` replaces only a failed or
unregistered attempt, and the replacement is disclosed. All verbs run on the serving host against the
store of 1.6 and the checkout's `serving/serving.lock.json` (or the installed copy the bootstrap
wrote, which must equal it).

14.10 The A1 amendment cites `x1.json`'s full-run choice per task and rotates the framings lock; a task
whose outcome is K1 is recorded as such (its tiers audit-only, `ols_rank` proposals, M4 probe-first).

## 15. What may still change after the first real run

**Nothing, except by amendment with the affected cells marked exploratory.** After the first run of
§14.4–§14.7, any change to this plan or to the code that would change a number, a choice, a flag or the
item set of a committed cell, a bug fix included, is made only by a DESIGN.md amendment that names the
change and why, and every cell it affects is marked `exploratory: true` (G1 freeze). A departure from
§14 is disclosed the same way. The A1 amendment that records X1's registered outcome is the planned
consequence of the run, not such a change.

## Appendix A: frozen phrase → section

| frozen phrase (PR) | section |
|---|---|
| Grid {F1, F4, F7, F9} × {clm-latest, clm-raw} × {term.fits, ontology_fits} | 1.2, 1.4, 1.5 |
| s_c = 100·(zs·zc − zs·za); scoring offline from the cache | 3.1–3.2, 1.6 |
| 200-pair cross-check ≤ 1e-4 | 1.6 |
| within-card state shuffle | 5 |
| candidate-only probe | 7.11 |
| pooled AUROC (cluster CI) | 4 |
| ΔAUROC(real − shuffle) | 5.3–5.4 |
| LOCO-Platt acc/ECE/NLL | 6 |
| collapse diagnostic; calls/tokens per target; p50 latency | 9.10, 11 |
| (1) qualifies if AUROC cluster-LB > 0.5, AUROC ≥ 0.60, ΔAUROC ≻ 0 | 7.2 |
| (2) winner = lowest LOCO-Platt NLL; more cacheable; F7/F9 tie | 7.4–7.7 |
| (3) model clm-latest unless clm-raw ≻ it on NLL | 7.8 |
| (4) F1 ≻ all eligible → amendment reopening D3 | 7.9 |
| (5) nothing qualifies → K1 | 7.3, 8.8 |
| (6) anchor variant (withdrawn) | 7.10 |
| Nesting; `fold_choices`; agree with A1 in ≥ 5/7 | 8 |
| guards 30 train / 5 held-out per class; floors 100 / 40 | 6.2–6.3 |
| Predictions pooled, never fold-averaged | 6.1 |
| Rule R | 5.4, 7.13 |
| Lookup as a probability model; `beats_lookup_novel` | 9.9 |
| Cell fields | 9.6–9.11 |
| X2 baselines | 10 |
| X3: zero_shot + calibrated on A1 | 9 |
| K1 | 7.3, 8.8, 14.10 |
| weights (D20) | 2.4, 6.4–6.6 |
| frozen snapshots (D30) | 1.1–1.3 |

## Appendix B: every constant

| constant | value | where |
|---|---|---|
| snapshot | `bench/snapshots/2026-09-29.parquet` | `bench.registered.REGISTERED` |
| labels_sha256 | `aafd18f8cea7992c9a8e0d12ded5a60ae2534a6677e911c28c2b05deaa1b752c` | same |
| labels_content_sha256 | `5c60a8a69cf71d90df776cc331ba1347a8f7f3f074e75dca6ee6f95efb987b2b` | same |
| AnyJev L2 dump sha256 | `557c53b7284749cc42f5a9ff1056f93721e891ff40b80713572e8457207c6483` | same |
| framings lock sha | `b432d32a7536c8f455098ae4a23139f6badbae7bc181a3a64853df3e80921eca` | `bench.registered.FRAMINGS_LOCK_SHA` |
| serving lock sha | `dd33f9fedbaee0129b09a21311227118461bed9326c942c2af7d0bd9b31b5237` | `bench.registered.SERVING_LOCK_SHA` |
| model fingerprints | encoder_fp `c3b3d5e1a283`; clm_model_fp `78be8c462b2e` (clm-latest), `9f44b0301ee3` (clm-raw); schema_sha256 `52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335` | `bench.registered.FINGERPRINTS` |
| bootstrap B, seed, α | 2000, 0 (`default_rng`), 0.05 one-sided | `stats.DEFAULT_*`, registration |
| rule R sign test | ≥ 10 items per card, ⌈0.8·m_c⌉ wins, m_c ≥ 4 | `stats.SIGN_*`, `MIN_CLUSTERS` |
| rule (1) | AUROC lower bound > 0.5; AUROC ≥ 0.60 | `framing.AUROC_LOWER_GATE`, `AUROC_FLOOR` |
| shuffle | K = 200 derangements per card, seed 0, rejection sampling | `framing.SHUFFLE_K`, `SHUFFLE_SEED` |
| guards | 30 training / 5 held-out per class, two-class tasks | `bench.tasks.base` |
| floors | 100 training items per calibrator; 40 per probe | `framing.CALIBRATION_FLOOR`, `PROBE_FLOOR`, `cells.CALIBRATED_FLOOR`, `PROBE_FLOOR` |
| Platt | hard targets; ridge 1e-6; ≤ 100 Newton iterations; gtol 1e-10; Armijo 1e-4; ≤ 60 halvings; quadratic cut 1e-12; start (0, 0) | `calibrate.FITTER_CONSTANTS` |
| temperature | T ∈ [1e-4, 1e4]; bisection in log(1/T) to 1e-12, ≤ 200 halvings | same |
| scales | `clm-latest` min(exp(logit_scale), 100); `clm-raw` 100; T = 1 | `offline`, `framing.TEMPERATURE` |
| PR #13 replica, probe | StandardScaler + LogisticRegression(C=1, max_iter=2000), unweighted | `x2.replica_pipeline` |
| ECE | 15 equal-mass bins, tie-invariant | `bench.metrics` |
| threshold_cp | ≥ 30 items, one-sided 95% Clopper–Pearson, risks 0.05 and 0.10 | `stats` |
| NLL clip | 1e-12 | vendored metrics |
| recompute tolerance | 1e-9 relative to max(1, \|x\|) | `framing.REPRO_TOLERANCE` |
| latency draw | 20 targets per (task, framing), sha256("task\|framing\|target") order; both models per target, clm-latest first at even positions | `framing.LATENCY_TARGETS` |
