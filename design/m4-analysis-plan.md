# M4 analysis plan (pre-run): X3 the probe tier, K2, X4, artifacts and promotion, the citation test, audits

*Written 2026-10-04 on `feat/m4-learned-tiers` (from `main` at `b46c36b`, the A6 merge), before
any M4 statistic was computed on real labels. This file and the code it describes are committed
and pushed **before the first M4 run on the snapshot's labels** (§12.1), so the commit is
tamper-evident evidence that every choice below was made before a result existed. Until then the
code ran only on synthetic data (generated labels with planted signal, fake feature stores, random
heads, the hermetic fixtures); nothing combined a model output with a silver label (DESIGN,
implementation note "M4 pre-run disclosure").*

The code: `src/mesa_clm/learn/linear.py` (the weighted fitters), `learn/probe.py` (the specs, the
nested selection, the out-of-fold calibrator, the probe artifact), `learn/teacher.py` (the D19
ingest), `learn/labels.py` (teacher rows, the silver-minus-Opus subset), `bench/x3.py` (the probe
cells), `bench/k2.py` (K2), `bench/x4.py` (the teacher ablation), `bench/e2e.py` (the end-to-end
measure, interface only), `bench/registered.py` (the M4 registration), `bench/run.py` and `cli.py`
(the verbs), `artifacts.py` (versions, `CURRENT.json`, `learn fit`, `learn promote`, publish, pull),
`providers/tiered.py` and `providers/live.py` (the probe tier served), `providers/base.py` (the
`feature_spec` invariant), `policy.py` (the citation test, the audit blocker), `audit.py` (the
production audit), `health.py` (the `artifacts` check), `scripts/record_ols_teacher.py` and
`tests/fixtures/ols-teacher/` (the corpus's OLS terms, recorded once). Symbols are in `code font`;
"PR" is DESIGN.md "Pre-registration (G1)"; "plan" is `design/plan-2026-09-28.md`; "M2 plan" is
`design/m2-analysis-plan.md`, whose readings of the folds, guards, floors, calibrators, metrics,
baselines and cell fields (its §6, §9) M4 keeps unchanged and does not restate.

This is a reading of the frozen text, not an amendment. It merges three drafts written in parallel
(the probe tier; the bench producers and the registration; the serving side) as revised after three
independent pre-run reviews (faithfulness, statistics, integrity). If this plan and the frozen text
disagree, the frozen text wins and the disagreement is a bug in this plan, fixed only by amendment
(§13).

## 0. How the frozen text is read

0.1 As in the M2 plan §0: each frozen rule is read as one deterministic algorithm; where the text
admits more than one reading, the reading chosen is named with every alternative found and why it
was rejected, and the more conservative reading wins (the one that admits less, cites less, or
promotes less). A reading that would let a result influence a choice is never taken.

0.2 Every constant is in Appendix B and `tests/unit/test_m4_plan_constants.py` holds Appendix B to
the code (`bench.registered.REGISTERED_M4`, `X3_GRID`, `K2_CONSTANTS`, `learn.linear.GRIDS`,
`FITTER_CONSTANTS`, `learn.probe.SPECS`, the artifact and audit constants).

0.3 The frozen text M4 runs: PR "X3 tier sweep" ("zero_shot + calibrated on A1; probe specs ×
fitters (nested); head (M7)"), PR "X4 teacher ablation", PR "Kill and pivot criteria" K2 (and K3,
K4 as they bind promotion and drift), PR "Citation test", PR "LOCO folds, leakage controls and
cells" (the probe/head floor of 40, the teacher-row rule, nested selection), plan §5.3 (the tier
fitters' table: the probe specs, the fitters, "spec/hyperparameters by grouped (card) inner-CV NLL,
then OOF temperature"), §5.5 (artifacts and promotion), §4.7 (policy and citation, the audit
requirement), §8 (the M4 row: deliverables and acceptance). Appendix A maps every frozen phrase to
its section.

0.4 Three readings settle the whole shape of X3 and are stated first:

- **The framing is not an axis of X3.** The probe of a task is fitted and served on the texts of the
  task's active framing after A1 (`framings.ACTIVE`: F9 for `column.ontology_fits`, F7 for `term.fits`
  and the closed choices). X3's frozen grid is "probe specs × fitters"; which framing a task asks is
  X1's decision (A1), and a probe serves the texts the active framing renders (`question_key` keys the
  artifact, plan §5.5). Rejected: a framing axis inside the inner selection (it would re-open X1
  inside X3, and a probe chosen on another framing's texts could not be served without a lock
  rotation).
- **A spec belongs to one model.** A spec that reads any head quantity (`lowdim.v1`, `pair512.v1`,
  `choice.state.v1`) is a `clm-latest` probe; one that reads raw vectors only (`pair4096.v1`,
  `joint4096@S1`, `joint4096@S1ns`, `choice.raw.v1`) is a `clm-raw` probe. K2's clause "if a
  `clm-raw` probe ≽ a `clm-latest` probe" is read on the two restricted nested cells (`@raw`, `@latest`,
  §4, B2.4), and a served probe lives in the artifact bundle of its spec's model (§6).
- **The floors are per fit.** The probe floor of 40 applies to every probe fit's training items (outer
  and inner, M2 plan §6.3), and the calibration floor of 100 to the out-of-fold pool the probe's
  calibrator is fitted on; a fold below either is skipped with the reason recorded. Stated in advance
  from the published counts: `column.annotate` (98 items) and `column.aspect` (60) never pool 100
  out-of-fold items, so their probe cells have no evaluated fold and `metrics: null`, with the
  uncalibrated probe reported in their diagnostics only (§2, P.6); under A6 those steps are decided
  by rules, so this changes nothing in production.

0.5 X3's `zero_shot + calibrated on A1` half is M2's (`tiers.json`, committed); M4 adds the probe
half and reads the committed calibrated cells only as K2's other candidate (§4) and as `learn
fit`'s calibrators (§6). The head half is M7's (K3).

## 1. Inputs and the M4 registration

1.1 **The registered snapshot** is M2's, unchanged: `bench/snapshots/2026-09-29.parquet`
(`labels_sha256` `aafd18f8…`, `labels_content_sha256` `5c60a8a6…`, the published counts), read only
through `bench.run.load_inputs` with M2's refusals (M2 plan §1.1, §1.3). The item sets of the five
tasks, their folds (`Task.leave_one_card_out`, sorted cards), guards (30/5 per class, two-class
tasks) and weights are exactly M2's (M2 plan §2, §6.1–§6.2).

1.2 **The feature store** is the one of the registered encoder (`encoder_fp` c3b3d5e1a283) under
the serving lock of G1 (`dd33f9fedbae…`), holding every text of the registered items under F4, F7,
F9, F1 and the X2 joint specs, and the closed choices' F7 texts (1,953 texts, 0 truncated, every
one projected on both sides under `clm-latest`, label-free `features stats`). X3 reads from it
only; X4 adds the teacher items' texts (§5, B3.5), built label-free before the run (§12.3).

1.3 **The committed M2 inputs** K2 reads, pinned by sha256: `bench/results/2026-10-03/tiers.json`
(`42501a293f45…`, the calibrated nested cells) and `x2.json` (`76e07c34ddae…`, the AnyJev L2
per-item cells). Other bytes are refused outright (B1.1).

1.4 **The framings lock** is A1's (`7c93cc3e0ff6…`), not G1's: M2's registration keeps G1's lock
so its replay still holds; an M4 producer under any other lock is unregistered (B1.1). The
fingerprints are G1's serving lock's, unchanged.

### B1. The M4 registration (`bench/registered.py`)

B1.1 **What is pinned.** `REGISTERED_M4` (`current_m4()`) adds to M2's `REGISTERED`, which it
never repeats (the snapshot `bench/snapshots/2026-09-29.parquet` with both hashes, the published
counts, B = 2000, seed 0, α = 0.05, the fingerprints of G1's serving lock come from `current()`):

- the **framings lock of A1**, `framings.lock_sha()` at the pre-run commit
  (`FRAMINGS_LOCK_SHA_M4 = 7c93cc3e0ff6c26918a631cb787ceb6623dfd797fa1bb90cc353e956c1856920`),
  and the **active framing per task** (`ACTIVE_FRAMINGS_M4`: F9 for column.ontology_fits, F7 for
  the others). M2's registration keeps G1's lock (`b432d32a7536…`); an M4 producer scoring under
  any other lock is unregistered (`m4_identity_deviations`).
- the **X3 grid** (`X3_GRID`): the specs per shape, each spec's model, the fitters with their
  hyperparameter values in declared order, the inner criterion, the floors (100 OOF items for the
  calibrator, 40 training items for a probe) and the standardization floor 1e-8.
  `learn.probe.DEFAULT_GRID` / `learn.linear.GRIDS` must equal it (`grid_deviations`); a run on
  another grid is unregistered.
- the **committed M2 inputs K2 reads**, by sha256: `bench/results/2026-10-03/tiers.json`
  (`42501a293f45…`) and `x2.json` (`76e07c34ddae…`). Other bytes are refused outright
  (`check_committed_input`, as M2's `check_anyjev`); a missing pin is a deviation.
- the **K2 constants** (`K2_CONSTANTS`): acc margin 0.02, the clm-raw clause's acc margin 0.01,
  ECE ≤ 0.08, ECE cluster upper bound ≤ 0.12, the card-sign fraction 0.8, 10 items per card, 4
  counting cards.
- **X4's teacher arms** `(None, (0.5, 0.3), (0.3, 0.1))` with the decision arm (0.5, 0.3).
- the **teacher inputs**, filled from the label-free ingest of §12.2 (run before this commit)
  and committed with this plan: the corpus content hash `teacher_corpus_sha256 =
  d90387928bed0a9c196a8efc4dce3cbf40b1e2e8cc879bfb50c5e3451b208785` (`teacher_corpus_sha256(dir)`
  = sha256 of `json.dumps(sorted((filename, sha256(bytes))))` over
  `curation/generic/*.validated.json`, default separators) as read at neon-ducklake commit
  `neon_ducklake_commit = b1fa52a8d3ffc56173a09794253ad86b0fd10111`; the teacher snapshot
  `bench/snapshots/2026-10-04-teacher.parquet` (sha256
  `397f98d4b28c1cbff951a0797a7c3dca73dfe5e21b7e59262ca79bbf7906153f`, content
  `deb0dc46e17bdf7879f11135cec8090f58f12f7ed50f114037852e93a79bb07b`); the silver-minus-Opus
  snapshot `bench/snapshots/2026-10-04-minus-opus.parquet` (sha256
  `2cd529ffa43986585bcb79988fbe683febc137d310ce34d9b94ace809a3676fe`, content
  `d8e26a0c7ac3b182e00741715d22e9e5a84c47ea898bf1d086a77c641acaa62c`). No pin is `UNPINNED`.
  `x4` compares the corpus hash only when a corpus directory is given: the corpus is pinned
  through the teacher snapshot's two hashes, and the live working tree (`~/neon-ducklake`) may
  move on after the commit it was read at (DESIGN M4 notes, item 4).

B1.2 **Every producer applies it itself.** `x3`, `k2`, `x4` compute their own deviations
(`run_deviations`, `m4_identity_deviations`, `grid_deviations`, `task_set_deviations`,
`committed_input_deviations`, `pinned_snapshot_deviations`, the active framings, X4's arms) and
write every cell `pre_registered: false`, `exploratory: true` with the deviations in the file's
first note when any exists, whatever the caller says. `learn fit` computes the same identity
and grid deviations (`m4_identity_deviations` over the framings lock and the fingerprints,
`grid_deviations` against `DEFAULT_GRID` per shape) and **refuses** on any (exit 1, nothing
written; C1.7): a bench cell written exploratory still informs, but an artifact is what
production serves, and an unregistered one has no cell to cite. Rejected: writing the version
with `registered: false` (promotion would refuse it, but a version number would be consumed
and the files would sit under the production layout). A teacher input left `UNPINNED` would
make `x4` unregistered (none is, B1.1); a **pinned** input whose bytes differ is **refused**
(another snapshot, other label content, other committed inputs: the M2 refusals). The verbs
expose no option for B, the seed, α, the grid, the framings or the models.

B1.3 Rejected: pinning the grid by importing A's constants into the registration (the
registration would then follow a code change it is meant to detect); a single `registered`
flag the caller sets (M2 rejected it too, §13.2).

## 2. The probe tier (`learn/linear.py`, `learn/probe.py`)

### P.1 Specs

Every vector comes from the feature store of the registered encoder (`encoder_fp` c3b3d5e1a283):
`zs`, `zc` are the 512-d `clm-latest` projections (`FeatureStore.project` under its
`clm_model_fp`), `xs`, `xc` the raw L2-normalised 4096-d vectors (`store.get`), `s_latest` and
`s_raw` the zero-shot `s_c = scale · (zs·zc − zs·za)` under each model's `OfflineScorer`
(`pairwise_s_c`, the tier cells' scores: `FeatureBuilder.zero_shot_logits` equals `score_arm`'s
`x`, tested). `⊕` is concatenation, `⊙` the elementwise product. The spec id names the formula
(`probe.rank_fit_features`, `probe.choice_features`: one implementation for the bench and for
serving).

| spec | shape | model | dim (K options) | rows | formula / texts |
|---|---|---|---|---|---|
| `lowdim.v1` | rank_fit | clm-latest | 2 | one per labelled (target, candidate) | `[s_latest, s_raw]` |
| `pair512.v1` | rank_fit | clm-latest | 1025 | same | `zs⊙zc ⊕ |zs−zc| ⊕ [s_latest]` |
| `pair4096.v1` | rank_fit | clm-raw | 4097 | same | `xs⊙xc ⊕ [s_raw]` |
| `joint4096@S1` | rank_fit | clm-raw | 4096 | same | the raw vector of the X2 joint context text (`features.X2_SPECS`, manifest role `context`, option = the candidate) |
| `joint4096@S1ns` | rank_fit | clm-raw | 4096 | same | same, no task suffix |
| `choice.state.v1` | choice | clm-latest | 512 + K | one per labelled target | `zs ⊕` the K zero-shot option logits under clm-latest |
| `choice.raw.v1` | choice | clm-raw | 4096 + K | same | `xs ⊕` the K zero-shot option logits under clm-raw |

The texts are the active framing's (`framings.ACTIVE`: F9 for `column.ontology_fits`, F7
otherwise), matched to each item's D1 identity through the manifest (`cells.TextIndex`); the
anchor text is read only to form `s_c`. A spec's **model** is `clm-latest` when it reads any
head quantity, else `clm-raw` (`SpecInfo.model`); the table above is reproduced
verbatim in `probe.SPECS`, in that **declared order**. `dim` is the registered dimension
(`SpecInfo.dim(K)`); the builder uses whatever dimension the store holds (tests use 64/16).

Reading: `pair512.v1`'s last column is `s_latest` (plan §5.3 writes `s_c`; under the
`clm-latest` head the only `s_c` is `s_latest`), `pair4096.v1`'s is `s_raw` (R1). A spec never
mixes the two models' vectors; only `lowdim.v1` reads both scores and is `clm-latest` because
it reads a head quantity.

### P.2 Fitters (`learn/linear.py`)

Inputs of every fit: the spec matrix rows of the training items `X [n, d]`, their option-index
labels (Yes = 0) and the label weights as **sample weights** (D20). Weights act on the data term
as a weighted mean: the objective is `Σᵢ wᵢ ℓᵢ / Σᵢ wᵢ + penalty`, so an integer weight equals
that many copies of a row and the scale of the weights is irrelevant (tested).

**Standardization** (P.2.1): per fit, every feature is centred by the training set's weighted
mean and divided by its weighted (biased, `/Σw`) standard deviation, floored at `1e-8`; a test
row is transformed with the training moments. A constant column maps to 0 on the training rows
and receives a zero coefficient (it spans nothing).

**Reduction** (P.2.2, an implementation that changes no optimum): the standardized training
matrix `X_std = U S Vᵀ` (thin SVD) is replaced by `Z = U S [n, r]` with `r` its numerical rank
(singular values above `s_max · max(n, d) · ε`, numpy's `matrix_rank` rule). The L2-penalised
logistic regression and the ridge have their optimum in the row span (a component of `β`
orthogonal to every training row leaves the data term unchanged and adds penalty), and `‖Vγ‖ =
‖γ‖`, so the reduced fit is the full-space optimum exactly; for LDA the class means and the
pooled within-class covariance lie in that span and the shrinkage identity acts on the complement
as a scalar every class shares, which the softmax cancels, so the posteriors are those of the
full-space formula exactly (both tested against direct full-space solves and scikit-learn to
1e-6–1e-15). Coefficients are stored in the full standardized space (`coef [K, d]`), so an
artifact needs no basis. Consequence: a 4097-d fit on ≤ 300 rows is a ≤ 300-dimensional problem
and the K = 8 multinomial on 60 items has a 8·61-parameter Hessian.

**logreg** (P.2.3): `J(β, b) = Σᵢ wᵢ (lse(ηᵢ) − η_{i,yᵢ}) / Σ wᵢ + (λ/2) ‖β‖²_F`, `ηᵢ = β zᵢ +
b`, the intercepts unpenalised. K = 2 is the binary form (one weight vector, Yes against No;
stored as row 0 with row 1 zero, so `probs = [σ(η₀), 1 − σ(η₀)]`). K > 2 is the symmetric
multinomial form (every class a row, all K rows penalised, which makes the coefficients
identifiable and the fit invariant to the option order) with the intercept gauge `b_{K−1} = 0`
(invisible in the probabilities). Newton's method from zero with Armijo step halving (constant
1e-4, at most 60 halvings; inside the quadratic region, a predicted decrease below 1e-12 of the
loss, the full step is taken), at most 100 iterations, stopping when every gradient component is
below 1e-8; a singular Newton system falls back to a gradient step. **A fit that does not
converge** (the cap, or no decrease along the Newton direction) **is used as it is** and recorded
`converged: false` in the fold's record (`diagnostics.per_fold.<card>.linear`, `fold_choices
.<card>.converged`), never refitted or skipped — the M2 calibrators' rule (§6.4).

**lda** (P.2.4): weighted class means `μ_k = Σ_{i∈k} wᵢ zᵢ / W_k`; weighted class priors `π_k =
W_k / W`; pooled weighted within-class covariance `Σ = Σᵢ wᵢ (zᵢ − μ_{yᵢ})(zᵢ − μ_{yᵢ})ᵀ / W`
(normalised by the total weight, the maximum-likelihood form); shrinkage toward the scaled
identity `Σ_γ = (1 − γ) Σ + γ ν I` with `ν = tr(Σ) / d`, the trace of the pooled covariance in
the full standardized space divided by its dimension `d` (the Ledoit–Wolf target, what
scikit-learn's `shrinkage=γ` computes); posteriors `softmax_k(zᵀ Σ_γ⁻¹ μ_k − ½ μ_kᵀ Σ_γ⁻¹ μ_k +
ln π_k)` (closed form, `iterations 0`). A class with no training weight has no mean and no prior:
the fit is **refused** (`LinearError`) and the nested selection records the configuration as
unfittable on that fold (P.3.4).

**ridge** (P.2.5): `J(β, b) = Σᵢ wᵢ ‖β zᵢ + b − e_{yᵢ}‖² / (2 Σ wᵢ) + (λ/2) ‖β‖²_F` on the
one-hot targets, solved through the √w row scaling of the normal equations `(ZᵀWZ/W + λI) γ =
ZᵀW(Y − b)/W`; the unpenalised intercept is the weighted class frequency (the standardized
features have weighted mean zero, so it decouples). The scores are the fitted regressions;
"probabilities" are their softmax (K = 2: `σ` of the score difference), a ranking that only the
OOF calibrator makes a probability (R1).

**Hyperparameter grids** (P.2.6, `linear.GRIDS`, declared order = tie order): logreg λ ∈
{1e-2, 1e-1, 1, 10}; lda γ ∈ {0.1, 0.5, 0.9}; ridge λ ∈ {1e-1, 1, 10}. Fitters in the order
`logreg, lda, ridge`.

Reading (P.2.7, the penalty's normalisation): λ multiplies the penalty against a **weighted
mean** data term (the convention of `calibrate.fit_platt`, `mean NLL + λ/2 (a² + b²)`), not a
weighted sum. Alternative rejected: the sum form (scikit-learn's `C Σ wᵢ ℓᵢ + ½‖β‖²`, the PR #13
replica's `C = 1`), under which λ = 1e-2 would be nearly unpenalised on a 4097-d matrix of ~300
rows and the fit would depend on the number and scale of the weights. Equivalence for the
record: the mean form at λ equals the sum form at `C = 1/(λ · Σw)`; on the snapshot's rank_fit
training sets (Σw ≈ 130–170 at the 0.5–0.8 weights) the grid spans `C ≈ 0.6 … 6e-4`, i.e. from
about the replica's regularisation to much stronger; the OOF calibrator restores the scale of a
shrunk probe. The mean form is the more conservative reading (never a near-unpenalised 4097-d
fit) and the one that keeps every objective of the code base in one convention. Stated in
advance: this is a choice of grid scale, made before any result, identical for every spec and
task, so it cannot favour one.

### P.3 The inner selection (`probe.inner_select`)

P.3.1 **Folds.** For an outer fold *c* (held-out card *c*, training cards *T*, the other six;
`Task.leave_one_card_out`, sorted card order), the inner folds are the grouped leave-one-card-out
over the cards of *T* in sorted order: inner held-out card *t*, inner training the other cards of
*T*. Nothing in the inner loop reads an item outside *T*.

P.3.2 **Guards and floor per inner fold, on the task's items only.** The 30/5 guard
(`bench.tasks.fold_guard`: two-class tasks, 30 per class in the inner training items, 5 per
class in the inner held-out card) then the probe floor: fewer than 40 inner training items skip
the inner fold `below_floor n < 40 training items (probe)`. Teacher rows (P.5) do not count
toward a guard or a floor (reading: the guards are the PR's "30 train / 5 held-out per class"
over fold-eligible items; teacher rows are not fold-eligible and are all Yes, so counting them
would let a training set with fewer than 30 silver Yes pass the guard; rejected).

P.3.3 **Every configuration on every passing inner fold.** The grid's configurations are the
spec × fitter × hyper product in declared order (`Grid.configurations`). Each is fitted on each
passing inner fold (the inner training items' spec rows, their weights, plus the teacher rows
not of the inner held-out card's product) and applied to the inner held-out card; the OOF
predictions are pooled over the passing inner folds (`oof_idx`, the union of their held-out
items, in item order). The inner criterion is the **pooled OOF NLL of the uncalibrated probe
probabilities** (the row-wise softmax of the scores; `stats.metric_value("nll")`, the vendored
NLL with the 1e-12 clip), **unweighted over the items**. The winner is the configuration with
the lowest inner NLL; a tie (equal to the bit) goes to the first in declared order (strict `<`
while scanning in order; tested with identical matrices under two names).

Reading (P.3.3a, weighted or unweighted criterion): unweighted. The weights are *sample weights
of the fits* (D20: "weighted everything" names the fits and the calibrators, and R1 says so for
both), while the criterion is the quantity the cell is judged on: §6.7's metrics are unweighted
over the pooled items, rule R compares unweighted NLL, and X1's inner criterion (the LOCO-Platt
NLL, `framing.loco_nll`) was unweighted. Rejected: a weighted inner NLL, which would select on a
quantity no cell reports and the frozen text does not name.

P.3.4 **Unfittable configurations.** A configuration whose fit is refused on any passing inner
fold (`LinearError`: in practice LDA with a class absent from the inner training set, which
happens on the K = 8 `column.aspect` with 60 items) is recorded in `inner_grid` with `nll:
null` and the `error` (the inner card and the message) and is not selectable on that outer
fold. If no configuration is selectable the outer fold is skipped
`inner_no_fittable_configuration`; if no inner fold passes, `inner_no_passing_fold` (with every
inner skip reason recorded). Reading: an unfittable configuration could instead be scored on the
inner folds it can be fitted on (a smaller pool); rejected because the pooled NLLs would then
not be comparable across configurations.

P.3.5 **Record.** `fold_choices[c]` carries `spec, fitter, hyper, model` (the spec's),
`clm_model_fp` (the model's, from the caller's fingerprints), `inner_nll, inner_n` (pooled OOF
items), `inner_folds` (passing inner folds), `inner_skips` ({inner card: reason}), `n_train`,
`n_teacher` (teacher rows in the outer training set), `converged` (the refit), the `calibrator`
record (P.4) and `evaluated`; `decision` is `configuration` or the skip reason; a skipped fold
lists `null` for the configuration fields it has none of. Every configuration's inner NLL is in
`diagnostics.per_fold[c].inner_grid` (report-only).

### P.4 The OOF calibrator and the outer fold (`probe.nested_probe`)

P.4.1 **Outer guard and floor.** The 30/5 guard on the outer fold (`fold_guard`) and the probe
floor on *T* (`below_floor n < 40 training items (probe)`) skip the fold before any inner work.

P.4.2 **Calibrator floor.** Fewer than 100 pooled OOF items of the winner skip the outer fold
`below_floor n < 100 OOF items (probe calibrator)`: no probe calibrator is ever fitted on fewer
(§6.3 applied to the OOF pool, R1). The winner is still refitted on *T* and applied to card *c*
so that the **uncalibrated** held-out metrics are reported (P.6); the fold's `fold_choices` entry
keeps the configuration with `evaluated: false` and the `skip_reason`.

P.4.3 **The calibrator**, fitted **weighted** (the OOF items' label weights as sample weights)
on the winner's pooled OOF predictions by `calibrate.fit_calibrator("choice", …)`: K = 2
(rank_fit and `column.annotate`) a Platt `σ(a·x + b)` on the OOF logit `x = ln p/(1−p) = score₀ −
score₁` (`PlattCalibrator.feature = "logit_difference"`, hard targets, the §6.4 Newton rule,
`a < 0` flagged `inverted` and kept); K > 2 a temperature on the OOF log-probabilities
(`log_softmax(scores)`; the §6.5 bisection; a bound flagged `at_bound` and kept). The record:
`{kind, params ({a, b} or {temperature}), n, converged | at_bound, inverted, nll}`; the cell's
`calibration` is `platt` or `temperature`.

P.4.4 **Refit and prediction.** The winner is refitted on all of *T* (its items' spec rows and
weights plus the teacher rows not of card *c*'s product), applied to card *c*'s rows, the
calibrator applied to the result (`calibrator.probs(calibrator_input(scores))`), and the held-out
predictions pooled over the evaluated folds in item order, never averaged. `NestedProbeResult`
holds `probs [n_pooled, K]`, `pooled`, `folds`, `skipped_folds`, `fold_choices`, `diagnostics`
and the per-fold fits; `cells.assemble_cell` builds the cell from the first six (tested: the
`probe` cell's metrics, `beats_lookup_novel`, items). `ProbeArtifact.predict` on the fold's fit
equals the pooled held-out probabilities bit for bit (tested).

P.4.5 **Variants.** `@latest` / `@raw`: the same procedure on `Grid.restrict(model)` (the specs
of one model). `@full` (`selection: "full"`, exploratory): `full_probe`'s inner LOCO over all
seven cards picks one configuration (`inner_select` on every item: the same guards, floor,
criterion and ties), and `nested_probe(fixed=that configuration)` evaluates it in every outer
fold with the fold's own OOF calibrator and refit (only the choice is fixed; `inner_grid` has
one entry).

P.4.6 **Cell identity** (for the cell builder, B): the nested cell's `feature_spec`, `model` and
`fingerprint.clm_model_fp` are the `@full` configuration's (the configuration production would
serve, as A1's arm is the nested calibrated cell's identity, §9.5); `fold_choices[c].spec /
.model` give the per-fold agreement with it (the citation test's ≥ 5/7, R6). Every spec is
`servable` (both models are served). Stated in advance: a nested cell whose folds chose specs of
both models is still one cell; its items are scored by each fold's own spec. Rejected: the
identity of the majority of the evaluated folds' choices (the configuration most folds chose)
— a majority is not what serving fits (a 3/2/2 split has none, and a majority configuration the
full-data selection does not choose would be cited for a probe production never serves); the
citation test's agreement count is against the served configuration for the same reason.

### P.5 Teacher rows (`probe.TeacherRows`, D19, plan §5.4)

A `TeacherRows` set (a feature source over the teacher items — a `FeatureBuilder` on the teacher
task under the same framing — with labels, weights and the `leak_group` of every row) is added
to **every training set, outer and inner**, except the rows whose `leak_group` equals the
held-out card's product: for outer fold *c*, the rows of card *c*'s product are dropped from
that fold's outer training set and from every inner training set of that fold; inside, the
inner held-out card's product is dropped too. Teacher rows are never in a test index (they are
not items of the task; `assemble_cell` refuses a teacher source among the pooled items, and
`diagnostics.teacher_in_test: false` is asserted), never counted by a guard or floor, and every
fold records `n_teacher`. Tested by counting the rows of every fit (outer and inner) against the
expected drops, and that the rows change the fit. The full-data artifact trains on every
teacher row (no held-out card); its 7-card inner LOCO drops each inner product.

### P.6 Report-only diagnostics (`NestedProbeResult.diagnostics`)

`per_fold[c]`: `n, n_train, n_teacher`, the selection record, `inner_grid` (every
configuration's `nll, n, error`), `linear` (`converged, iterations, d, r, n` of the refit),
`uncalibrated_heldout` (NLL, ECE, accuracy, Brier and the calibration in the large of the
uncalibrated probe on the held-out card: `cells.calibration_summary`), the fitted `calibrator`
(its full JSON). `uncalibrated_pooled`: the same summary over the uncalibrated predictions of
every fold that fitted a model — the evaluated folds and the folds skipped by the calibrator
floor alone — so `column.annotate` and `column.aspect` get numbers although no fold of theirs
reaches the calibrator floor (R1). Also `grid`, `criterion`, `floors`, `fitters` (every
constant of P.2), `calibrators` (`calibrate.FITTER_CONSTANTS`), `teacher`, `teacher_rows`,
`task_counts`, `excluded`, `chosen_specs`. The builder adds per spec (`FeatureBuilder
.spec_diagnostics`): the model, `dim`, `mean_state_cos` (on the spec's model's side; a joint
spec's own context vectors), `calls`, `input_tokens`, `ms_per_decision` (the §9.10 units) and
`build_ms`. No rule reads any of it.

Stated in advance from the published counts (§6.3; the per-card `n` of
`bench/results/2026-09-29/baselines.json#per_fold_n`): `column.annotate` (98 items) can never
pool 100 OOF items, so every outer fold is calibrator-floor skipped and its probe cell has
`metrics: null` with `uncalibrated_pooled` reported; `column.aspect` (60 items; per-card n 13,
11, 9, 9, 7, 6, 5, so outer training sets of 47–55 and inner training sets of 36–49) likewise
(≤ 55 OOF items < 100); its inner training sets fall to 36 or 38 < 40 on the six inner folds
where the 13-item card and an 11- or 9-item card are both held out (outer and inner), which the
probe floor skips, and LDA is unfittable on most of the rest (K = 8 over 60 items).
`avu.value_kind` (278; per-card 34–47), `term.fits` (285; 33–48) and `column.ontology_fits`
(190; 17–44) clear the 40 floor outright on every outer and inner training set (the smallest
inner training set is 185, 192 and 114 items); they clear the 100 OOF floor on every outer fold
(the smallest outer training set is 231, 237 and 146 items) **only if the inner 30/5 guards
pass**, since the OOF pool is the union of the passing inner folds' held-out cards and the
per-card class counts are not published, so that half is not stated in advance; a
guard-skipped inner fold shrinks the pool and is recorded in `inner_skips`.

### P.7 The artifact (`probe.ProbeArtifact`, `probes/<question_key>.json`)

`{format: "mesa-clm/probe/1", question_key, task_id, framing_id, spec, model, fingerprint (the
spec's model's), linear (LinearModel: kind, k, hyper, coef [K, d], intercept [K], standardizer
{mean, std}, n, weight_sum, converged, iterations), calibrator (learn.calibrate's JSON:
PlattCalibrator or TemperatureCalibrator, discriminated on kind), selection {selection: "full",
cards, spec, fitter, hyper, model, inner_nll, inner_n, inner_folds, inner_skips, n_teacher,
n_train, calibrator record, linear record}, labels_sha256, labels_content_sha256, n_train}`;
`extra = "forbid"`, frozen, finite floats only. `predict(X)` = `calibrator.probs
(calibrator_input(linear.scores(X)))`, the `[n, K]` calibrated distribution (K = 2: `[p_yes,
p_no]`); `scores(X)` the uncalibrated logits. `full_probe` refuses (`ProbeError`) an item set
below the probe floor or the 30-per-class guard, one without a selectable configuration, and
one whose 7-card OOF pool is below 100 (so `learn fit` writes no probe for `column.annotate` or
`column.aspect`, which R1's floors already exclude from a citable cell).

Note for C (serving): `calibrate.PlattCalibrator`'s JSON carries `feature, targets, weight_sum,
ridge, iterations, converged, nll` beside `kind, a, b, n`, and `TemperatureCalibrator` names
`temperature` (not `T`); `providers/tiered.py`'s runtime copies forbid extra keys, so the
conversion is a projection `{kind, a, b, n}` / `{kind, T: temperature, n}`. The spec features at
serving time are `probe.rank_fit_features` / `probe.choice_features` over the live vectors.

## 3. The X3 cells (`bench/x3.py`, `bench x3 --date <date>` → `x3.json`, `x3.md`)

3.1 **Four cells per task**, every one through `cells.assemble_cell` (M2 plan §9.6–§9.11: the same
identity fields, counts, metrics, lookup controls, `beats_lookup_novel`, `threshold_cp`, per-item
predictions): the **nested cell** `<task>.probe.<active framing>` (P.3–P.4 on the whole grid,
`selection: "nested"`, `pre_registered: true`, `exploratory: false` in the registered run; the
citable form), `@latest` and `@raw` (the same procedure on the specs of one model, P.4.5; nested,
pre-registered, not exploratory, never citable because a variant), and `@full` (one configuration
chosen by the inner LOCO over all seven cards, evaluated in every outer fold with that fold's own
calibrator and refit; `selection: "full"`, `exploratory: true`, D27). The tier is `probe`; the
`calibration` field is the calibrator's kind; `servable` is true for every spec (both models are
served; a cell whose full-data selection chooses nothing is `servable: false` with the reason in
`diagnostics.full_selection` and has no `@full` cell).

3.2 **Identity** (P.4.6): the nested cell's `feature_spec`, `model` and `fingerprint.clm_model_fp`
are the `@full` configuration's — the configuration production would serve, as A1's arm is the
nested calibrated cell's identity (M2 plan §9.5) — and `fold_choices[c]` records each outer fold's
own configuration, from which the citation test counts the agreement (§8, ≥ 5 of 7). A nested cell
whose folds chose specs of both models is one cell; its items are scored by each fold's own spec,
and `diagnostics.fold_agreement_with_full` reports how many folds chose the served configuration.
Rejected: identity from the majority of the evaluated folds' choices (P.4.6: a majority of folds
is not what serving fits, a 3/2/2 split has none, and a cell carrying a configuration production
does not serve could be cited for the one it does).

3.3 **What can be cited.** Only the nested cell, and only when the citation test passes (§8); for
the closed choices `beats_lookup_novel` is false by construction, each for its own reason:
`column.aspect` and `avu.value_kind` have no AUROC (K > 2, M2 plan §9.9); `column.annotate`
(K = 2) has an AUROC, but its lookup control is `insufficient_clusters` under the frozen MDE
statement (one card with ≥ 10 novel-key items; DESIGN "Minimum detectable effect", stated there
in advance) and its probe cell has no evaluated fold under the 100 OOF floor (0.4). So no
closed-choice probe cell can be cited in M4, stated in advance. `term.fits` can be cited for a numeric `auto` only through a promoted probe,
since its calibrated tier is K1's audit-only (A1).

3.4 **Registration.** The producer applies the M4 registration itself (B1.2): a run on other
labels, another lock, other fingerprints, another grid, another active framing, a subset of tasks,
or B/seed/α other than the registration's is written with every cell `pre_registered: false`,
`exploratory: true` and the deviations in the first note. The verb exposes no option for any of
them; `--force` replaces only a failed or unregistered attempt (§12).

3.5 **Diagnostics** (P.6; report-only): every configuration's inner NLL per fold, the uncalibrated
held-out and pooled metrics, the refit's convergence, the per-spec `mean_state_cos`, `calls`,
`input_tokens`, `ms_per_decision` in M2's units (M2 plan §9.10), the teacher counts (zero in X3).

## 4. K2 (`bench/k2.py`)

### B2. K2

B2.1 **Candidates.** Per task, the citable-form nested cells: `calibrated` =
`tiers.json#/cells/<task>.calibrated.<active framing>` with `selection: "nested"` (only
column.ontology_fits has one; term.fits' K1 audit cells are `selection: "none"` and the closed
choices' F7 cells are fixed arms, so they are listed as candidates with `eligible: false` and the
reason, never chosen), and `probe` = `x3.json#/cells/<task>.probe.<active framing>`. A
candidate is eligible when it is `pre_registered: true`, nested, servable, not exploratory and
has pooled items; the check is per candidate cell, whatever the file's note says (an
exploratory or unregistered cell is ineligible with that reason), so an unregistered `x3.json`
leaves K2 with the committed calibrated candidate alone.

B2.2 **The best servable tier** = the eligible candidate with the lowest pooled NLL (`metrics.nll`;
a tie goes to `calibrated`, the lower-rank tier — alphabetical, and the conservative choice).
Rule R "probe ≻ calibrated on NLL" is computed on the items both pooled, paired by identity, and
recorded as `probe_vs_calibrated_nll` / `promotion_probe_over_calibrated`: promotion (R4) needs
≻, K2 does not ("K2 judges the best tier"). Rejected: requiring ≻ before a probe may be best (the
frozen K2 text names "the best servable tier", not a promotion rule).

B2.3 **K2(a)** on the best cell, every condition with its numbers (`conditions`):

1. `beats_lookup_novel` = the cell's own `baselines.beats_lookup_novel` (its `beats_detail`
   copied); false with the reason when the cell has none.
2. **Non-inferiority to AnyJev L2 on the full item set.** The AnyJev side is
   `x2.json#/cells/<task>.baseline.anyjev_l2.items`, joined by `(target_sha256, option_key)`;
   labels and cards of a paired item must agree (both come from the snapshot; a disagreement is
   a `CellError`). "On the full item set" is read strictly: the tier must pool every item of the
   task (`diagnostics.task_counts.n`, else the registration's published count for the task; a
   cell with neither fails the condition `task_count_unknown` — the count is never inferred from
   the cell's own pooled items, which would make every tier "full") **and** every pooled item
   must match an AnyJev item; otherwise the condition fails with `tier_not_on_full_item_set` /
   `anyjev_join_incomplete` (the committed join is full, so only a tier that skipped a fold can
   fail this way). Then Δacc = acc(tier) − acc(AnyJev) with the
   card-cluster paired bootstrap (B = 2000, seed 0, α = 0.05 one-sided) lower bound > −0.02
   (`stats.non_inferior("acc", …, margin=0.02)`). Rejected: comparing on the intersection when
   the tier skipped a fold (the frozen text says "the full set"; a tier that cannot be scored on
   it is not shown non-inferior on it).
3. **The card-sign condition.** Rule R's sign test (ii), verbatim: on ≥ ⌈0.8·m_c⌉ of the m_c
   held-out cards with ≥ 10 items, the per-card Δacc = acc_c(tier) − acc_c(AnyJev) > 0, strictly
   (`stats.card_sign_test("acc", tier, AnyJev, …)` as it is, no margin; a card at exactly 0 does
   not win, §11.4); m_c < 4 is `insufficient_clusters` and fails. The per-card deltas are
   recorded. Reading: "the card-sign condition" of the frozen K2 text is the one card-sign
   condition the frozen text defines, rule R's (ii), which has no margin; the −0.02 of the same
   sentence belongs to its other half, the cluster lower bound ("Δacc cluster-LB > −0.02 +
   card-sign condition"). Rejected: the margin-shifted per-card test (a card wins when Δacc >
   −0.02), which reads a margin into a sign test the frozen text left plain and is the
   permissive reading (it admits more). It is **reported, not gated**:
   `numbers.card_sign_margin_report` counts the cards within the margin beside the gated count,
   computed in integers — a card is within the margin iff `50·(correct_tier − correct_anyjev) >
   −n_card`, which is Δacc > −0.02 with no float rounding. Stated in advance: the literal test
   asks the tier to *beat* AnyJev L2 on most counting cards, so K2(a) is demanding against
   AnyJev L2, and a tier non-inferior on the cluster bound but not better per card is (b) at
   best; a numeric `auto` stays null in 0.1.0 whatever the verdict (§12.11), so the gate decides
   a record, not a write.
4. **ECE** ≤ 0.08 on the cell's pooled items and the cluster upper bound
   `stats.metric_ci("ece", …).upper` ≤ 0.12 (the one-sided 95% bound over the card resamples; a
   cell with one card has no bound and fails).

Verdict `a` iff 1 ∧ 2 ∧ 3 ∧ 4; `b` iff 1 alone; `c` otherwise, including "no eligible candidate"
(a probe cell with no pooled items, as column.annotate's under the 100 OOF floor may be). A
closed choice is `c` by construction, stated in advance, each for its own reason: `column.aspect`
and `avu.value_kind` have no AUROC (K > 2), so 1 is false; `column.annotate` (K = 2) has an
AUROC, but its lookup control is `insufficient_clusters` under the frozen MDE statement (one
card with ≥ 10 novel-key items) and its probe cell pools nothing under the 100 OOF floor (0.4),
so 1 is false for it too. Under A6 those steps are rules anyway. Strict inequalities are strict.

B2.4 **The clm-raw clause** (`head_adds_nothing`, rank_fit tasks): on `x3.json`'s `@latest` and
`@raw` cells, paired by identity, `@latest` is **not** ≻ `@raw` on NLL (rule R, A = latest,
B = raw, `passed` false) **and** `@raw` ≽ `@latest` on acc within 0.01 (`stats.non_inferior`,
A = raw, B = latest, margin 0.01). A missing cell or one without pooled items leaves it
`evaluated: false`. Reading: the frozen "a clm-raw probe ≽ a clm-latest probe" names no metric
and no δ; "≽ within δ" is defined by rule R on one metric, and the only registered δ on acc is
the promotion margin 0.01, so the clause is read as "the head does not win on the metric the
tier is judged on (NLL, rule R) and the raw probe is non-inferior on acc within 0.01". Rejected:
`@raw` ≽ `@latest` on NLL within some δ (no δ on NLL is registered; one chosen now would be
chosen for this clause alone); `@raw` ≻ `@latest` (superiority, which "≽" does not say);
comparing the nested cell's `clm-raw`-spec folds with its `clm-latest`-spec folds (a fold's
choice is not a model comparison and the items would not pair); comparing the best single spec
of each model (a spec axis the frozen text does not name, chosen on a result). Stated in
advance: with 7 cards rule R's sign test needs a win on ⌈0.8·7⌉ = 6 of the counting cards, and
its first half ("`@latest` ≻ `@raw`") seldom passes, so the clause mostly reduces to its acc
half (`@raw` within 0.01 of `@latest` on acc), and a `head_adds_nothing: true` is weak evidence
that the head adds nothing, as much as a `false` would be weak evidence that it does. No
production configuration changes in the run; the consequences, RESEARCH.md's record included,
are the integrator's amendment (A7).

B2.5 **The file.** `k2.json` (`format mesa-clm/k2/1`): the inputs with their sha256, the
constants, `registered` and `deviations`, and per task the candidates (identity, NLL/acc/ECE,
`beats_lookup_novel`, eligibility), the rule-R comparison, the best tier and its cite, every
condition, the verdict with its reason, the clm-raw clause. `k2.md` is one row per task.
`bench table` lists the verdicts beside the cells of the other files. Deviations the producer
finds on its own: the three files not about the same labels, an `x3.json` that is not
pre-registered, B/seed/α other than the registration's; and, per candidate, a cell that is not
`pre_registered: true` is ineligible with that reason (B2.1), and a best cell without
`diagnostics.task_counts` and without a registered count fails condition 2 with
`task_count_unknown` (B2.3; fail closed).

## 5. X4 and the teacher corpus (`learn/teacher.py`, `bench/x4.py`)

### B3. X4 and the teacher corpus

B3.1 **The corpus** (`~/neon-ducklake/curation/generic/<DP>.validated.json`: 55 files, 389
`accepted` items, every file `model` claude-opus-5-5, read by the ingest of §12.2 at
neon-ducklake commit `b1fa52a8d3ff…` with corpus hash `d90387928bed…`, both pinned in B1.1; the
report is `.local/m4/ingest_teacher.log`, transcribed in DESIGN's M4 notes, item 6). D19: items
with `status ∈ {accepted, human_accepted}` (the `human_accepted` list is read when a file has
one; none does); a file whose `model`, or whose replicate proposal file's `model` when that
file exists under the root, starts with `mesa-clm` is refused (none was). The label-free check
of 2026-10-04 found no replicate proposal file under `curation/generic`; the ingest found **6**
under `sites/SRER/curation/proposals`, the per-site tree the generic files' `replicates[]
.proposal` point at, read each for its `model` only, and wrote **0** `teacher_implicit` rows:
the ingest reads a replicate proposal for the D19 refusal and synthesizes no row from its
unpicked `seen_curies` (the report counts `proposal_files: 6`, `implicit_rows: 0`). Stated in
advance: X4's `teacher_implicit` weights (0.3, 0.1) therefore weight nothing in this run, and
its arms differ in the `teacher` weight alone.

B3.2 **Drops, counted by reason** (`TeacherReport.dropped`): `out_of_registry` (a CURIE prefix
that is not a registry ontology: STATO, OBCS, CHEBI, GO, TO, ECOCORE, CHMO, UBERON, PPO, AGRO,
PO, OBA, FLOPO, SO), `unmapped_aspect` (`process`, `NEON_ASPECT_MAP`), `ontology_mismatch` (the
file's `ontologyId` is not the CURIE's prefix), `no_column` (a dataset-level item),
`unresolved_column` (no table card of the product carries the column), `term_missing` (OLS has
no usable record: root, obsolete, unknown), `collapsed_identity`. Rejected for `no_column` and
`unresolved_column`: synthesizing a dataset-scope `term.fits` state (plan §5.1's "else dataset
scope") — a dataset-scope state needs a scope the item does not name, and the aspect of such a
row would be invented; the plan marked the resolution "unverified → M4 task" and this is the
conservative answer. Label-free check (2026-10-04): 110 distinct in-registry, aspect-mapped
CURIEs; 231 (item, card) resolutions from 101 items with a column, 11 items with an unresolved
column, 101 dataset-level items without a column, 54 of 55 products have a card
(DP1.10092.001 has none). **The ingest's counts** (§12.2, 2026-10-05 01:44–01:46Z; DESIGN M4
notes, item 6): 55 files, 389 items; dropped 103 `out_of_registry`, 101 `no_column`, 22
`unmapped_aspect`, 17 `collapsed_identity`, 11 `unresolved_column`, 0 `ontology_mismatch`, 0
`term_missing` (`terms_missing` empty: every in-registry CURIE had a recorded OLS term); 5
ontology rows skipped as `ontology_not_aspect_allowed` (not a drop, B3.4); 231 (item, card)
resolutions; 435 rows inserted, 231 `term.fits` and 204 `column.ontology_fits`;
`products_without_card: [DP1.10092.001]`.

B3.3 **Target resolution.** A column resolves to **every** table card of the product
(`sites/SRER/cards/anyjev/<DP>.<table>.md`, `cards.py`'s grammar) whose column set contains it,
one row per card; the rows share `leak_group` = the product code (so a fold holding out a card
of that product drops all of them). The item's aspect is the mapped mesa-clm aspect.

B3.4 **Rows**, Yes only (no negative is synthesized): `term.fits` for (column state, CURIE)
with the candidate as `TermResolver` records it (`Candidate.as_state()`: the OLS label, CURIE,
ontology, description cut at 300 characters, synonyms, children; the framings render the
candidate text `"{label}: {description}"`), through the OLS layer replaying
`tests/fixtures/ols-teacher/` (110 `get_term` responses recorded once from live EBI OLS4 on
2026-10-04, 22:17:41Z–22:18:12Z, 110 requests, 0 failures; `scripts/record_ols_teacher.py`;
label-free); `column.ontology_fits` for (column state, ontology) when the ontology is
aspect-allowed for the item's aspect (`registry.allowed_for_aspect`; otherwise the ontology
row is skipped and counted as `ontology_not_aspect_allowed` in a counter of its own, outside
`dropped`, since the term.fits row is still written and the item is not lost). Every row: `teacher` at
weight 0.5, `fold_eligible=false`, `bench_card=false`, `leak_group=product_code`,
`origin="teacher:<file sha256[:12]>"`, the actor `ingest-teacher`. `labels ingest-teacher
--neon-root … [--cards-dir …]` prints the report; `labels snapshot` then writes
`bench/snapshots/<date>-teacher.parquet`, whose silver rows are byte-identical to the registered
snapshot (checked by the X4 producer: the content digest of its non-teacher rows equals
`labels_content_sha256`).

B3.5 **Features.** The teacher snapshot's manifest carries the teacher states by their identity
(the manifest reads identity and state columns, no source), so `features build --snapshot
<teacher snapshot> --tasks term.fits,column.ontology_fits --framings F7,F9` embeds their texts;
`bench x4` indexes the teacher snapshot's manifest under `F7,F9,X2` and builds the teacher rows'
spec matrices through a `FeatureBuilder` over the teacher item set under the task's active
framing (`learn.probe.TeacherRows.from_task`).

B3.6 **The silver-minus-Opus subset.** The `--exclude-models claude-opus-5-5` path
(`learn.labels.ingest_neon_eval(exclude_models=…)`, `cli.py` `labels ingest-neon-eval
--exclude-models`) rebuilds the silver from `neon-avu-eval/results/validated.json` with Opus's
runs removed: the model count drops to three, so a pair only Opus proposed disappears and a pair
Opus plus one other model proposed becomes `consensus_negative` (its label flips to No). It
**cannot** be applied to the committed snapshot alone: the snapshot does not carry the model set
of a consensus row. So the subset is produced by a label-free rebuild from
`tests/fixtures/neon-avu-eval` with the exclusion (`labels ingest-neon-eval --eval-root
tests/fixtures/neon-avu-eval --exclude-models claude-opus-5-5` into a separate store, then
`labels snapshot --out bench/snapshots/<date>-minus-opus.parquet`), committed and pinned before
the run. The **surviving** items (`learn.labels.surviving_identities`) are the registered items
whose identity the rebuilt task carries **with the same label**; a weight change alone
(`consensus_all` → `consensus_majority`) survives, a flipped label or a vanished identity does
not. The silver-minus-Opus scoring is the same predictions on that subset, scored on the
registered labels (equal to the rebuilt ones on it by construction). Rejected: "present with any
label" (a label Opus's vote created is not a surviving silver label).

B3.7 **The arms.** Three runs of the X3 nested procedure on **identical outer folds** (the
task's `leave_one_card_out`, a function of the silver items alone): off (no teacher rows),
(0.5, 0.3) and (0.3, 0.1) for (`teacher`, `teacher_implicit`) weights; the teacher rows enter
every training set, outer and inner, as sample weights (D20), except the rows whose `leak_group`
is the held-out card's product (both bench products have a corpus file, so the rule fires on
every fold); no teacher row is ever a test item (they are not items of the task; the pooled
indices are checked to be silver items; `teacher_in_test: false`). Cells
`<task>.probe.<framing>@teacher-off`, `@teacher-0.5`, `@teacher-0.3` and their `-minus-opus`
twins; every cell is a variant, pre-registered, `exploratory: true`, never citable; `teacher:
true` on the on-arms; the cells' identity is the off arm's full-data choice (what production
serves). Each cell's diagnostics record the arm, the weights, the teacher rows available and
weighted, the scoring, the subset size and `nested_probe`'s per-fold `n_teacher`.

B3.8 **Decision.** Keep teacher labels in production fits iff `on ≻ off` on **novel-key NLL**
under rule R for **both** scorings, for the (0.5, 0.3) arm; the (0.3, 0.1) arm is reported. The
novel keys are the items whose lookup key no training card of their fold carries
(`baselines.evaluate_lookup` over the off arm's evaluated folds), the comparison paired on the
items both arms pooled; fewer than two novel-key cards is recorded as such and counts as not
passing. The record (`diagnostics.teacher_decision` of every X4 cell and the file's notes) holds
rule R's full record per arm and scoring and `keep`. Expected ≈ null with few teacher rows in the
bench products' registry (stated in advance).

## 6. Artifacts, `learn fit`, `learn promote`, publish and pull (`artifacts.py`)

### C1. Artifacts (plan §5.5)

C1.1 **Layout.** `<artifacts.dir>/<encoder_fp>/<clm_model_fp>/<lock8>/v<N>/{manifest.json,
calibrators.json, probes/<question_key>.json}` and `CURRENT.json` at `<lock8>/`, `lock8` the
first 8 hex characters of `framings.lock_sha()` (`artifacts.LOCK_SHA8`). One bundle per
`clm_model_fp`: a probe lives under its spec's model (R1: `clm-latest` for `lowdim.v1`,
`pair512.v1`, `choice.state.v1`; `clm-raw` for `pair4096.v1`, `joint4096@S1`, `joint4096@S1ns`,
`choice.raw.v1`; `tiered.SPEC_MODELS`) and is served by the provider of that model. Rejected: one
bundle holding both models' probes (the record's `clm_model_fp` is the
served head's, and an artifact's `(question_key, encoder_fp, clm_model_fp)` must equal the
record's, plan §4.6 invariant 5).

C1.2 **Immutability.** A version is written to a temporary sibling directory (0700) and moved
into place with `os.rename` (`os.replace` would overwrite: `rename` onto an existing directory
fails, which is the refusal we want); an existing `v<N>` is never rewritten, by `write_version`,
`publish` or `pull`; `next_version()` is `max + 1`. Files 0600, directories 0700 (`perms.py`).

C1.3 **The manifest** (`artifacts.FORMAT = "mesa-clm/artifacts/1"`): `version`, `created_at`,
`encoder_fp`, `clm_model_fp`, `framings_lock_sha` (full), `labels_sha256`,
`labels_content_sha256`, per `question_key` a `ManifestEntry` (`task_id`, `framing_id`, `tier`
∈ {calibrated, probe}, `file`, `spec`, `fitter`, `hyper`, `calibrator` (the fitted
`learn.calibrate` JSON), `n_train`, `cite`, `cell`) and `files` (relative path → sha256 of every
file but the manifest). `cell` (`CellIdentity`) copies the cited cell's `path`, `key`, `task`,
`tier`, `labels_sha256`, `labels_content_sha256`, `question_key`, `fingerprint` and the
`fold_choices` agreement count with this artifact's configuration (`FoldAgreement(agree,
folds)`: calibrated → the fold's `(framing, model)` is the arm's; probe → the fold's `model` and
`spec` are the artifact's; a fold with `evaluated: false` never agrees). No metric is copied.
Reading: "`fold_choices` agreement count" counts evaluated folds that chose this configuration;
rejected: counting skipped folds as agreeing (a skipped fold chose nothing).

C1.4 **Strict load** (`load_version`, `cfg.artifacts.strict`): every listed file re-hashed (a
mismatch or an unlisted file is refused, strict or not: tampering is never relaxed), the
manifest's fingerprints must be the layout's and the live `Fingerprint`'s (K4), its framings
lock the layout's, and every entry's `question_key` must be its task's *active* framing's
(`framings.active_framing`); `strict=False` drops the mismatched or stale entries with a warning
and reports them (`LoadedVersion.dropped`), never silently. Reading of "stale question keys":
a key that is not the task's active framing's key today (a rotated template or a changed active
framing). Rejected: any key `framings.by_question_key` still knows (an inactive framing's probe
would then be served for the active framing's requests, which never asks it).

C1.5 **`CURRENT.json`** (`artifacts.CURRENT_FORMAT = "mesa-clm/artifacts-current/1"`): per
`task_id` `{version, tier, question_key, promoted_at, cite}`. The provider's bundle
(`bundle_from_current`) is the union of the promoted entries over their versions (a promotion
table may span versions; `ArtifactBundle.versions` keeps each key's version and the record's
`artifact_version` is that entry's). Under a promotion table only promoted entries are served:
an unpromoted calibrator or probe of the same version is `TierUnavailable` at `--tier` and
`zero_shot` at `auto`. A bundle without a table (a `calibrators.json` loaded directly, M1's
path) serves its calibrators as before. Rejected: serving every entry of the newest version
(promotion, not fitting, is what admits an artifact to production, plan §5.5).

C1.6 **The live provider** (`live.clm_provider`): loads `CURRENT.json` under
`<artifacts.dir>/<encoder_fp>/<clm_model_fp of clm.model>/<lock8>/`; a promoted probe whose
spec reads head quantities needs the pinned head's export (`heads/npz/<sha8>.npz` with the
lock's `source_sha256`, as the doctor loads it) and the live feature store, when it exists, is
read (never written) as a vector cache. A mismatch refuses (`ProviderSetupError`, K4) unless
`artifacts.strict` is false, which drops the mismatched entries with a note on the run.
Promoted artifacts filed under another framings lock than the live one (another `<lock8>`
directory with a `CURRENT.json`, none under the live lock) are refused the same way, not served
as zero_shot. Without a `CURRENT.json` the provider serves `zero_shot`.

C1.7 **`learn fit`** (`artifacts.fit_version`, the CLI): from the registered snapshot
(`bench.run.load_inputs`: the sha256 and content checks of M2) and the feature store of the
serving lock, per task (a) the `@full` probe (`learn.probe.full_probe`: inner LOCO over all
seven cards picks the configuration, the fit is on all items, the calibrator from the seven-card
OOF predictions; the probe's model names the layout it is written to) and (b) the calibrated
tier's calibrator: `learn.calibrate.fit_calibrator` on every item of the A1 arm
(`framings.ACTIVE[task]` on `clm-latest`) with the label weights (D20). Reading of "A1 arms":
every task's active framing on `clm-latest` except the tasks `decider.ols_rank_tasks` names
(K1, DESIGN A1: `term.fits` has no A1 arm; a calibrator for it would cite nothing), i.e.
`column.ontology_fits` (F9) and the closed choices (F7). One new version per served model that
received an artifact. **The registration applies** (B1.2): `--tiers` defaults to the pinned
`tiers.json` (`current_m4().tiers_path()`, after `check_committed_input`) and `--x3` to
`bench/results/<date>/x3.json`; the verb computes `m4_identity_deviations` (the framings lock,
the fingerprints) and `grid_deviations` (`DEFAULT_GRID` per shape) and **refuses** when any
deviation exists (exit 1, nothing written: no version directory, no export); the manifest
records `registered: true` and the registration's identity (the lock sha, the fingerprints, the
pinned input paths with their sha256). A `--tasks` subset is allowed, since promotion is per
task, and the manifest records `tasks_fitted`. The manifest entry cites
`bench/results/<date>/x3.json#<task>.probe.<framing>` and
`bench/results/<date>/tiers.json#<task>.calibrated.<framing>` when that nested cell exists in
the file, else `cite: null`, which promotion refuses: **by construction** the calibrated entries
of the closed choices (fixed F7 arms, not nested) and of `term.fits` (K1's `selection: "none"`
audit cells) get `cite: null` from the pinned `tiers.json`, and the CLI prints one line per such
entry saying promotion will refuse it — a fact about the cells, not an error. The export copy
goes to `<out-dir>/<date>/artifacts_v<N>/<clm_model_fp>/` (manifest and probes; the calibrators
are in the manifest's entries). `learn fit` never touches `CURRENT.json`.

C1.8 **`learn promote --version N --task T [--tier]`** (`artifacts.promote`): (i) the entry's
cite names a cell of the task and tier with `loco`, `pre_registered`, `selection: "nested"`, not
`exploratory`, whose `question_key`, `encoder_fp`/`clm_model_fp` and `labels_sha256` equal the
manifest's; (ii) `k2.json`'s verdict for that tier is `a` or `b` (`k2_verdict`: `tasks[task].best
== tier` gives `tasks[task].verdict`, and nothing else does; K2 judges the best servable tier
only, so another tier has no verdict and is refused; `c` is refused. Rejected: a per-tier
verdict field in `k2.json` read by promotion — K2 never judged that tier, so a verdict for it
would be a number the frozen K2 text does not define); (iii) when the task already serves a promoted
artifact, the new cell ≻ the current one on NLL under rule R (`stats.rule_r`, B = 2000, seed 0,
α = 0.05 from the registration) and ≽ on accuracy within `PROMOTE_ACC_MARGIN = 0.01`
(`stats.non_inferior`), both on the two cells' pooled `items` joined by `(target_sha256,
option_key)` (a partial join is refused); (iv) a head is refused until M7. `--tier` absent:
the version's probe for the task, else its calibrator. Nothing is written when a rule fails.

C1.9 **publish / pull.** `publish --to DIR` copies `v<N>` to `<DIR>/<encoder_fp>/<clm_model_fp>/
<lock8>/v<N>/` and verifies the copy against the manifest; `pull --from DIR --version N` copies
the version into a temporary sibling, re-hashes every file (`--no-verify` skips it; the strict
load refuses the tampered version later anyway) and refuses on a mismatch, placing nothing.
`CURRENT.json` is never pulled: promotion is a local decision. Local paths only in M4 (an ssh
or iRODS mount counts).

C1.10 **Health.** The doctor's `artifacts` check (after `feature store`): `CURRENT.json` parses,
each promoted version's manifest loads with the live lock's fingerprints and the code's framings
lock, each promotion's cite names a results file (under `policy.results_root`) that holds the
cell; other locks' promoted artifacts are a warning; the artifacts tree joins the
`permissions` targets.

## 7. The probe tier served (`providers/tiered.py`, `providers/live.py`)

### C2. The probe tier (plan §4.1, §4.6, §5.3)

C2.1 **Resolution.** `resolve_tier(qk)` = `probe` when `CURRENT.json` promotes a probe for `qk`,
else `calibrated` when a servable calibrator exists, else `zero_shot`; `--tier probe` without a
promoted probe is `TierUnavailable`; `annotate --tier probe` admits the tier when the provider
supports it for every task not decided by `ols_rank` (`cli._check_tier`).

C2.2 **The path.** The state text (`framings.context_text`, what `context_sha256` hashes) and the
option texts (the wire criteria, anchor last) go to the encoder through a `VectorCache`
(`VECTOR_CACHE_MAX = 2048` texts, LRU; the live feature store read first when configured; the
store is never written); the served head's export projects (`HeadProjector`); the spec
features are built with `learn.probe.rank_fit_features` / `choice_features` (the one place the
formulas live; `tiered.rank_fit_features` computes `s_latest = scale·(zs·zc − zs·za)`, `s_raw =
100·(xs·xc − xs·xa)`, the projections, and hands them over); the probe's `predict` gives the
calibrated `[m, K]` probabilities. clm-serve is never asked on this path.

C2.3 **The anchor's row.** A rank_fit probe scores candidates; the record needs `p_fit` per
option, the anchor included. Reading: the anchor's row is the features of a candidate whose
vector is the anchor's own (`s_latest = s_raw = 0`, `zc = za`, `xc = xa`; a joint spec's anchor
row is its own option vector), the probe analogue of Platt's anchor `p_fit = σ(b)` ("a
candidate scoring exactly like the anchor"). Rejected: `p_fit = 0.5` or `0` for the anchor (a
number the probe did not produce). The anchor's `p_fit` never enters `probs`.

C2.4 **The record.** `method = clm` (`fake` under the fake stack), `level = probe`,
`calibration` the probe calibrator's kind (`platt` at K = 2, `temperature` above),
`raw_probs` the zero-shot softmax over the same vectors under the served model (`clm-raw`:
`softmax(100·cos)`; a head: `softmax(scale·cos)` of its projections; CLM's own float64
softmax), `s_c` its log-ratios (anchor 0), `p_fit` the probe's calibrated Yes probability per
option, `probs = softmax(logit(p_fit) for candidates, 0 for the anchor)` — the mirror of
`PlattCalibrator.rank_fit`, whose candidate logits are `a·s_c + b` — with `p_fit` clipped to
`[PROBE_LOGIT_CLIP, 1 − PROBE_LOGIT_CLIP]`, `PROBE_LOGIT_CLIP = 1e-12` (the NLL clip), so each
candidate's odds against the anchor are exactly its `p_fit`; a closed choice's `probs` are
`predict`'s row. `confidence = max(probs)`, `margin = top1 − top2`, `served_model = None`,
`clm_confidence = None` (no CLM field exists), `feature_spec` the spec, `artifact_version` the
probe's version and `artifact` its reference, `latency_ms` the local wall time, `diagnostics`
`{probe, spec, feature_dim, n_embedded, token_source}`. The aspect mask applies after scoring
as for every tier. The token guard applies unchanged. The encoder call is reported to
`clm_calls` as `/v1/embeddings` with the tokens charged (only when something was embedded); the
encoder down gives `unavailable` records (`decider_unavailable`), a 401/403/404 or a port held
by another account refuses the run (`DeciderRefused`), exactly clm-serve's rules.

C2.5 **Joint specs at serving time.** `joint4096@S1`/`@S1ns` embed the X1 control framing's
per-candidate anyjev state (`framings.control_state` with `n_candidates` the group's size)
rendered by CLM's `state_text` with (S1) or without (S1ns) the task question, exactly
`learn.features._joint_rows`. Reading of `n_candidates`: the request's group size (the
snapshot's rows carry the evaluation's count, which serving cannot know); it only affects the
joint specs.

C2.6 **`_honest`.** `feature_spec` is set iff `level ∈ {probe, head}` (`providers.base
.FEATURE_SPEC_LEVELS`, `_check_feature_spec`). The DDL is unchanged (no CHECK added, so no
mirror in `DecisionRow._mirror_the_checks`): the column is free text, the rule lives in the
record validator, and a migration for a CHECK is not worth a schema version in M4.

## 8. The citation test (`policy.CellCitationValidator`)

### C3. The citation test (D8, plan §4.7, PR "Citation test")

C3.1 `policy.CellCitationValidator` is `Policy.load`'s default validator; `RefuseAllCitations`
stays for callers that want it. `results_root` = `policy.results_root`, default the checkout
root when `mesa_clm` is imported from a `src/` checkout (the directory holding `src/` and
`pyproject.toml`), else `~/.mesa/clm/results` (`DEFAULT_RESULTS_HOME`). The committed snapshots
are the sha256 of every `<results_root>/bench/snapshots/*.parquet` (`SNAPSHOTS_DIR`; hashed
bytes, never opened for their labels), or a configured set given to the validator.

C3.2 **Checks, in order** (the first failure is the `cite_refused` reason): the cite is
well-formed (`CITE_RE`); the file exists under `results_root` and loads; the cell exists and its
`task_id` is the record's; its `tier` is the record's `level` (C3.3); `loco`; `pre_registered`; `selection == "nested"`; not `exploratory`;
`servable`; `n_folds >= MIN_NESTED_FOLDS = 5`; `teacher_in_test is False`; `masked` equals the
record's framing's masking (`mask_rule` set ⇔ `masked`: the rank_fit task with an aspect mask,
`column.ontology_fits`, is `masked: true`); `labels_sha256` ∈ the committed snapshots;
`counts.n_neg >= GUARD_MIN = 30` (rank_fit) / `n_nonmodal >= 30` (choice); `question_key` and
`encoder_fp`, `clm_model_fp`, `schema_sha256` equal the record's; the cell's fingerprint equals
the live `Fingerprint`, `serving_lock_sha` included (the pipeline binds the provider's
fingerprint; unbound, every cite is refused: `serving_lock_sha` is not on a record and cannot
be checked otherwise); `fold_choices` agree with the served configuration in at least
`FOLD_AGREEMENT_MIN = 5` evaluated folds (calibrated cell: the fold's `(framing, model)` is the
cell's; probe cell: the fold's `model` is the record's `model` and its `spec` the record's
`feature_spec`; a record without one never agrees); `baselines.beats_lookup_novel is True`; `metrics.threshold_cp[f"{risk:.2f}"]`
exists (`None` refuses) and `thresholds.auto >=` it.

C3.3 **Reading of "the tier's level rank ≥ the record's level".** Taken at its most
conservative: equality of the cell's tier and the record's level, which satisfies the
inequality and is the only reading under which the threshold was derived from the numbers it
is applied to. Rejected: the literal inequality alone (it admits a probe cell's threshold for a
calibrated record, whose `p_fit` came from another model) and the reverse inequality (a
calibrated cell for a probe record).

C3.4 **The audit blocker** (`auto_requires_audit`, the `prod` profile). After every other gate
and the citation pass, an `auto` needs a passing `audits` row for the record's `(task_key,
artifact_version)` read from the `AuditCheck` the caller supplies (`StoreAuditCheck` over the
sidecar, bound by `Annotator`); without one the verdict is `proposed` with reason
`audit_required` (`AUTO_BLOCKERS` gains `audit_required`; `DEMOTION_REASONS` lists it as the
one reason a proposed row records), and the sidecar's `decisions.reason` says so. A record
without an `artifact_version` has no audit. A check that raises fails closed. `apply` is M3's;
the pipeline records the blocker now.

## 9. Audits (`audit.py`)

### C4. Audits (plan §4.7, §8 M4)

C4.1 **`audit sample --n N --min-cards M --runs|--since --out FILE`** (`audit.py`): from the
actor's annotate runs on non-bench cards (a `BENCH_CARDS` card or a card with silver labels in
the store, `DecisionService.is_bench_card`, refuses the whole sample: a curator label on a
pre-registered item is never minted here), the pool is every decision with a distribution
(rules, planners, unavailable records excluded; `ols_rank` proposals included, the K1
proposals among them), in three strata: `would_be_auto` (outcome `proposed`/`auto` with the
policy statistic — `p_fit` of the answer for a rank_fit, `confidence` for a choice — at or
above the task's cited cell's `threshold_cp[risk]` when the policy cites one, else at or above
the `TOP_DECILE = 0.90` quantile of the task's statistics in the pool), `proposed` (outcome
`proposed`, not would-be-auto) and `anchor_abstain` (outcome `abstain`, reason `anchor_won`).
Equal shares (`n // 3` each, the remainder to the first strata; a stratum short of its share
passes the rest on, in order), each stratum sorted by decision id, permuted with
`numpy.random.default_rng(SAMPLE_SEED = 0)` and then interleaved round-robin by card so that a draw spans the cards the pool allows *(added 2026-10-05 after the first review's draw, before any audit row exists: the first draw happened to span 7 of 7 cards; a 9-item test draw did not always span its 3 cards)*; the sample must span at least `min_cards` cards
(`DEFAULT_N = 100`, `DEFAULT_MIN_CARDS = 5`, plan §8 M4). The file (`audit.FORMAT =
"mesa-clm/audit-sample/1"`) holds ids, task keys, strata, the statistic, the card *names* and
the checklist; no card content.

C4.2 **`audit review --file`**: a terminal is required (like `review`; the answers are curator
labels via `cli`, DESIGN A2). Each decision is shown through `DecisionService.explain` (the run)
and its group's candidates; `c`/`i`/`s`/`q`. A verdict mints `audit.labels_for_verdict`: a
rank_fit decision's answered CURIE gets `Yes` (correct) or `No` (incorrect); a correct closed
choice gets its answer; an incorrect closed choice, an anchor answer or an abstain mints no
label (no true class is named); every row is `curator`, weight 1.0, `fold_eligible=false`,
`bench_card=false`, `origin = audit:<audit_id>`. The file is rewritten after every answer.

C4.3 **`audit record --file`**: one `AuditRow` per `(task_key, artifact_version)` among the
reviewed `would_be_auto` items (`n`, `n_cards`, `cards`, `reviewer`, `n_errors`, `cp95_upper =
clopper_pearson_upper(n_errors, n)`, `risk` the task's policy risk, `passed = n >= AUDIT_MIN_N
(50) ∧ n_cards >= AUDIT_MIN_CARDS (3) ∧ cp95_upper <= AUDIT_RISK_FACTOR (2) × risk`, enforced by
`AuditRow` itself); decisions that apply no artifact (`zero_shot`, `ols_rank`) get no row and
are reported, since an audit is of an artifact version. The proposed-precision Clopper-Pearson
interval (precision point; lower `1 − cp95_upper(errors)`; upper `cp95_upper(correct)`;
report-only) is printed per stratum.

## 10. `bench e2e --loco` (`bench/e2e.py`)

### B4. `bench e2e --loco`

B4.1 **Interface, first wave.** Report-only, no cell. Per held-out card: `consensus_all` recall =
the fraction of the card's `consensus_all` Yes `term.fits` CURIEs (`consensus_all_curies`, from
the label store) that the run proposes (`AnnotationRun.proposals`), against AnyJev's static 0.25
(`ANYJEV_STATIC_RECALL`); anchor-abstain precision = the fraction of the run's anchor-won groups
(`abstained` with `reason == "anchor_won"`) whose candidates include no consensus-all Yes (with
the group's candidates; without them a group counts as clean only when the card has no
consensus-all Yes at all, the conservative reading). Pooled over cards with the card-cluster
bootstrap bounds (`measure`, B = 2000, seed 0, one-sided 95%).

B4.2 **Implementation, second wave.** The fold provider (a fold's nested probe or calibrator
served from the store behind the `DecisionProvider` protocol, the encoder for candidates outside
the store, the OLS fixtures replaying) is not in the pre-run commit; `bench e2e --loco` says so
(a usage error naming R8) and the measure ships by amendment with its own disclosure.

## 11. Registration, determinism and constants

11.1 **The registered configuration** (`bench.registered.REGISTERED_M4`, `current_m4()`; B1): the
M2 registration's snapshot, counts, B = 2000, seed 0, α = 0.05 and fingerprints; A1's framings lock
and the active framing per task; the X3 grid (specs per shape with their models, the fitters with
their hyperparameter values in declared order, the inner criterion, the floors 100/40, the
standardization floor); the committed `tiers.json` and `x2.json` by sha256; the K2 constants; X4's
arms; the teacher corpus hash with the neon-ducklake commit it was read at, the teacher snapshot
and the silver-minus-Opus snapshot (both hashes each), filled from the label-free ingest of
§12.2 (run before this commit) and committed with this plan. The verbs expose no option for B,
the seed, α, the grid, the framings, the models or the inputs.

11.2 **Anything else is not the registered run** (M2 plan §13.2 applies verbatim): the producers
(`x3`, `k2`, `x4`) compute their own deviations and write every cell `pre_registered: false`,
`exploratory: true` with the deviations listed, whatever the caller says; `learn fit` computes
its own and refuses on any (B1.2, C1.7); a teacher input left unpinned would make `x4`
unregistered (none is); a pinned input whose bytes differ, another snapshot, other label content
or other committed inputs are refused outright.

11.3 **Determinism.** The fitters have no randomness (SVD, Newton, closed forms; P.2); ties go to
the first configuration in declared order (P.3.3); every bootstrap (CIs, rule R, non-inferiority,
full and inner) uses B = 2000 and seed 0 on the sorted cluster set; the audit sample uses seed 0
(C4.1); Platt starts from a fixed point and temperature bisects; items are in (target_sha256,
option_key) order; float64 throughout. Two runs of a producer on the same inputs are byte-identical
(tested on synthetic worlds).

11.4 **Strict inequalities are strict** (M2 plan §13.4): a card's Δ of exactly 0 (or of exactly the
margin) is not a win; a lower bound of exactly the bound does not pass; an NLL tie between
candidates goes to the lower-rank tier (B2.2).

11.5 On another platform a recomputed number can differ in its last bits (BLAS and SIMD summation
order; the SVD reduction, P.2.2); a discrete outcome (a fold's configuration, a K2 verdict, a
promotion) that flips under such a difference is recorded as a discrepancy, never silently
re-derived.

## 12. Run protocol

12.1 **Before the commit** nothing combines a model output with a real label: the code is exercised
on synthetic data only, and no M4 verb (`bench x3`, `bench k2`, `bench x4`, `learn fit`, `learn
promote`) runs on the registered snapshot, the live feature store or the sidecar. Commit and push
this plan, the code and DESIGN's M4 notes (the pre-run disclosure) together.

12.2 **The teacher inputs, label-free, once — done before this commit and disclosed** (B3; B5
step 1). Run on 2026-10-05 from 01:44Z to 01:46Z, before the pre-run commit, and reported in
DESIGN's M4 notes (item 6): `labels ingest-neon-eval --eval-root tests/fixtures/neon-avu-eval`
into a fresh store (`.local/m4/labels.duckdb`; its snapshot's content digest equals the
registered `5c60a8a6…`), then `labels ingest-teacher --neon-root ~/neon-ducklake` (the report
`.local/m4/ingest_teacher.log`, its counts in B3.1–B3.2) and `labels snapshot --out
bench/snapshots/2026-10-04-teacher.parquet` (1,369 labels); `labels ingest-neon-eval --eval-root
tests/fixtures/neon-avu-eval --exclude-models claude-opus-5-5` into a second fresh store
(`.local/m4/minus_opus.duckdb`) and `labels snapshot --out
bench/snapshots/2026-10-04-minus-opus.parquet` (679 labels). The pins (`teacher_corpus_sha256`,
`neon_ducklake_commit`, both snapshots' two hashes; B1.1) are written into `REGISTERED_M4`, the
snapshots committed with them and this plan, and the commit pushed **before** any M4 verb runs
on labels. These steps read labels (the registered ones copied, the teacher ones created) but no
model output, and nothing of them combined a label with a model output; the teacher snapshot's
silver rows are the registered rows (`bench x4` checks their content digest, B3.4).

12.3 **Features, label-free, once**: `features build --snapshot bench/snapshots/<date>-teacher.parquet
--tasks term.fits,column.ontology_fits --framings F7,F9` (the teacher states and candidates; the
registered items' texts are already in the store, §1.2), then `features project`. Disclosed with
the token count.

12.4 X3, once: `uv run mesa-clm bench x3 --date <date>` (every task, every variant; writes `x3.json`,
`x3.md`).

12.5 K2, once, after 12.4: `uv run mesa-clm bench k2 --date <date>` (reads the pinned `tiers.json`,
`x2.json` and this date's `x3.json`; writes `k2.json`, `k2.md`).

12.6 X4, once: `uv run mesa-clm bench x4 --date <date>` (the pinned teacher and silver-minus-Opus
snapshots; writes `x4.json`, `x4.md`).

12.7 `uv run mesa-clm bench table --date <date> --out bench/results/<date>/table.md`.

12.8 Every output of 12.4–12.7 is committed as produced, in one commit, before any of it is written
into prose. A verb that fails is re-run only after its error is recorded; `--force` replaces only a
failed or unregistered attempt, and the replacement is disclosed. All verbs run on the serving host
against the store of §1.2 and the checkout's `serving/serving.lock.json`.

12.9 **The artifacts, once, after 12.8**: `uv run mesa-clm learn fit --date <date>` (C1.7: the
full-data probes and calibrators, one version per served model, exported to
`bench/results/<date>/artifacts_v<N>/` and committed as produced). `learn fit` never promotes.

12.10 **Promotion** (C1.8) is decided by K2's verdicts and recorded, with X4's decision and K2's
consequences for the production configuration (`decider.ols_rank_tasks`, the served tier per task,
the `head_adds_nothing` record), in DESIGN amendment **A7**, after 12.8; `learn promote` runs once
per promoted task after A7 is written. No production configuration changes in the run itself.

12.11 **The production audit** (C4; plan §8 M4 acceptance): `annotate` on at least five non-bench
SRER cards under the promoted tiers, `audit sample --n 100 --min-cards 5`, the curator's `audit
review` (a terminal, the user), `audit record`; the proposed-precision interval and the `audits`
rows are reported in RESEARCH.md. A numeric `auto` stays null in 0.1.0 whatever K2 says (plan §8,
K2: "0.1.0 ships proposed-only").

12.12 `bench e2e --loco` (§10) is second-wave: it ships by amendment with its own disclosure, after
12.8.

## 13. What may still change after the first real run

**Nothing, except by amendment with the affected cells marked exploratory** (M2 plan §15 applies
verbatim). After the first run of 12.4–12.6, any change to this plan or to the code that would
change a number, a choice, a flag or the item set of a committed cell, a bug fix included, is made
only by a DESIGN.md amendment that names the change and why, and every cell it affects is marked
`exploratory: true`. A departure from §12 is disclosed the same way. The A7 amendment that records
K2's verdicts, X4's decision and the promotions is the planned consequence of the run, not such a
change; so is the second-wave `bench e2e` (§10, by its own amendment).

## Appendix A: frozen phrase → section

| frozen phrase (PR, plan) | section |
|---|---|
| "probe specs × fitters (nested)" (X3) | 0.4, P.1–P.4, 3 |
| "Specs `lowdim.v1`=[s_latest, s_raw]; `pair512.v1`=zs⊙zc ⊕ \|zs−zc\| ⊕ s_c; `pair4096.v1`=xs⊙xc ⊕ s_raw; `joint4096@S1`/`@S1ns`" (plan §5.3) | P.1 |
| "Weighted {logreg, lda, ridge}" (plan §5.3), "sample_weight in Platt, logreg, LDA and ridge" (D20) | P.2 |
| "spec/hyperparameters by grouped (card) inner-CV NLL, then OOF temperature" (plan §5.3) | P.3, P.4 |
| "floors 100 (calibrated) / 40 (probe/head)" (PR LOCO) | 0.4, P.3.2, P.4.1–P.4.2 |
| "train drops teacher rows whose leak_group equals the held-out product; … no fold_eligible=false, same-product teacher or bench_card curator row reaches a test fold" (PR LOCO) | P.5, B3.7 |
| "every choice among framings/models/anchors/specs/hyperparameters is re-made inside each outer fold … cells record fold_choices" (PR LOCO, D27) | P.3, P.4.5–P.4.6, 3.1–3.2 |
| "Predictions pooled, never fold-averaged" (PR LOCO) | P.4.4 |
| "Cell fields — identity … counts … metrics … baselines … diagnostics" (PR LOCO) | 3.1, M2 plan §9.6–§9.11 |
| "Rule R", "A ≽ B within δ = cluster LB > −δ" (PR Rule R) | B2.2–B2.4, C1.8 |
| "`beats_lookup_novel`" (PR Lookup) | B2.3 (1), M2 plan §9.9 |
| K2 "(a) auto-eligible = beats_lookup_novel ∧ paired non-inferiority to AnyJev L2 on the full set (Δacc cluster-LB > −0.02 + card-sign condition) ∧ ECE ≤0.08 with cluster-UB ≤0.12" | B2.3 (the card-sign condition is rule R's sign test (ii), no margin: B2.3 (3)) |
| K2 "(b) proposer-only = beats_lookup_novel only", "(c) tier killed → ols_rank proposals" | B2.3, 12.10 |
| K2 "If a clm-raw probe ≽ a clm-latest probe, RESEARCH.md records that CLM's head adds nothing" | 0.4, B2.4 |
| K2 "best servable tier, nested cells" | B2.1–B2.2 |
| X4 "Off vs (0.5/0.3) vs (0.3/0.1) on identical folds, scored on silver and silver-minus-Opus; keep teacher labels only if on ≻ off on novel-key NLL for both" | B3.6–B3.8 |
| D19 teacher labels (accepted items, in-registry, weights, leak group, refused files) | B3.1–B3.4 |
| "Teacher target resolution (column → table card, else dataset scope) is unverified → M4 task" (plan §5.1) | B3.2–B3.3 |
| "Citation test" (PR; plan §4.7; D8) | C3 |
| "With auto_requires_audit, apply also needs a passing audits row: ≥50 would-be-auto decisions from ≥3 non-bench cards, curator-reviewed, CP-95 error upper bound ≤ 2×risk" (PR Citation test) | C3.4, C4.3 |
| "Production audit: audit sample --n 100 over ≥5 non-bench SRER cards, stratified by outcome, curator-reviewed; proposed-precision CP CI reported" (plan §8 M4) | C4, 12.11 |
| "Artifacts and promotion" layout, "immutable", "strict load refuses fingerprint mismatches and stale keys" (plan §5.5) | C1.1–C1.6 |
| "learn promote … requires a pre-registered nested cell, new ≻ current on NLL and acc ≽ current within 0.01, and for a head, head ≻ probe (K3)" (plan §5.5) | C1.8 |
| "artifacts publish … manifest sha256 per file; artifacts pull --verify re-hashes and refuses on mismatch. Test: manifest LOCO equals the bench cell" (plan §5.5) | C1.3, C1.9 |
| "probe → EncoderClient.embed → headproj → LinearHead locally (bypasses clm-serve)" (plan §4.1) | C2 |
| "level ∉ {none, zero_shot} ⇒ artifact_version set, and the artifact's (question_key, encoder_fp, clm_model_fp) equals the record's" (plan §4.6 inv. 5) | C1.1, C2.4 |
| "Report-only with CIs: e2e consensus-all recall vs AnyJev static 0.25; anchor-abstain precision" (plan §8 M4) | B4, 12.12 |
| K4 drift | C1.4, C1.6, C1.10 |

## Appendix B: every constant

| constant | value | where |
|---|---|---|
| specs (declared order) | `lowdim.v1, pair512.v1, pair4096.v1, joint4096@S1, joint4096@S1ns` (rank_fit); `choice.state.v1, choice.raw.v1` (choice) | `probe.SPECS`, `probe.DEFAULT_GRID` |
| spec models | clm-latest: `lowdim.v1, pair512.v1, choice.state.v1`; clm-raw: the rest | `probe.SPECS` |
| registered dims | 2, 1025, 4097, 4096, 4096, 512 + K, 4096 + K | `probe.SPECS`, `PROJECTION_DIM = 512`, `encoder.EMBEDDING_DIM = 4096` |
| fitters (declared order) | `logreg, lda, ridge` | `linear.KINDS`, `probe.FITTERS` |
| logreg λ grid | 1e-2, 1e-1, 1, 10 | `linear.GRIDS["logreg"]` |
| lda γ grid | 0.1, 0.5, 0.9 | `linear.GRIDS["lda"]` |
| ridge λ grid | 1e-1, 1, 10 | `linear.GRIDS["ridge"]` |
| penalty convention | weighted-mean data term + λ/2‖coef‖²; intercepts unpenalised | `linear` (P.2.7) |
| standardization | weighted mean, weighted biased std, floor 1e-8 | `linear.STD_FLOOR` |
| reduction rank tolerance | `s_max · max(n, d) · ε`, ε = float64 eps | `linear.RANK_EPS` |
| logreg Newton | ≤ 100 iterations; gtol 1e-8; Armijo 1e-4; ≤ 60 halvings; quadratic cut 1e-12; start 0; non-converged used, flagged | `linear.LOGREG_*` |
| logreg form | K = 2 binary (row 0 vs 1); K > 2 symmetric multinomial, gauge `b[K−1] = 0` | `linear._fit_logreg` |
| lda | weighted means and priors; pooled within covariance / Σw; `(1−γ)Σ + γ (tr Σ / d) I`; absent class refused | `linear._fit_lda` |
| ridge | one-hot targets, √w row scaling, intercept = weighted class frequency | `linear._fit_ridge` |
| probe floor | 40 training items (outer T and every inner fold) | `cells.PROBE_FLOOR`, `probe.PROBE_FLOOR` |
| calibrator floor | 100 pooled OOF items (`below_floor n < 100 OOF items (probe calibrator)`) | `probe.CALIBRATION_FLOOR` |
| guards | 30 / 5 per class, two-class tasks, task items only | `bench.tasks.base` |
| inner criterion | pooled OOF NLL of the uncalibrated probe probabilities, unweighted, NLL clip 1e-12 | `probe.inner_select`, `stats.metric_value` |
| tie rule | first in declared order (specs, fitters, hypers); strict `<` | `probe.inner_select` |
| OOF calibrator | K = 2 Platt on the logit difference (hard targets, §6.4 constants); K > 2 temperature on log-probs (§6.5 constants); weighted | `probe._fit_probe_calibrator`, `calibrate.FITTER_CONSTANTS` |
| teacher rule | added to every training set except the held-out card's product; never in a test index; not counted by guards or floors | `probe.TeacherRows`, `nested_probe`, `inner_select` |
| artifact format | `mesa-clm/probe/1` | `probe.PROBE_FORMAT` |
| determinism | no randomness; SVD + Newton + closed forms; two runs byte-identical | tested |
| M4 framings lock sha | `7c93cc3e0ff6c26918a631cb787ceb6623dfd797fa1bb90cc353e956c1856920` | `bench.registered.FRAMINGS_LOCK_SHA_M4` |
| active framings | annotate F7, aspect F7, ontology_fits F9, term.fits F7, value_kind F7 | `bench.registered.ACTIVE_FRAMINGS_M4`, `framings.ACTIVE` |
| tiers.json pin | `bench/results/2026-10-03/tiers.json`, sha256 `42501a293f45eacdd5db70fc7ee6d28acd5d9c726bcf3b79e5f657ac52f8684d` | `bench.registered.REGISTERED_M4.tiers_*` |
| x2.json pin | `bench/results/2026-10-03/x2.json`, sha256 `76e07c34ddaed64e8987a2611c5af462457d7c398f49027533de3ea43cdb701d` | `REGISTERED_M4.x2_*` |
| X3 specs | rank_fit: lowdim.v1, pair512.v1, pair4096.v1, joint4096@S1, joint4096@S1ns; choice: choice.state.v1, choice.raw.v1 | `bench.registered.X3_GRID["specs"]`, `learn.probe.SPECS` |
| spec models | clm-latest: lowdim.v1, pair512.v1, choice.state.v1; clm-raw: pair4096.v1, joint4096@S1, joint4096@S1ns, choice.raw.v1 | `X3_GRID["spec_models"]` |
| fitters and grids | logreg λ ∈ {1e-2, 1e-1, 1, 10}; lda γ ∈ {0.1, 0.5, 0.9}; ridge λ ∈ {1e-1, 1, 10}; declared order = tie order | `X3_GRID["hypers"]`, `learn.linear.GRIDS` |
| inner criterion | pooled OOF NLL of the uncalibrated probe; ties → first in grid order | `X3_GRID["inner_criterion"]` |
| floors | 100 OOF items (probe calibrator); 40 training items (probe); std floor 1e-8 | `X3_GRID`, `bench.x3.CALIBRATION_FLOOR`, `PROBE_FLOOR`, `learn.linear.STD_FLOOR` |
| X3 variants | `@latest`, `@raw` (one model's specs), `@full` (fixed configuration) | `bench.x3.VARIANTS` |
| K2 acc margin | 0.02 (cluster LB of Δacc > −0.02; the gate. The per-card margin variant, Δacc > −0.02 per card, is report-only: `numbers.card_sign_margin_report`, in integers) | `bench.k2.ACC_MARGIN`, `K2_CONSTANTS` |
| K2 clm-raw clause margin | 0.01 | `bench.k2.HEAD_ACC_MARGIN` |
| K2 ECE | ≤ 0.08; cluster UB ≤ 0.12 | `bench.k2.ECE_MAX`, `ECE_UPPER_MAX` |
| card-sign condition | rule R's sign test (ii), no margin: per-card Δacc > 0 on ≥ ⌈0.8·m_c⌉ of the m_c cards with ≥ 10 items; m_c ≥ 4 | `stats.card_sign_test`, `SIGN_FRACTION`, `SIGN_MIN_ITEMS`, `MIN_CLUSTERS` |
| K2 tie on NLL | the lower-rank tier (calibrated) | `bench.k2.evaluate_task` |
| K2 candidate eligibility | `pre_registered: true`, nested, servable, not exploratory, pooled items; per cell | `bench.k2._candidate` |
| K2 full-set count | `diagnostics.task_counts.n`, else the registered count; neither → `task_count_unknown` (fails) | `bench.k2._task_n` |
| K2 verdict read by promotion | `tasks[task].verdict` iff `tasks[task].best == tier`; no per-tier verdict | `artifacts.k2_verdict` |
| X4 arms | off; (0.5, 0.3); (0.3, 0.1); decision arm (0.5, 0.3) | `bench.registered.X4_TEACHER_ARMS`, `X4_DECISION_ARM` |
| X4 scorings | silver; silver-minus-opus (variant suffix `-minus-opus`) | `bench.x4.SCORINGS`, `MINUS_OPUS_SUFFIX` |
| excluded model | claude-opus-5-5 | `bench.x4.OPUS` |
| teacher weights | teacher 0.5; teacher_implicit 0.3 (0 rows) | `learn.labels.WEIGHTS` |
| teacher statuses | accepted, human_accepted | `learn.teacher.TEACHER_STATUSES` |
| refused model prefix | `mesa-clm` | `learn.teacher.REFUSED_MODEL_PREFIX` |
| teacher origin | `teacher:<file sha256[:12]>` | `learn.teacher.TEACHER_ORIGIN` |
| corpus layout | `curation/generic/*.validated.json`; cards `sites/SRER/cards/anyjev` | `learn.teacher.CORPUS_DIR`, `CORPUS_PATTERN`, `CARDS_DIR` |
| corpus pin formula | sha256(json.dumps(sorted((name, sha256(bytes))))) | `bench.registered.teacher_corpus_sha256` |
| teacher corpus pin | `d90387928bed0a9c196a8efc4dce3cbf40b1e2e8cc879bfb50c5e3451b208785` at neon-ducklake commit `b1fa52a8d3ffc56173a09794253ad86b0fd10111` | `REGISTERED_M4.teacher_corpus_sha256`, `neon_ducklake_commit` |
| teacher snapshot pin | `bench/snapshots/2026-10-04-teacher.parquet`; sha256 `397f98d4b28c1cbff951a0797a7c3dca73dfe5e21b7e59262ca79bbf7906153f`; content `deb0dc46e17bdf7879f11135cec8090f58f12f7ed50f114037852e93a79bb07b` | `REGISTERED_M4.teacher_snapshot`, `teacher_labels_sha256`, `teacher_labels_content_sha256` |
| minus-Opus snapshot pin | `bench/snapshots/2026-10-04-minus-opus.parquet`; sha256 `2cd529ffa43986585bcb79988fbe683febc137d310ce34d9b94ace809a3676fe`; content `d8e26a0c7ac3b182e00741715d22e9e5a84c47ea898bf1d086a77c641acaa62c` | `REGISTERED_M4.minus_opus_snapshot`, `minus_opus_labels_sha256`, `minus_opus_labels_content_sha256` |
| `learn fit` on a deviation | refused (exit 1, nothing written); manifest `registered: true`, `tasks_fitted` | `artifacts.fit_version`, `cli._learn_fit` |
| M4 manifest framings | `F7,F9,X2` (rank_fit); `F7` (choice) | `bench.run.M4_FRAMINGS` |
| e2e static recall | 0.25 | `bench.e2e.ANYJEV_STATIC_RECALL` |
| e2e anchor reason | `anchor_won` | `bench.e2e.ANCHOR_REASON` |
| bootstrap B, seed, α | 2000, 0, 0.05 one-sided (from the M2 registration) | `stats.DEFAULT_*`, `registered.REGISTERED` |
| artifacts format | `mesa-clm/artifacts/1` | `artifacts.FORMAT` |
| CURRENT format | `mesa-clm/artifacts-current/1` | `artifacts.CURRENT_FORMAT` |
| audit sample format | `mesa-clm/audit-sample/1` | `audit.FORMAT` |
| lock directory | 8 hex characters of `framings.lock_sha()` | `artifacts.LOCK_SHA8` |
| version name | `v<N>`, N ≥ 1, next = max + 1 | `artifacts.VERSION_RE`, `ArtifactLayout.next_version` |
| export | `<out-dir>/<date>/artifacts_v<N>/<clm_model_fp>/` | `artifacts.EXPORT_DIR` |
| promotion acc margin | 0.01 | `artifacts.PROMOTE_ACC_MARGIN` |
| promotable K2 verdicts | {a, b} | `artifacts.PROMOTABLE_VERDICTS` |
| promotion rule R | B 2000, seed 0, α 0.05 (the registration's) | `registered.current()` via `promote` |
| spec → model | lowdim.v1, pair512.v1, choice.state.v1 → clm-latest; pair4096.v1, joint4096@S1, joint4096@S1ns, choice.raw.v1 → clm-raw | `tiered.SPEC_MODELS` (= `learn.probe.SPECS`) |
| spec dims | 2; 1025; 4097; 4096; 512 + K; 4096 + K | `tiered.spec_dim` (= `SpecInfo.dim`) |
| vector cache | 2048 texts (LRU) | `tiered.VECTOR_CACHE_MAX` |
| probe logit clip | 1e-12 | `tiered.PROBE_LOGIT_CLIP` |
| feature_spec levels | {probe, head} | `providers.base.FEATURE_SPEC_LEVELS` |
| citation: nested folds | n_folds ≥ 5 | `policy.MIN_NESTED_FOLDS` |
| citation: fold agreement | ≥ 5 evaluated folds | `policy.FOLD_AGREEMENT_MIN` |
| citation: guards | n_neg ≥ 30 (rank_fit), n_nonmodal ≥ 30 (choice) | `policy.GUARD_MIN` |
| citation: snapshots | `bench/snapshots/*.parquet` under results_root, hashed | `policy.SNAPSHOTS_DIR` |
| results root default | checkout root from `src/`, else `~/.mesa/clm/results` | `policy.default_results_root`, `DEFAULT_RESULTS_HOME` |
| citation: tier | cell tier == record level | `CellCitationValidator` |
| audit pass rule | n ≥ 50, n_cards ≥ 3, cp95_upper ≤ 2 × risk | `provenance.models.AUDIT_MIN_N`, `AUDIT_MIN_CARDS`, `AUDIT_RISK_FACTOR` |
| audit sample | n 100, ≥ 5 cards, seed 0, 3 equal strata, top decile 0.90 | `audit.DEFAULT_N`, `DEFAULT_MIN_CARDS`, `SAMPLE_SEED`, `STRATA`, `TOP_DECILE` |
| audit label weight | curator 1.0, fold_eligible false, bench_card false | `audit.labels_for_verdict`, `learn.labels.WEIGHTS` |
