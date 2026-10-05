---
title: "Learning and bench"
description: "Where mesa-clm's labels come from and how they are identified and frozen, the feature cache, the tier fitters, leave-one-card-out with nested selection, rule R, the lookup baseline every cell must beat, the pre-registered experiments X1-X4 with the AnyJev baseline numbers, how M2's analysis plan runs X1, the tier cells and X2, the registered M2 results (term.fits K1; F9 for column.ontology_fits), and how M4's analysis plan runs the probe tier (X3), K2, the teacher ablation (X4) and the artifacts, committed before any M4 result exists."
type: Guide
tags:
  - concepts
  - learning
  - bench
  - labels
  - calibration
generated:
  by: "claude/fable-5.1"
  at: "2026-10-04T23:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md)"
    author: "team:idss-mesa"
  - id: anyjev-bench
    resource: "https://github.com/idss-mesa/mesa-anyjev/blob/main/bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json"
    title: "mesa-anyjev local bench results, 2026-09-25 (Qwen/Qwen3-8B, raw/L0/L1/L2, leave-one-card-out)"
    author: "team:idss-mesa"
  - id: m0-baselines
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/bench/results/2026-09-29/baselines.json"
    title: "mesa-clm M0 no-model controls (lookup_prob, novel-key, leave-one-product-out), 2026-09-29"
    author: "team:idss-mesa"
  - id: m0-mde
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/bench/results/2026-09-29/mde.json"
    title: "mesa-clm M0 minimum detectable effect under rule R, 2026-09-29"
    author: "team:idss-mesa"
  - id: anyjev-l2-items
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/bench/baselines/anyjev_l2_2026-09-29.json"
    title: "mesa-anyjev per-item held-out L2 predictions (X2 baseline), 2026-09-29"
    author: "team:idss-mesa"
  - id: features-build
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/bench/results/2026-10-01/features_build.json"
    title: "mesa-clm feature store build under encoder_fp c3b3d5e1a283, 2026-10-01"
    author: "team:idss-mesa"
  - id: m2-plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/m2-analysis-plan.md"
    title: "mesa-clm M2 analysis plan (pre-run): X1, the zero_shot and calibrated tiers, X2"
    author: "team:idss-mesa"
  - id: m2-results
    resource: "https://github.com/idss-mesa/mesa-clm/tree/main/bench/results/2026-10-03"
    title: "mesa-clm M2 registered run, 2026-10-03 (x1.json, tiers.json, x2.json, table.md)"
    author: "team:idss-mesa"
  - id: m4-plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/m4-analysis-plan.md"
    title: "mesa-clm M4 analysis plan (pre-run): X3 the probe tier, K2, X4, artifacts and promotion, the citation test, audits"
    author: "team:idss-mesa"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# Learning and bench

## Labels

A label is identified by `(task_key, target_sha256, option_key, label_source)` (DESIGN D1), so
rotating a framing never orphans it; `option_key` is empty for closed choices. Each carries a
weight and the flags `fold_eligible`, `bench_card`, `product_code` and `leak_group`:

| Source | Weight | Fold-eligible | Producer |
|---|---|---|---|
| `curator` / `curator_implicit` | 1.0 / 0.7 | yes, except on bench cards | a pick in the MRTR elicitation or the CLI at an interactive terminal (`review`, `review --pick`, `feedback`), or a ticked neon `review.md` (DESIGN D21, A2) |
| `agent_pick` | 0 | no | a plain `mesa_clm_feedback` tool call, the same CLI verbs run without a terminal, and mesa-anyjev curator rows imported without `--trust-curator`; recorded, never learned from |
| `consensus_all` / `_majority` / `_negative` | 0.8 / 0.6 / 0.5 | yes | `labels ingest-neon-eval`: a term proposed by every, by at least two, or by a single one of the four agentic models in neon-avu-eval |
| `teacher` / `teacher_implicit` | 0.5 / 0.3 | **no** | `labels ingest-teacher` (M4): accepted items of neon-ducklake's Opus-validated generic curation, in-registry and aspect-mapped only, Yes only, one row per table card of the product that carries the column, leak group = product (DESIGN D19; `teacher_implicit` would weight the unpicked candidates of replicate proposal files, of which the corpus has none) |
| `gold` | 1.0 | yes | reserved; no hand-verified rows exist yet |

The silver is *agreement between models*, not truth, and Opus is both a silver labeler and the
teacher, which is why teacher labels never reach a test fold and X4 ablates them. Weights are
`sample_weight` in every fitter, including LDA and ridge (DESIGN D20). The bench never reads
the live label table: `labels snapshot` writes a frozen Parquet whose `labels_sha256` every
cell and manifest records, and curator labels on bench cards are excluded from pre-registered
cells (DESIGN D30). The bench cards are a fixed list (the seven neon-avu-eval tables), and the
bench drops such rows *before* it picks the highest-weight label of each target, so a curator
answer can neither change a pre-registered item's label nor remove the item. M0 ships `labels ingest-neon-eval|import-anyjev|snapshot|stats`; the
fixture ingestion yields 934 rows (term.fits 285: 86 Yes / 199 No), and the 303 valid (card,
CURIE) pairs of `validated.json` become 285 rows because five GAZ CURIEs come back from EBI OLS
as root terms (18 pairs, listed in `terms_missing`; RESEARCH.md).

## The feature cache (milestone M1)

Every text X1 and X2 embed is cached once per encoder fingerprint, at
`~/.mesa/clm/features/<encoder_fp>/features.duckdb` (owner-only; `learn/features.py`, DESIGN
D5): the text with its exact token count, the L2-normalised vector as **float32** with the
cosine of its float16 copy (the export's gate is 0.9999), the serving lock it was embedded under,
and the 512-d projections of the pinned head with the head they came from. The store records the
serving lock's vector recipe (image digest, `vllm serve` arguments, environment, truncation cap)
when it is created and refuses another `encoder_fp`, dimension, head, format or vector recipe.
`features build --snapshot` checks the running encoder container against the lock and renders
the texts from a frozen snapshot without reading a label: the F1
control (the mesa-anyjev per-candidate state with the task question), the F4, F7 and F9 rank_fit
requests (one per target, over its labelled candidates and the anchor) and the PR #13 replica
specs `joint4096@S1` (the per-candidate state with the question appended) and `@S1ns` (without
it), each state put back in the builders' key order first, since the snapshot's sorted-key JSON is
identity only. `features export-npz` writes CLM's `TextCache` file for `finetune.py
--embed-cache` (M7; the only place vectors are float16), and `learn/offline.py` scores rank_fit
and noul questions from the cached float32 vectors with clm-serve's maths (X1 scores offline).
The scores reproduce clm-serve only as closely as the vectors do: from float16 vectors the
pre-registered cross-check (≤ 1e-4 in probability against `/v1/systemone`, 200 groups) failed at
1.31e-3 and 3.59e-3 (`bench/results/2026-10-01/features_build.json#/rerun/crosscheck`), from
float32 vectors it passes at 5.2e-6 (`clm-latest`) and 7.47e-5 (`clm-raw`), and `/v1/rank`
agrees exactly (`bench/results/2026-10-01/x1_crosscheck.json`; `scripts/x1_crosscheck.py`).

The live build of the 2026-09-29 snapshot under `encoder_fp` c3b3d5e1a283 holds 1,563 distinct
texts from 5,018 manifest rows (1,400 state side, 163 action side), none over the window,
620,336 encoder tokens, built in 195.2 s one text per request, with a float16 round-trip minimum
cosine of 0.9999999 (`bench/results/2026-10-01/features_build.json`); rebuilt with float32
vectors it took 223.7 s, and every float16 cast of a new vector equals the first build's bit for
bit (`bench/results/2026-10-01/features_build_m1c.json`). On the same texts
rendered the pipeline's way, the label-free collapse diagnostic gives a mean pairwise cosine
between different term.fits contexts of 0.831 raw and 0.677 after the head with the state alone
(F7), and 0.993 and 0.965 with the question appended
(`bench/results/2026-10-01/collapse_spike.json`; report-only, RESEARCH.md).

## Tiers and fitters (milestones M2, M4, M7)

| Tier | rank_fit | choice | Artifact |
|---|---|---|---|
| `zero_shot` | `σ(s_c)` from `clm-latest` or `clm-raw` | softmax | none |
| `calibrated` | weighted Platt on `s_c` (`a < 0` flags `inverted`) | temperature (K>2), Platt on the logit difference (K=2) | `calibrators.json` |
| `probe` (M4) | weighted logistic regression, LDA or ridge on `lowdim.v1`, `pair512.v1`, `pair4096.v1`, `joint4096@S1` or `joint4096@S1ns`; spec, fitter and hyperparameters by grouped inner leave-one-card-out, then a weighted out-of-fold Platt on the logit difference | the same fitters on `choice.state.v1` or `choice.raw.v1`; an out-of-fold temperature at K > 2 (Platt at K = 2) | `probes/<question_key>.json` |
| `head` (M7) | per-candidate two-way rows through CLM's `finetune.py` in the serve venv, re-scored with `HeadPair`, then Platt | closed options | `heads/mesa-<task>-<qk8>-v<N>.pt` |

Artifacts live under `~/.mesa/clm/artifacts/<encoder_fp>/<clm_model_fp>/<lock8>/v<N>/`,
immutable, with a strict loader that refuses fingerprint mismatches; `CURRENT.json` beside the
versions says which version and tier each task serves, and only what it promotes is served.
`learn fit` writes a version and never promotes; `learn promote` puts one task's artifact into
`CURRENT.json` only when it cites a qualifying pre-registered nested cell, K2 judged that tier
`a` or `b`, and, when the task already serves one, the new cell beats the current on NLL under
rule R and is within 0.01 on accuracy; a head is promoted only if it beats the probe (DESIGN
D18, M7). Serving never fits (DESIGN D15). The M4 reading is below and on
[Serving](serving.md).

## Leave-one-card-out, nested selection, rule R

Folds are one per card over the fold-eligible cards; each needs 30 training and 5 held-out
labels per class, else it is skipped and reported; predictions are pooled, never fold-averaged.
Every choice of framing, model, anchor, feature spec or hyperparameter is re-made inside each
outer fold by grouped inner cross-validation on the six training cards (`selection: "nested"`,
DESIGN D27); a choice made on all seven cards is `exploratory: true` and cannot be cited.
Teacher rows of the held-out product never train that fold; a property test proves it.

**Rule R** ("A beats B on m"): the one-sided 95% lower bound of a card-cluster paired bootstrap
of the difference (resample held-out cards, B=2000, seed 0) is above zero **and** A beats B on
at least ⌈0.8·m_c⌉ of the held-out cards with at least 10 evaluable items. Effect floors are
always paired with a confidence-interval condition. `bench mde` simulates the minimum
detectable effect at the realised n and is committed with the pre-registration.

## The lookup baseline

A no-model lookup that copies the label of the same (scope, column, CURIE) from the other cards
already scores leave-one-card-out accuracy 0.772 on `term.fits` and 0.800 on
`column.ontology_fits`, above AnyJev L2's 0.765 on `term.fits` (reproduced by `mesa-clm bench
baselines`: `bench/results/2026-09-29/baselines.json`, cells `neon_term_fits.baseline.lookup_prob`
and `neon_ontology_fits.baseline.lookup_prob`; leave-one-product-out 0.723 / 0.721; ties between
training cards go to the alphabetically first card, RESEARCH.md). So every citable cell must
report `lookup_prob` (Laplace-smoothed key frequency over the training cards, falling back to
the training prior on unseen keys), the novel-key subset (159 items for term.fits, 99 for
ontology_fits) and leave-one-product-out, and **`beats_lookup_novel`** — novel-key AUROC
cluster lower bound above 0.5 and novel-key NLL beating `lookup_prob` under rule R — gates
every `auto` threshold (DESIGN D8).

## Pre-registered experiments (frozen at G1; full text in DESIGN.md)

* **X1 framing A/B**: {F1, F4, F7, F9} × {`clm-latest`, `clm-raw`} × {term.fits,
  ontology_fits}, scored offline from the feature cache with shuffle and candidate-only
  controls; a fixed decision rule picks the production framing A1 (M2). The rule's sixth step,
  an anchor variant chosen by NLL, was withdrawn before the freeze: no variant text was written
  down before the live stack had been looked at on a bench card, so X1 runs with the registry
  anchors only (DESIGN, "G1 freeze" and the implementation note "The X1 anchor variant").
* **X2 baselines**: majority, `lookup_prob`, novel-key, leave-one-product-out, the PR #13
  logistic-regression replica, and per-item AnyJev L2 predictions (dumped in M1, below), which
  make the K2(a) comparison paired.
* **X3 tier sweep**: zero-shot and calibrated on A1; probe specs and fitters; the head (M7).
* **X4 teacher ablation**: off vs (0.5/0.3) vs (0.3/0.1), scored on silver and
  silver-minus-Opus; with few teacher rows in the bench products' registry the expected effect
  is stated in advance as about null (D19's "13 usable items" described the SRER generics of
  2026-09-29 and is superseded by the corpus M4 reads, below).

Kill and pivot criteria K0–K4 are fixed with them: no state signal makes a task's CLM tiers
audit-only and its proposals `ols_rank`; a task earns auto-eligibility only by beating the
lookup on novel keys and matching AnyJev L2 in a paired test with ECE at most 0.08.

## Milestone M2: the analysis plan (pre-run)

`design/m2-analysis-plan.md` reads every frozen rule M2 runs as one deterministic algorithm,
names each alternative reading with the reason it was rejected (the more conservative one wins),
and lists every constant; it and its code were committed and pushed **before** the first run on
the snapshot's labels, so the commit shows every choice preceded the results (DESIGN,
"Implementation notes (M2)", with the pre-run disclosure of every label-free look). After the
first run it changes only by amendment, with the affected cells marked exploratory.

**Inputs.** Only the registered snapshot `bench/snapshots/2026-09-29.parquet`: `bench framing`,
`bench run` and `bench x2` refuse any other file by its sha256 and its label content, never write a
snapshot, load its rows into a scratch label store (the per-host sidecar is never read) and check
every task's items, class counts, items per card and `min_weight` against the counts M0
published, before any statistic. Every run records its configuration; anything but the
registered one (both X1 tasks, F1/F4/F7/F9 on `clm-latest` and `clm-raw` with every one of the 16
arms scored, the full run and the nesting, B = 2000, seed 0, α = 0.05, 200 shuffle draws; the
framings lock of G1 and each model's fingerprint under the serving lock of G1; for X2 the
registered AnyJev dump) is written with every cell `pre_registered: false`, `exploratory: true`,
and no verb takes an option for B, the seed or α. The producers check this themselves, so a library
call under another lock, fingerprint or dump is unregistered too.

### How X1 decides

Each arm (a framing on a model) scores the same items offline from the feature cache:
`s = scale·(zs·zc − zs·za)` for F4, F7 and F9, the noul logit difference for the F1 control.
**Rule (1)** qualifies an F4/F7/F9 arm when the AUROC of `s` has a card-cluster lower bound above
0.5 and is at least 0.60, and its ΔAUROC against the **within-card state shuffle** passes rule R:
the shuffle gives every target the context of another target of its own card, the comparator is
the mean AUROC of 200 seeded derangements, and both rule R conditions are computed on that
difference (the drafts' AUROC of an averaged score was biased against every arm and was replaced
before any run). If nothing qualifies on either model the outcome is **K1** (the task's CLM tiers
become audit-only, proposals use `ols_rank`, M4 goes probe-first). Otherwise **rule (2)** ranks
the qualified arms of `clm-latest` (of `clm-raw` when `clm-latest` has none) by the pooled NLL of
leave-one-card-out Platt (weighted, 30/5 guards, no fit on fewer than 100 items): an F7 or F9
winner stands; an F4 winner yields to a more cacheable qualified arm it does not beat under rule
R, and between F7 and F9 the lower NLL wins only if it beats the other under rule R, else the one
with fewer encoder tokens per target, then F7. **Rule (3)** keeps `clm-latest` unless `clm-raw`
beats it on NLL at that framing; **rule (4)** only recommends an amendment reopening D3 when F1
beats every eligible arm, never adopts it; rule (6) is withdrawn. The learned candidate-only probe
(the PR #13 recipe on the candidate vectors), the mean-context score, the collapse diagnostic,
calls and tokens per target and the p50 latency of a label-free timing run are reported, never
read by a rule. `bench framing --decide --from x1.json` replays every decision from the JSON
(each rule R verdict against its own numbers and the run's bootstrap settings, the arm records,
cells and fold choices against the traces, each cell's metrics against its own per-item
predictions) and recomputes the decisions, the arm records, the `p_loco` column, the cells'
predictions and the probe's statistics from `x1_items.parquet`; the arm cells' lookup controls and
`beats_lookup_novel` need the items' states and are report-only. "Cluster-LB" is the one-sided
95% bound of the card bootstrap, so every interval `[lower, upper]` the reports print is a 90%
interval, and they say so. The timing run asks each drawn target under both models back to back,
alternating which goes first, and flags each ask that sent a text the run had not sent before
(clm-serve's embedder cache is shared by both models).

### Nested selection

For each held-out card the whole decision is made again on the other six cards only (their
shuffles, their folds, their bootstraps); `fold_choices` records each fold's arm. A fold whose
inner outcome is K1 or undecidable chooses no arm and is skipped in both tiers. The full-data
choice is the production choice A1 (an amendment records it); a citable cell needs at least 5 of 7
folds to agree with A1 on framing **and** model.

### The tier cells and X2

`bench run --tiers zero_shot,calibrated --loco` writes `tiers.json`, the one producer of the
citable-form `<task>.<tier>.<A1>` cells: fold *c* is scored (and at `calibrated` fitted) with fold
*c*'s own arm, X1's outcome taken only from a registered `x1.json` that replays, recomputes and
names the same labels, framings lock and model fingerprints; each fold's `fold_choices` entry keeps
X1's pointer to its inner trace (`x1_trace`) and the file's note names `x1.json` by sha256. `zero_shot` is `σ(s_c)` or CLM's softmax;
`calibrated` is weighted Platt with hard targets on `s_c` or the logit difference, temperature
(T in [1e-4, 1e4]) at K > 2, with the unweighted fit reported beside it. `…@full` (A1 in every
fold) is exploratory; without an A1 the default F7@`clm-latest` is reported for audit only; the
closed choices are F7@`clm-latest` (not citable in M2) with `@clm-raw` reported. Every cell carries
the frozen fields, the lookup controls on its own folds, `beats_lookup_novel` with both of its
conditions and the reason when it fails, and its per-item predictions. `bench x2` writes the M0
controls (with a note on whether they equal the published M0 cells), the PR #13 replica (unweighted,
float32 vectors) and AnyJev L2 joined by D1 identity; X2 gates nothing.

## Milestone M2: the registered results (2026-10-03)

The run protocol (plan §14) ran once on the serving host, each step once, after the plan and its
code were pushed: the label-free timing run, `bench framing --decide`, the replay, `bench run`,
`bench x2`, `bench table`. Every output was committed as produced under
`bench/results/2026-10-03/` (`x1_latency.json`, `x1.json`, `x1_items.parquet`, `x1.md`,
`tiers.json`, `tiers.md`, `x2.json`, `x2.md`, `table.md`); the run is the registered one
(`x1.json#/x1/registered`), and `bench framing --decide --from` replays and recomputes it from the
files. DESIGN.md records the outcome as amendment A1 and RESEARCH.md ("M2 results") holds every
number with its JSON pointer; the main ones follow. Intervals are the pair of one-sided 95% card
bootstrap bounds (a 90% interval); silver labels are agreement between models, not truth.

**X1** (`x1.json#/x1/tasks/<task>/arms/<arm>`; AUROC of the raw score, rule (1) needs a lower
bound above 0.5, AUROC at least 0.60 and a ΔAUROC over the within-card shuffle that passes rule R):

| arm | term.fits AUROC | rule (1) | ontology_fits AUROC | rule (1) | ontology_fits LOCO-Platt NLL |
|---|---|---|---|---|---|
| F4@clm-latest | 0.497 [0.406, 0.572] | fails | 0.611 [0.529, 0.677] | qualifies | 0.666 |
| F4@clm-raw | 0.458 [0.413, 0.505] | fails | 0.501 [0.399, 0.614] | fails | 0.693 |
| F7@clm-latest | 0.511 [0.482, 0.544] | fails | 0.693 [0.611, 0.762] | qualifies | 0.633 |
| F7@clm-raw | 0.494 [0.428, 0.549] | fails | 0.592 [0.546, 0.637] | fails | 0.688 |
| F9@clm-latest | 0.557 [0.512, 0.596] | fails | 0.751 [0.697, 0.805] | qualifies | 0.602 |
| F9@clm-raw | 0.539 [0.495, 0.592] | fails | 0.745 [0.700, 0.804] | qualifies | 0.618 |

* **term.fits: K1.** No arm qualifies on either model (every one is under the 0.60 floor), and
  every nested fold's inner outcome is K1 too (`x1.json#/x1/tasks/neon_term_fits/decision`,
  `nested/fold_choices`). As the pre-registered K1 says, its zero_shot and calibrated tiers are
  audit-only, its proposals use `ols_rank`, and M4 goes probe-first.
* **column.ontology_fits: F9 on `clm-latest`.** Rule (2) ranks the qualified `clm-latest` arms by
  NLL (F9 lowest), rule (3) keeps `clm-latest` (`clm-raw` does not beat it at F9 under rule R),
  rule (4) does not fire (no D3 amendment), and all 7 nested folds choose F9@`clm-latest`
  (`x1.json#/x1/tasks/neon_ontology_fits/decision`, `a1`, `nested/fold_choices`).
* The candidate-only probe (report-only, the PR #13 recipe on the candidate vectors alone) reaches
  AUROC 0.808 [0.765, 0.858] on term.fits and 0.816 [0.758, 0.873] on ontology_fits
  (`x1.json#/x1/tasks/<task>/candidate_probe`).

**The tier cells** (`tiers.json`). The citable-form nested cells of column.ontology_fits
(`neon_ontology_fits.<tier>.F9`, 7 of 7 folds agreeing with A1): zero_shot accuracy 0.489, NLL
2.624, ECE 0.464, AUROC 0.751 [0.697, 0.805]; calibrated accuracy 0.642, NLL 0.602, ECE 0.123,
AUROC 0.732 [0.673, 0.802]. Neither has `beats_lookup_novel`: on the 99 novel-key items the
AUROC lower bound passes (0.597 and 0.584) but the NLL comparison with `lookup_prob` (NLL 0.653)
does not pass rule R (zero shot NLL 2.846; calibrated NLL 0.594, the bootstrap lower bound of the
difference −0.046), and neither cell has a
Clopper–Pearson threshold at 5% or 10% risk (`baselines/beats_detail`, `metrics/threshold_cp`).
The term.fits audit cells (F7@`clm-latest`, not pre-registered under K1): zero_shot accuracy 0.495,
AUROC 0.511; calibrated accuracy 0.698 (majority 0.698), AUROC 0.457. The closed choices
(F7@`clm-latest`) score below their majority rates at zero shot: column.annotate 0.357 against
0.643, column.aspect 0.050 against 0.300, avu.value_kind 0.324 against 0.478 (calibrated 0.324);
the calibrated cells of annotate and aspect are empty because no fold trains on 100 items. After
seeing these cells the user decided that serving answers the closed choices by deterministic rules
and keeps CLM's answers to them for audit only (amendment A6, 2026-10-04; see
[Decision model](decision-model.md)). That is a product-safety change, not a result: no
pre-registered rule covers these tasks, no cell changes or is re-read, and the bench never counts
the audit-only records (it reads label snapshots only). The aspect rule's lookup is M0's lookup
over the registered snapshot's 60 `column.aspect` items, frozen into the package
(`aspect_lookup.json`, rebuilt from the snapshot by a test); a run leaves out the annotated
card's own items, so a leave-one-card-out end-to-end bench stays leave-one-card-out.

**X2** (`x2.json`, baselines that gate nothing):

| cell | acc | AUROC | NLL | ECE | beats_lookup_novel |
|---|---|---|---|---|---|
| `neon_term_fits.baseline.pr13@S1` | 0.723 | 0.709 [0.643, 0.771] | 1.460 | 0.215 | false |
| `neon_term_fits.baseline.pr13@S1ns` | 0.747 | 0.786 [0.738, 0.839] | 1.019 | 0.184 | false |
| `neon_term_fits.baseline.anyjev_l2` | 0.765 | 0.782 [0.743, 0.833] | 0.499 | 0.058 | true |
| `neon_ontology_fits.baseline.pr13@S1` | 0.821 | 0.884 [0.842, 0.927] | 0.791 | 0.131 | false |
| `neon_ontology_fits.baseline.pr13@S1ns` | 0.821 | 0.905 [0.869, 0.941] | 0.568 | 0.137 | false |
| `neon_ontology_fits.baseline.anyjev_l2` | 0.837 | 0.908 [0.884, 0.940] | 0.375 | 0.078 | true |

The recomputed no-model controls equal M0's cells field for field (`x2.json#/notes/2`), and the
AnyJev L2 cells join all 285 and 190 items. Two independent recomputations after the run agree with
every number they recomputed (RESEARCH.md, which also names what the repository reproduces itself:
the replay, and X1's decisions under 200 other bootstrap seeds). What A1 changes in the pipeline is on
[Decision model](decision-model.md): Q3 asks F9, term.fits groups are `ols_rank` by default.

## Milestone M4: the analysis plan (pre-run)

`design/m4-analysis-plan.md` reads every frozen rule M4 runs (X3's probe tier, K2, X4, the
citation test, artifacts and promotion, the production audit) the way the M2 plan read X1: one
deterministic algorithm per rule, every alternative reading named with the reason it was
rejected (the more conservative one wins: the one that admits less, cites less or promotes
less), every constant in its Appendix B and held to the code by
`tests/unit/test_m4_plan_constants.py`. It and its code are committed and pushed **before the
first M4 run on the snapshot's labels**; until then the code ran on synthetic data only and
nothing combined a model output with a silver label. This section says what the verbs do as
the plan reads them; the registered run and its outcome are under "M4 results" at the end of
it. After the first run the plan changes only by amendment, with the affected cells marked
exploratory; the amendment that records K2's verdicts, X4's decision and the promotions (A7) is
the planned consequence of the run, not such a change.

**Inputs and the M4 registration.** The snapshot is M2's, unchanged (`bench/snapshots/2026-09-29.parquet`,
read through M2's refusals: both hashes, the published counts, never written); the folds, guards
and weights are M2's. The feature store is the registered encoder's under the serving lock of
G1, holding every registered text under F4, F7, F9, F1 and the X2 joint specs; X3 reads from it
only, X4 adds the teacher items' texts (built label-free before the run). The M4 registration
(`bench.registered.REGISTERED_M4`) adds to M2's, which it never repeats: the framings lock of
A1 (M2's registration keeps G1's so its replay still holds) with the active framing per task
(F9 for `column.ontology_fits`, F7 elsewhere), the X3 grid (the specs per shape, each spec's
model, the fitters with their hyperparameter values in declared order, the inner criterion, the
floors), the committed `tiers.json` and `x2.json` K2 reads by sha256 (other bytes are refused
outright), the K2 constants, X4's arms, and the teacher inputs, filled from the label-free
ingest run before the pre-run commit (plan §12.2; DESIGN "Implementation notes (M4)", item 6):
the corpus content hash `d90387928bed…` as read at neon-ducklake commit `b1fa52a8d3ff…`, the
teacher snapshot `bench/snapshots/2026-10-04-teacher.parquet` (sha256 `397f98d4b28c…`, content
`deb0dc46e17b…`) and the silver-minus-Opus snapshot `bench/snapshots/2026-10-04-minus-opus.parquet`
(sha256 `2cd529ffa439…`, content `d8e26a0c7ac3…`); no pin is a placeholder. Every producer
applies it itself: `bench x3`, `bench k2` and `bench x4` write a run on other labels, another
lock, other fingerprints, another grid, another active framing, a subset of tasks or other
bootstrap settings with every cell `pre_registered: false`, `exploratory: true` and the
deviations in the file's first note, whatever the caller says; `learn fit` computes the same
lock, fingerprint and grid deviations and **refuses** on any (exit 1, nothing written), since an
artifact is what production serves and an unregistered one has no cell to cite; a pinned input
whose bytes differ is refused everywhere. The verbs expose no option for B, the seed, alpha,
the grid, the framings or the models.

### The probe tier

Three readings settle X3's shape. The **framing is not an axis**: a task's probe is fitted and
served on the texts of its active framing after A1, since which framing a task asks is X1's
decision and a probe chosen on another framing's texts could not be served without a lock
rotation. **A spec belongs to one model**: a spec that reads any head quantity is a
`clm-latest` probe, one that reads raw vectors only is a `clm-raw` probe, and a served probe
lives in the artifact bundle of its spec's model. **The floors are per fit**: 40 training items
for every probe fit, outer and inner, and 100 out-of-fold items for the pool the probe's
calibrator is fitted on; a fold below either is skipped with the reason recorded.

**Specs** (`learn.probe.SPECS`, in declared order; `zs`, `zc` the 512-d `clm-latest`
projections, `xs`, `xc` the raw L2-normalised 4096-d vectors, `s_latest` and `s_raw` the
zero-shot `s_c` under each model, `⊕` concatenation, `⊙` the elementwise product):

| spec | shape | model | formula |
|---|---|---|---|
| `lowdim.v1` | rank_fit | clm-latest | `[s_latest, s_raw]` |
| `pair512.v1` | rank_fit | clm-latest | `zs⊙zc ⊕ |zs−zc| ⊕ [s_latest]` |
| `pair4096.v1` | rank_fit | clm-raw | `xs⊙xc ⊕ [s_raw]` |
| `joint4096@S1` / `@S1ns` | rank_fit | clm-raw | the raw vector of the X2 joint context text, with / without the task question |
| `choice.state.v1` | choice | clm-latest | `zs ⊕` the K zero-shot option logits under clm-latest |
| `choice.raw.v1` | choice | clm-raw | `xs ⊕` the K option logits under clm-raw |

A rank_fit spec has one row per labelled (target, candidate) pair, a choice spec one per
labelled target; the anchor text is read only to form `s_c`. The formulas live in one place
(`probe.rank_fit_features`, `probe.choice_features`) for the bench and for serving, and the
builder's zero-shot logits equal the tier cells' scores.

**Fitters** (`learn.linear`): weighted L2 logistic regression (binary at K = 2, the symmetric
multinomial form above), weighted shrinkage LDA (toward the scaled identity, the Ledoit–Wolf
target) and weighted ridge on one-hot targets, in numpy float64, the label weights as sample
weights (DESIGN D20: an integer weight equals that many copies of a row). Every data term is a
**weighted mean** with the penalty `λ/2 ‖coef‖²` on the standardized features and never on an
intercept (the convention of the M2 Platt fit; the sum form was rejected because it would make
a 4097-d fit at the smallest λ nearly unpenalised and dependent on the scale of the weights).
Features are standardized with the training set's weighted moments (std floored at 1e-8); a
thin-SVD reduction to the row span makes a wide fit exact and cheap. Grids in declared order,
which is the tie order: logreg λ ∈ {1e-2, 1e-1, 1, 10}; lda γ ∈ {0.1, 0.5, 0.9}; ridge λ ∈
{1e-1, 1, 10}; fitters `logreg, lda, ridge`. Newton's method with Armijo halving; a fit that
does not converge is used as it is and flagged `converged: false`, never refitted or skipped
(the M2 calibrators' rule). LDA with a class absent from a training set is refused and that
configuration is recorded unfittable on that fold.

**The nested inner selection** (`probe.inner_select`, DESIGN D27). For an outer fold (held-out
card *c*, training cards *T*) the inner folds are the grouped leave-one-card-out over the cards
of *T*; nothing in the inner loop reads an item outside *T*. Each inner fold applies the 30/5
guard and the probe floor of 40 to the task's items only (teacher rows count toward neither).
Every configuration of the grid (spec × fitter × hyperparameter, in declared order) is fitted
on every passing inner fold and applied to the inner held-out card; the out-of-fold
predictions are pooled and the criterion is the **pooled OOF NLL of the uncalibrated probe
probabilities, unweighted over the items** (the quantity the cell is judged on; the weights
are sample weights of the fits). The lowest wins; a tie goes to the first in declared order. A
configuration unfittable on any passing inner fold is not selectable on that outer fold; no
selectable configuration, or no passing inner fold, skips the outer fold with the reason.

**The OOF calibrator and the outer fold** (`probe.nested_probe`). After the outer guard and
floor, fewer than 100 pooled OOF items of the winner skip the fold (`below_floor n < 100 OOF
items (probe calibrator)`): no probe calibrator is ever fitted on fewer. Otherwise the
calibrator is fitted **weighted** on the winner's pooled OOF predictions: at K = 2 a Platt on
the OOF logit difference (`a < 0` flagged `inverted` and kept), at K > 2 a temperature on the
OOF log-probabilities (a bound flagged and kept); the cell's `calibration` is its kind. The
winner is refitted on all of *T*, applied to card *c*, the calibrator applied, and the held-out
predictions pooled in item order, never averaged; `ProbeArtifact.predict` on a fold's fit
equals the pooled held-out probabilities bit for bit. Stated in advance from the published
counts (`bench/results/2026-09-29/baselines.json#per_fold_n`): `column.annotate` (98 items) and
`column.aspect` (60) can never pool 100 OOF items, so their probe cells have no evaluated fold
and `metrics: null`, with the uncalibrated probe reported in their diagnostics only; with
per-card counts of 13, 11, 9, 9, 7, 6 and 5, `column.aspect`'s inner training sets also fall to
36 or 38 items on the six inner folds where the 13-item card and an 11- or 9-item card are both
held out, which the probe floor of 40 skips. `avu.value_kind`, `term.fits` and
`column.ontology_fits` clear the 40 floor on every outer and inner training set outright, and
the 100 OOF floor on every outer fold only if the inner 30/5 guards pass (the per-card class
counts are not published, so that half is not stated in advance). Under A6 the closed choices
are decided by rules, so none of this changes production.

**Teacher rows** (`probe.TeacherRows`, DESIGN D19) are added to every training set, outer and
inner, except the rows whose `leak_group` equals the held-out card's product; they are never in
a test index, never counted by a guard or floor, and every fold records `n_teacher`. In X3
there are none; X4 adds them.

### The X3 cells (`bench x3`)

Four cells per task, every one through the same cell builder as the M2 cells (`cells.assemble_cell`:
the identity fields, counts, metrics, lookup controls, `beats_lookup_novel`, `threshold_cp`
and the per-item predictions): the **nested cell** `<task>.probe.<active framing>` (the whole
grid, `selection: "nested"`, the only citable form), `@latest` and `@raw` (the same procedure
on the specs of one model; pre-registered, never citable because a variant; what K2's clm-raw
clause compares) and `@full` (one configuration chosen by the inner LOCO over all seven cards,
evaluated in every outer fold with that fold's own calibrator and refit; `selection: "full"`,
`exploratory: true`). The nested cell's identity (`feature_spec`, `model`,
`fingerprint.clm_model_fp`) is the **full-data** configuration's, the one production would
serve, as A1's arm is the identity of the M2 nested cells; `fold_choices` records each outer
fold's own configuration (spec, fitter, hyper, model, inner NLL, the calibrator record,
`n_teacher`, `converged`, `evaluated` or the skip reason) and `diagnostics.fold_agreement_with_full`
how many folds chose the served one. A nested cell whose folds chose specs of both models is
one cell, each item scored by its fold's own spec. Every spec is servable (both models are
served); a task whose full-data selection chooses nothing is `servable: false`. Report-only
diagnostics carry every configuration's inner NLL per fold, the uncalibrated held-out and
pooled metrics, the refit's convergence, each spec's `mean_state_cos`, calls, tokens and
milliseconds per decision in M2's units.

**What can be cited.** Only the nested cell, and only when the citation test passes
([Tiers and policy](tiers-and-policy.md)); the closed choices' probe cells cannot have
`beats_lookup_novel`, each for its own reason (`column.aspect` and `avu.value_kind` have no
AUROC at K > 2; `column.annotate`, K = 2, has one, but its lookup control is
`insufficient_clusters` under the frozen minimum-detectable-effect statement, one card having
at least 10 novel-key items, and its probe cell has no evaluated fold under the 100 OOF floor),
so no closed-choice probe cell can be cited in M4, stated in advance. `term.fits` can be cited for a numeric `auto` only through a promoted probe, since its
calibrated tier is K1's audit-only.

### K2 (`bench k2`)

K2 judges each task's **best servable tier on nested cells**. The candidates are the task's
citable-form nested cells: `calibrated` from the committed `tiers.json` (only
`column.ontology_fits` has one; `term.fits`' K1 audit cells and the closed choices' fixed F7
arms are listed with `eligible: false` and the reason) and `probe` from `x3.json`; a candidate
is eligible only when its own cell is `pre_registered: true`, nested, servable, not exploratory
and pools items. The best tier is the eligible candidate with the lowest pooled NLL (a tie goes
to `calibrated`, the lower rank); whether the probe beats the calibrated tier under rule R is
recorded beside it, because promotion needs it and K2 does not. On the best cell, with every
number recorded:

* **(a) auto-eligible**: `beats_lookup_novel` ∧ non-inferiority to AnyJev L2 on the full item
  set ∧ ECE ≤ 0.08 with cluster upper bound ≤ 0.12. The AnyJev side is `x2.json`'s per-item
  cell joined by identity; "the full set" is read strictly (the tier must pool every item of the
  task, counted from its `task_counts` or the registered count and failing closed when it has
  neither, and every one must match, so a tier that skipped a fold fails here); non-inferiority
  is two conditions, Δacc's card-cluster paired bootstrap lower bound above −0.02 **and** the
  card-sign condition, which is **rule R's sign test, verbatim and without a margin**: on at
  least ⌈0.8·m_c⌉ of the m_c cards with at least 10 items the per-card Δacc is above 0, and
  fewer than 4 such cards fails. The plan names and rejects the margin-shifted per-card test
  (Δacc above −0.02 per card) as the permissive reading; it is reported beside the gate
  (`card_sign_margin_report`, computed in integers), never gated on. Stated in advance: the
  literal test asks the tier to beat AnyJev L2 on most cards, so (a) is demanding, and a tier
  that is non-inferior on the cluster bound but not better per card is (b) at best; a numeric
  `auto` stays null in 0.1.0 whatever the verdict;
* **(b) proposer-only**: `beats_lookup_novel` alone;
* **(c)** otherwise, "no eligible candidate" included. A closed choice is (c) by construction
  (see "What can be cited" above for each one's reason).

Strict inequalities are strict. **The clm-raw clause** (`head_adds_nothing`, rank_fit tasks):
on the `@latest` and `@raw` cells, `@latest` does **not** beat `@raw` on NLL under rule R and
`@raw` is non-inferior to `@latest` on accuracy within 0.01; recorded, with no production
change in the run (the consequences are amendment A7's). `k2.json` holds the inputs with their
sha256, the constants, the candidates, the comparison, the best tier and its cite, every
condition and the verdict with its reason; `k2.md` is one row per task, and `bench table` lists
the verdicts beside the cells of the other files.

### X4 and the teacher corpus (`labels ingest-teacher`, `bench x4`)

**The corpus** (`<neon-root>/curation/generic/<DP>.validated.json`), ingested label-free on
2026-10-05 (01:44–01:46Z, before the pre-run commit; the report is transcribed in DESIGN
"Implementation notes (M4)", item 6) at neon-ducklake commit `b1fa52a8d3ff…`, corpus hash
`d90387928bed…`: 55 files, 389 `accepted` items, every file's `model` `claude-opus-5-5`, no
`human_accepted` list. DESIGN D19 as read: items with status `accepted` or `human_accepted`; a
file whose `model`, or whose replicate proposal file's `model`, starts with `mesa-clm` is
refused (none was). The ingest found 6 replicate proposal files under
`sites/SRER/curation/proposals`, read each for its `model` only, and wrote 0 `teacher_implicit`
rows: it synthesizes no row from a proposal's unpicked candidates, so X4's `teacher_implicit`
weights weight nothing in this run. Drops are counted by reason, and the ingest dropped 103
`out_of_registry` (a CURIE prefix that is not a registry ontology), 101 `no_column` (a
dataset-level item), 22 `unmapped_aspect` (`process`), 17 `collapsed_identity`, 11
`unresolved_column` (no table card of the product carries the column), 0 `ontology_mismatch`
and 0 `term_missing` (OLS has no usable record). **Target resolution**: a column resolves to
**every** table card of the product (`sites/SRER/cards/anyjev/<DP>.<table>.md`) whose column
set contains it, one row per card, the rows sharing `leak_group` = the product code; no
dataset-scope state is synthesized for an item without a resolvable column (the plan had left
that resolution to M4; this is the conservative answer); 231 (item, card) resolutions, and
`DP1.10092.001` has no card. Rows are Yes only: `term.fits` for (column state, CURIE) with the
candidate as OLS records it, and `column.ontology_fits` for (column state, ontology) when the
ontology is aspect-allowed (otherwise the ontology row is skipped and counted
`ontology_not_aspect_allowed` outside the drops, since the `term.fits` row is still written: 5
such); 435 rows inserted, 231 `term.fits` and 204 `column.ontology_fits`; every row `teacher`
at 0.5, `fold_eligible=false`, `bench_card=false`, origin `teacher:<file sha256[:12]>`. The OLS
records of the corpus's in-registry CURIEs were recorded once from live EMBL-EBI OLS4 into
`tests/fixtures/ols-teacher/` (110 `get_term` responses; `scripts/record_ols_teacher.py`), so
the ingest replays. `labels snapshot` then wrote `bench/snapshots/2026-10-04-teacher.parquet`
(1,369 labels), whose silver rows are byte-identical to the registered snapshot (the X4
producer checks the content digest of its non-teacher rows); `features build --snapshot
<teacher snapshot> --tasks term.fits,column.ontology_fits --framings F7,F9` embeds the teacher
texts and has not run yet.

**The silver-minus-Opus subset.** The committed snapshot does not carry the model set behind a
consensus row, so the subset comes from a label-free rebuild of the silver from
`tests/fixtures/neon-avu-eval` with Opus's runs removed (`labels ingest-neon-eval
--exclude-models claude-opus-5-5` into a separate store, then `labels snapshot --out
bench/snapshots/2026-10-04-minus-opus.parquet`, 679 labels, committed and pinned with the
pre-run commit). The surviving
items are the registered items whose identity the rebuilt task carries **with the same label**
(a weight change survives, a flipped label or a vanished identity does not); the scoring is the
same predictions on that subset.

**The arms.** Three runs of the X3 nested procedure on **identical outer folds**: off, (0.5, 0.3)
and (0.3, 0.1) for the (`teacher`, `teacher_implicit`) weights, the teacher rows entering every
training set as sample weights except the held-out card's product (both bench products have a
corpus file, so the rule fires on every fold), never a test item (`teacher_in_test: false`,
asserted). Cells `<task>.probe.<framing>@teacher-off|-0.5|-0.3` and their `-minus-opus` twins,
every one a variant, pre-registered and `exploratory: true`, never citable; their identity is
the off arm's full-data choice. **Decision**: keep teacher labels in production fits iff
`on ≻ off` on **novel-key NLL** under rule R for **both** scorings, for the (0.5, 0.3) arm; the
(0.3, 0.1) arm is reported; fewer than two novel-key cards counts as not passing. The record is
in every X4 cell's diagnostics and the file's notes.

### `bench e2e --loco` (planned, second wave)

The end-to-end measure is report-only and writes no cell: per held-out card, consensus-all
recall (the fraction of the card's `consensus_all` Yes `term.fits` CURIEs the run proposes,
against AnyJev's static recall of 0.25) and anchor-abstain precision (the fraction of the run's
anchor-won groups whose candidates include no consensus-all Yes), pooled with the card-cluster
bootstrap bounds. Its measurement half is in `bench/e2e.py`; the fold provider that serves a
fold's nested probe or calibrator from the store behind the decision-provider protocol is
**not in the pre-run commit**: `bench e2e --loco` exits with a usage error that says so, and
the measure ships by amendment with its own disclosure.

### The run protocol (plan §12)

Before the commit nothing combines a model output with a real label and no M4 verb runs on the
registered snapshot, the live feature store or the sidecar; the one step that reads labels, the
teacher inputs (`labels ingest-teacher`, `labels snapshot` for the teacher snapshot, the rebuild
without Opus and its snapshot), was run label-free with respect to model outputs on 2026-10-05
before the commit and is disclosed above, its pins written into `REGISTERED_M4` and committed
with the plan. Then, once each, on the serving host: `features build` for the teacher texts →
`bench x3 --date <date>` → `bench k2 --date <date>` → `bench x4 --date <date>` → `bench table`;
every output committed as produced, in one commit, before any of it is written into prose
(`--force` replaces only a failed or unregistered attempt, and the replacement is disclosed).
Then `learn fit --date <date>` (the full-data probes and calibrators, one version per served
model, exported to `bench/results/<date>/artifacts_v<N>/` and committed; refused outright on
any deviation from the registration), promotion decided by
K2's verdicts and recorded in DESIGN amendment A7 with X4's decision, `learn promote` once per
promoted task after A7, and the production audit (`annotate` on at least five non-bench SRER
cards under the promoted tiers, `audit sample --n 100 --min-cards 5`, the curator's `audit
review`, `audit record`). A numeric `auto` stays null in 0.1.0 whatever K2 says. Determinism:
the fitters have no randomness, ties go to the first configuration in declared order, every
bootstrap uses B = 2000 and seed 0, and two runs of a producer on the same inputs are
byte-identical (tested on synthetic worlds); a discrete outcome that flips under another
platform's last-bit differences is recorded as a discrepancy, never silently re-derived.

### M4 results (2026-10-04; DESIGN A7)

The run protocol ran once on the serving host and its outputs are committed as produced under
`bench/results/2026-10-04/` (`x3.json`, `k2.json`, `x4.json`, their `.md`, `table.md`,
`artifacts_v1/`); DESIGN A7 records the consequences and RESEARCH.md ("M4 results") every number.
**`term.fits`: K2(b), the probe is promoted.** Its best servable tier is the nested probe cell
`x3.json#neon_term_fits.probe.F7` (`pair512.v1` on `clm-latest`, logreg λ 1, Platt): accuracy
0.754, NLL 0.485, ECE 0.079, AUROC 0.804 [0.749, 0.864]; `beats_lookup_novel` true (novel-key
AUROC lower bound 0.753; NLL ≻ `lookup_prob` under rule R with lower bound 0.057); 6 of 7 folds
chose the served configuration. K2(a) fails on its other three conditions (Δacc against AnyJev
L2 −0.011 with cluster lower bound −0.053, the card-sign test 4 of 7 with 6 needed, ECE's
cluster upper bound 0.129 > 0.12), so the verdict is (b), proposer-only: artifact version 1's
`term.fits` probe (`a686a06aab14497a`) is promoted on the serving host, and a task
`decider.ols_rank_tasks` names is now decided by a promoted probe where the provider resolves one
([Tiers and policy](tiers-and-policy.md)). **`column.ontology_fits`: K2(c), nothing promoted.**
The probe (`x3.json#neon_ontology_fits.probe.F9`: accuracy 0.821, NLL 0.404, ECE 0.061, AUROC
0.892 [0.855, 0.933]) beats the calibrated F9 cell on NLL under rule R but `beats_lookup_novel`
is false (the novel-key NLL bootstrap bound passes at 0.047, the card sign test fails; K2 also
fails non-inferiority, Δacc −0.016 with lower bound −0.053, 2 of 7 cards, ECE's upper bound
0.131); production keeps A1's zero-shot F9 proposals, and probe investment for the task pauses
until ≥ 200 curator labels or CLM-35B. **The closed choices: K2(c) by construction** (nothing
pools under the calibrator floor; `avu.value_kind`'s probe cell scores accuracy 0.655, NLL 0.859
against the majority 0.478 and the lookup 0.680); A6's rules stand. **X4: teacher labels are not
kept.** `on ≻ off` on novel-key NLL fails for both scorings at (0.5, 0.3): `term.fits` Δ +0.006
(lower bound −0.062) on silver and +0.035 (−0.037) on silver-minus-Opus; `column.ontology_fits`
−0.106 (−0.166) and −0.114 (`insufficient_clusters` on the 66 surviving novel-key items); the
(0.3, 0.1) arm gives `term.fits` +0.035 (−0.008) / +0.047 (−0.013) and `column.ontology_fits`
−0.070 (−0.126) / −0.102. Production fits use the registered snapshot only. **CLM's head adds
signal**: `@latest` ≻ `@raw` on NLL under rule R for both rank_fit tasks (lower bounds 0.012 and
0.026). No numeric `auto` exists (every probe cell's `threshold_cp` is null at both risks; 0.1.0
ships proposed-only), and the production audit on the promoted `term.fits` probe is still to run.

## Baseline numbers

From mesa-anyjev's `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json` (Qwen/Qwen3-8B bf16,
leave-one-card-out over 7 cards): `neon_term_fits.L2` accuracy 0.765, ECE 0.058, coverage at 5%
risk 0.088 (n 285, 199 negatives); `neon_ontology_fits.L2` accuracy 0.837, ECE 0.078, coverage
at 5% risk 0.547 (n 190, 114 negatives). The per-item dump of those predictions,
`bench/baselines/anyjev_l2_2026-09-29.json` (run under mesa-anyjev's interpreter in the M1-A GPU
window; summary `bench/baselines/anyjev_l2_2026-09-29.md`), reproduces both cells and all 14
per-fold cells exactly, and each of its 285 and 190 items pairs with one row of
`bench/snapshots/2026-09-29.parquet` on the D1 identity with the same label and card, so
mesa-clm's cells can be compared item by item, card by card (K2(a), M4).

mesa-clm reports ECE, coverage at risk and AURC with tie-invariant versions of those metrics
(DESIGN D33): tied confidences share their group's mean correctness, so a cell no longer depends
on how a platform's sort orders ties. On tie-free probabilities such as AnyJev L2's the values are
identical, so the numbers above stay directly comparable.

The first mesa-clm cells are the M0 no-model controls in
`bench/results/2026-09-29/baselines.json` (five `<task>.baseline.lookup_prob` cells, snapshot
`bench/snapshots/2026-09-29.parquet`): `lookup_acc` 0.772 (term.fits, n 285), 0.800
(ontology_fits, n 190), 0.796 (annotate, n 98), 0.550 (aspect, n 60), 0.680 (value_kind, n 278);
leave-one-product-out 0.723 / 0.721 / 0.724 / 0.350 / 0.543; novel-key subsets of 159 / 99 / 39
/ 28 / 80 items. `bench/results/2026-09-29/mde.json` (200 simulated datasets per grid point,
B=2000, seed 0) puts the minimum detectable effect under rule R at AUROC 0.625 on term.fits
(0.675 on its novel keys), 0.650 on ontology_fits (0.750 on novel keys, with four counting
cards) and 0.700 on annotate, whose novel-key population has one counting card and therefore
cannot pass rule R at any effect. The model cells of milestone M2 (X1, the tier cells, X2) are
above, each named by its results file.
