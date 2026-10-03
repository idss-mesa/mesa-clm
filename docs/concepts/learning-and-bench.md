---
title: "Learning and bench"
description: "Where mesa-clm's labels come from and how they are identified and frozen, the feature cache, the tier fitters, leave-one-card-out with nested selection, rule R, the lookup baseline every cell must beat, the pre-registered experiments X1-X4 with the AnyJev baseline numbers, and how M2's analysis plan runs X1, the tier cells and X2 (committed before any result)."
type: Guide
tags:
  - concepts
  - learning
  - bench
  - labels
  - calibration
generated:
  by: "claude/fable-5.1"
  at: "2026-10-03T12:00:00Z"
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
| `teacher` / `teacher_implicit` | 0.5 / 0.3 | **no** | `labels ingest-teacher`: accepted items of neon-ducklake's Opus-validated generic curation, in-registry only, leak group = product (DESIGN D19) |
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
| `probe` | weighted logistic regression, LDA or ridge on `lowdim.v1`, `pair512.v1`, `pair4096.v1` or the PR #13 replica `joint4096@S1`; spec and hyperparameters by grouped inner CV, then temperature | `choice.state.v1` | `probes/<question_key>.json` |
| `head` | per-candidate two-way rows through CLM's `finetune.py` in the serve venv, re-scored with `HeadPair`, then Platt | closed options | `heads/mesa-<task>-<qk8>-v<N>.pt` |

Artifacts live under `~/.mesa/clm/artifacts/<encoder_fp>/<clm_model_fp>/<lock8>/v<N>/`,
immutable, with a strict loader that refuses fingerprint mismatches. `learn promote` moves
`CURRENT` only when the new version beats the current on NLL and is within 0.01 on accuracy
under rule R; a head is promoted only if it beats the probe (DESIGN D18). Serving never fits
(DESIGN D15).

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
  silver-minus-Opus; the expected effect with 13 usable items is stated in advance as about
  null.

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

### Planned results

None exist yet. The run protocol (plan §14) runs once each, after the push: the label-free timing
run, `bench framing --decide`, the replay, `bench run`, `bench x2`, `bench table`; every output is
committed as produced under `bench/results/<date>/` (`x1_latency.json`, `x1.json`,
`x1_items.parquet`, `tiers.json`, `x2.json`, `table.md`), and this page will cite those files.

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
cannot pass rule R at any effect. Model cells (X1, the tier cells, X2) follow in milestone M2
under the analysis plan above, and every cell will name its results JSON here.
