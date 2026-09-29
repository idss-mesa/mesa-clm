---
title: "Learning and bench"
description: "Where mesa-clm's labels come from and how they are identified and frozen, the tier fitters, leave-one-card-out with nested selection, rule R, the lookup baseline every cell must beat, and the pre-registered experiments X1-X4 with the AnyJev baseline numbers."
type: Guide
tags:
  - concepts
  - learning
  - bench
  - labels
  - calibration
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T00:00:00Z"
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
| `curator` / `curator_implicit` | 1.0 / 0.7 | yes, except on bench cards | a pick in the MRTR elicitation or the interactive CLI (`review`), or a ticked neon `review.md` (DESIGN D21) |
| `agent_pick` | 0 | no | a plain `mesa_clm_feedback` tool call; recorded, never learned from |
| `consensus_all` / `_majority` / `_negative` | 0.8 / 0.6 / 0.5 | yes | `labels ingest-neon-eval`: a term proposed by every, by at least two, or by a single one of the four agentic models in neon-avu-eval |
| `teacher` / `teacher_implicit` | 0.5 / 0.3 | **no** | `labels ingest-teacher`: accepted items of neon-ducklake's Opus-validated generic curation, in-registry only, leak group = product (DESIGN D19) |
| `gold` | 1.0 | yes | reserved; no hand-verified rows exist yet |

The silver is *agreement between models*, not truth, and Opus is both a silver labeler and the
teacher, which is why teacher labels never reach a test fold and X4 ablates them. Weights are
`sample_weight` in every fitter, including LDA and ridge (DESIGN D20). The bench never reads
the live label table: `labels snapshot` writes a frozen Parquet whose `labels_sha256` every
cell and manifest records, and curator labels on bench cards are excluded from pre-registered
cells (DESIGN D30). M0 ships `labels ingest-neon-eval|import-anyjev|snapshot|stats`; the
fixture ingestion yields 934 rows (term.fits 285: 86 Yes / 199 No), and the 303 valid (card,
CURIE) pairs of `validated.json` become 285 rows because five GAZ CURIEs come back from EBI OLS
as root terms (18 pairs, listed in `terms_missing`; RESEARCH.md).

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
  controls; a fixed decision rule picks the production framing A1 (M2).
* **X2 baselines**: majority, `lookup_prob`, novel-key, leave-one-product-out, the PR #13
  logistic-regression replica, and per-item AnyJev L2 predictions if the dump is feasible.
* **X3 tier sweep**: zero-shot and calibrated on A1; probe specs and fitters; the head (M7).
* **X4 teacher ablation**: off vs (0.5/0.3) vs (0.3/0.1), scored on silver and
  silver-minus-Opus; the expected effect with 13 usable items is stated in advance as about
  null.

Kill and pivot criteria K0–K4 are fixed with them: no state signal makes a task's CLM tiers
audit-only and its proposals `ols_rank`; a task earns auto-eligibility only by beating the
lookup on novel keys and matching AnyJev L2 in a paired test with ECE at most 0.08.

## Baseline numbers

From mesa-anyjev's `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json` (Qwen/Qwen3-8B bf16,
leave-one-card-out over 7 cards): `neon_term_fits.L2` accuracy 0.765, ECE 0.058, coverage at 5%
risk 0.088 (n 285, 199 negatives); `neon_ontology_fits.L2` accuracy 0.837, ECE 0.078, coverage
at 5% risk 0.547 (n 190, 114 negatives).

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
cannot pass rule R at any effect. Model cells (X1, X2) follow in milestone M2, and every cell
names its results JSON here.
