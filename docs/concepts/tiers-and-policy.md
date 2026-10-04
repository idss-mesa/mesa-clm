---
title: "Tiers and policy"
description: "What level and calibration promise on a mesa-clm decision, the record invariants, the rules that turn a decision into auto, proposed or abstain with a cited bench cell, and what drives each step of a run today (ols_rank for term.fits under K1; the closed choices by rule, CLM's answers to them audit-only, under amendment A6)."
type: Guide
tags:
  - concepts
  - calibration
  - policy
generated:
  by: "claude/opus-5.5"
  at: "2026-10-04T18:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/plan-2026-09-28.md"
    title: "mesa-clm implementation plan (v1.1, 2026-09-28)"
    author: "team:idss-mesa"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# Tiers and policy

Every stored decision carries two fields (DESIGN D6):

* `level`: the tier that produced the probability. `zero_shot` (CLM's released head, or
  `clm-raw`), `calibrated` (Platt or temperature fitted on labels), `probe` (a local linear
  model on the frozen 4096-d or 512-d features), `head` (a fine-tuned CLM head served by
  clm-serve), or `none` for anything that is not a CLM decision: a rule, a planner hint, the
  `ols_rank` fallback, a Claude second opinion, an unavailable decider.
* `calibration`: how the probabilities were mapped: `uncalibrated`, `platt`, `temperature`, or
  `none`. `probs` is null exactly when calibration is `none`; nothing is ever one-hot.

`LEVEL_RANK = {none: -1, zero_shot: 0, calibrated: 1, probe: 2, head: 2}`. The `_honest`
check refuses a record that breaks the invariants: `zero_shot ⇒ uncalibrated`;
`calibrated|probe|head ⇒ platt|temperature`; rank_fit records carry `anchor_index` and a
`p_fit` per option; `confidence == max(probs)`; a level above `zero_shot` names an
`artifact_version` whose `(question_key, encoder_fp, clm_model_fp)` match the record; an anchor
win or a truncated context is an abstain and builds no AVU; `ols_rank` records have null probs
and are never `auto`; every record carries `task_key, question_key, framing_id, encoder_fp,
clm_model_fp, schema_sha256, state_sha256, target_sha256, context_sha256, context_tokens`
(plan §4.6). The same invariants are CHECK constraints in the sidecar.

## Choosing a tier

`decider.tier` (`MESA_CLM_DECIDER__TIER`, or `annotate --tier`) asks for a tier: `auto` (the
best tier the provider can serve for each question key: `calibrated` when a promoted artifact
has its calibrator, else `zero_shot`), an explicit `zero_shot`, `calibrated`, `probe` or `head`
(refused before any request when the provider cannot serve it; `probe` arrives with M4, `head`
with M7), or `ols_rank`: every candidate group (`term.fits`, `column.ontology_fits`) is decided
by the degraded method, the OLS top-1 as `proposed` with no probabilities (DESIGN D28), while
the closed choices (annotate, aspect, value kind) are still asked. The pipeline also falls back
to `ols_rank` per group when a CLM answer is unavailable or the context was truncated, and the
run is marked `degraded`. Under the closed-choice rules (below) a column's aspects never depend
on CLM, so under `ols_rank` (or with clm-serve down) every column keeps its aspects and its
column-to-ontology groups are `ols_rank` too.

`decider.ols_rank_tasks` (`MESA_CLM_DECIDER__OLS_RANK_TASKS`, or `annotate --ols-rank-tasks`)
names the candidate-group tasks that `ols_rank` decides whatever the tier. Its shipped default is
`[term.fits]`: the pre-registered framing experiment found no qualifying framing for `term.fits`
on either model, so its kill criterion K1 ("no state signal") makes that task's zero_shot and
calibrated tiers audit-only, with `ols_rank` proposals until a probe is promoted (DESIGN A1;
`bench/results/2026-10-03/x1.json`). Those groups get no probabilities and no specificity
refinement (it ranks by `p_fit`; the group records that it was not asked), and the run is not
marked `degraded` for them. An audit run asks CLM about `term.fits` anyway: `--ols-rank-tasks
none` or `decider.ols_rank_tasks: []`; its `term.fits` decisions are `zero_shot`, never `auto`.
`column.ontology_fits` is asked with its A1 framing (F9).

`decider.closed_choice` (`MESA_CLM_DECIDER__CLOSED_CHOICE`, or `annotate --closed-choice`) says
who answers the closed choices. Its shipped default is `rules` (DESIGN A6): M2's registered tier
run measured CLM's zero-shot answers below the majority class on all three
(`bench/results/2026-10-03/tiers.json`: `column.annotate` 0.357 against 0.643, `column.aspect`
0.050 against 0.300, `avu.value_kind` 0.324 against 0.478), no pre-registered rule covers them,
and the user chose deterministic fallbacks after seeing that, as a product-safety change that makes
no scientific claim. `clm` keeps the M2 behaviour for audits and tests. What drives each step of a
run with the shipped defaults:

| Step | Driven by | Level of the deciding record |
|---|---|---|
| Q1 annotate | rule: every non-identifier column, whatever the planner says | `none` (`rule`) |
| Q2 aspect | the planner's hint, then the top two of the M0 lookup over the packaged table frozen from the registered labels snapshot (the card held out), up to three; else the first two aspects in the lookup's prior order (`unit` only for a column with a unit); no CLM answer read | `none` (`rule`) |
| Q3 ontology | CLM, F9 (A1), asked for each chosen aspect but `unit`; top two in-play ontologies kept whatever the outcome (D28) | `zero_shot` |
| Q4–Q6 terms | `ols_rank`, the OLS top-1 (K1, A1) | `none` (`ols_rank`) |
| Q4b specificity | not asked for an `ols_rank` group (A1) | — |
| Q7 value kind | the pre-rule, else "the term label" | `none` (`rule`) |
| Q8 keep | the rule: dedup, cap 25 (D25) | — |

Under `rules` CLM's answers to Q1, Q2 and Q7 are still requested in the same calls and stored, but
**audit-only**: outcome `abstain`, reason `audit_only_a6`, whatever the verdict below says (the
verdict is still computed, so a misconfigured threshold still fails the run). They never decide
anything, never become a link or a label, and are never offered for review; `outcomes` counts
them under `abstain` and `n_audit_only` counts them alone.

When `annotate --provider clm` finds clm-serve not answering before it starts, tier `auto` runs
`ols_rank` for the whole card only if `decider.ols_rank_fallback` is `true`
(`MESA_CLM_DECIDER__OLS_RANK_FALLBACK`); otherwise, and always for an explicit CLM tier, the
command refuses (exit 1), as the plugin tool will with `decider_unavailable`. A serving pair that
answers but cannot be used as configured (a missing or rejected key, an unserved `clm.model`, an
encoder that is not the serving lock's) is never degraded: the command stops (exit 2), and a
request clm-serve refuses mid-run (401, 403, 404) fails the run instead of turning its groups
into `ols_rank` proposals. Either way nothing is ever `auto` without a calibrated, cited tier.

## Outcomes

`outcome(record, thresholds, profile)` (`policy_defaults.yaml`, `mesa_clm.policy`) reads `p_fit`
for rank_fit questions and `confidence` for choices:

0. a `rule` record keeps its rule outcome; an `ols_rank` record is `proposed` if its OLS rank is
   within `ols_rank_top` (default 1), else `abstain`;
1. **abstain** if probs are null, the answer index is negative, the context was truncated, the
   anchor won, or every option is masked;
2. **auto** only if the task's `auto` threshold is set, the level is at or above both the
   task's `min_level` and the profile's `min_level_write`, the calibration is one the profile
   allows, the statistic clears `auto`, the margin clears `margin`, and the cited cell is valid
   and fingerprint-matched (below);
3. **proposed** if the statistic clears `propose`;
4. **abstain** otherwise. A group margin can only demote an auto.

What a run stores is the verdict except where the pipeline settles a record itself: a moot record
or a keep-rule drop is stored `rejected` (M1), and an audit-only closed-choice record `abstain`
with `audit_only_a6` (DESIGN A6).

Profiles: `prod` (`min_level_write: calibrated`, `auto_calibrations: [platt, temperature]`,
`auto_requires_audit: true`, `allow_history_none: false`) and `dev` (`min_level_write:
zero_shot`, `allow_auto_write: false`, `allow_history_none: true`). The shipped defaults are
proposed-only: `auto: null` everywhere, and `zero_shot` never auto-writes.

## Citations

A numeric `auto` must cite `bench/results/<date>/<file>.json#<task>.<tier>.<framing>`, and
`tests/test_policy_citations.py` parses the cell (DESIGN D8): leave-one-card-out,
pre-registered, `selection: "nested"`, not exploratory, at least 5 folds, no teacher row in a
test fold, servable, `masked` equal to serving, `labels_sha256` equal to a committed snapshot,
at least 30 negatives (rank_fit) or 30 non-modal items (choice), fingerprint and
`question_key` equal to the live ones, fold choices agreeing with the production framing in at
least 5 of 7 folds, `beats_lookup_novel` true, and `auto ≥ threshold_cp[risk]`, the smallest
threshold whose pooled held-out items (at least 30) have a one-sided 95% Clopper–Pearson error
bound within the risk. With `auto_requires_audit`, apply also needs a passing audit row: at
least 50 would-be-auto decisions from at least 3 non-bench cards, curator-reviewed, error bound
within twice the risk. `min_weight` has one source, the policy file, and the bench reads it too
(DESIGN D9).
