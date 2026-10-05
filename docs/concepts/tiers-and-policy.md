---
title: "Tiers and policy"
description: "What level and calibration promise on a mesa-clm decision, the record invariants, how a tier is chosen and how the probe tier is served without clm-serve (M4), the rules that turn a decision into auto, proposed or abstain with a cited bench cell (the citation test as implemented, the audit demotion), promotion and the production audits, and what drives each step of a run today (ols_rank for term.fits under K1; the closed choices by rule under amendment A6)."
type: Guide
tags:
  - concepts
  - calibration
  - policy
generated:
  by: "claude/fable-5.1"
  at: "2026-10-04T23:00:00Z"
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
  model over a feature spec built from the frozen 4096-d vectors and the pinned head's 512-d
  projections, served without clm-serve; M4), `head` (a fine-tuned CLM head served by
  clm-serve; M7), or `none` for anything that is not a CLM decision: a rule, a planner hint,
  the `ols_rank` fallback, a Claude second opinion, an unavailable decider.
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
(plan §4.6); from M4, `feature_spec` names the spec a record's numbers came from and is set
exactly when the level is `probe` or `head` (a rule of the record validator, not of the DDL).
The other invariants are CHECK constraints in the sidecar.

## Choosing a tier

`decider.tier` (`MESA_CLM_DECIDER__TIER`, or `annotate --tier`) asks for a tier: `auto` (the
best tier the provider can serve for each question key, in this order: `probe` when
`CURRENT.json` promotes a probe for it, else `calibrated` when it promotes a servable
calibrator, else `zero_shot`), an explicit `zero_shot`, `calibrated`, `probe` or `head`
(refused before any request, exit 1, unless the provider can serve it for every task that
`ols_rank` does not decide; `head` arrives with M7), or `ols_rank`: every candidate group (`term.fits`, `column.ontology_fits`) is decided
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

**A promoted probe lifts a task out of that set (DESIGN A7, 2026-10-05).** A task
`decider.ols_rank_tasks` names is decided by `ols_rank` *unless the provider resolves a promoted
probe for the task's active framing* (`CURRENT.json` promotes one for that exact `question_key`)
and the tier is `auto` or `probe`: then the probe decides it at level `probe` (the calibrator's
kind, `feature_spec` and `artifact_version` on every record; proposed-only, since every `auto`
threshold is null and no probe cell has a `threshold_cp`), the specificity refinement is asked
again for those groups (a probe record carries `p_fit`; D24 amended back), the run lists the task
under `probe_tasks` instead of `ols_rank_tasks` (which names only what `ols_rank` actually
decided), and the run is not `degraded`. The shipped default `[term.fits]` is unchanged: a host
without the promoted artifact proposes by `ols_rank` as before, and CLM's zero-shot `term.fits`
answers stay audit-only (an explicit `--tier zero_shot` or `calibrated` keeps K1). `--tier
ols_rank` still sends every candidate group to `ols_rank`, and `decider.ols_rank_fallback` is
unchanged: the encoder down under a promoted probe gives `unavailable` records, which fall back
to `ols_rank` per group and mark the run `degraded`. The registered M4 run promoted the version-1
`term.fits` probe (K2(b); `bench/results/2026-10-04/x3.json#neon_term_fits.probe.F7`) on the
serving host, so there Q4–Q6 are decided by that probe; `column.ontology_fits` keeps its zero-shot
F9 proposals (K2(c), nothing promoted) and the closed choices keep A6's rules.

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
| Q4–Q6 terms | `ols_rank`, the OLS top-1 (K1, A1); on a host whose `CURRENT.json` promotes the `term.fits` probe, that probe (A7) | `none` (`ols_rank`), or `probe` |
| Q4b specificity | not asked for an `ols_rank` group (A1); asked for a probe group (A7, D24) | — or `probe` |
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

### The probe tier served (milestone M4)

A promoted probe is served by the provider of its spec's model (`clm-latest` for `lowdim.v1`,
`pair512.v1` and `choice.state.v1`; `clm-raw` for `pair4096.v1`, `joint4096@S1`, `joint4096@S1ns`
and `choice.raw.v1`), read from `CURRENT.json` under that model's artifacts directory
([Serving](serving.md)); only what the promotion table names is served, so an unpromoted
calibrator or probe of the same version is `TierUnavailable` at an explicit `--tier` and
`zero_shot` at `auto`, and without a `CURRENT.json` the provider serves `zero_shot`. On the probe
path the state text (what `context_sha256` hashes) and the option texts (anchor last) go to the
encoder through a read-only vector cache (an in-memory LRU of 2,048 texts, then the live feature
store when one exists, never written), the pinned head's numpy export projects them when the
spec reads head quantities, the spec features are built with the bench's own formulas
(`learn.probe.rank_fit_features` / `choice_features`), and the probe's `predict` gives the
calibrated probabilities. **clm-serve is never asked on this path**: the encoder call is
reported to `clm_calls` as `/v1/embeddings` with the tokens charged, the encoder down gives
`unavailable` records, and a 401, 403 or 404 or a port held by another account refuses the run,
exactly clm-serve's rules. The joint specs embed the X1 control framing's per-candidate state
rendered with the request's group size.

The record: `level = probe`, `calibration` the probe calibrator's kind (`platt` at K = 2,
`temperature` above), `raw_probs` the zero-shot softmax over the same vectors under the served
model (what clm-serve would answer), `s_c` its log-ratios with the anchor at 0, `p_fit` the
probe's calibrated Yes probability per option, `probs = softmax(logit(p_fit))` over the
candidates with the anchor's logit at 0 (the mirror of the calibrated tier's `rank_fit`, so each
candidate's odds against the anchor are exactly its `p_fit`; `p_fit` clipped at 1e-12 from 0
and 1); the anchor's own row is the features of a candidate whose vector is the anchor's, the
probe analogue of Platt's `σ(b)`, and its `p_fit` never enters `probs`. `confidence = max(probs)`,
`clm_confidence` is null (no CLM field exists), `feature_spec` the spec, `artifact_version` the
probe's version and `artifact` its reference; the aspect mask and the token guard apply as for
every tier. A closed choice's `probs` are the probe's row.

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
   and fingerprint-matched (below); under a profile with `auto_requires_audit` (`prod`) it
   also needs a passing `audits` row for the record's `(task_key, artifact_version)`, read from
   the sidecar (M4): without one the verdict is **proposed** with reason `audit_required`, the
   one demotion reason a proposed row records, and a record without an `artifact_version` has
   no audit; a check that raises fails closed;
3. **proposed** if the statistic clears `propose`;
4. **abstain** otherwise. A group margin can only demote an auto.

What a run stores is the verdict except where the pipeline settles a record itself: a moot record
or a keep-rule drop is stored `rejected` (M1), and an audit-only closed-choice record `abstain`
with `audit_only_a6` (DESIGN A6). `apply` is M3's; the pipeline records the audit blocker now.

Profiles: `prod` (`min_level_write: calibrated`, `auto_calibrations: [platt, temperature]`,
`auto_requires_audit: true`, `allow_history_none: false`) and `dev` (`min_level_write:
zero_shot`, `allow_auto_write: false`, `allow_history_none: true`). The shipped defaults are
proposed-only: `auto: null` everywhere, and `zero_shot` never auto-writes.

## Citations

A numeric `auto` must cite `bench/results/<date>/<file>.json#<task>.<tier>.<framing>` (DESIGN
D8). From M4 the citation test is `policy.CellCitationValidator`, `Policy.load`'s default
validator (`RefuseAllCitations` stays for callers that want it; `tests/unit/test_policy_citations.py`
negates every field once). The cite resolves under `policy.results_root` (default the checkout
root when `mesa_clm` runs from a `src/` checkout, else `~/.mesa/clm/results`), and the
committed snapshots are the sha256 of every `<results_root>/bench/snapshots/*.parquet`, hashed,
never opened for their labels. The checks, in order, the first failure being the `cite_refused`
reason: the cite is well-formed; the file exists and loads; the cell exists and is the record's
task's; the cell's **tier equals the record's level** (the frozen "tier rank at or above the
level" read at its most conservative: a probe cell's threshold is never applied to a calibrated
record, whose `p_fit` came from another model, nor the reverse); leave-one-card-out;
pre-registered; `selection: "nested"`; not exploratory; servable; at least 5 folds; no teacher
row in a test fold; `masked` equal to the record's framing's masking; `labels_sha256` a
committed snapshot's; at least 30 negatives (rank_fit) or 30 non-modal items (choice);
`question_key`, `encoder_fp`, `clm_model_fp` and `schema_sha256` equal to the record's; the
cell's fingerprint equal to the live one, `serving_lock_sha` included (the pipeline binds the
provider's fingerprint; unbound, every cite is refused, since `serving_lock_sha` is not on a
record); `fold_choices` agreeing with the served configuration in at least 5 evaluated folds (a
calibrated cell: the fold's framing and model are the cell's; a probe cell: the fold's model is
the record's and its spec the record's `feature_spec`; a skipped fold never agrees);
`beats_lookup_novel` true; and `auto ≥ threshold_cp[risk]`, the smallest threshold whose pooled
held-out items (at least 30) have a one-sided 95% Clopper–Pearson error bound within the risk
(`None` refuses). `min_weight` has one source, the policy file, and the bench reads it too
(DESIGN D9). The shipped defaults are still `auto: null` everywhere, so no cite is read in a
default run; no M4 cell exists yet.

## Promotion and the audits (milestone M4)

**Promotion** (`learn promote --version N --task T [--tier probe|calibrated]`) is the only way an
artifact reaches production (serving never fits, DESIGN D15). It requires (i) the version
entry's cite to name a cell of that task and tier that is leave-one-card-out, pre-registered,
`selection: "nested"` and not exploratory, whose `question_key`, fingerprints and
`labels_sha256` equal the manifest's; (ii) `k2.json`'s verdict for that tier to be `a` or `b`
(the verdict is read only when the task's `best` tier is this tier: K2 judges the best servable
tier only, so another tier has no verdict and is refused, there is no per-tier verdict, and `c`
is refused; K2's (a) is demanding, since its card-sign condition is rule R's literal sign test
against AnyJev L2, but (b) suffices here); (iii) when the task already serves a promoted
artifact, the new cell to beat the
current one on NLL under rule R and to be non-inferior on accuracy within 0.01, both on the two
cells' pooled items joined by identity (a partial join is refused); (iv) no head before M7.
Nothing is written when a rule fails; a pass writes `CURRENT.json`. `learn fit` never promotes.
After the registered M4 run (DESIGN A7) one promotion exists: `learn promote --version 1 --task
term.fits --tier probe` on the serving host (K2(b) for `term.fits`), which is what makes the
`term.fits` groups probe-decided there (above); `column.ontology_fits` (K2(c)) and the closed
choices promote nothing.

**Audits** (plan §4.7, §8). `audit sample` draws from the actor's annotate runs on **non-bench**
cards (a bench card, or a card with silver labels in the store, refuses the whole sample: a
curator label on a pre-registered item is never minted here) every decision with a distribution
(rules, planners and unavailable records excluded; `ols_rank` proposals included) in three
strata of equal shares: `would_be_auto` (the policy statistic, `p_fit` of the answer for a
rank_fit or `confidence` for a choice, at or above the task's cited cell's `threshold_cp[risk]`
when the policy cites one, else at or above the top decile of the task's statistics in the
pool), `proposed` and `anchor_abstain`; each stratum sorted by id and permuted with seed 0; the
sample must span at least `--min-cards` cards (defaults 100 and 5). The file holds ids, task
keys, strata, the statistic, the card names and the checklist, no card content. `audit review`
needs a terminal, like `review` (its answers are curator labels, DESIGN A2): a rank_fit
decision's answered CURIE gets `Yes` (correct) or `No` (incorrect), a correct closed choice gets
its answer, an incorrect closed choice, an anchor answer or an abstain mints no label (no true
class is named); every row is `curator` at weight 1.0, `fold_eligible=false`, `bench_card=false`,
origin `audit:<audit_id>`. `audit record` writes one `audits` row per `(task_key,
artifact_version)` among the reviewed would-be-auto decisions with `n`, `n_cards`, `n_errors`,
`cp95_upper` (the one-sided 95% Clopper–Pearson bound on the error rate), the task's `risk` and
`passed = n ≥ 50 ∧ n_cards ≥ 3 ∧ cp95_upper ≤ 2 × risk`, enforced by the row model itself;
decisions that apply no artifact (`zero_shot`, `ols_rank`) get no row, since an audit is of an
artifact version. The proposed-precision interval is printed per stratum, report-only. The
production audit of plan §8 (at least five non-bench SRER cards under the promoted tiers) is part
of the M4 run protocol and has not run.
