---
title: "Decision model"
description: "How mesa-clm asks CLM: one rank per candidate group with a fixed abstain anchor, set-independent scores, the Q1-Q8 pipeline, contexts that end with the target, framings and the two keys, what the M2 framing experiment changed (F9 for column.ontology_fits; term.fits proposals by ols_rank under K1), and what drives each step since amendment A6 (the closed choices Q1, Q2, Q7 by deterministic rules, CLM's answers to them audit-only) and A8 (the aspect fallback for a column the lookup has not seen, by its dtype and unit, never taxon)."
type: Guide
tags:
  - concepts
  - decisions
  - clm
  - framings
generated:
  by: "claude/opus-5.5"
  at: "2026-10-04T19:42:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md)"
    author: "team:idss-mesa"
  - id: clm
    resource: "https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
    title: "Contrastive-LM/CLM at bb42c6c5 (schema.py, engine.py, server.py)"
    author: "org:Contrastive-LM"
status: draft
stale_after: "2026-12-31T00:00:00Z"
---

# Decision model

CLM scores candidate texts against a state: a frozen Qwen3-8B last-token encoder, two 512-d
projection heads, and a softmax over `scale · cosine` **within each question's candidate set**
(RESEARCH.md, text contract). There is no abstain option and the probabilities are relative to
the set, so mesa-clm never asks "does this term fit?" one candidate at a time. Instead
(DESIGN D2):

* every fit question is **one `/v1/systemone` Choice** whose criteria are
  `{<CURIE or registry id>: candidate_text, "__none__": anchor_text}`;
* the score of a candidate is `s_c = ln p_c − ln p_anchor = (scale/T)(cos_c − cos_anchor)`,
  which does not depend on which other candidates were offered;
* `p_fit = σ(s_c)` at zero shot and `σ(a·s_c + b)` once calibrated; the winner is the largest
  `s_c`, and the anchor winning is an abstain (`anchor_won`).

Score questions are never used (upstream #3: state-independent answers). Noul-per-candidate,
the shape mesa-anyjev used, survives only as the control arm F1 of the framing experiment
(DESIGN D3). Closed questions (aspect, annotate, value kind) are ordinary Choices without an
anchor, and `confidence = max(probs)` is computed locally (DESIGN D7). Requests always use
temperature 1; calibration is client-side.

## Pipeline (milestone M1; defaults since amendment A1)

Every context view ends with the target, because last-token pooling weights the tail
(DESIGN D23). Until amendment A1 every task was asked with F7, the state-only context. A1
(2026-10-03) records the registered framing experiment X1 (`bench/results/2026-10-03/x1.json`):

* `column.ontology_fits` (Q3) is asked with **F9** on `clm-latest`, the query-shaped context
  "NEON dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:"; its
  decisions are keyed by F9's `question_key` (`c95785008b523fd0`);
* `term.fits` had no qualifying framing on either model (**K1**): its groups (Q4, Q5, Q6) are
  decided by `ols_rank` by default (the OLS top-1 as `proposed`, no probabilities, never `auto`;
  DESIGN D28), Q4b is not asked for them because it ranks by `p_fit`, and CLM is asked about
  `term.fits` only in an audit run (`annotate --ols-rank-tasks none`); `term.fits` keeps F7 as its
  framing for those audit records;
* the closed choices (Q1, Q2, Q7) keep F7: no pre-registered rule covers them.

Amendment **A7** (2026-10-05, after the registered M4 run) adds the production rule for a
promoted probe: a task in `decider.ols_rank_tasks` is decided by `ols_rank` *unless the provider
resolves a promoted probe for its active framing* (`CURRENT.json`), in which case the probe
decides it at level `probe`, proposed-only. For `term.fits` (K2(b), the version-1 `pair512.v1`
probe promoted on the serving host) that means: Q4, Q5 and Q6 are the probe's calibrated ranks
there (`p_fit` per candidate, `probs` their odds against the anchor, never `auto`), Q4b is asked
again for those groups (D24 amended back, since the record carries `p_fit`), the run names the
task under `probe_tasks` and `ols_rank_tasks` lists only what `ols_rank` actually decided. A host
without the promoted artifact keeps K1 exactly as above. `column.ontology_fits` (K2(c)) keeps
its zero-shot F9 proposals; the closed choices keep A6's rules.

The numbers behind A1 are in RESEARCH.md, "M2 results (registered run)", and on
[Learning and bench](learning-and-bench.md).

## The closed choices by rule (amendment A6)

The same registered run measured CLM's zero-shot answers to the closed choices below the
majority class of their items: `column.annotate` accuracy 0.357 against 0.643, `column.aspect`
0.050 against 0.300, `avu.value_kind` 0.324 against 0.478
(`bench/results/2026-10-03/tiers.json`, cells `neon_annotate.zero_shot.F7`,
`neon_aspect.zero_shot.F7`, `neon_value_kind.zero_shot.F7`). No pre-registered rule covers those
tasks, and having seen the cells the user decided (2026-10-04) that they are answered by
deterministic rules by default. Amendment A6 records that as a product-safety change of serving
behaviour: it makes no scientific claim and touches neither the frozen pre-registration nor any
cell. `decider.closed_choice` selects it (`rules`, the default; `clm` keeps the M2 behaviour for
audits and tests; `annotate --closed-choice`). Under `rules` each step is driven by:

| Step | What decides | Recorded as |
|---|---|---|
| Q1 | a column is annotated iff it is not an identifier, whatever the planner says (its `annotate=False` stays in the run's plan and decides nothing; an identifier stays out even with its `annotate=True`) | a `rule` record answering `Yes`, reason `a6_not_identifier`; identifiers are `No` rules as before |
| Q2 | the planner's aspect hint, then the top two aspects of the M0 lookup (below), de-duplicated, `other` excluded: up to three; when that is empty, the fallback below | one `rule` record per aspect, reason `a6_aspect_hint`, `a6_aspect_lookup` or `a6_aspect_fallback`; the column's Q3 groups carry `search_json.a6` (the source of each aspect, the lookup's counts, the fallback's record: `rule: "a8"`, its `reason`, the prior order, the order considered, what was kept) |
| Q2 fallback (amendment A8, 2026-10-05) | for a column the lookup has not seen: a **numeric** column (the card's dtype is `real`, `integer`, `unsigned integer`, …) **or a column with a unit** gets `measurement` first, then `unit` when it has a unit, else the next aspect of the lookup's training prior (most labels first; aspects without labels after, in registry order) that is neither `taxon` nor `other`; a **string** column keeps the prior order with `taxon` removed. In both cases `other` is skipped, so is any aspect with no ontology in play and `unit` for a column without a unit, and the first two are kept; `taxon` never comes from the fallback (the hint and the lookup still give it). No CLM answer is read, so it works with `column.ontology_fits` decided by `ols_rank` or clm-serve down | `rule` records as above (reason `a6_aspect_fallback`); `search_json.a6.fallback` carries `rule: "a8"` and `reason` `unit`, `numeric` or `string`; Q3 then runs on the kept aspects like any other; a column is `no_aspect` only when no aspect has an ontology in play |
| Q7 | the deterministic pre-rule first; where it leaves the kind open, "the term label" | a `rule` record, reason `a6_value_kind_label`, under the proposal's group |
| Q3–Q6, Q4b, Q8 | unchanged (F9 for Q3, asked once per chosen aspect but `unit`; `ols_rank` for `term.fits` under K1) | as before |

The **M0 lookup** is the no-model control the bench reports beside every cell: it counts how often
each aspect labels a column of the same name on the other cards. Its items are frozen: the 60
`column.aspect` items of the M0 bench task on the registered labels snapshot
(`bench/snapshots/2026-09-29.parquet`), the ones behind the lookup's 0.550 on the aspect cell
above, shipped in the package as `aspect_lookup.json` with its sha256 pinned in the code
(`scripts/freeze_aspect_lookup.py` writes it; a test rebuilds it from the snapshot). Every host
reads the same table and no sidecar is read, so the same card gets the same aspects everywhere.
At annotate time the items of the card being annotated are left out, as the bench's
leave-one-card-out lookup leaves them out; the two most frequent labels of the column's name win,
a tie going to the label of the alphabetically first card. A name no other card carries
contributes nothing and the fallback decides. A run's `labels_sha256` names the snapshot. Two
points are the implementation's reading of the decision, which the user confirmed on 2026-10-04:
an unseen name goes to the fallback, and the fallback's order and its cap of two.

**Amendment A8 (2026-10-05) changes the fallback only.** Under A6 the fallback took the first two
aspects of the lookup's prior order, and that prior comes from the bench's bird and beetle
tables: on seven non-bench SRER runs (A7's appended note) it gave the `term.fits` column groups
`method` 53, `taxon` 44 and `measurement` 1 times, 25 of the 44 `taxon` to numeric or
unit-bearing columns (`archiveMass` in grams searched NCBITaxon for `gram` and proposed twelve
organisms). The user's decision: the fallback reads the card's column grammar instead. A numeric
column (dtype `real`, `integer`, `unsigned integer`, …) or a column with a unit gets `measurement`
first, then `unit` when it has a unit, else the next aspect of the prior that is neither `taxon`
nor `other`; a string column keeps the prior order with `taxon` removed; the cap of two, the
in-play filter and "`unit` only for a column with a unit" stay. `taxon` never comes from the
fallback; it still comes from the planner's hint or the lookup, which are unchanged, as is every
seen column. A fallback column's Q3 groups record `search_json.a6.fallback` with `rule: "a8"` and
`reason` (`unit`, `numeric` or `string`), so a run shows which rule produced its aspects; the
audit-only CLM records are untouched. On the fixture cards (bench cards, most columns seen) the
unseen numeric or unit-bearing columns move from `method, taxon` to `measurement, unit` (or
`measurement, method` without a unit) and the unseen string columns from `method, taxon` to
`method, environment` or `method, measurement`, by card.

**CLM still answers Q1, Q2 and Q7**, in the same requests as before (a column the planner
excluded, which M2 never asked about, gets no audit question), and every such record is
**audit-only**: stored with outcome `abstain` and reason `audit_only_a6` whatever the policy says,
its probabilities kept as served. It decides nothing, makes no link and no label; `explain` lists
it as a decision with that marker, and review, the pending groups and feedback never offer it
(they offer a group's deciding record, which is never one). A run's summary and `--out` JSON give
the mode (`closed_choice`) and the count (`n_audit_only`). Q1's rule and Q7's fallback answer the
majority class of the cells above (`Yes`; "the term label"), which describes them; it is not a
measurement of the rules on new cards.

## The pipeline steps

| Step | Task | Shape | Context view | Candidates |
|---|---|---|---|---|
| Plan | (the planner) | — | — | which ontologies, columns and queries; hints only |
| Q1 | `column.annotate` | choice K=2; by default a rule, CLM audit-only (A6) | `{card_header, column}` | `ANNOTATE_OPTIONS`; identifiers are a rule |
| Q2 | `column.aspect` | choice K=8; by default hint, lookup, else the fallback by dtype and unit, never `taxon` (A6, A8), CLM audit-only | same text as Q1 | `ASPECT_OPTIONS`; under `clm`: top-1 if proposed, else top-2 |
| Q3 | `column.ontology_fits` | rank_fit 12 + anchor | F9 (A1): `NEON dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:` | registry `option_text`, masked by aspect after scoring |
| S | candidates | OLS | — | `search_candidates` ≤12, fixed `unit_candidate` |
| Q4 | `term.fits` | rank_fit ≤12 + anchor; by default `ols_rank` (K1, A1), the promoted probe where `CURRENT.json` holds one (A7) | `target_state = {card_header, scope, aspect, column|site}` | `"{label}: {description[:300]}"` |
| Q4b | `term.fits` (specificity) | rank_fit; not asked for an `ols_rank` group (A1), asked for a probe group (A7) | same | {parent, ≤10 children, anchor}; a child wins at `p_fit +0.10`, proposed-only (DESIGN D24) |
| Q5 / Q6 | `term.fits` | rank_fit; by default `ols_rank` (K1, A1), the promoted probe where `CURRENT.json` holds one (A7) | `{card_header, site}` / `{card_header}` | ENVO biome descendants / NCBITaxon |
| Q7 | `avu.value_kind` | choice K=4; by default the pre-rule, else "the term label", CLM audit-only (A6) | `value_kind_state` | `VALUE_KINDS`, deterministic pre-rules first |
| Q8 | rule | — | — | exact-triple dedup, cap 25 by `p_fit` (DESIGN D25) |

Questions sharing a context go in one request: Q1 and Q2 for a column share one, and every
candidate group of one target shares one. Q3 ranks all twelve registry ontologies plus the
anchor and masks the answer afterwards by the column's aspect and by the ontologies the plan put
in play; the two best in-play ontologies are kept. Q6 keeps a second taxon only when it
out-ranks the anchor, and at most as `proposed`. Q8 drops (exact duplicates, over the cap, an
AVU that cannot be built) reject their groups and are listed with their reason among the
abstentions. While a task is uncalibrated, steps prune by top-k only; when CLM is down or a
tier is killed, the degraded method `ols_rank` proposes the OLS top-1 per group as `proposed`,
never `auto` (DESIGN D28); `decider.tier: ols_rank` selects it for every candidate group, and
`decider.ols_rank_tasks` per task: `[term.fits]` by default (K1, DESIGN A1), `[]` (or `annotate
--ols-rank-tasks none`) for an audit run that asks CLM about `term.fits` too. Under `ols_rank`
the second taxon of Q6 is not kept (only the top-1 is proposed), and where Q4b would have refined
a proposed winner with OLS children the group records `specificity: {asked: false, reason:
no_p_fit_ols_rank}` and the proposal's rationale says so. A run is marked `degraded` only when
CLM did not answer or the tier is `ols_rank`, not for the tasks `ols_rank` decides by design,
nor for a task a promoted probe decides instead (A7: at tier `auto` or `probe` the provider's
`resolve_tier` for the task's active framing is `probe`, the probe decides at level `probe`, Q4b
is asked, and the run's `probe_tasks` names it; `--tier ols_rank` overrides the probe).

Each CLM question is stored as one `decisions` row with one `decision_options` row per candidate
and one for the anchor (`s_c`, `p_fit`, rank, masked), and each candidate group as a
`decision_groups` row ([Provenance](provenance.md)). The groups a reviewer is asked about are
the `term.fits` groups that are proposed, escalated or anchor-won, or that only an agent has
answered, and that no curator has settled; a parent that a specificity child replaced is asked
through the refinement group, which offers the parent too. A curator's answer is final in M1,
an agent's is replaced by a curator's (DESIGN A2).

## Rendering, anchors and the token guard

`render.context(state, framing)` projects the builder-ordered state through the view and calls
the vendored `to_text` (prose, never JSON). Candidate texts are stable per CURIE so CLM's
action cache hits. The anchors are fixed strings in `registry.ANCHORS`: for terms "None of
these terms is the right concept for this target.", for ontologies "None of these ontologies
has a suitable term for this target." Raw cards are never sent: measured states are at most
1,165 tokens (2,272 synthetic worst case), the encoder runs at `--max-model-len 4096`, and a
client-side guard abstains with `truncated=true` above `max_len − 16` (RESEARCH.md, token
lengths).

## Two keys

`task_key = sha256(json{kind, text, options, scale, centers})[:16]` over the original question
texts is mesa-anyjev's lock key (term.fits `0ccc8d141ffd30ff`) and identifies labels together
with `(target_sha256, option_key, label_source)`. `question_key` hashes the framing (context
view and template, instructions, candidate template, anchor text, options, mask rule, render
version, `schema_sha256`) and keys decisions, features and artifacts (DESIGN D1).
The target identity includes the aspect the target was asked under, so a label on (column,
CURIE, aspect) does not carry over to the same column and CURIE under another aspect; this is
intended. `framings.lock.json` pins both keys and each task's `active_framing`; `mesa-clm
framings --check` fails CI on drift. Framings in the experiment grid: F1 (noul control), F4
(Choice + anchor with a task sentence), F7 (state-only target context), F9 (a query-shaped
sentence ending in "ontology term:"). A1 moved `column.ontology_fits`' active framing from F7 to
F9, which rotated the lock's `lock_sha` and no `question_key` (whether a framing is active is not
part of its key). F9's context names the product and the column, not the table, so the same
column in two tables of one product gets the same context and the same answer.
