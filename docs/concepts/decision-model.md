---
title: "Decision model"
description: "How mesa-clm asks CLM: one rank per candidate group with a fixed abstain anchor, set-independent scores, the Q1-Q8 pipeline, contexts that end with the target, framings and the two keys, and what the M2 framing experiment changed (F9 for column.ontology_fits; term.fits proposals by ols_rank under K1)."
type: Guide
tags:
  - concepts
  - decisions
  - clm
  - framings
generated:
  by: "claude/opus-5.5"
  at: "2026-10-03T18:30:00Z"
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

The numbers behind A1 are in RESEARCH.md, "M2 results (registered run)", and on
[Learning and bench](learning-and-bench.md).

| Step | Task | Shape | Context view | Candidates |
|---|---|---|---|---|
| Plan | (the planner) | — | — | which ontologies, columns and queries; hints only |
| Q1 | `column.annotate` | choice K=2 | `{card_header, column}` | `ANNOTATE_OPTIONS`; identifiers are a rule |
| Q2 | `column.aspect` | choice K=8 | same text as Q1 | `ASPECT_OPTIONS`; top-1 if proposed, else top-2 |
| Q3 | `column.ontology_fits` | rank_fit 12 + anchor | F9 (A1): `NEON dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:` | registry `option_text`, masked by aspect after scoring |
| S | candidates | OLS | — | `search_candidates` ≤12, fixed `unit_candidate` |
| Q4 | `term.fits` | rank_fit ≤12 + anchor; by default `ols_rank` (K1, A1) | `target_state = {card_header, scope, aspect, column|site}` | `"{label}: {description[:300]}"` |
| Q4b | `term.fits` (specificity) | rank_fit; not asked for an `ols_rank` group (A1) | same | {parent, ≤10 children, anchor}; a child wins at `p_fit +0.10`, proposed-only (DESIGN D24) |
| Q5 / Q6 | `term.fits` | rank_fit; by default `ols_rank` (K1, A1) | `{card_header, site}` / `{card_header}` | ENVO biome descendants / NCBITaxon |
| Q7 | `avu.value_kind` | choice K=4 | `value_kind_state` | `VALUE_KINDS`, deterministic pre-rules first |
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
CLM did not answer or the tier is `ols_rank`, not for the tasks `ols_rank` decides by design.

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
