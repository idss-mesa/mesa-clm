---
title: "Quickstart"
description: "Annotate a dataset card, read the run back, review its pending groups and export it: offline with the fake provider and the recorded OLS fixtures, then on the serving host with the real CLM provider."
type: Guide
tags:
  - getting-started
  - quickstart
  - annotate
  - review
generated:
  by: "claude/fable-5.1"
  at: "2026-10-01T18:00:00Z"
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

# Quickstart

This walks one dataset card through the decide phase (milestone M1): `annotate` decides and
records, `explain` reads the run back, `review` and `feedback` answer the groups that wait for a
reviewer, and `provenance export` copies the run out of the sidecar. Nothing here writes iRODS
or the MESA history; `apply` and `revert` arrive with milestone M3. Every outcome is
**proposed-only**: the shipped policy has `auto: null` for every task, so no decision becomes
`auto` until a calibrated tier is cited in milestone M4 (DESIGN D8, D10).

## Offline, with the fake provider

From a checkout (`uv sync --all-extras`, see [Install](install.md)). The fake provider is a
deterministic hashed-n-gram stand-in for CLM: it exercises the pipeline, the policy and the
sidecar, and **it is never evidence**; its probabilities mean nothing about the card. OLS is
replayed from the recorded fixtures, so nothing touches the network.

```bash
export MESA_CLM_OLS__FIXTURES=replay                  # recorded EMBL-EBI OLS4 responses only
export MESA_CLM_OLS__FIXTURES_DIR=tests/fixtures/ols
export MESA_CLM_PROVENANCE__DSN=duckdb:////tmp/mesa-clm-quickstart/provenance.duckdb

uv run mesa-clm annotate --card tests/fixtures/cards/DP1.10003.001.brd_countdata.md \
  --provider fake --fake-seed 5 --out run.json
```

Mind the slashes in a DuckDB DSN: the path starts after `duckdb:///`, so
`duckdb:///x.duckdb` is `x.duckdb` in the working directory and `duckdb:////tmp/x.duckdb` (four
slashes) is the absolute `/tmp/x.duckdb`. Without `--provenance` or the variable the sidecar is
`~/.mesa/clm/provenance.duckdb`. Without `MESA_CLM_OLS__FIXTURES` the default (`off`) queries the
live EMBL-EBI OLS. `--fake-seed` picks the fake head; seed 5 answers "Yes" to enough columns for
every pipeline step to run on the fixture cards (the default seed 0 annotates few columns).

`annotate` prints a summary:

* `run <run_id> (owner …, card …, tier …)`: the run id (commands below accept a unique prefix of
  at least 4 hex digits), the owner (`--owner`, default `--actor`, default `$USER`) and the tier;
* the counts: proposals kept, groups abstained, decisions recorded, CLM requests, the encoder
  tokens they cost, the wall time, and `degraded` when any group fell back to the `ols_rank`
  method or a CLM answer was unavailable;
* `outcomes`: decisions per outcome (`proposed`, `abstain`, `rejected`, `rule`, …);
* `abstained`: groups that propose nothing, by reason (`anchor_won`: "none of these" ranked
  first; `no_candidates`: OLS found nothing; `duplicate`, `over_cap`: the keep rule, DESIGN D25);
* `fingerprint`: `encoder_fp`, `clm_model_fp` and the first characters of `serving_lock_sha`,
  the D5 bundle stamped on every decision (the fake has its own, with route `fake`, so a fake
  run can never pass for a real one);
* the provider note and `next: mesa-clm review --run-id <run_id>`.

`--out run.json` holds the full run: every proposal with its attribute, value, unit, CURIE,
`p_fit`, level (`zero_shot` for CLM, `none` for `ols_rank`), calibration, method and rationale,
plus the abstentions. `--out -` prints it instead, and `--eval-result` writes the neon-avu-eval
result shape so its scoring scripts run unchanged.

### Read the run back

```bash
uv run mesa-clm explain --run-id <prefix>            # decisions, top-3 options per group, links, pending groups
uv run mesa-clm explain --irods-path /zone/home/you/data.csv   # your runs with links at that path (after apply, M3)
```

Only the owner's runs are shown: `explain`, `review` and `feedback` act as `--actor` and refuse
another owner's run (DESIGN D21).

### Review

```bash
uv run mesa-clm review --run-id <prefix>
```

Interactive, and it needs a terminal. For each group that waits for a reviewer (term groups
that are proposed, escalated to a human, or whose anchor won, and groups only an agent has
answered so far) it shows the target, aspect and ontology, then the offered candidates best
`p_fit` first, numbered, and `0. none of these (the anchor)`. Answer with a number to accept
that candidate, `0` or `n` for none of these, `d` to decline (you saw it but answer nothing),
`s` or Enter to skip, `q` (or end of input) to quit. Each answer is recorded with `via=cli`: a
pick writes curator labels (the pick, and implicit negatives for the other offered candidates),
none of these an anchor-positive row plus a negative per candidate, and a decline only an
override row (DESIGN D21, A2). The pick's link becomes `accepted`, ready for `apply` in M3. A
curator's answer settles the group: a different second answer is refused (the same one again
changes nothing).

The answers can also be given as arguments; every one is checked (a pending group, an offered
candidate, one answer per group) before any is recorded:

```bash
uv run mesa-clm review --run-id <prefix> --pick <group>=PATO:0000040 --pick <group2>=none --decline <group3>
uv run mesa-clm feedback --group-id <group> --action pick --option-key PATO:0000040
uv run mesa-clm feedback --group-id <group> --action reject      # none of these
```

Typed at a terminal these are curator answers too. **Run without a terminal** (from a script, a
pipe or an agent's shell) they are recorded exactly like a plain tool call: `via=tool`,
`agent_pick` labels at weight 0 that never enter a fit, links accepted by `agent`; the verb says
so on stderr, and the groups stay pending so a curator's answer can replace the agent's (DESIGN
A2). Group ids come from `explain` (`pending_groups`, and `group_id` per group); `review` takes
a unique prefix within the run, `feedback` a unique prefix among your runs' groups, and
`--action pick` needs `--option-key` (`none` for none of these).

### Export and prune

```bash
uv run mesa-clm provenance export --run-id <prefix> --out /tmp/mesa-clm-quickstart/export
uv run mesa-clm provenance prune --dry-run
```

The export is one Parquet file per run table plus a `manifest.json` with each file's sha256
and row count, owner-only like the sidecar (0700, files 0600); the run is marked exported only
after the copy is written and verified. `provenance import <dir>` commits it into another
sidecar. `prune` deletes only
terminal runs (applied, exported, every written link recorded in the history) and abandoned
runs past the TTL, so a freshly decided run stays.

## On the serving host, with CLM

The same commands with `--provider clm` (the default) talk to the encoder on `127.0.0.1:8090`
and clm-serve on `127.0.0.1:8700` ([Serving](../deploy/serving.md)). The keys `serve keys
--init` wrote (`~/.mesa/clm/secrets/clm.key`, `encoder.key`) are read by default; check the
stack first:

```bash
uv run mesa-clm doctor --serve        # binds, the 401 matrix, models, goldens, long input, parity, drift

MESA_CLM_OLS__FIXTURES=replay MESA_CLM_OLS__FIXTURES_DIR=tests/fixtures/ols-srer \
  uv run mesa-clm annotate --card tests/fixtures/cards-srer/DP1.00004.001.BP_30min.md \
  --tier zero_shot --out run.json
```

The live example uses a **non-bench** card (plan §9's smoke card, with its recorded OLS
responses): until the pre-registered M2 cells exist, no live annotate run looks at the seven
bench cards (DESIGN, "G1 freeze"). For other cards, live OLS is used with `ols.fixtures` `auto`
or `record`.

`--tier zero_shot` asks CLM's released head (`clm.model`, default `clm-latest`) and records
level `zero_shot`, calibration `uncalibrated`: `p_fit = σ(s_c)` with `s_c` the log-odds of a
candidate against the fixed anchor (DESIGN D2). The summary adds the serving lock the
fingerprint came from (`~/.mesa/clm/serving.lock.json`, or the checkout's
`serving/serving.lock.json`) and the pre-flight's answer. The fingerprint lines now describe
the real stack: `encoder_fp` hashes the encoder recipe in that lock (model, revision, dtype,
last-token pooling, normalisation, `max_len`, left truncation, prefix caching, route and batch
invariance; c3b3d5e1a283 on sparky-1, DESIGN A3), `clm_model_fp` the served head and the CLM
commit, and `serving_lock_sha` the lock itself. A calibration artifact is applied only under
the `encoder_fp` and `clm_model_fp` it was fitted with and only to its own `question_key`, and
a bench cell is cited only when its whole fingerprint and `question_key` equal the live ones
(DESIGN D5, D8, K4).

Before the first question `annotate` runs a pre-flight: clm-serve's `/health`, then, with the
keys, its model list and the encoder's (model, window and route against the lock), then the
running encoder container against the lock's recipe (a check docker cannot answer is noted as
"not verified"). A missing or rejected key, a model clm-serve does not serve or an encoder that
is not the lock's stops it with exit 2 at any CLM tier (`--tier ols_rank` skips the
pre-flight). When clm-serve does not answer,
`annotate` refuses with exit 1 and says why. Run
`--tier ols_rank` (the OLS top-1 of every candidate group, proposed-only, never auto; DESIGN
D28) to decide the candidate groups without it (the column questions Q1, Q2 and Q7 still try
CLM and are recorded as unavailable, so the run says `degraded` and columns without a planner
aspect get no groups), or set `decider.ols_rank_fallback: true`
(`MESA_CLM_DECIDER__OLS_RANK_FALLBACK=true`) to let tier `auto` degrade to `ols_rank` on its
own; an explicit CLM tier such as `zero_shot` always refuses. Review, feedback and export work
the same way against either provider's runs.
