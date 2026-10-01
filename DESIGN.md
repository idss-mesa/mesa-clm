# DESIGN.md — decisions register

Numbered and **append-only**. A decision is never rewritten in place (typo fixes excepted): a
change of mind is a new amendment `A<n>` under [Amendments](#amendments) that names the
decision it amends and why, and the register row gains a pointer to it. The pre-registered
experiments and criteria below are frozen at gate G1; after G1 they change only by amendment.
Facts the decisions rest on live in `RESEARCH.md`; the full plan, with its evidence and review
history, is the design record `design/plan-2026-09-28.md` (cited below as "plan §n").

## User decisions (U1–U4; not reopened)

- **U1 Lineage.** Standalone package `mesa_clm`, console script `mesa-clm`; it ports
  mesa-anyjev's generic layers and imports nothing from `anyjev` or `mesa_anyjev`. Naming: tool
  prefix `mesa_clm_`; entry point `clm` in `mesa_mcp.tools`; `io.mesa/surface="clm"`; env
  prefix `MESA_CLM_`; data under `~/.mesa/clm/`; source tag `mesa-clm:<tool>:<outcome>`;
  sidecar schema `mesa_clm`. Defects fixed in the port:
  (a) the `unit_candidate` lookup order (mesa-anyjev `ols.py:277`), including
  kilometer→UO:0000009;
  (b) Q7 reading `col.name` instead of `prop.column_name` (`pipeline.py:612`);
  (c) the literal `"the top profile value"` (`service.py:405`);
  (d) `auto` never reaching `accepted`;
  (e) docs and code disagreeing on the abstain reason;
  (f) anonymous apply marking links `written` and the run `applied` without writing iRODS
  (`apply.py:94-106`, `mcp_tools/__init__.py:384-392`);
  (g) plain-tool `feedback` picks becoming weight-1.0 curator labels credited to the
  authenticated user (`mcp_tools/__init__.py:465-485`, `_actor` at 165-169).
  Quirk fixed too: `record_human_pick` hard-codes `question_id="term.fits"`.
- **U2 Target.** 0.1.0 is a generic mesa-mcp plugin plus CLI (annotate → apply), benched on the
  neon-avu-eval cards; 0.2.0 is a neon-ducklake curate adapter emitting
  `sites/<SITE>/curation/proposals/<DP>.rep<N>.json`.
- **U3 Serving.** On sparky-1: `vllm/vllm-openai` v0.27.1 through `sg docker`
  (`Qwen/Qwen3-8B --runner pooling`) plus a patched `clm-serve` in its own venv; both bound to
  127.0.0.1 with API keys, about 0.2 of the GPU; CLM patches #6/#10/#11/#23 carried; the core is
  a torch-free HTTP client; an in-process transformers encoder is the documented fallback.
- **U4 Labels.** Train on silver consensus (term.fits 285, ontology_fits 190), curator picks
  (D21) and Opus-validated neon curation as down-weighted `teacher` labels; bench on held-out
  cards only; teacher labels are never in a test fold.

## Register

| # | Decision | Status |
|---|---|---|
| D0 | Pin mesa-mcp `c74f3aa` and mesa-ducklake `7bc143f` by git commit (moving to the M3 upstream merge commits once landed); the core never depends on `contrastive-lm`, torch or vllm; serving has its own venv | accepted |
| D1 | Task vs framing: `task_key` over the original texts equals mesa-anyjev's lock; label identity is `(task_key, target_sha256, option_key, label_source)`; the framing-specific `question_key` keys decisions, features and artifacts | accepted |
| D2 | Rank-first with a fixed abstain anchor: one `/v1/systemone` Choice per candidate group keyed by CURIE/registry id plus `__none__`; `s_c = ln p_c − ln p_anchor` | accepted |
| D3 | No Score questions; noul only as control arm F1, and F1 winning reopens D3 by amendment, never automatically | accepted |
| D4 | Vendor CLM `schema.py` + `LICENSE` byte-identical under `src/mesa_clm/_vendor/clm/`, reached only through the typed facade `render.py`; `client.py` only as a test wire oracle | accepted |
| D5 | Fingerprints (`encoder_fp`, `clm_model_fp`, `schema_sha256`, `serving_lock_sha`) on every decision, feature row, artifact and bench cell; exact match required | accepted; amended by [A3](#a3-2026-10-01--amends-d5), [A4](#a4-2026-10-01--amends-d16-d23) |
| D6 | `level ∈ {none, zero_shot, calibrated, probe, head}`, `calibration ∈ {none, uncalibrated, platt, temperature}`; `zero_shot` never `auto` | accepted |
| D7 | `confidence = max(probs)` computed locally; `p_fit = σ(a·s_c+b)`; CLM's own `confidence` stored only as `clm_confidence` | accepted |
| D8 | Citation rule: a numeric `auto` cites a pre-registered, nested-selection, fingerprint-matched LOCO cell with `beats_lookup_novel`, and sits at or above the CP-bounded risk threshold | accepted |
| D9 | `min_weight` has one source (`policy_defaults.yaml`), read by the bench too; a test pins each bench task's class counts at that weight | accepted |
| D10 | Two phases; apply never decides; annotate stores `auto` as `accepted` with `accepted_by=policy` | accepted |
| D11 | Sidecar `mesa_clm`: per-host DuckDB at `~/.mesa/clm/provenance.duckdb`, opened per operation under a flock, one transaction per run; Postgres optional; CHECK parity; `NOT NULL DEFAULT ''` sentinels instead of `NULLS NOT DISTINCT` | accepted |
| D12 | History backend `direct\|spool\|none\|auto`; the plugin always spools; `direct` is CLI-only and needs the whole lock set free; one spool format `mesa-spool/1`; Postgres never default | accepted |
| D13 | Direct mode writes one snapshot per (run, project) with a note listing paths; never an empty `record_changes` | accepted |
| D14 | Lazy `_register()` with explicit `meta=`; a name collision is logged and re-raised | accepted |
| D15 | Serving never fits; heads reach clm-serve only through promotion under unique names plus a restart | accepted |
| D16 | Topology = encoder URL (:8090) + clm-serve URL (:8700); the fallback is an in-process `Engine(embedder=CudaEncoder)` that also serves :8090; `route` enters `encoder_fp` | accepted; amended by [A4](#a4-2026-10-01--amends-d16-d23), [A5](#a5-2026-10-01--amends-d16-a4) |
| D17 | clm-serve heads run on CPU (`--device cpu --action-cache 512MiB`); the encoder gets the whole 0.20 GPU share | accepted |
| D18 | Probe before head; a head is promoted only if it beats the probe under rule R | accepted |
| D19 | Teacher labels from `accepted` items of neon `curation/generic/<DP>.validated.json`, in-registry only, weight 0.5 / implicit 0.3, never fold-eligible, leak group = product; refuse files whose replicate model starts with `mesa-clm`; corpus frozen by sha256 before M6 | accepted |
| D20 | Label weights are `sample_weight` in Platt, logreg, LDA and ridge; heads exclude teacher rows; a test shows the teacher weight changes the fit | accepted |
| D21 | Curator labels only from MRTR elicitation or the interactive CLI; a plain tool pick is `agent_pick` (weight 0, not fold-eligible); runs store `owner` and every follow-up checks it | accepted; amended by [A2](#a2-2026-09-29--amends-d21) |
| D22 | The planner plans, never decides; Claude is a second opinion only, never `auto` | accepted |
| D23 | Builders and `state_sha256` byte-identical to mesa-anyjev; contexts come from additive views that end with the target; `--max-model-len 4096` plus a client token guard | accepted; amended by [A4](#a4-2026-10-01--amends-d16-d23) |
| D24 | Specificity: one rank over {parent, ≤10 children, anchor}; a child wins at `p_fit(child) ≥ p_fit(parent)+0.10`; unbenched, so proposed-only | accepted |
| D25 | `avu.keep` is a rule: exact-triple dedup, then a cap of 25 by `p_fit` | accepted |
| D26 | MRTR state carries ids only (≤16 KiB) with a tamper guard; OLS and HTTP in `asyncio.to_thread`; `card_path` is CLI-only; iRODS card paths go through `assert_allowed`; parse errors carry line numbers, never content | accepted |
| D27 | Pre-registration and nested selection: X1–X4, metrics and rule R committed before any run; every selection re-made inside each outer fold; full-data selections are `exploratory:true` | accepted |
| D28 | Rank-and-cap while uncalibrated; degraded mode `method="ols_rank"` proposes the OLS top-1 per group as `proposed` (probs NULL, never auto) | accepted |
| D29 | Stream, not store: cards read into memory, runs exported to the project and pruned only once terminal, per-VM sidecars, artifacts and labels moved through the Data Store, secrets `env\|file\|keyring\|auto` | accepted |
| D30 | Frozen label snapshots: the bench reads only `labels snapshot` output; cells record `labels_sha256`; curator labels on bench cards are tagged and excluded from pre-registered cells | accepted |
| D31 | Reversibility: `mesa-clm revert` deletes exactly the written triples; neon replace mode backs up existing reps; the live-venv bump has a written rollback | accepted |
| D32 | Plugin visibility: a mesa-mcp allowlist `MESA_MCP_SERVER__PLUGINS` and a neon template setting it to `none` land before the live bump | accepted |
| D33 | Ranking metrics (ECE, coverage-at-risk, AURC) are tie-invariant: tied confidences share their group's mean correctness before binning; equal to the vendored AnyJev metrics on tie-free inputs | accepted |

## D0. Pinned git dependencies; a torch-free core

`[tool.uv.sources]` pins mesa-mcp `c74f3aa1a3caade558f063f4f637a9127dba82d9` and mesa-ducklake
`7bc143fefb093f0a2ace748a7036598147533ca0`; both move to the merge commits of the M3 upstream
PRs (plan §7.3, D32) once those land. Only c74f3aa has `load_plugins` and
`register_tool(meta=)`; the live stack's 8fbaedf predates both. The PyPI `contrastive-lm` 0.1.0
wheel hard-requires torch and vllm, so the core never depends on it: the text contract is
vendored (D4), the wire is plain httpx, and the serving side lives in its own venv under
`~/.mesa/clm/serve/` that never imports `mesa_clm`.

## D1. Task vs framing

`task_key = sha256(json{kind, text, options, scale, centers})[:16]` over the original question
texts, which is exactly mesa-anyjev's lock key (term.fits `0ccc8d141ffd30ff`). A *framing* is
how a task is rendered for CLM (context view, templates, anchor, instructions) and has its own
`question_key` (plan §4.4) that keys decisions, features and artifacts. Rotating a framing must
not orphan labels, so labels are identified by `(task_key, target_sha256, option_key,
label_source)` (`option_key=''` for choice tasks); `state_sha256` is kept for anyjev import and
parity. The anyjev `candidate_state` contains `n_candidates`, so one pair can have several
state shas, and an anchor pick has no candidate state at all; neither can be a label identity.

*Note (M1, 2026-09-29; a clarification, not a change).* `target_sha256` hashes the compact
target key `{dataset, scope, target, aspect?, term?}` (`identity.target_key`), and `aspect` is
part of it whenever the state carries one, as every `term.fits` `target_state` does. A
`term.fits` label on (column, CURIE, aspect) therefore does not transfer to the same column
and CURIE asked under another aspect: the question differs ("is PATO:0000040 the *measurement*
of observerDistance" vs its *method*), so the answer may too. This is intended.

## D2. Rank-first with a fixed abstain anchor

Each fit question is one `/v1/systemone` Choice whose criteria are
`{<CURIE|registry id>: candidate_text, "__none__": anchor_text}`. CLM's probabilities are a
softmax over the offered set with no abstain option, so a fixed anchor gives every candidate a
set-independent score `s_c = ln p_c − ln p_anchor = (scale/T)(cos_c − cos_anchor)`. The anchor
winning is an abstain (`anchor_won`).

## D3. No Score questions

Upstream issue #3 shows Score answers are state-independent, and #15 shows suffix collapse and
inverted calibration on noul-style questions. Score is never used. Noul-per-candidate (the
mesa-anyjev shape) survives only as control arm F1 in X1; if F1 beats every eligible arm under
rule R, D3 is reopened by an amendment, never adopted automatically.

## D4. Vendored text contract

`src/mesa_clm/_vendor/clm/schema.py` (sha256 `52cec58a…7335`) and its `LICENSE`
(`cfc7749b…d30`) are byte-identical copies from CLM `bb42c6c5`, pinned in `vendored.sha256`.
schema.py has bare `dict` types, so mypy ignores `mesa_clm._vendor.*` and ruff and coverage
exclude it, the same pattern mesa-anyjev uses for its vendored metrics; strict code reaches it
only through the typed facade `render.py`. CLM's `client.py` (`45f565c1…5738`) is vendored only
as `tests/_vendor/clm_client.py`, the wire oracle the fake transport's golden bodies are
checked against; `requests` and `types-requests` sit in the `dev` extra for it.

## D5. Fingerprints

`encoder_fp = sha256{model, revision, dtype, pooling:"LAST", normalize, max_len,
truncation_side:"left", prefix_caching, route}[:12]`, plus `clm_model_fp`, `schema_sha256` and
`serving_lock_sha`, are stamped on every decision, feature row, artifact and bench cell, and an
exact match is required to use any of them. CLM heads are encoder-locked, CLM-35B may change the
encoder, and the fallback route must not silently share artifacts with the vLLM route.

## D6. Honest levels

`level` says what produced the probability (`none` for rules, planners, `ols_rank`, Claude and
unavailable; `zero_shot`, `calibrated`, `probe`, `head` for CLM tiers) and `calibration` says
how it was mapped (`none`, `uncalibrated`, `platt`, `temperature`). `zero_shot` is never
`auto`: #15 reports inverted calibration at zero shot, and PR #13 shows a probe on frozen
embeddings beating the released head.

## D7. Local confidence

`confidence = max(probs)` is computed client-side, and for rank_fit `p_fit = σ(a·s_c+b)` once
calibrated (`σ(s_c)` at zero shot). CLM's own `confidence` is `p_top − mean(p_rest)`; at K=2
it equals the margin, not a probability (RESEARCH.md, contradiction 7), so it is stored only as
`clm_confidence` and never read by policy.

## D8. Citation rule

A numeric `auto` threshold must cite a pre-registered, nested-selection, fingerprint-matched
leave-one-card-out cell whose `beats_lookup_novel` is true (plan §5.4), and the threshold must
be at least the cell's Clopper–Pearson-bounded risk threshold. This fixes the
`masked`/`n_neg=0`/cov@5% weaknesses of mesa-anyjev's citation test and closes the
lookup-baseline leakage: a no-model lookup already scores LOCO acc 0.772 on term.fits.
`tests/test_policy_citations.py` (from M4) enforces every field (plan §4.7).

## D9. One source for min_weight

`policy_defaults.yaml` is the only source of `min_weight`, and the bench reads it too: **0.5**
for term.fits, column.ontology_fits, column.annotate and avu.value_kind; 0.6 for column.aspect;
`avu.keep` is removed (D25). mesa-anyjev had 0.6 in the policy and 0.5 in the bench
(`policy_defaults.yaml:17,65,73`; `bench/tasks/neon.py:53,61,62`); at 0.6 column.annotate loses
all 35 negatives, value_kind loses class 3 (82 rows) and ontology_fits drops to 76 Yes / 0 No.
`test_policy_min_weight_counts` asserts each bench task's class counts at the policy weight
equal the published cell.

## D10. Two phases; apply never decides

`annotate` decides, proposes and records; `apply` only writes what was already accepted.
Annotate stores a policy `auto` as `accepted` with `accepted_by=policy`, which fixes defect (d)
(mesa-anyjev's `auto` never reached `accepted`).

## D11. The provenance sidecar

Schema `mesa_clm` in a DuckDB file per host at `~/.mesa/clm/provenance.duckdb`, **opened per
operation under `flock(~/.mesa/clm/locks/provenance.lock)`**, with a run's rows buffered and
committed in one transaction; Postgres is an optional dialect with the same CHECKs (a parity
test covers both). mesa-anyjev's `DuckDBStore` held its connection for the process lifetime
(`store.py:169`), which blocks every other process. DuckDB 1.5.5 rejects `UNIQUE NULLS NOT
DISTINCT` (verified), so nullable key columns become `NOT NULL DEFAULT ''` sentinels; where a
sentinel does not fit, uniqueness is checked in code and the DDL-parity test exempts exactly
those constraints. DuckDB 1.5.5 enforces CHECK constraints (verified).

## D12. History backends and the lock set

`MESA_CLM_HISTORY__BACKEND=direct|spool|none|auto` (+ `MULTIWRITER=1` for DuckDB 2.0). The
plugin always spools and never writes through mesa-mcp's DuckLake singleton, which never closes
and so holds the single-writer catalog for its process lifetime. `direct` is CLI-only and
requires the whole lock set free and no `/proc` holders: `$MESA_HOME/locks/mesa-catalog.lock`,
plus neon's `<NEON_DUCKLAKE_ROOT>/work/locks/mesa-tag.lock` when configured, both through
`fcntl.flock` (the same call as neon's `env.file_lock`). There is one spool format,
`mesa-spool/1` (neon's column set plus manifest keys), proposed upstream to mesa-ducklake as the
central recorder. Postgres is never the default. This follows the stated preference: a central
recorder and spool now, DuckDB 2.0 multi-writer later.

## D13. One snapshot per (run, project)

Direct mode records one snapshot per (run, project) with a note listing the paths, never an
empty `record_changes`, and the recorder coalesces spooled batches. mesa-ducklake D7 caps a
project at 1000 snapshots with no compaction.

## D14. Plugin registration

`mcp_tools` registers lazily through `_register()` and passes `meta={"io.mesa/surface": "clm"}`
explicitly (mesa-mcp ≥ c74f3aa). A name collision is logged and re-raised, never suppressed:
mesa-anyjev's `contextlib.suppress(ValueError)` hid a half-registered plugin.

## D15. Serving never fits

Fitting happens only in `learn fit` / `learn promote`. clm-serve's `--ckpt-dir` globs `*.pt` at
startup, so a head reaches it only through promotion under a unique file name followed by a
restart of the unit.

## D16. Topology and fallback

The core talks to two URLs: the encoder (:8090, `/v1/embeddings`) and clm-serve (:8700). The
fallback is an in-process `Engine(embedder=CudaEncoder)` that also serves :8090, because
`Engine` only ever calls `embed` and `healthy` on its embedder. `route` (vllm or transformers)
enters `encoder_fp`, so the routes never share artifacts before a recorded parity run.

## D17. Heads on CPU

clm-serve runs its heads on CPU (`--device cpu --action-cache 512MiB`) and the encoder gets the
whole 0.20 GPU share. The two heads together are 18.9M parameters; if CARC's planned 0.83 share
of this GPU lands, about 7.7 GiB would remain.

## D18. Probe before head

A probe (local numpy on 4096-d or 512-d features) is fitted before any head, and a head is
promoted only if it beats the probe under rule R (K3). PR #13: logistic regression on frozen
embeddings 0.737 against the fine-tuned head's 0.686 on typed-decisions.

## D19. Teacher labels

Teacher labels come from `accepted` items (status `accepted` or `human_accepted`) of neon
`curation/generic/<DP>.validated.json`; `seen_curies` come from the `replicates[].proposal` files
(the generic files have none). Out-of-registry prefixes (GO, CHEBI, STATO, OBCS) and unmapped
aspects (`process`) are dropped, which today leaves **14 in-registry, 13 usable of 19**. Weight
0.5 (implicit 0.3), `fold_eligible=false`, `leak_group=product_code`. Any file whose replicate
proposal `model` starts with `mesa-clm` is refused, because validate's generic model inherits
the first replicate's (`validate.py:495`) and Opus is both teacher and a silver labeler. The
corpus is frozen by content sha256 before M6.

## D20. Weights everywhere

Label weights enter as `sample_weight` in Platt, logistic regression, **LDA (weighted class
means and covariance) and ridge (√w row scaling)** in the ported `learn/linear.py`. Heads exclude
teacher rows because `finetune.py` takes no weights. A min_weight filter with hard targets would
count teacher rows exactly like curator rows, contradicting U4;
`test_teacher_weight_changes_fit` shows that changing the teacher weight changes the fitted
model.

## D21. Curator labels and ownership

Curator labels come only from MRTR elicitation responses or the interactive CLI
(`human_overrides.via ∈ {elicitation, cli}`): the pick at 1.0, implicit negatives at 0.7; an
explicit "none" writes an anchor-positive row plus per-candidate negatives (anyjev-compatible).
A plain tool call is `agent_pick` (weight 0, `fold_eligible=false`, `via=tool`), which fixes
defect (g): an agent could otherwise mint curator-grade labels. Runs store `owner`, and
explain, feedback, apply and MRTR resume check it.

## D22. The planner never decides

The planner (static, Claude or gateway) proposes ontologies, columns and queries; it never
answers a question and never writes. Claude is a recorded second opinion only (level `none`,
no probability) and can never produce `auto`: "LLM reasons, CLM selects".

## D23. Builders and target-last contexts

`cards.py` and `states.py` builders and `state_sha256` stay byte-identical to mesa-anyjev (the
parity fixture proves it). Contexts come from additive views that **end with the target**:
`target_state = {card_header, scope, aspect, column|site}`, because last-token pooling weights
the tail most. `test_context_ends_with_target` covers every view. The encoder runs at
`--max-model-len 4096` with a client token guard: the largest synthetic state measured is
2,272 tokens.

## D24. Specificity

Refinement asks one rank over {parent, ≤10 children, anchor}; a child replaces the parent if
`p_fit(child) ≥ p_fit(parent) + 0.10`. `s_c` is comparable across calls (D2), which makes the
single rank possible. The heuristic is unbenched, so it only ever yields `proposed`.

## D25. avu.keep is a rule

Exact-triple dedup, then a cap of 25 by `p_fit`. mesa-anyjev's `avu.keep` question had 11
negatives and every LOCO fold was skipped, so it could never be learned.

## D26. Bounded, ids-only tool inputs

MRTR request state carries ids only (≤16 KiB, mesa-mcp's cap) with a tamper guard (tool, run,
owner, pending group, offered id). OLS and HTTP calls run in `asyncio.to_thread` because
mesa-mcp's OLS client is synchronous `requests`. `card_path` is CLI-only (a tool that reads
local files would expose the server host); iRODS card paths go through `assert_allowed`; parse
errors report line numbers, never content.

## D27. Pre-registration and nested selection

X1–X4, their cells, metrics and the comparison rule R are committed (see
[Pre-registration](#pre-registration-g1)) before any run. With 7 cards, choosing framing,
model or anchor on the full data and then evaluating on the same folds is winner's-curse
selection, so every such choice is re-made inside each outer fold (`selection:"nested"`);
full-data selections are recorded as `exploratory:true` and can never be cited.

## D28. Rank-and-cap and ols_rank

While a task is uncalibrated, steps prune by top-k only. The degraded mode `method="ols_rank"`
proposes the OLS top-1 per group as `proposed` (probs NULL, never auto) through an explicit
outcome branch and a DB CHECK. Without it, the K1 and K2(c) pivots would emit nothing, because
`outcome()` abstains whenever probs are NULL.

## D29. Stream, not store

Cards are read from iRODS into memory (≤1 MiB); runs are exported to
`<project>/.mesa/clm/runs/<run_id>/` and pruned locally only once terminal (plan §7.3);
sidecars are per VM; `artifacts publish|pull --verify` and `labels export|import` move state
through the Data Store; secrets resolve `env|file|keyring|auto`. There are many DE VMs and no
single host (standing preference).

## D30. Frozen label snapshots

The bench reads only `labels snapshot` output, and every cell and manifest records
`labels_sha256`. Curator labels on bench cards are tagged `bench_card=true` and excluded from
pre-registered cells. mesa-anyjev's `labelled_states` keeps the highest weight per state
(`learn/labels.py:377-396`), so a single 1.0 pick would otherwise silently rewrite a test fold.

## D31. Reversibility

`mesa-clm revert` deletes exactly the written triples (with the stored unit, mesa-ducklake D5)
and records history with `op=delete`; neon replace mode moves existing reps to
`proposals/.bak/<ts>/` first; the live-venv bump has a written rollback (plan §7.6). No undo
existed in mesa-anyjev.

## D32. Plugin visibility

Before the live venv bump, a mesa-mcp PR adds a plugin allowlist
`MESA_MCP_SERVER__PLUGINS=<names>|none` (new; c74f3aa has only `STRICT_PLUGINS`), and a
neon-ducklake PR sets it to `none` in the committed template `curation/mcp-mesa-ols.json`.
neon's `HIDDEN_TOOLS` is a static list of the 8fbaedf registry (`curate.py:101-104,324`), so new
plugin tools would change the Opus child's `tools_visible` and with it the teacher corpus.

## D33. Tie-invariant ranking metrics

The vendored AnyJev `bench/metrics.py` (byte-identical, never edited) orders items by confidence
with `np.argsort`'s default, unstable sort. With tied confidences the order of the tied items is
whatever the platform's sort kernel yields, and that order decides which items share an
equal-mass ECE bin and which prefix of the risk-coverage curve they join. The M0 lookup cell
gave ECE 0.1452 on sparky-1 (aarch64) and 0.1473 on x86-64 CI from identical inputs (PR #1).
Lookup probabilities take a handful of distinct values and a collapsed zero-shot CLM head
(issue #15) is almost all ties, so every mesa-clm cell reports `ece`, `cov@5%`, `cov@10%` and
`aurc` from `mesa_clm.bench.metrics`: each item's correctness is replaced by the mean over its
tie group before sorting (with a stable sort). For the risk-coverage curve this is the
expectation over uniformly random tie-breaking; for ECE it is a canonical order-free value. On
tie-free inputs the functions equal the vendored ones exactly (`tests/unit/test_tie_metrics.py`),
so mesa-anyjev's continuous-probability L2 cells stay directly comparable. Accuracy, macro-F1,
Brier and NLL do not sort and still come from the vendored module.

## Implementation notes (M1)

How M1 realised the decisions above where the plan left room. Each note refines a decision or
the plan's wording; none changes a decision (what D21's "interactive CLI" means is amendment
[A2](#a2-2026-09-29--amends-d21); the encoder recipe and auth changes are
[A3](#a3-2026-10-01--amends-d5), [A4](#a4-2026-10-01--amends-d16-d23) and [A5](#a5-2026-10-01--amends-d16-a4)).

- **One `decisions` row per CLM question (D2, plan §4.2 Q4).** A rank_fit group is one
  `/v1/systemone` Choice, so it is one `decisions` row (the answered candidate's `s_c`/`p_fit`
  on the row) with one `decision_options` row per candidate *and* one for the anchor, plus the
  `decision_groups` row, not one `decisions` row per candidate. Rationale: the row is the
  question that was asked; the options carry every per-candidate number (`rank`, `s_c`,
  `p_fit`, `prob`, `raw_prob`, `masked`), and the anchor row keeps "none of these" addressable.
- **Q3 asks all twelve registry ontologies, then masks (plan §4.2 Q3).** One rank over the full
  registry plus the anchor per (column, aspect), masked after scoring by the aspect and by the
  planner's in-play set (`masked=true` rows kept). Rationale: the question (hence its context,
  cache key and `question_key`) stays identical across aspects and plans; the mask is pure and
  idempotent.
- **Q1 and Q2 share one request per column (plan §4.1).** `column.annotate` and
  `column.aspect` render the same `column_state`, so they travel in one request. Rationale:
  "questions sharing a context go in one request"; one embedding of the context.
- **Selecting the degraded method (D28).** `ols_rank` runs for every rank_fit task through
  `decider.tier: ols_rank` (now a `config.Tier`), `annotate(tier="ols_rank")` or
  `--tier ols_rank`, per task through `ols_rank_tasks=`, and per group whenever the CLM record is
  `unavailable` or `truncated`. The closed choices (Q1, Q2, Q7) still go to CLM. Rationale: the
  K1/K2(c) pivots are per task; a whole-run switch is what an outage needs.
- **clm-serve down at the start (D28, plan §7.1).** `annotate --provider clm` asks clm-serve's
  `/health` first (inside the pre-flight below); when it does not answer, tier `auto` with
  `decider.ols_rank_fallback: true` runs `ols_rank`, anything else refuses (exit 1, the plugin's
  `decider_unavailable`); an explicit CLM tier always refuses. Rationale: an operator may prefer
  proposed-only OLS ranks to nothing, but only by saying so; the default never silently
  degrades.
- **`pending_groups` (plan §7.1 MRTR).** Only `term.fits` groups wait for a reviewer: proposed,
  escalated or anchor-won, or answered so far only by an agent (`via='tool'`, flagged
  `agent_answered`), and not settled by a curator, minus a parent whose D24 refinement group
  (which offers the parent too) is asked instead; a group settled (`human`, `rejected`) with no
  override row at all (an imported run) counts as a curator's. `review --pick/--decline` answer
  these groups only; `feedback` answers any group of the owner's runs. Rationale: ontology groups
  build no AVU; asking the parent twice would split one answer across two groups; an agent's
  answer must not hide a group from the curator (A2).
- **`record_human_pick` (D21, A2).** It requires `owner`; `reject` is an explicit "none of these"
  (anchor-positive row plus a negative per offered candidate); `decline` keeps its override row
  and writes no labels, links or group outcome. A curator's answer is final: a different answer
  or a decline afterwards raises `AlreadyAnswered`, the same answer again changes nothing (D1's
  `INSERT OR IGNORE`); an agent's answer is replaced by a curator's, whose accepted link displaces
  the agent's (one accepted link per group), and a different second agent answer is refused.
  `check_answer` runs every check without writing. Rationale: a declined question was seen, not
  answered; without the guard a second answer left two accepted links and contradicting curator
  rows (M1 integration review).
- **Keep-rule drops (D25).** What the dedup and the cap drop gets no link row and is listed in
  `abstained` with the reason `duplicate`, `over_cap` or `avu_unbuildable`. Rationale: a link row
  means "could be applied"; the reason keeps the drop explainable.
- **The second taxon (plan §4.2 Q6).** Q6 keeps the dataset's top two taxa, but the second is
  capped at `proposed` and must out-rank the anchor. Rationale: the policy judged only the
  winner; a runner-up below "none of these" is not a proposal.
- **Failed runs (D11).** A run that raises is committed with status `failed` (what it decided
  so far, its calls) and the error is re-raised. Rationale: the post-mortem needs the rows; the
  caller still sees the failure.
- **Bulk sidecar inserts (D11).** DuckDB 1.5.6 attempts `import pandas` twice for every Python
  value it binds when pandas is absent (RESEARCH.md, "mesa-ducklake"), so a per-row
  `executemany` commit paid that for every cell of every row. `provenance.store.bulk_insert`
  binds one JSON value per chunk and unpacks it with `from_json_strict` (a failed cast is an
  error, never a NULL); a chunk JSON cannot carry falls back to `executemany`. The rows are
  identical (`tests/unit/test_provenance_bulk_insert.py` compares both paths table by table) and
  the data never enters the SQL text.
- **The live fingerprint (D5, K4).** The live provider takes the serving lock the host runs,
  `~/.mesa/clm/serving.lock.json` (the bootstrap's copy), else the checkout's
  `serving/serving.lock.json`, and refuses (`providers.live.ProviderSetupError`) a lock that does
  not verify, a lock whose `schema_sha256` is not the vendored `schema.py` the client renders
  with, an installed lock that differs from the checkout's (the checkout moved and the bootstrap
  was not re-run), and a served head other than `clm-latest`/`clm-raw` until M7. Rationale: every
  record is stamped with the lock's fingerprint, so the lock must be the recipe the host runs and
  the code renders.
- **The annotate pre-flight (D28, plan §7.1).** Before any question `annotate --provider clm`
  runs `providers.live.preflight`: clm-serve's `/health` (unreachable, or `embedder: false`, is
  the only state the `decider.ols_rank_fallback` rule may degrade), then with the keys clm-serve's
  `/v1/models` (the configured `clm.model` served) and the encoder's `/v1/models` against the
  lock (served name, `root`, `max_model_len`, and `owned_by` against the lock's `route`: `vllm`
  for the image, `mesa-clm-fallback` for the in-process fallback). A missing or rejected key, an
  unserved model or another encoder is never degraded (exit 2). Mid-run, a 401, 403 or 404 from
  `/v1/systemone` raises `DeciderRefused` and fails the run (recorded `failed`, exit 1) instead of
  turning every group into an `ols_rank` proposal under the requested tier; transport errors,
  timeouts, 5xx and 422 still fall back per group, and the summary counts the failed calls.
  Rationale: before this, a missing key passed the unguarded `/health` check and an explicit
  `--tier zero_shot` run exited 0 with `ols_rank` proposals only (M1 integration review).
- **Doctor serve mode (plan §6.8).** The live serving probes run under `--serve` or when both
  `mesa-clm-*` user units are active; an unreachable endpoint fails under `--serve` (or
  auto-detected serve mode without `--quick`) and warns otherwise, so `doctor --quick` (the
  future `mesa_clm_health`) stays green during a planned GPU window. Every configured URL passes
  the clients' loopback rule before it is called and is printed redacted. The light probes
  (binds, `/health`, the 401 matrix, model lists, the golden `/v1/systemone` shape) run in any
  serve mode; the slow ones (encoder goldens, the long-input probe, the systemone parity, the
  drift probes) under `--serve` or in serve mode without `--quick`. The 401 matrix asks, without
  a key, every route vLLM's own `build_app` registers for the locked arguments (17 besides
  `/health`: `GET /v1/models`, `/metrics`, `/version`, `/load`, `/ping`; `POST /v1/embeddings`,
  `/pooling`, `/invocations`, `/score`, `/v1/score`, `/rerank`, `/v1/rerank`, `/v2/rerank`,
  `/v2/embed`, `/tokenize`, `/detokenize`, `/ping`; `bench/results/2026-10-01/serving_m1b.json`)
  plus a path no route has, and clm-serve's `GET /v1/models`, `POST /v1/systemone` and `POST
  /v1/rank` (it checks the key per handler, so each route is asked); any answer but 401 fails.
  Rationale: a 401 on one route proves nothing about another (M1-A found five open encoder
  routes behind a green `GET /v1/models` check). Serve mode also reads the **encoder container's
  network namespace** (`/proc/<pid>/net/{dev,tcp,tcp6}` of the container's main process, the pid
  from `docker inspect`): a listening socket on a non-loopback address in a namespace with an
  interface besides `lo` fails, a container it cannot inspect is a warning (A5: `ss` on the host
  never sees the engine's own rendezvous and collective sockets, which have no key); and it warns
  when the **headroom timer** is not active while serving (plan §6.5; the encoder unit pulls it
  in since A5). Every doctor run also checks the **feature store** of the live `encoder_fp`
  (format, the vector recipe it was built under against the live lock's, the export's float16
  gate; a format-1 store or another recipe fails), as plan §6.8's "Stores" asks. The doctor stays
  read-only: it reports permissions and keys, never repairs them.
- **Encoder goldens, long input, systemone parity (plan §6.8).** The plan's "golden cosine
  >= 0.9999" is checked as the A3 recipe allows: the texts of
  `encoder_golden_<encoder_fp>.npz` (the serving home's `goldens/`, else the checkout's
  `.local/serving/`, written by `scripts/serving_probes.py`) embedded one per request must equal
  the reference bitwise (equal within 0.9999 is a warning, below it a failure), and as one batch
  agree with one by one within `1 - 1e-6` when the recipe is batch-invariant. The "5,000-token
  tail probe" embeds a deterministic text of more than the window (8,741 tokens on the live
  encoder) and requires its vector to equal, within 0.9999, the vector of its last
  `max_len - 1` token ids, with `max_len - 1` tokens charged (A4's cap). The "golden
  `/v1/systemone` <= 1e-4" recomputes the golden question on the local route (encoder vectors,
  the pinned head's numpy export for `clm-latest`, raw cosines at scale 100 for `clm-raw`) and
  compares clm-serve's probabilities; without the head export only `clm-raw` is checked. The
  records of the live runs are `bench/results/2026-10-01/doctor_serve.json` (A3/A4) and
  `doctor_serve_m1c.json` (A5: 34 ok, the A3 drift warning, 0 failures). Rationale: under
  A3's kernels these hold exactly (M1-A's batch dependence made the plan's cosine tolerance
  necessary; it is kept as the warning band).
- **Upstream drift stays a warning (plan §6.8; the question A3 left to the doctor's owner).**
  The quickstart and tides references stay issue #15's and the model card's, with their
  tolerances; the doctor prints the unrounded values and differences. A deliberate kernel change
  rotates `encoder_fp` (A3) and records these probes under the new fingerprint (for
  c3b3d5e1a283, `bench/results/2026-10-01/serving_m1b.json#/drift`), so a warning whose values
  equal the current fingerprint's record is that recipe, and one that departs from it is drift.
  The doctor does not encode the record: a reference that moved with each recipe would stop
  detecting upstream drift.
- **Plan deviations kept for M1.** `serve keys --rotate` replaces both keys and prints the
  `systemctl --user restart` of both units instead of restarting them (plan §6.4): starting and
  stopping the units is the operator's, never a side effect of a key verb. The doctor has no
  `--only` (plan §6.8) and the headroom timer runs `deploy/bin/mesa-clm-check-headroom` (a shell
  check of `MemAvailable` that also names active CARC backends) rather than `doctor --quick
  --only host,gpu_budget` (plan §6.2): the same script is the encoder unit's start check (32 GiB)
  and the timer's (8 GiB), and the units reference only the serving home (`~/.mesa/clm/bin`, the
  serve venv), never a mesa-clm checkout or its venv. Both are revisited with the
  `mesa_clm_health` tool in M3. Since A5 the timer starts and stops with the encoder unit
  (`Wants=` there, `PartOf=` on the timer), because a timer the operator had to start separately
  was found stopped while both units served; none of the units is enabled at boot.
- **Owner-only state (D11, D29; A2).** Everything mesa-clm creates for its state is owner-only
  whatever the process umask (`mesa_clm.perms`): missing directories are created `0700` component
  by component (`chmod` after `mkdir`), files `0600`; every write also tightens the sidecar's
  existing `locks/` directory, lock file, DuckDB file and `.wal` (DuckDB creates the last two
  with the umask's mode, so until the chmod only the `0700` directory protects them), while a
  read changes no existing mode;
  the same for the label store, the feature store and its npz export, run exports
  (`provenance export`: the run directory `0700`, Parquet and manifest `0600`), `annotate
  --out`, and the missing parents of `secrets/`. Existing directories the user made (a sidecar
  under `/tmp`, `~/.mesa`) are never chmod-ed. The doctor's `permissions` check warns on group or
  other bits anywhere in the data home's `locks/` and `secrets/`, the sidecar and the feature
  stores and prints the `chmod` that fixes them. Rationale: the sidecar holds owners, curator
  labels and card-derived states, and A2's trust boundary is the account that owns it.
- **Default key files (plan §6.4).** `clm.api_key_file` and `encoder.api_key_file` default to
  `~/.mesa/clm/secrets/clm.key` and `encoder.key` (the serving home's `secrets/`, what `serve
  keys --init` writes) when those files exist, in the `auto` and `file` secrets modes only, after
  an inline value, a configured file and (in `auto`) the keyring; an explicit empty value turns the
  default off. The default is resolved when the key is read, not stored in the configuration, so
  `config_sha256` is unchanged by it, and the file is read with the 0600 and owner rules (a loose
  default fails loudly). The doctor names it as "(default)". Rationale: on the serving host the
  default provider is `clm`, and a missing key used to surface only as 401s.
- **Exports are stamped last (D29, plan §7.3 step 9).** `provenance export` writes the copy with
  the new `exported_at` on its run row, verifies it as an importer would (`read_export`), and only
  then stamps the store's run row; a failed export (an unwritable `--out`, a full disk, a
  read-back mismatch) leaves the store unchanged and exits 1 naming the path. Rationale: `prune`
  trusts `exported_at` to mean "a copy exists".
- **One DSN parser.** `config.duckdb_path` reads `duckdb:///<path>` and a bare `<path>.duckdb`
  for every verb (`provenance.store.duckdb_file` delegates to it), so the review verbs never
  create a database a bare path names, and `serve lock --check` reports an unreadable lock as a
  failed check. CLI usage errors (a blank `--actor`, a negative `--ttl-days`, an unsupported DSN
  for `provenance import`, end of input at the review prompt) end in `mesa-clm: <message>` with
  exit 1 or 2, never a traceback.

- **The feature store (plan §5.2; D5, D11, D23, D30).** One store per `encoder_fp` at
  `<features.dir>/<encoder_fp>/features.duckdb` (`learn/features.py`), opened per operation under
  its own flock like the sidecar. Beyond the plan's `texts` and `vectors` it keeps `meta` (the
  recorded `encoder_fp`, dimension and encoder spec, so a store under another fingerprint or
  width is refused), `vectors.fp16_cos` (the float32-to-float16 round-trip cosine per vector,
  the export's gate), `proj_heads` (which head a `clm_model_fp`'s projections came from, the
  head id a sha256 over all its weights; a second head under the same fingerprint is refused,
  K4), and truncated texts with their exact token count, never embedded (a truncated context is
  an abstain, D23). **Plan deviation (format 2):** `vectors` holds the **float32** vector
  (`vec32`), not the plan's float16 blob; float16 appears only in the `TextCache` export. Scores
  from float16 vectors missed X1's pre-registered "cross-check vs clm-serve ≤ 1e-4 in probs" by
  more than ten times (1.31e-3 `clm-latest`, 3.59e-3 `clm-raw` over 200 groups,
  `bench/results/2026-10-01/features_build.json#/rerun/crosscheck`): Qwen3-8B's last-token vectors
  carry three large dimensions where the float16 spacing is 2.4e-4, and CLM's scale of 100
  amplifies it; the plan's round-trip gate (cosine ≥ 0.9999, met at 0.9999999) bounds angles,
  not scores. From float32 vectors the same cross-check passes at 5.2e-6 and 7.47e-5
  (`bench/results/2026-10-01/x1_crosscheck.json`; `clm-raw`'s narrower margin is clm-serve's
  float32 cosine accumulation against the scorer's float64, the vectors being bitwise equal), so
  the pre-registered gate is kept. The store also records, when it is created, the serving lock's
  **vector recipe** (`ServingLock.vector_recipe()`: image digest, encoder spec, `vllm serve`
  arguments, non-secret environment, truncation cap) with its sha256, and stamps each vector with
  the `lock_sha` it was embedded under; every read and write through `FeatureStore.for_lock` (all
  CLI paths, `scripts/x1_crosscheck.py`, M2's X1) refuses a store recorded under another vector
  recipe, and a format-1 store is refused outright (it cannot be converted, only rebuilt). The
  first write of a vector still wins, which the recipe check makes safe. Rationale: the store is
  the input of every X1/X2 cell, so it must refuse what would mix recipes, and `encoder_fp` alone
  does not name everything a vector depends on: a new image with the same flags, `--enforce-eager`
  dropped, or K0 step 1's NGC image all keep it, and a kernel-level change moves vectors to min
  cosine 0.99937 (`bench/results/2026-10-01/batch_invariance.json#/secondary`), a few hundredths
  in probability at scale 100. The rebuilt store is
  `bench/results/2026-10-01/features_build_m1c.json`.
- **What the manifest embeds (plan §5.6 X1, X2).** `features.manifest` reads a frozen snapshot
  label-free (no `label` or `weight` column) and renders every text through `framings` and
  `render` only. `state_json` is canonical JSON (sorted keys, identity only, plan §4.3), so the
  builders' key order is restored first (`builder_ordered`, templates taken from the `states`
  builders themselves): rendering the sorted state would send text the pipeline never sends (a
  card header starting with "columns:"); a test rebuilds every context from the fixture cards and
  gets identical text. F4, F7 and F9 ask one request per target over its labelled candidates
  (`framings.build_request`, then `render.build_pairs`; `column.ontology_fits` states take the
  task scope `column`, as Q3 does). F1 and the X2 specs use the stored labelled mesa-anyjev state
  with its `n_candidates` (0 for the neon-avu-eval silver), the state the label was recorded on.
  PR #13's logistic regression embedded the state text with the question appended and never a
  candidate text, so for mesa-clm's pair questions **S1** is `render.state_text(anyjev state,
  task question)` (the same text as F1's context, stored once) and **S1ns** the same state with
  no question; both embed the candidate inside the state, because a probe on the target context
  alone cannot answer a pair question. Rationale: plan §5.3 named the specs without fixing which
  state "joint" meant.
- **Building, exporting, projecting (plan §5.2).** `features build` does not go through the live
  provider (that would also need a servable `clm.model`): it takes `EncoderClient.from_config`
  and the live lock, refuses an `encoder.max_len` other than the lock's (exit 2), an encoder that
  is not the lock's (the pre-flight's encoder check, then the container check below), a store
  recorded under another `encoder_fp`, format or vector recipe and approximate token counts (exit
  1), embeds one text per request by default (bitwise
  reproducible on every recipe) and commits every 64 texts and before reporting a failure, so a
  re-run embeds only what is missing. `features project` projects every stored vector on both
  sides (a text is stored once whatever its role) and refuses a head export whose
  `source_sha256` is not the lock's head; `--model clm-raw` has nothing to project. `features
  export-npz --out DIR` writes CLM's `TextCache` file (`finetune.py:422-446, 467-468` at
  `bb42c6c5`: `sha1(text)` keys as `<U40`, float16 `[N, 4096]` cast from the stored float32
  vectors, plain `np.savez`), refused below the 0.9999 round-trip gate. The live builds under
  c3b3d5e1a283 are `bench/results/2026-10-01/features_build.json` (format 1) and
  `features_build_m1c.json` (format 2, on the A5 recipe: every float16 cast bitwise equal to the
  format-1 vectors, the export byte-identical).
- **The X1 anchor variant: rule (6) withdrawn before G1 (plan §4.3, §5.6; D27).** X1's decision
  rule (6) chose "the anchor variant by NLL" and plan §4.3 says X1 "also tests a longer variant"
  of the registry anchors, but no variant text was ever written into `registry.ANCHORS` or the
  framings, so rule (6) had nothing to compare. Writing one now would not be a pre-registered
  choice: the live zero-shot stack has already been run on a bench card and its outputs read
  against that card's silver labels and its anchor (the annotate smoke, disclosed under "G1
  freeze"), so any text chosen after that could be steered by what was seen. Rule (6) is
  therefore withdrawn from the pre-registration before G1 (its words stay, struck through), X1
  runs with the two registry anchors only, and the 16-cell grid, rules (1)-(5) and the MDE are
  unchanged. A longer anchor can still be tested later as a new framing (a new `question_key`),
  by amendment, with its cells `exploratory:true` (D27). Rationale: a pre-registered rule with
  nothing to compare cannot be run, and the only way to give it something now is a post-hoc
  choice.
- **Bench-card membership and the fold filter (D30).** The bench cards are a fixed list, the
  seven neon-avu-eval tables of U2 (`learn.labels.BENCH_CARDS`, equal to the cards of the frozen
  snapshot `bench/snapshots/2026-09-29.parquet` and of the committed eval copy; a test pins
  both), not inferred from the silver labels a sidecar happens to hold: the serving host's
  sidecar holds none, so a curator answer on a bench card was tagged `bench_card=false` there.
  `DecisionService` tags curator rows on those cards (plus any card with silver labels in the
  store) and, failing closed, on a run whose card name is unknown; a caller may pass its own
  list (only tests do). The bench drops rows that may not enter a pre-registered fold **before**
  the per-identity highest-weight selection (`labelled_targets(fold_only=True)`):
  `fold_eligible=false` rows and curator rows on a bench card, whatever their tag, are counted in
  `meta["excluded"]`. Rationale: selecting first let a tagged curator row at 1.0 win its identity
  and then be dropped, deleting the silver item (term.fits 285 → 284 items in a reproduction),
  and an untagged one replace the silver label in a test fold; a curator answer must never change
  a pre-registered item's label or presence (`tests/unit/test_bench_tasks.py`).
- **The encoder container check (D5, K4).** `/v1/models` answers the same for every container of
  the pinned image, whatever its kernels or flags: a container like the batch-invariance
  experiment's B0 arm (the same image without `VLLM_BATCH_INVARIANT`) would pass it while its
  vectors differ (min cosine 0.99937, `batch_invariance.json#/secondary`). Before `features build` writes a vector and before an
  annotate run asks a question, `providers.live.container_check` compares the running encoder
  container with the lock (`serving.check_encoder_container`: image, arguments, environment,
  mounts, network); a container that departs from it is refused (`features build` exit 1,
  annotate exit 2, never degraded), and when docker cannot be asked from the session the
  result says "not verified" in the command's output (annotate's `--out` JSON carries it under
  `preflight`). The fallback server refuses `--batch` other than 1, since its fingerprint says
  `serial` (A3). Rationale: a fingerprint is only as good as the check that the running recipe
  is the fingerprinted one. The run row itself does not record the check (that needs a sidecar
  migration, left for M3).
- **K0 (plan §8, the M1-A gate).** K0 stops at step 1: the pinned `vllm/vllm-openai` v0.27.1
  serves Qwen3-8B pooling on the GB10 (sm_121) within the 0.20 share
  (`bench/results/2026-09-29/serving_m1.json`, `bench/results/2026-10-01/serving_m1c.json`), so
  neither the NGC image (step 1's alternative) nor the in-process fallback (step 2) is needed,
  step 3 (pause and escalate) does not arise, and step 3b's CPU throughput was not needed for the
  bench. The fallback still passes its two cosine parity gates at batch 1
  (`bench/results/2026-10-01/fallback_parity.json`).
- **The CARC allocation request (plan §6.5; a plan §1 user go-ahead).** The written request for
  the 0.20 share is drafted in the serving runbook (`docs/deploy/serving.md`, section 6) with the
  measured totals of the A5 recipe; it has **not** been sent, because sending it needs the user's
  go-ahead (plan §1). Until it is accepted the units stay not enabled at boot.

## Pre-registration (G1)

### G1 freeze

The pre-registration below (the LOCO folds and cells, rule R, lookup as a probability model,
the citation test, the minimum detectable effect, X1–X4 and the kill and pivot criteria K0–K4)
is **frozen as of the M1 merge commit**, the commit that merges `feat/m1-pipeline` into `main`.
From that commit on it changes only by amendment, and an amendment made after an experiment has
run marks every affected cell `exploratory:true`.

**Registered in M0, frozen at G1.** The text below is copied from plan §5.4, §5.6 and §8 and is
committed verbatim at gate G1, before any M2 run, with one edit made before G1 and shown in
place: X1's decision rule (6) is struck through and withdrawn (implementation note "The X1 anchor
variant"). After G1 it changes only by amendment, and an amendment made after an experiment has
run marks every affected cell `exploratory:true`.

**Disclosure: every look at labelled bench data before G1.** No pre-registered cell has been
computed. What M0 and M1 did with the seven bench cards' silver labels, or with model outputs on
those cards, and what was seen:

1. *The M0 controls* (`bench/results/2026-09-29/baselines.json`, plan §5.4's verified controls):
   lookup LOCO accuracy 0.772 / 0.800, leave-one-product-out 0.723 / 0.721, the novel-key
   subsets 159 and 99 items, `lookup_prob`'s novel-key AUROC 0.424 / 0.396 (term.fits /
   column.ontology_fits). These are the baselines the rules compare against, computed as planned.
2. *The MDE simulation* (`bench/results/2026-09-29/mde.json`): the realised class counts per task
   and card, and simulated power; no model output.
3. *AnyJev L2's held-out predictions* (`bench/baselines/anyjev_l2_2026-09-29.json`, X2's paired
   baseline): another model's per-item predictions against the silver labels, reproducing
   mesa-anyjev's committed cells exactly (term.fits accuracy 0.765, column.ontology_fits 0.837,
   and the per-card accuracies).
4. *Label-free runs over the labelled rows' texts*: the collapse spikes
   (`bench/results/2026-09-29/collapse_spike.json`, `2026-10-01/collapse_spike.json`), the
   feature-store builds (`features_build.json`, `features_build_m1c.json`), the X1 cross-checks
   (`features_build.json#/rerun/crosscheck`, `x1_crosscheck.json`) and the serving and
   batch-invariance probes' 50 systemone pairs (`serving_m1.json`, `serving_m1b.json`,
   `serving_m1c.json`, `batch_invariance.json`): they compare encoders, routes and stores on the
   snapshot's texts; no label was read and no score was set against one.
5. *The annotate smoke* (`bench/results/2026-10-01/annotate_smoke.md`, run 1): the live zero-shot
   stack on the bench card DP1.10003.001.brd_countdata, its outputs read against that card's
   silver labels and the anchor. Seen: zero-shot `column.annotate` answered No to all 14 of the
   card's non-identifier columns, where the silver labels mark 8 labelled columns Yes and 4 No;
   in the SRER biome group the top three candidates were near-ties (0.99880, 0.99853, 0.99834);
   the second taxon scored 0.5557, just above the anchor's 0.5; the D24 refinement replaced Aves
   by an NCBI bookkeeping child. This is zero-shot behaviour on one bench card, of the tasks
   column.annotate and term.fits, relative to the registry anchor: it is why X1's anchor variant
   was withdrawn rather than written now (the text could not be chosen blind).
6. *The live engine test* (`tests/engine/test_doctor_live.py`) annotated the same bench card at
   zero shot on 2026-10-01, asserting only the run's structure (no comparison with labels); it
   now annotates the non-bench SRER card `DP1.00004.001.BP_30min` (plan §9's smoke card), and
   live smokes and latency runs use non-bench cards only until the M2 cells exist
   (`annotate_latency.json`).

The hermetic test suite runs the deterministic fake provider (hashed n-grams, never evidence) on
the fixture cards; it says nothing about the model.

### LOCO folds, leakage controls and cells (plan §5.4)

- Folds: `for card in sorted(fold_eligible_cards)`; guards 30 train / 5 held-out per class (all
  14 term/ontology folds pass at 0.5, verified) + floors 100 (calibrated) / 40 (probe/head).
- Teacher rows: train drops teacher rows whose `leak_group` equals the held-out product;
  property test: no `fold_eligible=false`, same-product teacher or `bench_card` curator row
  reaches a test fold. Predictions pooled, never fold-averaged.
- Nested selection (D27): every choice among framings/models/anchors/specs/hyperparameters is
  re-made inside each outer fold by grouped inner CV on the 6 training cards; cells record
  `fold_choices`.
- Cell fields — identity: `question_key, fingerprint, labels_sha256, feature_spec,
  label_sources, teacher, teacher_in_test:false, masked, loco, selection, pre_registered,
  exploratory, servable, n_folds, skipped_folds, fold_choices` (plus `labels_content_sha256`,
  the ingestion-independent hash of the label rows, so a test can prove a committed cell was
  re-derived from the same rows while `labels_sha256` names the frozen snapshot); counts: `n, n_neg, n_nonmodal,
  class_counts`; metrics: `acc, macro_f1, brier, nll, ece, cov@5%, cov@10%, aurc, auroc[cluster
  CI], threshold_cp{0.05,0.10}`; baselines: `majority_acc, lookup_acc, lookup_nll,
  novel_key{n, acc, auroc, nll}, lopo{acc, nll}, beats_lookup_novel`; diagnostics:
  `mean_state_cos, calls, input_tokens, ms_per_decision`.

### Rule R (plan §5.4)

**Rule R** ("A ≻ B on m") needs both: (i) the one-sided 95% lower bound of a card-cluster
paired bootstrap of Δm (resample held-out cards, B=2000, seed 0) > 0; (ii) A beats B on
≥⌈0.8·m_c⌉ of the m_c held-out cards with ≥10 evaluable items (m_c<4 →
`insufficient_clusters`). "A ≽ B within δ" = cluster LB > −δ. Effect floors (AUROC ≥0.60, ECE
≤0.08) are always paired with a CI condition.

### Lookup as a probability model (plan §5.4)

`lookup_prob` = Laplace-smoothed (α=1) key frequency over training cards, falling back to the
training prior for unseen keys (so on novel keys AUROC 0.5, NLL = prior cross-entropy).
**`beats_lookup_novel` := novel-key AUROC cluster-LB > 0.5 ∧ novel-key NLL ≻ `lookup_prob`
(rule R).** Novel-key accuracy vs majority (0.717 term.fits, 0.667 ontology_fits; verified) is
reported, not gated. Controls, reproduced in M0 by `bench baselines`
(`bench/results/2026-09-29/baselines.json`, cells `neon_term_fits.baseline.lookup_prob` and
`neon_ontology_fits.baseline.lookup_prob`): lookup LOCO 0.772/0.800; leave-one-product-out
0.723/0.721; novel-key subsets term.fits 159 (45 Yes/114 No), ontology_fits 99 (33/66). The
lookup key is `(task, scope, target, option_key)` with target = column name, site code or `''`
for dataset scope; when training cards tie on a key, the label of the alphabetically first card
wins (`Counter.most_common(1)` over card-ordered rows), which is the rule that reproduces those
numbers (6 tied keys in term.fits, 2 in ontology_fits; a symmetric tie rule gives 221/151).
Observed: `lookup_prob`'s novel-key AUROC is 0.424 (term.fits) / 0.396 (ontology_fits), not
exactly 0.5, because the per-fold training prior anti-correlates with the held-out card's label
mix; the gate stays "cluster-LB > 0.5".

### Citation test (D8, plan §4.7)

Per numeric `auto`: `cite` matches
`^bench/results/\d{4}-\d{2}-\d{2}/[\w.\-]+\.json#<task>\.<tier>\.<framing>$`; the cell has
`loco, pre_registered, selection:"nested", exploratory:false, n_folds≥5, teacher_in_test:false,
servable`, `masked` equal to serving, `labels_sha256` equal to a committed snapshot; guards
`n_neg≥30` (rank_fit) or `n_nonmodal≥30` (choice); fingerprint and `question_key` equal the live
ones; `fold_choices` agree with A1 in ≥5/7 folds; `beats_lookup_novel == true`; `auto ≥
cell.threshold_cp[t.risk]` (smallest t whose pooled held-out {stat ≥ t} has ≥30 items and a
one-sided 95% Clopper-Pearson error upper bound ≤ risk). With `auto_requires_audit`, apply also
needs a passing `audits` row: ≥50 would-be-auto decisions from ≥3 non-bench cards,
curator-reviewed, CP-95 error upper bound ≤ 2×risk.

### Minimum detectable effect

`bench mde` simulates the minimum detectable effect at the realised n, base rates and 7 cards.
Committed in M0 as `bench/results/2026-09-29/mde.json` (200 simulated datasets per grid point,
grid 0.55–0.85 step 0.025, B=2000, seed 0; model A a calibrated binormal scorer at the grid
AUROC, model B the constant training prior, i.e. `lookup_prob` on novel keys; power = the
fraction of simulated datasets on which rule R passes; MDE = the smallest grid AUROC with power
≥ 0.8). Every number below names that file.

| population | n (pos/neg) | counting cards | AUROC gate: MDE (Δ over 0.5) | NLL gate: binormal AUROC needed |
|---|---|---|---|---|
| term.fits, full | 285 (86/199) | 7 | 0.625 (0.125) | 0.725 |
| term.fits, novel keys | 159 (45/114) | 7 | 0.675 (0.175) | 0.800 |
| column.ontology_fits, full | 190 (76/114) | 7 | 0.650 (0.150) | 0.775 |
| column.ontology_fits, novel keys | 99 (33/66) | 4 | 0.750 (0.250) | 0.850 |
| column.annotate, full | 98 (63/35) | 6 | 0.700 (0.200) | 0.825 |
| column.annotate, novel keys | 39 (24/15) | 1 | `insufficient_clusters` at every effect | same |
| column.aspect, avu.value_kind | 60, 278 | — | not applicable (8 and 4 classes) | — |

Stated in advance: `beats_lookup_novel` for column.annotate cannot pass rule R at any effect
(one card has ≥10 novel-key items), so column.annotate stays proposed-only until more cards
exist; ontology_fits on novel keys needs a large effect (AUROC 0.75) with four counting cards.

### X1 framing A/B (plan §5.6)

- Grid {F1 control, F4, F7, F9} × {`clm-latest`, `clm-raw`} × {term.fits, ontology_fits}.
- F1 noul per candidate over anyjev `candidate_state` + `Q_TERM_FITS`; F4 Choice + anchor with
  a task-sentence `instructions`; F7 state-only `target_state` + candidates + anchor; F9 "NEON
  dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:" + candidates
  + anchor (shared fixed tail → collapse diagnostic reported explicitly).
- Scoring offline from the cache, s_c = 100·(zs·zc − zs·za); 200-pair cross-check vs clm-serve
  ≤1e-4 in probs. Controls: within-card state shuffle; candidate-only probe.
- Metrics: pooled AUROC (cluster CI); ΔAUROC(real − shuffle); LOCO-Platt acc/ECE/NLL; collapse
  diagnostic (raw + projected mean pairwise cosine; report only); calls/tokens per target; p50
  latency.
- Decision rule (`bench framing --decide`, per task): (1) F4/F7/F9 arm qualifies if AUROC
  cluster-LB > 0.5, AUROC ≥0.60 and ΔAUROC(real − shuffle) ≻ 0; (2) winner = lowest LOCO-Platt
  NLL; if not ≻ a more cacheable arm, take the more cacheable (F7, F9 > F4); F7/F9 tie → fewer
  encoder tokens/target, then F7; (3) model `clm-latest` unless `clm-raw` ≻ it on NLL; (4) F1 ≻
  all eligible → amendment reopening D3; (5) nothing qualifies → K1; ~~(6) anchor variant by
  NLL~~ *(withdrawn before G1: no variant text was registered before bench outputs were seen;
  implementation note "The X1 anchor variant")*.
- Nesting: run on all 7 cards to pick the production A1 framing (`exploratory:true`) and again
  inside each outer fold for citable `selection:"nested"` cells.

### X2 baselines (plan §5.6)

Majority, `lookup_prob`, novel-key, LOPO; PR #13 replica
`StandardScaler+LogisticRegression(C=1, max_iter=2000)` on `joint4096@S1/@S1ns`; per-item
AnyJev L2 held-out predictions via `scripts/anyjev_l2_predictions.py` (feasibility checked in
M0; run in the M1-A GPU window, sequential with vLLM). Without per-item data the comparison is
unpaired vs `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json` (term.fits 0.765/0.058/0.088;
ontology_fits 0.837/0.078/0.547) and K2(a) is unattainable.

### X3 tier sweep (plan §5.6)

zero_shot + calibrated on A1; probe specs × fitters (nested); head (M7).

### X4 teacher ablation (plan §5.6)

Off vs (0.5/0.3) vs (0.3/0.1) on identical folds, scored on silver and silver-minus-Opus; keep
teacher labels only if on ≻ off on novel-key NLL for both. With 13 usable items the expected
effect is ≈ null, stated in advance.

### Kill and pivot criteria (plan §8; none reopens U1–U4)

- **K0 serving (M1-A):** (1) vllm-openai 0.27.1 pooling fails on sm_121 → NGC digest; (2) →
  in-process fallback; (3) none fits at ≤0.20 → pause GPU work, escalate to CARC; (3b) measure
  CPU bf16 throughput in the serve venv — if it can build the X1/X2 feature set within 48 h,
  build features on CPU and run M2 offline.
- **K1 no state signal (M2):** if no arm qualifies for a task on either model, its
  zero_shot/calibrated tiers become audit-only; proposals use `ols_rank` (`proposed`, D28)
  until a probe is promoted; M4 goes probe-first.
- **K2 value per task (M4, best servable tier, nested cells):** (a) auto-eligible =
  `beats_lookup_novel` ∧ paired non-inferiority to AnyJev L2 on the full set (Δacc cluster-LB >
  −0.02 + card-sign condition) ∧ ECE ≤0.08 with cluster-UB ≤0.12 (needs the paired AnyJev
  dump); (b) proposer-only = `beats_lookup_novel` only → ship proposed-only, docs recommend
  mesa-anyjev for auto; (c) tier killed → `ols_rank` proposals, investment pauses until ≥200
  curator labels or CLM-35B. If a `clm-raw` probe ≽ a `clm-latest` probe, RESEARCH.md records
  that CLM's head adds nothing and the default becomes encoder + local probe (clm-serve stays
  deployed for zero_shot/head, U3). 0.1.0 ships proposed-only whatever K2 says and always
  proposes something (`ols_rank` at worst).
- **K3 head (M7):** promote only if ≻ probe on NLL and ≽ on acc within 0.01.
- **K4 drift:** any schema/head/image/encoder sha mismatch fails doctor; the provider refuses
  until the task is re-benched.

## Amendments

Format for each entry: `### A<n> (YYYY-MM-DD) — amends D<k>` followed by the new decision, the
evidence (results JSON or RESEARCH.md fact) and which cells, locks or artifacts it rotates.
**A1 is reserved** for the production framing chosen by X1 (M2): the pre-registration and plan
§8 refer to it by that number ("the A1 framing", "fold_choices agree with A1"), so amendments
made before it take the numbers after it.

### A2 (2026-09-29) — amends D21

**Decision.** "The interactive CLI" in D21 means the `mesa-clm` command line run under the
account that holds the sidecar **with an interactive terminal on stdin** (`sys.stdin.isatty()`).
Only then does it record `via='cli'` and therefore curator labels (the pick at 1.0, implicit
negatives at 0.7, an explicit none as an anchor-positive row plus a negative per offered
candidate):

- the interactive `review` walk requires a terminal and refuses to start without one;
- `review --pick GROUP=OPTION_KEY|none --decline GROUP` and `feedback --action
  pick|reject|decline` record `via='cli'` at a terminal. Run without one (a script, a pipe, an
  agent's shell) they are recorded exactly like a plain tool call: `via='tool'`, label source
  `agent_pick`, weight 0, `fold_eligible=false`, links `accepted_by='agent'` (a link built for
  the answer carries the source `mesa-clm:feedback:tool`), and the verb says so on stderr before
  it records anything;
- `labels import-anyjev` takes the source file's `curator` and `curator_implicit` rows (which
  mesa-anyjev's defect (g) let an agent mint) as `agent_pick` at weight 0 unless
  `--trust-curator` is given, which needs a terminal too; trusted curator rows on a bench card
  are tagged `bench_card` (D30).

Every one of them acts as `--actor` (default `$USER`) and refuses a run whose `owner` differs;
on a shared Postgres sidecar `explain`, `review` and `feedback` act only as the OS account
(`--actor` must equal it), because there any database user could pass another owner's name.
`review --pick/--decline` answer the run's pending groups only, each at most once per call, and
check every answer before recording any. A curator's answer settles a group: a different
second answer, or a decline, is then refused (M1 has no amend flow, and the D1 `INSERT OR IGNORE`
would silently keep the first answer's rows), the same answer again is idempotent. An agent's answer does not settle
it: the group stays pending (`agent_answered`), a curator's answer replaces it (the agent's
accepted link goes back to `proposed`, so a group never has two accepted links; its weight-0
rows stay as the audit trail), and a different second agent answer is refused.

**Why.** Defect (g), which D21 closes, is an answer given by an agent on someone's behalf. A
first version of this amendment (unshipped) let `review --pick` and `feedback` record
`via='cli'` from any process of the account, which let a script or an agent with a shell mint
curator-grade labels, fold-eligible, at 1.0 (found in the M1 integration review). Requiring a
terminal keeps the curator's own tool (the interactive walk) and turns every unattended use into
what it is, an agent's answer. The CLI's trust boundary is still the OS account: the sidecar is
a per-user file under `~/.mesa/clm/`, owner-only whatever the umask (D11; DESIGN implementation
notes, "Owner-only state").

**Residual risk.** `isatty()` is a speed bump, not a boundary: a process under the curator's
account that deliberately allocates a pseudo-terminal (`script -qc`, `expect`, a pty library)
passes it, and any process under that account can also write the sidecar file directly. A
shell-capable agent running as the curator is therefore still trusted like the curator if it
goes out of its way; the protection is against the ordinary, unattended path. Environment
markers (an agent's `CLAUDECODE`, say) are not consulted: they are set by the agent and unset as
easily. Closing the hole needs a separate identity for the curator (an MRTR elicitation answered
in the client, M3, or a sidecar the agent's account cannot write). On a shared Postgres sidecar
curator labels are only as trustworthy as its database roles: mesa-clm does not check that the
connecting role belongs to the OS account.

**Evidence and rotations.** `tests/unit/test_cli_m1.py` (curator labels at a terminal, agent
labels without one with the stderr notice, the terminal requirement of the interactive walk,
pending-only and once-per-call answers, a curator answer superseding an agent's and then final,
`feedback` without a terminal, the Postgres actor rule), `tests/unit/test_feedback_agent_pick.py`
(the service rules), `tests/unit/test_label_identity.py` and `tests/unit/test_cli.py` (the import
demotion and `--trust-curator`). No framing, lock, cell or artifact rotates; no label existed to
re-tag (the hermetic fixtures carry no curator rows, and mesa-anyjev's `.local/labels.duckdb`
on the serving host holds none either: RESEARCH.md, "Labels and evaluation assets").

**History.** The first version of this amendment was pushed on the feature branch
`feat/m1-pipeline` in commit `8962c58` (2026-09-29) and never reached `main`. Its decision read:
"'The interactive CLI' in D21 means the `mesa-clm` command line run under the account that holds
the sidecar: `review` (interactive), its non-interactive form `review --pick GROUP=OPTION_KEY|none
--decline GROUP`, and `feedback --action pick|reject|decline` all record `via='cli'` and
therefore curator labels", with the residual risk that "an agent with a shell under the
curator's account can run these verbs and mint curator labels". The text above **supersedes it
before the M1 merge** (the change the "Why" paragraph describes: `via='cli'` only with a terminal
on stdin), so the register on `main` never carries the first version; it is recorded here because
the register is append-only and that commit was published.

### A3 (2026-10-01) — amends D5

**Decision.** `encoder_fp` also names whether a vector depends on what it was batched with.
`EncoderSpec` gains `batch_invariance ∈ {none, kernels, serial}`, hashed only when it is not
`none`, so every fingerprint of a recipe that never set it (the fake stack's included) is
unchanged: `none` is the M1-A recipe, whose vectors do depend on their batch; `kernels` is vLLM's
batch-invariant mode (`VLLM_BATCH_INVARIANT=1`: a persistent Triton matmul without split-k,
batch-invariant RMSNorm, softmax and `bmm`, FlashAttention with one split); `serial` is one
sequence per forward pass (the fallback at batch 1, or vLLM at `--max-num-seqs 1`). The vLLM route
runs `kernels`; the in-process fallback runs at batch 1 and is `serial`
(`mesa_clm.serving.FALLBACK_ENCODER`). The serving lock gains a `recipe` block that
`serving_lock_sha` covers: the whole `vllm serve` argument list, the container's non-secret
environment, the bearer guard of A4 with its sha256 and read-only mount, and the clients'
truncation cap. `verify_serving_lock` compares it with the running container (arguments; the
environment through a `docker inspect` template that prints the values of the locked names only,
never the key; the guard's mount), and a lock whose recipe contradicts its encoder spec (window,
cap, prefix caching, batch-invariance mechanism, middleware) is refused when it is parsed.

**Why.** In M1-A a text's vector moved by up to 1.3e-4 in cosine with its batch, so clm-serve's
`/v1/systemone` and the local route disagreed by 0.0434 in probability at CLM's scale 100 while
the head projection matched to 6.3e-7 (`bench/results/2026-09-29/serving_m1.json`), and the
fallback failed its parity gate at batch 8 (min cosine 0.998975,
`bench/results/2026-09-29/fallback_parity.json`). A fingerprint must name everything a vector
depends on, and batching did not appear in it.

**Evidence.** The experiment `bench/results/2026-10-01/batch_invariance.json`, with its adoption
rule set before the first arm ran but **not tamper-evidently**: the rule was written into
`scripts/batch_invariance.py`, which was neither committed nor hashed into the arm records before
the arms ran (the file is dated 08:51:48 MDT, after both arms, 08:39-08:50; the arm files carry
no pre-registration block, the verdict step copied it from the script). The margins make the
verdict insensitive to where its new thresholds were set: B0 misses the cosine criterion by a
factor of about 100 and the ≤ 1e-4 systemone gate (fixed in plan §5.6 and §8 since 2026-09-28) by
377-591 times; B1 meets the cosine criterion by more than eight orders of magnitude, that gate at
3.9e-6 and 6.6e-5, and the slowdown criterion at 1.16× of 4×. The script enters the repository
with the M1 merge; a future pre-registered experiment commits its rule, or writes the rule's
sha256 into every arm record as it is measured, before the first arm runs. With the old
kernels (B0) the four request patterns agree only to 1 − 1.05e-4 and clm-serve differs from the
local route by 0.0377 / 0.0591 (clm-latest / clm-raw); with `VLLM_BATCH_INVARIANT=1` (B1) they
agree to 1 − 3.3e-15 and differ by 3.9e-6 / 6.6e-5, at 1.16× M1-A's latency, so B1 is adopted and
B2 (serial) was not needed. On the A3/A4 recipe `bench/results/2026-10-01/serving_m1b.json` repeats
it (systemone 3.9e-6 / 6.6e-5; goldens 20/20 bitwise one text per request; batch against one by
one cosine 1.0), and `bench/results/2026-10-01/fallback_parity.json` passes plan §6.6's two cosine
gates at batch 1 (min cosine 0.999260, mean 0.999919; batch 8 would give 0.998663; the third gate,
probe agreement, waits for M4).

**Known consequence.** The batch-invariant kernels move every vector (min cosine 0.99937 against
the old recipe, `batch_invariance.json#/secondary`), and at scale 100 that moves the CLM README
quickstart's urgency to 0.8466, 0.0100116 from issue #15's 0.8366 against a tolerance of 0.01
(billing, frustration and the tides rank stay within; `serving_m1b.json#/drift`). That probe is
the doctor's warn-level upstream-drift check (plan §6.8), whose references were measured on
another stack. A3 re-tunes neither its references nor its tolerance: the warning stands until the
doctor's owner records how a deliberate kernel change is told apart from upstream drift.

**Rotations.** `encoder_fp` of the vLLM route 852efc921a8a → **c3b3d5e1a283**, of the fallback
82f4ff33b628 → **b171b1ba4536**; `serving_lock_sha` `3c032477…` → `59625b8e…`. Nothing durable was
keyed by 852efc921a8a: the serving home held no feature store or sidecar, and no artifact or
pre-registered cell existed; the label-free collapse spike
(`bench/results/2026-09-29/collapse_spike.json`, report-only) and the M1-A probes keep it as
history. The golden reference is now one text per request under c3b3d5e1a283
(`.local/serving/encoder_golden_c3b3d5e1a283.npz`, written by `scripts/serving_probes.py`). No
framing, task key, question key or pre-registered rule changes. The ≤ 1e-4 systemone gate of plan
§5.6 and §8 now holds end to end between clm-serve and the local route **for float32 vectors
only**: X1's offline scores from the float16 vectors the feature store first kept missed it
(1.31e-3 `clm-latest`, 3.59e-3 `clm-raw`, `features_build.json#/rerun/crosscheck`), and the
store therefore keeps float32 vectors (implementation note "The feature store"), from which the
200-group cross-check passes at 5.2e-6 and 7.47e-5 (`x1_crosscheck.json`). `clm-raw`'s margin is
the narrower one because clm-serve accumulates its 4096-d cosines in float32 and the offline
scorer in float64; the gate's scope is unchanged and it is not loosened.

### A4 (2026-10-01) — amends D16, D23

**Decision.** The encoder endpoint (:8090) answers nothing but `/health` without the key,
whichever route serves it. In the vLLM container the bearer guard `serving/vllm_auth.py` is
loaded with `--middleware vllm_auth.require_api_key`: the bootstrap installs it under
`~/.mesa/clm/serve/vllm-auth` (directory 0700, file 0600, sha256 checked against the lock), the
container mounts it read-only on `PYTHONPATH`, and it compares `Authorization: Bearer <key>` in
constant time on every path except exactly `/health`, refusing to start without
`VLLM_API_KEY`. `--disable-fastapi-docs` removes `/docs`, `/redoc` and `/openapi.json`, and
`VLLM_NO_USAGE_STATS=1` / `DO_NOT_TRACK=1` stop vLLM's default usage report to an outside
server. The fallback's :8090 applies the same rule. This withdraws plan §6.4 and §6.8's "open by
design on loopback" for `/tokenize` and `/metrics`. Every client also sends
`truncate_prompt_tokens = max_len − 1` (`EncoderClient`; clm-serve `--max-tokens 4095`; the
fallback's `embed`), which is CLM's training cap (`train/embed_utils.py` `Recipe`).
`EncoderSpec.max_len` stays 4096, the server's window; the cap lives in the lock's recipe and not
in `encoder_fp`, because no vector was ever produced at 4,096 tokens (such inputs never
completed), and D23's token guard (abstain above `max_len − 16`) is unchanged. A4 therefore
amends D23's "client token guard" (every client now also truncates at `max_len − 1`) and fixes
what D5's `max_len` names (the server's window, not the longest input a vector was made from); the
register rows of D5 and D23 point here.

**Why.** M1-A found that vLLM's API-key check covers only `/v1`, `/v2`, `/inference` and
`/cohere`: `POST /pooling`, `/invocations`, `/score`, `/rerank`, `/detokenize` (and `/tokenize`,
`/metrics`, `/version`) answered 200 without a key (`bench/results/2026-09-29/serving_m1.json`).
sparky-1 is a multi-user host, and the container's port is also reachable on its docker-bridge
address from any local account (`bench/results/2026-10-01/serving_m1b.json#/auth/docker_bridge_address`),
so binding the published port to loopback keeps no local account out. An input of exactly
4,096 tokens never completed on the pinned image.

**Evidence.** `bench/results/2026-10-01/serving_m1b.json`: vLLM's own `build_app`, run for the
locked arguments in a throwaway, network-less copy of the image, has 19 routes and no WebSocket
route, with the guard outermost; on the live encoder all 17 routes other than `/health` answer 401
to no key, a wrong key and clm-serve's key, and unknown paths do too; a 5,005-token text
completes through `EncoderClient` in 0.98 s (4,095 tokens charged; cosine 1.0000000 with the
embedding of its last 4,095 token ids) and as a clm-serve state in 1.08 s; a request of exactly
4,096 token ids still times out at 30 s and is aborted on disconnect. `serving/tests/` covers the
guard (`test_vllm_auth.py`) and the route probe (`test_vllm_routes.py`) in the serving CI.

**Rotations.** `serving_lock_sha` (shared with A3); `encoder_fp` rotates for neither the guard nor
the cap. Callers of `/tokenize`, `/metrics` or `/version` on :8090 need the key; `EncoderClient`
sends it on every route.

### A5 (2026-10-01) — amends D16, A4

**Decision.** The encoder container has **no network** (`docker run --network none`): vLLM serves
on the unix socket `/run/mesa-clm/encoder.sock` (`--uds`; the host directory
`~/.mesa/clm/run`, 0700, bind-mounted read-write), and the topology's encoder URL
`http://127.0.0.1:8090` (D16) is a systemd socket unit, `mesa-clm-encoder-proxy.socket`
(`ListenStream=127.0.0.1:8090`), whose service hands each connection to
`/usr/lib/systemd/systemd-socket-proxyd ~/.mesa/clm/run/encoder.sock`. Nothing is published with
`-p`. The engine's single-process rendezvous is put on loopback (`VLLM_HOST_IP=127.0.0.1`,
`GLOO_SOCKET_IFNAME=lo`). In the same recipe the KV cache is pinned to what `--max-num-seqs 8`
at `--max-model-len 4096` needs, `--kv-cache-memory-bytes 4831838208` (8 × 4,096 tokens ×
147,456 bytes, the 4.5 GiB plan §6.5's budget assumed), while `--gpu-memory-utilization 0.20`
stays as vLLM's start-up check; the encoder unit pulls in the socket unit and the headroom timer
(`Wants=`; both stop with it, `PartOf=`) and is started at most three times an hour
(`StartLimitIntervalSec=1h`, `StartLimitBurst=3`), and its run script refuses an `encoder.env`
without a usable `VLLM_API_KEY` line before docker runs. The lock's recipe gains `network: "none"`,
which `serving_lock_sha` covers and the container check compares (`HostConfig.NetworkMode`, the
socket directory's mount); a recipe with `network: "none"` must serve on `--uds` and name no
`--host` or `--port`. The bearer guard of A4 is unchanged and still answers 401 on every route
but `/health`. This withdraws A4's statement that "the container's port is also reachable on its
docker-bridge address from any local account": the container has no bridge address.

**Why.** Under A4 the container's network namespace held seven listeners besides the API, all in
the engine process and none behind the key: PyTorch's TCPStore of the rendezvous and six Gloo
collective sockets, on the docker-bridge address and the IPv6 wildcard
(`bench/results/2026-10-01/encoder_netns.json#/before`: interfaces eth0 and lo; `0.0.0.0:8090`,
six sockets on the bridge address, `[::]:53581`). PyTorch treats both as trusted-network-only (no
authentication; open key and value reads, writes and deletes; memory that grows in the engine), and
the bridge address is reachable from every local account on this multi-user host, so the key no
longer kept local accounts away from the container as a whole. `ss` on the host cannot see them,
which is why neither the doctor nor the runbook did. Binding each listener to loopback is not
enough (TCPStore binds the wildcard whatever the rendezvous host says); removing the namespace's
network is. Separately, vLLM sized its KV cache at every start to fill its 0.20 share (9.38-9.87
GiB over the seven recorded starts, `bench/results/2026-10-01/encoder_logs.json`), which put the
process at 25,153 MiB by `nvidia-smi` and 27.43 GiB by the plan's own measure (the
`MemAvailable` difference), above plan §8's "footprint ≤ 24.3 GiB vLLM"
(`serving_m1b.json#/footprint`). The start limit and the key check stop a recipe that dies after
the model load (14.11 GiB in 66-82 s, `encoder_logs.json`) from reloading it on the shared GPU
every two minutes (`Restart=on-failure` with `RestartSec=30` never reaches systemd's default start
limit of five starts in ten seconds), and a headroom timer the operator had to start by hand was
found stopped while both units served.

**Evidence.** `bench/results/2026-10-01/serving_m1c.json` on the restarted units: the container's
namespace has `lo` only, six Gloo sockets on 127.0.0.1 and the rendezvous store on `[::]` inside
it, no bridge address (`encoder_netns.json#/after`); the 401 matrix, the clm-serve routes and
`verify_serving_lock(require_live=True)` all pass; the start log shows
`distributed_init_method=tcp://127.0.0.1:…`, "reserved 4.5 GiB memory for KV Cache", a KV cache of
32,768 tokens (8.00× concurrency at 4,096 tokens) and "Starting vLLM server on
unix:/run/mesa-clm/encoder.sock"; the encoder measures **19,609 MiB** by `nvidia-smi` and
**21.45 GiB** by the `MemAvailable` difference (107.46 → 86.01 GiB), under the 24.3 GiB gate by
either measure; the 20 encoder goldens are **bitwise equal** to the c3b3d5e1a283 reference and
the 1,563 feature-store vectors bitwise equal (as float16 casts) to the A4 recipe's
(`features_build_m1c.json`); systemone parity 3.9e-6 / 6.6e-5. `mesa-clm doctor --serve`: 34 ok,
the A3 drift warning, 0 failures (`doctor_serve_m1c.json`).

**Rotations.** `serving_lock_sha` `59625b8e…` → **`dd33f9fe…`**; `encoder_fp` stays
c3b3d5e1a283 (no vector moved: the goldens and the store are bitwise equal, D5). The feature
store's vector recipe (`ServingLock.vector_recipe()`) changes with the arguments and environment,
so the store was rebuilt under it (it was rebuilt in format 2 anyway). Clients are unchanged:
they still call `http://127.0.0.1:8090` with the key. Units to install: the six of
`deploy/systemd/` (the two proxy units are new and cannot be enabled: they have no install
section); none is enabled at boot.

## Plan (summary; the full plan is `design/plan-2026-09-28.md`)

Milestones, each a `feat/mN-*` branch merged by PR after `scripts/wait_for_checks.sh`:
M0 scaffold, house files, vendored files, ports (config, secrets, cards, states, registry, ols,
avu, planner, labels, metrics), fixtures and the OLS fixture closure, anyjev parity, bench
baselines and MDE, labels snapshot → M1 two tracks: A serving on sparky-1 (encoder, patched
clm-serve, keys, lock, doctor, collapse spike, features, fallback parity) and B the hermetic
pipeline (framings and lock, render, `_honest`, fake transport, rank-first pipeline, sidecar,
service); gate G1 freezes D0–D32 and the pre-registration → M2 evidence (X1, X2, zero_shot and
calibrated cells; K1; amendment A1) → M3 live proposed-only loop (apply, revert, history,
tools, smoke; tag v0.1.0a1) → M4 learned tiers (X3, X4, artifacts, citations, audits; K2) → M5
release 0.1.0 → M6 neon adapter (0.2.0) → M7 head tier (K3) → M8 scale-out and evidence-gated
auto.
