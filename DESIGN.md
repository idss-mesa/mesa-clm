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
| D24 | Specificity: one rank over {parent, ≤10 children, anchor}; a child wins at `p_fit(child) ≥ p_fit(parent)+0.10`; unbenched, so proposed-only | accepted; amended by [A1](#a1-2026-10-03--amends-d24-d28) |
| D25 | `avu.keep` is a rule: exact-triple dedup, then a cap of 25 by `p_fit` | accepted |
| D26 | MRTR state carries ids only (≤16 KiB) with a tamper guard; OLS and HTTP in `asyncio.to_thread`; `card_path` is CLI-only; iRODS card paths go through `assert_allowed`; parse errors carry line numbers, never content | accepted |
| D27 | Pre-registration and nested selection: X1–X4, metrics and rule R committed before any run; every selection re-made inside each outer fold; full-data selections are `exploratory:true` | accepted |
| D28 | Rank-and-cap while uncalibrated; degraded mode `method="ols_rank"` proposes the OLS top-1 per group as `proposed` (probs NULL, never auto) | accepted; amended by [A1](#a1-2026-10-03--amends-d24-d28), [A6](#a6-2026-10-04--amends-d28-a1-and-plan-42-q1-q2-q7) |
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
  `agent_answered`), and not settled by a curator or by the pipeline, minus a parent whose D24
  refinement group (which offers the parent too) is asked instead. Who answered a group is read
  from its override rows only (run exports and imports carry them): the pipeline sets `rejected`
  itself (the Q8 keep rule's drops, a refinement that did not replace its parent), so such a group
  is not pending but has no curator answer, and only a `human` outcome without a row (which only
  `record_human_pick` sets) counts as a curator's. `review --pick/--decline` answer the pending
  groups only; `feedback` answers any group of the owner's runs. Rationale: ontology groups
  build no AVU; asking the parent twice would split one answer across two groups; an agent's
  answer must not hide a group from the curator (A2). The first M1 version also counted a
  `rejected` group without a row as a curator's, which made `feedback` refuse the pipeline's own
  rejections as "already has a curator answer" (a regression against `8962c58`, found in the
  pre-merge review; `tests/unit/test_feedback_agent_pick.py`, `tests/unit/test_cli_m1.py`).
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
  routes behind a green `GET /v1/models` check). The **binds** come from `ss -ltne`: loopback
  only, sockets of this account (another account's socket on a free port would receive a
  client's key; `ss` prints no `uid:` for root's), and while `mesa-clm-encoder-proxy.socket` is
  active the encoder's port in the user manager's `init.scope` cgroup (when `ss` reports cgroups);
  the encoder unit active without that socket unit fails too. The **encoder socket** check fails
  anything in `~/.mesa/clm/run` but `encoder.sock`, and that entry unless it is a socket of this
  account (the container can write the directory; A5). The **serving units** check
  (`systemctl --user show`) fails a unit systemd loaded that is not the checkout's rendering, one
  awaiting a `daemon-reload`, and a running proxy without the rendered command line or its
  sandbox (`/proc/<pid>/status`); units that are not installed (the in-process fallback serving)
  are a warning (A5, second revision: the binds and the run directory look the same under the old
  and the revised wiring). Serve mode also reads the **encoder
  container's network namespace** (`/proc/<pid>/net/{dev,tcp,tcp6}` of the container's main
  process, the pid
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
  records of the live runs are `bench/results/2026-10-01/doctor_serve.json` (A3/A4),
  `doctor_serve_m1c.json` (A5: 34 ok, the A3 drift warning, 0 failures) and
  `doctor_serve_m1d.json` (A5 as revised before the merge, with the binds' owners and the encoder
  socket: 35 ok, the drift warning, 0 failures; the golden question moved to the non-bench SRER
  card, G1 freeze item 7). Rationale: under A3's kernels these hold exactly (M1-A's batch
  dependence made the plan's cosine tolerance necessary; it is kept as the warning band).
- **Upstream drift stays a warning (plan §6.8; the question A3 left to the doctor's owner).**
  The quickstart and tides references stay issue #15's and the model card's, with their
  tolerances; the doctor prints the unrounded values and differences. A deliberate kernel change
  rotates `encoder_fp` (A3) and records these probes under the new fingerprint (for
  c3b3d5e1a283, `bench/results/2026-10-01/serving_m1b.json#/drift`), so a warning whose values
  equal the current fingerprint's record is that recipe, and one that departs from it is drift.
  The doctor does not encode the record: a reference that moved with each recipe would stop
  detecting upstream drift.
- **Plan deviations kept for M1.** `serve keys --rotate` replaces both keys and prints the
  `systemctl --user restart` of both units, preceded by a `reset-failed` of the encoder (its
  start limit counts manual starts), instead of restarting them (plan §6.4): starting and
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
  default off (so does an explicit YAML `null`, which is why `config.yaml.example` keeps the four
  key settings commented out; its nulls used to switch the defaults off, pre-merge review). The
  default is resolved when the key is read, not stored in the configuration, so
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
  `bench/results/2026-10-01/features_build.json#/rerun/crosscheck`): float16 rounding spread over
  the vector moves scores beyond 1e-4 at CLM's scale of 100, and the plan's round-trip gate
  (cosine ≥ 0.9999, met at 0.9999999) bounds angles, not scores. It is not the few large
  dimensions alone: putting back from float32 the three that hold every vector's largest
  component (2202, 2284, 3169; float16 spacing 2.4e-4 there) recovers most of `clm-raw`'s gap
  (to 8.0e-4, still eight times the gate) and none of `clm-latest`'s (1.37e-3), and renormalising
  the float16 vectors before the head leaves `clm-latest` at 1.26e-3
  (`features_build.json#/rerun/crosscheck/cause`), so only float32 vectors reproduce clm-serve.
  From float32 vectors the same cross-check passes at 5.2e-6 and 7.47e-5
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
  `fold_eligible=false` rows and every row on a bench card that is not the silver consensus
  (curator rows whatever their tag, the reserved `gold`, and a consensus row whose `origin` is not
  the neon-avu-eval ingestion's, `learn.labels.NEON_EVAL_ORIGIN`) are counted in
  `meta["excluded"]`; `labels import-anyjev` skips `gold` rows outright
  (`skipped["reserved_source"]`), since nothing in mesa-clm or mesa-anyjev produces that source
  (plan §5.1), and consensus rows on a bench card (`skipped["bench_silver"]`), since a bench
  card's silver labels come only from `labels ingest-neon-eval` (mesa-anyjev's copies of them are
  the same rows). Rationale: selecting first let a tagged curator row at 1.0 win its identity and
  then be dropped, deleting the silver item (term.fits 285 → 284 items in a reproduction), and an
  untagged one replace the silver label in a test fold; a gold row in a mesa-anyjev file,
  imported without a terminal or `--trust-curator`, did the same (pre-merge review), and so did a
  consensus row in such a file under a heavier consensus source, which a source name alone let
  through (`consensus_all` at 0.8 over a `consensus_negative` item: term.fits 86/199 → 87/198
  in a reproduction; the review of `70dbefe`); nothing but the silver label may change a
  pre-registered item's label or presence (`tests/unit/test_bench_tasks.py`). All 934 rows of the
  frozen snapshot (`bench/snapshots/2026-09-29.parquet`) carry the ingestion's origin, so the
  origin rule leaves its items as they were.
- **The encoder container check (D5, K4).** `/v1/models` answers the same for every container of
  the pinned image, whatever its kernels or flags: a container like the batch-invariance
  experiment's B0 arm (the same image without `VLLM_BATCH_INVARIANT`) would pass it while its
  vectors differ (min cosine 0.99937, `batch_invariance.json#/secondary`). Before `features build` writes a vector and before an
  annotate run asks a question, `providers.live.container_check` compares the running encoder
  container with the lock (`serving.check_encoder_container`: image, arguments, environment,
  mounts, network); a container that departs from it is refused (`features build` exit 1,
  annotate exit 2, never degraded), also when docker answers but the pinned image is not present
  (a container not created from the pinned digest, or one that was but runs another recipe; this
  used to pass as "not verified", pre-merge review), and when docker cannot be asked from the
  session, or no encoder container exists, the result says "not verified" in the command's
  output (annotate's `--out` JSON carries it under `preflight`, in the run shape and the
  `--eval-result` shape alike). The fallback server refuses `--batch` other than 1, since its fingerprint says
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

## Implementation notes (M2)

How M2 reads the frozen pre-registration where its text leaves room. None of these notes changes
the frozen section below; each says what the code does and where the reading is written down.

- **M2 analysis plan (pre-run).** Every frozen rule M2 runs (the LOCO folds and cells, rule R, the
  lookup gate, the citation test's fields, X1, X2, X3's zero_shot and calibrated tiers, K1) is
  read as one deterministic algorithm in `design/m2-analysis-plan.md`, committed and pushed with
  its code (`src/mesa_clm/bench/{framing,cells,x2,registered,run,stats,results}.py`,
  `learn/{calibrate,features,offline}.py`, `cli.py`, `scripts/x1_latency.py`) **before the first
  run on the snapshot's labels**, so the commit shows every choice was made before a result
  existed. Where the frozen text admits more than one reading the plan names each alternative and
  why it was rejected (the more conservative reading wins; §0). The readings the three pre-commit
  reviews changed, in short: ΔAUROC(real − shuffle) is the real AUROC minus the mean AUROC of 200
  seeded within-card derangements (§5; the drafts' AUROC of the expected shuffled score was biased
  against every arm); rule (2) is read literally, an F7 or F9 NLL winner stands and the token
  tie-break applies only when the cacheability step leaves both (§7.5–§7.6); no calibrator is fitted
  on fewer than 100 items, per fit (§6.3); Platt's targets are the hard labels and the temperature
  bounds [1e-4, 1e4] (§6.4–§6.5); the candidate-only probe is a learned probe on the PR #13 recipe
  (§7.11); the tier run (`tiers.json`) is the one producer of the citable-form
  `<task>.<tier>.<A1>` cells, X1 (`x1.json`) chooses (§8.6); a run is *registered* only on the
  registered snapshot (`bench/snapshots/2026-09-29.parquet`, both its hashes, the published counts)
  with the registered configuration, and any other run is written with every cell
  `pre_registered: false`, `exploratory: true` (§13). A finishing pass, still before any run, made
  every producer apply the registration itself (X1, the tier cells and X2 write a call on other
  labels or with other settings unregistered whatever its caller says, and refuse labels that claim
  the registered snapshot but give other counts; §1.3, §13.2), added X2's report-only comparison of
  the recomputed no-model controls with the M0 cells (§10.1) and a test that holds the plan's
  Appendix B and §1.3 to the code (§0.2). A second review round (faithfulness and code), still
  before any run, changed: the registration also pins the framings lock of G1 (`b432d32a7536…`),
  each model's D5 fingerprint under the serving lock of G1 (`dd33f9fedbae…`; `clm_model_fp`
  78be8c462b2e and 9f44b0301ee3) and, in X2's producer too, the AnyJev dump, and a run is
  registered only when its item tables score exactly the configured grid and it covers the
  registered tasks and tiers (§1.4-§1.5, §13); `bench framing --decide --from` re-derives the status
  from the identity `x1.json` records, holds the numbers `x1.md` prints to the traces and to the
  items file (the arm records, the arm cells' counts, metrics and per-item predictions, the
  `p_loco` column, the probe's statistics, the published counts) and names what it cannot replay
  (the arm cells' lookup controls and `beats_lookup_novel`, the probe's fits, the label-free
  diagnostics; §12.6), and `bench run` recomputes X1 before it takes the outcome (§12.7, §14.6); the tier
  cells keep X1's trace pointer (`x1_trace`, §9.5); the plan names the one-sided reading of
  "cluster-LB" and labels every reported interval as the 90% interval it is (§4.2), keeps
  "cluster-LB > 0.5" as the bound alone where the frozen MDE simulated rule R (§4.3, §9.9), corrects
  what `lopo` and a nested cell's diagnostics cover (§9.9-§9.10) and says a fit that does not
  converge is used as it is and flagged (§6.4, §6.5, §10.2); the timing run asks each target under
  both models back to back, alternating which goes first, and records which asks found new texts
  (§11.3). After the first run nothing in the plan or the code changes except by amendment with the
  affected cells marked exploratory (§15).
- **M2 pre-run disclosure: every look at labelled bench data from G1 to this commit.** Nothing in
  this phase computed a quantity that combines a silver label with a model output (from the feature
  store, clm-serve or anything else), and no M2 verb ran on the registered snapshot. What was read
  or run on real data, all label-free or already published:
  1. *Published counts*: the class counts, items per card (`per_fold_n`) and label-source counts of
     `bench/results/2026-09-29/baselines.json`, and `mde.json`'s per-card class counts (both M0
     files), to write the plan's floors, guards, weights and count check.
  2. *The snapshot, label-free*: its column names, per-task counts, identity columns and
     `state_json` (the manifests, the request builders, the AnyJev join test); the distinct targets
     per card (5–25); how many identities have several rows or states (no label column read). The
     hermetic suite now also reads, on every run: the snapshot's identity and state columns (the M2
     CLI tests build a synthetic world on them, with generated labels; the timing script's test
     builds requests from them for a stub that answers nothing), the snapshot's label-content digest
     against the published `labels_content_sha256` (`tests/unit/test_bench_registered.py`; a hash,
     no value), and the committed AnyJev dump's `state_json` (its identities equal the snapshot's,
     `tests/unit/test_x2.py`). The label-free manifest tests now overwrite every label-bearing column
     (label, index, weight, source, origin, actor, fold flags, ids) before comparing.
  3. *The AnyJev L2 dump*, structure only (keys, item counts, the presence of `state_json`, the null
     counts and lengths of its fields) and its `state_json`; no label or `p_yes` value was printed or
     used.
  4. *The feature store*: coverage and stats (1,563 X1/X2 texts, all embedded, 0 truncated); the
     label-free `features build` of the 390 closed-choice texts
     (`--tasks column.annotate,column.aspect,avu.value_kind --framings F7`, 2026-10-02T00:36:06Z to
     00:37:00Z, 165,575 encoder tokens, 0 truncated) and `features project` (00:37:07Z), after which
     the store holds 1,953 texts and vectors under the live recipe; the encoder token counts of the
     F4/F7/F9 contexts per target (§7.6 quotes them). The pinned head's `exp(logit_scale)` (100.811)
     was read from its export, and the committed collapse-spike reports were read.
  5. *The three reviews* (faithfulness, statistics, integrity), the integration and the finishing
     pass ran synthetic computations only (planted-signal worlds, fake stores, mutation runs of the
     test suite in `/tmp` copies; the M2 CLI test world also runs `bench baselines`' code on its
     generated labels), plus the label-free reads above; the integrity review verified that
     `bench/results` is unchanged since G1 and that the feature store holds only the manifests'
     texts. The duplicate-identity manifest test replaces every label-bearing column before it reads
     its one snapshot row. ~~One synthetic X1 output left in `/tmp` (labels_sha256 `aaaa…`, cards
     `DP0.0000k.001.synthetick`) was deleted.~~ *(Corrected after the run, 2026-10-03, by the
     post-run integrity check:)* one synthetic X1 output in `/tmp` was deleted, but another with
     that description remains (`/tmp/rev_m2/demo/out/2026-10-02/`: labels_sha256 `aaaa…`,
     labels_content_sha256 `bbbb…`, cards `DP0.0000k.001.synthetick`), and so do about 95 other
     X1-format files of the reviews and of test runs (pytest's temporary directories, the review
     scratch directories); all are synthetic (label hashes `aaaa…`/`bbbb…` or generated content
     `462967ff…`/`c2a2e677…`, fake encoder vectors and a random head), none carries the registered
     label content `5c60a8a6…`. The second review round (faithfulness, code) and its
     fixes did the same: synthetic worlds and mutation runs in `/tmp` copies of the checkout (which
     hold the committed snapshot file, read there only as the hermetic suite reads it), the
     registration's framings lock and fingerprints taken from the committed `framings.lock.json` and
     `serving/serving.lock.json`, and label-free counts of the timing run's requests against a
     text-keyed cache, before and after its order changed (requests built from the snapshot's
     identity and state columns, a stub that answered nothing). Its new tests read the snapshot as before: the manifest of the closed
     choices (with a request builder made to fail) and the timing script's requests, label-free.
  6. *After this commit*, before the X1 run, the label-free timing run of the plan's §14.3 will ask
     clm-serve 20 manifest targets per (task, framing), each under both models back to back,
     discarding every answer; it is added here when it has run. *(Added after the run,
     2026-10-03, by the post-run integrity check; it changes no cell.)* It ran once on `76c890f`,
     17:28:14Z to 17:28:50Z (35.4 s, exit 0): 2 tasks × 4 framings × 20 targets × 2 models, every
     answer discarded; `bench/results/2026-10-03/x1_latency.json` (sha256 `85e89397…`) holds
     identity fields, wall-clock ms, the two flags per ask (`first`, `new_text`) and their
     summaries only, no target id, option, answer or label. Just before it, at 17:28:09Z, a pre-flight `mesa-clm
     doctor --serve` that the protocol did not name ran once (exit 0: 36 ok and the A3 drift
     warning; no key in its output, which was deleted): label-free, it embedded the doctor's fixed
     texts (the 20 encoder goldens, whose seventh is the F9 context of a bench target, G1 freeze
     item 4) and asked the SRER golden question and the drift questions; the encoder journal
     (`journalctl --user -u mesa-clm-encoder.service --utc`) shows, for the two, 295
     `/v1/embeddings` requests from 17:28:09Z to 17:28:50Z: 294 answered and one refused with 401,
     the doctor's request without the key (its 401 matrix). Between §14.4 and §14.5
     `x1.md`, the report §14.4 had just written, was read; the decision was already made and
     recorded, and §14.5 leaves nothing to choose. The integrity check found nothing else between
     the commit and the run but `--help` calls.
  Anything else done before the first run that reads labelled bench data or asks a bench item is
  added here, in place, before that run.
- **M2 registered run (post-run; added after the run, 2026-10-03).** The protocol of plan §14 ran
  on the serving host from the pre-run commit `76c890f` (pushed with the branch at 17:27:39Z, PR
  #4 opened at 17:27:41Z), each step once and with exit 0 (`.local/m2/run_protocol.log`, outside
  the repository): §14.3 the timing run 17:28:14Z–17:28:50Z; §14.4 `bench framing --latency …
  --decide` 17:28:55Z–17:29:43Z; §14.5 `bench framing --decide --from …/x1.json`
  17:30:07Z–17:30:13Z; §14.6 `bench run --tiers zero_shot,calibrated --loco` 17:30:22Z–17:30:29Z;
  §14.7 `bench x2` 17:30:36Z–17:31:23Z; §14.8 `bench table` 17:31:23Z–17:31:24Z. The outputs were
  committed as produced in `eea525d` (17:31:40Z, pushed 17:31:44Z as a fast-forward; no code
  changed between the two commits), and **the run was not repeated**: no `--force`, no second
  attempt, no other results file (plan §14.9). The registered outputs under
  `bench/results/2026-10-03/`, by sha256:
  `x1_latency.json` `85e893978a16e068b3eb0d804b7270168eabebc4746b08ccac03b4c463bd3df4`;
  `x1.json` `bbef6cc9dbf094abba8e0fcdc404e00f09da391ce43e97b4dab8440d2b3455b8`;
  `x1_items.parquet` `b949664b42e3b6a55a4ecb7891a7f2531edc52b80e16b14b04e007256443d26f`;
  `x1.md` `4e93238f330a47fefa66e6d5e7ecd36c84f5c202c4a123286db35b70b90e2e1c`;
  `tiers.json` `42501a293f45eacdd5db70fc7ee6d28acd5d9c726bcf3b79e5f657ac52f8684d`;
  `tiers.md` `379b100b658c4bf3ce9921aa7a1180ffc498686aa90abfd05bd7b2dae3e196ca`;
  `x2.json` `76e07c34ddaed64e8987a2611c5af462457d7c398f49027533de3ea43cdb701d`;
  `x2.md` `f26bbfa0392cc351b3a20a269869b4faf3c22816bdcea9bb945668c6ef4daa9d`;
  `table.md` `288bc1de0ec43aeb70e8314eff3d02e4e5a5d641f23cbb45e42eb2d306d94063`.
  X1's outcome is amendment [A1](#a1-2026-10-03--amends-d24-d28) and every number is in RESEARCH.md,
  "M2 results (registered run)". After the commit, a post-run integrity check and two independent
  recomputations worked in scratch directories (exploratory; nothing written to the repository), and
  the replay was run again at 17:41:13Z and after A1's lock rotation (18:19:14Z), each time with the
  results files' hashes unchanged. One reading of the analysis plan is inexact without changing a
  number: §9.4 says the calibrated cells of `column.annotate` and `column.aspect` "list every fold
  `below_floor`"; aspect's do (training sets of 47–55 items), while annotate's record the first
  check that fails, the 30/5 guard in 6 folds and `below_floor` (81 items) only for `bet_fielddata`,
  because `cells.tier_cell` checks the guard before the floor
  (`tiers.json#/cells/neon_annotate.calibrated.F7/skipped_folds`). All 7 annotate training sets
  (77–92 items) are below 100, so the cell is empty under either order.

## Implementation notes (M4)

How M4 reads the frozen pre-registration where its text leaves room. None of these notes changes
the frozen section below; each says what the code does and where the reading is written down.

- **M4 analysis plan (pre-run).** Every frozen rule M4 runs (X3's probe tier, K2, X4 and the
  teacher corpus, the citation test, artifacts and promotion, the production audit) is read as one
  deterministic algorithm in `design/m4-analysis-plan.md`, committed and pushed with its code
  (`src/mesa_clm/learn/{linear,probe,teacher}.py`, `bench/{x3,k2,x4,e2e,registered,run}.py`,
  `artifacts.py`, `audit.py`, `policy.py`, `providers/{tiered,live,base}.py`, `health.py`, `cli.py`,
  `scripts/record_ols_teacher.py`) **before the first M4 run on the snapshot's labels**, so the
  commit shows every choice was made before a result existed. Where the frozen text admits more
  than one reading the plan names each alternative and why it was rejected (the more conservative
  reading wins; plan §0). The readings that settle the shape of X3, in short: the framing is not an
  axis of the grid (a probe is fitted and served on the active framing's texts after A1; plan §0.4);
  a spec belongs to one model, and K2's clm-raw clause is read on the model-restricted nested cells
  (§0.4, B2.4); the floors are per fit, the probe floor of 40 on every probe fit's training items and
  the calibration floor of 100 on the out-of-fold pool the probe's calibrator is fitted on (§0.4,
  P.3.2, P.4.2); the inner criterion is the pooled out-of-fold NLL of the uncalibrated probe,
  unweighted over items, with ties to the first configuration in declared order (P.3.3); the
  logistic penalty is normalised against a weighted-mean data term (P.2.7); K2's best servable tier
  is the candidate with the lowest pooled NLL, ties to `calibrated`, and "on the full set" is read
  strictly (B2.2–B2.3); teacher target resolution is column → every table card of the product that
  carries it, nothing synthesized for a column found in no card (B3.3); the silver-minus-Opus
  subset is the registered items whose label survives a rebuild without Opus (B3.6); the citation
  test requires the cell's tier to equal the record's level (C3.3); only promoted entries are served
  (C1.5). The three pre-commit reviews (faithfulness, statistics, integrity; read-only, synthetic
  data and `/tmp` copies, mutation runs of the suite) changed, still before any run: K2's
  "card-sign condition" is now the literal rule R sign test (per-card Δacc > 0 on ≥ ⌈0.8·m_c⌉
  counting cards), the margin-shifted per-card variant the drafts had gated on being kept
  report-only and computed in integers (`50·Δcorrect > −n_card`, no float rounding at exactly
  −0.02) — the more conservative reading, stated in advance to make K2(a) demanding against
  AnyJev L2 (B2.3); the citation test reads `masked` as M2's cells record it (`masked: true` with
  `mask` the framing's rule or none; the draft would have refused every real term.fits cell and
  its fake cells hid it; C3.2); `learn fit` applies the M4 registration and refuses on any
  deviation, cites only pre-registered nested cells (the closed choices' and term.fits' calibrated
  entries get `cite: null` by construction) and records the registration in the manifest (C1.7);
  `k2` requires `pre_registered: true` per candidate and fails closed on an unknown task count;
  `k2_verdict` has no per-tier hook; the "no AUROC" wording for `column.annotate` corrected; P.6's
  "stated in advance" claims qualified by the published per-card counts (aspect's inner training
  sets fall to 36 < 40); the alternatives for the nested cell's identity and the clm-raw clause
  named and rejected; the registration gained the neon-ducklake commit pin and name-wise constant
  tests; the suite gained the tests the mutation runs found missing (the calibrator fitted on
  inner out-of-fold predictions only, the inner 30/5 guard, strict boundaries of rule R's bound,
  ECE's cluster bound, an exploratory candidate's ineligibility, promotion's accuracy rule, X4's
  decision direction, the inner-fold refits reproducing the winner's OOF scores) and single-thread
  BLAS with reduced synthetic grids so it runs in 8 minutes; `ontology_not_aspect_allowed` left
  the ingest's `dropped` tally. Nothing changed a frozen rule or a committed cell.
- **M4 pre-run disclosure: every look at labelled bench data from the A6 merge (`b46c36b`) to this
  commit.** Nothing in this phase computed a quantity that combines a silver label with a model
  output (from the feature store, clm-serve, the encoder or anything else), and no M4 verb (`bench
  x3`, `bench k2`, `bench x4`, `learn fit`, `learn promote`) ran on the registered snapshot or the
  live feature store. What was read or run on real data, all label-free with respect to model
  outputs or already published:
  1. *Published counts and identity fields*: the cell keys, `tier`, `selection`, the flags,
     `question_key`, `model`, `n` and item counts of `bench/results/2026-10-03/{tiers,x2,x1}.json`
     and the two files' sha256 (the pins); `baselines.json`'s per-card counts for the plan's
     "stated in advance" paragraphs. No metric value of a committed cell entered code or a test.
  2. *The snapshot, label-free*: the hermetic M4 tests build their worlds on the registered
     snapshot's identity and state columns through the M2 CLI test world, every label-bearing column
     replaced first (`tests/unit/test_bench_m4_cli.py`); the learn, serving and bench unit tests of
     M4 are fully synthetic (random vectors, random heads, planted labels, fake results files, a
     snapshot file that exists only to be hashed).
  3. *The feature store*: `features stats --json` (1,953 texts, all embedded, 0 truncated, every
     one projected on both sides under `clm-latest`), read once to confirm the X3 inputs exist.
     Nothing was added to the store before this commit.
  4. *The teacher corpus and the SRER cards* (`~/neon-ducklake`, a live working tree): file bytes
     for the content hash, item counts per status, prefix, aspect and column, card column names for
     the resolution counts (RESEARCH.md "The teacher corpus"). One corpus file was rewritten on disk
     by a neon-ducklake commit at 2026-10-04 22:17:55Z while the pre-run work was reading the
     directory, so the hash first computed (`d97b1690b450…`) was superseded by `d90387928bed…`,
     the bytes the ingest read at neon-ducklake commit `b1fa52a8d3ff…` ("data: KONZ curation
     results and updated ontology maps", committed 2026-10-04 22:18Z, the writer of that file; the
     item set, 389, did not change); both are pinned. The hermetic suite now also reads the real
     corpus when `~/neon-ducklake` exists (`tests/unit/test_teacher.py`, the fixture-closure check:
     file names, bytes for the hash, CURIEs; label-free).
  5. *Live EMBL-EBI OLS4*: `scripts/record_ols_teacher.py` recorded the 110 `get_term` responses of
     the corpus's in-registry, aspect-mapped CURIEs on 2026-10-04 22:17:41Z–22:18:12Z (110
     requests, 0 failures, 0 retries, 31 s; `.local/ols_teacher_report.json`) into
     `tests/fixtures/ols-teacher/`; label-free.
  6. *The teacher ingest and the two snapshots* (plan §12.2; labels copied or created, no model
     output): on 2026-10-05 from 01:44Z to 01:46Z, `labels ingest-neon-eval --eval-root
     tests/fixtures/neon-avu-eval` into a fresh store (`.local/m4/labels.duckdb`; its snapshot's
     content digest equals the registered `5c60a8a6…`, the file hash differs by row ids and
     timestamps; that check snapshot is `.local/m4/check.parquet`, 934 rows, outside the
     repository), then `labels ingest-teacher --neon-root ~/neon-ducklake` (report
     `.local/m4/ingest_teacher.log`: 55 files, 389 items, corpus sha256 `d90387928bed…`, 6
     proposal files found under `sites/SRER/curation/proposals` but 0 `teacher_implicit` rows, 231
     (item, card) resolutions; dropped: 103 `out_of_registry`, 101 `no_column`, 22
     `unmapped_aspect`, 17 `collapsed_identity`, 11 `unresolved_column`, 5
     `ontology_not_aspect_allowed` (the term.fits row still written); inserted 435 rows: 231
     `term.fits`, 204 `column.ontology_fits`; `DP1.10092.001` has no card; `terms_missing` empty),
     then `labels snapshot --out bench/snapshots/2026-10-04-teacher.parquet` (1,369 labels;
     sha256 `397f98d4b28c…`, content `deb0dc46e17b…`); and `labels ingest-neon-eval --eval-root
     tests/fixtures/neon-avu-eval --exclude-models claude-opus-5-5` into a second fresh store
     (`.local/m4/minus_opus.duckdb`) with `labels snapshot --out
     bench/snapshots/2026-10-04-minus-opus.parquet` (679 labels; sha256 `2cd529ffa439…`, content
     `d8e26a0c7ac3…`). The pins are in `bench.registered.REGISTERED_M4`. The teacher snapshot's
     silver rows are the registered rows (`bench x4` checks their content digest).
  7. *The builders and the reviews*: the three builders (the probe tier; the bench producers and
     the registration; the serving side) reported synthetic-only work, the bench builder's
     label-free reads being items 1, 4 and 5 (their reports are in this session's transcript, not
     in the repository). The three pre-commit reviews (faithfulness, statistics, integrity) worked
     read-only, on synthetic data and in `/tmp` copies (planted worlds, mutation runs of the test
     suite); the integrity review also ran `features stats --json`, hashed the two snapshots,
     recomputed the teacher snapshot's silver content digest and counted rows per task, source and
     origin in the `.local/m4` stores and in seven leftover synthetic M2-CLI test worlds under
     `/tmp/mesa-clm-bench-*` (origin `…@synthetic000`, actor `synthetic`; deleted afterwards). No
     label value met a model output in any of it.
  8. *Found by the integrity check, outside this branch*: the A6 review round's two live probes of
     2026-10-04 that joined clm-serve's F9 Q3 answers on the bench cards to the `column.aspect`
     silver labels, recorded in the correction appended to A6 (exploratory, nothing written, no
     registered number touched, `column.aspect` has no citable cell in M2 or M4).
  Anything else done before the first run that reads labelled bench data or asks a bench item is
  added here, in place, before that run.

## Pre-registration (G1)

### G1 freeze

*(Added before G1, in `8947e8b`.)* The pre-registration below (the LOCO folds and cells, rule R,
lookup as a probability model, the citation test, the minimum detectable effect, X1–X4 and the
kill and pivot criteria K0–K4) is **frozen as of the M1 merge commit**, the commit that merges
`feat/m1-pipeline` into `main`. From that commit on it changes only by amendment, and an
amendment made after an experiment has run marks every affected cell `exploratory:true`.
*(Added before G1 by the convergence round after `9681eed`: the list above left the disclosure
out, so nothing made it amendment-only after the merge.)* The rest of this section is frozen at
the same commit and changes after it only by amendment too: this note, the note "Registered in
M0, frozen at G1" and the disclosure of every look at labelled bench data before G1 (its opening
paragraph, its numbered items and its closing paragraph on the hermetic suite).

**Registered in M0, frozen at G1.** ~~The text below is copied from plan §5.4, §5.6 and §8 and is
committed verbatim at gate G1, before any M2 run, with one edit made before G1 and shown in
place: X1's decision rule (6) is struck through and withdrawn (implementation note "The X1 anchor
variant").~~ The sections below from "LOCO folds" to "Kill and pivot criteria" are the text
registered in M0 and committed at gate G1, before any M2 run: plan §4.7 (the citation test),
§5.4, §5.6 and §8, worded by M0 in a few places, with M0's additions (the `labels_content_sha256`
parenthetical of the cell fields; in the lookup section the controls as M0 reproduced them, the
tie rule and what was observed; the minimum detectable effect). Since M0 its only edit is X1's
decision rule (6), struck through and withdrawn before G1 (implementation note "The X1 anchor
variant"); the G1 freeze note above and the disclosure below were added before G1. *(Corrected
before G1 by the convergence round after `9681eed`: the struck sentence, M0's own with a clause
`ec2e2d4` added unmarked, called the whole text plan §5.4, §5.6 and §8 verbatim and rule (6) its
one edit, but the citation test is plan §4.7's, M0 wrote the additions named, and the freeze note
and the disclosure are additions too.)* After G1 it changes only by amendment, and an amendment
made after an experiment has run marks every affected cell `exploratory:true`.

**Disclosure: every look at labelled bench data before G1.** ~~No pre-registered cell has been
computed.~~ No pre-registered **model** cell has been computed: the only cells flagged
`pre_registered: true` are the five `lookup_prob` control cells of item 1, one per neon task
(`baselines.json`), the M0 controls. *(Corrected before G1 by the pre-merge review: the struck
sentence overlooked those five cells. The same review corrected items 4 and 6 and added items 7
and 8; each change is marked where it was made. The verification round after `70dbefe`
completed items 4 and 7 and added item 9, marked the same way. The convergence round after
`9681eed` completed items 1, 3, 4 and 7 and the closing paragraph, corrected item 8 and the next
sentence (whose scope left out the planning stage and the reviews) and added item 10, marked the
same way.)* What ~~M0 and M1~~ the planning stage, M0, M1 and the reviews before the merge did
with the seven bench cards' silver labels, or with model outputs on those cards, and what was
seen:

1. *The M0 controls* (`bench/results/2026-09-29/baselines.json`, plan §5.4's verified controls):
   lookup LOCO accuracy 0.772 / 0.800, leave-one-product-out 0.723 / 0.721, the novel-key
   subsets 159 and 99 items, `lookup_prob`'s novel-key AUROC 0.424 / 0.396 (term.fits /
   column.ontology_fits). These are the baselines the rules compare against, computed as planned.
   *(Added before G1 by the convergence round after `9681eed`:)* The planning stage computed
   the same controls from the silver labels before M0 (plan §5.4's "verified controls
   (read-only)" and novel-key majority rates and plan §5.1's label counts, 2026-09-28; plan §5.6
   also cites mesa-anyjev's committed AnyJev L2 cells, the unpaired X2 baseline), and before the
   M0 commit (2026-09-29, 08:09-08:15 MDT) M0 found the tie rule that reproduces those controls
   by scoring alternative tie and novel-key rules against the silver labels in scratch scripts
   outside the repository, one of which printed the conflicting keys with their labels; the
   outcome is the lookup section's tie rule ("a symmetric tie rule gives 221/151").
2. *The MDE simulation* (`bench/results/2026-09-29/mde.json`): the realised class counts per task
   and card, and simulated power; no model output.
3. *AnyJev L2's held-out predictions* (`bench/baselines/anyjev_l2_2026-09-29.json`, X2's paired
   baseline): another model's per-item predictions against the silver labels, reproducing
   mesa-anyjev's committed cells exactly (term.fits accuracy 0.765, column.ontology_fits 0.837,
   and the per-card accuracies). *(Added before G1 by the convergence round after `9681eed`:)*
   Before that run, a dry run of the same script with its fake backend (hashed features, no
   model; `.local/serving/anyjev_l2_fake_dryrun.json`, 2026-09-29 12:37 MDT, outside the
   repository) fitted and scored the same 285 + 190 items against the silver labels.
4. *Label-free runs over the labelled rows' texts*: the collapse spikes
   (`bench/results/2026-09-29/collapse_spike.json`, `2026-10-01/collapse_spike.json`), the
   feature-store builds (`features_build.json`, `features_build_m1c.json`), the X1 cross-checks
   (`features_build.json#/rerun/crosscheck`, `x1_crosscheck.json`) and the serving and
   batch-invariance probes' 50 systemone pairs (`serving_m1.json`, `serving_m1b.json`,
   `serving_m1c.json`, `batch_invariance.json`): they compare encoders, routes and stores on the
   snapshot's texts; no label was read and no score was set against one. *(Added before G1, the
   same kind and equally label-free:)* the fallback-parity runs' 20 systemone pairs per route
   (`bench/results/2026-09-29/fallback_parity.json`, `2026-10-01/fallback_parity.json`), the
   sorted-keys collapse variant (`2026-10-01/collapse_spike_sorted_keys.json`), the latency
   probes' batch of 32 snapshot contexts (`embeddings_batch_32` in `serving_m1.json`,
   `serving_m1b.json`, `serving_m1c.json`), and the engine test
   `tests/engine/test_offline_parity.py` (12 snapshot groups through the live stack, offline
   against `/v1/systemone` and `/v1/rank`), which runs again whenever the engine tests run.
   *(Added before G1 by the review of `70dbefe`, the same kind and equally label-free:)*
   `batch_invariance.json`'s request patterns (a)-(d) over the first 100 contexts of the collapse
   spike and its 32-text latency batch of snapshot contexts (`embeddings_batch_32`), and the 200
   snapshot contexts the fallback-parity runs embedded through both routes (`contexts: 200`, the
   collapse spike's `.local/serving/collapse_contexts.json`). These runs do not stop at G1:
   whenever they run, the full `scripts/serving_probes.py` (its parity section, the 50 snapshot
   pairs on `clm-latest` and `clm-raw`, and its latency batch of 32), `scripts/batch_invariance.py
   measure` (the 100 contexts and the 50 pairs), `scripts/fallback_parity.py` (the 200 contexts
   and 20 pairs) and `tests/engine/test_offline_parity.py` (12 groups) ask or embed labelled
   snapshot items, label-free, and the scripts keep what the stack returned outside the
   repository (`.local/serving/`: the vectors, `systemone_pairs.json` with `clm-latest`'s answers
   to the 50 pairs, `m1b/vllm_ref_systemone.json` with clm-serve's answers to the 20).
   *(Added before G1 by the convergence round after `9681eed`: the item named only some of the
   records that keep the stack's answers, and left out the goldens.)* Two committed records keep
   answers too: `x1_crosscheck.json` records, for each of its 200 groups (141 `term.fits` and 59
   `column.ontology_fits`; F4 67, F7 67, F9 66), the top choice of the offline store and of
   clm-serve (`winner`: `store` and `served`) for `clm-latest` and for `clm-raw`, and
   `features_build.json#/rerun/crosscheck/flips` one group's winner with its probabilities and
   one noul pair's `p_true`; the silver labels being public in the same repository
   (`bench/snapshots/2026-09-29.parquet`, `tests/fixtures/neon-avu-eval/`), zero-shot accuracy
   on those groups can be computed from them without a model call. Outside the repository the
   stack's answers on bench items are also kept in `.local/serving/m1b/systemone_pairs.json` and
   `m1c/systemone_pairs.json` (`clm-latest`'s answers to the 50 pairs of `serving_m1b.json` and
   `serving_m1c.json`), `.local/serving/vllm_ref_systemone.json` (both models' answers to the 20
   pairs of the 2026-09-29 fallback-parity run) and `.local/live-final/crosscheck.json` (the
   first cross-check's probabilities, `s_c` and winners for its 200 groups, both models), and
   the doctor records of item 7. None of these was set against the labels. The 20 encoder
   goldens (`GOLDEN_TEXTS` in `scripts/serving_probes.py`, the texts of
   `.local/serving/encoder_golden.npz` and `encoder_golden_c3b3d5e1a283.npz`) include a bench
   target's context: the seventh is the F9 context of brd_countdata's `observerDistance`
   (`77bf030797bc…`, item 7), the eighth a definition of PATO's "distance". They were embedded,
   label-free, by the probes' golden section (`serving_m1.json`, `serving_m1b.json`,
   `serving_m1c.json`), as the first 20 texts of `batch_invariance.json` in both arms and of
   both fallback-parity runs on both routes (`texts.goldens: 20`), and by every serve-mode
   doctor (`encoder goldens`, the engine test's doctor check included), and they still are
   after G1: they are what items 8 to 10 call the doctor's fixed texts.
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
   ~~live smokes and latency runs use non-bench cards only~~ no live **annotate** run (smokes,
   latency) uses a bench card until the M2 cells exist (`annotate_latency.json`). *(Corrected
   before G1: the struck words also covered the probes of items 4 and 7, which did use bench
   texts.)*
7. *(Added before G1.)* *Probes that asked a labelled bench item.* Until 2026-10-01 the doctor's
   golden `/v1/systemone` question (`health.GOLDEN_STATE`: brd_countdata's `observerDistance`,
   aspect measurement, candidates PATO:0000040 and UO:0000008 plus the anchor) was a labelled
   `term.fits` target (its `target_sha256`, `77bf030797bc…`, has ~~two rows in the snapshot,
   options PATO:0000040 and PATO:0000122~~ six rows in the snapshot: two `term.fits` rows, options
   PATO:0000040 and PATO:0000122, and four `column.ontology_fits` rows, the target being labelled
   for that task too): every serve-mode doctor run answered it at zero shot (the engine test's
   and plan §9's live-smoke step 1 included), and the recorded answer is PATO:0000040
   (`2026-10-01/doctor_serve.json`, `doctor_serve_m1c.json`), never set against the labels
   *(until the verification round of item 9)*. *(Added before G1 by the convergence round after
   `9681eed`:)* Four scratch records outside the repository keep that answer too, never set
   against the labels: `.local/live-final/doctor_serve_1.txt` (10:49 MDT) and
   `doctor_serve_final.json` (11:00), `/tmp/mesa-clm-int/doctor_quick.txt` (10:47, a `--quick`
   doctor in serve mode, the units being up) and `/tmp/doctor_live.json` (13:20, item 8). The
   latency probes asked the same target against 12 PATO candidates and recorded ~~latencies
   only~~ latencies (`serving_m1.json`,
   `serving_m1b.json`, `serving_m1c.json`, `batch_invariance.json`) and, in two `--quick` runs of
   the same script, the answer's full distribution over those candidates and the anchor
   (`.local/serving/quick_probe.json`, 2026-09-29T18:43:42Z, and
   `quick_probe_after_restart.json`, 19:34:43Z; outside the repository, never set against the
   labels), and the long-input probe embedded the bench cards' text ending with that target
   (`serving_m1.json`) and asked it as a state against "distance", "temperature" and the anchor,
   recording the probabilities (`serving_m1b.json`, `serving_m1c.json`). From
   the pre-merge review on, the golden question and these probes use the non-bench SRER card
   (`health.GOLDEN_STATE`, `scripts/serving_probes.py`; `tests/unit/test_bench_tasks.py` keeps
   them off the snapshot's targets), so ~~routine doctor and latency runs after G1 ask no bench
   item~~ the doctor's golden question and the probes' latency and long-input questions ask no
   bench item after G1; the probes' parity section and the other runs item 4 names still ask
   labelled snapshot items, label-free. *(Corrected before G1 by the review of `70dbefe`: the
   struck words promised more than the code keeps, since a full probe run still asks the 50
   parity pairs of item 4; the struck row count missed the four `column.ontology_fits` rows,
   and the two quick-probe records were left out.)*
8. *(Added before G1.)* *The pre-merge review's own ~~label-free~~ work on 2026-10-01*: a re-run
   of `scripts/x1_crosscheck.py` (output outside the repository, ~~identical to
   `x1_crosscheck.json`~~ with results identical to `x1_crosscheck.json`'s; its start time,
   durations, latencies, `clm-latest`'s cache-miss token count, command and store path differ), a
   20-group `clm-raw` `/v1/systemone` probe on the same draw, ~~and~~ reading the silver labels of
   a few snapshot rows to build the tests of the fold filter and the gold source (none compared
   with a model output), one `mesa-clm doctor --serve` at 13:20 MDT on `ec2e2d4`'s code, which
   still asked item 7's bench golden question at zero shot (`clm golden` on `clm-latest`,
   `clm parity` on both models; the answer kept outside the repository in
   `/tmp/doctor_live.json`, never set against the labels), and at 13:25 copies of this host's
   five run sidecars into a scratch directory (the smoke sidecar holds item 5's zero-shot run on
   brd_countdata, the latency sidecar the runs of `annotate_latency.json` on non-bench cards,
   the other three fake-provider runs; whether a copy was opened is not recorded); the fixes'
   live checks (`bench/results/2026-10-01/serving_m1d.json`, `doctor_serve_m1d.json`, and one
   doctor run through a scratch port before the restart) asked only the SRER golden question and
   the upstream drift questions and embedded the doctor's fixed texts. *(Corrected before G1 by
   the convergence round after `9681eed`: the item called label-free work that read silver
   labels, left out the 13:20 doctor run and the sidecar copies, and called the re-run identical
   where its results are.)*
9. *(Added before G1.)* *The verification round after `70dbefe` (2026-10-01).* While checking
   item 7, two of its reviewers printed the six snapshot rows of the former golden target
   (`77bf030797bc…`, brd_countdata's `observerDistance`) with their silver labels (task, option,
   and the `label` or the `label_source` that encodes it); item 7 records the doctor's zero-shot
   answer to that question (PATO:0000040), so one recorded zero-shot answer was thereby seen next
   to its silver label. No other model output was compared with a label in the round's reports.
   The rest of the round was label-free: tests that printed only class counts `mde.json`
   publishes (item 2) and per-source counts; this host's mesa-anyjev sidecars imported into
   scratch stores (no change to the five bench tasks); one `mesa-clm doctor --serve` (the SRER
   golden question and the drift questions); the hermetic suite; proxy experiments on scratch
   sockets and ports; and the fixes' own work: the snapshot rows of that target counted per task
   (no label or option read), the snapshot's rows counted per label source and origin (the
   totals `labels stats` reports), the candidate keys of the two quick-probe records listed (no
   value read), and the live checks around the restart (`bench/results/2026-10-01/serving_m1e.json`,
   `doctor_serve_m1e.json`, one more `mesa-clm doctor --serve` and the engine test's doctor
   check), which asked only the SRER golden question and the drift questions and embedded the
   doctor's fixed texts.
10. *(Added before G1.)* *The convergence round after `9681eed` (2026-10-01), the last work
    before the merge.* Its two reviewers (of this section and of the encoder endpoint) and the
    judges of their findings reported reading no label value and no model answer on a bench
    item: they inspected the records and scratch files behind items 1, 3, 4, 7 and 8 and the
    closing paragraph by structure only (keys, types, counts, check names and statuses, the
    source of scratch scripts; a path-only diff of item 8's re-run), ran clients and the proxy
    against scratch sockets and ports with dummy keys, and counted, without printing, the
    encoder key's occurrences in `docker inspect`'s output (A5). The fixes' own work was of the
    same kind: the records named in items 1, 3, 4, 7 and 8 inspected by structure (one pair's
    candidate keys listed in `vllm_ref_systemone.json`, no value read; the smoke and gate runs'
    summaries read for their card, tier, fingerprint and counts), the seventh golden text
    compared with the F9 rendering of its target, the hermetic suite (below), the serving tests,
    a race experiment on a scratch port with a dummy key, the count-only check of
    `docker inspect` again (A5), and two runs of `mesa-clm doctor --serve`, which asked only the
    SRER golden question and the drift questions and embedded the doctor's fixed texts (item 4).
    Anything else done before the merge that reads labelled bench data or asks a bench item is
    added here, in place, before the merge commit.

The hermetic test suite runs the deterministic fake provider (hashed n-grams, never evidence) on
the fixture cards; it says nothing about the model. *(Added before G1 by the convergence round
after `9681eed`:)* It also recomputes items 1 and 2 from the fixture's silver labels
(`tests/unit/test_baselines.py`, `tests/unit/test_mde.py`), and its label-integrity tests flip
silver items and put curator rows on bench cards in temporary stores
(`tests/unit/test_bench_tasks.py`; while they were written, before `ec2e2d4`, a scratch store
reproduced the finding they test with such a row), each time it runs, before G1 and after; no
model output is involved.

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

### A1 (2026-10-03) — amends D24, D28

**Decision.** The production framing of each X1 task is the full-run choice of X1's registered
run, `bench/results/2026-10-03/x1.json` (`#/x1/registered` true, `#/x1/deviations` empty;
the reserved amendment of plan §8's M2 exit gate and analysis plan §14.10). Pointers below are
into that file unless another file is named.

- **`column.ontology_fits` → F9 on `clm-latest`** (`#/x1/tasks/neon_ontology_fits/a1`). Rule (1)
  qualified F4, F7 and F9 on `clm-latest` and F9 on `clm-raw`
  (`#/x1/tasks/neon_ontology_fits/decision/step4/eligible`). Rule (2), read literally (analysis
  plan §7.5), ranked the qualified `clm-latest` arms by pooled LOCO-Platt NLL, F9 0.6021, F7
  0.6331, F4 0.6656 (`#/x1/tasks/neon_ontology_fits/decision/arms/<arm>/evidence/nll`), and an F9
  winner stands: "F9 has the lowest LOCO-Platt NLL; no arm is more cacheable"
  (`#/x1/tasks/neon_ontology_fits/decision/step2`). Rule (3) keeps `clm-latest`: F9@`clm-raw` ≻
  F9@`clm-latest` on NLL fails both conditions, Δ −0.0162 with bootstrap lower bound −0.1069 and
  2 of 7 cards against 6 needed (`#/x1/tasks/neon_ontology_fits/decision/step3`,
  `#/x1/tasks/neon_ontology_fits/decision/comparisons/0`). The nesting agrees: every one of the 7
  outer folds chose F9@`clm-latest` (`#/x1/tasks/neon_ontology_fits/nested/fold_choices`), above
  the citation test's ≥ 5/7 (`bench/results/2026-10-03/tiers.json#/cells/neon_ontology_fits.calibrated.F9/diagnostics/fold_agreement_with_a1`:
  7 of 7).
- **`term.fits` → K1** (`#/x1/tasks/neon_term_fits/decision/outcome`;
  `#/x1/tasks/neon_term_fits/a1` null). No F4, F7 or F9 arm qualified on either model: every one is
  below the 0.60 AUROC floor (the best, F9@`clm-latest`, 0.557:
  `#/x1/tasks/neon_term_fits/decision/arms/F9@clm-latest/evidence/auroc`), and every nested fold's
  inner outcome is K1 too (`#/x1/tasks/neon_term_fits/nested/fold_choices`;
  `#/x1/tasks/neon_term_fits/nested/skipped_folds`: `inner_k1` in all 7). The frozen criterion
  applies as written: "**K1 no state signal (M2):** if no arm qualifies for a task on either model,
  its zero_shot/calibrated tiers become audit-only; proposals use `ols_rank` (`proposed`, D28) until
  a probe is promoted; M4 goes probe-first." For mesa-clm that means:
  - every `term.fits` group (Q4, Q5, Q6) is decided by `ols_rank` by default: the shipped
    `decider.ols_rank_tasks` is `[term.fits]` (the OLS top-1 per group as `proposed`,
    probabilities null, never `auto`; D28). **D28 is amended**: `ols_rank` is no longer only the
    degraded mode of an outage or a pivot named per call, but `term.fits`'s production method
    until a probe is promoted; a run that uses it by design is not `degraded` (that flag now
    means CLM did not answer, or tier `ols_rank`);
  - **D24 is amended**: its refinement ranks by `p_fit`, which an `ols_rank` record does not
    have, so it is never asked for an `ols_rank` group; where it would have been (specificity on,
    a proposed winner with OLS children) the group's `search_json` records `specificity:
    {asked: false, reason: no_p_fit_ols_rank}` and the proposal's rationale says so (the M1 code
    already skipped it for `ols_rank` fallback groups, silently);
  - CLM answers `term.fits` only in a run that asks for it, for audit: `annotate --ols-rank-tasks
    none`, or `decider.ols_rank_tasks: []` (`MESA_CLM_DECIDER__OLS_RANK_TASKS='[]'`). Such a run
    is recorded as it is (its `term.fits` decisions at level `zero_shot`, never `auto`, D6; its
    output lists the tasks `ols_rank` decided); `term.fits`'s zero_shot and calibrated tiers are
    audit-only, never the shipped default (K1 sets no end to that: its "until a probe is promoted"
    bounds the `ols_rank` proposals);
  - `term.fits` keeps F7 as its active framing, the framing of its audit records: the K1 audit
    cells are `tiers.json`'s `neon_term_fits.zero_shot.F7` and `neon_term_fits.calibrated.F7`
    (`selection: none`, `pre_registered: false`, `exploratory: true`; analysis plan §8.8);
  - M4 goes probe-first for `term.fits`: X3's probe specs come before any further work on its
    zero_shot or calibrated tiers. Its proposals leave `ols_rank` on the frozen terms only, with
    no condition added here: when a probe is promoted (plan §5.5: `learn promote` needs a
    pre-registered nested cell, new ≻ current on NLL and acc ≽ current within 0.01), and K2
    settles the task in M4 as it settles every task (K2(c): a killed tier → `ols_rank`
    proposals). Changing the shipped `decider.ols_rank_tasks` then is a changed decision, so it is
    recorded by an amendment like any other (AGENTS.md, "Definition of done"); that is procedure,
    not a condition of K1.
- **Rule (4) did not fire.** On `column.ontology_fits` neither F1 arm beats every eligible arm on
  NLL (`#/x1/tasks/neon_ontology_fits/decision/step4`: `d3_amendment_recommended: false`; the
  nearest comparison, F1@`clm-latest` ≻ F4@`clm-latest`, has lower bound −0.0549 and 3 of 7
  cards: `#/x1/tasks/neon_ontology_fits/decision/comparisons/1`); on `term.fits` it is not
  evaluated (no eligible arm). D3 stands.
- **The closed choices are unchanged.** X1 covers the two rank_fit tasks only, and no
  pre-registered rule decides anything for `column.annotate`, `column.aspect` or
  `avu.value_kind` from their tier cells: they keep F7 and their behaviour. Their measured cells
  are recorded in RESEARCH.md, "M2 results (registered run)".

**Why.** This is the outcome X1 was registered to produce (plan §8, M2: "**K1**; A1 amendment + lock
rotation"; analysis plan §9.5 and §15: A1 "is the outcome X1 was registered to produce, not a change
to these cells' pre-registration"). It changes no pre-registered rule and no analysis-plan reading:
K1 applies as written, and the `term.fits` bullets above say what its words mean for mesa-clm's
production configuration without adding a condition to them (the amendment they name for a later
change of the shipped default is the register's procedure for any changed decision). Nor does it
change a committed number, flag or cell; what it changes is the production configuration (the
active framing, the decider's default and, with it, D24 and D28) and, through the framings lock,
the registration status a new X1 or tier run would get (below). The run was made once
(implementation notes (M2), "M2 registered run").

**Evidence.** `bench/results/2026-10-03/x1.json` (sha256 `bbef6cc9…`) with its items file
`x1_items.parquet` (`b949664b…`), `tiers.json` and `x2.json` (the hashes and the protocol times are
in implementation notes (M2)); the facts, each with its pointer, are RESEARCH.md, "M2 results
(registered run)". `uv run mesa-clm bench framing --decide --from bench/results/2026-10-03/x1.json`
replays every decision from the JSON and recomputes it from the items file ("the pre-registered run;
every decision replays and recomputes"), at §14.5, again at 17:41:13Z and after this rotation
(18:19:14Z), the results files unchanged. Two independent recomputations made after the outputs were
committed (their own code, outside the repository; mesa-clm's label-free manifest supplied the item
texts) agree with every number they recomputed: X1's 16 arms (pooled AUROC exactly, the
card-bootstrap bounds identically, the shuffle comparator to 3.3e-15, LOCO-Platt NLL to 2.3e-11),
its 14 nested decisions, and every tier and X2 cell (1,923 comparisons, largest deviation 1.2e-12);
RESEARCH.md, "M2 results", keeps them as the reviews reported them and names what the repository
reproduces itself. Neither decision depends on the bootstrap seed: over 200 other seeds (1–200,
rerun with the repository's `framing.decide` and `framing.nested_selection` on the committed items
file) the full-run and nested decisions are identical, and term.fits' K1 rests on the 0.60 floor,
which involves no resampling (the best nested arm reaches 0.5708:
`#/x1/tasks/neon_term_fits/nested/folds/DP1.10022.001.bet_archivepooling/decision/arms/F9@clm-latest/evidence/auroc`).
The feature store the run read is named by its stamp (`#/x1/store`: `encoder_fp` c3b3d5e1a283, 1,953
texts and vectors, the vector recipe `ed486dcd…`), which does not bind the vectors' content; here it
is corroborated by the configured `features.dir` (no override; the store unmodified since 2026-10-01
18:37 MDT) and by the collapse diagnostics of F7, F9 and F1, which equal the M1 spike's to 4 dp
(`#/x1/tasks/<task>/collapse` against `bench/results/2026-10-01/collapse_spike.json`). A later
registered run should also record a digest of the vectors it reads.

**What F9 means for production.** F9's context names the product and the column, not the table,
so one column in two tables of a product gets one context and the same scores: on the snapshot
the 60 `column.ontology_fits` targets have 45 distinct F9 contexts and the 143 `term.fits`
targets 101 (`#/x1/tasks/neon_ontology_fits/collapse/F9/n`, `#/x1/tasks/neon_term_fits/collapse/F9/n`),
and 7 Yes/No pairs of `column.ontology_fits` items (22 of `term.fits`) in different tables have
bit-identical F9 scores while their silver labels disagree (`x1_items.parquet`, F9 rows; every
AUROC counts such a tie one half). F9 answers alike for such columns; the registered cells
include that.

**Rotations.** `framings.lock.json`: `column.ontology_fits` `active_framing` F7 → **F9** (F9's
`active` true, F7's false); `term.fits` stays F7, and the closed choices F7. `lock_sha`
`b432d32a7536…` → **`7c93cc3e0ff6…`**. No `question_key` rotates (`active` is not part of it, D1):
`column.ontology_fits` decisions, features and artifacts are now keyed by F9's `c95785008b523fd0`
instead of F7's `8e8c77ee40c3eacb` (`#/x1/question_keys`), and `term.fits` audit records keep F7's
`a686a06aab14497a`. No `task_key`, label, `encoder_fp` (c3b3d5e1a283), `clm_model_fp` (78be8c462b2e
`clm-latest`, 9f44b0301ee3 `clm-raw`), head, `schema_sha256` or `serving_lock_sha` changes. The
configuration gains `decider.ols_rank_tasks` (default `[term.fits]`), so every run's `config_sha256`
changes. The registration keeps G1's lock (`bench.registered.FRAMINGS_LOCK_SHA` `b432d32a…`,
`#/x1/framings_lock_sha`): the committed `x1.json` stays the registered run and still replays,
because the replay compares the lock it records with the registration, not with the checkout; G1's
lock is today's framings with every active framing F7 (`framings.lock_sha({task: "F7", …})`,
`tests/unit/test_bench_registered.py`). A new `bench framing` or `bench run` invocation runs under
the rotated lock and is written unregistered (analysis plan §13.2; X2 reads no framing and does not
check the lock), and `bench run` refuses to take X1's outcome from `2026-10-03/x1.json` (§12.7: the
framings lock must be the tier run's own); the M2 run is not repeated (§14.9). No committed cell
changes or becomes exploratory. Tests: `tests/unit/test_framings.py` (the active framings, no key
moved by the rotation), `tests/unit/test_bench_registered.py`, `tests/unit/test_k1_default.py`
(term.fits proposals are `ols_rank` by default, D24 not asked and recorded, Q3 asked with F9, the
audit run, the CLI).

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
(`ListenStream=127.0.0.1:8090`), whose service hands its connections to
`mesa-clm-encoder-proxy` (`serving/encoder_proxy.py`, installed by the bootstrap, run by the serve
venv's Python): it relays only this account's connections (for each one it asks the kernel's
socket diagnostics, `sock_diag`, who owns the client's socket, and closes another account's at
once, before it takes a slot or reaches the encoder), and for each connection it opens
`~/.mesa/clm/run` and then `encoder.sock` with `O_PATH | O_NOFOLLOW`, requires an owner-only
directory and a socket of this account, and connects through `/proc/self/fd/<fd>`, so it never
follows a symlink the container could put there; it holds at most 4,096 connections (two
descriptors each, `LimitNOFILE=16384`), closes one on which no byte has crossed for 900 s, logs
each kind of refusal at most once per 10 s, and runs in a systemd sandbox without namespaces
(`NoNewPrivileges=`, a system-call allow list, `AF_UNIX` and `AF_NETLINK` only). Nothing is
published with `-p`. The engine's single-process
rendezvous is put on loopback (`VLLM_HOST_IP=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo`). In the same
recipe the KV cache is pinned to what `--max-num-seqs 8` at `--max-model-len 4096` needs,
`--kv-cache-memory-bytes 4831838208` (8 × 4,096 tokens × 147,456 bytes, the 4.5 GiB plan §6.5's
budget assumed), while `--gpu-memory-utilization 0.20` stays as vLLM's start-up check; the
encoder unit requires the socket unit and starts after it (`Requires=`, `After=`: a port another
account holds fails the encoder, and clm-serve with it), pulls in the headroom timer (`Wants=`;
both stop with it, `PartOf=`) and is started at most three times an hour, manual starts included
(`StartLimitIntervalSec=1h`, `StartLimitBurst=3`; `systemctl --user reset-failed` before a manual
restart), and its run script refuses an `encoder.env` without a usable `VLLM_API_KEY` line before
docker runs. A client that sends a key to a loopback port first checks that the socket holding
it belongs to this account (`net.assert_listener_owner`, before every keyed request: the serving
pair's clients, the planner gateway client and the probe scripts' raw clients) and, once a
connection is made and before a byte is sent on it, that the kernel names this account as the
owner of the connection's server end (`net.OwnerCheckedTransport`), and the doctor's
`serving binds` requires sockets of this account and, while the socket unit is active,
the encoder's port in the user manager's cgroup, its `encoder socket` check fails anything in
`~/.mesa/clm/run` but a socket of this account, and its `serving units` check fails units that
are not the checkout's rendering and a running proxy without the rendered command line or its
sandbox. The lock's recipe gains `network: "none"`,
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

**Revision before the M1 merge (2026-10-01).** The version of this amendment in commit `ec2e2d4`
(pushed on `feat/m1-pipeline`, never on `main`) read "whose service hands each connection to
`/usr/lib/systemd/systemd-socket-proxyd ~/.mesa/clm/run/encoder.sock`" and "the encoder unit
pulls in the socket unit and the headroom timer (`Wants=`; both stop with it, `PartOf=`) and is
started at most three times an hour". The pre-merge security review found that endpoint failing
in three ways, and the text above supersedes it before the merge (it is recorded here because
the register is append-only and that commit was published):

- *Fail-open on a taken port.* With `Wants=` alone, a port another local account held (free
  whenever the units are down, which is the normal state since nothing is enabled at boot, and
  briefly on every restart, the socket being `PartOf=` the encoder) failed the socket unit while
  the encoder and clm-serve still started; clm-serve's wait for a 200 from `:8090/health`
  accepted the squatter, its embedder sent the encoder key with `GET /v1/models` at start and on
  every unauthenticated `GET :8700/health`, and it used the vectors that came back. Before A5,
  `docker run -p` refused to start on a taken port. Now the encoder requires the socket unit,
  keyed clients refuse another account's socket (also while the units are down) and the doctor
  checks owners.
- *A deputy for the container.* `systemd-socket-proxyd` connected by path for every connection,
  `connect(2)` follows symlinks, and the container writes that directory: code running in the
  container could point 127.0.0.1:8090 at any unix socket this account can reach (the session
  bus, the user manager's private socket), for every local account, with this account's
  credentials. The proxy now refuses anything but a socket of this account there (verified on
  this host with the same scratch setup for both: `systemd-socket-proxyd` reached the other
  socket through the symlink, `mesa-clm-encoder-proxy` refused it).
- *A low local denial of service.* It used six descriptors per connection under the user
  manager's soft `RLIMIT_NOFILE` of 1,024 and vLLM's server never closes a connection that sends
  nothing, so about 170 idle connections from any account blocked the endpoint; the proxy now
  takes two descriptors per connection under `LimitNOFILE=16384` and refuses beyond 4,096.

Evidence: `serving/tests/test_encoder_proxy.py` (the target checks, the relay, the connection
limit, the idle timeout), `tests/unit/test_listener_owner.py`, `tests/unit/test_health_serving.py`
(binds and the encoder socket), `tests/unit/test_serving.py` (the units); the restarted units are
`bench/results/2026-10-01/serving_m1d.json` and `doctor_serve_m1d.json`. Nothing rotates: units
and the proxy are not in the lock (`serving_lock_sha` stays `dd33f9fe…`, `encoder_fp`
c3b3d5e1a283). Residual risks, stated rather than fixed: the container still shares the host's
IPC namespace (`--ipc=host`, one of the two ways vLLM documents to give its processes shared
memory; the other, `--shm-size`, is untested on this recipe, and the lock does not cover the IPC
mode); ~~any local account can still hold up to 4,096 idle connections and so block the
endpoint~~ (withdrawn by the second revision below: only this account's processes can); the
checks name an account, not a process, so another process of this account can hold the
ports; and the start limit counts manual restarts. The review also exposed both serving keys in
its own session's output; they appear in no file of the repository, its history or
`bench/results` (checked in process, the keys never on a command line), and rotating them
(`mesa-clm serve keys --rotate`, then `reset-failed` and a restart of both units) is an operator
action, not a recipe change.

**Second revision before the M1 merge (the review of `70dbefe`, 2026-10-01).** That commit
(also pushed on `feat/m1-pipeline` only) let every local account's connection through the proxy,
the one component they can all reach, ran it with the serving account's whole authority
(`systemd-analyze --user security`: 9.8 UNSAFE), logged two kinds of refusal once per
connection, and its doctor could not tell the revised wiring from the old one. The text above
now says, and `70dbefe` did not:

- *Only this account's connections.* For each accepted connection the proxy asks the kernel's
  socket diagnostics (`sock_diag(7)`: one exact lookup of the client's address and port, which
  any account may make; `SO_PEERCRED` answers only on unix sockets) who owns the client's socket,
  and closes a connection of another account, or of an owner the kernel can no longer name (the
  client already gone), at once: it takes no slot, reaches no encoder and holds nothing. That
  withdraws the residual risk that any local account could hold the 4,096 connections and so
  block the endpoint; this account's own processes still can.
- *A sandbox without namespaces.* `NoNewPrivileges=`, `SystemCallFilter=@system-service` minus
  `@privileged` and `@resources` (`EPERM` otherwise), `RestrictAddressFamilies=AF_UNIX
  AF_NETLINK`, `MemoryDenyWriteExecute=`, `RestrictNamespaces=`, `RestrictRealtime=`,
  `RestrictSUIDSGID=`, `LockPersonality=`, `KeyringMode=private`, `UMask=0077`: 5.9 MEDIUM
  instead of 9.8. A rootless user manager cannot set up `PrivateDevices=`,
  `ProtectKernelModules=`, `ProtectKernelLogs=`, `ProtectClock=` or an empty
  `CapabilityBoundingSet=` (the start fails with status 218). The mount-namespace options
  (`ProtectSystem=`, `ProtectHome=`, `PrivateTmp=`, `ProtectProc=`, `ProcSubset=`, …) imply
  `PrivateUsers=` there, and on this host Ubuntu's AppArmor confines a process in an
  unprivileged user namespace (`unprivileged_userns`), which refused the proxy's `connect()` to
  the socket the container bound ("disconnected path"): installed with them, the proxy closed
  every connection to :8090 for ten minutes after the restart (`serving_m1e.md`), so they are
  left out. The proxy keeps the backlog its listening socket was given (read from the socket,
  `tcp_info`) instead of reading `net.core.somaxconn`.
- *Throttled refusals.* Each kind of refusal (another account, the encoder unreachable, a refused
  socket entry, the connection limit) is logged at most once per 10 s, the next line counting
  those left out, so connecting in a loop cannot write the journal once per connection.
- *The doctor tells the wiring apart.* `serving binds` and `encoder socket` read the same under
  the old units and the revised ones; `serving units` compares each unit systemd loaded with the
  checkout's rendering, wants none awaiting a `daemon-reload`, and checks that the running proxy
  has the rendered command line and its sandbox (`NoNewPrivs`, seccomp in `/proc/<pid>/status`),
  so a host that pulled a fix without installing the units fails it.
- *Every keyed client checks the port's owner.* The planner gateway client (its default port is
  the `carc-litellm-tunnel` unit's, free whenever the tunnel is down) and the raw clients of
  `scripts/serving_probes.py` now ask before each keyed request, as `EncoderClient` and
  `ClmHttpClient` did.

Evidence: `serving/tests/test_encoder_proxy.py` (the kernel's answer for this account's clients,
another account's connection closed unrelayed, the throttled logs, a swap after the check, the
backlog, a late answer after a half-close, a one-way stream, each owner check),
`tests/unit/test_serving.py` (the sandbox), `tests/unit/test_health_serving.py` (`serving
units`, `encoder socket`), `tests/unit/test_planner_gateway.py` and
`tests/unit/test_listener_owner.py` (the keyed clients, the owner checked before every attempt);
on the host, `bench/results/2026-10-01/serving_m1e.json` before and after the restart (an
unauthenticated `GET /health` from uid 0 and uid 65534 in throwaway containers on the host
network: the encoder's 92-byte answer before, nothing after; 9.8 → 5.9) and
`doctor_serve_m1e.json` (36 ok, the drift warning, 0 failures). Nothing rotates: units and the
proxy are not in the lock (`serving_lock_sha` `dd33f9fe…`, `encoder_fp` c3b3d5e1a283).

**Third revision before the M1 merge (the convergence round after `9681eed`, 2026-10-01).** The
text above now says, and `9681eed` did not, that each connection is checked and not only the
port; and a residual risk is stated:

- *The connection is checked, not only the port.* `assert_listener_owner` reads `/proc` and the
  request then connects in a second step, so another account could bind the port in between
  (while the units are down, the normal state, and on every restart) and receive the key: in
  the round's scratch run (a dummy key; the listener was this account's, counted as another's),
  a listener toggled 0.5 ms on and 4.5 ms off received the `Authorization` header on 16 of 400
  attempts the check had passed, and one toggled 2 ms on and 8 ms off on 3. Every keyed client
  (`HttpEndpoint`, hence the serving pair's clients, the planner gateway client and the probe
  scripts' raw clients) now sends through `net.OwnerCheckedTransport`: each new connection to a
  loopback address is checked once it is made and before TLS or a request byte, with one exact
  `sock_diag` lookup of the connection's server end (the request the proxy makes for the client
  end). On this kernel a connection not yet accepted already reports its listener's owner; an
  answer that names nobody (the handshake, or uid 0 without an inode, which some kernels report
  for a connection not yet accepted) is asked again for up to 5 s, and any owner but this
  account or root closes the connection unsent (`ListenerOwnerError`). The same scratch run
  against the fixed client: no byte reached the toggling listener in 800 attempts (101 refused
  by the `/proc` check, 99 on the connection, the rest failing to connect).
- *Residual risk: the docker group.* Three other local accounts are in the `docker` group (two
  with a login shell, one service account; `getent group docker` lists them),
  `/var/run/docker.sock` is `root:docker` 0660, and the daemon has no authorization plugin and
  no user-namespace remapping (`docker info`), so on this host those accounts are
  root-equivalent and the per-account boundary of this amendment does not hold against them:
  `docker inspect mesa-clm-encoder` shows the encoder key (the run script's `--env-file` puts
  `VLLM_API_KEY` into the container's `Config.Env`), `docker exec` and `docker cp` reach the
  container, a host-network container run as this account's uid is relayed by the proxy and
  passes the clients' checks as this account, and a bind mount of the home directory reaches
  `~/.mesa/clm/secrets` (the clm key included). The remedies are the host's, not the recipe's:
  the group's membership (for the host's administrators), rootless docker or `userns-remap`.
  Keeping the key out of the container's environment (a read-only 0400 key file the guard
  reads) would end the exposure through a routine `docker inspect` but not the others; it
  changes the guard, the lock and the container and is left to an amendment. Rotating both
  keys (`mesa-clm serve keys --rotate`, then `reset-failed` and a restart of both units) is an
  operator action, due after this round: its count-only check of `docker inspect` (no key
  printed) found the encoder key there, and its fixer wrote that command's output to a scratch
  file readable by every local account for about a second before deleting it.

Evidence: `tests/unit/test_listener_owner.py` (the kernel's answer for a connection's server end
before and after the accept, over IPv4, IPv6 and a dual-stack listener; the answer read
strictly; another account's server end refused, an undecided answer asked again; the trace on a
new connection only; a port taken after the check gets no byte; the planner and the probes'
raw client checking each connection). Nothing rotates: the clients are not in the lock, and
neither the proxy nor the units changed.

### A6 (2026-10-04) — amends D28, A1 and plan §4.2 Q1, Q2, Q7

**Decision (the user's, 2026-10-04, made after M2's results had been seen).** By default the
three closed choices are answered by deterministic rules, and CLM's answers to them are kept for
audit only. The user chose, verbatim: "Deterministic fallbacks: Q1: annotate every
non-identifier column (rule); Q2: the planner's aspect hint plus top-2 by the M0 lookup, else all
registry-allowed aspects capped; Q7: the existing pre-rule then 'the term label'. CLM still runs
for audit. Recorded as an amendment made after M2's results (product safety, not a scientific
claim)." A6 is a product-safety change of serving behaviour and **makes no scientific claim**: it
does not say that the rules or the lookup answer better than CLM on any card, nor that CLM has no
signal on these tasks, and no cell is read as evidence for the rules. It **does not touch the
frozen pre-registration (G1) or any cell**: no file under `bench/results/`, no `pre_registered`
or `exploratory` flag, no framing, `question_key`, `task_key`, label, fingerprint or lock changes,
and the M2 run is not repeated. `decider.closed_choice` selects the behaviour: `rules` (the
default, below) or `clm`, the M2 behaviour exactly (no A6 rule record, no audit marker, no
lookup table read), kept reachable for audits and tests (`MESA_CLM_DECIDER__CLOSED_CHOICE`,
`annotate --closed-choice rules|clm`, `Annotator(closed_choice=)`,
`DecisionService(closed_choice=)`). Where the decision's words leave room, the reading below is
the implementation's and is marked **(reading, confirmed by the user on 2026-10-04)**: the user
kept both readings as written when asked; each can be changed later by amendment without
touching anything else in A6. Under `rules`:

- **Q1 `column.annotate`.** A column is annotated iff it is not `cards.is_identifier`, whatever
  the planner says. A planner's `annotate=False` no longer keeps a non-identifier column out: it
  stays in the run's `plan_json` and decides nothing (under `clm` it is still M2's
  `planner_annotate_false` rule). A planner's `annotate=True` does not bring an identifier in: in
  M2 it only sent the column to CLM, whose answer no longer decides. Each annotated column gets a
  `rule` record answering `Yes` (`method` `rule`, `model` `rule`, `level` `none`, `calibration`
  `none`, `probs` NULL, outcome `rule`, reason `a6_not_identifier`); an identifier keeps M2's
  `No` rule (reason `is_identifier`).
- **Q2 `column.aspect`.** For an annotated column the aspects are, in this order, the planner's
  aspect hint for the column (if any) and the M0 lookup's top two (below), de-duplicated (the
  first source kept), `other` excluded: up to three aspects, as M2's CLM top two plus the
  planner's aspect. The decision caps only the fallback. The hint and the looked-up aspects
  are taken as given, `unit` included, as M2 took the planner's and CLM's (on the seven fixture
  cards every looked-up `unit` falls on a column with a unit). Each chosen aspect is a
  `column.aspect` `rule` record answering that aspect's option, reason `a6_aspect_hint` (model
  `planner`) or `a6_aspect_lookup` (model `lookup`); when the list is empty the fallback
  decides (below; reason `a6_aspect_fallback`, model `rule`). **No CLM answer chooses an
  aspect**: Q3 is asked only for the aspects chosen here, as M2 asked it for its aspects (F9, or
  `ols_rank`; `unit` is `uo` by rule; the two best in-play ontologies kept; the planner's
  ontology appended under the column's first aspect), so a column asks Q3 at most three times
  and a fallback column at most twice. A column left without an aspect (only when no aspect has
  an ontology in play, below) is reported `no_aspect`, as M2 reported one. How a column's
  aspects were chosen is on its rows: the reason of each `column.aspect` rule record, and
  `search_json.a6` of each of its Q3 groups (`aspect_source`, the column's aspects and their
  sources, the lookup's evidence, the fallback's order and choice).
  - **The lookup** is the M0 control (`mesa_clm.bench.baselines`: `lookup_key` and `Lookup`,
    plan §5.4) over a table frozen once from the registered snapshot.
    `src/mesa_clm/aspect_lookup.json` holds the 60 items of M0's `neon_aspect` bench task on
    `bench/snapshots/2026-09-29.parquet` (labels_sha256 `aafd18f8…`, labels_content_sha256
    `5c60a8a6…`; `column.aspect` at the policy's `min_weight` 0.6 with `fold_only`, D9, D30):
    card, column name and silver aspect.
    These are the rows behind M0's lookup on the very cell the decision cites
    (`tiers.json#/cells/neon_aspect.zero_shot.F7/baselines/lookup_acc` 0.550, n 60; M0's
    `bench/results/2026-09-29/baselines.json#/cells/neon_aspect.baseline.lookup_prob`).
    `scripts/freeze_aspect_lookup.py` writes the file through `closed_choice.freeze_table`, which
    refuses any snapshot but the registered one and checks M0's published counts. The file ships
    in the package; its sha256 (`12a38e76…`) is pinned in `closed_choice.TABLE_SHA256` and checked
    before any run, and a test rebuilds it from the snapshot byte for byte. Every host reads this
    table and no host's sidecar is read, so under one plan every host chooses the same aspects
    for a card.
    (The first build of A6 read the host's label store instead. That store holds no
    `column.aspect` labels on the serving host, so the lookup the decision names never answered
    there; review found this, and the frozen table replaces that read.) At annotate time the
    items of the card being annotated are left out, as M0's leave-one-card-out lookup leaves
    them out: a card never copies its own labels, so a bench card's silver labels never reach its
    own run, and an end-to-end leave-one-card-out bench (M4) stays leave-one-card-out. M0's
    `Lookup` then counts the remaining items; for every fold of the registered task this is
    M0's own lookup, with the same counts in the same order and the same prior (tested). The key
    is `(column.aspect, column, <column name>, '')`. The top two labels are those with the most
    rows, and a tie goes to the label of the alphabetically first card (M0's
    `Counter.most_common` over card-ordered rows). **A column name no other card carries
    contributes no aspect (reading, confirmed by the user on 2026-10-04).** M0's lookup would predict the
    training majority for such a key, and `lookup_prob` would give the training prior (28 of
    M0's 60 items had such a key: `…/baselines/novel_key/n`). A6 reads "top-2 by the M0 lookup"
    as the key's own top two, so the decision's "else" covers such a column as well as one whose
    looked-up labels are all `other`. The fallback then orders the aspects by that same training
    prior.
  - **The fallback** ("else all registry-allowed aspects capped"; reading, confirmed by the
    user on 2026-10-04). It takes the aspects of `registry.ASPECTS` except `other` that have an ontology in
    play (`allowed_for_aspect(aspect) ∩ in_play`), `unit` only for a column with a unit
    (`col.unit`; for a column without one, S would search UO for the literal word "unit"). They
    are ordered by the lookup's training prior (M0's `Lookup.prior` without the annotated card:
    most rows first, a tie to the label first seen in card order, as M0's `majority`; the aspects
    without rows follow in registry order), and the first **two** are kept. It reads no CLM
    answer and needs none, so it also chooses aspects when `column.ontology_fits` is decided by
    `ols_rank` or clm-serve is down; Q3 then runs on those aspects like any other. On the frozen
    table the prior starts with `method` (18 rows; 13 to 17 once a bench card's own items are
    left out), followed by `taxon` (12) or, for four of the seven bench cards, `measurement`, so
    a fallback column gets `method` and one of those two. The decision says "capped" but names
    neither the number nor the order: two is the lookup's own number, and the order is what M0
    gives a key it has not seen. Registry order, or reporting such a column `no_aspect`, would be
    the alternatives. (The first build of A6 asked Q3 once for every aspect and kept the two with
    the best zero-shot `p_fit`. Review found that this let CLM's uncalibrated `p_fit`, compared
    across seven aspect contexts, choose the aspects, a use of F9 no registered result covers,
    and that it offered `unit` to columns without a unit. The deterministic order replaces it.)
- **Q7 `avu.value_kind`.** `avu.pre_rule_value_kind` first, as before (no record). Where it
  returns `None`, exactly the proposals M2 asked CLM about, the value kind is "the term label"
  (`registry.VALUE_KINDS[0]`): a `rule` record (reason `a6_value_kind_label`) under the
  proposal's group, with the proposal's decision as its parent, as the CLM record was.
- **CLM still answers Q1, Q2 and Q7, in the same requests as in M2.** Q1 and Q2 are asked
  together for exactly the columns M2 sent: every non-identifier column the planner did not
  mark `annotate=False`, plus an identifier column the planner marked `annotate=True`. A
  non-identifier column the planner excluded is annotated without an audit record, since M2
  never asked about it. Q7 is asked for every proposal the pre-rule leaves open. Every such
  record is **audit-only**: stored with outcome `abstain` and reason `audit_only_a6` whatever the
  policy's verdict, with its numbers kept as served. The verdict is still computed, so a
  misconfigured threshold still fails the run. An audit-only record never decides whether a
  column is annotated, which aspect is used or which value kind is built, and never makes a link
  or a label. The vocabulary and the sidecar schema are unchanged (no migration): `abstain` is
  an existing outcome and `decisions.reason` is free text. `explain` lists these records as
  decisions with that outcome and reason. `review`, the pending groups and `feedback` offer only
  a group's deciding record, which is never one of them: a Q1 or Q2 record has no group, and a
  Q7 record hangs under a term group whose `winner_decision_id` is never it. The bench never
  counts them: it reads only frozen label snapshots (D30), and the one producer of labels from a
  run, a pick on a group, labels that group's deciding record. `AnnotationRun.n_audit_only`
  (also in the `--out` JSON) counts them, and `outcomes` counts them under `abstain`. An audit
  question CLM could not answer is marked audit-only too, and the run is still `degraded` (CLM
  did not answer, A1).
- **What a run records of the lookup.** In a `rules` run `runs.labels_sha256` (the existing
  column for the label snapshot in force) names the snapshot the table was frozen from
  (`aafd18f8…`); under `clm` it stays NULL, as in M2. Each Q3 group's `search_json.a6.lookup`
  holds the key, the counts and their Laplace(α=1) frequencies, the top two, the card held out,
  the number of items counted and the table's sha256. A column with no Q3 group (its only aspect
  `unit`, decided `uo` by rule, or aspects with no ontology in play) carries no copy of its
  counts. They still follow from the table (pinned by its sha256), the run's card and the
  column's name, since every host reads the same table.
- **What it amends.** **D28**: rank-and-cap no longer prunes Q1 and Q2 by CLM's top-k (Q1 by its
  top-1 `Yes`, Q2 by its top-1 when proposed, else top-2); it still prunes Q3's ontologies to the
  top two, and `ols_rank` is unchanged. **A1**: its bullet "The closed choices are unchanged …
  they keep F7 and their behaviour" now holds for their framing only (F7 stays, for the audit
  records and the rule records alike). **Plan §4.2**: the rows Q1, Q2 and Q7 are replaced by the
  rules above. Q1's identifier rule and Q7's pre-rule and fallback "the term label" are kept,
  with no CLM answer between them and the result. Q1's planner rule no longer decides. Row Q3 is
  unchanged: it is asked per chosen aspect, as in M2.
- **Unchanged.** Everything else is M2's: the identifier rule, Q3 for every chosen aspect, S,
  Q4–Q6 and Q4b, K1's `ols_rank` for `term.fits` (A1), Q8 (D25) and the policy (plan §4.7: a
  record's verdict is unchanged; only what is stored for an audit record is overridden, as M2
  already stored a moot aspect and the keep rule's drops as `rejected`). D10 holds: annotate
  decides, proposes and records, apply writes only what was accepted, and an audit record makes
  no link. D22 holds as M2 read it: the planner's aspect hint, appended to CLM's aspects in M2,
  now comes first, recorded as a rule (model `planner`, level `none`), and never yields a
  probability or an `auto`. **D15 stands**, read as it is written, about learned models: no
  calibrator, probe or head is fitted or promoted at serving time. The lookup is M0's no-model
  count table, counted at annotate time from items frozen offline, and nothing in a sidecar is
  read or written for it. K1 is not invoked: it concerns X1's two rank_fit tasks, and A6 only
  borrows its word "audit-only".

**Why.** In M2's registered tier run, CLM's zero-shot answers to the closed choices are worse
than answering each task's majority class on the same items:
`bench/results/2026-10-03/tiers.json#/cells/neon_annotate.zero_shot.F7` accuracy **0.357**
against majority **0.643** (`/metrics/acc`, `/baselines/majority_acc`; n 98),
`#/cells/neon_aspect.zero_shot.F7` **0.050** against **0.300** (n 60) and
`#/cells/neon_value_kind.zero_shot.F7` **0.324** against **0.478** (n 278); RESEARCH.md, "M2
results (registered run)". Under M2 those answers decide which columns are annotated, which
aspects are searched and which value an AVU carries. No pre-registered rule covers these tasks
(X1 and K1 concern the two rank_fit tasks; A1: "no pre-registered rule decides anything for
`column.annotate`, `column.aspect` or `avu.value_kind` from their tier cells"), so the frozen
criteria give no instruction, and A1 kept their behaviour. Having seen the cells, the user chose
the deterministic fallbacks above. Q1's rule and Q7's fallback coincide with the majority class
of those cells (`Yes`, 63 of 98; "the term label", 133 of 278: `/counts/class_counts`), which
the user had seen. Q2's lookup is the control whose accuracy the same aspect cell reports
(0.550). No cell is re-read or changed, and nothing is claimed about the rules' accuracy on any
card. Because A6 was decided after the results were seen, it is recorded as an amendment made
after M2's results, and no claim rests on it.

**What it changes in a run.** Every non-identifier column reaches S and Q4 (in M2, only the
columns CLM answered `Yes` did). The only exception is a column whose aspects have no ontology
in play. A column has at most three aspects (two from the fallback), each with at most two
ontologies, plus the planner's ontology as in M2. The OLS fixture closure already covers every
such column under every aspect (`ols_closure.enumerate_groups`), so the hermetic suite needs no
new fixture. A column-scoped AVU carries the term label unless the pre-rule says otherwise:
annotate no longer proposes "the column name" or "the most frequent data value". Which aspects
a run searches no longer depends on the host's sidecar. **Cost, and the M3 latency budget.** A
`rules` run asks CLM about more than an M2 run whose CLM answered `No`: every non-identifier
column gets Q3 for each chosen aspect but `unit` (at most three), S, Q4 and a Q7 audit question
for each open proposal. `bench/results/2026-10-01/annotate_latency.json` (M1: 9 to 26 CLM calls per
card, cold p50 3.34 s with live OLS) was measured before A6 and no longer describes the default
pipeline. It must be re-measured under `closed_choice: rules`, with live OLS, before the annotate
latency budget is fixed by amendment before M3 (plan §8, M3).

**Evidence and rotations.** The three cells above and the user's decision. No framing,
`question_key`, `task_key`, label, fingerprint, serving or framings lock, and no cell rotates;
no sidecar migration. New: the package file `src/mesa_clm/aspect_lookup.json` (NEON data CC BY
4.0, attributed in `THIRD_PARTY.md`) and its pinned sha256; `runs.labels_sha256` is set in a
`rules` run. Every run's `config_sha256` changes (the new `decider.closed_choice`). Building and
testing the table reads the snapshot's 60 `column.aspect` silver labels, the items M0's lookup
cell already read, and combines them with no model output.

Tests: `tests/unit/test_a6_closed_choice.py` covers:
- no CLM answer drives Q1, Q2 or Q7: runs whose closed-choice answers a test double pulls in
  opposite directions decide the same records, groups, links and proposals (the same pulls
  change them under `clm`), and runs whose Q3 answers it pulls apart keep the same aspects;
- the rule and audit records and their markers;
- Q1 whatever the planner says, both ways;
- the packaged table: rebuilt from the snapshot byte for byte, its sha256 pinned and checked,
  and per fold the lookup M0 computed;
- the hint and the lookup's top two uncapped, with M0's ties and the annotated card held out;
- the fallback in the prior's order, capped at two, never `unit` without a unit, needing no CLM
  answer (`ols_rank`, CLM down), and bounded by the ontologies the plan puts in play, the
  planner's ontology appended as in M2;
- a sidecar file without the schema;
- `clm` mode as M2;
- all seven fixture cards end to end in both modes under the shipped decider, with zero
  `ReplayMiss` and at most three Q3 questions per column;
- explain, the pending groups, the offered candidates and feedback never offering an audit
  record;
- the CLI.

Other tests: `tests/unit/test_pipeline_fake.py` runs all seven cards in both modes, with M2's
closed-choice assertions in `clm` mode, a planner's exclusion in both modes and CLM down in both
modes. `tests/unit/test_k1_default.py` and `tests/unit/test_cli_m1.py` no longer depend on how
many term groups a run searches, and `tests/unit/test_config.py` checks that the example files
document the new field.

Checks outside the repository, before the commit:
- 63 fake runs under `closed_choice: clm` gave rows identical field for field to `81e4495`. They
  covered the seven fixture cards under three fake heads, each in the audit and the shipped
  configuration, plus the CLM-down, CLM-down-with-hints and `ols_rank` paths: 4,114 decisions
  and 401 proposals. This is not committed as a test, because the fake's floating-point detail
  may differ across platforms.
- For each review finding about behaviour, one test failed on the reviewed build and passes on
  this one: the uncapped hint and lookup, the lookup answering on a host without labels, the
  planner's exclusion, the unseen key, the fallback without Q3 `p_fit`, `runs.labels_sha256`, Q3
  never choosing an aspect, `unit` without a unit, the bare sidecar file, and the Q3 count per
  column.

**Correction (2026-10-05, appended by the M4 pre-run integrity check; nothing above is edited).**
The sentence "Building and testing the table reads the snapshot's 60 `column.aspect` silver labels
… and combines them with no model output" is true of the table and its tests, but the A6 review
round of 2026-10-04 also ran two exploratory checks that did combine a model output with those
labels, which this register did not record: `/tmp/a6audit/live_fallback_probe.py` (14:56Z–14:57Z)
asked live clm-serve the F9 Q3 question for the 97 columns of the seven bench cards (679
`clm/zero_shot` answers) and joined the answers to the snapshot's 40 labelled `column.aspect`
items (`fallback_top1_equals_silver` 6, `silver_in_fallback_kept` 18, silver majority `method`
18 / `taxon` 12), and a skeptic's rerun (`/tmp/skeptic2/probe.py`, 15:27Z–15:28Z) reproduced it
(97/97 columns). Both were exploratory ("not evidence for any cell", their own docstrings), read
the snapshot read-only, wrote nothing into the repository, and touched no registered number:
`column.aspect` has no citable cell (below every floor; the M2 cells of the closed choices are
fixed arms, M2 plan §9.5), A6 is a product-safety decision not a scientific claim, and M4's
`column.aspect` probe cell is calibrator-floor skipped by construction (M4 plan §0.4). They are
recorded here and in the M4 pre-run disclosure so that the register's account of every look at
labelled bench data is complete.

## Plan (summary; the full plan is `design/plan-2026-09-28.md`)

Milestones, each a `feat/mN-*` branch merged by PR after `scripts/wait_for_checks.sh`:
M0 scaffold, house files, vendored files, ports (config, secrets, cards, states, registry, ols,
avu, planner, labels, metrics), fixtures and the OLS fixture closure, anyjev parity, bench
baselines and MDE, labels snapshot → M1 two tracks: A serving on sparky-1 (encoder, patched
clm-serve, keys, lock, doctor, collapse spike, features, fallback parity) and B the hermetic
pipeline (framings and lock, render, `_honest`, fake transport, rank-first pipeline, sidecar,
service); gate G1 freezes D0–D32 and the pre-registration → M2 evidence (X1, X2, zero_shot and
calibrated cells; K1; amendment A1; run 2026-10-03, [A1](#a1-2026-10-03--amends-d24-d28): F9 for
`column.ontology_fits`, K1 for `term.fits`; then, by the user's decision after the results,
[A6](#a6-2026-10-04--amends-d28-a1-and-plan-42-q1-q2-q7): the closed choices Q1, Q2, Q7 by
deterministic rules, CLM's answers to them audit-only) → M3 live proposed-only loop (apply,
revert, history, tools, smoke; tag v0.1.0a1; the annotate latency budget that plan §8 fixes by
amendment before M3 comes from a re-measurement under A6's `closed_choice: rules`, because
`bench/results/2026-10-01/annotate_latency.json` predates A6) → M4 learned tiers (X3, X4,
artifacts, citations, audits; K2) → M5 release 0.1.0 → M6 neon adapter (0.2.0) → M7 head tier
(K3) → M8 scale-out and evidence-gated auto.
