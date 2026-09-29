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
| D5 | Fingerprints (`encoder_fp`, `clm_model_fp`, `schema_sha256`, `serving_lock_sha`) on every decision, feature row, artifact and bench cell; exact match required | accepted |
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
| D16 | Topology = encoder URL (:8090) + clm-serve URL (:8700); the fallback is an in-process `Engine(embedder=CudaEncoder)` that also serves :8090; `route` enters `encoder_fp` | accepted |
| D17 | clm-serve heads run on CPU (`--device cpu --action-cache 512MiB`); the encoder gets the whole 0.20 GPU share | accepted |
| D18 | Probe before head; a head is promoted only if it beats the probe under rule R | accepted |
| D19 | Teacher labels from `accepted` items of neon `curation/generic/<DP>.validated.json`, in-registry only, weight 0.5 / implicit 0.3, never fold-eligible, leak group = product; refuse files whose replicate model starts with `mesa-clm`; corpus frozen by sha256 before M6 | accepted |
| D20 | Label weights are `sample_weight` in Platt, logreg, LDA and ridge; heads exclude teacher rows; a test shows the teacher weight changes the fit | accepted |
| D21 | Curator labels only from MRTR elicitation or the interactive CLI; a plain tool pick is `agent_pick` (weight 0, not fold-eligible); runs store `owner` and every follow-up checks it | accepted |
| D22 | The planner plans, never decides; Claude is a second opinion only, never `auto` | accepted |
| D23 | Builders and `state_sha256` byte-identical to mesa-anyjev; contexts come from additive views that end with the target; `--max-model-len 4096` plus a client token guard | accepted |
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

## Pre-registration (G1)

**Registered in M0, frozen at G1.** The text below is copied from plan §5.4, §5.6 and §8 and is
committed verbatim at gate G1, before any M2 run. After G1 it changes only by amendment, and an
amendment made after an experiment has run marks every affected cell `exploratory:true`.

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
  all eligible → amendment reopening D3; (5) nothing qualifies → K1; (6) anchor variant by NLL.
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

None yet. Format for each entry: `### A<n> (YYYY-MM-DD) — amends D<k>` followed by the new
decision, the evidence (results JSON or RESEARCH.md fact) and which cells, locks or artifacts
it rotates. The first expected amendment is A1, the production framing chosen by X1 (M2).

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
