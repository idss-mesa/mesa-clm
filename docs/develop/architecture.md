---
title: "Architecture"
description: "The mesa-clm modules and which milestone delivers each, the annotate and apply phases, the MCP tools, and the boundaries with CLM, mesa-mcp, mesa-ducklake and neon-ducklake."
type: Reference
tags:
  - develop
  - architecture
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

# Architecture

Two phases (DESIGN D10): `annotate` decides, proposes and records (no writes); `apply` writes
the accepted AVUs to iRODS through mesa-mcp's shared helpers in atomic batches and records
history through a backend (DESIGN D12). Decisions are written to the sidecar before any iRODS
write, and `revert` undoes exactly what `apply` wrote (DESIGN D31).

Layers: the CLM clients (`clm/http.py` for `/v1/systemone`, `/v1/rank`, `/v1/models`,
`/health`; `clm/encoder.py` for `/v1/embeddings` with the token guard; `clm/headproj.py`, a
numpy re-implementation of the head for local probes; `clm/fingerprint.py`, the D5 bundle and
the serving lock; `clm/fake.py`, a deterministic hashed n-gram fake with a collapse knob) below
the providers (`providers/base.py`: `DecisionRecord` with level, calibration and fingerprints
and its `_honest` invariants; `providers/tiered.py`: the tiered provider that resolves a tier
per `question_key`, the fake provider and the degraded `ols_rank` records;
`providers/claude_provider.py`: the recorded second opinion; `providers/live.py`: the live
provider, which from M4 loads the promoted artifacts of `CURRENT.json` and serves a promoted
probe locally through the encoder, the pinned head's export and `learn/probe.py`'s spec
formulas, never clm-serve) below the pipeline (`pipeline.py`, `Annotator.annotate`, Q1–Q8), the policy
(`policy.py`, every outcome; from M4 the citation test `CellCitationValidator` over
`policy.results_root` and the `audit_required` blocker), the artifacts (`artifacts.py`:
versions, manifests, `CURRENT.json`, promotion, publish, pull), the audits (`audit.py`) and the sidecar (`provenance/`). `service.py` puts one provider,
planner, OLS layer, policy and store behind the decider lock for the CLI and, from M3, the MCP
tools. The planner is a separate role that only proposes. `mesa_mcp.ols` provides the OLS
client and the canonical AVU transform; `mesa_ducklake.DuckLakeClient` records snapshots (M3);
the vendored `clm/schema.py` is reached only through `render.py` (DESIGN D4).

**The live provider** (`providers/live.py`). `clm_provider(cfg)` builds what `annotate
--provider clm` decides with: a `TieredProvider` over `ClmHttpClient` (the `clm` section:
loopback URL, key from the configured secrets source, by default
`~/.mesa/clm/secrets/clm.key`) with `EncoderClient` (the `encoder` section) as the token guard's
counter, stamped with the fingerprint of the serving lock the host runs
(`~/.mesa/clm/serving.lock.json`, the copy the bootstrap installed, else the checkout's
`serving/serving.lock.json`) under the served head `clm.model`. It refuses (K4) a lock that does
not verify, a lock whose `schema_sha256` is not the vendored `schema.py`, and an installed lock
that differs from the checkout's; only `clm-latest` and `clm-raw` are servable until promoted
heads arrive (M7). `preflight(stack, cfg)` runs before any question: clm-serve's `/health`
(unguarded on loopback, 5 s, no retry), which tells an unreachable clm-serve from one whose
encoder is down, then with the keys clm-serve's model list and the encoder's, checked against
the lock (`encoder_problems`: served name, `root`, `max_model_len`, `owned_by` against the
route). The CLI turns an unanswered pre-flight into a refusal or, with
`decider.ols_rank_fallback`, the degraded `ols_rank` method (DESIGN D28), and a key, model or
encoder problem into exit 2. During the run a 401, 403 or 404 from `/v1/systemone` raises
`DeciderRefused` (the run is recorded `failed`); transport errors, timeouts and 5xx fall back
per group.

**Sidecar writes** (`provenance/store.py`). A run's rows are buffered in a `RunBuffer` and
committed in one transaction under the flock (DESIGN D11). DuckDB 1.5.6 tries `import pandas`
twice for every Python value it binds when pandas is not installed (the failed import is not
cached; RESEARCH.md, "mesa-ducklake"), so a commit that binds one parameter per value spends most
of its time in failed imports; multi-row `VALUES` statements bind just as many values. `bulk_insert` therefore binds one value per chunk
of up to 1,000 rows, the rows as a JSON array unpacked in SQL by `from_json_strict` (a failed
cast is an error, never a silent NULL): the data never enters the SQL text, which carries only
checked column names and type names from a fixed set, and JSON and timestamp columns travel as
text cast by the INSERT. A chunk JSON cannot carry exactly (a non-finite float, a value of
another type) goes through `executemany` unchanged, so the rows and every error are what the
per-row path produced (`tests/unit/test_provenance_bulk_insert.py` compares the two table by
table). `set_link_status` updates chunks of ids with `link_id IN (…)` for the same reason.

**Owner-only state** (`perms.py`). Whatever the process umask, everything mesa-clm creates for
its state is owner-only: missing directories are created 0700 component by component and
files 0600, every write tightens the sidecar's existing `locks/` directory, lock file, DuckDB
file and `.wal` (DuckDB creates the last two with the umask's mode) while a read changes no
existing mode, and the label store,
the feature store and its npz export, run exports, `annotate --out` and the serving keys'
parents follow the same rule. Directories the user made are never chmod-ed.

**Curator answers** (`service.py`, `cli.py`; DESIGN A2). `record_human_pick(via=)` mints curator
labels only for `via` `elicitation` (MRTR, M3) or `cli`, and the CLI passes `cli` only when stdin
is an interactive terminal; without one `review --pick/--decline` and `feedback` pass `tool`
(`agent_pick`, weight 0). A curator's answer settles a group (a different second answer raises
`AlreadyAnswered`); an agent's leaves it pending and is replaced by a curator's, whose accepted
link displaces the agent's. `check_answer` runs every check without writing, so `review` records
a batch all or nothing.

## Modules by milestone

| Milestone | Modules |
|---|---|
| M0 | `config.py`, `secrets.py`; `cards.py`, `states.py`, `registry.py`; `tasks.py`, `identity.py`; `policy_defaults.py` + `policy_defaults.yaml`; `ols.py`, `avu.py`, `ols_closure.py`; `planner/`; `provenance/labels.py`; `learn/labels.py`; `bench/_metrics.py` (vendored), `bench/{stats,results,baselines,mde,metrics}.py`, `bench/tasks/`; `health.py` (pins, versions, plugin API, policy, stores); `mcp_tools/` (entry-point stub); `cli.py` (`labels`, `bench`, `doctor`); `_vendor/clm/` |
| M1 | `framings.py`, `render.py`, `framings.lock.json`; `vocab.py`; `perms.py`; `clm/{http,encoder,headproj,fingerprint,fake}.py`; `providers/{base,tiered,claude_provider,live}.py`; `pipeline.py`; `policy.py`; `provenance/{models,store,store_postgres,migrate,export}.py` and `migrations/0001_mesa_clm.sql`; `service.py`; `serving.py` (keys, units with the encoder's loopback socket proxy, the serving lock with its container recipe, its verification, the container's network namespace); `learn/features.py` (the feature store with float32 vectors and the lock's vector recipe, the X1/X2 manifest, the TextCache export) and `learn/offline.py` (offline scoring from cached vectors); `health.py` M1 checks (framings lock, schema sha256, sidecar schema, feature store, permissions, serving lock, host, `gpu_budget`, the serving reachability line and the serve-mode probes: the binds' owners, the encoder socket, the units against the rendering, the encoder network namespace, the headroom timer, the 401 matrix, goldens, long input, systemone parity, drift); `cli.py` M1 verbs (`framings`, `annotate`, `explain`, `review`, `feedback`, `provenance migrate\|export\|import\|prune`, `serve keys\|units\|lock`, `features build\|export-npz\|project\|stats`, `doctor --serve`); `serving/` (with the encoder's bearer guard `vllm_auth.py`, the route probe `vllm_routes.py` and the encoder's endpoint proxy `encoder_proxy.py`), `deploy/` (the serve-venv side, never importing `mesa_clm`); `net.py`'s port-owner check (keyed clients never send a key to another account's loopback socket); `scripts/` M1-A probes and records |
| M2 | `bench/{framing,cells,x2,registered,run}.py` (X1, the tier cells, X2, the registration, the verbs), `learn/calibrate.py` (weighted Platt, temperature); after the registered run, amendment A1: `framings.ACTIVE` (F9 for `column.ontology_fits`) and `decider.ols_rank_tasks` (`term.fits` decided by `ols_rank`, K1); amendment A6: `closed_choice.py`, `aspect_lookup.json` and `decider.closed_choice` (the closed choices by rule, CLM's answers audit-only; the aspect lookup reuses `bench.baselines.Lookup` over a table frozen from the registered snapshot by `scripts/freeze_aspect_lookup.py`) |
| M3 | `apply.py`, `revert.py`, `irods_io.py`, `history/`; the five `mesa_clm_*` tools in `mcp_tools/` |
| M4 (pre-run) | `learn/linear.py` (the weighted fitters: L2 logistic regression, shrinkage LDA, ridge; the SVD reduction; the grids), `learn/probe.py` (the specs, `FeatureBuilder`, the nested inner selection, the OOF calibrator, `TeacherRows`, `ProbeArtifact`, `full_probe`), `learn/teacher.py` (the D19 ingest and the corpus pin), `learn/labels.py` teacher rows and `surviving_identities`; `bench/x3.py`, `bench/k2.py`, `bench/x4.py` (the producers), `bench/e2e.py` (the measurement half; the fold provider is planned, second wave), `bench/registered.py` `REGISTERED_M4` and `bench/run.py`'s M4 verbs; `artifacts.py` (versions, the manifest, `CURRENT.json`, `fit_version`, `promote`, `publish`, `pull`); `audit.py` (`audit sample\|review\|record`); `policy.py` `CellCitationValidator`, `StoreAuditCheck`, `default_results_root`; `providers/tiered.py`'s probe path (`ServedProbe`, `VectorCache`, the spec formulas' serving side) and `providers/live.py`'s artifact loading; `providers/base.py` `FEATURE_SPEC_LEVELS`; `provenance/models.py` `AuditRow`'s pass rule; `health.py`'s `artifacts` check; `config.py` `policy.results_root`; `cli.py` (`labels ingest-teacher`, `bench x3\|k2\|x4\|e2e`, `learn fit\|promote`, `artifacts publish\|pull`, `audit sample\|review\|record`); `scripts/record_ols_teacher.py` and `tests/fixtures/ols-teacher/`. The analysis plan is `design/m4-analysis-plan.md`; nothing here has run on real labels yet |
| M6 | `adapters/neon.py` |
| M7 | `learn/finetune.py`, `serving/finetune_preflight.py`, `serving/rescore_head.py` |

Deferred to M8: DataCite questions, the chooser for mesa-mcp's own picker, dense retrieval
over pre-embedded registry ontologies, `mesa_clm_rank`.

## Service (milestone M1) and tools (milestone M3)

`service.py` holds the collaborators (provider, planner, OLS layer, policy, store) behind a
decider lock (`DeciderBusy` when it stays taken) and owns the human-feedback path
(`record_human_pick(via=)`), the candidate read-back for elicitations and the run-owner checks
(DESIGN D21). The CLI verbs `annotate`, `explain`, `review` and `feedback` build on it now; the
five tools build on it in M3:

| Tool | Phase | What it does |
|---|---|---|
| `mesa_clm_annotate` | decide | Runs the pipeline on a card (`card_text` or an iRODS path checked by `assert_allowed`; local paths are CLI-only), records every decision and proposal, returns proposals with `p_fit`, level, calibration and the fingerprint; writes nothing. |
| `mesa_clm_apply` | write | Writes accepted AVUs to an iRODS path (`dry_run=true` by default); `accept="proposed"` asks one elicitation per open group first with an ids-only state (DESIGN D26); anonymous callers are refused; history is spooled. |
| `mesa_clm_explain` | read | A run's decisions, top options per group, links and pending groups; the owner's runs only. |
| `mesa_clm_feedback` | curate | Pick, reject or decline on a group; recorded as `agent_pick` (weight 0) unless it came through an elicitation (DESIGN D21, A2). |
| `mesa_clm_health` | ops | `doctor --quick`: locks, vendored shas, transport, stores, fingerprint. |

Tools register through `register_tool(..., meta={"io.mesa/surface": "clm"})` with 2020-12
schemas; handlers run service calls in `asyncio.to_thread`.

## The neon-ducklake adapter (0.2.0, milestone M6)

`mesa-clm neon curate --site S --neon-root R --mode replace|verify` reads the site's table cards,
runs the pipeline with the neon profile (no `uo`/`gaz`/`ncbitaxon`, aspects through
`NEON_ASPECT_MAP`) and writes `sites/<S>/curation/proposals/<DP>.rep1.json` (proposed ∪ auto) and
`.rep2.json` (auto only) with neon's record fields, so an unchanged `neon-ducklake site
validate` evaluates them. It works in an isolated CLM root by default, never writes iRODS or
history, and refuses to overwrite non-CLM replicates without backing them up.
