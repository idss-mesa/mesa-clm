---
title: "Architecture"
description: "The mesa-clm modules and which milestone delivers each, the annotate and apply phases, the MCP tools, and the boundaries with CLM, mesa-mcp, mesa-ducklake and neon-ducklake."
type: Reference
tags:
  - develop
  - architecture
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T00:00:00Z"
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
numpy re-implementation of the head for local probes; `clm/fake.py`, a deterministic hashed
n-gram fake with a collapse knob) below the providers (`DecisionRecord` with level,
calibration and fingerprints; the tiered provider that resolves a tier per `question_key`)
below the pipeline (`Annotator.annotate`, Q1–Q8). The planner is a separate role that only
proposes. `mesa_mcp.ols` provides the OLS client and the canonical AVU transform;
`mesa_ducklake.DuckLakeClient` records snapshots; the vendored `clm/schema.py` is reached only
through `render.py` (DESIGN D4).

## Modules by milestone

| Milestone | Modules |
|---|---|
| M0 (this) | `config.py`, `secrets.py`; `cards.py`, `states.py`, `registry.py`; `tasks.py`, `identity.py`; `policy_defaults.py` + `policy_defaults.yaml`; `ols.py`, `avu.py`, `ols_closure.py`; `planner/`; `provenance/labels.py`; `learn/labels.py`; `bench/_metrics.py` (vendored), `bench/{stats,results,baselines,mde}.py`, `bench/tasks/`; `health.py` (pins, versions, plugin API, policy, stores); `mcp_tools/` (entry-point stub); `cli.py` (`labels`, `bench`, `doctor`); `_vendor/clm/` |
| M1 | `framings.py`, `render.py`, `framings.lock.json`; `clm/{http,encoder,headproj,fingerprint,fake}.py`; `providers/`; `pipeline.py`; `policy.py`; `provenance/{models,store,migrate,export}.py`; `service.py`; `health.py` serving probes; `learn/features.py`; `serving/`, `deploy/` |
| M2 | `bench/{run,framing}.py` |
| M3 | `apply.py`, `revert.py`, `irods_io.py`, `history/`; the five `mesa_clm_*` tools in `mcp_tools/` |
| M4 | `learn/{linear,calibrate,fit,teacher}.py`, `artifacts.py`, `bench/e2e.py`, audits |
| M6 | `adapters/neon.py` |
| M7 | `learn/finetune.py`, `serving/finetune_preflight.py`, `serving/rescore_head.py` |

Deferred to M8: DataCite questions, the chooser for mesa-mcp's own picker, dense retrieval
over pre-embedded registry ontologies, `mesa_clm_rank`.

## Service and tools (milestone M3)

`service.py` holds the collaborators (provider, planner, OLS layer, policy, store) and owns the
human-feedback path (`record_human_pick(via=)`), the candidate read-back for elicitations and
the run-owner checks (DESIGN D21). The CLI and the five tools build on it:

| Tool | Phase | What it does |
|---|---|---|
| `mesa_clm_annotate` | decide | Runs the pipeline on a card (`card_text` or an iRODS path checked by `assert_allowed`; local paths are CLI-only), records every decision and proposal, returns proposals with `p_fit`, level, calibration and the fingerprint; writes nothing. |
| `mesa_clm_apply` | write | Writes accepted AVUs to an iRODS path (`dry_run=true` by default); `accept="proposed"` asks one elicitation per open group first with an ids-only state (DESIGN D26); anonymous callers are refused; history is spooled. |
| `mesa_clm_explain` | read | A run's decisions, top options per group, links and pending groups; the owner's runs only. |
| `mesa_clm_feedback` | curate | Pick, reject or decline on a group; recorded as `agent_pick` (weight 0) unless it came through an elicitation. |
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
