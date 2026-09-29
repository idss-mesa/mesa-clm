---
title: "Install"
description: "Install mesa-clm with uv, choose the extras for the bench, Claude, Postgres or end-to-end tests, verify the vendored files, and know what a serving host needs."
type: Guide
tags:
  - getting-started
  - install
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T00:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md)"
    author: "team:idss-mesa"
  - id: plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/plan-2026-09-28.md"
    title: "mesa-clm implementation plan (v1.1, 2026-09-28)"
    author: "team:idss-mesa"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# Install

```bash
git clone https://github.com/idss-mesa/mesa-clm
cd mesa-clm
uv sync --all-extras                 # dev install; nothing here pulls torch or vllm
uv run pytest -q                     # hermetic suite
sha256sum -c vendored.sha256         # vendored CLM and AnyJev files are byte-identical
uv run mesa-clm doctor               # pins, versions, plugin API, policy, stores
```

Python 3.11 or newer. `mesa-mcp` and `mesa-ducklake` come from pinned git commits
(`pyproject.toml`, `[tool.uv.sources]`; DESIGN D0); neither is on PyPI. The core is a torch-free
HTTP client: it never depends on `contrastive-lm`, torch or vllm, and the vendored CLM
`schema.py` is the only upstream code it imports (DESIGN D4).

## Command line (milestone M0)

`mesa-clm --help` lists the verbs that exist. Global options go before the verb: `--config` (a
YAML file; `MESA_CLM_*` variables and flags override it), `--provenance` (the sidecar DSN,
`duckdb:///<path>`; also accepted after the verb) and `--actor` (recorded on labels; default
`$USER`). Exit codes: 0 ok, 1 the verb ran and failed, 2 a configuration or usage problem.

| Verb | What it does |
|---|---|
| `doctor [--quick] [--json]` | Vendored files against `vendored.sha256`, the configuration, Python, duckdb (1.5.x window), mesa-mcp and mesa-ducklake with their pinned commits, mesa-mcp's plugin API (`load_plugins`, `register_tool(meta=)`) and the `clm` entry point, the policy defaults, the provenance path, the label store, key files (mode 0600, never read), eval root and OLS fixtures. `--quick` skips the hashing; `--json` prints `{ok, summary, checks[]}`. The serving probes land in M1. |
| `labels ingest-neon-eval --eval-root DIR [--exclude-models a,b]` | Silver labels from a neon-avu-eval checkout (`tests/fixtures/neon-avu-eval` in this repository), terms resolved through OLS (`MESA_CLM_OLS__FIXTURES=replay` for the recorded fixtures). |
| `labels import-anyjev --dsn DSN` | Labels from a mesa-anyjev sidecar, opened read-only, re-identified per DESIGN D1. |
| `labels snapshot [--out PATH]` | Freeze the store to Parquet and print its `labels_sha256` (DESIGN D30); default `bench/snapshots/<today>.parquet`. |
| `labels stats` | Counts per task, source, label and weight as JSON. |
| `bench baselines [--out-dir DIR] [--date D] [--snapshot PATH] [--B N] [--seed S]` | The no-model controls per task (`lookup_prob`, novel-key, leave-one-product-out) into `<out-dir>/<date>/baselines.json` and `.md`, stamped with the snapshot's `labels_sha256`; the default snapshot `bench/snapshots/<date>.parquet` is written when missing and refused when the store has changed since. |
| `bench mde [--n-sims N] ...` | The minimum detectable effect under rule R at the realised label counts, into `<out-dir>/<date>/mde.json`. |

The M0 sequence, as run for `bench/results/2026-09-29/`: `labels ingest-neon-eval --eval-root
tests/fixtures/neon-avu-eval` → `labels snapshot --out bench/snapshots/2026-09-29.parquet` →
`bench baselines --date 2026-09-29` → `bench mde --date 2026-09-29`.

## Extras

| Extra | Adds | When |
|---|---|---|
| `bench` | `scikit-learn` | The PR #13 logistic-regression baseline replica in the bench (M2). |
| `tokenize` | `tokenizers` | Exact Qwen3 token counts for the client token guard when the encoder's `/tokenize` is unavailable (M1). |
| `claude` | `anthropic` | The Claude planner and the second-opinion provider. |
| `pg` | `psycopg` | The optional Postgres dialect of the provenance sidecar (DuckDB per host is the default). |
| `e2e` | `mcp` | The end-to-end test over MCP stdio (M3). |
| `docs` | `zensical`, `pyyaml` | Building this site. |
| `dev` | pytest, ruff, mypy, `requests` | Development; `requests` only for the vendored CLM client used as a test oracle. |

## Serving hosts (milestone M1)

Annotation needs the two serving processes described in [Serving](../concepts/serving.md): a
vLLM pooling container serving `Qwen/Qwen3-8B` on `127.0.0.1:8090` and a patched `clm-serve`
on `127.0.0.1:8700`, both keyed. In 0.1.0 they run on one GPU host (sparky-1); the core
refuses a non-loopback URL unless `MESA_CLM_CLM__ALLOW_REMOTE=1` is set **and** the URL is
`https://` (use an `ssh -L` forward otherwise). `mesa-clm doctor` reports what a host can reach.
Everything else (labels, snapshots, baselines, the hermetic pipeline with the fake transport)
runs without a GPU.

## As a mesa-mcp plugin (milestone M3)

Installed beside mesa-mcp (≥ `c74f3aa`, which has the entry-point loader), mesa-clm registers
the five `mesa_clm_*` tools under the surface `clm` through the `mesa_mcp.tools` entry point
`clm`. The umbrella installer (`idss-mesa/docs` `install.sh`) makes it opt-in
(`MESA_PLUGINS="mesa-anyjev mesa-clm"`) because serving exists only on the GPU host.
