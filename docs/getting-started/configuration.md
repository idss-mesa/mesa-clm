---
title: "Configuration"
description: "Configure mesa-clm with a YAML file, MESA_CLM_ environment variables or flags; precedence, the sections, secrets, and the CLM fallbacks."
type: Reference
tags:
  - getting-started
  - configuration
  - environment
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

# Configuration

Precedence, highest first: command-line flag, environment variable, YAML file
(`--config FILE`), built-in default. Environment variables use the `MESA_CLM_` prefix and `__`
to descend into a section: `MESA_CLM_HISTORY__BACKEND=spool` sets `history.backend`. The model
is `mesa_clm.config.Config`; a secret-free `config_sha256` is recorded on every run. Data lives
under `~/.mesa/clm/` (provenance, locks, features, artifacts, heads, secrets, spool).

Two CLM names are honoured as fallbacks so an upstream `clm` client environment keeps working:
`CLM_BASE_URL` and `CLM_API_KEY`. Every `*_API_KEY` field has a `*_API_KEY_FILE` twin that reads
a raw 0600 file; `MESA_CLM_SECRETS=auto|env|file|keyring` chooses where secrets come from. No
secret is ever logged, hashed into a fingerprint or written to a run record.

## Sections

| Section | What it configures |
|---|---|
| `clm` | The clm-serve URL (`http://127.0.0.1:8700`), key, model (`clm-latest`, `clm-raw` or a promoted head), `allow_remote` (needs https). |
| `encoder` | The vLLM pooling URL (`http://127.0.0.1:8090`), key, `max_len` (4096), the token guard. |
| `decider` | Tier request (`auto`), temperature (always 1 on the wire; calibration is client-side). |
| `planner` | The reasoning model that only plans: `static` (default in 0.1.0), `claude`, `gateway`. |
| `claude` | Credentials profile and model for the planner and the second opinion. |
| `policy` | Profile (`prod` needs `calibrated` for auto-writes and an audit; `dev` allows `--history none`), thresholds file (`policy_defaults.yaml`). |
| `artifacts`, `features` | Artifact root and feature cache under `~/.mesa/clm/`, keyed by fingerprint. |
| `provenance`, `ducklake` | Sidecar DSN (`duckdb:///~/.mesa/clm/provenance.duckdb` or `postgresql://`), TTL for abandoned runs; local DuckLake catalog for development. |
| `history` | `backend` (`direct`, `spool`, `none`, `auto`), `multiwriter`, spool root, lock paths (`$MESA_HOME/locks/mesa-catalog.lock` plus neon's when configured). |
| `apply` | Batch size (≤500 operations per call) and the token bucket (`max_calls_per_s`, default 5). |
| `ols` | OLS base URL and the fixture directory for replay. |
| `neon` | The neon-ducklake root, iRODS root and curation-sync switch for the adapter (0.2.0). |

The configuration module is ported from mesa-anyjev in M0; the serving-facing sections
(`clm`, `encoder`) become live in M1 and `history`/`apply` in M3. The authoritative field list is
the `mesa_clm.config` model and `config.yaml.example` in the repository.
