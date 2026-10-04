---
title: "Configuration"
description: "Configure mesa-clm with a YAML file, MESA_CLM_ environment variables or flags; precedence, the sections, secrets, and the CLM fallbacks."
type: Reference
tags:
  - getting-started
  - configuration
  - environment
generated:
  by: "claude/opus-5.5"
  at: "2026-10-04T18:00:00Z"
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
a raw 0600 file; `MESA_CLM_SECRETS=auto|env|file|keyring` chooses where secrets come from. When
neither `clm.api_key` nor `clm.api_key_file` is set, the key is read from
`~/.mesa/clm/secrets/clm.key` if that file exists (what `mesa-clm serve keys --init` writes),
and likewise `encoder.key` for the encoder: only in the `auto` and `file` modes, after an
inline value, a configured file and (in `auto`) the keyring; an explicit empty value
(`MESA_CLM_CLM__API_KEY_FILE=`) turns the default off, and so does an explicit `null` in the
YAML file, which is why `config.yaml.example` keeps the four key settings commented out. The default is resolved when the key is
read, never stored in the configuration, so `config_sha256` does not change with it, and the file
must pass the same 0600 and owner rules as any key file. No secret is ever logged, hashed into
a fingerprint or written to a run record, and a configuration error never shows a value or a
snippet of the file (a YAML syntax error names the file, line and column).

## Sections

| Section | What it configures |
|---|---|
| `clm` | The clm-serve URL (`http://127.0.0.1:8700`), key (default file `~/.mesa/clm/secrets/clm.key`), model (`clm-latest`, `clm-raw` or a promoted head), `allow_remote` (needs https). |
| `encoder` | The vLLM pooling URL (`http://127.0.0.1:8090`), key (default file `~/.mesa/clm/secrets/encoder.key`), `max_len` (4096, the server's window; clients truncate at 4,095), and `tokenizer_json` (the pinned Qwen3 `tokenizer.json`, counted by the `tokenize` extra when the encoder has no `/tokenize`; unset, the token guard over-counts with `ceil(chars / 2)`). |
| `decider` | `tier`: `auto` (the best promoted artifact, else `zero_shot`), `zero_shot`, `calibrated`, `probe`, `head`, or `ols_rank` (the degraded OLS top-1 method for every candidate group, proposed-only, DESIGN D28); `ols_rank_fallback` (default `false`): whether `annotate --provider clm` at tier `auto` runs `ols_rank` when clm-serve does not answer instead of refusing; `ols_rank_tasks` (default `[term.fits]`, DESIGN A1's K1): the candidate-group tasks `ols_rank` decides whatever the tier (a YAML list, or `MESA_CLM_DECIDER__OLS_RANK_TASKS` as a list, a comma-separated string or `none`); `[]` asks CLM about `term.fits` too, an audit run; `closed_choice` (default `rules`, DESIGN A6): who answers the closed choices, `rules` (every non-identifier column annotated whatever the planner says; aspects from the planner's hint and the M0 lookup over the packaged table frozen from the registered labels snapshot, else the first two in the lookup's prior order; value kind "the term label"; CLM's answers recorded audit-only) or `clm` (the M2 behaviour; `MESA_CLM_DECIDER__CLOSED_CHOICE`, `annotate --closed-choice`). Requests always use temperature 1; calibration is client-side. |
| `planner` | The reasoning model that only plans: `static` (default in 0.1.0), `claude`, `gateway`. |
| `claude` | Credentials profile and model for the planner and the second opinion. |
| `policy` | Profile (`prod` needs `calibrated` for auto-writes and an audit; `dev` allows `--history none`), thresholds file (`policy_defaults.yaml`). |
| `artifacts`, `features` | Artifact root and feature cache under `~/.mesa/clm/`, keyed by fingerprint. |
| `provenance`, `ducklake` | Sidecar DSN (`duckdb:///~/.mesa/clm/provenance.duckdb`, a bare `*.duckdb` path, or `postgresql://`), TTL for abandoned runs; local DuckLake catalog for development. The DuckDB sidecar, its lock and everything mesa-clm creates for it are owner-only (0700/0600). |
| `history` | `backend` (`direct`, `spool`, `none`, `auto`), `multiwriter`, spool root, lock paths (`$MESA_HOME/locks/mesa-catalog.lock` plus neon's when configured). |
| `apply` | Batch size (≤500 operations per call) and the token bucket (`max_calls_per_s`, default 5). |
| `ols` | OLS base URL and the fixture directory for replay. |
| `neon` | The neon-ducklake root, iRODS root and curation-sync switch for the adapter (0.2.0). |

The configuration module is ported from mesa-anyjev in M0; the serving-facing sections
(`clm`, `encoder`, `decider`) are live since M1 (`annotate --provider clm`, `doctor --serve`) and
`history`/`apply` become live in M3. The authoritative field list is
the `mesa_clm.config` model and `config.yaml.example` in the repository.
