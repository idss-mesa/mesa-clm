---
title: "Install"
description: "Install mesa-clm with uv, the command line that exists after milestone M1 (annotate, explain, review, feedback, framings, provenance, serve, doctor), the extras, and what a serving host needs."
type: Guide
tags:
  - getting-started
  - install
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T20:00:00Z"
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

## Command line (milestone M1)

`mesa-clm --help` lists the verbs that exist. Global options go before the verb: `--config` (a
YAML file; `MESA_CLM_*` variables and flags override it), `--provenance` (the sidecar DSN,
`duckdb:///<path>` or `postgresql://…`) and `--actor` (recorded on labels, runs and overrides;
default `$USER`); `--provenance` and `--actor` are also accepted after the verb. In a
`duckdb:///` DSN the path follows the third slash, so `duckdb:///x.duckdb` is relative to the
working directory and `duckdb:////tmp/x.duckdb` is absolute. Exit codes: 0 ok; 1 the verb ran
and failed (a doctor failure, framing drift, an unreachable clm-serve, an OLS replay miss,
another owner's run); 2 a configuration or usage problem (an unreadable config file or card, a
bad argument). No verb prints a key: a configuration error names the field and the problem,
never the value, and `serve keys` names files only.

| Verb | What it does |
|---|---|
| `doctor [--quick] [--serve] [--json]` | What this host can run (below). `--quick` skips the vendored-file hashing and the serving-lock verification; `--serve` forces the live serving probes; `--json` prints `{ok, summary, checks[]}`. |
| `framings --check \| --update-lock [--lock PATH]` | The framing keys against `framings.lock.json` (DESIGN D1): `--check` exits 1 on drift (CI runs it); `--update-lock` rewrites the lock after a deliberate framing change, which rotates that framing's `question_key`. |
| `annotate --card PATH [--provider clm\|fake] [--planner static\|gateway\|claude] [--tier auto\|zero_shot\|calibrated\|probe\|head\|ols_rank] [--owner O] [--out FILE\|-] [--eval-result] [--fake-seed N]` | Decide one dataset card (a local file; CLI only, DESIGN D26) and record the run in the sidecar. Prints a summary (run id, proposals, abstentions by reason, decisions, CLM calls, input tokens, seconds, fingerprint) and the next step; `--out` writes the run as JSON (`-` for stdout), `--eval-result` the neon-avu-eval result shape instead. Every outcome is proposed-only: the shipped policy has `auto: null` everywhere. |
| `explain --run-id ID \| --irods-path P [--limit N]` | A run read back from the sidecar: slim decisions, the top three options per group, links and the groups waiting for a reviewer. Run ids may be given as a unique prefix (at least 4 hex digits). |
| `review --run-id ID [--pick GROUP=OPTION_KEY\|none]... [--decline GROUP]...` | Answer a run's pending groups. Without `--pick`/`--decline` it is interactive and needs a terminal: each group's candidates with `p_fit` and the anchor, then a number picks, `0` or `n` is "none of these", `d` declines, `s` (or Enter) skips, `q` quits. The non-interactive form checks every answer before recording any. Answers are curator labels (`via=cli`, DESIGN D21); a decline keeps only an override row. |
| `feedback --group-id G --action pick\|reject\|decline [--option-key CURIE]` | One answer for one group (`via=cli`); a pick without `--option-key` and a `reject` are "none of these". |
| `provenance migrate [--dsn DSN]` | Create or upgrade the `mesa_clm` schema (DuckDB: the bootstrap; Postgres: the packaged migrations, `pg` extra). |
| `provenance export --run-id ID --out DIR` | Parquet copy of a run's rows plus `manifest.json` under `DIR/<run_id>/`. |
| `provenance import PATH` | Commit an exported run into the configured sidecar (refused when the run exists). |
| `provenance prune [--ttl-days N] [--dry-run]` | Delete the local rows of terminal runs and of abandoned runs past the TTL (default `provenance.ttl_days`); `--dry-run` lists what would go. |
| `serve keys --init \| --rotate [--secrets-dir DIR]` | The two bearer key files and the units' env files, all 0600 in a 0700 directory (default `~/.mesa/clm/secrets`). Prints paths and the `export MESA_CLM_*__API_KEY_FILE=…` lines, never a key; `--rotate` also prints the `systemctl --user restart` command to run (the running units keep the old keys until restarted). |
| `serve units [--home DIR]` | Print the rendered systemd user units (installing and enabling them stays a user action). |
| `serve lock --check [--lock PATH] [--home DIR] [--require-live]` | Verify `serving/serving.lock.json` against this host (patch files, installed lock copy, serve clone and applied patches, schema, head file, image and running container); exit 1 on any failure. Absent pieces are skipped unless `--require-live`. |
| `labels ingest-neon-eval --eval-root DIR [--exclude-models a,b]` | Silver labels from a neon-avu-eval checkout (`tests/fixtures/neon-avu-eval` in this repository), terms resolved through OLS (`MESA_CLM_OLS__FIXTURES=replay` for the recorded fixtures). |
| `labels import-anyjev --dsn DSN` | Labels from a mesa-anyjev sidecar, opened read-only, re-identified per DESIGN D1. |
| `labels snapshot [--out PATH]` | Freeze the store to Parquet and print its `labels_sha256` (DESIGN D30); default `bench/snapshots/<today>.parquet`. |
| `labels stats` | Counts per task, source, label and weight as JSON. |
| `bench baselines [--out-dir DIR] [--date D] [--snapshot PATH] [--B N] [--seed S]` | The no-model controls per task (`lookup_prob`, novel-key, leave-one-product-out) into `<out-dir>/<date>/baselines.json` and `.md`, stamped with the snapshot's `labels_sha256`; the default snapshot `bench/snapshots/<date>.parquet` is written when missing and refused when the store has changed since. |
| `bench mde [--n-sims N] ...` | The minimum detectable effect under rule R at the realised label counts, into `<out-dir>/<date>/mde.json`. |

The label verbs (`labels`, `bench`) need a DuckDB sidecar; the annotation and provenance verbs
also take a Postgres DSN.

**Owners (DESIGN D21).** A run belongs to its owner: `annotate --owner` (default `--actor`,
itself `$USER`). `explain`, `review` and `feedback` act as `--actor` and refuse another owner's
run with exit 1 (the message never names that owner).

**Providers and an unreachable clm-serve.** `--provider fake` is a deterministic offline fake
(hashed n-grams; `--fake-seed` picks the fake head): it exercises the pipeline, the policy and
the sidecar and is never evidence. `--provider clm` (the default) talks to the loopback
serving pair with the keys from the configuration, stamps every record with the fingerprint of
the serving lock the host runs (`~/.mesa/clm/serving.lock.json`, else the checkout's
`serving/serving.lock.json`), and asks clm-serve's `/health` first. When clm-serve does not
answer:

* tier `auto` with `decider.ols_rank_fallback: true` (`MESA_CLM_DECIDER__OLS_RANK_FALLBACK`)
  runs the degraded `ols_rank` method instead (the OLS top-1 per group, proposed-only, never
  auto; DESIGN D28) and the run says `degraded`;
* otherwise the verb refuses with exit 1 (the plugin's `decider_unavailable`), and an explicit
  CLM tier (`zero_shot`, `calibrated`, …) always refuses;
* `--tier ols_rank` (or `decider.tier: ols_rank`) never needs clm-serve for the candidate
  groups; the closed choices (annotate, aspect, value kind) are still asked and recorded
  `unavailable` when it is down.

A learned tier the provider cannot serve (`calibrated`, `probe`, `head` before M4/M7) is refused
before any request. An OLS replay miss fails the run (exit 1): what was decided before it is
committed with status `failed`.

The M0 bench sequence, as run for `bench/results/2026-09-29/`: `labels ingest-neon-eval
--eval-root tests/fixtures/neon-avu-eval` → `labels snapshot --out
bench/snapshots/2026-09-29.parquet` → `bench baselines --date 2026-09-29` → `bench mde --date
2026-09-29`. The [quickstart](quickstart.md) walks through annotate, explain and review.

## The doctor

Every check prints `ok`, `WARN` or `FAIL`; the doctor exits 1 on any failure. The M0 checks
cover the vendored files (`vendored.sha256`), the configuration's secret-free hash, Python,
duckdb (the 1.5.x window), mesa-mcp and mesa-ducklake against their pinned commits, mesa-mcp's
plugin API and the `clm` entry point, the policy defaults, the provenance path, the label store,
key files (stat-ed for mode 0600, never read), the eval root and the OLS fixtures. M1 adds:

| Check | What it compares |
|---|---|
| `framings lock` | The framing keys against `framings.lock.json` (the `framings --check` rule). |
| `schema sha256` | The vendored CLM `schema.py` hash every record cites against its `vendored.sha256` entry and the serving pin. |
| `sidecar schema` | The `mesa_clm` schema version of the configured sidecar, read without creating anything (a DuckDB file opened read-only under the shared flock; Postgres through the migration table). Older: warn and `provenance migrate`; newer than this mesa-clm: fail. |
| `serving lock` | `serve lock --check` when the serving home `~/.mesa/clm` exists; skipped by `--quick` unless in serve mode. |
| `host` | `MemAvailable`; a warning below the encoder's 8 GiB running floor. |
| `gpu_budget` | Active CARC vLLM backends (`systemctl list-units carc-vllm@*`, read-only); a warning when they are active and `MemAvailable` is below the 32 GiB the encoder needs to start. |
| `serving` | Outside serve mode, one line: whether the encoder and clm-serve answer `/health` (a warning when not). |

**Serve mode** (`--serve`, or automatically when both `mesa-clm-encoder.service` and
`mesa-clm-serve.service` are active) replaces the `serving` line with live probes:
`serving binds` (both ports listen on loopback only, from `ss -ltn`), `encoder health` and
`clm-serve health` (`/health` 200), `encoder auth` and `clm-serve auth` (an unauthenticated
`GET /v1/models` gets 401), `encoder models` (`qwen3-8b` listed, with the configured key) and
`clm-serve models` (`clm-latest` and `clm-raw`), and `clm golden` (one fixed
`/v1/systemone` question answered over every key, each probability above 0, summing to 1). An
unreachable endpoint or a missing key fails under `--serve`, and under auto-detected serve mode
without `--quick`; otherwise it is a warning. A key the server rejects (401) always fails.

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

Annotation with `--provider clm` needs the two serving processes described in
[Serving](../concepts/serving.md) and the runbook [Serving](../deploy/serving.md): a vLLM pooling
container serving `Qwen/Qwen3-8B` on `127.0.0.1:8090` and a patched `clm-serve` on
`127.0.0.1:8700`, both keyed. In 0.1.0 they run on one GPU host (sparky-1); the core refuses a
non-loopback URL unless `MESA_CLM_CLM__ALLOW_REMOTE=1` is set **and** the URL is `https://`
(use an `ssh -L` forward otherwise). Point mesa-clm at the key files (paths, never the keys);
`mesa-clm serve keys --init` prints these two lines:

```bash
export MESA_CLM_CLM__API_KEY_FILE=~/.mesa/clm/secrets/clm.key
export MESA_CLM_ENCODER__API_KEY_FILE=~/.mesa/clm/secrets/encoder.key
# optional: exact token counts for the guard when the encoder has no /tokenize (tokenize extra)
export MESA_CLM_ENCODER__TOKENIZER_JSON=~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/<revision>/tokenizer.json
uv run mesa-clm doctor --serve
```

The token guard counts each context through the encoder's `/tokenize` first, then the
`tokenize` extra over `encoder.tokenizer_json`, else a conservative `ceil(chars / 2)`; a context
over `max_len - 16` tokens is never sent and abstains as `truncated`. Everything else (labels,
snapshots, baselines, `annotate --provider fake`, review, provenance) runs without a GPU.

## As a mesa-mcp plugin (milestone M3)

Installed beside mesa-mcp (≥ `c74f3aa`, which has the entry-point loader), mesa-clm registers
the five `mesa_clm_*` tools under the surface `clm` through the `mesa_mcp.tools` entry point
`clm`. The umbrella installer (`idss-mesa/docs` `install.sh`) makes it opt-in
(`MESA_PLUGINS="mesa-anyjev mesa-clm"`) because serving exists only on the GPU host.
