---
title: "Install"
description: "Install mesa-clm with uv, the command line that exists after milestone M1 (annotate, explain, review, feedback, framings, provenance, serve, features, doctor), the extras, and what a serving host needs."
type: Guide
tags:
  - getting-started
  - install
generated:
  by: "claude/opus-5.5"
  at: "2026-10-03T18:30:00Z"
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
`duckdb:///<path>`, a bare `<path>.duckdb` or `postgresql://…`) and `--actor` (recorded on
labels, runs and overrides; default `$USER`; never blank). The `labels`, `bench`, `annotate`,
`explain`, `review`, `feedback` and `provenance` verbs also accept `--provenance` and `--actor`
after the verb; `doctor`, `framings`, `serve` and `features` do not. In a `duckdb:///` DSN the
path follows the third slash, so `duckdb:///x.duckdb` is relative to the working directory and
`duckdb:////tmp/x.duckdb` is absolute. Exit codes: 0 ok; 1 the verb ran and failed (a doctor
failure, framing drift, an unreachable clm-serve, an OLS replay miss, another owner's run, an
export that could not be written); 2 a configuration or usage problem (an unreadable config file
or card, a missing or rejected key, a bad argument). No verb prints a key: a configuration error
names the field and the problem (or, for a YAML syntax error, the file, line and column), never
the value or a snippet of the file, and `serve keys` names files only. Everything mesa-clm writes
for its state (the sidecar and its lock, the feature store, run exports, `annotate --out`) is
owner-only, directories 0700 and files 0600, whatever the umask.

| Verb | What it does |
|---|---|
| `doctor [--quick] [--serve] [--json]` | What this host can run (below). `--quick` skips the vendored-file hashing and, outside serve mode, the serving-lock verification; in serve mode it keeps the light probes and skips the slow ones unless `--serve` is given. `--serve` forces every live serving probe; `--json` prints `{ok, summary, checks[]}`. |
| `framings --check \| --update-lock [--lock PATH]` | The framing keys against `framings.lock.json` (DESIGN D1): `--check` exits 1 on drift (CI runs it); `--update-lock` rewrites the lock after a deliberate framing change, which rotates that framing's `question_key`. |
| `annotate --card PATH [--provider clm\|fake] [--planner static\|gateway\|claude] [--tier auto\|zero_shot\|calibrated\|probe\|head\|ols_rank] [--ols-rank-tasks TASKS\|none] [--owner O] [--out FILE\|-] [--eval-result] [--fake-seed N]` | Decide one dataset card (a local file; CLI only, DESIGN D26) and record the run in the sidecar. Prints a summary (run id, proposals, abstentions by reason, decisions, CLM calls, input tokens, seconds, fingerprint, the tasks `ols_rank` decided) and the next step; `--out` writes the run as JSON (`-` for stdout), `--eval-result` the neon-avu-eval result shape instead. `term.fits` groups are decided by `ols_rank` by default (`decider.ols_rank_tasks`, K1 in DESIGN A1); `--ols-rank-tasks none` asks CLM about them too, an audit run. Every outcome is proposed-only: the shipped policy has `auto: null` everywhere. |
| `explain --run-id ID \| --irods-path P [--limit N]` | A run read back from the sidecar: slim decisions, the top three options per group, links and the groups waiting for a reviewer. Run ids may be given as a unique prefix (at least 4 hex digits). |
| `review --run-id ID [--pick GROUP=OPTION_KEY\|none]... [--decline GROUP]...` | Answer a run's pending groups. Without `--pick`/`--decline` it is interactive and needs a terminal: each group's candidates with `p_fit` and the anchor, then a number picks, `0` or `n` is "none of these", `d` declines, `s` (or Enter) skips, `q` (or end of input) quits; its answers are curator labels (`via=cli`, DESIGN A2). The non-interactive form answers pending groups only, each once per call, and checks every answer before recording any; at a terminal its answers are curator labels, without one (a script, a pipe, an agent's shell) they are recorded like a plain tool call (`via=tool`: `agent_pick` at weight 0, never fold-eligible) and the verb says so on stderr. A decline keeps only an override row. A curator's answer settles a group; an agent's leaves it pending for a curator, whose answer replaces it. |
| `feedback --group-id G --action pick\|reject\|decline [--option-key KEY\|none]` | One answer for any group of your runs (a full id or a unique prefix); `pick` needs `--option-key` (`none` is "none of these"), `reject` is "none of these". At a terminal `via=cli`, otherwise `via=tool` as above. |
| `provenance migrate [--dsn DSN]` | Create or upgrade the `mesa_clm` schema (DuckDB: the bootstrap; Postgres: the packaged migrations, `pg` extra). |
| `provenance export --run-id ID --out DIR` | Parquet copy of a run's rows plus `manifest.json` under `DIR/<run_id>/` (0700, files 0600); the run is marked exported only once the copy is written and verified. |
| `provenance import PATH` | Commit an exported run into the configured sidecar (refused when the run exists). |
| `provenance prune [--ttl-days N] [--dry-run]` | Delete the local rows of terminal runs and of abandoned runs past the TTL (default `provenance.ttl_days`); `--dry-run` lists what would go. |
| `serve keys --init \| --rotate [--secrets-dir DIR]` | The two bearer key files and the units' env files, all 0600 in a 0700 directory (default `~/.mesa/clm/secrets`, which mesa-clm reads by default; another directory prints the `export MESA_CLM_*__API_KEY_FILE=…` lines). Prints paths, never a key; `--rotate` also prints the `systemctl --user restart` command to run (the running units keep the old keys until restarted). |
| `serve units [--home DIR]` | Print the rendered systemd user units (installing and enabling them stays a user action). |
| `serve lock --check [--lock PATH] [--home DIR] [--require-live]` | Verify `serving/serving.lock.json` against this host (patch files, installed lock copy, serve clone and applied patches, schema, head file, image and running container); exit 1 on any failure. Absent pieces are skipped unless `--require-live`. |
| `labels ingest-neon-eval --eval-root DIR [--exclude-models a,b]` | Silver labels from a neon-avu-eval checkout (`tests/fixtures/neon-avu-eval` in this repository), terms resolved through OLS (`MESA_CLM_OLS__FIXTURES=replay` for the recorded fixtures). |
| `labels import-anyjev --dsn DSN [--trust-curator]` | Labels from a mesa-anyjev sidecar, opened read-only, re-identified per DESIGN D1. Its `curator` rows arrive as `agent_pick` at weight 0 (mesa-anyjev let a plain tool call mint them) unless `--trust-curator`, which needs a terminal (DESIGN A2). |
| `labels snapshot [--out PATH]` | Freeze the store to Parquet and print its `labels_sha256` (DESIGN D30); default `bench/snapshots/<today>.parquet`. |
| `labels stats` | Counts per task, source, label and weight as JSON. |
| `bench baselines [--out-dir DIR] [--date D] [--snapshot PATH] [--B N] [--seed S]` | The no-model controls per task (`lookup_prob`, novel-key, leave-one-product-out) into `<out-dir>/<date>/baselines.json` and `.md`, stamped with the snapshot's `labels_sha256`; the default snapshot `bench/snapshots/<date>.parquet` is written when missing and refused when the store has changed since. |
| `bench mde [--n-sims N] ...` | The minimum detectable effect under rule R at the realised label counts, into `<out-dir>/<date>/mde.json`. |
| `features build --snapshot PATH [--tasks T] [--framings F] [--batch N]` | Embed every X1/X2 text of a frozen snapshot that the store of the live `encoder_fp` lacks (`~/.mesa/clm/features/<encoder_fp>/`), one text per request by default, after the exact token guard; refuses an encoder that is not the serving lock's. Serving host only. |
| `features export-npz --out DIR` | The store as CLM's `TextCache` file for `finetune.py --embed-cache DIR` (refused below the 0.9999 fp16 round-trip gate). |
| `features project [--model clm-latest\|clm-raw]` | Cache the pinned head's 512-d projections of every stored vector. |
| `features stats [--json]` | Every feature store under `features.dir`. |

The label verbs (`labels`, `bench`) need a DuckDB sidecar; the annotation and provenance verbs
also take a Postgres DSN.

**Owners and curator labels (DESIGN D21, A2).** A run belongs to its owner: `annotate
--owner` (default `--actor`, itself `$USER`). `explain`, `review` and `feedback` act as
`--actor` and refuse another owner's run with exit 1 (the message never names that owner); on a
shared Postgres sidecar they act only as the OS account. Curator labels come only from an
interactive terminal (or, from M3, an MRTR elicitation): `isatty()` is the check, so a process
that deliberately allocates a pseudo-terminal under the curator's account still passes it
(DESIGN A2, "Residual risk").

**Providers and an unreachable clm-serve.** `--provider fake` is a deterministic offline fake
(hashed n-grams; `--fake-seed` picks the fake head): it exercises the pipeline, the policy and
the sidecar and is never evidence. `--provider clm` (the default) talks to the loopback
serving pair with the keys from the configuration (by default `~/.mesa/clm/secrets/clm.key`
and `encoder.key`), stamps every record with the fingerprint of the serving lock the host runs
(`~/.mesa/clm/serving.lock.json`, else the checkout's `serving/serving.lock.json`; refused when
it contradicts the vendored schema or the checkout's lock), and runs a pre-flight first:
clm-serve's `/health`, then, with the keys, the served models and the encoder's model, window and
route against the lock. A missing or rejected key, a `clm.model` clm-serve does not serve or an
encoder that is not the lock's is never degraded (exit 2), and a request clm-serve refuses
mid-run (401, 403, 404) fails the run (recorded `failed`, exit 1). When clm-serve does not
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
key files (configured or the serving pair's defaults, named "(default)"; stat-ed for mode 0600,
never read), the eval root and the OLS fixtures. M1 adds:

| Check | What it compares |
|---|---|
| `framings lock` | The framing keys against `framings.lock.json` (the `framings --check` rule). |
| `schema sha256` | The vendored CLM `schema.py` hash every record cites against its `vendored.sha256` entry and the serving pin. |
| `sidecar schema` | The `mesa_clm` schema version of the configured sidecar, read without creating anything but the D11 lock file (a DuckDB file opened read-only under the shared flock; Postgres through the migration table). Older: warn and `provenance migrate`; newer than this mesa-clm: fail. |
| `feature store` | The store of the live `encoder_fp` under `features.dir` (M1): format 2 (float32 vectors), the vector recipe it was built under against the live serving lock's (another one, or a format-1 store, fails: every read would be refused), the export's float16 gate; stores of other fingerprints are listed, never read. |
| `permissions` | Group or other bits on mesa-clm's state (the data home and its `locks/` and `secrets/`, the sidecar with its `.wal` and lock, the feature stores): a warning naming the paths and the `chmod` that fixes them. The doctor never repairs; the next write through mesa-clm tightens its own lock and store files. |
| `serving lock` | `serve lock --check` when the serving home `~/.mesa/clm` exists; skipped by `--quick` unless in serve mode. |
| `host` | `MemAvailable`; a warning below the encoder's 8 GiB running floor. |
| `gpu_budget` | Active CARC vLLM backends (`systemctl list-units carc-vllm@*`, read-only); a warning when they are active and `MemAvailable` is below the 32 GiB the encoder needs to start. |
| `serving` | Outside serve mode, one line: whether the encoder and clm-serve answer `/health` (a warning when not). |

Every configured URL first passes the clients' loopback rule (loopback, or
`MESA_CLM_CLM__ALLOW_REMOTE=1` and https; never credentials in the URL) and is printed
redacted; a URL that does not is a failure and is never called.

**Serve mode** (`--serve`, or automatically when both `mesa-clm-encoder.service` and
`mesa-clm-serve.service` are active) replaces the `serving` line with live probes:
`serving binds` (both ports listen on loopback only, in sockets of this account, from
`ss -ltne`; while the socket unit is active the encoder's port is the user manager's),
`encoder socket` (`~/.mesa/clm/run` holds only the encoder's socket), `serving units` (the
units systemd loaded are the checkout's rendering and the running proxy was started from them,
sandboxed), `encoder health` and
`clm-serve health` (`/health` 200 without a key), `encoder auth` and `clm-serve auth` (the 401
matrix: without a key, every route vLLM registers on the encoder and an unknown path, and
clm-serve's `GET /v1/models`, `POST /v1/systemone` and `POST /v1/rank`, must answer 401; any
other answer fails), `encoder models` (`qwen3-8b` listed with the lock's model, window and
route) and `clm-serve models` (`clm-latest` and `clm-raw`), and `clm golden` (one fixed
`/v1/systemone` question answered over every key, each probability above 0, summing to 1).
Under `--serve`, or in serve mode without `--quick`, the slow probes follow: `encoder goldens`
(the reference texts of `encoder_golden_<encoder_fp>.npz` embedded one per request must equal
the reference bitwise, and as one batch agree within 1 − 1e-6), `encoder long input` (a text
longer than the window completes, truncated to 4,095 tokens from the left, with cosine ≥ 0.9999
against its last 4,095 token ids), `clm parity` (the golden question recomputed locally, encoder
plus the pinned head's numpy export, within 1e-4 of clm-serve) and `clm drift` (a warning only:
the CLM README quickstart against issue #15 with the unrounded differences, and the tides rank
against the model card). An unreachable endpoint, a missing key or a failed slow probe fails
under `--serve`, and under auto-detected serve mode without `--quick`; otherwise it is a
warning. An open route and a key the server rejects (401) always fail. Serve mode also reads
the encoder container's network namespace (`encoder network`: a listener on a non-loopback
address next to an interface besides `lo` fails) and warns when the headroom timer is stopped;
every doctor run checks the live `encoder_fp`'s `feature store` (its format and the vector recipe
it was built under). The records of the live runs are `bench/results/2026-10-01/doctor_serve.json`
(DESIGN A3/A4) and `doctor_serve_m1c.json` (A5).

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
(use an `ssh -L` forward otherwise). The keys `mesa-clm serve keys --init` writes,
`~/.mesa/clm/secrets/clm.key` and `encoder.key`, are read by default; point mesa-clm elsewhere
with the key-file settings (paths, never the keys):

```bash
# only when the keys live elsewhere; an explicit empty value switches the default off
export MESA_CLM_CLM__API_KEY_FILE=/path/to/clm.key
export MESA_CLM_ENCODER__API_KEY_FILE=/path/to/encoder.key
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
