---
title: "Provenance"
description: "The mesa_clm sidecar: a per-host DuckDB opened under a flock, its tables and fingerprints, the migrate, export, import and prune verbs, the direct/spool/none history backends and the lock set, source tags, and revert."
type: Reference
tags:
  - concepts
  - provenance
  - ducklake
  - history
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
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# Provenance

Decision provenance is a sidecar owned by mesa-clm: schema `mesa_clm` in a DuckDB file per host
at `~/.mesa/clm/provenance.duckdb`, **opened per operation under
`flock(~/.mesa/clm/locks/provenance.lock)`**, with a run's rows buffered and committed in one
transaction (DESIGN D11). Postgres is an optional dialect with the same CHECK constraints; a
parity test compares both DDLs. It is never a column on mesa-ducklake's `avu_changes` (the
13-column Parquet layout is a hard contract) and never a table in schema `mesa`.

Tables (`migrations/0001_mesa_clm.sql`, milestone M1): `runs` (owner, fingerprints, lock shas,
`labels_sha256`, history backend and waiver, counters, `exported_at`, `terminal_at`),
`decisions` (method, shape, level, calibration, probs, `raw_probs`, confidence,
`clm_confidence`, `s_c`, `p_fit`, `anchor_index`, artifact version, the two keys, the
fingerprints, context sha and token count, outcome and reason), `decision_options` (rank, `s_c`,
`p_fit`, `action_sha256`), `decision_groups` (the ranking over a candidate group), `avu_links`
(`write_status ∈ {proposed, accepted, dry_run, writing, written, spooled, reverted,
mirror_failed, local_only}`, `accepted_by ∈ {policy, human, agent}`, spool batch and snapshot
ids), `human_overrides` (`via ∈ {elicitation, cli, tool}` and what was offered), `labels`
(identity `(task_key, target_sha256, option_key, label_source)`, product code, leak group,
`fold_eligible`, `bench_card`), `audits`, `clm_calls`, `schema_versions`. DuckDB 1.5.5 rejects
`UNIQUE NULLS NOT DISTINCT`, so nullable key columns are `NOT NULL DEFAULT ''` sentinels.

A rank_fit question is one CLM Choice over a candidate group, so it is one `decisions` row
(the answered candidate's `s_c` and `p_fit` on the row) with one `decision_options` row per
candidate and one for the anchor, plus a `decision_groups` row with the OLS search log, the
top `p_fit`, the group margin and whether the anchor won. Every CLM request of a run is a
`clm_calls` row (latency, input tokens, status), and a run that fails is still committed with
status `failed` so the post-mortem has its rows. Writes buffer the whole run and commit it in
one transaction; the DuckDB store passes each table's rows as one JSON parameter per chunk
(DuckDB 1.5.6 pays a failed `pandas` import per bound value otherwise; see
[Architecture](../develop/architecture.md)).

Decisions are never updated; only a link's write status and snapshot id, and a run's status,
change. Corrections are new rows. Every AVU written by mesa-clm carries a mesa-ducklake `source`
of the form `mesa-clm:<tool>:<outcome>` (`mesa-clm:apply:auto|human|agent`,
`mesa-clm:revert:human`), so `get_history` shows what mesa-clm wrote without a join.

## History backends (milestone M3)

`MESA_CLM_HISTORY__BACKEND=direct|spool|none|auto` (DESIGN D12):

* **spool** (the plugin's only mode): a `mesa-spool/1` batch (Parquet with neon-ducklake's
  column set, a manifest with sha256, row count and sizes verified by read-back) under
  `<spool_root>/<vm>/`; links become `spooled`. A recorder (`mesa-clm history record`, a timer on
  the catalog host) folds batches into one mesa-ducklake snapshot per project, de-duplicates by
  `sha1(path, a, v, u, op, ts)`, moves batches to `spool-done/` or `spool-quarantine/`, and
  `provenance reconcile` then fills `snapshot_id`.
* **direct** (CLI only): takes the whole lock set through `fcntl.flock`, the same call
  neon-ducklake uses (`$MESA_HOME/locks/mesa-catalog.lock`, plus
  `<NEON_DUCKLAKE_ROOT>/work/locks/mesa-tag.lock` when configured), refuses if a `/proc` scan
  finds another holder (falls back to spool), records one snapshot per (run, project) with a
  note listing the paths, never an empty `record_changes` (DESIGN D13), and closes. mesa-mcp's
  own DuckLake singleton never closes, which is why the plugin never writes through it.
* **none**: only with `profile.allow_history_none` or `--allow-no-history`; the run records a
  `history_waiver_actor`.

Upstream, the plan proposes the lock and the spool recorder to mesa-ducklake and the
per-write client to mesa-mcp (plan §7.3); until then mesa-clm's recorder is interim with the
identical format.

## Schema, export, import, prune (milestone M1)

```bash
mesa-clm provenance migrate [--dsn DSN]              # create or upgrade the mesa_clm schema
mesa-clm provenance export --run-id R --out DIR      # DIR/<run_id>/<table>.parquet + manifest.json
mesa-clm provenance import DIR/<run_id>              # commit an exported run into this sidecar
mesa-clm provenance prune [--ttl-days N] [--dry-run] # delete terminal and long-abandoned runs' rows
```

`migrate` runs the DuckDB bootstrap (`CREATE … IF NOT EXISTS`) or applies the packaged Postgres
migrations (`pg` extra); `mesa-clm doctor` reports the schema version without creating
anything (`sidecar schema`). `export` stamps the run's `exported_at`, stages its rows through an
in-memory DuckDB carrying the sidecar DDL (so the Parquet columns are the schema's types and the
CHECKs re-validate every row), writes one file per table read back and hashed, and writes
`manifest.json` last. `import` verifies every file's sha256 and row count against the manifest
and never overwrites a run the store already holds. `prune` deletes local rows only for
**terminal** runs (applied, exported, every `written` or `spooled` link with a snapshot id, no
group escalated to a human) or abandoned ones past `MESA_CLM_PROVENANCE__TTL_DAYS` (DESIGN D29);
`--dry-run` lists what would go and deletes nothing. Labels and audits are never pruned
(DESIGN D30).

## Project export, retry, revert (milestone M3, planned)

Once `apply` writes, the export lands in `<project>/.mesa/clm/runs/<run_id>/` (never under
`.mesa/ducklake/`) and is re-exported after reconcile (`export --push`). `history retry
--run-id R` closes the crash window between an iRODS write and the link update. `mesa-clm revert
--run-id R` deletes exactly the written triples with the stored unit, records `op=delete`, and
marks links `reverted` (DESIGN D31).
