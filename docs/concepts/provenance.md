---
title: "Provenance"
description: "The mesa_clm sidecar: a per-host DuckDB opened under a flock, its tables and fingerprints, the direct/spool/none history backends and the lock set, source tags, export, prune and revert."
type: Reference
tags:
  - concepts
  - provenance
  - ducklake
  - history
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

## Export, prune, retry, revert

`provenance export --run-id R` writes a run's rows as Parquet to
`<project>/.mesa/clm/runs/<run_id>/` (read-back verified, never under `.mesa/ducklake/`) and
re-exports after reconcile; `prune` deletes local rows only for **terminal** runs (applied,
every link `written|spooled` with a snapshot id, no open elicitation) or abandoned ones past
`MESA_CLM_PROVENANCE__TTL_DAYS` (DESIGN D29). `history retry --run-id R` closes the crash window
between an iRODS write and the link update. `mesa-clm revert --run-id R` deletes exactly the
written triples with the stored unit, records `op=delete`, and marks links `reverted`
(DESIGN D31).
