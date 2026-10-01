---
title: "Testing"
description: "How to run the hermetic mesa-clm test suite, what the fake CLM transport, the recorded OLS fixtures and the offline doctor cover, the vendored-file and parity checks, the M1 tests, and the opt-in live, engine, GPU, Postgres, end-to-end and neon tiers."
type: Guide
tags:
  - develop
  - testing
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T20:00:00Z"
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

# Testing

```bash
uv run pytest -q                                   # hermetic: fake CLM transport, OLS fixtures, DuckDB under tmp_path
uv run ruff check src tests scripts && uv run ruff format --check src tests
uv run mypy --strict src
sha256sum -c vendored.sha256                       # vendored files are byte-identical to upstream
uv run mesa-clm framings --check                   # framing keys match framings.lock.json (CI runs it)
uv run mesa-clm doctor --quick                     # pins, versions, plugin API, policy, stores (no hashing)
uv run mesa-clm annotate --card tests/fixtures/cards/DP1.10003.001.brd_countdata.md --provider fake --out /tmp/run.json
MESA_CLM_ENGINE=1 uv run pytest -q -m engine       # the live stack: doctor --serve and a zero-shot annotate
```

Unit tests never touch the network or a GPU, and they never read the sibling mesa-anyjev
checkout: parity data (`tests/fixtures/anyjev_parity.json`: states, `state_sha256`, task keys
and AVUs) are committed fixtures. The seven neon-avu-eval cards, 1,884 recorded EMBL-EBI OLS4
responses (777 from mesa-anyjev, 15 for the unit table, 1,092 for the fixture closure recorded
once with `scripts/record_ols_closure.py`) and `results/validated.json` live under
`tests/fixtures/`. Opt-in tiers are excluded by default (`addopts` in `pyproject.toml`)
and selected with a marker and an environment variable:

| Marker | Variable | Needs |
|---|---|---|
| `live` | `MESA_CLM_LIVE=1` | the real EMBL-EBI OLS API; only for recording fixtures and the closure check |
| `engine` | `MESA_CLM_ENGINE=1` | the real encoder on :8090 and clm-serve on :8700 with `MESA_CLM_CLM__API_KEY_FILE` and `MESA_CLM_ENCODER__API_KEY_FILE` set (`tests/engine/test_doctor_live.py`: `doctor --serve` green and a zero-shot annotate with the live fingerprint and `clm_calls` rows; later goldens, wire parity, `s_c` vs `/v1/rank`) |
| `gpu`, `serve` | `MESA_CLM_GPU=1`, run with the serve venv's Python | a CUDA device and the serve venv (`tests/serve`, `tests/gpu`) |
| `requires_postgres` | `MESA_CLM_TEST_PG_DSN=postgresql://…` | a Postgres 16 (docker) for the sidecar dialect parity |
| `e2e` | `MESA_CLM_E2E=1` | the real `mesa-mcp --transport stdio` and the `e2e` extra |
| `neon` | `MESA_CLM_NEON_ROOT=<clm root>` | a neon-ducklake root for the adapter |

## The fake CLM (milestone M1)

`tests/fakes/clm_transport.py` is an `httpx.MockTransport` serving `/v1/systemone`, `/v1/rank`,
`/v1/models`, `/health`, `/v1/embeddings` and `/tokenize`. It scores with the vendored
`build_pairs` and `answer_from_logits` over hashed n-gram vectors and has a `collapse` knob to
reproduce upstream #15; OLS is replayed from the fixture closure, and a replay miss names the
missing query. Its request bodies are checked against the vendored CLM client's
`question_to_dict` output, so the wire shape stays the upstream one.

## The doctor offline

The doctor's host and serving checks go through `mesa_clm.health.ServeProbes` (the serving home,
the httpx transport, the command runner for `ss`, `systemctl`, git and docker, and the meminfo
file). An autouse fixture in `tests/conftest.py` replaces `health.default_probes` with
`ServeProbes.offline()`: no serving home, every connection refused, every command missing, no
meminfo. The hermetic suite therefore never probes a port or runs `systemctl`, whatever host it
runs on; `tests/unit/test_health_serving.py` injects its own probes (the fake CLM transport,
scripted `ss -ltn` and `systemctl` output, a meminfo file) to drive every serving check, and
`tests/engine/test_doctor_live.py` passes the real `ServeProbes()`.

## Key tests

Vendored hashes; `test_anyjev_parity` (state shas, task keys and AVUs equal mesa-anyjev's);
`test_policy_min_weight_counts` (class counts at the policy `min_weight` equal the published
cells); `test_unit_lookup` (defect (a), including kilometer's UO id against a recorded
`get_term`); `test_ols_fixture_closure` (0 missing queries); `test_cli` (the M0 verbs end to end
on the fixtures: ingest, stats, snapshot, baselines and MDE stamped with the snapshot's
`labels_sha256`, exit codes); `test_health` (every doctor check, each failure path forced, no
secret in a report line); `test_mcp_tools` (the `clm` entry point loads under `load_plugins`
and registers nothing). From M1: `test_framings` (framing keys and the lock),
`test_honest` (at least 12 rejected records), `test_context_ends_with_target`,
`test_delta_recovery` (`s_c` set-invariant), `test_pipeline_fake` (7 cards end to end, including
`decider.tier: ols_rank` from the configuration), `test_ols_rank_proposes_never_auto`,
`test_provenance_ddl`, `test_sidecar_flock`, `test_service`, `test_feedback_agent_pick` and
`test_run_owner_checks`; `test_cli_m1` (every M1 verb through `main()`: `framings --check`
drift and `--update-lock`, `annotate` with the fake provider and with `--provider clm` over the
fake transport, the unreachable-clm-serve rule and the `ols_rank_fallback` degradation, replay
misses committed as `failed` runs, `explain` by prefix and owner, `review` interactive with
scripted input and with `--pick`/`--decline` validated before anything is recorded, `feedback`,
`provenance migrate|export|import|prune --dry-run`, `serve keys` never printing a key, `serve
units`, `serve lock --check`); `test_health_serving` (framings drift, the schema contract,
sidecar schema versions on DuckDB and Postgres, the serving lock on a serving home, the host and
GPU budget checks, and serve mode against the fake stack: loopback binds, 401s, health, model
lists, the golden call, missing and rejected keys, unreachable as fail or warn);
`test_provenance_bulk_insert` (the JSON bulk insert writes exactly the rows the per-row path
wrote on real fake runs, hostile strings stay data, failed casts raise, non-finite floats fall
back to `executemany`, and seven fake runs commit within a loose time bound); the encoder's
`tokenizer_json` from the configuration in `test_clm_encoder`. From M3: `test_apply_anonymous_rejected`,
`test_apply_crash_retry`, `test_revert`, `test_history_spool_recorder` (two VMs, dedup,
quarantine), `test_mcp_conformance`. From
M4: `test_policy_citations`, `test_loco_leakage`, `test_nested_selection`, `test_rule_r`,
`test_lookup_prob`, `test_cp_threshold`, `test_teacher_weight_changes_fit`,
`test_artifact_identity`. Coverage must stay at or above 80% (`fail_under`).
