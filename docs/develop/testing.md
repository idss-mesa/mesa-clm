---
title: "Testing"
description: "How to run the hermetic mesa-clm test suite, what the fake CLM transport and the recorded OLS fixtures cover, the vendored-file and parity checks, and the opt-in live, engine, GPU, Postgres, end-to-end and neon tiers."
type: Guide
tags:
  - develop
  - testing
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

# Testing

```bash
uv run pytest -q                                   # hermetic: fake CLM transport, OLS fixtures, DuckDB under tmp_path
uv run ruff check src tests scripts && uv run ruff format --check src tests
uv run mypy --strict src
sha256sum -c vendored.sha256                       # vendored files are byte-identical to upstream
uv run mesa-clm doctor --quick                     # pins, versions, plugin API, policy, stores (no hashing)
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
| `engine` | `MESA_CLM_ENGINE=1` | the real encoder on :8090 and clm-serve on :8700 (goldens, wire parity, `s_c` vs `/v1/rank`) |
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

## Key tests

Vendored hashes; `test_anyjev_parity` (state shas, task keys and AVUs equal mesa-anyjev's);
`test_policy_min_weight_counts` (class counts at the policy `min_weight` equal the published
cells); `test_unit_lookup` (defect (a), including kilometer's UO id against a recorded
`get_term`); `test_ols_fixture_closure` (0 missing queries); `test_cli` (the M0 verbs end to end
on the fixtures: ingest, stats, snapshot, baselines and MDE stamped with the snapshot's
`labels_sha256`, exit codes); `test_health` (every doctor check, each failure path forced, no
secret in a report line); `test_mcp_tools` (the `clm` entry point loads under `load_plugins`
and registers nothing). From M1: `test_framing_keys`,
`test_honest` (at least 12 rejected records), `test_context_ends_with_target`,
`test_pipeline_fake` (7 cards end to end), `test_ols_rank_proposes_never_auto`,
`test_provenance_ddl` and `test_sidecar_flock`. From M3: `test_apply_anonymous_rejected`,
`test_apply_crash_retry`, `test_revert`, `test_history_spool_recorder` (two VMs, dedup,
quarantine), `test_feedback_agent_pick`, `test_run_owner_checks`, `test_mcp_conformance`. From
M4: `test_policy_citations`, `test_loco_leakage`, `test_nested_selection`, `test_rule_r`,
`test_lookup_prob`, `test_cp_threshold`, `test_teacher_weight_changes_fit`,
`test_artifact_identity`. Coverage must stay at or above 80% (`fail_under`).
