# Changelog

All notable changes to the mesa-clm package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Documentation changes are logged separately in
`docs/log.md`; decisions in `DESIGN.md` (append-only, amendments A1…); verified facts in
`RESEARCH.md`.

## [Unreleased]

### Fixed

- Bench ranking metrics no longer depend on the order of tied confidences (DESIGN D33):
  `mesa_clm.bench.metrics` replaces the vendored `ece`, `coverage_at_risk` and `aurc` in every
  cell (tied items share their group's mean correctness; identical to the vendored values on
  tie-free inputs). The committed `bench/results/2026-09-29/baselines.json` was regenerated;
  only `ece`, `cov@5%`, `cov@10%` and `aurc` moved. Found by x86-64 CI disagreeing with aarch64.

### Added

- Milestone M0: repository scaffold and contracts. The hatchling `src/` package `mesa_clm`
  (typed, `py.typed`) with the console script `mesa-clm` and the `mesa_mcp.tools` entry point
  `clm` (`mesa_clm.mcp_tools`, an M0 stub that imports cleanly under mesa-mcp's `load_plugins`
  and registers nothing until the five `mesa_clm_*` tools land in M3, DESIGN D14); mesa-mcp
  `c74f3aa` and mesa-ducklake `7bc143f` pinned by git commit, `requests` declared for the OLS
  layer, a torch-free core (no `contrastive-lm`, torch or vllm anywhere in the dependency tree,
  DESIGN D0); `policy_defaults.yaml` shipped inside the wheel.
- Vendored, byte-identical and hash-pinned in `vendored.sha256`: CLM `schema.py` and its
  Apache-2.0 `LICENSE` (`bb42c6c5`) under `src/mesa_clm/_vendor/clm/`, AnyJev's
  `bench/metrics.py` as `bench/_metrics.py`, and CLM's `client.py` as the test-only wire oracle
  `tests/_vendor/clm_client.py` (DESIGN D4); mypy and ruff overrides for the vendored modules.
- Ports from mesa-anyjev (imports adapted, behaviour kept, DESIGN U1): configuration model
  (`MESA_CLM_<SECTION>__<FIELD>`, `*_FILE` secrets held as `SecretStr`, secret-free
  `config_sha256`, string fields taken verbatim from the environment so `OLS__FIXTURES=off`
  stays a word), dataset-card parser and JSON state builders (`state_sha256` byte-identical,
  additive `target_state` view that ends with the target), frozen vocabularies (`registry.py`
  with `ANCHORS`, `ANNOTATE_OPTIONS`, `NEON_ASPECT_MAP`), the 17 frozen tasks (`tasks.py`,
  `task_key` bit-exact to mesa-anyjev's `questions.lock.json`) and the D1 label identity
  (`identity.py`: `target_sha256`, `option_key`), the policy defaults reader (`policy_defaults.py`,
  the one source of `min_weight`, DESIGN D9), the deterministic OLS candidate layer with recorded
  fixtures and the AVU builder, the planner protocol with the Static, Claude and Gateway
  planners, the `mesa_clm.labels` table with its per-operation flock store
  (`provenance/labels.py`, DESIGN D11; the flock is `<dir>/locks/provenance.lock`, the D11
  path the M1 sidecar store shares, `lock_path_for`) and the label ingest. Defects fixed in
  the port: the `unit_candidate` lookup order (a) — exact key, then whole-word suffix,
  camelCase units normalised, a compound unit (`metersPerSecond`, `squareMeter`,
  `celsiusSquared`) never resolving to its trailing base unit but to its own UO term (meter per
  second UO:0000094, watt per square meter UO:0000155, milligram per kilogram UO:0000308,
  square meter UO:0000080, cubic meter UO:0000096) or missing so the pipeline searches, every
  `UNIT_TABLE` CURIE checked against a recorded `get_term` (kilometer is UO:0010066; UO:0000009
  is kilogram) — the literal `"the top profile value"` (c), an OLS outage propagating out of
  `search_candidates` (now logged), and a non-HTTP failure crashing the gateway planner instead
  of falling back to the static plan.
- Labels: `labels ingest-neon-eval [--eval-root DIR] [--exclude-models a,b]` writes the
  neon-avu-eval silver with the D1 identity `(task_key, target_sha256, option_key,
  label_source)`, `product_code`, `leak_group`, `fold_eligible` and `bench_card` (934 rows from
  the committed fixtures; the 303 valid (card, CURIE) pairs become 285 `term.fits` rows because
  five GAZ CURIEs come back from EBI OLS as root terms, 18 pairs listed in `terms_missing`; the
  450 column-linked AVUs collapse onto 301 `avu.value_kind` identities before the insert, the
  first winning as in mesa-anyjev, reported as `skipped.collapsed_identity` 149 and `conflicts`
  28, so `per_task` and `inserted` count what the store holds);
  `labels import-anyjev --dsn DSN` reads a mesa-anyjev sidecar read-only and re-identifies every
  row (float32 weights snapped back to the table weight); `labels snapshot [--out PATH]` writes
  the frozen Parquet snapshot the bench reads and prints its `labels_sha256` (DESIGN D30);
  `labels stats` prints counts per task, source, label and weight.
- Bench: `bench baselines` computes the no-model controls per task — `lookup_acc` and
  `lookup_prob` (Laplace α=1) over the key `(task, scope, target, option_key)`, ties between
  training cards going to the alphabetically first card, the novel-key subset and
  leave-one-product-out — with class counts at the policy `min_weight`; `bench mde` simulates
  the minimum detectable effect under rule R (200 simulated datasets per grid point, B=2000,
  seed 0). Both stamp the `labels_sha256` of `bench/snapshots/<date>.parquet` (written from the
  store when missing, refused when the store's `labels_content_sha256`, a second content-only
  hash carried by every results file, no longer matches). Committed:
  `bench/snapshots/2026-09-29.parquet` and `bench/results/2026-09-29/{baselines.json,
  baselines.md, mde.json}`, reproducing the planning numbers lookup LOCO 0.772 / 0.800, LOPO
  0.723 / 0.721 and novel-key subsets of 159 / 99 items; `tests/unit/test_baselines.py` and
  `test_mde.py` re-derive the committed cells from the fixtures, and
  `test_policy_min_weight_counts` pins the class counts at the policy weight.
- `mesa-clm doctor [--quick] [--json]` (`health.py`): the vendored files against
  `vendored.sha256` (skipped, and said so, when running from a wheel), the configuration's
  secret-free hash, Python, duckdb (the 1.5.x window of DESIGN D11), mesa-mcp and mesa-ducklake
  with their installed git commits against the D0 pins, mesa-mcp's plugin API (`load_plugins`,
  `register_tool(meta=)`, DESIGN D14) and the `clm` entry point, the policy defaults, the
  provenance path, the label store, key files (stat-ed for mode 0600, never read), the eval root
  and the OLS fixture directory; `ok`/`warn`/`fail` per check, exit 1 on any failure. The
  serving probes follow in M1.
- The `mesa-clm` CLI with the global `--config`, `--provenance` and `--actor` options (the last
  two also accepted after the verb), exit codes 0 ok / 1 failed / 2 configuration or usage, and
  the M0 verbs `labels ingest-neon-eval|import-anyjev|snapshot|stats`, `bench baselines|mde`
  and `doctor`. A configuration validation error names the field and the problem, never the
  value (`Config` sets `hide_input_in_errors`; a key on a mistyped `MESA_CLM_*__API_KEY` name
  cannot reach stderr), and an `--eval-root` without `results/validated.json` and `cards/` is a
  usage error (exit 2, the doctor's rule) instead of a traceback.
- Fixtures under `tests/fixtures/`: the 7 neon-avu-eval dataset cards and `validated.json`
  (CC BY 4.0 NEON metadata); 1,884 recorded EMBL-EBI OLS4 responses (777 from mesa-anyjev, 15
  for the unit table, 1,092 for the OLS fixture closure — every non-identifier column × every
  aspect-allowed ontology × the static planner's queries, the unit fallback, the site biome and
  dataset taxon queries and the children of every candidate — enumerated by `ols_closure.py`,
  recorded once by `scripts/record_ols_closure.py` and checked by `test_ols_fixture_closure`,
  0 missing); the mesa-anyjev parity data (`anyjev_parity.json`, 949 records, and
  `anyjev_questions.lock.json`) so tests never read the sibling checkout.
- House files: `CLAUDE.md`, `AGENTS.md`, `DESIGN.md` (U1–U4, D0–D32, the pre-registration for
  X1–X4, rule R, the committed MDE and K0–K4), `RESEARCH.md`, `THIRD_PARTY.md`, the OKF
  documentation bundle under `docs/` with `zensical.toml`, and the CI (`ci.yml` with the
  `vendored` job and `mesa-clm doctor --quick`), `serving.yml` skeleton, `docs.yml` and
  `release.yml` workflows.
