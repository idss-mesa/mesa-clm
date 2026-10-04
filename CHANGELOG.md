# Changelog

All notable changes to the mesa-clm package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Documentation changes are logged separately in
`docs/log.md`; decisions in `DESIGN.md` (append-only, amendments A1…); verified facts in
`RESEARCH.md`.

## [Unreleased]

### Changed (DESIGN A6: the closed choices by rule, 2026-10-04)

- **The closed choices Q1, Q2 and Q7 are answered by deterministic rules by default**, CLM's
  answers to them recorded audit-only. Decided by the user after M2's results had been seen
  (`bench/results/2026-10-03/tiers.json`: CLM's zero-shot `column.annotate` 0.357 against majority
  0.643, `column.aspect` 0.050 against 0.300, `avu.value_kind` 0.324 against 0.478; no
  pre-registered rule covers these tasks) and recorded as amendment A6, a product-safety change of
  serving behaviour that makes no scientific claim and touches neither the frozen
  pre-registration nor any cell. New `decider.closed_choice: rules | clm` (default `rules`;
  `MESA_CLM_DECIDER__CLOSED_CHOICE`; `annotate --closed-choice`; `Annotator(closed_choice=)`,
  `DecisionService(closed_choice=)`); `clm` is the M2 behaviour, row for row (checked against
  `81e4495` on 63 fake runs). Under `rules`:
  - Q1: a column is annotated iff it is not an identifier, whatever the planner says (a `rule`
    record, `Yes`, reason `a6_not_identifier`): a planner's `annotate=False` stays in the run's
    `plan_json` and decides nothing, and an identifier stays out even when the planner says
    `annotate=True`;
  - Q2: the planner's aspect hint, then the top two aspects of the M0 lookup
    (`bench.baselines.Lookup` and `lookup_key`), de-duplicated, `other` excluded, up to three
    (`rule` records, reasons `a6_aspect_hint`, `a6_aspect_lookup`); else the fallback: the
    registry's aspects in the order of the lookup's training prior, the first two that have an
    ontology in play, `unit` only for a column with a unit (reason `a6_aspect_fallback`). No CLM
    answer chooses an aspect: Q3 is asked only for the chosen aspects, as in M2, so at most three
    times per column. The lookup counts a **table frozen from the registered snapshot**
    (`src/mesa_clm/aspect_lookup.json`: the 60 items of M0's `neon_aspect` task; written by
    `scripts/freeze_aspect_lookup.py`, sha256 pinned in `closed_choice.TABLE_SHA256`), the
    annotated card's items held out; no sidecar is read, so every host chooses the same
    aspects. A name no other card carries goes to the fallback. Each Q3 group records how its
    column's aspects were chosen in `search_json.a6` (sources, the lookup's counts and Laplace
    frequencies, the table's sha256, the fallback's order), and `runs.labels_sha256` names the
    snapshot the table came from;
  - Q7: the pre-rule, else "the term label" (a `rule` record, reason `a6_value_kind_label`);
  - CLM is asked Q1, Q2 and Q7 in the same requests as in M2 and every such record is stored with
    outcome `abstain` and reason `audit_only_a6` (no new vocabulary value, no migration); it
    never decides, never makes a link or a label and is never offered for review.
    `AnnotationRun.closed_choice` and `n_audit_only` (also in the `--out` JSON, both shapes) and
    a summary line say which mode ran and how many records are audit-only; the doctor's `config`
    line names `closed_choice`.
- More columns reach the OLS search (every non-identifier column), each with at most three
  aspects (two from the fallback); the OLS fixture closure already covered every such search. The
  annotate latency measured in M1 predates A6 and is to be re-measured under `rules` before M3's
  latency budget. Every run's `config_sha256` changes (the new field); no framing,
  `question_key`, `task_key`, label, fingerprint, lock or cell changes. A6 marks two readings for
  the user to confirm: a column name no other card carries goes to the fallback, and the
  fallback's order and its cap of two.

### Added (DESIGN A6)

- `closed_choice.py` (the A6 constants, `candidate_aspects`, `fallback_aspects`, the frozen
  `AspectTable` with `freeze_table`/`packaged_table`, `AspectLookup`, `is_audit_only`),
  `aspect_lookup.json` (shipped in the wheel; NEON data CC BY 4.0, `THIRD_PARTY.md`) and
  `scripts/freeze_aspect_lookup.py` (`--check` compares the packaged table with one rebuilt
  from the snapshot). Tests: `test_a6_closed_choice` (no CLM answer drives Q1, Q2 or Q7: runs
  whose closed-choice answers or whose Q3 answers are pulled in opposite directions decide the
  same things; the rule and audit records and their markers; Q1 whatever the planner says; the
  packaged table rebuilt from the snapshot byte for byte and, per fold, M0's own lookup; the hint
  and the lookup's top two uncapped, with M0's ties and the card held out; the fallback in the
  prior's order, capped at two, never `unit` without a unit, needing no CLM answer and bounded by
  the plan's ontologies; a sidecar file without the schema; `clm` mode as M2; all fixture cards in
  both modes; explain, review and feedback never offering an audit record; the CLI).
  `test_pipeline_fake` runs all seven cards in both modes (M2's closed-choice assertions in `clm`
  mode); `test_k1_default` and `test_cli_m1` no longer depend on how many term groups a run
  searches.

### Added (milestone M2: the registered run)

- `bench/results/2026-10-03/`: the registered M2 outputs, committed as produced by the protocol of
  `design/m2-analysis-plan.md` §14, run once (`x1_latency.json`, `x1.json`, `x1_items.parquet`,
  `x1.md`, `tiers.json`, `tiers.md`, `x2.json`, `x2.md`, `table.md`). X1's outcome: `term.fits`
  **K1** (no F4/F7/F9 arm qualifies on either model; every nested fold K1 too);
  `column.ontology_fits` **F9 on `clm-latest`** (7 of 7 nested folds agree). `RESEARCH.md`, "M2
  results (registered run)", gives every number with its file and JSON pointer (X1 per arm, the
  nested and audit cells, the closed choices, X2, the independent recomputation); DESIGN.md
  records the decision as amendment A1 and, in its M2 implementation notes, the outputs' sha256,
  the protocol times and the post-run disclosure.
- Tests: `test_k1_default` (the shipped defaults after A1: `term.fits` proposals are `ols_rank`, D24
  not asked and recorded, Q3 asked with F9, the audit run, the CLI), the replay of the committed
  registered run under the rotated framings lock (`test_bench_registered`), and the A1 rotation
  moving no `question_key` (`test_framings`).

### Changed (milestone M2: DESIGN A1, the X1 framing and K1)

- `column.ontology_fits` is asked with **F9** (X1's registered choice; `framings.ACTIVE`).
  `framings.lock.json` rotates: `column.ontology_fits` `active_framing` F7 → F9, `lock_sha`
  `b432d32a7536…` → `7c93cc3e0ff6…`; no `question_key`, `task_key`, label or fingerprint changes.
  `term.fits` and the closed choices keep F7. `framings.lock_payload(active=…)` and
  `lock_sha(active=…)` build the lock under another active map (G1's, every task at F7, is the
  lock M2's registration keeps, so the committed run still replays; a new X1 or tier run under
  the rotated lock is written unregistered).
- **K1 for `term.fits`**: the new `decider.ols_rank_tasks` (default `[term.fits]`;
  `MESA_CLM_DECIDER__OLS_RANK_TASKS`, a list, a comma-separated string or `none`; `annotate
  --ols-rank-tasks TASKS|none`) sends every `term.fits` group (Q4, Q5, Q6) to `ols_rank`, the OLS
  top-1 as `proposed`, never auto (D28). `none` or `[]` asks CLM for `term.fits` too, an audit
  run. D24's refinement needs `p_fit`, so it is not asked for an `ols_rank` group, which its group
  records (`search_json.specificity: {asked: false, reason: no_p_fit_ols_rank}`) as does the
  proposal's rationale. A run is `degraded` only when CLM did not answer or the tier is
  `ols_rank`, not for the tasks `ols_rank` decides by design. The annotate summary and the
  `--out` JSON (both shapes) name the `ols_rank_tasks`; an explicit learned tier is checked only
  for the tasks CLM answers. `Annotator` and `DecisionService` take `ols_rank_tasks=None` for the
  configured default. The closed choices are unchanged.

### Added (milestone M2, pre-run: the analysis plan and its code, before any result)

- `design/m2-analysis-plan.md`: the M2 analysis plan, every frozen rule M2 runs (X1, the
  zero_shot and calibrated tiers, X2, K1) read as one deterministic algorithm, each alternative
  reading named with the reason it was rejected, every constant in Appendix B, the run protocol
  (§14). Committed and pushed with its code before the first run on the snapshot's labels; until
  then nothing combined a model output with a silver label (DESIGN, "Implementation notes (M2)":
  the pre-run disclosure).
- `bench/framing.py` (X1): per-item scores offline from the feature store; the AUROC of `s` with
  its card-cluster CI; ΔAUROC(real − shuffle) against the mean AUROC of 200 seeded within-card
  derangements under rule R (`stats.rule_r_auroc_mean`, `stats.mean_auroc_ci`); weighted
  LOCO-Platt with the 30/5 guards and a floor of 100 training items per fit; the decision rules
  (1)–(5) as a pure function (rule (2) read literally: an F7 or F9 NLL winner stands, the token
  tie-break only when the cacheability step leaves both); the nesting on each fold's six training
  cards; the learned candidate-only probe (the PR #13 recipe) and the mean-context score
  (report-only); `x1.json`, `x1_items.parquet`, `x1.md`; `decide_from_json` replays every trace
  from the JSON (each rule R verdict against its own numbers and the run's bootstrap settings, the
  arm records and cells against their traces, each cell's counts, metrics and novel-key block
  against its own items, `a1`, `fold_choices`, the published counts) and recomputes the decisions,
  every arm record's statistics, the `p_loco` column, the cells' per-item predictions and the
  probe's statistics from the items file; what it cannot replay (the arm cells' lookup controls and
  `beats_lookup_novel`, the probe's fits) is named in the plan and in `x1.md`.
- `bench/cells.py`: the zero_shot and calibrated cells of all five tasks (`tiers.json`), the one
  producer of the citable-form `<task>.<tier>.<A1>` cells (nested: each fold scored by its own
  fold's X1 arm, its `fold_choices` entry keeping X1's `outcome` and `x1_trace` pointer),
  `@full` and audit-only K1 cells, `beats_lookup_novel` with both conditions and its reason,
  per-item predictions, the unweighted-calibrator sensitivity (report-only).
  `bench/x2.py`: the M0 controls, the PR #13 replica (`joint4096@S1`/`@S1ns`, unweighted, float32)
  and the AnyJev L2 cell joined by D1 identity. `learn/calibrate.py`: the code base's one weighted
  Platt (hard targets) and temperature (T in [1e-4, 1e4]) with every fitter constant recorded.
- `bench/registered.py`: the registered snapshot (both hashes), the published counts, the
  AnyJev dump's sha256, B = 2000, seed 0, alpha 0.05, the framings lock of G1 and each model's D5
  fingerprint under the serving lock of G1; any other run, through the verbs or the library
  (another framings lock or fingerprint, another or no dump, a subset of the grid, tasks or
  tiers), is written with every cell `pre_registered: false`, `exploratory: true`, and X1 refuses
  an item table that does not score exactly the configured arms.
- `mesa-clm bench framing [--decide [--from X1_JSON]] [--latency FILE]`, `bench run --tiers
  zero_shot,calibrated --loco`, `bench x2` and `bench table` (`bench/run.py`): the labels come from
  the registered snapshot only (refused by hash, label content and published counts; no snapshot
  is ever written), no option for B, the seed, alpha, the models or the framings, and no results
  file is replaced without `--force` (also for `bench baselines` and `bench mde`). `bench x2`
  notes whether its recomputed controls equal M0's cells; `bench run` replays and recomputes the
  `x1.json` it takes X1's outcome from and names it by sha256. `scripts/x1_latency.py`: the
  label-free live timing run for X1's p50 latency (answers discarded), each target asked under both
  models back to back with the first model alternating, every ask flagged `first` and `new_text`;
  `bench framing --latency` refuses a timing file about other labels or another serving lock.
  Every reported AUROC interval is labelled as what it is, the pair of one-sided 95% bounds (a 90%
  interval).
- The feature manifest covers the closed-choice tasks (F7 options), `FeatureStore.token_counts`,
  `offline.store_vectors`; the cell schema gains `model`, `variant` (`…@<variant>` keys are never
  citable), per-item `items`, `novel_key_lookup` and `beats_detail`.
- Tests (synthetic data only): `test_framing_x1`, `test_cells`, `test_x2`, `test_calibrate`,
  `test_bench_m2_cli`, `test_bench_registered`, `test_features_choice`, `test_x1_latency_script`
  and `test_m2_plan_constants` (Appendix B and §1.3 of the plan against the code).

### Added (milestone M1)

- Track B, the hermetic pipeline. `framings.py` and `framings.lock.json` (tasks vs framings,
  `question_key`, the F1/F4/F7/F9 framings with F7 active, the lock and its drift report;
  DESIGN D1) and `render.py`, the typed facade over the vendored CLM `schema.py` (D4). The
  torch-free CLM clients: `clm/http.py` (`ClmHttpClient`: `/v1/systemone`, `/v1/rank`,
  `/v1/models`, `/health`; retries, circuit breaker, loopback rule, keys redacted from every
  error), `clm/encoder.py` (`EncoderClient`: `/v1/embeddings` with `truncate_prompt_tokens =
  max_len - 1` and `truncation_side: left`, base64 float32, L2, the token guard over `/tokenize`,
  the `tokenize` extra or chars/2, token-id inputs; the key on every route), `clm/headproj.py`
  (CLM's head MLP in numpy over an exported `.npz`), `clm/fingerprint.py` (`encoder_fp` with
  `batch_invariance`, `clm_model_fp`, `Fingerprint`, the self-verifying `serving.lock.json`
  with its container `recipe`; D5, A3) and `clm/fake.py` with `tests/fakes/clm_transport.py` (a
  deterministic fake encoder behind the A4 bearer guard and a fake clm-serve, one
  `httpx.MockTransport`), wire parity against the vendored CLM client and committed goldens.
- Providers (D2, D6, D7, D22, D28): `DecisionRecord` with the `_honest` invariants mirrored in
  the sidecar CHECKs, `TieredProvider` (zero_shot and calibrated tiers, `s_c` recovery against
  the anchor, one request per shared context, the aspect mask, per-call `clm_calls`; a 401, 403
  or 404 refuses the run with `DeciderRefused`), `FakeProvider`, the degraded `ols_rank` records
  and `OlsRankProvider`, the Claude structured-output second opinion, and `providers/live.py`
  (the live provider over the serving lock the host runs, refusing a lock that contradicts the
  vendored schema or the checkout's lock; `preflight`: `/health`, then the keys, the served
  model and the encoder's model, window and route against the lock). `policy.py`: `outcome()`
  with the explicit rule and `ols_rank` branches, masks, the keep rule and specificity helpers.
- The `mesa_clm` sidecar (D11): row models, the DuckDB store opened per operation under the
  shared flock with a `RunBuffer` committed in one transaction, the optional Postgres store with
  its migration rendered from the DuckDB DDL, and Parquet export (stamped `exported_at` only
  after the copy verifies), import and terminal-only prune.
- `pipeline.py`, `Annotator.annotate` Q1–Q8 (rank-first with the anchor, rank-and-cap,
  specificity, value kinds, the keep rule, `ols_rank` per group, task or run; failed runs
  committed as `failed`; failed calls counted), and `service.py`, `DecisionService` (decider
  lock, owners, `run_summary` with `pending` groups, `candidates_for_group`, `explain`,
  `check_answer` and `record_human_pick` with `via`; one answer per group: a curator's is
  final, an agent's can be replaced by a curator's; D21, A2). Where M1 refined the plan is
  recorded in DESIGN.md "Implementation notes (M1)".
- The feature cache (plan §5.2; `learn/features.py`): `FeatureStore` per `encoder_fp`
  (`texts`, float16 `vectors` with their round-trip cosine, 512-d head projections with the
  head they came from, `meta`), the X1/X2 text `manifest` built label-free from a frozen
  snapshot in the builders' key order (`builder_ordered`; F1, F4, F7, F9 and the PR #13
  replica specs `joint4096@S1`/`@S1ns`), `export_npz` (CLM's `TextCache` format for
  `finetune.py --embed-cache`, refused below the 0.9999 fp16 gate). `learn/offline.py`:
  `OfflineScorer` (rank_fit and noul scoring from cached vectors, equal to the fake's
  `/v1/systemone` within 1e-9), `pairwise_s_c`, `sigmoid`.
- `mesa_clm/perms.py`: owner-only directories (0700) and files (0600) whatever the umask, for
  the sidecar and its lock, the label store, the feature store and its export, run exports,
  `annotate --out` and the serving keys' parents.
- Track A, the serving stack for sparky-1: CLM patches 0001–0006, `serving/serving.lock.json`
  (with the container recipe: arguments, non-secret environment, the bearer guard's sha256 and
  read-only mount, the 4,095-token cap), the serve-venv requirements,
  `serving/{export_head,cuda_encoder,fallback_serve}.py` with their tests, the encoder's bearer
  guard `serving/vllm_auth.py` (401 on every route but `/health`; DESIGN A4) and the route probe
  `serving/vllm_routes.py`, `deploy/bin/mesa-clm-{encoder-run,serve-bootstrap,wait-http,
  check-headroom}`, the systemd user units (encoder with `VLLM_BATCH_INVARIANT=1`, clm-serve with
  `--max-tokens 4095`, headroom service and timer), `serving.py` (keys, rendered units, lock
  verification against the host and the running container) and the `serving.yml` CI job.
- CLI verbs (plan §7.2): `framings --check|--update-lock`; `annotate --card PATH [--provider
  clm|fake] [--planner static|gateway|claude] [--tier auto|zero_shot|calibrated|probe|head|ols_rank]
  [--owner] [--out FILE|-] [--eval-result] [--fake-seed N]`; `explain --run-id|--irods-path
  [--limit]`; `review --run-id` (interactive, or `--pick GROUP=OPTION_KEY|none` and `--decline
  GROUP` on pending groups, once each, all-or-nothing); `feedback --group-id --action
  pick|reject|decline [--option-key KEY|none]`; `provenance migrate [--dsn]|export --run-id --out
  DIR|import PATH|prune [--ttl-days] [--dry-run]`; `serve keys --init|--rotate` (paths only;
  `--rotate` prints the `systemctl --user restart` to run), `serve units`, `serve lock --check
  [--require-live]`; `features build --snapshot PATH [--tasks] [--framings] [--batch]|export-npz
  --out DIR|project [--model]|stats [--json]`. Run ids, and group ids within your runs, accept
  unique prefixes.
- Doctor checks (plan §6.8): `framings lock`, `schema sha256` (against `vendored.sha256` and the
  serving pin), `sidecar schema`, `permissions` (group or other bits on mesa-clm's state, with
  the `chmod` that fixes them), the default key files named as such, `serving lock` (when
  `~/.mesa/clm` exists), `host`, `gpu_budget`, a `serving` reachability line, and in serve mode
  (`--serve`, or both units active) loopback binds, `/health`, the 401 matrix (every encoder
  route vLLM registers and an unknown path, clm-serve's three `/v1` routes), the model lists
  with the encoder checked against the lock, one golden `/v1/systemone` call and, under
  `--serve` or without `--quick`, the encoder goldens (bitwise one text per request, batch within
  1 − 1e-6), the long-input probe, the systemone parity against the local route (≤ 1e-4) and the
  warn-level upstream drift probes with unrounded differences. Every URL passes the loopback
  rule before it is probed and is printed redacted.
- M1-A records (bench/results; every number in the docs names one): `2026-09-29/serving_m1`,
  `collapse_spike`, `fallback_parity`; `2026-10-01/serving_m1b`, `batch_invariance` (the
  pre-registered batch-invariance experiment), `fallback_parity` (passes at batch 1),
  `collapse_spike` and `collapse_spike_sorted_keys` (the pipeline's own rendering, and the first
  run's), `features_build` (the feature store under c3b3d5e1a283) and `doctor_serve` (`doctor
  --serve` green); `bench/baselines/anyjev_l2_2026-09-29.{json,md}`, mesa-anyjev's per-item L2
  predictions, reproducing its committed cells exactly. Scripts: `serving_probes.py`,
  `collapse_spike.py`, `fallback_parity.py`, `batch_invariance.py`, `doctor_record.py`,
  `anyjev_l2_predictions.py`.
- `tests/engine/test_doctor_live.py` (marker `engine`): `doctor --serve` and a zero-shot
  annotate against the live stack, on the non-bench SRER card
  `tests/fixtures/cards-srer/DP1.00004.001.BP_30min.md` with its recorded OLS responses
  (`tests/fixtures/ols-srer/`); `tests/engine/test_offline_parity.py`: offline float32 scores
  against `/v1/systemone` and `/v1/rank` within plan §5.6's 1e-4.
- `scripts/x1_crosscheck.py` (X1's 200-group cross-check of the feature store against clm-serve,
  `/v1/rank` and a float16 reference route; `bench/results/2026-10-01/x1_crosscheck.json`) and
  `scripts/annotate_latency.py` (per-card annotate latency on non-bench cards, cold and warm;
  `annotate_latency.json`).
- The doctor checks the live `encoder_fp`'s feature store (format, vector recipe, the export's
  float16 gate), and in serve mode the encoder container's network namespace
  (`/proc/<pid>/net`: a non-loopback listener next to an interface besides `lo` fails) and the
  headroom timer (a warning when stopped). `serving.read_netns`, `netns_problems`,
  `container_pid` and `check_encoder_container`; `providers.live.container_check`.

### Changed

- `decider.tier` accepts `ols_rank` (the degraded D28 method for every rank_fit task);
  `decider.ols_rank_fallback` (default false) lets `annotate --provider clm` at tier `auto` run
  `ols_rank` when clm-serve does not answer instead of refusing; `encoder.tokenizer_json`
  (`MESA_CLM_ENCODER__TOKENIZER_JSON`) feeds `EncoderClient.from_config`.
- `clm.api_key_file` and `encoder.api_key_file` default to `~/.mesa/clm/secrets/clm.key` and
  `encoder.key` when those exist (`auto` and `file` modes; explicit settings win, an explicit
  empty value turns the default off; `config_sha256` unchanged). `serve keys --init` says so
  instead of printing exports for the default directory.
- The encoder recipe is batch-invariant (`VLLM_BATCH_INVARIANT=1`, DESIGN A3): `encoder_fp` of
  the vLLM route 852efc921a8a → c3b3d5e1a283, of the fallback (batch 1, `serial`) 82f4ff33b628
  → b171b1ba4536; `serving_lock_sha` `3c032477…` → `59625b8e…`. Clients truncate at 4,095 tokens
  (A4).
- `labels import-anyjev` imports mesa-anyjev's `curator`/`curator_implicit` rows as
  `agent_pick` at weight 0 unless `--trust-curator` (at a terminal); trusted curator rows on a
  bench card are tagged `bench_card`; the summary counts `demoted`.
- `provenance.export.prune` takes `dry_run`; its summary carries `dry_run`. The run JSON of
  `annotate` carries `n_failed_calls`.
- `config.duckdb_path` is the one DSN parser (`duckdb:///<path>` or a bare `<path>.duckdb`);
  `provenance.store.duckdb_file` delegates to it.
- CI runs `mesa-clm framings --check`.
- The feature store is format 2 (`mesa-clm-features/2`): vectors are stored as float32 (`vec32`)
  and cast to float16 only for the `TextCache` export; the store records the serving lock's vector
  recipe (`ServingLock.vector_recipe()`: image digest, encoder spec, `vllm serve` arguments,
  environment, cap) and stamps `lock_sha` on each vector, and `FeatureStore.for_lock` refuses a
  store of another vector recipe or format. A format-1 store is refused (rebuild it). Offline
  scores from float16 vectors missed X1's pre-registered ≤ 1e-4 cross-check (1.31e-3 / 3.59e-3);
  from float32 vectors it passes (5.2e-6 / 7.47e-5).
- The encoder recipe of DESIGN A5: the container runs with `--network none` and serves on the
  unix socket `~/.mesa/clm/run/encoder.sock` behind the new user units
  `mesa-clm-encoder-proxy.socket` (127.0.0.1:8090) and `mesa-clm-encoder-proxy.service`
  (`systemd-socket-proxyd`); the engine's rendezvous is on loopback; the KV cache is pinned to
  `--kv-cache-memory-bytes 4831838208` (19,609 MiB by `nvidia-smi`, 21.45 GiB of `MemAvailable`;
  was 25,153 MiB and 27.43 GiB). The encoder unit pulls in the socket and the headroom timer
  (`Wants=`/`PartOf=`) and starts at most three times an hour; the run script refuses an
  `encoder.env` without a usable key before docker runs. The lock's recipe carries `network`;
  `serving_lock_sha` `59625b8e…` → `dd33f9fe…`, `encoder_fp` unchanged (vectors bitwise equal).
- Bench-card membership is the fixed `learn.labels.BENCH_CARDS` (the seven neon-avu-eval tables),
  failing closed on an unknown card; `labelled_targets(fold_only=True)` (used by the neon bench
  tasks) drops non-fold-eligible rows and curator rows on bench cards before the per-target
  selection and counts them in `excluded`.
- X1's decision rule (6), the anchor variant, is withdrawn from the pre-registration before G1
  (DESIGN, "G1 freeze").
- `fallback_serve.py` refuses `--batch` other than 1 (its fingerprint says `serial`).
- `docs.yml`: read-only permissions by default, Pages permissions and a non-cancelling `pages`
  concurrency group on the deploy job only, the conformance check grouped per ref.
  `serving.yml`: `systemd-analyze verify` is a gate again (system mode, `%h` replaced by a stub
  home, the socket unit included) and the recipe check covers the network mode.

### Security

- Curator labels from the CLI need an interactive terminal (DESIGN A2): `review --pick/--decline`
  and `feedback` without one are recorded like a plain tool call (`via=tool`, `agent_pick`,
  weight 0, never fold-eligible) and say so on stderr; the interactive `review` requires one.
  On a shared Postgres sidecar `explain`, `review` and `feedback` act only as the OS account.
- The encoder answers nothing but `/health` without its key (DESIGN A4; M1-A found `/pooling`,
  `/invocations`, `/score`, `/rerank`, `/detokenize`, `/tokenize`, `/metrics` and `/version`
  open, `bench/results/2026-09-29/serving_m1.json`), its FastAPI docs are off and vLLM's usage
  report to an outside server is disabled.
- mesa-clm's state is owner-only whatever the umask (the sidecar and its lock were 0664/0775
  under umask 002).
- A YAML syntax error in `--config` names the file, line and column, never PyYAML's snippet of
  the line (which could carry an `api_key:`); any other unexpected configuration error prints
  only its type.
- The doctor refuses to probe a URL the clients would refuse (non-loopback without
  `allow_remote` and https, or with userinfo) and prints every URL redacted.
- The encoder container's engine sockets (the rendezvous store and six Gloo collective sockets,
  no authentication) were reachable on the docker-bridge address from every local account; the
  container now has no network (DESIGN A5, `bench/results/2026-10-01/encoder_netns.json`).
- `features build` and `annotate --provider clm` compare the running encoder container with the
  lock's recipe before trusting `encoder_fp` (a container without `VLLM_BATCH_INVARIANT` answers
  `/v1/models` identically); a departure is refused, an unreachable docker is reported as "not
  verified". Since the pre-merge review a container is also compared when docker answers but the
  pinned image is absent (one started from another image used to pass as "not verified").
- The encoder endpoint fails closed (DESIGN A5, revision before the merge): the encoder unit
  requires its socket unit and starts after it, so a port another local account holds stops the
  encoder and clm-serve from starting instead of letting clm-serve send the encoder key to that
  account's socket; every keyed client (`EncoderClient`, `ClmHttpClient`) refuses to send its key
  to a loopback port held by another account's socket (`net.assert_listener_owner`; the
  pre-flight exits 2, a refusal mid-run fails the run), also while the units are down; the
  doctor's `serving binds` reads `ss -ltne` and requires sockets of this account (and the
  encoder's port in the user manager's socket unit).
- The endpoint proxy `systemd-socket-proxyd` (which followed a symlink the container could plant
  in the run directory, and blocked at about 170 idle connections under the user manager's
  descriptor limit) is replaced by `serving/encoder_proxy.py` (`mesa-clm-encoder-proxy`): a socket
  of this account only, opened without following symlinks, at most 4,096 connections under
  `LimitNOFILE=16384`, idle ones closed after 900 s; the doctor's new `encoder socket` check fails
  anything else in `~/.mesa/clm/run`.
- The proxy relays only this account's connections (DESIGN A5, second revision before the
  merge): for each one it asks the kernel's socket diagnostics (`sock_diag`) who owns the
  client's socket and closes another account's at once, so no other local account can hold the
  4,096 connections or reach the encoder at all. Its unit runs it in a systemd sandbox without
  namespaces (`NoNewPrivileges=`, a system-call allow list, `AF_UNIX` and `AF_NETLINK` only;
  `systemd-analyze --user security` 9.8 → 5.9; the namespace options are left out because
  Ubuntu's AppArmor refuses a namespaced proxy's connect to the container's socket); it logs each
  kind of refusal at most once per 10 s and keeps the backlog its socket was given. The doctor's
  new `serving units` check fails units that are not the checkout's rendering and a running
  proxy without the rendered command line or its sandbox (`bench/results/2026-10-01/serving_m1e.md`,
  `doctor_serve_m1e.md`).
- Keyed clients check the connection itself, not only the port (DESIGN A5, third revision before
  the merge): `net.assert_listener_owner` reads `/proc` and the request connected in a second
  step, so another local account could bind the port in between and receive the key (a scratch
  run with a dummy key captured it on 16 of 400 attempts against a toggling listener). Every
  keyed client (`HttpEndpoint`, so `EncoderClient` and `ClmHttpClient`, the planner gateway
  client and the raw clients of `scripts/serving_probes.py`) now sends through
  `net.OwnerCheckedTransport`, which asks the kernel's socket diagnostics (`sock_diag`) which
  account holds the server end of each new loopback connection once it is made and before a
  byte is sent, and closes the connection unsent otherwise. DESIGN A5 also states the `docker`
  group as a residual risk: its members are root-equivalent on the serving host and can read
  the encoder key with `docker inspect`.

### Fixed

- `annotate --provider clm` with a missing or wrong key, or a model clm-serve does not serve,
  refuses before the run (exit 2) instead of passing the unguarded `/health` pre-flight and
  turning every group into an `ols_rank` proposal under an explicit `--tier zero_shot` (exit 0).
- Answering a settled group no longer leaves two accepted links or contradicting curator labels:
  a different second curator answer is refused, the same one is idempotent.
- `provenance export` stamps the run `exported_at` only after the copy is written and verified;
  a failed export (an unwritable or non-directory `--out`) leaves the store unchanged and exits
  1, so `prune` cannot delete the only copy of a run.
- `feedback --action pick` without `--option-key` no longer records a curator "none of these";
  `--option-key none` works as in `review --pick`.
- `annotate` expands `~` in `ols.fixtures_dir` like the other verbs.
- A curator answer on a bench card can no longer delete a pre-registered bench item (a tagged
  1.0 row used to win its identity and then be dropped: term.fits 285 → 284 items) or replace its
  silver label (an untagged row, written while the sidecar held no silver labels). Neither can a
  row of the reserved `gold` source: `labels import-anyjev` skips it (nothing produces it), and
  the fold filter drops every non-consensus row on a bench card.
- `feedback` answers a group the pipeline itself rejected (the keep rule's duplicate, a
  refinement that did not replace its parent) again; it was refused as "already has a curator
  answer" because a `rejected` outcome without an override row counted as a curator's.
- `annotate --eval-result --out` keeps the pre-flight notes (`preflight`, including whether the
  encoder container was verified), as the run shape does.
- `config.yaml.example` leaves the four key settings commented out: its explicit nulls switched
  the default key files off, so a configuration started from it resolved no key.
- The doctor's golden `/v1/systemone` question and the serving probes' latency and long-input
  requests use the non-bench SRER card instead of a labelled bench item (DESIGN, "G1 freeze").
- A consensus row in a mesa-anyjev file can no longer replace a bench item's silver label (a
  `consensus_all` row at 0.8 over a `consensus_negative` item entered without a terminal or
  `--trust-curator`): `labels import-anyjev` skips consensus rows on a bench card
  (`skipped["bench_silver"]`), and the fold filter counts a bench card's consensus rows as silver
  only when the neon-avu-eval ingestion wrote them (`learn.labels.NEON_EVAL_ORIGIN`).
- The planner gateway client and the raw clients of `scripts/serving_probes.py` ask who holds a
  loopback port before every keyed request, like `EncoderClient` and `ClmHttpClient`; the
  gateway's default port (the CARC tunnel's) is free whenever the tunnel is down.
- CLI error paths end in `mesa-clm: <message>` instead of a traceback: `provenance prune
  --ttl-days -1`, `provenance import` on an unsupported DSN, `provenance export --out` to a file,
  `serve lock --check --lock <directory>`, a blank `--actor`, end of input or Ctrl-C at the review
  prompt; a bare `*.duckdb` DSN is no longer created by the review verbs.
- Sidecar commits no longer pay DuckDB 1.5.6's failed `import pandas` per bound value (pandas
  is not a dependency): `provenance.store.bulk_insert` binds one JSON value per chunk and
  unpacks it with `from_json_strict`, falling back to `executemany` for a chunk JSON cannot
  carry; `set_link_status` updates a chunk of ids per statement. Rows are identical to the old
  path (compared table by table in `tests/unit/test_provenance_bulk_insert.py`); export staging
  uses the same path.
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
