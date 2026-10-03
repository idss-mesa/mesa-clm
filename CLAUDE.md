# CLAUDE.md — mesa-clm

mesa-clm proposes calibrated, ontology-grounded AVUs for the MESA stack with Contrastive LM
(CLM): rank-first `/v1/systemone` Choice questions with a fixed abstain anchor over the OBO/OLS
tools of mesa-mcp, every decision recorded in a sidecar next to the mesa-ducklake AVU history.
It is the standalone successor-sibling of mesa-anyjev: the generic layers are ported, the AnyJev
logprob readout is replaced by a torch-free HTTP client for a vLLM pooling encoder plus a
patched `clm-serve`. Python 3.11+, hatchling `src/` layout, package `mesa_clm`, console script
`mesa-clm`, MCP tools `mesa_clm_*`. House style follows `idss-mesa/mesa-anyjev` (and through
it `neon-mcp`). `DESIGN.md` records every decision (U1–U4, D0–D33, the M1 and M2 implementation notes with
M2's pre-run disclosure and post-run record, the pre-registration frozen at G1 = the M1 merge
commit, amendments A1–A5; A1 is X1's registered outcome) and `RESEARCH.md` the verified facts
(the M2 results with their JSON pointers included); read both before changing behaviour. The full
plan is `design/plan-2026-09-28.md`; how M2 reads the frozen rules (X1, the tier cells, X2, every
constant) is `design/m2-analysis-plan.md`, committed before any M2 result and changed after the
first run only by amendment.

**Current state (2026-10-03): M2 done, on `feat/m2-evidence` (draft PR #4).** The registered run
(plan §14, once; outputs committed as produced in `bench/results/2026-10-03/`, never edited)
decided, and DESIGN **A1** records: `column.ontology_fits` → **F9 on `clm-latest`**
(`framings.ACTIVE`, `framings.lock.json` rotated, lock_sha `7c93cc3e0ff6…`; no question_key
moved); `term.fits` → **K1** (no arm qualified): its proposals are `ols_rank` by default
(`decider.ols_rank_tasks: [term.fits]`), D24 is not asked for those groups (recorded), CLM
answers `term.fits` only in an audit run (`annotate --ols-rank-tasks none`), M4 goes probe-first;
the closed choices keep F7. The M2 registration keeps G1's framings lock (`b432d32a7536…`), so
`bench framing --decide --from bench/results/2026-10-03/x1.json` still replays, and a new
`bench framing` or `bench run` is written unregistered. Next: M3 (the live proposed-only loop).

## Commands

```bash
uv sync --all-extras                                # dev install; nothing here pulls torch or vllm
uv run pytest -q                                    # hermetic suite (fake CLM transport, OLS fixtures, DuckDB)
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run mypy --strict src
sha256sum -c vendored.sha256                        # vendored CLM schema.py/LICENSE, AnyJev metrics, CLM client oracle
uv run mesa-clm doctor [--quick] [--serve] [--json] # pins, locks, stores, host; --serve (or both units active): live probes
uv run mesa-clm framings --check                    # framing keys match framings.lock.json (--update-lock after a deliberate edit)
MESA_CLM_OLS__FIXTURES=replay MESA_CLM_OLS__FIXTURES_DIR=tests/fixtures/ols \
  uv run mesa-clm --provenance duckdb:////tmp/p.duckdb annotate --card tests/fixtures/cards/DP1.10003.001.brd_countdata.md --provider fake --out /tmp/run.json
#   term.fits groups: ols_rank by default (K1, DESIGN A1); --ols-rank-tasks none asks CLM too (an audit run)
uv run mesa-clm explain --run-id <id|prefix>        # owner = --actor ($USER); review --run-id (TTY) | review --pick G=KEY|none --decline G
uv run mesa-clm feedback --group-id G --action pick|reject|decline [--option-key KEY|none]  # TTY: via=cli curator; no TTY: via=tool agent_pick (A2)
uv run mesa-clm provenance migrate|export --run-id R --out DIR|import PATH|prune [--dry-run]
uv run mesa-clm serve keys --init|--rotate | serve units | serve lock --check   # never prints a key
uv run mesa-clm features build --snapshot bench/snapshots/2026-09-29.parquet | export-npz --out DIR | project | stats [--json]
uv run python scripts/doctor_record.py --out bench/results/<date>/doctor_serve.json   # doctor --serve as a results file
uv run python scripts/x1_latency.py --out bench/results/<date>/x1_latency.json   # X1's p50: label-free live timing, answers discarded
uv run mesa-clm bench framing --date <date> --latency bench/results/<date>/x1_latency.json --decide   # X1: x1.json, x1_items.parquet, x1.md
uv run mesa-clm bench framing --decide --from bench/results/<date>/x1.json   # replay + recompute X1 from the files alone
uv run mesa-clm bench run --tiers zero_shot,calibrated --loco --date <date>   # tier cells of all five tasks (tiers.json)
uv run mesa-clm bench x2 --date <date>                # X2: M0 controls, PR #13 replica, AnyJev L2 (x2.json)
uv run mesa-clm bench table --date <date> [--out bench/results/<date>/table.md]   # every row names file#cell
MESA_CLM_LIVE=1 uv run pytest -q -m live            # real EMBL-EBI OLS: fixture recording and closure only
MESA_CLM_ENGINE=1 uv run pytest -q -m engine        # real encoder :8090 + clm-serve :8700 (doctor --serve, non-bench annotate, offline vs /v1/systemone and /v1/rank)
MESA_CLM_CLM_SRC=~/.mesa/clm/serve/CLM/src uvx --with fastapi --with httpx --with numpy --with requests \
  --with uvicorn --with pytest-asyncio pytest -q -p no:cacheprovider serving/tests   # serve side, CPU only (as serving.yml)
uv run python scripts/x1_crosscheck.py --out bench/results/<date>/x1_crosscheck.json   # X1's 200-group store vs clm-serve check
MESA_CLM_TEST_PG_DSN=postgresql://... uv run pytest -q -m requires_postgres          # docker postgres:16
MESA_CLM_E2E=1 uv run pytest -q -m e2e              # spawns the real mesa-mcp over MCP stdio
MESA_CLM_NEON_ROOT=<clm root> uv run pytest -q -m neon                              # neon-ducklake adapter
```

Bench sequence (plan §9; verbs land milestone by milestone, `mesa-clm --help` lists what exists):
`labels ingest-neon-eval --eval-root tests/fixtures/neon-avu-eval` → `labels snapshot` →
`features build` → `bench mde` → `bench baselines` → `bench framing --decide` → `bench run
--loco` → `bench e2e --loco` → `bench table`. M0 ships `labels
ingest-neon-eval|import-anyjev|snapshot|stats`, `bench baselines` (lookup_prob, novel-key, LOPO)
and `bench mde`, both stamped with the `labels_sha256` of `bench/snapshots/<date>.parquet`
(written from the store when missing, refused when the store has changed since; DESIGN D30), and
`doctor`. M2 (pre-run) adds `bench framing|run|x2|table`: they read only the registered
snapshot (`bench/snapshots/2026-09-29.parquet`, refused by both hashes and the published counts,
never written), take B, the seed and alpha from the registration (no option for them), write any
other invocation with every cell `pre_registered: false, exploratory: true` (the producers also
hold the framings lock, each model's fingerprint, X2's AnyJev dump and the evaluated grid to the
registration), and replace no results file without `--force`; `bench run` replays and recomputes
its `x1.json` before using it. **M2 run protocol** (plan §14; once each, nothing run on real
labels before the plan and code are pushed, outputs committed as produced): `x1_latency.py` →
`bench framing --latency … --decide` → `bench framing --decide --from x1.json` → `bench run --tiers
zero_shot,calibrated --loco` → `bench x2` → `bench table --out …`. M1 adds `framings`, `annotate`
(`--provider clm|fake`, `--tier
auto|zero_shot|…|ols_rank`; clm-serve down at tier auto runs `ols_rank` only with
`decider.ols_rank_fallback: true`, otherwise exit 1), `explain`, `review`, `feedback`,
`provenance migrate|export|import|prune`, `serve keys|units|lock` and `features
build|export-npz|project|stats`. Global options `--config`, `--provenance`, `--actor` go before
the verb (`labels`, `bench`, `annotate`, `explain`, `review`, `feedback` and `provenance` also
take the last two after it; `doctor`, `framings`, `serve` and `features` do not); exit codes 0
ok, 1 the verb failed, 2 config or usage. A `duckdb:///` DSN with a relative path is relative to
the working directory; use four slashes for an absolute path (a bare `*.duckdb` path works
too). Keys default to `~/.mesa/clm/secrets/{clm,encoder}.key` when those exist; everything
mesa-clm writes under `~/.mesa/clm` is owner-only (0700/0600) whatever the umask.

## Fixed decisions (do not re-litigate; details in DESIGN.md)

- **U1–U4.** Standalone `mesa_clm`, nothing imported from `anyjev`/`mesa_anyjev`; tool prefix
  `mesa_clm_`, entry point `clm`, surface `clm`, env prefix `MESA_CLM_`, data `~/.mesa/clm/`,
  source tag `mesa-clm:<tool>:<outcome>`, sidecar schema `mesa_clm`. 0.1.0 is the plugin + CLI
  benched on the neon-avu-eval cards; 0.2.0 the neon-ducklake curate adapter. Serving is a
  vLLM pooling container plus a patched `clm-serve` on sparky-1 loopback with keys. Labels are
  silver consensus, curator picks and down-weighted teacher labels; teacher labels never reach
  a test fold.
- **Rank-first with an anchor (D2, D3).** One Choice per candidate group keyed by CURIE plus
  `__none__`; `s_c = ln p_c − ln p_anchor` is set-independent; the anchor winning is an abstain.
  No Score questions; noul only as control arm F1.
- **Task vs framing (D1).** `task_key` equals mesa-anyjev's lock key; labels are identified by
  `(task_key, target_sha256, option_key, label_source)`; the framing-specific `question_key`
  keys decisions, features and artifacts. A framing edit rotates `question_key` and must update
  `framings.lock.json`.
- **Honest levels (D6, D7).** `level ∈ {none, zero_shot, calibrated, probe, head}`,
  `calibration ∈ {none, uncalibrated, platt, temperature}`; `probs IS NULL iff
  calibration='none'`; `zero_shot` never auto-writes; `confidence = max(probs)` is computed
  locally, CLM's field is only `clm_confidence`. Never fabricate a distribution.
- **Fingerprints (D5).** `encoder_fp`, `clm_model_fp`, `schema_sha256` and `serving_lock_sha` on
  every decision, feature row, artifact and bench cell; exact match or refuse (K4).
- **Citations (D8, D9, D27, D30).** Every numeric `auto` cites a pre-registered,
  nested-selection LOCO cell that beats the lookup baseline on novel keys and sits above the
  Clopper–Pearson risk threshold; `policy_defaults.yaml` is the one source of `min_weight`; the
  bench reads only frozen label snapshots. Shipped defaults are `auto: null` everywhere.
- **Two phases (D10, D28).** `annotate` decides and proposes (policy `auto` stored as
  `accepted`, `accepted_by=policy`); `apply` writes. While uncalibrated, steps rank-and-cap;
  the degraded `ols_rank` method proposes and never autos.
- **X1's outcome (A1, D24, D28).** `column.ontology_fits` is asked with F9 on `clm-latest`;
  `term.fits` is K1: `ols_rank` decides its groups by default (`decider.ols_rank_tasks`), D24 is
  not asked for them (no `p_fit`; the group records it), its zero_shot/calibrated tiers are
  audit-only (`--ols-rank-tasks none`) until a probe is promoted, M4 goes probe-first. The closed
  choices are untouched by A1. The committed M2 results are never edited or re-run; the
  registration keeps G1's framings lock.
- **Sidecar and history (D11, D12, D13).** Per-host DuckDB opened per operation under a flock;
  the plugin always spools (`mesa-spool/1`); `direct` is CLI-only and needs the whole lock set
  free; one snapshot per (run, project); never an empty `record_changes`.
- **Labels (D19, D20, D21, D30).** The bench cards are the fixed seven neon-avu-eval tables
  (`learn.labels.BENCH_CARDS`, fail-closed), and the bench drops bench-card curator rows and
  non-fold-eligible rows before it picks a target's label. Curator labels only from MRTR elicitation or the CLI at an
  interactive terminal (`review`, `review --pick`, `feedback`: `via='cli'` only when stdin is a
  TTY, DESIGN A2); without one, and from a plain tool call, an answer is `agent_pick` at weight
  0; a curator's answer settles a group; runs carry an `owner`; weights are sample weights in
  every fitter.
- **Planner (D22).** The planner plans; Claude is a recorded second opinion; neither ever
  decides or writes.
- **Contexts (D23, D26).** Builders and `state_sha256` byte-identical to mesa-anyjev; every
  context view ends with the target; raw cards are never sent; `card_path` is CLI-only.
- **Serving (D15, D16, D17, A3, A4, A5).** Serving never fits; heads reach clm-serve only through
  promotion plus a restart; encoder :8090 and clm-serve :8700 on loopback with keys, the
  encoder answering nothing but `/health` without its key (A4); the vLLM encoder runs its
  batch-invariant kernels (`encoder_fp` c3b3d5e1a283, A3) in a container with no network, on a
  unix socket that a loopback socket unit proxies (`serving/encoder_proxy.py`: a socket of this
  account only, never through a symlink; the encoder requires the socket unit, so a taken port
  fails closed), with its KV cache pinned to 4.5 GiB (A5), and clients truncate at 4,095 tokens
  and never send a key to another account's loopback socket (`net.assert_listener_owner` before
  each request, `net.OwnerCheckedTransport` on each new connection before a byte is sent);
  `features build` and annotate check the running container against the lock first;
  remote URLs only with `MESA_CLM_CLM__ALLOW_REMOTE=1` **and** https. Never start, stop or
  enable the units as a side effect: that is the operator's.
- MIT, Copyright (c) 2026 The Regents of the University of New Mexico. "anyjev" appears only
  where mesa-anyjev is named as the source of a port.

## Code map (plan §3; **M0**/**M1** mark what exists after that milestone, the rest is planned)

`config.py`, `secrets.py` (settings, flag > env > YAML > default, `MESA_CLM_<SECTION>__<FIELD>`,
`*_FILE` secrets, `SecretStr` keys, `duckdb_path`) **M0** · `cards.py`, `states.py`, `registry.py`
(card grammar, JSON state builders, `target_state`, frozen vocabularies, `ANCHORS`,
`ANNOTATE_OPTIONS`, `NEON_ASPECT_MAP`) **M0** · `tasks.py`, `identity.py` (the 17 frozen tasks
with `task_key` bit-exact to mesa-anyjev's lock; the D1 label identity `target_sha256` /
`option_key`) **M0** · `policy_defaults.py` + `policy_defaults.yaml` (thresholds, profiles, the one
`min_weight`; shipped inside the wheel) **M0** · `ols.py`, `avu.py`, `ols_closure.py` (`OLSLayer`,
`RecordingOLS`, `UNIT_TABLE`, `build_avu`; defects (a) and (c) fixed; the fixture closure
enumeration behind `scripts/record_ols_closure.py`) **M0** · `planner/` (Static, Claude, Gateway
planners, `make_planner`) **M0** · `provenance/labels.py` (`LabelStore`, `LABELS_DDL`, per-operation
flock) **M0** · `learn/labels.py` (`ingest_neon_eval`, `import_anyjev`, `labelled_targets`,
`snapshot`, `stats`) **M0** · `bench/_metrics.py` (vendored AnyJev metrics) **M0** ·
`bench/{stats,results,baselines,mde}.py`, `bench/tasks/{base,neon}.py` (cluster bootstrap, rule R,
CP thresholds, typed cells, lookup_prob, novel-key, LOPO, MDE simulation, neon tasks reading the
policy `min_weight`) **M0** · `health.py` (doctor: vendored shas, config, versions and D0 pins,
mesa-mcp plugin API, policy, provenance path, key files) **M0**; framings lock, schema sha,
sidecar schema, feature store, permissions, serving lock, host/gpu_budget and the serve-mode
probes (encoder network namespace, headroom timer, 401 matrix, encoder goldens, long input,
systemone parity, drift) through injectable `ServeProbes` **M1** ·
`perms.py` (owner-only directories and files) **M1** · `mcp_tools/__init__.py` (entry-point
stub, registers nothing) **M0**, the five `mesa_clm_*` tools M3 · `cli.py` (`labels`, `bench`,
`doctor`) **M0**; `framings`, `annotate`, `explain`, `review`, `feedback`, `provenance`,
`serve`, `features` **M1**; the other verbs with their milestones · `_vendor/clm/` (CLM `schema.py` + `LICENSE`, byte-identical) **M0** ·
`framings.py`, `render.py` + `framings.lock.json` (framings, `question_key`, lock; typed facade
over the vendored `to_text`/`build_pairs`) **M1** · `clm/{http,encoder,headproj,fingerprint,fake}.py`
(`ClmHttpClient`, `EncoderClient`, numpy head projection, fingerprints, deterministic fake)
**M1** · `providers/{base,tiered,claude_provider,live}.py` (`DecisionRecord`, `_honest`, tier
routing, `DeciderRefused`, second opinion; `live.clm_provider` builds the real provider over the
serving lock the host runs and refuses a lock that contradicts the schema or the checkout,
`live.preflight` the keyed pre-flight, `live.encoder_problems` the encoder against the lock,
`live.container_check` the running container against the lock's recipe)
**M1** · `pipeline.py` (`Annotator.annotate` Q1–Q8) **M1** · `policy.py` (`outcome()`, `masked()`) **M1** ·
`provenance/{models,store,store_postgres,migrate,export}.py` (sidecar `mesa_clm`, migrations
including `LABELS_DDL`, export/import/prune; `store.bulk_insert` binds one JSON value per chunk
because DuckDB 1.5.6 probes `import pandas` per bound value) **M1** · `service.py`
(`DecisionService`: owners, pending groups, one answer per group) **M1** · `serving.py` (keys,
units incl. the encoder's loopback socket proxy, the lock and its container recipe, lock
verification, the container's network namespace) **M1** · `learn/features.py`
(`FeatureStore`: float32 vectors, the lock's vector recipe and `lock_sha` stamped, format 2; the
X1/X2 manifest, `builder_ordered`, TextCache npz export) and `learn/offline.py` (offline
rank_fit/noul scoring from the cached float32 vectors; `scripts/x1_crosscheck.py` checks it
against clm-serve) **M1** · `bench/framing.py` (X1: scores, the within-card shuffle, LOCO-Platt,
the decision rules, nesting, `decide_from_json`), `bench/cells.py` (the zero_shot and calibrated
cells, the one producer of citable `<task>.<tier>.<A1>` cells), `bench/x2.py`,
`bench/registered.py` (the registered snapshot, counts, B/seed/alpha, framings lock, model
fingerprints, AnyJev dump), `bench/run.py` (the M2
verbs), `learn/calibrate.py` (weighted Platt, temperature) **M2** (registered run
`bench/results/2026-10-03/`; A1: `framings.ACTIVE`, `decider.ols_rank_tasks`) · `apply.py`,
`revert.py`, `irods_io.py`, `history/` (two-phase apply, revert, backends, spool, recorder, lock
set) M3 · `learn/{linear,fit}.py`, `artifacts.py`,
`learn/teacher.py` M4 · `bench/e2e.py` M4 · `adapters/neon.py` M6 ·
`learn/finetune.py` M7. `serving/` and `deploy/` hold the serve-venv side (patches, encoder
scripts, units) and never import `mesa_clm`.

## Testing rules

- Unit tests never touch the network or a GPU: the fake CLM transport (`httpx.MockTransport`),
  `RecordingOLS` replaying `tests/fixtures/ols/`, DuckDB files under `tmp_path`. Tests never read
  the sibling mesa-anyjev checkout; parity data are committed fixtures. Markers `live`, `engine`,
  `gpu`, `serve`, `requires_postgres`, `e2e`, `neon` are opt-in through the environment
  variables above and excluded by default (`addopts` in `pyproject.toml`). An autouse fixture
  in `tests/conftest.py` makes the doctor offline (`health.ServeProbes.offline`: no serving
  home, refused connections, missing commands) and points `serving.DEFAULT_HOME` and
  `net.PROC_NET` at empty per-test paths (no installed lock, no default key files, no host
  listeners); tests that exercise serving checks inject their own `ServeProbes`, home or
  `/proc/net` files. The live checks are `tests/engine/`
  (`MESA_CLM_ENGINE=1`; keys from the configuration or the default key files).
- Vendored files (`vendored.sha256`) are byte-identical to upstream and never edited; CI checks
  the hashes.
- Every number in docs or tables names the results JSON it came from (AnyJev ground rule 1);
  numbers marked *to reproduce* in RESEARCH.md are not cited until the bench reproduces them.
- M2 pre-commitment: until the analysis plan and its code are pushed, nothing combines a model
  output with the snapshot's silver labels and no M2 verb runs on the registered snapshot; the M2
  tests run on synthetic labels only (from the snapshot they read its identity and state columns,
  rows whose label-bearing columns were replaced first, and its label digest).
  `tests/unit/test_m2_plan_constants.py` holds the plan's Appendix B to the code. Since the
  registered run (2026-10-03) one test also replays the committed `x1.json` from its files
  (`test_bench_registered.py`); the synthetic registrations stand in the checkout's framings lock.
  The pipeline test helpers (`tests/fakes/pipeline.py` `config()`) are audit configurations
  (`decider.ols_rank_tasks: []`, CLM on `term.fits`); `shipped_config()` and
  `tests/unit/test_k1_default.py` cover the shipped K1 default.
- No token, key or password is ever logged, printed, hashed into a fingerprint or committed;
  secrets resolve through `env|file|keyring|auto`.

## Git

Agents commit on `feat/`, `fix/`, `docs/`, `chore/` branches with Conventional Commits and open
PRs; never amend shared branches. `CHANGELOG.md` (Keep a Changelog) for the package, `docs/log.md`
for the docs, `DESIGN.md` amendments for decisions, `RESEARCH.md` for facts.

## Merging

Wait for a PR's checks with `scripts/wait_for_checks.sh <pr-number|branch>` before
`gh pr merge`. It refuses to settle until checks exist (at least three), none is pending, the
set is stable across two polls and none failed; an empty check list means CI has not started,
never that it passed. Do not replace it with an ad hoc `until ... grep pending` loop.
