---
title: "Testing"
description: "How to run the hermetic mesa-clm test suite, what the fake CLM transport, the recorded OLS fixtures and the offline doctor cover, the vendored-file and parity checks, the M1 tests, the M2 and M4 bench tests on synthetic labels only (and the plan-constants tests), the tests of amendment A1 after the registered run, and the opt-in live, engine, GPU, Postgres, end-to-end and neon tiers."
type: Guide
tags:
  - develop
  - testing
generated:
  by: "claude/fable-5.1"
  at: "2026-10-04T23:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/plan-2026-09-28.md"
    title: "mesa-clm implementation plan (v1.1, 2026-09-28)"
    author: "team:idss-mesa"
  - id: m2-plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/m2-analysis-plan.md"
    title: "mesa-clm M2 analysis plan (pre-run)"
    author: "team:idss-mesa"
  - id: m4-plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/m4-analysis-plan.md"
    title: "mesa-clm M4 analysis plan (pre-run)"
    author: "team:idss-mesa"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# Testing

```bash
uv run pytest -q                                   # hermetic: fake CLM transport, OLS fixtures, DuckDB under tmp_path
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
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
once with `scripts/record_ols_closure.py`), the 110 `get_term` responses of the teacher corpus's
in-registry CURIEs under `tests/fixtures/ols-teacher/` (recorded once with
`scripts/record_ols_teacher.py`, M4) and `results/validated.json` live under
`tests/fixtures/`. Opt-in tiers are excluded by default (`addopts` in `pyproject.toml`)
and selected with a marker and an environment variable:

| Marker | Variable | Needs |
|---|---|---|
| `live` | `MESA_CLM_LIVE=1` | the real EMBL-EBI OLS API; only for recording fixtures and the closure check |
| `engine` | `MESA_CLM_ENGINE=1` | the real encoder on :8090 and clm-serve on :8700 with keys from the configuration or the default `~/.mesa/clm/secrets/{clm,encoder}.key` and the local docker daemon (`tests/engine/test_doctor_live.py`: `doctor --serve` green and a zero-shot annotate of the non-bench SRER card `tests/fixtures/cards-srer/DP1.00004.001.BP_30min.md`, OLS replayed from `tests/fixtures/ols-srer/`, with the container check, the live fingerprint and `clm_calls` rows; `tests/engine/test_offline_parity.py`: offline scores from float32 vectors in a scratch feature store against `/v1/systemone` and `/v1/rank`, within plan §5.6's 1e-4) |
| `gpu`, `serve` | `MESA_CLM_GPU=1` | reserved: no test carries these markers yet. The serve side's tests (the patched `create_app`, the fallback app with a stub encoder, the vLLM bearer guard and route probe) live in `serving/tests/` and run CPU-only, as the serving CI runs them, with the patched clone's source on the path: `MESA_CLM_CLM_SRC=~/.mesa/clm/serve/CLM/src uvx --with fastapi --with httpx --with numpy --with requests --with uvicorn --with pytest-asyncio pytest -q -p no:cacheprovider serving/tests` |
| `requires_postgres` | `MESA_CLM_TEST_PG_DSN=postgresql://…` | a Postgres 16 (docker) for the sidecar dialect parity |
| `e2e` | `MESA_CLM_E2E=1` | the real `mesa-mcp --transport stdio` and the `e2e` extra |
| `neon` | `MESA_CLM_NEON_ROOT=<clm root>` | a neon-ducklake root for the adapter |

## The fake CLM (milestone M1)

`tests/fakes/clm_transport.py` is an `httpx.MockTransport` serving `/v1/systemone`, `/v1/rank`,
`/v1/models`, `/health`, `/v1/embeddings` (texts or token-id lists) and `/tokenize`, the encoder
side behind the same rule as the real bearer guard (DESIGN A4: every path but `/health` answers
401 without the key, unknown paths included) and with a settable `owned_by` and `root` so a test
can impersonate the in-process fallback. It scores with the vendored `build_pairs` and
`answer_from_logits` over hashed n-gram vectors and has a `collapse` knob to reproduce upstream
#15; OLS is replayed from the fixture closure, and a replay miss names the missing query. Its
request bodies are checked against the vendored CLM client's `question_to_dict` output, so the
wire shape stays the upstream one.

## The doctor offline

The doctor's host and serving checks go through `mesa_clm.health.ServeProbes` (the serving home,
the httpx transport, the command runner for `ss`, `systemctl`, git and docker, and the meminfo
file, the directories searched for encoder goldens). An autouse fixture in `tests/conftest.py`
replaces `health.default_probes` with `ServeProbes.offline()`: no serving home, every connection
refused, every command missing, no meminfo, no goldens. It also points `serving.DEFAULT_HOME` at
an empty per-test path, so neither the installed serving lock nor the default key files under
`~/.mesa/clm/secrets` reach a test (the `engine` tests keep the real home). The hermetic suite
therefore never probes a port, runs `systemctl` or reads a key, whatever host it runs on; `tests/unit/test_health_serving.py` injects its own probes (the fake CLM transport,
scripted `ss -ltne` and `systemctl` output, a meminfo file, a scratch run directory with a real
unix socket) to drive every serving check, and `tests/engine/test_doctor_live.py` passes the
real `ServeProbes()`. The same fixture points `net.PROC_NET` at an empty path, so the keyed
clients' port-owner check reads no listener of the host; `tests/unit/test_listener_owner.py`
drives it with fixture `/proc/net/tcp` and `tcp6` files (another account's socket on 8090 or
8700 is refused before the key is sent, the pre-flight exits 2, a refusal mid-run fails the
run, a retry asks again, the planner gateway and the probe scripts' raw clients ask too) and the
check of each new connection against real loopback listeners of the test's own account (the
kernel's answer for the server end before and after the accept, over IPv4, IPv6 and dual stack;
another account's answer substituted, a port taken after the `/proc` check gets no byte), and
`serving/tests/test_encoder_proxy.py` the encoder's endpoint proxy (the kernel's answer for the
owner of a real loopback client, another account's connection closed without reaching the
encoder, a symlinked or foreign socket refused, a swap after the check, the relay with a late
answer after a half-close and a one-way stream, the connection limit, the idle timeout, the
throttled refusal logs, the backlog kept from the socket unit). `tests/unit/test_health_serving.py`
also drives `serving units` with unit files under `tmp_path`, a scripted `systemctl --user show`
and a fake `/proc/<pid>` (the old units, a stale unsandboxed proxy, a pending `daemon-reload`).

## The M2 bench tests (pre-run)

The M2 analysis code was written, reviewed and tested **before** the first run on the snapshot's
labels (`design/m2-analysis-plan.md` §14), so its tests run on synthetic labels only: generated
cards and labels with planted signal, fake feature stores, a random head exported under the lock's
sha, and a stand-in registration (`mesa_clm.bench.registered.REGISTERED` monkeypatched with the
synthetic snapshot's hashes and counts). No test combines a silver label with a model output.
What they read from the committed snapshot is label-free: its identity and state columns (the M2
CLI world and the timing script's requests), its label-content digest against the published one
(`test_bench_registered`), and, in the manifest tests, rows whose every label-bearing column was
replaced first.

- `test_framing_x1`: weighted Platt; the 30/5 guards and the floor of 100 at its boundary; the
  shuffle's seeded derangements, its mean-AUROC rule R against brute force over all 200 draws and
  its null calibration (the drafts' comparator was biased); rules (1)–(5) one by one, rule (2)'s
  literal reading and the F7/F9 tie; nesting and K1 folds, a fold's choice blind to its held-out
  card; the registration (every run parameter is a deviation, any deviation writes every cell
  unregistered, registered labels must give the published counts, another framings lock or model
  fingerprint is a deviation, an item table that is not the configured grid is refused, a file
  without a configured task or arm is refused); the replay's checks (each edit of a verdict, its
  bootstrap settings, a cell's identity field, metrics, counts, novel-key block or diagnostics, an
  arm record's AUROC, shuffle verdict or NLL, `a1`, `fold_choices`, `skipped_folds`, a nested
  trace or the record's shape is refused) and the recompute from `x1_items.parquet` (an edited
  sensitivity or mean-context record, `p_loco`, item, label or shuffle draw is refused); the
  settings at every call site (each nested fold equals `decide` on an evidence built with the
  literal registered settings, the intervals and rule R the statistics' own at seed 0, the shuffle
  draws an independent rejection sampler and golden draws); the floor, the guards and the tokens
  per target inside a nested fold, and an inner fold that pools nothing; the mean-AUROC rule R's
  card eligibility and ties; the report-only controls' values; the learned candidate-only probe
  equal to scikit-learn fold by fold, blind to weights and held-out labels.
- `test_cells`: arm scores equal the offline scorer's; zero shot is `σ(s_c)`; the calibrated tier
  recovers a planted Platt or temperature; a held-out card never sees its own labels; each fold is
  scored by its own fold's arm, exactly; a guarded fold without an arm is accounted for in both
  tiers; `beats_lookup_novel` fails when either of its two conditions fails (its rule R at seed
  0); the label weights change the fit and the unweighted sensitivity ignores them; a tier run off
  the registration (labels, settings, a fingerprint, the framings lock, a subset of tiers) is
  unregistered; on cards interleaved in the bench's identity order every item's prediction, the
  NLL, the sensitivity block and the raw-score AUROC equal a computation keyed by identity (rank_fit,
  K = 2, K > 2); `n_saturated` and annotate's raw-score AUROC by value; X1's trace pointer kept in
  `fold_choices`, a colliding selector key refused.
- `test_x2` (the replica equals scikit-learn, unweighted, training rows only, also on
  interleaved cards; the AnyJev join; another or no dump, no replica or another fingerprint
  unregistered; probabilities outside [0, 1] refused, a card without a prediction skipped),
  `test_calibrate`, `test_features_choice` (the closed-choice manifest; a request whose keys are not
  the closed options refused), `test_x1_latency_script` (every drawn target's requests asked once
  per model in the alternating order, the `first` and `new_text` flags, the sha256 draw order, a
  request that does not render the manifest refused, no key written, every answer discarded),
  `test_bench_registered` (the registration against the published files and the committed locks,
  by digest and count).
- `test_bench_m2_cli`: `bench framing`, `--decide --from`, `bench run`, `bench x2` and
  `bench table` end to end on the synthetic world: the registered run and its replay, partial runs
  written unregistered, another or a missing snapshot refused (none written), the published counts
  and the scratch store checked, results never overwritten without `--force`, a tampered or
  foreign `x1.json` refused (an items file edited behind the replay too: `bench run` recomputes),
  `tiers.json`'s fold pointers resolving into `x1.json`, another AnyJev dump, a timing file about
  other labels or another serving lock and a store of another vector recipe refused, an
  unregistered run's own replay, X2's controls equal to what `bench baselines` publishes.
- `test_m2_plan_constants`: parses the plan's Appendix B and §1.3 and fails when a value is not
  the one the code applies or a name in the "where" column does not exist.

Before the commit the suite was also checked by mutation (in `/tmp` copies, never the working
tree): every property above was shown to fail on code with the corresponding bug (for example
rule (2)'s drafts' reading, a fixed arm in the nested cell, a calibrator fitted on the held-out
card, either half of `beats_lookup_novel` removed, a single shuffle draw, an unchecked deviation).
A second review round mutated the code again and found paths no test guarded (the realignment of
fold-pooled predictions, the seeds and settings inside the nesting, the inner-fold floor, the
timing run's request order, several refusals); each fix above came with a test that fails on the
reverted code or the reviewers' mutant.

## After the registered run (DESIGN A1)

The registered M2 run is committed (`bench/results/2026-10-03/`) and amendment A1 records its
outcome; these tests hold the code to it:

- `test_bench_registered` replays and recomputes the committed `x1.json` from its own files (no
  feature store, no label store) under the rotated framings lock, and checks that the registration
  keeps G1's lock, which is today's framings with every task's active framing F7.
- `test_framings`: the active framings are A1's (F9 for `column.ontology_fits`, F7 elsewhere) and
  the rotation moved no `question_key`.
- `test_k1_default` (the fake provider): under the shipped defaults every `term.fits` decision and
  proposal is `ols_rank` (no probabilities, proposed, never auto), no D24 refinement is asked and
  the groups that would have had one record it, the run is not `degraded`, Q3 is asked with F9
  (its `question_key` and context hash), an audit run (`decider.ols_rank_tasks: []` or
  `--ols-rank-tasks none`) asks CLM about `term.fits` again, and the CLI prints and writes the
  tasks `ols_rank` decided.
- `test_a6_closed_choice` (DESIGN A6, the fake provider): by default no CLM answer drives Q1, Q2
  or Q7 (two runs whose closed-choice answers a test double pulls in opposite directions decide
  the same records, groups, links and proposals, which the same skews change under
  `closed_choice: clm`; two runs whose Q3 answers it pulls apart keep the same aspects); the rule
  records and the audit-only records (`abstain`, `audit_only_a6`) carry their markers and never
  make a link or a label; Q1 annotates every non-identifier column whatever the planner says; the
  packaged aspect lookup table rebuilds byte for byte from the registered snapshot, its sha256 is
  pinned and checked, and for every leave-one-card-out fold the lookup built from it is M0's
  (it reads the snapshot's 60 `column.aspect` labels, which M0's lookup cell already read); the
  hint and the lookup's top two are not capped, with M0's tie rule and the annotated card held
  out; the fallback (DESIGN A8) gives an unseen numeric or unit-bearing column `measurement`
  then `unit` or the prior's next non-`taxon` aspect and a string column the lookup's prior
  without `taxon`, keeps two, never offers `taxon` or `unit` to a column without a unit, records
  `rule: "a8"` and its reason in `search_json.a6.fallback`, leaves a planner hint of `taxon`
  and every seen column unchanged, needs no CLM answer (`ols_rank`, CLM down) and asks only the aspects with an
  ontology the plan puts in play, the planner's ontology still appended; a sidecar file without
  the schema; `clm` mode is the M2 behaviour; all seven fixture cards annotate in both modes under
  the shipped decider with at most three Q3 questions per column; explain, the pending groups,
  the offered candidates and feedback never offer an audit record; the CLI's `--closed-choice`
  and `MESA_CLM_DECIDER__CLOSED_CHOICE`. `scripts/freeze_aspect_lookup.py --check` compares the
  packaged table with one rebuilt from the snapshot.

The pipeline helpers (`tests/fakes/pipeline.py` `config()`) are audit configurations
(`decider.ols_rank_tasks: []`), so `test_pipeline_fake`, `test_service` and the review tests keep
exercising CLM's `term.fits` path (specificity, the second opinion, runner-ups);
`shipped_config()` gives the shipped defaults. Both keep the shipped `decider.closed_choice`
(`rules`, DESIGN A6); `test_pipeline_fake` runs all seven cards under both modes, M2's assertions
on Q1, Q2 and Q7 in `clm` mode. The synthetic M2 registrations stand in the checkout's framings
lock.

## The M4 tests (pre-run)

The M4 code (`design/m4-analysis-plan.md`) is tested before any M4 verb runs on the snapshot's
labels, so **every M4 test is synthetic only**: generated labels with planted signal, fake
feature sources and feature-store worlds with planted vectors, a probe fitted on random
features under the fake stack's fingerprint, synthetic corpora and table cards, a stand-in
registration with a small grid, a results root under `tmp_path` and a snapshot file that exists
only to be hashed. No test combines a silver label with a model output, reads a committed
results file as data, or touches the live feature store or the sidecar. `tests/fakes/m4.py`
holds the shared fixtures (a synthetic probe artifact, planted items and `fold_choices`, a cell
with every frozen field, a results file, a snapshot file to hash, a `k2.json` shaped like the
producer's).

- `test_linear`: each fitter reaches scikit-learn's optimum for the same objective (binary and
  multinomial logreg, shrinkage LDA against the `lsqr` form and the direct full-space formula
  when d > n, ridge against its normal equations); the SVD reduction is exact; weights are sample
  weights (an integer weight equals repeated rows); the training moments and the std floor; a
  constant column gets a zero coefficient; a fit at the iteration cap is used as it is and
  flagged; refusals; JSON round trip and determinism; the grids and constants are the declared
  ones.
- `test_probe`: every spec matrix is its formula over the vectors `score_arm` reads and the
  builder's zero-shot logits equal the tier cells' scores; the nested selection recovers the
  planted spec in every fold, ties go to the first configuration, the criterion is the pooled
  OOF NLL; the weights change the fit; teacher rows never reach a test fold, are dropped for the
  held-out product (outer and inner) and change the fit; two runs are byte-identical; the 30/5
  guards, the 40 floor, the inner floors and the 100 OOF calibrator floor skip folds with the
  stated reasons; an unfittable configuration is recorded and never chosen;
  `ProbeArtifact.predict` equals the bench's held-out probabilities; the full-data artifact.
- `test_x3`, `test_k2`, `test_x4`: the four probe cells with every frozen field, the identity as
  the full-data choice and `fold_agreement_with_full`, the `@latest`/`@raw` restriction, `@full`
  fixed, a planted signal passing `beats_lookup_novel`, the registration (another grid, other
  labels, a task subset stamp every cell exploratory; a stand-in makes the run registered); K2's
  candidates and best tier, every condition with its numbers, the verdicts a, b and c, a closed
  choice c by construction, a tier that does not pool the full set, a missing AnyJev cell, the
  clm-raw clause, the files and the `bench table` rows; X4's three arms on identical folds, the
  on-arm fits changed by teacher rows that never reach a test fold, the silver-minus-Opus cells
  on the surviving subset, the decision record, the unpinned-input rule.
- `test_teacher`: every drop reason on a synthetic corpus, the resolution to every card of the
  product that carries the column, Yes-only rows for both tasks, the aspect-allowed rule, the
  row flags, a `mesa-clm` model refused, the corpus pin, the rows read back as `TeacherRows`,
  the survivor rule; one label-free closure check of `tests/fixtures/ols-teacher` runs only when
  the real corpus is present.
- `test_bench_m4_cli`: `bench x3|k2|x4|e2e`, `bench table` over the new files and `labels
  ingest-teacher` end to end on `test_bench_m2_cli`'s synthetic world plus a synthetic corpus:
  registered runs and refusals, `x3.json` required by K2, the pinned inputs refusing other bytes,
  `bench e2e` saying it is second-wave work, a teacher snapshot whose silver rows differ refused.
- `test_artifacts`: the layout and manifest, the cited cell's identity equal to the bench cell's,
  immutability and owner-only modes, the strict load's refusals (tampering, another
  fingerprint, another framings lock, a stale question key), `CURRENT.json` and the bundle it
  gives the provider, promotion's four rules, `learn fit`'s writer with an injected fitter, the
  export copy, publish and pull with a tampered pull refused.
- `test_probe_provider`: a promoted synthetic probe served over the fake stack (the fake encoder
  embeds, the fake head projects), honest records (`level='probe'`, the calibrator's kind,
  `feature_spec`, `artifact_version`, the artifact reference, `raw_probs` the zero-shot parity
  distribution, `s_c`, `probs` the mirror of the calibrated tier over `logit(p_fit)`), closed
  choices, an unpromoted probe `TierUnavailable`, K4 refusing another stack, the encoder down
  and a rejected key, the vector cache reading the store and never writing it, a pipeline run
  storing `feature_spec` rows.
- `test_policy_citations`: a qualifying synthetic cell lets a passing record `auto`, and every
  field of the citation test, negated once, refuses it with a reason naming the field; the audit
  blocker (`audit_required`), `Policy.load`'s default validator and the pipeline's binding of the
  live fingerprint.
- `test_audit`: sampling refuses a bench card, stratifies by outcome from a seeded draw spanning
  enough cards, the file holds ids only, a verdict mints curator labels outside every fold, the
  `audits` rows follow the pass rule (none for decisions that apply no artifact), and `audit
  sample|review|record` run end to end with the review at a terminal only, over fake-provider
  runs on synthetic non-bench copies of the fixture card.
- `test_m4_plan_constants`: Appendix B of the M4 plan against the code (the K2 constants, the
  artifact and audit formats, the grids and floors, the citation test's numbers, the serving
  constants, the registration's pins: the filled teacher corpus hash, the neon-ducklake commit
  and the two snapshots' file hashes, recomputed from the committed bytes without opening a
  label).

`test_health`, `test_policy`, `test_tiered_provider` and `test_cli` are adjusted for M4 (the
doctor's `artifacts` check among the expected lines, `feature_spec` on the probe and head
records the policy tests build and the audit blocker in their verdicts, the probe tier's
`TierUnavailable` message).

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
`test_provenance_ddl`, `test_sidecar_flock`, `test_service`, `test_feedback_agent_pick` (the
`via` rules and one answer per group: a curator's is final, an agent's is replaced by a
curator's) and `test_run_owner_checks`; `test_cli_m1` (every M1 verb through `main()`:
`framings --check` drift and `--update-lock`, `annotate` with the fake provider and with
`--provider clm` over the fake transport, the unreachable-clm-serve rule and the
`ols_rank_fallback` degradation, replay misses committed as `failed` runs, `explain` by prefix
and owner, `review` and `feedback` with `sys.stdin.isatty` monkeypatched both ways (curator
labels at a terminal, `agent_pick` and the stderr notice without one, DESIGN A2), pending-only
and once-per-call answers validated before anything is recorded, the Postgres actor rule,
`provenance migrate|export|import|prune --dry-run` and their error paths, `serve keys` never
printing a key, `serve units`, `serve lock --check` without touching docker); `test_live_provider`
(a lock with another schema or an installed lock unlike the checkout's refused, the keyed
pre-flight refusing a missing or rejected key, an unserved model and an encoder of another
route, root or window, a mid-run 401 failing the run, failed calls counted);
`test_health_serving` (framings drift, the schema contract, sidecar schema versions on DuckDB
and Postgres, the serving lock on a serving home, the host and GPU budget checks, permissions,
the default key files, URLs refused by the loopback rule and printed redacted, and serve mode
against the fake stack: loopback binds, the 401 matrix, health, model lists checked against the
lock, the golden call, encoder goldens bitwise, the long-input probe, systemone parity with an
exported head, the drift warning, missing and rejected keys, unreachable as fail or warn);
`test_perms` (owner-only modes under umask 002 for the sidecar, the label store, run exports,
the feature store and its npz, the serving keys' parents and `annotate --out`); `test_config`
(YAML errors that never echo the file, the default key files); `test_features` and `test_offline`
(the manifest rebuilt from the fixture cards in builder order, the store's refusals, the
TextCache format, offline scores equal to the fake's `/v1/systemone`);
`test_provenance_bulk_insert` (the JSON bulk insert writes exactly the rows the per-row path
wrote on real fake runs, hostile strings stay data, failed casts raise, non-finite floats fall
back to `executemany`, and seven fake runs commit within a loose time bound); the encoder's
`tokenizer_json` from the configuration in `test_clm_encoder`. From M2: the bench tests and the A1
tests above. From M3: `test_apply_anonymous_rejected`,
`test_apply_crash_retry`, `test_revert`, `test_history_spool_recorder` (two VMs, dedup,
quarantine), `test_mcp_conformance`. From
M4 (pre-run): the tests above (`test_linear`, `test_probe`, `test_x3`, `test_k2`, `test_x4`,
`test_teacher`, `test_bench_m4_cli`, `test_artifacts`, `test_probe_provider`,
`test_policy_citations`, `test_audit`, `test_m4_plan_constants`). Coverage must stay at or
above 80% (`fail_under`).
