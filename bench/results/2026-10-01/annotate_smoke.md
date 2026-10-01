# Annotate smoke through the live stack, sparky-1, 2026-10-01

**Disclosure (DESIGN, "G1 freeze").** Run 1 below ran the live zero-shot stack on the bench card
DP1.10003.001.brd_countdata before G1, and the observations compare its outputs with that card's
silver labels and with the anchor. DESIGN lists it among the pre-G1 looks at labelled bench data;
X1's anchor variant (rule 6) was withdrawn rather than chosen after it, and live smokes use
non-bench cards until the M2 cells exist (`annotate_latency.md`, `tests/engine/`).

Source: `annotate_smoke.json` (this directory; `<repo>` is the checkout's absolute path, `~` the
home directory). Diagnostics, not evidence: tier `zero_shot` is uncalibrated and never `auto`
(D6), and none of these runs is a bench cell. The CLM runs went through the live units on the
final recipe of DESIGN A3/A4 (`encoder_fp` c3b3d5e1a283, `clm_model_fp` 78be8c462b2e, `lock_sha`
`59625b8e…`), each with its own sidecar under `.local/smoke/` (the default sidecar was never
touched), from 16:57Z to 16:58Z; neither unit was restarted, and both stayed active and not
enabled at boot.

    # 1. bench card, live stack, OLS replayed from the committed fixtures
    MESA_CLM_OLS__FIXTURES=replay MESA_CLM_OLS__FIXTURES_DIR=tests/fixtures/ols \
      uv run mesa-clm --provenance duckdb:///<repo>/.local/smoke/prov.duckdb annotate \
      --card tests/fixtures/cards/DP1.10003.001.brd_countdata.md --provider clm --tier zero_shot \
      --out <repo>/.local/smoke/run-clm.json
    # 2. the same with the deterministic fake, separate sidecar .local/smoke/prov-fake.duckdb
    # 3. a non-bench SRER card, live EMBL-EBI OLS4 recorded to .local/smoke/ols-live
    MESA_CLM_OLS__FIXTURES=record MESA_CLM_OLS__FIXTURES_DIR=<repo>/.local/smoke/ols-live \
      uv run mesa-clm --provenance duckdb:///<repo>/.local/smoke/prov.duckdb annotate \
      --card ~/neon-ducklake/sites/SRER/cards/anyjev/DP1.00004.001.BP_30min.md --provider clm \
      --tier zero_shot --out <repo>/.local/smoke/run-srer.json
    # then, per run
    uv run mesa-clm --provenance <its sidecar> explain --run-id <run_id>

| | clm, bench card | fake, bench card | clm, SRER card (live OLS) |
|---|---|---|---|
| Card | DP1.10003.001.brd_countdata | DP1.10003.001.brd_countdata | DP1.00004.001.BP_30min |
| Run | `277c09cb…` | `9f8d4f3b…` | `d0ce4560…` |
| Exit, status | 0, decided | 0, decided | 0, decided |
| Decisions | 42 | 42 | 39 |
| Proposals | 4 | 3 | 1 |
| Abstained (by reason) | 0 | 1 (`anchor_won` 1) | 0 |
| Outcomes | proposed 18, rejected 14, rule 10 | proposed 17, rejected 14, rule 10, abstain 1 | proposed 17, rejected 16, rule 6 |
| `auto` | **0** | **0** | **0** |
| CLM calls (failed) | 18 (0) | 18 (0) | 17 (0) |
| Encoder input tokens | 0 (all cached, see below) | 2,953 (the fake's count) | 7,144 |
| Seconds (run; CLI wall clock) | 0.08; 0.60 | 0.03; 0.50 | 3.36; 3.84 |
| Degraded | no | no | no |
| Fingerprint | c3b3d5e1a283 / 78be8c462b2e / lock `59625b8e…` | fake: f2521831a25e / c715fc0bb42f / lock `f7d62d55…` | c3b3d5e1a283 / 78be8c462b2e / lock `59625b8e…` |
| `explain` | exit 0; 42 decisions, 4 groups, 4 links, 3 pending | exit 0; 42 decisions, 4 groups, 3 links, 3 pending | exit 0; 39 decisions, 1 group, 1 link, 1 pending |

Every fingerprint also carries `schema_sha256` `52cec58a…` and `framings_lock_sha` `b432d32a…`;
each `explain` agrees with its run's JSON on calls, tokens, fingerprint and decision count.

Example proposals (attribute / value / unit / `p_fit`):

| Run | Attribute | Value | Unit | p_fit |
|---|---|---|---|---|
| clm, bench card | `envo.temperate_mixed_forest_biome` | HARV | ENVO:01000212 | 0.99994 |
| clm, bench card | `envo.tropical_desert_biome` | SRER | ENVO:01000183 | 0.99880 |
| clm, bench card | `ncbitaxon.environmental_samples_birdsclass_aves` | environmental samples <birds,class Aves> | NCBITaxon:1126251 | 0.99165 |
| fake, bench card | `ncbitaxon.parastagonospora_avenae_f_sp_tritici` | Parastagonospora avenae f. sp. tritici | NCBITaxon:54790 | 0.54314 |
| fake, bench card | `envo.subtropical_desert_biome` | SRER | ENVO:01000184 | 0.54285 |
| fake, bench card | `ncbitaxon.avena_strigosa_x_avena_wiestii` | Avena strigosa x Avena wiestii | NCBITaxon:1080143 | 0.53766 |
| clm, SRER card | `envo.desert_biome` | SRER | ENVO:01000179 | 0.99975 |

The fourth proposal of the bench-card CLM run is `ncbitaxon.avena_atlantica_x_avena_hirtula`
(NCBITaxon:1080144, `p_fit` 0.5557), the dataset's second taxon.

## Observations

- **No auto, no degradation, no failed call.** All three runs propose at `zero_shot` only, as D6
  requires; OLS replay missed no fixture (a miss fails the run).
- **Zero-shot `column.annotate` said No to every column.** On the bench card all 14 columns not
  ruled out as identifiers got "No" (confidence 0.82 to 0.99992), on the SRER card all 16
  (0.9986 to 0.99999), so no column reached the aspect, ontology or term steps and every proposal
  is a site biome or a dataset taxon. The bench card's silver labels mark 8 of its labelled
  columns Yes and 4 No (`bench/results/2026-09-29/mde.json`, `neon_annotate` per card): at
  zero shot the live stack proposes no column AVU at all. Report-only; M2 measures it.
- **Zero-shot term choices are near-ties or wrong in places.** SRER's biome group on the bench card
  ranks tropical desert biome 0.99880 over 0.99853 and 0.99834; the D24 refinement replaced Aves
  (NCBITaxon:8782, 0.8672) by its child "environmental samples <birds,class Aves>" (0.9916), an
  NCBI bookkeeping node; the second taxon is an oat hybrid (`Avena`, 0.5557) just above the
  anchor's 0.5. All are `proposed`, for a curator.
- **Live OLS was light.** The SRER card made three sequential EMBL-EBI OLS4 calls (ENVO biome
  descendants for "xeric shrubland biome", "shrubland biome" and "desert biome": 1, 9 and 6 hits)
  inside a 3.36 s run, since no column reached a term search and the card has no taxon.
- **Encoder tokens depend on clm-serve's cache.** `input_tokens` counts encoder tokens on cache
  misses only. The bench-card CLM run found every text already cached (0 tokens, median call
  0.55 ms; consistent with `tests/engine/test_doctor_live.py`, which annotates the same card at
  zero shot and ran against the live stack earlier today): it is not a cold-run cost, and its 17
  distinct contexts are 6,425 tokens by the token guard. The SRER run's 7,144 tokens are exactly
  its 17 distinct contexts (median call 131.0 ms), so its candidate, option and anchor texts were
  cached too and 7,144 is a lower bound.

`doctor --serve` after the runs (17:00:41Z): exit 0, 30 ok, 2 warn (the `~/.mesa/clm/locks`
permissions and the quickstart drift of DESIGN A3), 0 fail.
