# mesa-clm

Calibrated, ontology-grounded metadata annotation for the MESA stack with Contrastive LM.

**mesa-clm** proposes OBO-grounded AVUs for a dataset the way its sibling
[mesa-anyjev](https://github.com/idss-mesa/mesa-anyjev) does, but asks
[Contrastive LM](https://github.com/Contrastive-LM/CLM) (CLM) instead of AnyJev: every fit
question is one rank over the candidate terms plus a fixed "none of these" anchor, scored by a
frozen Qwen3-8B encoder and CLM's projection heads behind a loopback HTTP server. A reasoning
model only *plans* (which ontologies are in play, what to search); deterministic code drives
[mesa-mcp](https://github.com/idss-mesa/mesa-mcp)'s OBO/OLS tools; every decision is recorded in
a provenance sidecar next to the AVU history kept by
[mesa-ducklake](https://github.com/idss-mesa/mesa-ducklake). The core is a torch-free client;
the encoder and `clm-serve` run as separate processes.

Status: pre-alpha, milestone M1 (the serving stack on sparky-1 and the hermetic rank-first
pipeline: `annotate`, `explain`, `review`, `feedback`, proposed-only). See `DESIGN.md` for the
decisions (U1–U4, D0–D33, the pre-registered experiments), `RESEARCH.md` for the verified facts
the design rests on, `design/plan-2026-09-28.md` for the plan, and `CLAUDE.md` for how to work
here.

## Why

mesa-anyjev's AnyJev readout works through next-token logprobs: a top-20 logprob cap, one slow
prompt per candidate, K>8 choices split into yes/no twins. CLM scores candidate texts against a
state with cacheable candidate vectors and no option limit. Its released head is weak zero-shot
on typed questions (state-independent Score answers, suffix collapse, inverted calibration,
and a logistic regression on the frozen embeddings that beats the fine-tuned head; upstream
issues #3, #13, #15), and a no-model lookup that copies labels across cards already scores
leave-one-card-out accuracy 0.772 on `term.fits`, above AnyJev L2's 0.765. So mesa-clm asks
rank-first questions with an anchor, fingerprints everything that depends on the encoder,
earns trust only through pre-registered, leakage-aware, nested-selection bench cells that beat
that lookup on novel keys, and ships proposed-only first.

## Install

```bash
uv sync --all-extras                      # dev install; nothing here pulls torch or vllm
uv run pytest -q                          # hermetic suite
sha256sum -c vendored.sha256              # vendored CLM and AnyJev files are byte-identical
```

Requires Python 3.11+. `mesa-mcp` and `mesa-ducklake` are installed from pinned git commits
(neither is on PyPI). Serving (a vLLM pooling container plus a patched `clm-serve`) runs on the
GPU host; see `docs/concepts/serving.md` and the runbook `docs/deploy/serving.md`.

## Command line

```bash
uv run mesa-clm doctor                    # pins, versions, mesa-mcp plugin API, policy, stores (--quick, --json)
export MESA_CLM_OLS__FIXTURES=replay MESA_CLM_PROVENANCE__DSN=duckdb:///labels.duckdb
uv run mesa-clm labels ingest-neon-eval --eval-root tests/fixtures/neon-avu-eval   # silver labels
uv run mesa-clm labels stats                                                         # counts per task, source, label
uv run mesa-clm labels snapshot --out bench/snapshots/today.parquet                  # frozen labels + labels_sha256
uv run mesa-clm bench baselines --snapshot bench/snapshots/today.parquet             # lookup_prob, novel-key, LOPO
uv run mesa-clm bench mde --snapshot bench/snapshots/today.parquet                   # minimum detectable effect
```

`labels import-anyjev --dsn <sidecar>` reads a mesa-anyjev sidecar read-only (its curator rows
arrive as weight-0 `agent_pick` unless `--trust-curator` is given at a terminal).

Annotation (milestone M1; every outcome is proposed-only until evidence-cited autos land in M4):

```bash
export MESA_CLM_OLS__FIXTURES=replay MESA_CLM_OLS__FIXTURES_DIR=tests/fixtures/ols
uv run mesa-clm --provenance duckdb:////tmp/clm.duckdb annotate \
    --card tests/fixtures/cards/DP1.10003.001.brd_countdata.md --provider fake --out run.json
uv run mesa-clm --provenance duckdb:////tmp/clm.duckdb explain --run-id <id or prefix>
uv run mesa-clm --provenance duckdb:////tmp/clm.duckdb review --run-id <id or prefix>   # interactive
uv run mesa-clm framings --check          # framing keys match framings.lock.json
```

Answers given at an interactive terminal are curator labels; `review --pick` and `feedback` run
without one (a script, an agent's shell) are recorded as an agent's answers at weight 0 and say
so (DESIGN A2). On the serving host, `--provider clm` (the default) talks to the loopback encoder
and clm-serve with the keys `mesa-clm serve keys --init` writes, `~/.mesa/clm/secrets/clm.key`
and `encoder.key`, read by default (`MESA_CLM_CLM__API_KEY_FILE` and
`MESA_CLM_ENCODER__API_KEY_FILE` name others); `mesa-clm doctor --serve` checks the stack first.
`features build|export-npz|project|stats` keep the per-encoder feature cache, and `provenance
migrate|export|import|prune` maintain the sidecar. `plan`, `apply`, `revert` and the
`mesa_clm_*` MCP tools arrive with M3; `mesa-clm --help` lists what exists.

## Links

- Documentation: https://idss-mesa.github.io/mesa-clm/ (OKF v0.2 bundle; `llms.txt`,
  `llms-full.txt`)
- Decisions: `DESIGN.md` · Facts: `RESEARCH.md` · Plan: `design/plan-2026-09-28.md`
- Sibling: https://github.com/idss-mesa/mesa-anyjev · Upstream: https://github.com/Contrastive-LM/CLM

## License

MIT, Copyright (c) 2026 The Regents of the University of New Mexico. CLM and its vendored
`schema.py` are Apache-2.0 and not affiliated with TypeSafe AI or Jev; NEON metadata in the
fixtures is CC BY 4.0; see `THIRD_PARTY.md`.
