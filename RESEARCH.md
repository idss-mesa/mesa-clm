# RESEARCH.md — verified facts (2026-09-29, M1 additions 2026-10-01, M2 results 2026-10-03)

Every fact names where it was verified: a URL, a commit, a `file:line`, or the command that was
run. Every section carries a `stale_after`: upstream CLM facts expire on 2026-12-31 (the project
is days old and unreleased), everything else on 2027-03-31. Re-check anything past its date
before relying on it. Nothing here is a credential, a host network address, a key path or a
tunnel; the serving host is referred to as `sparky-1` only. Items marked *to reproduce* are
numbers the design rests on that mesa-clm's own bench must reproduce before they are cited.
Sources abbreviated below: "plan" is `design/plan-2026-09-28.md`; "exploration" is the
read-only exploration record the plan was built from (`.local/evidence/exploration.json`,
local, not committed).

## Contrastive LM upstream (`stale_after: 2026-12-31`)

Source unless stated: https://github.com/Contrastive-LM/CLM at
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` ("README: note how to raise the 2048-token limit",
2026-09-24T23:09Z), read through the GitHub API on 2026-09-28.

- **Maturity.** Repository created 2026-09-23; 8 commits, all by one author on 2026-09-24; no
  tags, no releases, no test suite (PR #21 would add the first tests); no external PR merged;
  no commits after `bb42c6c` as of 2026-09-28; no maintainer reply on any issue read. License
  Apache-2.0. Console scripts `clm-serve = clm.server:main`, `clm-download =
  clm.heads:download_main`.
- **PyPI `contrastive-lm`** (https://pypi.org/pypi/contrastive-lm/json): one version, 0.1.0,
  uploaded 2026-09-24T21:12Z; `requires_python >=3.10`; `requires_dist` `numpy>=1.24,
  requests>=2.28, torch>=2.1, fastapi>=0.100, uvicorn>=0.23, vllm>=0.6` (vllm unconditional, no
  platform marker); the wheel lacks `pyarrow`, which the repository's `pyproject.toml` added
  ten minutes after the upload (commit `d43894a`). Consequence: the core never depends on it
  (DESIGN D0).
- **Hugging Face `Contrastive-LM/CLM-v0.1-8B`** revision
  `e939398d4556fcd9400c76fa8c5a513202f42b0a`: `CLM_v0.1-8B.pt` 75,557,149 bytes, sha256
  `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`; `config.json`
  `{"model_type":"clm","base_model":"Qwen/Qwen3-8B","encoder_pooling":"last-token",
  "embedding_dim":4096}`; card: "Encoder-locked", "probabilities are relative to that set",
  "Verifier results need fine-tuning". A "multimodal CLM-35B" is announced for early October
  2026 (blog and card); a new encoder would rotate `encoder_fp` (DESIGN D5).
- **Head shape.** 4096→1536→1536→512, GELU, LayerNorm on the hidden layer (blog architecture
  note); PR #10 reports 18,887,680 parameters for both heads (9,443,840 each by arithmetic),
  75.5 MB in fp32, matching the file size. `heads.py` `make_head(width, depth=2, proj=512,
  activation="gelu", layernorm=False, residual=False, hidden=4096)`; the checkpoint is a
  `torch.save` dict `{state_head, action_head, logit_scale, cfg{width, depth, projection_dim,
  hidden_size, activation, layernorm, residual, …}, epoch, metrics}` loaded with
  `torch.load(path, map_location="cpu")` (no `weights_only`, PR #10). Heads hot-reload on file
  mtime and `--ckpt-dir` serves every `*.pt` under its stem.
- **Vendored files** (`vendored.sha256`): `src/clm/schema.py` sha256
  `52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335`; `LICENSE`
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`; `src/clm/client.py`
  `45f565c1b30964dedc8f66ed4d4448ad43233dc4a8eb62558d539d87d2485738` (test-only wire oracle).
  `schema.py` imports only `json` and `math`; `client.py` only `requests`; there is no
  upstream NOTICE file.

### Text contract (`src/clm/schema.py`, `engine.py`)

- `to_text(x)`: `str` passes through; `bool` → `true`/`false`; numbers → `str(x)`; a `dict`
  becomes `key: value` lines with a blank line between top-level fields and nested values
  indented by 2; a `list` becomes `- item` lines. Docstring: "the heads are trained on prose,
  not on JSON". `to_text(None)` is `""`.
- `state_text(state, instructions)` (`schema.py:62-65`): `f"{s}\n\n{i}" if s and i else (s or
  i)` — context first, question last; with `instructions` None or empty the encoder sees the
  state alone. There is no instruction prefix and no chat template at serving time.
- `candidates(q)` (`schema.py:75-101`): a **Choice** option is embedded as its description only
  (the key when the description is empty; the key is never prefixed and instructions never
  reach the candidates); **Score** levels verbatim, keys `"0".."n-1"`; **Noul** candidates are
  `"true: Yes. This is true: {ins}"` / `"false: No. This is false: {ins}"` (or `"true:
  {desc}"` with custom criteria), so a noul instruction appears in the state text and in both
  candidates.
- `build_pairs(state, questions)` (`schema.py:104-112`) → `{qid: (state_text, keys,
  candidate_texts)}`: **one state text per question**, so N questions with different
  instructions cost N state embeddings; questions with byte-identical state text are
  de-duplicated.
- Scoring (`engine.py:132-136`): per question `cos = za[k:k+len(texts)] @ zq[i]` over
  L2-normalised head outputs, logits `scale * cos / temperature`, then `answer_from_logits`.
  **The softmax runs per question over that question's candidates only**: no cross-question
  normalisation, no abstain. Raw cosines are not returned over the wire.
- `scale = exp(logit_scale).clamp(max=100.0)` (`heads.py:95`); the released head's
  `logit_scale` is 4.6132 (#15) so `exp` ≈ 100.8 and the scale is **clamped to 100**: a cosine
  gap of 0.01 is one logit. `temperature ∈ (0, 100]`, default 1.0. `clm-raw` scores cosines in
  the raw 4096-d space with `RAW_SCALE = 100.0`.
- The softmax is float64 `math.exp` with no rounding (`schema.py:115-119`); with |cos| ≤ 1 and
  scale 100 the largest logit gap is 200, so the smallest probability is about e^-200 > 0 and
  no underflow re-query is needed (DESIGN, plan §4.1).
- `confidence(probs)` (`schema.py:131-145`) is `clip(p_top − mean(p_rest), 0, 1)`, algebraically
  TypeSafe's `(n·p_max − 1)/(n − 1)`; at K=2 it equals the margin `p_top − p_other`, not a
  probability. Noul `p_true = σ(scale·(cos_true − cos_false)/T)`; Score `score = Σ i·p_i`
  (an expected index). mesa-clm computes `confidence = max(probs)` itself (DESIGN D7).
- `Engine.rank(state, candidates, instructions=None, …)` wraps candidates as a Choice with keys
  `"0".."n-1"` and passes `instructions` through, including None (`engine.py:140-151`);
  `CLMClient.rank(context, question, answers)` has a different argument order and always sends
  `question`, which may be null. `/v1/rank` maps `question` onto `instructions`
  (`server.py:121-153`).
- `Engine` calls only `self.embedder.embed(missing)` (`engine.py:78`) and
  `engine.embedder.healthy()` (`server.py:84`), so any object with `embed(texts) ->
  (ndarray[n,4096], tokens)` and `healthy()` can be injected (DESIGN D16).

### Embedder, caches, server (`src/clm/embedder.py`, `cache.py`, `server.py`)

- `Embedder(url="http://127.0.0.1:8090/v1/embeddings", model="qwen3-8b", max_tokens=2048,
  cache_size=200_000, batch=32, timeout=300.0, api_key=None)`; `_fetch` (`embedder.py:40-43`)
  sends `{"model", "input", "encoding_format": "base64", "truncate_prompt_tokens": max_tokens}`
  and **no `truncation_side`** (PR #6); vectors are L2-normalised client-side (`embedder.py:56`).
- Host LRU keyed by raw text only, up to 200,000 fp32 4096-d vectors (about 3.3 GB at the
  cap, arithmetic), shared across heads and both sides; `server.py:203` builds the Embedder
  without overriding `cache_size` — hence local patch 0005 (plan §6.3) and
  `CLM_EMB_CACHE_SIZE=20000`.
- Device `VectorArena` (`cache.py`): projected 512-d vectors keyed
  `f"{head}@{gen}/state\x00{text}"` / `…/action\x00{text}` plus a 4096-d pool for `clm-raw`
  (`RAW_SHARE=0.125`); budget `--action-cache` / `CLM_ACTION_CACHE`, default `"0.02"` of
  device memory, capped at 90% of currently free memory; LRU, no persistence. A state vector is
  reused only when the whole state text is byte-identical.
- Server: default `--host 0.0.0.0` with auth off; `CLM_API_KEY` optional Bearer; `auth()` is not
  applied to `/health`, which lists model names, and compares the key with `!=`
  (`server.py:78-89`; local patch 0006 uses `hmac.compare_digest`). Endpoints `POST
  /v1/systemone`, `POST /v1/rank`, `GET /v1/models`, `GET /health`, `GET /` (playground,
  `--no-ui`). Errors: 401 bad key, 422 malformed or unknown model, 502 embedder unreachable.
  Latency in the `X-CLM-Latency-Ms` header; `usage.billing_units = len(questions)`,
  `input_tokens` counts encoder tokens on cache misses only. Env: `CLM_PORT, CLM_EMB_URL,
  CLM_EMB_MODEL, CLM_EMB_MAX_TOKENS, CLM_CKPT, CLM_CKPT_DIR, CLM_DEVICE, CLM_ACTION_CACHE,
  CLM_API_KEY`; client `CLM_BASE_URL` (default `http://127.0.0.1:8700`).
- `Engine()` never downloads the head; only `clm-serve` does (`--no-download` disables it).

### Training path (`train/finetune.py`, `train/embed_utils.py`)

- `TextCache` (`finetune.py:422-446`): `sha1(text)` → fp16 vector, persisted as `.npz` with a
  `<U40` key array and `float16 [N, 4096]` values, already L2-normalised. Path
  `{--embed-cache or OUT/embeddings}/choice_{_slug(embed_model)}_{max_len}.npz`
  (`finetune.py:467-468`); `_slug("Qwen/Qwen3-8B") == "Qwen_Qwen3-8B"` (verified);
  **`--embed-cache` takes a directory**. `run_choice` builds an embedding backend only when
  texts are missing (`:470-475`); `OfflineBackend` imports vllm lazily
  (`embed_utils.py:52-67`), `ServerBackend` posts token-id lists (`:70-101`).
- `--data` and `--out-dir` are required (`finetune.py:647, 728-729`); `--workflow` defaults to
  `all` (`:657`); a single parquet `--data` is used for both train and test
  (`load_typed_rows:404-405`). Row layout: `id, workflow, state, questions (System One wire
  JSON), gold {qid: {label, probabilities}}` (the `LocalLLaMA/typed-decisions` layout).
- `--task choice` defaults: width 1536, depth 3, GELU, LayerNorm, proj 512; `--init-ckpt`
  warm-starts cfg and `logit_scale`; `--loss infonce` (default) or `softce`; `--targets
  soft|hard`; AdamW lr 5e-4, OneCycle, batch 256, epochs 20, patience 5, max_len 2048 (8192
  for `--task clm`). `torch.load(..., weights_only=False)`.
- `Recipe.state_ids` (`embed_utils.py:45-49`) applies the chat template to message-list states
  and keeps the **tail** (`cap = max_len − 1`); string states are tokenised raw, tail kept.
  Serving never applies a chat template (train/serve layout mismatch, unanswered in #8/#15).

### Issues and pull requests (`stale_after: 2026-12-31`)

- **#3 "Unexpected scores"** (issue, 5 comments): Score returns "Very angry" at 0.999 for "I'm
  very calm.", also with the levels reversed, reproduced on vLLM/CUDA, MLX and PyTorch-MPS.
  Noul ("Is the customer angry?") and Choice with descriptive options do track the state
  (p 0.20/0.39/0.48/0.97 and 0.35/0.03/0.83/0.93 across calm→furious). The tides rank example
  reproduces at 0.993 across backends (model card 0.993, README 0.997). → DESIGN D3.
- **#6** (PR, head `11211fcab1c2`, open): vLLM's pooling runner truncates from the right, so
  states over the limit lose the appended question; the fix is `body["truncation_side"] =
  "left"` next to `truncate_prompt_tokens`; cosine to the tail-kept vector −0.0006 before,
  1.0000 after. Carried as serving patch 0001.
- **#10** (PR, head `cf230d2a5337`, open): `torch.load(..., weights_only=True)` in `heads.py`
  and `finetune.py`. Patch 0003.
- **#11** (PR, head `fc4d8d85b41b`, open): loopback default host and refusal to start an
  exposed unkeyed server (`server.py` +39 −3). Patch 0002; local patch 0005 (Embedder api_key
  and cache size) is generated against the 0002-applied tree, where the `Engine(Embedder(…))`
  call moves from `bb42c6c:203` to about line 240.
- **#13** (PR by an outside contributor, head `sabyasm/CLM@8c2d65a4`, open, 0 reviews):
  zero-shot Banking77 0.051, Bitext 0.056, typed-decisions 0.355 (below the 0.483 majority
  baseline), the same answer for every input on 14 of 20 question types, rewording no help;
  fine-tuned heads 0.775/0.795, 0.991/0.989, 0.686/0.661; `StandardScaler +
  LogisticRegression(max_iter=2000)` on the frozen suffixed 4096-d state vectors 0.900, 0.997,
  **0.737**. Question-first and state-only layouts were not tested. → DESIGN D18, X2 replica.
- **#15 "README Quickstart … noul 0.84 vs documented 0.41"** (issue, open, one comment):
  README numbers do not reproduce (noul 0.84 vs 0.41; 98 vs 106 tokens); zero-shot BoolQ noul
  ≈0.133 on every item (498/500 "no"); Emotion picks "sadness" 460/500; mean cosine between
  *different* states **0.958 with the question suffix, 0.452 for the state alone, 0.947 after
  the head (suffixed text)**; the reporter's hypothesis is that the shared suffix dominates
  last-token pooling. The single comment (2026-09-28T13:05:48Z) ran PyPI 0.1.0 with **vLLM
  0.30.0 `--runner pooling` (`seq_pooling_type='LAST'`), `--max-model-len 8192`, `clm-serve
  --max-tokens 8192` on a DGX Spark GB10 (aarch64, CUDA 13.0, torch 2.13)**: quickstart urgency
  0.8366 (README 0.4102), billing 0.9888, frustration 2.0000, 98 tokens; an 8-scenario ×
  4-phrasing eval scored 22% (`clm-latest`) and 31% (`clm-raw`), about chance; **inverted
  calibration**: 0% accuracy at confidence 0.9–1.0, 41% below 0.3. These are the doctor's
  upstream-drift probes (plan §6.8). → DESIGN D3, D6, D23, X1 F7/F9.
- **#22** (issue, 0 comments): Chinese inputs below 30% accuracy.
- **#23** (PR, head `b618c1d89940`, open): `VectorArena` slot leak on concurrent misses. Patch
  0004.
- **#14 / PR #20**: `run_choice`'s majority baseline compares option indices across splits
  (bug; fix unmerged).
- **#16/#17**: install fails on macOS because `vllm` is unconditional; PRs #5/#7/#9/#18 add
  Apple Silicon and llama.cpp encoders (PR #7 `MpsEncoder`: `AutoModel`, right padding,
  `truncation_side="left"`, `last_hidden_state` at the last non-pad token — the recipe the
  fallback `CudaEncoder` follows, plus L2).
- Encoder-recipe cross-checks: the #15 author's custom `/v1/embeddings` (final-norm hidden
  state at the last token, L2) matched real vLLM pooling at min cosine 0.999998 over 109 texts
  on Qwen3-0.6B; fp32 vs bf16 quickstart 0.842 vs 0.837; PR #7 MPS 0.84301, CPU bf16 0.84629 —
  every encoder route lands near 0.84, so the README gap is head or layout, not encoder.

### Collapse on mesa-clm's own contexts (M1-A; `stale_after: 2027-03-31`)

Label-free, report-only (plan §8 M1-A, X1's collapse diagnostic): mean pairwise cosine between
*different* contexts of the 285 term.fits and 190 column.ontology_fits rows of
`bench/snapshots/2026-09-29.parquet`, raw 4096-d and after the released state head (512-d).

- *Measured 2026-10-01 on the pipeline's rendering* (`bench/results/2026-10-01/collapse_spike.json`,
  `encoder_fp` c3b3d5e1a283; the state-only, F9 and F1 texts equal the feature manifest's F7, F9
  and F1 contexts): term.fits state-only (F7) **0.831** raw / **0.677** projected, with the
  question appended **0.993 / 0.965**, F9 0.945 / 0.749, the F1 control 0.992 / 0.959;
  column.ontology_fits state-only **0.949 / 0.767**, suffixed **0.995 / 0.971**, F9 0.939 / 0.727,
  F1 0.991 / 0.946. Issue #15's direction (0.452 state only against 0.958 suffixed, raw) holds:
  appending the question collapses different targets on mesa-clm's contexts too, and the head
  keeps the state-only contexts further apart than the raw space does.
- *The first run used a rendering the pipeline never sends.* `bench/results/2026-09-29/collapse_spike.json`
  (M1-A recipe 852efc921a8a) rendered the snapshot's canonical, sorted-key `state_json`
  directly, so nested objects (the card header first) were in sorted key order: state-only
  0.730 / 0.675 (term.fits) and 0.854 / 0.740 (ontology_fits). Re-run on the A3/A4 recipe with the
  same sorted-key texts (`bench/results/2026-10-01/collapse_spike_sorted_keys.json`) it gives
  0.7297 and 0.8536 raw, within about 5e-4 of the first run's means (0.8541 there for
  ontology_fits state-only; term.fits F9 0.9448 against 0.945): the encoder recipe barely moves
  these numbers, the key order does (term.fits within-card and across-card means 0.832 / 0.831 on the
  pipeline's rendering against 0.744 / 0.727 on the sorted one). The 2026-09-29 state-only texts
  remain the fixed inputs of the fallback parity and batch-invariance runs, which compare
  encoder routes on the same texts and do not depend on the rendering.

## vLLM pooling (`stale_after: 2027-03-31`)

- `truncation_side` was added in vLLM **v0.18.0**
  (`vllm/entrypoints/pooling/base/protocol.py@v0.18.0` L27; wired in
  `embed/protocol.py@v0.18.0` L67, L97; handled in `vllm/renderers/params.py@v0.18.0`
  `get_encode_kwargs` ≈L300 and `_token_truncation` ≈L397, where `"left"` keeps
  `tokens[-max_length:]`). In v0.17.1 the field is absent; `OpenAIBaseModel` is
  `ConfigDict(extra="allow")` so older servers accept and silently ignore it. Pooling's default
  truncation side is `"right"` (`vllm/tokenizers/registry.py@v0.13.0` L128-132; the same
  mapping exists in v0.11.0–v0.12.0 `init_tokenizer_from_config`). Only vLLM ≥0.18 plus
  `truncation_side="left"` is safe; both local images qualify.
- A generative model under `--runner pooling` is converted to `embed` with **LAST** pooling and
  normalisation on (https://docs.vllm.ai/en/latest/models/pooling_models.html); the #15
  commenter's engine config confirmed `seq_pooling_type='LAST'`.
- v0.27.1 enables prefix caching by default for decoder-attention LAST-pooling models
  (`vllm/engine/arg_utils.py` `_set_default_chunked_prefill_and_prefix_caching_args`
  `default_prefix_caching`; `config/model.py:1952-1977`), so the encoder unit passes
  `--no-enable-prefix-caching` explicitly until golden-vector parity with caching on is recorded
  (plan §6.1). API-key auth guards only `/v1`, `/v2`, `/inference` and `/cohere`
  (`vllm/entrypoints/openai/server_utils.py:42`); `/health`, `/tokenize` and `/metrics` are open.
  `/tokenize` works under `--runner pooling` (measured 2026-09-29, below), and the M1-A probes
  found more open routes than this list (DESIGN A4).
- vLLM's startup check compares `gpu_memory_utilization × total` against *currently free*
  memory (a CARC spec note: 0.40 of 121.7 GiB failed with 42 GiB free). Whether GB10 page cache
  counts as used there is still unverified (see "Unverified, scheduled").
- `/v1/embeddings` requests default to `add_special_tokens=True`
  (`vllm/entrypoints/pooling/base/protocol.py@v0.30.0` L202-203), which adds nothing for Qwen3
  (below).
- PyPI vLLM 0.30.0 (2026-09-22) ships `manylinux_2_28_aarch64` and pins `torch==2.13.0`; the
  mesa-anyjev venv has torch 2.14.0, so vLLM can never share a venv with it. mesa-clm uses the
  container image instead (U3) and the serve venv has no vllm (plan §3).

- *Measured 2026-09-29 on sparky-1 (M1-A recipe, `encoder_fp` 852efc921a8a;
  `bench/results/2026-09-29/serving_m1.json` unless stated).* vLLM 0.27.1 `--runner pooling`
  runs Qwen3-8B bf16 on the GB10 (sm_121) from the pinned image, so neither the NGC image nor the
  in-process fallback was needed (K0 steps 1-2): the log shows the model loaded (14.11 GiB in
  66.0 s), `seq_pooling_type='LAST'` with `use_activation=True`, `enable_prefix_caching=False`
  (`--no-enable-prefix-caching` accepted), 105.2 of 121.69 GiB free at start-up and a 9.49 GiB KV
  cache (69,072 tokens; the log lines are `bench/results/2026-10-01/encoder_logs.json`, the start
  of 2026-09-29 18:30Z). `VLLM_API_KEY` through `docker run --env-file` works: `GET /v1/models` and
  `POST /v1/embeddings` answered 401 to no key, a wrong key and clm-serve's key, 200 with the
  right one. `/tokenize` works under `--runner pooling` (200; unguarded on that recipe, guarded
  since DESIGN A4). Without a key `POST /pooling`, `/invocations`, `/score`, `/rerank` and
  `/detokenize` answered 200 (embeddings and scores served unauthenticated) while `/v1/score` and
  `/v2/embed` answered 401: vLLM's key check covers path prefixes, not routes (DESIGN A4).
- *Measured 2026-09-29 (`serving_m1.json`).* Left truncation works as PR #6 says: a
  ~5,005-token text sent with `truncate_prompt_tokens` 4,095 and `truncation_side: left` embeds
  to cosine 1.0000000 with its last 4,095 token ids; the default (right) truncation keeps the
  first 4,095 ids (cosine 0.8107 with the left-truncated vector). The M1-A recipe was not
  batch-invariant: 20 golden texts one per request were bitwise identical on repeat (20/20), the
  same 20 as one batch against one by one reached min cosine 0.9999507, and two identical batches
  0.9998659. The numpy head projection equals CLM's torch `HeadPair` on identical vectors (max abs
  projection difference 6.3e-7, probability 3.8e-6, 50 questions / 159 candidates), and the
  released head loads with `torch.load(..., weights_only=True)` (patch 0003). clm-serve's
  `/v1/systemone` against the local route differed by up to 0.0434 (`clm-latest`) and 0.0591
  (`clm-raw`) in probability, and the local route against itself by 0.0248 / 0.0363: the batch
  dependence, not the head, which DESIGN A3 removed (3.9e-6 / 6.6e-5 on the A3/A4 recipe,
  `bench/results/2026-10-01/serving_m1b.json`, and again on A5's, `serving_m1c.json`).
  The CLM README quickstart gave urgency 0.8449, billing 0.9878, frustration 2.0000 and the
  model-card tides rank 0.9939 on that recipe.
- *Measured 2026-09-29 (`serving_m1.json#/latency`).* A rank-fit Choice (a 371-token
  `target_state`, 12 PATO candidates plus the anchor, `clm-latest`) takes p50 1.0 ms / p95
  1.6 ms when the state was seen before and 110.1 / 111.6 ms for a new state with cached
  candidates (heads on CPU, D17); `/v1/embeddings` for a batch of 32 contexts (14,004 tokens)
  p50 2,820 ms / p95 2,841 ms (30 repeats each).
- *Measured 2026-10-01 on the pinned image (vLLM 0.27.1, Python 3.12.13, starlette 1.6.0,
  fastapi 0.136.3; `docker run --entrypoint python3` on the image).* Under `--runner pooling` for
  Qwen3-8B, vLLM's own `build_app` registers 18 routes plus a `/metrics` mount and no WebSocket
  route: `GET /load`, `/version`, `/health`, `/metrics`, `/v1/models`, `/ping`; `POST
  /tokenize`, `/detokenize`, `/ping`, `/invocations`, `/pooling`, `/v1/embeddings`, `/v2/embed`,
  `/score`, `/v1/score`, `/rerank`, `/v1/rerank`, `/v2/rerank`
  (`bench/results/2026-10-01/serving_m1b.json#/auth/enumeration`, built by
  `serving/vllm_routes.py`). The startup `Route:` log lines omit mounts. `--middleware` with an
  async function goes through FastAPI's `app.middleware("http")` after vLLM's own middleware,
  so it is the outermost layer (`vllm/entrypoints/openai/api_server.py` `build_app`, the
  `args.middleware` loop; the enumeration's `middleware_outermost_first`).
  `--disable-fastapi-docs` removes `/docs`, `/redoc` and `/openapi.json`. The image sets no
  `PYTHONPATH` (`docker image inspect`). mesa-clm's guard `serving/vllm_auth.py` now answers 401
  on every route but `/health` (DESIGN A4).
- *Measured 2026-10-01.* An input of exactly `--max-model-len` (4,096) token ids still never
  completes (timeout at 30 s), and the abandoned request is aborted when the client disconnects,
  also with a function middleware in the stack: 5 s later the engine reports 0 running and 0
  waiting requests and a 4,095-id request answers in 1.03 s
  (`bench/results/2026-10-01/serving_m1b.json#/boundary`). Clients truncate to 4,095 (DESIGN A4).
- *Measured 2026-10-01.* `VLLM_BATCH_INVARIANT=1` starts and applies on the GB10 (sm_121): the
  image's `model_executor/layers/batch_invariant.py` `enable_batch_invariant_mode` takes its
  non-SM80 branch (`CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUBLASLT_WORKSPACE_SIZE=1`, cuBLASLt
  preferred) and registers batch-invariant softmax, mean and `bmm`; unquantized linear layers
  call the persistent Triton matmul (`model_executor/layers/linear.py`, the
  `VLLM_BATCH_INVARIANT` branch of `apply`), RMSNorm a Triton kernel (`layernorm.py`), and
  FlashAttention runs with `num_splits=1`. On start the attention selector drops FLASHINFER
  (`bench/results/2026-10-01/batch_invariance.json#/arms/B1/container/log_lines`); the log
  (`journalctl --user -u mesa-clm-encoder.service --since "2026-10-01 08:43:58"`) shows torch's
  `preferred_blas_library` warning and the JIT compilation of `matmul_kernel_persistent` on the
  first requests. Effect: one text per request, batches of 32 and 8 concurrent requests agree to
  1 − 3.3e-15 in cosine (1 − 1.05e-4 without it), at 3,266.6 ms against 2,885.2 ms for a 32-text
  batch measured the same day (`batch_invariance.json#/arms`). One text per request is bitwise
  reproducible with and without it.
- *Verified 2026-10-01 in the image.* vLLM reports usage to `https://stats.vllm.ai` by default
  (`vllm/usage/usage_lib.py` `is_usage_stats_enabled`; `envs.py` `VLLM_USAGE_STATS_SERVER`);
  `VLLM_NO_USAGE_STATS=1` or `DO_NOT_TRACK=1` turns it off, and the encoder recipe sets both.

## Encoder model: Qwen/Qwen3-8B (`stale_after: 2027-03-31`)

- The HF cache on sparky-1 holds `models--Qwen--Qwen3-8B` snapshot
  **`b968826d9c46dd6066d109eabc6255188de91218`** with all 5 safetensors shards (about 16 GB
  bf16), config and tokenizer — the same revision #15 used; it is the pinned `--revision`.
  `models--RedHatAI--Qwen3-8B-NVFP4` holds tokenizer and config only. No CLM head was in the
  cache at exploration time (`~/.cache/clm` absent).
- `config.json`: hidden 4096, 36 layers, 8 KV heads, `head_dim` 128, vocab 151,936,
  `max_position_embeddings` 40,960, `torch_dtype bfloat16`. KV cache cost 2×36×8×128×2 B =
  **144 KiB per token** (288 MiB per 2048-token sequence).
- Tokenizer (`Qwen2Tokenizer`, transformers 5.17.0, offline): `add_bos_token: False`,
  `bos_token: None`, `eos_token <|im_end|>`, `model_max_length 131072`; the `post_processor` is
  plain ByteLevel, so `add_special_tokens` changes nothing (`tok("hello world")` → `[14990,
  1879]` either way).
- `transformers/models/qwen3/modeling_qwen3.py:356,423-425`: `last_hidden_state` is
  post-final-norm; `capture_outputs(tie_last_hidden_states=True)` makes `hidden_states[36]`
  the same tensor. AnyJev's `HFBackend.hidden_states(prompts, layers=[36])` followed by L2 is
  CLM-compatible (fallback recipe, plan §6.6). Qwen3-8B bf16 loads through transformers in
  about two minutes at 16.4 GB resident (mesa-anyjev RESEARCH.md, 2026-09-25); torch 2.14's
  Triton rotary kernel needs `Python.h` at first use (mesa-anyjev D19), which the uv CPython
  3.11.16 interpreter ships and the system Python 3.12.3 does not.
- Qwen3-Embedding-8B is a different model (base `Qwen3-8B-Base`, vocab 151,665,
  `Instruct:/Query:` prefix) and is not usable with the CLM head.

### Rendered-state token lengths (measured 2026-09-28, mesa-anyjev venv, offline tokenizer)

States from mesa-anyjev `.local/labels.duckdb` (`state_json`), rendered as
`state_text(json.loads(state_json), lock[q]["text"])`:

| task | n | question tokens | p50 | p95 | max |
|---|--:|--:|--:|--:|--:|
| term.fits | 285 | 29 | 502 | 628 | 685 |
| column.ontology_fits | 190 | 19 | 470 | 587 | 630 |
| column.annotate | 98 | 25 | 434 | 535 | 591 |
| avu.keep | 313 | 24 | 862 | 1116 | 1165 |

None exceeds 2048. Logged decisions in mesa-anyjev's provenance files max at 1,337 tokens.
Synthetic worst cases over the 145 SRER table cards (real builders, max-length candidates):
`card_header` max 1,076; column.annotate max 1,330; term.fits max 1,517; avu.keep with 25
siblings max **2,272** (2 of 145 over 2048). Each extra site in `card_header` costs about 39
tokens. Raw cards are far larger (fixture cards p50 2,852 tokens; SRER product cards p50
5,369, max 21,987) and are never sent (DESIGN D23, D26). Median chars per token on these cards
is 2.4–2.7, hence the conservative chars/2.0 fallback in the token guard (plan §4.3).

## sparky-1 host facts (`stale_after: 2027-03-31`)

Read-only commands on 2026-09-28 (`nvidia-smi`, `nvcc --version`, `free -h`, `uname -a`, `id`,
`getent group docker`, `sg docker -c "docker images"`, `docker inspect`, `ss -ltn`,
`journalctl`, `systemctl show-user`).

- NVIDIA **GB10** (DGX Spark class), aarch64, 20 cores (10× Cortex-X925 + 10× Cortex-A725),
  driver 580.173.02, CUDA 13.0 (nvcc 13.0.88), compute capability 12.1; **121 GiB unified
  memory** (vLLM sees 121.7 GiB total, so 0.01 of utilisation ≈ 1.22 GiB); `nvidia-smi` reports
  memory as "Not Supported" on unified memory, so footprints are measured as the
  `/proc/meminfo` `MemAvailable` delta plus `docker stats` (plan §6.5). 3.7 TB NVMe.
- Python: system `python3` 3.12.3 has no `Python.h`; uv-managed CPython **3.11.16** at
  `~/.local/share/uv/python/cpython-3.11.16-linux-aarch64-gnu/` ships
  `include/python3.11/Python.h` (this repository's `.venv` and `~/.mesa/.venv` use it). uv
  0.12.9, Docker 29.6.2.
- Docker: the socket is `root:docker 660`; `/etc/group` lists this user in `docker` (gid 988),
  but login sessions and the user systemd manager do not carry the gid, so every docker call
  runs through **`sg docker -c "…"`** (units use `ExecStart=/usr/bin/sg docker -c …`, plan
  §6.1). `loginctl` `Linger=yes`. nvidia-container-toolkit is installed (`nvidia-ctk`,
  `/var/run/cdi/nvidia.yaml`).
- Images present: **`vllm/vllm-openai:latest` = `VLLM_IMAGE_TAG=v0.27.1`, arm64, digest
  `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`**, `CUDA_VERSION
  13.0.2`, `TORCH_CUDA_ARCH_LIST 8.0 8.7 8.9 9.0 10.0 11.0 12.0`, entrypoint `vllm serve`,
  22.5 GB — the pinned encoder image; fallback **`nvcr.io/nvidia/vllm@sha256:9204569b17ee4c0eff75194b8e6e458479c8aee18953b5ab9cf359fcdac659e2`**
  = `VLLM_VERSION 0.21.0+2325b6f0`, `NVIDIA_VLLM_VERSION 26.05.post1`, CUDA 13.2.1 (runs on the
  13.0 driver through the compat libraries), `12.0+PTX`, 20.4 GB, proven on this GB10 by CARC's
  two-week run (2026-09-09 to 2026-09-23). torch 2.14.0+cu130 also lists only sm_120 and still
  runs Qwen3-8B bf16 here, so a missing sm_121 entry is not a blocker by itself; vLLM's custom
  kernels and packaging are the risk (K0).
- Ports 8090 and 8700 were free at exploration time; mesa-clm binds only those two, on
  loopback. Other loopback ports on this host belong to other services and are never reused.
- *Checked 2026-10-01 (pre-merge review).* A free loopback port can be bound by any local
  account, and the units are not enabled at boot, so 8090 and 8700 are usually free. `ss -ltne`
  shows each listener's owner (`uid:`, omitted for root) and cgroup: the socket unit's
  127.0.0.1:8090 is `uid:1000` in `user@1000.service/init.scope` (the user manager),
  clm-serve's :8700 `uid:1000` in `app.slice/mesa-clm-serve.service`; `/proc/net/tcp` carries
  the same owner. `/proc`
  is mounted without `hidepid` (`rw,nosuid,nodev,noexec,relatime`), so any account can read
  another's command lines: a key passed on argv is exposed. The user manager gives its services a
  soft `RLIMIT_NOFILE` of 1,024 (hard 500,000); `systemd-socket-proxyd` (systemd 255.4) takes six
  descriptors per connection (two sockets, two splice pipes) and connects to its target by path,
  following symlinks. `kernel.apparmor_restrict_unprivileged_userns=1`, so `--user` units cannot
  sandbox with `TemporaryFileSystem=`/`BindPaths=`. Hence `serving/encoder_proxy.py` (DESIGN A5,
  revision before the merge).
- CARC co-tenancy: `/opt/carc/carc-agents/models/*.yaml` (2026-09-09) pins four vLLM backends
  to this host at `gpu_memory_utilization` 0.05 + 0.12 + 0.30 + 0.36 = **0.83**; all four
  stopped 2026-09-23 15:28 MDT and `carc-vllm@.service` is disabled, but the specs still name
  sparky-1. If they return, about 7.7 GiB remain (arithmetic), not enough for a bf16 Qwen3-8B
  encoder: hence the 0.20 share, `mesa-clm-check-headroom`, and the written allocation request
  (M1). The CARC gateway itself serves no bf16 Qwen3-8B pooling model (`carc-embed` is
  Qwen3-Embedding-0.6B; `carc-fast` is NVFP4 and generative), so it cannot back CLM.
- GPU budget at 0.20: 24.3 GiB (weights ≈15–16.4 GiB, KV 8×4096×144 KiB = 4.5 GiB). clm-serve
  on CPU: unpatched worst case ≈4–5 GiB (200k-entry LRU ≈3.3 GB + 512 MiB arena + torch CPU +
  heads); with `CLM_EMB_CACHE_SIZE=20000` the LRU is ≈0.33 GB. Measured in M1: clm-serve VmRSS
  1.20 GiB (2026-09-29) and 1.19 GiB (2026-10-01), below.

- *Measured 2026-09-29 (M1-A recipe; `bench/results/2026-09-29/serving_m1.json#/footprint`).*
  MemAvailable 107.8 GiB with both units stopped, 80.3 with the encoder alone, 78.15 with both
  (Δ 29.65 GiB: encoder 27.5, clm-serve 2.15); `nvidia-smi` VLLM::EngineCore 24,799 MiB;
  `docker stats` 4.79 GiB; clm-serve VmRSS 1.20 GiB (VmHWM 1.20 GiB, unit `MemoryPeak` 1.06 GB),
  with `CLM_EMB_CACHE_SIZE=20000` and the 512 MiB action arena on CPU (`/health`: 536.9 MB
  reserved). The encoder ran from the read-only HF cache mount with `HF_HUB_OFFLINE=1`, as the
  unit is written (plan §6.1).
- *Measured 2026-09-29 (`bench/results/2026-09-29/fallback_parity.json`).* The in-process
  fallback (`serving/cuda_encoder.py`, bf16) at batch 8 failed plan §6.6's parity gate against
  the vLLM route: min cosine 0.998975 (gate 0.999; 52 of 220 texts below 0.9999), mean 0.999907;
  re-embedded in the same process at batch 1 it reached min 0.999401, so right padding inside
  bf16 batches made most of the gap. Systemone through the fallback differed from the vLLM route by
  up to 0.0328 (`clm-latest`) and 0.0717 (`clm-raw`) in probability. Model load 88.3 s; 220
  texts (84,660 tokens) in 25.95 s at batch 8; peak MemAvailable Δ 18.15 GiB,
  `torch.cuda.max_memory_allocated` 14.6 GiB, process max RSS 4.89 GiB. The batch-1 recipe passes
  (2026-10-01, below; DESIGN A3).
- *Measured 2026-10-01.* A container port that docker publishes on `127.0.0.1` is still
  reachable on the container's docker-bridge address from the host, by any local account: the
  encoder answered `/health` there (`bench/results/2026-10-01/serving_m1b.json#/auth/docker_bridge_address`;
  the address is not recorded). `ss -ltn` shows only `127.0.0.1:8090` because the container
  listens in its own network namespace. The bearer guard on every route but `/health` is the
  control (DESIGN A4).
- *Measured 2026-10-01 (`bench/results/2026-10-01/encoder_netns.json`).* That namespace held seven
  more listeners, all in the engine process and none behind the key: PyTorch's TCPStore of the
  single-process rendezvous on the IPv6 wildcard (`[::]:53581`, dual-stack, so on the bridge
  address too) and six Gloo collective sockets on the bridge address (read from
  `/proc/<container pid>/net/{dev,tcp,tcp6}`; vLLM's `UniProcExecutor` builds
  `distributed_init_method` from `get_ip()`, which picks the bridge address). After DESIGN A5
  (`--network none`, the API on a unix socket behind a loopback socket unit, `VLLM_HOST_IP=127.0.0.1`,
  `GLOO_SOCKET_IFNAME=lo`) the namespace has `lo` only: the six Gloo sockets listen on 127.0.0.1 and
  the TCPStore still binds `[::]`, now inside a namespace with no other interface, and the
  container has no bridge address.
- *Measured 2026-10-01 on the DESIGN A5 recipe* (`bench/results/2026-10-01/serving_m1c.json`):
  with `--kv-cache-memory-bytes 4831838208` vLLM reserves exactly 4.5 GiB of KV cache (32,768
  tokens, 8.00× concurrency at 4,096 tokens; `encoder_logs.json`); MemAvailable 107.46 GiB with
  both units stopped, 86.01 with the encoder alone (Δ **21.45 GiB**), 84.79 with both (Δ 22.67;
  clm-serve 1.22); `nvidia-smi` VLLM::EngineCore **19,609 MiB**; `docker stats` 3.10 GiB;
  clm-serve VmRSS 1.18 GiB. The encoder goldens are bitwise equal to the A4 recipe's (20/20), so
  neither the KV pin nor the transport moves a vector.
- *Measured 2026-10-01 on the A3/A4 recipe* (`bench/results/2026-10-01/serving_m1b.json#/footprint`):
  MemAvailable 107.46 GiB with both units stopped, 80.03 with the encoder alone, 78.31 with both
  (Δ 29.15 GiB; clm-serve 1.72); `nvidia-smi` VLLM::EngineCore 25,153 MiB (24.56 GiB, against
  24,799 MiB in M1-A, `bench/results/2026-09-29/serving_m1.json`); clm-serve VmRSS 1.19 GiB. vLLM
  sizes its KV cache at start-up so that its own accounting fills the 0.20 share ("Desired GPU
  memory utilization is (0.2, 24.34 GiB)" in the start-up log, `bench/results/2026-10-01/encoder_logs.json`);
  that start chose 9.87 GiB of KV cache where M1-A's measured start chose 9.49 GiB (the same
  file), and `nvidia-smi` also counts the CUDA context, so the process total moves by a few
  hundred MiB between starts. That recipe was above plan §8's 24.3 GiB by either measure; the KV
  pin of DESIGN A5 brings it under (below).
- *Measured 2026-10-01.* The in-process fallback (`serving/cuda_encoder.py`, bf16, batch 1)
  embeds 220 texts (84,660 tokens) in 27.61 s, 125.5 ms per text, after a 104.8 s model load, at a
  peak MemAvailable Δ of 17.78 GiB (`bench/results/2026-10-01/fallback_parity.json`).

- *Measured 2026-10-01 (`bench/results/2026-10-01/features_build.json`).* `features build` over
  every X1/X2 text of the 2026-09-29 snapshot (1,563 distinct texts, 620,336 encoder tokens, none
  over the window) took 195.2 s one text per request on the A3/A4 recipe, about 125 ms per text;
  the fp16 copies keep min cosine 0.9999999 with the float32 vectors, the store is 31.2 MB, the
  `clm-latest` projections of all 1,563 vectors took 2.81 s wall clock in numpy on the CPU, and the
  `TextCache` export is 13.1 MB.
- *Measured 2026-10-01 (`bench/results/2026-10-01/features_build.json#/rerun/crosscheck`,
  `x1_crosscheck.json`).* Offline scores from **float16** vectors miss clm-serve by up to 1.31e-3
  (`clm-latest`) and 3.59e-3 (`clm-raw`) in probability over X1's 200 groups, though every float16
  copy keeps cosine ≥ 0.9999999 with its float32 vector: float16 rounding spread over the vector
  moves scores beyond 1e-4 at CLM's scale of 100, and a round-trip cosine bounds angles, not
  scores. The few large dimensions are not the whole cause: Qwen3-8B's last-token vectors hold
  their largest component in one of three dimensions (2202, 2284, 3169; median |value| 0.27-0.47
  against a median component of 0.0038; float16 spacing 2.4e-4 there), but putting those three
  back from float32 brings `clm-raw` to 8.0e-4 (still eight times the 1e-4 gate) and leaves
  `clm-latest` at 1.37e-3, and renormalising the float16 vectors before the head leaves it at
  1.26e-3 (`features_build.json#/rerun/crosscheck/cause`); only float32 vectors reproduce
  clm-serve. From **float32** vectors the same 200 groups match `/v1/systemone` to 5.2e-6 and
  7.47e-5 and `/v1/rank` identically (rank and systemone return the same probabilities);
  `clm-raw`'s larger residual is clm-serve's float32 accumulation of 4096-d cosines (`engine.py`
  `za @ zq` on CPU float32) against the offline scorer's float64.
- *Measured 2026-10-01 (`bench/results/2026-10-01/features_build_m1c.json`).* The feature store in
  format 2 (float32 vectors) on the A5 recipe: 1,563 texts embedded one per request in 223.7 s;
  every float16 cast of a new vector is bitwise equal to the format-1 store's (1,563/1,563), the
  `TextCache` export is byte-identical, and the store is 57.4 MB.
- *Measured 2026-10-01 (`bench/results/2026-10-01/annotate_latency.json`).* Annotate at zero shot
  on ten non-bench SRER cards through the live stack: the first run after a restart (OLS live)
  took 1.10-4.33 s (p50 3.34 s, p95 4.18 s across the cards), most of it in clm-serve calls on
  cache misses (1.06-3.90 s summed); ten warm repeats (OLS replayed) took 12.6-41.5 ms per card at
  the p50 and at most 47.2 ms at the p95 for the decide phase, plus about 115-180 ms for the
  sidecar commit. Per card (one cold run each; ten warm repeats; the wall clock includes the
  commit):

  | card | cold s | warm p50 / p95 ms | warm wall p50 / p95 ms |
  |---|--:|--:|--:|
  | DP1.00004.001.BP_30min | 4.33 | 16.3 / 24.9 | 133 / 141 |
  | DP1.10047.001.spc_particlesize | 4.01 | 32.3 / 41.1 | 170 / 177 |
  | DP1.10107.001.mms_metagenomeSequencing | 2.40 | 31.2 / 36.4 | 179 / 187 |
  | DP1.10109.001.mga_soilGroupAbundances | 1.90 | 17.6 / 21.2 | 163 / 173 |
  | DP1.10038.001.mos_BOLDtaxonomy | 3.51 | 20.5 / 22.1 | 163 / 170 |
  | DP1.00022.001.SRPP_1min | 1.83 | 13.0 / 17.6 | 165 / 176 |
  | DP1.10043.001.mos_expertTaxonomistIDProcessed | 3.49 | 33.0 / 47.2 | 204 / 215 |
  | DP1.10067.001.bbc_percore | 3.19 | 41.5 / 45.7 | 217 / 228 |
  | DP1.00094.001.SWS_30_minute | 3.64 | 28.2 / 40.8 | 206 / 219 |
  | DP1.00046.001.THRPRE_1min | 1.10 | 12.6 / 15.0 | 180 / 184 |
- *Measured 2026-10-01 (`bench/results/2026-10-01/doctor_serve_m1c.json`).* On the A5 recipe,
  after the operator's `chmod` of `~/.mesa/clm/locks`, `mesa-clm doctor --serve` reports 34 ok,
  1 warning (the quickstart urgency of DESIGN A3) and 0 failures in 4.7 s, with the new feature
  store, encoder network and headroom timer checks ok.
- *Measured 2026-10-01 (`bench/results/2026-10-01/doctor_serve.json`).* `mesa-clm doctor --serve`
  is green on the A3/A4 recipe (30 ok, 2 warnings, 0 failures, 4.6 s): every encoder route but
  `/health` and an unknown path answer 401 without a key, as do clm-serve's three `/v1` routes;
  the 20 encoder goldens are bitwise equal one text per request and within 1 − 3.6e-15 as one
  batch; an 8,741-token text completes with 4,095 tokens charged and cosine 1.0000000 with its
  last 4,095 token ids; the golden `/v1/systemone` question matches the local route to 1.7e-8
  (`clm-latest`) and 1.1e-5 (`clm-raw`). The warnings are the quickstart urgency of DESIGN A3 and
  `~/.mesa/clm/locks` (0775) and its lock file (0664), created before mesa-clm set modes itself.
- *Measured 2026-10-01 (the review of `70dbefe`; `bench/results/2026-10-01/serving_m1e.json`).*
  The user manager (systemd 255) refuses to start a unit with `PrivateDevices=`,
  `ProtectKernelModules=`, `ProtectKernelLogs=`, `ProtectClock=` or an empty
  `CapabilityBoundingSet=` (exit status 218, each tried alone in a transient
  `systemd-run --user` unit); `ProtectHostname=` is ignored with a notice ("UTS namespace setup
  is prohibited"). Its mount-namespace options put the service in a user namespace mapping this
  account alone, and the kernel runs with `kernel.apparmor_restrict_unprivileged_userns=1`, so
  AppArmor confines such a process as `unprivileged_userns`: it denies `capable(sys_admin)` and a
  `connect()` to the unix socket the encoder container bound in a bind-mounted directory
  ("Failed name lookup - disconnected path", name `run/mesa-clm/encoder.sock`), while a socket
  bound on the host in the same kind of directory was reached. The kernel's socket
  diagnostics answer an exact lookup (`SOCK_DIAG_BY_FAMILY` for one address pair) from an
  unprivileged process, naming the client socket's owner for IPv4, IPv6 and dual-stack clients
  (`serving/tests/test_encoder_proxy.py`); a socket that has closed comes back with inode 0 and
  uid 0.
  `systemd-analyze --user security` rates the proxy unit 9.8 UNSAFE without a sandbox and 5.9
  MEDIUM with the one the review installed.

## mesa-mcp: pin `c74f3aa` vs live `8fbaedf` (`stale_after: 2027-03-31`)

Source: https://github.com/idss-mesa/mesa-mcp; `git log --oneline 8fbaedf..c74f3aa` and `git
diff --stat` in a checkout; the live stack under `~/.mesa/`.

- **c74f3aa1a3caade558f063f4f637a9127dba82d9** (merge of PR #5, 2026-09-25) is the pin. It
  adds, over `8fbaedf850301146b38b997774ab7aed1d0cb179`, exactly four commits (`09e90c0`,
  `638813d`, `95a636a`, `c74f3aa`) touching `config.py` (+`ServerConfig.strict_plugins`, env
  `MESA_MCP_SERVER__STRICT_PLUGINS`), `ducklake/client.py` (`record_avu_change(s)` return the
  committed `mesa_ducklake.Snapshot | None` instead of `None`) and `server.py`
  (`register_tool(name, description, *, input_model=None, output_model=None, meta=None)`,
  `PLUGIN_ENTRY_POINT_GROUP = "mesa_mcp.tools"`, `load_plugins(*, strict=False)` called from
  `MesaServer.__post_init__`, `_meta` = `{**spec.meta, "io.mesa/surface": …}`). `ols/`,
  `ducklake/tools`, `datacite`, `auth` and `irods` are unchanged between the two commits.
- **8fbaedf** is what the live `~/.mesa/.venv` runs (editable install; Python 3.11; six
  `mesa-mcp --transport stdio` processes at exploration time); it has **no `load_plugins`**, so
  no plugin is loadable there until the live bump (plan §7.6). neon-ducklake also pins
  8fbaedf; the bump to c74f3aa changes none of the APIs it calls (exploration followup 5 §1).
  The live checkout carries an uncommitted local patch in `auth/irods_auth.py` that the bump
  runbook preserves and the rollback re-applies.
- No plugin allowlist exists at c74f3aa (only `STRICT_PLUGINS`) → DESIGN D32. `register_tool`
  raises `ValueError` on a duplicate name; `load_plugins` skips an entry point whose name is
  already loaded, and its load order follows `importlib.metadata` discovery (not guaranteed),
  so mesa-clm uses entry point `clm` and prefix `mesa_clm_` (no collision with mesa-anyjev's
  `decide` / `mesa_decide_*`).
- MRTR: `InputRequired(message, schema, state, key)`; the server encodes `state` as base64 JSON,
  **unsigned**, capped at `MAX_REQUEST_STATE_BYTES = 16 KiB`; on resume the handler sees
  `elicited={"responses": {key: {"action", "content"}}, "state": {...}}`. mesa-mcp's own picker
  (`mesa_avu_apply_term`) offers ≤8 IRIs and accepts only one it offered (the whitelist check
  `_resolve_elicited_choice` is the pattern for the tamper guard, DESIGN D26).
- Handlers are `async def h(args, auth_value=None, elicited=None)`; `assert_allowed(path,
  auth_value)` (`mesa_mcp.irods.access`) normalises or raises `ToolError("forbidden")`;
  `reject_anonymous_write` lives in `mesa_mcp.irods._helpers` (defect (f) fix); `ToolError(code,
  message, details=None)` in `mesa_mcp.errors`.
- The DuckLake mirror uses a process-wide singleton `get_default_client()`
  (`mesa_mcp.ducklake.client`), opened lazily and **never closed**: the first stdio server to
  write an AVU keeps the single-writer catalog lock for its lifetime (DESIGN D12).
  `record_avu_change(s)` hard-code `source=f"mesa-mcp:{tool_name}"`, so mesa-clm calls
  `DuckLakeClient.record_changes` itself to keep its `mesa-clm:` source tags.
- OLS client (`mesa_mcp.ols`): base URL `https://www.ebi.ac.uk/ols4/api/v2` (`config.py:55`);
  term dict `{label, iri, curie, description, ontologyId, isRoot, hasChildren, synonyms[:5]}`;
  no parents/ancestors API; `search_terms(ontology_id=…)` leaks imported terms;
  `search_term_descendants` raises `requests.HTTPError`; synchronous `requests` (hence
  `asyncio.to_thread`, D26). `_label_to_snake` and `ontology_annotations_to_avus`
  (`ols.transform`) produce `attribute='<ontology>.<snake_label>', unit=CURIE`;
  `RESERVED_PREFIXES` is exported there.
- Conformance (`tests/test_spec_conformance_2026_07_28.py`): a 2020-12 `$schema` on input and
  output schemas that passes `check_schema`, and a non-empty `_meta` with `io.mesa/surface`.

## mesa-ducklake `7bc143f` (`stale_after: 2027-03-31`)

Source: https://github.com/idss-mesa/mesa-ducklake at
`7bc143fefb093f0a2ace748a7036598147533ca0` (live main, editable in `~/.mesa/.venv`).

- Public API: `DuckLakeClient`, `AvuChange`, `Project`, `Snapshot`. `AvuChange` is
  `extra="forbid"` with 13 fields (`project_id, snapshot_id, irods_path, target_type ∈
  {data_object, collection, resource, user}, attribute, value, unit="", op ∈ {add, delete},
  actor, ts (UTC now), source="mesa-mcp", via_ticket, rule_invocation`); `actor` and `source`
  reject blanks (`models.py:112`), but the `source` default is not validated, so an unset
  source silently records as `mesa-mcp`.
- `DuckLakeClient(catalog_dsn: str | None = None, irods_session=None, *, cache_dir=None,
  cache_cap_bytes=1<<30 (0 = no eviction), lake_root_override=None, data_collection=None)`;
  `register_project(irods_path, actor, zone) -> Project`; `find_project_by_path(irods_path)`
  is an **exact match** (the apply path walks ancestors itself, plan §7.3);
  `record_changes(project_id, actor, changes, note=None, *, session=None) -> Snapshot` raises
  `ValueError` on empty `changes` and `KeyError` on an unknown project; `get_avus`,
  `get_avus_as_of(ts)`, `get_history(limit=100)`, `list_snapshots`, `diff`,
  `recover_pending_pushes`, `close()`. With a session: create snapshot (`pending`) → pending
  push → local Parquet → iRODS put + checksum → commit the file name → delete pending → evict.
- The lake is plain Parquet, one file per snapshot with **13 columns in a fixed order**
  (`lake.py:102`, "Never reorder"); it does not use the DuckDB `ducklake` extension. The DuckDB
  catalog is **single-writer**: one OS process holds the file lock (module docstring and
  `docs/deploy/duckdb-catalog.md`). Snapshot ids are global across projects. Reads with a
  session pull at most `_ENSURE_CACHED_LIMIT = 1000` snapshots; mesa-ducklake D7 caps a project
  at 1000 snapshots with no compaction (DESIGN D13); D5: a delete must send the exact stored
  unit or the add stays effective (DESIGN D31); D4: an empty or unknown checksum counts as a
  match, and `data.cyverse.org` returns `checksum=None`, so the recorder verifies sha256 and row
  counts itself (plan §7.3).
- Dependencies: `duckdb>=1.1, platformdirs, psycopg[binary]>=3.2, pydantic>=2.6,
  python-irodsclient>=2.0, pytz`; every venv on the stack runs **duckdb 1.5.5**, hence
  `duckdb>=1.5.5,<1.6` here.
- DuckDB 1.5.5 rejects `UNIQUE NULLS NOT DISTINCT (a, b)` with a ParserException and enforces
  CHECK constraints (in-memory test in the mesa-anyjev venv, 2026-09-28) → DESIGN D11.
- *Verified 2026-10-01 in this repository's venv (duckdb 1.5.6, pandas not installed), with a
  `sys.meta_path` finder counting lookups of `pandas`:* DuckDB attempts `import pandas` twice for
  every Python value it binds: `executemany` of 1, 10 and 100 rows of two values made 4, 40 and
  400 attempts, a bound list of 100 integers 202, and the same 100 rows bound as one JSON string
  2. Each failed attempt walks the import path, which is what made a per-row sidecar commit slow
  → DESIGN implementation notes (M1), "Bulk sidecar inserts".

## neon-ducklake seam (`stale_after: 2027-03-31`)

Source: `~/neon-ducklake` (HEAD `b0f2ef2`, "chore: prepare public release", 0.1.0) with a large
**uncommitted** streaming layer (+3,931/−383 lines, new `avu/recorder.py`, `site/stream.py`,
`coord.py`, …). Line numbers below are those of that working tree as verified for the plan and
will move when it lands.

- Curation: `site curate` runs `claude -p --model claude-opus-5-5 --strict-mcp-config
  --mcp-config curation/mcp-mesa-ols.json --allowedTools
  mcp__mesa-ols__{mesa_ols_search_terms,mesa_ols_get_term,mesa_avu_from_term} --permission-mode
  dontAsk`, two replicates, in a scrubbed environment; the template runs
  `${MESA_HOME}/.venv/bin/mesa-mcp`. `HIDDEN_TOOLS` is a static list of the 8fbaedf registry
  (`site/curate.py:101-104,324`) → DESIGN D32. Allowed aspects `organism_group,
  measured_property, environmental_material, method, data_type, process`; forbidden:
  species-level taxa, UO, biomes, places, non-OBO terms.
- Records (`site/curate.py:516-525`): `kind:"product", key, rep, productCode, site, card,
  card_sha256, model, modelReported, claudeVersion (line 517), prompt_sha256, input_sha256,
  seconds, attempts, returncodes, timedOut, usage, total_cost_usd, num_turns, tool_calls,
  tool_call_counts, seen_curies, tools_visible, mcp_servers, permission_denials, stop,
  session_id, trace, createdAt, avus[{attribute, value, unit, ontology_id, curie, iri, label,
  aspect, column, rationale}], parse_ok` — the golden contract for `adapters/neon.py`
  (`test_neon_contract`).
- Validation (`site/validate.py`): `check_avu` normalises the CURIE, forbids reserved
  prefixes, UO and GAZ, requires an active OBO Foundry prefix, a resolving non-obsolete term
  with a matching label, no species-rank NCBITaxon, no ENVO biome descendant, no duplicate of a
  deterministic CURIE, value ≤2,000 bytes naming no site (`:360`), an existing column, and a
  canonical rebuild through `ontology_annotations_to_avus`. `evaluate_product`
  (`validate.py:492-526`): **accepted** = valid in every replicate and grounded in at least one;
  **proposed** = valid in one replicate or never grounded; rejected otherwise; cap 25 per
  product; rep2 is mandatory (`:1239-1242`); the generic result's `model` inherits the first
  replicate's (`:495`) → DESIGN D19. `publish_generic_results` is a no-op when
  `NEON_DUCKLAKE_CURATION_SYNC=off` (`:626-635`); `select_products` skips products that already
  have a generic result (`curate.py:594`); `NEON_DUCKLAKE_ROOT` is honoured at `env.py:62`
  (isolated-root mode, plan §7.4).
- Output files: `sites/<SITE>/curation/proposals/<DP>.rep<N>.json` (carry `seen_curies`) and
  `curation/generic/<DP>.validated.json` with keys `accepted` (items carry status
  `accepted`/`human_accepted`), `proposed`, `rejected`, `rules`, `replicates` (`card_sha256`,
  `prompt_sha256`); the generic file has **no** `seen_curies`. Today's SRER generics hold 19
  accepted items: 14 in-registry (GO 2, CHEBI 1, STATO 1, OBCS 1 out), 13 usable after
  dropping the unmapped `process` aspect (DESIGN D19).
- History: stream mode always spools; `run.py` refuses `direct` in stream mode unless
  `NEON_DUCKLAKE_MESA_MULTIWRITER=1`; the recorder takes `<root>/work/locks/mesa-tag.lock`
  through `fcntl.flock` (`avu/recorder.py:361`, `env.file_lock`) and writes snapshots straight
  through `DuckLakeClient.record_changes` (`recorder.py:440`, `avu/history.py:250`);
  `env.catalog_lock_holders()` scans `/proc` for holders of the shared catalog. These are the
  lock set and the spool format (`mesa-spool/1`) that DESIGN D12 reimplements.
- Table cards `sites/<SITE>/cards/anyjev/<DP>.<table>.md` use the mesa-anyjev `DatasetCard`
  grammar (regexes copied at `site/cards.py:86-91`) and all 145 SRER cards parse with
  `load_card` (`reports/cards.json`); nothing in neon-ducklake consumes them for decisions
  today.

## Labels and evaluation assets (`stale_after: 2027-03-31`)

- neon-avu-eval (vendored as `tests/fixtures/neon-avu-eval/`, CC BY 4.0 NEON metadata): 7
  dataset cards (DP1.10003.001 `brd_countdata`, `brd_perpoint`; DP1.10022.001
  `bet_archivepooling`, `bet_expertTaxonomistIDProcessed`, `bet_fielddata`,
  `bet_parataxonomistID`, `bet_sorting`), 42 runs (carc-tools 7, claude-haiku-4-5 7,
  claude-opus-5-5 14, claude-sonnet-5 14); `results/validated.json` has 572 valid AVUs and 303
  unique valid (card, CURIE) candidate states (92 by ≥2 models, 211 by one). Cross-model
  exact-CURIE Jaccard 0.113–0.230, self-consistency Opus 0.501 / Sonnet 0.388 (mesa-anyjev
  RESEARCH.md; `results/metrics.json`). No human gold labels exist.
- mesa-anyjev's ingest (`.local/labels.duckdb`, min_weight 0.5, highest weight per state)
  yields **285 term.fits states (86 Yes, 199 No)** and 190 column.ontology_fits (76 Yes, 114
  No). The 285 vs 303 gap is reconciled (M0, `tests/unit/test_labels.py`, `IngestReport.
  terms_missing`): 18 of the 303 pairs name one of five GAZ CURIEs (GAZ:00002518 ×6,
  GAZ:00002537 ×6, GAZ:00162932 ×3, GAZ:00000448 ×2, GAZ:00057784 ×1) whose recorded EBI OLS
  `get_term` responses carry `isRoot: true` (OLS loads GAZ without its hierarchy), and
  `ols.to_candidates` drops root terms, so `TermResolver` returns `None`; mesa-anyjev's own
  ingest reports the same 285 resolved / 18 missing. mesa-clm's fixture ingestion inserts 934
  rows (1,083 derived, 149 collapse onto an earlier D1 identity). Silver weights (plan §5.1,
  verified): term.fits consensus_all/majority/negative 16/70/199; ontology_fits 76/114;
  column.annotate 63 (0.6) / 35 (0.5); value_kind class 3 = 82 *rows* at 0.5, which is 59 of
  278 *targets* once the highest weight per target is kept (`labelled_targets`; identical to
  mesa-anyjev's `labelled_states`).
- mesa-anyjev's `.local/labels.duckdb` on sparky-1 holds no curator labels: a read-only count
  of a copy on 2026-10-01 found `consensus_all` 107, `consensus_majority` 758,
  `consensus_negative` 441 rows and nothing else, so DESIGN A2's demotion of imported curator
  rows to `agent_pick` changes no label that exists today.
- mesa-anyjev's policy and bench disagree on `min_weight` (`policy_defaults.yaml:17,65,73` =
  0.6; `bench/tasks/neon.py:53,61,62` = 0.5) → DESIGN D9. `labelled_states` keeps the highest
  weight per state (`learn/labels.py:377-396`) → DESIGN D30.
- mesa-anyjev lock task keys the ported tasks must reproduce (`questions.lock.json` at
  `6159281`): term.fits `0ccc8d141ffd30ff`, column.annotate `8b8c7f14be35925d`, column.aspect
  `982e2baa2c1c7762`, column.ontology_fits `f45eb7fe2fbefdb5`, avu.value_kind
  `18a3a5159c3a8492`.
- Defect (a), verified by running mesa-anyjev's `unit_candidate` in its venv: `endswith` in
  dict order maps centimeter→meter (UO:0000008), kilometer→meter, kilogram→gram (UO:0000021),
  milligram→gram, "kilometer per hour"→hour (UO:0000032); the table maps `kilometer` to
  UO:0000009, which is the same id it uses for kilogram. Settled in M0 by a recorded `get_term`
  for every UNIT_TABLE CURIE (18 CURIEs, every live label equal to the table label): kilometer
  is **UO:0010066** and UO:0000009 is kilogram; `lookup_unit` now uses an exact key, then a
  whole-word suffix (`micrometer` no longer becomes meter), with camelCase units normalised
  (`kilometersPerHour` → kilometer per hour, UO:0010008). A compound unit never resolves to its
  trailing base unit (`metersPerSecond` was second, `wattsPerSquareMeter` meter): EBI OLS4
  `get_term` on 2026-09-29 confirms UO:0000094 meter per second, UO:0000155 watt per square
  meter, UO:0000308 milligram per kilogram, UO:0000080 square meter and UO:0000096 cubic meter
  (recorded fixtures, rows in `UNIT_TABLE`); a `search_terms(..., "uo")` for "millimole per
  liter", "nanomole per gram" and "microsiemens per centimeter" returned no UO term and
  "micromole per square meter per second" only "microeinstein per square meter per second"
  (UO:0000160), so those miss and reach the pipeline's OLS search.
- OLS fixtures: 1,884 recorded OLS4 responses under `tests/fixtures/ols/`: 777 from mesa-anyjev
  (they cover only the queries its fake backend happened to make), 15 recorded for `UNIT_TABLE`
  (one `get_term` per new CURIE plus `search_terms("kilometer", "uo")`), and the fixture closure
  recorded once on 2026-09-29 with `scripts/record_ols_closure.py` at 4 requests/s
  (`.local/ols_closure_report.json`): 1,568 unique calls over the 7 cards and 1,163 candidate
  groups (1,346 `search_terms`, 6 `search_term_descendants`, 216 `get_term_children` at
  `max_candidates` 12), 0 failures, 1.63 MB in all; `test_ols_fixture_closure` re-enumerates
  the closure and finds 0 missing. EMBL-EBI OLS4 terms are under the OBO Foundry ontologies'
  own licenses (`THIRD_PARTY.md`).

- **The teacher corpus (D19), read label-free on 2026-10-04** (`~/neon-ducklake/curation/generic/`,
  a live working tree: the neon-ducklake checkout was at commit "data: KONZ curation results and
  updated ontology maps" of 2026-10-04 16:18 -0600, which rewrote `DP1.00001.001.validated.json`
  at 16:17:55 while M4's pre-run work was reading the directory; the content hash
  `bench.registered.teacher_corpus_sha256` therefore moved from `d97b1690b450…` to
  `d90387928bed…` that afternoon, and M4 pins the bytes its ingest reads together with the
  neon-ducklake commit). The corpus holds **55** `<DP>.validated.json` files and **389** `accepted`
  items (every status `accepted`; no `human_accepted` list exists), every file's `model`
  `claude-opus-5-5`, and **no replicate proposal files** anywhere under `curation/` (so
  `teacher_implicit` rows are zero by construction). Both bench products have a file
  (`DP1.10003.001`, `DP1.10022.001`). CURIE prefixes: envo 114, obi 55, ncbitaxon 38, stato 36,
  pato 30, iao 29, chebi 14, go 11, to 10, obcs 9, pco 9, bco 9, chmo 4, uberon 4, so 4, ecocore 3,
  genepio 2, ppo 2, agro 2, oba 2, po 1, flopo 1; aspects: measured_property 114, data_type 80,
  method 63, environmental_material 55, organism_group 41, process 36. In-registry and
  aspect-mapped: 110 distinct CURIEs; 101 items name a column (231 (item, table card)
  resolutions over the SRER cards, 11 columns found in no card), 101 are dataset-level without a
  column; 54 of 55 products have a table card (`DP1.10092.001` has none). D19's "19 accepted, 14
  in-registry, 13 usable" described the SRER generics of 2026-09-29 and is superseded. The OLS
  records of the 110 CURIEs were recorded once from live EMBL-EBI OLS4 on 2026-10-04
  (22:17:41Z–22:18:12Z, 110 `get_term` requests, 0 failures; `scripts/record_ols_teacher.py`,
  `.local/ols_teacher_report.json`) into `tests/fixtures/ols-teacher/`. The ingest's own counts
  (rows per task, drops per reason) are in DESIGN "Implementation notes (M4)" once it has run.

## Baselines (`stale_after: 2027-03-31`)

- **AnyJev L2, leave-one-card-out over 7 cards** — mesa-anyjev
  `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json` (repo `6159281`, file sha256
  `74c7ad5035293e44a4800a045afa6ca1fcee3a692f8f46d416b0c0aa715d9824`; `environment.model
  Qwen/Qwen3-8B`, bf16 on this GB10, `questions_lock_sha 0190586d…`): `neon_term_fits.L2` acc
  **0.765**, ECE **0.058**, cov@5% **0.088**, cov@10% 0.396, NLL 0.499, Brier 0.327, macro-F1
  0.708, n 285, n_neg 199, 7 folds; `neon_ontology_fits.L2` acc **0.837**, ECE **0.078**,
  cov@5% **0.547**, cov@10% 0.768, NLL 0.375, n 190, n_neg 114, 7 folds. `neon_annotate.L2`
  passed the fold guards in only 1 of 7 folds (n 17, n_neg 5; class counts 63/35), so
  column.annotate has no citable AnyJev cell. These are the K2(a) comparison targets.
- **AnyJev L2 per item (M1-A, 2026-09-29), reproduced exactly.**
  `bench/baselines/anyjev_l2_2026-09-29.json` (written by `scripts/anyjev_l2_predictions.py`
  under mesa-anyjev's interpreter in the GPU window, the encoder stopped; summary
  `bench/baselines/anyjev_l2_2026-09-29.md`) holds every held-out L2 prediction: 285 term.fits
  and 190 column.ontology_fits items with card, state, target identity, label and `p_yes`. Its
  pooled and per-fold cells equal mesa-anyjev's committed
  `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json` float for float in every metric field
  (compared field by field on 2026-10-01; term.fits acc 0.765, ECE 0.058; ontology_fits 0.837,
  0.078); only the fields one file has and the other lacks differ (`head` per fold here;
  `masked`, `ms_per_decision`, `prompts`, `seconds` there), and the environment records
  `questions_lock_sha` `9f9370e0…` against `0190586d…` with the same two task keys. Every item
  maps to its own D1 identity and pairs with a row of `bench/snapshots/2026-09-29.parquet`
  with the same label and card (285 / 190 of 285 / 190), so the K2(a) comparison can be paired
  (plan §5.6 X2).
- **Lookup and control baselines** (first a read-only recomputation from mesa-anyjev
  `.local/labels.duckdb` during planning, plan §1 and §5.4; **reproduced in M0** by `mesa-clm
  bench baselines` from the fixture ingestion: `bench/results/2026-09-29/baselines.json`,
  `labels_content_sha256 5c60a8a6…`, snapshot `bench/snapshots/2026-09-29.parquet`): copying
  the most common training label of `(task, scope, target, option_key)` from the other cards
  scores LOCO acc **0.772** (term.fits, 220/285) / **0.800** (column.ontology_fits, 152/190);
  leave-one-product-out **0.723 / 0.721**; novel-key subsets (keys unseen in training cards)
  **159** items (45 Yes / 114 No) for term.fits and **99** (33 / 66) for ontology_fits; majority
  accuracy on those subsets 0.717 / 0.667. The reproduction needed one rule: when training
  cards tie on a key, the label of the alphabetically first card wins (`Counter.most_common(1)`
  over card-ordered rows); a symmetric tie rule gives 221/151 (6 tied keys in term.fits, 2 in
  ontology_fits). `lookup_prob`'s novel-key AUROC is 0.424 / 0.396, not 0.5, because the
  per-fold prior anti-correlates with the held-out card's label mix. The other tasks: annotate
  lookup 0.796 (n 98, LOPO 0.724, 6 of 7 folds would fail the 30/5 guards), aspect 0.550 (n 60,
  LOPO 0.350), value_kind 0.680 (n 278, LOPO 0.543). The lookup beats AnyJev L2's 0.765 on
  term.fits, which is why every citable cell must beat `lookup_prob` on novel keys (DESIGN D8).
- **Minimum detectable effect** (`bench/results/2026-09-29/mde.json`: 200 simulated datasets
  per grid point, grid 0.55–0.85 step 0.025, B=2000, seed 0, binormal model A vs the constant
  prior B, power 0.8 under rule R): term.fits needs AUROC ≥ 0.625 on the full set and 0.675 on
  novel keys; ontology_fits 0.650 / 0.750 (only 4 novel-key cards have ≥10 items); annotate
  0.700 on the full set and `insufficient_clusters` at every effect on novel keys (1 counting
  card); for the NLL gate the binormal scorer must reach AUROC ≈ 0.725 / 0.80 (term.fits) and
  0.775 / 0.85 (ontology_fits); aspect and value_kind are not applicable (8 and 4 classes). MDE
  took about 2 minutes on the GB10 CPU.
- mesa-anyjev end-to-end (`bench/results/2026-09-25/RedHatAI__Qwen3-8B-NVFP4.gateway.e2e.json`,
  carc-fast L0, static planner): rep-to-rep Jaccard 0.37, consensus-all recall 0.25 — the
  report-only e2e comparison in M4.

- **Vendored ranking metrics are tie-order dependent** (found by PR #1 CI, 2026-09-29): AnyJev
  `bench/metrics.py` `ece` and `coverage_risk` sort with `np.argsort` (unstable); on the M0
  lookup cell the same inputs gave `neon_annotate` ECE 0.1452 on aarch64 and 0.1473 on the
  x86-64 GitHub runners. With the tie-invariant metrics of DESIGN D33 the committed cells
  (`bench/results/2026-09-29/baselines.json`) report lookup_prob ECE 0.131 (annotate), 0.401
  (aspect), 0.152 (ontology_fits), 0.070 (term.fits), 0.142 (value_kind); accuracy, NLL, AUROC,
  the novel-key and LOPO blocks are unchanged.

## M2 results (registered run, 2026-10-03; `stale_after: 2027-03-31`)

The registered M2 run: plan §14 run once on sparky-1 from the pre-run commit `76c890f`, outputs
committed as produced in `eea525d` (DESIGN, implementation notes (M2), "M2 registered run", with
every file's sha256 and the protocol times; the decisions are DESIGN A1). Files are under
`bench/results/2026-10-03/`, and `file#/pointer` is a JSON pointer into that file. The run is the
registered one (`x1.json#/x1/registered` true, `#/x1/deviations` empty): the snapshot
`bench/snapshots/2026-09-29.parquet`, term.fits 285 items (86 Yes / 199 No, 143 targets),
column.ontology_fits 190 (76 / 114, 60 targets), 7 cards (`x1.json#/x1/tasks/<task>/n`,
`class_counts`, `n_targets`, `cards`). Silver labels are four-model agreement, not truth. Every
interval `[lower, upper]` is the pair of one-sided 95% card-cluster bootstrap bounds (B = 2000,
seed 0), a 90% interval; "cards won" is rule R's sign test (cards with ≥ 10 items; 6 of 7 needed).

**X1, per arm** (`x1.json#/x1/tasks/<task>/arms/<arm>`: `auroc`, `shuffle/auroc`, `shuffle/rule_r`,
`qualified`, `failed`, `loco/nll`). AUROC is of the raw score `s`; the shuffle comparator is the
mean AUROC of 200 within-card derangements; rule (1) needs the AUROC lower bound > 0.5 ("bound"),
AUROC ≥ 0.60 ("floor") and ΔAUROC ≻ 0 under rule R ("shuffle"). term.fits
(`#/x1/tasks/neon_term_fits/arms`):

| arm | AUROC(s) [bounds] | shuffle mean AUROC | ΔAUROC (lower bound; cards won) | rule (1) | LOCO-Platt NLL |
|---|---|---|---|---|---|
| F1@clm-latest | 0.629 [0.579, 0.675] | — | — | control | 0.6038 |
| F1@clm-raw | 0.436 [0.407, 0.471] | — | — | control | 0.6178 |
| F4@clm-latest | 0.497 [0.406, 0.572] | 0.472 | +0.0256 (−0.0112; 5) | fails: bound, floor, shuffle | 0.6285 |
| F4@clm-raw | 0.458 [0.413, 0.505] | 0.444 | +0.0136 (+0.0008; 6), passes | fails: bound, floor | 0.6227 |
| F7@clm-latest | 0.511 [0.482, 0.544] | 0.563 | −0.0521 (−0.0765; 0) | fails: bound, floor, shuffle | 0.6213 |
| F7@clm-raw | 0.494 [0.428, 0.549] | 0.480 | +0.0135 (−0.0241; 3) | fails: bound, floor, shuffle | 0.6235 |
| F9@clm-latest | 0.557 [0.512, 0.596] | 0.519 | +0.0375 (+0.0112; 5) | fails: floor, shuffle | 0.6187 |
| F9@clm-raw | 0.539 [0.495, 0.592] | 0.494 | +0.0448 (+0.0145; 5) | fails: bound, floor, shuffle | 0.6192 |

column.ontology_fits (`#/x1/tasks/neon_ontology_fits/arms`):

| arm | AUROC(s) [bounds] | shuffle mean AUROC | ΔAUROC (lower bound; cards won) | rule (1) | LOCO-Platt NLL |
|---|---|---|---|---|---|
| F1@clm-latest | 0.595 [0.528, 0.663] | — | — | control | 0.6821 |
| F1@clm-raw | 0.470 [0.414, 0.514] | — | — | control | 0.6850 |
| F4@clm-latest | 0.611 [0.529, 0.677] | 0.538 | +0.0731 (+0.0531; 7), passes | qualifies | 0.6656 |
| F4@clm-raw | 0.501 [0.399, 0.614] | 0.545 | −0.0448 (−0.0596; 0) | fails: bound, floor, shuffle | 0.6935 |
| F7@clm-latest | 0.693 [0.611, 0.762] | 0.608 | +0.0849 (+0.0478; 7), passes | qualifies | 0.6331 |
| F7@clm-raw | 0.592 [0.546, 0.637] | 0.579 | +0.0128 (−0.0386; 4) | fails: floor, shuffle | 0.6879 |
| F9@clm-latest | 0.751 [0.697, 0.805] | 0.622 | +0.1288 (+0.0898; 7), passes | qualifies | 0.6021 |
| F9@clm-raw | 0.745 [0.700, 0.804] | 0.646 | +0.0987 (+0.0729; 7), passes | qualifies | 0.6183 |

- **Decisions** (`#/x1/tasks/<task>/decision`, `a1`): term.fits **K1**, no arm qualifying on
  either model, the best AUROC 0.557 (F9@clm-latest) under the 0.60 floor; every nested fold's
  inner outcome is K1 as well (`#/x1/tasks/neon_term_fits/nested/fold_choices`; the best nested
  AUROC is 0.571). column.ontology_fits **F9@clm-latest**: eligible F4, F7, F9 on clm-latest and
  F9 on clm-raw (`#/x1/tasks/neon_ontology_fits/decision/step4/eligible`); rule (2) F9 has the
  lowest NLL among the clm-latest arms (`decision/step2`); rule (3) F9@clm-raw ≻ F9@clm-latest
  fails, Δ −0.0162, lower bound −0.1069, 2 cards won (`decision/comparisons/0`); rule (4) does not
  fire (`decision/step4/d3_amendment_recommended` false; the nearest, F1@clm-latest ≻
  F4@clm-latest, lower bound −0.0549, 3 cards won: `decision/comparisons/1`); 7 of 7 nested folds
  chose F9@clm-latest (`#/x1/tasks/neon_ontology_fits/nested/fold_choices`).
- **Candidate-only probe** (report-only; the PR #13 recipe on the candidate vectors alone,
  `#/x1/tasks/<task>/candidate_probe`): term.fits AUROC **0.808** [0.765, 0.858], NLL 0.850,
  accuracy 0.804; column.ontology_fits AUROC **0.816** [0.758, 0.873], NLL 0.458, accuracy 0.747.
- **Cost and latency** (report-only): encoder tokens per target F9 100.3 / 85.9, F7 483.9 / 481.7,
  F4 498.2 / 495.8, F1 1,166.1 / 1,707.1 (term.fits / ontology_fits,
  `#/x1/tasks/<task>/decision/tokens_per_target`). The timing run's p50 per target
  (`#/x1/tasks/<task>/arms/<arm>/diagnostics/latency`): ontology_fits F9@clm-latest 12.2 ms (91.6
  ms on a target's first ask, 11.6 ms on its second, whose texts are cached), F7@clm-latest 13.2 ms
  (133.9 / 11.8); term.fits F9@clm-latest 56.4 ms (138.9 / 18.2), F7@clm-latest 68.8 ms (205.8 /
  17.6).
- **Collapse diagnostic** (raw / projected mean pairwise cosine of distinct contexts,
  `#/x1/tasks/<task>/collapse`): term.fits F7 0.831 / 0.677, F9 0.945 / 0.749, F4 0.990 / 0.932,
  F1 0.992 / 0.959; ontology_fits F7 0.949 / 0.767, F9 0.939 / 0.727, F4 0.994 / 0.950, F1 0.991 /
  0.946; the F7, F9 and F1 values equal the M1 spike's to 4 dp
  (`bench/results/2026-10-01/collapse_spike.json`). F9's
  context names the product and the column, not the table: 101 distinct F9 contexts for 143
  term.fits targets and 45 for 60 ontology_fits targets (`collapse/F9/n`); 22 Yes/No pairs of
  term.fits items and 7 of ontology_fits in different tables have bit-identical F9 scores
  (`x1_items.parquet`, F9 rows, the same under both models).
- **Monte Carlo sensitivity** (report-only; rerun 2026-10-03 with the repository's code on the
  committed items file, nothing written: `uv run python` with `mesa_clm.bench.framing`, `items =
  read_items("bench/results/2026-10-03/x1_items.parquet")`, then for each shuffle seed s = 0–5
  `ev = ItemEvidence(items[task], shuffle_seed=s)`, `ev.shuffle(arm)` (the per-card Δ is its rule
  R's `sign.per_card`), `decide(ev)` and `nested_selection(items[task], shuffle_seed=s)`):
  term.fits F9@clm-raw's shuffle verdict (5 cards won, 6 needed) hinges on one card,
  brd_countdata, whose Δ is −0.0026 at seed 0 and ranges from −0.0039 to +0.0003 over shuffle
  seeds 0–5 (6 cards won, the verdict passing, at seeds 3 and 5); the arm fails the bound and the
  floor in any case. In ontology_fits' nested fold without brd_countdata, F7@clm-raw becomes
  eligible at shuffle seed 2 only (AUROC 0.6052, lower bound 0.5589); that fold's choice stays
  F9@clm-latest. No full-run or nested decision moves at any of the six seeds.

**The nested column.ontology_fits cells** (`tiers.json#/cells/neon_ontology_fits.<tier>.F9`;
`selection: nested`, `pre_registered: true`, `exploratory: false`, `fold_choices` 7 of 7
F9@clm-latest; the `@full` cells equal them):

| tier | acc | NLL | ECE | AUROC [bounds] | novel keys (n 99: 33 / 66): acc, AUROC (lower bound), NLL vs lookup_prob | beats_lookup_novel | threshold_cp 0.05 / 0.10 |
|---|---|---|---|---|---|---|---|
| zero_shot | 0.489 | 2.624 | 0.464 | 0.751 [0.697, 0.805] | 0.434, 0.725 (0.597), 2.846 vs 0.653 | false: NLL rule R, Δ −2.193, lower bound −2.496, 0 of 4 cards | null / null |
| calibrated | 0.642 | 0.602 | 0.123 | 0.732 [0.673, 0.802] | 0.657, 0.709 (0.584), 0.594 vs 0.653 | false: NLL rule R, Δ +0.0582, lower bound −0.0463, 4 of 4 cards | null / null |

Pointers: `metrics`, `baselines/novel_key`, `baselines/novel_key_lookup`, `baselines/beats_detail`
of each cell. The calibrated cell's per-fold Platt slopes are 0.211–0.324, none inverted
(`diagnostics/per_fold`); fitted without the label weights it gives NLL 0.599 and ECE 0.094
(`diagnostics/sensitivity_unweighted`, report-only). The paired no-model controls on the same
folds: majority 0.600, lookup 0.800 (lookup NLL 0.522) (`baselines`); the unpaired
leave-one-product-out control (task-level, its 2 product folds over all 190 items, analysis plan
§9.9): 0.721 (`baselines/lopo`). The **term.fits audit cells** (K1: `neon_term_fits.<tier>.F7`,
F7@clm-latest, `selection: none`, `pre_registered: false`, `exploratory: true`): zero_shot
accuracy 0.495, NLL 2.047, ECE 0.374, AUROC 0.511 [0.482, 0.544]; calibrated accuracy 0.698, NLL
0.621, ECE 0.105, AUROC 0.457 [0.435, 0.491], one fold inverted (bet_sorting, a = −0.0026);
`beats_lookup_novel` false in both (novel-key AUROC lower bound 0.468 and 0.396); majority 0.698,
lookup 0.772.

**The closed choices** (F7@clm-latest, `pre_registered: true`, `selection: none`, not citable in
M2; `tiers.json#/cells/<task>.<tier>.F7`): column.annotate zero_shot accuracy **0.357** against
majority **0.643** (n 98, NLL 3.971, ECE 0.620, AUROC 0.498 [0.401, 0.588]); column.aspect zero_shot
**0.050** against **0.300** (n 60, NLL 5.918, ECE 0.705); avu.value_kind zero_shot **0.324**
against **0.478** (n 278, NLL 1.987, ECE 0.274) and calibrated 0.324 (NLL 1.399, ECE 0.129;
temperature 7.62–25.99 per fold). The lookup scores 0.796, 0.550 and 0.680 on the same items.
column.annotate's and column.aspect's calibrated cells are empty: no fold trains on 100 items
(aspect 47–55, every fold `below_floor`; annotate 77–92, recorded as the 30/5 guard in 6 folds and
`below_floor` in bet_fielddata; `skipped_folds`). The `@clm-raw` cells (exploratory): annotate
0.347 (NLL 1.070), aspect 0.200 (2.251), value_kind 0.392 (1.406; calibrated 0.392, 1.343).

**X2** (`x2.json#/cells/<task>.baseline.<spec>`; baselines, never servable or cited):

| cell | acc | AUROC [bounds] | NLL | ECE | novel-key AUROC (lower bound) | beats_lookup_novel |
|---|---|---|---|---|---|---|
| neon_term_fits.baseline.lookup_prob | 0.754 (lookup 0.772) | 0.622 [0.591, 0.663] | 0.555 | 0.070 | 0.424 | — |
| neon_term_fits.baseline.pr13@S1 | 0.723 | 0.709 [0.643, 0.771] | 1.460 | 0.215 | 0.636 (0.556) | false |
| neon_term_fits.baseline.pr13@S1ns | 0.747 | 0.786 [0.738, 0.839] | 1.019 | 0.184 | 0.757 (0.714) | false |
| neon_term_fits.baseline.anyjev_l2 | 0.765 | 0.782 [0.743, 0.833] | 0.499 | 0.058 | 0.783 (0.729) | true |
| neon_ontology_fits.baseline.lookup_prob | 0.805 (lookup 0.800) | 0.818 [0.772, 0.870] | 0.522 | 0.152 | 0.396 | — |
| neon_ontology_fits.baseline.pr13@S1 | 0.821 | 0.884 [0.842, 0.927] | 0.791 | 0.131 | 0.834 (0.720) | false |
| neon_ontology_fits.baseline.pr13@S1ns | 0.821 | 0.905 [0.869, 0.941] | 0.568 | 0.137 | 0.848 (0.759) | false |
| neon_ontology_fits.baseline.anyjev_l2 | 0.837 | 0.908 [0.884, 0.940] | 0.375 | 0.078 | 0.875 (0.802) | true |

The PR #13 replica fails `beats_lookup_novel` on its NLL half (`baselines/beats_detail`); every
fold converged (`diagnostics/per_fold`). The AnyJev L2 cells join all 285 and 190 items by D1
identity with no label or card difference (`diagnostics/join`). The five recomputed M0 controls
equal `bench/results/2026-09-29/baselines.json` field for field (`x2.json#/notes/2`).

**Recomputation from the repository** (2026-10-03, on the committed files, nothing written).
`mesa-clm bench framing --decide --from bench/results/2026-10-03/x1.json` replays and recomputes
the X1 file (§14.5, 17:41:13Z, and 18:19:14Z after DESIGN A1's lock rotation). scikit-learn's
`roc_auc_score` on each arm's `s` with the positives of `framing.read_items` over
`x1_items.parquet` equals `x1.json#/x1/tasks/<task>/arms/<arm>/auroc/point` for all 16 arms to
1.1e-16. With `mesa_clm.bench.framing`, `decide(ItemEvidence(items[task], seed=b))` and
`nested_selection(items[task], seed=b)` for every bootstrap seed b = 1–200 give each task's
registered full-run outcome and all 14 fold choices (`#/x1/tasks/<task>/decision`,
`nested/fold_choices`; seed 0 reproduces them too): no decision changes.

**Independent recomputation** (reported by the two post-run reviews; not reproducible from the
repository: their own code, in scratch directories outside it, with mesa-clm's label-free manifest
supplying the item texts): X1's 16 arms reproduce (pooled AUROC exactly, the bootstrap bounds
identically, the shuffle comparator to 3.3e-15, LOCO-Platt NLL to 2.3e-11, the `p_loco` column to
5.9e-10), all 14 nested decisions are identical, and over 200 other bootstrap seeds the full-run
and nested decisions never change; every tier and X2 cell reproduces (1,923 comparisons, 0
failures, largest deviation 1.2e-12).

**What K2 will compare (M4).** K2, frozen at G1, judges each task's best servable tier on nested
cells: auto-eligible needs `beats_lookup_novel`, paired non-inferiority to AnyJev L2 on the full
item set (Δacc cluster lower bound > −0.02 with the card-sign condition) and ECE ≤ 0.08 with a
cluster upper bound ≤ 0.12; proposer-only needs `beats_lookup_novel` alone; otherwise the tier is
killed and proposals use `ols_rank`. The AnyJev side is the per-item `x2.json` cells
`<task>.baseline.anyjev_l2` (285 and 190 items, fully joined). Under K1, term.fits enters M4
through probe cells first; column.ontology_fits through its F9 tiers and probes.

## M4 results (registered run, 2026-10-05; `stale_after: 2027-03-31`)

The registered M4 run: the M4 plan §12 run on sparky-1 from the pre-run commit `07caecf`, outputs
committed as produced in `7a198b6` and `b92d762` (DESIGN, "Implementation notes (M4)", "M4
registered run": every file's sha256, the attempts and their causes, the protocol times; the
decisions are DESIGN A7). Files are under `bench/results/2026-10-04/`; `file#/pointer` is a JSON
pointer. The registered snapshot, folds, item sets and controls are M2's (RESEARCH "M2 results").
Silver labels are four-model agreement, not truth. Intervals are the pair of one-sided 95%
card-cluster bootstrap bounds (B = 2000, seed 0).

**X3, the probe cells** (`x3.json#/cells/<cell>`; a cell without `@` is the nested, citable-form
cell; `@latest`/`@raw` restrict the grid to one model's specs; `@full` is the full-data
configuration in every fold, exploratory):

| cell | n | acc | NLL | ECE | AUROC [bounds] | majority / lookup | novel keys: AUROC, NLL | beats_lookup_novel | identity (full-data choice) |
|---|---|---|---|---|---|---|---|---|---|
| neon_term_fits.probe.F7 | 285 | 0.754 | 0.485 | 0.079 | 0.804 [0.749, 0.864] | 0.698 / 0.772 | 0.785, 0.503 | **true** (AUROC lower bound 0.753; NLL ≻ lookup_prob, lower bound 0.057, rule R passed) | pair512.v1, clm-latest, logreg λ 1, Platt; 6/7 folds agree |
| neon_term_fits.probe.F7@latest | 285 | 0.768 | 0.469 | 0.077 | 0.817 [0.774, 0.866] | | 0.806, 0.479 | | pair512.v1 |
| neon_term_fits.probe.F7@raw | 285 | 0.786 | 0.509 | 0.071 | 0.770 [0.700, 0.843] | | 0.735, 0.527 | | joint4096@S1ns |
| neon_ontology_fits.probe.F9 | 190 | 0.821 | 0.404 | 0.061 | 0.892 [0.855, 0.933] | 0.600 / 0.800 | 0.833, 0.490 | **false** (AUROC lower bound 0.751; NLL rule R bound 0.047 passes, its card sign test fails) | pair512.v1, clm-latest, logreg λ 0.1, Platt; 4/7 folds (5/7 by the artifact's count) |
| neon_ontology_fits.probe.F9@latest | 190 | 0.858 | 0.358 | 0.049 | 0.916 [0.884, 0.953] | | 0.878, 0.413 | | pair512.v1 |
| neon_ontology_fits.probe.F9@raw | 190 | 0.811 | 0.424 | 0.102 | 0.882 [0.848, 0.919] | | 0.809, 0.519 | | joint4096@S1 |
| neon_value_kind.probe.F7 | 278 | 0.655 | 0.859 | 0.109 | — (K = 4) | 0.478 / 0.680 | —, 1.070 | false (no AUROC) | choice.state.v1, logreg λ 1, temperature; 3/7 |
| neon_annotate.probe.F7, neon_aspect.probe.F7 | 0 | — | — | — | — | | | false | every fold below the 100-OOF calibrator floor, as stated in advance |

Per-fold choices are `fold_choices` (term.fits: pair512.v1/logreg λ 1 in 6 folds, joint4096@S1ns in
brd_countdata; ontology_fits: pair512.v1 λ 0.1 in 4 folds, λ 1 in one, joint4096@S1/@S1ns in two;
value_kind: choice.raw.v1 in 4 folds, choice.state.v1 in 3); no fold's Platt is inverted; every
`threshold_cp` is `None` at both risks (no statistic with ≥ 30 items clears the Clopper–Pearson
bound), so no numeric `auto` can be cited from any probe cell.

**K2** (`k2.json#/tasks/<task>`): `neon_term_fits` best tier probe, verdict **b** (proposer-only):
`beats_lookup_novel` true; non-inferiority to AnyJev L2 fails (Δacc −0.011, cluster lower bound
−0.053 < −0.02); the literal card-sign test fails (4 of 7 cards, 6 needed; the report-only
margin variant also fails); ECE 0.079 ≤ 0.08 but its cluster upper bound 0.129 > 0.12. The probe
≻ the K1 calibrated audit cell on NLL (that cell is `selection: none`, not a candidate).
`neon_ontology_fits` best tier probe (≻ the calibrated F9 cell on NLL under rule R), verdict
**c**: `beats_lookup_novel` false (above); non-inferiority fails (Δacc −0.016, lower bound −0.053;
2 of 7 cards); ECE 0.061 with upper bound 0.131. `neon_value_kind`, `neon_annotate`,
`neon_aspect`: **c** (no AUROC; no evaluated fold; no AnyJev cell). The clm-raw clause:
`head_adds_nothing` **false** for both rank_fit tasks — `@latest` ≻ `@raw` on NLL under rule R
(lower bounds 0.012 and 0.026), so CLM's head adds signal over the raw encoder.

**X4** (`x4.json#/cells/<task>.probe.<framing>@teacher-<arm>[-minus-opus]`, all exploratory
variants): teacher rows (231 term.fits, 204 column.ontology_fits; none in a test fold) at
(0.5, 0.3) versus off on novel-key NLL under rule R — `term.fits` Δ +0.006 (lower bound −0.062)
on silver, +0.035 (−0.037) on the 187 surviving silver-minus-Opus items; `column.ontology_fits`
−0.106 (−0.166) and −0.114 (`insufficient_clusters`, 66 surviving novel-key items); (0.3, 0.1):
term.fits +0.035 (−0.008) / +0.047 (−0.013), ontology_fits −0.070 (−0.126) / −0.102. **keep =
false** for both tasks (`diagnostics.teacher_decision`). Cell metrics: term.fits off 0.761 /
0.507 / 0.073 (acc / NLL / ECE), 0.5 arm 0.765 / 0.508 / 0.086, 0.3 arm 0.761 / 0.494 / 0.116;
ontology_fits off 0.821 / 0.405 / 0.050, 0.5 arm 0.768 / 0.457 / 0.110, 0.3 arm 0.768 / 0.458 /
0.093. The expected ≈ null held for term.fits and the teacher rows hurt ontology_fits.

**Artifacts** (`artifacts_v1/78be8c462b2e/manifest.json`, version 1 under clm-latest, registered):
probes `term.fits` (`a686a06aab14497a`: pair512.v1, logreg λ 1, Platt, n_train 285, cite
`x3.json#neon_term_fits.probe.F7`, 6/7 folds), `column.ontology_fits` (`c95785008b523fd0`:
pair512.v1, λ 0.1, cite its probe cell, 5/7), `avu.value_kind` (`082178504fb4a3c9`: choice.state.v1,
λ 1, temperature, 3/7); calibrators for the five tasks, only `column.ontology_fits`'s citing a
nested cell (`tiers.json#neon_ontology_fits.calibrated.F9`, 7/7). Promoted on the serving host
by `learn promote --version 1 --task term.fits --tier probe` (DESIGN A7); nothing else promoted.

**Production audit: deferred, not run** (DESIGN A7, the appended note): no `audits` row exists;
the seven live runs on non-bench SRER cards (2026-10-05 04:57Z–05:06Z: `DP1.00004.001.BP_30min`,
`DP1.00002.001.SAAT_30min`, `DP1.00013.001.wdp_collectionChem`, `DP1.00038.001.wdi_isoPerSample`,
`DP1.10047.001.spc_biogeochem`, `DP1.10026.001.cfc_carbonNitrogen`, `DP1.00024.001.PARPAR_30min`;
1,380 decisions, the promoted `term.fits` probe deciding every `term.fits` group) and the
artifact-only sample of 100 probe decisions (7 would-be-auto by the top decile, 60 proposed, 33
anchor-abstains) are in the sidecar and `.local/m4/audit/`. Label-free, from those runs' groups:
the A6 aspect fallback assigned `method` 53, `taxon` 44, `measurement` 1 to the `term.fits`
column groups; 25 of the 44 `taxon` assignments went to numeric or unit-bearing columns.

**A8 on the seven SRER cards** (DESIGN A8; label-free production runs on 2026-10-05, the same
cards as the audit runs, `--provider clm`, the promoted `term.fits` probe deciding): under A6 the
seven runs made 1,380 decisions, 49 proposals (`taxon` 28, `method` 21; 27 of the 49 NCBITaxon
terms, 13 OBI, 4 GENEPIO, 3 IAO) and 545 abstains in 519 s / 238,973 encoder tokens, the
`term.fits` column groups' aspects being `method` 298, `taxon` 293, `measurement` 2; under A8
(runs `1f13e4db…`, `c23d717b…`, `d6bfbfc8…`, `0ac32a08…`, `813f8149…`, `9d1dd27b…`, `272c1c22…`)
1,496 decisions, 65 proposals (`measurement` 54, `method` 6, `unit` 4, `taxon` 1; ENVO 34, OBI 12,
PATO 11, UO 4, GENEPIO 3, NCBITaxon 1) and 436 abstains in 1,843 s / 160,779 tokens, the aspects
`measurement` 292, `method` 153, `unit` 72, `taxon` 0. The longer wall time is the live OLS
searches the new aspects needed (first asks, not cached); the encoder tokens fell with the
shorter candidate lists. Whether the new proposals are right is what the deferred production
audit would measure (A7's appended note); these counts say only what the pipeline now asks.

**Recomputation.** A scratch rerun of X3 after the SVD fallback (`linear._thin_svd`, commit
`034672e`) reproduced all 20 committed X3 cells exactly (timing diagnostics excluded).

## Contradictions found during exploration and how they were settled

1. HF caches were said to be under the repo tree; they are under `~/.cache/huggingface/hub`, and
   the NVFP4 snapshot has no weights.
2. `--task choice` training loss: `infonce` (default) or `softce`, not "soft-target CE plus
   optional InfoNCE".
3. The live stack (8fbaedf) predates `load_plugins`; neither mesa-anyjev nor mesa-clm is
   loadable there until the bump.
4. torch listing only sm_120 is not the GB10 blocker; vLLM packaging is.
5. 285 (labels.duckdb) vs 303 (unique candidate states): settled in M0 — 18 pairs name five
   GAZ CURIEs that EBI OLS returns with `isRoot: true`, dropped as root terms (see Labels).
6. CLM `confidence` is Jev's formula but not AnyJev's `max(probs)`; mesa-clm computes its own
   (D7).
7. The PyPI wheel lacks `pyarrow`; git HEAD has it; a PyPI install cannot train.
8. Issue #15 gained a comment on 2026-09-28 (the GB10 report above); no upstream commit since
   `bb42c6c`, so the zero-shot findings stand.

## Unverified, scheduled

- M1 (settled above, `bench/results/2026-09-29/serving_m1.json` and the 2026-10-01 records):
  vLLM 0.27.1 pooling on sm_121 (works; no NGC image needed), `VLLM_API_KEY` through
  `--env-file` (works; the bearer guard of DESIGN A4 now covers every route), prefix caching off
  (logged), `/tokenize` under `--runner pooling` (works), the `:ro` HF mount offline (the unit
  runs from it), `weights_only=True` with the released head (loads), CPU head latency (110 ms for
  a new 371-token state, 1 ms when cached) and clm-serve RSS (1.20 GiB). Still open: whether GB10
  page cache counts toward vLLM's free-memory check (start-up reported 105.2 of 121.69 GiB free
  with MemAvailable 107.8 GiB, which does not separate the two).
- M0 (settled above): the 285/303 reconciliation (five GAZ root terms) and kilometer's UO id
  (UO:0010066; UO:0000009 is kilogram). The AnyJev per-item L2 dump ran in M1-A's GPU window
  and reproduces the committed cells (see Baselines).
- M4 (settled in `design/m4-analysis-plan.md` B3.3, pre-run): teacher target resolution is column →
  every table card of the product that carries the column; a column found in no card, or an item
  without a column, is dropped and counted (no dataset-scope state is synthesized).
- M7: the local `--data`/`--workflow` directory layout `finetune.py` expects.
- Whether the released head was trained at max_len 8192 or 2048 is not stated upstream.
