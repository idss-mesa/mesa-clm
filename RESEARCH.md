# RESEARCH.md — verified facts (2026-09-29)

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
  Whether `/tokenize` works under `--runner pooling` is unverified (M1).
- vLLM's startup check compares `gpu_memory_utilization × total` against *currently free*
  memory (a CARC spec note: 0.40 of 121.7 GiB failed with 42 GiB free). Whether GB10 page cache
  counts as used there is unverified (M1).
- `/v1/embeddings` requests default to `add_special_tokens=True`
  (`vllm/entrypoints/pooling/base/protocol.py@v0.30.0` L202-203), which adds nothing for Qwen3
  (below).
- PyPI vLLM 0.30.0 (2026-09-22) ships `manylinux_2_28_aarch64` and pins `torch==2.13.0`; the
  mesa-anyjev venv has torch 2.14.0, so vLLM can never share a venv with it. mesa-clm uses the
  container image instead (U3) and the serve venv has no vllm (plan §3).

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
- CARC co-tenancy: `/opt/carc/carc-agents/models/*.yaml` (2026-09-09) pins four vLLM backends
  to this host at `gpu_memory_utilization` 0.05 + 0.12 + 0.30 + 0.36 = **0.83**; all four
  stopped 2026-09-23 15:28 MDT and `carc-vllm@.service` is disabled, but the specs still name
  sparky-1. If they return, about 7.7 GiB remain (arithmetic), not enough for a bf16 Qwen3-8B
  encoder: hence the 0.20 share, `mesa-clm-check-headroom`, and the written allocation request
  (M1). The CARC gateway itself serves no bf16 Qwen3-8B pooling model (`carc-embed` is
  Qwen3-Embedding-0.6B; `carc-fast` is NVFP4 and generative), so it cannot back CLM.
- GPU budget at 0.20: 24.3 GiB (weights ≈15–16.4 GiB, KV 8×4096×144 KiB = 4.5 GiB). clm-serve
  on CPU: unpatched worst case ≈4–5 GiB (200k-entry LRU ≈3.3 GB + 512 MiB arena + torch CPU +
  heads); with `CLM_EMB_CACHE_SIZE=20000` the LRU is ≈0.33 GB. RSS to be measured in M1.

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

## Baselines (`stale_after: 2027-03-31`)

- **AnyJev L2, leave-one-card-out over 7 cards** — mesa-anyjev
  `bench/results/2026-09-25/Qwen__Qwen3-8B.hf.json` (repo `6159281`, file sha256
  `74c7ad5035293e44a4800a045afa6ca1fcee3a692f8f46d416b0c0aa715d9824`; `environment.model
  Qwen/Qwen3-8B`, bf16 on this GB10, `questions_lock_sha 0190586d…`): `neon_term_fits.L2` acc
  **0.765**, ECE **0.058**, cov@5% **0.088**, cov@10% 0.396, NLL 0.499, Brier 0.327, macro-F1
  0.708, n 285, n_neg 199, 7 folds; `neon_ontology_fits.L2` acc **0.837**, ECE **0.078**,
  cov@5% **0.547**, cov@10% 0.768, NLL 0.375, n 190, n_neg 114, 7 folds. `neon_annotate.L2`
  passed the fold guards in only 1 of 7 folds (n 17, n_neg 5; class counts 63/35), so
  column.annotate has no citable AnyJev cell. These are the K2(a) comparison targets; without
  the per-item dump (`scripts/anyjev_l2_predictions.py`, feasibility read in M0) the comparison
  is unpaired.
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

- M1: vLLM pooling on sm_121 in v0.27.1 (else the NGC digest); `VLLM_API_KEY` through
  `--env-file`; `--no-enable-prefix-caching` accepted and logged; `/tokenize` under `--runner
  pooling`; the `:ro` HF mount offline; `weights_only=True` with the released head; CPU head
  latency; clm-serve RSS; whether GB10 page cache counts toward vLLM's free-memory check.
- M0 (settled above): the 285/303 reconciliation (five GAZ root terms) and kilometer's UO id
  (UO:0010066; UO:0000009 is kilogram). The AnyJev per-item L2 dump is feasible
  (`scripts/anyjev_l2_predictions.py`: mesa-anyjev's `_fit_eval_loco` builds the per-fold
  `DecisionRecord` list before pooling, so the script keeps it; ≈12–15 min on the GB10 with
  vLLM stopped) and runs in M1-A's GPU window; it has not been run.
- M4: teacher target resolution (column → table card, else dataset scope).
- M7: the local `--data`/`--workflow` directory layout `finetune.py` expects.
- Whether the released head was trained at max_len 8192 or 2048 is not stated upstream.
