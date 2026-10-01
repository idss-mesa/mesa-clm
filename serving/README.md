# serving/ — the serve-venv side of mesa-clm

Everything here runs on the serving host in its own venv (`~/.mesa/clm/serve/.venv`, uv CPython
3.11.16, aarch64, CUDA 13.0 driver) or in the vLLM container, and never imports `mesa_clm`
(DESIGN D0; plan §3, §6). The runbook is `docs/deploy/serving.md`; the systemd units and shell
helpers are under `deploy/`; the core's view (keys, units, lock checks) is
`src/mesa_clm/serving.py`.

| Path | What |
|---|---|
| `serving.lock.json` | The pins (plan §6.7): CLM commit, patch series, `schema.py` sha256, head file, encoder spec (incl. `batch_invariance`), image digest, and the container `recipe` (arguments, non-secret environment, the bearer guard's sha256 and mount, the clients' truncation cap, the network mode `none`; DESIGN A3, A4, A5); `encoder_fp` and `lock_sha` from `mesa_clm.clm.fingerprint.sign_lock_body` |
| `vllm_auth.py` | The encoder container's bearer guard: `--middleware vllm_auth.require_api_key`, 401 on every route but `/health` without `Authorization: Bearer <VLLM_API_KEY>` (DESIGN A4) |
| `vllm_routes.py` | Run inside a throwaway copy of the image: vLLM's real route table for the locked arguments and the guard's answer on each route (used by `scripts/serving_probes.py`) |
| `encoder_proxy.py` | The encoder's loopback endpoint (installed as `~/.mesa/clm/bin/mesa-clm-encoder-proxy`, run by `mesa-clm-encoder-proxy.service` under the serve venv's Python, socket-activated on 127.0.0.1:8090): relays only this account's connections (the kernel's socket diagnostics, `sock_diag`, name the client socket's owner; another account's is closed at once), each to `~/.mesa/clm/run/encoder.sock` only when that entry is a socket of this account in an owner-only directory, opened with `O_PATH`/`O_NOFOLLOW` and connected through `/proc/self/fd`, so never through a symlink the container could plant; at most 4,096 connections, idle ones closed after 900 s, refusals logged at most once per 10 s per reason, sandboxed by its unit without namespaces (DESIGN A5); standard library only |
| `patches/0001…0006-*.patch` | The CLM patch series, applied in order with `git apply` to Contrastive-LM/CLM `bb42c6c5` |
| `requirements-serve.in` / `.txt` | The serve venv's requirements and their hash-pinned lock (no vllm) |
| `export_head.py` | `.pt` head checkpoint → `heads/npz/<sha8>.npz` for the torch-free `HeadProjector` |
| `cuda_encoder.py` | `CudaEncoder`: Qwen3-8B last-token embeddings in process (the fallback recipe) |
| `fallback_serve.py` | The fallback route: `Engine(embedder=CudaEncoder)` on :8700 plus a vLLM-compatible :8090 |
| `mypy.ini`, `tests/` | CI configuration and the serve-side tests (`.github/workflows/serving.yml`) |

## The patch series

None touches `src/clm/schema.py` or `src/clm/client.py` (their sha256 are pinned; DESIGN D4).
Each file starts with a provenance header that `git apply` ignores; 0001–0004 carry the pull
request diffs verbatim. Applied to `bb42c6c5` in order, the tracked files form git tree
`f4146fa8b90024b554a5404e0993bd20cc64f64b` (the bootstrap records it in
`~/.mesa/clm/serve/patches.applied.json` and the doctor diffs the clone against it).

| Patch | Upstream | Head sha | Files |
|---|---|---|---|
| 0001 `embedder-truncation-side-left` | PR #6 | `11211fcab1c27e4779323ff646714dbce475568e` | `src/clm/embedder.py` |
| 0002 `server-loopback-default` | PR #11 | `fc4d8d85b41bbe4dc6cf17c6260f7de3c3ff69e8` | `README.md`, `src/clm/server.py` |
| 0003 `torch-load-weights-only` | PR #10 | `cf230d2a5337a98706a9497e96663f9d9f150b33` | `evaluation/bon_eval.py`, `preprocessing/hf_embeddings.py`, `src/clm/heads.py`, `train/finetune.py` |
| 0004 `cache-arena-slot-leak` | PR #23 | `b618c1d8994042bd4cc205f293476e2b295f298c` | `src/clm/cache.py` |
| 0005 `server-embedder-key-and-cache-size` | local | — | `src/clm/server.py` `main()`: `Embedder(..., api_key=os.environ.get("CLM_EMB_API_KEY"), cache_size=int(os.environ.get("CLM_EMB_CACHE_SIZE", "200000")))` |
| 0006 `server-constant-time-auth` | local | — | `src/clm/server.py` `auth()`: `hmac.compare_digest` instead of `!=` |

Check the series against a fresh clone:

```bash
git clone --no-checkout https://github.com/Contrastive-LM/CLM.git /tmp/CLM
git -C /tmp/CLM checkout --detach bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
for p in serving/patches/000*.patch; do git -C /tmp/CLM apply --check "$PWD/$p" && git -C /tmp/CLM apply "$PWD/$p"; done
```

After changing a patch, `vllm_auth.py` or any pin (the container recipe lives in
`mesa_clm.serving`: `ENCODER`, `ENCODER_ARGS`, `ENCODER_ENV`, `TRUNCATE_PROMPT_TOKENS`, and
`deploy/bin/mesa-clm-encoder-run` must pass the same arguments and `-e` flags), re-sign the lock
and commit both; `test_committed_lock_verifies_and_equals_the_build`,
`test_committed_lock_pins_the_container_recipe` and `test_encoder_run_matches_the_lock` fail
until they agree:

```bash
uv run python -c "from mesa_clm import serving; open('serving/serving.lock.json', 'w').write(serving.render_lock('serving/patches'))"
```

## Requirements

`requirements-serve.in` pins `torch==2.14.0` (PyPI's aarch64 wheel is a CUDA 13.0 build:
`cuda-toolkit==13.0.3`, `nvidia-*-cu13`, `triton~=3.8.0`) and `transformers==5.17.0` (the version
Qwen3-8B was verified with on the host), plus contrastive-lm's own runtime requirements without
vllm. The lock is regenerated with:

```bash
uv pip compile serving/requirements-serve.in --python-version 3.11 \
  --python-platform aarch64-unknown-linux-gnu --generate-hashes -o serving/requirements-serve.txt
```

The bootstrap installs it with `--require-hashes`, then the patched clone with `--no-deps
--no-build-isolation -e` (contrastive-lm declares an unconditional `vllm>=0.6`).

## Head export (`export_head.py`)

`python serving/export_head.py CKPT [--out FILE | --out-dir DIR] [--expect-sha256 HEX]` loads the
checkpoint with `torch.load(..., weights_only=True)`, resolves its configuration exactly as CLM's
`HeadPair._load` does, maps `make_head`'s parameters onto the contract below, writes the file
atomically (uncompressed `numpy.savez`), and checks it: a float64 numpy forward pass over the
exported arrays must match CLM's own `make_head` on seeded unit vectors within 1e-5 (the plan's
headproj bound), and the clamped scale must match `HeadPair`'s. An existing export of the same
checkpoint is verified and kept; one of another checkpoint is refused without `--force`.

The npz keys (`mesa_clm.clm.headproj`'s contract, format `mesa-clm-head-npz/1`):

| Key | dtype / shape | Meaning |
|---|---|---|
| `format` | `<U` scalar | `mesa-clm-head-npz/1` |
| `cfg` | `<U` scalar | JSON with exactly `width, depth, activation, layernorm, residual, projection_dim, hidden_size` |
| `logit_scale` | float32 scalar | the checkpoint's raw `logit_scale` (the reader applies `min(exp, 100)`) |
| `source_sha256` | `<U` scalar | sha256 of the `.pt` |
| `<side>.linear.<i>.weight` | float32 `[out, in]` | torch layout; `i` in `0..depth-1` |
| `<side>.linear.<i>.bias` | float32 `[out]` | |
| `<side>.norm.<i>.weight` / `.bias` | float32 `[width]` | only with `layernorm`; `i` in `1..depth-2` |

`<side>` is `state` (`state_head`) or `action` (`action_head`). The mapping from `make_head`
(`heads.py` at `bb42c6c5`): `inp` → `linear.0`; `hidden.<j>` → `linear.<j+1>`; `norms.<j>` →
`norm.<j+1>`; `out` → `linear.<depth-1>`. For the released head (4096 → 1536 → 1536 → 512, GELU,
LayerNorm) that is `linear.0..2` and `norm.1` per side.

## The encoder's bearer guard (`vllm_auth.py`)

vLLM 0.27.1's own key check covers only `/v1`, `/v2`, `/inference` and `/cohere`; the pooling
server also answers `/pooling`, `/invocations`, `/score`, `/rerank`, `/tokenize`, `/detokenize`,
`/metrics`, `/version`, `/load` and `/ping` (`bench/results/2026-09-29/serving_m1.json`). The
guard is an async function middleware (stdlib and starlette only, the container's Python 3.12):
the bootstrap installs it under `~/.mesa/clm/serve/vllm-auth/` (0700, the file 0600, sha256
checked against the lock), `mesa-clm-encoder-run` mounts that directory read-only at
`/opt/mesa-clm-auth`, puts it on `PYTHONPATH` and passes `--middleware
vllm_auth.require_api_key --disable-fastapi-docs`. vLLM wraps the function with FastAPI's
`app.middleware("http")` after its own middleware, so the guard is the outermost layer: every
request but one to exactly `/health` needs `Authorization: Bearer <VLLM_API_KEY>` (scheme
case-insensitive, token compared with `hmac.compare_digest`) and gets `401 {"error":
"Unauthorized"}` otherwise. Resolving `require_api_key` without `VLLM_API_KEY` raises, so the
container refuses to start rather than serve unguarded. `vllm_routes.py` checks it against
vLLM's real route table (19 routes on this image, no WebSocket route;
`bench/results/2026-10-01/serving_m1b.json`).

## Fallback encoder (`fallback_serve.py`, `cuda_encoder.py`)

`CudaEncoder` loads `Qwen/Qwen3-8B` bf16 at `b968826d…` from the local HF cache only, embeds one
sequence per forward pass by default (`batch=1`: no padding, so a vector does not depend on what
else is embedded; batch 8 failed the parity gate in M1-A), truncates token ids from the left to
the limit (`embed`: `max_len - 1`, the vLLM route's cap), reads `last_hidden_state` at the last
token and L2-normalises in float32; it deregisters torch's Triton eager overrides when
`torch._native` exists (mesa-anyjev D19; `--native-triton` keeps them). `fallback_serve.py` runs
the patched `create_app` over `Engine(embedder=CudaEncoder, device="cpu",
action_cache="512MiB")` on `127.0.0.1:8700` and a vLLM-compatible `/v1/embeddings`,
`/v1/models`, `/tokenize`, `/health` on `127.0.0.1:8090`, with `CLM_API_KEY` and the encoder key
(`VLLM_API_KEY` or `CLM_EMB_API_KEY`) compared in constant time on every :8090 route but
`/health`, as the guard does in the container. It never runs next to the vLLM container; its
`route: transformers`, `batch_invariance: serial` is a different `encoder_fp` (b171b1ba4536),
and its parity with the vLLM route passed the two cosine gates at batch 1
(`bench/results/2026-10-01/fallback_parity.md`; the probe-agreement gate waits for M4, so nothing is
shared between the routes yet). `--batch` other than 1 is refused: the fingerprint names one
sequence per forward pass.

## Tests

`serving/tests/` runs in CI with fastapi, httpx, numpy, requests and pytest, and the patched
clone in `MESA_CLM_CLM_SRC` (`test_serve_side.py`: the patched clm-serve and the fallback;
`test_vllm_auth.py`: the guard on a FastAPI app shaped like vLLM's, installed the way vLLM
installs it; `test_vllm_routes.py`: the route-table walk and per-route probe):

```bash
MESA_CLM_CLM_SRC=/tmp/CLM/src uvx --with fastapi --with httpx --with numpy --with requests \
  --with uvicorn --with pytest-asyncio pytest -q -p no:cacheprovider serving/tests
```
