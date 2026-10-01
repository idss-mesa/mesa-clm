---
title: "Serving runbook"
description: "Bring up, check, rotate, pause and remove the mesa-clm serving stack on the GPU host: the bootstrap, the bearer keys and the encoder's bearer guard, the network-less encoder container behind a loopback socket unit, the systemd user units (installed, never enabled by default), start and stop order, health checks, batch-invariant encoding, the GPU budget and the CARC allocation request, the fallback encoder and rollback."
type: Guide
tags:
  - deploy
  - serving
  - runbook
  - systemd
  - vllm
  - clm
generated:
  by: "claude/fable-5.1"
  at: "2026-10-01T18:00:00Z"
sources:
  - id: plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/plan-2026-09-28.md"
    title: "mesa-clm implementation plan (v1.1, 2026-09-28), section 6"
    author: "team:idss-mesa"
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md), D5, D15-D17, amendments A3, A4 and A5"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md): CLM upstream, vLLM pooling, host facts"
    author: "team:idss-mesa"
  - id: serving-readme
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/serving/README.md"
    title: "serving/ README: patch series, requirements, head export, fallback"
    author: "team:idss-mesa"
  - id: m1b
    resource: "https://github.com/idss-mesa/mesa-clm/tree/main/bench/results/2026-10-01"
    title: "M1-B measurements: serving_m1b, serving_m1c, batch_invariance, fallback_parity, annotate_latency (2026-10-01)"
    author: "team:idss-mesa"
  - id: clm
    resource: "https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
    title: "Contrastive-LM/CLM at bb42c6c5"
    author: "org:Contrastive-LM"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# Serving runbook

The operational side of [Serving](../concepts/serving.md) for milestone M1 (track A). Two
processes run on the GPU host (sparky-1, a GB10 with 121 GiB of unified memory), both reached on
loopback and both behind bearer keys:

| Unit | Port | What |
|---|---|---|
| `mesa-clm-encoder.service` | (a unix socket) | `vllm/vllm-openai` v0.27.1, pinned by digest, serving `Qwen/Qwen3-8B` bf16 with last-token pooling, offline, `--gpu-memory-utilization 0.20` with the KV cache pinned to 4.5 GiB, batch-invariant kernels (`VLLM_BATCH_INVARIANT=1`), every route but `/health` behind the bearer guard `serving/vllm_auth.py`; the container has no network (`--network none`) and listens on `~/.mesa/clm/run/encoder.sock` (DESIGN A5) |
| `mesa-clm-encoder-proxy.socket`, `mesa-clm-encoder-proxy.service` | `127.0.0.1:8090` | the encoder's endpoint: systemd listens on loopback and `systemd-socket-proxyd` forwards each connection to the socket; started with the encoder (`Wants=`) and stopped with it (`PartOf=`) |
| `mesa-clm-serve.service` | `127.0.0.1:8700` | the patched `clm-serve` (CLM `bb42c6c5` + patches 0001–0006) in `~/.mesa/clm/serve/.venv`, heads on CPU, the pinned head plus promoted heads, inputs truncated to 4,095 tokens |
| `mesa-clm-headroom.timer` | — | every 5 minutes, fails when `MemAvailable` drops below 8 GiB; started and stopped with the encoder |

Every pin lives in `serving/serving.lock.json`, including the container recipe (the whole `vllm
serve` argument list, the non-secret environment, the bearer guard's sha256, the clients'
truncation cap and the network mode; DESIGN A3, A4, A5); `deploy/` holds the scripts and the units; `serving/README.md`
documents the patch series, the requirements, the head export and the guard. None of the
commands below prints a key, and nothing here enables a unit at boot.

## Layout under `~/.mesa/clm`

| Path | Written by | Contents |
|---|---|---|
| `serve/CLM`, `serve/.venv`, `serve/patches.applied.json` | bootstrap | patched clone, serve venv, the applied series and its git tree |
| `heads/CLM_v0.1-8B.pt`, `heads/npz/<sha8>.npz`, `heads/served/` | bootstrap; promotion (M4) | the pinned head, its numpy export, promoted heads (clm-serve globs `*.pt` at start) |
| `bin/` | bootstrap | `mesa-clm-encoder-run`, `mesa-clm-wait-http`, `mesa-clm-check-headroom`, `mesa-clm-serve-bootstrap` |
| `serve/vllm-auth/vllm_auth.py` (0700 directory, 0600 file) | bootstrap | the encoder's bearer guard, mounted read-only into the container (DESIGN A4) |
| `serving.lock.json` | bootstrap | the lock the host was built from |
| `secrets/` (0700) | keys step | `clm.key`, `encoder.key` (raw, 0600) and the derived `clm.env`, `encoder.env` (0600) |
| `run/` (0700) | `mesa-clm-encoder-run` | `encoder.sock`, the encoder's API socket (bind-mounted at `/run/mesa-clm` in the container; removed before each start) |

## 1. Bootstrap

Prerequisites, all read-only to check: this user is listed in the `docker` group (every docker
call goes through `sg docker -c …`, because login sessions and the user systemd manager do not
carry the gid); the pinned image is present
(`sg docker -c "docker image inspect vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"`);
the Qwen3-8B snapshot `b968826d…` is in `~/.cache/huggingface/hub` (the container runs offline
with the cache mounted read-only); `uv` and its CPython 3.11.16 (which ships `Python.h`);
`loginctl show-user "$USER" -p Linger` is `yes`.

From a mesa-clm checkout:

```bash
deploy/bin/mesa-clm-serve-bootstrap
```

It re-verifies the lock's `lock_sha`, clones CLM at the pinned commit, asserts the pristine
`schema.py` sha256, applies the patch series in lock order, creates the venv, installs
`serving/requirements-serve.txt` with `--require-hashes` and then the clone with `--no-deps`
(vllm is never installed there), downloads the head at the pinned revision and verifies its size
(75,557,149 bytes) and sha256, exports it to `heads/npz/`, installs the encoder's bearer guard
(`serving/vllm_auth.py`, its sha256 checked against the lock's recipe) into
`~/.mesa/clm/serve/vllm-auth/` with modes 0700/0600, and installs the scripts and the lock. A
running encoder reads the guard only when it starts: restart the units after a bootstrap that
changed it.
Every step is skipped when it is already done and verified, so re-running it is safe; it refuses
(and changes nothing) when the clone carries changes other than the series, or when a head file
of another sha256 is in the way. It never creates keys, installs units or starts anything.

## 2. Keys

Create both keys and derive the two env files:

```bash
uv run mesa-clm serve keys --init
```

It writes the raw keys `~/.mesa/clm/secrets/{clm,encoder}.key` and the units' env files
`clm.env` and `encoder.env`, all 0600 in a 0700 directory (`--secrets-dir` for another place),
and prints the file paths, never a key. Re-running keeps existing keys and re-derives the env
files. The core reads the raw key files (never the env files), and these two by default: no
setting is needed on the serving host. Keys kept elsewhere are named with the key-file settings
(the command prints them for a `--secrets-dir`), and an explicit empty value turns the default
off:

```bash
export MESA_CLM_CLM__API_KEY_FILE=/path/to/clm.key        # only when not ~/.mesa/clm/secrets/clm.key
export MESA_CLM_ENCODER__API_KEY_FILE=/path/to/encoder.key
```

Rotate both keys, then restart both units so the running processes pick them up:

```bash
uv run mesa-clm serve keys --rotate
systemctl --user restart mesa-clm-encoder.service mesa-clm-serve.service
```

`--rotate` replaces the keys and prints that `systemctl --user restart` line; it restarts nothing
itself, and the running units keep the old keys until they are restarted. The library
equivalent is `mesa_clm.serving.init_keys()` (`rotate=True`), which returns paths only.

## 3. Install the units without enabling them

```bash
mkdir -p ~/.config/systemd/user
install -m 0644 deploy/systemd/mesa-clm-* ~/.config/systemd/user/
systemctl --user daemon-reload
systemd-analyze --user verify ~/.config/systemd/user/mesa-clm-*.service \
  ~/.config/systemd/user/mesa-clm-*.timer ~/.config/systemd/user/mesa-clm-*.socket
```

`deploy/systemd/` is the default rendering of `mesa_clm.serving.render_units()` (a unit test
keeps them identical); `uv run mesa-clm serve units [--home DIR]` prints the rendering for a
serving home, and the units spell the home as `%h`. `systemd-analyze verify` only complains
about missing executables under `%h/.mesa/clm/` until the bootstrap has run. Do **not** run
`systemctl --user enable`: with lingering on, an enabled unit starts at boot, and that waits for
the CARC allocation below. The two proxy units have no install section and cannot be enabled at
all (`systemctl --user list-unit-files 'mesa-clm*'` shows them `static`, the others `disabled`).

## 4. Start and stop

Start the encoder first; loading the model takes minutes:

```bash
systemctl --user start mesa-clm-encoder.service   # also starts the :8090 socket and the headroom timer;
                                                  # refuses to start below 32 GiB MemAvailable
systemctl --user start mesa-clm-serve.service     # waits up to 600 s for the encoder's /health
```

Starting `mesa-clm-serve.service` alone also starts the encoder (`Requires=`). Stop in the
reverse order; stopping the encoder stops clm-serve, the socket, the proxy and the timer as well:

```bash
systemctl --user stop mesa-clm-serve.service mesa-clm-encoder.service
```

The encoder is started at most three times an hour (`StartLimitIntervalSec=1h`,
`StartLimitBurst=3`), so a recipe that fails after its model load does not reload it on the
shared GPU every few minutes; after the cause is fixed, `systemctl --user reset-failed
mesa-clm-encoder.service` allows the next start. The run script refuses an `encoder.env` without
a usable `VLLM_API_KEY` line before docker runs (the bearer guard is only resolved after the model
has loaded).

Logs: `journalctl --user -u mesa-clm-encoder.service -u mesa-clm-serve.service -e`. An encoder
start logs last-token pooling (`seq_pooling_type='LAST'`) with normalisation, prefix caching
disabled, `'uds': '/run/mesa-clm/encoder.sock'`, `'kv_cache_memory_bytes': 4831838208`,
`'middleware': ['vllm_auth.require_api_key']` and `'disable_fastapi_docs': True` among the
non-default arguments, `distributed_init_method=tcp://127.0.0.1:…`, "reserved 4.5 GiB memory for
KV Cache", "Starting vLLM server on unix:/run/mesa-clm/encoder.sock" and, because of
`VLLM_BATCH_INVARIANT=1`, an attention-backend list
without FLASHINFER and torch's `preferred_blas_library` warning; the first requests JIT-compile
Triton's `matmul_kernel_persistent` (RESEARCH.md, vLLM pooling). A start that dies with
`vllm_auth: VLLM_API_KEY is empty or unset` means `encoder.env` lacks the key: the guard refuses
to serve without one.

## 5. Health checks

```bash
# Both ports listen on loopback only (8090 belongs to the systemd socket unit; nothing is published
# by docker, and the container has no network of its own).
ss -ltn '( sport = :8090 or sport = :8700 )'
sg docker -c "docker inspect mesa-clm-encoder --format '{{.HostConfig.NetworkMode}}'"   # none
# Only /health answers without a key (both units).
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/health        # 200
curl -s http://127.0.0.1:8700/health                                          # ok, model names
# Without a key every other encoder route answers 401 (DESIGN A4), not only /v1.
for r in /v1/models /metrics /version; do
  curl -s -o /dev/null -w "GET $r %{http_code}\n" "http://127.0.0.1:8090$r"             # 401 each
done
for r in /tokenize /pooling /score; do
  curl -s -o /dev/null -w "POST $r %{http_code}\n" -X POST "http://127.0.0.1:8090$r"    # 401 each
done
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8700/v1/models     # 401
# With a key: the header is read from a process substitution, so the key never appears in argv.
curl -s -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.mesa/clm/secrets/encoder.key)") \
  http://127.0.0.1:8090/v1/models
# The host against the lock: patch files, installed lock copy, clone commit and applied series,
# schema.py in the clone and the serve venv, head size and sha256, image digest and tag, and the
# running container's image and recipe. Exit 1 on any failure; absent pieces fail with --require-live.
uv run mesa-clm serve lock --check --require-live
# Everything above at once (the default key files, section 2):
uv run mesa-clm doctor --serve
```

`mesa-clm doctor --serve` runs the lock check and the live probes: loopback-only binds on both
ports (`ss -ltn`); the encoder container's own network namespace (`/proc/<pid>/net`, which `ss`
on the host cannot see: a listener on a non-loopback address next to an interface besides `lo`
fails); the headroom timer active (a warning otherwise); `/health` 200 on both without a key; the
401 matrix (without a key every
encoder route vLLM registers, 17 besides `/health`, and an unknown path, and clm-serve's
`/v1/models`, `/v1/systemone` and `/v1/rank` answer 401); the authenticated model lists
(`qwen3-8b` on the encoder, checked against the lock's model, window and route; `clm-latest` and
`clm-raw` on clm-serve); one golden `/v1/systemone` call whose answer must cover every key with
probabilities above 0 summing to 1; and the slow probes: the encoder goldens of
`encoder_golden_<encoder_fp>.npz` (the serving home's `goldens/` or the checkout's
`.local/serving/`, written by `scripts/serving_probes.py`; bitwise one text per request, within
1 − 1e-6 as one batch), the long-input probe (a text over the window completes at 4,095 tokens,
cosine ≥ 0.9999 with its last 4,095 token ids), the systemone parity against the local route
(≤ 1e-4, `clm-latest` with the head's numpy export and `clm-raw`) and the upstream drift probes
(a warning only). It also reports `host` (`MemAvailable`), `gpu_budget` (active CARC backends),
`permissions` (group or other bits on mesa-clm's files, with the `chmod` that fixes them) and
`feature store` (the live `encoder_fp`'s store: its format and the vector recipe it was built
under, against the live lock's).
The lock check (`mesa-clm serve lock --check`) also verifies the installed guard (owner-only,
the locked sha256) and the running container against the lock's recipe: its whole argument
list, its environment (values read only for the locked names, so the key is never read) and the
guard's read-only mount. The doctor runs the light probes automatically whenever both units are
active (the slow ones too without `--quick`); an unreachable endpoint or a failed probe then fails
unless `--quick` is given (a warning), and always fails under `--serve`; an open route always
fails. Any lock mismatch fails the doctor. The live provider refuses a lock that does not verify,
whose `schema_sha256` is not the vendored `schema.py`, or (when both exist) an installed copy
that differs from the checkout's, and its pre-flight refuses a missing or rejected key, an
unserved model and an encoder that is not the lock's; an artifact or bench cell under another
fingerprint is refused until the task is re-benched (K4). `uv run python scripts/doctor_record.py
--out bench/results/<date>/doctor_serve.json` records a run (home and host names shortened, the
keys checked absent); the 2026-10-01 records are `bench/results/2026-10-01/doctor_serve.md`
(A3/A4) and `doctor_serve_m1c.md` (A5: 34 ok, the drift warning, no failure).

The full live record is `scripts/serving_probes.py` (it reads the keys from their files inside
the process; pass the `MemAvailable` readings of the restart for the footprint):

```bash
uv run python scripts/serving_probes.py --mem-pre-start <GiB, both stopped> --mem-encoder-only <GiB> \
  --out bench/results/<date>/serving_<tag>.json --golden-out .local/serving/<tag>/encoder_golden_<encoder_fp>.npz
```

It enumerates the encoder's real route table (vLLM's `build_app` for the locked arguments, in a
throwaway copy of the image without network or model, `serving/vllm_routes.py`), requests every
route with no key, a wrong key, clm-serve's key and the right key, embeds a ~5,000-token text
(it must complete, truncated to its last 4,095 tokens), checks golden determinism (one text per
request, bitwise, and against the existing reference of the same `encoder_fp`) and clm-serve
against the local head route (≤ 1e-4), reads the container's network namespace, and records
latency and footprint; the 2026-10-01 records are `bench/results/2026-10-01/serving_m1b.md`
(A3/A4) and `serving_m1c.md` (A5). Per-card annotate latency on non-bench cards is
`scripts/annotate_latency.py` (`bench/results/2026-10-01/annotate_latency.md`).

## 5a. Batch-invariant encoding

The encoder runs vLLM's batch-invariant kernels (`VLLM_BATCH_INVARIANT=1`; DESIGN A3), so a
text's vector no longer depends on what else is in its batch: clm-serve, the local head route and
the bench embed identically whatever their batching, and clm-serve agrees with the local route
to 3.9e-6 (clm-latest) and 6.6e-5 (clm-raw) in probability on 50 real questions
(`bench/results/2026-10-01/serving_m1b.json`). The mode was adopted by a pre-registered
experiment (`bench/results/2026-10-01/batch_invariance.md`) at 1.16× the M1-A latency of a
32-text batch. It is part of `encoder_fp` (`batch_invariance: kernels`, c3b3d5e1a283): removing
the variable, or any other recipe change, makes the lock check fail and rotates the fingerprint.
Inputs of exactly 4,096 tokens never complete on this image, so every client sends
`truncate_prompt_tokens` 4,095 (`EncoderClient`, clm-serve's `--max-tokens 4095`; DESIGN A4).

## 6. GPU budget and CARC coexistence

The encoder is budgeted 0.20 of the GPU's view of memory: about 24.3 GiB of 121.7 GiB (weights
about 14.1 GiB as loaded, KV cache 8 sequences × 4096 tokens × 144 KiB = 4.5 GiB). vLLM's start-up
check still requires that share free, but the KV cache is pinned to the 4.5 GiB those 8
sequences need (`--kv-cache-memory-bytes 4831838208`, DESIGN A5); without the pin vLLM filled
the share with 9.38-9.87 GiB of KV cache (`bench/results/2026-10-01/encoder_logs.json`), which put the process at 25,153 MiB by `nvidia-smi`
(`bench/results/2026-10-01/serving_m1b.json`). With it the encoder measures **19,609 MiB** by
`nvidia-smi` and **21.45 GiB** by the `MemAvailable` difference (107.46 GiB with both units
stopped, 86.01 with the encoder alone; `bench/results/2026-10-01/serving_m1c.json`). clm-serve
runs on the CPU from the same unified memory: its host cache is capped at 20,000 vectors (about
0.33 GB) by `CLM_EMB_CACHE_SIZE`, plus a 512 MiB vector arena, torch and the heads; it adds 1.22
GiB (RSS 1.18 GiB), so the stack takes 22.67 GiB of `MemAvailable` in all. `nvidia-smi` reports
total memory as "Not Supported" on this GPU, so footprints are the `/proc/meminfo`
`MemAvailable` difference before and after a start, plus `sg docker -c "docker stats --no-stream
mesa-clm-encoder"`.

CARC's deployed plan pins four vLLM backends to this host at 0.83 of the GPU; they have been
stopped since 2026-09-23, and if they return about 7.7 GiB would remain, too little for the
encoder. Hence the 32 GiB start check, the 8 GiB timer, and this rule: **if CARC's backends come
back** (the headroom timer fails, `systemctl list-units 'carc-vllm@*' --state=active` lists
them, or the doctor reports `gpu_budget`), stop both mesa-clm units (section 4) and leave them
stopped until the allocation is agreed. While they are down the MCP tools answer
`decider_unavailable`; the bench is unaffected because features are cached. The units are
enabled at boot only after the written 0.20 allocation request is accepted.

**The allocation request (draft; not sent).** Sending it needs the user's go-ahead (plan §1). The
text to send to CARC's owner, with the measured totals of the A5 recipe:

> mesa-clm asks for a standing share of sparky-1's GB10 for one encoder: vLLM 0.27.1 serving
> Qwen/Qwen3-8B bf16 (pooling) at `--gpu-memory-utilization 0.20` with the KV cache pinned to
> 4.5 GiB, plus clm-serve on the CPU. Measured on 2026-10-01: the encoder takes 21.45 GiB of
> `MemAvailable` (19,609 MiB by `nvidia-smi`) and clm-serve 1.22 GiB, 22.67 GiB in all; vLLM
> requires 24.34 GiB free to start. Both run as user units on loopback with keys, the encoder in
> a container without network. They stop when `MemAvailable` falls below 8 GiB (checked every 5
> minutes) and do not start below 32 GiB, and they are not enabled at boot until this share is
> agreed. With CARC's deployed plan (0.83 of the GPU) about 7.7 GiB would remain, too little for
> both; the request is for CARC's backends to leave 0.20 (about 24.3 GiB) to mesa-clm, or for a
> schedule of windows in which they do.

The numbers are `bench/results/2026-10-01/serving_m1c.json#/footprint`; the 24.34 GiB start-up
requirement is vLLM's own log line ("Desired GPU memory utilization is (0.2, 24.34 GiB)",
`bench/results/2026-10-01/encoder_logs.json`).

## 7. Fallback encoder

If the container cannot serve pooling on this GPU (K0 step 1 then 2), the in-process fallback
serves both ports from one process, with the same rule on :8090 (only `/health` without the
key) and one sequence per forward pass (`--batch 1`, no padding; DESIGN A3). Stop both units
first (they hold the same ports and the fallback never runs next to vLLM), then:

```bash
systemd-run --user --unit mesa-clm-fallback -p EnvironmentFile=$HOME/.mesa/clm/secrets/clm.env \
  ~/.mesa/clm/serve/.venv/bin/python "$PWD/serving/fallback_serve.py"
systemctl --user stop mesa-clm-fallback    # when done
```

Its `route: transformers` with `batch_invariance: serial` is a different `encoder_fp`
(b171b1ba4536; DESIGN D5, D16, A3). The sequential parity run (`scripts/fallback_parity.py`:
dump the vLLM vectors one text per request, stop the units, run the fallback, compare: minimum
cosine ≥ 0.999, mean ≥ 0.9999) passed at batch 1 against the final recipe: min 0.999260, mean
0.999919 (`bench/results/2026-10-01/fallback_parity.md`; batch 8 failed in M1-A and would still
give 0.998663). Artifacts stay keyed by `encoder_fp`. `--device cpu --no-clm` measures CPU bf16
throughput for K0 step 3b.

## 8. Rollback and removal

Pause: stop the units (section 4). To go back to an earlier lock, check out the earlier mesa-clm
commit, move `~/.mesa/clm/serve/CLM` aside (the bootstrap refuses a clone patched with another
series), re-run the bootstrap and restart both units.

Remove everything mesa-clm installed for serving:

```bash
systemctl --user stop mesa-clm-serve.service mesa-clm-encoder.service
systemctl --user disable mesa-clm-headroom.timer mesa-clm-serve.service mesa-clm-encoder.service 2>/dev/null || true
rm -f ~/.config/systemd/user/mesa-clm-*.service ~/.config/systemd/user/mesa-clm-*.timer \
  ~/.config/systemd/user/mesa-clm-*.socket
systemctl --user daemon-reload
sg docker -c "docker rm -f mesa-clm-encoder" 2>/dev/null || true
rm -rf ~/.mesa/clm/serve ~/.mesa/clm/bin ~/.mesa/clm/run ~/.mesa/clm/serving.lock.json
rm -rf ~/.mesa/clm/heads      # keep heads/served if promoted heads must survive
rm -rf ~/.mesa/clm/secrets    # keys are random; a reinstall creates new ones
```

The container image and the Hugging Face cache are left alone: they predate mesa-clm on this
host. The core's own data under `~/.mesa/clm` (sidecar, labels, features) is not touched by any
of the above.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| encoder unit fails at `ExecStartPre` | `MemAvailable` below 32 GiB: another tenant holds the memory (section 6) |
| `permission denied` on the docker socket | this user is not listed in the `docker` group, or a unit calls docker without `sg` |
| clm-serve times out waiting for :8090 | the model is still loading or the container failed: `journalctl --user -u mesa-clm-encoder.service`; until vLLM has created its socket, connections to :8090 are accepted by the socket unit and closed by the proxy (`journalctl --user -u mesa-clm-encoder-proxy.service`) |
| `systemctl --user start mesa-clm-encoder.service` reports `start-limit-hit` | the encoder failed three times within an hour (DESIGN A5); fix the cause (`journalctl --user -u mesa-clm-encoder.service`), then `systemctl --user reset-failed mesa-clm-encoder.service` |
| encoder exits at once with `encoder.env has no usable VLLM_API_KEY line` | the env file lacks the key or carries characters an env file cannot hold literally: `uv run mesa-clm serve keys --init` (or `--rotate`) |
| the doctor fails `encoder network` | the container runs on a docker network again (a hand-started container, an old unit or run script): re-run the bootstrap and install the units from `deploy/systemd/`, then restart both units |
| clm-serve 502 on `/v1/systemone` | the encoder rejected clm-serve's key: `clm.env` and `encoder.env` disagree; re-run the keys step without `rotate`, then restart both units |
| `verify_serving_lock` reports `applied patches` | the clone was edited after the bootstrap, or the lock changed: move the clone aside and re-run the bootstrap |
| encoder exits at start with `vllm_auth.py is missing` or `VLLM_API_KEY is empty or unset` | the bootstrap has not installed the guard, or `encoder.env` lacks the key: re-run the bootstrap or the keys step |
| `verify_serving_lock` reports `auth module` or `encoder container: env …` | the guard was edited or loosened, or the container runs another recipe (a leftover drop-in, a hand-started container): re-run the bootstrap, remove drop-ins under `~/.config/systemd/user/mesa-clm-encoder.service.d/`, restart both units |
| a client gets 401 from `/tokenize` or `/metrics` | expected: every encoder route but `/health` needs the key (DESIGN A4) |
| doctor warns on the quickstart drift (urgency about 0.8466 against 0.8366 ± 0.01) | the batch-invariant kernels move vectors slightly; a known consequence recorded in DESIGN A3, not upstream drift |
