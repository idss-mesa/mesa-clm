---
title: "Serving runbook"
description: "Bring up, check, rotate, pause and remove the mesa-clm serving stack on the GPU host: the bootstrap, the bearer keys, the systemd user units (installed, never enabled by default), start and stop order, health checks, the GPU budget next to CARC, the fallback encoder and rollback."
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
  at: "2026-09-29T20:00:00Z"
sources:
  - id: plan
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/design/plan-2026-09-28.md"
    title: "mesa-clm implementation plan (v1.1, 2026-09-28), section 6"
    author: "team:idss-mesa"
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md), D5, D15-D17"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md): CLM upstream, vLLM pooling, host facts"
    author: "team:idss-mesa"
  - id: serving-readme
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/serving/README.md"
    title: "serving/ README: patch series, requirements, head export, fallback"
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
processes run on the GPU host (sparky-1, a GB10 with 121 GiB of unified memory), both on
loopback and both behind bearer keys:

| Unit | Port | What |
|---|---|---|
| `mesa-clm-encoder.service` | `127.0.0.1:8090` | `vllm/vllm-openai` v0.27.1, pinned by digest, serving `Qwen/Qwen3-8B` bf16 with last-token pooling, offline, `--gpu-memory-utilization 0.20` |
| `mesa-clm-serve.service` | `127.0.0.1:8700` | the patched `clm-serve` (CLM `bb42c6c5` + patches 0001–0006) in `~/.mesa/clm/serve/.venv`, heads on CPU, the pinned head plus promoted heads |
| `mesa-clm-headroom.timer` | — | every 5 minutes, fails when `MemAvailable` drops below 8 GiB |

Every pin lives in `serving/serving.lock.json`; `deploy/` holds the scripts and the units;
`serving/README.md` documents the patch series, the requirements and the head export. None of
the commands below prints a key, and nothing here enables a unit at boot.

## Layout under `~/.mesa/clm`

| Path | Written by | Contents |
|---|---|---|
| `serve/CLM`, `serve/.venv`, `serve/patches.applied.json` | bootstrap | patched clone, serve venv, the applied series and its git tree |
| `heads/CLM_v0.1-8B.pt`, `heads/npz/<sha8>.npz`, `heads/served/` | bootstrap; promotion (M4) | the pinned head, its numpy export, promoted heads (clm-serve globs `*.pt` at start) |
| `bin/` | bootstrap | `mesa-clm-encoder-run`, `mesa-clm-wait-http`, `mesa-clm-check-headroom`, `mesa-clm-serve-bootstrap` |
| `serving.lock.json` | bootstrap | the lock the host was built from |
| `secrets/` (0700) | keys step | `clm.key`, `encoder.key` (raw, 0600) and the derived `clm.env`, `encoder.env` (0600) |

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
(75,557,149 bytes) and sha256, exports it to `heads/npz/`, and installs the scripts and the lock.
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
and prints the file paths and the two lines that point the core at the key files, never a key.
Re-running keeps existing keys and re-derives the env files. The core reads the raw key files
(never the env files):

```bash
export MESA_CLM_CLM__API_KEY_FILE=~/.mesa/clm/secrets/clm.key
export MESA_CLM_ENCODER__API_KEY_FILE=~/.mesa/clm/secrets/encoder.key
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
systemd-analyze --user verify ~/.config/systemd/user/mesa-clm-*.service ~/.config/systemd/user/mesa-clm-*.timer
```

`deploy/systemd/` is the default rendering of `mesa_clm.serving.render_units()` (a unit test
keeps them identical); `uv run mesa-clm serve units [--home DIR]` prints the rendering for a
serving home, and the units spell the home as `%h`. `systemd-analyze verify` only complains
about missing executables under `%h/.mesa/clm/` until the bootstrap has run. Do **not** run
`systemctl --user enable`: with lingering on, an enabled unit starts at boot, and that waits for
the CARC allocation below.

## 4. Start and stop

Start the encoder first; loading the model takes minutes:

```bash
systemctl --user start mesa-clm-encoder.service   # refuses to start below 32 GiB MemAvailable
systemctl --user start mesa-clm-serve.service     # waits up to 600 s for the encoder's /health
systemctl --user start mesa-clm-headroom.timer    # started, not enabled
```

Starting `mesa-clm-serve.service` alone also starts the encoder (`Requires=`). Stop in the
reverse order; stopping the encoder stops clm-serve as well:

```bash
systemctl --user stop mesa-clm-headroom.timer mesa-clm-serve.service mesa-clm-encoder.service
```

Logs: `journalctl --user -u mesa-clm-encoder.service -u mesa-clm-serve.service -e`. The first
encoder start should log last-token pooling (`seq_pooling_type='LAST'`) with normalisation and
prefix caching disabled; record what it logs in RESEARCH.md (plan §6.1 lists these as M1
verifications).

## 5. Health checks

```bash
# Both ports listen on loopback only.
ss -ltn '( sport = :8090 or sport = :8700 )'
# Open by design on loopback: /health (both), /tokenize and /metrics (encoder).
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/health        # 200
curl -s http://127.0.0.1:8700/health                                          # ok, model names
# Without a key every /v1 route answers 401.
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/v1/models     # 401
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8700/v1/models     # 401
# With a key: the header is read from a process substitution, so the key never appears in argv.
curl -s -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.mesa/clm/secrets/encoder.key)") \
  http://127.0.0.1:8090/v1/models
# The host against the lock: patch files, installed lock copy, clone commit and applied series,
# schema.py in the clone and the serve venv, head size and sha256, image digest and tag, and the
# running container's image and recipe. Exit 1 on any failure; absent pieces fail with --require-live.
uv run mesa-clm serve lock --check --require-live
# Everything above at once (with the key files exported, section 2):
uv run mesa-clm doctor --serve
```

`mesa-clm doctor --serve` runs the lock check and the live probes: loopback-only binds on both
ports (`ss -ltn`), an unauthenticated `GET /v1/models` answered 401 on both, `/health` 200 on
both, the authenticated model lists (`qwen3-8b` on the encoder; `clm-latest` and `clm-raw` on
clm-serve), and one golden `/v1/systemone` call whose answer must cover every key with
probabilities above 0 summing to 1. It also reports `host` (`MemAvailable`) and `gpu_budget`
(active CARC backends). The doctor runs these probes automatically whenever both units are
active; an unreachable endpoint then fails unless `--quick` is given (a warning), and always
fails under `--serve`. The golden encoder vectors, the 5,000-token tail probe and the
clm-serve-vs-local head parity are not doctor checks: they are the M1 track A measurements of
`scripts/serving_probes.py`. Any lock mismatch fails the doctor and the provider refuses until
the task is re-benched (K4).

## 6. GPU budget and CARC coexistence

The encoder takes a fixed 0.20 of the GPU's view of memory: about 24.3 GiB of 121.7 GiB (weights
about 16 GiB, KV cache 8 sequences × 4096 tokens × 144 KiB = 4.5 GiB). clm-serve runs on the CPU
from the same unified memory: its host cache is capped at 20,000 vectors (about 0.33 GB) by
`CLM_EMB_CACHE_SIZE`, plus a 512 MiB vector arena, torch and the heads; its RSS is an M1
measurement. `nvidia-smi` reports memory as "Not Supported" on this GPU, so footprints are the
`/proc/meminfo` `MemAvailable` difference before and after a start, plus
`sg docker -c "docker stats --no-stream mesa-clm-encoder"`.

CARC's deployed plan pins four vLLM backends to this host at 0.83 of the GPU; they have been
stopped since 2026-09-23, and if they return about 7.7 GiB would remain, too little for the
encoder. Hence the 32 GiB start check, the 8 GiB timer, and this rule: **if CARC's backends come
back** (the headroom timer fails, `systemctl list-units 'carc-vllm@*' --state=active` lists
them, or the doctor reports `gpu_budget`), stop both mesa-clm units (section 4) and leave them
stopped until the allocation is agreed. While they are down the MCP tools answer
`decider_unavailable`; the bench is unaffected because features are cached. The units are
enabled at boot only after the written 0.20 allocation request (an M1 task) is accepted.

## 7. Fallback encoder

If the container cannot serve pooling on this GPU (K0 step 1 then 2), the in-process fallback
serves both ports from one process. Stop both units first (they hold the same ports and the
fallback never runs next to vLLM), then:

```bash
systemd-run --user --unit mesa-clm-fallback -p EnvironmentFile=$HOME/.mesa/clm/secrets/clm.env \
  ~/.mesa/clm/serve/.venv/bin/python "$PWD/serving/fallback_serve.py"
systemctl --user stop mesa-clm-fallback    # when done
```

Its `route: transformers` is a different `encoder_fp` (DESIGN D5, D16): nothing it produces is
shared with the vLLM route until a sequential parity run (dump vLLM vectors, stop the units, run
the fallback, compare: minimum cosine ≥ 0.999, mean ≥ 0.9999, probe agreement ≥ 99%) is
recorded. `--device cpu --no-clm` measures CPU bf16 throughput for K0 step 3b.

## 8. Rollback and removal

Pause: stop the units (section 4). To go back to an earlier lock, check out the earlier mesa-clm
commit, move `~/.mesa/clm/serve/CLM` aside (the bootstrap refuses a clone patched with another
series), re-run the bootstrap and restart both units.

Remove everything mesa-clm installed for serving:

```bash
systemctl --user stop mesa-clm-headroom.timer mesa-clm-serve.service mesa-clm-encoder.service
systemctl --user disable mesa-clm-headroom.timer mesa-clm-serve.service mesa-clm-encoder.service 2>/dev/null || true
rm -f ~/.config/systemd/user/mesa-clm-*.service ~/.config/systemd/user/mesa-clm-*.timer
systemctl --user daemon-reload
sg docker -c "docker rm -f mesa-clm-encoder" 2>/dev/null || true
rm -rf ~/.mesa/clm/serve ~/.mesa/clm/bin ~/.mesa/clm/serving.lock.json
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
| clm-serve times out waiting for :8090 | the model is still loading or the container failed: `journalctl --user -u mesa-clm-encoder.service` |
| clm-serve 502 on `/v1/systemone` | the encoder rejected clm-serve's key: `clm.env` and `encoder.env` disagree; re-run the keys step without `rotate`, then restart both units |
| `verify_serving_lock` reports `applied patches` | the clone was edited after the bootstrap, or the lock changed: move the clone aside and re-run the bootstrap |
