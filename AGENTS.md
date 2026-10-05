# Agent guide — mesa-clm

This repository holds **mesa-clm** (package `mesa_clm`, script `mesa-clm`, MCP tools
`mesa_clm_*`, tests under `tests/`) and its documentation site, built with
[Zensical](https://zensical.org) from `docs/`, an **Open Knowledge Format (OKF) v0.2** bundle:
every content page carries YAML frontmatter (`type`, `title`, `description`, `tags`,
`generated`, `sources`, optional `status`/`stale_after`); section `index.md` files carry none;
`docs/log.md` is the dated log. The serving side (`serving/`, `deploy/`) is a separate venv that
never imports `mesa_clm`.

## Ground rules (adopted from AnyJev's AGENTS.md through mesa-anyjev, adapted to CLM)

1. **No fabricated numbers, ever.** Bench numbers come from committed `bench/results/<date>/`
   JSON; every number in prose names its JSON. If a run did not happen, the cell is empty.
   Numbers RESEARCH.md marks *to reproduce* are not cited until `bench` reproduces them.
2. **No hidden generation in decision mode.** A decision is a softmax over scaled cosines from
   CLM (or a local probe over the same embeddings), never sampled text. Any code path that
   samples tokens to answer a question is a bug. Planners generate; deciders read probabilities.
3. **Level and calibration are mandatory.** Every decision record carries `level` (`none`,
   `zero_shot`, `calibrated`, `probe`, `head`) and `calibration` (`none`, `uncalibrated`,
   `platt`, `temperature`), and `_honest` (from M1) refuses records that violate the DESIGN §4.6
   invariants. `auto` is a request, never a stored level. `confidence` is `max(probs)`
   computed here; CLM's own field is only recorded as `clm_confidence`.
4. **Thresholds cite nested LOCO cells.** A numeric write threshold names a pre-registered,
   nested-selection, fingerprint-matched leave-one-card-out cell that beats the lookup baseline
   on novel keys and clears the Clopper–Pearson risk bound; `tests/unit/test_policy_citations.py`
   (from M4) refuses anything else. Selections made on the full data are `exploratory:true` and
   cannot be cited.
5. **The planner never decides.** The planner (static, Claude, gateway) proposes ontologies,
   columns and queries; Claude is a recorded second opinion without a probability; neither can
   produce `auto` or write.
6. **Data stays on the host.** Cards are streamed into memory and never stored; the encoder and
   clm-serve bind loopback with keys; a remote URL needs `MESA_CLM_CLM__ALLOW_REMOTE=1` and
   https; there is no hosted provider. Raw cards are never sent to the encoder; parse errors
   carry line numbers, never content.
7. **Fingerprints must match.** `encoder_fp`, `clm_model_fp`, `schema_sha256` and
   `serving_lock_sha` are stamped everywhere and compared exactly; a mismatch fails the doctor
   and the provider refuses until the task is re-benched (K4). Vendored files stay
   byte-identical (`vendored.sha256`).
8. **Licenses are checked** before any dataset or third-party code is used, and recorded in
   `THIRD_PARTY.md`.
9. **Labels are honest.** Curator labels come only from MRTR elicitation or the interactive
   CLI; an agent's tool pick is `agent_pick` at weight 0; teacher labels are down-weighted and
   never in a test fold; the bench reads only frozen snapshots and records `labels_sha256`.
10. **Git is shared.** Commit on feature branches, Conventional Commits, PRs; never amend a
    shared branch; merge only after `scripts/wait_for_checks.sh`.

## Definition of done

Code + tests + docstring + a `CHANGELOG.md` line; a `DESIGN.md` amendment for any new or
changed decision (the register is append-only); a `RESEARCH.md` entry with its source and
`stale_after` for any new fact the code relies on; a `docs/log.md` entry for any docs change;
`framings.lock.json` updated (with a key-rotation note) whenever a framing's view, template,
anchor or options change (from M1); `vendored.sha256` untouched; `ruff`, `ruff format --check`,
`mypy --strict`, `pytest -q` and `sha256sum -c vendored.sha256` green; nothing that imports
`anyjev` or `mesa_anyjev` under `src/`; no secret in a log, a fingerprint or a commit.

## Documentation commands

```bash
python scripts/okf_validate.py docs           # OKF conformance (CI-enforced; 0 errors)
python scripts/gen_llms_txt.py                # regenerate docs/llms.txt + docs/llms-full.txt
uv run zensical build --clean --strict        # static site -> site/
python scripts/postbuild_agent_surface.py site
```

Editing rules are those of neon-mcp's AGENTS.md: frontmatter on every content page, `index.md`
listings without frontmatter, a dated `docs/log.md` entry per change, generated files
regenerated and committed, new pages added to `zensical.toml`'s `nav`, future-milestone
features marked as planned, never add `verified:` yourself.
