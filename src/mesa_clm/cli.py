"""The ``mesa-clm`` console script (plan §7.2).

Verbs so far: ``labels ingest-neon-eval|import-anyjev|snapshot|stats`` and ``bench
baselines|mde`` (M0); ``doctor [--quick] [--serve] [--json]``; ``framings --check|--update-lock``;
``annotate``, ``explain``, ``review`` and ``feedback`` (the decide phase and its review, M1);
``provenance migrate|export|import|prune``; ``serve keys|units|lock``. ``plan``, ``apply``,
``revert``, ``features``, ``learn``, ``artifacts``, ``history`` and ``neon curate`` land with
their milestones; ``mesa-clm --help`` lists what exists. Ported from mesa-anyjev ``cli.py``
(``6159281``): the same argparse shape, exit codes and lazy imports inside the commands, so
``--help`` and ``doctor`` never import the modules they do not need.

Exit codes: ``EXIT_OK`` 0; ``EXIT_FAIL`` 1 when the verb ran and failed (a doctor ``fail``, an
empty store, a snapshot that no longer matches the store, framing drift, an unreachable
clm-serve, a replay miss, another owner's run); ``EXIT_CONFIG`` 2 for a configuration or usage
problem (an unreadable config file or card, a missing ``--eval-root`` or one without
``results/validated.json`` and ``cards/``, a Postgres DSN for the M0 label verbs). A
configuration validation error is reported as ``config error: <field>: <problem>`` without the
offending value, so a mistyped ``MESA_CLM_<SECTION>__API_KEY`` never echoes the key (``Config``
also hides inputs in its errors), and no verb prints a key: ``serve keys`` names files only.
The global ``--config``, ``--provenance`` and ``--actor`` are read before the verb (``mesa-clm
--provenance DSN labels stats``); ``--provenance`` and ``--actor`` are also accepted after it,
as mesa-anyjev's users type them.

**Owners (D21).** A run belongs to its ``owner``: ``annotate --owner`` (default ``--actor``,
itself ``$USER``). ``explain``, ``review`` and ``feedback`` act as ``--actor`` and refuse another
owner's run. ``review`` and ``feedback`` record ``via='cli'``: curator labels (D21 "the
interactive CLI"); ``review`` without ``--pick``/``--decline`` needs a terminal.

**annotate --provider clm** builds the live provider (:mod:`mesa_clm.providers.live`: loopback
URLs and keys from the ``clm``/``encoder`` sections, the fingerprint of the serving lock the host
runs) and asks clm-serve's ``/health`` first. When it does not answer: tier ``auto`` with
``decider.ols_rank_fallback: true`` runs the degraded ``ols_rank`` method (proposed-only, never
auto, D28; the run says ``degraded``); otherwise, and always for an explicit CLM tier, the verb
refuses with exit 1 (the plugin's ``decider_unavailable``). ``--tier ols_rank`` never needs
clm-serve for the candidate groups. ``--provider fake`` is the deterministic offline fake; it
exercises the pipeline and is never evidence.

``bench baselines`` and ``bench mde`` stamp their cells with the ``labels_sha256`` of a frozen
snapshot (DESIGN D30): ``--snapshot PATH`` names one written by ``labels snapshot``, otherwise
``bench/snapshots/<date>.parquet`` is used and written from the store when missing. An existing
snapshot must still describe the store (same label content hash), or the verb refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import ValidationError

from mesa_clm import __version__
from mesa_clm.config import (
    Config,
    Tier,
    duckdb_path,
    expand_path,
    load_config,
    redact_dsn,
    set_active_config,
)

if TYPE_CHECKING:
    from mesa_clm.service import DecisionService

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2

DEFAULT_OUT_DIR = "bench/results"
DEFAULT_SNAPSHOT_DIR = "bench/snapshots"
# Mirrors bench.stats.DEFAULT_B / DEFAULT_SEED and bench.mde.DEFAULT_N_SIMS (asserted by the
# CLI tests) so building the parser never imports numpy.
DEFAULT_B = 2000
DEFAULT_SEED = 0
DEFAULT_N_SIMS = 200


class UsageError(Exception):
    """A problem ``main`` reports on stderr as ``mesa-clm: <message>`` and turns into ``code``."""

    def __init__(self, message: str, code: int = EXIT_CONFIG) -> None:
        super().__init__(message)
        self.code = code


def _today() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def label_store(cfg: Config, dsn: str | None = None) -> Any:
    """The M0 label store behind ``dsn`` (default: the configured sidecar DSN).

    The labels table lives in the per-host DuckDB sidecar (DESIGN D11), locked through
    ``<dir>/locks/provenance.lock`` (``LabelStore``'s default, the path the M1 sidecar store
    shares); the Postgres dialect lands with the M1 store, so any other DSN is a usage error
    (redacted in the message).
    """
    from mesa_clm.provenance.labels import LabelStore

    chosen = dsn or cfg.provenance.dsn
    path = duckdb_path(chosen)
    if path is None:
        raise UsageError(
            f"{redact_dsn(chosen)}: the M0 label store needs a duckdb:///<path> DSN "
            "(the Postgres dialect lands in M1)"
        )
    return LabelStore(path)


def _ols_client(cfg: Config) -> Any:
    """The OLS client label ingestion resolves terms through: mesa-mcp's live client, wrapped in
    ``RecordingOLS`` unless ``ols.fixtures`` is ``off``; ``replay`` needs no network at all."""
    from mesa_clm.ols import RecordingOLS

    inner: Any = None
    if cfg.ols.fixtures != "replay":
        from mesa_mcp.ols.client import OLSClient

        inner = OLSClient(cfg.ols.base_url)
    if cfg.ols.fixtures == "off":
        return inner
    return RecordingOLS(inner, expand_path(cfg.ols.fixtures_dir), cfg.ols.fixtures)


def _print_report(report: Any) -> int:
    print(json.dumps(report.summary(), indent=1))
    if report.terms_missing:
        missing = sorted(set(report.terms_missing))
        print(
            f"unresolved CURIEs ({len(missing)}): {', '.join(missing[:20])}",
            file=sys.stderr,
        )
    return EXIT_OK


def _require_labels(store: Any) -> int:
    """The number of labels in ``store``; an empty or missing store is ``EXIT_FAIL``."""
    n: int = store.count()
    if n == 0:
        raise UsageError(
            f"no labels in {store.path}; run `mesa-clm labels ingest-neon-eval` first", EXIT_FAIL
        )
    return n


def _cmd_labels(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm.learn import labels as learn

    store = label_store(cfg)
    if args.verb == "ingest-neon-eval":
        root = args.eval_root or cfg.eval_root
        if not root:
            raise UsageError("labels ingest-neon-eval needs --eval-root or MESA_CLM_EVAL_ROOT")
        try:
            learn.check_eval_root(root)  # the doctor's "eval root" rule, before anything opens
        except FileNotFoundError as exc:
            raise UsageError(str(exc)) from exc
        excluded = [m.strip() for m in (args.exclude_models or "").split(",") if m.strip()]
        report = learn.ingest_neon_eval(
            store,
            root,
            learn.TermResolver(_ols_client(cfg)),
            exclude_models=excluded,
            actor=args.actor,
        )
        return _print_report(report)
    if args.verb == "import-anyjev":
        import duckdb

        if not args.dsn:
            raise UsageError("labels import-anyjev needs --dsn <mesa-anyjev sidecar DSN or path>")
        try:
            report = learn.import_anyjev(args.dsn, store, actor=args.actor)
        except FileNotFoundError as exc:
            raise UsageError(str(exc)) from exc
        except duckdb.Error as exc:
            raise UsageError(
                f"{args.dsn}: not a readable mesa-anyjev sidecar: {exc}", EXIT_FAIL
            ) from exc
        return _print_report(report)
    if args.verb == "snapshot":
        n = _require_labels(store)
        out = Path(args.out) if args.out else Path(DEFAULT_SNAPSHOT_DIR) / f"{_today()}.parquet"
        sha = learn.snapshot(store, out)
        print(f"wrote {out} ({n} labels); labels_sha256 {sha}")
        return EXIT_OK
    print(json.dumps(learn.stats(store), indent=1))
    return EXIT_OK


def _frozen_labels(store: Any, args: argparse.Namespace, date: str) -> tuple[str, Path]:
    """``(labels_sha256, path)`` of the snapshot a bench run is stamped with (DESIGN D30).

    ``--snapshot`` names an existing snapshot; otherwise ``bench/snapshots/<date>.parquet`` is
    used and written from the store when it does not exist yet. An existing snapshot must carry
    the store's label content (``labels_content_sha256``), else the store changed since it was
    frozen and the run is refused.
    """
    from mesa_clm.bench.results import labels_content_sha256, snapshot_content_sha256
    from mesa_clm.learn.labels import snapshot

    _require_labels(store)
    path = Path(args.snapshot) if args.snapshot else Path(DEFAULT_SNAPSHOT_DIR) / f"{date}.parquet"
    if not path.exists():
        if args.snapshot:
            raise UsageError(f"--snapshot {path}: no such file (write one with `labels snapshot`)")
        return snapshot(store, path), path
    if snapshot_content_sha256(path) != labels_content_sha256(store):
        raise UsageError(
            f"{path} does not hold the labels of {store.path} (the store changed since the "
            f"snapshot); re-run `mesa-clm labels snapshot --out {path}` or name another --snapshot",
            EXIT_FAIL,
        )
    return hashlib.sha256(path.read_bytes()).hexdigest(), path


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    store = label_store(cfg)
    date = args.date or _today()
    sha, snap = _frozen_labels(store, args, date)
    print(f"labels snapshot {snap}; labels_sha256 {sha}", file=sys.stderr)
    if args.verb == "baselines":
        from mesa_clm.bench.baselines import run_baselines
        from mesa_clm.bench.results import markdown_table, write_results

        results = run_baselines(
            store,
            labels_sha256=sha,
            date=date,
            policy_path=cfg.policy.policy_path,
            B=args.B,
            seed=args.seed,
        )
        path = write_results(results, args.out_dir)
        print(path)
        print(markdown_table(results))
        return EXIT_OK
    from mesa_clm.bench.mde import run_mde, write_mde
    from mesa_clm.bench.tasks.neon import tasks_from_store

    tasks = tasks_from_store(store, policy_path=cfg.policy.policy_path)
    mde = run_mde(tasks, labels_sha256=sha, date=date, n_sims=args.n_sims, B=args.B, seed=args.seed)
    path = write_mde(mde, args.out_dir)
    print(path)
    for c in mde.curves:
        line = (
            f"{c.task:20s} {c.population:10s} {c.metric:6s} n={c.n:4d} "
            f"cards_counting={c.cards_counting} mde_auroc={_fmt(c.mde_auroc)} "
            f"mde_delta={_fmt(c.mde_delta)}"
        )
        if c.not_applicable:
            line += f" ({c.not_applicable})"
        print(line)
    return EXIT_OK


def _cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm.health import doctor

    # --serve forces the live probes; otherwise they run when both serving units are active.
    report = doctor(cfg, quick=args.quick, serve=True if args.serve else None)
    if args.json:
        print(json.dumps(report.as_dict(), indent=1))
    else:
        print("\n".join(report.lines()))
    return EXIT_OK if report.ok else EXIT_FAIL


# -- M1: framings -----------------------------------------------------------------------------


def _cmd_framings(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm import framings

    path = Path(args.lock) if args.lock else framings.LOCK_PATH
    if args.update_lock:
        sha = framings.write_lock(path)
        print(f"wrote {path}; lock_sha {sha}")
        return EXIT_OK
    problems = framings.lock_drift(path)
    if problems:
        print(f"{path.name} does not match the framings in the code:")
        for line in problems:
            print(f"  {line}")
        print(
            "a framing edit rotates its question_key (D1): if it was deliberate, run "
            "`mesa-clm framings --update-lock` and commit the lock"
        )
        return EXIT_FAIL
    print(f"{path.name} in sync; lock_sha {framings.lock_sha()}")
    return EXIT_OK


# -- M1: the decide phase and its review --------------------------------------------------------

_ID_CHARS = frozenset("0123456789abcdef-")
_MIN_PREFIX = 4


def _json_out(data: Any) -> str:
    return json.dumps(data, indent=1, default=str, ensure_ascii=False)


def _open_store(cfg: Config) -> Any:
    """The configured sidecar for the review verbs: an existing DuckDB file (never created
    here) or Postgres (schema ensured)."""
    from mesa_clm.provenance.store import DuckDBStore, open_store

    dsn = cfg.provenance.dsn
    path = duckdb_path(dsn)
    if path is not None:
        if not path.is_file():
            raise UsageError(
                f"no sidecar at {path}; `mesa-clm annotate` records runs there", EXIT_FAIL
            )
        return DuckDBStore(path)
    try:
        return open_store(dsn)
    except ImportError as exc:
        raise UsageError(f"{redact_dsn(dsn)}: {exc} (install the pg extra)") from exc
    except ValueError as exc:
        raise UsageError(f"{redact_dsn(dsn)}: {exc}") from exc


def _reader_service(cfg: Config, store: Any | None = None) -> DecisionService:
    """A :class:`~mesa_clm.service.DecisionService` for the verbs that read runs and record
    picks: no CLM provider is reached (a degraded ``ols_rank`` stand-in over the fake
    fingerprint), OLS replays only, the static planner."""
    from mesa_clm.ols import OLSLayer, RecordingOLS
    from mesa_clm.planner.static_planner import StaticPlanner
    from mesa_clm.policy import Policy
    from mesa_clm.providers.tiered import OlsRankProvider, fake_fingerprint
    from mesa_clm.service import DecisionService

    return DecisionService(
        cfg,
        provider=OlsRankProvider(fake_fingerprint()),
        planner=StaticPlanner(),
        ols=OLSLayer(RecordingOLS(None, expand_path(cfg.ols.fixtures_dir), "replay")),
        policy=Policy.from_config(cfg.policy),
        store=store if store is not None else _open_store(cfg),
    )


def _by_prefix(ids: list[str], text: str, what: str) -> str:
    """The one id in ``ids`` equal to ``text`` or starting with it (at least 4 hex chars)."""
    needle = text.strip().lower()
    if needle in ids:
        return needle
    if len(needle) < _MIN_PREFIX or not set(needle) <= _ID_CHARS:
        raise UsageError(f"{what} {text!r}: not an id or an id prefix of at least 4 hex digits")
    hits = [i for i in ids if i.startswith(needle)]
    if len(hits) != 1:
        raise UsageError(
            f"{what} {text!r}: {'no' if not hits else f'{len(hits)}'} match(es)", EXIT_FAIL
        )
    return hits[0]


def _resolve_run(store: Any, text: str, owner: str | None) -> Any:
    """A run id from ``text`` (a UUID, or a prefix among ``owner``'s runs, or all runs)."""
    from uuid import UUID

    try:
        return UUID(text)
    except ValueError:
        pass
    ids = [str(r["run_id"]) for r in store.runs(owner=owner, limit=None)]
    return UUID(_by_prefix(ids, text, "--run-id"))


def _abstain_counts(abstained: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in abstained:
        reason = str(a.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _load_card_arg(path_text: str) -> Any:
    from mesa_clm.cards import load_card

    path = Path(path_text).expanduser()
    try:
        return load_card(path)
    except FileNotFoundError as exc:
        raise UsageError(f"--card {path}: no such file") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"--card {path}: cannot read ({type(exc).__name__})") from exc
    except ValueError as exc:  # the card grammar; messages carry line numbers, not content
        raise UsageError(f"--card {path}: {exc}") from exc


def _annotate_provider(
    cfg: Config, args: argparse.Namespace, tier: str
) -> tuple[Any, str, list[str], Any]:
    """``(provider, effective tier, notes, close)`` for ``--provider`` (module docstring)."""
    if args.provider == "fake":
        from mesa_clm.providers.tiered import FakeProvider

        return (
            FakeProvider(seed=args.fake_seed),
            tier,
            [f"provider fake (seed {args.fake_seed}): deterministic, never evidence"],
            lambda: None,
        )
    from mesa_clm.net import EndpointError
    from mesa_clm.providers import live
    from mesa_clm.secrets import SecretError

    try:
        stack = live.clm_provider(cfg)
    except live.ProviderSetupError as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    except (EndpointError, SecretError) as exc:
        raise UsageError(str(exc)) from exc
    notes = [f"serving lock {stack.lock_path} (lock_sha {stack.lock.lock_sha[:12]})"]
    if tier != "ols_rank":
        answering, detail = live.clm_status(stack.client)
        if answering:
            notes.append(detail)
        elif tier == "auto" and cfg.decider.ols_rank_fallback:
            notes.append(f"{detail}: degraded to ols_rank (decider.ols_rank_fallback, D28)")
            tier = "ols_rank"
        else:
            stack.close()
            raise UsageError(
                f"{detail}; start the serving units (docs/deploy/serving.md), or run "
                "--tier ols_rank (proposed-only, D28), or set decider.ols_rank_fallback for "
                "tier auto, or use --provider fake",
                EXIT_FAIL,
            )
    return stack.provider, tier, notes, stack.close


def _check_tier(provider: Any, tier: str) -> None:
    """Refuse an explicit learned tier the provider cannot serve before any request is spent."""
    if tier not in ("calibrated", "probe", "head"):
        return
    from mesa_clm.framings import FRAMINGS

    missing = [t for t in FRAMINGS if not provider.supports_tier(t, tier)]
    if missing:
        raise UsageError(
            f"tier {tier} is not servable for {', '.join(missing)}: no promoted artifacts "
            "(calibrated/probe arrive with M4, head with M7); use --tier auto or zero_shot",
            EXIT_FAIL,
        )


def _cmd_annotate(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm.ols import ReplayMiss
    from mesa_clm.providers.base import TierUnavailable
    from mesa_clm.service import DeciderBusy, DecisionService

    card = _load_card_arg(args.card)
    owner = args.owner or args.actor
    tier = args.tier or cfg.decider.tier
    provider, tier, notes, close = _annotate_provider(cfg, args, tier)
    try:
        _check_tier(provider, tier)
        svc = DecisionService.from_config(cfg, provider, planner_kind=args.planner)
        try:
            run = svc.annotate(card, args.actor, owner=owner, tier=tier)
        except DeciderBusy as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
        except ReplayMiss as exc:
            raise UsageError(
                f"OLS replay miss ({exc}); the run was recorded with status failed. Record the "
                "fixture (ols.fixtures=record) or run with ols.fixtures=auto",
                EXIT_FAIL,
            ) from exc
        except TierUnavailable as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
    finally:
        close()
    out = run.to_eval_result() if args.eval_result else run.to_dict()
    if not args.eval_result:
        out["tier"] = tier
        out["next_step"] = f"mesa-clm review --run-id {run.run_id}"
    if args.out == "-":
        print(_json_out(out))
    elif args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(_json_out(out) + "\n", encoding="utf-8")
    report = sys.stderr if args.out == "-" else sys.stdout
    fp = run.fingerprint
    lines = [
        f"run {run.run_id} (owner {run.owner}, card {run.card.name}, tier {tier})",
        f"  {len(run.proposals)} proposals, {len(run.abstained)} abstained, "
        f"{run.n_decisions} decisions, {run.n_calls} CLM calls, {run.input_tokens} input "
        f"tokens, {run.seconds:.2f} s{', degraded' if run.degraded else ''}",
        "  outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(run.outcomes.items())),
        "  abstained: "
        + (", ".join(f"{k}={v}" for k, v in _abstain_counts(run.abstained).items()) or "none"),
        f"  fingerprint: encoder_fp {fp.get('encoder_fp')} clm_model_fp {fp.get('clm_model_fp')} "
        f"serving_lock_sha {str(fp.get('serving_lock_sha'))[:12]}",
        *(f"  {n}" for n in notes),
    ]
    if args.out and args.out != "-":
        lines.append(f"  wrote {args.out}")
    lines.append(f"next: mesa-clm review --run-id {run.run_id}")
    print("\n".join(lines), file=report)
    return EXIT_OK


def _cmd_explain(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm.service import NotOwner

    svc = _reader_service(cfg)
    owner = args.actor
    run_id = _resolve_run(svc.store, args.run_id, owner) if args.run_id else None
    try:
        out = svc.explain(owner=owner, run_id=run_id, irods_path=args.irods_path, limit=args.limit)
    except NotOwner as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    except KeyError as exc:
        raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    print(_json_out(out))
    return EXIT_OK


def _fmt_p(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _describe_group(g: dict[str, Any], i: int, n: int) -> str:
    if g.get("column_name"):
        target = f"column {g['column_name']}"
    elif g.get("site_code"):
        target = f"site {g['site_code']}"
    else:
        target = str(g.get("scope") or "dataset")
    state = "anchor won" if g.get("anchor_won") else str(g.get("outcome"))
    return (
        f"[{i}/{n}] {g.get('task_id')} {target}  aspect {g.get('aspect') or '-'}  "
        f"ontology {g.get('ontology_id') or '-'}  ({state}, top p_fit {_fmt_p(g.get('top_p_fit'))})"
    )


def _candidate_lines(cands: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Printable candidate lines (the anchor as 0) and the option keys by number (1-based)."""
    lines: list[str] = []
    keys: list[str] = []
    anchor = next((c for c in cands if c["is_anchor"]), None)
    for c in cands:
        if c["is_anchor"]:
            continue
        keys.append(str(c["option_key"]))
        link = f"  [{c['write_status']} link]" if c.get("write_status") else ""
        lines.append(
            f"  {len(keys)}. {c.get('label') or c['option_key']} ({c['option_key']})  "
            f"p_fit {_fmt_p(c.get('p_fit'))}  rank {c.get('rank') if c.get('rank') else '-'}"
            f"{link}"
        )
    p_anchor = _fmt_p(anchor.get("p_fit")) if anchor else "-"
    lines.append(f"  0. none of these (the anchor)  p_fit {p_anchor}")
    return lines, keys


def review_interactive(
    svc: Any,
    pending: list[dict[str, Any]],
    *,
    owner: str,
    actor: str,
    ask: Any,
    say: Any,
) -> dict[str, int]:
    """Walk the pending groups: show each group's offered candidates with ``p_fit`` and the
    anchor, record the answer through ``record_human_pick(via='cli')``. ``ask(prompt) -> str``
    and ``say(text)`` are the terminal (``input``/``print``), injectable for tests. Answers: a
    number picks that candidate, ``0``/``n`` is "none of these", ``d`` declines, ``s`` or empty
    skips, ``q`` stops."""
    from uuid import UUID

    from mesa_clm.registry import ANCHOR_KEY

    done = {"picked": 0, "none": 0, "declined": 0, "skipped": 0}
    for i, g in enumerate(pending, 1):
        cands = svc.candidates_for_group(UUID(g["group_id"]))
        say(_describe_group(g, i, len(pending)))
        lines, keys = _candidate_lines(cands)
        for line in lines:
            say(line)
        while True:
            answer = ask(
                f"pick 1-{len(keys)}, 0 = none of these, d = decline, s = skip, q = quit [s]: "
            )
            answer = answer.strip().lower()
            if answer in ("", "s"):
                done["skipped"] += 1
                break
            if answer == "q":
                return done
            if answer == "d":
                res = svc.record_human_pick(
                    UUID(g["group_id"]), actor, via="cli", owner=owner, action="decline"
                )
                done["declined"] += 1
            elif answer in ("0", "n"):
                res = svc.record_human_pick(
                    UUID(g["group_id"]), actor, via="cli", owner=owner, option_key=ANCHOR_KEY
                )
                done["none"] += 1
            elif answer.isdigit() and 1 <= int(answer) <= len(keys):
                res = svc.record_human_pick(
                    UUID(g["group_id"]),
                    actor,
                    via="cli",
                    owner=owner,
                    option_key=keys[int(answer) - 1],
                )
                done["picked"] += 1
            else:
                say(f"  '{answer}' is not an answer")
                continue
            say(
                f"  recorded {res['action']}: group {res['outcome']}, "
                f"{res['labels_written']} label(s) {res['label_source'] or ''}".rstrip()
            )
            break
    return done


def _cmd_review(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_clm.registry import ANCHOR_KEY
    from mesa_clm.service import NotOwner

    svc = _reader_service(cfg)
    owner = args.actor
    run_id = _resolve_run(svc.store, args.run_id, owner)
    try:
        summary = svc.run_summary(run_id, owner=owner)
    except NotOwner as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    except KeyError as exc:
        raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
    if not (args.pick or args.decline):
        if not sys.stdin.isatty():
            raise UsageError(
                "review is interactive (it needs a terminal); from a script use "
                "--pick GROUP=OPTION_KEY|none and --decline GROUP"
            )
        pending = summary["pending"]
        if not pending:
            print(f"run {run_id}: nothing waits for a reviewer")
            return EXIT_OK
        print(
            f"run {run_id}: {len(pending)} group(s) wait for a reviewer (answers are curator labels)"
        )
        done = review_interactive(svc, pending, owner=owner, actor=args.actor, ask=input, say=print)
        print(", ".join(f"{k} {v}" for k, v in done.items()))
        return EXIT_OK
    # Every answer is resolved and checked against the offered candidates before any is
    # recorded, so a typo in the third --pick records nothing.
    group_ids = [str(g["group_id"]) for g in summary["groups"]]
    answers: list[tuple[UUID, str | None, Literal["pick", "decline"]]] = []
    for spec in args.pick:
        gid_text, sep, key = spec.partition("=")
        if not sep or not key.strip():
            raise UsageError(f"--pick {spec!r}: expected GROUP=OPTION_KEY or GROUP=none")
        gid = UUID(_by_prefix(group_ids, gid_text, "group"))
        option = None if key.strip().lower() in ("none", ANCHOR_KEY) else key.strip()
        offered = {c["option_key"] for c in svc.candidates_for_group(gid)}
        if option is not None and option not in offered:
            raise UsageError(
                f"--pick {spec!r}: {option} is not offered in group {gid} "
                f"(offered: {', '.join(sorted(offered - {ANCHOR_KEY}))})",
                EXIT_FAIL,
            )
        answers.append((gid, option, "pick"))
    for gid_text in args.decline:
        answers.append((UUID(_by_prefix(group_ids, gid_text, "group")), None, "decline"))
    results = []
    try:
        for gid, option, action in answers:
            results.append(
                svc.record_human_pick(
                    gid, args.actor, via="cli", owner=owner, option_key=option, action=action
                )
            )
    except (NotOwner, KeyError, ValueError) as exc:
        raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
    print(_json_out(results))
    return EXIT_OK


def _cmd_feedback(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_clm.service import NotOwner

    try:
        gid = UUID(args.group_id)
    except ValueError as exc:
        raise UsageError(f"--group-id {args.group_id!r} is not a group id") from exc
    svc = _reader_service(cfg)
    try:
        res = svc.record_human_pick(
            gid,
            args.actor,
            via="cli",
            owner=args.actor,
            option_key=args.option_key,
            action=args.action,
        )
    except (NotOwner, KeyError, ValueError) as exc:
        raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
    print(_json_out(res))
    return EXIT_OK


# -- M1: provenance ---------------------------------------------------------------------------


def _cmd_provenance(args: argparse.Namespace, cfg: Config) -> int:
    if args.verb == "migrate":
        from mesa_clm.provenance.migrate import apply_migrations

        dsn = args.dsn or cfg.provenance.dsn
        try:
            version = apply_migrations(dsn)
        except ImportError as exc:
            raise UsageError(f"{redact_dsn(dsn)}: {exc} (install the pg extra)") from exc
        except ValueError as exc:
            raise UsageError(f"{redact_dsn(dsn)}: {exc}") from exc
        print(f"{redact_dsn(dsn)}: mesa_clm schema v{version}")
        return EXIT_OK
    from mesa_clm.provenance import export

    if args.verb == "import":
        from mesa_clm.provenance.store import open_store

        store = open_store(cfg.provenance.dsn)
        try:
            run_id = export.import_run(store, args.path)
        except FileNotFoundError as exc:
            raise UsageError(str(exc)) from exc
        except ValueError as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
        print(f"imported run {run_id} into {redact_dsn(cfg.provenance.dsn)}")
        return EXIT_OK
    store = _open_store(cfg)
    if args.verb == "export":
        run_id = _resolve_run(store, args.run_id, None)
        try:
            written = export.export_run(store, run_id, args.out)
        except KeyError as exc:
            raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
        print(f"exported run {run_id}: {written['manifest']}")
        return EXIT_OK
    ttl = args.ttl_days if args.ttl_days is not None else cfg.provenance.ttl_days
    report = export.prune(store, ttl_days=ttl, dry_run=args.dry_run)
    print(_json_out(report.summary()))
    return EXIT_OK


# -- M1: serve ----------------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm import serving
    from mesa_clm.secrets import SecretError

    if args.verb == "keys":
        try:
            res = serving.init_keys(args.secrets_dir, rotate=args.rotate)
        except (SecretError, serving.ServingError) as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
        if res.rotated:
            state = "rotated (both keys replaced)"
        elif res.created:
            state = f"created {', '.join(res.created)}"
        else:
            state = "kept (both keys already present)"
        print(f"keys in {res.secrets_dir} (0700): {state}; env files re-derived (0600):")
        for path in (res.clm_key, res.encoder_key, res.clm_env, res.encoder_env):
            print(f"  {path}")
        print("point mesa-clm at the key files (paths only; keys are never printed):")
        print(f"  export MESA_CLM_CLM__API_KEY_FILE={res.clm_key}")
        print(f"  export MESA_CLM_ENCODER__API_KEY_FILE={res.encoder_key}")
        if res.rotated:
            units = " ".join(serving.UNIT_NAMES[:2])
            print("the running units keep the old keys until restarted; run:")
            print(f"  systemctl --user restart {units}")
        return EXIT_OK
    if args.verb == "units":
        try:
            rendered = serving.render_units(args.home)
        except serving.ServingError as exc:
            raise UsageError(str(exc)) from exc
        for name, text in rendered.items():
            print(f"# ---- {name} ----")
            print(text, end="" if text.endswith("\n") else "\n")
        print(
            f"# install: copy into {serving.unit_install_dir()} and "
            "`systemctl --user daemon-reload` (installing does not enable them)"
        )
        return EXIT_OK
    checks = serving.check_serving_lock(args.lock, home=args.home, require_live=args.require_live)
    tags = {"ok": "ok", "skip": "skip", "fail": "FAIL"}
    for c in checks:
        print(f"[{tags[c.status]}] {c.name}: {c.detail}")
    return EXIT_FAIL if any(c.status == "fail" for c in checks) else EXIT_OK


def _m1_parsers(sub: Any, common: argparse.ArgumentParser) -> None:
    # config.Tier (= pipeline.TIERS, asserted by the tests): building the parser never imports
    # the pipeline.
    tiers = get_args(Tier)
    fr = sub.add_parser("framings", help="framing keys against framings.lock.json (D1)")
    mode = fr.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 when the keys drifted")
    mode.add_argument(
        "--update-lock", action="store_true", help="rewrite the lock after a deliberate change"
    )
    fr.add_argument("--lock", help="lock file (default: the checkout's framings.lock.json)")
    fr.set_defaults(func=_cmd_framings)

    an = sub.add_parser(
        "annotate",
        parents=[common],
        help="decide one dataset card and record the run in the sidecar (proposed-only)",
    )
    an.add_argument("--card", required=True, help="dataset card file (CLI only, D26)")
    an.add_argument(
        "--provider",
        choices=["clm", "fake"],
        default="clm",
        help="clm: the loopback serving pair (default); fake: deterministic, offline, not evidence",
    )
    an.add_argument(
        "--planner", choices=["static", "gateway", "claude"], help="default: planner.kind"
    )
    an.add_argument("--tier", choices=tiers, help="default: decider.tier (auto)")
    an.add_argument("--owner", help="who the run belongs to (default: --actor)")
    an.add_argument("--out", help="write the run as JSON to this file ('-': stdout)")
    an.add_argument(
        "--eval-result",
        action="store_true",
        help="write the neon-avu-eval result shape instead of the run",
    )
    an.add_argument(
        "--fake-seed", type=int, default=0, help="seed of the fake head (--provider fake)"
    )
    an.set_defaults(func=_cmd_annotate)

    ex = sub.add_parser(
        "explain", parents=[common], help="a run (or an iRODS path's runs), read-only"
    )
    which = ex.add_mutually_exclusive_group(required=True)
    which.add_argument("--run-id", help="run id or a unique prefix of one of your runs")
    which.add_argument("--irods-path", help="your runs that proposed AVUs for this path")
    ex.add_argument("--limit", type=int, default=50, help="rows per section (default 50)")
    ex.set_defaults(func=_cmd_explain)

    rv = sub.add_parser(
        "review",
        parents=[common],
        help="answer a run's pending groups (curator labels, via=cli); interactive by default",
    )
    rv.add_argument("--run-id", required=True, help="run id or a unique prefix")
    rv.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="GROUP=OPTION_KEY|none",
        help="non-interactive: accept a candidate (or none of them) for a group id or prefix",
    )
    rv.add_argument(
        "--decline",
        action="append",
        default=[],
        metavar="GROUP",
        help="non-interactive: answer nothing for a group (kept as an override row only)",
    )
    rv.set_defaults(func=_cmd_review)

    fb = sub.add_parser(
        "feedback", parents=[common], help="pick, reject or decline one group (via=cli)"
    )
    fb.add_argument("--group-id", required=True)
    fb.add_argument("--action", choices=["pick", "reject", "decline"], required=True)
    fb.add_argument("--option-key", help="the candidate's CURIE for a pick (none: 'none of these')")
    fb.set_defaults(func=_cmd_feedback)

    pv = sub.add_parser("provenance", help="the mesa_clm sidecar: schema, export, import, prune")
    pv_sub = pv.add_subparsers(dest="verb", required=True)
    mig = pv_sub.add_parser("migrate", parents=[common], help="create or upgrade the schema")
    mig.add_argument("--dsn", help="sidecar DSN (default: provenance.dsn)")
    exp = pv_sub.add_parser("export", parents=[common], help="Parquet copy of a run + manifest")
    exp.add_argument("--run-id", required=True)
    exp.add_argument("--out", required=True, help="directory; the run lands in <out>/<run_id>/")
    imp = pv_sub.add_parser("import", parents=[common], help="commit an exported run")
    imp.add_argument("path", help="an exported run directory (or its manifest.json)")
    pru = pv_sub.add_parser(
        "prune", parents=[common], help="delete terminal (or long-abandoned) runs' local rows"
    )
    pru.add_argument("--ttl-days", type=int, help="default: provenance.ttl_days")
    pru.add_argument("--dry-run", action="store_true", help="say what would go; delete nothing")
    pv.set_defaults(func=_cmd_provenance)

    se = sub.add_parser("serve", help="the serving host: keys, systemd units, the serving lock")
    se_sub = se.add_subparsers(dest="verb", required=True)
    keys = se_sub.add_parser("keys", help="bearer key files and the units' env files (0600)")
    kmode = keys.add_mutually_exclusive_group(required=True)
    kmode.add_argument("--init", action="store_true", help="create missing keys (idempotent)")
    kmode.add_argument("--rotate", action="store_true", help="replace both keys")
    keys.add_argument("--secrets-dir", help="default ~/.mesa/clm/secrets")
    units = se_sub.add_parser("units", help="print the rendered systemd --user units")
    units.add_argument("--home", default="~/.mesa/clm", help="serving home (default ~/.mesa/clm)")
    lock = se_sub.add_parser("lock", help="verify serving.lock.json against this host")
    lock.add_argument("--check", action="store_true", required=True)
    lock.add_argument(
        "--lock", help="lock file (default: the checkout's serving/serving.lock.json)"
    )
    lock.add_argument("--home", help="serving home (default ~/.mesa/clm)")
    lock.add_argument(
        "--require-live",
        action="store_true",
        help="absent pieces (clone, venv, head, image) fail instead of being skipped",
    )
    se.set_defaults(func=_cmd_serve)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mesa-clm",
        description="Calibrated, ontology-grounded MESA decisions with Contrastive LM.",
    )
    p.add_argument("--version", action="version", version=f"mesa-clm {__version__}")
    p.add_argument("--config", help="YAML config file (env MESA_CLM_* and flags override it)")
    p.add_argument("--provenance", help="sidecar DSN, duckdb:///<path> (default: config)")
    p.add_argument(
        "--actor",
        default=os.environ.get("USER", "mesa-clm"),
        help="actor recorded on labels, runs and AVUs (default: $USER)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # The same two options after the verb; SUPPRESS keeps the sub-parser from clobbering the
    # values parsed before it with its own defaults.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provenance", default=argparse.SUPPRESS, help="sidecar DSN (default: config)"
    )
    common.add_argument("--actor", default=argparse.SUPPRESS, help="actor recorded on labels")

    la = sub.add_parser(
        "labels",
        parents=[common],
        help="labels in the sidecar: ingest the neon-avu-eval silver, import a mesa-anyjev "
        "sidecar, freeze a snapshot, print counts",
    )
    la.add_argument("verb", choices=["ingest-neon-eval", "import-anyjev", "snapshot", "stats"])
    la.add_argument(
        "--eval-root", help="neon-avu-eval checkout (ingest-neon-eval; default: config eval_root)"
    )
    la.add_argument(
        "--exclude-models",
        help="comma-separated neon-avu-eval model names to drop first (X4 silver-minus-Opus)",
    )
    la.add_argument("--dsn", help="mesa-anyjev sidecar, duckdb:///<path> or a path (import-anyjev)")
    la.add_argument(
        "--out",
        help=f"snapshot Parquet path (snapshot; default {DEFAULT_SNAPSHOT_DIR}/<today>.parquet)",
    )
    la.set_defaults(func=_cmd_labels)

    be = sub.add_parser(
        "bench",
        parents=[common],
        help="no-model controls (baselines: lookup_prob, novel-key, LOPO) and the MDE simulation",
    )
    be.add_argument("verb", choices=["baselines", "mde"])
    be.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="results root (<out-dir>/<date>/)")
    be.add_argument("--date", help="results directory name (default: today, UTC)")
    be.add_argument(
        "--snapshot",
        help=(
            "frozen labels snapshot to stamp the cells with "
            f"(default {DEFAULT_SNAPSHOT_DIR}/<date>.parquet, written when missing)"
        ),
    )
    be.add_argument("--B", dest="B", type=int, default=DEFAULT_B, help="bootstrap replicates")
    be.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed")
    be.add_argument(
        "--n-sims", type=int, default=DEFAULT_N_SIMS, help="simulated datasets per grid point (mde)"
    )
    be.set_defaults(func=_cmd_bench)

    d = sub.add_parser(
        "doctor", help="what this host can run: pins, versions, stores, paths, serving"
    )
    d.add_argument(
        "--quick",
        action="store_true",
        help="skip the vendored-file hashing and the serving-lock verification",
    )
    d.add_argument(
        "--serve",
        action="store_true",
        help="run the live serving probes (binds, 401s, health, models, a golden systemone "
        "call; automatic when both mesa-clm units are active) and fail when unreachable",
    )
    d.add_argument("--json", action="store_true", help="print the report as JSON")
    d.set_defaults(func=_cmd_doctor)
    _m1_parsers(sub, common)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.provenance:
        # Folded into the config so every verb, the doctor included, sees the same sidecar.
        overrides["provenance"] = {"dsn": args.provenance}
    try:
        cfg = load_config(args.config, flag_overrides=overrides)
    except ValidationError as exc:
        # Field and problem only, never the input: a value meant for *_API_KEY that landed on a
        # mistyped name (or failed its type check) must not reach stderr or a CI log.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
        print(f"config error: {problems}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    set_active_config(cfg)
    try:
        result: int = args.func(args, cfg)
    except UsageError as exc:
        print(f"mesa-clm: {exc}", file=sys.stderr)
        return exc.code
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
