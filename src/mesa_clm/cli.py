"""The ``mesa-clm`` console script (plan §7.2).

Verbs so far: ``labels ingest-neon-eval|import-anyjev|snapshot|stats`` and ``bench
baselines|mde`` (M0); ``doctor [--quick] [--serve] [--json]``; ``framings --check|--update-lock``;
``annotate``, ``explain``, ``review`` and ``feedback`` (the decide phase and its review, M1);
``provenance migrate|export|import|prune``; ``serve keys|units|lock``; ``features
build|export-npz|project|stats`` (the feature cache, M1). ``plan``, ``apply``, ``revert``,
``learn``, ``artifacts``, ``history`` and ``neon curate`` land with their milestones;
``mesa-clm --help`` lists what exists. Ported from mesa-anyjev ``cli.py``
(``6159281``): the same argparse shape, exit codes and lazy imports inside the commands, so
``--help`` and ``doctor`` never import the modules they do not need.

Exit codes: ``EXIT_OK`` 0; ``EXIT_FAIL`` 1 when the verb ran and failed (a doctor ``fail``, an
empty store, a snapshot that no longer matches the store, framing drift, an unreachable
clm-serve, a replay miss, another owner's run); ``EXIT_CONFIG`` 2 for a configuration or usage
problem (an unreadable config file or card, a missing ``--eval-root`` or one without
``results/validated.json`` and ``cards/``, a Postgres DSN for the M0 label verbs). A
configuration error is reported as ``config error: <field>: <problem>`` (validation) or ``config
error: <file>: invalid YAML at line L, column C`` (syntax) without the offending value or a
snippet of the file, so a mistyped ``MESA_CLM_<SECTION>__API_KEY`` or a broken ``api_key:`` line
never echoes the key (``Config`` also hides inputs in its errors), and no verb prints a key:
``serve keys`` names files only. The global ``--config``, ``--provenance`` and ``--actor`` are
read before the verb (``mesa-clm --provenance DSN labels stats``); the ``labels``, ``bench``,
``annotate``, ``explain``, ``review``, ``feedback`` and ``provenance`` verbs also accept
``--provenance`` and ``--actor`` after the verb, as mesa-anyjev's users type them (``doctor``,
``framings``, ``serve`` and ``features`` do not). ``--actor`` must not be blank.

**Owners and curator labels (D21, DESIGN A2).** A run belongs to its ``owner``: ``annotate
--owner`` (default ``--actor``, itself ``$USER``). ``explain``, ``review`` and ``feedback`` act
as ``--actor`` and refuse another owner's run; on a shared Postgres sidecar ``--actor`` must be
the OS account. Curator labels (``via='cli'``) need an interactive terminal on stdin: the
interactive ``review`` requires one, and ``review --pick/--decline`` or ``feedback`` run without
one are recorded exactly like a plain tool call (``via='tool'``: ``agent_pick`` at weight 0,
never fold-eligible), which the verb says on stderr. A curator's answer settles a group (a
different second answer is refused); an agent's answer leaves it pending for a curator, whose
answer replaces it. ``review --pick/--decline`` answer pending groups only, each at most once per
call, and check every answer before recording any; ``feedback --action pick`` needs
``--option-key`` (``none`` for "none of these"), and ``--group-id`` takes a unique prefix among
your runs' groups.

**annotate --provider clm** builds the live provider (:mod:`mesa_clm.providers.live`: loopback
URLs and keys from the ``clm``/``encoder`` sections, the fingerprint of the serving lock the host
runs, refused when that lock contradicts the vendored schema or the checkout's lock) and runs
the pre-flight (``live.preflight``) first. When clm-serve or its encoder does not answer: tier
``auto`` with ``decider.ols_rank_fallback: true`` runs the degraded ``ols_rank`` method
(proposed-only, never auto, D28; the run says ``degraded``); otherwise, and always for an
explicit CLM tier, the verb refuses with exit 1 (the plugin's ``decider_unavailable``). A
missing or rejected key, a ``clm.model`` clm-serve does not serve or an encoder that is not the
lock's is never degraded: exit 2. A request clm-serve refuses mid-run (401, 403, 404) fails the
run (recorded ``failed``, exit 1) instead of turning every group into ``ols_rank``; transport
errors, timeouts and 5xx still fall back per group, and the summary counts the failed calls.
After the pre-flight the running encoder container is compared with the lock's recipe
(``providers.live.container_check``): a departure is refused (exit 2), and a check docker could
not answer is noted as "not verified" in the summary and the ``--out`` JSON (``preflight``, in
the run shape and the ``--eval-result`` shape alike). A loopback port held by another account's
socket is refused before the key is sent (exit 2; ``net.assert_listener_owner``, DESIGN A5).
``--tier ols_rank`` skips the pre-flight and never needs clm-serve for the candidate groups.
``--provider fake`` is the deterministic offline fake; it exercises the pipeline and is never
evidence.

``bench baselines`` and ``bench mde`` stamp their cells with the ``labels_sha256`` of a frozen
snapshot (DESIGN D30): ``--snapshot PATH`` names one written by ``labels snapshot``, otherwise
``bench/snapshots/<date>.parquet`` is used and written from the store when missing. An existing
snapshot must still describe the store (same label content hash), or the verb refuses.

**features** (plan §5.2, §5.6; DESIGN D5; :mod:`mesa_clm.learn.features`). ``features build
--snapshot PATH`` embeds every X1/X2 text of a frozen label snapshot that the store of the live
``encoder_fp`` still lacks, through ``EncoderClient`` (the ``encoder`` section; the live
fingerprint is the serving lock this host runs, ``providers.live.live_lock_path``), one text per
request unless ``--batch`` says otherwise (a text embedded alone gets the same vector bit for
bit every time; under the batch-invariant recipe of DESIGN A3 a batch agrees to 1 - 3.3e-15 in
cosine, ``bench/results/2026-10-01/batch_invariance.json``), after the token guard: counts must
be exact (``/tokenize`` or the ``tokenize`` extra) and a text over ``max_len - 16`` tokens is
recorded as truncated and never embedded. It refuses a store whose recorded ``encoder_fp``,
format or vector recipe is not the live lock's, an encoder that does not serve the lock's model,
window and route, and a running encoder container that departs from the lock's recipe
(``providers.live.container_check``; exit 1), and an ``encoder.max_len`` other than the lock's
(exit 2). The store keeps float32 vectors, each stamped with the live ``lock_sha``. Vectors are
committed every 64 texts and before a failure is reported, so a re-run embeds only what is still
missing.
``features export-npz --out DIR`` writes CLM's ``TextCache`` file for ``finetune.py
--embed-cache DIR`` (refused below the fp16 round-trip gate; owner-only), ``features project``
caches the pinned head's projections of every stored vector (``--model clm-raw`` has none to
cache) and ``features stats [--json]`` describes every store under ``features.dir``.
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
    ConfigError,
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
# The M2 verbs' one snapshot (mesa_clm.bench.registered; asserted by the CLI tests) so building
# the parser imports nothing.
REGISTERED_SNAPSHOT = "bench/snapshots/2026-09-29.parquet"
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


def _non_blank(text: str) -> str:
    """An argparse type: a value that is not empty or whitespace (``--actor``)."""
    if not text.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return text


def _non_negative_int(text: str) -> int:
    """An argparse type: an integer >= 0 (``--ttl-days``)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


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
        if args.trust_curator and not stdin_is_terminal():
            raise UsageError(
                "--trust-curator vouches for the file's curator labels and needs an interactive "
                "terminal (DESIGN A2); without it they are imported as agent_pick (weight 0)"
            )
        try:
            report = learn.import_anyjev(
                args.dsn, store, actor=args.actor, trust_curator=args.trust_curator
            )
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
    if args.verb in ("framing", "run", "x2", "table"):
        return _cmd_bench_m2(args, cfg)
    from mesa_clm.bench.results import ResultsExist

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
        try:
            path = write_results(results, args.out_dir, force=args.force)
        except ResultsExist as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
        print(path)
        print(markdown_table(results))
        return EXIT_OK
    from mesa_clm.bench.mde import run_mde, write_mde
    from mesa_clm.bench.tasks.neon import tasks_from_store

    tasks = tasks_from_store(store, policy_path=cfg.policy.policy_path)
    mde = run_mde(tasks, labels_sha256=sha, date=date, n_sims=args.n_sims, B=args.B, seed=args.seed)
    out = Path(args.out_dir) / date / "mde.json"
    if out.exists() and not args.force:
        raise UsageError(
            f"{out} exists: one run per results file (pass --force to replace)", EXIT_FAIL
        )
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


def _csv_arg(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [t.strip() for t in text.split(",") if t.strip()]


def _cmd_bench_m2(args: argparse.Namespace, cfg: Config) -> int:
    """``bench framing|run|x2|table`` (M2; ``design/m2-analysis-plan.md`` §12, §14): refusals of
    the registered inputs and of an X1 file that does not reproduce exit 1, usage problems 2."""
    from mesa_clm.bench import run as m2
    from mesa_clm.bench.framing import X1DataError, X1Exists, X1ReproError
    from mesa_clm.bench.registered import RegistrationError
    from mesa_clm.bench.results import ResultsExist
    from mesa_clm.learn.features import FeatureMissing, FeatureStoreError, ManifestError

    date = getattr(args, "date", None) or _today()
    try:
        if args.verb == "framing":
            return _bench_framing(args, cfg, date)
        if args.verb == "run":
            if not args.loco:
                raise UsageError(
                    "bench run needs --loco: leave-one-card-out is the only split a cell may be "
                    "cited from (D8)"
                )
            path, results = m2.run_tiers(
                cfg,
                date=date,
                out_dir=args.out_dir,
                framing_from=args.framing_from,
                snapshot=args.snapshot,
                tiers=_csv_arg(args.tiers) or [],
                force=args.force,
            )
            print(path)
            for key, cell in results.cells.items():
                print(
                    f"{key}: n={cell.counts.n} selection={cell.selection} "
                    f"pre_registered={cell.pre_registered} exploratory={cell.exploratory}"
                )
            return EXIT_OK
        if args.verb == "x2":
            path, results = m2.run_x2(
                cfg, date=date, out_dir=args.out_dir, snapshot=args.snapshot, force=args.force
            )
            print(path)
            print("\n".join(sorted(results.cells)))
            return EXIT_OK
        paths = (
            [Path(p) for p in args.results]
            if args.results
            else m2.results_files(args.out_dir, date)
        )
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise UsageError(f"{missing[0]}: no such results file")
        text = m2.table(paths)
        if args.out:
            out = Path(args.out)
            if out.exists() and not args.force:
                raise UsageError(f"{out} exists (pass --force to replace)", EXIT_FAIL)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(out)
        else:
            print(text, end="")
        return EXIT_OK
    except m2.BenchRunError as exc:
        raise UsageError(str(exc), EXIT_CONFIG if exc.usage else EXIT_FAIL) from exc
    except (RegistrationError, X1DataError, X1ReproError, FeatureMissing, FeatureStoreError) as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    except (X1Exists, ResultsExist) as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    except ManifestError as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc


def _print_decisions(decisions: dict[str, Any]) -> None:
    for task, d in decisions.items():
        if d is None:
            print(f"{task}: no full run")
            continue
        choice = "-" if d.choice is None else d.choice.arm
        print(f"{task}: {d.outcome} -> {choice}")


def _bench_framing(args: argparse.Namespace, cfg: Config, date: str) -> int:
    from mesa_clm.bench import framing
    from mesa_clm.bench import run as m2

    if args.from_ is not None:
        if not args.decide:
            raise UsageError("--from replays an x1.json and needs --decide")
        path = Path(args.from_)
        if not path.is_file():
            raise UsageError(f"--from {path}: no such file")
        decisions = framing.decide_from_json(path)
        print(f"{path}: the pre-registered run; every decision replays and recomputes")
        _print_decisions(decisions)
        return EXIT_OK
    full, nested = {"full": (True, False), "nested": (False, True)}.get(args.part, (True, True))
    path, run = m2.run_framing(
        cfg,
        date=date,
        out_dir=args.out_dir,
        snapshot=args.snapshot,
        tasks=_csv_arg(args.tasks),
        full=full,
        nested=nested,
        latency=args.latency,
        force=args.force,
    )
    block = run.results.x1
    print(path)
    print(
        "registered run"
        if block.registered
        else f"NOT the registered run ({'; '.join(block.deviations)}): every cell exploratory"
    )
    if block.latency is None:
        print(
            "no --latency file: p50 latency is not reported (design/m2-analysis-plan.md §11.3)",
            file=sys.stderr,
        )
    _print_decisions({t: r.decision for t, r in block.tasks.items()})
    if args.decide:
        framing.decide_from_json(path, require_registered=block.registered)
        print(
            f"{path}: replayed and recomputed from {block.items_file.name if block.items_file else '-'}"
        )
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
        try:
            answering, detail = live.preflight(stack, cfg)
        except live.PreflightError as exc:  # a key, model or encoder problem: never degraded
            stack.close()
            raise UsageError(str(exc)) from exc
        except SecretError as exc:
            stack.close()
            raise UsageError(str(exc)) from exc
        if answering:
            notes.append(detail)
            # /v1/models cannot tell two recipes of one image apart (live.container_check).
            check = live.container_check(stack.lock)
            if not check.ok:
                stack.close()
                raise UsageError(
                    f"{check.note()}: its vectors are not encoder_fp {stack.lock.encoder_fp} "
                    "(D5, K4); restart the units from the installed lock"
                )
            notes.append(check.note())
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
    from mesa_clm.providers.base import DeciderRefused, TierUnavailable
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
        except DeciderRefused as exc:
            raise UsageError(
                f"{exc}; the run was recorded with status failed (check "
                "MESA_CLM_CLM__API_KEY_FILE and clm.base_url, then `mesa-clm doctor --serve`)",
                EXIT_FAIL,
            ) from exc
    finally:
        close()
    out = run.to_eval_result() if args.eval_result else run.to_dict()
    # Whether the encoder recipe was verified travels with either shape (the run row does not
    # record it until M3); an extra key leaves neon-avu-eval's scoring unchanged.
    out["preflight"] = notes
    if not args.eval_result:
        out["tier"] = tier
        out["next_step"] = f"mesa-clm review --run-id {run.run_id}"
    if args.out == "-":
        print(_json_out(out))
    elif args.out:
        from mesa_clm.perms import private_dir, write_private_text

        # A copy of sidecar content: owner-only like the sidecar (mesa_clm.perms).
        target = Path(args.out).expanduser()
        try:
            private_dir(target.parent, tighten=False)
            write_private_text(target, _json_out(out) + "\n")
        except OSError as exc:
            raise UsageError(
                f"--out {args.out}: cannot write ({type(exc).__name__}); the run {run.run_id} "
                "is recorded in the sidecar",
                EXIT_FAIL,
            ) from exc
    report = sys.stderr if args.out == "-" else sys.stdout
    fp = run.fingerprint
    lines = [
        f"run {run.run_id} (owner {run.owner}, card {run.card.name}, tier {tier})",
        f"  {len(run.proposals)} proposals, {len(run.abstained)} abstained, "
        f"{run.n_decisions} decisions, {run.n_calls} CLM calls"
        f"{f' ({run.n_failed_calls} failed)' if run.n_failed_calls else ''}, "
        f"{run.input_tokens} input tokens, {run.seconds:.2f} s"
        f"{', degraded' if run.degraded else ''}",
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

    owner = _review_owner(args, cfg)
    svc = _reader_service(cfg)
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


def stdin_is_terminal() -> bool:
    """Whether stdin is an interactive terminal: the condition for ``via='cli'``, i.e. for
    curator labels from the CLI (DESIGN A2). A closed or replaced stdin counts as no terminal."""
    stdin = sys.stdin
    try:
        return stdin is not None and bool(stdin.isatty())
    except (AttributeError, OSError, ValueError):
        return False


AGENT_NOTICE = (
    "stdin is not a terminal: the answers are recorded like a plain tool call (via=tool: "
    "agent_pick labels at weight 0, never fold-eligible; links accepted_by=agent). Curator "
    "labels need an interactive terminal (DESIGN A2)."
)


def _answer_via() -> Literal["cli", "tool"]:
    """``cli`` at an interactive terminal, else ``tool`` with :data:`AGENT_NOTICE` on stderr."""
    if stdin_is_terminal():
        return "cli"
    print(f"mesa-clm: {AGENT_NOTICE}", file=sys.stderr)
    return "tool"


def _os_user() -> str:
    """The OS account this process runs as (``$USER`` when the passwd entry is missing)."""
    import pwd

    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return os.environ.get("USER", "")


def _review_owner(args: argparse.Namespace, cfg: Config) -> str:
    """The owner the review verbs (``explain``, ``review``, ``feedback``) act as: ``--actor``.
    On a per-user DuckDB sidecar (``0700``, D11) the OS account is the trust boundary and
    ``--actor`` is free; on a shared Postgres sidecar any database user could pass another
    owner's name, so there it must be the OS account (DESIGN A2)."""
    from mesa_clm.provenance.store import is_postgres_dsn

    owner: str = args.actor
    if is_postgres_dsn(cfg.provenance.dsn):
        me = _os_user()
        if owner != me:
            raise UsageError(
                f"--actor {owner!r}: on a shared Postgres sidecar explain, review and feedback "
                f"act only as the OS account ({me or 'unknown'}) (DESIGN A2)"
            )
    return owner


def review_interactive(
    svc: Any,
    pending: list[dict[str, Any]],
    *,
    owner: str,
    actor: str,
    ask: Any,
    say: Any,
    via: Literal["cli", "tool"] = "cli",
) -> dict[str, int]:
    """Walk the pending groups: show each group's offered candidates with ``p_fit`` and the
    anchor, record the answer through ``record_human_pick(via=via)`` (``cli``: the caller made
    sure stdin is a terminal, DESIGN A2). ``ask(prompt) -> str`` and ``say(text)`` are the
    terminal (``input``/``print``), injectable for tests. Answers: a number picks that
    candidate, ``0``/``n`` is "none of these", ``d`` declines, ``s`` or empty skips, ``q`` (or
    end of input, or Ctrl-C) stops."""
    from uuid import UUID

    from mesa_clm.registry import ANCHOR_KEY

    done = {"picked": 0, "none": 0, "declined": 0, "skipped": 0}
    for i, g in enumerate(pending, 1):
        cands = svc.candidates_for_group(UUID(g["group_id"]))
        say(_describe_group(g, i, len(pending)))
        if g.get("agent_answered"):
            say("  (an agent answered this group; your answer replaces it)")
        lines, keys = _candidate_lines(cands)
        for line in lines:
            say(line)
        while True:
            try:
                answer = ask(
                    f"pick 1-{len(keys)}, 0 = none of these, d = decline, s = skip, q = quit [s]: "
                )
            except (EOFError, KeyboardInterrupt):
                say("")
                return done
            answer = answer.strip().lower()
            if answer in ("", "s"):
                done["skipped"] += 1
                break
            if answer == "q":
                return done
            if answer == "d":
                res = svc.record_human_pick(
                    UUID(g["group_id"]), actor, via=via, owner=owner, action="decline"
                )
                done["declined"] += 1
            elif answer in ("0", "n"):
                res = svc.record_human_pick(
                    UUID(g["group_id"]), actor, via=via, owner=owner, option_key=ANCHOR_KEY
                )
                done["none"] += 1
            elif answer.isdigit() and 1 <= int(answer) <= len(keys):
                res = svc.record_human_pick(
                    UUID(g["group_id"]),
                    actor,
                    via=via,
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


def _pending_group(summary: dict[str, Any], text: str) -> str:
    """The pending group of a run that ``text`` names (an id or a unique prefix); a group of the
    run that does not wait for a reviewer is refused with a pointer to ``feedback``."""
    pending = [str(g) for g in summary["pending_groups"]]
    try:
        return _by_prefix(pending, text, "group")
    except UsageError:
        others = [str(g["group_id"]) for g in summary["groups"]]
        needle = text.strip().lower()
        hits = [
            g
            for g in others
            if g == needle or (len(needle) >= _MIN_PREFIX and g.startswith(needle))
        ]
        if len(hits) == 1:
            raise UsageError(
                f"group {hits[0]} does not wait for a reviewer (review answers pending groups "
                "only; `mesa-clm feedback --group-id` answers any group)",
                EXIT_FAIL,
            ) from None
        raise


def _cmd_review(args: argparse.Namespace, cfg: Config) -> int:
    from uuid import UUID

    from mesa_clm.registry import ANCHOR_KEY
    from mesa_clm.service import NotOwner

    owner = _review_owner(args, cfg)
    svc = _reader_service(cfg)
    run_id = _resolve_run(svc.store, args.run_id, owner)
    try:
        summary = svc.run_summary(run_id, owner=owner)
    except NotOwner as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    except KeyError as exc:
        raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
    if not (args.pick or args.decline):
        if not stdin_is_terminal():
            raise UsageError(
                "review is interactive (it needs a terminal); without one, --pick "
                "GROUP=OPTION_KEY|none and --decline GROUP record an agent's answers (via=tool, "
                "DESIGN A2)"
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
    # Every answer is resolved and checked (pending group, offered candidate, one answer per
    # group, the settled-group rules) before any is recorded, so a typo in the third --pick
    # records nothing.
    via = _answer_via()
    answers: list[tuple[UUID, str | None, Literal["pick", "decline"]]] = []
    for spec in args.pick:
        gid_text, sep, key = spec.partition("=")
        if not sep or not key.strip():
            raise UsageError(f"--pick {spec!r}: expected GROUP=OPTION_KEY or GROUP=none")
        gid = UUID(_pending_group(summary, gid_text))
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
        answers.append((UUID(_pending_group(summary, gid_text)), None, "decline"))
    seen: set[UUID] = set()
    for gid, _, _ in answers:
        if gid in seen:
            raise UsageError(f"group {gid} is answered more than once in one call")
        seen.add(gid)
    try:
        for gid, option, action in answers:
            svc.check_answer(gid, via=via, owner=owner, option_key=option, action=action)
        results = [
            svc.record_human_pick(
                gid, args.actor, via=via, owner=owner, option_key=option, action=action
            )
            for gid, option, action in answers
        ]
    except (NotOwner, KeyError, ValueError) as exc:
        raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
    print(_json_out(results))
    return EXIT_OK


def _resolve_group(svc: Any, text: str, owner: str) -> Any:
    """A group id from ``text``: a UUID, or a unique prefix among the groups of ``owner``'s
    runs (at least 4 hex digits)."""
    from uuid import UUID

    try:
        return UUID(text)
    except ValueError:
        pass
    ids = [
        str(g["group_id"])
        for r in svc.store.runs(owner=owner, limit=None)
        for g in svc.store.groups(UUID(str(r["run_id"])))
    ]
    return UUID(_by_prefix(ids, text, "--group-id"))


def _cmd_feedback(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm.registry import ANCHOR_KEY
    from mesa_clm.service import NotOwner

    option = args.option_key.strip() if args.option_key else None
    if args.action == "pick" and not option:
        raise UsageError(
            "--action pick needs --option-key (a candidate's key, or 'none' for none of these)"
        )
    if option is not None and option.lower() in ("none", ANCHOR_KEY):
        option = ANCHOR_KEY
    owner = _review_owner(args, cfg)
    svc = _reader_service(cfg)
    gid = _resolve_group(svc, args.group_id, owner)
    via = _answer_via()
    try:
        res = svc.record_human_pick(
            gid,
            args.actor,
            via=via,
            owner=owner,
            option_key=option,
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

        dsn = cfg.provenance.dsn
        try:
            store = open_store(dsn)
        except ImportError as exc:
            raise UsageError(f"{redact_dsn(dsn)}: {exc} (install the pg extra)") from exc
        except ValueError as exc:
            raise UsageError(f"{redact_dsn(dsn)}: {exc}") from exc
        try:
            run_id = export.import_run(store, args.path)
        except FileNotFoundError as exc:
            raise UsageError(str(exc)) from exc
        except ValueError as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
        except OSError as exc:
            raise UsageError(f"{args.path}: cannot read ({type(exc).__name__})", EXIT_FAIL) from exc
        print(f"imported run {run_id} into {redact_dsn(dsn)}")
        return EXIT_OK
    store = _open_store(cfg)
    if args.verb == "export":
        run_id = _resolve_run(store, args.run_id, None)
        try:
            written = export.export_run(store, run_id, args.out)
        except KeyError as exc:
            raise UsageError(str(exc).strip("'\""), EXIT_FAIL) from exc
        except OSError as exc:
            # An unwritable or non-directory --out, a full disk: the store is unchanged.
            raise UsageError(
                f"--out {args.out}: cannot write the export ({type(exc).__name__}); the run "
                "was not marked exported",
                EXIT_FAIL,
            ) from exc
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
        if res.secrets_dir == serving.serving_home() / serving.SECRETS_DIR:
            print(
                "mesa-clm reads these key files by default (the configuration's api_key_file, "
                "MESA_CLM_*__API_KEY_FILE, still wins; keys are never printed)"
            )
        else:
            print("point mesa-clm at the key files (paths only; keys are never printed):")
            print(f"  export MESA_CLM_CLM__API_KEY_FILE={res.clm_key}")
            print(f"  export MESA_CLM_ENCODER__API_KEY_FILE={res.encoder_key}")
        if res.rotated:
            units = " ".join(serving.UNIT_NAMES[:2])
            # Every start of the encoder counts against its limit of three an hour (DESIGN A5).
            print("the running units keep the old keys until restarted; run:")
            print(f"  systemctl --user reset-failed {serving.ENCODER_UNIT}")
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


# -- M1: features ----------------------------------------------------------------------------------

DEFAULT_FEATURE_TASKS = "term.fits,column.ontology_fits"
DEFAULT_FEATURE_FRAMINGS = "F1,F4,F7,F9,X2"
# One text per /v1/embeddings request: bitwise reproducible on every recipe (serving_m1.md, the
# M1-A recipe, was batch-dependent; under DESIGN A3's kernels a batch agrees to 1 - 3.3e-15).
DEFAULT_FEATURE_BATCH = 1
_FEATURE_PROGRESS = 200
# Vectors are committed to the store every this many texts (one DuckDB transaction each).
_FEATURE_FLUSH = 64
_PROJECT_CHUNK = 256


def _live_lock() -> tuple[Any, Path]:
    """The serving lock this host runs (``providers.live.live_lock_path``), verified."""
    from mesa_clm.clm.fingerprint import LockError, load_serving_lock
    from mesa_clm.providers.live import live_lock_path

    path = live_lock_path()
    try:
        return load_serving_lock(path), path
    except LockError as exc:
        raise UsageError(f"serving lock: {exc}", EXIT_FAIL) from exc


def _existing_feature_store(cfg: Config, lock: Any) -> Any:
    """The store of the live ``encoder_fp``, checked against the live lock's vector recipe; a
    missing one is ``EXIT_FAIL``."""
    from mesa_clm.learn.features import FeatureStore

    store = FeatureStore.for_lock(cfg, lock)
    if not store.exists():
        raise UsageError(
            f"no feature store at {store.path} (encoder_fp {lock.encoder_fp}); run "
            "`mesa-clm features build --snapshot <labels snapshot>` first",
            EXIT_FAIL,
        )
    return store


def _check_live_encoder(enc: Any, cfg: Config, lock: Any) -> None:
    """The encoder answers ``/v1/models`` with the configured served name, and what it reports
    of that model (``root``, ``max_model_len``, ``owned_by`` against the route) is the lock's
    recipe (D5; :func:`mesa_clm.providers.live.encoder_problems`)."""
    from mesa_clm.clm.http import ClmError
    from mesa_clm.net import EndpointError
    from mesa_clm.providers.live import encoder_problems

    if enc.max_len != lock.encoder.max_len:
        raise UsageError(
            f"encoder.max_len {enc.max_len} is not the serving lock's {lock.encoder.max_len}: "
            "the vectors would not be the fingerprinted recipe (D5)"
        )
    try:
        models = enc.models()
    except (ClmError, EndpointError) as exc:
        raise UsageError(
            f"{exc}; start the serving units (docs/deploy/serving.md)", EXIT_FAIL
        ) from exc
    problems = encoder_problems(models, cfg.encoder.model, lock)
    if problems:
        raise UsageError(f"encoder at {enc.endpoint.shown}: {'; '.join(problems)} (D5)", EXIT_FAIL)


def _features_build(args: argparse.Namespace, cfg: Config) -> int:
    import time

    from mesa_clm.clm.encoder import EncoderClient
    from mesa_clm.clm.http import ClmError
    from mesa_clm.learn import features as feat
    from mesa_clm.net import EndpointError
    from mesa_clm.providers.live import container_check
    from mesa_clm.secrets import SecretError

    if args.batch < 1:
        raise UsageError("--batch must be at least 1")
    try:
        manifest = feat.manifest(args.snapshot, args.tasks, args.framings)
    except feat.ManifestError as exc:
        raise UsageError(str(exc)) from exc
    lock, lock_path = _live_lock()
    try:
        # The encoder section with the clm section's one remote switch (plan §6.4), as annotate.
        enc = EncoderClient.from_config(cfg.encoder, allow_remote=cfg.clm.allow_remote)
    except (EndpointError, SecretError) as exc:
        raise UsageError(str(exc)) from exc
    started = time.monotonic()
    try:
        _check_live_encoder(enc, cfg, lock)
        # /v1/models cannot tell two recipes of one image apart: inspect the container too.
        container = container_check(lock)
        if not container.ok:
            raise UsageError(
                f"{container.note()}: its vectors would not be the lock's recipe (D5)", EXIT_FAIL
            )
        store = feat.FeatureStore.for_lock(cfg, lock)
        try:
            store.ensure(encoder_spec=lock.encoder.as_dict())
        except feat.FeatureStoreError as exc:
            raise UsageError(str(exc), EXIT_FAIL) from exc
        texts = manifest.texts()
        todo = store.missing(texts)
        enc.batch = args.batch
        try:
            counts = enc.token_guard(todo)
        except (ClmError, EndpointError) as exc:
            raise UsageError(f"token guard: {exc}", EXIT_FAIL) from exc
        if any(c.source == "chars" for c in counts):
            raise UsageError(
                "the token guard has no exact counter (the encoder answers no /tokenize and no "
                "encoder.tokenizer_json is set): features build records exact token counts only",
                EXIT_FAIL,
            )
        over = [(t, c.tokens) for t, c in zip(todo, counts, strict=True) if c.truncated]
        keep = [(t, c.tokens) for t, c in zip(todo, counts, strict=True) if not c.truncated]
        if over:
            store.add_truncated([t for t, _ in over], [n for _, n in over])
        done = new = charged = 0
        pending: list[tuple[str, int, Any]] = []

        def flush() -> int:
            if not pending:
                return 0
            added: int = store.add(
                [t for t, _, _ in pending], [v for _, _, v in pending], [n for _, n, _ in pending]
            )
            pending.clear()
            return added

        for start in range(0, len(keep), args.batch):
            chunk = keep[start : start + args.batch]
            try:
                vectors, tokens = enc.embed([t for t, _ in chunk])
            except (ClmError, EndpointError) as exc:
                new += flush()
                raise UsageError(
                    f"{exc}; stopped after {done} of {len(keep)} texts (those are stored: "
                    "re-run to embed the rest)",
                    EXIT_FAIL,
                ) from exc
            pending.extend((t, n, v) for (t, n), v in zip(chunk, vectors, strict=True))
            if len(pending) >= _FEATURE_FLUSH:
                new += flush()
            charged += tokens
            done += len(chunk)
            if done % _FEATURE_PROGRESS < len(chunk) or done == len(keep):
                print(f"embedded {done}/{len(keep)} texts", file=sys.stderr)
        new += flush()
    finally:
        enc.close()
    summary = manifest.summary()
    gate = feat.fp16_roundtrip_min_cosine(store)
    print(
        "\n".join(
            [
                f"feature store {store.path} (encoder_fp {lock.encoder_fp})",
                f"  serving lock {lock_path} (lock_sha {lock.lock_sha[:12]}, vector recipe "
                f"{lock.vector_recipe_sha256()[:12]}); encoder {enc.endpoint.shown} serving "
                f"{cfg.encoder.model}",
                f"  {container.note()}",
                f"  snapshot {manifest.snapshot}; labels_sha256 {manifest.labels_sha256}",
                f"  tasks {', '.join(manifest.tasks)}; framings {', '.join(manifest.framings)}",
                f"  manifest: {summary['rows']} rows, {summary['texts']} distinct texts "
                f"(state {summary['texts_by_side']['state']}, "
                f"action {summary['texts_by_side']['action']})",
                f"  already stored {len(texts) - len(todo)}, embedded {done} ({new} new vectors), "
                f"truncated {len(over)} (recorded, not embedded), encoder tokens {charged}, "
                f"batch {args.batch}, {time.monotonic() - started:.1f} s",
                "  fp16 round-trip min cosine "
                + ("-" if gate is None else f"{gate:.7f}")
                + f" (gate >= {feat.FP16_MIN_COSINE})",
            ]
        )
    )
    return EXIT_OK


def _features_export(args: argparse.Namespace, cfg: Config) -> int:
    from mesa_clm.learn import features as feat

    lock, _ = _live_lock()
    store = _existing_feature_store(cfg, lock)
    try:
        res = feat.export_npz(
            store, args.out, embed_model=lock.encoder.model, max_len=lock.encoder.max_len
        )
    except feat.FeatureStoreError as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    print(
        f"wrote {res.path}: {res.n} texts (keys sha1 <U40, vecs float16 [{res.n}, {store.dim}]); "
        f"fp16 round-trip min cosine {res.fp16_min_cosine:.7f} (gate >= {feat.FP16_MIN_COSINE})"
    )
    print(
        f"finetune.py: --embed-cache {res.path.parent} --embed-model {lock.encoder.model} "
        f"--max-len {lock.encoder.max_len}"
    )
    return EXIT_OK


def _features_project(args: argparse.Namespace, cfg: Config) -> int:
    if args.model == "clm-raw":
        print(
            "clm-raw scores raw 4096-d cosines at scale 100: there is nothing to project "
            "(offline scoring reads the stored vectors)"
        )
        return EXIT_OK
    from mesa_clm.clm.fingerprint import clm_model_fp
    from mesa_clm.clm.headproj import SIDES, HeadError, HeadProjector
    from mesa_clm.learn.features import FeatureStoreError
    from mesa_clm.serving import HEADS_DIR, serving_home

    lock, _ = _live_lock()
    store = _existing_feature_store(cfg, lock)
    fp = clm_model_fp(lock.model_spec(args.model))
    npz = serving_home() / HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    try:
        projector = HeadProjector.from_npz(npz)
    except HeadError as exc:
        raise UsageError(f"{exc} (the bootstrap exports it, plan §6.2)", EXIT_FAIL) from exc
    if projector.source_sha256 != lock.head.sha256:
        raise UsageError(
            f"{npz}: exported from {projector.source_sha256[:12] or 'an unknown checkpoint'}, "
            f"the serving lock pins {lock.head.sha256[:12]} (D5)",
            EXIT_FAIL,
        )
    texts = store.embedded_texts()
    try:
        for side in SIDES:
            for start in range(0, len(texts), _PROJECT_CHUNK):
                store.project(projector, fp, side, texts[start : start + _PROJECT_CHUNK])
    except FeatureStoreError as exc:
        raise UsageError(str(exc), EXIT_FAIL) from exc
    done = store.stats()["projections"].get(fp, {})
    print(
        f"{args.model} (clm_model_fp {fp}, head {lock.head.sha256[:12]}): "
        + ", ".join(f"{side} {done.get(side, 0)}" for side in SIDES)
        + f" projections of {len(texts)} stored vectors in {store.path}"
    )
    return EXIT_OK


def _features_stats(args: argparse.Namespace, cfg: Config) -> int:
    import re

    from mesa_clm.learn.features import DB_FILE, FeatureStore, FeatureStoreError

    root = expand_path(cfg.features.dir)
    out: dict[str, Any] = {"root": str(root), "live_encoder_fp": None, "stores": []}
    lock: Any = None
    try:
        lock, _ = _live_lock()
        out["live_encoder_fp"] = lock.encoder_fp
    except UsageError as exc:
        out["note"] = str(exc)
    fps = (
        sorted(
            p.name
            for p in root.iterdir()
            if re.fullmatch(r"[0-9a-f]{12}", p.name) and (p / DB_FILE).is_file()
        )
        if root.is_dir()
        else []
    )
    live_recipe = lock.vector_recipe_sha256() if lock is not None else None
    for fp in fps:
        try:
            info = FeatureStore(root, fp).stats()
        except FeatureStoreError as exc:
            out["stores"].append({"encoder_fp": fp, "error": str(exc)})
            continue
        if fp == out["live_encoder_fp"]:
            info["vector_recipe_matches_live"] = info.get("vector_recipe_sha256") == live_recipe
        out["stores"].append(info)
    if args.json:
        print(_json_out(out))
        return EXIT_OK
    lines = [f"feature stores under {root} (live encoder_fp {out['live_encoder_fp'] or '?'}):"]
    if "note" in out:
        lines.append(f"  {out['note']}")
    for s in out["stores"]:
        live_tag = " (live)" if s["encoder_fp"] == out["live_encoder_fp"] else ""
        if "error" in s:
            lines.append(f"  {s['encoder_fp']}{live_tag}: {s['error']}")
            continue
        tok = s["tokens"]
        gate = s["fp16_min_cosine"]
        lines.append(
            f"  {s['encoder_fp']}{live_tag}: {s['texts']} texts, {s['vectors']} vectors, "
            f"{s['truncated']} truncated, {tok['total']} tokens (max {tok['max']}), "
            "fp16 min cosine " + ("-" if gate is None else f"{gate:.7f}") + f", {s['bytes']} bytes"
        )
        if "vector_recipe_matches_live" in s:
            recipe = str(s.get("vector_recipe_sha256") or "none")[:12]
            lines.append(
                f"    vector recipe {recipe}: "
                + (
                    "the live lock's"
                    if s["vector_recipe_matches_live"]
                    else "NOT the live lock's (reads and writes are refused; rebuild)"
                )
            )
        for fp, sides in s["projections"].items():
            lines.append(
                f"    projections {fp}: "
                + ", ".join(f"{side} {n}" for side, n in sorted(sides.items()))
            )
    if not out["stores"]:
        lines.append("  none yet; run `mesa-clm features build --snapshot <labels snapshot>`")
    print("\n".join(lines))
    return EXIT_OK


def _cmd_features(args: argparse.Namespace, cfg: Config) -> int:
    if args.verb == "build":
        return _features_build(args, cfg)
    if args.verb == "export-npz":
        return _features_export(args, cfg)
    if args.verb == "project":
        return _features_project(args, cfg)
    return _features_stats(args, cfg)


def _features_parsers(sub: Any) -> None:
    fe = sub.add_parser(
        "features",
        help="the per-encoder feature cache: build, export-npz, project, stats (plan §5.2)",
    )
    fe_sub = fe.add_subparsers(dest="verb", required=True)
    b = fe_sub.add_parser(
        "build",
        help="embed every X1/X2 text of a labels snapshot the live encoder_fp's store lacks",
    )
    b.add_argument("--snapshot", required=True, help="frozen labels snapshot (`labels snapshot`)")
    b.add_argument(
        "--tasks", default=DEFAULT_FEATURE_TASKS, help=f"default {DEFAULT_FEATURE_TASKS}"
    )
    b.add_argument(
        "--framings",
        default=DEFAULT_FEATURE_FRAMINGS,
        help=f"default {DEFAULT_FEATURE_FRAMINGS} (X2 = joint4096@S1,joint4096@S1ns)",
    )
    b.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_FEATURE_BATCH,
        help="texts per /v1/embeddings request (default 1: vectors independent of batching)",
    )
    ex = fe_sub.add_parser(
        "export-npz", help="the store as CLM's TextCache npz for finetune.py --embed-cache"
    )
    ex.add_argument("--out", required=True, help="directory; writes choice_<model>_<max_len>.npz")
    pr = fe_sub.add_parser(
        "project", help="cache the pinned head's 512-d projections of every stored vector"
    )
    pr.add_argument("--model", choices=["clm-latest", "clm-raw"], default="clm-latest")
    st = fe_sub.add_parser("stats", help="every feature store under features.dir")
    st.add_argument("--json", action="store_true", help="print the stores as JSON")
    fe.set_defaults(func=_cmd_features)


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
        help="answer a run's pending groups: at a terminal curator labels (via=cli), "
        "otherwise an agent's answers (via=tool, DESIGN A2); interactive by default",
    )
    rv.add_argument("--run-id", required=True, help="run id or a unique prefix")
    rv.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="GROUP=OPTION_KEY|none",
        help="non-interactive: accept a candidate (or none of them) for a pending group id or "
        "prefix",
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
        "feedback",
        parents=[common],
        help="pick, reject or decline one group (at a terminal via=cli, otherwise via=tool)",
    )
    fb.add_argument("--group-id", required=True, help="group id or a unique prefix")
    fb.add_argument("--action", choices=["pick", "reject", "decline"], required=True)
    fb.add_argument(
        "--option-key",
        help="the candidate's key for --action pick (required; 'none': none of these)",
    )
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
    pru.add_argument(
        "--ttl-days", type=_non_negative_int, help="default: provenance.ttl_days (>= 0)"
    )
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
        type=_non_blank,
        default=os.environ.get("USER") or "mesa-clm",
        help="actor recorded on labels, runs and AVUs (default: $USER)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # The same two options after the verb; SUPPRESS keeps the sub-parser from clobbering the
    # values parsed before it with its own defaults.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provenance", default=argparse.SUPPRESS, help="sidecar DSN (default: config)"
    )
    common.add_argument(
        "--actor", type=_non_blank, default=argparse.SUPPRESS, help="actor recorded on labels"
    )

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
        "--trust-curator",
        action="store_true",
        help="import-anyjev: keep the file's curator labels as curator labels (needs a terminal; "
        "default: agent_pick at weight 0, DESIGN A2)",
    )
    la.add_argument(
        "--out",
        help=f"snapshot Parquet path (snapshot; default {DEFAULT_SNAPSHOT_DIR}/<today>.parquet)",
    )
    la.set_defaults(func=_cmd_labels)

    be = sub.add_parser(
        "bench",
        help="the bench: no-model controls and the MDE (M0); X1, the tier cells, X2 and the "
        "table (M2)",
    )
    be_sub = be.add_subparsers(dest="verb", required=True)
    be.set_defaults(func=_cmd_bench)
    m0 = {
        "baselines": "no-model controls: lookup_prob, novel-key, LOPO",
        "mde": "the minimum detectable effect simulation",
    }
    for verb, text in m0.items():
        sp = be_sub.add_parser(verb, parents=[common], help=text)
        sp.add_argument(
            "--out-dir", default=DEFAULT_OUT_DIR, help="results root (<out-dir>/<date>/)"
        )
        sp.add_argument("--date", help="results directory name (default: today, UTC)")
        sp.add_argument(
            "--snapshot",
            help=(
                "frozen labels snapshot to stamp the cells with "
                f"(default {DEFAULT_SNAPSHOT_DIR}/<date>.parquet, written when missing)"
            ),
        )
        sp.add_argument("--B", dest="B", type=int, default=DEFAULT_B, help="bootstrap replicates")
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed")
        if verb == "mde":
            sp.add_argument(
                "--n-sims",
                type=int,
                default=DEFAULT_N_SIMS,
                help="simulated datasets per grid point",
            )
        sp.add_argument("--force", action="store_true", help="replace an existing results file")
    registered = (
        f"the registered labels snapshot (default {REGISTERED_SNAPSHOT}; any other file is "
        "refused, none is ever written)"
    )
    fr = be_sub.add_parser(
        "framing",
        parents=[common],
        help="X1 framing A/B: x1.json, x1_items.parquet, x1.md; --decide --from replays one",
    )
    fr.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="results root (<out-dir>/<date>/)")
    fr.add_argument("--date", help="results directory name (default: today, UTC)")
    fr.add_argument("--snapshot", help=registered)
    fr.add_argument(
        "--tasks",
        help="term.fits,column.ontology_fits (default both; a subset is not the registered run)",
    )
    part = fr.add_mutually_exclusive_group()
    part.add_argument(
        "--both",
        dest="part",
        action="store_const",
        const="both",
        help="the full run and the nesting (default; the registered run)",
    )
    part.add_argument(
        "--full", dest="part", action="store_const", const="full", help="the full run only"
    )
    part.add_argument(
        "--nested", dest="part", action="store_const", const="nested", help="the nesting only"
    )
    fr.add_argument(
        "--latency",
        help="the live timing run's file (scripts/x1_latency.py) for the p50 latency",
    )
    fr.add_argument("--force", action="store_true", help="replace an existing x1.json")
    fr.add_argument(
        "--decide",
        action="store_true",
        help="replay and recompute the decision from the JSON (after the run, or --from FILE)",
    )
    fr.add_argument("--from", dest="from_", metavar="X1_JSON", help="an x1.json to replay")
    fr.set_defaults(part="both")
    rn = be_sub.add_parser(
        "run",
        parents=[common],
        help="the zero_shot and calibrated cells of every task: tiers.json",
    )
    rn.add_argument("--tiers", default="zero_shot,calibrated", help="zero_shot,calibrated (M2)")
    rn.add_argument("--loco", action="store_true", help="leave-one-card-out (required)")
    rn.add_argument(
        "--framing-from",
        metavar="X1_JSON",
        help="X1's outcome (default <out-dir>/<date>/x1.json), replayed and recomputed before use",
    )
    rn.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="results root (<out-dir>/<date>/)")
    rn.add_argument("--date", help="results directory name (default: today, UTC)")
    rn.add_argument("--snapshot", help=registered)
    rn.add_argument("--force", action="store_true", help="replace an existing tiers.json")
    x2 = be_sub.add_parser(
        "x2",
        parents=[common],
        help="X2 baselines: no-model controls, the PR #13 replica, AnyJev L2: x2.json",
    )
    x2.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="results root (<out-dir>/<date>/)")
    x2.add_argument("--date", help="results directory name (default: today, UTC)")
    x2.add_argument("--snapshot", help=registered)
    x2.add_argument("--force", action="store_true", help="replace an existing x2.json")
    tb = be_sub.add_parser("table", help="one markdown table over results files")
    tb.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="results root (<out-dir>/<date>/)")
    tb.add_argument("--date", help="every results file of <out-dir>/<date>/ (default: today)")
    tb.add_argument("--results", nargs="+", metavar="JSON", help="these results files instead")
    tb.add_argument("--out", help="write the table here instead of printing it")
    tb.add_argument("--force", action="store_true", help="replace an existing --out file")

    d = sub.add_parser(
        "doctor", help="what this host can run: pins, versions, stores, paths, serving"
    )
    d.add_argument(
        "--quick",
        action="store_true",
        help="skip the vendored-file hashing; outside serve mode also the serving-lock "
        "verification, in serve mode the slow probes (goldens, long input, parity, drift) "
        "unless --serve is given",
    )
    d.add_argument(
        "--serve",
        action="store_true",
        help="run every live serving probe (binds, the 401 matrix, health, models, the golden "
        "systemone call, encoder goldens, the long-input probe, systemone parity, drift; the "
        "light ones are automatic when both mesa-clm units are active) and fail when unreachable",
    )
    d.add_argument("--json", action="store_true", help="print the report as JSON")
    d.set_defaults(func=_cmd_doctor)
    _m1_parsers(sub, common)
    _features_parsers(sub)
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
    except ConfigError as exc:  # names the file and the line, never its content
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:  # anything else: its type only, never a message that may quote
        where = f"{args.config}: " if args.config else ""
        print(f"config error: {where}{type(exc).__name__}", file=sys.stderr)
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
