"""The ``mesa-clm`` console script (plan §7.2).

Milestone M0 verbs: ``labels ingest-neon-eval|import-anyjev|snapshot|stats``, ``bench
baselines|mde`` and ``doctor [--quick] [--json]``. The annotation verbs (``plan``, ``annotate``,
``review``, ``feedback``, ``apply``, ``revert``, ``explain``), ``framings``, ``provenance``,
``features``, ``learn``, ``artifacts``, ``history``, ``serve`` and ``neon curate`` land with
their milestones; ``mesa-clm --help`` lists what exists. Ported from mesa-anyjev ``cli.py``
(``6159281``): the same argparse shape, exit codes and lazy imports inside the commands, so
``--help`` and ``doctor`` never import the modules they do not need.

Exit codes: ``EXIT_OK`` 0; ``EXIT_FAIL`` 1 when the verb ran and failed (a doctor ``fail``, an
empty store, a snapshot that no longer matches the store); ``EXIT_CONFIG`` 2 for a configuration
or usage problem (an unreadable config file, a missing ``--eval-root`` or one without
``results/validated.json`` and ``cards/``, a Postgres DSN in M0). A configuration validation
error is reported as ``config error: <field>: <problem>`` without the offending value, so a
mistyped ``MESA_CLM_<SECTION>__API_KEY`` never echoes the key (``Config`` also hides inputs in
its errors). The global ``--config``, ``--provenance`` and ``--actor`` are read before the verb
(``mesa-clm --provenance DSN labels stats``); ``--provenance`` and ``--actor`` are also accepted
after it, as mesa-anyjev's users type them.

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
from typing import Any

from pydantic import ValidationError

from mesa_clm import __version__
from mesa_clm.config import (
    Config,
    duckdb_path,
    expand_path,
    load_config,
    redact_dsn,
    set_active_config,
)

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

    report = doctor(cfg, quick=args.quick)
    if args.json:
        print(json.dumps(report.as_dict(), indent=1))
    else:
        print("\n".join(report.lines()))
    return EXIT_OK if report.ok else EXIT_FAIL


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

    d = sub.add_parser("doctor", help="what this host can run: pins, versions, stores, paths")
    d.add_argument(
        "--quick",
        action="store_true",
        help="skip the vendored-file hashing (and, from M1, serving)",
    )
    d.add_argument("--json", action="store_true", help="print the report as JSON")
    d.set_defaults(func=_cmd_doctor)
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
