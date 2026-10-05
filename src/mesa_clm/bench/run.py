"""The M2 bench verbs behind the console script: ``bench framing``, ``bench run``, ``bench x2``
and ``bench table`` (``design/m2-analysis-plan.md`` §12, §14; plan §8 M2, §9).

Every verb reads its labels from the **registered snapshot** and nothing else
(:mod:`mesa_clm.bench.registered`): :func:`load_inputs` refuses another file (by sha256 and by
label content), loads the snapshot's rows into a scratch label store so the bench tasks are the
snapshot's by construction (the per-host sidecar is never read, and no snapshot is ever written),
builds the five bench tasks at the shipped policy's ``min_weight`` (D9) and checks their counts
against the ones published in M0 before any statistic. The vectors come from the feature store of
the serving lock this host runs, opened with its vector recipe; ``clm-latest`` scores through the
lock's pinned head (:func:`mesa_clm.bench.framing.scorers_for_lock`, K4).

* :func:`run_framing`: X1 (:func:`mesa_clm.bench.framing.run_x1`) to ``x1.json``,
  ``x1_items.parquet`` and ``x1.md``; registered only for the pre-registered invocation (both
  tasks, the full run and the nesting); a ``--latency`` file must be about the same labels and
  serving lock.
* :func:`run_tiers`: the zero_shot and calibrated cells of every task
  (:func:`mesa_clm.bench.cells.run_tier_cells`) to ``tiers.json``; X1's outcome only through
  :func:`mesa_clm.bench.framing.selections_from_json` (which replays the file, recomputes it from
  its items file and refuses an unregistered one), only from an ``x1.json`` whose labels, label
  content, framings lock and model fingerprints are this run's, and named in the file's notes by
  its sha256.
* :func:`run_x2`: the X2 baselines (:func:`mesa_clm.bench.x2.run_x2`) to ``x2.json``, the AnyJev
  L2 dump checked against its registered sha256, and a note saying whether the recomputed
  no-model controls equal the cells published in M0 (:func:`m0_comparison`, report-only).
* :func:`table`: one markdown table over results files; every number names its file.

**M4** (plan §8 M4; ``design/m4-analysis-plan.md``, the M4 registration
:func:`mesa_clm.bench.registered.current_m4`): :func:`run_x3` the probe cells of every task
(:func:`mesa_clm.bench.x3.run_x3`) to ``x3.json``; :func:`run_k2` the K2 verdict per task
(:func:`mesa_clm.bench.k2.evaluate_k2`) over the committed ``tiers.json`` and ``x2.json`` (by
their pinned sha256) and this date's ``x3.json`` to ``k2.json``; :func:`run_x4` the teacher
ablation (:func:`mesa_clm.bench.x4.run_x4`) from the teacher snapshot (whose silver rows must be
byte-for-byte the registered snapshot's, checked by their content digest) and the
silver-minus-Opus snapshot to ``x4.json``; :func:`run_e2e` the end-to-end measure, second wave
(:mod:`mesa_clm.bench.e2e`). Every M4 producer applies the M4 registration itself (R5) and
writes unregistered otherwise; ``bench table`` lists the new files beside the M2 ones.

Results files are never overwritten unless asked (``--force``): one run per file, committed as
produced (§14).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from mesa_clm.bench import registered as reg

if TYPE_CHECKING:
    from mesa_clm.bench.framing import X1Run
    from mesa_clm.bench.results import BenchResults
    from mesa_clm.bench.tasks.base import Task
    from mesa_clm.clm.fingerprint import ServingLock
    from mesa_clm.config import Config
    from mesa_clm.learn.features import FeatureStore
    from mesa_clm.learn.offline import OfflineScorer

__all__ = [
    "M4_FRAMINGS",
    "REGISTERED_TIERS",
    "BenchRunError",
    "Inputs",
    "Serving",
    "TeacherInputs",
    "load_inputs",
    "load_minus_opus",
    "load_teacher_inputs",
    "m0_comparison",
    "results_files",
    "run_e2e",
    "run_framing",
    "run_k2",
    "run_tiers",
    "run_x2",
    "run_x3",
    "run_x4",
    "serving",
    "snapshot_store",
    "table",
]

REGISTERED_TIERS: Final[tuple[str, ...]] = ("zero_shot", "calibrated")
# The framings whose texts the M4 verbs read from the manifest: every active framing after A1
# (F7, F9) and the X2 joint specs the probe specs ``joint4096@*`` reuse (rank_fit tasks), F7 for
# the closed choices.
M4_FRAMINGS: Final[str] = "F7,F9,X2"


class BenchRunError(RuntimeError):
    """A verb cannot run on what it was given (missing store, head, x1 file; mismatched
    identity). ``usage`` marks a problem with the command line rather than the inputs."""

    def __init__(self, message: str, *, usage: bool = False) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Inputs:
    """The registered labels: the snapshot's path, its two hashes and the five bench tasks built
    from its rows."""

    snapshot: Path
    labels_sha256: str
    labels_content_sha256: str
    tasks: dict[str, Task] = field(default_factory=dict)


def snapshot_store(snapshot: str | Path, path: str | Path) -> Any:
    """A label store at ``path`` holding exactly the rows of ``snapshot`` (every column, as
    ``labels snapshot`` wrote them), so the bench reads the frozen rows and nothing else (D30)."""
    from mesa_clm.provenance.labels import TABLE, LabelStore

    store = LabelStore(path)
    store.ensure_schema()
    cols = ", ".join(name for name, _ in store.columns())
    with store.connect() as con:
        con.execute(
            f"INSERT INTO {TABLE} ({cols}) SELECT {cols} FROM read_parquet(?)",  # noqa: S608
            [str(Path(snapshot))],
        )
    return store


def load_inputs(snapshot: str | Path | None, workdir: str | Path) -> Inputs:
    """The registered snapshot (``snapshot``, default the registration's) checked by sha256 and
    label content, its rows in a scratch store under ``workdir``, the five bench tasks at the
    shipped policy's ``min_weight`` and their counts checked against the published ones (§1).
    :class:`~mesa_clm.bench.registered.RegistrationError` for anything else."""
    from mesa_clm.bench.results import labels_content_sha256
    from mesa_clm.bench.tasks.neon import tasks_from_store

    path = Path(snapshot) if snapshot is not None else reg.current().snapshot_path()
    sha, content = reg.check_snapshot(path)
    store = snapshot_store(path, Path(workdir) / "labels.duckdb")
    if labels_content_sha256(store) != content:
        raise reg.RegistrationError(f"{path}: the scratch store does not hold the snapshot's rows")
    tasks = tasks_from_store(store)
    reg.check_tasks(tasks)
    return Inputs(path, sha, content, tasks)


@contextmanager
def scratch() -> Iterator[Path]:
    """A private temporary directory for the scratch label store, removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="mesa-clm-bench-") as tmp:
        yield Path(tmp)


@dataclass(frozen=True)
class Serving:
    """What scoring needs: the live lock, its feature store, the scorers (``None`` when the
    verb scores nothing through a head) and each model's fingerprint."""

    lock: ServingLock
    store: FeatureStore
    scorers: dict[str, OfflineScorer] | None
    fingerprints: dict[str, dict[str, str]]

    def require_scorers(self) -> dict[str, OfflineScorer]:
        if self.scorers is None:
            raise BenchRunError("no scorers: the head was not loaded")
        return self.scorers


def serving(cfg: Config, *, head: bool = True) -> Serving:
    """The serving lock this host runs (checked as the live provider checks it: the vendored
    schema's, and an installed copy only when it equals the checkout's), its feature store (which
    must exist and carry the lock's vector recipe) and, with ``head``, the scorers through the
    lock's pinned head (K4)."""
    from mesa_clm.bench.framing import fingerprints_for_lock, scorers_for_lock
    from mesa_clm.clm.headproj import HeadError, HeadProjector
    from mesa_clm.learn.features import FeatureStore
    from mesa_clm.providers.live import ProviderSetupError, _checked_lock, live_lock_path
    from mesa_clm.serving import HEADS_DIR, serving_home

    try:
        # the live provider's rule (K4): the vendored schema's lock, and an installed lock only
        # when it equals the checkout's
        lock = _checked_lock(live_lock_path(), None, explicit=False)
    except ProviderSetupError as exc:
        raise BenchRunError(str(exc)) from exc
    store = FeatureStore.for_lock(cfg, lock)
    if not store.exists():
        raise BenchRunError(
            f"no feature store at {store.path} (encoder_fp {lock.encoder_fp}); run "
            "`mesa-clm features build` first"
        )
    store.meta()  # refuses another encoder_fp, format or vector recipe before any read
    scorers = None
    if head:
        npz = serving_home() / HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
        try:
            projector = HeadProjector.from_npz(npz)
        except HeadError as exc:
            raise BenchRunError(f"{exc} (the bootstrap exports it, plan §6.2)") from exc
        scorers = scorers_for_lock(lock, projector)
    return Serving(lock, store, scorers, fingerprints_for_lock(lock))


def _task_ids(tasks: Sequence[str] | None, allowed: Sequence[str]) -> list[str]:
    if tasks is None:
        return list(allowed)
    out = [t.strip() for t in tasks if t.strip()]
    unknown = [t for t in out if t not in allowed]
    if unknown or not out:
        raise BenchRunError(
            f"--tasks takes {', '.join(allowed)}; got {', '.join(out) or 'nothing'}", usage=True
        )
    return [t for t in allowed if t in out]


def run_framing(
    cfg: Config,
    *,
    date: str,
    out_dir: str | Path,
    snapshot: str | Path | None = None,
    tasks: Sequence[str] | None = None,
    full: bool = True,
    nested: bool = True,
    latency: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, X1Run]:
    """``bench framing``: X1 on the registered snapshot (§12.1, §14.4). Anything but both tasks
    with the full run and the nesting is written unregistered (every cell exploratory)."""
    from mesa_clm.bench import framing
    from mesa_clm.learn.features import X1_FRAMINGS, X1_TASKS, manifest

    task_ids = _task_ids(tasks, X1_TASKS)
    with scratch() as tmp:
        inputs = load_inputs(snapshot, tmp)
        sv = serving(cfg)
        # the timing run must be about these labels and this serving lock (§11.3)
        timing = (
            framing.load_latency(
                latency, labels_sha256=inputs.labels_sha256, lock_sha=sv.lock.lock_sha
            )
            if latency is not None
            else None
        )
        man = manifest(inputs.snapshot, task_ids, X1_FRAMINGS)
        chosen = {name: t for name, t in inputs.tasks.items() if t.task_id in task_ids}
        run = framing.run_x1(
            chosen,
            man,
            sv.store,
            sv.require_scorers(),
            date=date,
            labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256,
            fingerprints=sv.fingerprints,
            latency=timing,
            full=full,
            nested=nested,
            B=reg.current().B,
            seed=reg.current().seed,
            alpha=reg.current().alpha,
        )
    path = framing.write_x1(run, out_dir, force=force)
    return path, run


def _check_x1_identity(x1_path: Path, inputs: Inputs, sv: Serving) -> None:
    """The ``x1.json`` a tier run takes X1's outcome from must be about the same labels, the
    same framings and the same models (D5, D30)."""
    from mesa_clm import framings as fr
    from mesa_clm.bench.framing import load_x1

    res = load_x1(x1_path)
    problems = []
    if res.labels_sha256 != inputs.labels_sha256:
        problems.append("labels_sha256")
    if res.labels_content_sha256 != inputs.labels_content_sha256:
        problems.append("labels_content_sha256")
    if res.x1.framings_lock_sha != fr.lock_sha():
        problems.append("framings lock")
    for model, fp in sv.fingerprints.items():
        if (res.x1.models.get(model) or {}).get("fingerprint") != fp:
            problems.append(f"{model} fingerprint")
    if problems:
        raise BenchRunError(
            f"{x1_path} is not about this run's inputs ({', '.join(problems)} differ)"
        )


def run_tiers(
    cfg: Config,
    *,
    date: str,
    out_dir: str | Path,
    framing_from: str | Path | None = None,
    snapshot: str | Path | None = None,
    tiers: Sequence[str] = REGISTERED_TIERS,
    force: bool = False,
) -> tuple[Path, BenchResults]:
    """``bench run --tiers zero_shot,calibrated --loco``: the tier cells of every task (§9,
    §12.3), X1's outcome from ``framing_from`` (default ``<out_dir>/<date>/x1.json``)."""
    from mesa_clm.bench.cells import TIERS, TextIndex, TierName, X1Selection, run_tier_cells
    from mesa_clm.bench.framing import X1ReproError, selections_from_json
    from mesa_clm.bench.results import results_path, write_results
    from mesa_clm.learn.features import CHOICE_TASKS, X1_TASKS, manifest

    unknown = [t for t in tiers if t not in REGISTERED_TIERS]
    if unknown or not tiers:
        later = {"probe": "M4", "head": "M7"}
        hint = "; ".join(f"{t} lands in {later[t]}" for t in unknown if t in later)
        raise BenchRunError(
            f"--tiers takes {', '.join(REGISTERED_TIERS)} in M2" + (f" ({hint})" if hint else ""),
            usage=True,
        )
    x1_path = Path(framing_from) if framing_from else results_path(out_dir, date, "x1")
    if not x1_path.is_file():
        raise BenchRunError(
            f"{x1_path}: no X1 results file; run `mesa-clm bench framing` first or pass "
            "--framing-from",
            usage=True,
        )
    chosen_tiers: list[TierName] = [t for t in TIERS if t in tiers]
    with scratch() as tmp:
        inputs = load_inputs(snapshot, tmp)
        sv = serving(cfg)
        _check_x1_identity(x1_path, inputs, sv)
        try:
            raw = selections_from_json(x1_path)  # replayed and recomputed from its items file
        except X1ReproError as exc:
            raise BenchRunError(f"{x1_path}: {exc}") from exc
        x1_sha = hashlib.sha256(x1_path.read_bytes()).hexdigest()
        selections = {task_id: X1Selection.of(**sel) for task_id, sel in raw.items()}
        missing = [t for t in X1_TASKS if t not in selections]
        if missing:
            raise BenchRunError(f"{x1_path} has no X1 outcome for {', '.join(missing)}")
        index = TextIndex.from_manifests(
            manifest(inputs.snapshot, X1_TASKS, "F4,F7,F9"),
            manifest(inputs.snapshot, CHOICE_TASKS, "F7"),
        )
        results = run_tier_cells(
            inputs.tasks,
            index,
            sv.store,
            sv.require_scorers(),
            sv.fingerprints,
            selections=selections,
            labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256,
            date=date,
            registered=True,  # the producer applies the registration (a subset of tiers is not)
            tiers=chosen_tiers,
            B=reg.current().B,
            seed=reg.current().seed,
        )
    note = f"X1's outcome: {x1_path} (sha256 {x1_sha}), replayed and recomputed before use."
    results = results.model_copy(update={"notes": [*results.notes, note]})
    return write_results(results, out_dir, force=force), results


def run_x2(
    cfg: Config,
    *,
    date: str,
    out_dir: str | Path,
    snapshot: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, BenchResults]:
    """``bench x2``: the X2 baselines (§10, §12.4) on the registered snapshot, the PR #13
    replica over the feature store and the registered AnyJev L2 dump."""
    from mesa_clm.bench import x2
    from mesa_clm.bench.cells import TextIndex
    from mesa_clm.bench.results import write_results
    from mesa_clm.learn.features import X1_TASKS, manifest

    with scratch() as tmp:
        inputs = load_inputs(snapshot, tmp)
        sv = serving(cfg, head=False)
        dump_path = reg.current().anyjev_path()
        reg.check_anyjev(dump_path)
        dump = x2.load_anyjev(dump_path)
        index = TextIndex.from_manifests(manifest(inputs.snapshot, X1_TASKS, "X2"))
        results = x2.run_x2(
            inputs.tasks,
            labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256,
            date=date,
            index=index,
            store=sv.store,
            fingerprint=sv.lock.fingerprint("clm-latest").as_dict(),
            anyjev=dump,
            registered=True,
            B=reg.current().B,
            seed=reg.current().seed,
        )
    results = results.model_copy(
        update={"notes": [*results.notes, m0_comparison(results, reg.current().published_path())]}
    )
    return write_results(results, out_dir, force=force), results


def m0_comparison(results: BenchResults, published: Path) -> str:
    """§10.1, report-only: whether X2's recomputed no-model controls
    (``<task>.baseline.lookup_prob``) are the cells ``bench baselines`` published in M0, field
    for field. A difference, or a published file that cannot be read, is recorded in the note,
    never repaired or refused (the comparison never fails the run)."""
    from mesa_clm.bench.results import load_results

    if not published.is_file():
        return f"M0 controls: no published file at {published} to compare with."
    try:
        ref = load_results(published)
    except (OSError, ValueError) as exc:
        return f"M0 controls: {published} could not be read ({type(exc).__name__}); not compared."
    same: list[str] = []
    differ: list[str] = []
    for key, cell in sorted(results.cells.items()):
        if not key.endswith(".baseline.lookup_prob"):
            continue
        want = ref.cells.get(key)
        if want is None:
            differ.append(f"{key} (not in the published file)")
            continue
        mine, theirs = cell.model_dump(mode="json"), want.model_dump(mode="json")
        fields = sorted(k for k in mine.keys() | theirs.keys() if mine.get(k) != theirs.get(k))
        if fields:
            differ.append(f"{key} ({', '.join(fields)})")
        else:
            same.append(key)
    if differ:
        return (
            f"M0 controls recomputed: {len(same)} equal to {published.name}; NOT equal: "
            + "; ".join(differ)
            + " (report-only, design/m2-analysis-plan.md §10.1)."
        )
    return (
        f"M0 controls recomputed: all {len(same)} equal to {published.name}, field for field "
        "(design/m2-analysis-plan.md §10.1)."
    )


def _cells_of(path: Path) -> tuple[str, Any]:
    """``(format, BenchResults)`` of a results file, or ``(format, None)`` for one without
    cells (``mde.json``, ``k2.json``, whose verdicts :func:`table` lists separately)."""
    from mesa_clm.bench import framing
    from mesa_clm.bench.results import FORMAT, load_results

    data = json.loads(path.read_text(encoding="utf-8"))
    fmt = str(data.get("format", ""))
    if fmt == framing.FORMAT:
        return fmt, framing.x1_cells(path)
    if fmt == FORMAT:
        return fmt, load_results(path)
    return fmt, None


def _k2_rows(path: Path, shown: str) -> list[str]:
    """The K2 verdict rows of a ``k2.json`` (``bench table``; empty for another format)."""
    from mesa_clm.bench import k2

    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("format", "")) != k2.FORMAT:
        return []
    res = k2.load_k2(path)
    rows = [
        "",
        "| file#task | best tier | best cite | verdict | reason | head_adds_nothing |",
        "|---|---|---|---|---|---|",
    ]
    for task, t in res.tasks.items():
        head = "" if t.head_adds_nothing is None else str(t.head_adds_nothing.head_adds_nothing)
        rows.append(
            f"| {shown}#{task} | {t.best or '-'} | {t.best_cite or '-'} | **{t.verdict}** | "
            f"{t.reason} | {head} |"
        )
    return rows


def _f(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def table(paths: Sequence[str | Path], *, root: str | Path | None = None) -> str:
    """``bench table``: one markdown table over the cells of ``paths`` (``x1.json``,
    ``tiers.json``, ``x2.json``, ``baselines.json``; files without cells are listed and
    skipped). Every row names its file and cell, so every number names the JSON it came from."""
    base = Path(root) if root is not None else None
    lines = [
        "# bench table",
        "",
        "Silver labels are four-model agreement, not truth. A cell is citable only if it is "
        "pre-registered, nested, not exploratory and passes the rest of the citation test "
        "(DESIGN, 'Citation test').",
        "",
        "| file#cell | n | acc | nll | ece | auroc [one-sided 95% bounds; 90% interval] | "
        "beats_lookup_novel | selection | pre_registered | exploratory |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    skipped: list[str] = []
    k2_rows: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        shown = (
            path.relative_to(base).as_posix() if base and path.is_relative_to(base) else str(path)
        )
        fmt, results = _cells_of(path)
        if results is None:
            verdicts = _k2_rows(path, shown)
            if verdicts:
                k2_rows.extend(verdicts)
            else:
                skipped.append(f"{shown} ({fmt or 'no format'}: no cells)")
            continue
        for key, cell in results.cells.items():
            m = cell.metrics
            b = cell.baselines
            ci = m.auroc_ci if m is not None else None
            auroc = (
                ""
                if ci is None or ci.point is None
                else (f"{_f(ci.point)} [{_f(ci.lower)}, {_f(ci.upper)}]")
            )
            lines.append(
                f"| {shown}#{key} | {cell.counts.n} | {_f(m.acc) if m else ''} | "
                f"{_f(m.nll) if m else ''} | {_f(m.ece) if m else ''} | {auroc} | "
                f"{'' if b is None else b.beats_lookup_novel} | {cell.selection} | "
                f"{cell.pre_registered} | {cell.exploratory} |"
            )
    lines += k2_rows
    if skipped:
        lines += ["", "Not tables of cells: " + "; ".join(skipped) + "."]
    return "\n".join(lines) + "\n"


def results_files(out_dir: str | Path, date: str) -> list[Path]:
    """The results JSON files of one date, sorted (``bench table --date``)."""
    folder = Path(out_dir) / date
    if not folder.is_dir():
        raise BenchRunError(f"{folder}: no results for {date}", usage=True)
    return sorted(p for p in folder.glob("*.json") if p.is_file())


# -- M4 ---------------------------------------------------------------------------------------------------


def _m4_index(inputs: Inputs) -> Any:
    """The manifest index of every text the probe specs read: the rank_fit tasks under
    ``F7,F9,X2`` and the closed choices under F7, from the snapshot the inputs name."""
    from mesa_clm.bench.cells import TextIndex
    from mesa_clm.learn.features import CHOICE_TASKS, X1_TASKS, manifest

    return TextIndex.from_manifests(
        manifest(inputs.snapshot, X1_TASKS, M4_FRAMINGS),
        manifest(inputs.snapshot, CHOICE_TASKS, "F7"),
    )


def run_x3(
    cfg: Config,
    *,
    date: str,
    out_dir: str | Path,
    snapshot: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, BenchResults]:
    """``bench x3``: the probe cells of every task on the registered snapshot (R1), to
    ``x3.json``. The verb exposes no option for B, the seed, the grid, the framings or the
    models (R5): the producer takes them from the registrations."""
    from mesa_clm.bench import x3
    from mesa_clm.bench.results import write_results

    _refuse_existing(out_dir, date, "x3", force)
    with scratch() as tmp:
        inputs = load_inputs(snapshot, tmp)
        sv = serving(cfg)
        index = _m4_index(inputs)
        results = x3.run_x3(
            inputs.tasks,
            index,
            sv.store,
            sv.require_scorers(),
            sv.fingerprints,
            labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256,
            date=date,
            registered=True,  # the producer applies the M4 registration itself
            B=reg.current().B,
            seed=reg.current().seed,
        )
    return write_results(results, out_dir, force=force), results


def _refuse_existing(out_dir: str | Path, date: str, name: str, force: bool) -> None:
    """One run per results file: refuse before anything is computed when
    ``<out_dir>/<date>/<name>.json`` exists and ``force`` is not given (``write_results``
    repeats the check at the end)."""
    from mesa_clm.bench.results import ResultsExist, results_path

    path = results_path(out_dir, date, name)
    if path.exists() and not force:
        raise ResultsExist(f"{path} exists: one run per results file (pass --force to replace)")


def _committed(name: str, path: Path, sha256: str) -> dict[str, str]:
    """``{"path", "sha256"}`` of a committed input after :func:`registered.check_committed_input`
    (a refusal for other bytes; an unpinned input is a deviation the caller records)."""
    got = reg.check_committed_input(name, path, sha256)
    return {"path": path.as_posix(), "sha256": got}


def run_k2(
    cfg: Config,
    *,
    date: str,
    out_dir: str | Path,
    x3_from: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, Any]:
    """``bench k2``: the K2 verdict per task (R2) over the committed ``tiers.json`` and
    ``x2.json`` (the M4 registration's pins, refused when their bytes differ) and ``x3.json``
    (default ``<out_dir>/<date>/x3.json``), to ``k2.json`` and ``k2.md``. ``cfg`` is unused but
    taken so every bench verb has one shape."""
    del cfg
    from mesa_clm.bench import k2
    from mesa_clm.bench.results import load_results, results_path

    m4 = reg.current_m4()
    x3_path = Path(x3_from) if x3_from else results_path(out_dir, date, "x3")
    if not x3_path.is_file():
        raise BenchRunError(
            f"{x3_path}: no X3 results file; run `mesa-clm bench x3` first or pass --x3-from",
            usage=True,
        )
    inputs = {
        "tiers": _committed("tiers.json", m4.tiers_path(), m4.tiers_sha256),
        "x2": _committed("x2.json", m4.x2_path(), m4.x2_sha256),
        "x3": {"path": x3_path.as_posix(), "sha256": reg.file_sha256(x3_path)},
    }
    deviations = [
        *reg.committed_input_deviations("tiers.json", m4.tiers_path(), m4.tiers_sha256),
        *reg.committed_input_deviations("x2.json", m4.x2_path(), m4.x2_sha256),
    ]
    results = k2.evaluate_k2(
        tiers=load_results(m4.tiers_path()),
        x2=load_results(m4.x2_path()),
        x3=load_results(x3_path),
        date=date,
        inputs=inputs,
        registered=not deviations,
        deviations=deviations,
        B=reg.current().B,
        seed=reg.current().seed,
        alpha=reg.current().alpha,
    )
    return k2.write_k2(results, out_dir, force=force), results


@dataclass(frozen=True)
class TeacherInputs:
    """The teacher snapshot (R3): its path and hashes, the teacher rows per task id (training
    only), and the digest of its non-teacher rows, which must equal the registered
    ``labels_content_sha256``."""

    snapshot: Path
    labels_sha256: str
    labels_content_sha256: str
    silver_content_sha256: str
    teacher: dict[str, Any] = field(default_factory=dict)


def _silver_content_sha256(snapshot: Path) -> str:
    """The content digest (``results.labels_content_sha256``'s rows and canon) of a snapshot's
    non-teacher rows: what must equal the registered snapshot's digest."""
    import duckdb

    from mesa_clm.bench.results import _content_sha256
    from mesa_clm.learn.labels import TEACHER_SOURCES
    from mesa_clm.provenance.labels import IDENTITY_COLUMNS

    order = ", ".join(IDENTITY_COLUMNS)
    sources = ", ".join(f"'{s}'" for s in TEACHER_SOURCES)
    target = str(snapshot).replace("'", "''")
    con = duckdb.connect()
    try:
        rows = [
            list(r)
            for r in con.execute(
                f"SELECT {order}, label_index, weight, card, product_code, leak_group, "  # noqa: S608
                f"fold_eligible, bench_card FROM read_parquet('{target}') "
                f"WHERE label_source NOT IN ({sources}) ORDER BY {order}"
            ).fetchall()
        ]
    finally:
        con.close()
    return _content_sha256(rows)


def load_teacher_inputs(
    snapshot: str | Path | None, workdir: str | Path, inputs: Inputs
) -> TeacherInputs:
    """The teacher snapshot (``snapshot``, default the M4 registration's pin; a usage error
    when none is pinned and none given): refused when its non-teacher rows are not the
    registered snapshot's rows (content digest), or when the pin names other bytes; its
    teacher rows per task id from a scratch store. Reads no silver label value."""
    from mesa_clm.bench.results import snapshot_content_sha256
    from mesa_clm.learn.labels import teacher_rows
    from mesa_clm.learn.teacher import TEACHER_TASKS

    m4 = reg.current_m4()
    path = Path(snapshot) if snapshot is not None else m4.teacher_snapshot_path()
    if path is None:
        raise BenchRunError(
            "no teacher snapshot: pass --teacher-snapshot (the M4 registration pins none yet)",
            usage=True,
        )
    if not path.is_file():
        raise reg.RegistrationError(f"{path}: no such teacher snapshot")
    sha = reg.file_sha256(path)
    if m4.teacher_labels_sha256 and sha != m4.teacher_labels_sha256:
        raise reg.RegistrationError(
            f"{path}: labels_sha256 {sha[:12]}… is not the pinned teacher snapshot's "
            f"{m4.teacher_labels_sha256[:12]}…"
        )
    silver = _silver_content_sha256(path)
    if silver != inputs.labels_content_sha256:
        raise reg.RegistrationError(
            f"{path}: its non-teacher rows (content digest {silver[:12]}…) are not the "
            f"registered snapshot's ({inputs.labels_content_sha256[:12]}…)"
        )
    store = snapshot_store(path, Path(workdir) / "teacher.duckdb")
    return TeacherInputs(
        path,
        sha,
        snapshot_content_sha256(path),
        silver,
        {task_id: teacher_rows(store, task_id) for task_id in TEACHER_TASKS},
    )


def load_minus_opus(
    snapshot: str | Path | None, workdir: str | Path
) -> tuple[Path, str, str, dict[str, Task]] | None:
    """The silver-minus-Opus snapshot (default the M4 pin; ``None`` when neither is given):
    its path, both hashes and the bench tasks built from it (the rebuilt silver, no model
    output); refused when the pin names other bytes."""
    from mesa_clm.bench.results import snapshot_content_sha256
    from mesa_clm.bench.tasks.neon import tasks_from_store

    m4 = reg.current_m4()
    path = Path(snapshot) if snapshot is not None else m4.minus_opus_snapshot_path()
    if path is None:
        return None
    if not path.is_file():
        raise reg.RegistrationError(f"{path}: no such silver-minus-Opus snapshot")
    sha = reg.file_sha256(path)
    if m4.minus_opus_labels_sha256 and sha != m4.minus_opus_labels_sha256:
        raise reg.RegistrationError(
            f"{path}: labels_sha256 {sha[:12]}… is not the pinned silver-minus-Opus snapshot's "
            f"{m4.minus_opus_labels_sha256[:12]}…"
        )
    store = snapshot_store(path, Path(workdir) / "minus_opus.duckdb")
    return path, sha, snapshot_content_sha256(path), tasks_from_store(store)


def run_x4(
    cfg: Config,
    *,
    date: str,
    out_dir: str | Path,
    snapshot: str | Path | None = None,
    teacher_snapshot: str | Path | None = None,
    minus_opus_snapshot: str | Path | None = None,
    corpus_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, BenchResults]:
    """``bench x4``: the teacher ablation on the two rank_fit tasks (R3) to ``x4.json``: the
    silver items of the registered snapshot, the teacher rows of the teacher snapshot (its
    manifest, which carries the teacher states, indexes the texts) and the silver-minus-Opus
    subset from its snapshot. ``corpus_dir`` (optional) is hashed
    (``registered.teacher_corpus_sha256``) and compared with the pin; without it the corpus is
    not checked (the teacher snapshot pins it; the live corpus may move)."""
    from mesa_clm.bench import x4
    from mesa_clm.bench.cells import TextIndex
    from mesa_clm.bench.results import write_results
    from mesa_clm.learn.features import X1_TASKS, manifest

    _refuse_existing(out_dir, date, "x4", force)
    corpus_sha = None if corpus_dir is None else reg.teacher_corpus_sha256(corpus_dir)
    with scratch() as tmp:
        inputs = load_inputs(snapshot, tmp)
        sv = serving(cfg)
        teacher = load_teacher_inputs(teacher_snapshot, tmp, inputs)
        minus = load_minus_opus(minus_opus_snapshot, tmp)
        index = TextIndex.from_manifests(manifest(teacher.snapshot, X1_TASKS, M4_FRAMINGS))
        chosen = {name: t for name, t in inputs.tasks.items() if t.task_id in x4.X4_TASKS}
        results = x4.run_x4(
            chosen,
            index,
            sv.store,
            sv.require_scorers(),
            sv.fingerprints,
            teacher=teacher.teacher,
            minus_opus=None if minus is None else minus[3],
            labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256,
            date=date,
            registered=True,  # the producer applies the M4 registration itself
            teacher_hashes=(teacher.labels_sha256, teacher.labels_content_sha256),
            minus_opus_hashes=None if minus is None else (minus[1], minus[2]),
            teacher_corpus_sha256=corpus_sha,
            B=reg.current().B,
            seed=reg.current().seed,
        )
    note = (
        f"teacher snapshot {teacher.snapshot} (labels_sha256 {teacher.labels_sha256}); its "
        f"non-teacher rows equal the registered snapshot (content digest "
        f"{teacher.silver_content_sha256[:12]}…)"
    )
    if minus is not None:
        note += f"; silver-minus-Opus snapshot {minus[0]} (labels_sha256 {minus[1]})"
    results = results.model_copy(update={"notes": [*results.notes, note + "."]})
    return write_results(results, out_dir, force=force), results


def run_e2e(cfg: Config, *, date: str, out_dir: str | Path, force: bool = False) -> Path:
    """``bench e2e --loco`` (R8): second-wave work; :func:`mesa_clm.bench.e2e.run_e2e` says so."""
    del cfg, date, out_dir, force
    from mesa_clm.bench import e2e

    try:
        e2e.run_e2e()
    except e2e.NotImplementedYet as exc:
        raise BenchRunError(str(exc), usage=True) from exc
    raise BenchRunError("bench e2e: unreachable", usage=True)  # pragma: no cover
