"""The pre-registered inputs of the M2 runs (``design/m2-analysis-plan.md`` §1, §13): the frozen
labels snapshot, its two hashes, the item counts it must give, the policy weights, the bootstrap
settings, the framings and the models (the framings lock of G1, each model's D5 fingerprint under
the serving lock of G1) and X2's AnyJev dump. Everything here was fixed and committed before the
first run on real labels.

A run is *registered* only on exactly these inputs: ``bench framing``, ``bench run`` and ``bench
x2`` read the snapshot at :attr:`Registration.snapshot` and nothing else, refuse a file whose
sha256 is not :attr:`Registration.labels_sha256` (they never write a snapshot), build the bench
tasks from that file's rows (so the label store is the snapshot by construction, D30), refuse
label content other than :attr:`Registration.labels_content_sha256`, and check the realised
counts, item for item per card, against the counts ``bench baselines`` published in M0
(``bench/results/2026-09-29/baselines.json``) before any statistic. The counts are counts: no
label value beyond those published class counts is read here. The producers (X1, the tier cells,
X2) also compare the framings lock, the models' fingerprints and the AnyJev dump they ran on with
the registration (:func:`identity_deviations`, :func:`anyjev_deviations`): any other value makes
the run unregistered, whatever its caller says (§13.2).

The values are module data on one frozen object, :data:`REGISTERED`, read through
:func:`current` at call time, so the hermetic tests can stand a synthetic registration in for
it (they never touch the real snapshot's labels: the M2 pre-commitment rule).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from mesa_clm.bench.tasks.base import Task

__all__ = [
    "FINGERPRINTS",
    "FRAMINGS_LOCK_SHA",
    "FRAMINGS_LOCK_SHA_M4",
    "K2_CONSTANTS",
    "REGISTERED",
    "REGISTERED_M4",
    "ROOT",
    "SERVING_LOCK_SHA",
    "UNPINNED",
    "X3_GRID",
    "X4_TEACHER_ARMS",
    "Registration",
    "RegistrationError",
    "RegistrationM4",
    "TaskCounts",
    "anyjev_deviations",
    "check_anyjev",
    "check_committed_input",
    "check_snapshot",
    "check_task",
    "check_tasks",
    "committed_input_deviations",
    "current",
    "current_m4",
    "data_deviations",
    "file_sha256",
    "grid_deviations",
    "identity_deviations",
    "m4_identity_deviations",
    "pinned_snapshot_deviations",
    "run_deviations",
    "task_set_deviations",
    "teacher_corpus_sha256",
]

ROOT: Final[Path] = Path(__file__).resolve().parents[3]


class RegistrationError(ValueError):
    """An input that is not the pre-registered one: another snapshot, other label content,
    other counts or another ``min_weight``. Raised before any statistic."""


@dataclass(frozen=True)
class TaskCounts:
    """The published shape of one bench task: ``n`` items, the class counts (label index ->
    items), the items per card and the policy ``min_weight`` (D9)."""

    n: int
    class_counts: Mapping[int, int]
    per_card: Mapping[str, int]
    min_weight: float


# The framings and models of G1 (plan §1.4-§1.6), checked against the committed files by
# tests/unit/test_bench_registered.py: the lock_sha of `framings.lock.json` at G1 (F1, F4, F7, F9
# of every task, F7 active everywhere; DESIGN A1 later moved column.ontology_fits' active framing
# to F9, which rotated the checkout's lock_sha and no question_key, so the test rebuilds G1's lock
# from today's framings and the registered x1.json still replays) and, under
# `serving/serving.lock.json` (lock_sha below), the D5 fingerprint of each X1 model: the encoder
# c3b3d5e1a283 (A3), clm-latest through its pinned head (clm_model_fp 78be8c462b2e), clm-raw (the
# raw encoder vectors, 9f44b0301ee3), the vendored schema's sha256.
FRAMINGS_LOCK_SHA: Final[str] = "b432d32a7536c8f455098ae4a23139f6badbae7bc181a3a64853df3e80921eca"
SERVING_LOCK_SHA: Final[str] = "dd33f9fedbaee0129b09a21311227118461bed9326c942c2af7d0bd9b31b5237"
_ENCODER_FP: Final[str] = "c3b3d5e1a283"
_SCHEMA_SHA256: Final[str] = "52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335"
FINGERPRINTS: Final[Mapping[str, Mapping[str, str]]] = {
    model: {
        "encoder_fp": _ENCODER_FP,
        "clm_model_fp": clm_model_fp,
        "schema_sha256": _SCHEMA_SHA256,
        "serving_lock_sha": SERVING_LOCK_SHA,
    }
    for model, clm_model_fp in (("clm-latest", "78be8c462b2e"), ("clm-raw", "9f44b0301ee3"))
}


def _fingerprints() -> dict[str, dict[str, str]]:
    return {m: dict(fp) for m, fp in FINGERPRINTS.items()}


@dataclass(frozen=True)
class Registration:
    """The pre-registered M2 inputs (module docstring). ``snapshot`` is relative to the
    repository root; ``published`` names the file the counts come from; ``framings_lock_sha``
    is ``framings.lock.json``'s at G1 and ``fingerprints`` each model's D5 bundle
    (``encoder_fp``, ``clm_model_fp``, ``schema_sha256``, ``serving_lock_sha``) under the
    serving lock of G1."""

    snapshot: str
    labels_sha256: str
    labels_content_sha256: str
    published: str
    counts: Mapping[str, TaskCounts] = field(default_factory=dict)
    anyjev_dump: str = "bench/baselines/anyjev_l2_2026-09-29.json"
    anyjev_sha256: str = "557c53b7284749cc42f5a9ff1056f93721e891ff40b80713572e8457207c6483"
    B: int = 2000
    seed: int = 0
    alpha: float = 0.05
    framings_lock_sha: str = FRAMINGS_LOCK_SHA
    fingerprints: Mapping[str, Mapping[str, str]] = field(default_factory=_fingerprints)

    @property
    def serving_lock_sha(self) -> str | None:
        """The serving lock every registered fingerprint names (``None`` if they disagree)."""
        shas = {fp.get("serving_lock_sha") for fp in self.fingerprints.values()}
        return next(iter(shas)) if len(shas) == 1 else None

    def snapshot_path(self, root: str | Path | None = None) -> Path:
        """The snapshot's path under ``root`` (default: this checkout)."""
        return _under(self.snapshot, root)

    def anyjev_path(self, root: str | Path | None = None) -> Path:
        """X2's AnyJev L2 dump under ``root`` (default: this checkout)."""
        return _under(self.anyjev_dump, root)

    def published_path(self, root: str | Path | None = None) -> Path:
        """The M0 results file the counts were published in (``bench baselines``' output),
        under ``root`` (default: this checkout)."""
        return _under(self.published, root)


def _under(name: str, root: str | Path | None) -> Path:
    path = Path(name)
    return path if path.is_absolute() else Path(root or ROOT) / path


_CARDS: Final[tuple[str, ...]] = (
    "DP1.10003.001.brd_countdata",
    "DP1.10003.001.brd_perpoint",
    "DP1.10022.001.bet_archivepooling",
    "DP1.10022.001.bet_expertTaxonomistIDProcessed",
    "DP1.10022.001.bet_fielddata",
    "DP1.10022.001.bet_parataxonomistID",
    "DP1.10022.001.bet_sorting",
)


def _per_card(*n: int) -> dict[str, int]:
    return dict(zip(_CARDS, n, strict=True))


# Every number below is `bench/results/2026-09-29/baselines.json`'s (cells
# `<task>.baseline.lookup_prob`: counts.n, counts.class_counts, diagnostics.per_fold_n,
# min_weight), published in M0; tests/unit/test_bench_registered.py checks them against that file.
REGISTERED: Registration = Registration(
    snapshot="bench/snapshots/2026-09-29.parquet",
    labels_sha256="aafd18f8cea7992c9a8e0d12ded5a60ae2534a6677e911c28c2b05deaa1b752c",
    labels_content_sha256="5c60a8a69cf71d90df776cc331ba1347a8f7f3f074e75dca6ee6f95efb987b2b",
    published="bench/results/2026-09-29/baselines.json",
    counts={
        "neon_annotate": TaskCounts(98, {0: 63, 1: 35}, _per_card(12, 20, 6, 21, 17, 10, 12), 0.5),
        "neon_aspect": TaskCounts(
            60, {0: 12, 1: 10, 2: 18, 3: 11, 4: 6, 5: 3}, _per_card(6, 9, 5, 13, 11, 7, 9), 0.6
        ),
        "neon_ontology_fits": TaskCounts(
            190, {0: 76, 1: 114}, _per_card(21, 27, 17, 44, 28, 21, 32), 0.5
        ),
        "neon_term_fits": TaskCounts(
            285, {0: 86, 1: 199}, _per_card(37, 40, 41, 48, 45, 33, 41), 0.5
        ),
        "neon_value_kind": TaskCounts(
            278, {0: 133, 1: 62, 2: 24, 3: 59}, _per_card(34, 38, 35, 46, 47, 35, 43), 0.5
        ),
    },
)


def current() -> Registration:
    """The registration in force (:data:`REGISTERED`; a test may stand another in)."""
    return REGISTERED


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes (``labels_sha256`` of a snapshot, D30)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_snapshot(path: str | Path) -> tuple[str, str]:
    """``(labels_sha256, labels_content_sha256)`` of ``path`` after checking both against the
    registration; :class:`RegistrationError` otherwise (or for a missing file). Reads the
    snapshot's bytes and its label content digest, no label value."""
    from mesa_clm.bench.results import snapshot_content_sha256

    reg = current()
    file = Path(path)
    if not file.is_file():
        raise RegistrationError(f"{file}: no such labels snapshot")
    sha = file_sha256(file)
    if sha != reg.labels_sha256:
        raise RegistrationError(
            f"{file}: labels_sha256 {sha[:12]}… is not the pre-registered snapshot's "
            f"{reg.labels_sha256[:12]}… ({reg.snapshot}); M2 runs only on that file"
        )
    content = snapshot_content_sha256(file)
    if content != reg.labels_content_sha256:
        raise RegistrationError(
            f"{file}: labels_content_sha256 {content[:12]}… is not the pre-registered "
            f"{reg.labels_content_sha256[:12]}…"
        )
    return sha, content


def check_anyjev(path: str | Path) -> str:
    """sha256 of X2's AnyJev L2 dump after checking it is the registered file (M0, frozen at
    G1); :class:`RegistrationError` otherwise. Hashes bytes only."""
    file = Path(path)
    if not file.is_file():
        raise RegistrationError(f"{file}: no such AnyJev L2 dump")
    sha = file_sha256(file)
    if sha != current().anyjev_sha256:
        raise RegistrationError(
            f"{file}: sha256 {sha[:12]}… is not the registered AnyJev L2 dump's "
            f"{current().anyjev_sha256[:12]}…"
        )
    return sha


def data_deviations(labels_sha256: str | None, labels_content_sha256: str | None) -> list[str]:
    """How the labels a run names depart from the registration (empty: none)."""
    reg = current()
    out: list[str] = []
    if labels_sha256 != reg.labels_sha256:
        out.append(f"labels_sha256 {str(labels_sha256)[:12]} is not {reg.labels_sha256[:12]}")
    if labels_content_sha256 != reg.labels_content_sha256:
        out.append(
            f"labels_content_sha256 {str(labels_content_sha256)[:12]} is not "
            f"{reg.labels_content_sha256[:12]}"
        )
    return out


def run_deviations(
    *,
    labels_sha256: str | None,
    labels_content_sha256: str | None,
    B: int,
    seed: int,
    alpha: float | None = None,
) -> list[str]:
    """How a run's labels and bootstrap settings depart from the registration (empty: none):
    :func:`data_deviations` plus ``B``, ``seed`` and (when the run has one) ``alpha``. The tier
    and X2 producers apply it themselves, so a library call on other labels or with other
    settings is written unregistered whatever its caller says (§13.2)."""
    reg = current()
    out = data_deviations(labels_sha256, labels_content_sha256)
    if B != reg.B:
        out.append(f"B {B!r} is not {reg.B!r}")
    if seed != reg.seed:
        out.append(f"seed {seed!r} is not {reg.seed!r}")
    if alpha is not None and alpha != reg.alpha:
        out.append(f"alpha {alpha!r} is not {reg.alpha!r}")
    return out


def _short(value: object) -> str:
    return str(value)[:12]


def identity_deviations(
    *,
    framings_lock_sha: str | None,
    fingerprints: Mapping[str, Mapping[str, str] | None] | None,
    models: Sequence[str],
    ignore: Sequence[str] = (),
    registered_lock_sha: str | None = None,
) -> list[str]:
    """How the framings and models a run used depart from the registration (empty: none): the
    framings lock sha it scored under (``None``: not part of the run) and, for each of
    ``models``, its D5 fingerprint (every field but those in ``ignore``; a missing fingerprint or
    a model the registration does not name is a deviation). The producers apply it themselves
    (X1, the tier cells, X2), and ``bench framing --decide --from`` applies it to what
    ``x1.json`` records (§12.6 (0)). ``registered_lock_sha`` names another registered framings
    lock than M2's (the M4 producers pass A1's, :func:`m4_identity_deviations`)."""
    reg = current()
    want_lock = registered_lock_sha if registered_lock_sha is not None else reg.framings_lock_sha
    out: list[str] = []
    if framings_lock_sha is not None and framings_lock_sha != want_lock:
        out.append(f"framings lock_sha {_short(framings_lock_sha)} is not {_short(want_lock)}")
    for model in models:
        want = reg.fingerprints.get(model)
        got = None if fingerprints is None else fingerprints.get(model)
        if want is None:
            out.append(f"{model} is not a registered model")
            continue
        if got is None:
            out.append(f"{model}: no fingerprint (the registration's: {_fp(want, ignore)})")
            continue
        keys = sorted((set(want) | set(got)) - set(ignore))
        differ = [k for k in keys if got.get(k) != want.get(k)]
        if differ:
            out.append(
                f"{model} fingerprint: "
                + ", ".join(
                    f"{k} {_short(got.get(k))} (registered {_short(want.get(k))})" for k in differ
                )
            )
    return out


def _fp(fp: Mapping[str, str], ignore: Sequence[str]) -> str:
    return ", ".join(f"{k} {_short(v)}" for k, v in sorted(fp.items()) if k not in ignore)


def anyjev_deviations(sha256: str | None) -> list[str]:
    """How X2's AnyJev L2 dump departs from the registration (``None``: no dump given)."""
    want = current().anyjev_sha256
    if sha256 is None:
        return [f"no AnyJev L2 dump (the registered one: sha256 {_short(want)})"]
    if sha256 != want:
        return [f"AnyJev L2 dump sha256 {_short(sha256)} is not {_short(want)}"]
    return []


def task_set_deviations(names: Sequence[str]) -> list[str]:
    """The registered tasks a run leaves out (empty: none). A registration without counts (a
    test's stand-in) names no tasks and checks nothing."""
    missing = sorted(set(current().counts) - set(names))
    return [f"tasks {missing} are not run"] if missing else []


def check_task(name: str, task: Task) -> None:
    """One bench task built from the registered snapshot must have exactly its published shape:
    ``n``, class counts, items per card and ``min_weight``; :class:`RegistrationError` otherwise,
    and for a task the registration publishes no counts for (while it publishes some). A
    registration without counts (a test's stand-in) checks nothing. Counts only."""
    reg = current()
    if not reg.counts:
        return
    want = reg.counts.get(name)
    if want is None:
        raise RegistrationError(f"{name}: the registration publishes no counts for this task")
    per_card: dict[str, int] = {}
    for card in task.cards:
        per_card[card] = per_card.get(card, 0) + 1
    got = TaskCounts(
        len(task.items),
        task.class_counts(),
        dict(sorted(per_card.items())),
        float(task.meta.get("min_weight", -1.0)),
    )
    if got.min_weight != want.min_weight:
        raise RegistrationError(
            f"{name}: min_weight {got.min_weight} is not the shipped policy's "
            f"{want.min_weight} (D9)"
        )
    if got.n != want.n or dict(got.class_counts) != dict(want.class_counts):
        raise RegistrationError(
            f"{name}: {got.n} items {dict(got.class_counts)} are not the published "
            f"{want.n} {dict(want.class_counts)} ({reg.published})"
        )
    if dict(got.per_card) != dict(sorted(want.per_card.items())):
        raise RegistrationError(
            f"{name}: the items per card are not the published ones ({reg.published})"
        )


def check_tasks(tasks: Mapping[str, Task]) -> None:
    """The bench tasks built from the registered snapshot must have exactly the published shape:
    the same tasks, ``n``, class counts, items per card and ``min_weight`` (:func:`check_task`).
    Counts only; the first difference raises :class:`RegistrationError` (before any statistic)."""
    reg = current()
    missing = sorted(set(reg.counts) - set(tasks))
    if missing:
        raise RegistrationError(f"the snapshot gives no items for {', '.join(missing)}")
    for name in reg.counts:
        check_task(name, tasks[name])


# -- M4: the learned tiers (plan §8 M4, K2; DESIGN D19, D20, D27; design/m4-analysis-plan.md) ----
#
# The M4 producers (``bench x3``, ``bench k2``, ``bench x4``, ``learn fit``) apply this
# registration themselves, exactly as the M2 producers apply :data:`REGISTERED`: the same
# snapshot (both hashes), counts, B/seed/α and model fingerprints come from :data:`REGISTERED`
# (through :func:`current`); M4 adds the framings lock of DESIGN A1 (the pre-run commit's
# ``framings.lock.json``; M2 keeps G1's), the active framing per task, the X3 grid, the committed
# M2 inputs K2 reads (by sha256), the K2 constants, X4's teacher weights and the teacher inputs.
# A pin that reads :data:`UNPINNED` is a placeholder the integrator fills after the teacher
# ingest (the corpus hash, the teacher snapshot, the silver-minus-Opus snapshot); until then a
# producer that needs it writes every cell unregistered (``pre_registered: false``,
# ``exploratory: true``) with the missing pin named among its deviations. Nothing here reads a
# label value.

# A pin the integrator has not filled yet (module comment above).
UNPINNED: Final[str] = ""

# ``framings.lock_sha()`` at the M4 pre-run commit: G1's lock rotated by DESIGN A1
# (column.ontology_fits' active framing F7 -> F9, no question_key moved); the lock M4 scores
# under (tests/unit/test_m4_plan_constants.py holds it to the checkout).
FRAMINGS_LOCK_SHA_M4: Final[str] = (
    "7c93cc3e0ff6c26918a631cb787ceb6623dfd797fa1bb90cc353e956c1856920"
)

# The active framing per task after A1 (``framings.ACTIVE``), the framing every probe cell
# serves (R1: the framing is X1's decision, not an X3 axis).
ACTIVE_FRAMINGS_M4: Final[Mapping[str, str]] = {
    "column.annotate": "F7",
    "column.aspect": "F7",
    "column.ontology_fits": "F9",
    "term.fits": "F7",
    "avu.value_kind": "F7",
}

# The X3 grid (plan §5.3, R1): probe specs per shape, the fitters with their hyperparameter
# grids (declared order = tie order; values only, the key name is ``learn.linear``'s), the inner
# criterion and the floors. ``learn.probe.DEFAULT_GRID`` / ``learn.linear.GRIDS`` must equal it
# (:func:`grid_deviations`); a run on another grid is unregistered.
X3_GRID: Final[Mapping[str, Any]] = {
    "specs": {
        "rank_fit": (
            "lowdim.v1",
            "pair512.v1",
            "pair4096.v1",
            "joint4096@S1",
            "joint4096@S1ns",
        ),
        "choice": ("choice.state.v1", "choice.raw.v1"),
    },
    "spec_models": {
        "lowdim.v1": "clm-latest",
        "pair512.v1": "clm-latest",
        "pair4096.v1": "clm-raw",
        "joint4096@S1": "clm-raw",
        "joint4096@S1ns": "clm-raw",
        "choice.state.v1": "clm-latest",
        "choice.raw.v1": "clm-raw",
    },
    "fitters": ("logreg", "lda", "ridge"),
    "hypers": {
        "logreg": (1e-2, 1e-1, 1.0, 10.0),
        "lda": (0.1, 0.5, 0.9),
        "ridge": (1e-1, 1.0, 10.0),
    },
    "inner_criterion": "pooled OOF NLL of the uncalibrated probe probabilities over the training "
    "cards' inner LOCO; ties -> the first configuration in the grid's declared order",
    "calibration_floor": 100,
    "probe_floor": 40,
    "std_floor": 1e-8,
}

# K2 (plan §8, R2): the non-inferiority margin on accuracy against AnyJev L2, the clm-raw clause's
# margin, the ECE ceiling and its cluster upper bound, and the card-sign condition (the rule R
# sign test's fraction, minimum items per card and minimum counting cards).
K2_CONSTANTS: Final[Mapping[str, float | int]] = {
    "acc_margin": 0.02,
    "head_acc_margin": 0.01,
    "ece_max": 0.08,
    "ece_upper_max": 0.12,
    "sign_fraction": 0.8,
    "sign_min_items": 10,
    "min_clusters": 4,
}

# X4 (plan §5.6, R3): the teacher arms as (teacher weight, teacher_implicit weight); ``None`` is
# the off arm. The decision rule reads the (0.5, 0.3) arm; (0.3, 0.1) is reported.
X4_TEACHER_ARMS: Final[tuple[tuple[float, float] | None, ...]] = (None, (0.5, 0.3), (0.3, 0.1))
X4_DECISION_ARM: Final[tuple[float, float]] = (0.5, 0.3)


@dataclass(frozen=True)
class RegistrationM4:
    """The pre-registered M4 inputs (module comment). Paths are relative to the repository root;
    the snapshot, counts, bootstrap settings and model fingerprints are :data:`REGISTERED`'s
    (:func:`current`), never repeated here."""

    framings_lock_sha: str = FRAMINGS_LOCK_SHA_M4
    active: Mapping[str, str] = field(default_factory=lambda: dict(ACTIVE_FRAMINGS_M4))
    tiers_results: str = "bench/results/2026-10-03/tiers.json"
    tiers_sha256: str = "42501a293f45eacdd5db70fc7ee6d28acd5d9c726bcf3b79e5f657ac52f8684d"
    x2_results: str = "bench/results/2026-10-03/x2.json"
    x2_sha256: str = "76e07c34ddaed64e8987a2611c5af462457d7c398f49027533de3ea43cdb701d"
    grid: Mapping[str, Any] = field(default_factory=lambda: dict(X3_GRID))
    k2: Mapping[str, float | int] = field(default_factory=lambda: dict(K2_CONSTANTS))
    teacher_arms: tuple[tuple[float, float] | None, ...] = X4_TEACHER_ARMS
    # Pinned 2026-10-05T01:45Z after the label-free `labels ingest-teacher` (analysis plan §12.2):
    # sha256 over the sorted (filename, file sha256) pairs of the corpus files
    # (:func:`teacher_corpus_sha256`) as read at that neon-ducklake commit, the teacher snapshot
    # with both hashes, the silver-minus-Opus snapshot with both hashes.
    teacher_corpus_sha256: str = "d90387928bed0a9c196a8efc4dce3cbf40b1e2e8cc879bfb50c5e3451b208785"
    neon_ducklake_commit: str = "b1fa52a8d3ffc56173a09794253ad86b0fd10111"
    teacher_snapshot: str = "bench/snapshots/2026-10-04-teacher.parquet"
    teacher_labels_sha256: str = "397f98d4b28c1cbff951a0797a7c3dca73dfe5e21b7e59262ca79bbf7906153f"
    teacher_labels_content_sha256: str = (
        "deb0dc46e17bdf7879f11135cec8090f58f12f7ed50f114037852e93a79bb07b"
    )
    minus_opus_snapshot: str = "bench/snapshots/2026-10-04-minus-opus.parquet"
    minus_opus_labels_sha256: str = (
        "2cd529ffa43986585bcb79988fbe683febc137d310ce34d9b94ace809a3676fe"
    )
    minus_opus_labels_content_sha256: str = (
        "d8e26a0c7ac3b182e00741715d22e9e5a84c47ea898bf1d086a77c641acaa62c"
    )

    def tiers_path(self, root: str | Path | None = None) -> Path:
        return _under(self.tiers_results, root)

    def x2_path(self, root: str | Path | None = None) -> Path:
        return _under(self.x2_results, root)

    def teacher_snapshot_path(self, root: str | Path | None = None) -> Path | None:
        return None if not self.teacher_snapshot else _under(self.teacher_snapshot, root)

    def minus_opus_snapshot_path(self, root: str | Path | None = None) -> Path | None:
        return None if not self.minus_opus_snapshot else _under(self.minus_opus_snapshot, root)


REGISTERED_M4: RegistrationM4 = RegistrationM4()


def current_m4() -> RegistrationM4:
    """The M4 registration in force (:data:`REGISTERED_M4`; a test may stand another in)."""
    return REGISTERED_M4


def teacher_corpus_sha256(corpus_dir: str | Path, pattern: str = "*.validated.json") -> str:
    """The teacher corpus content hash the M4 registration pins (R5): sha256 of
    ``json.dumps(sorted((filename, sha256(file bytes)) pairs))`` over the files matching
    ``pattern`` directly under ``corpus_dir`` (default separators, so the pin reproduces from
    the formula alone). Hashes bytes only."""
    folder = Path(corpus_dir).expanduser()
    pairs = sorted((p.name, file_sha256(p)) for p in folder.glob(pattern) if p.is_file())
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()


def m4_identity_deviations(
    *,
    framings_lock_sha: str | None,
    fingerprints: Mapping[str, Mapping[str, str] | None] | None,
    models: Sequence[str],
    ignore: Sequence[str] = (),
) -> list[str]:
    """:func:`identity_deviations` against the M4 registration: the framings lock of A1 and the
    same model fingerprints as M2 (the serving lock of G1 is unchanged)."""
    return identity_deviations(
        framings_lock_sha=framings_lock_sha,
        fingerprints=fingerprints,
        models=models,
        ignore=ignore,
        registered_lock_sha=current_m4().framings_lock_sha,
    )


def committed_input_deviations(name: str, path: str | Path | None, sha256: str) -> list[str]:
    """How a committed input a producer reads (``tiers.json``, ``x2.json``, ``x3.json``)
    departs from its pin (empty: none): a missing file, or bytes whose sha256 is not ``sha256``.
    An :data:`UNPINNED` pin is itself a deviation."""
    if not sha256:
        return [f"{name} is not pinned by the M4 registration"]
    if path is None or not Path(path).is_file():
        return [f"{name}: no such committed input ({path})"]
    got = file_sha256(path)
    if got != sha256:
        return [f"{name} sha256 {_short(got)} is not the pinned {_short(sha256)}"]
    return []


def check_committed_input(name: str, path: str | Path, sha256: str) -> str:
    """sha256 of a committed input after checking it is the pinned file;
    :class:`RegistrationError` for a missing file or other bytes (the refusal of the M4 verbs,
    as M2's :func:`check_anyjev`). An unpinned input is checked to exist only and its sha256
    returned (the producer records the missing pin as a deviation)."""
    file = Path(path)
    if not file.is_file():
        raise RegistrationError(f"{file}: no such committed input ({name})")
    got = file_sha256(file)
    if sha256 and got != sha256:
        raise RegistrationError(
            f"{file}: sha256 {_short(got)}… is not the pinned {name} ({_short(sha256)}…); the "
            "M4 verbs read the committed M2 outputs and nothing else"
        )
    return got


def pinned_snapshot_deviations(
    name: str, labels_sha256: str | None, labels_content_sha256: str | None
) -> list[str]:
    """How a pinned M4 snapshot (the teacher or the silver-minus-Opus one) a run names departs
    from its pins (empty: none); unpinned hashes are deviations."""
    m4 = current_m4()
    want_sha = getattr(m4, f"{name}_labels_sha256")
    want_content = getattr(m4, f"{name}_labels_content_sha256")
    out: list[str] = []
    if not want_sha or not want_content:
        out.append(f"{name} snapshot is not pinned by the M4 registration")
        return out
    if labels_sha256 != want_sha:
        out.append(f"{name} labels_sha256 {_short(labels_sha256)} is not {_short(want_sha)}")
    if labels_content_sha256 != want_content:
        out.append(
            f"{name} labels_content_sha256 {_short(labels_content_sha256)} is not "
            f"{_short(want_content)}"
        )
    return out


def _hyper_values(hypers: Any) -> dict[str, tuple[float, ...]]:
    """``{fitter: values}`` of a grid's hyperparameter table, whether its entries are plain
    values or one-entry ``{name: value}`` mappings (``learn.linear.GRIDS``)."""
    out: dict[str, tuple[float, ...]] = {}
    for fitter, entries in dict(hypers).items():
        values: list[float] = []
        for h in entries:
            if isinstance(h, Mapping):
                if len(h) != 1:
                    raise RegistrationError(f"{fitter}: a hyperparameter entry must have one value")
                values.append(float(next(iter(h.values()))))
            else:
                values.append(float(h))
        out[str(fitter)] = tuple(values)
    return out


def grid_deviations(grid: Any, *, shape: str) -> list[str]:
    """How a probe grid (``learn.probe.Grid``: ``specs``, ``fitters``, ``hypers``) departs from
    the registered X3 grid for ``shape`` (empty: none). A variant cell's grid (``@latest``,
    ``@raw``: the specs of one model; ``@full``: one configuration) is compared by the producer
    against the restriction it declares, not here."""
    want = current_m4().grid
    out: list[str] = []
    specs = tuple(getattr(grid, "specs", ()))
    if specs != tuple(want["specs"][shape]):
        out.append(f"{shape} specs {list(specs)} are not {list(want['specs'][shape])}")
    fitters = tuple(getattr(grid, "fitters", ()))
    if fitters != tuple(want["fitters"]):
        out.append(f"fitters {list(fitters)} are not {list(want['fitters'])}")
    got = _hyper_values(getattr(grid, "hypers", {}))
    for fitter, values in dict(want["hypers"]).items():
        if got.get(fitter) != tuple(float(v) for v in values):
            out.append(f"{fitter} hyperparameters {got.get(fitter)} are not {tuple(values)}")
    return out
