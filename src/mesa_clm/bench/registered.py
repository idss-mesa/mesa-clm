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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mesa_clm.bench.tasks.base import Task

__all__ = [
    "FINGERPRINTS",
    "FRAMINGS_LOCK_SHA",
    "REGISTERED",
    "ROOT",
    "SERVING_LOCK_SHA",
    "Registration",
    "RegistrationError",
    "TaskCounts",
    "anyjev_deviations",
    "check_anyjev",
    "check_snapshot",
    "check_task",
    "check_tasks",
    "current",
    "data_deviations",
    "file_sha256",
    "identity_deviations",
    "run_deviations",
    "task_set_deviations",
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
) -> list[str]:
    """How the framings and models a run used depart from the registration (empty: none): the
    framings lock sha it scored under (``None``: not part of the run) and, for each of
    ``models``, its D5 fingerprint (every field but those in ``ignore``; a missing fingerprint or
    a model the registration does not name is a deviation). The producers apply it themselves
    (X1, the tier cells, X2), and ``bench framing --decide --from`` applies it to what
    ``x1.json`` records (§12.6 (0))."""
    reg = current()
    out: list[str] = []
    if framings_lock_sha is not None and framings_lock_sha != reg.framings_lock_sha:
        out.append(
            f"framings lock_sha {_short(framings_lock_sha)} is not {_short(reg.framings_lock_sha)}"
        )
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
