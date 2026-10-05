"""Learned artifacts on disk: versions, the manifest, promotion, publish and pull (plan §5.5,
§8 M4; DESIGN D5, D8, D15, D18, D27; ``design/m4-analysis-plan.md`` C1).

**Layout.** ``<artifacts.dir>/<encoder_fp>/<clm_model_fp>/<framings lock sha8>/v<N>/`` holds one
immutable version: ``manifest.json``, ``calibrators.json`` (the
:class:`~mesa_clm.providers.tiered.ArtifactBundle` form) and ``probes/<question_key>.json``
(:class:`mesa_clm.learn.probe.ProbeArtifact`); ``CURRENT.json`` beside the versions says which
version and tier each task serves. One bundle per ``clm_model_fp``: a probe whose spec belongs
to ``clm-raw`` lives under the ``clm-raw`` fingerprint's directory and is served by a provider
of that model. A version is written to a temporary directory and moved into place with
``os.rename`` (which refuses an existing directory); an existing version is never rewritten
(:class:`ArtifactError`). Everything is owner-only (:mod:`mesa_clm.perms`).

**The manifest** (:class:`Manifest`, format :data:`FORMAT`) records the version, when it was
written, the fingerprints, the framings lock sha and the labels snapshot (both hashes), one
:class:`ManifestEntry` per question key (task, framing, tier, spec, fitter, hyper, the
calibrator's parameters, the file that holds the artifact, ``n_train``, the ``cite`` of the
pre-registered nested cell it stands on and that cell's identity: ``labels_sha256``,
``question_key``, fingerprint, and the ``fold_choices`` agreement count with this artifact's
configuration) and the sha256 of every file. :func:`load_version` is strict: a file whose hash
differs from the manifest's, a manifest whose fingerprints are not the live ones (K4), whose
framings lock is not the live one, or an entry whose question key is stale (not its task's active
framing's) is refused; ``strict=False`` drops such entries with a warning instead.

**Fitting** (``learn fit``, :func:`fit_version`) is the only place a probe or a calibrator is
fitted (D15): per task the probe configuration chosen by inner LOCO over all seven cards (the
``@full`` selection of the X3 plan, ``learn.probe.full_probe``, the fit on all items, its
calibrator from the seven-card OOF predictions) and the Platt/temperature calibrator of the
calibrated tier fitted on all items of the A1 arm (``learn.calibrate.fit_calibrator``). It
writes ``v<N>`` (``N`` = the next integer) and exports a copy of the manifest and the probes to
``bench/results/<date>/artifacts_v<N>/`` (:func:`export_copy`; committed, probes are small).
``learn fit`` applies the M4 registration itself (R5; ``cli._learn_fit``): the pinned
``tiers.json`` by sha256 (``registered.check_committed_input``), the framings lock of A1 and the
model fingerprints (``registered.m4_identity_deviations``), the registered X3 grid
(``registered.grid_deviations``) — any deviation is a refusal (exit 1, nothing written) — and
the manifest records ``registered: true`` with that identity and ``tasks_fitted`` (a ``--tasks``
subset is allowed: promotion is per task). A calibrated entry whose task has no pre-registered
nested calibrated cell in the pinned ``tiers.json`` (the closed choices, term.fits) is written
with ``cite: null``, which promotion refuses (not an error; the CLI says so per entry).
``learn fit`` never changes ``CURRENT.json``.

**Promotion** (``learn promote``, :func:`promote`) requires (i) the entry's ``cite`` to name a
pre-registered nested cell of that task and tier whose ``question_key``, fingerprint and
``labels_sha256`` equal the artifact's; (ii) the task's K2 verdict for that tier (``k2.json``)
to be ``a`` or ``b`` (a ``c`` tier is never promoted); (iii) when the task already serves a
promoted artifact, the new cell ≻ the current one on NLL (rule R on the two cells' pooled items)
and ≽ on accuracy within 0.01; (iv) a head to beat the probe (M7: refused here). It writes
``CURRENT.json`` (format :data:`CURRENT_FORMAT`).

**Transport.** :func:`publish` copies a version directory under another root (a local path; an
ssh or iRODS mount counts, the iRODS transport of plan §5.5 needs M3's ``irods_io``) and
:func:`pull` copies one back, re-hashing every file against the manifest (``--verify``) and
refusing a mismatch; neither overwrites an existing version.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mesa_clm import framings
from mesa_clm.bench import registered as reg
from mesa_clm.bench import stats
from mesa_clm.bench.results import BenchCell, BenchResults, load_results
from mesa_clm.clm.fingerprint import Fingerprint
from mesa_clm.perms import PRIVATE_DIR, private_dir, tighten_dir, tighten_file, write_private_text
from mesa_clm.policy import CITE_RE, CiteRef, parse_cite
from mesa_clm.providers.tiered import (
    ArtifactBundle,
    ArtifactError,
    PlattCalibrator,
    Promotion,
    ServedProbe,
    TemperatureCalibrator,
    load_calibrator,
    probe_from_json,
)
from mesa_clm.tasks import TASKS

if TYPE_CHECKING:
    from mesa_clm.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATORS_FILE",
    "CURRENT_FILE",
    "CURRENT_FORMAT",
    "EXPORT_DIR",
    "FORMAT",
    "LOCK_SHA8",
    "MANIFEST_FILE",
    "PROBES_DIR",
    "PROMOTABLE_VERDICTS",
    "PROMOTE_ACC_MARGIN",
    "VERSION_RE",
    "ArtifactError",
    "ArtifactLayout",
    "CellIdentity",
    "Current",
    "CurrentEntry",
    "FitResult",
    "FoldAgreement",
    "LoadedVersion",
    "Manifest",
    "ManifestEntry",
    "PromotionReport",
    "bundle_from_current",
    "calibrated_entry",
    "cell_identity",
    "entry_key",
    "export_copy",
    "file_sha256",
    "fit_version",
    "fold_agreement",
    "k2_verdict",
    "load_manifest",
    "load_version",
    "probe_entry",
    "promote",
    "publish",
    "pull",
    "verify_files",
    "version_name",
    "write_version",
]

FORMAT: Final = "mesa-clm/artifacts/1"
CURRENT_FORMAT: Final = "mesa-clm/artifacts-current/1"
MANIFEST_FILE: Final[str] = "manifest.json"
CALIBRATORS_FILE: Final[str] = "calibrators.json"
CURRENT_FILE: Final[str] = "CURRENT.json"
PROBES_DIR: Final[str] = "probes"
# ``bench/results/<date>/artifacts_v<N>/``: the committed copy of a version's manifest and probes.
EXPORT_DIR: Final[str] = "artifacts_v{n}"
# The framings lock directory is the lock sha's first 8 hex characters (plan §5.5 "lock8").
LOCK_SHA8: Final[int] = 8
VERSION_RE: Final[re.Pattern[str]] = re.compile(r"^v([1-9]\d*)$")
# Promotion rule (iii): the new cell must be ≽ the current one on accuracy within this margin
# (plan §5.5 "acc ≽ current within 0.01").
PROMOTE_ACC_MARGIN: Final[float] = 0.01
# Promotion rule (ii): the K2 verdicts that admit a tier (plan §8 K2; a ``c`` tier is killed).
PROMOTABLE_VERDICTS: Final[frozenset[str]] = frozenset({"a", "b"})
# The tiers an artifacts version holds in M4; heads join in M7.
Tier = Literal["calibrated", "probe"]
_TIERS: Final[tuple[str, ...]] = ("calibrated", "probe")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- the manifest ------------------------------------------------------------------------------


class FoldAgreement(_Frozen):
    """How many of a cited nested cell's outer folds chose this artifact's configuration
    (plan §4.7 "fold_choices agree ... in ≥5 of 7 folds"; the citation test's rule)."""

    agree: int = Field(ge=0)
    folds: int = Field(ge=0)


class CellIdentity(_Frozen):
    """The identity of the cell an entry cites, as the manifest records it (R4): the results
    path and cell key, the cell's ``labels_sha256`` and ``labels_content_sha256``, its
    ``question_key`` and fingerprint, its tier and bench task name, and the fold agreement."""

    path: str
    key: str
    task: str
    tier: str
    labels_sha256: str
    labels_content_sha256: str | None = None
    question_key: str | None = None
    fingerprint: dict[str, str] | None = None
    fold_agreement: FoldAgreement = Field(default_factory=lambda: FoldAgreement(agree=0, folds=0))


class ManifestEntry(_Frozen):
    """One artifact of a version, keyed by its question key (module docstring)."""

    question_key: str
    task_id: str
    framing_id: str
    tier: Tier
    file: str
    spec: str | None = None
    fitter: str | None = None
    hyper: dict[str, float] | None = None
    calibrator: dict[str, Any] = Field(default_factory=dict)
    n_train: int | None = None
    cite: str | None = None
    cell: CellIdentity | None = None

    @field_validator("question_key")
    @classmethod
    def _qk(cls, v: str) -> str:
        if len(v) != 16 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("question_key must be 16 lower-case hex characters")
        return v

    @field_validator("cite")
    @classmethod
    def _cite(cls, v: str | None) -> str | None:
        if v is not None and CITE_RE.match(v) is None:
            raise ValueError(
                "cite must be bench/results/<date>/<file>.json#<task>.<tier>.<framing>"
            )
        return v


class Manifest(_Frozen):
    """``manifest.json`` (module docstring). ``files`` maps every file of the version (relative
    path) to its sha256; ``entries`` the artifacts by question key."""

    format: Literal["mesa-clm/artifacts/1"] = FORMAT
    version: str
    created_at: str
    encoder_fp: str
    clm_model_fp: str
    framings_lock_sha: str
    labels_sha256: str
    labels_content_sha256: str | None = None
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    # ``learn fit`` applies the M4 registration (R5) and refuses on any deviation, so a version
    # it wrote records ``registered: true`` with the registration's identity (the framings lock
    # sha, the model fingerprints, the pinned input paths and hashes) and the tasks it fitted
    # (``--tasks`` may name a subset: promotion is per task).
    registered: bool = False
    registration: dict[str, Any] = Field(default_factory=dict)
    tasks_fitted: list[str] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        if VERSION_RE.match(v) is None:
            raise ValueError("version must be v<N> with N a positive integer")
        return v

    @property
    def number(self) -> int:
        return int(self.version[1:])

    def entries_for(self, task_id: str, tier: str | None = None) -> list[ManifestEntry]:
        return [
            e
            for e in self.entries.values()
            if e.task_id == task_id and (tier is None or e.tier == tier)
        ]


class CurrentEntry(_Frozen):
    """One task's production artifact (``CURRENT.json``): version, tier, question key, when it
    was promoted and the cell it was promoted on."""

    version: str
    tier: Literal["calibrated", "probe", "head"]
    question_key: str
    promoted_at: str
    cite: str | None = None


class Current(_Frozen):
    """``CURRENT.json`` (format :data:`CURRENT_FORMAT`): per task its :class:`CurrentEntry`."""

    format: Literal["mesa-clm/artifacts-current/1"] = CURRENT_FORMAT
    tasks: dict[str, CurrentEntry] = Field(default_factory=dict)


def entry_key(tier: str, question_key: str) -> str:
    """The manifest's key of an entry: ``<tier>:<question_key>`` (a calibrated and a probe
    artifact of one framing share the question key, plan §4.4; they never share a key here)."""
    return f"{tier}:{question_key}"


def version_name(n: int) -> str:
    if n < 1:
        raise ArtifactError("an artifacts version is a positive integer")
    return f"v{n}"


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes (``manifest.files``)."""
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


# -- the layout ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactLayout:
    """Where one ``(encoder_fp, clm_model_fp, framings lock)`` keeps its versions."""

    root: Path
    encoder_fp: str
    clm_model_fp: str
    framings_lock_sha: str

    @classmethod
    def for_fingerprint(
        cls, root: str | Path, fp: Fingerprint, framings_lock_sha: str | None = None
    ) -> ArtifactLayout:
        """The layout of a live fingerprint under ``root`` (``artifacts.dir``, ``~`` expanded)
        and the framings lock in force (default: the code's :func:`mesa_clm.framings.lock_sha`)."""
        return cls(
            Path(root).expanduser(),
            fp.encoder_fp,
            fp.clm_model_fp,
            framings_lock_sha or framings.lock_sha(),
        )

    @classmethod
    def from_config(
        cls, cfg: Config, fp: Fingerprint, framings_lock_sha: str | None = None
    ) -> ArtifactLayout:
        return cls.for_fingerprint(cfg.artifacts.dir, fp, framings_lock_sha)

    @property
    def model_dir(self) -> Path:
        return self.root / self.encoder_fp / self.clm_model_fp

    @property
    def dir(self) -> Path:
        return self.model_dir / self.framings_lock_sha[:LOCK_SHA8]

    def version_dir(self, n: int) -> Path:
        return self.dir / version_name(n)

    @property
    def current_path(self) -> Path:
        return self.dir / CURRENT_FILE

    def versions(self) -> list[int]:
        """The version numbers present, ascending."""
        if not self.dir.is_dir():
            return []
        out = []
        for p in self.dir.iterdir():
            m = VERSION_RE.match(p.name)
            if m is not None and p.is_dir():
                out.append(int(m.group(1)))
        return sorted(out)

    def next_version(self) -> int:
        found = self.versions()
        return (found[-1] + 1) if found else 1

    def read_current(self) -> Current | None:
        """``CURRENT.json`` when present; :class:`ArtifactError` when it does not parse."""
        path = self.current_path
        if not path.is_file():
            return None
        try:
            return Current.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ArtifactError(f"{path}: not a valid CURRENT.json ({exc})") from None

    def write_current(self, current: Current) -> Path:
        private_dir(self.dir)
        return write_private_text(self.current_path, _json_text(current.model_dump(mode="json")))

    def other_locks_with_current(self) -> list[str]:
        """Lock directories of this model other than this layout's that hold a ``CURRENT.json``
        (promoted artifacts of a rotated framings lock, which K4 refuses to serve silently)."""
        mine = self.framings_lock_sha[:LOCK_SHA8]
        if not self.model_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in self.model_dir.iterdir()
            if p.is_dir() and p.name != mine and (p / CURRENT_FILE).is_file()
        )


# -- cells and the manifest's identity record ----------------------------------------------------


def _fold_matches(choice: Mapping[str, Any], *, tier: str, config: Mapping[str, Any]) -> bool:
    """Whether one ``fold_choices`` entry chose ``config``: for a calibrated cell the arm
    (``framing``, ``model``); for a probe cell the ``model`` and the ``spec`` (plan §4.7 as the
    brief reads it for probes)."""
    if not choice.get("evaluated", True):
        return False
    if tier == "probe":
        return choice.get("model") == config.get("model") and choice.get("spec") == config.get(
            "spec"
        )
    return choice.get("framing") == config.get("framing") and choice.get("model") == config.get(
        "model"
    )


def fold_agreement(cell: BenchCell, *, config: Mapping[str, Any]) -> FoldAgreement:
    """The agreement of ``cell``'s ``fold_choices`` with ``config`` (:func:`_fold_matches`)
    over every outer fold the cell lists."""
    folds = cell.fold_choices or {}
    agree = sum(
        1
        for c in folds.values()
        if isinstance(c, Mapping) and _fold_matches(c, tier=cell.tier, config=config)
    )
    return FoldAgreement(agree=agree, folds=len(folds))


def cell_identity(path: str, cell: BenchCell, *, config: Mapping[str, Any]) -> CellIdentity:
    """The manifest's record of a cited cell (:class:`CellIdentity`): what the citation test
    compares with the artifact, copied from the cell (never a metric)."""
    return CellIdentity(
        path=path,
        key=cell.key,
        task=cell.task,
        tier=cell.tier,
        labels_sha256=cell.labels_sha256,
        labels_content_sha256=cell.labels_content_sha256,
        question_key=cell.question_key,
        fingerprint=dict(cell.fingerprint) if cell.fingerprint else None,
        fold_agreement=fold_agreement(cell, config=config),
    )


def _results_cell(results_root: Path, ref: CiteRef) -> tuple[BenchResults, BenchCell]:
    path = results_root / ref.path
    if not path.is_file():
        raise ArtifactError(f"{ref.path}: no such results file under {results_root}")
    try:
        results = load_results(path)
    except (ValueError, OSError) as exc:
        raise ArtifactError(f"{path}: not a results file ({type(exc).__name__})") from None
    key = f"{ref.task}.{ref.tier}.{ref.framing}"
    cell = results.cells.get(key)
    if cell is None:
        raise ArtifactError(f"{ref.path}: no cell {key}")
    return results, cell


# -- writing a version ---------------------------------------------------------------------------


def _write_tree(
    tmp: Path,
    *,
    bundle: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """The version's files under ``tmp`` (owner-only), returning path -> sha256."""
    files: dict[str, str] = {}
    cal_path = tmp / CALIBRATORS_FILE
    write_private_text(cal_path, _json_text(bundle))
    files[CALIBRATORS_FILE] = file_sha256(cal_path)
    if probes:
        private_dir(tmp / PROBES_DIR)
    for qk, data in probes.items():
        rel = f"{PROBES_DIR}/{qk}.json"
        target = tmp / rel
        write_private_text(target, _json_text(data))
        files[rel] = file_sha256(target)
    return files


def write_version(
    layout: ArtifactLayout,
    *,
    entries: Mapping[str, ManifestEntry],
    calibrators: Mapping[str, PlattCalibrator | TemperatureCalibrator],
    probes: Mapping[str, Mapping[str, Any]],
    labels_sha256: str,
    labels_content_sha256: str | None,
    notes: Sequence[str] = (),
    created_at: str | None = None,
    registered: bool = False,
    registration: Mapping[str, Any] | None = None,
    tasks_fitted: Sequence[str] = (),
) -> tuple[int, Path]:
    """Write the next version ``v<N>`` of ``layout`` (module docstring): ``calibrators.json``
    from ``calibrators`` (question key -> runtime calibrator), ``probes/<qk>.json`` from
    ``probes`` (question key -> the ``ProbeArtifact`` JSON), ``manifest.json`` from ``entries``
    (its ``file`` fields are set here), with ``registered``, ``registration`` and
    ``tasks_fitted`` recorded as given (:class:`Manifest`). Every entry must name a loaded
    artifact and vice versa.
    Returns ``(N, directory)``. Refuses (:class:`ArtifactError`) an entry without an artifact,
    an artifact without an entry, a probe whose spec/model/fingerprint disagree with the layout,
    and an existing ``v<N>`` (immutability: nothing is ever rewritten)."""
    keys = set(entries)
    held = {entry_key("calibrated", qk) for qk in calibrators} | {
        entry_key("probe", qk) for qk in probes
    }
    if keys != held:
        raise ArtifactError(
            f"manifest entries {sorted(keys - held)} have no artifact; artifacts "
            f"{sorted(held - keys)} have no entry"
        )
    for key, e in entries.items():
        if entry_key(e.tier, e.question_key) != key:
            raise ArtifactError(f"entry {e.tier}:{e.question_key} is filed under {key!r}")
    for qk, data in probes.items():
        fp = dict(data.get("fingerprint") or {})
        mine = {"encoder_fp": layout.encoder_fp, "clm_model_fp": layout.clm_model_fp}
        if {k: fp.get(k) for k in mine} != mine:
            raise ArtifactError(f"probe {qk}: fingerprint {fp} is not this layout's {mine} (K4)")
        if str(data.get("question_key")) != qk:
            raise ArtifactError(f"probe {qk}: its question_key is {data.get('question_key')!r}")
    n = layout.next_version()
    version = version_name(n)
    target = layout.version_dir(n)
    if target.exists():  # pragma: no cover - next_version skips existing ones
        raise ArtifactError(f"{target} exists: a version is never rewritten")
    private_dir(layout.root, tighten=False)
    private_dir(layout.dir)
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=layout.dir))
    os.chmod(tmp, PRIVATE_DIR)
    try:
        bundle = {
            "version": version,
            "encoder_fp": layout.encoder_fp,
            "clm_model_fp": layout.clm_model_fp,
            "framings_lock_sha": layout.framings_lock_sha,
            "calibrators": {
                qk: cal.model_dump(mode="json") for qk, cal in sorted(calibrators.items())
            },
        }
        files = _write_tree(tmp, bundle=bundle, probes=probes)
        filed = {
            key: e.model_copy(
                update={
                    "file": CALIBRATORS_FILE
                    if e.tier == "calibrated"
                    else f"{PROBES_DIR}/{e.question_key}.json"
                }
            )
            for key, e in sorted(entries.items())
        }
        manifest = Manifest(
            version=version,
            created_at=created_at or _now(),
            encoder_fp=layout.encoder_fp,
            clm_model_fp=layout.clm_model_fp,
            framings_lock_sha=layout.framings_lock_sha,
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            entries=filed,
            files=files,
            notes=list(notes),
            registered=registered,
            registration=dict(registration or {}),
            tasks_fitted=list(tasks_fitted),
        )
        write_private_text(tmp / MANIFEST_FILE, _json_text(manifest.model_dump(mode="json")))
        try:
            os.rename(tmp, target)  # fails if target exists: a version is never rewritten
        except OSError as exc:
            raise ArtifactError(f"{target}: cannot place the version ({exc})") from None
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return n, target


# -- loading ---------------------------------------------------------------------------------


def load_manifest(version_dir: str | Path) -> Manifest:
    """``manifest.json`` of a version directory, validated; :class:`ArtifactError` otherwise."""
    path = Path(version_dir) / MANIFEST_FILE
    try:
        return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise ArtifactError(f"{path}: no manifest") from None
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ArtifactError(f"{path}: not a valid manifest ({exc})") from None


def verify_files(version_dir: str | Path, manifest: Manifest) -> list[str]:
    """Every file the manifest lists, re-hashed: the problems found (empty: intact). A file in
    the directory the manifest does not list is a problem too (``probes/`` and the root only)."""
    base = Path(version_dir)
    problems: list[str] = []
    for rel, sha in sorted(manifest.files.items()):
        path = base / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
            continue
        got = file_sha256(path)
        if got != sha:
            problems.append(f"{rel}: sha256 {got[:12]} is not the manifest's {sha[:12]}")
    listed = set(manifest.files) | {MANIFEST_FILE}
    present = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
    for rel in sorted(present - listed):
        problems.append(f"{rel}: not in the manifest")
    return problems


@dataclass
class LoadedVersion:
    """A version read back: its manifest, the runtime calibrators and the served probes by
    question key, and the entries ``strict=False`` dropped (reason per key)."""

    manifest: Manifest
    directory: Path
    calibrators: dict[str, PlattCalibrator | TemperatureCalibrator] = field(default_factory=dict)
    probes: dict[str, ServedProbe] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)


def _stale(entry: ManifestEntry) -> str | None:
    """Why an entry's question key is stale: its task is unknown, its framing is not the
    task's active one, or its key is not that framing's (a rotated template)."""
    if entry.task_id not in framings.FRAMINGS:
        return f"unknown task {entry.task_id!r}"
    try:
        active = framings.active_framing(entry.task_id)
    except framings.FramingError as exc:
        return str(exc)
    if active.id != entry.framing_id:
        return f"framing {entry.framing_id} is not {entry.task_id}'s active framing {active.id}"
    if active.question_key != entry.question_key:
        return (
            f"question_key {entry.question_key} is not {entry.task_id}/{active.id}'s "
            f"{active.question_key} (the framing rotated: re-bench before use)"
        )
    return None


def load_version(
    layout: ArtifactLayout,
    n: int,
    *,
    live: Fingerprint | None = None,
    strict: bool = True,
    verify: bool = True,
) -> LoadedVersion:
    """Read ``v<n>`` of ``layout`` (module docstring). ``live`` is the live fingerprint the
    manifest must match (``encoder_fp``, ``clm_model_fp``; K4); the framings lock must be the
    layout's. ``verify`` re-hashes every file (always refused on mismatch, strict or not).
    ``strict`` refuses a fingerprint or lock mismatch and a stale entry; otherwise those are
    dropped with a warning and listed in ``dropped``."""
    directory = layout.version_dir(n)
    if not directory.is_dir():
        raise ArtifactError(f"{directory}: no such artifacts version")
    manifest = load_manifest(directory)
    if manifest.version != version_name(n):
        raise ArtifactError(f"{directory}: the manifest says {manifest.version}")
    if verify:
        problems = verify_files(directory, manifest)
        if problems:
            raise ArtifactError(
                f"{directory}: the files do not match the manifest (tampered or incomplete): "
                + "; ".join(problems)
            )
    problems = []
    if (manifest.encoder_fp, manifest.clm_model_fp) != (layout.encoder_fp, layout.clm_model_fp):
        problems.append(
            f"fingerprints ({manifest.encoder_fp}, {manifest.clm_model_fp}) are not the "
            f"layout's ({layout.encoder_fp}, {layout.clm_model_fp})"
        )
    if live is not None and (manifest.encoder_fp, manifest.clm_model_fp) != (
        live.encoder_fp,
        live.clm_model_fp,
    ):
        problems.append(
            f"fingerprints ({manifest.encoder_fp}, {manifest.clm_model_fp}) are not the live "
            f"({live.encoder_fp}, {live.clm_model_fp})"
        )
    if manifest.framings_lock_sha != layout.framings_lock_sha:
        problems.append(
            f"framings lock {manifest.framings_lock_sha[:12]} is not the live "
            f"{layout.framings_lock_sha[:12]}"
        )
    loaded = LoadedVersion(manifest, directory)
    if problems:
        message = f"{directory}: " + "; ".join(problems) + " (K4: re-bench before use)"
        if strict:
            raise ArtifactError(message)
        logger.warning("%s; every entry dropped (artifacts.strict is false)", message)
        loaded.dropped = {key: "; ".join(problems) for key in manifest.entries}
        return loaded
    bundle_data: Mapping[str, Any] | None = None
    for key, entry in manifest.entries.items():
        qk = entry.question_key
        why = _stale(entry)
        if why is not None:
            if strict:
                raise ArtifactError(f"{directory}: entry {key} is stale: {why}")
            logger.warning("%s: entry %s dropped: %s", directory, key, why)
            loaded.dropped[key] = why
            continue
        if entry.tier == "calibrated":
            if bundle_data is None:
                bundle_data = ArtifactBundle.load(directory / CALIBRATORS_FILE).model_dump(
                    mode="json"
                )
            cal = dict(bundle_data.get("calibrators", {})).get(qk)
            if cal is None:
                raise ArtifactError(f"{directory}: {CALIBRATORS_FILE} has no calibrator {qk}")
            loaded.calibrators[qk] = load_calibrator(cal)
        else:
            path = directory / entry.file
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"{path}: not a probe file ({exc})") from None
            probe = probe_from_json(data, version=manifest.version)
            if probe.question_key != qk or probe.task_id != entry.task_id:
                raise ArtifactError(f"{path}: the probe is not entry {qk}'s")
            if probe.spec != entry.spec:
                raise ArtifactError(f"{path}: spec {probe.spec} is not the manifest's {entry.spec}")
            loaded.probes[qk] = probe
    return loaded


def bundle_from_current(
    layout: ArtifactLayout,
    *,
    live: Fingerprint | None = None,
    strict: bool = True,
) -> ArtifactBundle | None:
    """The :class:`~mesa_clm.providers.tiered.ArtifactBundle` ``CURRENT.json`` names: every
    promoted task's artifact from its version (versions loaded once, :func:`load_version`),
    with the promotion table; ``None`` when there is no ``CURRENT.json`` or nothing promoted
    survives (``strict=False``). A promotion naming a version, tier or key the version does
    not hold is refused."""
    current = layout.read_current()
    if current is None or not current.tasks:
        return None
    versions: dict[int, LoadedVersion] = {}
    calibrators: dict[str, PlattCalibrator | TemperatureCalibrator] = {}
    probes: dict[str, ServedProbe] = {}
    promoted: dict[str, Promotion] = {}
    version_of: dict[str, str] = {}
    for task_id, entry in sorted(current.tasks.items()):
        m = VERSION_RE.match(entry.version)
        if m is None:
            raise ArtifactError(f"{layout.current_path}: {task_id} names version {entry.version!r}")
        n = int(m.group(1))
        if n not in versions:
            versions[n] = load_version(layout, n, live=live, strict=strict)
        loaded = versions[n]
        qk = entry.question_key
        key = entry_key(entry.tier, qk)
        if key in loaded.dropped:
            logger.warning("%s: promoted %s dropped (%s)", task_id, key, loaded.dropped[key])
            continue
        manifest_entry = loaded.manifest.entries.get(key)
        if manifest_entry is None or manifest_entry.task_id != task_id:
            raise ArtifactError(
                f"{layout.current_path}: {task_id} promotes {qk} of {entry.version}, which "
                "holds no such entry for that task"
            )
        if manifest_entry.tier != entry.tier:
            raise ArtifactError(
                f"{layout.current_path}: {task_id} promotes {qk} as {entry.tier}, the manifest "
                f"says {manifest_entry.tier}"
            )
        if entry.tier == "probe":
            probes[qk] = loaded.probes[qk]
        else:
            calibrators[qk] = loaded.calibrators[qk]
        version_of[qk] = entry.version
        promoted[task_id] = Promotion(
            tier=entry.tier, question_key=qk, version=entry.version, cite=entry.cite
        )
    if not promoted:
        return None
    newest = version_name(max(versions))
    return ArtifactBundle(
        version=newest,
        encoder_fp=layout.encoder_fp,
        clm_model_fp=layout.clm_model_fp,
        framings_lock_sha=layout.framings_lock_sha,
        calibrators=calibrators,
        probes=probes,
        promoted=promoted,
        versions={qk: v for qk, v in version_of.items() if v != newest},
    )


# -- export, publish, pull ----------------------------------------------------------------------


def export_copy(version_dir: str | Path, out_dir: str | Path, date: str) -> Path:
    """Copy the manifest and the probes of a version to ``<out_dir>/<date>/artifacts_v<N>/
    <clm_model_fp>/`` (``out_dir`` is ``bench/results``: committed next to the results, one
    bundle per served model; the calibrators are in the manifest's entries already). Refuses
    an existing export."""
    manifest = load_manifest(version_dir)
    target = Path(out_dir) / date / EXPORT_DIR.format(n=manifest.number) / manifest.clm_model_fp
    if target.exists():
        raise ArtifactError(f"{target} exists: one export per version")
    target.mkdir(parents=True)
    shutil.copyfile(Path(version_dir) / MANIFEST_FILE, target / MANIFEST_FILE)
    probes = Path(version_dir) / PROBES_DIR
    if probes.is_dir():
        shutil.copytree(probes, target / PROBES_DIR)
    return target


def _copy_version(source: Path, dest: Path, *, verify: bool) -> Path:
    """Copy ``source`` (a version directory) to ``dest`` through a temporary sibling, every
    file re-hashed against the manifest when ``verify``; never over an existing ``dest``."""
    if not source.is_dir():
        raise ArtifactError(f"{source}: no such artifacts version")
    manifest = load_manifest(source)
    if dest.exists():
        raise ArtifactError(f"{dest} exists: a version is never rewritten (pull another N)")
    private_dir(dest.parent.parent.parent.parent, tighten=False)
    private_dir(dest.parent)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    os.chmod(tmp, PRIVATE_DIR)
    try:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            rel = path.relative_to(source)
            target = tmp / rel
            if target.parent != tmp:
                private_dir(target.parent)
            shutil.copyfile(path, target)
            tighten_file(target)
        for d in tmp.rglob("*"):
            if d.is_dir():
                tighten_dir(d)
        if verify:
            problems = verify_files(tmp, manifest)
            if problems:
                raise ArtifactError(
                    f"{source}: the copied files do not match the manifest (refused): "
                    + "; ".join(problems)
                )
        try:
            os.rename(tmp, dest)
        except OSError as exc:
            raise ArtifactError(f"{dest}: cannot place the version ({exc})") from None
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def publish(layout: ArtifactLayout, n: int, to_root: str | Path) -> Path:
    """Copy ``v<n>`` to ``<to_root>/<encoder_fp>/<clm_model_fp>/<lock8>/v<n>/`` (plan §5.5),
    the manifest's hashes verified after the copy."""
    source = layout.version_dir(n)
    other = ArtifactLayout(
        Path(to_root).expanduser(), layout.encoder_fp, layout.clm_model_fp, layout.framings_lock_sha
    )
    return _copy_version(source, other.version_dir(n), verify=True)


def pull(layout: ArtifactLayout, n: int, from_root: str | Path, *, verify: bool = True) -> Path:
    """Copy ``v<n>`` from ``<from_root>/<encoder_fp>/<clm_model_fp>/<lock8>/`` into ``layout``,
    every file re-hashed against the manifest (``verify``; a mismatch is refused and nothing is
    placed). ``CURRENT.json`` is never pulled: promotion is a local decision."""
    other = ArtifactLayout(
        Path(from_root).expanduser(),
        layout.encoder_fp,
        layout.clm_model_fp,
        layout.framings_lock_sha,
    )
    return _copy_version(other.version_dir(n), layout.version_dir(n), verify=verify)


# -- promotion ---------------------------------------------------------------------------------


def k2_verdict(k2: Mapping[str, Any], task_id: str, tier: str) -> str | None:
    """The K2 verdict of ``tier`` for ``task_id`` as ``k2.json`` records it
    (:mod:`mesa_clm.bench.k2`: ``tasks[task_id]`` with ``best`` the best servable tier and
    ``verdict`` its K2 verdict; a task entry keyed otherwise is found by its ``task_id``). K2
    judges the best tier only, so a verdict exists iff ``best == tier``; another tier has none
    (``None``) and is never promoted. Nothing else in the file is read (no per-tier hook)."""
    tasks = k2.get("tasks")
    if not isinstance(tasks, Mapping):
        return None
    entry = tasks.get(task_id)
    if not isinstance(entry, Mapping):
        entry = next(
            (v for v in tasks.values() if isinstance(v, Mapping) and v.get("task_id") == task_id),
            None,
        )
    if not isinstance(entry, Mapping):
        return None
    best = entry.get("best", entry.get("best_tier"))
    if best == tier and entry.get("verdict") is not None:
        return str(entry["verdict"])
    return None


@dataclass(frozen=True)
class _Pooled:
    probs: np.ndarray
    labels: np.ndarray
    cards: list[str]


def _pooled(cell: BenchCell, *, what: str) -> dict[tuple[str, str], tuple[list[float], int, str]]:
    if not cell.items:
        raise ArtifactError(f"{what}: the cell carries no items (needed for rule R)")
    return {(i.target_sha256, i.option_key): (i.probs, i.label, i.card) for i in cell.items}


def _paired(new: BenchCell, current: BenchCell) -> tuple[_Pooled, _Pooled]:
    a = _pooled(new, what="the new cell")
    b = _pooled(current, what="the current cell")
    if set(a) != set(b):
        raise ArtifactError(
            "the new and the current cells pool different items (rule R needs a full join): "
            f"{len(set(a) ^ set(b))} differ"
        )
    keys = sorted(a)
    labels = np.asarray([a[k][1] for k in keys], dtype=np.int64)
    if any(b[k][1] != a[k][1] for k in keys):
        raise ArtifactError("the two cells disagree on an item's label")
    return (
        _Pooled(
            np.asarray([a[k][0] for k in keys], dtype=np.float64), labels, [a[k][2] for k in keys]
        ),
        _Pooled(
            np.asarray([b[k][0] for k in keys], dtype=np.float64), labels, [b[k][2] for k in keys]
        ),
    )


@dataclass
class PromotionReport:
    """What :func:`promote` checked, for the verb's output."""

    task_id: str
    tier: str
    version: str
    question_key: str
    cite: str
    k2_verdict: str
    replaced: CurrentEntry | None = None
    comparison: dict[str, Any] = field(default_factory=dict)
    current: Current | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "version": self.version,
            "question_key": self.question_key,
            "cite": self.cite,
            "k2_verdict": self.k2_verdict,
            "replaced": None if self.replaced is None else self.replaced.model_dump(mode="json"),
            "comparison": self.comparison,
        }


def _check_cell(
    results_root: Path, manifest: Manifest, entry: ManifestEntry, task_id: str, tier: str
) -> tuple[CiteRef, BenchCell]:
    """Promotion rule (i)."""
    if entry.cite is None:
        raise ArtifactError(f"{entry.question_key}: the entry cites no cell (rule i)")
    ref = parse_cite(entry.cite)
    if ref is None:  # pragma: no cover - the manifest validator checks the pattern
        raise ArtifactError(f"{entry.cite}: not a cite")
    _, cell = _results_cell(results_root, ref)
    problems: list[str] = []
    if cell.task_id != task_id:
        problems.append(f"the cell is {cell.task_id}'s, not {task_id}'s")
    if cell.tier != tier:
        problems.append(f"the cell is a {cell.tier} cell, not {tier}")
    if not (
        cell.loco and cell.pre_registered and cell.selection == "nested" and not cell.exploratory
    ):
        problems.append(
            "the cell is not a pre-registered nested LOCO cell "
            f"(loco={cell.loco}, pre_registered={cell.pre_registered}, "
            f"selection={cell.selection}, exploratory={cell.exploratory})"
        )
    if cell.question_key != entry.question_key:
        problems.append(
            f"question_key {cell.question_key} is not the artifact's {entry.question_key}"
        )
    fp = cell.fingerprint or {}
    mine = {"encoder_fp": manifest.encoder_fp, "clm_model_fp": manifest.clm_model_fp}
    if {k: fp.get(k) for k in mine} != mine:
        problems.append(f"fingerprint {fp} is not the artifact's {mine}")
    if cell.labels_sha256 != manifest.labels_sha256:
        problems.append(
            f"labels_sha256 {cell.labels_sha256[:12]} is not the artifact's "
            f"{manifest.labels_sha256[:12]}"
        )
    if problems:
        raise ArtifactError(f"{entry.cite} (rule i): " + "; ".join(problems))
    return ref, cell


def promote(
    layout: ArtifactLayout,
    *,
    version: int,
    task_id: str,
    tier: str | None,
    results_root: str | Path,
    k2: Mapping[str, Any],
    live: Fingerprint | None = None,
    now: str | None = None,
) -> PromotionReport:
    """Promote ``task_id``'s ``tier`` artifact of ``v<version>`` (module docstring, rules
    i-iv); ``tier`` ``None`` takes the version's probe for the task when it has one, else its
    calibrator. ``k2`` is the parsed ``k2.json``. Writes ``CURRENT.json`` and returns the
    report. Nothing is written when a rule fails (:class:`ArtifactError`)."""
    if task_id not in TASKS:
        raise ArtifactError(f"unknown task {task_id!r}")
    if tier == "head":
        raise ArtifactError("a head is promoted only once it beats the probe (rule iv, M7)")
    if tier is not None and tier not in _TIERS:
        raise ArtifactError(f"tier must be one of {_TIERS}, not {tier!r}")
    loaded = load_version(layout, version, live=live, strict=True)
    manifest = loaded.manifest
    candidates = manifest.entries_for(task_id, tier)
    if tier is None:
        probes = [e for e in candidates if e.tier == "probe"]
        candidates = probes or [e for e in candidates if e.tier == "calibrated"]
    if not candidates:
        raise ArtifactError(
            f"{manifest.version} holds no {tier or 'probe or calibrated'} artifact for {task_id}"
        )
    if len(candidates) > 1:
        raise ArtifactError(f"{manifest.version} holds {len(candidates)} {task_id} entries")
    entry = candidates[0]
    chosen = entry.tier
    root = Path(results_root)
    _, cell = _check_cell(root, manifest, entry, task_id, chosen)
    verdict = k2_verdict(k2, task_id, chosen)
    if verdict is None:
        raise ArtifactError(f"k2.json records no verdict for {task_id}'s {chosen} tier (rule ii)")
    if verdict not in PROMOTABLE_VERDICTS:
        raise ArtifactError(
            f"{task_id}'s {chosen} tier is K2({verdict}): never promoted (rule ii; a c tier is "
            "killed, plan §8 K2)"
        )
    current = layout.read_current() or Current()
    replaced = current.tasks.get(task_id)
    comparison: dict[str, Any] = {}
    if replaced is not None:
        if replaced.cite is None:
            raise ArtifactError(f"{task_id}: the current promotion cites no cell (rule iii)")
        cur_ref = parse_cite(replaced.cite)
        if cur_ref is None:  # pragma: no cover - CURRENT.json is written by promote
            raise ArtifactError(f"{replaced.cite}: not a cite")
        _, cur_cell = _results_cell(root, cur_ref)
        if (replaced.version, replaced.question_key, replaced.tier) == (
            manifest.version,
            entry.question_key,
            chosen,
        ):
            raise ArtifactError(f"{task_id}: {manifest.version}'s {chosen} is already promoted")
        new_p, cur_p = _paired(cell, cur_cell)
        registration = reg.current()
        rr = stats.rule_r(
            "nll",
            new_p.probs,
            cur_p.probs,
            new_p.labels,
            new_p.cards,
            B=registration.B,
            seed=registration.seed,
            alpha=registration.alpha,
        )
        ni = stats.non_inferior(
            "acc",
            new_p.probs,
            cur_p.probs,
            new_p.labels,
            new_p.cards,
            PROMOTE_ACC_MARGIN,
            B=registration.B,
            seed=registration.seed,
            alpha=registration.alpha,
        )
        comparison = {
            "current": replaced.model_dump(mode="json"),
            "n": len(new_p.labels),
            "nll_rule_r": _json_safe(rr.as_dict()),
            "acc_non_inferior": _json_safe(ni.as_dict()),
        }
        if not rr.passed:
            raise ArtifactError(
                f"{task_id}: the new cell is not ≻ the current one on NLL (rule R: {rr.reason}; "
                "rule iii)"
            )
        if not ni.passed:
            raise ArtifactError(
                f"{task_id}: the new cell is not ≽ the current one on acc within "
                f"{PROMOTE_ACC_MARGIN} (lower bound {ni.lower_bound:.4f}; rule iii)"
            )
    promoted = CurrentEntry(
        version=manifest.version,
        tier=chosen,
        question_key=entry.question_key,
        promoted_at=now or _now(),
        cite=entry.cite,
    )
    updated = Current(tasks={**current.tasks, task_id: promoted})
    layout.write_current(updated)
    return PromotionReport(
        task_id=task_id,
        tier=chosen,
        version=manifest.version,
        question_key=entry.question_key,
        cite=str(entry.cite),
        k2_verdict=verdict,
        replaced=replaced,
        comparison=comparison,
        current=updated,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


# -- fitting (learn fit) -----------------------------------------------------------------------


def _cite_for(
    results_root: Path, path: str | None, task_name: str, tier: str, framing: str
) -> tuple[str | None, BenchCell | None]:
    """The cite of the cell ``<task>.<tier>.<framing>`` in ``path`` when the file holds it as
    a pre-registered, non-exploratory **nested** cell (``None`` otherwise: the entry is written
    with ``cite: null`` and promotion refuses it, rule i). The closed choices' and term.fits'
    calibrated cells in the pinned ``tiers.json`` are fixed arms or K1 audit cells
    (``selection: "none"``), so their calibrators cite nothing by construction."""
    if path is None:
        return None, None
    rel = Path(path).as_posix()
    cite = f"{rel}#{task_name}.{tier}.{framing}"
    ref = parse_cite(cite)
    if ref is None:
        raise ArtifactError(f"{cite}: not a cite (the results path must be bench/results/...)")
    try:
        _, cell = _results_cell(results_root, ref)
    except ArtifactError as exc:
        logger.warning("no cite for %s (%s)", cite, exc)
        return None, None
    if not (cell.pre_registered and cell.selection == "nested" and not cell.exploratory):
        logger.warning(
            "no cite for %s: not a pre-registered nested cell (selection %s, pre_registered "
            "%s, exploratory %s)",
            cite,
            cell.selection,
            cell.pre_registered,
            cell.exploratory,
        )
        return None, None
    return cite, cell


def calibrated_entry(
    task: Any,
    scores: Any,
    *,
    results_root: Path,
    tiers_path: str | None,
) -> tuple[ManifestEntry, PlattCalibrator | TemperatureCalibrator]:
    """The calibrated-tier artifact of ``task`` under the arm ``scores`` (a
    :class:`~mesa_clm.bench.cells.ArmScores`): the plan §5.3 calibrator fitted on every item
    with the label weights (D20), cited to ``<tiers_path>#<task>.calibrated.<framing>`` when
    that nested cell exists."""
    from mesa_clm.learn import calibrate

    labels = np.asarray(task.labels, dtype=np.int64)
    weights = np.asarray(task.weights or [1.0] * len(task.items), dtype=np.float64)
    fitted = calibrate.fit_calibrator(scores.shape, scores.x, labels, weights)
    runtime = load_calibrator(fitted.model_dump(mode="json"))
    frame = scores.arm.framing
    cite, cell = _cite_for(results_root, tiers_path, task.name, "calibrated", frame)
    config = {"framing": frame, "model": scores.arm.model}
    return (
        ManifestEntry(
            question_key=scores.question_key,
            task_id=task.task_id,
            framing_id=frame,
            tier="calibrated",
            file=CALIBRATORS_FILE,
            calibrator=fitted.model_dump(mode="json"),
            n_train=len(task.items),
            cite=cite,
            cell=None
            if cell is None
            else cell_identity(str(cite).split("#")[0], cell, config=config),
        ),
        runtime,
    )


def probe_entry(
    task: Any,
    artifact: Any,
    selection: Mapping[str, Any],
    *,
    results_root: Path,
    x3_path: str | None,
) -> tuple[ManifestEntry, dict[str, Any]]:
    """The probe artifact of ``task`` (``learn.probe.full_probe``'s ``ProbeArtifact`` and its
    selection record) as a manifest entry and the probe's JSON, cited to
    ``<x3_path>#<task>.probe.<framing>`` when that nested cell exists."""
    data = artifact.model_dump(mode="json")
    linear = data.get("linear") or {}
    frame = str(data["framing_id"])
    cite, cell = _cite_for(results_root, x3_path, task.name, "probe", frame)
    config = {"spec": str(data["spec"]), "model": str(data["model"]), "framing": frame}
    hyper = linear.get("hyper")
    return (
        ManifestEntry(
            question_key=str(data["question_key"]),
            task_id=str(data["task_id"]),
            framing_id=frame,
            tier="probe",
            file=f"{PROBES_DIR}/{data['question_key']}.json",
            spec=str(data["spec"]),
            fitter=str(linear.get("kind")) if linear.get("kind") is not None else None,
            hyper={str(k): float(v) for k, v in dict(hyper).items()} if hyper else None,
            calibrator=dict(data.get("calibrator") or {}),
            n_train=int(data.get("n_train") or len(task.items)),
            cite=cite,
            cell=None
            if cell is None
            else cell_identity(str(cite).split("#")[0], cell, config=config),
        ),
        data,
    )


@dataclass
class FitResult:
    """What :func:`fit_version` wrote: per served model the version number and directory
    (``None`` when nothing of that model was fitted), the manifests, and what was skipped."""

    written: dict[str, tuple[int, Path]] = field(default_factory=dict)
    manifests: dict[str, Manifest] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def fit_version(
    layouts: Mapping[str, ArtifactLayout],
    *,
    tasks: Mapping[str, Any],
    arm_scores: Mapping[str, Any],
    probe_fitter: Any | None,
    labels_sha256: str,
    labels_content_sha256: str | None,
    results_root: str | Path,
    x3_path: str | None,
    tiers_path: str | None,
    notes: Sequence[str] = (),
    registered: bool = False,
    registration: Mapping[str, Any] | None = None,
) -> FitResult:
    """``learn fit`` (module docstring): for every task in ``tasks`` (bench name -> task) the
    calibrated tier's calibrator under the arm ``arm_scores[name]`` gives (an
    :class:`~mesa_clm.bench.cells.ArmScores` of the A1 arm; absent: no calibrator for that
    task, K1) and, when ``probe_fitter`` is given, the probe ``probe_fitter(task)`` returns as
    ``(ProbeArtifact, selection record)`` (``learn.probe.full_probe``; ``None`` or an
    ``ArtifactError`` from the fitter skips the task's probe with the reason). Each artifact is
    written into the layout of its model (``layouts``: model -> layout; a probe whose spec
    belongs to a model without a layout is skipped and reported), one new version per model
    that received anything, its manifest recording ``registered``/``registration`` as the
    caller found them (the CLI refuses before calling when the M4 registration deviates) and
    ``tasks_fitted`` = the task ids of ``tasks``. ``CURRENT.json`` is untouched."""
    root = Path(results_root)
    task_ids = sorted({str(getattr(t, "task_id", name)) for name, t in tasks.items()})
    per_model: dict[str, dict[str, Any]] = {
        m: {"entries": {}, "calibrators": {}, "probes": {}} for m in layouts
    }
    result = FitResult()
    for name, task in tasks.items():
        scores = arm_scores.get(name)
        if scores is not None:
            model = str(scores.arm.model)
            if model not in per_model:
                result.skipped[f"{name}.calibrated"] = f"no layout for model {model}"
            else:
                entry, runtime = calibrated_entry(
                    task, scores, results_root=root, tiers_path=tiers_path
                )
                per_model[model]["entries"][entry_key("calibrated", entry.question_key)] = entry
                per_model[model]["calibrators"][entry.question_key] = runtime
        else:
            result.skipped[f"{name}.calibrated"] = "no A1 arm (K1): no calibrator fitted"
        if probe_fitter is None:
            continue
        try:
            fitted = probe_fitter(task)
        except ArtifactError as exc:
            result.skipped[f"{name}.probe"] = str(exc)
            continue
        if fitted is None:
            result.skipped[f"{name}.probe"] = "the probe fitter chose nothing"
            continue
        artifact, selection = fitted
        entry, data = probe_entry(task, artifact, selection, results_root=root, x3_path=x3_path)
        model = str(data["model"])
        if model not in per_model:
            result.skipped[f"{name}.probe"] = f"spec {entry.spec} belongs to {model}: no layout"
            continue
        per_model[model]["entries"][entry_key("probe", entry.question_key)] = entry
        per_model[model]["probes"][entry.question_key] = data
    for model, parts in per_model.items():
        if not parts["entries"]:
            continue
        n, path = write_version(
            layouts[model],
            entries=parts["entries"],
            calibrators=parts["calibrators"],
            probes=parts["probes"],
            labels_sha256=labels_sha256,
            labels_content_sha256=labels_content_sha256,
            notes=[*notes, f"model {model}"],
            registered=registered,
            registration=registration,
            tasks_fitted=task_ids,
        )
        result.written[model] = (n, path)
        result.manifests[model] = load_manifest(path)
    return result
