"""The per-encoder feature cache, the X1/X2 text manifest and the trainer export (plan §5.2,
§5.3, §5.6; DESIGN D5, D11, D23, D30).

**Store.** One :class:`FeatureStore` per encoder fingerprint at
``<features.dir>/<encoder_fp>/features.duckdb`` (default ``~/.mesa/clm/features``): vectors are
encoder-locked (D5), so a store never mixes routes, revisions or recipes, and a store whose
recorded ``encoder_fp`` disagrees with its directory or with the live one is refused. Because a
change of image, kernel flag or truncation cap moves vectors without changing
:class:`~mesa_clm.clm.fingerprint.EncoderSpec`, the store also records, when it is created, the
serving lock's **vector recipe** (:meth:`~mesa_clm.clm.fingerprint.ServingLock.vector_recipe`:
image digest, encoder spec, ``vllm serve`` arguments, non-secret environment, cap) and its
sha256, and refuses every read and write under a lock whose vector recipe differs
(:meth:`FeatureStore.for_lock`); each vector row carries the ``lock_sha`` it was embedded under.
The directory is created ``0700`` and the database and its sibling lock file ``features.lock``
``0600``. Like the sidecar (D11) nothing holds the file between calls: every method is one
operation that takes ``fcntl.flock`` on the lock file (exclusive for a write, shared for a read),
opens DuckDB (``read_only`` for reads), runs one transaction, closes and releases, so one writer
at a time and any number of readers serialise on the lock instead of failing on DuckDB's own
file lock. A read of a store that does not exist creates nothing.

Tables (DuckDB, schema ``main``):

* ``texts(text_sha256 PRIMARY KEY, text, tokens, truncated, first_seen)`` - every text the
  cache has seen; ``tokens`` is the token guard's exact count (``EncoderClient.token_guard``),
  ``truncated`` means the text is over ``max_len - 16`` tokens and was never embedded (plan
  §4.3: a truncated context is an abstain, so it never needs a vector);
* ``vectors(text_sha256 PRIMARY KEY, vec32, fp16_cos, lock_sha)`` - ``vec32`` is the
  little-endian float32 ``[dim]`` BLOB of the L2-normalised vector (normalised in float64 and
  cast to float32 as ``headproj.l2`` does), the vector offline scoring reads; ``fp16_cos`` the
  cosine between it and its float16 copy, the plan §5.2 gate of the ``TextCache`` export
  (:func:`fp16_roundtrip_min_cosine` ``>= 0.9999``); ``lock_sha`` the serving lock it was embedded
  under. Format 1 kept only the float16 copy, and scores from it missed clm-serve by up to
  1.31e-3 (``clm-latest``) and 3.59e-3 (``clm-raw``) in probability against X1's pre-registered
  1e-4: float16 rounding spread over the vector moves scores beyond 1e-4 at CLM's scale of 100
  (putting the three largest dimensions back from float32 recovers most of ``clm-raw``'s gap and
  none of ``clm-latest``'s, ``bench/results/2026-10-01/features_build.json#/rerun/crosscheck/cause``),
  and only float32 vectors meet it (``bench/results/2026-10-01/x1_crosscheck.json``). A format-1
  store is refused: it cannot be converted, only rebuilt;
* ``proj(clm_model_fp, side, text_sha256, vec)`` - the 512-d float32 head projections
  (``side`` ``state`` | ``action``), computed by :meth:`FeatureStore.project` from the float32
  vectors through :class:`~mesa_clm.clm.headproj.HeadProjector`;
* ``proj_heads(clm_model_fp, head_id, ...)`` - which head a ``clm_model_fp``'s projections came
  from, so a different head under the same fingerprint is refused (K4);
* ``meta(key, value)`` - ``format``, ``encoder_fp``, ``dim``, ``created_at``, the encoder spec,
  the vector recipe with its sha256 and the ``lock_sha`` the store was created under.

DuckDB 1.5.6 probes ``import pandas`` for every bound Python value when pandas is absent
(DESIGN, "Bulk sidecar inserts"), so rows travel the way ``provenance.store.bulk_insert`` sends
them: one JSON array per chunk unpacked by ``from_json_strict`` (BLOBs as base64, decoded by
``from_base64``), and lookups bind one JSON list. Only ``learn/`` and the ``features`` verbs
write here; serving keeps its own in-memory caches (plan §5.2).

**Manifest.** :func:`manifest` lists every text X1 and X2 embed for ``term.fits`` and
``column.ontology_fits`` and, for M2's tier cells, every text the closed-choice tasks
``column.annotate``, ``column.aspect`` and ``avu.value_kind`` are asked with, built from a
frozen label snapshot (``labels snapshot``, D30) through :mod:`mesa_clm.framings` and
:mod:`mesa_clm.render` only: per labelled (target, option) pair and framing, the state text and
the action texts exactly as CLM's engine renders them (``render.build_pairs`` over
``framings.build_request``). The snapshot is read label-free (no ``label`` or ``weight`` column
is read), its file sha256 is ``labels_sha256``, and the order is deterministic (tasks and
framings as requested, then targets, options and roles sorted). Every requested (task, framing)
pair must exist: the rank_fit tasks take F1, F4, F7, F9 and the X2 specs, a closed-choice task
only its one framing F7 (``framings.FRAMINGS``), so a request such as ``--tasks column.aspect``
with the default X1/X2 framings is refused rather than silently narrowed.

* **F4, F7, F9** (non-control rank_fit arms): one ``context`` row per target (option ``''``; the
  ``target_state`` view, with the task's scope added to a ``column.ontology_fits`` state, which
  has none), one ``anchor`` row per target (option ``__none__``) and one ``candidate`` row per
  labelled option (``FramingCandidate.from_term`` on the stored candidate for ``term.fits``,
  ``framings.ontology_candidates`` for ``column.ontology_fits``).
* **F1** (the noul control): per labelled pair a ``context`` row (the stored mesa-anyjev state,
  ``candidate_state`` or ``ontology_state``, with the task sentence appended; option = the
  candidate) and its two noul candidate texts (roles ``noul_true`` / ``noul_false``). The stored
  state keeps its ``n_candidates`` (0 for the neon-avu-eval silver), because that is the state
  the label was recorded on and what RESEARCH.md's token table and the collapse spike measured.
* **X2 ``joint4096@S1`` / ``joint4096@S1ns``** (the PR #13 replica, plan §5.3/§5.6): PR #13's
  logistic regression read ``cache[e.state_text]``, the raw 4096-d vector of CLM's state text =
  state + ``"\\n\\n"`` + instructions, and never a candidate text (exploration followup 4 §3).
  For mesa-clm's pair questions the "joint" state is the mesa-anyjev per-candidate state (the
  target and the candidate in one text, as the label was recorded), so **S1** is
  ``render.state_text(anyjev state, task question)`` - the PR #13 suffixed layout, identical
  text to F1's context - and **S1ns** the same state with no suffix
  (``render.state_text(anyjev state, None)``). One ``context`` row per labelled pair (option =
  the candidate), no action rows. A probe on the target context alone is invalid for a pair
  question, which is why both specs embed the candidate.

* **Closed choices** (``column.annotate`` K=2, ``column.aspect`` K=8, ``avu.value_kind`` K=4;
  framing F7, plan §4.2 Q1, Q2, Q7): one ``context`` row per labelled target (the
  ``column_state`` or ``value_kind_state`` view, no instructions; ``column.annotate`` and
  ``column.aspect`` render the same column text, which serving asks once per column) and one
  ``option`` row per closed option, keyed by its wire key (the ``ANNOTATE_OPTIONS`` keys, the
  ``ASPECTS``, ``framings.VALUE_KIND_KEYS``), in option order. Option texts are the same for every
  target; they are listed per target, as the anchor is, so a row set describes one request.

*Builder order.* ``state_json`` is stored as canonical JSON (sorted keys at every depth; it is
identity only, DESIGN D23 and plan §4.3), but CLM renders a dict in its key order and the
pipeline sends builder-ordered states. Before rendering, :func:`builder_ordered` restores the key
order of every nested object (``card`` and its ``sites``, ``column``, ``site``, ``candidate``,
``ontology``, ``term``) from templates the :mod:`mesa_clm.states` builders produce themselves;
:func:`mesa_clm.framings.project` orders the top level. ``tests/unit/test_features.py`` proves
that every manifest context equals the one built from the fixture card by the builders.

**Export.** :func:`export_npz` writes ``<dir>/choice_<slug(model)>_<max_len>.npz`` in the format
of CLM's ``train/finetune.py`` ``TextCache`` (``bb42c6c5``, ``finetune.py:422-446, 467-468``):
``keys`` = ``sha1(text)`` hex as a ``<U40`` array, ``vecs`` = float16 ``[N, dim]`` (the stored
float32 vectors cast to float16, the only place float16 appears), uncompressed ``np.savez``, so
``finetune.py --embed-cache <dir> --embed-model Qwen/Qwen3-8B --max-len 4096`` finds every text
and never builds an embedding backend (plan §5.3). ``_slug`` is copied from that file.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, get_args

import duckdb
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, model_validator

from mesa_clm import framings, render
from mesa_clm.cards import ColumnInfo, DatasetCard, SiteInfo
from mesa_clm.clm.encoder import EMBEDDING_DIM
from mesa_clm.clm.fingerprint import ServingLock, canonical_json
from mesa_clm.clm.headproj import SIDES, HeadProjector, Side, l2
from mesa_clm.config import Config, expand_path
from mesa_clm.identity import IdentityError, option_key, target_sha256
from mesa_clm.perms import open_private, private_dir, tighten_file
from mesa_clm.registry import ANCHOR_KEY
from mesa_clm.states import candidate_state, ontology_state, target_state, value_kind_state
from mesa_clm.tasks import TASKS

__all__ = [
    "CHOICE_FRAMINGS",
    "CHOICE_TASKS",
    "DB_FILE",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_FRAMINGS",
    "DEFAULT_MAX_LEN",
    "FORMAT",
    "FP16_MIN_COSINE",
    "LOCK_FILE",
    "MANIFEST_TASKS",
    "X1_FRAMINGS",
    "X1_TASKS",
    "X2_ALIAS",
    "X2_SPECS",
    "FeatureMissing",
    "FeatureStore",
    "FeatureStoreError",
    "Manifest",
    "ManifestError",
    "ManifestRow",
    "NpzExport",
    "builder_ordered",
    "expand_framings",
    "export_npz",
    "fp16_roundtrip_min_cosine",
    "manifest",
    "recipe_sha256",
    "text_sha1",
    "text_sha256",
    "textcache_filename",
]

# Format 2 stores float32 vectors and the vector recipe; format 1 (float16 vectors) is refused.
FORMAT: Final[str] = "mesa-clm-features/2"
DB_FILE: Final[str] = "features.duckdb"
LOCK_FILE: Final[str] = "features.lock"
# plan §5.2: the float16 copy must keep the float32 vector's direction to this cosine.
FP16_MIN_COSINE: Final[float] = 0.9999
# CLM train/finetune.py at this commit is the TextCache format export_npz writes.
TEXTCACHE_COMMIT: Final[str] = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
DEFAULT_EMBED_MODEL: Final[str] = "Qwen/Qwen3-8B"
DEFAULT_MAX_LEN: Final[int] = 4096

Role = Literal["context", "candidate", "anchor", "noul_true", "noul_false", "option"]
ROLES: Final[tuple[str, ...]] = get_args(Role)
# The side of the head a role's text goes through (CLM embeds contexts with the state head).
ROLE_SIDE: Final[dict[str, Side]] = {
    "context": "state",
    "candidate": "action",
    "anchor": "action",
    "noul_true": "action",
    "noul_false": "action",
    "option": "action",
}
X1_TASKS: Final[tuple[str, ...]] = ("term.fits", "column.ontology_fits")
# The closed-choice tasks of M2's tier cells (plan §8 M2) and the one framing each has.
CHOICE_TASKS: Final[tuple[str, ...]] = ("column.annotate", "column.aspect", "avu.value_kind")
CHOICE_FRAMINGS: Final[tuple[str, ...]] = ("F7",)
MANIFEST_TASKS: Final[tuple[str, ...]] = (*X1_TASKS, *CHOICE_TASKS)
X1_FRAMINGS: Final[tuple[str, ...]] = ("F1", "F4", "F7", "F9")
X2_SPECS: Final[tuple[str, ...]] = ("joint4096@S1", "joint4096@S1ns")
X2_ALIAS: Final[str] = "X2"
DEFAULT_FRAMINGS: Final[tuple[str, ...]] = (*X1_FRAMINGS, *X2_SPECS)
# The X1 framing whose view and task sentence the X2 joint specs reuse (module docstring).
_CONTROL_ID: Final[str] = "F1"

_FP_RE: Final = re.compile(r"[0-9a-f]{12}")
_CHUNK: Final[int] = 1000
# FeatureStore.head_id per projector object (hashing 19M weights once, not per call).
_HEAD_IDS: Final[weakref.WeakKeyDictionary[HeadProjector, str]] = weakref.WeakKeyDictionary()

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


class FeatureStoreError(RuntimeError):
    """A feature store that cannot be used as asked: a fingerprint or dimension mismatch, a
    vector that cannot be stored, a head that does not match its ``clm_model_fp``. Messages name
    paths, hashes and counts, never a key."""


class FeatureMissing(FeatureStoreError):
    """Texts without a vector in the store; ``missing`` lists their ``text_sha256``."""

    def __init__(self, missing: Sequence[str], what: str = "vector") -> None:
        self.missing = list(missing)
        head = ", ".join(s[:12] for s in self.missing[:3])
        more = "" if len(self.missing) <= 3 else f" (+{len(self.missing) - 3} more)"
        super().__init__(
            f"{len(self.missing)} text(s) have no {what} in the feature store "
            f"(text_sha256 {head}{more}); run `mesa-clm features build`"
        )


class ManifestError(ValueError):
    """A snapshot, task or framing the manifest cannot be built from (bad path, unknown id, a
    row whose identity does not match its state)."""


def text_sha256(text: str) -> str:
    """sha256 of the text's UTF-8 bytes: the store's key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_sha1(text: str) -> str:
    """CLM ``TextCache.key``: ``hashlib.sha1(t.encode()).hexdigest()`` (UTF-8)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324 - CLM's cache key, not security


def _slug(s: str) -> str:
    """CLM ``finetune.py:_slug`` (``bb42c6c5``), verbatim."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def textcache_filename(
    embed_model: str = DEFAULT_EMBED_MODEL, max_len: int = DEFAULT_MAX_LEN
) -> str:
    """``choice_<slug(embed_model)>_<max_len>.npz`` (``finetune.py:467-468``)."""
    return f"choice_{_slug(embed_model)}_{int(max_len)}.npz"


def _unit_f16(vectors: npt.ArrayLike, dim: int) -> tuple[F32, npt.NDArray[np.float16], F64]:
    """``(float32 unit rows, their float16 copies, per-row cosine between the two)``; refuses a
    wrong shape, a non-finite value and a zero vector (whose direction is undefined)."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise FeatureStoreError(f"vectors must be [n, {dim}], got {list(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise FeatureStoreError("vectors contain a non-finite value")
    if arr.shape[0] and float(np.min(np.linalg.norm(arr, axis=1))) <= 0.0:
        raise FeatureStoreError("a zero vector has no direction to store")
    x32 = l2(arr)
    x16 = x32.astype(np.float16)
    a = x32.astype(np.float64)
    b = x16.astype(np.float64)
    cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return x32, x16, np.asarray(cos, dtype=np.float64)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def recipe_sha256(vector_recipe: Mapping[str, Any]) -> str:
    """sha256 of a vector recipe's canonical JSON (sorted keys, compact separators), equal to
    :meth:`~mesa_clm.clm.fingerprint.ServingLock.vector_recipe_sha256` for the lock's recipe."""
    return hashlib.sha256(canonical_json(dict(vector_recipe)).encode("utf-8")).hexdigest()


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
        ).fetchall()
    }


def _has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return name in _tables(con)


def _json_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


# -- the store ----------------------------------------------------------------------------------


class FeatureStore:
    """The feature cache of one encoder fingerprint (module docstring).

    ``root`` is the ``features.dir`` setting (``~`` expanded); the store lives in
    ``<root>/<encoder_fp>/``. The constructor touches nothing: the first write creates the
    directory (0700), the lock file and the database (0600), the tables and the ``meta`` rows.

    ``vector_recipe`` is the serving lock's
    :meth:`~mesa_clm.clm.fingerprint.ServingLock.vector_recipe` (:meth:`for_lock` passes it): a
    store created with it records it, and every read and write refuses a store whose recorded
    recipe sha256 differs (one recorded without a recipe included). ``None`` skips that check
    (tests, and tools that hold no lock); every CLI path passes the live lock's. ``lock_sha`` is
    stamped on the vector rows this object writes.
    """

    def __init__(
        self,
        root: str | Path,
        encoder_fp: str,
        *,
        dim: int = EMBEDDING_DIM,
        vector_recipe: Mapping[str, Any] | None = None,
        lock_sha: str = "",
    ) -> None:
        if not _FP_RE.fullmatch(encoder_fp):
            raise FeatureStoreError(
                f"encoder_fp must be 12 lower-case hex characters (D5), got {encoder_fp!r}"
            )
        if dim < 1:
            raise FeatureStoreError("dim must be positive")
        self.root = Path(root).expanduser()
        self.encoder_fp = encoder_fp
        self.dim = int(dim)
        self.dir = self.root / encoder_fp
        self.path = self.dir / DB_FILE
        self.lock_path = self.dir / LOCK_FILE
        self.vector_recipe = None if vector_recipe is None else dict(vector_recipe)
        self.recipe_sha256 = None if vector_recipe is None else recipe_sha256(vector_recipe)
        self.lock_sha = lock_sha

    @classmethod
    def from_config(cls, cfg: Config, encoder_fp: str, *, dim: int = EMBEDDING_DIM) -> FeatureStore:
        """The store of ``encoder_fp`` under ``cfg.features.dir``, without a recipe check (use
        :meth:`for_lock` wherever a serving lock is at hand)."""
        return cls(expand_path(cfg.features.dir), encoder_fp, dim=dim)

    @classmethod
    def for_lock(
        cls, root: str | Path | Config, lock: ServingLock, *, dim: int = EMBEDDING_DIM
    ) -> FeatureStore:
        """The store of ``lock``'s ``encoder_fp`` under ``root`` (a directory, or a
        :class:`~mesa_clm.config.Config` whose ``features.dir`` is used), checked against the
        lock's vector recipe and stamping its ``lock_sha`` on new vectors."""
        base = expand_path(root.features.dir) if isinstance(root, Config) else Path(root)
        return cls(
            base,
            lock.encoder_fp,
            dim=dim,
            vector_recipe=lock.vector_recipe(),
            lock_sha=lock.lock_sha,
        )

    def exists(self) -> bool:
        return self.path.is_file()

    def __repr__(self) -> str:
        return f"FeatureStore({str(self.path)!r})"

    # -- files, lock, connection -------------------------------------------------------------------

    def _make_dirs(self) -> None:
        # Missing components of features.dir (``~/.mesa/clm`` included) 0700 whatever the umask;
        # an existing root is left as it is, the store's own directory is tightened.
        private_dir(self.root, tighten=False)
        private_dir(self.dir)

    def _chmod_files(self) -> None:
        for path in (self.path, self.path.with_name(self.path.name + ".wal")):
            tighten_file(path)

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        """One operation: flock the sibling lock file (exclusive to write, shared to read), open
        DuckDB (``read_only`` unless ``write``), yield, close, release. A read needs the database
        to exist (callers check :meth:`exists` first)."""
        if write:
            self._make_dirs()
        fd = open_private(self.lock_path, tighten=write)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            try:
                con = duckdb.connect(str(self.path), read_only=not write)
                try:
                    if write:
                        self._chmod_files()
                    con.execute("SET TimeZone = 'UTC'")
                    yield con
                finally:
                    con.close()
                    if write:
                        self._chmod_files()
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _ddl(self) -> tuple[str, ...]:
        vec_bytes = 4 * self.dim
        return (
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """CREATE TABLE IF NOT EXISTS texts (
                text_sha256 TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                tokens INTEGER NOT NULL CHECK (tokens >= 0),
                truncated BOOLEAN NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL DEFAULT now())""",
            f"""CREATE TABLE IF NOT EXISTS vectors (
                text_sha256 TEXT PRIMARY KEY,
                vec32 BLOB NOT NULL CHECK (octet_length(vec32) = {vec_bytes}),
                fp16_cos DOUBLE NOT NULL,
                lock_sha TEXT NOT NULL DEFAULT '')""",
            """CREATE TABLE IF NOT EXISTS proj (
                clm_model_fp TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('state', 'action')),
                text_sha256 TEXT NOT NULL,
                vec BLOB NOT NULL,
                PRIMARY KEY (clm_model_fp, side, text_sha256))""",
            """CREATE TABLE IF NOT EXISTS proj_heads (
                clm_model_fp TEXT PRIMARY KEY,
                head_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                projection_dim INTEGER NOT NULL CHECK (projection_dim > 0),
                first_seen TIMESTAMPTZ NOT NULL DEFAULT now())""",
        )

    def _bootstrap(
        self, con: duckdb.DuckDBPyConnection, encoder_spec: Mapping[str, Any] | None = None
    ) -> None:
        """Create the tables and ``meta`` rows of a new store; an existing store's ``meta`` is
        checked first, so nothing is added to a store this object would refuse."""
        if _has_table(con, "meta"):
            self._check_meta(dict(con.execute("SELECT key, value FROM meta").fetchall()))
        for stmt in self._ddl():
            con.execute(stmt)
        rows = [
            {"key": "format", "value": FORMAT},
            {"key": "encoder_fp", "value": self.encoder_fp},
            {"key": "dim", "value": str(self.dim)},
            {"key": "created_at", "value": datetime.now(tz=UTC).isoformat(timespec="seconds")},
            {"key": "vector_recipe_sha256", "value": self.recipe_sha256 or ""},
            {"key": "serving_lock_sha", "value": self.lock_sha},
        ]
        if encoder_spec is not None:
            spec = json.dumps(dict(encoder_spec), sort_keys=True, separators=(",", ":"))
            rows.append({"key": "encoder_spec", "value": spec})
        if self.vector_recipe is not None:
            recipe = json.dumps(self.vector_recipe, sort_keys=True, separators=(",", ":"))
            rows.append({"key": "vector_recipe", "value": recipe})
        _insert(con, "meta", {"key": "VARCHAR", "value": "VARCHAR"}, rows)
        self._check_meta(dict(con.execute("SELECT key, value FROM meta").fetchall()))

    def _check_meta(self, meta: Mapping[str, str]) -> None:
        """Refuse a store whose recorded format, fingerprint, dimension or (when this object
        carries one) vector recipe is not this one's."""
        problems = []
        if meta.get("format") != FORMAT:
            problems.append(
                f"format {meta.get('format')!r} (expected {FORMAT!r}; a format-1 store kept "
                "float16 vectors only, which cannot reproduce clm-serve: move the directory "
                "aside and run `mesa-clm features build` again)"
            )
        if meta.get("encoder_fp") != self.encoder_fp:
            problems.append(
                f"encoder_fp {meta.get('encoder_fp')!r} (expected {self.encoder_fp!r}; vectors "
                "are encoder-locked, D5/K4)"
            )
        if meta.get("dim") != str(self.dim):
            problems.append(f"dim {meta.get('dim')!r} (expected {self.dim})")
        if self.recipe_sha256 is not None and meta.get("format") == FORMAT:
            recorded = meta.get("vector_recipe_sha256") or ""
            if recorded != self.recipe_sha256:
                problems.append(
                    f"vector recipe {recorded[:12] or 'none'} (the serving lock's is "
                    f"{self.recipe_sha256[:12]}: another image, vllm argument, environment or "
                    "cap produced these vectors, D5/K4; move the directory aside and rebuild)"
                )
        if problems:
            raise FeatureStoreError(
                f"{self.path}: refused, the store records " + "; ".join(problems)
            )

    @contextmanager
    def _write(
        self, encoder_spec: Mapping[str, Any] | None = None
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        """A write operation: bootstrap (idempotent) and check ``meta``, then one transaction."""
        with self.connect(write=True) as con:
            self._bootstrap(con, encoder_spec)
            con.execute("BEGIN")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")

    @contextmanager
    def _read(self) -> Iterator[duckdb.DuckDBPyConnection | None]:
        """A read operation: ``None`` when the store (or its schema) does not exist yet."""
        if not self.exists():
            yield None
            return
        with self.connect() as con:
            tables = _tables(con)
            if "meta" in tables:
                self._check_meta(dict(con.execute("SELECT key, value FROM meta").fetchall()))
            if not {"meta", "texts", "vectors", "proj", "proj_heads"} <= tables:
                yield None
                return
            yield con

    def ensure(self, *, encoder_spec: Mapping[str, Any] | None = None) -> None:
        """Create the store if missing (recording ``encoder_spec`` when given) and check that it
        belongs to this fingerprint and dimension; :class:`FeatureStoreError` otherwise."""
        with self._write(encoder_spec):
            pass

    def meta(self) -> dict[str, str]:
        """The ``meta`` rows (empty for a missing store)."""
        with self._read() as con:
            if con is None:
                return {}
            return {
                str(k): str(v) for k, v in con.execute("SELECT key, value FROM meta").fetchall()
            }

    # -- texts and vectors --------------------------------------------------------------------------

    def _known(self, shas: Sequence[str]) -> set[str]:
        """The shas that need no embedding: a vector exists, or the text is recorded truncated."""
        if not shas:
            return set()
        with self._read() as con:
            if con is None:
                return set()
            rows = con.execute(
                "SELECT t.text_sha256 FROM texts t LEFT JOIN vectors v USING (text_sha256) "
                "WHERE t.text_sha256 IN (SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]'))) "
                "AND (t.truncated OR v.text_sha256 IS NOT NULL)",
                [_json_list(shas)],
            ).fetchall()
        return {str(r[0]) for r in rows}

    def missing(self, texts: Iterable[str]) -> list[str]:
        """The distinct texts (first-seen order) that still need an embedding: no vector and not
        recorded as truncated."""
        distinct = list(dict.fromkeys(texts))
        shas = [text_sha256(t) for t in distinct]
        known = self._known(shas)
        return [t for t, s in zip(distinct, shas, strict=True) if s not in known]

    @staticmethod
    def _check_tokens(texts: Sequence[str], tokens: Sequence[int]) -> list[int]:
        if len(tokens) != len(texts):
            raise FeatureStoreError(f"{len(texts)} texts but {len(tokens)} token counts")
        out = [int(n) for n in tokens]
        if any(n < 0 for n in out):
            raise FeatureStoreError("token counts must be non-negative")
        return out

    def add(self, texts: Sequence[str], vectors: npt.ArrayLike, tokens: Sequence[int]) -> int:
        """Store ``texts`` with their encoder ``vectors`` (``[n, dim]``, normalised here and kept
        as float32) and the token guard's ``tokens``; returns how many vectors were new. A text
        that already has a vector keeps it (first write wins, which the vector recipe check makes
        safe: only one recipe ever writes to a store); a text recorded as truncated is refused (it
        was never meant to be embedded)."""
        counts = self._check_tokens(texts, tokens)
        x32, _, cos = _unit_f16(vectors, self.dim)
        if x32.shape[0] != len(texts):
            raise FeatureStoreError(f"{len(texts)} texts but {x32.shape[0]} vectors")
        rows: dict[str, tuple[str, int, int]] = {}
        for i, text in enumerate(texts):
            rows.setdefault(text_sha256(text), (text, counts[i], i))
        if not rows:
            return 0
        with self._write() as con:
            truncated = [
                str(r[0])
                for r in con.execute(
                    "SELECT text_sha256 FROM texts WHERE truncated AND text_sha256 IN "
                    "(SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                    [_json_list(list(rows))],
                ).fetchall()
            ]
            if truncated:
                raise FeatureStoreError(
                    f"{len(truncated)} text(s) are recorded as truncated and have no vector by "
                    f"design (text_sha256 {truncated[0][:12]})"
                )
            have = {
                str(r[0])
                for r in con.execute(
                    "SELECT text_sha256 FROM vectors WHERE text_sha256 IN "
                    "(SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                    [_json_list(list(rows))],
                ).fetchall()
            }
            new = [(sha, row) for sha, row in rows.items() if sha not in have]
            _insert(
                con,
                "texts",
                {
                    "text_sha256": "VARCHAR",
                    "text": "VARCHAR",
                    "tokens": "INTEGER",
                    "truncated": "BOOLEAN",
                },
                [
                    {"text_sha256": sha, "text": text, "tokens": n, "truncated": False}
                    for sha, (text, n, _) in rows.items()
                ],
            )
            _insert(
                con,
                "vectors",
                {
                    "text_sha256": "VARCHAR",
                    "vec32": "BLOB",
                    "fp16_cos": "DOUBLE",
                    "lock_sha": "VARCHAR",
                },
                [
                    {
                        "text_sha256": sha,
                        "vec32": _b64(x32[i].astype("<f4").tobytes()),
                        "fp16_cos": float(cos[i]),
                        "lock_sha": self.lock_sha,
                    }
                    for sha, (_, _, i) in new
                ],
            )
        return len(new)

    def add_truncated(self, texts: Sequence[str], tokens: Sequence[int]) -> int:
        """Record ``texts`` the token guard found over the limit (``truncated=true``, no vector);
        returns how many were new. A text that already has a vector is refused."""
        counts = self._check_tokens(texts, tokens)
        rows: dict[str, tuple[str, int]] = {}
        for text, n in zip(texts, counts, strict=True):
            rows.setdefault(text_sha256(text), (text, n))
        if not rows:
            return 0
        with self._write() as con:
            shas = _json_list(list(rows))
            clash = con.execute(
                "SELECT count(*) FROM vectors WHERE text_sha256 IN "
                "(SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                [shas],
            ).fetchone()
            if clash and int(clash[0]):
                raise FeatureStoreError(f"{int(clash[0])} text(s) already have a vector")
            have = {
                str(r[0])
                for r in con.execute(
                    "SELECT text_sha256 FROM texts WHERE text_sha256 IN "
                    "(SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                    [shas],
                ).fetchall()
            }
            _insert(
                con,
                "texts",
                {
                    "text_sha256": "VARCHAR",
                    "text": "VARCHAR",
                    "tokens": "INTEGER",
                    "truncated": "BOOLEAN",
                },
                [
                    {"text_sha256": sha, "text": text, "tokens": n, "truncated": True}
                    for sha, (text, n) in rows.items()
                ],
            )
        return sum(1 for sha in rows if sha not in have)

    def _vectors(self, shas: Sequence[str]) -> dict[str, bytes]:
        if not shas:
            return {}
        with self._read() as con:
            if con is None:
                return {}
            found: dict[str, bytes] = {}
            for start in range(0, len(shas), _CHUNK):
                chunk = shas[start : start + _CHUNK]
                for sha, blob in con.execute(
                    "SELECT text_sha256, vec32 FROM vectors WHERE text_sha256 IN "
                    "(SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                    [_json_list(chunk)],
                ).fetchall():
                    found[str(sha)] = bytes(blob)
        return found

    def get(self, texts: Sequence[str]) -> F32:
        """The stored vectors of ``texts`` in input order as float32 ``[n, dim]``, exactly as
        they were stored (L2-normalised float32, the vectors clm-serve scores; module
        docstring). :class:`FeatureMissing` names the texts without one."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        shas = [text_sha256(t) for t in texts]
        distinct = list(dict.fromkeys(shas))
        found = self._vectors(distinct)
        missing = [s for s in distinct if s not in found]
        if missing:
            raise FeatureMissing(missing)
        decoded = {s: np.frombuffer(found[s], dtype="<f4") for s in distinct}
        return np.stack([decoded[s] for s in shas]).astype(np.float32)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """The token guard's exact count of each of ``texts`` as stored (``texts.tokens``), in
        input order; :class:`FeatureMissing` names texts the store has never seen."""
        if not texts:
            return []
        shas = [text_sha256(t) for t in texts]
        distinct = list(dict.fromkeys(shas))
        found: dict[str, int] = {}
        with self._read() as con:
            if con is not None:
                for start in range(0, len(distinct), _CHUNK):
                    chunk = distinct[start : start + _CHUNK]
                    for sha, tokens in con.execute(
                        "SELECT text_sha256, tokens FROM texts WHERE text_sha256 IN "
                        "(SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                        [_json_list(chunk)],
                    ).fetchall():
                        found[str(sha)] = int(tokens)
        missing = [s for s in distinct if s not in found]
        if missing:
            raise FeatureMissing(missing, what="token count")
        return [found[s] for s in shas]

    def embedded_texts(self) -> list[str]:
        """Every text that has a vector, in ``text_sha256`` order."""
        with self._read() as con:
            if con is None:
                return []
            rows = con.execute(
                "SELECT t.text FROM vectors v JOIN texts t USING (text_sha256) "
                "ORDER BY v.text_sha256"
            ).fetchall()
        return [str(r[0]) for r in rows]

    # -- projections --------------------------------------------------------------------------------

    @staticmethod
    def head_id(projector: HeadProjector) -> str:
        """sha256 over the head's exporter arrays (config, logit scale, source checkpoint sha
        and every weight, in key order): what the ``proj`` cache is keyed to besides
        ``clm_model_fp``, so a re-trained head under the same fingerprint is refused. Computed
        once per projector object."""
        cached = _HEAD_IDS.get(projector)
        if cached is None:
            digest = hashlib.sha256()
            arrays = projector.to_arrays()
            for key in sorted(arrays):
                arr = np.ascontiguousarray(arrays[key])
                digest.update(f"{key}:{arr.dtype.str}:{arr.shape};".encode())
                digest.update(arr.tobytes())
            cached = _HEAD_IDS[projector] = digest.hexdigest()
        return cached

    def _check_head(self, con: duckdb.DuckDBPyConnection, clm_model_fp: str, head_id: str) -> bool:
        """Whether ``clm_model_fp`` is registered; refuses one registered for another head."""
        row = con.execute(
            "SELECT head_id FROM proj_heads WHERE clm_model_fp = ?", [clm_model_fp]
        ).fetchone()
        if row is None:
            return False
        if str(row[0]) != head_id:
            raise FeatureStoreError(
                f"{self.path}: clm_model_fp {clm_model_fp} was projected with another head "
                f"(head_id {str(row[0])[:12]} vs {head_id[:12]}); a head is identified by its "
                "fingerprint (D5/K4)"
            )
        return True

    def project(
        self,
        projector: HeadProjector,
        clm_model_fp: str,
        side: Side,
        texts: Sequence[str],
    ) -> F32:
        """The L2-normalised ``side`` projections of ``texts`` under ``projector`` as float32
        ``[n, projection_dim]``, in input order. Cached rows are read; the missing ones are
        computed in one batch (first-seen order) from the stored vectors and written under
        ``(clm_model_fp, side, text_sha256)``. ``clm_model_fp`` must name this head (the first
        projection registers it; another head under the same fingerprint is refused)."""
        if side not in SIDES:
            raise FeatureStoreError(f"side must be one of {SIDES}, got {side!r}")
        if not _FP_RE.fullmatch(clm_model_fp):
            raise FeatureStoreError("clm_model_fp must be 12 lower-case hex characters (D5)")
        if projector.cfg.hidden_size != self.dim:
            raise FeatureStoreError(
                f"the head takes {projector.cfg.hidden_size}-d input, the store holds {self.dim}-d"
            )
        width = projector.cfg.projection_dim
        if not texts:
            return np.zeros((0, width), dtype=np.float32)
        head_id = self.head_id(projector)
        shas = [text_sha256(t) for t in texts]
        order = list(dict.fromkeys(shas))
        cached: dict[str, bytes] = {}
        with self._read() as con:
            if con is not None:
                self._check_head(con, clm_model_fp, head_id)
                for start in range(0, len(order), _CHUNK):
                    chunk = order[start : start + _CHUNK]
                    for sha, blob in con.execute(
                        "SELECT text_sha256, vec FROM proj WHERE clm_model_fp = ? AND side = ? "
                        "AND text_sha256 IN (SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]')))",
                        [clm_model_fp, side, _json_list(chunk)],
                    ).fetchall():
                        cached[str(sha)] = bytes(blob)
        todo = [s for s in order if s not in cached]
        if todo:
            first = {s: t for s, t in zip(shas, texts, strict=True)}
            x = self.get([first[s] for s in todo])
            z = projector.project_states(x) if side == "state" else projector.project_actions(x)
            blobs = {s: z[i].astype("<f4").tobytes() for i, s in enumerate(todo)}
            with self._write() as con:
                if not self._check_head(con, clm_model_fp, head_id):
                    _insert(
                        con,
                        "proj_heads",
                        {
                            "clm_model_fp": "VARCHAR",
                            "head_id": "VARCHAR",
                            "source_sha256": "VARCHAR",
                            "projection_dim": "INTEGER",
                        },
                        [
                            {
                                "clm_model_fp": clm_model_fp,
                                "head_id": head_id,
                                "source_sha256": projector.source_sha256,
                                "projection_dim": width,
                            }
                        ],
                    )
                _insert(
                    con,
                    "proj",
                    {
                        "clm_model_fp": "VARCHAR",
                        "side": "VARCHAR",
                        "text_sha256": "VARCHAR",
                        "vec": "BLOB",
                    },
                    [
                        {
                            "clm_model_fp": clm_model_fp,
                            "side": side,
                            "text_sha256": s,
                            "vec": _b64(b),
                        }
                        for s, b in blobs.items()
                    ],
                )
            cached.update(blobs)
        decoded = {s: np.frombuffer(cached[s], dtype="<f4") for s in order}
        if any(v.shape != (width,) for v in decoded.values()):
            raise FeatureStoreError(
                f"cached projections for {clm_model_fp} are not {width}-d; the cache is corrupt"
            )
        return np.stack([decoded[s] for s in shas]).astype(np.float32)

    # -- reporting ----------------------------------------------------------------------------------

    def fp16_cosines(self) -> F64:
        """Every stored vector's float32-vs-float16 cosine (``text_sha256`` order)."""
        with self._read() as con:
            if con is None:
                return np.zeros(0, dtype=np.float64)
            rows = con.execute("SELECT fp16_cos FROM vectors ORDER BY text_sha256").fetchall()
        return np.asarray([float(r[0]) for r in rows], dtype=np.float64)

    def stats(self) -> dict[str, Any]:
        """Counts and sizes: texts, vectors, truncated texts, token totals and quantiles, the
        fp16 gate value, projections per ``clm_model_fp`` and side, the file size."""
        out: dict[str, Any] = {
            "path": str(self.path),
            "encoder_fp": self.encoder_fp,
            "exists": self.exists(),
        }
        with self._read() as con:
            if con is None:
                return out
            meta = {
                str(k): str(v) for k, v in con.execute("SELECT key, value FROM meta").fetchall()
            }
            text_row = con.execute(
                "SELECT count(*), count(*) FILTER (WHERE truncated), coalesce(sum(tokens), 0), "
                "max(tokens), quantile_cont(tokens, 0.5), quantile_cont(tokens, 0.95) FROM texts"
            ).fetchone()
            n_texts, n_trunc, tok_sum, tok_max, p50, p95 = text_row or (0, 0, 0, None, None, None)
            vec_row = con.execute("SELECT count(*), min(fp16_cos) FROM vectors").fetchone()
            n_vec, min_cos = vec_row or (0, None)
            by_lock = {
                str(sha): int(n)
                for sha, n in con.execute(
                    "SELECT lock_sha, count(*) FROM vectors GROUP BY ALL ORDER BY ALL"
                ).fetchall()
            }
            proj: dict[str, dict[str, int]] = {}
            for fp, side, n in con.execute(
                "SELECT clm_model_fp, side, count(*) FROM proj GROUP BY ALL ORDER BY ALL"
            ).fetchall():
                proj.setdefault(str(fp), {})[str(side)] = int(n)
        out.update(
            format=meta.get("format"),
            dim=int(meta.get("dim", self.dim)),
            created_at=meta.get("created_at"),
            encoder_spec=json.loads(meta["encoder_spec"]) if "encoder_spec" in meta else None,
            vector_recipe_sha256=meta.get("vector_recipe_sha256") or None,
            serving_lock_sha=meta.get("serving_lock_sha") or None,
            vectors_by_lock_sha=by_lock,
            texts=int(n_texts),
            vectors=int(n_vec),
            truncated=int(n_trunc),
            tokens={
                "total": int(tok_sum),
                "max": None if tok_max is None else int(tok_max),
                "p50": None if p50 is None else float(p50),
                "p95": None if p95 is None else float(p95),
            },
            fp16_min_cosine=None if min_cos is None else float(min_cos),
            projections=proj,
            bytes=self.path.stat().st_size,
        )
        return out


def _insert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """``INSERT OR IGNORE`` ``rows`` into ``table`` in chunks of one bound JSON array each
    (the ``provenance.store.bulk_insert`` technique); ``BLOB`` columns travel as base64 text and
    are decoded by ``from_base64``. Table and column names are this module's constants."""
    if not rows:
        return
    structure = {c: ("VARCHAR" if kind == "BLOB" else kind) for c, kind in columns.items()}
    picked = ", ".join(
        f'from_base64(r."{c}")' if kind == "BLOB" else f'r."{c}"' for c, kind in columns.items()
    )
    literal = json.dumps([structure], separators=(",", ":"))
    sql = (
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "  # noqa: S608 - fixed names
        f"SELECT {picked} FROM (SELECT unnest(from_json_strict(?, '{literal}')) AS r)"
    )
    for start in range(0, len(rows), _CHUNK):
        chunk = [{c: row[c] for c in columns} for row in rows[start : start + _CHUNK]]
        con.execute(sql, [json.dumps(chunk, ensure_ascii=False, allow_nan=False)])


# -- the fp16 gate and the TextCache export ------------------------------------------------------


def fp16_roundtrip_min_cosine(store: FeatureStore) -> float | None:
    """The smallest cosine between a stored vector's float32 form and its float16 copy (plan
    §5.2 gate: ``>= FP16_MIN_COSINE``); ``None`` for a store without vectors."""
    cos = store.fp16_cosines()
    return float(cos.min()) if cos.size else None


class NpzExport(NamedTuple):
    """What :func:`export_npz` wrote: the file, the number of texts and the fp16 gate value."""

    path: Path
    n: int
    fp16_min_cosine: float


def export_npz(
    store: FeatureStore,
    out_dir: str | Path,
    embed_model: str = DEFAULT_EMBED_MODEL,
    max_len: int = DEFAULT_MAX_LEN,
) -> NpzExport:
    """Write every stored vector as CLM's ``TextCache`` file ``<out_dir>/choice_<slug>_<max_len>.npz``
    (module docstring): ``keys`` ``sha1(text)`` (``<U40``), ``vecs`` float16 ``[N, dim]``, plain
    ``np.savez``, rows in key order. The file is written next to its final name (owner-only,
    ``0600``; a missing directory is created ``0700``) and moved into place, then read back and
    compared. Refused for a store without vectors and when the fp16 gate fails."""
    gate = fp16_roundtrip_min_cosine(store)
    if gate is None:
        raise FeatureStoreError(
            f"{store.path}: no vectors to export; run `mesa-clm features build`"
        )
    if gate < FP16_MIN_COSINE:
        raise FeatureStoreError(
            f"fp16 round-trip min cosine {gate:.7f} is below the {FP16_MIN_COSINE} gate (plan §5.2)"
        )
    with store._read() as con:
        if con is None:  # vectors were just counted; only a concurrent delete gets here
            raise FeatureStoreError(f"{store.path}: the store disappeared during the export")
        rows = con.execute(
            "SELECT t.text, v.vec32 FROM vectors v JOIN texts t USING (text_sha256)"
        ).fetchall()
    by_key: dict[str, bytes] = {}
    for text, blob in rows:
        key = text_sha1(str(text))
        if key in by_key:
            raise FeatureStoreError(f"two stored texts share the sha1 key {key}")
        by_key[key] = bytes(blob)
    keys = sorted(by_key)
    # CLM's TextCache keeps float16; the store's float32 vectors are cast exactly as format 1 did.
    vecs = np.stack([np.frombuffer(by_key[k], dtype="<f4") for k in keys]).astype(np.float16)
    out = private_dir(out_dir, tighten=False)  # a missing directory is created 0700
    path = out / textcache_filename(embed_model, max_len)
    tmp = out / f".{path.stem}.{os.getpid()}.tmp.npz"
    try:
        fd = open_private(tmp, os.O_WRONLY | os.O_TRUNC)
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, keys=np.array(keys), vecs=vecs)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    with np.load(path, allow_pickle=False) as z:
        ok = (
            z["keys"].dtype == np.dtype("<U40")
            and z["keys"].tolist() == keys
            and z["vecs"].dtype == np.float16
            and z["vecs"].shape == vecs.shape
            and np.array_equal(z["vecs"].view(np.uint16), vecs.view(np.uint16))
        )
    if not ok:
        raise FeatureStoreError(f"{path}: the read-back does not match what was written")
    return NpzExport(path, len(keys), gate)


# -- builder order ------------------------------------------------------------------------------


def _templates() -> dict[str, Any]:
    """Key-order templates for the nested objects of every state, produced by the
    :mod:`mesa_clm.states` builders themselves over a synthetic one-column, one-site card, so
    they cannot drift from the builders."""
    site = SiteInfo(code="SITE", name="n", state="s", domain="D00", domain_name="d", habitat="h")
    col = ColumnInfo(name="c", description="d", dtype="t", unit="u", profile="p")
    card = DatasetCard(
        product_code="DP0.00000.000",
        table="t",
        product_title="t",
        product_description="d",
        source="s",
        sites=[site],
        rows=1,
        months_from="a",
        months_to="b",
        columns=[col],
        sha256="0" * 64,
    )
    cand = candidate_state(card, "column", col, "a", {"label": "l", "curie": "X:1"}, 1)
    ont = ontology_state(card, col, "a", "o", "O: text", ["a"])
    site_view = target_state(card, "site", "a", site=site)
    vk = value_kind_state(card, col, {"label": "l", "curie": "X:1"}, "a")
    return {
        "card": cand["card"],
        "column": cand["column"],
        "candidate": cand["candidate"],
        "ontology": ont["ontology"],
        "site": site_view["site"],
        "term": vk["term"],
    }


_TEMPLATES: Final[dict[str, Any]] = _templates()


def _reorder(value: Any, template: Any, where: str) -> Any:
    if isinstance(value, dict) and isinstance(template, dict):
        unknown = [k for k in value if k not in template]
        if unknown:
            raise ManifestError(f"state key {where}.{unknown[0]} has no builder order")
        return {k: _reorder(value[k], template[k], f"{where}.{k}") for k in template if k in value}
    if isinstance(value, list) and isinstance(template, list) and template:
        return [_reorder(v, template[0], f"{where}[]") for v in value]
    return value


def builder_ordered(state: Mapping[str, Any]) -> dict[str, Any]:
    """``state`` with every nested object's keys in the order the :mod:`mesa_clm.states`
    builders emit them (a sorted-key ``state_json`` back to what the pipeline renders). The top
    level keeps its order (:func:`mesa_clm.framings.project` orders it per view); a nested key the
    builders never emit is a :class:`ManifestError`."""
    return {k: _reorder(v, _TEMPLATES[k], k) if k in _TEMPLATES else v for k, v in state.items()}


# -- the manifest -------------------------------------------------------------------------------


class ManifestRow(BaseModel):
    """One text a task and framing needs for one target (and option). ``option_key`` is ``''``
    for a per-target context, ``__none__`` for the anchor, the wire key of a closed option
    (role ``option``) and the candidate's key otherwise; ``side`` is the head the text goes
    through (``context`` -> ``state``, the rest -> ``action``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    framing_id: str
    target_sha256: str
    option_key: str
    role: Role
    side: Side
    text_sha256: str
    text: str

    @model_validator(mode="after")
    def _consistent(self) -> ManifestRow:
        if ROLE_SIDE[self.role] != self.side:
            raise ValueError(f"role {self.role} goes through the {ROLE_SIDE[self.role]} head")
        if text_sha256(self.text) != self.text_sha256:
            raise ValueError("text_sha256 does not hash the text")
        return self


class Manifest(BaseModel):
    """The X1/X2 texts of one snapshot (:func:`manifest`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: str
    labels_sha256: str
    tasks: tuple[str, ...]
    framings: tuple[str, ...]
    rows: tuple[ManifestRow, ...]
    # Targets whose labelled rows render different per-target contexts (the first row wins).
    context_conflicts: int = 0

    def texts(self) -> list[str]:
        """The distinct texts, in row order (what ``features build`` embeds)."""
        return list(dict.fromkeys(r.text for r in self.rows))

    def summary(self) -> dict[str, Any]:
        """Row counts per task, framing and role, plus the distinct texts per side."""
        counts: dict[str, dict[str, dict[str, int]]] = {}
        for r in self.rows:
            by_role = counts.setdefault(r.task_id, {}).setdefault(r.framing_id, {})
            by_role[r.role] = by_role.get(r.role, 0) + 1
        sides: dict[str, set[str]] = {s: set() for s in SIDES}
        for r in self.rows:
            sides[r.side].add(r.text_sha256)
        return {
            "snapshot": self.snapshot,
            "labels_sha256": self.labels_sha256,
            "tasks": list(self.tasks),
            "framings": list(self.framings),
            "rows": len(self.rows),
            "texts": len({r.text_sha256 for r in self.rows}),
            "texts_by_side": {s: len(v) for s, v in sides.items()},
            "context_conflicts": self.context_conflicts,
            "counts": counts,
        }


def _csv(values: str | Iterable[str]) -> list[str]:
    items = values.split(",") if isinstance(values, str) else list(values)
    return [v.strip() for v in items if v and v.strip()]


def expand_framings(ids: str | Iterable[str]) -> tuple[str, ...]:
    """Framing ids in the order given, ``X2`` expanded to the two joint specs and duplicates
    dropped; an unknown id is a :class:`ManifestError`."""
    out: list[str] = []
    for fid in _csv(ids):
        for one in X2_SPECS if fid == X2_ALIAS else (fid,):
            if one not in DEFAULT_FRAMINGS:
                raise ManifestError(
                    f"unknown framing {one!r}; expected some of {', '.join(DEFAULT_FRAMINGS)} "
                    f"or {X2_ALIAS}"
                )
            if one not in out:
                out.append(one)
    if not out:
        raise ManifestError("no framings given")
    return tuple(out)


def _check_tasks(ids: str | Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for tid in _csv(ids):
        if tid not in MANIFEST_TASKS:
            raise ManifestError(
                f"unknown task {tid!r}; X1/X2 cover {', '.join(X1_TASKS)} and the closed "
                f"choices {', '.join(CHOICE_TASKS)} take {', '.join(CHOICE_FRAMINGS)}"
            )
        if tid not in out:
            out.append(tid)
    if not out:
        raise ManifestError("no tasks given")
    return tuple(out)


def _check_pairs(task_ids: Sequence[str], framing_ids: Sequence[str]) -> None:
    """Every requested (task, framing) pair must exist: a closed-choice task has F7 only."""
    for tid in task_ids:
        if tid not in CHOICE_TASKS:
            continue
        other = [fid for fid in framing_ids if fid not in CHOICE_FRAMINGS]
        if other:
            raise ManifestError(
                f"unknown task {tid!r} for framing {other[0]!r}: the X1/X2 framings cover "
                f"{', '.join(X1_TASKS)}; the closed choice {tid} is framed only as "
                f"{', '.join(CHOICE_FRAMINGS)} (request it with --framings F7)"
            )


class _Pair(NamedTuple):
    """One labelled (target, option) of the snapshot, its state in builder order."""

    target_sha256: str
    option_key: str
    state: dict[str, Any]


def _snapshot_pairs(path: Path, task_ids: Sequence[str]) -> dict[str, list[_Pair]]:
    """Per task, one pair per (target, option) in that order (the row with the smallest
    ``state_sha256`` when several label sources share a pair), each checked against its D1
    identity. No label column is read."""
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT task_id, task_key, target_sha256, option_key, state_sha256, state_json "
            "FROM read_parquet(?) "
            "WHERE task_id IN (SELECT unnest(from_json_strict(?, '[\"VARCHAR\"]'))) "
            "ORDER BY task_id, target_sha256, option_key, state_sha256",
            [str(path), _json_list(list(task_ids))],
        ).fetchall()
    except duckdb.Error as exc:
        raise ManifestError(
            f"{path}: not a readable labels snapshot ({type(exc).__name__})"
        ) from exc
    finally:
        con.close()
    out: dict[str, list[_Pair]] = {t: [] for t in task_ids}
    seen: set[tuple[str, str, str]] = set()
    for task_id, task_key, tsha, okey, _ssha, state_json in rows:
        task_id, tsha, okey = str(task_id), str(tsha), str(okey)
        if (task_id, tsha, okey) in seen:
            continue
        seen.add((task_id, tsha, okey))
        where = f"{task_id} target {tsha[:12]} option {okey!r}"
        if str(task_key) != TASKS[task_id].key:
            raise ManifestError(f"{where}: task_key {task_key} is not {TASKS[task_id].key} (D1)")
        state = json.loads(state_json) if isinstance(state_json, str) else dict(state_json)
        try:
            if target_sha256(task_id, state) != tsha or option_key(task_id, state) != okey:
                raise ManifestError(f"{where}: the stored identity does not match its state (D1)")
        except IdentityError as exc:
            raise ManifestError(f"{where}: {exc}") from exc
        out[task_id].append(_Pair(tsha, okey, builder_ordered(state)))
    return out


def _candidate(task_id: str, pair: _Pair) -> framings.FramingCandidate:
    """The labelled option as a framing candidate (its key must be the label's option_key)."""
    if task_id == "term.fits":
        cand = framings.FramingCandidate.from_term(pair.state["candidate"])
    else:
        cand = framings.ontology_candidates([pair.option_key])[0]
    if cand.key != pair.option_key:
        raise ManifestError(f"{task_id}: candidate key {cand.key!r} is not {pair.option_key!r}")
    return cand


def _row(
    task_id: str, framing_id: str, target: str, option: str, role: Role, text: str
) -> ManifestRow:
    return ManifestRow(
        task_id=task_id,
        framing_id=framing_id,
        target_sha256=target,
        option_key=option,
        role=role,
        side=ROLE_SIDE[role],
        text_sha256=text_sha256(text),
        text=text,
    )


def _arm_rows(
    task_id: str, f: framings.Framing, pairs: Sequence[_Pair]
) -> tuple[list[ManifestRow], int]:
    """F4/F7/F9 rows: per target the context, the anchor and every labelled candidate, from one
    rank_fit request over the target's labelled candidates (what one candidate group costs)."""
    rows: list[ManifestRow] = []
    conflicts = 0
    by_target: dict[str, list[_Pair]] = {}
    for p in pairs:
        by_target.setdefault(p.target_sha256, []).append(p)
    scope = TASKS[task_id].scope
    for target, group in by_target.items():
        states = [{**p.state, "scope": p.state.get("scope") or scope} for p in group]
        contexts = {framings.context_text(f, s) for s in states}
        conflicts += len(contexts) > 1
        cands = [_candidate(task_id, p) for p in group]
        state, questions = framings.build_request(f, states[0], cands)
        state_text, keys, texts = render.build_pairs(state, questions)[task_id]
        if keys[-1] != ANCHOR_KEY or keys[:-1] != [c.key for c in cands]:
            raise ManifestError(f"{task_id}/{f.id}: the request keys are not candidates + anchor")
        rows.append(_row(task_id, f.id, target, "", "context", state_text))
        rows.append(_row(task_id, f.id, target, ANCHOR_KEY, "anchor", texts[-1]))
        rows.extend(
            _row(task_id, f.id, target, key, "candidate", text)
            for key, text in zip(keys[:-1], texts[:-1], strict=True)
        )
    return rows, conflicts


def _choice_rows(
    task_id: str, f: framings.Framing, pairs: Sequence[_Pair]
) -> tuple[list[ManifestRow], int]:
    """Closed-choice rows: per labelled target the context and every closed option, from the one
    request ``framings.build_request`` makes for it (what serving sends, plan §4.2)."""
    rows: list[ManifestRow] = []
    conflicts = 0
    by_target: dict[str, list[_Pair]] = {}
    for p in pairs:
        by_target.setdefault(p.target_sha256, []).append(p)
    wire = list(f.closed_options or {})
    for target, group in by_target.items():
        contexts = {framings.context_text(f, p.state) for p in group}
        conflicts += len(contexts) > 1
        state, questions = framings.build_request(f, group[0].state)
        state_text, keys, texts = render.build_pairs(state, questions)[task_id]
        if keys != wire:
            raise ManifestError(f"{task_id}/{f.id}: the request keys are not the closed options")
        rows.append(_row(task_id, f.id, target, "", "context", state_text))
        rows.extend(
            _row(task_id, f.id, target, key, "option", text)
            for key, text in zip(keys, texts, strict=True)
        )
    return rows, conflicts


def _control_rows(task_id: str, f: framings.Framing, pairs: Sequence[_Pair]) -> list[ManifestRow]:
    """F1 rows: per labelled pair the stored anyjev state + task sentence and the two noul
    candidates CLM derives from that sentence."""
    rows: list[ManifestRow] = []
    question = framings.noul_question(f)
    for p in pairs:
        state_text, keys, texts = render.build_pairs(
            framings.build_context(f, p.state), {p.option_key: question}
        )[p.option_key]
        noul = dict(zip(keys, texts, strict=True))
        rows.append(_row(task_id, f.id, p.target_sha256, p.option_key, "context", state_text))
        rows.append(_row(task_id, f.id, p.target_sha256, p.option_key, "noul_true", noul["true"]))
        rows.append(_row(task_id, f.id, p.target_sha256, p.option_key, "noul_false", noul["false"]))
    return rows


def _joint_rows(task_id: str, spec: str, pairs: Sequence[_Pair]) -> list[ManifestRow]:
    """X2 rows: per labelled pair the joint anyjev state, with (S1) or without (S1ns) the task
    question appended by CLM's ``state_text``."""
    control = framings.framing(task_id, _CONTROL_ID)
    suffix = control.instructions if spec == "joint4096@S1" else None
    return [
        _row(
            task_id,
            spec,
            p.target_sha256,
            p.option_key,
            "context",
            render.state_text(framings.build_context(control, p.state), suffix),
        )
        for p in pairs
    ]


def manifest(
    snapshot_path: str | Path,
    tasks: str | Iterable[str] = X1_TASKS,
    framing_ids: str | Iterable[str] = DEFAULT_FRAMINGS,
) -> Manifest:
    """Every X1/X2 text of ``tasks`` under ``framing_ids`` for the labelled pairs of a frozen
    snapshot, and the F7 texts of the closed-choice tasks (module docstring). ``framing_ids``
    accepts ``X2`` for both joint specs. :class:`ManifestError` for a missing or unreadable
    snapshot, an unknown task or framing, a (task, framing) pair that does not exist (a closed
    choice under anything but F7), or a row whose identity does not match its state."""
    path = Path(snapshot_path).expanduser()
    if not path.is_file():
        raise ManifestError(f"{path}: no such labels snapshot (write one with `labels snapshot`)")
    task_ids = _check_tasks(tasks)
    framing_list = expand_framings(framing_ids)
    _check_pairs(task_ids, framing_list)
    labels_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    pairs = _snapshot_pairs(path, task_ids)
    rows: list[ManifestRow] = []
    conflicts = 0
    for task_id in task_ids:
        for fid in framing_list:
            if fid in X2_SPECS:
                rows.extend(_joint_rows(task_id, fid, pairs[task_id]))
                continue
            f = framings.framing(task_id, fid)
            if f.control:
                rows.extend(_control_rows(task_id, f, pairs[task_id]))
            elif f.shape == "choice":
                choice, n = _choice_rows(task_id, f, pairs[task_id])
                rows.extend(choice)
                conflicts += n
            else:
                arm, n = _arm_rows(task_id, f, pairs[task_id])
                rows.extend(arm)
                conflicts += n
    return Manifest(
        snapshot=str(path),
        labels_sha256=labels_sha256,
        tasks=task_ids,
        framings=framing_list,
        rows=tuple(rows),
        context_conflicts=conflicts,
    )
