"""Fingerprints of everything a decision's probabilities depend on (DESIGN D5; plan §6.7).

CLM heads are encoder-locked: a head trained over one encoder recipe (model, revision, dtype,
pooling, truncation) is meaningless over another, and the in-process fallback encoder must never
share artifacts with the vLLM route before a recorded parity run (D16). Four short hashes are
therefore stamped on every decision, feature row, artifact and bench cell and compared *exactly*
before any of them is used (K4):

* ``encoder_fp`` - sha256 of the canonical JSON of :class:`EncoderSpec` (``model``, ``revision``,
  ``dtype``, ``pooling``, ``normalize``, ``max_len``, ``truncation_side``, ``prefix_caching``,
  ``route``), first 12 hex characters;
* ``clm_model_fp`` - sha256 of the canonical JSON of :class:`ClmModelSpec` (``head_name``,
  ``head_sha256``, ``clm_commit``), first 12 hex characters. ``head_name`` is the model name the
  request carries (``clm-latest``, ``clm-raw`` or a promoted head's unique name, D15),
  ``head_sha256`` the sha256 of the ``.pt`` file that name serves (empty for ``clm-raw``, which
  scores raw cosines without a head) and ``clm_commit`` the 40-hex commit of the CLM clone whose
  engine does the scoring, so a re-trained head under the same name, a head served by a
  different engine, or the raw ablation each get their own artifacts;
* ``schema_sha256`` - the vendored text contract (:func:`mesa_clm.render.schema_sha256`);
* ``serving_lock_sha`` - ``lock_sha`` of ``serving/serving.lock.json``, which pins the CLM commit,
  the patches, the head file, the encoder recipe and the container image.

Canonical JSON is the same everywhere in mesa-clm (``states.state_sha256``,
``identity.target_sha256``): sorted keys, compact separators, UTF-8 with non-ASCII kept. A
fingerprint never contains a secret: keys and URLs are not part of any spec.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mesa_clm import render

FP_HEX: Final[int] = 12
LOCK_SHA_FIELD: Final[str] = "lock_sha"
_SHA256_HEX = 64
_COMMIT_HEX = 40


class FingerprintMismatch(RuntimeError):
    """Two fingerprint bundles differ; the message names the differing fields, never a secret."""


def canonical_json(obj: Any) -> str:
    """Sorted keys, compact separators, non-ASCII kept: the one canonical form mesa-clm hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_fp(obj: Any) -> str:
    """First ``FP_HEX`` hex characters of the sha256 of ``obj``'s canonical JSON."""
    return sha256_hex(canonical_json(obj))[:FP_HEX]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _hex(value: str, length: int, what: str) -> str:
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{what} must be {length} lower-case hex characters")
    return value


class EncoderSpec(_Frozen):
    """The encoder recipe that produced a 4096-d vector (D5; plan §6.1, §6.6).

    ``pooling`` is always ``LAST`` (vLLM's pooling runner on a decoder converts it to last-token
    pooling; the fallback reads ``last_hidden_state`` at the last non-pad token) and
    ``truncation_side`` ``left`` (PR #6: the tail carries the question). ``prefix_caching`` is
    off until golden-vector parity with it on is recorded; enabling it rotates the fingerprint.
    ``route`` names the serving path: ``vllm`` (the container) or ``transformers`` (the
    in-process fallback).
    """

    model: str
    revision: str
    dtype: str = "bfloat16"
    pooling: Literal["LAST"] = "LAST"
    normalize: bool = True
    max_len: int = Field(default=4096, gt=0)
    truncation_side: Literal["left", "right"] = "left"
    prefix_caching: bool = False
    route: Literal["vllm", "transformers", "fake"] = "vllm"

    def as_dict(self) -> dict[str, Any]:
        """The nine hashed fields, in the D5 order (order does not change the hash)."""
        return {
            "model": self.model,
            "revision": self.revision,
            "dtype": self.dtype,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_len": self.max_len,
            "truncation_side": self.truncation_side,
            "prefix_caching": self.prefix_caching,
            "route": self.route,
        }


def encoder_fp(spec: EncoderSpec) -> str:
    """``sha256(canonical json of the spec)[:12]`` (D5)."""
    return short_fp(spec.as_dict())


class ClmModelSpec(_Frozen):
    """What scored the candidates: the served head and the engine commit (module docstring)."""

    head_name: str = Field(min_length=1)
    # sha256 of the served .pt; '' for clm-raw (no head).
    head_sha256: str = ""
    clm_commit: str

    @field_validator("head_sha256")
    @classmethod
    def _head_sha(cls, v: str) -> str:
        return v if v == "" else _hex(v, _SHA256_HEX, "head_sha256")

    @field_validator("clm_commit")
    @classmethod
    def _commit(cls, v: str) -> str:
        return _hex(v, _COMMIT_HEX, "clm_commit")

    def as_dict(self) -> dict[str, Any]:
        return {
            "head_name": self.head_name,
            "head_sha256": self.head_sha256,
            "clm_commit": self.clm_commit,
        }


def clm_model_fp(spec: ClmModelSpec) -> str:
    """``sha256(canonical json {head_name, head_sha256, clm_commit})[:12]``."""
    return short_fp(spec.as_dict())


class Fingerprint(_Frozen):
    """The four D5 hashes every record carries. ``serving_lock_sha`` is the full 64-hex
    ``lock_sha``; the two ``*_fp`` fields are 12 hex characters; ``schema_sha256`` is 64."""

    encoder_fp: str = Field(min_length=FP_HEX, max_length=FP_HEX)
    clm_model_fp: str = Field(min_length=FP_HEX, max_length=FP_HEX)
    schema_sha256: str
    serving_lock_sha: str

    @field_validator("schema_sha256", "serving_lock_sha")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _hex(v, _SHA256_HEX, "sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "encoder_fp": self.encoder_fp,
            "clm_model_fp": self.clm_model_fp,
            "schema_sha256": self.schema_sha256,
            "serving_lock_sha": self.serving_lock_sha,
        }

    def check(self, other: Fingerprint | Mapping[str, Any], *, what: str = "fingerprint") -> None:
        """Raise :class:`FingerprintMismatch` unless ``other`` equals this bundle field by field.

        ``other`` may be a stored ``as_dict()`` (an artifact manifest, a bench cell); a missing
        field counts as a mismatch. ``what`` names the thing being checked in the message.
        """
        mine = self.as_dict()
        theirs = other.as_dict() if isinstance(other, Fingerprint) else dict(other)
        diffs = [
            f"{k}: expected {mine[k]!r}, got {theirs.get(k)!r}"
            for k in mine
            if theirs.get(k) != mine[k]
        ]
        if diffs:
            raise FingerprintMismatch(
                f"{what} does not match the live fingerprint (K4: re-bench before use): "
                + "; ".join(diffs)
            )

    def matches(self, other: Fingerprint | Mapping[str, Any]) -> bool:
        try:
            self.check(other)
        except FingerprintMismatch:
            return False
        return True


def fingerprint(encoder: EncoderSpec, model: ClmModelSpec, *, serving_lock_sha: str) -> Fingerprint:
    """The bundle for a live (encoder, head) pair under the vendored contract and a lock."""
    return Fingerprint(
        encoder_fp=encoder_fp(encoder),
        clm_model_fp=clm_model_fp(model),
        schema_sha256=render.schema_sha256(),
        serving_lock_sha=serving_lock_sha,
    )


# -- serving.lock.json (plan §6.7) -------------------------------------------------------------


class LockPatch(_Frozen):
    """One carried patch: ``pr`` and ``pr_head_sha`` are null for the local patches (0005, 0006)."""

    id: str
    pr: int | None = None
    pr_head_sha: str | None = None
    sha256: str

    @field_validator("pr_head_sha")
    @classmethod
    def _head(cls, v: str | None) -> str | None:
        return None if v is None else _hex(v, _COMMIT_HEX, "pr_head_sha")

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _hex(v, _SHA256_HEX, "patch sha256")


class LockHead(_Frozen):
    """The released head file (HF repo, revision, file, size, sha256)."""

    repo: str
    revision: str
    file: str
    size: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _hex(v, _SHA256_HEX, "head sha256")


class LockImage(_Frozen):
    ref: str
    tag: str
    digest: str


class ServingLock(_Frozen):
    """``serving/serving.lock.json``: the pins the doctor compares the host against (§6.7, §6.8).

    ``lock_sha`` is the sha256 of the canonical JSON of every other field (``lock_sha`` itself
    removed), so the file is self-verifying; ``encoder_fp`` must equal
    :func:`encoder_fp` of ``encoder``. :func:`load_serving_lock` checks both.
    """

    clm_commit: str
    patches: list[LockPatch]
    schema_sha256: str
    head: LockHead
    encoder: EncoderSpec
    image: LockImage
    encoder_fp: str
    lock_sha: str

    @field_validator("clm_commit")
    @classmethod
    def _commit(cls, v: str) -> str:
        return _hex(v, _COMMIT_HEX, "clm_commit")

    @field_validator("schema_sha256", "lock_sha")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _hex(v, _SHA256_HEX, "sha256")

    def body(self) -> dict[str, Any]:
        """Every field but ``lock_sha``, as plain JSON data (what ``lock_sha`` hashes)."""
        data = self.model_dump(mode="json")
        data.pop(LOCK_SHA_FIELD)
        return data

    def model_spec(self, head_name: str = "clm-latest") -> ClmModelSpec:
        """The :class:`ClmModelSpec` of a served name under this lock: the pinned head file for
        ``clm-latest`` (and any promoted head, whose own sha the caller passes through
        :class:`ClmModelSpec` directly), no head for ``clm-raw``."""
        sha = "" if head_name == "clm-raw" else self.head.sha256
        return ClmModelSpec(head_name=head_name, head_sha256=sha, clm_commit=self.clm_commit)

    def fingerprint(self, head_name: str = "clm-latest") -> Fingerprint:
        """The live bundle for ``head_name`` under this lock and the vendored contract."""
        return Fingerprint(
            encoder_fp=self.encoder_fp,
            clm_model_fp=clm_model_fp(self.model_spec(head_name)),
            schema_sha256=render.schema_sha256(),
            serving_lock_sha=self.lock_sha,
        )


class LockError(ValueError):
    """``serving.lock.json`` is malformed or its recorded hashes do not match its content."""


def lock_sha_of(body: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a lock body (``lock_sha`` removed if present)."""
    data = {k: v for k, v in body.items() if k != LOCK_SHA_FIELD}
    return sha256_hex(canonical_json(data))


def parse_serving_lock(
    data: Mapping[str, Any], *, source: str = "serving.lock.json"
) -> ServingLock:
    """Validate a lock document and its two recorded hashes (``lock_sha``, ``encoder_fp``).

    The vendored ``schema_sha256`` is *not* compared here: a lock may legitimately describe a
    serving host that has not been rebuilt yet; the doctor reports that difference (§6.8), and
    :meth:`Fingerprint.check` refuses the resulting mismatch at use time (K4).
    """
    try:
        lock = ServingLock.model_validate(dict(data))
    except ValidationError as exc:
        # Pydantic's message names fields and problems, never the (non-secret) values we care about.
        raise LockError(f"{source}: {exc.error_count()} validation error(s): {exc}") from None
    expected_sha = lock_sha_of(lock.body())
    if lock.lock_sha != expected_sha:
        raise LockError(
            f"{source}: lock_sha {lock.lock_sha[:12]}… does not match the content "
            f"({expected_sha[:12]}…); the file was edited without re-signing"
        )
    expected_fp = encoder_fp(lock.encoder)
    if lock.encoder_fp != expected_fp:
        raise LockError(
            f"{source}: encoder_fp {lock.encoder_fp} does not match the encoder block "
            f"({expected_fp})"
        )
    return lock


def load_serving_lock(path: str | Path) -> ServingLock:
    """Read and verify ``serving.lock.json`` (:func:`parse_serving_lock`)."""
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LockError(f"{p}: serving lock not found") from None
    except json.JSONDecodeError as exc:
        raise LockError(f"{p}: not valid JSON ({exc.msg} at line {exc.lineno})") from None
    if not isinstance(data, dict):
        raise LockError(f"{p}: the lock must be a JSON object")
    return parse_serving_lock(data, source=str(p))


def sign_lock_body(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``body`` with ``encoder_fp`` and ``lock_sha`` (re)computed: what the M1-A writer
    of ``serving.lock.json`` calls before serialising, so the file always verifies."""
    data = {k: v for k, v in body.items() if k != LOCK_SHA_FIELD}
    data["encoder_fp"] = encoder_fp(EncoderSpec.model_validate(dict(data["encoder"])))
    data[LOCK_SHA_FIELD] = lock_sha_of(data)
    return data
