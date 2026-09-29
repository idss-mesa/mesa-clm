"""Fingerprints (``mesa_clm.clm.fingerprint``, DESIGN D5; plan §6.7): the D5 field set and
12-hex length of ``encoder_fp`` / ``clm_model_fp``, ``schema_sha256`` from the vendored file,
the bundle's exact-match check, and the self-verifying ``serving.lock.json``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import render
from mesa_clm.clm.fingerprint import (
    ClmModelSpec,
    EncoderSpec,
    Fingerprint,
    FingerprintMismatch,
    LockError,
    ServingLock,
    canonical_json,
    clm_model_fp,
    encoder_fp,
    fingerprint,
    load_serving_lock,
    lock_sha_of,
    parse_serving_lock,
    sign_lock_body,
)

SCHEMA_SHA = "52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335"
CLM_COMMIT = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
HEAD_SHA = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"


def spec(**over: Any) -> EncoderSpec:
    base: dict[str, Any] = {"model": "Qwen/Qwen3-8B", "revision": QWEN_REV}
    base.update(over)
    return EncoderSpec(**base)


def lock_body() -> dict[str, Any]:
    return {
        "clm_commit": CLM_COMMIT,
        "patches": [
            {
                "id": "0001-truncation-side",
                "pr": 6,
                "pr_head_sha": "11211fcab1c2" + "0" * 28,
                "sha256": "a" * 64,
            },
            {"id": "0005-embedder-key", "pr": None, "pr_head_sha": None, "sha256": "b" * 64},
        ],
        "schema_sha256": SCHEMA_SHA,
        "head": {
            "repo": "Contrastive-LM/CLM-v0.1-8B",
            "revision": "e939398d4556fcd9400c76fa8c5a513202f42b0a",
            "file": "CLM_v0.1-8B.pt",
            "size": 75557149,
            "sha256": HEAD_SHA,
        },
        "encoder": spec().as_dict(),
        "image": {"ref": "vllm/vllm-openai", "tag": "v0.27.1", "digest": "sha256:" + "c" * 64},
    }


# -- encoder_fp -----------------------------------------------------------------------------------


def test_encoder_fp_is_sha256_of_the_d5_fields() -> None:
    s = spec()
    fields = {
        "model": "Qwen/Qwen3-8B",
        "revision": QWEN_REV,
        "dtype": "bfloat16",
        "pooling": "LAST",
        "normalize": True,
        "max_len": 4096,
        "truncation_side": "left",
        "prefix_caching": False,
        "route": "vllm",
    }
    assert s.as_dict() == fields
    expected = hashlib.sha256(canonical_json(fields).encode()).hexdigest()[:12]
    assert encoder_fp(s) == expected and len(expected) == 12
    # Sorted-key canonical form: field order cannot change the hash.
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == '{"a":[1,{"c":3,"d":2}],"b":1}'


@pytest.mark.parametrize(
    "change",
    [
        {"route": "transformers"},
        {"prefix_caching": True},
        {"max_len": 2048},
        {"truncation_side": "right"},
        {"revision": "0" * 40},
        {"dtype": "float16"},
        {"normalize": False},
        {"model": "Qwen/Qwen3-8B-Base"},
    ],
)
def test_every_d5_field_rotates_encoder_fp(change: dict[str, Any]) -> None:
    assert encoder_fp(spec(**change)) != encoder_fp(spec())


def test_encoder_spec_is_strict() -> None:
    with pytest.raises(ValueError):
        EncoderSpec(model="m", revision="r", pooling="MEAN")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EncoderSpec(model="m", revision="r", extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        EncoderSpec(model="m", revision="r", max_len=0)


# -- clm_model_fp ---------------------------------------------------------------------------------


def test_clm_model_fp_definition() -> None:
    m = ClmModelSpec(head_name="clm-latest", head_sha256=HEAD_SHA, clm_commit=CLM_COMMIT)
    fields = {"head_name": "clm-latest", "head_sha256": HEAD_SHA, "clm_commit": CLM_COMMIT}
    assert m.as_dict() == fields
    assert clm_model_fp(m) == hashlib.sha256(canonical_json(fields).encode()).hexdigest()[:12]
    raw = ClmModelSpec(head_name="clm-raw", clm_commit=CLM_COMMIT)
    assert raw.head_sha256 == "" and clm_model_fp(raw) != clm_model_fp(m)
    # Same name, re-trained head: a different fingerprint (artifacts never silently carry over).
    retrained = ClmModelSpec(head_name="clm-latest", head_sha256="0" * 64, clm_commit=CLM_COMMIT)
    assert clm_model_fp(retrained) != clm_model_fp(m)
    other_engine = ClmModelSpec(head_name="clm-latest", head_sha256=HEAD_SHA, clm_commit="1" * 40)
    assert clm_model_fp(other_engine) != clm_model_fp(m)


def test_clm_model_spec_validates_hex() -> None:
    with pytest.raises(ValueError, match="clm_commit"):
        ClmModelSpec(head_name="x", clm_commit="short")
    with pytest.raises(ValueError, match="head_sha256"):
        ClmModelSpec(head_name="x", head_sha256="G" * 64, clm_commit=CLM_COMMIT)
    with pytest.raises(ValueError):
        ClmModelSpec(head_name="", clm_commit=CLM_COMMIT)


# -- the bundle -----------------------------------------------------------------------------------


def test_bundle_uses_the_vendored_schema_sha() -> None:
    fp = fingerprint(
        spec(),
        ClmModelSpec(head_name="clm-latest", head_sha256=HEAD_SHA, clm_commit=CLM_COMMIT),
        serving_lock_sha="d" * 64,
    )
    assert fp.schema_sha256 == render.schema_sha256() == SCHEMA_SHA
    assert set(fp.as_dict()) == {"encoder_fp", "clm_model_fp", "schema_sha256", "serving_lock_sha"}
    assert fp.encoder_fp == encoder_fp(spec()) and len(fp.clm_model_fp) == 12


def test_bundle_check_is_exact_and_names_the_field() -> None:
    fp = fingerprint(
        spec(),
        ClmModelSpec(head_name="clm-latest", head_sha256=HEAD_SHA, clm_commit=CLM_COMMIT),
        serving_lock_sha="d" * 64,
    )
    fp.check(fp)
    fp.check(fp.as_dict())
    assert fp.matches(fp.as_dict())
    drifted = fingerprint(
        spec(route="transformers"),
        ClmModelSpec(head_name="clm-latest", head_sha256=HEAD_SHA, clm_commit=CLM_COMMIT),
        serving_lock_sha="d" * 64,
    )
    with pytest.raises(FingerprintMismatch, match="encoder_fp") as info:
        fp.check(drifted, what="artifact v3")
    assert "artifact v3" in str(info.value) and "K4" in str(info.value)
    assert "clm_model_fp" not in str(info.value)
    assert not fp.matches({**fp.as_dict(), "serving_lock_sha": "e" * 64})
    with pytest.raises(FingerprintMismatch, match="serving_lock_sha"):
        fp.check({k: v for k, v in fp.as_dict().items() if k != "serving_lock_sha"})


def test_bundle_validates_lengths() -> None:
    with pytest.raises(ValueError):
        Fingerprint(
            encoder_fp="abc",
            clm_model_fp="0" * 12,
            schema_sha256=SCHEMA_SHA,
            serving_lock_sha="d" * 64,
        )
    with pytest.raises(ValueError):
        Fingerprint(
            encoder_fp="0" * 12,
            clm_model_fp="0" * 12,
            schema_sha256="nothex",
            serving_lock_sha="d" * 64,
        )


# -- serving.lock.json ----------------------------------------------------------------------------


def test_sign_parse_and_load_lock(tmp_path: Path) -> None:
    signed = sign_lock_body(lock_body())
    assert signed["encoder_fp"] == encoder_fp(spec())
    assert signed["lock_sha"] == lock_sha_of({k: v for k, v in signed.items() if k != "lock_sha"})
    # Signing is idempotent and ignores a stale lock_sha in the input.
    assert sign_lock_body({**signed, "lock_sha": "0" * 64}) == signed
    lock = parse_serving_lock(signed)
    assert isinstance(lock, ServingLock) and lock.clm_commit == CLM_COMMIT
    assert lock.patches[1].pr is None and lock.patches[0].pr == 6
    path = tmp_path / "serving.lock.json"
    path.write_text(json.dumps(signed, indent=2))
    loaded = load_serving_lock(path)
    assert loaded == lock
    fp = loaded.fingerprint()
    assert fp.serving_lock_sha == lock.lock_sha and fp.encoder_fp == lock.encoder_fp
    assert fp.schema_sha256 == render.schema_sha256()
    assert fp.clm_model_fp == clm_model_fp(
        ClmModelSpec(head_name="clm-latest", head_sha256=HEAD_SHA, clm_commit=CLM_COMMIT)
    )
    raw = loaded.fingerprint("clm-raw")
    assert raw.clm_model_fp == clm_model_fp(
        ClmModelSpec(head_name="clm-raw", clm_commit=CLM_COMMIT)
    )
    assert raw.clm_model_fp != fp.clm_model_fp and raw.encoder_fp == fp.encoder_fp
    assert (
        loaded.model_spec("mesa-term-v2").head_sha256 == HEAD_SHA
    )  # the caller overrides for promoted heads


def test_lock_refuses_tampering_and_bad_files(tmp_path: Path) -> None:
    signed = sign_lock_body(lock_body())
    edited = {**signed, "clm_commit": "1" * 40}
    with pytest.raises(LockError, match="lock_sha"):
        parse_serving_lock(edited)
    wrong_fp = {**signed, "encoder_fp": "000000000000"}
    wrong_fp["lock_sha"] = lock_sha_of(wrong_fp)
    with pytest.raises(LockError, match="encoder_fp"):
        parse_serving_lock(wrong_fp)
    with pytest.raises(LockError, match="validation error"):
        parse_serving_lock({**signed, "unexpected": 1})
    with pytest.raises(LockError, match="validation error"):
        parse_serving_lock({k: v for k, v in signed.items() if k != "head"})
    with pytest.raises(LockError, match="not found"):
        load_serving_lock(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(LockError, match="not valid JSON"):
        load_serving_lock(bad)
    arr = tmp_path / "arr.json"
    arr.write_text("[]")
    with pytest.raises(LockError, match="JSON object"):
        load_serving_lock(arr)


def test_lock_encoder_block_is_the_d5_spec() -> None:
    lock = parse_serving_lock(sign_lock_body(lock_body()))
    assert lock.encoder == spec()
    assert lock.body()["encoder"] == spec().as_dict()
    assert "lock_sha" not in lock.body() and "encoder_fp" in lock.body()
