"""Fingerprints (``mesa_clm.clm.fingerprint``, DESIGN D5; plan §6.7): the D5 field set and
12-hex length of ``encoder_fp`` / ``clm_model_fp``, ``batch_invariance`` hashed only when set
(DESIGN A3), ``schema_sha256`` from the vendored file, the bundle's exact-match check, and the
self-verifying ``serving.lock.json`` with its container recipe."""

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
    LockRecipe,
    ServingLock,
    canonical_json,
    clm_model_fp,
    encoder_fp,
    fingerprint,
    load_serving_lock,
    lock_sha_of,
    parse_serving_lock,
    recipe_conflicts,
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


def test_batch_invariance_enters_the_hash_only_when_set() -> None:
    """DESIGN A3: ``none`` (the default, the M1-A recipe) leaves the D5 dict and every existing
    fingerprint unchanged; ``kernels`` and ``serial`` rotate it and each other."""
    base = spec()
    assert base.batch_invariance == "none"
    assert "batch_invariance" not in base.as_dict()
    assert encoder_fp(spec(batch_invariance="none")) == encoder_fp(base)
    # The M1-A vLLM recipe's fingerprint (serving_m1.json, collapse_spike.json) stays put.
    assert encoder_fp(base) == "852efc921a8a"
    kernels, serial = spec(batch_invariance="kernels"), spec(batch_invariance="serial")
    assert kernels.as_dict()["batch_invariance"] == "kernels"
    fps = {encoder_fp(base), encoder_fp(kernels), encoder_fp(serial)}
    assert len(fps) == 3
    expected = hashlib.sha256(canonical_json(kernels.as_dict()).encode()).hexdigest()[:12]
    assert encoder_fp(kernels) == expected
    with pytest.raises(ValueError):
        spec(batch_invariance="sometimes")


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


def recipe_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "args": [
            "Qwen/Qwen3-8B",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--no-enable-prefix-caching",
            "--middleware",
            "vllm_auth.require_api_key",
        ],
        "env": {"PYTHONPATH": "/opt/mesa-clm-auth", "HF_HUB_OFFLINE": "1"},
        "auth": {
            "middleware": "vllm_auth.require_api_key",
            "file": "vllm_auth.py",
            "sha256": "e" * 64,
            "mount": "/opt/mesa-clm-auth",
            "open_paths": ["/health"],
        },
        "truncate_prompt_tokens": 4095,
    }
    body.update(over)
    return body


def test_lock_with_a_recipe_signs_parses_and_hashes_it() -> None:
    signed = sign_lock_body({**lock_body(), "recipe": recipe_body()})
    lock = parse_serving_lock(signed)
    assert isinstance(lock.recipe, LockRecipe) and lock.recipe.truncate_prompt_tokens == 4095
    assert lock.recipe.flag("--max-num-seqs") == "8" and lock.recipe.flag("--nope") is None
    assert lock.body()["recipe"] == recipe_body()
    assert lock.encoder_fp == encoder_fp(spec())  # the recipe is not part of encoder_fp
    plain = sign_lock_body(lock_body())
    assert plain["lock_sha"] != signed["lock_sha"]  # ... but it is part of the lock
    edited = {**signed, "recipe": recipe_body(truncate_prompt_tokens=4000)}
    with pytest.raises(LockError, match="lock_sha"):
        parse_serving_lock(edited)


def test_a_lock_without_a_recipe_hashes_as_before() -> None:
    """Locks written before DESIGN A3 (no recipe, no batch_invariance) still verify."""
    plain = sign_lock_body(lock_body())
    lock = parse_serving_lock(plain)
    assert lock.recipe is None and "recipe" not in lock.body()
    assert "batch_invariance" not in lock.body()["encoder"]
    assert lock_sha_of(lock.body()) == plain["lock_sha"]


@pytest.mark.parametrize(
    ("encoder", "recipe", "match"),
    [
        ({}, {"truncate_prompt_tokens": 4096}, "must stay below max_len"),
        ({"max_len": 2048}, {"truncate_prompt_tokens": 2047}, "--max-model-len '4096'"),
        ({"prefix_caching": True}, {}, "prefix caching flag"),
        ({"batch_invariance": "kernels"}, {}, "VLLM_BATCH_INVARIANT None"),
        (
            {},
            {"env": {"PYTHONPATH": "/opt/mesa-clm-auth", "VLLM_BATCH_INVARIANT": "1"}},
            "VLLM_BATCH_INVARIANT '1'",
        ),
        ({"batch_invariance": "serial"}, {}, "--max-num-seqs 1"),
        ({}, {"env": {"PYTHONPATH": "/elsewhere"}}, "PYTHONPATH '/elsewhere'"),
    ],
)
def test_a_recipe_that_contradicts_the_encoder_is_refused(
    encoder: dict[str, Any], recipe: dict[str, Any], match: str
) -> None:
    body = {**lock_body(), "encoder": spec(**encoder).as_dict(), "recipe": recipe_body(**recipe)}
    with pytest.raises(LockError, match="contradicts the encoder") as info:
        parse_serving_lock(sign_lock_body(body))
    assert match in str(info.value)


def test_recipe_conflicts_accepts_the_consistent_variants() -> None:
    kernels_env = {"PYTHONPATH": "/opt/mesa-clm-auth", "VLLM_BATCH_INVARIANT": "1"}
    kernels = LockRecipe.model_validate(recipe_body(env=kernels_env))
    assert recipe_conflicts(spec(batch_invariance="kernels"), kernels) == []
    serial_args = [a if a != "8" else "1" for a in recipe_body()["args"]]
    serial = LockRecipe.model_validate(recipe_body(args=serial_args))
    assert recipe_conflicts(spec(batch_invariance="serial"), serial) == []
    unguarded = LockRecipe.model_validate(recipe_body(args=recipe_body()["args"][:-2]))
    assert any("--middleware" in p for p in recipe_conflicts(spec(), unguarded))


def test_a_recipe_never_names_a_key() -> None:
    for name in ("VLLM_API_KEY", "CLM_API_KEY", "CLM_EMB_API_KEY"):
        with pytest.raises(ValueError, match=name):
            LockRecipe.model_validate(recipe_body(env={name: "x", "PYTHONPATH": "/p"}))
    with pytest.raises(ValueError):
        LockRecipe.model_validate(recipe_body(truncate_prompt_tokens=0))


def test_lock_encoder_block_is_the_d5_spec() -> None:
    lock = parse_serving_lock(sign_lock_body(lock_body()))
    assert lock.encoder == spec()
    assert lock.body()["encoder"] == spec().as_dict()
    assert "lock_sha" not in lock.body() and "encoder_fp" in lock.body()


# -- DESIGN A5: the network mode, and the vector recipe the feature store keys on -----------------


def test_a_recipe_without_network_hashes_as_before_and_with_it_rotates_the_lock() -> None:
    """``network`` is absent from locks written before A5 and leaves their ``lock_sha`` alone;
    setting it changes the lock (it is part of the recipe) but never ``encoder_fp``."""
    old = sign_lock_body({**lock_body(), "recipe": recipe_body()})
    lock = parse_serving_lock(old)
    assert lock.recipe is not None and lock.recipe.network is None
    assert "network" not in lock.body()["recipe"]
    assert lock_sha_of(lock.body()) == old["lock_sha"]
    args = [*recipe_body()["args"], "--uds", "/run/mesa-clm/encoder.sock"]
    new = sign_lock_body({**lock_body(), "recipe": recipe_body(args=args, network="none")})
    isolated = parse_serving_lock(new)
    assert isolated.recipe is not None and isolated.recipe.network == "none"
    assert isolated.body()["recipe"]["network"] == "none"
    assert new["lock_sha"] != old["lock_sha"] and new["encoder_fp"] == old["encoder_fp"]


@pytest.mark.parametrize(
    ("args_extra", "network", "match"),
    [
        ([], "none", "needs --uds"),
        (["--uds", "/run/x.sock", "--host", "0.0.0.0"], "none", "nothing for --host"),  # noqa: S104
        (["--uds", "/run/x.sock", "--port", "8090"], "none", "nothing for --port"),
        ([], "bridge", "network 'bridge'"),
    ],
)
def test_a_network_none_recipe_must_serve_on_a_socket(
    args_extra: list[str], network: str, match: str
) -> None:
    body = recipe_body(args=[*recipe_body()["args"], *args_extra], network=network)
    problems = recipe_conflicts(spec(), LockRecipe.model_validate(body))
    assert any(match in p for p in problems), problems


def test_the_vector_recipe_names_what_a_vector_depends_on() -> None:
    """Image digest, encoder spec, arguments, environment and cap enter the vector recipe; the
    bearer guard, the patches and the head do not (they never touch a vector)."""
    base = {**lock_body(), "recipe": recipe_body()}
    lock = parse_serving_lock(sign_lock_body(base))
    vr = lock.vector_recipe()
    assert vr["image_digest"] == "sha256:" + "c" * 64 and vr["encoder_fp"] == lock.encoder_fp
    assert vr["args"] == recipe_body()["args"] and vr["truncate_prompt_tokens"] == 4095
    sha = lock.vector_recipe_sha256()
    assert len(sha) == 64 and sha == hashlib.sha256(canonical_json(vr).encode()).hexdigest()

    def other(**over: Any) -> str:
        body = {**base, **over}
        return parse_serving_lock(sign_lock_body(body)).vector_recipe_sha256()

    image = {"ref": "vllm/vllm-openai", "tag": "v0.30.0", "digest": "sha256:" + "d" * 64}
    assert other(image=image) != sha
    assert other(recipe=recipe_body(args=[*recipe_body()["args"], "--enforce-eager"])) != sha
    assert other(recipe=recipe_body(truncate_prompt_tokens=2047)) != sha
    env = recipe_body(env={**recipe_body()["env"], "VLLM_BATCH_INVARIANT": "1"})
    assert other(recipe=env, encoder=spec(batch_invariance="kernels").as_dict()) != sha
    guard = recipe_body(auth={**recipe_body()["auth"], "sha256": "f" * 64})
    assert other(recipe=guard) == sha
    head = {**lock_body()["head"], "sha256": "0" * 64}
    assert other(head=head) == sha
    plain = parse_serving_lock(sign_lock_body(lock_body()))
    assert plain.vector_recipe()["args"] is None and plain.vector_recipe_sha256() != sha
