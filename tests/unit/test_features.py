"""The feature cache, the X1/X2 text manifest and the TextCache export
(``mesa_clm.learn.features``; plan §5.2, §5.3, §5.6; DESIGN D5, D11, D23, D30).

The store: owner-only layout, the per-operation flock (shared to read, exclusive to write), the
float32 vectors returned bit for bit, the fp16 round-trip gate of the export, truncated texts,
the encoder_fp, dimension, format-1 and vector-recipe refusals, cached head projections that
refuse a second head under one ``clm_model_fp``. The manifest:
built from the committed labels snapshot through the framings and the vendored contract only,
deterministic, label-free, every context equal to what the builders make from the fixture card
(the snapshot's ``state_json`` has sorted keys; rendering needs builder order), the X2 joint
specs (S1 = the anyjev state + the task question, S1ns = without it). The export: read back with
a copy of CLM's own ``TextCache``. No network, no GPU; vectors come from the fake encoder."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from mesa_clm import framings, render
from mesa_clm.cards import load_card
from mesa_clm.clm.fake import FakeEncoder
from mesa_clm.clm.fingerprint import (
    ClmModelSpec,
    ServingLock,
    clm_model_fp,
    load_serving_lock,
    parse_serving_lock,
    sign_lock_body,
)
from mesa_clm.clm.headproj import HeadProjector, l2, random_head
from mesa_clm.config import Config
from mesa_clm.learn import features as feat
from mesa_clm.learn.features import (
    FP16_MIN_COSINE,
    FeatureMissing,
    FeatureStore,
    FeatureStoreError,
    ManifestError,
    builder_ordered,
    export_npz,
    fp16_roundtrip_min_cosine,
    manifest,
    text_sha256,
)
from mesa_clm.providers.tiered import fake_fingerprint
from mesa_clm.registry import ANCHOR_KEY, ANCHORS, entry
from mesa_clm.serving import CLM_COMMIT
from mesa_clm.states import candidate_state, ontology_state, state_sha256, target_state
from mesa_clm.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "bench" / "snapshots" / "2026-09-29.parquet"
LOCK = ROOT / "serving" / "serving.lock.json"
CARDS = ROOT / "tests" / "fixtures" / "cards"
# Computed, never a literal: the encoder spec may gain fields (DESIGN A3).
FP = fake_fingerprint().encoder_fp
TEXTS = [
    "card: DP1.10003.001.brd_countdata\n\ncolumn: observerDistance",
    "distance: A 1-D extent quality equal to the distance between two points.",
    "None of these terms is the right concept for this target.",
]


def _store(tmp_path: Path, fp: str = FP) -> FeatureStore:
    return FeatureStore(tmp_path / "features", fp)


def _vectors(texts: list[str]) -> np.ndarray:
    vecs, _ = FakeEncoder().embed(texts)
    return vecs


def _tokens(texts: list[str]) -> list[int]:
    return [len(t.split()) for t in texts]


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# -- the store ---------------------------------------------------------------------------------


def test_layout_is_owner_only_and_records_the_fingerprint(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.path == tmp_path / "features" / FP / "features.duckdb"
    assert not (tmp_path / "features").exists()  # the constructor touches nothing
    assert store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS)) == 3
    assert _mode(store.dir) == 0o700
    assert _mode(store.path) == 0o600 and _mode(store.lock_path) == 0o600
    assert store.lock_path == store.dir / "features.lock"
    meta = store.meta()
    assert meta["format"] == feat.FORMAT and meta["encoder_fp"] == FP and meta["dim"] == "4096"
    assert "encoder_spec" not in meta
    spec = {"model": "fake-ngram", "route": "fake"}
    store.ensure(encoder_spec=spec)
    assert json.loads(store.meta()["encoder_spec"]) == spec


def test_add_get_missing_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    vecs = _vectors(TEXTS)
    assert store.missing([*TEXTS, TEXTS[0]]) == TEXTS  # distinct, first-seen order
    assert store.add(TEXTS[:2], vecs[:2], _tokens(TEXTS[:2])) == 2
    assert store.missing([TEXTS[2], TEXTS[0], TEXTS[2]]) == [TEXTS[2]]
    got = store.get([TEXTS[1], TEXTS[0], TEXTS[1]])
    # The float32 unit vectors themselves, not a float16 copy (format 2; X1's 1e-4 needs them).
    want = l2(vecs)
    assert got.dtype == np.float32 and got.shape == (3, 4096)
    np.testing.assert_array_equal(got, want[[1, 0, 1]])
    assert not np.array_equal(got, want[[1, 0, 1]].astype(np.float16).astype(np.float32))
    # First write wins: another vector for a stored text is ignored, the new text is stored.
    other = _vectors(["something else entirely"])
    assert store.add([TEXTS[0], TEXTS[2]], np.vstack([other, vecs[2:]]), [1, 2]) == 1
    np.testing.assert_array_equal(store.get([TEXTS[0]])[0], want[0])
    assert store.missing(TEXTS) == []
    assert store.get([]).shape == (0, 4096)
    assert store.embedded_texts() == sorted(TEXTS, key=text_sha256)
    with pytest.raises(FeatureMissing) as err:
        store.get([TEXTS[0], "never embedded"])
    assert err.value.missing == [text_sha256("never embedded")]
    assert "features build" in str(err.value)


def test_fp16_gate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert fp16_roundtrip_min_cosine(store) is None
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5, 4096)) * 3.0  # not unit: the store normalises first
    texts = [f"text {i}" for i in range(5)]
    store.add(texts, raw, [2] * 5)
    x32 = l2(raw)
    x16 = x32.astype(np.float16).astype(np.float64)
    a = x32.astype(np.float64)
    cos = (a * x16).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(x16, axis=1))
    np.testing.assert_allclose(np.sort(store.fp16_cosines()), np.sort(cos), rtol=0, atol=1e-15)
    gate = fp16_roundtrip_min_cosine(store)
    assert gate is not None and gate >= FP16_MIN_COSINE and gate == pytest.approx(cos.min())
    assert store.stats()["fp16_min_cosine"] == gate


def test_truncated_texts_are_recorded_never_embedded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    long_text = "word " * 50
    assert store.add_truncated([long_text, long_text], [5000, 5000]) == 1
    assert store.add_truncated([long_text], [5000]) == 0
    assert store.missing([long_text, TEXTS[0]]) == [TEXTS[0]]
    with pytest.raises(FeatureMissing):
        store.get([long_text])
    with pytest.raises(FeatureStoreError, match="recorded as truncated"):
        store.add([long_text], _vectors([long_text]), [5000])
    store.add(TEXTS[:1], _vectors(TEXTS[:1]), [7])
    with pytest.raises(FeatureStoreError, match="already have a vector"):
        store.add_truncated(TEXTS[:1], [7])
    stats = store.stats()
    assert stats["texts"] == 2 and stats["vectors"] == 1 and stats["truncated"] == 1
    assert stats["tokens"]["total"] == 5007 and stats["tokens"]["max"] == 5000


@pytest.mark.parametrize(
    ("vectors", "tokens", "match"),
    [
        (np.ones((3, 8)), [1, 1, 1], r"\[n, 4096\]"),
        (np.zeros((3, 4096)), [1, 1, 1], "zero vector"),
        (np.full((3, 4096), np.nan), [1, 1, 1], "non-finite"),
        (np.ones((2, 4096)), [1, 1, 1], "3 texts but 2 vectors"),
        (np.ones((3, 4096)), [1, -1, 1], "non-negative"),
        (np.ones((3, 4096)), [1, 1], "3 texts but 2 token counts"),
    ],
)
def test_add_refuses_bad_input_before_writing(
    tmp_path: Path, vectors: np.ndarray, tokens: list[int], match: str
) -> None:
    store = _store(tmp_path)
    with pytest.raises(FeatureStoreError, match=match):
        store.add(TEXTS, vectors, tokens)
    assert not store.exists()


def test_reads_never_create_anything(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.missing(TEXTS) == TEXTS
    assert store.meta() == {} and store.embedded_texts() == []
    assert store.stats() == {"path": str(store.path), "encoder_fp": FP, "exists": False}
    with pytest.raises(FeatureMissing):
        store.get(TEXTS)
    assert fp16_roundtrip_min_cosine(store) is None
    assert not (tmp_path / "features").exists()


def test_store_refuses_another_fingerprint_or_dimension(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    other_fp = hashlib.sha256(FP.encode()).hexdigest()[:12]
    shutil.copytree(store.dir, store.root / other_fp)
    moved = FeatureStore(store.root, other_fp)
    with pytest.raises(FeatureStoreError, match="encoder_fp") as err:
        moved.ensure()
    assert "D5" in str(err.value)
    with pytest.raises(FeatureStoreError, match="encoder_fp"):
        moved.missing(TEXTS)
    with pytest.raises(FeatureStoreError, match="dim"):
        FeatureStore(store.root, FP, dim=8).ensure()
    for bad in ("852EFC921A8A", "short", "../escape12", ""):
        with pytest.raises(FeatureStoreError, match="12 lower-case hex"):
            FeatureStore(tmp_path, bad)


def _lock_variant(**over: Any) -> ServingLock:
    """The committed serving lock with some fields replaced, re-signed."""
    body = json.loads(LOCK.read_text(encoding="utf-8"))
    body.update(over)
    return parse_serving_lock(sign_lock_body(body))


def test_a_format_1_store_is_refused_not_converted(tmp_path: Path) -> None:
    """Format 1 kept float16 vectors only; X1 cannot score from them (module docstring), so
    the store is refused with the fix instead of being read or extended."""
    store = _store(tmp_path)
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    con = duckdb.connect(str(store.path))
    try:
        con.execute("UPDATE meta SET value = 'mesa-clm-features/1' WHERE key = 'format'")
    finally:
        con.close()
    for call in (store.ensure, lambda: store.get(TEXTS), store.stats):
        with pytest.raises(FeatureStoreError, match="format 'mesa-clm-features/1'") as err:
            call()
        assert "float16" in str(err.value) and "features build" in str(err.value)
    with pytest.raises(FeatureStoreError, match="format"):
        store.add(["new"], _vectors(["new"]), [1])


def test_the_store_records_and_enforces_the_vector_recipe(tmp_path: Path) -> None:
    """D5: a store built under one lock's vector recipe refuses reads and writes under a lock
    whose image, arguments, environment or cap differ, keeps the lock_sha on every vector row,
    and does not care about the bearer guard or the head (they never touch a vector)."""
    lock = load_serving_lock(LOCK)
    root = tmp_path / "features"
    store = FeatureStore.for_lock(root, lock)
    assert store.encoder_fp == lock.encoder_fp and store.lock_sha == lock.lock_sha
    assert (
        store.recipe_sha256
        == lock.vector_recipe_sha256()
        == feat.recipe_sha256(lock.vector_recipe())
    )
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    meta = store.meta()
    assert meta["vector_recipe_sha256"] == lock.vector_recipe_sha256()
    assert json.loads(meta["vector_recipe"]) == lock.vector_recipe()
    assert meta["serving_lock_sha"] == lock.lock_sha
    stats = store.stats()
    assert stats["vectors_by_lock_sha"] == {lock.lock_sha: 3}
    assert stats["vector_recipe_sha256"] == lock.vector_recipe_sha256()
    # Another image under the same encoder_fp: every operation is refused.
    image = {**lock.image.model_dump(), "digest": "sha256:" + "d" * 64}
    bumped = FeatureStore.for_lock(root, _lock_variant(image=image))
    assert bumped.encoder_fp == store.encoder_fp and bumped.path == store.path
    for call in (
        bumped.ensure,
        lambda: bumped.get(TEXTS),
        lambda: bumped.missing(TEXTS),
        lambda: bumped.add(["x"], _vectors(["x"]), [1]),
    ):
        with pytest.raises(FeatureStoreError, match="vector recipe") as err:
            call()
        assert "rebuild" in str(err.value)
    # A guard or head change rotates lock_sha only: the vectors stay usable, stamped per row.
    guard = lock.recipe.model_dump() if lock.recipe else {}
    guard["auth"] = {**guard["auth"], "sha256": "f" * 64}
    regarded = _lock_variant(recipe=guard)
    assert regarded.lock_sha != lock.lock_sha
    again = FeatureStore.for_lock(root, regarded)
    assert again.add(["one more"], _vectors(["one more"]), [2]) == 1
    assert again.stats()["vectors_by_lock_sha"] == {lock.lock_sha: 3, regarded.lock_sha: 1}
    # A store recorded without a recipe is refused under a lock, and a recipe-less object
    # (tests, tools without a lock) skips the check.
    plain = FeatureStore(tmp_path / "plain", lock.encoder_fp)
    plain.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    with pytest.raises(FeatureStoreError, match="vector recipe none"):
        FeatureStore.for_lock(tmp_path / "plain", lock).get(TEXTS)
    assert FeatureStore(root, lock.encoder_fp).get(TEXTS).shape == (3, 4096)
    cfg = Config.model_validate({"features": {"dir": str(root)}})
    assert FeatureStore.for_lock(cfg, lock).path == store.path


def test_writes_take_the_exclusive_lock_reads_the_shared_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    calls: list[int] = []
    real = fcntl.flock

    def spy(fd: int, op: int) -> None:
        calls.append(op)
        real(fd, op)

    monkeypatch.setattr(feat.fcntl, "flock", spy)
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    assert calls[0] == fcntl.LOCK_EX
    calls.clear()
    store.get(TEXTS)
    assert calls[0] == fcntl.LOCK_SH and fcntl.LOCK_EX not in calls


def test_a_held_lock_serialises_a_reader(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    finished = threading.Event()

    def read() -> None:
        store.get(TEXTS)
        finished.set()

    with store.lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        worker = threading.Thread(target=read)
        worker.start()
        time.sleep(0.3)
        assert not finished.is_set()  # waiting on the lock, not failing on DuckDB's
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    worker.join(timeout=10)
    assert finished.is_set()


# -- projections -------------------------------------------------------------------------------


def _head(seed: int = 1, source_sha256: str = "") -> HeadProjector:
    h = random_head(seed, hidden_size=4096)
    return HeadProjector(h.cfg, h.state, h.action, h.logit_scale, source_sha256)


def _model_fp(name: str = "clm-latest") -> str:
    return clm_model_fp(ClmModelSpec(head_name=name, head_sha256="", clm_commit=CLM_COMMIT))


def test_project_caches_the_head_projections(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    head, fp = _head(), _model_fp()
    zs = store.project(head, fp, "state", TEXTS[:1])
    np.testing.assert_array_equal(zs, head.project_states(store.get(TEXTS[:1])))
    za = store.project(head, fp, "action", [TEXTS[2], TEXTS[1], TEXTS[2]])
    np.testing.assert_array_equal(
        za, head.project_actions(store.get([TEXTS[2], TEXTS[1], TEXTS[2]]))
    )
    assert za.dtype == np.float32 and za.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(za, axis=1), 1.0, atol=1e-6)

    def boom(x: Any) -> Any:
        raise AssertionError("a cached projection was recomputed")

    cached = _head()
    cached.project_actions = boom  # type: ignore[method-assign]
    np.testing.assert_array_equal(store.project(cached, fp, "action", TEXTS[1:]), za[[1, 0]])
    assert store.stats()["projections"] == {fp: {"action": 2, "state": 1}}
    assert store.project(head, fp, "state", []).shape == (0, 512)


def test_project_refuses_another_head_and_bad_arguments(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    fp = _model_fp()
    store.project(_head(1), fp, "state", TEXTS)
    with pytest.raises(FeatureStoreError, match="another head"):
        store.project(_head(2), fp, "state", TEXTS)
    with pytest.raises(FeatureStoreError, match="another head"):
        store.project(_head(1, source_sha256="a" * 64), fp, "action", TEXTS)
    # The same head under its own fingerprint is fine.
    store.project(_head(2), _model_fp("clm-other"), "state", TEXTS)
    with pytest.raises(FeatureStoreError, match="side"):
        store.project(_head(1), fp, "both", TEXTS)  # type: ignore[arg-type]
    with pytest.raises(FeatureStoreError, match="12 lower-case hex"):
        store.project(_head(1), "nope", "state", TEXTS)
    with pytest.raises(FeatureStoreError, match="4096-d"):
        store.project(random_head(1, hidden_size=96), fp, "state", TEXTS)
    with pytest.raises(FeatureMissing):
        store.project(_head(1), fp, "action", ["not embedded"])


# -- the TextCache export ----------------------------------------------------------------------
#
# Copied from Contrastive-LM/CLM train/finetune.py at bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
# (lines 57-58 and 422-446; Apache-2.0, src/mesa_clm/_vendor/clm/LICENSE): the cache reader and
# writer finetune.py itself uses, so the export is checked against the real format, not ours.


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


class TextCache:
    """sha1(text) -> embedding, persisted as .npz."""

    def __init__(self, path: str):
        self.path, self.vecs = path, {}
        if os.path.exists(path):
            z = np.load(path)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))  # noqa: B905
            print(f"[choice] embedding cache: {len(self.vecs)} texts from {path}", flush=True)

    @staticmethod
    def key(t: str) -> str:
        return hashlib.sha1(t.encode()).hexdigest()  # noqa: S324

    def missing(self, texts):  # type: ignore[no-untyped-def]
        return [t for t in dict.fromkeys(texts) if self.key(t) not in self.vecs]

    def add(self, texts, vecs):  # type: ignore[no-untyped-def]
        self.vecs.update(
            {self.key(t): np.asarray(v, dtype=np.float16) for t, v in zip(texts, vecs)}  # noqa: B905
        )
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        ks = list(self.vecs)
        np.savez(
            self.path,
            keys=np.array(ks),
            vecs=np.stack([self.vecs[k] for k in ks]).astype(np.float16),
        )

    def __getitem__(self, t):  # type: ignore[no-untyped-def]
        return self.vecs[self.key(t)]


def _members(path: Path) -> list[tuple[str, int]]:
    with zipfile.ZipFile(path) as zf:
        return sorted((i.filename, i.compress_type) for i in zf.infolist())


def test_export_is_clm_textcache(tmp_path: Path) -> None:
    store = _store(tmp_path)
    texts = [*TEXTS, "ünïcode: Ångström — 1-D extent"]
    store.add(texts, _vectors(texts), _tokens(texts))
    res = export_npz(store, tmp_path / "npz")
    assert res.path == tmp_path / "npz" / "choice_Qwen_Qwen3-8B_4096.npz"
    assert res.n == 4 and res.fp16_min_cosine >= FP16_MIN_COSINE
    assert feat.textcache_filename("Qwen/Qwen3-8B", 4096) == res.path.name
    for name in ("Qwen/Qwen3-8B", "org/model v2:beta", "__x__", "a..b-c_d"):
        assert feat._slug(name) == _slug(name)
    cache = TextCache(str(res.path))
    assert cache.missing(texts) == []
    for text, want in zip(texts, store.get(texts), strict=True):
        got = cache[text]
        assert got.dtype == np.float16 and got.shape == (4096,)
        # The only float16 in the pipeline: the export casts the stored float32 vectors.
        np.testing.assert_array_equal(got, want.astype(np.float16))
    with np.load(res.path) as z:
        assert z["keys"].dtype == np.dtype("<U40") and z["vecs"].dtype == np.float16
        assert z["keys"].tolist() == sorted(feat.text_sha1(t) for t in texts)
    # Same members and compression as a file written by TextCache.add itself (np.savez).
    theirs = TextCache(str(tmp_path / "theirs" / res.path.name))
    theirs.add(texts, store.get(texts))
    assert _members(res.path) == _members(Path(theirs.path))
    assert {c for _, c in _members(res.path)} == {zipfile.ZIP_STORED}
    # A second export replaces the file in place and leaves no temporary behind.
    store.add(["one more text"], _vectors(["one more text"]), [3])
    assert export_npz(store, tmp_path / "npz").n == 5
    assert sorted(p.name for p in (tmp_path / "npz").iterdir()) == [res.path.name]


def test_export_refuses_an_empty_store_and_a_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    with pytest.raises(FeatureStoreError, match="no vectors"):
        export_npz(store, tmp_path / "npz")
    store.add(TEXTS, _vectors(TEXTS), _tokens(TEXTS))
    monkeypatch.setattr(feat, "fp16_roundtrip_min_cosine", lambda s: 0.99)
    with pytest.raises(FeatureStoreError, match=r"0\.9999 gate"):
        export_npz(store, tmp_path / "npz")
    assert not (tmp_path / "npz").exists()


# -- builder order -----------------------------------------------------------------------------


def _resorted(state: dict[str, Any]) -> dict[str, Any]:
    """What a labels snapshot stores: canonical JSON, sorted keys at every depth."""
    loaded: dict[str, Any] = json.loads(json.dumps(state, sort_keys=True))
    return loaded


def _ordered_json(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False)


def test_builder_order_is_restored_from_sorted_json(fixture_cards: list[Path]) -> None:
    for path in fixture_cards:
        card = load_card(path)
        col = card.columns[1]
        e = entry("envo")
        built = [
            target_state(card, "column", "measurement", column=col),
            target_state(card, "site", "environment", site=card.sites[0]),
            target_state(card, "dataset", "taxon"),
            candidate_state(card, "column", col, "method", {"label": "l", "curie": "X:1"}, 4),
            ontology_state(card, col, "measurement", e.id, e.option_text, sorted(e.aspects)),
        ]
        for state in built:
            restored = builder_ordered(_resorted(state))
            assert restored == state
            # Nested order is the builders'; the top level is the framing view's business.
            for key, value in state.items():
                assert _ordered_json(restored[key]) == _ordered_json(value)
    with pytest.raises(ManifestError, match=r"card\.extra"):
        builder_ordered({"card": {"dataset": "d", "extra": 1}})


# -- the manifest ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full() -> feat.Manifest:
    return manifest(SNAPSHOT)


def _snapshot_counts(task_id: str) -> tuple[int, int]:
    """(distinct targets, distinct (target, option) pairs) of a task in the snapshot."""
    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT count(DISTINCT target_sha256), count(DISTINCT (target_sha256, option_key)) "
            "FROM read_parquet(?) WHERE task_id = ?",
            [str(SNAPSHOT), task_id],
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    return int(row[0]), int(row[1])


def test_manifest_shape_counts_and_determinism(full: feat.Manifest) -> None:
    assert full.tasks == feat.X1_TASKS and full.framings == feat.DEFAULT_FRAMINGS
    assert full.labels_sha256 == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert full.context_conflicts == 0
    summary = full.summary()
    for task_id in feat.X1_TASKS:
        n_targets, n_pairs = _snapshot_counts(task_id)
        counts = summary["counts"][task_id]
        for fid in ("F4", "F7", "F9"):
            assert counts[fid] == {"context": n_targets, "anchor": n_targets, "candidate": n_pairs}
        assert counts["F1"] == {"context": n_pairs, "noul_true": n_pairs, "noul_false": n_pairs}
        for spec in feat.X2_SPECS:
            assert counts[spec] == {"context": n_pairs}
    assert summary["rows"] == len(full.rows)
    assert summary["texts"] == len(full.texts()) == len({r.text_sha256 for r in full.rows})
    assert manifest(SNAPSHOT).rows == full.rows  # deterministic
    for r in full.rows:
        assert r.side == ("state" if r.role == "context" else "action")
        assert r.text_sha256 == text_sha256(r.text)
    # Rows are ordered: per task and framing, targets then options sorted.
    keys = [(r.task_id, r.framing_id, r.target_sha256) for r in full.rows]
    for task_id in feat.X1_TASKS:
        for fid in feat.DEFAULT_FRAMINGS:
            targets = [k[2] for k in keys if k[:2] == (task_id, fid)]
            assert targets == sorted(targets)


def _built_target(task_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """The production context state of a snapshot row, rebuilt from the fixture card."""
    card = load_card(CARDS / f"{state['card']['dataset']}.md")
    scope = state.get("scope") or TASKS[task_id].scope
    if "column" in state:
        return target_state(
            card, scope, state["aspect"], column=card.column(state["column"]["name"])
        )
    if "site" in state:
        site = next(s for s in card.sites if s.code == state["site"]["code"])
        return target_state(card, scope, state["aspect"], site=site)
    return target_state(card, scope, state["aspect"])


def _built_anyjev(task_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """The stored mesa-anyjev per-candidate state, rebuilt from the fixture card."""
    card = load_card(CARDS / f"{state['card']['dataset']}.md")
    if task_id == "column.ontology_fits":
        e = entry(state["ontology"]["id"])
        col = card.column(state["column"]["name"])
        return ontology_state(card, col, state["aspect"], e.id, e.option_text, sorted(e.aspects))
    target = card.column(state["column"]["name"]) if "column" in state else None
    return candidate_state(
        card, state["scope"], target, state["aspect"], state["candidate"], state["n_candidates"]
    )


def _snapshot_states() -> dict[tuple[str, str, str], dict[str, Any]]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT task_id, target_sha256, option_key, state_json FROM read_parquet(?) "
            "WHERE task_id IN ('term.fits', 'column.ontology_fits')",
            [str(SNAPSHOT)],
        ).fetchall()
    finally:
        con.close()
    return {(str(t), str(ts), str(ok)): json.loads(sj) for t, ts, ok, sj in rows}


def test_every_context_is_the_one_the_builders_make(full: feat.Manifest) -> None:
    """The snapshot keeps sorted-key state JSON; the manifest renders builder order, so each
    context equals the text the pipeline would send for the same target (D23, plan §4.3)."""
    states = _snapshot_states()
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for (task_id, target, _), state in sorted(states.items()):
        first.setdefault((task_id, target), state)
    checked = 0
    for r in full.rows:
        if r.role != "context":
            continue
        if r.framing_id in ("F4", "F7", "F9"):
            built = _built_target(r.task_id, first[(r.task_id, r.target_sha256)])
            want = framings.context_text(framings.framing(r.task_id, r.framing_id), built)
        else:
            anyjev = _built_anyjev(r.task_id, states[(r.task_id, r.target_sha256, r.option_key)])
            f1 = framings.framing(r.task_id, "F1")
            suffix = None if r.framing_id == "joint4096@S1ns" else f1.instructions
            want = render.state_text(framings.build_context(f1, anyjev), suffix)
        assert r.text == want, (r.task_id, r.framing_id, r.target_sha256[:12], r.option_key)
        checked += 1
    assert checked == sum(1 for r in full.rows if r.role == "context")
    # The sorted-key rendering would not have matched (which is why builder order is restored).
    task_id, target, _ = next(iter(sorted(states)))
    raw = {**states[(task_id, target, _)], "scope": states[(task_id, target, _)].get("scope")}
    f7 = framings.framing(task_id, "F7")
    assert framings.context_text(f7, raw) != framings.context_text(f7, builder_ordered(raw))


def test_action_texts_are_the_framings(full: feat.Manifest) -> None:
    states = _snapshot_states()
    for r in full.rows:
        if r.role == "anchor":
            assert r.option_key == ANCHOR_KEY
            assert r.text == ANCHORS["term" if r.task_id == "term.fits" else "ontology"]
        elif r.role == "candidate":
            f = framings.framing(r.task_id, r.framing_id)
            if r.task_id == "term.fits":
                cand = framings.FramingCandidate.from_term(
                    states[(r.task_id, r.target_sha256, r.option_key)]["candidate"]
                )
            else:
                (cand,) = framings.ontology_candidates([r.option_key])
            assert r.text == framings.candidate_text(f, cand)
        elif r.role in ("noul_true", "noul_false"):
            f1 = framings.framing(r.task_id, "F1")
            keys, texts = render.candidates(framings.noul_question(f1))
            assert r.text == dict(zip(keys, texts, strict=True))[r.role.removeprefix("noul_")]


def test_x2_joint_specs_are_the_pr13_layout(full: feat.Manifest) -> None:
    """S1 = CLM's state text of the anyjev (target + candidate) state with the task question
    appended: the same text as F1's context. S1ns = the same state without the suffix."""
    by: dict[str, dict[tuple[str, str, str], str]] = {}
    for r in full.rows:
        if r.role == "context" and r.framing_id in ("F1", *feat.X2_SPECS):
            by.setdefault(r.framing_id, {})[(r.task_id, r.target_sha256, r.option_key)] = r.text
    assert by["joint4096@S1"] == by["F1"]
    for key, s1 in by["joint4096@S1"].items():
        s1ns = by["joint4096@S1ns"][key]
        assert s1 == f"{s1ns}\n\n{TASKS[key[0]].text}"
        assert key[2] in s1ns  # the candidate is inside the joint state


# Every column that carries a label or where it came from, overwritten without reading it: the
# label-bearing columns get values derived from the identity columns only, label_source ordered
# against state_sha256 so a manifest that sorted (or filtered) by it would change.
LABEL_FREE_REPLACE = (
    "SELECT * REPLACE ("
    "'synthetic-' || md5(state_sha256 || task_id || option_key) AS label_id, "
    "CASE WHEN substr(md5(target_sha256 || option_key), 1, 1) < '8' THEN 'consensus_all' "
    "ELSE 'consensus_negative' END AS label_source, "
    "'x' AS label, 0 AS label_index, 0.1 AS weight, false AS fold_eligible, true AS bench_card, "
    "'synthetic' AS origin, 'nobody' AS actor, "
    "TIMESTAMPTZ '2000-01-01 00:00:00+00' AS created_at) FROM read_parquet(?)"
)


def test_manifest_is_label_free(tmp_path: Path, full: feat.Manifest) -> None:
    flipped = tmp_path / "flipped.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY ({LABEL_FREE_REPLACE}) TO '{flipped}' (FORMAT PARQUET)",
            [str(SNAPSHOT)],
        )
    finally:
        con.close()
    other = manifest(flipped)
    assert other.rows == full.rows and other.labels_sha256 != full.labels_sha256


def test_a_duplicate_identity_takes_the_smallest_state_sha_whatever_its_source(
    tmp_path: Path,
) -> None:
    """Two label rows of one D1 identity on states that differ in ``n_candidates`` (as mesa-anyjev
    records them): the manifest renders the state with the smaller ``state_sha256`` (D1), never
    one chosen by ``label_source`` or any other label column."""
    con = duckdb.connect()
    try:
        # identity and state of one row; every label-bearing column replaced before it is read
        first = f"SELECT * FROM ({LABEL_FREE_REPLACE}) WHERE task_id = 'term.fits'"  # noqa: S608
        con.execute(
            f"CREATE TABLE t AS {first} ORDER BY target_sha256, option_key LIMIT 1",
            [str(SNAPSHOT)],
        )
        (state_json,) = con.execute("SELECT state_json FROM t").fetchone() or ("{}",)
        state = json.loads(state_json)
        other = {**state, "n_candidates": int(state.get("n_candidates", 0)) + 5}
        con.execute(
            "INSERT INTO t SELECT * REPLACE (? AS label_id, ? AS state_sha256, ? AS state_json) "
            "FROM t",
            ["synthetic-dup", state_sha256(other), json.dumps(other, sort_keys=True)],
        )
        small = min(state_sha256(state), state_sha256(other))
        picks = []
        for first in ("consensus_all", "consensus_negative"):
            last = "consensus_negative" if first == "consensus_all" else "consensus_all"
            out = tmp_path / f"{first}.parquet"
            con.execute(
                "UPDATE t SET label_source = CASE WHEN state_sha256 = ? THEN ? ELSE ? END, "
                "label = 'x', label_index = 0, weight = 0.1",
                [small, last, first],
            )
            con.execute(f"COPY t TO '{out}' (FORMAT PARQUET)")
            picks.append(manifest(out, "term.fits", "F1"))
    finally:
        con.close()
    assert picks[0].rows == picks[1].rows
    contexts = [r.text for r in picks[0].rows if r.role == "context"]
    chosen = state if state_sha256(state) == small else other
    assert len(contexts) == 1 and f"n_candidates: {chosen['n_candidates']}" in contexts[0]


def test_manifest_subsets_aliases_and_refusals(tmp_path: Path) -> None:
    sub = manifest(SNAPSHOT, "column.ontology_fits", "X2,F7,joint4096@S1")
    assert sub.tasks == ("column.ontology_fits",)
    assert sub.framings == ("joint4096@S1", "joint4096@S1ns", "F7")
    assert next(r.framing_id for r in sub.rows) == "joint4096@S1"
    assert {r.task_id for r in sub.rows} == {"column.ontology_fits"}
    assert feat.expand_framings(["F7", " F1 ", "F7"]) == ("F7", "F1")
    with pytest.raises(ManifestError, match="no such labels snapshot"):
        manifest(tmp_path / "missing.parquet")
    with pytest.raises(ManifestError, match="unknown task"):
        manifest(SNAPSHOT, "column.aspect")
    with pytest.raises(ManifestError, match="unknown framing"):
        manifest(SNAPSHOT, framing_ids="F2")
    with pytest.raises(ManifestError, match="no framings"):
        manifest(SNAPSHOT, framing_ids=" , ")
    with pytest.raises(ManifestError, match="no tasks"):
        manifest(SNAPSHOT, tasks="")
    bad = tmp_path / "not-parquet.parquet"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ManifestError, match="not a readable labels snapshot"):
        manifest(bad)


def test_manifest_refuses_a_row_whose_identity_does_not_match(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * REPLACE ('{'0' * 64}' AS target_sha256) FROM read_parquet(?) "  # noqa: S608
            f"WHERE task_id = 'term.fits' LIMIT 1) TO '{tampered}' (FORMAT PARQUET)",
            [str(SNAPSHOT)],
        )
    finally:
        con.close()
    with pytest.raises(ManifestError, match="identity does not match"):
        manifest(tampered, "term.fits", "F7")
