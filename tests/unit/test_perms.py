"""Owner-only state whatever the umask (``mesa_clm.perms``; DESIGN implementation notes M1,
"Owner-only state"): the sidecar (DuckDB file, ``.wal``, ``locks/`` and its lock file), the label
store on the same file, a run export, the feature store and its npz export, the serving keys'
missing parents and ``annotate --out``. Each test runs under a permissive umask (``002``, the
serving host's) and checks modes ``0700`` / ``0600``; existing parents are never touched."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest

from mesa_clm import perms
from mesa_clm.provenance.labels import LabelStore, lock_path_for
from mesa_clm.provenance.store import DuckDBStore

ROOT = Path(__file__).resolve().parents[2]


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture(autouse=True)
def _loose_umask() -> Iterator[None]:
    old = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(old)


def test_private_dir_creates_every_missing_component_0700(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o775)
    os.chmod(shared, 0o775)  # noqa: S103 - a deliberately loose mode
    target = perms.private_dir(shared / "a" / "b" / "c")
    for d in (shared / "a", shared / "a" / "b", target):
        assert _mode(d) == 0o700
    assert _mode(shared) == 0o775  # an existing parent is never touched
    loose = shared / "loose"
    loose.mkdir()
    os.chmod(loose, 0o775)  # noqa: S103 - a deliberately loose mode
    perms.private_dir(loose, tighten=False)
    assert _mode(loose) == 0o775
    perms.private_dir(loose)
    assert _mode(loose) == 0o700
    (tmp_path / "file").write_text("x")
    with pytest.raises(NotADirectoryError):
        perms.private_dir(tmp_path / "file")


def test_open_private_and_tighten_file(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    os.close(perms.open_private(path))
    assert _mode(path) == 0o600
    os.chmod(path, 0o664)
    os.close(perms.open_private(path))  # an existing loose file is repaired
    assert _mode(path) == 0o600
    other = tmp_path / "data"
    other.write_text("x")
    os.chmod(other, 0o644)
    assert perms.tighten_file(other) and _mode(other) == 0o600
    assert not perms.tighten_file(other) and not perms.tighten_file(tmp_path / "missing")
    link = tmp_path / "link"
    link.symlink_to(other)
    os.chmod(other, 0o644)
    assert not perms.tighten_file(link) and _mode(other) == 0o644  # symlinks are left alone
    assert perms.loose([other, link, tmp_path / "missing"]) == [(other, 0o644)]


def test_write_private_text_is_atomic_and_0600(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    perms.write_private_text(path, "one\n")
    perms.write_private_text(path, "two\n")
    assert path.read_text() == "two\n" and _mode(path) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_the_sidecar_is_owner_only(tmp_path: Path) -> None:
    db = tmp_path / "home" / ".mesa" / "clm" / "provenance.duckdb"
    store = DuckDBStore(db)
    store.ensure_schema()
    lock = lock_path_for(db)
    assert _mode(db.parent) == _mode(db.parent.parent) == 0o700
    assert _mode(lock.parent) == 0o700 and _mode(lock) == 0o600 and _mode(db) == 0o600
    wal = db.with_name(db.name + ".wal")
    assert not wal.exists() or _mode(wal) == 0o600
    # A read changes no existing mode (the doctor reports, never repairs); the next write
    # repairs a lock directory, lock file and database a looser umask left behind.
    os.chmod(lock.parent, 0o775)  # noqa: S103 - a deliberately loose mode
    os.chmod(lock, 0o664)
    os.chmod(db, 0o664)
    assert store.runs() == []
    assert (_mode(lock.parent), _mode(lock), _mode(db)) == (0o775, 0o664, 0o664)
    store.ensure_schema()
    assert (_mode(lock.parent), _mode(lock), _mode(db)) == (0o700, 0o600, 0o600)


def test_a_read_creates_a_missing_lock_privately_and_leaves_the_rest(tmp_path: Path) -> None:
    db = tmp_path / "side" / "p.duckdb"
    DuckDBStore(db).ensure_schema()
    lock = lock_path_for(db)
    lock.unlink()
    os.chmod(db, 0o644)
    assert DuckDBStore(db).runs() == []
    assert _mode(lock) == 0o600 and _mode(db) == 0o644


def test_the_label_store_is_owner_only(tmp_path: Path) -> None:
    db = tmp_path / "labels" / "prov.duckdb"
    LabelStore(db).ensure_schema()
    assert _mode(db.parent) == 0o700 and _mode(db) == 0o600
    assert _mode(lock_path_for(db)) == 0o600 and _mode(lock_path_for(db).parent) == 0o700


def test_a_sidecar_in_a_shared_directory_leaves_the_directory_alone(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o1777)  # noqa: S103 - a deliberately loose mode
    DuckDBStore(shared / "p.duckdb").ensure_schema()
    assert _mode(shared) == 0o1777 and _mode(shared / "p.duckdb") == 0o600
    assert _mode(shared / "locks") == 0o700


def test_a_run_export_is_owner_only(tmp_path: Path) -> None:
    from mesa_clm.provenance import export
    from tests.unit.test_provenance_export import _populate

    store = DuckDBStore(tmp_path / "prov.duckdb")
    run_id = _populate(store)
    out = tmp_path / "exports" / "deep"
    written = export.export_run(store, run_id, out)
    target = out / str(run_id)
    assert _mode(tmp_path / "exports") == _mode(out) == _mode(target) == 0o700
    for path in map(Path, written.values()):
        assert _mode(path) == 0o600, path
    assert sorted(p.name for p in target.iterdir() if p.name.startswith(".")) == []


def test_the_feature_store_and_its_export_are_owner_only(tmp_path: Path) -> None:
    import numpy as np

    from mesa_clm.learn.features import FeatureStore, export_npz

    root = tmp_path / "home" / "features"
    store = FeatureStore(root, "0123456789ab", dim=8)
    store.ensure(encoder_spec={"model": "x"})
    rng = np.random.default_rng(0)
    store.add(["a", "b"], rng.normal(size=(2, 8)), [1, 1])
    assert _mode(tmp_path / "home") == _mode(root) == _mode(store.dir) == 0o700
    assert _mode(store.path) == 0o600 and _mode(store.lock_path) == 0o600
    res = export_npz(store, tmp_path / "npz" / "dir", embed_model="m", max_len=16)
    assert _mode(tmp_path / "npz") == _mode(res.path.parent) == 0o700
    assert _mode(res.path) == 0o600


def test_serving_keys_create_private_parents(tmp_path: Path) -> None:
    from mesa_clm import serving

    secrets = tmp_path / "home" / ".mesa" / "clm" / "secrets"
    res = serving.init_keys(secrets)
    for d in (tmp_path / "home" / ".mesa", tmp_path / "home" / ".mesa" / "clm", secrets):
        assert _mode(d) == 0o700, d
    for path in (res.clm_key, res.encoder_key, res.clm_env, res.encoder_env):
        assert _mode(path) == 0o600


def test_annotate_out_is_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mesa_clm.cli import EXIT_OK, main

    for key in list(os.environ):
        if key.startswith(("MESA_CLM_", "CLM_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES", "replay")
    monkeypatch.setenv("MESA_CLM_OLS__FIXTURES_DIR", str(ROOT / "tests" / "fixtures" / "ols"))
    monkeypatch.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    db = tmp_path / "side" / "prov.duckdb"
    out = tmp_path / "runs" / "run.json"
    card = ROOT / "tests" / "fixtures" / "cards" / "DP1.10003.001.brd_countdata.md"
    argv = ["--provenance", f"duckdb:///{db}", "annotate", "--card", str(card)]
    assert main([*argv, "--provider", "fake", "--out", str(out)]) == EXIT_OK
    capsys.readouterr()
    assert _mode(out.parent) == 0o700 and _mode(out) == 0o600
    assert _mode(db) == 0o600 and _mode(db.parent) == 0o700
    assert UUID(str(DuckDBStore(db).runs()[0]["run_id"]))
