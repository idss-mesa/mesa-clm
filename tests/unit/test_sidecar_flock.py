"""D11: the sidecar store opens the DuckDB file per operation under ``flock`` of the lock file the
M0 ``LabelStore`` uses, never holds it between calls, takes a shared lock for read-only opens and
the exclusive lock for writes, and serialises with other threads and other processes."""

from __future__ import annotations

import fcntl
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pytest

from mesa_clm.provenance import store as store_module
from mesa_clm.provenance.labels import LabelRow, LabelStore, lock_path_for
from mesa_clm.provenance.models import RunRow
from mesa_clm.provenance.store import SCHEMA, DuckDBStore
from mesa_clm.tasks import TASKS


def _run(**over: Any) -> RunRow:
    base: dict[str, Any] = {
        "owner": "alice",
        "card_name": "DP1.x",
        "card_sha256": "s" * 64,
        "planner": "static",
        "provider": "fake",
        "clm_model": "clm-latest",
        "encoder_model": "qwen3-8b",
        "schema_sha256": "h" * 64,
        "encoder_fp": "e" * 12,
        "clm_model_fp": "m" * 12,
        "framings_lock_sha": "l" * 64,
        "policy_profile": "dev",
        "config_sha256": "c" * 64,
        "vm_id": "testhost",
    }
    base.update(over)
    return RunRow(**base)


def _try_lock(lock_path: Path) -> bool:
    """Whether the exclusive flock can be taken right now (nobody holds the file)."""
    with lock_path.open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return True


def test_lock_path_is_the_label_store_lock(tmp_path: Path) -> None:
    path = tmp_path / "prov.duckdb"
    store = DuckDBStore(path)
    assert store.lock_path == lock_path_for(path) == LabelStore(path).lock_path
    assert store.lock_path == tmp_path / "locks" / "provenance.lock"
    assert not store.lock_path.exists() and not path.exists()  # the constructor touches nothing
    custom = DuckDBStore(path, lock_path=tmp_path / "other.lock")
    assert custom.lock_path == tmp_path / "other.lock"
    custom.ensure_schema()
    assert (tmp_path / "other.lock").exists()


def test_lock_and_connection_are_released_after_each_operation(tmp_path: Path) -> None:
    path = tmp_path / "prov.duckdb"
    store = DuckDBStore(path)
    store.begin_run(_run())
    assert store.lock_path.exists() and _try_lock(store.lock_path)
    # another process-like opener gets the file: no lingering DuckDB handle ("Could not set lock")
    con = duckdb.connect(str(path))
    assert con.execute(f"SELECT count(*) FROM {SCHEMA}.runs").fetchone() == (1,)  # noqa: S608
    con.close()
    assert store.runs() and _try_lock(store.lock_path)
    # the M0 label store shares the file and the lock without stepping on the sidecar
    LabelStore(path).insert_labels(
        [
            LabelRow(
                task_id="term.fits",
                task_key=TASKS["term.fits"].key,
                target_sha256="d" * 64,
                option_key="PATO:1",
                label_source="curator",
                label="Yes",
                label_index=0,
                weight=1.0,
                state_sha256="a" * 64,
                state_json={},
            )
        ]
    )
    assert len(store.labels_for("term.fits")) == 1 and len(store.runs()) == 1


def test_reads_are_read_only_shared_locked_and_writes_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    run = _run()
    store.begin_run(run)
    opens: list[bool] = []
    locks: list[int] = []
    real_connect, real_flock = duckdb.connect, fcntl.flock

    def spy_connect(*args: Any, **kwargs: Any) -> Any:
        opens.append(bool(kwargs.get("read_only", False)))
        return real_connect(*args, **kwargs)

    def spy_flock(fd: int, op: int) -> None:
        locks.append(op)
        real_flock(fd, op)

    monkeypatch.setattr(store_module.duckdb, "connect", spy_connect)
    monkeypatch.setattr(store_module.fcntl, "flock", spy_flock)
    assert store.run(run.run_id) is not None
    assert store.decisions(run.run_id) == [] and store.runs() and store.columns("runs")
    assert opens == [True, True, True, True]
    assert locks == [fcntl.LOCK_SH, fcntl.LOCK_UN] * 4
    opens.clear()
    locks.clear()
    store.finish_run(run.run_id, "decided")
    store.insert_links([])
    assert opens == [False, False]
    assert locks == [fcntl.LOCK_EX, fcntl.LOCK_UN] * 2


def test_write_waits_for_a_held_lock_and_reads_share_it(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    store.ensure_schema()
    first = _run()
    store.begin_run(first)
    done = threading.Event()
    result: list[int] = []

    def writer() -> None:
        store.begin_run(_run())
        result.append(len(store.runs()))
        done.set()

    with store.lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_SH)  # a reader elsewhere
        assert store.run(first.run_id) is not None  # our read shares the lock
        t = threading.Thread(target=writer)
        t.start()
        time.sleep(0.3)
        assert not done.is_set() and result == []  # the write waits behind the reader
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    assert done.wait(10) and result == [2]
    t.join()


def test_threads_contending_on_one_store_serialise(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "prov.duckdb")
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for _ in range(n):
                store.begin_run(_run())
                store.runs(limit=1)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(4,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert errors == []
    assert len(store.runs(limit=None)) == 16 and _try_lock(store.lock_path)


_CHILD = r"""
import sys
from uuid import uuid4
from mesa_clm.provenance.models import RunRow
from mesa_clm.provenance.store import DuckDBStore
path, n = sys.argv[1], int(sys.argv[2])
store = DuckDBStore(path)
for i in range(n):
    store.begin_run(RunRow(owner="child", card_name="DP1.x", card_sha256="s"*64, planner="static",
        provider="fake", clm_model="clm-latest", encoder_model="qwen3-8b", schema_sha256="h"*64,
        encoder_fp="e"*12, clm_model_fp="m"*12, framings_lock_sha="l"*64, policy_profile="dev",
        config_sha256="c"*64, vm_id="child"))
    store.runs(limit=1)
print(len(store.runs(limit=None)))
"""


def test_processes_contending_on_one_file_serialise(tmp_path: Path) -> None:
    """Real cross-process contention: several interpreters write the same file at once; the flock
    (not DuckDB's own file lock, which would raise "Could not set lock") serialises them."""
    path = tmp_path / "prov.duckdb"
    per_process, n_processes = 5, 3
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(path), str(per_process)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(n_processes)
    ]
    outputs = [p.communicate(timeout=120) for p in procs]
    for p, (out, err) in zip(procs, outputs, strict=True):
        assert p.returncode == 0, err
        assert int(out.strip()) >= per_process
    assert len(DuckDBStore(path).runs(limit=None)) == per_process * n_processes
    assert _try_lock(lock_path_for(path))


def test_missing_file_read_takes_no_lock_and_makes_no_directory(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "nested" / "prov.duckdb")
    assert store.run(uuid4()) is None
    assert not (tmp_path / "nested").exists()
