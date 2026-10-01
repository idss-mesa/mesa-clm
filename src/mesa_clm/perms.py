"""Owner-only files and directories for what mesa-clm keeps on disk (DESIGN implementation
notes (M1), "Owner-only state"; D11, D29).

The sidecar holds runs, owners, curator labels and card-derived states, the feature store holds
the texts it embedded, an export holds a whole run, and ``secrets/`` holds the bearer keys. On a
multi-user host (sparky-1, the DE VMs of D29) home directories are often traversable and the
process umask is often ``002``, which would give every new file group and world read bits. So
mesa-clm never leaves a mode to the umask:

* :func:`private_dir` creates every *missing* component of a directory with mode ``0700``
  (``chmod`` right after ``mkdir``, so no umask can loosen or tighten it) and, with
  ``tighten=True``, brings the directory itself down to ``0700`` when this user owns it and it
  has group or other bits. Existing parents are never touched: a sidecar under ``/tmp`` must not
  ``chmod /tmp``.
* :func:`open_private` opens a file, creating it with mode ``0600``, and (with ``tighten``, the
  default) sets an existing file this user owns to exactly ``0600``, which repairs a lock file a
  looser umask created earlier; a read-only operation passes ``tighten=False`` and changes no
  existing mode.
* :func:`tighten_file` sets an existing regular file this user owns to ``0600``. DuckDB creates
  its database and ``.wal`` with the umask's mode, so the stores call it after every write; until
  then only the ``0700`` directory protects them (a ``.wal`` is re-created on every write).
* :func:`write_private_text` replaces a file atomically at ``0600`` (temp file, fsync, rename).
* :func:`loose` lists the paths among a set whose mode carries group or other bits, for the
  doctor's ``permissions`` check.

Nothing here reads a file's content or follows a symbolic link into a ``chmod`` of something
else: symlinks are left alone.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Final

__all__ = [
    "PRIVATE_DIR",
    "PRIVATE_FILE",
    "loose",
    "open_private",
    "private_dir",
    "tighten_dir",
    "tighten_file",
    "write_private_bytes",
    "write_private_text",
]

PRIVATE_DIR: Final[int] = 0o700
PRIVATE_FILE: Final[int] = 0o600
# Group and other permission bits: none may be set on mesa-clm's state.
LOOSE_BITS: Final[int] = 0o077


def _owned(st: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)  # absent on Windows; the stack runs on Linux
    return getuid is None or st.st_uid == getuid()


def tighten_dir(path: str | Path) -> bool:
    """Set ``path`` to ``0700`` when it is a real directory this user owns with group or other
    bits; ``True`` when it changed anything. Symlinks and other owners' directories are left."""
    p = Path(path)
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(st.st_mode) or not _owned(st):
        return False
    if stat.S_IMODE(st.st_mode) == PRIVATE_DIR:
        return False
    os.chmod(p, PRIVATE_DIR)
    return True


def private_dir(path: str | Path, *, tighten: bool = True) -> Path:
    """``path`` (home expanded) as a directory: every missing component created ``0700``, the
    directory itself tightened to ``0700`` when ``tighten`` (module docstring). Returns it."""
    target = Path(path).expanduser()
    missing: list[Path] = []
    probe = target
    while not os.path.lexists(probe):
        missing.append(probe)
        if probe.parent == probe:  # pragma: no cover - the root always exists
            break
        probe = probe.parent
    for d in reversed(missing):
        try:
            os.mkdir(d, PRIVATE_DIR)
        except FileExistsError:
            # Created concurrently (another mesa-clm process): it must still be a directory.
            if not d.is_dir():
                raise
            continue
        os.chmod(d, PRIVATE_DIR)
    if not target.is_dir():
        raise NotADirectoryError(f"{target}: not a directory")
    if tighten:
        tighten_dir(target)
    return target


def open_private(path: str | Path, flags: int = os.O_RDWR, *, tighten: bool = True) -> int:
    """A file descriptor for ``path`` opened with ``flags``, the file created ``0600`` when
    missing and, with ``tighten``, set to exactly ``0600`` when this user owns it with another
    mode. The caller closes the descriptor. A newly created file is always ``0600``."""
    target = Path(path)
    existed = os.path.lexists(target)
    fd = os.open(target, flags | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), PRIVATE_FILE)
    try:
        st = os.fstat(fd)
        wrong = stat.S_ISREG(st.st_mode) and _owned(st) and stat.S_IMODE(st.st_mode) != PRIVATE_FILE
        if wrong and (tighten or not existed):
            os.fchmod(fd, PRIVATE_FILE)
    except BaseException:
        os.close(fd)
        raise
    return fd


def tighten_file(path: str | Path) -> bool:
    """Set an existing regular file this user owns to ``0600`` when it has group or other bits;
    ``True`` when it changed anything (missing files, symlinks and others' files are left)."""
    p = Path(path)
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(st.st_mode) or not _owned(st):
        return False
    if not stat.S_IMODE(st.st_mode) & LOOSE_BITS:
        return False
    with contextlib.suppress(FileNotFoundError):  # a .wal can vanish at a checkpoint
        os.chmod(p, PRIVATE_FILE)
        return True
    return False


def write_private_bytes(path: str | Path, data: bytes) -> Path:
    """Atomically replace ``path`` with ``data`` at mode ``0600`` (a temp file in the same
    directory, fsync, rename, fsync of the directory). The directory must exist."""
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), PRIVATE_FILE)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    dir_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return target


def write_private_text(path: str | Path, text: str) -> Path:
    """:func:`write_private_bytes` for UTF-8 text."""
    return write_private_bytes(path, text.encode("utf-8"))


def loose(paths: Iterable[str | Path]) -> list[tuple[Path, int]]:
    """``(path, mode)`` for every existing path among ``paths`` (symlinks skipped) whose mode has
    group or other bits, in the order given."""
    out: list[tuple[Path, int]] = []
    for path in paths:
        p = Path(path)
        try:
            st = os.lstat(p)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if stat.S_ISLNK(st.st_mode):
            continue
        mode = stat.S_IMODE(st.st_mode)
        if mode & LOOSE_BITS:
            out.append((p, mode))
    return out
