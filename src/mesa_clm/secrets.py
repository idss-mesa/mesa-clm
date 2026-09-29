"""Secret resolution: an inline value, a 0600 key file, or the system keyring (DESIGN D29).

``MESA_CLM_SECRETS`` picks the source for every API key the config carries:

* ``env`` - only the configured value (``MESA_CLM_<SECTION>__API_KEY``, a flag or the YAML file);
* ``file`` - only the key file (``MESA_CLM_<SECTION>__API_KEY_FILE``);
* ``keyring`` - only the system keyring (service ``mesa-clm``); the ``keyring`` package is
  optional and imported lazily, it is not a dependency of mesa-clm;
* ``auto`` (default) - the first *configured* of value, file, keyring.

A key file must be a regular file owned by the current user with no group or other permission
bits (0600 or stricter, the ssh rule); anything else is refused with an error that names the file
and the fix, never its content. Values are never logged, hashed or put in an exception message.
"""

from __future__ import annotations

import importlib
import logging
import os
import stat
from pathlib import Path
from typing import Literal, get_args

logger = logging.getLogger(__name__)

SecretsMode = Literal["auto", "env", "file", "keyring"]
SECRETS_MODES: tuple[str, ...] = get_args(SecretsMode)

# A bearer key is a few dozen bytes; anything bigger is not a key file (a Parquet file, a log).
MAX_KEY_FILE_BYTES = 64 * 1024
# Permission bits a key file must not have: any group or other access.
_FORBIDDEN_MODE_BITS = 0o077


class SecretError(RuntimeError):
    """A configured secret source could not be used. The message never carries the value."""


def resolve_secret(
    value: str | None,
    file: str | Path | None,
    *,
    keyring_service: str | None = None,
    keyring_user: str | None = None,
    mode: SecretsMode = "auto",
) -> str | None:
    """The secret from the source ``mode`` selects, or ``None`` when that source is not configured.

    ``value`` is the inline value (flag, environment or YAML), ``file`` the path named by the
    matching ``*_API_KEY_FILE`` setting, and the keyring entry is ``(keyring_service,
    keyring_user)``. In ``auto`` the first configured source wins, in that order; a source that
    is configured but unusable (a key file with loose permissions, a locked keyring in
    ``keyring`` mode) raises :class:`SecretError` rather than falling through, so a
    misconfiguration never silently becomes an anonymous request.
    """
    if mode not in SECRETS_MODES:
        raise ValueError(
            f"unknown secrets mode {mode!r}; expected one of {', '.join(SECRETS_MODES)}"
        )
    if mode in ("env", "auto") and value:
        logger.debug("secret resolved from the configured value")
        return value
    if mode == "env":
        return None
    if mode in ("file", "auto") and file is not None and str(file):
        secret = read_secret_file(file)
        logger.debug("secret resolved from key file %s", Path(file).expanduser())
        return secret
    if mode == "file":
        return None
    return _from_keyring(keyring_service, keyring_user, required=mode == "keyring")


def read_secret_file(path: str | Path) -> str:
    """The stripped content of a key file that passes :func:`check_secret_file_stat`.

    The file is opened first and checked through ``fstat`` on the open descriptor, so the file
    that is read is the file that was checked. ``O_NONBLOCK`` keeps a FIFO from blocking the
    open; it is then refused as not a regular file.
    """
    p = Path(path).expanduser()
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(p, flags)
    except FileNotFoundError:
        raise SecretError(f"key file {p} does not exist") from None
    except OSError as exc:
        raise SecretError(f"key file {p} cannot be opened: {exc.strerror}") from None
    try:
        check_secret_file_stat(p, os.fstat(fd))
        chunks: list[bytes] = []
        size = 0
        while size <= MAX_KEY_FILE_BYTES:
            chunk = os.read(fd, MAX_KEY_FILE_BYTES + 1)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    finally:
        os.close(fd)
    if size > MAX_KEY_FILE_BYTES:
        raise SecretError(f"key file {p} is larger than {MAX_KEY_FILE_BYTES} bytes; not a key file")
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        raise SecretError(f"key file {p} is not UTF-8 text") from None
    secret = text.strip()
    if not secret:
        raise SecretError(f"key file {p} is empty")
    return secret


def check_secret_file_stat(path: Path, st: os.stat_result) -> None:
    """Refuse anything but a regular file owned by the current user with mode 0600 or stricter."""
    if not stat.S_ISREG(st.st_mode):
        raise SecretError(f"key file {path} is not a regular file")
    getuid = getattr(os, "getuid", None)  # absent on Windows; the stack runs on Linux
    if getuid is not None and st.st_uid != getuid():
        raise SecretError(
            f"key file {path} is owned by uid {st.st_uid}, not the current user (uid {getuid()})"
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & _FORBIDDEN_MODE_BITS:
        raise SecretError(
            f"key file {path} has mode {mode:04o}; it must be 0600 or stricter (run: chmod 600 {path})"
        )


def _from_keyring(service: str | None, user: str | None, *, required: bool) -> str | None:
    """The keyring entry, or ``None``. ``required`` (mode ``keyring``) turns every obstacle into
    a :class:`SecretError`; in ``auto`` a missing package or backend is merely logged (by type,
    never by value)."""
    if not service or not user:
        if required:
            raise SecretError("secrets mode 'keyring' needs a keyring service and user name")
        return None
    try:
        keyring = importlib.import_module("keyring")
    except ImportError:
        if required:
            raise SecretError(
                "MESA_CLM_SECRETS=keyring but the optional 'keyring' package is not installed "
                "(install it into the same environment, e.g. `uv pip install keyring`)"
            ) from None
        logger.debug(
            "keyring is not installed; skipping the keyring lookup for %s/%s", service, user
        )
        return None
    try:
        value = keyring.get_password(service, user)
    except Exception as exc:  # backends raise their own types: no backend, locked, D-Bus down
        if required:
            raise SecretError(
                f"keyring lookup for {service}/{user} failed: {type(exc).__name__}"
            ) from None
        logger.warning(
            "keyring lookup for %s/%s failed (%s); continuing without it",
            service,
            user,
            type(exc).__name__,
        )
        return None
    if value is None or not str(value):
        return None
    logger.debug("secret resolved from the keyring (%s/%s)", service, user)
    return str(value)
