"""resolve_secret: the 0600 file check, source precedence per mode, and the optional keyring."""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from mesa_clm.secrets import (
    MAX_KEY_FILE_BYTES,
    SECRETS_MODES,
    SecretError,
    read_secret_file,
    resolve_secret,
)

SECRET = "sk-clm-not-for-logs-7f3a"


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    p = tmp_path / "clm.key"
    p.write_text(f"{SECRET}\n", encoding="utf-8")
    p.chmod(0o600)
    return p


@pytest.fixture
def no_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import keyring`` fail whether or not the package is installed here."""
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A stand-in ``keyring`` module: ``store[(service, user)]`` answers ``get_password``;
    ``store["raise"]`` makes every lookup raise."""
    store: dict[Any, Any] = {}
    module = types.ModuleType("keyring")

    def get_password(service: str, user: str) -> str | None:
        if "raise" in store:
            raise RuntimeError("no backend, and the value would be: nothing")
        value = store.get((service, user))
        return None if value is None else str(value)

    module.get_password = get_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", module)
    return store


# -- key files ---------------------------------------------------------------------------------


def test_file_is_read_and_stripped(key_file: Path) -> None:
    assert read_secret_file(key_file) == SECRET
    assert read_secret_file(str(key_file)) == SECRET
    key_file.chmod(0o400)
    assert read_secret_file(key_file) == SECRET  # stricter is fine


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o606, 0o666, 0o610])
def test_group_or_other_bits_are_refused(key_file: Path, mode: int) -> None:
    key_file.chmod(mode)
    with pytest.raises(SecretError, match="0600") as info:
        read_secret_file(key_file)
    assert f"{mode:04o}" in str(info.value) and "chmod 600" in str(info.value)
    assert SECRET not in str(info.value)


def test_other_owner_is_refused(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    me = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: me + 1)
    with pytest.raises(SecretError, match="owned by uid"):
        read_secret_file(key_file)


def test_missing_directory_empty_binary_and_huge_files(tmp_path: Path) -> None:
    with pytest.raises(SecretError, match="does not exist"):
        read_secret_file(tmp_path / "absent.key")
    with pytest.raises(SecretError, match="not a regular file"):
        read_secret_file(tmp_path)
    empty = tmp_path / "empty.key"
    empty.write_text("\n  \n", encoding="utf-8")
    empty.chmod(0o600)
    with pytest.raises(SecretError, match="empty"):
        read_secret_file(empty)
    binary = tmp_path / "bin.key"
    binary.write_bytes(b"\xff\xfe\x00")
    binary.chmod(0o600)
    with pytest.raises(SecretError, match="UTF-8"):
        read_secret_file(binary)
    huge = tmp_path / "huge.key"
    huge.write_bytes(b"k" * (MAX_KEY_FILE_BYTES + 1))
    huge.chmod(0o600)
    with pytest.raises(SecretError, match="larger than"):
        read_secret_file(huge)


def test_tilde_is_expanded(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(key_file.parent))
    assert read_secret_file("~/clm.key") == SECRET
    assert resolve_secret(None, "~/clm.key", mode="file") == SECRET


# -- modes and precedence ----------------------------------------------------------------------


def test_modes_are_the_documented_four() -> None:
    assert SECRETS_MODES == ("auto", "env", "file", "keyring")
    with pytest.raises(ValueError, match="unknown secrets mode"):
        resolve_secret("v", None, mode="vault")  # type: ignore[arg-type]


def test_auto_prefers_value_then_file_then_keyring(
    key_file: Path, fake_keyring: dict[str, Any]
) -> None:
    fake_keyring[("mesa-clm", "clm")] = "from-keyring"
    kr = {"keyring_service": "mesa-clm", "keyring_user": "clm"}
    assert resolve_secret("inline", key_file, mode="auto", **kr) == "inline"
    assert resolve_secret(None, key_file, mode="auto", **kr) == SECRET
    assert resolve_secret("", key_file, mode="auto", **kr) == SECRET  # empty value is unset
    assert resolve_secret(None, None, mode="auto", **kr) == "from-keyring"
    assert resolve_secret(None, "", mode="auto", **kr) == "from-keyring"  # empty path is unset
    del fake_keyring[("mesa-clm", "clm")]
    assert resolve_secret(None, None, mode="auto", **kr) is None
    assert resolve_secret(None, None, mode="auto") is None  # no keyring entry configured


def test_auto_surfaces_a_bad_file_instead_of_falling_through(key_file: Path) -> None:
    key_file.chmod(0o644)
    with pytest.raises(SecretError):
        resolve_secret(None, key_file, mode="auto")


def test_env_mode_uses_only_the_value(key_file: Path, fake_keyring: dict[str, Any]) -> None:
    fake_keyring[("s", "u")] = "kr"
    assert resolve_secret("inline", key_file, mode="env") == "inline"
    assert resolve_secret(None, key_file, mode="env", keyring_service="s", keyring_user="u") is None
    key_file.chmod(0o644)  # never even looked at
    assert resolve_secret("inline", key_file, mode="env") == "inline"


def test_file_mode_uses_only_the_file(key_file: Path, fake_keyring: dict[str, Any]) -> None:
    fake_keyring[("s", "u")] = "kr"
    assert resolve_secret("inline", key_file, mode="file") == SECRET
    assert (
        resolve_secret("inline", None, mode="file", keyring_service="s", keyring_user="u") is None
    )


# -- keyring -----------------------------------------------------------------------------------


def test_keyring_absent_is_silent_in_auto_and_an_error_in_keyring_mode(
    no_keyring: None, caplog: pytest.LogCaptureFixture
) -> None:
    kr = {"keyring_service": "mesa-clm", "keyring_user": "clm"}
    with caplog.at_level(logging.DEBUG, logger="mesa_clm.secrets"):
        assert resolve_secret(None, None, mode="auto", **kr) is None
    assert "not installed" in caplog.text
    with pytest.raises(SecretError, match=r"keyring.*not installed"):
        resolve_secret(None, None, mode="keyring", **kr)
    with pytest.raises(SecretError, match="service and user"):
        resolve_secret(None, None, mode="keyring")


def test_keyring_mode_uses_only_the_keyring(key_file: Path, fake_keyring: dict[str, Any]) -> None:
    kr = {"keyring_service": "mesa-clm", "keyring_user": "encoder"}
    fake_keyring[("mesa-clm", "encoder")] = "from-keyring"
    assert resolve_secret("inline", key_file, mode="keyring", **kr) == "from-keyring"
    fake_keyring[("mesa-clm", "encoder")] = ""
    assert resolve_secret("inline", key_file, mode="keyring", **kr) is None
    del fake_keyring[("mesa-clm", "encoder")]
    assert resolve_secret("inline", key_file, mode="keyring", **kr) is None


def test_keyring_backend_errors(
    fake_keyring: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    fake_keyring["raise"] = True
    kr = {"keyring_service": "mesa-clm", "keyring_user": "clm"}
    with caplog.at_level(logging.WARNING, logger="mesa_clm.secrets"):
        assert resolve_secret(None, None, mode="auto", **kr) is None
    assert "RuntimeError" in caplog.text and "would be" not in caplog.text  # type, not message
    with pytest.raises(SecretError, match="RuntimeError") as info:
        resolve_secret(None, None, mode="keyring", **kr)
    assert "would be" not in str(info.value)


# -- nothing leaks ----------------------------------------------------------------------------


def test_values_never_reach_the_log(
    key_file: Path, fake_keyring: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    fake_keyring[("mesa-clm", "clm")] = "kr-" + SECRET
    kr = {"keyring_service": "mesa-clm", "keyring_user": "clm"}
    with caplog.at_level(logging.DEBUG, logger="mesa_clm.secrets"):
        assert resolve_secret("v-" + SECRET, None, mode="auto") == "v-" + SECRET
        assert resolve_secret(None, key_file, mode="auto") == SECRET
        assert resolve_secret(None, None, mode="auto", **kr) == "kr-" + SECRET
    assert caplog.text  # something was logged...
    assert SECRET not in caplog.text  # ...but never a value
    assert str(key_file) in caplog.text  # the path is fine
