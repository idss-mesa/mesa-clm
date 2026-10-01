"""The real CLM provider on the serving host (plan §4.1, §6.4, §6.7; DESIGN D5, D16, D28).

:func:`clm_provider` builds what ``mesa-clm annotate --provider clm`` (and, from M3, the
``mesa_clm_*`` tools) decide with: a :class:`~mesa_clm.providers.tiered.TieredProvider` over
:class:`~mesa_clm.clm.http.ClmHttpClient` (``clm`` section: loopback URL, key from the
configured secrets source) with :class:`~mesa_clm.clm.encoder.EncoderClient` (``encoder``
section) as the token guard's counter, stamped with the *live* fingerprint: the serving lock the
host runs (:func:`live_lock_path`: ``~/.mesa/clm/serving.lock.json``, the copy the bootstrap
installed, else the checkout's ``serving/serving.lock.json``) under the served head
``clm.model``. A lock that does not verify refuses the build (K4); promoted heads (M7) are not
servable yet.

:func:`clm_status` is the pre-flight ``GET /health`` (unguarded by design on loopback, 5 s, no
retry) that tells an unreachable clm-serve from a reachable one whose encoder is down
(``embedder: false``). The CLI's rule for a failed pre-flight (``decider.ols_rank_fallback``) is
documented in :mod:`mesa_clm.cli`. Nothing here reads, logs or returns a key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.fingerprint import ServingLock, load_serving_lock
from mesa_clm.clm.http import HEALTH_TIMEOUT_S, ClmHttpClient
from mesa_clm.config import Config
from mesa_clm.providers.tiered import TieredProvider
from mesa_clm.serving import INSTALLED_LOCK, default_lock_path, serving_home

__all__ = [
    "SERVABLE_MODELS",
    "LiveStack",
    "ProviderSetupError",
    "clm_provider",
    "clm_status",
    "live_lock_path",
]

# The heads clm-serve serves under the pinned lock; promoted heads arrive with M7 (D15).
SERVABLE_MODELS: Final[tuple[str, ...]] = ("clm-latest", "clm-raw")


class ProviderSetupError(RuntimeError):
    """The live provider cannot be built (unservable head, missing or tampered lock)."""


def live_lock_path(home: str | Path | None = None) -> Path:
    """The serving lock this host runs: the installed copy under the serving home when there is
    one, else the checkout's ``serving/serving.lock.json``."""
    installed = serving_home(home) / INSTALLED_LOCK
    return installed if installed.is_file() else default_lock_path(home)


@dataclass
class LiveStack:
    """The live provider and the two clients behind it (closed together)."""

    provider: TieredProvider
    client: ClmHttpClient
    encoder: EncoderClient
    lock: ServingLock
    lock_path: Path

    def close(self) -> None:
        self.client.close()
        self.encoder.close()


def clm_provider(
    cfg: Config,
    *,
    home: str | Path | None = None,
    lock_path: str | Path | None = None,
    transport: httpx.BaseTransport | None = None,
) -> LiveStack:
    """The live CLM provider (module docstring). ``transport`` is for tests (one
    ``httpx.MockTransport`` answers both ports); ``lock_path`` overrides :func:`live_lock_path`.
    :class:`ProviderSetupError` for an unservable ``clm.model`` or a lock that does not verify;
    :class:`~mesa_clm.net.EndpointError` for a URL the loopback rule refuses."""
    model = cfg.clm.model
    if model not in SERVABLE_MODELS:
        raise ProviderSetupError(
            f"clm.model {model!r} is not servable yet: promoted heads arrive with M7 (D15); "
            f"use one of {', '.join(SERVABLE_MODELS)}"
        )
    path = Path(lock_path).expanduser() if lock_path is not None else live_lock_path(home)
    try:
        lock = load_serving_lock(path)
    except ValueError as exc:  # LockError: missing, malformed or re-signed without its content
        raise ProviderSetupError(f"serving lock: {exc}") from None
    client = ClmHttpClient.from_config(cfg.clm, transport=transport)
    try:
        encoder = EncoderClient.from_config(
            cfg.encoder, allow_remote=cfg.clm.allow_remote, transport=transport
        )
    except Exception:
        client.close()
        raise
    provider = TieredProvider(
        client, encoder, lock.fingerprint(model), method="clm", name="clm", model=model
    )
    return LiveStack(provider, client, encoder, lock, path)


def clm_status(client: ClmHttpClient) -> tuple[bool, str]:
    """``(answering, detail)`` from clm-serve's ``GET /health``: answering means ``ok`` and an
    encoder behind it (``embedder`` not false). The detail names models, never a key."""
    try:
        j: Any = client.endpoint.get_json("/health", timeout=HEALTH_TIMEOUT_S, retry=False)
    except Exception as exc:  # a probe: every failure is "not answering", with its reason
        return False, f"clm-serve at {client.endpoint.shown} unreachable ({type(exc).__name__})"
    if not isinstance(j, dict) or not j.get("ok"):
        return False, f"clm-serve at {client.endpoint.shown} answered /health without ok"
    if j.get("embedder") is False:
        return False, (
            f"clm-serve at {client.endpoint.shown} is up but its encoder is not (embedder: false)"
        )
    models = j.get("models")
    names = ", ".join(str(m) for m in models) if isinstance(models, list) else "?"
    return True, f"clm-serve at {client.endpoint.shown} ok (models: {names})"
