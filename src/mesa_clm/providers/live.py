"""The real CLM provider on the serving host (plan §4.1, §6.4, §6.7; DESIGN D5, D16, D28, K4).

:func:`clm_provider` builds what ``mesa-clm annotate --provider clm`` (and, from M3, the
``mesa_clm_*`` tools) decide with: a :class:`~mesa_clm.providers.tiered.TieredProvider` over
:class:`~mesa_clm.clm.http.ClmHttpClient` (``clm`` section: loopback URL, key from the
configured secrets source) with :class:`~mesa_clm.clm.encoder.EncoderClient` (``encoder``
section) as the token guard's counter, stamped with the *live* fingerprint: the serving lock the
host runs (:func:`live_lock_path`: ``~/.mesa/clm/serving.lock.json``, the copy the bootstrap
installed, else the checkout's ``serving/serving.lock.json``) under the served head
``clm.model``. Before anything is asked it refuses (:class:`ProviderSetupError`, K4):

* a lock that does not verify (``lock_sha``, ``encoder_fp``);
* a lock whose ``schema_sha256`` is not the vendored ``schema.py`` the client renders with
  (``render.schema_sha256()``): every record would cite a schema the server does not use;
* an installed lock that differs from the checkout's (the checkout moved and the bootstrap was
  not re-run, so the host runs another recipe than the code describes);
* a served head other than ``clm-latest``/``clm-raw`` (promoted heads arrive with M7).

**Artifacts (M4, plan §5.5).** When ``<artifacts.dir>/<encoder_fp>/<clm_model_fp>/<lock8>/
CURRENT.json`` names promoted artifacts for the served head, the provider serves them
(:func:`mesa_clm.artifacts.bundle_from_current`): every version is re-hashed against its
manifest, its fingerprints must be the live ones and its framings lock the code's, and a
promoted probe needs the pinned head's export (:func:`head_export`, as the doctor loads it)
when its spec reads head quantities; a mismatch refuses (K4), never a silent ``zero_shot``,
unless ``artifacts.strict`` is false, which drops the mismatched entries with a warning noted
on the run. Promoted artifacts filed under another framings lock than the live one are
refused the same way. Without a ``CURRENT.json`` the provider serves ``zero_shot``.

:func:`preflight` is what the CLI asks next, before a single question: clm-serve's ``/health``
(unguarded on loopback, 5 s, no retry), which tells an unreachable clm-serve, or one whose
encoder is down (``embedder: false``), from a reachable one (the caller may degrade to
``ols_rank``, D28); then, with the keys, what no fallback may paper over and which raises
:class:`PreflightError`: a missing key, a key the server rejects (401/403), a ``clm.model`` the
server does not list, and an encoder that is not the lock's (served name, ``root`` model,
``max_model_len`` window, and ``owned_by`` against the lock's ``route``: the vLLM image says
``vllm``, the in-process fallback ``mesa-clm-fallback``, so a fallback serving :8090 under the
vLLM lock is refused instead of stamping the vLLM ``encoder_fp`` on its answers), and a loopback
port held by another account's socket (:class:`mesa_clm.net.ListenerOwnerError`: the keyed
clients refuse to send the key there, DESIGN A5).
:func:`encoder_problems` is that last check on its own (``features build`` uses it too).
:func:`clm_status` is the ``/health`` half alone.

``/v1/models`` cannot tell two containers of the same image apart: one started without
``VLLM_BATCH_INVARIANT=1``, or with other ``vllm serve`` flags, answers exactly the same, yet
its vectors differ (min cosine 0.99937, ``bench/results/2026-10-01/batch_invariance.json``).
:func:`container_check` therefore inspects the running encoder container against the lock
(:func:`mesa_clm.serving.check_encoder_container`: image, arguments, environment, mounts,
network) before ``features build`` writes a vector and before an annotate run; a container that
departs from the lock is refused, and when docker cannot be asked (another host, no docker
group) the caller records that the recipe was not verified. Nothing here reads, logs or returns
a key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx

from mesa_clm import render
from mesa_clm.artifacts import ArtifactLayout, bundle_from_current
from mesa_clm.clm.encoder import EncoderClient
from mesa_clm.clm.fingerprint import Fingerprint, LockError, ServingLock, load_serving_lock
from mesa_clm.clm.headproj import HeadError, HeadProjector
from mesa_clm.clm.http import HEALTH_TIMEOUT_S, ClmError, ClmHttpClient
from mesa_clm.config import Config
from mesa_clm.net import EndpointError, ListenerOwnerError
from mesa_clm.providers.tiered import ArtifactBundle, ArtifactError, TieredProvider
from mesa_clm.serving import (
    HEADS_DIR,
    INSTALLED_LOCK,
    DockerArgv,
    Runner,
    check_encoder_container,
    default_lock_path,
    docker_argv,
    run_command,
    serving_home,
)

__all__ = [
    "ROUTE_OWNED_BY",
    "SERVABLE_MODELS",
    "ContainerCheck",
    "LiveStack",
    "PreflightError",
    "ProviderSetupError",
    "artifacts_for",
    "clm_provider",
    "clm_status",
    "container_check",
    "encoder_problems",
    "head_export",
    "live_lock_path",
    "preflight",
]

# The heads clm-serve serves under the pinned lock; promoted heads arrive with M7 (D15).
SERVABLE_MODELS: Final[tuple[str, ...]] = ("clm-latest", "clm-raw")
# What each encoder route answers as ``owned_by`` in ``/v1/models`` (vLLM's server;
# serving/fallback_serve.py): the lock's ``route`` (D16) must be the one serving :8090.
ROUTE_OWNED_BY: Final[dict[str, str]] = {"vllm": "vllm", "transformers": "mesa-clm-fallback"}
_REJECTED: Final[frozenset[int]] = frozenset({401, 403})
_KEY_HINT: Final[dict[str, str]] = {
    "clm": "MESA_CLM_CLM__API_KEY_FILE (default ~/.mesa/clm/secrets/clm.key)",
    "encoder": "MESA_CLM_ENCODER__API_KEY_FILE (default ~/.mesa/clm/secrets/encoder.key)",
}


class ProviderSetupError(RuntimeError):
    """The live provider cannot be built (unservable head, missing, tampered or contradicting
    lock)."""


class PreflightError(RuntimeError):
    """The serving pair answers but cannot be used as configured: a missing or rejected key, an
    unserved model, an encoder that is not the lock's. Never degraded to ``ols_rank``."""


def live_lock_path(home: str | Path | None = None) -> Path:
    """The serving lock this host runs: the installed copy under the serving home when there is
    one, else the checkout's ``serving/serving.lock.json``."""
    installed = serving_home(home) / INSTALLED_LOCK
    return installed if installed.is_file() else default_lock_path(home)


@dataclass
class LiveStack:
    """The live provider and the two clients behind it (closed together); ``notes`` what the
    artifacts loading wants recorded on the run (module docstring)."""

    provider: TieredProvider
    client: ClmHttpClient
    encoder: EncoderClient
    lock: ServingLock
    lock_path: Path
    artifacts: ArtifactBundle | None = None
    notes: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.client.close()
        self.encoder.close()


def head_export(lock: ServingLock, home: str | Path | None = None) -> HeadProjector | None:
    """The pinned head's numpy export under the serving home (``heads/npz/<sha8>.npz``, plan
    §6.2) when present and exported from the lock's checkpoint (``source_sha256``; K4), else
    ``None`` (the doctor's rule, ``health._head``)."""
    path = serving_home(home) / HEADS_DIR / "npz" / f"{lock.head.sha256[:8]}.npz"
    if not path.is_file():
        return None
    try:
        head = HeadProjector.from_npz(path)
    except HeadError:
        return None
    return head if head.source_sha256 == lock.head.sha256 else None


def artifacts_for(cfg: Config, fp: Fingerprint) -> tuple[ArtifactBundle | None, list[str]]:
    """The promoted artifacts of the served head (module docstring): ``(bundle, notes)``;
    ``None`` when nothing is promoted. :class:`ProviderSetupError` for a bundle that must not
    be served under ``artifacts.strict`` (K4)."""
    layout = ArtifactLayout.from_config(cfg, fp)
    notes: list[str] = []
    strict = cfg.artifacts.strict
    others = layout.other_locks_with_current()
    if others and not layout.current_path.is_file():
        message = (
            f"promoted artifacts under {layout.model_dir} are filed under framings lock(s) "
            f"{', '.join(others)}, not the live {layout.framings_lock_sha[:8]} (K4: re-bench "
            "and re-promote under the live lock)"
        )
        if strict:
            raise ProviderSetupError(message + "; or set artifacts.strict false to serve zero_shot")
        notes.append(f"artifacts: {message}; serving zero_shot (artifacts.strict false)")
        return None, notes
    if not layout.current_path.is_file():
        return None, notes
    try:
        bundle = bundle_from_current(layout, live=fp, strict=strict)
    except ArtifactError as exc:
        raise ProviderSetupError(f"artifacts: {exc}") from None
    if bundle is None:
        notes.append(
            f"artifacts: {layout.current_path} promotes nothing servable (artifacts.strict "
            f"{'true' if strict else 'false'}); serving zero_shot"
        )
        return None, notes
    served = ", ".join(
        f"{task} {promo.tier} {promo.version}" for task, promo in sorted(bundle.promoted.items())
    )
    notes.append(f"artifacts {bundle.version} ({layout.current_path}): {served}")
    return bundle, notes


def _checked_lock(path: Path, home: str | Path | None, explicit: bool) -> ServingLock:
    """The lock at ``path``, verified and consistent with the code and, unless the caller named
    the lock (``explicit``), with the checkout's copy (module docstring)."""
    try:
        lock = load_serving_lock(path)
    except (LockError, ValueError) as exc:  # missing, malformed or re-signed without content
        raise ProviderSetupError(f"serving lock: {exc}") from None
    vendored = render.schema_sha256()
    if lock.schema_sha256 != vendored:
        raise ProviderSetupError(
            f"serving lock {path}: schema_sha256 {lock.schema_sha256[:12]} is not the vendored "
            f"schema.py {vendored[:12]} the client renders with (K4: rebuild the serving host "
            "from this checkout)"
        )
    if not explicit:
        checkout = default_lock_path(home)
        if checkout != path and checkout.is_file():
            try:
                theirs = load_serving_lock(checkout)
            except (LockError, ValueError) as exc:
                raise ProviderSetupError(f"serving lock: {exc}") from None
            if theirs.lock_sha != lock.lock_sha:
                raise ProviderSetupError(
                    f"the installed serving lock {path} (lock_sha {lock.lock_sha[:12]}) differs "
                    f"from the checkout's {checkout} ({theirs.lock_sha[:12]}): the host runs "
                    "another recipe than this code describes; re-run "
                    "deploy/bin/mesa-clm-serve-bootstrap, which installs the checkout's lock "
                    "(K4)"
                )
    return lock


def clm_provider(
    cfg: Config,
    *,
    home: str | Path | None = None,
    lock_path: str | Path | None = None,
    transport: httpx.BaseTransport | None = None,
) -> LiveStack:
    """The live CLM provider (module docstring). ``transport`` is for tests (one
    ``httpx.MockTransport`` answers both ports); ``lock_path`` overrides :func:`live_lock_path`
    (and skips the installed-versus-checkout comparison). :class:`ProviderSetupError` for an
    unservable ``clm.model`` or a lock the module docstring refuses;
    :class:`~mesa_clm.net.EndpointError` for a URL the loopback rule refuses."""
    model = cfg.clm.model
    if model not in SERVABLE_MODELS:
        raise ProviderSetupError(
            f"clm.model {model!r} is not servable yet: promoted heads arrive with M7 (D15); "
            f"use one of {', '.join(SERVABLE_MODELS)}"
        )
    path = Path(lock_path).expanduser() if lock_path is not None else live_lock_path(home)
    lock = _checked_lock(path, home, explicit=lock_path is not None)
    fp = lock.fingerprint(model)
    bundle, notes = artifacts_for(cfg, fp)
    head: HeadProjector | None = None
    store = None
    if bundle is not None and bundle.probes:
        from mesa_clm.learn.features import FeatureStore

        promoted = [p for p in bundle.probes.values() if bundle.promotion_for(p.question_key)]
        if any(p.needs_head for p in promoted):
            head = head_export(lock, home)
            if head is None:
                raise ProviderSetupError(
                    "a promoted probe reads head quantities but the pinned head's export "
                    f"(heads/npz/{lock.head.sha256[:8]}.npz) is missing or not the lock's (K4)"
                )
        live_store = FeatureStore.for_lock(cfg, lock)
        if live_store.exists():
            store = live_store  # read-only on the probe path; never written by a provider
            notes.append(f"artifacts: encoder vectors cached from {live_store.path} (read-only)")
    client = ClmHttpClient.from_config(cfg.clm, transport=transport)
    try:
        encoder = EncoderClient.from_config(
            cfg.encoder, allow_remote=cfg.clm.allow_remote, transport=transport
        )
    except Exception:
        client.close()
        raise
    try:
        provider = TieredProvider(
            client,
            encoder,
            fp,
            bundle,
            method="clm",
            name="clm",
            model=model,
            head=head,
            feature_store=store,
        )
    except ArtifactError as exc:
        client.close()
        encoder.close()
        raise ProviderSetupError(f"artifacts: {exc}") from None
    return LiveStack(provider, client, encoder, lock, path, bundle, notes)


def clm_status(client: ClmHttpClient) -> tuple[bool, str]:
    """``(answering, detail)`` from clm-serve's ``GET /health``: answering means ``ok`` and an
    encoder behind it (``embedder`` not false). The detail names models, never a key. A port
    another account holds is :class:`PreflightError` (the key is not sent; DESIGN A5)."""
    try:
        j: Any = client.endpoint.get_json("/health", timeout=HEALTH_TIMEOUT_S, retry=False)
    except ListenerOwnerError as exc:  # another account holds the port: never degraded
        raise PreflightError(str(exc)) from None
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


def encoder_problems(
    models: Sequence[dict[str, Any]], served_name: str, lock: ServingLock
) -> list[str]:
    """What the encoder's ``/v1/models`` says against the lock (module docstring): the
    configured served name listed, its ``root`` the lock's model, its ``max_model_len`` the
    lock's window, its ``owned_by`` the lock's route. Empty when it matches."""
    entry = next((m for m in models if str(m.get("id")) == served_name), None)
    if entry is None:
        served = ", ".join(str(m.get("id")) for m in models) or "nothing"
        return [f"the encoder does not serve {served_name!r} (serves {served})"]
    problems: list[str] = []
    root, window, owner = entry.get("root"), entry.get("max_model_len"), entry.get("owned_by")
    if root is not None and str(root) != lock.encoder.model:
        problems.append(
            f"the encoder serves {root} as {served_name!r}, the lock pins {lock.encoder.model}"
        )
    if window is not None and int(window) != lock.encoder.max_len:
        problems.append(
            f"the encoder's max_model_len {window} is not the lock's window {lock.encoder.max_len}"
        )
    want = ROUTE_OWNED_BY.get(lock.encoder.route)
    if owner is not None and want is not None and str(owner) != want:
        problems.append(
            f"the encoder is owned_by {owner!r}, not the lock's route {lock.encoder.route!r} "
            f"({want!r}): its vectors are not encoder_fp {lock.encoder_fp}"
        )
    return problems


@dataclass(frozen=True)
class ContainerCheck:
    """What :func:`container_check` found: ``problems`` (the container departs from the lock;
    the caller refuses), else ``verified`` (it was inspected and matches) or not (``detail``
    says why it could not be inspected, which the caller records)."""

    verified: bool
    problems: tuple[str, ...]
    detail: str

    @property
    def ok(self) -> bool:
        return not self.problems

    def note(self) -> str:
        """One line for a run's notes and summaries."""
        if self.problems:
            return "encoder container: " + "; ".join(self.problems)
        if self.verified:
            return f"encoder container: {self.detail}"
        return f"encoder container: not verified ({self.detail})"


def container_check(
    lock: ServingLock, *, runner: Runner | None = None, docker: DockerArgv | None = None
) -> ContainerCheck:
    """The running encoder container against ``lock`` (module docstring). ``runner`` and
    ``docker`` default to this module's :func:`~mesa_clm.serving.run_command` and
    :func:`~mesa_clm.serving.docker_argv` (the hermetic suite replaces the latter)."""
    checks = check_encoder_container(
        lock, runner=runner or run_command, docker=docker or docker_argv
    )
    problems = tuple(f"{c.name}: {c.detail}" for c in checks if c.status == "fail")
    if problems:
        return ContainerCheck(False, problems, "")
    skipped = [c.detail for c in checks if c.status == "skip"]
    if skipped:
        return ContainerCheck(False, (), "; ".join(skipped))
    container = next((c.detail for c in checks if c.name == "encoder container"), "")
    return ContainerCheck(True, (), container or "matches the lock")


def _models(what: str, call: Any, shown: str) -> list[dict[str, Any]] | None:
    """The authenticated model list, ``None`` when nothing answered, :class:`PreflightError`
    for a rejected key."""
    try:
        models: list[dict[str, Any]] = call()
    except ClmError as exc:
        if exc.status in _REJECTED:
            raise PreflightError(
                f"{what} at {shown} rejected the configured key ({exc.status}); set "
                f"{_KEY_HINT['clm' if what == 'clm-serve' else 'encoder']} to the unit's key "
                "(`mesa-clm serve keys --init` writes both)"
            ) from None
        if exc.status == 0:
            return None
        raise PreflightError(f"{what} at {shown}: GET /v1/models -> {exc.status}") from None
    except ListenerOwnerError as exc:
        raise PreflightError(str(exc)) from None
    except EndpointError:
        return None
    return models


def preflight(stack: LiveStack, cfg: Config) -> tuple[bool, str]:
    """``(answering, detail)`` before a run (module docstring). ``(False, why)`` when clm-serve
    or its encoder does not answer (the CLI's degrade rule applies); :class:`PreflightError`
    for a missing or rejected key, an unserved ``clm.model`` or an encoder other than the
    lock's."""
    for what, endpoint in (("clm", stack.client.endpoint), ("encoder", stack.encoder.endpoint)):
        if not endpoint.has_key:
            raise PreflightError(
                f"no {what} key configured: set {_KEY_HINT[what]} "
                "(`mesa-clm serve keys --init` writes both)"
            )
    answering, detail = clm_status(stack.client)
    if not answering:
        return False, detail
    shown = stack.client.endpoint.shown
    served = _models("clm-serve", stack.client.models, shown)
    if served is None:
        return False, f"clm-serve at {shown} stopped answering after /health"
    names = [str(m.get("name")) for m in served]
    if cfg.clm.model not in names:
        raise PreflightError(
            f"clm-serve at {shown} does not serve clm.model {cfg.clm.model!r} "
            f"(serves {', '.join(names) or 'nothing'})"
        )
    enc_shown = stack.encoder.endpoint.shown
    enc_models = _models("encoder", stack.encoder.models, enc_shown)
    if enc_models is None:
        return False, f"the encoder at {enc_shown} does not answer"
    problems = encoder_problems(enc_models, cfg.encoder.model, stack.lock)
    if problems:
        raise PreflightError(f"encoder at {enc_shown}: " + "; ".join(problems) + " (D5, K4)")
    return True, f"{detail}; encoder {cfg.encoder.model} at {enc_shown} matches the lock"
