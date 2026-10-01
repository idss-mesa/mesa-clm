"""Configuration model and loader.

Precedence (highest wins): explicit flag > environment variable > YAML file > default.
Environment variables use the ``MESA_CLM_`` prefix and ``__`` to descend into a section:
``MESA_CLM_CLM__BASE_URL=http://127.0.0.1:8700`` maps to ``config.clm.base_url``. A value for a
string-typed field is taken verbatim (an empty value or a YAML null spelling means unset); every
other value is parsed with ``yaml.safe_load`` so ``true`` and ``12`` mean what they say. That
split fixes two mesa-anyjev traps: ``OLS__FIXTURES=off`` became ``False`` (YAML 1.1) and an
all-digit key became an ``int``.

Foreign names are honoured as fallbacks, each only when the native ``MESA_CLM_`` name is absent
from the environment (plan §6.4; mesa-anyjev's mesa-mcp compatibility rule):

* ``CLM_BASE_URL`` / ``CLM_API_KEY`` (CLM's own client) -> ``clm.base_url`` / ``clm.api_key``;
* ``MESA_LLM_BASE_URL`` (a trailing ``/v1`` is stripped, the planner appends its own paths) and
  ``MESA_LLM_API_KEY`` (mesa-mcp's names) -> ``planner.gateway_base_url`` / ``gateway_api_key``;
* ``MESA_HOME`` moves the *default* of ``history.lock_path`` to
  ``$MESA_HOME/locks/mesa-catalog.lock`` (DESIGN D12); the YAML file and the native name win.

API keys are :class:`pydantic.SecretStr`, so a printed or logged config shows ``**********``;
code reads them through ``resolved_api_key()``, which applies the ``MESA_CLM_SECRETS`` source
mode (:mod:`mesa_clm.secrets`). :func:`config_sha256` hashes the configuration with every secret
removed and every DSN password redacted. Shape is validated here; whether a host answers is the
doctor's job and whether a URL may be called at all is :func:`mesa_clm.net.assert_loopback`'s.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)

from mesa_clm.secrets import SecretsMode, resolve_secret

logger = logging.getLogger(__name__)

ENV_PREFIX = "MESA_CLM_"
ENV_DELIM = "__"
# The keyring service every mesa-clm key lives under; the user names are the section keys below.
KEYRING_SERVICE = "mesa-clm"
MiB = 1024 * 1024

PlannerKind = Literal["static", "gateway", "claude"]
# ``ols_rank`` is the degraded D28 method for every rank_fit task (proposed-only, never auto).
Tier = Literal["auto", "zero_shot", "calibrated", "probe", "head", "ols_rank"]
Profile = Literal["prod", "dev"]
FixtureMode = Literal["off", "record", "replay", "auto"]
HistoryBackend = Literal["direct", "spool", "none", "auto"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


# ``hide_input_in_errors``: a validation error names the field and the problem, never the value,
# so a mistyped ``MESA_CLM_CLM__API_KEY`` (or a key given where a string is expected) cannot
# echo the secret into stderr or a CI log (``cli.main`` prints the errors).
class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class _KeyedSection(_Section):
    """A section that carries an API key. :class:`Config` stamps its secrets mode after
    validation so ``resolved_api_key()`` needs no argument; a section built on its own resolves
    in ``auto`` mode."""

    _secrets_mode: SecretsMode = PrivateAttr(default="auto")

    def _resolve(
        self, value: SecretStr | None, file: str | None, user: str, mode: SecretsMode | None
    ) -> str | None:
        return resolve_secret(
            value.get_secret_value() if value is not None else None,
            file,
            keyring_service=KEYRING_SERVICE,
            keyring_user=user,
            mode=mode or self._secrets_mode,
        )


class ClmConfig(_KeyedSection):
    """The clm-serve endpoint that scores candidate texts against a state (plan §4.1, §6.2)."""

    # Loopback only; a remote host needs allow_remote *and* https (net.assert_loopback, D16).
    base_url: str = "http://127.0.0.1:8700"
    api_key: SecretStr | None = None
    # A 0600 file holding the raw key (`mesa-clm serve keys --init` writes it; plan §6.4).
    api_key_file: str | None = None
    # The served head: clm-latest | clm-raw | a promoted head's unique name (D15).
    model: str = "clm-latest"
    timeout: float = Field(default=120.0, gt=0)
    # Applies to the encoder URL as well: one switch for the serving pair (plan §6.4).
    allow_remote: bool = False

    def resolved_api_key(self, mode: SecretsMode | None = None) -> str | None:
        """The bearer key for clm-serve from value, key file or keyring (``mesa-clm``/``clm``)."""
        return self._resolve(self.api_key, self.api_key_file, "clm", mode)


class EncoderConfig(_KeyedSection):
    """The vLLM pooling server (Qwen3-8B, last-token, L2) behind ``/v1/embeddings`` (§6.1)."""

    url: str = "http://127.0.0.1:8090"
    api_key: SecretStr | None = None
    api_key_file: str | None = None
    # --served-model-name of the encoder unit.
    model: str = "qwen3-8b"
    # --max-model-len; the client token guard abstains above max_len - 16 (D23, §4.3).
    max_len: int = Field(default=4096, gt=16)
    timeout: float = Field(default=120.0, gt=0)
    # The pinned Qwen3 revision's tokenizer.json for the token guard's second counter (the
    # `tokenize` extra) when the encoder has no /tokenize; unset: chars/2 (plan §4.3).
    tokenizer_json: str | None = None

    def resolved_api_key(self, mode: SecretsMode | None = None) -> str | None:
        """The bearer key for the encoder (keyring entry ``mesa-clm``/``encoder``)."""
        return self._resolve(self.api_key, self.api_key_file, "encoder", mode)


class DeciderConfig(_Section):
    """Which tier answers (D6): ``auto`` takes the best promoted artifact, else zero_shot;
    ``ols_rank`` sends every rank_fit task to the degraded method (D28)."""

    tier: Tier = "auto"
    # `annotate --provider clm` with tier auto when clm-serve does not answer /health: False
    # refuses (decider_unavailable, the plugin's rule); True runs the degraded ols_rank method
    # instead (proposed-only, never auto; D28). An explicit CLM tier always refuses.
    ols_rank_fallback: bool = False


class PlannerConfig(_KeyedSection):
    """The reasoning model: it plans, never decides (DESIGN D22)."""

    kind: PlannerKind = "static"
    # Loopback only; 127.0.0.1:8000 is the carc-litellm-tunnel user unit (mesa-anyjev RESEARCH.md).
    gateway_base_url: str = "http://127.0.0.1:8000"
    gateway_api_key: SecretStr | None = None
    gateway_api_key_file: str | None = None
    gateway_model: str = "carc-tools"
    claude_model: str = "claude-opus-5"
    claude_effort: Effort = "high"
    timeout: float = Field(default=600.0, gt=0)

    def resolved_gateway_api_key(self, mode: SecretsMode | None = None) -> str | None:
        """The bearer key for the planner gateway (keyring entry ``mesa-clm``/``gateway``)."""
        return self._resolve(self.gateway_api_key, self.gateway_api_key_file, "gateway", mode)


class ClaudeConfig(_Section):
    """The Claude structured-output provider (second opinion only, never ``auto``; D22).
    Credentials resolve through the SDK (``ANTHROPIC_API_KEY``), not here."""

    model: str = "claude-opus-5"
    effort: Effort = "low"
    max_tokens: int = Field(default=1024, gt=0)
    # Ask Claude the term.fits question over a proposed group's top candidates and record the
    # answers as a second opinion (level none); a disagreement escalates to a human.
    second_opinion: bool = False
    second_opinion_top_k: int = Field(default=8, gt=0)


class PolicyConfig(_Section):
    profile: Profile = "prod"
    # None: the packaged policy_defaults.yaml (D9); a path replaces it wholesale.
    policy_path: str | None = None
    max_candidates: int = Field(default=12, gt=0)
    max_avus: int = Field(default=25, gt=0)
    proposal_size: int = Field(default=8, gt=0)
    max_wait_s: float = Field(default=30.0, ge=0)
    # Specificity (D24): a child term replaces its parent when p_fit(child) >= p_fit(parent) + delta.
    specificity: bool = True
    specificity_delta: float = Field(default=0.10, ge=0)


class ArtifactsConfig(_Section):
    dir: str = "~/.mesa/clm/artifacts"
    # Refuse fingerprint mismatches and stale question keys on load (§5.5).
    strict: bool = True


class FeaturesConfig(_Section):
    """The per-encoder feature cache (§5.2): ``<dir>/<encoder_fp>/features.duckdb``."""

    dir: str = "~/.mesa/clm/features"


class ProvenanceConfig(_Section):
    """The ``mesa_clm`` sidecar (D11): DuckDB per host, Postgres optional."""

    dsn: str = "duckdb:///~/.mesa/clm/provenance.duckdb"
    # Abandoned runs are pruned after this many days; terminal runs once exported (§7.3 step 9).
    ttl_days: int = Field(default=30, ge=1)


class DuckLakeConfig(_Section):
    """mesa-ducklake for history ``direct`` mode and the local CLI/test lake (§7.3)."""

    # The shared MESA catalog (duckdb:///~/.mesa/ducklake/catalog.duckdb) must be named
    # explicitly; the default is a private local lake so nothing touches it by accident.
    catalog_dsn: str | None = "duckdb:///~/.mesa/clm/ducklake-catalog.duckdb"
    cache_dir: str | None = "~/.mesa/clm/ducklake-cache"
    # Removed after close; 0 would mean unbounded, so it is refused.
    cache_cap_bytes: int = Field(default=256 * MiB, gt=0)
    local_project_root: str = "/local/mesa-clm"
    local_zone: str = "local"


class HistoryConfig(_Section):
    """How applied AVUs reach the MESA history (D12): the plugin always spools; ``direct`` is
    CLI-only and needs the whole lock set free."""

    backend: HistoryBackend = "auto"
    # DuckDB 2.0 multi-writer switch; until then direct mode is single-writer under the lock set.
    multiwriter: bool = False
    # Batches land in <spool_root>/<vm_id>/ as mesa-spool/1 (Parquet + manifest).
    spool_root: str = "~/.mesa/clm/spool"
    # load_config() moves this default to $MESA_HOME/locks/mesa-catalog.lock when MESA_HOME is set.
    lock_path: str = "~/.mesa/locks/mesa-catalog.lock"
    # neon-ducklake's <NEON_DUCKLAKE_ROOT>/work/locks/mesa-tag.lock when neon shares the catalog.
    neon_lock_path: str | None = None

    def lock_paths(self) -> list[Path]:
        """The lock set direct mode and the recorder must hold, expanded (D12)."""
        paths = [expand_path(self.lock_path)]
        if self.neon_lock_path:
            paths.append(expand_path(self.neon_lock_path))
        return paths


class ApplyConfig(_Section):
    # Token bucket over the iRODS write calls (atomic batches and paced add_avu; §7.3 step 5).
    max_calls_per_s: float = Field(default=5.0, gt=0)


class OLSConfig(_Section):
    base_url: str = "https://www.ebi.ac.uk/ols4/api/v2"
    fixtures: FixtureMode = "off"
    fixtures_dir: str = "tests/fixtures/ols"

    @field_validator("fixtures", mode="before")
    @classmethod
    def _yaml_off_is_a_word(cls, value: Any) -> Any:
        # YAML 1.1 reads a bare `off` as False; this field means the word.
        return "off" if value is False else value


class NeonConfig(_Section):
    """The neon-ducklake root the 0.2.0 adapter reads cards from (§7.4)."""

    root: str | None = None


class CardConfig(_Section):
    # Cards are streamed from iRODS in memory and never stored (D29); bigger is refused.
    max_bytes: int = Field(default=1 * MiB, gt=0)


_VM_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _default_vm_id() -> str:
    """The host name as one safe path segment (spool batches live under ``<root>/<vm_id>/``)."""
    name = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname()).lstrip("._-")[:64]
    return name or "localhost"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    clm: ClmConfig = Field(default_factory=ClmConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decider: DeciderConfig = Field(default_factory=DeciderConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    ducklake: DuckLakeConfig = Field(default_factory=DuckLakeConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    apply: ApplyConfig = Field(default_factory=ApplyConfig)
    ols: OLSConfig = Field(default_factory=OLSConfig)
    neon: NeonConfig = Field(default_factory=NeonConfig)
    card: CardConfig = Field(default_factory=CardConfig)
    # The neon-avu-eval checkout used for silver labels and the bench. No default path: a
    # personal home directory must not be baked into the package.
    eval_root: str | None = None
    # Where API keys come from (MESA_CLM_SECRETS; mesa_clm.secrets).
    secrets: SecretsMode = "auto"
    # This host's name on runs and spool batches (D29); one path segment, default the hostname.
    vm_id: str = Field(default_factory=_default_vm_id)

    @field_validator("vm_id")
    @classmethod
    def _vm_id_is_one_path_segment(cls, value: str) -> str:
        if not _VM_ID_RE.fullmatch(value):
            raise ValueError(
                "vm_id must be 1-64 characters of letters, digits, '.', '_' or '-' and start "
                "with a letter or digit (it names a spool directory)"
            )
        return value

    @model_validator(mode="after")
    def _stamp_secrets_mode(self) -> Self:
        for section in (self.clm, self.encoder, self.planner):
            section._secrets_mode = self.secrets
        return self


# -- environment mapping ----------------------------------------------------------------------

# YAML's spellings of null; an environment value equal to one of these unsets a nullable string.
_YAML_NULLS = frozenset({"", "~", "null", "Null", "NULL"})


def _coerce(raw: str) -> Any:
    """Environment values are strings; let YAML decide what they mean (``true``, ``12``)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _field_kind(annotation: Any) -> tuple[bool, bool]:
    """``(string-like, nullable)`` for a field annotation: ``str``, ``SecretStr`` and string
    ``Literal`` fields keep their environment value verbatim instead of going through YAML."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = get_args(annotation)
        members = [a for a in args if a is not type(None)]
        stringy = bool(members) and all(_field_kind(a)[0] for a in members)
        return stringy, len(members) < len(args)
    if origin is Literal:
        return all(isinstance(a, str) for a in get_args(annotation)), False
    return annotation in (str, SecretStr), False


def _annotation(section: str, field: str | None) -> Any | None:
    """The annotation of ``Config.<section>`` or ``Config.<section>.<field>``, if it exists."""
    top = Config.model_fields.get(section)
    if top is None:
        return None
    if field is None:
        return top.annotation
    model = top.annotation
    if isinstance(model, type) and issubclass(model, BaseModel):
        info = model.model_fields.get(field)
        return None if info is None else info.annotation
    return None


def _coerce_for(annotation: Any | None, raw: str) -> Any:
    if annotation is not None:
        stringy, nullable = _field_kind(annotation)
        if stringy:
            return None if nullable and raw.strip() in _YAML_NULLS else raw
    return _coerce(raw)


def _bucket(out: dict[str, Any], section: str) -> dict[str, Any] | None:
    """The section dict in ``out`` (created on demand), or ``None`` when the whole section was
    given as a non-mapping value (validation reports that)."""
    existing = out.get(section)
    if existing is None:
        existing = out[section] = {}
    return existing if isinstance(existing, dict) else None


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        rest = key[len(ENV_PREFIX) :]
        if ENV_DELIM in rest:
            section, field = (part.lower() for part in rest.split(ENV_DELIM, 1))
            bucket = _bucket(out, section)
            if bucket is not None:
                bucket[field] = _coerce_for(_annotation(section, field), value)
        elif rest.lower() in Config.model_fields:
            name = rest.lower()
            parsed = _coerce_for(_annotation(name, None), value)
            existing = out.get(name)
            # A whole section given as YAML merges under its MESA_CLM_<SECTION>__<FIELD> entries.
            out[name] = (
                _deep_merge(parsed, existing)
                if isinstance(parsed, dict) and isinstance(existing, dict)
                else parsed
            )
        # Anything else at the top level (MESA_CLM_LIVE, _ENGINE, _GPU, _E2E, _NEON_ROOT,
        # _TEST_PG_DSN) is a test-tier gate read by conftest, not a setting.
    # Foreign-name fallbacks (only when the native name is absent from the environment).
    if env.get("CLM_BASE_URL") or env.get("CLM_API_KEY"):
        clm = _bucket(out, "clm")
        if clm is not None:
            if "base_url" not in clm and env.get("CLM_BASE_URL"):
                clm["base_url"] = env["CLM_BASE_URL"].rstrip("/")
            if not {"api_key", "api_key_file"} & clm.keys() and env.get("CLM_API_KEY"):
                clm["api_key"] = env["CLM_API_KEY"]
    if env.get("MESA_LLM_BASE_URL") or env.get("MESA_LLM_API_KEY"):
        planner = _bucket(out, "planner")
        if planner is not None:
            if "gateway_base_url" not in planner and env.get("MESA_LLM_BASE_URL"):
                base = env["MESA_LLM_BASE_URL"].rstrip("/")
                planner["gateway_base_url"] = base[:-3] if base.endswith("/v1") else base
            if not {"gateway_api_key", "gateway_api_key_file"} & planner.keys() and env.get(
                "MESA_LLM_API_KEY"
            ):
                planner["gateway_api_key"] = env["MESA_LLM_API_KEY"]
    return {k: v for k, v in out.items() if v != {}}


def _env_defaults(env: Mapping[str, str]) -> dict[str, Any]:
    """Defaults that follow the environment but rank *below* the YAML file (D12)."""
    home = env.get("MESA_HOME")
    if home:
        return {"history": {"lock_path": str(Path(home) / "locks" / "mesa-catalog.lock")}}
    return {}


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None,
    *,
    flag_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Build a :class:`Config` from YAML, environment and flags (flags win)."""
    environ = os.environ if env is None else env
    data: dict[str, Any] = _env_defaults(environ)
    if path is not None:
        loaded = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: the configuration file must be a mapping")
        known = set(Config.model_fields)
        for section in list(loaded):
            if section not in known:
                logger.warning("config: unknown section %r ignored", section)
                loaded.pop(section)
        data = _deep_merge(data, loaded)
    data = _deep_merge(data, _env_overrides(environ))
    if flag_overrides:
        data = _deep_merge(data, flag_overrides)
    return Config.model_validate(data)


# -- secret-free views ------------------------------------------------------------------------


def _mentions(annotation: Any, target: type) -> bool:
    if annotation is target:
        return True
    return any(_mentions(a, target) for a in get_args(annotation))


def _secret_fields() -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for name, info in Config.model_fields.items():
        model = info.annotation
        if isinstance(model, type) and issubclass(model, BaseModel):
            found.extend(
                (name, fname)
                for fname, finfo in model.model_fields.items()
                if _mentions(finfo.annotation, SecretStr)
            )
    return tuple(found)


# Every (section, field) holding a SecretStr: stripped from hashes and public dumps.
SECRET_FIELDS: tuple[tuple[str, str], ...] = _secret_fields()
# DSNs may embed a password (postgresql://user:pw@host/db, `password=` key-value); redacted.
DSN_FIELDS: tuple[tuple[str, str], ...] = (("provenance", "dsn"), ("ducklake", "catalog_dsn"))

_KV_PASSWORD_RE = re.compile(r"(?i)\bpassword\s*=\s*('(?:[^'\\]|\\.)*'|[^&\s']+)")


def redact_dsn(dsn: str) -> str:
    """``dsn`` with any password replaced by ``***`` (URL userinfo, ``?password=`` or libpq
    ``password=`` key-value form), for hashes and log lines."""
    parts = urlsplit(dsn)
    if "@" in parts.netloc:
        userinfo, host = parts.netloc.rsplit("@", 1)
        if ":" in userinfo:
            user = userinfo.split(":", 1)[0]
            dsn = urlunsplit(parts._replace(netloc=f"{user}:***@{host}"))
    return _KV_PASSWORD_RE.sub("password=***", dsn)


def public_config(cfg: Config) -> dict[str, Any]:
    """``cfg`` as JSON-ready data with every secret removed and every DSN password redacted:
    what ``config show``, the doctor and :func:`config_sha256` may expose."""
    public = cfg.model_dump(mode="json")
    for section, field in SECRET_FIELDS:
        public[section][field] = None
    for section, field in DSN_FIELDS:
        if public[section][field]:
            public[section][field] = redact_dsn(public[section][field])
    return public


def config_sha256(cfg: Config) -> str:
    """A stable hash of the effective configuration with secrets removed (stored on runs)."""
    payload = json.dumps(public_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- paths ------------------------------------------------------------------------------------


def expand_path(value: str | Path) -> Path:
    """``~`` expanded; the config keeps paths as written so hashes do not depend on the home."""
    return Path(value).expanduser()


def expand_dsn(dsn: str) -> str:
    """``duckdb:///~/...`` with the home expanded; other DSNs unchanged."""
    if dsn.startswith("duckdb:///~"):
        return "duckdb:///" + str(Path(dsn[len("duckdb:///") :]).expanduser())
    return dsn


def duckdb_path(dsn: str) -> Path | None:
    """The file behind a ``duckdb:///<path>`` DSN, home expanded; ``None`` for any other dialect
    (a Postgres DSN), so callers can tell the two apart without parsing the DSN themselves."""
    prefix = "duckdb:///"
    if not dsn.startswith(prefix):
        return None
    return Path(dsn[len(prefix) :]).expanduser()


# -- the process-wide configuration --------------------------------------------------------------

_active: Config | None = None


def set_active_config(cfg: Config | None) -> None:
    global _active
    _active = cfg


def get_active_config() -> Config:
    global _active
    if _active is None:
        _active = load_config()
    return _active
