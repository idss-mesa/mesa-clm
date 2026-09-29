"""Loader semantics ported from mesa-anyjev's test_config.py plus the mesa-clm additions:
CLM_* fallbacks, MESA_HOME lock default, verbatim string values, SecretStr keys, the
secret-free hash, and the two example files staying in step with the model."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, SecretStr, ValidationError

from mesa_clm.config import (
    DSN_FIELDS,
    ENV_PREFIX,
    SECRET_FIELDS,
    Config,
    config_sha256,
    expand_dsn,
    expand_path,
    get_active_config,
    load_config,
    public_config,
    redact_dsn,
    set_active_config,
)

REPO = Path(__file__).resolve().parents[2]


def test_defaults() -> None:
    cfg = load_config(env={})
    assert cfg.clm.base_url == "http://127.0.0.1:8700"
    assert cfg.clm.model == "clm-latest" and cfg.clm.allow_remote is False
    assert cfg.clm.api_key is None and cfg.clm.api_key_file is None
    assert cfg.encoder.url == "http://127.0.0.1:8090"
    assert cfg.encoder.model == "qwen3-8b" and cfg.encoder.max_len == 4096
    assert cfg.decider.tier == "auto"
    assert cfg.planner.kind == "static"
    assert cfg.planner.gateway_base_url == "http://127.0.0.1:8000"
    assert cfg.planner.gateway_model == "carc-tools"
    assert cfg.claude.effort == "low" and cfg.claude.second_opinion is False
    assert cfg.policy.profile == "prod" and cfg.policy.policy_path is None
    assert (cfg.policy.max_candidates, cfg.policy.max_avus, cfg.policy.proposal_size) == (12, 25, 8)
    assert cfg.policy.max_wait_s == 30.0
    assert cfg.policy.specificity is True and cfg.policy.specificity_delta == 0.10
    assert cfg.artifacts.dir == "~/.mesa/clm/artifacts" and cfg.artifacts.strict is True
    assert cfg.features.dir == "~/.mesa/clm/features"
    assert cfg.provenance.dsn == "duckdb:///~/.mesa/clm/provenance.duckdb"
    assert cfg.provenance.ttl_days == 30
    assert cfg.ducklake.cache_dir == "~/.mesa/clm/ducklake-cache"
    assert cfg.ducklake.cache_cap_bytes == 256 * 1024 * 1024
    assert cfg.history.backend == "auto" and cfg.history.multiwriter is False
    assert cfg.history.lock_path == "~/.mesa/locks/mesa-catalog.lock"
    assert cfg.history.neon_lock_path is None
    assert cfg.history.spool_root == "~/.mesa/clm/spool"
    assert cfg.apply.max_calls_per_s == 5.0
    assert cfg.ols.base_url == "https://www.ebi.ac.uk/ols4/api/v2" and cfg.ols.fixtures == "off"
    assert cfg.neon.root is None
    assert cfg.card.max_bytes == 1024 * 1024
    assert cfg.eval_root is None
    assert cfg.secrets == "auto"
    assert cfg.vm_id and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", cfg.vm_id)


def test_env_precedence_and_mesa_mcp_fallbacks(tmp_path: Path) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text(
        "planner:\n  kind: claude\n  timeout: 30\nunknown_section:\n  x: 1\n", encoding="utf-8"
    )
    env = {
        "MESA_CLM_PLANNER__TIMEOUT": "45",
        "MESA_LLM_BASE_URL": "http://127.0.0.1:18000/v1",
        "MESA_LLM_API_KEY": "sk-test",
        "MESA_CLM_POLICY__PROFILE": "dev",
    }
    cfg = load_config(yaml_file, env=env, flag_overrides={"planner": {"timeout": 90}})
    assert cfg.planner.kind == "claude"
    assert cfg.planner.timeout == 90  # flag > env > yaml
    assert cfg.planner.gateway_base_url == "http://127.0.0.1:18000"  # /v1 stripped
    assert cfg.planner.resolved_gateway_api_key() == "sk-test"
    assert cfg.policy.profile == "dev"


def test_native_name_wins_over_fallback() -> None:
    cfg = load_config(
        env={
            "MESA_CLM_PLANNER__GATEWAY_BASE_URL": "http://127.0.0.1:8000",
            "MESA_LLM_BASE_URL": "http://127.0.0.1:18000/v1",
            "MESA_CLM_CLM__BASE_URL": "http://127.0.0.1:8701",
            "CLM_BASE_URL": "http://127.0.0.1:9999",
        }
    )
    assert cfg.planner.gateway_base_url == "http://127.0.0.1:8000"
    assert cfg.clm.base_url == "http://127.0.0.1:8701"


def test_clm_client_fallbacks() -> None:
    cfg = load_config(env={"CLM_BASE_URL": "http://127.0.0.1:8700/", "CLM_API_KEY": "k1"})
    assert cfg.clm.base_url == "http://127.0.0.1:8700"
    assert cfg.clm.resolved_api_key() == "k1"
    # A configured key file (or native key) suppresses the foreign key name entirely.
    cfg = load_config(env={"CLM_API_KEY": "k1", "MESA_CLM_CLM__API_KEY_FILE": "/nonexistent"})
    assert cfg.clm.api_key is None and cfg.clm.api_key_file == "/nonexistent"
    cfg = load_config(env={"MESA_LLM_API_KEY": "g", "MESA_CLM_PLANNER__GATEWAY_API_KEY_FILE": "/f"})
    assert cfg.planner.gateway_api_key is None


def test_mesa_home_moves_the_lock_default_but_yields_to_yaml_and_env(tmp_path: Path) -> None:
    cfg = load_config(env={"MESA_HOME": "/opt/mesa/"})
    assert cfg.history.lock_path == "/opt/mesa/locks/mesa-catalog.lock"
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("history:\n  lock_path: /yaml/lock\n", encoding="utf-8")
    assert load_config(yaml_file, env={"MESA_HOME": "/opt/mesa"}).history.lock_path == "/yaml/lock"
    cfg = load_config(
        yaml_file, env={"MESA_HOME": "/opt/mesa", "MESA_CLM_HISTORY__LOCK_PATH": "/env/lock"}
    )
    assert cfg.history.lock_path == "/env/lock"
    assert cfg.history.lock_paths() == [Path("/env/lock")]
    cfg = load_config(env={"MESA_CLM_HISTORY__NEON_LOCK_PATH": "~/neon/work/locks/mesa-tag.lock"})
    assert cfg.history.lock_paths()[1] == Path("~/neon/work/locks/mesa-tag.lock").expanduser()


def test_config_sha_excludes_every_key_and_dsn_password() -> None:
    base = load_config(env={})
    for section, field in SECRET_FIELDS:
        a = load_config(env={f"MESA_CLM_{section}__{field}".upper(): "one"})
        b = load_config(env={f"MESA_CLM_{section}__{field}".upper(): "two"})
        assert config_sha256(a) == config_sha256(b) == config_sha256(base), (section, field)
    assert "one" not in config_sha256(base)
    with_pw = load_config(env={"MESA_CLM_PROVENANCE__DSN": "postgresql://u:hunter2@db/mesa"})
    without = load_config(env={"MESA_CLM_PROVENANCE__DSN": "postgresql://u:other@db/mesa"})
    assert config_sha256(with_pw) == config_sha256(without)
    assert "hunter2" not in str(public_config(with_pw))
    assert public_config(with_pw)["provenance"]["dsn"] == "postgresql://u:***@db/mesa"
    assert set(DSN_FIELDS) == {("provenance", "dsn"), ("ducklake", "catalog_dsn")}


def test_every_key_like_field_is_a_secret() -> None:
    """A future ``*_api_key``/``*password``/``*token`` field must be a SecretStr, or the hash and
    ``public_config`` would leak it."""
    assert set(SECRET_FIELDS) == {
        ("clm", "api_key"),
        ("encoder", "api_key"),
        ("planner", "gateway_api_key"),
    }
    for name, info in Config.model_fields.items():
        model = info.annotation
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            continue
        for fname in model.model_fields:
            if fname.endswith(("api_key", "password", "token", "secret")):
                assert (name, fname) in SECRET_FIELDS, (name, fname)


def test_secrets_never_show_in_repr_or_dump() -> None:
    cfg = load_config(env={"MESA_CLM_CLM__API_KEY": "sk-visible-nowhere"})
    assert isinstance(cfg.clm.api_key, SecretStr)
    assert "sk-visible-nowhere" not in repr(cfg)
    assert "sk-visible-nowhere" not in str(cfg.model_dump())
    assert "sk-visible-nowhere" not in cfg.model_dump_json()
    assert cfg.clm.resolved_api_key() == "sk-visible-nowhere"


def test_redact_dsn_forms() -> None:
    assert redact_dsn("postgresql://u:p%40ss@h:5432/db") == "postgresql://u:***@h:5432/db"
    assert redact_dsn("postgresql://h/db?password=x&sslmode=require") == (
        "postgresql://h/db?password=***&sslmode=require"
    )
    assert redact_dsn("host=h password=abc dbname=x") == "host=h password=*** dbname=x"
    assert redact_dsn("host=h password='a b' dbname=x") == "host=h password=*** dbname=x"
    assert redact_dsn("duckdb:///~/x.duckdb") == "duckdb:///~/x.duckdb"
    assert redact_dsn("postgresql://user@h/db") == "postgresql://user@h/db"


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Config.model_validate({"clm": {"bogus": 1}})
    with pytest.raises(ValidationError, match="extra"):
        Config.model_validate({"bogus": {"x": 1}})
    with pytest.raises(ValidationError, match="extra"):
        load_config(env={"MESA_CLM_CLM__BOGUS": "1"})
    with pytest.raises(ValidationError, match="extra"):
        load_config(env={"MESA_CLM_GATEWAY__KIND": "x"})  # no AnyJev backend sections


def test_gate_variables_are_not_settings() -> None:
    cfg = load_config(
        env={
            "MESA_CLM_ENGINE": "1",
            "MESA_CLM_LIVE": "1",
            "MESA_CLM_GPU": "1",
            "MESA_CLM_E2E": "1",
            "MESA_CLM_NEON_ROOT": "/neon",
            "MESA_CLM_TEST_PG_DSN": "postgresql://x",
            "MESA_CLM_EVAL_ROOT": "/x",
        }
    )
    assert cfg.eval_root == "/x"
    assert cfg.neon.root is None


def test_string_fields_take_env_values_verbatim() -> None:
    cfg = load_config(
        env={
            "MESA_CLM_OLS__FIXTURES": "off",  # YAML 1.1 would read False
            "MESA_CLM_CLM__API_KEY": "0123",  # YAML would read 83 (octal)
            "MESA_CLM_ENCODER__API_KEY": "yes",
            "MESA_CLM_VM_ID": "007",
            "MESA_CLM_CLM__MODEL": "1e10",
            "MESA_CLM_EVAL_ROOT": "",  # a nullable string: unset
            "MESA_CLM_NEON__ROOT": "null",
            "MESA_CLM_HISTORY__BACKEND": "none",
        }
    )
    assert cfg.ols.fixtures == "off"
    assert cfg.clm.resolved_api_key() == "0123"
    assert cfg.encoder.resolved_api_key() == "yes"
    assert cfg.vm_id == "007" and cfg.clm.model == "1e10"
    assert cfg.eval_root is None and cfg.neon.root is None
    assert cfg.history.backend == "none"


def test_non_string_fields_are_yaml_coerced() -> None:
    cfg = load_config(
        env={
            "MESA_CLM_CLM__ALLOW_REMOTE": "1",
            "MESA_CLM_HISTORY__MULTIWRITER": "true",
            "MESA_CLM_POLICY__SPECIFICITY": "no",
            "MESA_CLM_ENCODER__MAX_LEN": "8192",
            "MESA_CLM_APPLY__MAX_CALLS_PER_S": "2.5",
            "MESA_CLM_CARD__MAX_BYTES": "2097152",
        }
    )
    assert cfg.clm.allow_remote is True and cfg.history.multiwriter is True
    assert cfg.policy.specificity is False
    assert cfg.encoder.max_len == 8192 and cfg.apply.max_calls_per_s == 2.5
    assert cfg.card.max_bytes == 2 * 1024 * 1024
    with pytest.raises(ValidationError):
        load_config(env={"MESA_CLM_DUCKLAKE__CACHE_CAP_BYTES": "0"})  # 0 would mean unbounded
    with pytest.raises(ValidationError):
        load_config(env={"MESA_CLM_DECIDER__TIER": "L2"})  # AnyJev levels are gone


def test_whole_section_from_env_merges_with_field_entries() -> None:
    cfg = load_config(
        env={
            "MESA_CLM_CLM": "{model: clm-raw, timeout: 7}",
            "MESA_CLM_CLM__MODEL": "clm-latest",
        }
    )
    assert cfg.clm.model == "clm-latest" and cfg.clm.timeout == 7


def test_yaml_bare_off_means_the_word(tmp_path: Path) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("ols:\n  fixtures: off\n", encoding="utf-8")
    assert load_config(yaml_file, env={}).ols.fixtures == "off"


def test_yaml_must_be_a_mapping(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(yaml_file, env={})
    yaml_file.write_text("", encoding="utf-8")
    assert load_config(yaml_file, env={}) == load_config(env={})
    yaml_file.write_text("motherduck:\n  database: x\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="mesa_clm.config"):
        load_config(yaml_file, env={})
    assert "unknown section 'motherduck'" in caplog.text


def test_vm_id_is_one_path_segment() -> None:
    for bad in ("../x", "a/b", "", ".hidden", "x" * 65, "with space"):
        with pytest.raises(ValidationError, match="vm_id"):
            Config.model_validate({"vm_id": bad})
    assert Config.model_validate({"vm_id": "de-vm-01.cyverse.org"}).vm_id == "de-vm-01.cyverse.org"


def test_secrets_mode_reaches_the_sections(tmp_path: Path) -> None:
    key_file = tmp_path / "clm.key"
    key_file.write_text("from-file\n", encoding="utf-8")
    key_file.chmod(0o600)
    env = {"MESA_CLM_CLM__API_KEY": "from-env", "MESA_CLM_CLM__API_KEY_FILE": str(key_file)}
    assert load_config(env=env).clm.resolved_api_key() == "from-env"  # auto: value first
    assert (
        load_config(env={**env, "MESA_CLM_SECRETS": "file"}).clm.resolved_api_key() == "from-file"
    )
    assert load_config(env={**env, "MESA_CLM_SECRETS": "env"}).clm.resolved_api_key() == "from-env"
    cfg = load_config(env={**env, "MESA_CLM_SECRETS": "file"})
    assert cfg.clm.resolved_api_key("env") == "from-env"  # an explicit mode still wins
    assert cfg.encoder.resolved_api_key() is None
    assert cfg.planner.resolved_gateway_api_key() is None
    with pytest.raises(ValidationError):
        load_config(env={"MESA_CLM_SECRETS": "vault"})


def test_active_config_round_trip() -> None:
    set_active_config(None)
    try:
        cfg = load_config(env={"MESA_CLM_POLICY__PROFILE": "dev"})
        set_active_config(cfg)
        assert get_active_config() is cfg
    finally:
        set_active_config(None)


def test_expand_helpers() -> None:
    home = Path.home()
    assert expand_dsn("duckdb:///~/.mesa/clm/p.duckdb") == f"duckdb:///{home}/.mesa/clm/p.duckdb"
    assert expand_dsn("postgresql://h/db") == "postgresql://h/db"
    assert expand_dsn("duckdb:////abs/p.duckdb") == "duckdb:////abs/p.duckdb"
    assert expand_path("~/.mesa/clm") == home / ".mesa" / "clm"


def test_planner_config_carries_what_the_planners_read() -> None:
    cfg = load_config(env={"MESA_CLM_PLANNER__KIND": "gateway"})
    assert cfg.planner.kind == "gateway"
    assert (cfg.planner.claude_model, cfg.planner.claude_effort, cfg.planner.timeout) == (
        "claude-opus-5",
        "high",
        600.0,
    )
    with pytest.raises(ValidationError):
        load_config(env={"MESA_CLM_PLANNER__KIND": "hf"})


# -- the example files stay in step with the model ---------------------------------------------


def _env_names() -> set[str]:
    names: set[str] = set()
    for section, info in Config.model_fields.items():
        model = info.annotation
        if isinstance(model, type) and issubclass(model, BaseModel):
            names.update(f"{ENV_PREFIX}{section}__{field}".upper() for field in model.model_fields)
        else:
            names.add(f"{ENV_PREFIX}{section}".upper())
    return names


def test_env_example_documents_every_name() -> None:
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, flags=re.M))
    missing = _env_names() - documented
    assert not missing, sorted(missing)
    for fallback in (
        "CLM_BASE_URL",
        "CLM_API_KEY",
        "MESA_LLM_BASE_URL",
        "MESA_LLM_API_KEY",
        "MESA_HOME",
    ):
        assert fallback in documented, fallback
    # Names only: no line sets a value.
    assert not re.findall(r"^[A-Z][A-Z0-9_]+=.+$", text, flags=re.M)
    # Every documented MESA_CLM_ name is either a setting or a listed test gate.
    gates = {
        "MESA_CLM_LIVE", "MESA_CLM_ENGINE", "MESA_CLM_GPU", "MESA_CLM_E2E",
        "MESA_CLM_TEST_PG_DSN", "MESA_CLM_NEON_ROOT",
    }  # fmt: skip
    unknown = {n for n in documented if n.startswith(ENV_PREFIX)} - _env_names() - gates
    assert not unknown, sorted(unknown)


def test_config_yaml_example_is_the_defaults() -> None:
    example = REPO / "config.yaml.example"
    loaded = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    # Every section and field is present (vm_id is host-specific and documented as a comment).
    for section, info in Config.model_fields.items():
        model = info.annotation
        if isinstance(model, type) and issubclass(model, BaseModel):
            assert set(loaded[section]) == set(model.model_fields), section
        elif section != "vm_id":
            assert section in loaded, section
    assert "vm_id" in example.read_text(encoding="utf-8")
    cfg = load_config(example, env={})
    defaults = load_config(env={})
    assert cfg.model_dump() == defaults.model_dump()
    assert config_sha256(cfg) == config_sha256(defaults)


def test_load_config_reads_the_real_environment_when_env_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(os.environ):
        if name.startswith(ENV_PREFIX) or name in {"CLM_BASE_URL", "CLM_API_KEY", "MESA_HOME"}:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MESA_CLM_POLICY__PROFILE", "dev")
    assert load_config().policy.profile == "dev"
