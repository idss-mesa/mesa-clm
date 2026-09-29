"""mesa_clm.serving: keys, units, the serving lock and its live checks (hermetic; plan §6)."""

from __future__ import annotations

import grp
import hashlib
import json
import logging
import os
import pwd
import re
import shlex
import shutil
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from mesa_clm import render, serving
from mesa_clm.clm.fingerprint import LockError, load_serving_lock, sign_lock_body
from mesa_clm.secrets import SecretError

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "serving" / "serving.lock.json"
PATCHES = ROOT / "serving" / "patches"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            out[key] = value
    return out


def _keys(result: serving.KeysResult) -> tuple[str, str]:
    return (
        result.clm_key.read_text(encoding="utf-8").strip(),
        result.encoder_key.read_text(encoding="utf-8").strip(),
    )


# -- keys ----------------------------------------------------------------------------------------


def test_init_keys_writes_private_files(tmp_path: Path) -> None:
    result = serving.init_keys(tmp_path / "secrets")
    assert _mode(tmp_path / "secrets") == 0o700
    for path in (result.clm_key, result.encoder_key, result.clm_env, result.encoder_env):
        assert _mode(path) == 0o600, path
    assert result.created == ("clm.key", "encoder.key")
    assert result.rotated is False
    assert not list((tmp_path / "secrets").glob(".*.tmp"))


def test_env_files_are_derived_from_the_keys(tmp_path: Path) -> None:
    result = serving.init_keys(tmp_path)
    clm_key, encoder_key = _keys(result)
    assert clm_key != encoder_key
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", clm_key)
    assert _env(result.clm_env) == {
        "CLM_API_KEY": clm_key,
        "CLM_EMB_API_KEY": encoder_key,
        "CLM_EMB_CACHE_SIZE": "20000",
    }
    assert _env(result.encoder_env) == {"VLLM_API_KEY": encoder_key}


def test_init_keys_is_idempotent_and_repairs_env_files(tmp_path: Path) -> None:
    first = serving.init_keys(tmp_path)
    keys = _keys(first)
    first.clm_env.write_text("CLM_API_KEY=stale\n", encoding="utf-8")
    second = serving.init_keys(tmp_path)
    assert second.created == ()
    assert _keys(second) == keys
    assert _env(second.clm_env)["CLM_API_KEY"] == keys[0]


def test_rotate_replaces_both_keys(tmp_path: Path) -> None:
    before = _keys(serving.init_keys(tmp_path))
    result = serving.init_keys(tmp_path, rotate=True)
    after = _keys(result)
    assert result.rotated is True
    assert result.created == ("clm.key", "encoder.key")
    assert after[0] != before[0] and after[1] != before[1]
    assert _env(result.clm_env)["CLM_EMB_API_KEY"] == after[1]
    assert _env(result.encoder_env)["VLLM_API_KEY"] == after[1]


def test_partial_state_creates_only_the_missing_key(tmp_path: Path) -> None:
    serving.init_keys(tmp_path)
    kept = (tmp_path / "clm.key").read_text(encoding="utf-8")
    (tmp_path / "encoder.key").unlink()
    result = serving.init_keys(tmp_path)
    assert result.created == ("encoder.key",)
    assert (tmp_path / "clm.key").read_text(encoding="utf-8") == kept


def test_no_key_text_in_results_or_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    results = [serving.init_keys(tmp_path), serving.init_keys(tmp_path, rotate=True)]
    results.append(serving.init_keys(tmp_path))
    old_and_new: set[str] = set()
    for r in results:
        old_and_new.update(_keys(r))
    texts = (
        [caplog.text]
        + [repr(r) for r in results]
        + [str(v) for r in results for v in vars(r).values()]
    )
    for key in old_and_new:
        digest = hashlib.sha256(key.encode()).hexdigest()
        for text in texts:
            assert key not in text
            assert digest[:8] not in text


def test_loose_secrets_dir_is_tightened(tmp_path: Path) -> None:
    directory = tmp_path / "secrets"
    directory.mkdir()
    directory.chmod(0o755)
    serving.init_keys(directory)
    assert _mode(directory) == 0o700


def test_a_symlinked_secrets_dir_is_refused(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir(mode=0o700)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(serving.ServingError, match="real directory"):
        serving.init_keys(tmp_path / "link")


def test_a_loose_key_file_is_refused_until_rotated(tmp_path: Path) -> None:
    serving.init_keys(tmp_path)
    key = tmp_path / "clm.key"
    secret = key.read_text(encoding="utf-8").strip()
    key.chmod(0o644)
    with pytest.raises(SecretError) as exc:
        serving.init_keys(tmp_path)
    assert secret not in str(exc.value)
    serving.init_keys(tmp_path, rotate=True)
    assert _mode(key) == 0o600


def test_an_env_unsafe_key_is_refused_without_echoing_it(tmp_path: Path) -> None:
    serving.init_keys(tmp_path)
    bad = 'has space and "quotes" 0123456789'
    (tmp_path / "encoder.key").write_text(bad + "\n", encoding="utf-8")
    (tmp_path / "encoder.key").chmod(0o600)
    with pytest.raises(serving.ServingError, match="--rotate") as exc:
        serving.init_keys(tmp_path)
    assert bad not in str(exc.value)


# -- units ---------------------------------------------------------------------------------------


def _joined(text: str) -> list[str]:
    """Unit lines with backslash continuations joined."""
    return text.replace("\\\n", " ").splitlines()


def _directive(text: str, name: str) -> list[str]:
    return [line.split("=", 1)[1] for line in _joined(text) if line.startswith(f"{name}=")]


def test_render_units_keeps_percent_h() -> None:
    units = serving.render_units()
    assert tuple(units) == serving.UNIT_NAMES
    for name, text in units.items():
        assert name.endswith(".timer") or "%h/.mesa/clm/" in text, name
        assert str(Path.home()) not in text, name


def test_committed_units_equal_the_rendering() -> None:
    for name, text in serving.render_units().items():
        assert (ROOT / "deploy" / "systemd" / name).read_text(encoding="utf-8") == text, name


def test_encoder_unit_matches_the_plan() -> None:
    text = serving.render_units()["mesa-clm-encoder.service"]
    assert _directive(text, "ExecStartPre") == [
        '-/usr/bin/sg docker -c "docker rm -f mesa-clm-encoder"',
        "%h/.mesa/clm/bin/mesa-clm-check-headroom 32",
    ]
    assert _directive(text, "ExecStart") == [
        '/usr/bin/sg docker -c "exec %h/.mesa/clm/bin/mesa-clm-encoder-run"'
    ]
    assert _directive(text, "ExecStop") == [
        '/usr/bin/sg docker -c "docker stop -t 30 mesa-clm-encoder"'
    ]
    assert _directive(text, "Restart") == ["on-failure"]
    assert _directive(text, "TimeoutStartSec") == ["900"]


def test_serve_unit_matches_the_plan() -> None:
    text = serving.render_units()["mesa-clm-serve.service"]
    assert _directive(text, "Requires") == ["mesa-clm-encoder.service"]
    assert _directive(text, "After") == ["mesa-clm-encoder.service"]
    assert _directive(text, "EnvironmentFile") == ["%h/.mesa/clm/secrets/clm.env"]
    assert _directive(text, "ExecStartPre") == [
        "%h/.mesa/clm/bin/mesa-clm-wait-http http://127.0.0.1:8090/health 600"
    ]
    (start,) = _directive(text, "ExecStart")
    assert shlex.split(start) == [
        "%h/.mesa/clm/serve/.venv/bin/clm-serve",
        *("--host", "127.0.0.1", "--port", "8700"),
        *("--emb-url", "http://127.0.0.1:8090/v1/embeddings", "--emb-model", "qwen3-8b"),
        *("--max-tokens", "4096", "--ckpt", "%h/.mesa/clm/heads/CLM_v0.1-8B.pt"),
        *("--ckpt-dir", "%h/.mesa/clm/heads/served", "--device", "cpu"),
        *("--action-cache", "512MiB", "--no-download", "--no-ui"),
    ]


def test_headroom_timer_runs_every_five_minutes() -> None:
    units = serving.render_units()
    assert _directive(units["mesa-clm-headroom.service"], "ExecStart") == [
        "%h/.mesa/clm/bin/mesa-clm-check-headroom 8"
    ]
    timer = units["mesa-clm-headroom.timer"]
    assert _directive(timer, "OnUnitActiveSec") == ["5min"]
    assert _directive(timer, "Unit") == ["mesa-clm-headroom.service"]


def test_install_sections_only_name_targets() -> None:
    for name, text in serving.render_units().items():
        install = text.split("[Install]", 1)[1] if "[Install]" in text else ""
        assert all(line.startswith("WantedBy=") for line in install.split() if "=" in line), name


@pytest.mark.parametrize(
    ("home", "spelled"),
    [
        ("~/.mesa/clm", "%h/.mesa/clm"),
        ("%h/.mesa/clm/", "%h/.mesa/clm"),
        ("/opt/mesa-clm", "/opt/mesa-clm"),
    ],
)
def test_unit_home(home: str, spelled: str) -> None:
    assert serving.unit_home(home) == spelled


def test_unit_home_under_the_home_directory() -> None:
    assert serving.unit_home(Path.home() / "x") == "%h/x"
    assert serving.unit_home(Path.home()) == "%h"


@pytest.mark.parametrize("home", ["relative/dir", "/opt/with space"])
def test_unit_home_refuses(home: str) -> None:
    with pytest.raises(serving.ServingError):
        serving.unit_home(home)


def test_custom_home_is_rendered() -> None:
    text = serving.render_units("/srv/clm")["mesa-clm-serve.service"]
    assert "EnvironmentFile=/srv/clm/secrets/clm.env" in text


# -- the lock ------------------------------------------------------------------------------------


def test_committed_lock_verifies_and_equals_the_build() -> None:
    lock = load_serving_lock(LOCK)
    assert LOCK.read_text(encoding="utf-8") == serving.render_lock(PATCHES)
    assert lock.clm_commit == serving.CLM_COMMIT
    assert lock.schema_sha256 == render.schema_sha256()
    assert lock.encoder == serving.ENCODER
    assert lock.encoder.route == "vllm" and lock.encoder.prefix_caching is False
    assert lock.image.tag == "v0.27.1"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", lock.image.digest)


def test_lock_patches() -> None:
    lock = load_serving_lock(LOCK)
    assert [p.id[:4] for p in lock.patches] == ["0001", "0002", "0003", "0004", "0005", "0006"]
    assert [p.pr for p in lock.patches] == [6, 11, 10, 23, None, None]
    for patch in lock.patches:
        path = PATCHES / f"{patch.id}.patch"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == patch.sha256
        if patch.pr is None:
            assert patch.pr_head_sha is None
        else:
            assert patch.pr_head_sha is not None and len(patch.pr_head_sha) == 40
    assert sorted(p.name for p in PATCHES.glob("*.patch")) == [
        f"{p.id}.patch" for p in lock.patches
    ]


def test_no_patch_touches_the_vendored_files() -> None:
    for path in PATCHES.glob("*.patch"):
        touched = re.findall(r"^diff --git a/(\S+) b/", path.read_text(encoding="utf-8"), re.M)
        assert touched, path
        assert not {"src/clm/schema.py", "src/clm/client.py"} & set(touched), path


def _encoder_run_args() -> tuple[str, list[str]]:
    script = (ROOT / "deploy" / "bin" / "mesa-clm-encoder-run").read_text(encoding="utf-8")
    joined = script.replace("\\\n", " ")
    (line,) = [ln for ln in joined.splitlines() if ln.startswith("exec docker run")]
    argv = shlex.split(line)
    (i,) = [k for k, a in enumerate(argv) if a.startswith("vllm/vllm-openai@")]
    return argv[i], argv[i + 1 :]


def test_encoder_run_matches_the_lock() -> None:
    lock = load_serving_lock(LOCK)
    image, args = _encoder_run_args()
    assert image == f"{lock.image.ref}@{lock.image.digest}"
    assert serving.recipe_mismatches(lock.encoder, args) == []
    assert args[args.index("--served-model-name") + 1] == "qwen3-8b"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.20"


def test_recipe_mismatches_names_the_differences() -> None:
    _, args = _encoder_run_args()
    changed = [a for a in args if a != "--no-enable-prefix-caching"]
    changed[changed.index("--max-model-len") + 1] = "2048"
    wrong = serving.recipe_mismatches(serving.ENCODER, changed)
    assert any("--max-model-len" in w for w in wrong)
    assert any("prefix caching" in w for w in wrong)
    assert serving.recipe_mismatches(serving.ENCODER, []) != []


# -- live checks with fakes ----------------------------------------------------------------------

COMMIT = serving.CLM_COMMIT
IMAGE_ID = "sha256:" + "ab" * 32
TAG_ENV = json.dumps(["PATH=/usr/bin", "VLLM_IMAGE_TAG=vllm/vllm-openai:v0.27.1"])


class FakeHost:
    """A scripted runner: git, the venv's python and docker answer from a table."""

    def __init__(self, schema: Path | None = None) -> None:
        self.docker_up = True
        self.image_present = True
        self.image_env = TAG_ENV
        self.container: tuple[str, list[str]] | None = None
        self.schema = schema
        self.commit = COMMIT
        self.diff_rc = 0
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str]) -> serving.CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "git":
            if "diff" in argv:
                return serving.CommandResult(self.diff_rc, "")
            return serving.CommandResult(0, self.commit + "\n")
        if argv[0].endswith("python"):
            return serving.CommandResult(0 if self.schema else 1, f"{self.schema or ''}\n")
        assert argv[0] == "docker"
        if argv[1] == "version":
            return serving.CommandResult(0 if self.docker_up else 1, "29.6.2\n")
        if argv[1:3] == ["image", "inspect"]:
            if not self.image_present:
                return serving.CommandResult(1, "")
            return serving.CommandResult(0, f"{IMAGE_ID}\t{self.image_env}\n")
        if argv[1] == "inspect":
            if self.container is None:
                return serving.CommandResult(1, "")
            image, cmd = self.container
            return serving.CommandResult(0, f"{image}\t{json.dumps(cmd)}\n")
        raise AssertionError(argv)


def _docker(args: Sequence[str]) -> list[str]:
    return ["docker", *args]


def _statuses(checks: list[serving.LockCheck]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in checks:
        out.setdefault(c.name, []).append(c.status)
    return out


def _check(tmp_path: Path, host: FakeHost, **kw: Any) -> list[serving.LockCheck]:
    return serving.check_serving_lock(
        kw.pop("lock", LOCK),
        home=tmp_path / "home",
        repo=kw.pop("repo", ROOT),
        runner=host.run,
        docker=kw.pop("docker", _docker),
        **kw,
    )


def test_nothing_installed_is_skipped_not_failed(tmp_path: Path) -> None:
    host = FakeHost()
    host.docker_up = False
    checks = _check(tmp_path, host)
    assert [c for c in checks if c.status == "fail"] == []
    assert _statuses(checks)["lock"] == ["ok"]
    assert _statuses(checks)["patch files"] == ["ok"]
    assert {c.name for c in checks if c.status == "skip"} >= {
        "installed lock",
        "serve clone",
        "venv schema",
        "head",
        "image",
    }


def test_require_live_turns_absence_into_problems(tmp_path: Path) -> None:
    host = FakeHost()
    host.docker_up = False
    problems = serving.verify_serving_lock(
        LOCK, home=tmp_path / "home", repo=ROOT, require_live=True, runner=host.run, docker=_docker
    )
    names = {p.split(":", 1)[0] for p in problems}
    assert names == {"installed lock", "serve clone", "venv schema", "head", "image"}


def test_docker_unreachable_is_a_skip(tmp_path: Path) -> None:
    checks = _check(tmp_path, FakeHost(), docker=lambda args: None)
    assert _statuses(checks)["image"] == ["skip"]


def _signed_lock(tmp_path: Path, head: Path) -> Path:
    """A lock like the committed one whose head pin is a small local file."""
    body = json.loads(LOCK.read_text(encoding="utf-8"))
    data = head.read_bytes()
    body["head"].update(file=head.name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    path = tmp_path / "serving.lock.json"
    path.write_text(json.dumps(sign_lock_body(body)), encoding="utf-8")
    return path


def _install(tmp_path: Path) -> tuple[Path, Path, FakeHost]:
    """A fake serving home: installed lock, clone with the vendored schema, stamp, venv, head."""
    home = tmp_path / "home"
    heads = home / "heads"
    heads.mkdir(parents=True)
    head = heads / "head.pt"
    head.write_bytes(b"not really a head")
    lock_path = _signed_lock(tmp_path, head)
    lock = load_serving_lock(lock_path)
    (home / "serving.lock.json").write_text(lock_path.read_text(encoding="utf-8"), encoding="utf-8")
    clone = home / "serve" / "CLM"
    (clone / ".git").mkdir(parents=True)
    (clone / "src" / "clm").mkdir(parents=True)
    schema = clone / "src" / "clm" / "schema.py"
    vendored = ROOT / "src" / "mesa_clm" / "_vendor" / "clm" / "schema.py"
    schema.write_bytes(vendored.read_bytes())
    stamp = {
        "clm_commit": lock.clm_commit,
        "tree": "0" * 40,
        "patches": [{"id": p.id, "sha256": p.sha256} for p in lock.patches],
    }
    (home / "serve" / "patches.applied.json").write_text(json.dumps(stamp), encoding="utf-8")
    python = home / "serve" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    return lock_path, home, FakeHost(schema=schema)


def test_a_complete_host_is_all_ok(tmp_path: Path) -> None:
    lock_path, _, host = _install(tmp_path)
    _, args = _encoder_run_args()
    host.container = (IMAGE_ID, args)
    checks = _check(tmp_path, host, lock=lock_path)
    assert {c.name: c.status for c in checks if c.status != "ok"} == {}
    assert _statuses(checks)["encoder container"] == ["ok"]


def test_mismatches_are_problems(tmp_path: Path) -> None:
    lock_path, home, host = _install(tmp_path)
    (home / "heads" / "head.pt").write_bytes(b"not really a head, but longer")
    host.commit = "f" * 40
    host.image_env = json.dumps(["VLLM_IMAGE_TAG=vllm/vllm-openai:v0.30.0"])
    host.container = ("sha256:" + "cd" * 32, [])
    stamp = home / "serve" / "patches.applied.json"
    doc = json.loads(stamp.read_text(encoding="utf-8"))
    doc["patches"] = doc["patches"][:-1]
    stamp.write_text(json.dumps(doc), encoding="utf-8")
    problems = serving.verify_serving_lock(
        lock_path, home=home, repo=ROOT, runner=host.run, docker=_docker
    )
    joined = "\n".join(problems)
    assert "serve clone: " in joined and "HEAD ffffffffffff" in joined
    assert "applied patches: " in joined
    assert "head: " in joined and "bytes" in joined
    assert "VLLM_IMAGE_TAG vllm/vllm-openai:v0.30.0" in joined
    assert "encoder container: " in joined and "not the pinned digest" in joined


def test_a_clone_edited_after_bootstrap_is_a_problem(tmp_path: Path) -> None:
    lock_path, home, host = _install(tmp_path)
    host.diff_rc = 1
    problems = serving.verify_serving_lock(
        lock_path, home=home, repo=ROOT, runner=host.run, docker=lambda args: None
    )
    assert any(p.startswith("applied patches: ") and "edited after" in p for p in problems)
    (diff,) = [c for c in host.calls if "diff" in c]
    assert diff[:2] == ["git", "--no-optional-locks"] and diff[-2:] == ["0" * 40, "--"]


def test_a_container_with_another_recipe_is_a_problem(tmp_path: Path) -> None:
    lock_path, home, host = _install(tmp_path)
    _, args = _encoder_run_args()
    host.container = (IMAGE_ID, [a for a in args if a != "--no-enable-prefix-caching"])
    problems = serving.verify_serving_lock(
        lock_path, home=home, repo=ROOT, runner=host.run, docker=_docker
    )
    assert any(p.startswith("encoder container: ") and "prefix caching" in p for p in problems)


def test_schema_and_installed_lock_drift(tmp_path: Path) -> None:
    lock_path, home, host = _install(tmp_path)
    (home / "serve" / "CLM" / "src" / "clm" / "schema.py").write_text(
        "# edited\n", encoding="utf-8"
    )
    (home / "serving.lock.json").write_text(LOCK.read_text(encoding="utf-8"), encoding="utf-8")
    problems = serving.verify_serving_lock(
        lock_path, home=home, repo=ROOT, runner=host.run, docker=_docker
    )
    names = {p.split(":", 1)[0] for p in problems}
    assert {"clone schema", "venv schema", "installed lock"} <= names


def test_patch_file_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "serving" / "patches").mkdir(parents=True)
    for path in PATCHES.glob("*.patch"):
        (repo / "serving" / "patches" / path.name).write_bytes(path.read_bytes())
    first = sorted((repo / "serving" / "patches").glob("*.patch"))[0]
    first.write_bytes(first.read_bytes() + b"\n")
    (repo / "serving" / "patches" / "0007-extra.patch").write_text("", encoding="utf-8")
    host = FakeHost()
    host.docker_up = False
    checks = _check(tmp_path, host, repo=repo)
    (patch_check,) = [c for c in checks if c.name == "patch files"]
    assert patch_check.status == "fail"
    assert "sha256 differs" in patch_check.detail and "0007-extra.patch" in patch_check.detail


def test_a_tampered_lock_stops_the_checks(tmp_path: Path) -> None:
    body = json.loads(LOCK.read_text(encoding="utf-8"))
    body["encoder"]["max_len"] = 2048
    path = tmp_path / "serving.lock.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    checks = _check(tmp_path, FakeHost(), lock=path)
    assert [(c.name, c.status) for c in checks] == [("lock", "fail")]
    with pytest.raises(LockError):
        serving.load_lock(path)


def test_image_absent_is_a_skip_unless_required(tmp_path: Path) -> None:
    host = FakeHost()
    host.image_present = False
    assert _statuses(_check(tmp_path, host))["image"] == ["skip"]
    assert _statuses(_check(tmp_path, host, require_live=True))["image"] == ["fail"]


# -- docker_argv and run_command -----------------------------------------------------------------


class _Group:
    gr_gid = 988
    gr_mem = ["someone"]


def test_docker_argv_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(grp, "getgrnam", lambda name: _Group())
    monkeypatch.setattr(os, "getgroups", lambda: [988])
    assert serving.docker_argv(["version"]) == ["docker", "version"]
    monkeypatch.setattr(os, "getgroups", lambda: [1000])

    class _Pw:
        pw_name = "someone"

    monkeypatch.setattr(pwd, "getpwuid", lambda uid: _Pw())
    argv = serving.docker_argv(["inspect", "x", "--format", "{{.Image}}\t{{json .Config.Cmd}}"])
    assert argv is not None and argv[:3] == ["sg", "docker", "-c"]
    assert shlex.split(argv[3]) == [
        "docker",
        "inspect",
        "x",
        "--format",
        "{{.Image}}\t{{json .Config.Cmd}}",
    ]
    _Pw.pw_name = "stranger"
    assert serving.docker_argv(["version"]) is None
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert serving.docker_argv(["version"]) is None


def test_run_command() -> None:
    assert serving.run_command(["mesa-clm-no-such-program"]).returncode == 127
    res = serving.run_command([sys.executable, "-c", "print('hello')"])
    assert res == serving.CommandResult(0, "hello\n")


def test_default_lock_path_is_the_checkout() -> None:
    assert serving.default_lock_path() == LOCK
    assert serving.repo_root() == ROOT
    assert serving.load_lock().lock_sha == load_serving_lock(LOCK).lock_sha


def test_serving_home_default() -> None:
    assert serving.serving_home() == Path(os.path.expanduser("~/.mesa/clm"))
