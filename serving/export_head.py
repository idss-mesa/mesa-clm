#!/usr/bin/env python3
"""Export a CLM head checkpoint (``.pt``) to mesa-clm's numpy ``.npz`` layout (plan §6.2).

Runs in the serve venv (it needs torch); the torch-free core reads the result with
``mesa_clm.clm.headproj.HeadProjector.from_npz``, whose module docstring is the contract this
file writes (format ``mesa-clm-head-npz/1``). The checkpoint is a ``torch.save`` dict
``{state_head, action_head, logit_scale, cfg, ...}`` and is loaded with
``torch.load(..., weights_only=True)`` (patch 0003's rule). The configuration is resolved exactly
as CLM's ``HeadPair._load`` does (``heads.py`` at bb42c6c5): ``width``, ``depth`` from ``cfg``;
``projection_dim`` from the top level, else ``cfg``, else 512; ``activation`` (gelu),
``layernorm`` (false), ``residual`` (false) and ``hidden_size`` (4096) from ``cfg`` with those
defaults.

``make_head`` names its parameters ``inp`` (``hidden -> width``), ``hidden.<j>`` (``width ->
width``, ``j`` in ``0..depth-3``), ``norms.<j>`` (LayerNorm after ``hidden.<j>``, only with
``layernorm``; ``nn.Identity`` has no parameters) and ``out`` (``width -> projection_dim``). The
export maps them onto the linear index ``i`` of the contract::

    inp.{weight,bias}        -> <side>.linear.0.{weight,bias}
    hidden.<j>.{weight,bias} -> <side>.linear.<j+1>.{weight,bias}
    norms.<j>.{weight,bias}  -> <side>.norm.<j+1>.{weight,bias}
    out.{weight,bias}        -> <side>.linear.<depth-1>.{weight,bias}

with ``<side>`` ``state`` (``state_head``) or ``action`` (``action_head``), float32 in torch's
``[out, in]`` layout, plus ``format``, ``cfg`` (JSON of exactly ``width, depth, activation,
layernorm, residual, projection_dim, hidden_size``), ``logit_scale`` (float32 scalar, the
checkpoint's raw value; the clamp ``min(exp, 100)`` is applied by the reader) and
``source_sha256`` (sha256 of the ``.pt``). A state dict with a missing or unexpected key, or a
shape that disagrees with the configuration, is refused, as ``load_state_dict(strict=True)``
would.

After writing, the export is checked: every array is compared with the checkpoint, and a numpy
forward pass over the exported arrays (float64, exact-erf GELU, torch's LayerNorm) is compared
with CLM's own ``make_head`` (or an identical local build when ``clm`` is not importable) on
seeded random unit vectors; the L2-normalised outputs must agree within ``--tolerance``
(default 1e-5, the plan's headproj parity bound).

Usage::

    python serving/export_head.py ~/.mesa/clm/heads/CLM_v0.1-8B.pt \
        --out ~/.mesa/clm/heads/npz/b2b4a8c9.npz --expect-sha256 b2b4a8c9...

Without ``--out`` the file is ``<out-dir>/<sha8>.npz`` (``--out-dir`` default
``~/.mesa/clm/heads/npz``). An existing export of the same checkpoint is verified and kept
(exit 0); an existing file from another checkpoint is refused unless ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

NPZ_FORMAT: Final[str] = "mesa-clm-head-npz/1"
DEFAULT_OUT_DIR: Final[str] = "~/.mesa/clm/heads/npz"
PROJ_DIM: Final[int] = 512
HIDDEN: Final[int] = 4096
ACTIVATIONS: Final[frozenset[str]] = frozenset({"gelu", "relu"})
SIDES: Final[tuple[tuple[str, str], ...]] = (("state", "state_head"), ("action", "action_head"))
LAYER_NORM_EPS: Final[float] = 1e-5
F64 = npt.NDArray[np.float64]


class ExportError(RuntimeError):
    """The checkpoint or an existing export cannot be used; the message says why."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_cfg(ck: Mapping[str, Any]) -> dict[str, Any]:
    """The seven forward-pass fields, resolved as ``HeadPair._load`` resolves them."""
    raw = ck.get("cfg")
    if not isinstance(raw, Mapping):
        raise ExportError("checkpoint has no 'cfg' mapping")
    cfg = dict(raw)
    try:
        width, depth = int(cfg["width"]), int(cfg["depth"])
    except KeyError as exc:
        raise ExportError(f"checkpoint cfg lacks {exc.args[0]!r}") from None
    activation = str(cfg.get("activation", "gelu"))
    proj = int(ck.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)))
    hidden = int(cfg.get("hidden_size", HIDDEN))
    if activation not in ACTIVATIONS:
        raise ExportError(
            f"activation {activation!r} is not supported by mesa-clm's headproj "
            f"(one of {sorted(ACTIVATIONS)})"
        )
    if depth < 2 or min(width, proj, hidden) < 1:
        raise ExportError(f"implausible head shape: width {width}, depth {depth}, proj {proj}")
    return {
        "width": width,
        "depth": depth,
        "activation": activation,
        "layernorm": bool(cfg.get("layernorm", False)),
        "residual": bool(cfg.get("residual", False)),
        "projection_dim": proj,
        "hidden_size": hidden,
    }


def expected_params(cfg: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """``make_head``'s state-dict keys and shapes for ``cfg``."""
    w, d, p, h = cfg["width"], cfg["depth"], cfg["projection_dim"], cfg["hidden_size"]
    shapes: dict[str, tuple[int, ...]] = {"inp.weight": (w, h), "inp.bias": (w,)}
    for j in range(d - 2):
        shapes[f"hidden.{j}.weight"] = (w, w)
        shapes[f"hidden.{j}.bias"] = (w,)
        if cfg["layernorm"]:
            shapes[f"norms.{j}.weight"] = (w,)
            shapes[f"norms.{j}.bias"] = (w,)
    shapes["out.weight"] = (p, w)
    shapes["out.bias"] = (p,)
    return shapes


def contract_key(side: str, torch_key: str, depth: int) -> str:
    """Map one ``make_head`` parameter name onto the npz contract (module docstring)."""
    module, _, leaf = torch_key.rpartition(".")
    if module == "inp":
        return f"{side}.linear.0.{leaf}"
    if module == "out":
        return f"{side}.linear.{depth - 1}.{leaf}"
    kind, _, j = module.partition(".")
    if kind == "hidden":
        return f"{side}.linear.{int(j) + 1}.{leaf}"
    if kind == "norms":
        return f"{side}.norm.{int(j) + 1}.{leaf}"
    raise ExportError(f"unexpected parameter {torch_key!r}")


def to_numpy(value: Any) -> npt.NDArray[np.float32]:
    """A tensor (or array-like) as a float32 numpy array on the host."""
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").float().numpy()
    return np.asarray(value, dtype=np.float32)


def build_arrays(
    ck: Mapping[str, Any], source_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(arrays, cfg)``: the contract's arrays for a loaded checkpoint."""
    cfg = resolve_cfg(ck)
    shapes = expected_params(cfg)
    arrays: dict[str, Any] = {
        "format": np.array(NPZ_FORMAT),
        "cfg": np.array(json.dumps(cfg, sort_keys=True)),
        "source_sha256": np.array(source_sha256),
    }
    scale = to_numpy(ck.get("logit_scale"))
    if scale.size != 1:
        raise ExportError(f"logit_scale must hold one value, got shape {scale.shape}")
    arrays["logit_scale"] = np.array(scale.reshape(()), dtype=np.float32)
    for side, key in SIDES:
        sd = ck.get(key)
        if not isinstance(sd, Mapping):
            raise ExportError(f"checkpoint has no {key!r} state dict")
        present, wanted = set(sd), set(shapes)
        if present != wanted:
            raise ExportError(
                f"{key}: keys differ from make_head(cfg): missing {sorted(wanted - present)}, "
                f"unexpected {sorted(present - wanted)}"
            )
        for tkey, shape in shapes.items():
            arr = to_numpy(sd[tkey])
            if arr.shape != shape:
                raise ExportError(f"{key}.{tkey}: shape {arr.shape}, expected {shape}")
            if not np.all(np.isfinite(arr)):
                raise ExportError(f"{key}.{tkey}: non-finite values")
            arrays[contract_key(side, tkey, cfg["depth"])] = arr
    return arrays, cfg


# -- verification ------------------------------------------------------------------------------


def _gelu(x: F64) -> F64:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return np.asarray(0.5 * x * (1.0 + erf(x / math.sqrt(2.0))), dtype=np.float64)


def numpy_forward(arrays: Mapping[str, Any], side: str, x: F64) -> F64:
    """The contract's forward pass (what ``headproj.Mlp`` computes), then L2."""
    cfg = json.loads(str(arrays["cfg"]))
    act = _gelu if cfg["activation"] == "gelu" else (lambda z: np.maximum(z, 0.0))

    def lin(i: int, h: F64) -> F64:
        w = np.asarray(arrays[f"{side}.linear.{i}.weight"], np.float64)
        b = np.asarray(arrays[f"{side}.linear.{i}.bias"], np.float64)
        return np.asarray(h @ w.T + b, dtype=np.float64)

    h = act(lin(0, x))
    for i in range(1, cfg["depth"] - 1):
        z = lin(i, h)
        if cfg["layernorm"]:
            nw = np.asarray(arrays[f"{side}.norm.{i}.weight"], np.float64)
            nb = np.asarray(arrays[f"{side}.norm.{i}.bias"], np.float64)
            mean = z.mean(axis=-1, keepdims=True)
            var = z.var(axis=-1, keepdims=True)
            z = (z - mean) / np.sqrt(var + LAYER_NORM_EPS) * nw + nb
        z = act(z)
        h = h + z if cfg["residual"] else z
    out = lin(cfg["depth"] - 1, h)
    return np.asarray(out / np.linalg.norm(out, axis=-1, keepdims=True), dtype=np.float64)


def _make_head(cfg: Mapping[str, Any]) -> Any:
    """CLM's ``make_head`` when importable, else an identical local build (bb42c6c5)."""
    kw = {
        "width": cfg["width"],
        "depth": cfg["depth"],
        "proj": cfg["projection_dim"],
        "activation": cfg["activation"],
        "layernorm": cfg["layernorm"],
        "residual": cfg["residual"],
        "hidden": cfg["hidden_size"],
    }
    try:
        from clm.heads import make_head
    except ImportError:
        return _local_make_head(**kw)
    return make_head(**kw)


def _local_make_head(
    width: int, depth: int, proj: int, activation: str, layernorm: bool, residual: bool, hidden: int
) -> Any:
    import torch.nn as nn

    act = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]

    class Head(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x: Any) -> Any:
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms, strict=True):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


def verify(
    arrays: Mapping[str, Any], ck: Mapping[str, Any], *, tolerance: float, n: int = 8
) -> float:
    """Max |numpy - torch| over both heads' L2-normalised outputs on seeded unit vectors."""
    import torch

    cfg = json.loads(str(arrays["cfg"]))
    rng = np.random.default_rng(20260929)
    x = rng.standard_normal((n, cfg["hidden_size"]))
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    worst = 0.0
    for side, key in SIDES:
        head = _make_head(cfg)
        head.load_state_dict(ck[key], strict=True)
        head.eval()
        with torch.no_grad():
            ref = torch.nn.functional.normalize(head(torch.from_numpy(x).float()), dim=-1)
        diff = float(np.max(np.abs(numpy_forward(arrays, side, x) - ref.double().numpy())))
        worst = max(worst, diff)
    scale_np = min(math.exp(float(arrays["logit_scale"])), 100.0)
    scale_torch = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
    if abs(scale_np - scale_torch) > 1e-4 * max(1.0, scale_torch):
        raise ExportError(f"scale {scale_np} differs from HeadPair's {scale_torch}")
    if not worst <= tolerance:
        raise ExportError(f"numpy and torch projections differ by {worst:.3g} > {tolerance:g}")
    return worst


def check_existing(path: Path, arrays: Mapping[str, Any]) -> bool:
    """True when ``path`` already holds exactly ``arrays`` (same keys, equal values)."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            if set(npz.files) != set(arrays):
                return False
            return all(
                np.array_equal(npz[k], np.asarray(v)) and npz[k].dtype == np.asarray(v).dtype
                for k, v in arrays.items()
            )
    except (OSError, ValueError):
        return False


def write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Uncompressed ``np.savez`` to a temp file in the target directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a CLM head checkpoint to the mesa-clm npz layout (headproj contract)."
    )
    ap.add_argument("ckpt", type=Path, help="the .pt checkpoint (torch.save dict)")
    ap.add_argument(
        "--out", type=Path, default=None, help="output file (default: <out-dir>/<sha8>.npz)"
    )
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    ap.add_argument("--expect-sha256", default=None, help="refuse unless the .pt has this sha256")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    ap.add_argument("--force", action="store_true", help="replace an export of another checkpoint")
    ap.add_argument(
        "--no-verify", action="store_true", help="skip the numpy vs torch forward check"
    )
    args = ap.parse_args(argv)

    ckpt = args.ckpt.expanduser()
    if not ckpt.is_file():
        print(f"export_head: {ckpt}: not a file", file=sys.stderr)
        return 1
    source = sha256_file(ckpt)
    if args.expect_sha256 and source != args.expect_sha256:
        print(
            f"export_head: {ckpt}: sha256 {source}, expected {args.expect_sha256}", file=sys.stderr
        )
        return 1
    out = (args.out or args.out_dir / f"{source[:8]}.npz").expanduser()

    import torch

    try:
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        if not isinstance(ck, Mapping):
            raise ExportError("the checkpoint is not a dict")
        arrays, cfg = build_arrays(ck, source)
        if out.exists():
            if check_existing(out, arrays):
                print(
                    json.dumps({"npz": str(out), "source_sha256": source, "status": "up to date"})
                )
                return 0
            if not args.force:
                raise ExportError(f"{out} exists with other content; pass --force to replace it")
        worst = None if args.no_verify else verify(arrays, ck, tolerance=args.tolerance)
        write_npz(out, arrays)
        if not check_existing(out, arrays):
            raise ExportError(f"{out}: read-back differs from what was written")
    except ExportError as exc:
        print(f"export_head: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "npz": str(out),
                "source_sha256": source,
                "cfg": cfg,
                "max_abs_diff": worst,
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
