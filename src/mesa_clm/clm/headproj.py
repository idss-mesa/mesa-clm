"""CLM's head MLP in numpy, over an exported ``.npz`` (plan §3 ``clm/headproj.py``, §4.1).

The probe tier scores locally: ``EncoderClient.embed`` gives the 4096-d encoder vectors,
this module projects them through the released (or a promoted) head exactly as CLM's ``HeadPair``
does, and ``learn/`` fits probes over the 512-d projections (``pair512.v1``) or the raw vectors.
Nothing here imports torch: the serve venv exports a head once (``serving/export_head.py``,
M1-A) and the core reads the arrays. The plan's acceptance is parity ``<= 1e-5`` against
``HeadPair.project_states`` / ``project_actions`` on the same inputs.

Architecture (CLM ``heads.py`` ``make_head(width, depth=2, proj=512, activation="gelu",
layernorm=False, residual=False, hidden=4096)``, RESEARCH.md): ``Linear(hidden -> width)`` +
activation, then ``(depth - 2)`` hidden blocks ``Linear(width -> width) -> LayerNorm? ->
activation``, optionally with a residual skip around the block, then ``Linear(width -> proj)``;
the output is L2-normalised. ``depth`` counts the Linear layers. The released CLM-v0.1-8B head
is ``4096 -> 1536 -> 1536 -> 512`` (depth 3, GELU, LayerNorm on the hidden block, no residual).
GELU is the exact erf form (torch's default); LayerNorm uses torch's default ``eps=1e-5`` with
affine weight and bias. The residual reading (``h = h + block(h)``) follows the docstring's
"act (±residual)" and is confirmed against ``HeadPair`` in M1-A: the released head has
``residual=False`` so nothing shipped depends on it.

Scale: CLM's logits are ``exp(logit_scale).clamp(max=100) * cos / temperature`` (``heads.py:95``,
``engine.py:132-136``); :attr:`HeadProjector.scale` applies the clamp.

Exporter contract (``heads/npz/<sha8>.npz``, written by ``serving/export_head.py`` with
``numpy.savez`` from the ``torch.save`` dict ``{state_head, action_head, logit_scale, cfg, ...}``):

==========================  ===============  ==========================================================
key                         dtype / shape    meaning
==========================  ===============  ==========================================================
``format``                  ``<U`` scalar    :data:`NPZ_FORMAT` (``mesa-clm-head-npz/1``)
``cfg``                     ``<U`` scalar    JSON of :class:`HeadConfig`: ``width, depth, activation,
                                             layernorm, residual, projection_dim, hidden_size``
``logit_scale``             float32 scalar   the checkpoint's ``logit_scale`` (log of 1/tau)
``source_sha256``           ``<U`` scalar    sha256 of the ``.pt`` the arrays came from ('' if unknown)
``<side>.linear.<i>.weight``  float32 [out, in]  torch layout (``y = x @ W.T + b``), ``i`` in ``0..depth-1``
``<side>.linear.<i>.bias``    float32 [out]
``<side>.norm.<i>.weight``    float32 [width]  only when ``layernorm``; ``i`` in ``1..depth-2``
``<side>.norm.<i>.bias``      float32 [width]
==========================  ===============  ==========================================================

``<side>`` is ``state`` or ``action``. Linear ``0`` maps ``hidden_size -> width``, linears
``1..depth-2`` ``width -> width``, linear ``depth-1`` ``width -> projection_dim``. The loader
refuses a missing or unexpected key and any shape that disagrees with ``cfg``
(:class:`HeadError`), so a truncated or mis-exported file cannot project silently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

NPZ_FORMAT: Final[str] = "mesa-clm-head-npz/1"
# CLM clamps exp(logit_scale) at 100 (heads.py:95); clm-raw uses RAW_SCALE = 100 outright.
MAX_SCALE: Final[float] = 100.0
RAW_SCALE: Final[float] = 100.0
LAYER_NORM_EPS: Final[float] = 1e-5
L2_EPS: Final[float] = 1e-12

Side = Literal["state", "action"]
SIDES: Final[tuple[Side, Side]] = ("state", "action")

FloatArray = npt.NDArray[np.float64]
# Inputs may arrive as float32 (encoder vectors) or float64; outputs are typed exactly.
AnyFloat = npt.NDArray[np.floating[Any]]
F32 = npt.NDArray[np.float32]


class HeadError(ValueError):
    """An exported head that cannot be loaded: bad format, missing key, wrong shape or config."""


class HeadConfig(BaseModel):
    """The ``cfg`` block of a CLM checkpoint, restricted to what the forward pass needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0)
    depth: int = Field(ge=2)
    activation: Literal["gelu", "relu"] = "gelu"
    layernorm: bool = False
    residual: bool = False
    projection_dim: int = Field(default=512, gt=0)
    hidden_size: int = Field(default=4096, gt=0)

    def linear_shapes(self) -> list[tuple[int, int]]:
        """``(out, in)`` of every Linear, index order (torch weight layout)."""
        shapes = [(self.width, self.hidden_size)]
        shapes += [(self.width, self.width)] * (self.depth - 2)
        shapes.append((self.projection_dim, self.width))
        return shapes

    def norm_indices(self) -> range:
        """Linear indices followed by a LayerNorm (the hidden blocks) when ``layernorm``."""
        return range(1, self.depth - 1) if self.layernorm else range(0)


# -- primitives ---------------------------------------------------------------------------------


def gelu(x: FloatArray) -> FloatArray:
    """Exact (erf) GELU, torch's default ``nn.GELU()``."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: FloatArray) -> FloatArray:
    # numpy has no vectorised erf outside scipy; math.erf through frompyfunc is exact and the
    # arrays here are small (n x width), so the per-element call is not a bottleneck.
    return np.asarray(np.frompyfunc(math.erf, 1, 1)(x), dtype=np.float64)


def relu(x: FloatArray) -> FloatArray:
    return np.maximum(x, 0.0)


def layer_norm(
    x: FloatArray, weight: FloatArray, bias: FloatArray, eps: float = LAYER_NORM_EPS
) -> FloatArray:
    """torch ``nn.LayerNorm`` over the last axis (biased variance, then affine)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return np.asarray((x - mean) / np.sqrt(var + eps) * weight + bias, dtype=np.float64)


def l2(x: AnyFloat, eps: float = L2_EPS) -> F32:
    """Row-wise L2 normalisation, as CLM ``embedder.py`` ``l2`` (norm floored at ``eps``)."""
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.asarray(arr / np.maximum(norms, eps), dtype=np.float32)


# -- the MLP ------------------------------------------------------------------------------------


class Mlp:
    """One head (state or action): the Linear/LayerNorm stack without the final L2."""

    def __init__(
        self,
        cfg: HeadConfig,
        linears: list[tuple[FloatArray, FloatArray]],
        norms: dict[int, tuple[FloatArray, FloatArray]],
    ) -> None:
        self.cfg = cfg
        self.linears = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in linears]
        self.norms = {
            i: (np.asarray(w, np.float64), np.asarray(b, np.float64)) for i, (w, b) in norms.items()
        }
        self._act = gelu if cfg.activation == "gelu" else relu

    def __call__(self, x: AnyFloat) -> FloatArray:
        h = np.asarray(x, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.cfg.hidden_size:
            raise HeadError(f"expected input [n, {self.cfg.hidden_size}], got {h.shape}")
        w, b = self.linears[0]
        h = self._act(np.asarray(h @ w.T + b, dtype=np.float64))
        for i in range(1, self.cfg.depth - 1):
            w, b = self.linears[i]
            z = np.asarray(h @ w.T + b, dtype=np.float64)
            if i in self.norms:
                nw, nb = self.norms[i]
                z = layer_norm(z, nw, nb)
            z = self._act(z)
            h = h + z if self.cfg.residual else z
        w, b = self.linears[-1]
        return np.asarray(h @ w.T + b, dtype=np.float64)


class HeadProjector:
    """The pair of heads plus the logit scale: what ``clm-latest`` (or a promoted head) applies.

    ``project_states`` / ``project_actions`` return L2-normalised float32 ``[n, projection_dim]``
    arrays; ``logits(zs, za, temperature)`` gives ``scale * zs @ za.T / temperature`` so that
    ``render.answer_from_logits`` over one row reproduces CLM's answer for that state.
    """

    def __init__(
        self, cfg: HeadConfig, state: Mlp, action: Mlp, logit_scale: float, source_sha256: str = ""
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.action = action
        self.logit_scale = float(logit_scale)
        self.source_sha256 = source_sha256

    @property
    def scale(self) -> float:
        """``min(exp(logit_scale), 100)``: CLM's clamp (``heads.py:95``)."""
        return min(math.exp(self.logit_scale), MAX_SCALE)

    def project_states(self, x: AnyFloat) -> F32:
        return l2(self.state(x))

    def project_actions(self, x: AnyFloat) -> F32:
        return l2(self.action(x))

    def logits(self, zs: AnyFloat, za: AnyFloat, temperature: float = 1.0) -> FloatArray:
        """``scale * cos / temperature`` for every (state row, action row) pair, float64."""
        if not 0.0 < temperature <= 100.0:
            raise ValueError("temperature must be in (0, 100]")
        cos = np.asarray(zs, np.float64) @ np.asarray(za, np.float64).T
        return self.scale * cos / temperature

    # -- npz ---------------------------------------------------------------------------------------

    @staticmethod
    def expected_keys(cfg: HeadConfig) -> set[str]:
        """Every array key an export of ``cfg`` must contain (scalars included)."""
        keys = {"format", "cfg", "logit_scale", "source_sha256"}
        for side in SIDES:
            for i in range(cfg.depth):
                keys |= {f"{side}.linear.{i}.weight", f"{side}.linear.{i}.bias"}
            for i in cfg.norm_indices():
                keys |= {f"{side}.norm.{i}.weight", f"{side}.norm.{i}.bias"}
        return keys

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any]) -> HeadProjector:
        """Build from the exporter's key/array mapping, checking format, keys and shapes."""
        fmt = str(arrays.get("format", ""))
        if fmt != NPZ_FORMAT:
            raise HeadError(f"unsupported head export format {fmt!r}; expected {NPZ_FORMAT!r}")
        try:
            cfg = HeadConfig.model_validate(json.loads(str(arrays["cfg"])))
        except KeyError:
            raise HeadError("head export has no 'cfg' entry") from None
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HeadError(f"head export 'cfg' is invalid: {exc}") from None
        expected = cls.expected_keys(cfg)
        present = set(arrays)
        if present != expected:
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            raise HeadError(
                f"head export keys differ from cfg: missing {missing}, unexpected {extra}"
            )
        heads: dict[str, Mlp] = {}
        for side in SIDES:
            linears: list[tuple[FloatArray, FloatArray]] = []
            for i, (out, inp) in enumerate(cfg.linear_shapes()):
                w = _array(arrays, f"{side}.linear.{i}.weight", (out, inp))
                b = _array(arrays, f"{side}.linear.{i}.bias", (out,))
                linears.append((w, b))
            norms: dict[int, tuple[FloatArray, FloatArray]] = {}
            for i in cfg.norm_indices():
                norms[i] = (
                    _array(arrays, f"{side}.norm.{i}.weight", (cfg.width,)),
                    _array(arrays, f"{side}.norm.{i}.bias", (cfg.width,)),
                )
            heads[side] = Mlp(cfg, linears, norms)
        logit_scale = np.asarray(arrays["logit_scale"], dtype=np.float64)
        if logit_scale.shape != ():
            raise HeadError(f"logit_scale must be a scalar, got shape {logit_scale.shape}")
        return cls(
            cfg,
            heads["state"],
            heads["action"],
            float(logit_scale),
            str(arrays.get("source_sha256", "")),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> HeadProjector:
        """Load an exported head (module docstring); the file is read once and closed."""
        p = Path(path).expanduser()
        try:
            with np.load(p, allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files}
        except FileNotFoundError:
            raise HeadError(f"{p}: head export not found") from None
        except (OSError, ValueError) as exc:
            raise HeadError(f"{p}: not a readable npz ({exc})") from None
        return cls.from_arrays(arrays)

    def to_arrays(self) -> dict[str, Any]:
        """The exporter layout of this head (the inverse of :meth:`from_arrays`)."""
        out: dict[str, Any] = {
            "format": np.array(NPZ_FORMAT),
            "cfg": np.array(json.dumps(self.cfg.model_dump(), sort_keys=True)),
            "logit_scale": np.array(self.logit_scale, dtype=np.float32),
            "source_sha256": np.array(self.source_sha256),
        }
        for side, mlp in (("state", self.state), ("action", self.action)):
            for i, (w, b) in enumerate(mlp.linears):
                out[f"{side}.linear.{i}.weight"] = w.astype(np.float32)
                out[f"{side}.linear.{i}.bias"] = b.astype(np.float32)
            for i, (w, b) in mlp.norms.items():
                out[f"{side}.norm.{i}.weight"] = w.astype(np.float32)
                out[f"{side}.norm.{i}.bias"] = b.astype(np.float32)
        return out

    def to_npz(self, path: str | Path) -> Path:
        """Write the exporter layout (uncompressed ``savez``); returns the path written."""
        p = Path(path).expanduser()
        with p.open("wb") as fh:
            np.savez(fh, **self.to_arrays())
        return p


def _array(arrays: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> FloatArray:
    arr = np.asarray(arrays[key])
    if arr.shape != shape:
        raise HeadError(f"{key}: expected shape {shape}, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise HeadError(f"{key}: expected a float array, got {arr.dtype}")
    return arr.astype(np.float64)


def random_head(
    seed: int = 0,
    *,
    width: int = 64,
    depth: int = 3,
    activation: Literal["gelu", "relu"] = "gelu",
    layernorm: bool = True,
    residual: bool = False,
    projection_dim: int = 512,
    hidden_size: int = 4096,
    logit_scale: float = math.log(MAX_SCALE),
) -> HeadProjector:
    """A small deterministic head (Kaiming-uniform-like weights) for tests and the fake CLM.

    Weights follow torch's ``nn.Linear`` default range ``U(-1/sqrt(in), 1/sqrt(in))``, LayerNorm
    affine parameters are perturbed around (1, 0) so a wrong LayerNorm implementation is caught,
    and the default ``logit_scale`` reproduces the released head's clamped scale of 100.
    """
    cfg = HeadConfig(
        width=width,
        depth=depth,
        activation=activation,
        layernorm=layernorm,
        residual=residual,
        projection_dim=projection_dim,
        hidden_size=hidden_size,
    )
    rng = np.random.default_rng(seed)
    mlps: dict[str, Mlp] = {}
    for side in SIDES:
        linears: list[tuple[FloatArray, FloatArray]] = []
        for out, inp in cfg.linear_shapes():
            bound = 1.0 / math.sqrt(inp)
            linears.append(
                (rng.uniform(-bound, bound, size=(out, inp)), rng.uniform(-bound, bound, size=out))
            )
        norms: dict[int, tuple[FloatArray, FloatArray]] = {
            i: (1.0 + 0.1 * rng.standard_normal(width), 0.1 * rng.standard_normal(width))
            for i in cfg.norm_indices()
        }
        mlps[side] = Mlp(cfg, linears, norms)
    return HeadProjector(cfg, mlps["state"], mlps["action"], logit_scale)
