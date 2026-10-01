"""The numpy head projection (``mesa_clm.clm.headproj``): a tiny random head generated in-test
is compared against an independent pure-Python/numpy reference of CLM's ``make_head`` forward
pass (Linear, exact-erf GELU, LayerNorm, optional residual, L2), the npz exporter contract round
trips and is validated, and the logit scale is clamped at 100 like ``heads.py``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mesa_clm.clm.headproj import (
    LAYER_NORM_EPS,
    MAX_SCALE,
    NPZ_FORMAT,
    RAW_SCALE,
    HeadConfig,
    HeadError,
    HeadProjector,
    gelu,
    l2,
    layer_norm,
    random_head,
)

HIDDEN = 96  # a small "4096" keeps the reference loop fast


def reference_forward(head: HeadProjector, side: str, x: np.ndarray) -> np.ndarray:
    """Element-wise reference: Python loops over rows, ``math.erf`` GELU, hand-written LayerNorm.
    Shares no code path with ``Mlp.__call__`` beyond the stored weights."""
    cfg = head.cfg
    mlp = head.state if side == "state" else head.action

    def act(v: list[float]) -> list[float]:
        if cfg.activation == "gelu":
            return [0.5 * t * (1.0 + math.erf(t / math.sqrt(2.0))) for t in v]
        return [max(t, 0.0) for t in v]

    def linear(v: list[float], w: np.ndarray, b: np.ndarray) -> list[float]:
        return [float(sum(w[o, i] * v[i] for i in range(len(v))) + b[o]) for o in range(w.shape[0])]

    def ln(v: list[float], w: np.ndarray, b: np.ndarray) -> list[float]:
        mean = sum(v) / len(v)
        var = sum((t - mean) ** 2 for t in v) / len(v)
        return [
            (t - mean) / math.sqrt(var + LAYER_NORM_EPS) * float(w[i]) + float(b[i])
            for i, t in enumerate(v)
        ]

    rows = []
    for row in x:
        h = act(linear([float(t) for t in row], *mlp.linears[0]))
        for i in range(1, cfg.depth - 1):
            z = linear(h, *mlp.linears[i])
            if i in mlp.norms:
                z = ln(z, *mlp.norms[i])
            z = act(z)
            h = [a + b for a, b in zip(h, z, strict=True)] if cfg.residual else z
        out = linear(h, *mlp.linears[-1])
        norm = math.sqrt(sum(t * t for t in out))
        rows.append([t / max(norm, 1e-12) for t in out])
    return np.array(rows, dtype=np.float64)


@pytest.mark.parametrize(
    ("depth", "layernorm", "residual", "activation"),
    [
        (2, False, False, "gelu"),
        (3, True, False, "gelu"),
        (4, True, True, "gelu"),
        (3, False, False, "relu"),
    ],
)
def test_projection_matches_reference(
    depth: int, layernorm: bool, residual: bool, activation: Any
) -> None:
    head = random_head(
        7,
        width=12,
        depth=depth,
        layernorm=layernorm,
        residual=residual,
        activation=activation,
        projection_dim=8,
        hidden_size=HIDDEN,
    )
    x = np.random.default_rng(3).standard_normal((5, HIDDEN)).astype(np.float32)
    for side, project in (("state", head.project_states), ("action", head.project_actions)):
        got = project(x)
        assert got.dtype == np.float32 and got.shape == (5, 8)
        np.testing.assert_allclose(got, reference_forward(head, side, x), atol=1e-5, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-6)
    # The two heads are different networks.
    assert not np.allclose(head.project_states(x), head.project_actions(x))


def test_released_head_shape_is_expressible() -> None:
    cfg = HeadConfig(width=1536, depth=3, layernorm=True)
    assert cfg.linear_shapes() == [(1536, 4096), (1536, 1536), (512, 1536)]
    assert list(cfg.norm_indices()) == [1]
    params = sum(o * i + o for o, i in cfg.linear_shapes()) + 2 * 1536
    assert params == 9_443_840  # RESEARCH.md: 18,887,680 for both heads (PR #10)
    assert list(HeadConfig(width=8, depth=2).norm_indices()) == []
    assert list(HeadConfig(width=8, depth=5, layernorm=True).norm_indices()) == [1, 2, 3]


def test_primitives_against_torch_definitions() -> None:
    x = np.array([[-3.0, -1.0, 0.0, 0.5, 2.0]])
    # torch.nn.functional.gelu reference values (exact erf form).
    np.testing.assert_allclose(
        gelu(x), [[-0.00404969, -0.15865525, 0.0, 0.34573123, 1.95449974]], atol=1e-7
    )
    w = np.array([1.0, 2.0, 0.5, 1.0, 1.0])
    b = np.array([0.0, 0.1, 0.0, -0.1, 0.0])
    out = layer_norm(x, w, b)
    mean, var = x.mean(), x.var()
    np.testing.assert_allclose(out, (x - mean) / np.sqrt(var + 1e-5) * w + b, atol=1e-12)
    np.testing.assert_allclose(l2(np.array([[3.0, 4.0]])), [[0.6, 0.8]], atol=1e-7)
    assert l2(np.zeros((1, 3))).tolist() == [[0.0, 0.0, 0.0]]  # floored norm, no NaN
    assert l2(np.ones((2, 3), dtype=np.float32)).dtype == np.float32


def test_scale_is_clamped_and_logits_follow_engine_maths() -> None:
    head = random_head(
        1, width=8, depth=2, projection_dim=4, hidden_size=HIDDEN, logit_scale=4.6132
    )
    assert (
        math.exp(4.6132) > 100.0 and head.scale == MAX_SCALE == RAW_SCALE
    )  # the released head (#15)
    soft = random_head(
        1, width=8, depth=2, projection_dim=4, hidden_size=HIDDEN, logit_scale=math.log(20.0)
    )
    assert soft.scale == pytest.approx(20.0)
    x = np.random.default_rng(0).standard_normal((2, HIDDEN))
    zs, za = head.project_states(x[:1]), head.project_actions(x)
    logits = head.logits(zs, za)
    np.testing.assert_allclose(
        logits, 100.0 * (zs.astype(np.float64) @ za.astype(np.float64).T), atol=1e-9
    )
    np.testing.assert_allclose(head.logits(zs, za, temperature=2.0), logits / 2.0)
    assert logits.shape == (1, 2) and np.all(np.abs(logits) <= 100.0 + 1e-9)
    with pytest.raises(ValueError, match="temperature"):
        head.logits(zs, za, temperature=0.0)


def test_npz_round_trip_and_contract(tmp_path: Path) -> None:
    head = random_head(11, width=16, depth=3, layernorm=True, projection_dim=8, hidden_size=HIDDEN)
    head.source_sha256 = "f" * 64
    path = head.to_npz(tmp_path / "head.npz")
    with np.load(path) as npz:
        keys = set(npz.files)
        assert str(npz["format"]) == NPZ_FORMAT
        cfg = json.loads(str(npz["cfg"]))
        assert cfg == {
            "width": 16,
            "depth": 3,
            "activation": "gelu",
            "layernorm": True,
            "residual": False,
            "projection_dim": 8,
            "hidden_size": HIDDEN,
        }
        assert npz["logit_scale"].dtype == np.float32 and npz["state.linear.0.weight"].shape == (
            16,
            HIDDEN,
        )
        assert npz["action.linear.2.weight"].shape == (8, 16) and npz[
            "state.norm.1.weight"
        ].shape == (16,)
    assert keys == HeadProjector.expected_keys(head.cfg)
    assert keys == {
        "format", "cfg", "logit_scale", "source_sha256",
        *(f"{s}.linear.{i}.{p}" for s in ("state", "action") for i in range(3) for p in ("weight", "bias")),
        *(f"{s}.norm.1.{p}" for s in ("state", "action") for p in ("weight", "bias")),
    }  # fmt: skip
    loaded = HeadProjector.from_npz(path)
    assert loaded.cfg == head.cfg and loaded.source_sha256 == "f" * 64
    assert loaded.scale == pytest.approx(head.scale, rel=1e-6)
    x = np.random.default_rng(5).standard_normal((4, HIDDEN)).astype(np.float32)
    np.testing.assert_allclose(loaded.project_states(x), head.project_states(x), atol=1e-6)
    np.testing.assert_allclose(loaded.project_actions(x), head.project_actions(x), atol=1e-6)


def test_loader_refuses_bad_exports(tmp_path: Path) -> None:
    head = random_head(2, width=8, depth=3, layernorm=True, projection_dim=4, hidden_size=HIDDEN)
    arrays = head.to_arrays()
    with pytest.raises(HeadError, match="format"):
        HeadProjector.from_arrays({**arrays, "format": np.array("other/1")})
    missing = dict(arrays)
    del missing["action.norm.1.bias"]
    with pytest.raises(HeadError, match="missing"):
        HeadProjector.from_arrays(missing)
    with pytest.raises(HeadError, match="unexpected"):
        HeadProjector.from_arrays({**arrays, "state.linear.9.weight": np.zeros((1, 1))})
    with pytest.raises(HeadError, match="shape"):
        HeadProjector.from_arrays(
            {**arrays, "state.linear.0.weight": np.zeros((8, HIDDEN + 1), np.float32)}
        )
    with pytest.raises(HeadError, match="float"):
        HeadProjector.from_arrays({**arrays, "state.linear.0.bias": np.zeros(8, dtype=np.int32)})
    with pytest.raises(HeadError, match="cfg"):
        HeadProjector.from_arrays({**arrays, "cfg": np.array('{"width": 8}')})
    with pytest.raises(HeadError, match="cfg"):
        HeadProjector.from_arrays({**arrays, "cfg": np.array("{not json")})
    with pytest.raises(HeadError, match="scalar"):
        HeadProjector.from_arrays({**arrays, "logit_scale": np.zeros(2, np.float32)})
    no_cfg = dict(arrays)
    del no_cfg["cfg"]
    with pytest.raises(HeadError, match="cfg"):
        HeadProjector.from_arrays(no_cfg)
    with pytest.raises(HeadError, match="not found"):
        HeadProjector.from_npz(tmp_path / "nope.npz")
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not an npz")
    with pytest.raises(HeadError, match="npz"):
        HeadProjector.from_npz(junk)
    with pytest.raises(HeadError, match="expected input"):
        head.project_states(np.zeros((2, HIDDEN + 1)))
    with pytest.raises(ValueError):
        HeadConfig(width=8, depth=1)
    with pytest.raises(ValueError):
        HeadConfig(width=8, depth=2, activation="silu")  # type: ignore[arg-type]


def test_random_head_is_deterministic_per_seed() -> None:
    a = random_head(3, width=8, depth=2, projection_dim=4, hidden_size=HIDDEN)
    b = random_head(3, width=8, depth=2, projection_dim=4, hidden_size=HIDDEN)
    c = random_head(4, width=8, depth=2, projection_dim=4, hidden_size=HIDDEN)
    x = np.ones((1, HIDDEN))
    np.testing.assert_array_equal(a.project_states(x), b.project_states(x))
    assert not np.allclose(a.project_states(x), c.project_states(x))
