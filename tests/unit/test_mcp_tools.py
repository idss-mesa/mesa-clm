"""The ``mesa_mcp.tools`` entry point ``clm`` (``mesa_clm.mcp_tools``): an M0 stub that imports
cleanly under ``load_plugins`` and registers nothing until the five ``mesa_clm_*`` tools land in
M3 (DESIGN D14)."""

from __future__ import annotations

from mesa_mcp.server import get_registered_tools, load_plugins

import mesa_clm.mcp_tools as tools


def _clm_tools() -> list[str]:
    return sorted(t.name for t in get_registered_tools() if t.name.startswith("mesa_clm_"))


def test_stub_registers_nothing() -> None:
    before = sorted(t.name for t in get_registered_tools())
    tools._register()
    assert sorted(t.name for t in get_registered_tools()) == before
    assert _clm_tools() == []
    assert tools.TOOL_SURFACE == "clm" and tools.TOOL_NAMES == ()
    assert tools.__all__ == ["TOOL_NAMES", "TOOL_SURFACE"]


def test_entry_point_loads_under_load_plugins() -> None:
    """mesa-mcp's loader imports ``clm = mesa_clm.mcp_tools`` and reports it loaded; strict mode
    would raise on an import error, and the registry still has no ``mesa_clm_*`` tool."""
    loaded = load_plugins(strict=True)
    assert loaded.get("clm") == "loaded"
    assert _clm_tools() == []


def test_docstring_names_the_m3_tools() -> None:
    doc = tools.__doc__ or ""
    for name in (
        "mesa_clm_annotate",
        "mesa_clm_apply",
        "mesa_clm_explain",
        "mesa_clm_feedback",
        "mesa_clm_health",
    ):
        assert name in doc
    assert "M3" in doc and "register_tool" in doc
