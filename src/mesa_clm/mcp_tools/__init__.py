"""``mesa_clm_*``: mesa-clm's tools inside mesa-mcp's registry (DESIGN D14, D32; plan §7.1).

Milestone M0 ships this module as a stub so that the ``mesa_mcp.tools`` entry point ``clm``
(``pyproject.toml``) imports cleanly under ``mesa_mcp.server.load_plugins()``: the loader reports
``clm: loaded`` and the registry gains nothing. The five tools arrive in milestone M3 with the
service and the apply path:

* ``mesa_clm_annotate``: the decide phase on a dataset card (no writes);
* ``mesa_clm_apply``: the write phase, ``dry_run`` by default, one elicitation per open group;
* ``mesa_clm_explain``, ``mesa_clm_feedback`` (a plain tool pick is ``agent_pick``, D21);
* ``mesa_clm_health``: :func:`mesa_clm.health.doctor` with ``quick=True``.

``_register()`` will then call ``mesa_mcp.server.register_tool(name, description,
input_model=, output_model=, meta={"io.mesa/surface": TOOL_SURFACE})`` for each, as mesa-anyjev's
``mcp_tools`` does for ``mesa_decide_*``. Until then importing this module has no side effect.
"""

from __future__ import annotations

from typing import Final

#: The surface family published in every tool's ``_meta`` (``io.mesa/surface``; U1).
TOOL_SURFACE: Final = "clm"
#: The registered tool names; empty until M3 registers the five ``mesa_clm_*`` tools.
TOOL_NAMES: Final[tuple[str, ...]] = ()


def _register() -> None:
    """Register the ``mesa_clm_*`` tools with mesa-mcp (M3). In M0 this registers nothing."""
    return None


_register()

__all__ = ["TOOL_NAMES", "TOOL_SURFACE"]
