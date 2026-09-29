"""Serve-side tests (plan §9, serving CI): the patched CLM and the fallback, CPU only, no torch.

They need fastapi, httpx, numpy and requests, and the patched CLM clone's ``src`` directory in
``MESA_CLM_CLM_SRC`` (CI clones Contrastive-LM/CLM at the pinned commit and applies
serving/patches). Tests that need either are skipped when it is missing, so the core venv
(which has neither) collects this directory harmlessly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING))
_clm_src = os.environ.get("MESA_CLM_CLM_SRC")
if _clm_src:
    sys.path.insert(0, str(Path(_clm_src).expanduser()))
