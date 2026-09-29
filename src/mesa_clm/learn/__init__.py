"""Learning: labels (M0), features and tier fitters (M1, M4), the finetune driver (M7).

Nothing here imports torch; fitting runs on numpy over cached CLM features, and the head tier
is a subprocess into the serve venv (plan §5.3).
"""

from __future__ import annotations
