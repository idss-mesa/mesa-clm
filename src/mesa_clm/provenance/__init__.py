"""The ``mesa_clm`` provenance sidecar (DESIGN D11).

M0 ships the ``labels`` table (:mod:`mesa_clm.provenance.labels`); the run, decision, link,
override and audit tables and the migrations land with the pipeline in M1, and their DDL
includes ``LABELS_DDL`` verbatim so one schema version covers both.
"""

from __future__ import annotations
