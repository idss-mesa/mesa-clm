"""The ``mesa_clm`` provenance sidecar (DESIGN D11).

M0 ships the ``labels`` table (:mod:`mesa_clm.provenance.labels`); the run, decision, link,
override and audit tables and the migrations land with the pipeline in M1, and their DDL
includes ``LABELS_DDL`` verbatim so one schema version covers both.
"""

from __future__ import annotations

from mesa_clm.provenance.labels import LABELS_DDL, LabelRow, LabelStore, lock_path_for
from mesa_clm.provenance.models import (
    AuditRow,
    AvuLinkRow,
    ClmCallRow,
    DecisionGroupRow,
    DecisionOptionRow,
    DecisionRow,
    HumanOverrideRow,
    RunRow,
)
from mesa_clm.provenance.store import (
    DUCKDB_DDL,
    SCHEMA_VERSION,
    TABLES,
    DuckDBStore,
    ProvenanceStore,
    RunBuffer,
    open_store,
)

__all__ = [
    "DUCKDB_DDL",
    "LABELS_DDL",
    "SCHEMA_VERSION",
    "TABLES",
    "AuditRow",
    "AvuLinkRow",
    "ClmCallRow",
    "DecisionGroupRow",
    "DecisionOptionRow",
    "DecisionRow",
    "DuckDBStore",
    "HumanOverrideRow",
    "LabelRow",
    "LabelStore",
    "ProvenanceStore",
    "RunBuffer",
    "RunRow",
    "lock_path_for",
    "open_store",
]
