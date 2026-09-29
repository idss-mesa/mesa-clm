"""Bench results: typed cells, the results file and its markdown table (DESIGN D5, D8, D27,
D30; plan §5.4).

A *cell* is one (task, tier, framing) measurement. Its fields are the plan §5.4 set: identity
(``question_key``, ``fingerprint``, ``labels_sha256``, ``feature_spec``, ``label_sources``,
``teacher``, ``teacher_in_test``, ``masked``, ``loco``, ``selection``, ``pre_registered``,
``exploratory``, ``servable``, ``n_folds``, ``skipped_folds``, ``fold_choices``), counts (``n``,
``n_neg``, ``n_nonmodal``, ``class_counts``), metrics (``acc``, ``macro_f1``, ``brier``, ``nll``,
``ece``, ``cov@5%``, ``cov@10%``, ``aurc``, ``auroc`` with its cluster CI, ``threshold_cp``), the
baselines block (``majority_acc``, ``lookup_acc``, ``lookup_nll``, ``novel_key``, ``lopo``,
``beats_lookup_novel``) and diagnostics. A baseline cell has no framing, fingerprint or feature
spec (``None``) and is never ``servable``; the M2+ runner fills those for model tiers.

Every results file is stamped with the ``labels_sha256`` of the frozen snapshot it was computed
from (D30), the ``mesa_clm`` version and a secret-free environment record, and lives at
``bench/results/<date>/<name>.json`` beside its ``.md`` table, so a policy ``cite`` of the form
``bench/results/<date>/<name>.json#<task>.<tier>.<framing>`` resolves to ``cells[<key>]``.

``labels_content_sha256`` is a second, content-only hash of the label rows (identity, label,
weight, card, product and fold flags, in identity order): two ingestions of the same fixtures
produce different ``labels_sha256`` values (``label_id`` and ``created_at`` differ, which is
what D30 wants from a snapshot hash) but the same ``labels_content_sha256``, so a test can
prove it re-derived the committed cell from the same rows.

Every model forbids unknown keys. ``cov@5%`` and ``cov@10%`` keep their AnyJev names in the JSON
through field aliases (``cov_at_5``, ``cov_at_10`` in Python).
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Final, Literal

import duckdb
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mesa_clm import __version__
from mesa_clm.provenance.labels import IDENTITY_COLUMNS, TABLE, LabelStore

FORMAT: Final = "mesa-clm/bench-results/1"
DEFAULT_OUT_DIR: Final = "bench/results"

Tier = Literal["baseline", "zero_shot", "calibrated", "probe", "head"]
Selection = Literal["none", "nested", "full"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ClusterCI(_Model):
    """A cluster-bootstrap interval (:class:`mesa_clm.bench.stats.BootstrapCI`)."""

    point: float | None
    lower: float | None
    upper: float | None
    B: int
    seed: int
    alpha: float
    n_clusters: int
    n_valid: int


class NovelKey(_Model):
    """The held-out items whose lookup key no training card carried (plan §5.4): the subset on
    which a model has to know something the labels of other cards do not say."""

    n: int
    class_counts: dict[int, int]
    majority_acc: float | None
    acc: float | None
    auroc: float | None
    auroc_ci: ClusterCI | None
    nll: float | None
    brier: float | None


class Lopo(_Model):
    """The leave-one-product-out control: how much of the lookup survives when the sibling
    tables of the held-out NEON product are gone too."""

    n_folds: int
    n: int
    acc: float | None
    nll: float | None
    novel_n: int
    majority_acc: float | None


class CellBaselines(_Model):
    """The baselines block of a cell. ``lookup_acc`` is the deterministic lookup (copy the most
    common label of the key among the training cards), ``lookup_nll`` and ``lookup_prob_acc``
    the Laplace-smoothed ``lookup_prob`` model; ``beats_lookup_novel`` is ``None`` on a baseline
    cell and a verdict on a model cell."""

    lookup_key: str
    lookup_rule: str
    majority_acc: float
    lookup_acc: float
    lookup_nll: float
    lookup_prob_acc: float
    novel_key: NovelKey
    lopo: Lopo
    beats_lookup_novel: bool | None = None


class CellCounts(_Model):
    n: int
    n_neg: int | None
    n_nonmodal: int
    class_counts: dict[int, int]


class CellMetrics(_Model):
    """Pooled held-out metrics (never fold-averaged, plan §5.4). ``auroc`` is ``None`` for a
    task wider than two classes."""

    acc: float
    macro_f1: float
    brier: float
    nll: float
    ece: float
    cov_at_5: float = Field(alias="cov@5%")
    cov_at_10: float = Field(alias="cov@10%")
    aurc: float
    auroc: float | None
    auroc_ci: ClusterCI | None
    threshold_cp: dict[str, float | None]


class BenchCell(_Model):
    """One measurement, keyed in the results file as ``<task>.<tier>.<framing>``."""

    task: str
    task_id: str
    task_key: str
    tier: Tier
    framing: str
    question_key: str | None
    fingerprint: dict[str, str] | None
    labels_sha256: str
    labels_content_sha256: str | None
    feature_spec: str | None
    label_sources: dict[str, int]
    min_weight: float
    teacher: bool
    teacher_in_test: bool
    masked: bool
    mask: str | None
    loco: bool
    selection: Selection
    pre_registered: bool
    exploratory: bool
    servable: bool
    n_folds: int
    skipped_folds: dict[str, str]
    guard_skipped_folds: dict[str, str]
    fold_choices: dict[str, Any]
    counts: CellCounts
    metrics: CellMetrics | None
    baselines: CellBaselines | None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @property
    def key(self) -> str:
        return cell_key(self.task, self.tier, self.framing)


class BenchResults(_Model):
    """A results file: ``cells`` keyed by :func:`cell_key`."""

    format: str = FORMAT
    date: str
    name: str
    mesa_clm: str
    labels_sha256: str
    labels_content_sha256: str | None
    environment: dict[str, Any]
    cells: dict[str, BenchCell]
    notes: list[str] = Field(default_factory=list)


def cell_key(task: str, tier: str, framing: str) -> str:
    """The dotted cell path a policy ``cite`` names after ``#`` (plan §4.7)."""
    return f"{task}.{tier}.{framing}"


def cite(path: str | Path, task: str, tier: str, framing: str) -> str:
    """``bench/results/<date>/<name>.json#<task>.<tier>.<framing>`` for ``path``."""
    return f"{Path(path).as_posix()}#{cell_key(task, tier, framing)}"


def environment() -> dict[str, Any]:
    """A secret-free record of where the bench ran (mesa-anyjev ``bench/run.py::environment``
    minus the model fields, which the fingerprint carries)."""
    return {
        "mesa_clm": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "numpy": np.__version__,
        "duckdb": duckdb.__version__,
    }


def finite(value: float | np.floating[Any] | None) -> float | None:
    """A JSON-safe float: ``None`` for ``None``, ``nan`` and ``inf``."""
    if value is None:
        return None
    f = float(value)
    return f if np.isfinite(f) else None


def results_path(out_dir: str | Path, date: str, name: str) -> Path:
    return Path(out_dir) / date / f"{name}.json"


def write_results(results: BenchResults, out_dir: str | Path = DEFAULT_OUT_DIR) -> Path:
    """Write ``<out_dir>/<date>/<name>.json`` (sorted keys, one-space indent, like mesa-anyjev)
    and the ``.md`` table beside it; returns the JSON path."""
    path = results_path(out_dir, results.date, results.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.model_dump(by_alias=True, mode="json")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown_table(results), encoding="utf-8")
    return path


def load_results(path: str | Path) -> BenchResults:
    return BenchResults.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _content_rows(con: duckdb.DuckDBPyConnection, relation: str) -> list[list[Any]]:
    """The content columns of every label row in ``relation``, in identity order."""
    order = ", ".join(IDENTITY_COLUMNS)
    return [
        list(r)
        for r in con.execute(
            f"SELECT {order}, label_index, weight, card, product_code, leak_group, "  # noqa: S608
            f"fold_eligible, bench_card FROM {relation} ORDER BY {order}"
        ).fetchall()
    ]


def _content_sha256(rows: list[list[Any]]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def labels_content_sha256(store: LabelStore) -> str:
    """sha256 over the content of every label row (identity, label index, weight, card, product,
    leak group, fold flags), in identity order and compact JSON, independent of ``label_id``,
    ``created_at``, ``origin``, ``actor`` and the stored state JSON. Empty store: the hash of
    ``[]``."""
    rows: list[list[Any]] = []
    if store.path.exists():
        with store.connect(read_only=True) as con:
            rows = _content_rows(con, TABLE)
    return _content_sha256(rows)


def snapshot_content_sha256(path: str | Path) -> str:
    """:func:`labels_content_sha256` over a snapshot Parquet file (``labels snapshot``), so the
    CLI can prove the store it benches still holds exactly the frozen rows (D30)."""
    target = str(Path(path).expanduser()).replace("'", "''")
    con = duckdb.connect()
    try:
        rows = _content_rows(con, f"read_parquet('{target}')")
    finally:
        con.close()
    return _content_sha256(rows)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _ci(ci: ClusterCI | None) -> str:
    if ci is None or ci.point is None:
        return ""
    return f"{_fmt(ci.point)} [{_fmt(ci.lower)}, {_fmt(ci.upper)}]"


def markdown_table(results: BenchResults) -> str:
    """The table the ``.md`` file carries: one row per cell, every number naming its JSON."""
    lines = [
        f"# bench {results.date} · {results.name} · mesa-clm {results.mesa_clm}",
        "",
        f"Every number below names `bench/results/{results.date}/{results.name}.json` "
        f"(labels_sha256 `{results.labels_sha256[:12]}…`). Silver labels are four-model agreement, "
        "not truth.",
        "",
        "| cell | n | n_neg | acc | nll | ece | auroc [95% cluster] | majority | lookup | "
        "novel n | novel auroc | novel nll | lopo acc |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, cell in results.cells.items():
        m = cell.metrics
        b = cell.baselines
        cols = [
            str(cell.counts.n),
            "" if cell.counts.n_neg is None else str(cell.counts.n_neg),
            _fmt(m.acc) if m else "",
            _fmt(m.nll) if m else "",
            _fmt(m.ece) if m else "",
            _ci(m.auroc_ci) if m else "",
            _fmt(b.majority_acc) if b else "",
            _fmt(b.lookup_acc) if b else "",
            str(b.novel_key.n) if b else "",
            _fmt(b.novel_key.auroc) if b else "",
            _fmt(b.novel_key.nll) if b else "",
            _fmt(b.lopo.acc) if b else "",
        ]
        lines.append(f"| {key} | " + " | ".join(cols) + " |")
    for note in results.notes:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"
