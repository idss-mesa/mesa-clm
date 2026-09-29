"""Label ingestion into ``mesa_clm.labels`` (DESIGN D1, D9, D19, D21, D30; plan §5.1).

Ported from mesa-anyjev ``learn/labels.py`` (``6159281``). Every label row carries the exact
state the task was asked on, so a fit reads ``(state, label_index, weight)`` triples per task,
but its *identity* is ``(task_key, target_sha256, option_key, label_source)``
(:mod:`mesa_clm.identity`): the same finding on states that differ only in ``n_candidates`` is
one row. Sources and weights, the highest winning when one target has several: ``curator`` 1.0
(an explicit pick or "none of these"), ``curator_implicit`` 0.7 (the other offered
candidates of a pick), ``agent_pick`` 0.0 (a plain tool call; recorded, never fitted),
``consensus_all`` 0.8 and ``consensus_majority`` 0.6 (neon-avu-eval agreement),
``consensus_negative`` 0.5, ``teacher`` 0.5 / ``teacher_implicit`` 0.3 (Opus-validated neon
curation, never fold-eligible), ``gold`` 1.0.

The neon-avu-eval silver is *agreement between four agentic models*, not truth, and it is
circular with the agentic baselines the bench compares against; the docs say so wherever a
number derived from it appears. ``ingest_neon_eval`` writes labels for the five tasks the bench
fits (``term.fits``, ``column.ontology_fits``, ``column.annotate``, ``column.aspect``,
``avu.value_kind``); ``avu.keep`` is a rule in mesa-clm (D25) and ``column.ontology`` is asked as
``column.ontology_fits`` (plan §4.2), so neither gets labels.

Reproduced counts (``tests/unit/test_labels.py``): term.fits 16 consensus_all / 70
consensus_majority / 199 consensus_negative; column.ontology_fits 76 / 114; column.annotate 63
(0.6) / 35 (0.5); column.aspect 60; avu.value_kind 219 (0.6) / 82 class 3 (0.5). The 303 unique
valid (card, CURIE) pairs of ``validated.json`` become 285 term.fits rows because 18 pairs name
one of five GAZ CURIEs, and EBI OLS returns every GAZ term with ``isRoot: true``, which
``ols.to_candidates`` drops as a root term; ``IngestReport.terms_missing`` lists them. The 450
valid column-linked AVUs collapse onto 301 ``avu.value_kind`` identities (the state carries the
column and the term, not the value): the first row per identity wins, as mesa-anyjev's ``INSERT
OR IGNORE`` did, the 149 others are ``skipped["collapsed_identity"]`` and the 28 that named
another value kind are ``conflicts``, so ``per_task`` and ``inserted`` count what the store holds.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import duckdb

from mesa_clm.cards import DatasetCard, is_identifier, load_card
from mesa_clm.identity import IdentityError, identity
from mesa_clm.ols import Candidate, OLSLike, iri_for, to_candidates
from mesa_clm.provenance.labels import IDENTITY_COLUMNS, TABLE, LabelRow, LabelStore
from mesa_clm.registry import ASPECTS, ONTOLOGY_REGISTRY, VALUE_KINDS, allowed_for_aspect, prefix_of
from mesa_clm.states import (
    candidate_state,
    column_state,
    ontology_state,
    state_sha256,
    value_kind_state,
)
from mesa_clm.tasks import TASKS

WEIGHTS: Final[dict[str, float]] = {
    "curator": 1.0,
    "curator_implicit": 0.7,
    "agent_pick": 0.0,
    "consensus_all": 0.8,
    "consensus_majority": 0.6,
    "consensus_negative": 0.5,
    "teacher": 0.5,
    "teacher_implicit": 0.3,
    "gold": 1.0,
}

# Sources whose rows never enter a fold (D19, D21); ``labelled_targets`` still returns them.
NOT_FOLD_ELIGIBLE: Final[frozenset[str]] = frozenset({"agent_pick", "teacher", "teacher_implicit"})

# The tasks ``ingest_neon_eval`` writes, in ingestion order.
INGESTED_TASKS: Final[tuple[str, ...]] = (
    "term.fits",
    "column.annotate",
    "column.aspect",
    "column.ontology_fits",
    "avu.value_kind",
)

# mesa-anyjev label sources with no mesa-clm counterpart: ``accepted_avu`` was never produced
# and ``hosted_jev`` answers were a different model's. ``import_anyjev`` skips them by name.
_ANYJEV_SOURCE_MAP: Final[dict[str, str]] = {s: s for s in WEIGHTS}

_REGISTRY_IDS: Final[list[str]] = [e.id for e in ONTOLOGY_REGISTRY]
_PRODUCT_CODE = re.compile(r"^(DP\d\.\d{5}\.\d{3})")
_VALUE_KIND_TOP: Final = VALUE_KINDS.index("the most frequent data value")


def product_code_of(card_name: str) -> str:
    """The NEON product code of a card name (``DP1.10003.001.brd_countdata`` ->
    ``DP1.10003.001``), or the whole name when it has no such prefix."""
    m = _PRODUCT_CODE.match(card_name)
    return m.group(1) if m else card_name


@dataclass
class IngestReport:
    """What an ingestion or import did; ``summary()`` is what the CLI prints.

    ``inserted`` + ``skipped_existing`` is the number of rows submitted to the store, one per
    identity; ``per_task`` counts those rows. Derived rows that collapsed onto an earlier row's
    identity are ``skipped["collapsed_identity"]``, and ``conflicts`` says how many of them
    carried a different label than the row that won.
    """

    inserted: int = 0
    skipped_existing: int = 0
    per_task: dict[str, collections.Counter[str]] = field(default_factory=dict)
    terms_resolved: int = 0
    terms_missing: list[str] = field(default_factory=list)
    skipped: collections.Counter[str] = field(default_factory=collections.Counter)
    conflicts: int = 0
    source_ref: str = ""

    def count(self, task_id: str, source: str, n: int = 1) -> None:
        self.per_task.setdefault(task_id, collections.Counter())[source] += n

    def submit(self, store: LabelStore, rows: Sequence[LabelRow]) -> None:
        """Insert ``rows`` and fill ``inserted``, ``skipped_existing`` and ``per_task`` from
        what was submitted."""
        for row in rows:
            self.count(row.task_id, row.label_source)
        self.inserted = store.insert_labels(rows)
        self.skipped_existing = len(rows) - self.inserted

    def summary(self) -> dict[str, Any]:
        return {
            "inserted": self.inserted,
            "skipped_existing": self.skipped_existing,
            "terms_resolved": self.terms_resolved,
            "terms_missing": len(self.terms_missing),
            "terms_missing_curies": sorted(set(self.terms_missing)),
            "skipped": dict(self.skipped),
            "conflicts": self.conflicts,
            "per_task": {t: dict(c) for t, c in self.per_task.items()},
            "source_ref": self.source_ref,
        }


def _row(
    task_id: str,
    state: dict[str, Any],
    label_index: int,
    source: str,
    card: DatasetCard,
    origin: str,
    *,
    actor: str,
    option: str | None = None,
) -> LabelRow:
    """One label row with the D1 identity derived from ``state`` (``option`` names the option
    of a rank_fit row whose state carries no candidate)."""
    t = TASKS[task_id]
    ident = identity(task_id, state, option)
    return LabelRow(
        task_id=task_id,
        task_key=ident.task_key,
        target_sha256=ident.target_sha256,
        option_key=ident.option_key,
        label_source=source,  # type: ignore[arg-type]
        label=t.options[label_index],
        label_index=label_index,
        weight=WEIGHTS[source],
        state_sha256=state_sha256(state),
        state_json=state,
        card=card.name,
        product_code=card.product_code,
        leak_group=card.product_code,
        fold_eligible=source not in NOT_FOLD_ELIGIBLE,
        bench_card=False,
        origin=origin,
        actor=actor,
    )


def _value_kind(value: str, ols_label: str, column: str | None, site_codes: set[str]) -> int:
    v = value.strip()
    if v.lower() == ols_label.strip().lower():
        return VALUE_KINDS.index("the term label")
    if v in site_codes:
        return VALUE_KINDS.index("the site code")
    if column and v == column:
        return VALUE_KINDS.index("the column name")
    return _VALUE_KIND_TOP


class TermResolver:
    """``get_term`` through the OLS layer (recorded fixtures), memoised per CURIE.

    A CURIE resolves to ``None`` when OLS has no record, when the call fails, or when
    ``to_candidates`` drops the hit (obsolete, root or incomplete); the caller reports those
    as ``terms_missing``.
    """

    def __init__(self, client: OLSLike) -> None:
        self.client = client
        self.cache: dict[str, Candidate | None] = {}

    def resolve(self, curie: str) -> Candidate | None:
        if curie in self.cache:
            return self.cache[curie]
        prefix = prefix_of(curie)
        ontology_id = prefix.lower()
        try:
            hit = self.client.get_term(ontology_id, iri_for(curie))
        except Exception:  # any OLS or replay failure means "unresolved" (as mesa-anyjev)
            hit = None
        cand = None
        if hit:
            cands = to_candidates([hit], ontology_id, f"get_term:{curie}")
            cand = cands[0] if cands else None
        self.cache[curie] = cand
        return cand


def _valid(avu: dict[str, Any]) -> bool:
    return bool(
        avu.get("resolves")
        and not avu.get("obsolete")
        and avu.get("canonical")
        and avu.get("curie")
    )


def _dedupe(rows: Sequence[LabelRow], report: IngestReport) -> list[LabelRow]:
    """One row per identity key, the first wins (what ``INSERT OR IGNORE`` did silently);
    the others are counted in ``report.skipped["collapsed_identity"]`` and, when they carry
    another label, in ``report.conflicts``."""
    first: dict[tuple[str, str, str, str], LabelRow] = {}
    for row in rows:
        key = (row.task_key, row.target_sha256, row.option_key, row.label_source)
        kept = first.get(key)
        if kept is None:
            first[key] = row
            continue
        report.skipped["collapsed_identity"] += 1
        if kept.label_index != row.label_index:
            report.conflicts += 1
    return list(first.values())


# What a neon-avu-eval checkout must hold; the doctor's ``eval root`` check applies the same rule.
EVAL_ROOT_LAYOUT: Final[tuple[str, ...]] = ("results/validated.json", "cards/")


def check_eval_root(eval_root: str | Path) -> Path:
    """The expanded neon-avu-eval root, or ``FileNotFoundError`` naming what is missing."""
    root = Path(eval_root).expanduser()
    checks = {
        "results/validated.json": (root / "results" / "validated.json").is_file(),
        "cards/": (root / "cards").is_dir(),
    }
    missing = [rel for rel, ok in checks.items() if not ok]
    if missing:
        raise FileNotFoundError(
            f"{root}: not a neon-avu-eval checkout (missing {', '.join(missing)})"
        )
    return root


def ingest_neon_eval(
    store: LabelStore,
    eval_root: str | Path,
    resolver: TermResolver,
    *,
    exclude_models: Sequence[str] = (),
    actor: str = "ingest-neon-eval",
) -> IngestReport:
    """Silver labels from ``<eval_root>/results/validated.json`` and ``<eval_root>/cards/*.md``.

    The derivation is mesa-anyjev's, step for step: one ``term.fits`` row per unique valid
    (card, CURIE) pair (all models -> ``consensus_all``, at least two -> ``consensus_majority``,
    one -> ``consensus_negative`` with label No); ``column.annotate`` Yes for a column at least two
    models annotated, No for an unannotated non-identifier column; ``column.aspect`` for a
    column with at least two votes for its top aspect; ``column.ontology_fits`` Yes/No per
    aspect-allowed registry entry of such a column; ``avu.value_kind`` per valid column-linked
    AVU, one row per identity (the first AVU wins, the rest are ``collapsed_identity``).
    ``exclude_models`` drops those runs first (silver-minus-Opus for X4; ingest into a separate
    store, the identity does not carry the model set). ``FileNotFoundError`` when ``eval_root``
    lacks ``results/validated.json`` or ``cards/``.
    """
    root = check_eval_root(eval_root)
    validated_path = root / "results" / "validated.json"
    runs: list[dict[str, Any]] = json.loads(validated_path.read_text(encoding="utf-8"))
    excluded = set(exclude_models)
    runs = [r for r in runs if r["model"] not in excluded]
    ref = (
        "neon-avu-eval/results/validated.json@"
        f"{hashlib.sha256(validated_path.read_bytes()).hexdigest()[:12]}"
    )
    if excluded:
        ref += " minus:" + ",".join(sorted(excluded))
    report = IngestReport(source_ref=ref)
    cards = {c.stem: load_card(c) for c in sorted((root / "cards").glob("*.md"))}
    n_models = len({r["model"] for r in runs})
    rows: list[LabelRow] = []

    def add(
        task_id: str, state: dict[str, Any], label_index: int, source: str, card: DatasetCard
    ) -> None:
        rows.append(_row(task_id, state, label_index, source, card, ref, actor=actor))

    # ---- gather: valid AVUs per run, models per (card, curie), columns per (card, curie) ----
    valid: list[tuple[dict[str, Any], dict[str, Any]]] = []
    models_by_pair: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    columns_by_pair: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    aspects_by_pair: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for run in runs:
        for avu in run["avus"]:
            if not _valid(avu):
                continue
            valid.append((run, avu))
            pair = (run["card"], avu["curie"])
            models_by_pair[pair].add(run["model"])
            if avu.get("column"):
                columns_by_pair[pair][avu["column"]] += 1
            if avu.get("aspect") in ASPECTS:
                aspects_by_pair[pair][avu["aspect"]] += 1

    def consensus_source(pair: tuple[str, str]) -> str:
        n = len(models_by_pair[pair])
        if n >= n_models:
            return "consensus_all"
        if n >= 2:
            return "consensus_majority"
        return "consensus_negative"

    # ---- term.fits: one row per unique (card, curie) -----------------------------------------
    for pair in sorted(models_by_pair):
        card_name, curie = pair
        card = cards.get(card_name)
        if card is None:
            report.skipped["unknown_card"] += 1
            continue
        cand = resolver.resolve(curie)
        if cand is None:
            report.terms_missing.append(curie)
            continue
        report.terms_resolved += 1
        column: str | None = (
            columns_by_pair[pair].most_common(1)[0][0] if columns_by_pair[pair] else None
        )
        aspect = aspects_by_pair[pair].most_common(1)[0][0] if aspects_by_pair[pair] else "other"
        target: Any = None
        scope = "dataset"
        if column and any(c.name == column for c in card.columns):
            target, scope = card.column(column), "column"
        elif aspect in ("environment", "location") and len(card.sites) == 1:
            target, scope = card.sites[0], "site"
        state = candidate_state(card, scope, target, aspect, cand.as_state(), 0)
        source = consensus_source(pair)
        add("term.fits", state, 0 if source != "consensus_negative" else 1, source, card)

    # ---- column-level tasks --------------------------------------------------------------------
    col_models: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    col_aspects: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    col_prefixes: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for run, avu in valid:
        if avu.get("column"):
            key = (run["card"], avu["column"])
            col_models[key].add(run["model"])
            if avu.get("aspect") in ASPECTS:
                col_aspects[key][avu["aspect"]] += 1
            col_prefixes[key][prefix_of(avu["curie"]).lower()] += 1
    for card_name, card in cards.items():
        for col in card.columns:
            key = (card_name, col.name)
            n = len(col_models.get(key, set()))
            st = column_state(card, col)
            if n >= 2:
                add("column.annotate", st, 0, "consensus_majority", card)
            elif n == 0 and not is_identifier(col):
                add("column.annotate", st, 1, "consensus_negative", card)
            if n >= 2 and col_aspects.get(key):
                aspect, votes = col_aspects[key].most_common(1)[0]
                if votes >= 2:
                    add("column.aspect", st, ASPECTS.index(aspect), "consensus_majority", card)
                    used = {p for p in col_prefixes[key] if p in _REGISTRY_IDS}
                    if used:
                        # mesa-anyjev also wrote a column.ontology row here; mesa-clm asks the
                        # registry as ontology_fits only (plan §4.2 Q3), so no row for it.
                        for e in ONTOLOGY_REGISTRY:
                            if e.id not in allowed_for_aspect(aspect):
                                continue
                            ost = ontology_state(
                                card, col, aspect, e.id, e.option_text, sorted(e.aspects)
                            )
                            if e.id in used:
                                add("column.ontology_fits", ost, 0, "consensus_majority", card)
                            else:
                                add("column.ontology_fits", ost, 1, "consensus_negative", card)

    # ---- avu.value_kind per valid column-linked AVU ------------------------------------------
    # (mesa-anyjev also derived avu.keep here; it is a rule in mesa-clm, D25.)
    for run, avu in valid:
        card = cards.get(run["card"])
        if card is None:
            continue
        cand = resolver.resolve(avu["curie"])
        if cand is None:
            continue
        column = None
        if avu.get("column") and any(c.name == avu.get("column") for c in card.columns):
            column = str(avu["column"])
        if not column:
            continue
        aspect = str(avu.get("aspect")) if avu.get("aspect") in ASPECTS else "other"
        site_codes = {s.code for s in card.sites}
        kind = _value_kind(
            str(avu.get("value") or ""), str(avu.get("ols_label") or ""), column, site_codes
        )
        vst = value_kind_state(card, card.column(column), cand.as_state(), aspect)
        source = "consensus_majority" if kind != _VALUE_KIND_TOP else "consensus_negative"
        add("avu.value_kind", vst, kind, source, card)

    report.submit(store, _dedupe(rows, report))
    return report


# -- mesa-anyjev import -------------------------------------------------------------------------

_DUCKDB_RE = re.compile(r"^duckdb:///(.+)$")


def _snap_weight(recorded: float, source: str) -> float:
    """mesa-anyjev stored ``weight`` as REAL (float32): 0.7 comes back as 0.699999988. A value
    within float32 rounding of the source's table weight is that weight; anything else is kept
    (rounded to 6 places) so a deliberately different weight survives the import."""
    table = WEIGHTS[source]
    return table if abs(recorded - table) < 1e-6 else round(recorded, 6)


def import_anyjev(
    dsn: str | Path,
    store: LabelStore,
    *,
    actor: str = "import-anyjev",
) -> IngestReport:
    """Import ``mesa_anyjev.labels`` from a mesa-anyjev DuckDB sidecar (opened read-only).

    Identity is derived from each row's ``state_json`` (D1). Rows of inactive tasks, of a
    ``question_key`` that is not the task's current key, of a source mesa-clm has no counterpart
    for (``accepted_avu``, ``hosted_jev``) or whose state carries no target identity are counted
    in ``skipped``. When several anyjev rows collapse onto one identity (the same pair with
    different ``n_candidates``), the highest weight wins, then the latest ``ts``; a collapse
    with a different label is counted in ``conflicts``. The recorded weight is kept as is.
    """
    text = str(dsn)
    m = _DUCKDB_RE.match(text)
    path = Path(m.group(1) if m else text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no mesa-anyjev labels file at {path}")
    con = duckdb.connect(str(path), read_only=True)
    try:
        cur = con.execute(
            "SELECT question_id, question_key, state_sha256, state_json, label_index, "
            "label_source, weight, source_ref, card, ts, label_id FROM mesa_anyjev.labels "
            "ORDER BY ts, label_id"
        )
        names = [d[0] for d in cur.description or []]
        raw = [dict(zip(names, values, strict=True)) for values in cur.fetchall()]
    finally:
        con.close()
    ref = f"anyjev-import:{path.name}@{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
    report = IngestReport(source_ref=ref)

    best: dict[tuple[str, str, str, str], tuple[float, Any, LabelRow]] = {}
    for r in raw:
        task_id = str(r["question_id"])
        t = TASKS.get(task_id)
        if t is None or not t.active:
            report.skipped["inactive_task"] += 1
            continue
        if str(r["question_key"]) != t.key:
            report.skipped["key_mismatch"] += 1
            continue
        source = _ANYJEV_SOURCE_MAP.get(str(r["label_source"]))
        if source is None:
            report.skipped["unknown_source"] += 1
            continue
        state = r["state_json"]
        if isinstance(state, str):
            state = json.loads(state)
        try:
            ident = identity(task_id, state)
        except IdentityError:
            report.skipped["no_identity"] += 1
            continue
        label_index = int(r["label_index"])
        if not 0 <= label_index < t.k:
            report.skipped["bad_label_index"] += 1
            continue
        card_name = str(r["card"] or state["card"]["dataset"])
        product = product_code_of(card_name)
        row = LabelRow(
            task_id=task_id,
            task_key=ident.task_key,
            target_sha256=ident.target_sha256,
            option_key=ident.option_key,
            label_source=source,  # type: ignore[arg-type]
            label=t.options[label_index],
            label_index=label_index,
            weight=_snap_weight(float(r["weight"]), source),
            state_sha256=str(r["state_sha256"]),
            state_json=state,
            card=card_name,
            product_code=product,
            leak_group=product,
            fold_eligible=source not in NOT_FOLD_ELIGIBLE,
            origin=f"{ref} {r['source_ref'] or ''}".strip(),
            actor=actor,
        )
        key = (row.task_key, row.target_sha256, row.option_key, row.label_source)
        cur_best = best.get(key)
        if cur_best is None:
            best[key] = (row.weight, r["ts"], row)
            continue
        if cur_best[2].label_index != row.label_index:
            report.conflicts += 1
        # Highest weight wins, then the latest ts (rows arrive in ts order).
        if row.weight >= cur_best[0]:
            best[key] = (row.weight, r["ts"], row)
        report.skipped["collapsed_identity"] += 1
    report.submit(store, [b[2] for b in best.values()])
    return report


# -- reading ------------------------------------------------------------------------------------


@dataclass
class LabelledSet:
    """Parallel lists, one entry per ``(target_sha256, option_key)``; ``labels`` are option
    indices. ``fold_eligible``, ``bench_card`` and ``leak_group`` let the bench build
    leakage-aware folds (D19, D30)."""

    states: list[dict[str, Any]]
    labels: list[int]
    weights: list[float]
    cards: list[str]
    target_sha256: list[str]
    option_key: list[str]
    sources: list[str] = field(default_factory=list)
    fold_eligible: list[bool] = field(default_factory=list)
    bench_card: list[bool] = field(default_factory=list)
    leak_group: list[str] = field(default_factory=list)

    def class_counts(self) -> dict[int, int]:
        return dict(sorted(collections.Counter(self.labels).items()))

    def __len__(self) -> int:
        return len(self.labels)


def labelled_targets(
    store: LabelStore,
    task_id: str,
    *,
    min_weight: float = 0.0,
    exclude_cards: Sequence[str] = (),
    sources: Sequence[str] | None = None,
) -> LabelledSet:
    """One entry per ``(target_sha256, option_key)`` with ``weight >= min_weight``, the
    highest-weight source winning (ties: the earliest row), sorted by that key.

    ``min_weight`` should come from :func:`mesa_clm.policy_defaults.min_weight_for` (D9). A
    fitter or the bench must exclude ``fold_eligible=False`` and ``bench_card=True`` entries
    from its folds itself (D19, D30); they are returned so the caller can report them.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in store.labels_for(
        task_id, min_weight=min_weight, exclude_cards=exclude_cards, sources=sources
    ):
        key = (str(row["target_sha256"]), str(row["option_key"]))
        cur = best.get(key)
        if cur is None or float(row["weight"]) > float(cur["weight"]):
            best[key] = row
    rows = [best[k] for k in sorted(best)]
    return LabelledSet(
        states=[r["state_json"] for r in rows],
        labels=[int(r["label_index"]) for r in rows],
        weights=[float(r["weight"]) for r in rows],
        cards=[str(r["card"]) for r in rows],
        target_sha256=[str(r["target_sha256"]) for r in rows],
        option_key=[str(r["option_key"]) for r in rows],
        sources=[str(r["label_source"]) for r in rows],
        fold_eligible=[bool(r["fold_eligible"]) for r in rows],
        bench_card=[bool(r["bench_card"]) for r in rows],
        leak_group=[str(r["leak_group"]) for r in rows],
    )


def snapshot(store: LabelStore, out_path: str | Path) -> str:
    """Write every label to a Parquet file in identity order and return the sha256 of the
    file's bytes (``labels_sha256``, recorded by every bench cell and manifest, D30).

    The same store snapshotted twice gives identical bytes; two ingestions differ in
    ``label_id`` and ``created_at`` and therefore in ``labels_sha256``, which is the point: the
    hash names one frozen set of rows. Uses DuckDB ``COPY`` (no pyarrow dependency).
    """
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    order = ", ".join(IDENTITY_COLUMNS)
    target = str(out).replace("'", "''")
    with store.connect(read_only=True) as con:
        con.execute(
            f"COPY (SELECT * FROM {TABLE} ORDER BY {order}) TO '{target}' (FORMAT PARQUET)"  # noqa: S608
        )
    return hashlib.sha256(out.read_bytes()).hexdigest()


def stats(store: LabelStore) -> dict[str, Any]:
    """Counts per task, source and label: ``{"n_labels", "tasks": {task_id: {"n", "sources":
    {source: n}, "labels": {label: n}, "weights": {weight: n}}}}``."""
    tasks: dict[str, dict[str, Any]] = {}
    for c in store.counts():
        t = tasks.setdefault(
            str(c["task_id"]),
            {
                "n": 0,
                "sources": collections.Counter(),
                "labels": collections.Counter(),
                "weights": collections.Counter(),
            },
        )
        n = int(c["n"])
        t["n"] += n
        t["sources"][str(c["label_source"])] += n
        t["labels"][str(c["label"])] += n
        t["weights"][float(c["weight"])] += n
    return {
        "n_labels": sum(t["n"] for t in tasks.values()),
        "tasks": {
            task_id: {
                "n": t["n"],
                "sources": dict(sorted(t["sources"].items())),
                "labels": dict(sorted(t["labels"].items())),
                "weights": {str(w): n for w, n in sorted(t["weights"].items())},
            }
            for task_id, t in sorted(tasks.items())
        },
    }
