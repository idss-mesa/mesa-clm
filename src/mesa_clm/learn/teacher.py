"""Teacher labels from the neon-ducklake curation corpus (DESIGN D19, D20; plan §5.1, §5.4,
§7.4; the M4 analysis plan's R3 reading, ``design/m4-analysis-plan.md``).

**Corpus.** ``<neon_root>/curation/generic/<DP>.validated.json``: one file per NEON product with
the ``accepted`` items of neon-ducklake's ``site validate`` (every item ``status`` ``accepted``;
``human_accepted`` items, when a file lists them, are taken too). Every file names its ``model``
and the two replicates whose agreement produced the items; a file whose ``model``, or whose
replicate proposal file's ``model`` (when the proposal file exists under ``neon_root``), starts
with ``mesa-clm`` is refused (D19: Opus is both the teacher and a silver labeler; mesa-clm must
never teach itself). The corpus is pinned by content: :func:`corpus_sha256` is sha256 over
``json.dumps(sorted((filename, file sha256) pairs))`` (``bench.registered.teacher_corpus_sha256``).

**Rows** (D19). An item is dropped, and counted by reason, when its CURIE prefix is not a
registry ontology (``out_of_registry``: GO, CHEBI, STATO, OBCS, …), when its neon aspect has no
mesa-clm aspect (``unmapped_aspect``: ``process``), when it names no column (``no_column``: a
dataset-level item; a dataset-scope ``term.fits`` state needs a scope the item does not name,
so none is synthesized), when its column is in no table card of the product (``unresolved_column``),
when the file's ``ontologyId`` disagrees with the CURIE's prefix (``ontology_mismatch``) or when
OLS has no usable record of the term (``term_missing``: a root, obsolete or unknown term, as
:class:`~mesa_clm.learn.labels.TermResolver` reports). **Target resolution** (plan §5.1, R3): a
column resolves to **every** table card of the product under ``<neon_root>/sites/SRER/cards/anyjev/
<DP>.<table>.md`` whose column set contains it, one row per card (the states differ by card; the
rows share ``leak_group`` = the product code). Labels produced, both **Yes only** (no negative is
synthesized): ``term.fits`` for (column state, CURIE) with the candidate as OLS records it
(``Candidate.as_state()``: label, CURIE, ontology, the description cut at 300 characters, which
the framings render as ``"{label}: {description}"``), and ``column.ontology_fits`` for (column
state, ontology) when the ontology is aspect-allowed for the item's aspect
(``registry.allowed_for_aspect``; otherwise the ``column.ontology_fits`` row is skipped and
counted ``ontology_rows_skipped`` — the ``term.fits`` row is still written, so this is not a
drop and is not among ``dropped``). Weight ``teacher`` 0.5; ``teacher_implicit`` 0.3 is
reserved for the unpicked ``seen_curies`` of the replicate proposal files: the ingest opens
the proposal files that exist under ``neon_root`` for their ``model`` only (the D19 refusal)
and counts them (``proposal_files``: 6 in the 2026-10-04 corpus), but synthesizes no implicit
row (``implicit_rows`` 0): those files carry no ``seen_curies`` the ingest can use, so nothing
is weighted 0.3. Every row:
``fold_eligible=False``, ``bench_card=False``, ``leak_group=product_code``,
``origin="teacher:<file sha256[:12]>"``. Two items that give one identity collapse onto the
first (``collapsed_identity``).

The ingest reads the corpus and the cards, never a silver label or a model output; the OLS
records come through the OLS layer's recorded fixtures (``tests/fixtures/ols-teacher``, recorded
once from live EBI OLS4 by ``scripts/record_ols_teacher.py`` and disclosed in DESIGN's M4 notes).
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from mesa_clm.cards import DatasetCard, load_card
from mesa_clm.identity import IdentityError, identity
from mesa_clm.learn.labels import WEIGHTS, TermResolver
from mesa_clm.provenance.labels import LabelRow, LabelStore
from mesa_clm.registry import (
    NEON_ASPECT_MAP,
    ONTOLOGY_REGISTRY,
    allowed_for_aspect,
    entry,
    prefix_of,
)
from mesa_clm.states import candidate_state, ontology_state, state_sha256
from mesa_clm.tasks import TASKS

__all__ = [
    "CARDS_DIR",
    "CORPUS_DIR",
    "CORPUS_PATTERN",
    "REFUSED_MODEL_PREFIX",
    "TEACHER_ORIGIN",
    "TEACHER_STATUSES",
    "TEACHER_TASKS",
    "CorpusFile",
    "TeacherError",
    "TeacherReport",
    "cards_by_product",
    "corpus_sha256",
    "ingest_teacher",
    "load_corpus",
    "teacher_curies",
]

CORPUS_DIR: Final[str] = "curation/generic"
CORPUS_PATTERN: Final[str] = "*.validated.json"
CARDS_DIR: Final[str] = "sites/SRER/cards/anyjev"
TEACHER_STATUSES: Final[frozenset[str]] = frozenset({"accepted", "human_accepted"})
REFUSED_MODEL_PREFIX: Final[str] = "mesa-clm"
TEACHER_ORIGIN: Final[str] = "teacher:"
TEACHER_TASKS: Final[tuple[str, ...]] = ("term.fits", "column.ontology_fits")
_PRODUCT_RE: Final = re.compile(r"^(DP\d\.\d{5}\.\d{3})\.")
_REGISTRY_IDS: Final[frozenset[str]] = frozenset(e.id for e in ONTOLOGY_REGISTRY)


class TeacherError(ValueError):
    """A corpus the ingest refuses: a file of a ``mesa-clm`` model (D19), a missing corpus or
    cards directory, an unreadable file."""


@dataclass(frozen=True)
class CorpusFile:
    """One ``<DP>.validated.json``: its product, model, sha256 and the items of a teacher
    status."""

    path: Path
    product_code: str
    model: str
    sha256: str
    items: tuple[dict[str, Any], ...]
    replicate_models: tuple[str, ...] = ()


@dataclass
class TeacherReport:
    """What :func:`ingest_teacher` did: the corpus pin, the files and items read, every drop
    reason with its count, the rows per task and source, and what the store took."""

    corpus_dir: str
    corpus_sha256: str
    files: int = 0
    items: int = 0
    proposal_files: int = 0
    implicit_rows: int = 0
    dropped: collections.Counter[str] = field(default_factory=collections.Counter)
    # column.ontology_fits rows not written because the ontology is not aspect-allowed (the
    # term.fits row of the same item is written: a skip, not a drop).
    ontology_rows_skipped: int = 0
    resolved_cards: int = 0
    per_task: dict[str, collections.Counter[str]] = field(default_factory=dict)
    inserted: int = 0
    skipped_existing: int = 0
    products_without_card: list[str] = field(default_factory=list)
    terms_missing: list[str] = field(default_factory=list)

    def count(self, task_id: str, source: str, n: int = 1) -> None:
        self.per_task.setdefault(task_id, collections.Counter())[source] += n

    def summary(self) -> dict[str, Any]:
        return {
            "corpus_dir": self.corpus_dir,
            "corpus_sha256": self.corpus_sha256,
            "files": self.files,
            "items": self.items,
            "proposal_files": self.proposal_files,
            "implicit_rows": self.implicit_rows,
            "dropped": dict(sorted(self.dropped.items())),
            "ontology_rows_skipped": self.ontology_rows_skipped,
            "resolved_cards": self.resolved_cards,
            "per_task": {t: dict(c) for t, c in sorted(self.per_task.items())},
            "inserted": self.inserted,
            "skipped_existing": self.skipped_existing,
            "products_without_card": sorted(self.products_without_card),
            "terms_missing": sorted(set(self.terms_missing)),
        }


def corpus_sha256(corpus_dir: str | Path) -> str:
    """The corpus pin (module docstring): the same formula as
    :func:`mesa_clm.bench.registered.teacher_corpus_sha256`."""
    from mesa_clm.bench.registered import teacher_corpus_sha256

    return teacher_corpus_sha256(corpus_dir, CORPUS_PATTERN)


def _models_of_replicates(data: dict[str, Any], neon_root: Path) -> tuple[str, ...]:
    """The ``model`` of each replicate proposal file that exists under ``neon_root`` (the
    generic files point at ``sites/<SITE>/curation/proposals/...``; the files that exist are
    opened for their ``model`` only and counted as ``proposal_files``)."""
    out: list[str] = []
    for rep in data.get("replicates") or []:
        rel = rep.get("proposal") if isinstance(rep, dict) else None
        if not rel:
            continue
        path = neon_root / str(rel)
        if not path.is_file():
            continue
        try:
            prop = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TeacherError(
                f"{path}: unreadable replicate proposal ({type(exc).__name__})"
            ) from exc
        out.append(str(prop.get("model") or ""))
    return tuple(out)


def load_corpus(neon_root: str | Path) -> list[CorpusFile]:
    """Every corpus file under ``<neon_root>/curation/generic``, sorted by name, with its
    teacher-status items. :class:`TeacherError` for a missing directory, an unreadable file or a
    file whose model (or a replicate proposal's) starts with ``mesa-clm`` (D19)."""
    root = Path(neon_root).expanduser()
    folder = root / CORPUS_DIR
    if not folder.is_dir():
        raise TeacherError(f"{folder}: no such corpus directory")
    out: list[CorpusFile] = []
    for path in sorted(folder.glob(CORPUS_PATTERN)):
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise TeacherError(f"{path}: not JSON ({type(exc).__name__})") from exc
        model = str(data.get("model") or "")
        rep_models = _models_of_replicates(data, root)
        for m in (model, *rep_models):
            if m.startswith(REFUSED_MODEL_PREFIX):
                raise TeacherError(
                    f"{path}: model {m!r} starts with {REFUSED_MODEL_PREFIX!r}; mesa-clm never "
                    "teaches itself (D19)"
                )
        items: list[dict[str, Any]] = []
        for key in ("accepted", "human_accepted"):
            block = data.get(key)
            if isinstance(block, list):
                items.extend(
                    it
                    for it in block
                    if isinstance(it, dict) and it.get("status") in TEACHER_STATUSES
                )
        product = str(data.get("productCode") or path.name.split(".validated")[0])
        out.append(
            CorpusFile(
                path=path,
                product_code=product,
                model=model,
                sha256=hashlib.sha256(raw).hexdigest(),
                items=tuple(items),
                replicate_models=rep_models,
            )
        )
    return out


def cards_by_product(cards_dir: str | Path) -> dict[str, list[DatasetCard]]:
    """Every table card under ``cards_dir`` grouped by product code, in file order."""
    folder = Path(cards_dir).expanduser()
    if not folder.is_dir():
        raise TeacherError(f"{folder}: no such cards directory")
    out: dict[str, list[DatasetCard]] = {}
    for path in sorted(folder.glob("*.md")):
        m = _PRODUCT_RE.match(path.name)
        if m is None:
            continue
        out.setdefault(m.group(1), []).append(load_card(path))
    return out


def _mapped_aspect(item: dict[str, Any]) -> str | None:
    """The mesa-clm aspect of an item's neon aspect, ``None`` when unmapped."""
    return NEON_ASPECT_MAP.get(str(item.get("aspect") or ""))


def _ontology_of(item: dict[str, Any]) -> str | None:
    """The registry id of the item's CURIE prefix, ``None`` when out of the registry."""
    curie = str(item.get("curie") or "")
    prefix = prefix_of(curie).lower()
    return prefix if prefix in _REGISTRY_IDS else None


def teacher_curies(files: Iterable[CorpusFile]) -> list[str]:
    """The distinct in-registry, aspect-mapped CURIEs of the corpus items, sorted: the OLS
    ``get_term`` closure the ingest needs (what ``scripts/record_ols_teacher.py`` records)."""
    out: set[str] = set()
    for f in files:
        for it in f.items:
            if _ontology_of(it) is not None and _mapped_aspect(it) is not None:
                out.add(str(it["curie"]))
    return sorted(out)


def _row(
    task_id: str,
    state: dict[str, Any],
    card: DatasetCard,
    origin: str,
    *,
    actor: str,
    source: str = "teacher",
) -> LabelRow:
    t = TASKS[task_id]
    ident = identity(task_id, state)
    return LabelRow(
        task_id=task_id,
        task_key=ident.task_key,
        target_sha256=ident.target_sha256,
        option_key=ident.option_key,
        label_source=source,  # type: ignore[arg-type]
        label=t.options[0],
        label_index=0,
        weight=WEIGHTS[source],
        state_sha256=state_sha256(state),
        state_json=state,
        card=card.name,
        product_code=card.product_code,
        leak_group=card.product_code,
        fold_eligible=False,
        bench_card=False,
        origin=origin,
        actor=actor,
    )


def ingest_teacher(
    store: LabelStore,
    neon_root: str | Path,
    resolver: TermResolver,
    *,
    cards_dir: str | Path | None = None,
    actor: str = "ingest-teacher",
    files: Sequence[CorpusFile] | None = None,
) -> TeacherReport:
    """Write the teacher rows of the corpus under ``neon_root`` into ``store`` (module
    docstring). ``cards_dir`` defaults to ``<neon_root>/sites/SRER/cards/anyjev``; ``files``
    (tests) replaces :func:`load_corpus`. Returns the report with every drop reason counted."""
    root = Path(neon_root).expanduser()
    corpus = list(files) if files is not None else load_corpus(root)
    cards = cards_by_product(cards_dir if cards_dir is not None else root / CARDS_DIR)
    report = TeacherReport(
        corpus_dir=str(root / CORPUS_DIR), corpus_sha256=corpus_sha256(root / CORPUS_DIR)
    )
    rows: list[LabelRow] = []
    seen: set[tuple[str, str, str]] = set()

    def add(task_id: str, state: dict[str, Any], card: DatasetCard, origin: str) -> None:
        try:
            row = _row(task_id, state, card, origin, actor=actor)
        except IdentityError:
            report.dropped["no_identity"] += 1
            return
        key = (row.task_key, row.target_sha256, row.option_key)
        if key in seen:
            report.dropped["collapsed_identity"] += 1
            return
        seen.add(key)
        rows.append(row)
        report.count(task_id, row.label_source)

    for f in corpus:
        report.files += 1
        report.proposal_files += sum(1 for _ in f.replicate_models)
        origin = f"{TEACHER_ORIGIN}{f.sha256[:12]}"
        product_cards = cards.get(f.product_code, [])
        if not product_cards:
            report.products_without_card.append(f.product_code)
        for it in f.items:
            report.items += 1
            ontology = _ontology_of(it)
            if ontology is None:
                report.dropped["out_of_registry"] += 1
                continue
            aspect = _mapped_aspect(it)
            if aspect is None:
                report.dropped["unmapped_aspect"] += 1
                continue
            declared = str(it.get("ontologyId") or "").lower()
            if declared and declared != ontology:
                report.dropped["ontology_mismatch"] += 1
                continue
            column = str(it.get("column") or "")
            if not column:
                report.dropped["no_column"] += 1
                continue
            targets = [c for c in product_cards if any(col.name == column for col in c.columns)]
            if not targets:
                report.dropped["unresolved_column"] += 1
                continue
            curie = str(it["curie"])
            cand = resolver.resolve(curie)
            if cand is None:
                report.dropped["term_missing"] += 1
                report.terms_missing.append(curie)
                continue
            e = entry(ontology)
            allowed = e.id in allowed_for_aspect(aspect)
            if not allowed:
                report.ontology_rows_skipped += 1  # the term.fits row is still written
            for card in targets:
                report.resolved_cards += 1
                col = card.column(column)
                add(
                    "term.fits",
                    candidate_state(card, "column", col, aspect, cand.as_state(), 0),
                    card,
                    origin,
                )
                if allowed:
                    add(
                        "column.ontology_fits",
                        ontology_state(card, col, aspect, e.id, e.option_text, sorted(e.aspects)),
                        card,
                        origin,
                    )
    report.inserted = store.insert_labels(rows)
    report.skipped_existing = len(rows) - report.inserted
    return report
