"""The closed choices by deterministic rules; CLM's answers to them audit-only (DESIGN A6).

M2's registered tier run found CLM's zero-shot answers to the three closed choices worse than the
majority class of their items (``bench/results/2026-10-03/tiers.json``, cells
``neon_annotate.zero_shot.F7``, ``neon_aspect.zero_shot.F7``, ``neon_value_kind.zero_shot.F7``),
and no pre-registered rule covers those tasks. After seeing that, the user decided (2026-10-04)
that by default they are answered by rules, CLM still answering them for audit only. A6 is a
product-safety change of serving behaviour, not a scientific claim. ``decider.closed_choice``
selects it: ``rules`` (the default) or ``clm`` (the M2 behaviour, for audits and tests).

Under ``rules`` the pipeline (:mod:`mesa_clm.pipeline`) applies, per column and proposal:

* **Q1 ``column.annotate``**: a column is annotated iff it is not ``cards.is_identifier``,
  whatever the planner says (a ``rule`` record answering ``Yes``, :data:`ANNOTATE_REASON`).
* **Q2 ``column.aspect``**: :func:`candidate_aspects`, the planner's aspect hint followed by the
  :class:`AspectLookup`'s top two, de-duplicated, ``other`` excluded (up to three aspects); when
  that is empty, :func:`fallback_aspects`: the registry's aspects in the order of the lookup's
  training prior, capped at :data:`FALLBACK_CAP` (:data:`FALLBACK_RULE`). Each chosen aspect is a
  ``rule`` record whose reason (:data:`ASPECT_REASONS`) says where it came from. No CLM answer is
  read: Q3 is asked only for the aspects chosen here, as in M2.
* **Q7 ``avu.value_kind``**: ``avu.pre_rule_value_kind`` first, else "the term label" as a
  ``rule`` record (:data:`VALUE_KIND_REASON`).

CLM is asked Q1, Q2 and Q7 in the same requests as in M2, and every such record is stored with
outcome ``abstain`` and reason :data:`AUDIT_ONLY_REASON` whatever the policy says (no vocabulary
value and no migration is added: ``abstain`` exists, ``decisions.reason`` is free text). It never
decides, makes no link and no label.

**The lookup is M0's, over a frozen table.** :data:`TABLE_PATH` (shipped in the package, its
sha256 pinned in :data:`TABLE_SHA256`) holds the items of M0's ``neon_aspect`` bench task on the
registered snapshot (``bench/snapshots/2026-09-29.parquet``: card, column name, silver aspect;
``column.aspect`` at the policy's ``min_weight``, ``fold_only``), the rows behind
``bench/results/2026-09-29/baselines.json#/cells/neon_aspect.baseline.lookup_prob``.
``scripts/freeze_aspect_lookup.py`` writes it (:func:`freeze_table`) and the tests rebuild it from
the snapshot. Every host reads the same table, and no sidecar is read. At annotate time the items
of the card being annotated are left out, as M0's leave-one-card-out lookup leaves them out, and
:class:`mesa_clm.bench.baselines.Lookup` counts the rest over
:func:`mesa_clm.bench.baselines.lookup_key`: a count table, no model (DESIGN D15 stands).
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

from mesa_clm.config import ClosedChoice
from mesa_clm.registry import ASPECTS, allowed_for_aspect
from mesa_clm.tasks import TASKS

if TYPE_CHECKING:
    from mesa_clm.bench.baselines import Lookup
    from mesa_clm.bench.tasks.base import Task

__all__ = [
    "ANNOTATE_REASON",
    "ASPECT_MODELS",
    "ASPECT_REASONS",
    "AUDIT_ONLY_REASON",
    "CLOSED_CHOICE_MODES",
    "CLOSED_CHOICE_TASKS",
    "FALLBACK_CAP",
    "FALLBACK_RULE",
    "LOOKUP_TOP",
    "TABLE_FORMAT",
    "TABLE_PATH",
    "TABLE_SHA256",
    "VALUE_KIND_REASON",
    "AspectItem",
    "AspectLookup",
    "AspectSource",
    "AspectTable",
    "AspectTableError",
    "candidate_aspects",
    "fallback_aspects",
    "freeze_table",
    "is_audit_only",
    "packaged_table",
    "render_table",
    "table_document",
]

TASK_ASPECT: Final[str] = "column.aspect"

# ``decider.closed_choice`` (``config.ClosedChoice``): ``rules`` (DESIGN A6, the default) or
# ``clm`` (the M2 behaviour: CLM's answers decide, rank-and-cap, D28).
CLOSED_CHOICE_MODES: Final[tuple[str, ...]] = get_args(ClosedChoice)
# The closed-choice tasks A6 answers by rule (plan §4.2 Q1, Q2, Q7).
CLOSED_CHOICE_TASKS: Final[frozenset[str]] = frozenset(
    {"column.annotate", "column.aspect", "avu.value_kind"}
)

# The marker of a CLM record kept for audit only: stored with outcome ``abstain`` and this reason.
AUDIT_ONLY_REASON: Final[str] = "audit_only_a6"
# The reasons of A6's rule records.
ANNOTATE_REASON: Final[str] = "a6_not_identifier"  # Q1: annotated, not an identifier
VALUE_KIND_REASON: Final[str] = "a6_value_kind_label"  # Q7: "the term label"

AspectSource = Literal["hint", "lookup", "fallback"]
# Q2's rule records: the reason and the ``model`` column per source of an aspect.
ASPECT_REASONS: Final[dict[str, str]] = {
    "hint": "a6_aspect_hint",
    "lookup": "a6_aspect_lookup",
    "fallback": "a6_aspect_fallback",
}
ASPECT_MODELS: Final[dict[str, str]] = {"hint": "planner", "lookup": "lookup", "fallback": "rule"}

# Q2's numbers. The lookup gives its top two (the decision's "top-2"); the hint and the lookup
# together are not capped (up to three aspects, as M2's top two plus the planner's aspect). The
# fallback keeps two: the decision says "capped" and names no number, so two is the
# implementation's reading (DESIGN A6), the lookup's own number.
LOOKUP_TOP: Final[int] = 2
FALLBACK_CAP: Final[int] = 2
FALLBACK_RULE: Final[str] = (
    "the first two of registry.ASPECTS in the order of the lookup's training prior (most rows "
    "first, a tie to the label seen first in card order; aspects without rows after, in registry "
    "order), skipping other, an aspect with no ontology in play, and unit for a column without a "
    "unit; no CLM answer is read"
)

# The frozen table (module docstring).
TABLE_FORMAT: Final[str] = "mesa-clm/aspect-lookup/1"
TABLE_PATH: Final[Path] = Path(__file__).resolve().parent / "aspect_lookup.json"
# sha256 of TABLE_PATH's bytes, as ``scripts/freeze_aspect_lookup.py`` prints it; a table with
# other bytes is refused (a tampered or regenerated file must come with this constant).
TABLE_SHA256: Final[str] = "12a38e76d6170c94a3bc268933f2b19abf272f9fc16626e389d8a419ff90e038"
# The bench task whose items the table holds (``mesa_clm.bench.tasks.neon.NEON_TASKS``).
TABLE_BENCH_TASK: Final[str] = "neon_aspect"


def is_audit_only(row: Mapping[str, Any]) -> bool:
    """Whether a stored decision (or record dict) is one of A6's audit-only CLM records."""
    return row.get("reason") == AUDIT_ONLY_REASON


def candidate_aspects(hint: str | None, looked_up: Sequence[str]) -> list[tuple[str, AspectSource]]:
    """Q2 under A6 before the fallback: the planner's aspect ``hint`` (if any) followed by the
    first :data:`LOOKUP_TOP` of ``looked_up`` (the lookup's labels, best first), de-duplicated
    (the first source kept), ``other`` excluded; up to three aspects, never capped further.
    Empty means the fallback decides."""
    listed: list[tuple[str, AspectSource]] = []
    if hint:
        listed.append((hint, "hint"))
    listed.extend((a, "lookup") for a in looked_up[:LOOKUP_TOP])
    out: list[tuple[str, AspectSource]] = []
    seen: set[str] = set()
    for aspect, source in listed:
        if aspect in seen or aspect not in ASPECTS:
            continue
        seen.add(aspect)
        if aspect != "other":
            out.append((aspect, source))
    return out


def fallback_aspects(
    order: Sequence[str], *, in_play: Collection[str], has_unit: bool
) -> list[str]:
    """The fallback's aspects (:data:`FALLBACK_RULE`): the first :data:`FALLBACK_CAP` aspects of
    ``order`` (the lookup's prior order, :meth:`AspectLookup.prior_order`) that are not
    ``other``, have an ontology in play (``allowed_for_aspect(aspect) & in_play``) and are not
    ``unit`` for a column without a unit (``has_unit``: the column's ``unit``; S would otherwise
    search UO for the literal word "unit"). Empty when no aspect qualifies."""
    play = frozenset(in_play)
    out: list[str] = []
    for aspect in order:
        if aspect in out or aspect not in ASPECTS or aspect == "other":
            continue
        if aspect == "unit" and not has_unit:
            continue
        if not allowed_for_aspect(aspect) & play:
            continue
        out.append(aspect)
        if len(out) == FALLBACK_CAP:
            break
    return out


# -- the frozen table ----------------------------------------------------------------------------


class AspectTableError(ValueError):
    """The lookup table is not the one this code pins, or not a table of ``column.aspect``."""


@dataclass(frozen=True)
class AspectItem:
    """One frozen item: a column of a card and its silver aspect."""

    card: str
    column: str
    aspect: str


@dataclass(frozen=True)
class AspectTable:
    """The items the lookup counts (module docstring). ``sha256`` names the table (the file's
    bytes for the packaged one); ``labels_sha256`` and ``labels_content_sha256`` the snapshot it
    was frozen from, ``min_weight`` the policy weight its rows were read at."""

    items: tuple[AspectItem, ...]
    sha256: str
    labels_sha256: str = ""
    labels_content_sha256: str = ""
    min_weight: float = 0.0
    snapshot: str = ""

    @classmethod
    def load(
        cls, path: str | Path = TABLE_PATH, *, sha256: str | None = TABLE_SHA256
    ) -> AspectTable:
        """Read a table file; :class:`AspectTableError` when its bytes are not ``sha256`` (unless
        ``None``) or it is not a ``column.aspect`` table under the current ``task_key``."""
        file = Path(path)
        raw = file.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if sha256 is not None and digest != sha256:
            raise AspectTableError(
                f"{file.name}: sha256 {digest[:12]}… is not the pinned {sha256[:12]}… "
                "(DESIGN A6; scripts/freeze_aspect_lookup.py writes the table and prints its sha256)"
            )
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            raise AspectTableError(f"{file.name}: not JSON ({exc.__class__.__name__})") from exc
        return cls.from_document(doc, sha256=digest)

    @classmethod
    def from_document(cls, doc: Mapping[str, Any], *, sha256: str) -> AspectTable:
        """A table from its JSON document (:func:`table_document`)."""
        if doc.get("format") != TABLE_FORMAT:
            raise AspectTableError(f"not an aspect lookup table (format {doc.get('format')!r})")
        if doc.get("task_id") != TASK_ASPECT or doc.get("task_key") != TASKS[TASK_ASPECT].key:
            raise AspectTableError(
                f"the table is for {doc.get('task_id')!r}/{doc.get('task_key')!r}, not "
                f"{TASK_ASPECT}/{TASKS[TASK_ASPECT].key} (a rotated task_key needs its labels "
                "migrated, DESIGN D1)"
            )
        items: list[AspectItem] = []
        for entry in doc.get("items", []):
            aspect = str(entry["aspect"])
            if aspect not in ASPECTS:
                raise AspectTableError(f"unknown aspect {aspect!r}")
            items.append(AspectItem(str(entry["card"]), str(entry["column"]), aspect))
        if len(items) != doc.get("n_items"):
            raise AspectTableError(f"{len(items)} items, but n_items is {doc.get('n_items')!r}")
        return cls(
            tuple(items),
            sha256,
            labels_sha256=str(doc.get("labels_sha256") or ""),
            labels_content_sha256=str(doc.get("labels_content_sha256") or ""),
            min_weight=float(doc.get("min_weight") or 0.0),
            snapshot=str(doc.get("snapshot") or ""),
        )

    @classmethod
    def from_items(cls, items: Sequence[tuple[str, str, str]], **meta: Any) -> AspectTable:
        """A table of ``(card, column, aspect)`` items (tests); ``sha256`` is over the items."""
        rows = tuple(AspectItem(card, column, aspect) for card, column, aspect in items)
        for row in rows:
            if row.aspect not in ASPECTS:
                raise AspectTableError(f"unknown aspect {row.aspect!r}")
        payload = json.dumps([[r.card, r.column, r.aspect] for r in rows], separators=(",", ":"))
        return cls(rows, hashlib.sha256(payload.encode("utf-8")).hexdigest(), **meta)


@functools.cache
def packaged_table() -> AspectTable:
    """The shipped table (:data:`TABLE_PATH`), checked against :data:`TABLE_SHA256`; read once
    per process."""
    return AspectTable.load()


def table_document(
    task: Task, *, labels_sha256: str, labels_content_sha256: str, snapshot: str
) -> dict[str, Any]:
    """The JSON document of the table frozen from M0's ``neon_aspect`` bench ``task``: one item
    per task item, in the task's order (``Lookup.fit`` orders by card, then by this order, as it
    does over the task), its lookup key reduced to the column name."""
    from mesa_clm.bench.baselines import LOOKUP_KEY_DOC, task_keys
    from mesa_clm.bench.tasks.neon import LICENSE, SOURCE

    if task.task_id != TASK_ASPECT:
        raise ValueError(f"{task.name}: a {task.task_id} task, not {TASK_ASPECT}")
    items: list[dict[str, str]] = []
    for (_, label), card, key in zip(task.items, task.cards, task_keys(task), strict=True):
        if (key[0], key[1], key[3]) != (TASK_ASPECT, "column", ""):
            raise ValueError(f"{task.name}: lookup key {key!r} is not a column key")
        items.append({"card": card, "column": key[2], "aspect": ASPECTS[label]})
    if len({(i["card"], i["column"]) for i in items}) != len(items):
        raise ValueError(f"{task.name}: a column of one card has two items")
    return {
        "format": TABLE_FORMAT,
        "design": "DESIGN A6: the M0 lookup's items for Q2 (column.aspect) under closed_choice: rules",
        "task_id": TASK_ASPECT,
        "task_key": task.spec.key,
        "bench_task": task.name,
        "snapshot": snapshot,
        "labels_sha256": labels_sha256,
        "labels_content_sha256": labels_content_sha256,
        "min_weight": float(task.meta.get("min_weight", 0.0)),
        "source": SOURCE,
        "license": LICENSE,
        "lookup_key": LOOKUP_KEY_DOC,
        "n_items": len(items),
        "items": items,
    }


def render_table(doc: Mapping[str, Any]) -> str:
    """The table file's text: two-space JSON in document order (``items`` last), one item per
    line, a final newline."""
    head = json.dumps({k: v for k, v in doc.items() if k != "items"}, indent=2, ensure_ascii=False)
    body = ",\n".join("    " + json.dumps(item, ensure_ascii=False) for item in doc["items"])
    return head[:-2] + ',\n  "items": [\n' + body + "\n  ]\n}\n"


def freeze_table(snapshot: str | Path | None = None) -> str:
    """The table's text frozen from the registered snapshot (default the registration's;
    another file is refused by :func:`mesa_clm.bench.run.load_inputs`, which also checks the
    tasks' counts against M0's published ones): what ``scripts/freeze_aspect_lookup.py``
    writes to :data:`TABLE_PATH`."""
    from mesa_clm.bench import registered as reg
    from mesa_clm.bench.run import load_inputs, scratch

    with scratch() as tmp:
        inputs = load_inputs(snapshot, tmp)
    return render_table(
        table_document(
            inputs.tasks[TABLE_BENCH_TASK],
            labels_sha256=inputs.labels_sha256,
            labels_content_sha256=inputs.labels_content_sha256,
            snapshot=reg.current().snapshot,
        )
    )


# -- the lookup ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class AspectLookup:
    """M0's lookup over ``table`` with the items of ``held_out`` (the card being annotated) left
    out: ``lookup`` is :class:`mesa_clm.bench.baselines.Lookup` fitted (counted) on the other
    items, ``n_items`` how many."""

    table: AspectTable
    held_out: str
    lookup: Lookup
    n_items: int

    @classmethod
    def for_card(cls, table: AspectTable, held_out: str) -> AspectLookup:
        from mesa_clm.bench.baselines import Lookup
        from mesa_clm.bench.tasks.base import Task

        kept = [item for item in table.items if item.card != held_out]
        task = Task(
            TABLE_BENCH_TASK,
            TASKS[TASK_ASPECT],
            [({}, ASPECTS.index(item.aspect)) for item in kept],
            "",
            "",
            cards=[item.card for item in kept],
        )
        keys = [(TASK_ASPECT, "column", item.column, "") for item in kept]
        return cls(table, held_out, Lookup.fit(task, list(range(len(kept))), keys), len(kept))

    @staticmethod
    def key(state: Mapping[str, Any]) -> tuple[str, str, str, str]:
        """M0's lookup key of a ``column_state``: ``(column.aspect, column, <name>, '')``."""
        from mesa_clm.bench.baselines import lookup_key

        return lookup_key(TASK_ASPECT, dict(state), "")

    def ranked(self, state: Mapping[str, Any]) -> list[tuple[str, int]]:
        """Every aspect with rows for the lookup key of ``state`` (a ``column_state``), most rows
        first, a tie going to the label seen first in card order (M0's ``Lookup``:
        ``Counter.most_common`` over card-ordered rows). Empty for a key no other card carries:
        unlike M0, whose lookup predicts the training majority for it, an unseen key contributes
        no aspect, and the fallback decides (DESIGN A6)."""
        counts = self.lookup.table.get(self.key(state))
        if not counts:
            return []
        return [(ASPECTS[label], int(n)) for label, n in counts.most_common()]

    def top(self, state: Mapping[str, Any]) -> list[str]:
        """The lookup's top :data:`LOOKUP_TOP` aspects for ``state`` (``other`` included: the
        caller excludes it after merging with the hint, :func:`candidate_aspects`)."""
        return [aspect for aspect, _ in self.ranked(state)[:LOOKUP_TOP]]

    def prior_order(self) -> list[tuple[str, int]]:
        """Every aspect with its rows in the training prior (M0's ``Lookup.prior``, the held-out
        card's items left out), most rows first, a tie going to the label seen first in card
        order (as M0's ``majority``); the aspects without rows after them, in registry order. The
        fallback's order: M0's ``lookup_prob`` gives an unseen key this prior."""
        counted = [(ASPECTS[label], int(n)) for label, n in self.lookup.prior.most_common()]
        seen = {aspect for aspect, _ in counted}
        return counted + [(aspect, 0) for aspect in ASPECTS if aspect not in seen]

    def evidence(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """What a Q3 group records of the lookup for its column (``search_json.a6.lookup``): the
        key, every label's rows and Laplace(α=1) frequency, the top two, and which table was read
        with which card left out."""
        key = self.key(state)
        ranked = self.ranked(state)
        laplace: dict[str, float] = {}
        if ranked:
            probs = self.lookup.prob(key)
            laplace = {aspect: float(probs[ASPECTS.index(aspect)]) for aspect, _ in ranked}
        return {
            "key": list(key),
            "seen": bool(ranked),
            "counts": dict(ranked),
            "laplace": laplace,
            "top": [aspect for aspect, _ in ranked[:LOOKUP_TOP]],
            "held_out": self.held_out,
            "n_items": self.n_items,
            "table_sha256": self.table.sha256,
            "labels_sha256": self.table.labels_sha256,
            "min_weight": self.table.min_weight,
        }

    def fallback(
        self, *, in_play: Collection[str], has_unit: bool
    ) -> tuple[list[str], dict[str, Any]]:
        """The fallback's aspects for a column (:func:`fallback_aspects` over
        :meth:`prior_order`) and what its Q3 groups record of it (``search_json.a6.fallback``)."""
        order = self.prior_order()
        kept = fallback_aspects([a for a, _ in order], in_play=in_play, has_unit=has_unit)
        return kept, {
            "rule": FALLBACK_RULE,
            "cap": FALLBACK_CAP,
            "order": [[aspect, n] for aspect, n in order],
            "has_unit": has_unit,
            "kept": list(kept),
        }
