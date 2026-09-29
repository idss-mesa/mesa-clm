"""The production tasks and their keys (DESIGN D1; mesa-anyjev D1, D7).

A *task* is what mesa-anyjev called a question: a kind (``choice`` | ``noul`` | ``score``), a
wording and a frozen option list. ``task_key`` hashes exactly what AnyJev's ``Question.key``
hashed (``sha256(json.dumps({kind, text, options, scale, centers}, sort_keys=True,
ensure_ascii=False))[:16]``), so every key below equals the mesa-anyjev lock key and labels
recorded by mesa-anyjev carry over unchanged. Texts and option lists are copied character for
character from mesa-anyjev ``questions.py`` (``6159281``); the option tuples come from
:mod:`mesa_clm.registry`, which mirrors mesa-anyjev's registry.

How a task is *asked* of CLM (the rank-first Choice with an anchor, the context view, the
candidate template) is a *framing*, keyed separately by ``question_key`` (plan §4.4). Rewording
a framing rotates decisions, features and artifacts, never labels; rewording a task rotates its
``task_key`` and therefore orphans its labels, so the texts here are frozen and
``tests/unit/test_tasks.py`` pins every key against the committed copy of the mesa-anyjev lock.

Only six tasks are ``active`` in mesa-clm (plan §4.2). The others are kept for key parity only:
``avu.keep`` became a rule (D25), ``term.fits.chooser`` and the DataCite tasks are deferred to
M8, and ``dataset.ontology_applies`` belongs to the planner audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal

from mesa_clm.registry import (
    ASPECT_OPTIONS,
    DATACITE_CONTRIBUTOR_TYPES,
    DATACITE_DATE_TYPES,
    DATACITE_DESCRIPTION_TYPES,
    DATACITE_RELATION_TYPES,
    ONTOLOGY_OPTIONS,
    VALUE_KINDS,
)

Kind = Literal["choice", "noul", "score"]
Scope = Literal["dataset", "column", "site", "avu"]

# AnyJev's letter-readout limit; kept so a choice task can never grow past what the mesa-anyjev
# keys were computed for.
MAX_OPTIONS: Final = 26

# The noul option labels AnyJev fixes (``Question.noul``): index 0 is "Yes".
NOUL_OPTIONS: Final[tuple[str, str]] = ("Yes", "No")


class TaskError(ValueError):
    """A task spec that AnyJev's constructors would have rejected."""


def task_key(
    kind: str,
    text: str,
    options: tuple[str, ...] | list[str],
    scale: tuple[float, float] = (0.0, 1.0),
    centers: tuple[float, ...] | None = None,
) -> str:
    """AnyJev ``Question.key``, reproduced bit for bit: the payload is ``{"kind", "text",
    "options": list, "scale": list, "centers": list | None}`` dumped with ``sort_keys=True``,
    ``ensure_ascii=False`` and the default separators, hashed with sha256 and cut to 16 hex.

    ``scale`` entries are floats (AnyJev stores ``(0.0, 1.0)``), so ``0`` renders as ``0.0``.
    An empty ``centers`` tuple hashes as ``None``, as AnyJev's ``if self.centers`` does.
    """
    payload = json.dumps(
        {
            "kind": kind,
            "text": text,
            "options": list(options),
            "scale": [float(scale[0]), float(scale[1])],
            "centers": [float(c) for c in centers] if centers else None,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Task:
    """One frozen task: what mesa-anyjev's ``QuestionSpec`` carried, without the AnyJev object.

    ``options`` are the label strings a label row stores in ``label`` (``label_index`` is the
    position here). ``scale`` and ``centers`` only matter for ``score`` tasks, which mesa-clm
    never asks (D3); they are kept so the key of such a task would still be AnyJev's.
    """

    id: str
    kind: Kind
    text: str
    options: tuple[str, ...]
    scope: Scope
    active: bool
    scale: tuple[float, float] = (0.0, 1.0)
    centers: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind == "noul":
            if self.options != NOUL_OPTIONS:
                raise TaskError(f"{self.id}: a noul task has the options {NOUL_OPTIONS}")
        elif self.kind == "choice":
            if len(self.options) < 2:
                raise TaskError(f"{self.id}: choice needs at least 2 options")
            if len(self.options) > MAX_OPTIONS:
                raise TaskError(f"{self.id}: choice supports at most {MAX_OPTIONS} options")
            if len(set(self.options)) != len(self.options):
                raise TaskError(f"{self.id}: choice options must be unique")
        elif self.kind == "score":
            if not 2 <= len(self.options) <= 10:
                raise TaskError(f"{self.id}: score supports 2 to 10 bins or levels")
        else:
            raise TaskError(f"{self.id}: unknown kind {self.kind!r}")
        if not self.text.strip():
            raise TaskError(f"{self.id}: empty text")

    @property
    def key(self) -> str:
        """The stable 16-hex ``task_key`` (equals the mesa-anyjev lock key)."""
        return task_key(self.kind, self.text, self.options, self.scale, self.centers)

    @property
    def k(self) -> int:
        return len(self.options)

    def index_of(self, label: str) -> int:
        """The ``label_index`` of an option label; ``TaskError`` when the label is not an option."""
        try:
            return self.options.index(label)
        except ValueError as exc:
            raise TaskError(f"{self.id}: {label!r} is not one of its options") from exc


def _noul(task_id: str, text: str, scope: Scope, *, active: bool = False) -> Task:
    return Task(task_id, "noul", text, NOUL_OPTIONS, scope, active)


def _choice(
    task_id: str, text: str, options: tuple[str, ...], scope: Scope, *, active: bool = False
) -> Task:
    return Task(task_id, "choice", text, tuple(options), scope, active)


# -- the mesa-anyjev questions, texts verbatim -------------------------------------------------

T_COLUMN_ANNOTATE: Final = _noul(
    "column.annotate",
    "Should this column receive an ontology-grounded annotation? Answer No for record "
    "identifiers, internal codes, timestamps and bookkeeping fields.",
    "column",
    active=True,
)

T_COLUMN_ASPECT: Final = _choice(
    "column.aspect",
    "Which aspect of the dataset does this column describe?",
    ASPECT_OPTIONS,
    "column",
    active=True,
)

T_COLUMN_ONTOLOGY: Final = _choice(
    "column.ontology",
    "Which ontology should be searched for a term that annotates this column?",
    ONTOLOGY_OPTIONS,
    "column",
    active=True,
)

# mesa-anyjev's noul twin of column.ontology; in mesa-clm the rank_fit over the registry (Q3).
T_COLUMN_ONTOLOGY_FITS: Final = _noul(
    "column.ontology_fits",
    "Does the ontology described in the state contain the right kind of term for annotating "
    "this column?",
    "column",
    active=True,
)

T_TERM_FITS: Final = _noul(
    "term.fits",
    "Is the candidate ontology term a correct annotation for the described column, site or "
    "dataset: the right concept at an appropriate specificity, not merely related?",
    "column",
    active=True,
)

# mesa-mcp's own term picker carries only (ontology, value, candidates): its own key (M8).
T_TERM_FITS_CHOOSER: Final = _noul(
    "term.fits.chooser",
    "Is the candidate ontology term the correct term for the value the user wants to "
    "annotate with: the right concept at an appropriate specificity, not merely related?",
    "column",
)

T_VALUE_KIND: Final = _choice(
    "avu.value_kind",
    "What should the AVU value be?",
    VALUE_KINDS,
    "avu",
    active=True,
)

# A rule in mesa-clm (D25): exact-triple dedup, then cap 25 by p_fit. Key kept for import.
T_KEEP_AVU: Final = _noul(
    "avu.keep",
    "Given the annotations already selected for this dataset, does this annotation add "
    "correct, non-redundant information worth keeping?",
    "avu",
)

T_DATASET_ONTOLOGY_APPLIES: Final = _noul(
    "dataset.ontology_applies",
    "Does this ontology apply to the dataset as a whole: would at least one of its terms "
    "correctly describe what the dataset measures, observes, samples or records?",
    "dataset",
)

# -- DataCite (deferred to M8; keys kept) ------------------------------------------------------
T_DATACITE_RESOURCE_TYPE_FITS: Final = _noul(
    "datacite.resource_type_fits",
    "Is the DataCite ResourceTypeGeneral value the right general type for this resource?",
    "dataset",
)
T_DATACITE_CONTRIBUTOR_TYPE: Final = _choice(
    "datacite.contributor_type",
    "Which DataCite ContributorType best describes this contributor's role?",
    DATACITE_CONTRIBUTOR_TYPES,
    "dataset",
)
T_DATACITE_CONTRIBUTOR_TYPE_FITS: Final = _noul(
    "datacite.contributor_type_fits",
    "Is the DataCite ContributorType value the right role for this contributor?",
    "dataset",
)
T_DATACITE_RELATION_TYPE: Final = _choice(
    "datacite.relation_type",
    "Which DataCite RelationType describes how this resource relates to the related identifier?",
    DATACITE_RELATION_TYPES,
    "dataset",
)
T_DATACITE_RELATION_TYPE_FITS: Final = _noul(
    "datacite.relation_type_fits",
    "Is the DataCite RelationType value the right relation from this resource to the related "
    "identifier?",
    "dataset",
)
T_DATACITE_DATE_TYPE: Final = _choice(
    "datacite.date_type",
    "Which DataCite DateType describes this date?",
    DATACITE_DATE_TYPES,
    "dataset",
)
T_DATACITE_DATE_TYPE_FITS: Final = _noul(
    "datacite.date_type_fits",
    "Is the DataCite DateType value the right type for this date?",
    "dataset",
)
T_DATACITE_DESCRIPTION_TYPE: Final = _choice(
    "datacite.description_type",
    "Which DataCite DescriptionType describes this description text?",
    DATACITE_DESCRIPTION_TYPES,
    "dataset",
)

# In mesa-anyjev's lock order, so a diff against it reads top to bottom.
TASKS: Final[dict[str, Task]] = {
    t.id: t
    for t in (
        T_COLUMN_ANNOTATE,
        T_COLUMN_ASPECT,
        T_COLUMN_ONTOLOGY,
        T_COLUMN_ONTOLOGY_FITS,
        T_TERM_FITS,
        T_TERM_FITS_CHOOSER,
        T_VALUE_KIND,
        T_KEEP_AVU,
        T_DATASET_ONTOLOGY_APPLIES,
        T_DATACITE_RESOURCE_TYPE_FITS,
        T_DATACITE_CONTRIBUTOR_TYPE,
        T_DATACITE_CONTRIBUTOR_TYPE_FITS,
        T_DATACITE_RELATION_TYPE,
        T_DATACITE_RELATION_TYPE_FITS,
        T_DATACITE_DATE_TYPE,
        T_DATACITE_DATE_TYPE_FITS,
        T_DATACITE_DESCRIPTION_TYPE,
    )
}

ACTIVE_TASKS: Final[tuple[str, ...]] = tuple(t.id for t in TASKS.values() if t.active)

# The tasks CLM answers as a rank_fit (one Choice over candidates plus the anchor, D2): a
# label on them names the candidate in ``option_key`` (D1). Every other task is a closed choice.
RANK_FIT_TASKS: Final[frozenset[str]] = frozenset({"term.fits", "column.ontology_fits"})

# The pairs mesa-anyjev asked as a wide choice and its noul twin (mesa-anyjev D11); mesa-clm
# asks only the rank_fit side but keeps both keys.
TWINS: Final[dict[str, str]] = {
    "column.ontology": "column.ontology_fits",
    "datacite.contributor_type": "datacite.contributor_type_fits",
    "datacite.relation_type": "datacite.relation_type_fits",
    "datacite.date_type": "datacite.date_type_fits",
}


def task(task_id: str) -> Task:
    """The task called ``task_id``; ``KeyError`` when there is none."""
    return TASKS[task_id]


def task_keys(*, active_only: bool = False) -> dict[str, str]:
    """``{task_id: task_key}`` for every task (or the active ones), in lock order."""
    return {t.id: t.key for t in TASKS.values() if t.active or not active_only}


def by_key(key: str) -> Task:
    """The task whose ``task_key`` is ``key``; ``KeyError`` when no task has it."""
    for t in TASKS.values():
        if t.key == key:
            return t
    raise KeyError(key)
