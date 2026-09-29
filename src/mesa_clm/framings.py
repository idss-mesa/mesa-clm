"""Framings: how a frozen task is asked of CLM (DESIGN D1, D2, D3, D23; plan §4.1–4.4, §5.6).

A *task* (:mod:`mesa_clm.tasks`) is the frozen question mesa-anyjev asked, keyed by ``task_key``
so that labels carry over. A *framing* is everything about how CLM is asked that same task:
which state view is the context (``target_state``, ``column_state``, ...), whether that view is
sent as a dict (rendered by CLM's ``to_text``) or through a short template, the optional
``instructions`` appended after the context, how a candidate becomes its option text, the fixed
abstain anchor and, for closed choices, the option texts and their map back to the anyjev
labels. Each framing has a ``question_key`` (plan §4.4) that keys decisions, features and
artifacts; rewording a framing rotates that key and must update ``framings.lock.json``, never
the labels (D1). Encoder and head identity never enter the key: they are the D5 fingerprints.

Shapes (D2, D3): a ``rank_fit`` framing asks one ``/v1/systemone`` Choice per candidate group,
``criteria = {<CURIE|registry id>: candidate_text, "__none__": anchor_text}``, so
``s_c = ln p_c − ln p_anchor`` is independent of the rest of the set; a ``choice`` framing asks
the closed option set with no anchor. Noul-per-candidate (the mesa-anyjev shape) survives only
as the control arm F1 of the pre-registered X1 framing A/B, marked ``control=True`` and asked as
one request per candidate over the anyjev ``candidate_state`` / ``ontology_state``.

Contexts (D23): every production view ends with the target because CLM pools the encoder's
*last* token. :func:`build_context` projects a builder-ordered state dict through the framing's
view (a superset such as a stored ``candidate_state`` projects onto its ``target_state``) and
``tests/unit/test_context_ends_with_target.py`` checks the tail over the fixture cards. Raw
cards never enter a context. Only :mod:`mesa_clm.render` touches the vendored text contract
(D4); this module builds wire-shaped dicts and hands them to it.

X1 arms for ``term.fits`` and ``column.ontology_fits`` (plan §5.6): F1 control, F4 Choice with a
task-sentence ``instructions``, F7 state-only (the default until amendment A1), F9 the short
query-shaped template ending in ``... ontology term:``. ``column.annotate``, ``column.aspect``
and ``avu.value_kind`` have one state-only framing F7 each. ``column.ontology`` (the wide
mesa-anyjev choice) is never asked: its registry is ranked as ``column.ontology_fits`` (Q3).
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from mesa_clm import render
from mesa_clm.registry import (
    ANCHOR_KEY,
    ANCHORS,
    ANNOTATE_OPTIONS,
    ASPECT_OPTIONS,
    ASPECTS,
    ONTOLOGY_REGISTRY,
    VALUE_KINDS,
    OntologyEntry,
    entry,
)
from mesa_clm.states import MAX_DESCRIPTION, target_state_from
from mesa_clm.tasks import NOUL_OPTIONS, TASKS, Task
from mesa_clm.vocab import SHAPES, Shape

# -- types and constants -----------------------------------------------------------------------

ContextView = Literal[
    "target_state", "column_state", "value_kind_state", "candidate_state", "ontology_state"
]
MaskRule = Literal["aspect"]
TargetKind = Literal["column", "site", "dataset"]

# The state side of a request: a template framing renders a string, every other framing sends the
# view dict and lets CLM's ``to_text`` render it (so the wire state is the builder-ordered dict).
Context = str | dict[str, Any]
Question = render.Question
Request = tuple[Context, dict[str, Question]]

RENDER_VERSION: Final[str] = "1"
DEFAULT_CANDIDATE_TEMPLATE: Final[str] = "{label}: {description}"

# Wire keys of the closed ``avu.value_kind`` choice, aligned with ``registry.VALUE_KINDS``. They
# are framing texts (part of ``question_key``); the anyjev labels stay the VALUE_KINDS strings.
VALUE_KIND_KEYS: Final[tuple[str, ...]] = ("term_label", "site_code", "column_name", "top_value")

# The anyjev noul labels a CLM noul answer maps back to (``NOUL_KEYS`` -> ``NOUL_OPTIONS``).
NOUL_LABEL_MAP: Final[dict[str, str]] = {"true": NOUL_OPTIONS[0], "false": NOUL_OPTIONS[1]}

# The views a framing may name, with the top-level keys a state must carry to be projected onto
# them (``target_state`` and ``candidate_state`` also take ``column`` or ``site``).
VIEW_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "target_state": ("card", "scope", "aspect"),
    "column_state": ("card", "column"),
    "value_kind_state": ("card", "column", "aspect", "term"),
    "candidate_state": ("card", "scope", "aspect", "candidate", "n_candidates"),
    "ontology_state": ("card", "column", "aspect", "ontology"),
}
# The mesa-anyjev per-candidate views: only a control framing may use them (they end with the
# candidate, not the target, which is the point of the control).
CONTROL_VIEWS: Final[frozenset[str]] = frozenset({"candidate_state", "ontology_state"})

# The fields a context template may reference, per target kind (plus the card-level ones).
_CARD_FIELDS: Final[tuple[str, ...]] = (
    "dataset",
    "title",
    "product_description",
    "sites",
    "scope",
    "aspect",
)
_TARGET_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "column": ("name", "description", "dtype", "unit", "profile"),
    "site": ("code", "site_name", "domain", "habitat"),
    "dataset": (),
}
_TERM_FIELDS: Final[tuple[str, ...]] = ("term_label", "term_curie")

# The fields of the ``question_key`` payload, in plan §4.4 order. ``control`` and ``active`` are
# deliberately absent: flipping either must not rotate decisions, features or artifacts.
QUESTION_KEY_FIELDS: Final[tuple[str, ...]] = (
    "task_key",
    "shape",
    "framing_id",
    "context_view",
    "context_template",
    "instructions",
    "candidate_template",
    "anchor_text",
    "closed_options",
    "option_keys",
    "label_map",
    "mask_rule",
    "render_version",
    "schema_sha256",
)

LOCK_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "framings.lock.json"


class FramingError(ValueError):
    """A framing spec, state or candidate set the framing cannot be asked with."""


# -- candidates --------------------------------------------------------------------------------


class CandidateLike(Protocol):
    """What a rank_fit framing needs from a candidate: the wire key (CURIE or registry id, also
    the label ``option_key``, D1), its label and its description."""

    @property
    def key(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def description(self) -> str: ...


@dataclass(frozen=True)
class FramingCandidate:
    """A candidate as the framings see it. ``key`` is the CURIE (``term.fits``) or the registry
    ontology id (``column.ontology_fits``). The extra fields only feed the F1 control's anyjev
    ``candidate_state`` and default to what :func:`mesa_clm.states.candidate_state` would coerce
    a sparse dict to."""

    key: str
    label: str
    description: str = ""
    ontology_id: str = ""
    synonyms: tuple[str, ...] = ()
    has_children: bool = False

    @classmethod
    def from_term(cls, term: Mapping[str, Any]) -> FramingCandidate:
        """From an OLS term dict (``ols.Candidate.as_state()``: label, curie, ontology_id,
        description, synonyms, has_children); ``curie`` is the key."""
        return cls(
            key=str(term.get("curie", "")),
            label=str(term.get("label", "")),
            description=str(term.get("description", "")),
            ontology_id=str(term.get("ontology_id", "")),
            synonyms=tuple(str(s) for s in list(term.get("synonyms") or [])),
            has_children=bool(term.get("has_children", False)),
        )

    @classmethod
    def from_registry(cls, e: OntologyEntry) -> FramingCandidate:
        """From a registry entry: key = ontology id, label = CURIE prefix, description = the
        option text without its ``PREFIX: `` head, so the default candidate template renders the
        registry ``option_text`` verbatim (plan §4.2 Q3)."""
        head = f"{e.curie_prefix}: "
        if not e.option_text.startswith(head):
            raise FramingError(f"registry option text for {e.id!r} does not start with {head!r}")
        return cls(key=e.id, label=e.curie_prefix, description=e.option_text[len(head) :])


def ontology_candidates(ids: Iterable[str] | None = None) -> list[FramingCandidate]:
    """The ``column.ontology_fits`` candidates in registry order, optionally restricted to
    ``ids`` (unknown ids raise :class:`FramingError`)."""
    if ids is None:
        return [FramingCandidate.from_registry(e) for e in ONTOLOGY_REGISTRY]
    wanted = {i.lower() for i in ids}
    unknown = wanted - {e.id for e in ONTOLOGY_REGISTRY}
    if unknown:
        raise FramingError(f"unknown registry ontology id(s): {sorted(unknown)}")
    return [FramingCandidate.from_registry(e) for e in ONTOLOGY_REGISTRY if e.id in wanted]


# -- the framing -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Framing:
    """One way of asking a task of CLM. Every field except ``anchor_key``, ``control`` and
    ``active`` enters :func:`question_key`.

    * ``context_view`` names the :mod:`mesa_clm.states` view the context is projected onto;
      ``context_template`` is ``None`` (send the view dict) or a ``str.format`` template (or one
      per target kind ``column`` / ``site`` / ``dataset``) over :func:`template_fields`.
    * ``instructions`` is appended after the context by CLM (``None`` = state alone).
    * ``candidate_template`` formats a rank_fit candidate (``key``, ``label``, ``description``,
      the description cut at :data:`mesa_clm.states.MAX_DESCRIPTION`; an empty description
      gives the label alone).
    * ``anchor_key`` / ``anchor_text`` are the fixed abstain option of a rank_fit (D2), ``None``
      for a closed choice and for the noul control.
    * ``closed_options`` maps the wire keys of a closed choice to the option texts CLM embeds, in
      task option order; ``option_keys`` are the anyjev labels in the same order (what a label
      row stores in ``label``); ``label_map`` maps every wire key (for the F1 control: the noul
      keys) back to its anyjev label. All three are ``None`` for a rank_fit, whose wire key *is*
      the ``option_key`` (D1).
    * ``mask_rule`` ``"aspect"`` applies ``registry.mask_for_aspect`` after scoring (Q3).
    """

    id: str
    task_id: str
    shape: Shape
    context_view: ContextView
    context_template: str | dict[str, str] | None = None
    instructions: str | None = None
    candidate_template: str = DEFAULT_CANDIDATE_TEMPLATE
    anchor_key: str | None = None
    anchor_text: str | None = None
    closed_options: dict[str, str] | None = None
    option_keys: tuple[str, ...] | None = None
    label_map: dict[str, str] | None = None
    mask_rule: MaskRule | None = None
    render_version: str = RENDER_VERSION
    control: bool = False
    active: bool = True

    def __post_init__(self) -> None:
        if self.task_id not in TASKS:
            raise FramingError(f"{self.id}: unknown task {self.task_id!r}")
        if not self.id or not self.id.strip():
            raise FramingError(f"{self.task_id}: a framing needs an id")
        if self.shape not in SHAPES:
            raise FramingError(f"{self.id}: unknown shape {self.shape!r}; expected one of {SHAPES}")
        if self.context_view not in VIEW_KEYS:
            raise FramingError(f"{self.id}: unknown context view {self.context_view!r}")
        if (self.context_view in CONTROL_VIEWS) != self.control:
            raise FramingError(
                f"{self.id}: the per-candidate views {sorted(CONTROL_VIEWS)} are for control "
                "framings only, and a control framing must use one"
            )
        if self.mask_rule not in (None, "aspect"):
            raise FramingError(f"{self.id}: unknown mask rule {self.mask_rule!r}")
        if self.mask_rule == "aspect" and self.task_id != "column.ontology_fits":
            raise FramingError(f"{self.id}: the aspect mask is defined over the ontology registry")
        if self.instructions is not None and not self.instructions.strip():
            raise FramingError(f"{self.id}: instructions must be None or a non-empty sentence")
        self._check_templates()
        if self.control:
            self._check_control()
        elif self.shape == "rank_fit":
            self._check_rank_fit()
        else:
            self._check_choice()

    def _check_templates(self) -> None:
        """Every placeholder must be a known field; the candidate template must format."""
        if isinstance(self.context_template, str):
            _check_placeholders(self.id, self.context_template, None)
        elif isinstance(self.context_template, dict):
            if not self.context_template:
                raise FramingError(f"{self.id}: an empty template map renders nothing")
            for kind, template in self.context_template.items():
                if kind not in _TARGET_FIELDS:
                    raise FramingError(f"{self.id}: unknown target kind {kind!r} in templates")
                _check_placeholders(self.id, template, kind)
        try:
            self.candidate_template.format(key="k", label="l", description="d")
        except (KeyError, IndexError, ValueError) as exc:
            raise FramingError(f"{self.id}: bad candidate template: {exc}") from exc

    def _check_control(self) -> None:
        if self.shape != "rank_fit":
            raise FramingError(f"{self.id}: the noul control is a rank_fit arm")
        if self.anchor_key is not None or self.anchor_text is not None:
            raise FramingError(f"{self.id}: a noul has no anchor option")
        if not self.instructions:
            raise FramingError(f"{self.id}: the noul control needs the task sentence")
        if self.closed_options is not None or self.option_keys != NOUL_OPTIONS:
            raise FramingError(f"{self.id}: the control's options are the anyjev {NOUL_OPTIONS}")
        if self.label_map != NOUL_LABEL_MAP:
            raise FramingError(f"{self.id}: the control's label map is {NOUL_LABEL_MAP}")

    def _check_rank_fit(self) -> None:
        if self.anchor_key != ANCHOR_KEY or not self.anchor_text:
            raise FramingError(f"{self.id}: a rank_fit carries the {ANCHOR_KEY!r} anchor (D2)")
        if not (
            self.closed_options is None and self.option_keys is None and self.label_map is None
        ):
            raise FramingError(f"{self.id}: a rank_fit has open candidates, not closed options")

    def _check_choice(self) -> None:
        if self.anchor_key is not None or self.anchor_text is not None:
            raise FramingError(f"{self.id}: a closed choice has no anchor (plan §4.1)")
        if not self.closed_options or self.option_keys is None or self.label_map is None:
            raise FramingError(
                f"{self.id}: a closed choice needs closed_options, option_keys, label_map"
            )
        if self.option_keys != self.task.options:
            raise FramingError(f"{self.id}: option_keys must be the task's options, in order")
        if len(self.closed_options) != len(self.option_keys):
            raise FramingError(f"{self.id}: closed_options and option_keys differ in length")
        if any(not k or k == ANCHOR_KEY or not t for k, t in self.closed_options.items()):
            raise FramingError(f"{self.id}: every closed option needs a key and a text")
        if self.label_map != dict(zip(self.closed_options, self.option_keys, strict=True)):
            raise FramingError(f"{self.id}: label_map must map each wire key to its anyjev label")

    def __hash__(self) -> int:
        # Frozen but with dict fields: identity is (task, framing id), which the registry keeps
        # unique; equality stays field-wise (the generated ``__eq__``).
        return hash((self.task_id, self.id))

    @property
    def task(self) -> Task:
        return TASKS[self.task_id]

    @property
    def task_key(self) -> str:
        """The frozen task's key (= mesa-anyjev's lock key, D1)."""
        return self.task.key

    @property
    def question_key(self) -> str:
        return question_key(self)


def _check_placeholders(framing_id: str, template: str, kind: str | None) -> None:
    allowed = set(_CARD_FIELDS) | set(_TERM_FIELDS)
    for k, names in _TARGET_FIELDS.items():
        if kind is None or kind == k:
            allowed |= set(names)
    for _literal, name, _spec, _conv in string.Formatter().parse(template):
        if name is None:
            continue
        if not name or name not in allowed:
            raise FramingError(f"{framing_id}: unknown template field {name!r}")


# -- keys --------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Sorted keys, compact separators, non-ASCII kept: the same canon as ``state_sha256``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def question_payload(framing: Framing) -> dict[str, Any]:
    """The plan §4.4 payload ``question_key`` hashes. ``closed_options`` is a list of
    ``[key, text]`` pairs so their order (the option index) enters the key; ``schema_sha256``
    is the vendored text contract's (D4), so a CLM schema change rotates every key."""
    closed = (
        [[k, v] for k, v in framing.closed_options.items()]
        if framing.closed_options is not None
        else None
    )
    return {
        "task_key": framing.task_key,
        "shape": framing.shape,
        "framing_id": framing.id,
        "context_view": framing.context_view,
        "context_template": framing.context_template,
        "instructions": framing.instructions,
        "candidate_template": framing.candidate_template,
        "anchor_text": framing.anchor_text,
        "closed_options": closed,
        "option_keys": list(framing.option_keys) if framing.option_keys is not None else None,
        "label_map": framing.label_map,
        "mask_rule": framing.mask_rule,
        "render_version": framing.render_version,
        "schema_sha256": render.schema_sha256(),
    }


def question_key(framing: Framing) -> str:
    """``sha256(canonical_json(question_payload))[:16]`` (plan §4.4). Encoder and head identity
    are not inputs: an encoder change never rotates a question key (that is ``encoder_fp``,
    D5); a template, instruction, anchor, option or schema edit always does."""
    payload = canonical_json(question_payload(framing))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# -- contexts ----------------------------------------------------------------------------------


def target_kind(state: Mapping[str, Any]) -> TargetKind:
    """``column`` / ``site`` / ``dataset`` from what the state carries (both is an error)."""
    if "column" in state and "site" in state:
        raise FramingError("state has both a column and a site target")
    if "column" in state:
        return "column"
    if "site" in state:
        return "site"
    return "dataset"


def _require(view: str, state: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in state]
    if missing:
        raise FramingError(f"state lacks {missing} for view {view!r}")


def project(view: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Project a builder-ordered state (or a superset of one) onto ``view``, key order included,
    so a stored ``candidate_state`` renders as the ``target_state`` it was asked about and a
    ``column_state`` never carries a stray candidate. :class:`FramingError` when the state lacks
    what the view needs."""
    if view not in VIEW_KEYS:
        raise FramingError(f"unknown context view {view!r}")
    _require(view, state, VIEW_KEYS[view])
    if view == "target_state":
        try:
            return target_state_from(dict(state))
        except ValueError as exc:
            raise FramingError(str(exc)) from exc
    if view == "column_state":
        return {"card": state["card"], "column": state["column"]}
    if view == "value_kind_state":
        return {k: state[k] for k in ("card", "column", "aspect", "term")}
    if view == "ontology_state":
        return {k: state[k] for k in ("card", "column", "aspect", "ontology")}
    # candidate_state: the mesa-anyjev key order (card, scope, column|site, aspect, candidate,
    # n_candidates), which is what its ``state_sha256`` and the imported labels were computed on.
    kind = target_kind(state)
    out: dict[str, Any] = {"card": state["card"], "scope": state["scope"]}
    if kind != "dataset":
        out[kind] = state[kind]
    out["aspect"] = state["aspect"]
    out["candidate"] = state["candidate"]
    out["n_candidates"] = state["n_candidates"]
    return out


def template_fields(view: Mapping[str, Any]) -> dict[str, str]:
    """The flat ``str.format`` namespace of a projected view: ``dataset``, ``title``,
    ``product_description``, ``sites`` (codes, comma-separated), ``scope``, ``aspect``; for a
    column target ``name``, ``description``, ``dtype``, ``unit`` (the card's unit, or the
    dtype when the card shows none, so a template's ``({unit})`` never renders empty) and
    ``profile``; for a site target ``code``, ``site_name``, ``domain``, ``habitat``; for a
    ``value_kind_state`` also ``term_label`` and ``term_curie``."""
    card = view.get("card")
    if not isinstance(card, dict):
        raise FramingError("state has no card header")
    out: dict[str, str] = {
        "dataset": str(card.get("dataset", "")),
        "title": str(card.get("product_title", "")),
        "product_description": str(card.get("product_description", "")),
        "sites": ", ".join(str(s.get("code", "")) for s in card.get("sites", [])),
        "scope": str(view.get("scope", "")),
        "aspect": str(view.get("aspect", "")),
    }
    kind = target_kind(view)
    if kind == "column":
        col = view["column"]
        out.update(
            name=str(col.get("name", "")),
            description=str(col.get("description", "")),
            dtype=str(col.get("dtype", "")),
            unit=str(col.get("unit") or col.get("dtype") or ""),
            profile=str(col.get("profile", "")),
        )
    elif kind == "site":
        site = view["site"]
        out.update(
            code=str(site.get("code", "")),
            site_name=str(site.get("name", "")),
            domain=str(site.get("domain", "")),
            habitat=str(site.get("habitat", "")),
        )
    if isinstance(view.get("term"), dict):
        out["term_label"] = str(view["term"].get("label", ""))
        out["term_curie"] = str(view["term"].get("curie", ""))
    return out


def build_context(framing: Framing, state: Mapping[str, Any]) -> Context:
    """The state side of a request: the view dict (CLM renders it with ``to_text``) or, for a
    template framing, the rendered string. The wire state is exactly this value."""
    view = project(framing.context_view, state)
    if framing.context_template is None:
        return view
    if isinstance(framing.context_template, str):
        template = framing.context_template
    else:
        kind = target_kind(view)
        try:
            template = framing.context_template[kind]
        except KeyError as exc:
            raise FramingError(f"{framing.id}: no template for a {kind} target") from exc
    try:
        return template.format_map(template_fields(view))
    except KeyError as exc:
        raise FramingError(f"{framing.id}: template field {exc} is not in this view") from exc


def context_text(framing: Framing, state: Mapping[str, Any]) -> str:
    """The exact text the state head sees: the context, then ``instructions`` after a blank
    line when the framing has any (``render.state_text``). What ``context_sha256`` and the
    token guard are computed over."""
    return render.state_text(build_context(framing, state), framing.instructions)


# -- questions and requests --------------------------------------------------------------------


def candidate_text(framing: Framing, candidate: CandidateLike) -> str:
    """One rank_fit candidate's option text: the description cut at ``MAX_DESCRIPTION`` through
    ``candidate_template``; an empty description gives the label alone. Stable per key, so the
    action-side vector is cached across states."""
    description = str(candidate.description or "")[:MAX_DESCRIPTION]
    if not description:
        return str(candidate.label)
    return framing.candidate_template.format(
        key=str(candidate.key), label=str(candidate.label), description=description
    )


def build_question(framing: Framing, candidates: Sequence[CandidateLike] | None = None) -> Question:
    """The wire question of a non-control framing: a Choice with ``criteria`` = the closed
    options (``candidates`` must then be ``None``) or the candidates keyed by CURIE / registry id
    plus the anchor last, and ``instructions`` (``None`` for a state-only framing). Key order
    follows CLM's client (``type``, ``instructions``, ``criteria``) so bodies match its goldens.
    """
    if framing.control:
        raise FramingError(
            f"{framing.id}: a control framing asks one noul per candidate; use build_requests"
        )
    criteria: dict[str, Any]
    if framing.shape == "choice":
        if candidates is not None:
            raise FramingError(f"{framing.id}: a closed choice takes no candidates")
        criteria = dict(framing.closed_options or {})
    else:
        if not candidates:
            raise FramingError(f"{framing.id}: a rank_fit needs at least one candidate")
        criteria = {}
        for c in candidates:
            key = str(c.key)
            if not key:
                raise FramingError(f"{framing.id}: a candidate has no key")
            if key == framing.anchor_key:
                raise FramingError(f"{framing.id}: {key!r} is the anchor key")
            if key in criteria:
                raise FramingError(f"{framing.id}: duplicate candidate {key!r}")
            criteria[key] = candidate_text(framing, c)
        criteria[str(framing.anchor_key)] = framing.anchor_text
    return {"type": "choice", "instructions": framing.instructions, "criteria": criteria}


def noul_question(framing: Framing) -> Question:
    """The F1 control's wire question: a bare noul carrying the anyjev task sentence, whose
    candidates CLM derives itself (``true: Yes. This is true: ...`` / ``false: ...``)."""
    if not framing.control:
        raise FramingError(f"{framing.id}: only a control framing asks a noul (D3)")
    return {"type": "noul", "instructions": framing.instructions}


def build_request(
    framing: Framing,
    state: Mapping[str, Any],
    candidates: Sequence[CandidateLike] | None = None,
) -> Request:
    """``(context, {task_id: question})`` for one non-control framing: the one request a
    candidate group costs. Control framings need one request per candidate: see
    :func:`build_requests`."""
    if framing.control:
        raise FramingError(
            f"{framing.id}: a control framing needs one request per candidate; use build_requests"
        )
    return build_context(framing, state), {framing.task_id: build_question(framing, candidates)}


def _candidate_dict(candidate: CandidateLike) -> dict[str, Any]:
    """The ``candidate`` entry of an anyjev ``candidate_state``, with its coercions."""
    return {
        "label": str(candidate.label),
        "curie": str(candidate.key),
        "ontology_id": str(getattr(candidate, "ontology_id", "") or ""),
        "description": str(candidate.description or "")[:MAX_DESCRIPTION],
        "synonyms": [str(s) for s in list(getattr(candidate, "synonyms", ()) or ())[:5]],
        "has_children": bool(getattr(candidate, "has_children", False)),
    }


def control_state(
    framing: Framing, state: Mapping[str, Any], candidate: CandidateLike, n_candidates: int
) -> dict[str, Any]:
    """The per-candidate anyjev state of the F1 control, built from the target's state:
    ``candidate_state`` (``term.fits``) or ``ontology_state`` (``column.ontology_fits``, the
    candidate being a registry id), byte-identical in keys and order to :mod:`mesa_clm.states`
    so its ``state_sha256`` joins the imported labels."""
    if not framing.control:
        raise FramingError(f"{framing.id}: not a control framing")
    if framing.context_view == "ontology_state":
        _require("ontology_state", state, ("card", "column", "aspect"))
        try:
            e = entry(str(candidate.key))
        except KeyError as exc:
            raise FramingError(
                f"{framing.id}: {candidate.key!r} is not a registry ontology"
            ) from exc
        return {
            "card": state["card"],
            "column": state["column"],
            "aspect": state["aspect"],
            "ontology": {"id": e.id, "description": e.option_text, "aspects": sorted(e.aspects)},
        }
    target = project("target_state", state)
    full: dict[str, Any] = dict(target)
    full["candidate"] = _candidate_dict(candidate)
    full["n_candidates"] = int(n_candidates)
    return project("candidate_state", full)


def build_requests(
    framing: Framing,
    state: Mapping[str, Any],
    candidates: Sequence[CandidateLike] | None = None,
) -> list[Request]:
    """Every request a candidate group costs under ``framing``: one for a Choice framing, one
    per candidate for the F1 control (each ``(candidate_state, {candidate.key: noul})``)."""
    if not framing.control:
        return [build_request(framing, state, candidates)]
    if not candidates:
        raise FramingError(f"{framing.id}: the control needs at least one candidate")
    keys = [str(c.key) for c in candidates]
    if len(set(keys)) != len(keys) or not all(keys):
        raise FramingError(f"{framing.id}: candidate keys must be unique and non-empty")
    n = len(candidates)
    return [
        (
            build_context(framing, control_state(framing, state, c, n)),
            {str(c.key): noul_question(framing)},
        )
        for c in candidates
    ]


# -- the registry ------------------------------------------------------------------------------

_F4_INSTRUCTIONS: Final[dict[str, str]] = {
    "term.fits": "Which term is the right concept for this target, not merely related?",
    "column.ontology_fits": (
        "Which ontology has the right kind of term for annotating this column?"
    ),
}

# F9: the query-shaped context (plan §5.6). The column form is the plan's template verbatim;
# ``term.fits`` also targets sites (Q5) and the dataset (Q6), which get the same shape.
_F9_COLUMN: Final[str] = (
    "NEON dataset {title}. Column {name}: {description} ({unit}). {aspect} ontology term:"
)
_F9_TERM: Final[dict[str, str]] = {
    "column": _F9_COLUMN,
    "site": "NEON dataset {title}. Site {code}: {site_name}, {habitat}. {aspect} ontology term:",
    "dataset": "NEON dataset {title}: {product_description}. {aspect} ontology term:",
}


def _fit_framings(
    task_id: str, anchor: str, control_view: ContextView, f9: str | dict[str, str]
) -> dict[str, Framing]:
    """The four X1 arms of a rank_fit task (plan §5.6): F1 control, F4, F7 (active), F9."""
    mask: MaskRule | None = "aspect" if task_id == "column.ontology_fits" else None
    return {
        "F1": Framing(
            "F1",
            task_id,
            "rank_fit",
            control_view,
            instructions=TASKS[task_id].text,
            option_keys=NOUL_OPTIONS,
            label_map=dict(NOUL_LABEL_MAP),
            mask_rule=mask,
            control=True,
            active=False,
        ),
        "F4": Framing(
            "F4",
            task_id,
            "rank_fit",
            "target_state",
            instructions=_F4_INSTRUCTIONS[task_id],
            anchor_key=ANCHOR_KEY,
            anchor_text=anchor,
            mask_rule=mask,
            active=False,
        ),
        "F7": Framing(
            "F7",
            task_id,
            "rank_fit",
            "target_state",
            anchor_key=ANCHOR_KEY,
            anchor_text=anchor,
            mask_rule=mask,
        ),
        "F9": Framing(
            "F9",
            task_id,
            "rank_fit",
            "target_state",
            context_template=f9,
            anchor_key=ANCHOR_KEY,
            anchor_text=anchor,
            mask_rule=mask,
            active=False,
        ),
    }


def _closed_f7(task_id: str, view: ContextView, keys: Sequence[str]) -> Framing:
    """The one state-only framing of a closed choice: no anchor, no instructions, the anyjev
    option texts as candidates keyed by ``keys``."""
    options = TASKS[task_id].options
    return Framing(
        "F7",
        task_id,
        "choice",
        view,
        closed_options=dict(zip(keys, options, strict=True)),
        option_keys=options,
        label_map=dict(zip(keys, options, strict=True)),
    )


# ``column.annotate``: the anyjev noul labels are the wire keys, the sentences of
# ``registry.ANNOTATE_OPTIONS`` the candidate texts (index 0 = "Yes", as the labels store it).
_ANNOTATE_F7: Final = Framing(
    "F7",
    "column.annotate",
    "choice",
    "column_state",
    closed_options=dict(ANNOTATE_OPTIONS),
    option_keys=NOUL_OPTIONS,
    label_map={k: k for k in NOUL_OPTIONS},
)

FRAMINGS: Final[dict[str, dict[str, Framing]]] = {
    "column.annotate": {"F7": _ANNOTATE_F7},
    "column.aspect": {"F7": _closed_f7("column.aspect", "column_state", ASPECTS)},
    "column.ontology_fits": _fit_framings(
        "column.ontology_fits", ANCHORS["ontology"], "ontology_state", _F9_COLUMN
    ),
    "term.fits": _fit_framings("term.fits", ANCHORS["term"], "candidate_state", _F9_TERM),
    "avu.value_kind": {"F7": _closed_f7("avu.value_kind", "value_kind_state", VALUE_KIND_KEYS)},
}

# The production framing per task: F7 everywhere until amendment A1 (the X1 winner) says otherwise.
ACTIVE: Final[dict[str, str]] = {task_id: "F7" for task_id in FRAMINGS}

if (
    TASKS["column.aspect"].options != ASPECT_OPTIONS
    or TASKS["avu.value_kind"].options != VALUE_KINDS
):
    raise RuntimeError("mesa_clm.framings: registry options drifted from the task options")

for _task_id, _arms in FRAMINGS.items():
    if len({f.shape for f in _arms.values()}) != 1:
        raise RuntimeError(f"mesa_clm.framings: {_task_id} mixes shapes across its framings")
    if any(f.task_id != _task_id or f.id != fid for fid, f in _arms.items()):
        raise RuntimeError(
            f"mesa_clm.framings: {_task_id} registry keys disagree with the framings"
        )
    if not _arms[ACTIVE[_task_id]].active or _arms[ACTIVE[_task_id]].control:
        raise RuntimeError(
            f"mesa_clm.framings: {_task_id} active framing must be active, not control"
        )
    if sum(f.active for f in _arms.values()) != 1:
        raise RuntimeError(f"mesa_clm.framings: {_task_id} has more than one active framing")


def framings_for(task_id: str) -> tuple[Framing, ...]:
    """Every framing of a task in registry order (F1, F4, F7, F9 for the fit tasks)."""
    try:
        return tuple(FRAMINGS[task_id].values())
    except KeyError as exc:
        raise FramingError(f"no framings for task {task_id!r}") from exc


def framing(task_id: str, framing_id: str) -> Framing:
    """One framing by task and id; :class:`FramingError` when there is none."""
    try:
        return FRAMINGS[task_id][framing_id]
    except KeyError as exc:
        raise FramingError(f"no framing {framing_id!r} for task {task_id!r}") from exc


def active_framing(task_id: str) -> Framing:
    """The production framing of a task (F7 until A1)."""
    try:
        return FRAMINGS[task_id][ACTIVE[task_id]]
    except KeyError as exc:
        raise FramingError(f"no active framing for task {task_id!r}") from exc


def by_question_key(key: str) -> Framing:
    """The framing whose ``question_key`` is ``key``; :class:`FramingError` when none has it."""
    for arms in FRAMINGS.values():
        for f in arms.values():
            if f.question_key == key:
                return f
    raise FramingError(f"no framing has question_key {key!r}")


# -- the lock ----------------------------------------------------------------------------------


def framing_entry(f: Framing) -> dict[str, Any]:
    """What the lock pins per framing: ``question_key`` first, then every field."""
    out: dict[str, Any] = {"question_key": f.question_key}
    for fld in fields(f):
        if fld.name in ("id", "task_id"):
            continue
        value = getattr(f, fld.name)
        out[fld.name] = list(value) if isinstance(value, tuple) else value
    return out


def _sha(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def lock_payload() -> dict[str, Any]:
    """``{schema_sha256, tasks: {task_id: {task_key, shape, active_framing, framings: {id:
    entry}}}, lock_sha}``; ``lock_sha`` hashes the canonical JSON of the rest."""
    tasks = {
        task_id: {
            "task_key": TASKS[task_id].key,
            "shape": next(iter(arms.values())).shape,
            "active_framing": ACTIVE[task_id],
            "framings": {fid: framing_entry(f) for fid, f in arms.items()},
        }
        for task_id, arms in FRAMINGS.items()
    }
    body = {"schema_sha256": render.schema_sha256(), "tasks": tasks}
    return {**body, "lock_sha": _sha(body)}


def lock_sha() -> str:
    return str(lock_payload()["lock_sha"])


def read_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def write_lock(path: Path = LOCK_PATH) -> str:
    """Write the lock (``indent=1``, trailing newline) and return its ``lock_sha``."""
    payload = lock_payload()
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(payload["lock_sha"])


def lock_drift(path: Path = LOCK_PATH) -> list[str]:
    """Human-readable differences between the code and the lock file (empty means in sync):
    a missing or tampered lock, a schema change, new or dropped tasks and framings, a rotated
    ``task_key`` or ``question_key`` (naming the fields that changed), a moved ``active_framing``.
    """
    current = lock_payload()
    try:
        locked = read_lock(path)
    except FileNotFoundError:
        return [f"{path.name} is missing; run `mesa-clm framings --update-lock`"]
    problems: list[str] = []
    body = {k: v for k, v in locked.items() if k != "lock_sha"}
    if locked.get("lock_sha") != _sha(body):
        problems.append(f"{path.name}: lock_sha does not match its content (edited by hand?)")
    if locked.get("schema_sha256") != current["schema_sha256"]:
        problems.append(
            f"schema_sha256 {locked.get('schema_sha256')} -> {current['schema_sha256']} "
            "(the vendored CLM text contract changed; every question_key rotates, D4/D5)"
        )
    cur_tasks: dict[str, Any] = current["tasks"]
    old_tasks: dict[str, Any] = locked.get("tasks") or {}
    for task_id in sorted(set(cur_tasks) | set(old_tasks)):
        if task_id not in old_tasks:
            problems.append(f"{task_id}: new task, not in the lock")
            continue
        if task_id not in cur_tasks:
            problems.append(f"{task_id}: in the lock but no longer framed")
            continue
        cur, old = cur_tasks[task_id], old_tasks[task_id]
        if cur["task_key"] != old.get("task_key"):
            problems.append(
                f"{task_id}: task_key {old.get('task_key')} -> {cur['task_key']} "
                "(task wording or options changed; labels need migration, D1)"
            )
        if cur["shape"] != old.get("shape"):
            problems.append(f"{task_id}: shape {old.get('shape')} -> {cur['shape']}")
        if cur["active_framing"] != old.get("active_framing"):
            problems.append(
                f"{task_id}: active framing {old.get('active_framing')} -> "
                f"{cur['active_framing']} (needs a DESIGN.md amendment)"
            )
        cur_f: dict[str, Any] = cur["framings"]
        old_f: dict[str, Any] = old.get("framings") or {}
        for fid in sorted(set(cur_f) | set(old_f)):
            if fid not in old_f:
                problems.append(f"{task_id}/{fid}: new framing, not in the lock")
            elif fid not in cur_f:
                problems.append(f"{task_id}/{fid}: in the lock but no longer defined")
            elif cur_f[fid]["question_key"] != old_f[fid].get("question_key"):
                changed = sorted(
                    k
                    for k in set(cur_f[fid]) | set(old_f[fid])
                    if cur_f[fid].get(k) != old_f[fid].get(k)
                )
                problems.append(
                    f"{task_id}/{fid}: question_key {old_f[fid].get('question_key')} -> "
                    f"{cur_f[fid]['question_key']} (changed: {', '.join(changed)}; decisions, "
                    "features and artifacts rotate, labels do not, DESIGN D1)"
                )
            elif cur_f[fid].get("control") != old_f[fid].get("control") or cur_f[fid].get(
                "active"
            ) != old_f[fid].get("active"):
                problems.append(f"{task_id}/{fid}: control/active flags changed")
    return problems
