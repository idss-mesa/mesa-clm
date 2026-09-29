#!/usr/bin/env python3
"""Record mesa-anyjev's cards, states and registry as the mesa-clm parity fixture (DESIGN D23).

Run with mesa-anyjev's own interpreter, never mesa-clm's (mesa-clm imports nothing from
``mesa_anyjev``, DESIGN U1; the sibling checkout is only read), from the mesa-clm root::

    <mesa-anyjev checkout>/.venv/bin/python scripts/gen_anyjev_parity.py \\
        [--anyjev-root <mesa-anyjev checkout>]

``--anyjev-root`` defaults to the checkout the interpreter's ``mesa_anyjev`` is installed from.
For each of the seven fixture cards in ``tests/fixtures/cards`` it calls every public builder
of ``mesa_anyjev.states`` with deterministic inputs: every column, every site, and a fixed set
of synthetic candidates, AVUs, sibling lists, values and DataCite texts that hit the truncation
caps (``MAX_PROFILE``, ``MAX_DESCRIPTION``, ``MAX_SIBLINGS``, the five-synonym cap, the
``MAX_DESCRIPTION * 4`` DataCite text cap) and the ``str()``/``bool()``/``int()`` coercions. It
writes

- ``tests/fixtures/anyjev_parity.json``: per record ``{builder, card, kwargs, state,
  state_sha256, ordered_sha256}``, plus per card its parsed model, header, sha256
  and name, a set of grammar edge cases, the registry vocabularies and helper outputs, and the
  mesa-anyjev commit;
- ``tests/fixtures/anyjev_questions.lock.json``: a byte-for-byte copy of mesa-anyjev's
  ``questions.lock.json`` (the task keys that mesa-clm's ``task_key`` must reproduce, D1).

Encoding, so the fixture stays small and ``tests/unit/test_anyjev_parity.py`` can rebuild each
state with mesa-clm from the same inputs:

- a ``kwargs`` value ``{"$card": NAME}``, ``{"$column": NAME}``, ``{"$site": CODE}`` or
  ``{"$synthetic": KEY}`` stands for that card, a column or site of the record's card, or
  ``synthetic[KEY]``; every other value is passed as is;
- inside ``state``, a ``"card"`` value equal to the record card's ``card_header`` is stored as
  ``{"$card_header": NAME}`` (the header is recorded once per card). ``state_sha256`` and
  ``ordered_sha256`` are always over the full, unabbreviated state; ``ordered_sha256`` is the
  sha256 of the compact JSON *without* ``sort_keys``, so it pins the builder key order at
  every depth even though the file itself is written with sorted keys.

``--check`` rebuilds both files in memory and exits 1 when either committed file differs.
The output is deterministic: fixed inputs, sorted keys, no timestamps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import mesa_anyjev
    from mesa_anyjev import cards as a_cards
    from mesa_anyjev import registry as a_registry
    from mesa_anyjev import states as a_states
except ImportError:  # pragma: no cover - operator error
    sys.exit(
        "gen_anyjev_parity.py must run under mesa-anyjev's interpreter, e.g.\n"
        "  <mesa-anyjev checkout>/.venv/bin/python scripts/gen_anyjev_parity.py "
        "[--anyjev-root <mesa-anyjev checkout>]"
    )

FORMAT = "mesa-clm/anyjev-parity/1"
REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
PARITY_NAME = "anyjev_parity.json"
LOCK_NAME = "anyjev_questions.lock.json"

# -- deterministic synthetic inputs ---------------------------------------------------------

_LONG_DESC = (
    "The température de l'air — measured 2 m above the ground in µmol·m⁻²·s⁻¹ units — is a "
    "physical quality inhering in a bearer by virtue of the bearer's thermal energy. "
) * 4  # ~440 characters, non-ASCII throughout: exercises MAX_DESCRIPTION and ensure_ascii
_CAP_DESC = ("abcdefghij" * 30)[: a_states.MAX_DESCRIPTION]
_LONG_TEXT = (
    "Breeding landbird point counts — Zählungen, conteos, décomptes — at HARV and SRER. "
) * 20  # ~1,660 characters: over the DataCite text cap of MAX_DESCRIPTION * 4

SYNTHETIC: dict[str, Any] = {
    "cand_pato_distance": {
        "label": "distance",
        "curie": "PATO:0000040",
        "ontology_id": "pato",
        "description": "A size quality inhering in a bearer by virtue of the bearer's extension.",
        "synonyms": ["length", "extent"],
        "has_children": True,
    },
    "cand_long_unicode": {
        "label": "température de l'air",
        "curie": "ENVO:09200001",
        "ontology_id": "envo",
        "description": _LONG_DESC,
        "synonyms": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        "has_children": False,
    },
    "cand_sparse": {"label": "Aves", "curie": "NCBITaxon:8782"},
    "cand_coerced": {
        "label": 42,
        "curie": "ENVO:00000428",
        "ontology_id": "envo",
        "description": None,
        "synonyms": [1, "two", 3.5],
        "has_children": 1,
    },
    "cand_exact_cap": {
        "label": "point count",
        "curie": "OBI:0000070",
        "ontology_id": "obi",
        "description": _CAP_DESC,
        "synonyms": ["a", "b", "c", "d", "e"],
        "has_children": False,
        "iri": "http://purl.obolibrary.org/obo/OBI_0000070",
    },
    "avu_basic": {"attribute": "pato.distance", "value": "distance", "unit": "PATO:0000040"},
    "avu_extra_keys": {
        "attribute": "envo.biome",
        "value": "desert grassland – Sonoran",
        "unit": "ENVO:01000178",
        "curie": "ENVO:01000178",
        "note": "dropped by avu_state",
    },
    "siblings_none": [],
    "siblings_3": [
        {"attribute": "ncbitaxon.aves", "value": "Aves", "unit": "NCBITaxon:8782"},
        {"attribute": "obi.point_count", "value": "point count", "unit": "OBI:0000070", "x": "y"},
        {"attribute": "uo.meter", "value": "observerDistance", "unit": "UO:0000008"},
    ],
    "siblings_30": [
        {
            "attribute": f"pato.quality_{i:02d}",
            "value": f"valeur {i:02d} é",
            "unit": f"PATO:{i:07d}",
        }
        for i in range(30)
    ],
    "text_short": "Count, distance from observer, and taxonomic identification of landbirds.",
    "text_long_unicode": _LONG_TEXT,
    "text_exact_cap": ("0123456789" * 120)[: a_states.MAX_DESCRIPTION * 4],
}

CANDIDATES = (
    "cand_pato_distance",
    "cand_long_unicode",
    "cand_sparse",
    "cand_coerced",
    "cand_exact_cap",
)
AVUS = ("avu_basic", "avu_extra_keys")
TEXTS = ("text_short", "text_long_unicode", "text_exact_cap")
CHOOSER_VALUES = ("distance", "Zenaida macroura", "température", "")
CHOOSER_ONTOLOGIES = ("pato", "envo", "ncbitaxon", "obi", "uo")

# Grammar edge cases parsed by both implementations (results or error messages recorded).
PARSE_CASES: dict[str, str] = {
    "minimal_header_only": "# Dataset: DP1.00001.001 / tbl\n",
    "not_a_card": "# Something else\nProduct: nothing\n",
    "empty": "",
    "hyphen_product_crlf_one_site": (
        "# Dataset: DP1.00098.001 / RH_30min\r\n"
        "Product: DP1.00098.001 - Relative humidity. Humidity above the canopy\r\n"
        "Source: NEON\r\n"
        "Sites: SRER (Santa Rita Experimental Range NEON, Arizona, domain D14 Desert Southwest;"
        " semi-arid desert grassland/shrubland).\r\n"
        "Rows: 12; months covered: 2023-01 to 2023-02 (2 distinct months).\r\n"
        "\r\n"
        "## Columns (name | NEON description | type | unit | profile)\r\n"
        "- RHMean | Mean relative humidity | real | percent | numeric, n=12, min=1, max=99\r\n"
        "- RHFinalQF | Quality flag | integer | - | 2 distinct; top: 0 (11) | 1 (1)\r\n"
        "- remarks | Free text | string | - | all blank\r\n"
    ),
    "en_dash_three_sites_odd_lines": (
        "  #   Dataset:DP4.00200.001/nsae  \n"
        "Product: DP4.00200.001 – Bundled eddy covariance data (no period)\n"
        "Sites: HARV (Harvard Forest, Massachusetts, domain D01 Northeast; forest) and "
        "SRER (Santa Rita, Arizona, domain D14 Desert Southwest; shrubland) and "
        "ONAQ (Onaqui, Utah, domain D15 Great Basin;  sagebrush ).\n"
        "Rows: 7; months covered: 2024-01\n"
        "Rows: 9; months covered: 2024-01 to 2024-03 (3 distinct months).\n"
        "- only | three | pipes\n"
        "-   padded   |  spaced description  |  real  |  -  |  numeric  \n"
        "- empty||||\n"
    ),
}

IDENTIFIER_CASES: tuple[dict[str, str], ...] = (
    {"name": "taxonID", "description": "", "dtype": "string", "unit": "", "profile": "x"},
    {"name": "plotID", "description": "", "dtype": "string", "unit": "", "profile": "x"},
    {"name": "Remarks", "description": "", "dtype": "string", "unit": "", "profile": "x"},
    {"name": "release", "description": "", "dtype": "string", "unit": "", "profile": "x"},
    {"name": "collectDate", "description": "", "dtype": "Date", "unit": "", "profile": "x"},
    {"name": "setDate", "description": "", "dtype": "dateTime", "unit": "", "profile": "x"},
    {"name": "measuredBy", "description": "", "dtype": "string", "unit": "", "profile": "x"},
    {
        "name": "individualCount",
        "description": "",
        "dtype": "integer",
        "unit": "",
        "profile": "All blank",
    },
    {
        "name": "sampleCondition",
        "description": "",
        "dtype": "string",
        "unit": "",
        "profile": "(not profiled)",
    },
    {
        "name": "scientificName",
        "description": "",
        "dtype": "string",
        "unit": "",
        "profile": "3 distinct",
    },
    {
        "name": "barcode",
        "description": "",
        "dtype": "string",
        "unit": "",
        "profile": "blank mostly",
    },
)

PREFIX_CASES = (
    "ncbitaxon:8782",
    "NCBITAXON:9606",
    "NCBITaxon:1",
    "envo:01000178",
    "GO:0007601",
    "go:1",
    "Chebi:15377",
    "nocolon",
    "",
    ":empty",
    "obo:PATO_0000001:extra",
)
IN_PLAY = ("envo", "iao", "ncbitaxon", "uo")


# -- helpers --------------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ordered_sha256(state: dict[str, Any]) -> str:
    payload = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _builders() -> list[str]:
    """Every public function defined in ``mesa_anyjev.states`` except the hash itself."""
    return sorted(
        name
        for name, fn in vars(a_states).items()
        if inspect.isfunction(fn)
        and fn.__module__ == a_states.__name__
        and not name.startswith("_")
        and name != "state_sha256"
    )


class _Recorder:
    def __init__(self, cards: dict[str, Any], headers: dict[str, dict[str, Any]]) -> None:
        self.cards = cards
        self.headers = headers
        self.records: list[dict[str, Any]] = []
        self._seen: set[str] = set()

    def _resolve(self, value: Any, card: Any) -> Any:
        if isinstance(value, dict) and len(value) == 1:
            ((tag, ref),) = value.items()
            if tag == "$card":
                return self.cards[ref]
            if tag == "$column":
                return card.column(ref)
            if tag == "$site":
                return next(s for s in card.sites if s.code == ref)
            if tag == "$synthetic":
                return copy.deepcopy(SYNTHETIC[ref])
        return value

    def add(self, builder: str, card_name: str | None, **kwargs: Any) -> None:
        ident = json.dumps([builder, card_name, kwargs], sort_keys=True)
        if ident in self._seen:
            return
        self._seen.add(ident)
        card = self.cards[card_name] if card_name else None
        args = {k: self._resolve(v, card) for k, v in kwargs.items()}
        state = getattr(a_states, builder)(**args)
        stored = dict(state)
        if card_name and stored.get("card") == self.headers[card_name]:
            stored["card"] = {"$card_header": card_name}
        self.records.append(
            {
                "id": f"{len(self.records):05d}",
                "builder": builder,
                "card": card_name,
                "kwargs": kwargs,
                "state": stored,
                "state_sha256": a_states.state_sha256(state),
                "ordered_sha256": _ordered_sha256(state),
            }
        )


def _record_card(rec: _Recorder, ci: int, name: str) -> None:
    card = rec.cards[name]
    ref = {"$card": name}
    reg = a_registry.ONTOLOGY_REGISTRY
    aspects = a_registry.ASPECTS
    rec.add("card_header", name, card=ref)
    # Every column goes through column_state (the shared column dict, MAX_PROFILE) and
    # candidate_state (term.fits); the builders that only add a small tail to the same column
    # dict sample every second or third column to keep the fixture small.
    for i, col in enumerate(card.columns):
        c = {"$column": col.name}
        rec.add("column_state", name, card=ref, col=c)
        rec.add(
            "candidate_state",
            name,
            card=ref,
            scope="column",
            target=c,
            aspect=aspects[(i + 2) % len(aspects)],
            candidate={"$synthetic": CANDIDATES[i % len(CANDIDATES)]},
            n_candidates=i % 13,
        )
        if i % 3 == 0:
            e = reg[(i + ci) % len(reg)]
            rec.add(
                "ontology_state",
                name,
                card=ref,
                col=c,
                aspect=aspects[i % len(aspects)],
                ontology_id=e.id,
                option_text=e.option_text,
                # reverse-sorted on purpose: the builder must sort
                aspects=sorted(e.aspects, reverse=True),
            )
        elif i % 3 == 1:
            rec.add(
                "value_kind_state",
                name,
                card=ref,
                col=c,
                term={"$synthetic": CANDIDATES[(i + 1) % len(CANDIDATES)]},
                aspect=aspects[(i + 3) % len(aspects)],
            )
        else:
            if i % 9 == 2:
                sib = "siblings_30"
            elif i % 2:
                sib = "siblings_3"
            else:
                sib = "siblings_none"
            rec.add(
                "avu_state",
                name,
                card=ref,
                avu={"$synthetic": AVUS[i % len(AVUS)]},
                term={"$synthetic": CANDIDATES[(i + 2) % len(CANDIDATES)]},
                scope="column",
                target_name=col.name,
                siblings={"$synthetic": sib},
            )
    # every ontology of the registry for the first column (the column.ontology_fits group)
    first = card.columns[0]
    for e in reg:
        rec.add(
            "ontology_state",
            name,
            card=ref,
            col={"$column": first.name},
            aspect="other",
            ontology_id=e.id,
            option_text=e.option_text,
            aspects=sorted(e.aspects),
        )
    for site in card.sites:
        s = {"$site": site.code}
        for k, cand in enumerate(CANDIDATES):
            rec.add(
                "candidate_state",
                name,
                card=ref,
                scope="site",
                target=s,
                aspect="environment",
                candidate={"$synthetic": cand},
                n_candidates=k + 1,
            )
        rec.add(
            "avu_state",
            name,
            card=ref,
            avu={"$synthetic": "avu_extra_keys"},
            term={"$synthetic": "cand_long_unicode"},
            scope="site",
            target_name=site.code,
            siblings={"$synthetic": "siblings_3"},
        )
    for cand in CANDIDATES:
        rec.add(
            "candidate_state",
            name,
            card=ref,
            scope="dataset",
            target=None,
            aspect="taxon",
            candidate={"$synthetic": cand},
            n_candidates=0,
        )
    rec.add(
        "avu_state",
        name,
        card=ref,
        avu={"$synthetic": "avu_basic"},
        term={"$synthetic": "cand_sparse"},
        scope="dataset",
        target_name=None,
        siblings={"$synthetic": "siblings_30"},
    )
    for e in reg:
        rec.add("dataset_ontology_state", name, card=ref, ontology_option=e.option_text)
    # DataCite with a card: the short text for every vocabulary, one long text per card
    for j, vocab in enumerate(a_registry.DATACITE_VOCABULARIES):
        members = a_registry.DATACITE_VOCABULARIES[vocab]
        text = {"$synthetic": TEXTS[0]}
        if j == ci % len(a_registry.DATACITE_VOCABULARIES):
            text = {"$synthetic": TEXTS[1 + ci % 2]}
            rec.add("datacite_state", name, card=ref, vocabulary=vocab, text=text)
        rec.add(
            "datacite_state",
            name,
            card=ref,
            vocabulary=vocab,
            text=text,
            value=members[(j + ci) % len(members)],
        )


def _record_card_free(rec: _Recorder) -> None:
    for k, cand in enumerate(CANDIDATES):
        for m in range(2):
            rec.add(
                "chooser_state",
                None,
                ontology_id=CHOOSER_ONTOLOGIES[(k + m) % len(CHOOSER_ONTOLOGIES)],
                value=CHOOSER_VALUES[(k + m) % len(CHOOSER_VALUES)],
                candidate={"$synthetic": cand},
                n_candidates=k * 3 + m,
            )
    for j, vocab in enumerate(a_registry.DATACITE_VOCABULARIES):
        members = a_registry.DATACITE_VOCABULARIES[vocab]
        rec.add(
            "datacite_state",
            None,
            card=None,
            vocabulary=vocab,
            text={"$synthetic": TEXTS[j % len(TEXTS)]},
        )
        rec.add(
            "datacite_state",
            None,
            card=None,
            vocabulary=vocab,
            text={"$synthetic": "text_long_unicode"},
            value=members[-1],
        )
    # ``str(text)`` coercion of a non-string text
    rec.add("datacite_state", None, card=None, vocabulary="DateType", text=2024, value="Issued")


def _registry() -> dict[str, Any]:
    r = a_registry
    return {
        "ASPECTS": list(r.ASPECTS),
        "ASPECT_OPTIONS": list(r.ASPECT_OPTIONS),
        "ONTOLOGY_REGISTRY": [
            {
                "id": e.id,
                "curie_prefix": e.curie_prefix,
                "option_text": e.option_text,
                "aspects": sorted(e.aspects),
            }
            for e in r.ONTOLOGY_REGISTRY
        ],
        "ONTOLOGY_OPTIONS": list(r.ONTOLOGY_OPTIONS),
        "VALUE_KINDS": list(r.VALUE_KINDS),
        "DATACITE_VOCABULARIES": {k: list(v) for k, v in r.DATACITE_VOCABULARIES.items()},
        "allowed_for_aspect": {
            a: sorted(r.allowed_for_aspect(a)) for a in (*r.ASPECTS, "not_an_aspect")
        },
        "mask_for_aspect": {a: [bool(x) for x in r.mask_for_aspect(a)] for a in r.ASPECTS},
        "mask_for_aspect_in_play": {
            "in_play": list(IN_PLAY),
            "masks": {
                a: [bool(x) for x in r.mask_for_aspect(a, frozenset(IN_PLAY))] for a in r.ASPECTS
            },
        },
        "prefix_of": {c: r.prefix_of(c) for c in PREFIX_CASES},
        "entry": {i: r.entry(i).id for i in ("ENVO", "NCBITaxon", "uo", "GenEpiO")},
    }


def _parse_cases() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, text in PARSE_CASES.items():
        try:
            out[key] = {"text": text, "parsed": a_cards.parse_card(text).model_dump(mode="json")}
        except ValueError as exc:
            out[key] = {"text": text, "error": {"type": type(exc).__name__, "message": str(exc)}}
    return out


def build(cards_dir: Path, anyjev_root: Path) -> tuple[bytes, bytes]:
    """The parity fixture and the verbatim lock copy, as bytes."""
    lock_bytes = (anyjev_root / "questions.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    files = sorted(cards_dir.glob("*.md"))
    if len(files) != 7:
        raise SystemExit(f"expected the 7 fixture cards in {cards_dir}, found {len(files)}")
    cards: dict[str, Any] = {}
    card_files: dict[str, str] = {}
    for f in files:
        card = a_cards.load_card(f)
        cards[card.name] = card
        card_files[card.name] = f.name
    headers = {n: a_states.card_header(c) for n, c in cards.items()}
    rec = _Recorder(cards, headers)
    for ci, name in enumerate(sorted(cards)):
        _record_card(rec, ci, name)
    _record_card_free(rec)

    builders = _builders()
    counts = {b: sum(1 for r in rec.records if r["builder"] == b) for b in builders}
    missing = [b for b, n in counts.items() if n == 0]
    if missing:
        raise SystemExit(f"no records for mesa_anyjev.states builders {missing}: extend the script")

    src = Path(a_states.__file__).resolve().parent
    status = _git(
        anyjev_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "src",
        "questions.lock.json",
    )
    fixture = {
        "format": FORMAT,
        "generator": "scripts/gen_anyjev_parity.py",
        "anyjev": {
            "commit": _git(anyjev_root, "rev-parse", "HEAD"),
            "dirty": bool(status),
            "version": mesa_anyjev.__version__,
            "sources_sha256": {
                p: _sha256_bytes((src / p).read_bytes())
                for p in ("cards.py", "states.py", "registry.py")
            },
        },
        "constants": {
            "MAX_PROFILE": a_states.MAX_PROFILE,
            "MAX_DESCRIPTION": a_states.MAX_DESCRIPTION,
            "MAX_SIBLINGS": a_states.MAX_SIBLINGS,
            "IDENTIFIER_SUFFIXES": list(a_cards.IDENTIFIER_SUFFIXES),
        },
        "builders": builders,
        "counts": counts,
        "questions_lock": {
            "file": LOCK_NAME,
            "sha256": _sha256_bytes(lock_bytes),
            "lock_sha": lock["lock_sha"],
            "keys": {q: v["key"] for q, v in lock["questions"].items()},
        },
        "synthetic": SYNTHETIC,
        "cards": {
            n: {
                "file": card_files[n],
                "name": c.name,
                "sha256": c.sha256,
                "card_header": headers[n],
                "parsed": c.model_dump(mode="json"),
                "identifiers": [col.name for col in c.columns if a_cards.is_identifier(col)],
            }
            for n, c in cards.items()
        },
        "parse_cases": _parse_cases(),
        "is_identifier_cases": [
            {"column": dict(col), "result": a_cards.is_identifier(a_cards.ColumnInfo(**col))}
            for col in IDENTIFIER_CASES
        ],
        "registry": _registry(),
        "records": rec.records,
    }
    return _dump(fixture), lock_bytes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--anyjev-root", type=Path, default=None, help="mesa-anyjev checkout")
    ap.add_argument("--cards", type=Path, default=FIXTURES / "cards", help="fixture cards dir")
    ap.add_argument("--out-dir", type=Path, default=FIXTURES, help="where the fixtures go")
    ap.add_argument("--check", action="store_true", help="compare, do not write")
    args = ap.parse_args(argv)
    root = args.anyjev_root or Path(mesa_anyjev.__file__).resolve().parents[2]
    if not (root / "questions.lock.json").is_file():
        ap.error(
            f"{root} is not a mesa-anyjev checkout (no questions.lock.json); pass --anyjev-root"
        )
    parity, lock = build(args.cards, root)
    targets = {args.out_dir / PARITY_NAME: parity, args.out_dir / LOCK_NAME: lock}
    if args.check:
        stale = [p for p, data in targets.items() if not p.is_file() or p.read_bytes() != data]
        for p in stale:
            print(f"stale: {p}", file=sys.stderr)
        return 1 if stale else 0
    for path, data in targets.items():
        path.write_bytes(data)
        print(f"wrote {path} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
