"""Dataset cards: the neon-avu-eval markdown grammar parsed into a frozen model.

A card (``neon-avu-eval/scripts/make_cards.py``) is::

    # Dataset: <productCode> / <table>
    Product: <code> — <title>. <description>
    Source: ...
    Sites: <CODE> (<name>, <state>, domain <Dxx> <domain name>; <habitat>) and <CODE> (...).
    Rows: <n>; months covered: <from> to <to> (<k> distinct months).

    ## Columns (name | NEON description | type | unit | profile)
    - <name> | <description> | <dtype> | <unit or -> | <profile>

Ported verbatim in behaviour from mesa-anyjev ``cards.py`` (``6159281``; DESIGN D23): the
parsed model feeds ``states.py``, whose ``state_sha256`` must stay byte-identical so labels
and states recorded by mesa-anyjev remain joinable (``tests/unit/test_anyjev_parity.py``).
neon-ducklake writes its table cards (``sites/<SITE>/cards/anyjev/<DP>.<table>.md``) in the
same grammar, with the regexes copied, so the neon adapter reads them through this parser.

Only the standard library and pydantic are used.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_HEADER = re.compile(r"^#\s*Dataset:\s*(?P<product>\S+)\s*/\s*(?P<table>\S+)\s*$")
_PRODUCT = re.compile(r"^Product:\s*(?P<code>\S+)\s*[—-]+\s*(?P<title>[^.]+)\.?\s*(?P<desc>.*)$")
_ROWS = re.compile(r"^Rows:\s*(?P<rows>\d+);\s*months covered:\s*(?P<a>\S+)\s+to\s+(?P<b>\S+)")
_SITE = re.compile(
    r"(?P<code>[A-Z]{4})\s*\((?P<name>[^,]+),\s*(?P<state>[^,]+),\s*domain\s+(?P<domain>D\d{2})"
    r"\s*(?P<dname>[^;]*);\s*(?P<habitat>[^)]*)\)"
)
_COLUMN = re.compile(
    r"^-\s*(?P<name>[^|]+)\|(?P<desc>[^|]*)\|(?P<dtype>[^|]*)\|(?P<unit>[^|]*)\|(?P<profile>.*)$"
)

IDENTIFIER_SUFFIXES = ("id", "uid", "code", "identifiedby", "recordedby", "measuredby")


class SiteInfo(BaseModel):
    """One NEON site from the card's ``Sites:`` line."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    name: str
    state: str
    domain: str
    domain_name: str
    habitat: str


class ColumnInfo(BaseModel):
    """One ``- name | description | dtype | unit | profile`` column line."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    dtype: str
    unit: str  # '' when the card shows '-'
    profile: str


class DatasetCard(BaseModel):
    """A parsed dataset card; ``sha256`` is over the exact card text (UTF-8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product_code: str
    table: str
    product_title: str
    product_description: str
    source: str
    sites: list[SiteInfo] = Field(default_factory=list)
    rows: int
    months_from: str
    months_to: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    sha256: str

    @property
    def name(self) -> str:
        """``<productCode>.<table>``, the card's identity everywhere (runs, labels, folds)."""
        return f"{self.product_code}.{self.table}"

    def column(self, name: str) -> ColumnInfo:
        """The column called ``name``; ``KeyError`` when the card has none."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


def card_sha256(text: str) -> str:
    """sha256 hex digest of the card text as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_card(text: str) -> DatasetCard:
    """Parse one card; unknown lines are ignored, a missing ``# Dataset:`` header raises
    ``ValueError`` (the message never echoes card content)."""
    product_code = table = ""
    title = desc = source = ""
    rows = 0
    months_from = months_to = ""
    sites: list[SiteInfo] = []
    columns: list[ColumnInfo] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := _HEADER.match(line):
            product_code, table = m["product"], m["table"]
        elif m := _PRODUCT.match(line):
            title, desc = m["title"].strip(), m["desc"].strip()
        elif line.startswith("Source:"):
            source = line[len("Source:") :].strip()
        elif line.startswith("Sites:"):
            sites = [
                SiteInfo(
                    code=s["code"],
                    name=s["name"].strip(),
                    state=s["state"].strip(),
                    domain=s["domain"],
                    domain_name=s["dname"].strip(),
                    habitat=s["habitat"].strip(),
                )
                for s in _SITE.finditer(line)
            ]
        elif m := _ROWS.match(line):
            rows, months_from, months_to = int(m["rows"]), m["a"], m["b"]
        elif m := _COLUMN.match(line):
            unit = m["unit"].strip()
            columns.append(
                ColumnInfo(
                    name=m["name"].strip(),
                    description=m["desc"].strip(),
                    dtype=m["dtype"].strip(),
                    unit="" if unit == "-" else unit,
                    profile=m["profile"].strip(),
                )
            )
    if not product_code or not table:
        raise ValueError("not a dataset card: missing '# Dataset: <product> / <table>' header")
    return DatasetCard(
        product_code=product_code,
        table=table,
        product_title=title,
        product_description=desc,
        source=source,
        sites=sites,
        rows=rows,
        months_from=months_from,
        months_to=months_to,
        columns=columns,
        sha256=card_sha256(text),
    )


def load_card(path: str | Path) -> DatasetCard:
    """Read and parse a card file (UTF-8)."""
    return parse_card(Path(path).read_text(encoding="utf-8"))


def is_identifier(col: ColumnInfo) -> bool:
    """The deterministic pre-filter: identifiers, bookkeeping and timestamps are never
    annotated (recorded as a ``rule`` decision, never a model call)."""
    name = col.name.lower()
    if name in {"uid", "remarks", "publicationdate", "release"}:
        return True
    if name.endswith(IDENTIFIER_SUFFIXES) and name not in {"taxonid"}:
        return True
    if col.dtype.lower() in {"datetime", "date"}:
        return True
    profile = col.profile.lower()
    return profile.startswith("all blank") or "not profiled" in profile


# The grammar's numeric dtypes, normalised by :func:`is_numeric_dtype` (lower case, no spaces or
# underscores): NEON's variables files write ``real``, ``integer``, ``unsigned integer`` and
# ``signed integer``; the generic names are kept so a card from another adapter counts the same
# way. ``dateTime``, ``date``, ``string`` and ``uri`` are not numeric.
NUMERIC_DTYPES: frozenset[str] = frozenset(
    {
        "real",
        "integer",
        "unsignedinteger",
        "signedinteger",
        "int",
        "bigint",
        "smallint",
        "float",
        "double",
        "decimal",
        "numeric",
        "number",
    }
)


def is_numeric_dtype(dtype: str) -> bool:
    """Whether a card's ``dtype`` names a numeric kind (:data:`NUMERIC_DTYPES`, compared without
    case, spaces or underscores: ``unsigned integer`` and ``unsignedInteger`` both count)."""
    return "".join(dtype.lower().split()).replace("_", "") in NUMERIC_DTYPES


def is_numeric(col: ColumnInfo) -> bool:
    """Whether a column is numeric by its dtype (DESIGN A8 reads this for Q2's fallback)."""
    return is_numeric_dtype(col.dtype)
