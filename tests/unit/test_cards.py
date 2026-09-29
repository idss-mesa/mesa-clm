"""The dataset-card grammar (``mesa_clm.cards``), ported from mesa-anyjev ``tests/test_cards.py``
plus the fixture cards, model immutability and the identifier pre-filter's edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mesa_clm.cards import (
    IDENTIFIER_SUFFIXES,
    ColumnInfo,
    DatasetCard,
    SiteInfo,
    card_sha256,
    is_identifier,
    load_card,
    parse_card,
)


def test_parse_card_header_sites_rows_columns(card: DatasetCard) -> None:
    assert card.product_code == "DP1.10003.001"
    assert card.table == "brd_countdata"
    assert card.name == "DP1.10003.001.brd_countdata"
    assert card.product_title == "Breeding landbird point counts"
    assert card.product_description.startswith("Count, distance from observer")
    assert card.source.startswith("NEON (National Ecological Observatory Network)")
    assert [s.code for s in card.sites] == ["HARV", "SRER"]
    assert card.sites[0].state == "Massachusetts"
    assert card.sites[0].domain_name == "Northeast"
    assert card.sites[1].domain == "D14"
    assert card.sites[1].habitat == "semi-arid desert grassland/shrubland"
    assert card.rows == 15484
    assert (card.months_from, card.months_to) == ("2020-04", "2024-06")
    assert len(card.columns) == 7
    assert card.column("observerDistance").unit == "meter"
    assert card.column("observerDistance").dtype == "real"
    assert card.column("siteID").unit == ""
    assert len(card.sha256) == 64


def test_sha256_is_over_the_exact_text(card_text: str, card: DatasetCard) -> None:
    assert card.sha256 == card_sha256(card_text)
    assert parse_card(card_text + "\n").sha256 != card.sha256  # any byte change moves it
    assert parse_card(card_text + "\n").columns == card.columns


def test_identifier_rule(card: DatasetCard) -> None:
    flagged = {c.name for c in card.columns if is_identifier(c)}
    assert flagged == {"uid", "siteID", "startDate", "identificationHistoryID"}
    assert not is_identifier(card.column("scientificName"))


@pytest.mark.parametrize(
    ("name", "dtype", "profile", "expected"),
    [
        ("taxonID", "string", "10 distinct", False),  # the one *ID column that is annotated
        ("plotID", "string", "10 distinct", True),
        ("Remarks", "string", "10 distinct", True),
        ("publicationDate", "string", "3 distinct", True),
        ("release", "string", "1 distinct", True),
        ("measuredBy", "string", "3 distinct", True),
        ("recordedBy", "string", "3 distinct", True),
        ("collectDate", "Date", "10 distinct", True),
        ("setDate", "dateTime", "10 distinct", True),
        ("individualCount", "integer", "All blank", True),
        ("sampleCondition", "string", "(identifier; not profiled)", True),
        ("scientificName", "string", "179 distinct; top: x (1)", False),
        ("tagValue", "string", "blank mostly", False),  # only a leading "all blank" counts
    ],
)
def test_identifier_rule_edges(name: str, dtype: str, profile: str, expected: bool) -> None:
    col = ColumnInfo(name=name, description="", dtype=dtype, unit="", profile=profile)
    assert is_identifier(col) is expected
    assert IDENTIFIER_SUFFIXES == ("id", "uid", "code", "identifiedby", "recordedby", "measuredby")


def test_not_a_card_raises_without_echoing_content() -> None:
    with pytest.raises(ValueError, match="Dataset") as info:
        parse_card("# Something else\nProduct: SECRET-CONTENT\n")
    assert "SECRET-CONTENT" not in str(info.value)
    with pytest.raises(ValueError, match="Dataset"):
        parse_card("")


def test_unknown_column_raises_keyerror(card: DatasetCard) -> None:
    with pytest.raises(KeyError):
        card.column("nope")


def test_models_are_frozen_and_forbid_extras(card: DatasetCard) -> None:
    with pytest.raises(ValidationError):
        card.rows = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        card.columns[0].name = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ColumnInfo(name="a", description="", dtype="", unit="", profile="", extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SiteInfo(  # type: ignore[call-arg]
            code="HARV", name="", state="", domain="", domain_name="", habitat="", x=1
        )


def test_grammar_tolerates_crlf_hyphen_and_dash_units() -> None:
    text = (
        "# Dataset: DP1.00098.001 / RH_30min\r\n"
        "Product: DP1.00098.001 - Relative humidity. Humidity above the canopy\r\n"
        "Sites: SRER (Santa Rita Experimental Range NEON, Arizona, domain D14 Desert Southwest;"
        " semi-arid desert grassland/shrubland).\r\n"
        "Rows: 12; months covered: 2023-01 to 2023-02 (2 distinct months).\r\n"
        "## Columns (name | NEON description | type | unit | profile)\r\n"
        "- RHMean | Mean relative humidity | real | percent | numeric, n=12\r\n"
        "- RHFinalQF | Quality flag | integer | - | 2 distinct; top: 0 (11) | 1 (1)\r\n"
    )
    card = parse_card(text)
    assert card.product_title == "Relative humidity"
    assert card.product_description == "Humidity above the canopy"
    assert card.source == ""
    assert [s.code for s in card.sites] == ["SRER"]
    assert (card.rows, card.months_from, card.months_to) == (12, "2023-01", "2023-02")
    assert card.column("RHMean").unit == "percent"
    assert card.column("RHFinalQF").unit == ""
    assert card.column("RHFinalQF").profile == "2 distinct; top: 0 (11) | 1 (1)"  # rest of line


def test_fixture_cards_load(fixture_cards: list[Path]) -> None:
    assert len(fixture_cards) == 7
    for path in fixture_cards:
        card = load_card(path)
        assert card.name == path.stem
        assert card.sha256 == card_sha256(path.read_text(encoding="utf-8"))
        assert [s.code for s in card.sites] == ["HARV", "SRER"]
        assert card.rows > 0 and card.columns
        assert len({c.name for c in card.columns}) == len(card.columns)
