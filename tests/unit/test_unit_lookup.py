"""Defect (a) regression (DESIGN U1): ``unit_candidate`` matches the exact key first, then the
longest whole-word suffix, never a dict-order ``endswith`` (mesa-anyjev ``ols.py:277`` sent
centimeter to meter, kilogram to gram and kilometersPerHour to hour); a compound unit without
its own row misses instead of resolving to its trailing base unit (metersPerSecond is not
``second``); and every ``UNIT_TABLE`` CURIE is checked against its recorded ``get_term`` fixture
(kilometer was UO:0000009, kilogram)."""

from __future__ import annotations

from pathlib import Path

import pytest

import mesa_clm.ols as ols
from mesa_clm.ols import UNIT_TABLE, OLSLayer, RecordingOLS, iri_for, lookup_unit, normalize_unit

OLS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ols"


@pytest.fixture
def layer() -> OLSLayer:
    return OLSLayer(RecordingOLS(None, OLS_DIR, "replay"))


@pytest.mark.parametrize(
    ("unit", "curie", "label"),
    [
        ("centimeter", "UO:0000015", "centimeter"),
        ("kilometer", "UO:0010066", "kilometer"),
        ("kilogram", "UO:0000009", "kilogram"),
        ("milligram", "UO:0000022", "milligram"),
        ("kilometer per hour", "UO:0010008", "kilometer per hour"),
        ("degree Celsius", "UO:0000027", "degree Celsius"),
        ("percent", "UO:0000187", "percent"),
        ("meter", "UO:0000008", "meter"),
        ("millimeter", "UO:0000016", "millimeter"),
        ("gram", "UO:0000021", "gram"),
        ("second", "UO:0000010", "second"),
        ("hour", "UO:0000032", "hour"),
        ("liter", "UO:0000099", "liter"),
        ("milliliter", "UO:0000098", "milliliter"),
        # The NEON card spellings (tests/fixtures/cards): camelCase, plurals, bare "celsius".
        ("kilometersPerHour", "UO:0010008", "kilometer per hour"),
        # Compound units NEON cards use that UO has an exact term for (recorded get_term).
        ("metersPerSecond", "UO:0000094", "meter per second"),
        ("meter per second", "UO:0000094", "meter per second"),
        ("wattsPerSquareMeter", "UO:0000155", "watt per square meter"),
        ("milligramsPerKilogram", "UO:0000308", "milligram per kilogram"),
        ("squareMeter", "UO:0000080", "square meter"),
        ("cubicMeter", "UO:0000096", "cubic meter"),
        ("decimalDegree", "UO:0000185", "degree"),
        ("nominalDay", "UO:0000033", "day"),
        ("number", "UO:0000189", "count unit"),
        ("count", "UO:0000189", "count unit"),
        ("celsius", "UO:0000027", "degree Celsius"),
        ("degreeCelsius", "UO:0000027", "degree Celsius"),
        ("degrees_celsius", "UO:0000027", "degree Celsius"),
        ("Celsius degrees", "UO:0000027", "degree Celsius"),
        ("  Meter ", "UO:0000008", "meter"),
    ],
)
def test_unit_candidate(layer: OLSLayer, unit: str, curie: str, label: str) -> None:
    cand = layer.unit_candidate(unit)
    assert cand is not None, unit
    assert (cand.curie, cand.label) == (curie, label)
    assert cand.iri == iri_for(curie) and cand.ontology_id == "uo"
    assert cand.query == unit and cand.rank == 0 and cand.description == ""


@pytest.mark.parametrize(
    "unit",
    [
        "furlong",
        "micrometer",  # a substring endswith gave meter (UO:0000008): wrong unit
        "millisecond",
        "nanogram",
        "microliter",
        "km/h",
        "",
        "   ",
        "-",
        # Compound units without a UO row: the trailing base unit is the wrong CURIE (the
        # pipeline takes a table hit as the only uo candidate), so they must reach the search.
        "micromolesPerSquareMeterPerSecond",  # was second, UO:0000010
        "millimolesPerLiter",  # was liter, UO:0000099
        "nanomolesPerGram",  # was gram, UO:0000021
        "microsiemensPerCentimeter",  # was centimeter, UO:0000015
        "gramsPerSquareMeter",  # was meter, UO:0000008
        "celsiusSquared",  # was degree Celsius, UO:0000027
        "metersSquared",
        "cubicCentimeter",
        "cubic meters per second",
        "meters per hour",  # no row: not kilometer per hour, not hour
    ],
)
def test_unit_candidate_misses_fall_through_to_search(layer: OLSLayer, unit: str) -> None:
    assert layer.unit_candidate(unit) is None
    assert lookup_unit(unit) is None


def test_compound_units_never_resolve_to_the_trailing_base_unit() -> None:
    """Every ``X per Y`` / ``square X`` / ``X squared`` spelling either has its own row (the
    exact UO term) or misses; none returns the row of its last word."""
    for unit in (
        "metersPerSecond",
        "wattsPerSquareMeter",
        "micromolesPerSquareMeterPerSecond",
        "milligramsPerKilogram",
        "millimolesPerLiter",
        "nanomolesPerGram",
        "microsiemensPerCentimeter",
        "squareMeter",
        "cubicMeter",
        "celsiusSquared",
    ):
        got = lookup_unit(unit)
        units = [w for w in normalize_unit(unit).split(" ") if w not in ols._COMPOUND_WORDS]
        base = UNIT_TABLE.get(units[-1])
        assert base is not None, f"{unit}: the test needs a base-unit row for {units[-1]!r}"
        assert got is not base, (unit, got)
        if got is not None:
            assert normalize_unit(unit) in UNIT_TABLE  # an exact row, never a suffix hit
    assert lookup_unit("metersPerSecond") == UNIT_TABLE["meter per second"]
    assert lookup_unit("wattsPerSquareMeter") == UNIT_TABLE["watt per square meter"]
    assert lookup_unit("micromolesPerSquareMeterPerSecond") is None
    assert lookup_unit("celsiusSquared") is None  # the celsius containment rule does not apply
    # A plain modifier in front of a compound row still reaches that row (as decimalDegree).
    assert lookup_unit("meanMetersPerSecond") == UNIT_TABLE["meter per second"]


def test_kilometer_is_not_kilogram() -> None:
    assert UNIT_TABLE["kilogram"]["curie"] == "UO:0000009"
    assert UNIT_TABLE["kilometer"]["curie"] == "UO:0010066"
    assert UNIT_TABLE["kilometer"]["curie"] != UNIT_TABLE["kilogram"]["curie"]
    # Prefix units never resolve to their base unit any more.
    assert lookup_unit("centimeter") is not UNIT_TABLE["meter"]
    assert lookup_unit("kilogram") is not UNIT_TABLE["gram"]
    assert lookup_unit("milligram") is not UNIT_TABLE["gram"]
    assert lookup_unit("kilometersPerHour") is not UNIT_TABLE["hour"]


def test_lookup_order_is_exact_then_longest_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the shortest names first in the table, a dict-order ``endswith`` would return
    ``hour``; the exact key and the longest suffix win regardless of order."""
    table = {
        "hour": {"label": "hour", "curie": "UO:0000032"},
        "per hour": {"label": "bogus", "curie": "UO:0000000"},
        "kilometer per hour": {"label": "kilometer per hour", "curie": "UO:0010008"},
        "meter": {"label": "meter", "curie": "UO:0000008"},
    }
    monkeypatch.setattr(ols, "UNIT_TABLE", table)
    assert lookup_unit("kilometer per hour") is table["kilometer per hour"]
    assert lookup_unit("mean kilometer per hour") is table["kilometer per hour"]
    assert lookup_unit("meanKilometerPerHour") is table["kilometer per hour"]
    assert lookup_unit("hour") is table["hour"]
    assert lookup_unit("workHour") is table["hour"]
    assert lookup_unit("kilometer") is None  # not a whole-word suffix of "meter"
    assert lookup_unit("squareMeter") is None  # compound modifier: never the "meter" suffix


def test_normalize_unit() -> None:
    assert normalize_unit("kilometersPerHour") == "kilometers per hour"
    assert normalize_unit("decimalDegree") == "decimal degree"
    assert normalize_unit("degree Celsius") == "degree celsius"
    assert normalize_unit("degrees_celsius") == "degrees celsius"
    assert normalize_unit("degree-Celsius") == "degree celsius"
    assert normalize_unit("  Meter\t") == "meter"
    assert normalize_unit("UO:0000008") == "uo:0000008"
    assert normalize_unit("") == ""


def test_table_keys_are_normalised_and_labels_consistent() -> None:
    for name, spec in UNIT_TABLE.items():
        assert normalize_unit(name) == name, name
        assert set(spec) == {"label", "curie"} and spec["curie"].startswith("UO:")
    by_curie: dict[str, set[str]] = {}
    for spec in UNIT_TABLE.values():
        by_curie.setdefault(spec["curie"], set()).add(spec["label"])
    assert all(len(labels) == 1 for labels in by_curie.values()), by_curie


def test_every_table_curie_matches_its_recorded_get_term(layer: OLSLayer) -> None:
    """The closure recorded once against EMBL-EBI OLS4 (plan M0): each CURIE's ``get_term``
    label equals the table label, so no row can point at another unit again."""
    replay = layer.client
    assert isinstance(replay, RecordingOLS)
    seen: set[str] = set()
    for name, spec in UNIT_TABLE.items():
        term = replay.get_term("uo", iri_for(spec["curie"]))
        assert term is not None, f"{name}: no term for {spec['curie']}"
        assert term["label"] == spec["label"], (name, spec, term["label"])
        assert term["curie"] == spec["curie"] and term["ontologyId"] == "uo"
        assert not term["isRoot"]
        cand = layer.unit_candidate(name)
        assert cand is not None and cand.iri == term["iri"]
        seen.add(spec["curie"])
    assert replay.missed == [] and replay.misses == 0
    assert len(seen) == 23  # 18 single units + 5 compound units UO names exactly


def test_kilometer_get_term_regression(layer: OLSLayer) -> None:
    replay = layer.client
    assert isinstance(replay, RecordingOLS)
    cand = layer.unit_candidate("kilometer")
    assert cand is not None
    term = replay.get_term("uo", cand.iri)
    assert term is not None and term["label"] == "kilometer" == cand.label
    kilogram = replay.get_term("uo", iri_for("UO:0000009"))
    assert kilogram is not None and kilogram["label"] == "kilogram"
