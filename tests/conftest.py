"""Shared hermetic fixtures: no network, no GPU, DuckDB files under tmp_path.

Module-specific fixtures live next to their tests; imports of mesa_clm modules happen
inside fixtures so one unfinished module never breaks collection of the others.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CARDS_DIR = FIXTURES / "cards"
OLS_DIR = FIXTURES / "ols"
NEON_EVAL_ROOT = FIXTURES / "neon-avu-eval"

CARD_TEXT = """# Dataset: DP1.10003.001 / brd_countdata
Product: DP1.10003.001 — Breeding landbird point counts. Count, distance from observer, and taxonomic identification of breeding landbirds observed during point counts
Source: NEON (National Ecological Observatory Network) Data API, basic package, stacked across site-months.
Sites: HARV (Harvard Forest & Quabbin Watershed NEON, Massachusetts, domain D01 Northeast; temperate deciduous/mixed forest) and SRER (Santa Rita Experimental Range NEON, Arizona, domain D14 Desert Southwest; semi-arid desert grassland/shrubland).
Rows: 15484; months covered: 2020-04 to 2024-06 (12 distinct months).

## Columns (name | NEON description | type | unit | profile)
- uid | Unique ID within NEON database; an identifier for the record | string | - | (identifier; not profiled)
- siteID | NEON site code | string | - | 2 distinct; top: SRER (9416), HARV (6068)
- startDate | The start date-time or interval during which an event occurred | dateTime | - | 898 distinct; top: 2024-05-03T12:19Z (36)
- scientificName | Scientific name, associated with the taxonID. | string | - | 179 distinct; top: Campylorhynchus brunneicapillus (1098), Zenaida macroura (1010)
- observerDistance | Radial distance between the observer and the individual(s) being observed | real | meter | numeric, n=14670, min=1, median=59, max=999
- detectionMethod | How the individual(s) was (were) first detected by the observer | string | - | 10 distinct; top: singing (7644), calling (5600), visual (958)
- identificationHistoryID | Identifier for linking records related to this identification history | string | - | all blank
"""


@pytest.fixture
def card_text() -> str:
    return CARD_TEXT


@pytest.fixture
def card() -> Any:
    from mesa_clm.cards import parse_card

    return parse_card(CARD_TEXT)


@pytest.fixture
def fixture_cards() -> list[Path]:
    return sorted(CARDS_DIR.glob("*.md"))


@pytest.fixture(autouse=True)
def _offline_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """The doctor never probes this host in the hermetic suite: no serving home, refused
    connections, missing commands (``health.ServeProbes.offline``). Tests that exercise the
    serving checks pass their own ``ServeProbes``."""
    from mesa_clm import health

    monkeypatch.setattr(health, "default_probes", health.ServeProbes.offline)
