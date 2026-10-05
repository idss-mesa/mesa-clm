"""Shared hermetic fixtures: no network, no GPU, DuckDB files under tmp_path.

Module-specific fixtures live next to their tests; imports of mesa_clm modules happen
inside fixtures so one unfinished module never breaks collection of the others.

The BLAS thread pools are pinned to one thread here, at import and before numpy loads
(``os.environ.setdefault``: a value set in the environment wins): the fitters are small dense
problems where thread fan-out only costs, and a loaded host makes a multi-threaded suite
slower, not faster.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from collections.abc import Iterator  # noqa: E402  (after the thread pins above)
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

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


# The shipped default ``artifacts.dir`` (``config.DEFAULT_ARTIFACTS_DIR``), which the hermetic
# session replaces (below) and the opt-in ``engine`` tests get back.
SHIPPED_ARTIFACTS_DIR = "~/.mesa/clm/artifacts"


@pytest.fixture(scope="session", autouse=True)
def _session_artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """This host's promoted artifacts (``~/.mesa/clm/artifacts/.../CURRENT.json``, what ``learn
    promote`` wrote) never reach the hermetic suite: the default ``artifacts.dir`` is an empty
    per-session path that is never created. Session-scoped, one value for every configuration
    built in the session (module-scoped fixtures build theirs before any function-scoped
    fixture runs, and ``config_sha256`` covers the field); ``_offline_serving`` restores the
    shipped value for the ``engine`` tests."""
    from mesa_clm import config

    mp = pytest.MonkeyPatch()
    mp.setattr(
        config,
        "DEFAULT_ARTIFACTS_DIR",
        str(tmp_path_factory.getbasetemp() / ".mesa-clm-artifacts"),
    )
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def _offline_serving(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hermetic suite never sees this host's serving stack. The doctor gets no serving
    home, refused connections and missing commands (``health.ServeProbes.offline``; tests that
    exercise the serving checks pass their own ``ServeProbes``), and the serving home
    (``serving.DEFAULT_HOME``) is an empty per-test path, so neither the installed serving lock
    nor the default key files under ``~/.mesa/clm/secrets`` (config's default ``*_api_key_file``)
    can reach a test (nor this host's promoted ``CURRENT.json``: ``_session_artifacts_dir``),
    and the encoder-container check of ``features build`` and ``annotate``
    finds no docker (``providers.live.docker_argv``), nor the keyed clients' port-owner check
    this host's listeners (``net.PROC_NET``). Tests that need a serving home point
    ``DEFAULT_HOME`` elsewhere. The live ``engine`` tests keep the real home and docker."""
    from mesa_clm import config, health, net, serving
    from mesa_clm.providers import live

    monkeypatch.setattr(health, "default_probes", health.ServeProbes.offline)
    if request.node.get_closest_marker("engine") is None:
        monkeypatch.setattr(serving, "DEFAULT_HOME", str(tmp_path / ".mesa-clm-serving-home"))
        # The keyed clients' port-owner check (net.assert_listener_owner) reads no listener of
        # this host; tests that exercise it point PROC_NET at their own tcp/tcp6 files.
        monkeypatch.setattr(net, "PROC_NET", tmp_path / ".mesa-clm-proc-net")
        # The container check of features build and annotate (live.container_check) sees no
        # docker; tests that exercise it inject their own runner and docker argv.
        monkeypatch.setattr(live, "docker_argv", lambda args: None)
    else:
        # The live engine tests see the real artifacts of this host, as the operator does.
        monkeypatch.setattr(config, "DEFAULT_ARTIFACTS_DIR", SHIPPED_ARTIFACTS_DIR)
