"""Hermetic collaborators for the pipeline and service tests (plan §9 "Fake CLM").

Everything here runs offline: :class:`~mesa_clm.providers.tiered.FakeProvider` for CLM,
``RecordingOLS`` in ``replay`` mode over ``tests/fixtures/ols`` (a missing fixture raises
``ReplayMiss``; nothing is ever recorded live), the shipped policy and a DuckDB sidecar under
``tmp_path``. ``FAKE_SEED``/``FAKE_MODEL`` pick a fake head under which the fixture cards reach
every step (Q1 says "Yes" to some columns, Q3/Q4/Q4b/Q7 and the keep rule all run); the
default seed answers "No" to nearly every column, which would leave the column paths untested.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mesa_clm.cards import ColumnInfo, DatasetCard, load_card
from mesa_clm.clm.http import ClmError, Question, SystemOneResponse
from mesa_clm.config import Config, load_config
from mesa_clm.ols import OLSLayer, RecordingOLS
from mesa_clm.pipeline import Annotator
from mesa_clm.planner.base import PlanResult
from mesa_clm.planner.static_planner import StaticPlanner
from mesa_clm.policy import Policy
from mesa_clm.provenance.store import DuckDBStore
from mesa_clm.providers.tiered import FakeProvider, TieredProvider, fake_fingerprint
from mesa_clm.service import DecisionService

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "fixtures" / "cards"
OLS_DIR = ROOT / "fixtures" / "ols"
CARD_PATHS = sorted(CARDS_DIR.glob("*.md"))

# Fake heads that exercise every pipeline step on the fixture cards (see the module docstring).
FAKE_SEED = 5
FAKE_MODEL = "clm-latest"
# A second head: more column proposals and a D24 child that replaces its parent.
RAW_SEED = 3
RAW_MODEL = "clm-raw"
SERVICE_CARD = "DP1.10003.001.brd_countdata"


def card(name: str) -> DatasetCard:
    return load_card(CARDS_DIR / f"{name}.md")


def config(**env: str) -> Config:
    """The default configuration plus ``MESA_CLM_*`` overrides (``POLICY__PROFILE='dev'`` by
    default), never the process environment.

    ``DECIDER__OLS_RANK_TASKS='[]'`` by default: these helpers exercise every CLM step, the
    ``term.fits`` groups (Q4-Q6, D24's refinement, the second opinion) included, as an audit run
    does. The shipped default decides ``term.fits`` by ``ols_rank`` (K1, DESIGN A1); pass
    ``DECIDER__OLS_RANK_TASKS=...`` (or use :func:`shipped_config`) to run it, as
    ``tests/unit/test_k1_default.py`` does."""
    base = {
        "MESA_CLM_POLICY__PROFILE": "dev",
        "MESA_CLM_OLS__FIXTURES": "replay",
        "MESA_CLM_DECIDER__OLS_RANK_TASKS": "[]",
    }
    return load_config(env={**base, **{f"MESA_CLM_{k}": v for k, v in env.items()}})


def shipped_config(**env: str) -> Config:
    """:func:`config` with the shipped ``decider`` defaults (``ols_rank_tasks`` included)."""
    base = {"MESA_CLM_POLICY__PROFILE": "dev", "MESA_CLM_OLS__FIXTURES": "replay"}
    return load_config(env={**base, **{f"MESA_CLM_{k}": v for k, v in env.items()}})


def replay_ols(max_candidates: int = 12) -> OLSLayer:
    return OLSLayer(RecordingOLS(None, OLS_DIR, "replay"), max_candidates=max_candidates)


def fake_provider(seed: int = FAKE_SEED, model: str = FAKE_MODEL) -> FakeProvider:
    return FakeProvider(seed=seed, model=model)


class DownClient:
    """A clm-serve that is not there: every request fails like a refused connection."""

    model = "clm-latest"

    def __init__(self) -> None:
        self.calls = 0

    def system_one(
        self,
        state: Any,
        questions: dict[str, Question],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        self.calls += 1
        raise ClmError(0, "ConnectError: connection refused")


def down_provider() -> TieredProvider:
    """A provider whose clm-serve is unreachable: every CLM record is ``unavailable``."""
    return TieredProvider(DownClient(), None, fake_fingerprint(), method="fake", name="fake")


class HintPlanner(StaticPlanner):
    """The static plan with extra hints on the columns it would annotate: ``hint(column)``
    returns ``ColumnHint`` fields (``aspect``, ``annotate``, ``ontology``). Queries stay the
    static ones, which the OLS fixture closure covers."""

    name = "hint-static"

    def __init__(self, hint: Callable[[ColumnInfo], dict[str, Any]]) -> None:
        self.hint = hint

    def plan(self, card: DatasetCard) -> PlanResult:
        result = super().plan(card)
        columns = {
            name: h if h.annotate is False else h.model_copy(update=self.hint(card.column(name)))
            for name, h in result.plan.columns.items()
        }
        plan = result.plan.model_copy(update={"columns": columns})
        return PlanResult(plan=plan, planner=self.name)


class AspectPlanner(HintPlanner):
    """An aspect hint on every column the static plan would annotate (a unit column is a
    ``measurement``, anything else ``taxon``), so a run whose Q2 cannot be answered still has
    aspects."""

    name = "aspect-static"

    def __init__(self) -> None:
        super().__init__(lambda col: {"aspect": "measurement" if col.unit else "taxon"})


def annotator(
    store: DuckDBStore | None,
    *,
    provider: Any | None = None,
    cfg: Config | None = None,
    planner: Any | None = None,
    owner: str = "alice",
    **kw: Any,
) -> Annotator:
    cfg = cfg or config()
    return Annotator(
        provider=provider or fake_provider(),
        planner=planner or StaticPlanner(),
        ols=replay_ols(cfg.policy.max_candidates),
        policy=Policy.from_config(cfg.policy),
        cfg=cfg,
        owner=owner,
        store=store,
        **kw,
    )


def service(
    store: DuckDBStore,
    *,
    provider: Any | None = None,
    cfg: Config | None = None,
    planner: Any | None = None,
    **kw: Any,
) -> DecisionService:
    cfg = cfg or config()
    return DecisionService(
        cfg,
        provider=provider or fake_provider(),
        planner=planner or StaticPlanner(),
        ols=replay_ols(cfg.policy.max_candidates),
        policy=Policy.from_config(cfg.policy),
        store=store,
        **kw,
    )


class FakeClaude:
    """An ``anthropic`` client stand-in for the second opinion: ``messages.parse`` answers with
    ``pick(options)`` (the options are read off the structured-output model it is given)."""

    def __init__(self, pick: Any) -> None:
        self.pick = pick
        self.calls = 0

    @property
    def messages(self) -> FakeClaude:
        return self

    def parse(self, **kw: Any) -> Any:
        from types import SimpleNamespace
        from typing import get_args

        self.calls += 1
        options = get_args(kw["output_format"].model_fields["answer"].annotation)
        return SimpleNamespace(
            parsed_output=SimpleNamespace(answer=self.pick(options)),
            usage=SimpleNamespace(input_tokens=42, output_tokens=3),
            model="claude-fake",
            stop_reason="end_turn",
        )


def annotated_template(directory: Path, **kw: Any) -> tuple[Path, Any]:
    """A sidecar file holding one annotated run of :data:`SERVICE_CARD` (owner ``alice``, actor
    ``agent-x``), for tests that copy it instead of annotating again."""
    path = directory / "prov.duckdb"
    svc = service(DuckDBStore(path), **kw)
    run = svc.annotate(card(SERVICE_CARD), "agent-x", owner="alice")
    return path, run


def copy_store(template: Path, directory: Path) -> DuckDBStore:
    import shutil

    target = directory / "prov.duckdb"
    shutil.copy(template, target)
    return DuckDBStore(target)
