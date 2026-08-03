"""Shared test fixtures.

The central idea: fake the network boundary and nothing else. ``FakeAPIProvider``
overrides only ``_call_api``, so cassette keying, cost accounting, budget
enforcement and structured-output parsing all execute for real in tests.
Stubbing at a higher level would test the stub instead of the system.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from scholar_graph.config import LLMMode, Settings, set_settings
from scholar_graph.domain import Document
from scholar_graph.llm.budget import BudgetTracker
from scholar_graph.llm.pricing import estimate_cost
from scholar_graph.llm.provider import LLMProvider, LLMResponse
from scholar_graph.state import Deps
from scholar_graph.tools import SearchService
from scholar_graph.tools.http import HttpFetcher

# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

DOC_A = Document(
    source_id=Document.make_source_id("openalex", "W1"),
    provider="openalex",
    external_id="W1",
    title="Sparse attention reduces transformer inference cost",
    abstract=(
        "We show that sparse attention patterns reduce inference cost by up to 40 percent "
        "on long sequences while preserving downstream accuracy within one point."
    ),
    authors=("Ada Lovelace", "Alan Turing"),
    published=date(2024, 3, 1),
    venue="Journal of Efficient Models",
    url="https://doi.org/10.0000/sparse",
    citation_count=120,
)

DOC_B = Document(
    source_id=Document.make_source_id("arxiv", "2401.00001"),
    provider="arxiv",
    external_id="2401.00001",
    title="Quantisation trade-offs in large language model serving",
    abstract=(
        "Eight-bit quantisation halves memory footprint with negligible quality loss, "
        "whereas four-bit quantisation degrades reasoning benchmarks substantially."
    ),
    authors=("Grace Hopper",),
    published=date(2024, 1, 2),
    venue="arXiv",
    url="https://arxiv.org/abs/2401.00001",
)

DOC_IRRELEVANT = Document(
    source_id=Document.make_source_id("openalex", "W99"),
    provider="openalex",
    external_id="W99",
    title="Soil nitrogen dynamics in temperate grassland",
    abstract="Nitrogen mineralisation rates were measured across three grassland sites.",
    authors=("Carl Linnaeus",),
    published=date(2020, 6, 1),
    venue="Ecology Letters",
)


# --------------------------------------------------------------------------
# Canned model output, keyed by the node that asks for it
# --------------------------------------------------------------------------

PLAN_JSON = json.dumps(
    {
        "interpretation": "Which techniques cut LLM inference cost without hurting quality?",
        "sub_questions": ["Attention-level techniques", "Numerical precision techniques"],
        "search_queries": ["sparse attention inference cost", "llm quantisation quality"],
        "success_criteria": [
            "Names at least two distinct cost-reduction techniques",
            "States the quality trade-off for each technique",
        ],
    }
)

SCREEN_JSON = json.dumps(
    {
        "verdicts": [
            {"source_id": DOC_A.source_id, "relevant": True, "reason": "Directly on topic."},
            {"source_id": DOC_B.source_id, "relevant": True, "reason": "Covers quantisation."},
            {"source_id": DOC_IRRELEVANT.source_id, "relevant": False, "reason": "Soil science."},
        ]
    }
)

EXTRACT_JSON = json.dumps(
    {
        "notes": [
            {
                "source_id": DOC_A.source_id,
                "claim": "Sparse attention cuts inference cost substantially.",
                "quote": "sparse attention patterns reduce inference cost by up to 40 percent",
            },
            {
                "source_id": DOC_B.source_id,
                "claim": "Eight-bit quantisation is nearly lossless; four-bit is not.",
                "quote": (
                    "Eight-bit quantisation halves memory footprint with negligible quality loss"
                ),
            },
        ]
    }
)

COVERAGE_SATISFIED_JSON = json.dumps(
    {
        "satisfied_criteria": [
            "Names at least two distinct cost-reduction techniques",
            "States the quality trade-off for each technique",
        ],
        "unmet_criteria": [],
        "follow_up_queries": [],
    }
)

COVERAGE_GAP_JSON = json.dumps(
    {
        "satisfied_criteria": ["Names at least two distinct cost-reduction techniques"],
        "unmet_criteria": ["States the quality trade-off for each technique"],
        "follow_up_queries": ["inference cost quality trade-off benchmark"],
    }
)

GOOD_DRAFT_JSON = json.dumps(
    {
        "summary": "Sparse attention and 8-bit quantisation both cut inference cost [S1][S2].",
        "body": (
            "Sparse attention patterns reduce inference cost by up to 40 percent on long "
            "sequences, and the accuracy cost is under one point, which makes it the cheaper "
            "of the two interventions to adopt in production serving stacks [S1].\n\n"
            "Quantisation is the other lever. Eight-bit quantisation halves the memory "
            "footprint with negligible quality loss, while four-bit quantisation degrades "
            "reasoning benchmarks substantially and should be adopted with care [S2]."
        ),
        "limitations": "Neither source reports latency under production serving load.",
    }
)

BAD_DRAFT_JSON = json.dumps(
    {
        "summary": "Sparse attention eliminates all inference cost [S7].",
        "body": (
            "An entirely uncited paragraph asserting that inference cost is a solved problem "
            "and that no trade-offs remain for practitioners deploying these systems today, "
            "which is a claim no retrieved source supports in any form.\n\n"
            "A second claim resting on a source number that was never retrieved [S7]."
        ),
        "limitations": "None.",
    }
)

PANEL_TEXT = "The draft is well supported by its citations. No blocking concerns. accept"
ADJUDICATOR_TEXT = "accept - the citations resolve and the trade-offs are stated."


DEFAULT_RESPONSES: dict[str, list[str]] = {
    "plan": [PLAN_JSON],
    "screen": [SCREEN_JSON],
    "extract": [EXTRACT_JSON],
    "coverage": [COVERAGE_SATISFIED_JSON],
    "synthesize": [GOOD_DRAFT_JSON],
    "revise": [GOOD_DRAFT_JSON],
    "panel:methodologist": [PANEL_TEXT],
    "panel:editor": [PANEL_TEXT],
    "panel:adjudicator": [ADJUDICATOR_TEXT],
}


class FakeAPIProvider(LLMProvider):
    """A provider whose only fake is the HTTP call to Anthropic.

    Responses are drawn per ``purpose``; the last entry repeats, so a node
    called twice (a second search round, a revision) keeps working without
    the test having to predict the call count.
    """

    def __init__(
        self,
        settings: Settings,
        budget: BudgetTracker,
        responses: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(settings, budget)
        self.responses = {k: list(v) for k, v in (responses or DEFAULT_RESPONSES).items()}
        self.calls: list[str] = []

    async def _call_api(self, request: dict[str, Any], purpose: str) -> LLMResponse:
        self.calls.append(purpose)
        queue = self.responses.get(purpose)
        if not queue:
            raise AssertionError(f"No canned response for purpose {purpose!r}")
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        # Price it through the real table rather than inventing a dollar
        # figure, so a recorded cost and its replayed re-computation agree.
        return LLMResponse(
            text=text,
            cost=estimate_cost(request["model"], input_tokens=800, output_tokens=200),
            stop_reason="end_turn",
        )


class FakeSearchService(SearchService):
    """Returns a fixed corpus without touching the network."""

    def __init__(self, settings: Settings, documents: list[Document] | None = None) -> None:
        super().__init__(settings)
        self.documents = documents if documents is not None else [DOC_A, DOC_B, DOC_IRRELEVANT]
        self.queries_seen: list[str] = []

    async def search(self, queries: list[str], *, per_query: int = 6) -> list[Document]:
        self.queries_seen.extend(queries)
        return list(self.documents)

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    configured = Settings(
        # `live` so requests flow through _call_api, which FakeAPIProvider
        # replaces. Cassette read/write paths get their own dedicated tests
        # rather than being implicitly exercised by every graph test.
        llm_mode=LLMMode.live,
        anthropic_api_key="test-key-not-used",
        cassette_dir=tmp_path / "cassettes",
        # In-memory by default: most tests do not exercise cross-process
        # durability, and a leaked sqlite connection per test is just noise.
        # The one test that cares about durability opts into a real file.
        checkpoint_db=None,
        max_usd_per_run=1.0,
        require_approval_over_usd=1000.0,  # off by default; tests opt in
        max_search_rounds=2,
        max_documents=10,
        enable_review_panel=False,
        log_level="WARNING",
    )
    set_settings(configured)
    yield configured
    set_settings(Settings())


@pytest.fixture
def budget(settings: Settings) -> BudgetTracker:
    return BudgetTracker(settings.max_usd_per_run)


@pytest.fixture
def provider(settings: Settings, budget: BudgetTracker) -> FakeAPIProvider:
    return FakeAPIProvider(settings, budget)


@pytest.fixture
def deps(settings: Settings, budget: BudgetTracker, provider: FakeAPIProvider) -> Deps:
    return Deps(
        settings=settings,
        provider=provider,
        search=FakeSearchService(settings),
        budget=budget,
    )


@pytest.fixture
def fetcher(settings: Settings) -> HttpFetcher:
    live = settings.model_copy(update={"llm_mode": LLMMode.live})
    return HttpFetcher(live)
