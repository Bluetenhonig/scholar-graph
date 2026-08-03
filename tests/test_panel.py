"""The AutoGen review board, and the adapter that keeps it on our rails."""

from __future__ import annotations

import pytest
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from scholar_graph.config import Settings
from scholar_graph.domain import DraftReport, PanelCritique, ResearchReport
from scholar_graph.llm.budget import BudgetTracker
from scholar_graph.panel.model_client import ProviderChatCompletionClient, _split_messages
from scholar_graph.panel.review_board import panel_requires_revision, review
from scholar_graph.service import ResearchService
from scholar_graph.state import Deps
from tests.conftest import DEFAULT_RESPONSES, FakeAPIProvider, FakeSearchService

DRAFT = DraftReport(
    summary="Sparse attention and quantisation both cut cost [S1][S2].",
    body="Body text with citations [S1].",
    limitations="No latency data.",
)


class TestMessageFlattening:
    def test_system_and_dialogue_are_separated(self) -> None:
        system, dialogue = _split_messages(
            [
                SystemMessage(content="You are a reviewer."),
                UserMessage(content="Review this.", source="user"),
                AssistantMessage(content="Looks fine.", source="methodologist"),
            ]
        )
        assert system == "You are a reviewer."
        assert "user: Review this." in dialogue
        assert "methodologist: Looks fine." in dialogue

    def test_empty_messages_are_dropped(self) -> None:
        system, dialogue = _split_messages([UserMessage(content="   ", source="user")])
        assert system == ""
        assert dialogue == ""

    def test_multiple_system_messages_are_joined(self) -> None:
        system, _ = _split_messages(
            [SystemMessage(content="First."), SystemMessage(content="Second.")]
        )
        assert "First." in system and "Second." in system


class TestAdapter:
    async def test_calls_route_through_our_provider(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        provider = FakeAPIProvider(settings, budget, {"panel:test": ["a critique"]})
        client = ProviderChatCompletionClient(
            provider, model="claude-haiku-4-5", purpose="panel:test"
        )
        result = await client.create(
            [SystemMessage(content="sys"), UserMessage(content="hi", source="user")]
        )

        assert result.content == "a critique"
        assert result.finish_reason == "stop"
        # The point of the adapter: panel spend lands in the run's budget.
        assert budget.spent_usd > 0

    async def test_usage_accumulates_across_calls(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        provider = FakeAPIProvider(settings, budget, {"panel:test": ["x"]})
        client = ProviderChatCompletionClient(
            provider, model="claude-haiku-4-5", purpose="panel:test"
        )
        messages = [UserMessage(content="hi", source="user")]
        await client.create(messages)
        await client.create(messages)

        assert client.total_usage().prompt_tokens == 1600
        assert client.actual_usage().prompt_tokens == 800

    async def test_tools_are_refused_loudly(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        client = ProviderChatCompletionClient(
            FakeAPIProvider(settings, budget), model="claude-haiku-4-5", purpose="panel:test"
        )
        with pytest.raises(NotImplementedError):
            await client.create([UserMessage(content="hi", source="user")], tools=[object()])

    async def test_stream_yields_the_completed_result(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        provider = FakeAPIProvider(settings, budget, {"panel:test": ["streamed"]})
        client = ProviderChatCompletionClient(
            provider, model="claude-haiku-4-5", purpose="panel:test"
        )
        chunks = [c async for c in client.create_stream([UserMessage(content="hi", source="user")])]
        assert chunks[-1].content == "streamed"  # type: ignore[union-attr]


class TestReviewBoard:
    async def test_panel_returns_one_critique_per_reviewer(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        provider = FakeAPIProvider(settings, budget, DEFAULT_RESPONSES)
        critiques = await review(
            question="How can inference cost be reduced?",
            draft=DRAFT,
            provider=provider,
            model="claude-haiku-4-5",
        )

        reviewers = {c.reviewer for c in critiques}
        assert {"methodologist", "editor", "adjudicator"} <= reviewers

    async def test_adjudicator_verdict_is_the_binding_one(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        provider = FakeAPIProvider(
            settings,
            budget,
            {
                **DEFAULT_RESPONSES,
                "panel:methodologist": ["I would revise this heavily."],
                "panel:adjudicator": ["accept - the concerns are stylistic."],
            },
        )
        critiques = await review(
            question="q", draft=DRAFT, provider=provider, model="claude-haiku-4-5"
        )
        # A reviewer saying "revise" does not by itself trigger a revision.
        assert not panel_requires_revision(critiques)

    async def test_revise_verdict_is_detected(self) -> None:
        critiques = [
            PanelCritique(reviewer="editor", verdict="accept", comments="fine"),
            PanelCritique(reviewer="adjudicator", verdict="revise", comments="fix S2"),
        ]
        assert panel_requires_revision(critiques)


class TestPanelInTheGraph:
    async def test_enabled_panel_attaches_critiques_to_the_report(self, settings: Settings) -> None:
        with_panel = settings.model_copy(update={"enable_review_panel": True})
        budget = BudgetTracker(with_panel.max_usd_per_run)
        deps = Deps(
            settings=with_panel,
            provider=FakeAPIProvider(with_panel, budget, DEFAULT_RESPONSES),
            search=FakeSearchService(with_panel),
            budget=budget,
        )
        report = await ResearchService(with_panel).run("How to cut inference cost?", deps=deps)

        assert isinstance(report, ResearchReport)
        assert len(report.critiques) >= 3
        assert report.verification.passed

    async def test_panel_failure_does_not_sink_the_run(self, settings: Settings) -> None:
        with_panel = settings.model_copy(update={"enable_review_panel": True})
        budget = BudgetTracker(with_panel.max_usd_per_run)
        # No canned responses for the panel purposes: every panel call raises.
        responses = {k: v for k, v in DEFAULT_RESPONSES.items() if not k.startswith("panel:")}
        deps = Deps(
            settings=with_panel,
            provider=FakeAPIProvider(with_panel, budget, responses),
            search=FakeSearchService(with_panel),
            budget=budget,
        )
        report = await ResearchService(with_panel).run("How to cut inference cost?", deps=deps)

        assert isinstance(report, ResearchReport)
        assert report.critiques == []
        assert any("panel" in w.lower() for w in report.warnings)
        # The actual report survived intact.
        assert report.verification.passed
