"""End-to-end graph behaviour: routing, loops, bounds, and human-in-the-loop."""

from __future__ import annotations

import json

from scholar_graph.config import Settings
from scholar_graph.domain import (
    CostBreakdown,
    CoverageAssessment,
    ResearchReport,
    VerificationReport,
)
from scholar_graph.graph import (
    route_after_approval,
    route_after_coverage,
    route_after_panel,
    route_after_verification,
)
from scholar_graph.llm.budget import BudgetTracker
from scholar_graph.service import PendingApproval, ResearchService
from scholar_graph.state import Deps, ResearchState
from tests.conftest import (
    BAD_DRAFT_JSON,
    COVERAGE_GAP_JSON,
    COVERAGE_SATISFIED_JSON,
    DEFAULT_RESPONSES,
    DOC_IRRELEVANT,
    GOOD_DRAFT_JSON,
    FakeAPIProvider,
    FakeSearchService,
)

QUESTION = "How can inference cost be reduced for large language models?"


def _deps(settings: Settings, responses: dict[str, list[str]] | None = None) -> Deps:
    budget = BudgetTracker(settings.max_usd_per_run)
    merged = {**DEFAULT_RESPONSES, **(responses or {})}
    return Deps(
        settings=settings,
        provider=FakeAPIProvider(settings, budget, merged),
        search=FakeSearchService(settings),
        budget=budget,
    )


# --------------------------------------------------------------------------
# Routing (pure functions, tested directly)
# --------------------------------------------------------------------------


class TestRouting:
    def test_rejected_run_ends(self) -> None:
        assert route_after_approval(ResearchState(approval_decision="reject")) == "__end__"

    def test_approved_run_searches(self) -> None:
        assert route_after_approval(ResearchState(approval_decision="approve")) == "search"

    def test_coverage_gap_triggers_another_search(self) -> None:
        state = ResearchState(
            coverage=CoverageAssessment(
                satisfied_criteria=[], unmet_criteria=["gap"], follow_up_queries=["q"]
            ),
            search_round=1,
            max_search_rounds=3,
        )
        assert route_after_coverage(state) == "search"

    def test_search_round_cap_is_respected_even_with_gaps(self) -> None:
        state = ResearchState(
            coverage=CoverageAssessment(
                satisfied_criteria=[], unmet_criteria=["gap"], follow_up_queries=["q"]
            ),
            search_round=3,
            max_search_rounds=3,
        )
        assert route_after_coverage(state) == "synthesize"

    def test_no_follow_up_queries_means_stop(self) -> None:
        state = ResearchState(
            coverage=CoverageAssessment(
                satisfied_criteria=[], unmet_criteria=["gap"], follow_up_queries=[]
            ),
            search_round=1,
            max_search_rounds=3,
        )
        assert route_after_coverage(state) == "synthesize"

    def test_clean_verification_skips_revision(self) -> None:
        clean = VerificationReport(
            checked_markers=2, valid_markers=2, grounded_notes=1, total_notes=1
        )
        assert route_after_verification(ResearchState(verification=clean)) == "panel"

    def test_failed_verification_revises(self) -> None:
        from scholar_graph.domain import CitationIssue

        failing = VerificationReport(
            checked_markers=1,
            valid_markers=0,
            grounded_notes=0,
            total_notes=1,
            issues=(CitationIssue(kind="unknown_marker", detail="bad", marker=9),),
        )
        state = ResearchState(verification=failing, revision_round=0)
        assert route_after_verification(state) == "revise"

    def test_revision_cap_ships_the_draft_with_warnings(self) -> None:
        from scholar_graph.domain import CitationIssue

        failing = VerificationReport(
            checked_markers=1,
            valid_markers=0,
            grounded_notes=0,
            total_notes=1,
            issues=(CitationIssue(kind="unknown_marker", detail="bad", marker=9),),
        )
        state = ResearchState(verification=failing, revision_round=99)
        assert route_after_verification(state) == "panel"

    def test_panel_cannot_loop_past_the_revision_cap(self) -> None:
        from scholar_graph.domain import PanelCritique

        critiques = [PanelCritique(reviewer="adjudicator", verdict="revise", comments="fix it")]
        assert route_after_panel(ResearchState(critiques=critiques, revision_round=99)) == "__end__"


# --------------------------------------------------------------------------
# Full runs
# --------------------------------------------------------------------------


class TestHappyPath:
    async def test_produces_a_verified_report(self, settings: Settings) -> None:
        deps = _deps(settings)
        report = await ResearchService(settings).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        assert report.verification.passed
        assert report.verification.citation_precision == 1.0
        assert report.verification.groundedness == 1.0
        assert "[S1]" in report.summary
        assert report.cost.usd > 0
        assert report.cost.calls >= 4

    async def test_irrelevant_sources_are_screened_out_of_the_citation_list(
        self, settings: Settings
    ) -> None:
        deps = _deps(settings)
        report = await ResearchService(settings).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        cited_ids = {doc.source_id for doc in report.sources}
        assert DOC_IRRELEVANT.source_id not in cited_ids
        assert len(report.sources) == 2

    async def test_markdown_render_lists_every_source(self, settings: Settings) -> None:
        report = await ResearchService(settings).run(QUESTION, deps=_deps(settings))
        assert isinstance(report, ResearchReport)
        markdown = report.to_markdown()
        assert "## Sources" in markdown
        for i, doc in enumerate(report.sources, start=1):
            assert f"**[S{i}]**" in markdown
            assert doc.title in markdown


class TestSearchLoop:
    async def test_coverage_gap_causes_a_second_search_round(self, settings: Settings) -> None:
        # First assessment reports a gap, second is satisfied.
        deps = _deps(settings, {"coverage": [COVERAGE_GAP_JSON, COVERAGE_SATISFIED_JSON]})
        report = await ResearchService(settings).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        assert report.search_rounds == 2

    async def test_search_rounds_are_hard_capped(self, settings: Settings) -> None:
        # Coverage never satisfied: only the cap can stop this.
        capped = settings.model_copy(update={"max_search_rounds": 2})
        deps = _deps(capped, {"coverage": [COVERAGE_GAP_JSON]})
        report = await ResearchService(capped).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        assert report.search_rounds == 2


class TestVerificationLoop:
    async def test_a_bad_draft_is_revised_into_a_good_one(self, settings: Settings) -> None:
        deps = _deps(
            settings,
            {"synthesize": [BAD_DRAFT_JSON], "revise": [GOOD_DRAFT_JSON]},
        )
        report = await ResearchService(settings).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        assert "revise" in deps.provider.calls  # type: ignore[attr-defined]
        assert report.verification.passed

    async def test_an_unfixable_draft_ships_with_warnings_rather_than_looping(
        self, settings: Settings
    ) -> None:
        # Revision keeps returning the same broken draft.
        deps = _deps(
            settings,
            {"synthesize": [BAD_DRAFT_JSON], "revise": [BAD_DRAFT_JSON]},
        )
        report = await ResearchService(settings).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        assert not report.verification.passed
        assert any("citation issue" in w for w in report.warnings)
        # Bounded: two revisions, not infinite.
        revise_calls = [c for c in deps.provider.calls if c == "revise"]  # type: ignore[attr-defined]
        assert len(revise_calls) == 2


class TestHumanInTheLoop:
    async def test_expensive_run_suspends_for_approval(self, settings: Settings) -> None:
        gated = settings.model_copy(update={"require_approval_over_usd": 0.0001})
        result = await ResearchService(gated).run(QUESTION, deps=_deps(gated))

        assert isinstance(result, PendingApproval)
        assert result.payload["reason"] == "cost_approval_required"
        assert result.payload["projected_usd"] > 0

    async def test_approval_resumes_the_run_to_completion(self, settings: Settings) -> None:
        gated = settings.model_copy(update={"require_approval_over_usd": 0.0001})
        service = ResearchService(gated)
        deps = _deps(gated)

        pending = await service.run(QUESTION, run_id="hitl-approve", deps=deps)
        assert isinstance(pending, PendingApproval)

        report = await service.resume("hitl-approve", "approve", deps=deps)
        assert isinstance(report, ResearchReport)
        assert report.verification.passed

    async def test_rejection_ends_the_run_without_spending_on_search(
        self, settings: Settings
    ) -> None:
        gated = settings.model_copy(update={"require_approval_over_usd": 0.0001})
        service = ResearchService(gated)
        deps = _deps(gated)

        await service.run(QUESTION, run_id="hitl-reject", deps=deps)
        report = await service.resume("hitl-reject", "reject", deps=deps)

        assert isinstance(report, ResearchReport)
        assert any("rejected" in w.lower() for w in report.warnings)
        # Only planning happened; nothing was screened, read or written.
        assert "synthesize" not in deps.provider.calls  # type: ignore[attr-defined]
        assert "screen" not in deps.provider.calls  # type: ignore[attr-defined]

    async def test_cheap_run_is_not_gated(self, settings: Settings) -> None:
        ungated = settings.model_copy(update={"require_approval_over_usd": 1_000.0})
        result = await ResearchService(ungated).run(QUESTION, deps=_deps(ungated))
        assert isinstance(result, ResearchReport)


class TestDegradation:
    async def test_no_sources_yields_an_honest_report_not_a_crash(self, settings: Settings) -> None:
        budget = BudgetTracker(settings.max_usd_per_run)
        deps = Deps(
            settings=settings,
            provider=FakeAPIProvider(settings, budget, DEFAULT_RESPONSES),
            search=FakeSearchService(settings, documents=[]),
            budget=budget,
        )
        report = await ResearchService(settings).run(QUESTION, deps=deps)

        assert isinstance(report, ResearchReport)
        assert report.sources == []
        assert "No sources" in report.summary or "no findings" in report.body.lower()

    async def test_budget_exhaustion_returns_findings_instead_of_crashing(
        self, settings: Settings
    ) -> None:
        # Enough budget to plan, screen, extract and assess coverage (~$0.022),
        # but not the headroom synthesis requires on top of that.
        broke = settings.model_copy(update={"max_usd_per_run": 0.025})
        deps = _deps(broke)
        result = await ResearchService(broke).run(QUESTION, deps=deps)

        assert isinstance(result, ResearchReport)
        # The evidence already paid for is handed back, correctly cited.
        assert "[S1]" in result.body
        assert result.verification.passed
        assert any("budget" in w.lower() for w in result.warnings)

    async def test_budget_exhaustion_mid_run_still_returns_a_report(
        self, settings: Settings
    ) -> None:
        # So little budget that a node raises rather than degrading locally;
        # the service-level salvage path must still produce a report.
        broke = settings.model_copy(update={"max_usd_per_run": 0.0095})
        deps = _deps(broke)
        result = await ResearchService(broke).run(QUESTION, deps=deps)

        assert isinstance(result, ResearchReport)
        assert any("budget" in w.lower() or "exhaust" in w.lower() for w in result.warnings)
        assert deps.budget.spent_usd < 0.05  # bounded, not runaway


class TestDurability:
    async def test_a_paused_run_resumes_in_a_fresh_service_instance(
        self, settings: Settings, tmp_path: object
    ) -> None:
        """The point of a durable checkpointer: approval can arrive later.

        Two service instances over one SQLite file stand in for two
        processes — the second knows nothing except the run id.
        """
        durable = settings.model_copy(
            update={
                "require_approval_over_usd": 0.0001,
                "checkpoint_db": tmp_path / "checkpoints.sqlite",  # type: ignore[operator]
            }
        )

        first = ResearchService(durable)
        pending = await first.run(QUESTION, run_id="durable-1", deps=_deps(durable))
        assert isinstance(pending, PendingApproval)
        await first.aclose()

        second = ResearchService(durable)
        report = await second.resume("durable-1", "approve", deps=_deps(durable))
        await second.aclose()

        assert isinstance(report, ResearchReport)
        assert report.verification.passed
        # The question survived the restart; the caller only supplied a run id.
        assert report.question == QUESTION


class TestReportSerialisation:
    def test_report_round_trips_through_json(self, settings: Settings) -> None:
        report = ResearchService.build_report(
            "r1",
            QUESTION,
            {"draft": None, "warnings": [], "search_round": 1},
            CostBreakdown(usd=0.01, calls=1),
        )
        payload = json.loads(report.model_dump_json())
        restored = ResearchReport.model_validate(payload)
        assert restored.run_id == "r1"
