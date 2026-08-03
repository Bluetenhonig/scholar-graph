"""Planning, the human-approval gate, and the search-coverage decision."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from scholar_graph import prompts
from scholar_graph.domain import CoverageAssessment, ResearchPlan
from scholar_graph.observability import get_logger
from scholar_graph.state import ResearchState, deps_from_config

log = get_logger(__name__)


async def plan(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    deps = deps_from_config(dict(config))

    research_plan, _ = await deps.provider.structured(
        purpose="plan",
        system=prompts.PLANNER,
        user=f"Research question:\n{state['question']}",
        response_model=ResearchPlan,
        model=deps.settings.reasoning_model,
        effort="medium",
    )

    log.info(
        "plan.ready",
        sub_questions=len(research_plan.sub_questions),
        queries=len(research_plan.search_queries),
    )
    return {
        "plan": research_plan,
        "pending_queries": research_plan.search_queries,
    }


async def approval_gate(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Pause for a human when a run looks expensive.

    The estimate is deliberately crude — it only needs to be good enough to
    separate "routine" from "worth a look". It is computed from the plan
    because that is the first moment the shape of the run is known and the
    last moment before the expensive part begins.
    """
    deps = deps_from_config(dict(config))
    settings = deps.settings

    query_count = len(state.get("pending_queries") or [])
    projected = _project_cost(
        query_count=query_count,
        rounds=settings.max_search_rounds,
        documents=settings.max_documents,
    )

    if projected <= settings.require_approval_over_usd:
        log.info("approval.auto", projected_usd=round(projected, 4))
        return {"approval_decision": "auto"}

    if state.get("approval_decision") in {"approve", "reject"}:
        return {}

    # Suspends the graph. The checkpointer persists everything above; the run
    # resumes from exactly here when a decision arrives, possibly in another
    # process, days later.
    decision = interrupt(
        {
            "reason": "cost_approval_required",
            "question": state["question"],
            "projected_usd": round(projected, 4),
            "threshold_usd": settings.require_approval_over_usd,
            "budget_usd": settings.max_usd_per_run,
            "queries": state.get("pending_queries"),
        }
    )

    normalised = str(decision).strip().lower()
    if normalised not in {"approve", "reject"}:
        normalised = "reject"

    log.info("approval.decided", decision=normalised, projected_usd=round(projected, 4))
    return {"approval_decision": normalised}


def _project_cost(*, query_count: int, rounds: int, documents: int) -> float:
    """Rough upper bound on a run's spend, in USD.

    Constants come from the eval suite's measured averages; they are a
    planning aid, not accounting. The BudgetTracker is the real enforcement.
    """
    screening = documents * 0.0004
    extraction = documents * 0.0012
    writing = 0.02 * rounds
    planning = 0.01
    return planning + (screening + extraction) * max(1, query_count) / 4 + writing


async def assess_coverage(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Decide whether the evidence gathered answers the question yet."""
    deps = deps_from_config(dict(config))
    plan_obj = state.get("plan")
    notes = state.get("notes") or []

    if plan_obj is None:
        return {"coverage": None}

    # Nothing found at all: another round of the same queries will not help.
    if not notes:
        log.warning("coverage.no_notes")
        return {
            "coverage": CoverageAssessment(
                satisfied_criteria=[],
                unmet_criteria=plan_obj.success_criteria,
                follow_up_queries=[],
            ),
            "warnings": ["No usable findings were extracted from the retrieved sources."],
        }

    findings = "\n".join(f"- [{n.source_id}] {n.claim}" for n in notes[:60])
    already_run = ", ".join(state.get("executed_queries") or [])

    assessment, _ = await deps.provider.structured(
        purpose="coverage",
        system=prompts.COVERAGE,
        user=(
            f"Question: {state['question']}\n\n"
            f"Success criteria:\n"
            + "\n".join(f"- {c}" for c in plan_obj.success_criteria)
            + f"\n\nQueries already run: {already_run}\n\n"
            f"Findings collected ({len(notes)}):\n{findings}"
        ),
        response_model=CoverageAssessment,
        model=deps.settings.reasoning_model,
        effort="low",
    )

    log.info(
        "coverage.assessed",
        satisfied=len(assessment.satisfied_criteria),
        unmet=len(assessment.unmet_criteria),
        follow_ups=len(assessment.follow_up_queries),
    )
    return {"coverage": assessment, "pending_queries": assessment.follow_up_queries}
