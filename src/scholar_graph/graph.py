"""The LangGraph state machine and the routing rules between its nodes.

    plan -> approval -> search -> screen -> extract -> coverage
                                              ^           |
                                              |           v
                                              +------ (more evidence needed?)
                                                          |
                                                          v
                                  synthesize -> verify -> (repair?) -> revise
                                                          |               |
                                                          v               |
                                                        panel <-----------+
                                                          |
                                                          v
                                                        finalize

Every loop in that diagram is bounded — by search rounds, revision rounds,
document count, and a hard USD budget. An agent whose termination depends on
a model deciding it is finished is an agent that eventually does not finish.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from scholar_graph.nodes import (
    approval_gate,
    assess_coverage,
    check_citations,
    convene_panel,
    extract,
    plan,
    revise,
    screen,
    search,
    synthesize,
)
from scholar_graph.nodes.writing import MAX_REVISIONS
from scholar_graph.observability import get_logger
from scholar_graph.panel.review_board import panel_requires_revision
from scholar_graph.state import ResearchState

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def route_after_approval(state: ResearchState) -> Literal["search", "__end__"]:
    if state.get("approval_decision") == "reject":
        log.info("route.rejected_by_human")
        return "__end__"
    return "search"


def route_after_coverage(state: ResearchState) -> Literal["search", "synthesize"]:
    """Search again only if there is a gap, a query to close it, and room to run."""
    coverage = state.get("coverage")
    rounds_used = state.get("search_round", 0)
    max_rounds = state.get("max_search_rounds", 3)

    if coverage is None:
        return "synthesize"
    if rounds_used >= max_rounds:
        log.info("route.search_capped", rounds=rounds_used)
        return "synthesize"
    if not coverage.follow_up_queries:
        return "synthesize"
    if not coverage.unmet_criteria:
        return "synthesize"

    log.info("route.search_again", round=rounds_used + 1, gaps=len(coverage.unmet_criteria))
    return "search"


def route_after_verification(state: ResearchState) -> Literal["revise", "panel"]:
    report = state.get("verification")
    revisions = state.get("revision_round", 0)

    if report is None or report.passed:
        return "panel"
    if revisions >= MAX_REVISIONS:
        # Ship it with the failures recorded rather than looping forever. A
        # report that says where it is weak is more useful than no report.
        log.warning("route.revision_capped", issues=len(report.issues), rounds=revisions)
        return "panel"

    log.info("route.revise", issues=len(report.issues), round=revisions + 1)
    return "revise"


def route_after_panel(state: ResearchState) -> Literal["revise", "__end__"]:
    critiques = state.get("critiques") or []
    revisions = state.get("revision_round", 0)

    if revisions >= MAX_REVISIONS:
        return "__end__"
    if panel_requires_revision(critiques):
        log.info("route.panel_requested_revision", round=revisions + 1)
        return "revise"
    return "__end__"


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_graph(checkpointer: Any = None) -> Any:
    """Compile the research graph.

    ``checkpointer`` is required for human-in-the-loop: without one, an
    ``interrupt`` has nowhere to persist the state it suspends.
    """
    builder = StateGraph(ResearchState)

    builder.add_node("plan", plan)
    builder.add_node("approval", approval_gate)
    builder.add_node("search", search)
    builder.add_node("screen", screen)
    builder.add_node("extract", extract)
    builder.add_node("coverage", assess_coverage)
    builder.add_node("synthesize", synthesize)
    builder.add_node("verify", check_citations)
    builder.add_node("revise", revise)
    builder.add_node("panel", convene_panel)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approval")
    builder.add_conditional_edges(
        "approval", route_after_approval, {"search": "search", "__end__": END}
    )

    builder.add_edge("search", "screen")
    builder.add_edge("screen", "extract")
    builder.add_edge("extract", "coverage")
    builder.add_conditional_edges(
        "coverage", route_after_coverage, {"search": "search", "synthesize": "synthesize"}
    )

    builder.add_edge("synthesize", "verify")
    builder.add_conditional_edges(
        "verify", route_after_verification, {"revise": "revise", "panel": "panel"}
    )
    # A revision is a new draft, so it goes back through the same mechanical
    # verification the first draft did. No draft reaches the reader unchecked.
    builder.add_edge("revise", "verify")
    builder.add_conditional_edges("panel", route_after_panel, {"revise": "revise", "__end__": END})

    return builder.compile(checkpointer=checkpointer)
