"""Graph state and the dependency bundle nodes run against.

The state is deliberately flat and JSON-serialisable: it is what the
checkpointer persists, and a run that cannot be persisted cannot be resumed
after an interrupt.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from scholar_graph.config import Settings
from scholar_graph.domain import (
    CoverageAssessment,
    Document,
    DraftReport,
    Note,
    PanelCritique,
    ResearchPlan,
    VerificationReport,
)
from scholar_graph.llm.budget import BudgetTracker
from scholar_graph.llm.provider import LLMProvider
from scholar_graph.tools import SearchService


def merge_documents(left: list[Document], right: list[Document]) -> list[Document]:
    """Reducer: accumulate documents across search rounds without duplicates.

    Order is stable — the citation markers in a draft refer to positions in
    this list, so reordering it mid-run would silently rewrite every citation.
    """
    seen = {doc.source_id for doc in left}
    merged = list(left)
    for doc in right:
        if doc.source_id not in seen:
            seen.add(doc.source_id)
            merged.append(doc)
    return merged


class ResearchState(TypedDict, total=False):
    run_id: str
    question: str

    plan: ResearchPlan | None
    pending_queries: list[str]
    executed_queries: Annotated[list[str], operator.add]

    documents: Annotated[list[Document], merge_documents]
    """Everything ever retrieved. Append-only, so citation indices never shift."""

    screened_ids: Annotated[list[str], operator.add]
    """Sources the screener has ruled on, so it never pays to judge one twice."""

    rejected_ids: Annotated[list[str], operator.add]
    """Sources the screener ruled irrelevant. Kept rather than deleted so a
    run's discard decisions stay auditable after the fact."""

    cited_sources: list[Document]
    """The exact ordered list the writer saw, frozen at synthesis time.
    Verification and rendering index into this, never into `documents` —
    otherwise a later retrieval could silently renumber every citation."""

    notes: Annotated[list[Note], operator.add]

    coverage: CoverageAssessment | None
    draft: DraftReport | None
    verification: VerificationReport | None
    critiques: list[PanelCritique]

    search_round: int
    revision_round: int
    max_search_rounds: int
    """Seeded from settings so routing decisions read only from state."""
    warnings: Annotated[list[str], operator.add]

    approval_decision: str | None
    """Set by a human when the run pauses at the cost gate: 'approve' | 'reject'."""


@dataclass
class Deps:
    """Everything a node needs that is not state.

    Passed through LangGraph's ``configurable`` rather than captured in
    closures so a graph instance stays reusable across runs and testable
    with substituted collaborators.
    """

    settings: Settings
    provider: LLMProvider
    search: SearchService
    budget: BudgetTracker

    async def aclose(self) -> None:
        await self.provider.aclose()
        await self.search.aclose()


def deps_from_config(config: dict[str, Any]) -> Deps:
    configurable = config.get("configurable") or {}
    deps = configurable.get("deps")
    if not isinstance(deps, Deps):
        raise RuntimeError(
            "Graph invoked without dependencies. Pass "
            "config={'configurable': {'deps': Deps(...), 'thread_id': ...}}."
        )
    return deps


def active_documents(state: ResearchState) -> list[Document]:
    """Retrieved sources that survived screening, in stable retrieval order."""
    rejected = set(state.get("rejected_ids") or [])
    return [doc for doc in (state.get("documents") or []) if doc.source_id not in rejected]


def initial_state(run_id: str, question: str, *, max_search_rounds: int = 3) -> ResearchState:
    return ResearchState(
        run_id=run_id,
        question=question,
        plan=None,
        pending_queries=[],
        executed_queries=[],
        documents=[],
        screened_ids=[],
        rejected_ids=[],
        cited_sources=[],
        notes=[],
        coverage=None,
        draft=None,
        verification=None,
        critiques=[],
        search_round=0,
        revision_round=0,
        max_search_rounds=max_search_rounds,
        warnings=[],
        approval_decision=None,
    )
