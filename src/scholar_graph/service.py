"""The application-level entry point: run a research question end to end.

The CLI, the HTTP API and the eval harness all go through here, so they
cannot drift apart in how they build dependencies, handle interrupts, or
assemble the final report.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scholar_graph.config import Settings, get_settings
from scholar_graph.domain import (
    CostBreakdown,
    DraftReport,
    ResearchReport,
    VerificationReport,
)
from scholar_graph.graph import build_graph
from scholar_graph.llm.budget import BudgetExceeded, BudgetTracker
from scholar_graph.llm.provider import LLMProvider
from scholar_graph.observability import get_logger, run_context, timed
from scholar_graph.state import Deps, active_documents, initial_state
from scholar_graph.tools import SearchService

log = get_logger(__name__)


@dataclass
class PendingApproval:
    """A run suspended at the cost gate, waiting on a human."""

    run_id: str
    payload: dict[str, Any]


@asynccontextmanager
async def build_deps(settings: Settings | None = None) -> AsyncIterator[Deps]:
    settings = settings or get_settings()
    budget = BudgetTracker(settings.max_usd_per_run)
    provider = LLMProvider(settings, budget)
    search = SearchService(settings)
    deps = Deps(settings=settings, provider=provider, search=search, budget=budget)
    try:
        yield deps
    finally:
        await deps.aclose()


async def _make_checkpointer(settings: Settings) -> tuple[Any, Any]:
    """Return ``(checkpointer, closeable)``.

    Durable checkpoints are what let an approval arrive tomorrow rather than
    within one process lifetime. The graph is driven with ``ainvoke``, so the
    checkpointer has to be the async SQLite saver — the synchronous one raises
    on every async call path.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    path = settings.checkpoint_db
    if path is None:
        return InMemorySaver(), None

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(str(path))
        return AsyncSqliteSaver(connection), connection
    except Exception as exc:  # noqa: BLE001 - degrade to memory rather than fail to start
        log.warning("checkpointer.sqlite_unavailable", error=str(exc))
        return InMemorySaver(), None


class ResearchService:
    """Owns the compiled graph and its checkpointer.

    Construction is synchronous but the checkpointer is not, so the graph is
    built lazily on first use. Callers may ``aclose()`` when done; the CLI's
    one-shot runs let process exit handle it.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.graph: Any = None
        self.checkpointer: Any = None
        self._closeable: Any = None

    async def _ensure_graph(self) -> Any:
        if self.graph is None:
            self.checkpointer, self._closeable = await _make_checkpointer(self.settings)
            self.graph = build_graph(self.checkpointer)
        return self.graph

    async def aclose(self) -> None:
        if self._closeable is not None:
            await self._closeable.close()
            self._closeable = None

    async def run(
        self,
        question: str,
        *,
        run_id: str | None = None,
        deps: Deps | None = None,
    ) -> ResearchReport | PendingApproval:
        """Execute a research run, or suspend it awaiting human approval."""
        run_id = run_id or uuid.uuid4().hex[:12]

        if deps is not None:
            return await self._run_with(question, run_id, deps)

        async with build_deps(self.settings) as owned:
            return await self._run_with(question, run_id, owned)

    async def resume(
        self,
        run_id: str,
        decision: str,
        *,
        deps: Deps | None = None,
    ) -> ResearchReport | PendingApproval:
        """Resume a suspended run with a human decision."""
        if deps is not None:
            return await self._resume_with(run_id, decision, deps)

        async with build_deps(self.settings) as owned:
            return await self._resume_with(run_id, decision, owned)

    # -- internals --------------------------------------------------------

    def _config(self, run_id: str, deps: Deps) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id, "deps": deps},
            "recursion_limit": 60,
        }

    async def _run_with(
        self, question: str, run_id: str, deps: Deps
    ) -> ResearchReport | PendingApproval:
        graph = await self._ensure_graph()
        with run_context(run_id), timed(log, "research.run", question=question[:120]):
            state = initial_state(
                run_id, question, max_search_rounds=deps.settings.max_search_rounds
            )
            config = self._config(run_id, deps)
            try:
                result = await graph.ainvoke(state, config=config)
            except BudgetExceeded as exc:
                result = await self._salvage(graph, config, exc)
            return self._interpret(run_id, question, result, deps)

    async def _resume_with(
        self, run_id: str, decision: str, deps: Deps
    ) -> ResearchReport | PendingApproval:
        from langgraph.types import Command

        graph = await self._ensure_graph()
        with run_context(run_id), timed(log, "research.resume", decision=decision):
            result = await graph.ainvoke(
                Command(resume=decision), config=self._config(run_id, deps)
            )
            snapshot = await graph.aget_state(self._config(run_id, deps))
            question = str((snapshot.values or {}).get("question", ""))
            return self._interpret(run_id, question, result, deps)

    @staticmethod
    async def _salvage(graph: Any, config: dict[str, Any], exc: BudgetExceeded) -> dict[str, Any]:
        """Recover whatever the run had accumulated when the budget ran out.

        Nodes degrade locally where they can. This is the backstop for the
        ones that cannot: rather than losing a part-finished run entirely,
        read the last checkpoint and report from it, saying plainly why it is
        incomplete.
        """
        log.warning("research.budget_exhausted", error=str(exc))
        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values or {})
        values["warnings"] = [*(values.get("warnings") or []), str(exc)]
        return values

    def _interpret(
        self, run_id: str, question: str, result: dict[str, Any], deps: Deps
    ) -> ResearchReport | PendingApproval:
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = getattr(interrupts[0], "value", interrupts[0])
            log.info("research.awaiting_approval", payload=payload)
            return PendingApproval(run_id=run_id, payload=dict(payload))

        return self.build_report(run_id, question, result, deps.budget.total)

    @staticmethod
    def build_report(
        run_id: str,
        question: str,
        state: dict[str, Any],
        cost: CostBreakdown,
    ) -> ResearchReport:
        draft: DraftReport | None = state.get("draft")
        warnings = list(state.get("warnings") or [])

        if draft is None:
            # Reached when a human rejects the run at the cost gate.
            draft = DraftReport(
                summary="This run did not produce a report.",
                body="No draft was generated. See warnings for why.",
                limitations="Not applicable.",
            )
            if state.get("approval_decision") == "reject":
                warnings.append("Run was rejected at the cost-approval gate.")

        verification = state.get("verification") or VerificationReport(
            checked_markers=0, valid_markers=0, grounded_notes=0, total_notes=0
        )
        if verification.issues:
            warnings.append(
                f"{len(verification.issues)} citation issue(s) remained after revision."
            )

        sources = state.get("cited_sources") or active_documents(state)  # type: ignore[arg-type]

        return ResearchReport(
            run_id=run_id,
            question=question,
            summary=draft.summary,
            body=draft.body,
            limitations=draft.limitations,
            sources=sources,
            verification=verification,
            critiques=list(state.get("critiques") or []),
            cost=cost,
            search_rounds=state.get("search_round", 0),
            warnings=warnings,
        )
