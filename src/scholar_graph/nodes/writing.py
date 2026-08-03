"""Synthesis, mechanical verification, and targeted revision."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from scholar_graph import prompts
from scholar_graph.domain import Document, DraftReport, Note
from scholar_graph.observability import get_logger
from scholar_graph.state import ResearchState, active_documents, deps_from_config
from scholar_graph.verification import format_issues_for_revision, verify

log = get_logger(__name__)

MAX_REVISIONS = 2
"""Revision rounds before shipping with warnings. Verification failures that
survive two targeted repairs are usually evidence problems, not prose
problems, and a third rewrite just burns budget."""


def _render_sources(sources: list[Document]) -> str:
    return "\n\n".join(doc.as_context(i) for i, doc in enumerate(sources, start=1))


def _render_notes(notes: list[Note], sources: list[Document]) -> str:
    index_by_id = {doc.source_id: i for i, doc in enumerate(sources, start=1)}
    lines = []
    for note in notes:
        marker = index_by_id.get(note.source_id)
        if marker is None:
            continue
        lines.append(f'[S{marker}] {note.claim}\n      evidence: "{note.quote}"')
    return "\n".join(lines)


async def synthesize(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    deps = deps_from_config(dict(config))
    sources = active_documents(state)
    notes = state.get("notes") or []

    if not sources:
        return {
            "draft": DraftReport(
                summary="No sources could be retrieved for this question.",
                body="The search returned no usable results, so no findings can be reported.",
                limitations="This is a retrieval failure, not a finding about the literature.",
            ),
            "cited_sources": [],
            "warnings": ["Synthesis ran with an empty evidence base."],
        }

    if not deps.budget.can_afford(0.005):
        # Out of budget at the last step. The evidence was already gathered and
        # paid for, so hand it back as a plain findings list rather than
        # throwing away the whole run.
        log.warning("synthesize.degraded", reason="insufficient_budget")
        return {
            "draft": _evidence_only_draft(notes, sources),
            "cited_sources": sources,
            "warnings": [
                "Budget ran out before synthesis; returning extracted findings verbatim "
                "instead of a written summary."
            ],
        }

    draft, _ = await deps.provider.structured(
        purpose="synthesize",
        system=prompts.WRITER,
        user=(
            f"Research question: {state['question']}\n\n"
            f"Numbered sources:\n{_render_sources(sources)}\n\n"
            f"Verified findings:\n{_render_notes(notes, sources)}"
        ),
        response_model=DraftReport,
        model=deps.settings.reasoning_model,
        effort="high",
    )

    log.info("synthesize.done", sources=len(sources), notes=len(notes))
    # Freeze the source list the writer actually saw: every [Sn] in this draft
    # is an index into exactly this list.
    return {"draft": draft, "cited_sources": sources}


def _evidence_only_draft(notes: list[Note], sources: list[Document]) -> DraftReport:
    """Assemble a report from notes alone, with no model call.

    Every line is a verified quote plus its citation, so this degraded output
    passes the same verification the written version does.
    """
    index_by_id = {doc.source_id: i for i, doc in enumerate(sources, start=1)}
    lines = []
    for note in notes:
        marker = index_by_id.get(note.source_id)
        if marker is not None:
            lines.append(f"- {note.claim} [S{marker}]")

    return DraftReport(
        summary=(
            f"Budget was exhausted before the summary could be written. "
            f"{len(lines)} verified findings from {len(sources)} sources are listed below."
        ),
        body="\n".join(lines) if lines else "No findings were extracted.",
        limitations=(
            "These are raw extracted findings, not a synthesis. They have not been "
            "weighed against one another or checked for contradictions."
        ),
    )


async def check_citations(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Mechanically verify the draft. No model involved, by design."""
    draft = state.get("draft")
    if draft is None:
        return {}

    sources = state.get("cited_sources") or active_documents(state)
    notes = state.get("notes") or []
    report = verify(draft, notes, sources)

    log.info(
        "verify.done",
        citation_precision=round(report.citation_precision, 3),
        groundedness=round(report.groundedness, 3),
        issues=len(report.issues),
    )
    return {"verification": report}


async def revise(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Repair exactly the defects verification found, and nothing else."""
    deps = deps_from_config(dict(config))
    draft = state.get("draft")
    report = state.get("verification")
    if draft is None or report is None:
        return {}

    sources = state.get("cited_sources") or active_documents(state)
    round_number = state.get("revision_round", 0) + 1

    revised, _ = await deps.provider.structured(
        purpose="revise",
        system=prompts.REVISER,
        user=(
            f"Research question: {state['question']}\n\n"
            f"Numbered sources:\n{_render_sources(sources)}\n\n"
            f"Current summary:\n{draft.summary}\n\n"
            f"Current body:\n{draft.body}\n\n"
            f"Current limitations:\n{draft.limitations}\n\n"
            f"Verification failures to fix:\n{format_issues_for_revision(report)}"
        ),
        response_model=DraftReport,
        model=deps.settings.reasoning_model,
        effort="high",
    )

    log.info("revise.done", round=round_number, issues_addressed=len(report.issues))
    return {"draft": revised, "revision_round": round_number}
