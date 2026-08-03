"""AutoGen review board.

Mechanical verification (``scholar_graph.verification``) catches citations
that do not resolve. It cannot catch a correctly-cited sentence that
overstates what its source found, or a summary that answers a question
nobody asked. Those need judgement, and judgement from a single model that
just wrote the draft is not independent.

So: two reviewers with deliberately different mandates, then an adjudicator
that turns their prose into a decision. The panel advises — the graph decides
whether to act on it.
"""

from __future__ import annotations

from typing import Any

from scholar_graph import prompts
from scholar_graph.domain import DraftReport, PanelCritique
from scholar_graph.llm.provider import LLMProvider
from scholar_graph.observability import get_logger
from scholar_graph.panel.model_client import ProviderChatCompletionClient

log = get_logger(__name__)

MAX_TURNS = 3
"""One turn per reviewer plus the adjudicator. The termination condition is a
hard turn cap rather than a model-decided stop: a group chat that decides for
itself when to stop is a group chat that can decide not to."""


class PanelUnavailable(RuntimeError):
    """AutoGen is not installed, or the panel could not run."""


def _draft_brief(question: str, draft: DraftReport) -> str:
    return (
        f"Research question: {question}\n\n"
        f"Summary:\n{draft.summary}\n\n"
        f"Body:\n{draft.body}\n\n"
        f"Stated limitations:\n{draft.limitations}"
    )


async def review(
    *,
    question: str,
    draft: DraftReport,
    provider: LLMProvider,
    model: str,
) -> list[PanelCritique]:
    """Run the panel and return its critiques.

    Raises :class:`PanelUnavailable` if AutoGen is not installed; callers
    treat that as "skip the panel", not as a failed run.
    """
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
    except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
        raise PanelUnavailable(
            "The review panel needs the 'panel' extra: pip install 'scholar-graph[panel]'"
        ) from exc

    def agent(name: str, system: str) -> Any:
        return AssistantAgent(
            name=name,
            system_message=system,
            model_client=ProviderChatCompletionClient(
                provider,
                model=model,
                purpose=f"panel:{name}",
                effort="low",
                max_tokens=1024,
            ),
        )

    methodologist = agent("methodologist", prompts.PANEL_METHODOLOGIST)
    editor = agent("editor", prompts.PANEL_EDITOR)
    adjudicator = agent("adjudicator", prompts.PANEL_ADJUDICATOR)

    team = RoundRobinGroupChat(
        [methodologist, editor, adjudicator],
        termination_condition=MaxMessageTermination(MAX_TURNS + 1),
    )

    result = await team.run(task=_draft_brief(question, draft))

    critiques: list[PanelCritique] = []
    for message in result.messages:
        source = getattr(message, "source", "")
        content = getattr(message, "content", "")
        if source in {"user", ""} or not isinstance(content, str):
            continue
        critiques.append(
            PanelCritique(
                reviewer=source,
                verdict=_verdict_for(source, content),
                comments=content.strip(),
            )
        )

    log.info("panel.done", reviewers=len(critiques))
    return critiques


def _verdict_for(source: str, content: str) -> str:
    """Only the adjudicator's verdict is binding; reviewers merely advise."""
    if source != "adjudicator":
        return "accept"
    lowered = content.lower()
    # Bias to accept: a revision round is expensive, and an ambiguous review
    # is not evidence of a defect.
    return "revise" if "revise" in lowered and "accept" not in lowered else "accept"


def panel_requires_revision(critiques: list[PanelCritique]) -> bool:
    return any(c.reviewer == "adjudicator" and c.verdict == "revise" for c in critiques)
