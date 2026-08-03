"""The review-panel node: advisory, optional, and never fatal."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from scholar_graph.llm.budget import BudgetExceeded
from scholar_graph.llm.cassette import CassetteMiss
from scholar_graph.observability import get_logger
from scholar_graph.panel.review_board import PanelUnavailable, review
from scholar_graph.state import ResearchState, deps_from_config

log = get_logger(__name__)


async def convene_panel(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    deps = deps_from_config(dict(config))
    draft = state.get("draft")

    if draft is None or not deps.settings.enable_review_panel:
        return {"critiques": []}

    if not deps.budget.can_afford(0.02):
        log.info("panel.skipped", reason="insufficient_budget")
        return {
            "critiques": [],
            "warnings": ["Review panel skipped: not enough budget remaining."],
        }

    try:
        critiques = await review(
            question=state["question"],
            draft=draft,
            provider=deps.provider,
            model=deps.settings.worker_model,
        )
    except PanelUnavailable as exc:
        log.info("panel.unavailable", error=str(exc))
        return {"critiques": [], "warnings": [f"Review panel unavailable: {exc}"]}
    except BudgetExceeded as exc:
        log.warning("panel.budget_exceeded", error=str(exc))
        return {"critiques": [], "warnings": [f"Review panel stopped: {exc}"]}
    except CassetteMiss:
        # Surface stale recordings rather than quietly shipping without a panel.
        raise
    except Exception as exc:  # noqa: BLE001 - advisory stage, must not sink the run
        log.warning("panel.failed", error=str(exc))
        return {"critiques": [], "warnings": [f"Review panel failed: {exc}"]}

    return {"critiques": critiques}
