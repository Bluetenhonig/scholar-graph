"""Search, screening, and note extraction.

Screening and extraction run on the cheap model: they are high-volume,
low-judgement work, and spending Opus tokens deciding whether an abstract is
on-topic is how agent costs get away from you.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig

from scholar_graph import prompts
from scholar_graph.domain import Document, Note, NoteSet, ScreeningResult
from scholar_graph.llm.budget import BudgetExceeded
from scholar_graph.llm.cassette import CassetteMiss
from scholar_graph.observability import get_logger
from scholar_graph.state import Deps, ResearchState, active_documents, deps_from_config

log = get_logger(__name__)

EXTRACTION_BATCH_SIZE = 4
"""Sources per extraction call. Small batches keep quotes attributable —
large ones make the model blur spans across documents."""


async def search(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    deps = deps_from_config(dict(config))
    queries = list(state.get("pending_queries") or [])
    round_number = state.get("search_round", 0) + 1

    if not queries:
        return {"search_round": round_number}

    already_have = len(state.get("documents") or [])
    room_left = max(0, deps.settings.max_documents - already_have)
    if room_left == 0:
        log.info("search.skipped", reason="document_cap_reached", cap=deps.settings.max_documents)
        return {"search_round": round_number, "pending_queries": []}

    per_query = max(2, min(8, room_left // max(1, len(queries)) or 2))

    try:
        documents = await deps.search.search(queries, per_query=per_query)
    except CassetteMiss:
        # A missing cassette is a developer error, not a flaky provider.
        # Degrading here would produce a confident, empty report instead of
        # telling you the recording is stale.
        raise
    except Exception as exc:  # noqa: BLE001 - a dead provider degrades the run, not ends it
        log.error("search.failed", error=str(exc), queries=queries)
        return {
            "search_round": round_number,
            "pending_queries": [],
            "warnings": [f"Search round {round_number} failed: {exc}"],
        }

    documents = documents[:room_left]
    log.info("search.done", round=round_number, queries=len(queries), retrieved=len(documents))

    return {
        "documents": documents,
        "executed_queries": queries,
        "pending_queries": [],
        "search_round": round_number,
    }


async def screen(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Drop off-topic sources before paying to read them."""
    deps = deps_from_config(dict(config))
    documents = state.get("documents") or []
    already_judged = set(state.get("screened_ids") or [])
    unscreened = [d for d in documents if d.source_id not in already_judged]

    if not unscreened:
        return {}

    catalogue = "\n\n".join(
        f"source_id: {d.source_id}\ntitle: {d.title}\nabstract: {d.abstract[:900]}"
        for d in unscreened
    )

    try:
        result, _ = await deps.provider.structured(
            purpose="screen",
            system=prompts.SCREENER,
            user=f"Research question: {state['question']}\n\nSources:\n\n{catalogue}",
            response_model=ScreeningResult,
            model=deps.settings.worker_model,
            effort="low",
        )
    except BudgetExceeded as exc:
        # Out of money mid-screen: keep everything unjudged rather than
        # discarding sources we simply could not afford to look at.
        log.warning("screen.budget_exceeded", error=str(exc))
        return {"warnings": [f"Screening stopped early, sources kept unfiltered: {exc}"]}

    verdicts = {v.source_id: v.relevant for v in result.verdicts}
    rejected = [d.source_id for d in unscreened if verdicts.get(d.source_id) is False]
    judged = [d.source_id for d in unscreened]

    log.info("screen.done", considered=len(unscreened), rejected=len(rejected))

    out: dict[str, Any] = {"screened_ids": judged, "rejected_ids": rejected}
    if len(rejected) == len(unscreened):
        out["warnings"] = ["Screening rejected every source retrieved in this round."]
    return out


async def extract(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Pull quote-backed findings out of the sources that survived screening."""
    deps = deps_from_config(dict(config))
    covered = {note.source_id for note in state.get("notes") or []}
    todo = [d for d in active_documents(state) if d.source_id not in covered]

    if not todo:
        return {}

    batches = [
        todo[i : i + EXTRACTION_BATCH_SIZE] for i in range(0, len(todo), EXTRACTION_BATCH_SIZE)
    ]
    results = await asyncio.gather(
        *(_extract_batch(deps, state["question"], batch) for batch in batches),
        return_exceptions=True,
    )

    notes: list[Note] = []
    warnings: list[str] = []
    for batch, outcome in zip(batches, results, strict=True):
        if isinstance(outcome, BaseException):
            ids = ", ".join(d.source_id for d in batch)
            log.warning("extract.batch_failed", sources=ids, error=str(outcome))
            warnings.append(f"Could not extract findings from {len(batch)} source(s).")
            continue
        notes.extend(outcome)

    log.info("extract.done", sources=len(todo), notes=len(notes))
    out: dict[str, Any] = {"notes": notes}
    if warnings:
        out["warnings"] = warnings
    return out


async def _extract_batch(deps: Deps, question: str, batch: list[Document]) -> list[Note]:
    catalogue = "\n\n".join(
        f"source_id: {d.source_id}\ntitle: {d.title}\nabstract: {d.abstract}" for d in batch
    )
    valid_ids = {d.source_id for d in batch}

    note_set, _ = await deps.provider.structured(
        purpose="extract",
        system=prompts.EXTRACTOR,
        user=f"Research question: {question}\n\nSources:\n\n{catalogue}",
        response_model=NoteSet,
        model=deps.settings.worker_model,
        effort="low",
    )

    # Guard against the model attributing a finding to a source outside this
    # batch. Verification would catch it later; discarding it here keeps the
    # failure local and cheap.
    return [n for n in note_set.notes if n.source_id in valid_ids]
