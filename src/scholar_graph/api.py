"""HTTP API.

Deliberately small: submit a question, poll or stream a run, approve a run
that paused. Runs execute in the background because a research run takes
minutes, and a request that holds a connection open for minutes is a request
that dies to the first proxy timeout.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from scholar_graph.config import get_settings
from scholar_graph.domain import ResearchReport
from scholar_graph.observability import configure_logging, get_logger
from scholar_graph.service import PendingApproval, ResearchService

log = get_logger(__name__)

RunStatus = Literal["running", "awaiting_approval", "completed", "failed"]


@dataclass
class RunRecord:
    run_id: str
    question: str
    status: RunStatus = "running"
    report: ResearchReport | None = None
    approval: dict[str, Any] | None = None
    error: str | None = None
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)


class RunStore:
    """In-process run registry.

    Fine for a single instance; a multi-replica deployment would back this
    with the same store as the LangGraph checkpointer. Called out in
    docs/operations.md rather than hidden here.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, question: str) -> RunRecord:
        run_id = uuid.uuid4().hex[:12]
        record = RunRecord(run_id=run_id, question=question)
        self._runs[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown run {run_id!r}.")
        return record


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    question: str
    approval: dict[str, Any] | None = None
    error: str | None = None
    report: ResearchReport | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    app.state.store = RunStore()
    app.state.service = ResearchService(settings)
    log.info("api.started", llm_mode=settings.llm_mode.value)
    yield


app = FastAPI(
    title="scholar-graph",
    version="0.1.0",
    summary="Evidence-grounded research agent with verified citations.",
    lifespan=lifespan,
)


async def _execute(record: RunRecord, resume_decision: str | None = None) -> None:
    service: ResearchService = app.state.service
    try:
        await record.events.put({"event": "started", "run_id": record.run_id})
        if resume_decision is None:
            result = await service.run(record.question, run_id=record.run_id)
        else:
            result = await service.resume(record.run_id, resume_decision)

        if isinstance(result, PendingApproval):
            record.status = "awaiting_approval"
            record.approval = result.payload
            await record.events.put({"event": "awaiting_approval", "data": result.payload})
        else:
            record.status = "completed"
            record.report = result
            await record.events.put(
                {
                    "event": "completed",
                    "data": {
                        "citation_precision": result.verification.citation_precision,
                        "usd": result.cost.usd,
                        "sources": len(result.sources),
                    },
                }
            )
    except Exception as exc:
        log.exception("api.run_failed", run_id=record.run_id)
        record.status = "failed"
        record.error = str(exc)
        await record.events.put({"event": "failed", "data": {"error": str(exc)}})
    finally:
        await record.events.put({"event": "done"})


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "mode": get_settings().llm_mode.value}


@app.post("/runs", response_model=RunSummary, status_code=202)
async def create_run(request: ResearchRequest, background: BackgroundTasks) -> RunSummary:
    record: RunRecord = app.state.store.create(request.question)
    background.add_task(_execute, record)
    return RunSummary(run_id=record.run_id, status=record.status, question=record.question)


@app.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str) -> RunSummary:
    record: RunRecord = app.state.store.get(run_id)
    return RunSummary(
        run_id=record.run_id,
        status=record.status,
        question=record.question,
        approval=record.approval,
        error=record.error,
        report=record.report,
    )


@app.post("/runs/{run_id}/approval", response_model=RunSummary, status_code=202)
async def approve_run(
    run_id: str, request: ApprovalRequest, background: BackgroundTasks
) -> RunSummary:
    record: RunRecord = app.state.store.get(run_id)
    if record.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is {record.status}, not awaiting approval.",
        )
    record.status = "running"
    record.approval = None
    background.add_task(_execute, record, request.decision)
    return RunSummary(run_id=record.run_id, status=record.status, question=record.question)


@app.get("/runs/{run_id}/events")
async def stream_run(run_id: str) -> EventSourceResponse:
    """Server-sent progress events for one run."""
    record: RunRecord = app.state.store.get(run_id)

    async def publisher() -> AsyncIterator[dict[str, Any]]:
        while True:
            message = await record.events.get()
            if message.get("event") == "done":
                yield {"event": "done", "data": "{}"}
                return
            import json

            yield {
                "event": message.get("event", "message"),
                "data": json.dumps(message.get("data", {})),
            }

    return EventSourceResponse(publisher())
