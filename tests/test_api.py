"""HTTP API behaviour, exercised against the real graph with a faked network."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from scholar_graph import service as service_module
from scholar_graph.config import Settings
from scholar_graph.llm.budget import BudgetTracker
from scholar_graph.state import Deps
from tests.conftest import DEFAULT_RESPONSES, FakeAPIProvider, FakeSearchService

QUESTION = "How can inference cost be reduced for large language models?"


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient whose runs use fake collaborators but the real graph."""

    @asynccontextmanager
    async def fake_build_deps(_: Settings | None = None) -> AsyncIterator[Deps]:
        budget = BudgetTracker(settings.max_usd_per_run)
        yield Deps(
            settings=settings,
            provider=FakeAPIProvider(settings, budget, DEFAULT_RESPONSES),
            search=FakeSearchService(settings),
            budget=budget,
        )

    monkeypatch.setattr(service_module, "build_deps", fake_build_deps)

    from scholar_graph.api import app

    with TestClient(app) as test_client:
        yield test_client


def _wait_for(client: TestClient, run_id: str, *statuses: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/runs/{run_id}").json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} never reached {statuses}: last was {payload['status']}")


class TestHealth:
    def test_healthz_reports_the_mode(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "mode" in response.json()


class TestValidation:
    def test_a_too_short_question_is_rejected(self, client: TestClient) -> None:
        assert client.post("/runs", json={"question": "hi"}).status_code == 422

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/runs/does-not-exist").status_code == 404

    def test_approving_a_run_that_is_not_waiting_is_409(self, client: TestClient) -> None:
        created = client.post("/runs", json={"question": QUESTION}).json()
        _wait_for(client, created["run_id"], "completed", "failed")
        response = client.post(f"/runs/{created['run_id']}/approval", json={"decision": "approve"})
        assert response.status_code == 409


class TestRunLifecycle:
    def test_a_run_completes_and_returns_a_verified_report(self, client: TestClient) -> None:
        created = client.post("/runs", json={"question": QUESTION})
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        payload = _wait_for(client, run_id, "completed", "failed")
        assert payload["status"] == "completed", payload.get("error")

        report = payload["report"]
        assert report["verification"]["valid_markers"] > 0
        assert len(report["sources"]) == 2
        assert report["cost"]["usd"] > 0

    def test_progress_events_stream(self, client: TestClient) -> None:
        run_id = client.post("/runs", json={"question": QUESTION}).json()["run_id"]
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            events = [
                line[len("event: ") :] for line in stream.iter_lines() if line.startswith("event: ")
            ]
        assert "started" in events
        assert "done" in events


class TestApprovalOverHttp:
    @pytest.fixture
    def gated_client(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[TestClient]:
        gated = settings.model_copy(update={"require_approval_over_usd": 0.0001})
        from scholar_graph.config import set_settings

        set_settings(gated)

        @asynccontextmanager
        async def fake_build_deps(_: Settings | None = None) -> AsyncIterator[Deps]:
            budget = BudgetTracker(gated.max_usd_per_run)
            yield Deps(
                settings=gated,
                provider=FakeAPIProvider(gated, budget, DEFAULT_RESPONSES),
                search=FakeSearchService(gated),
                budget=budget,
            )

        monkeypatch.setattr(service_module, "build_deps", fake_build_deps)
        from scholar_graph.api import app

        with TestClient(app) as test_client:
            yield test_client

    def test_expensive_run_pauses_then_resumes_on_approval(self, gated_client: TestClient) -> None:
        run_id = gated_client.post("/runs", json={"question": QUESTION}).json()["run_id"]

        paused = _wait_for(gated_client, run_id, "awaiting_approval", "failed")
        assert paused["status"] == "awaiting_approval", paused.get("error")
        assert paused["approval"]["reason"] == "cost_approval_required"

        accepted = gated_client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        assert accepted.status_code == 202

        finished = _wait_for(gated_client, run_id, "completed", "failed")
        assert finished["status"] == "completed", finished.get("error")
        assert finished["report"]["summary"]

    def test_rejection_finishes_the_run_without_a_draft(self, gated_client: TestClient) -> None:
        run_id = gated_client.post("/runs", json={"question": QUESTION}).json()["run_id"]
        _wait_for(gated_client, run_id, "awaiting_approval", "failed")

        gated_client.post(f"/runs/{run_id}/approval", json={"decision": "reject"})
        finished = _wait_for(gated_client, run_id, "completed", "failed")

        assert finished["status"] == "completed"
        assert any("rejected" in w.lower() for w in finished["report"]["warnings"])

    def test_an_invalid_decision_is_rejected(self, gated_client: TestClient) -> None:
        run_id = gated_client.post("/runs", json={"question": QUESTION}).json()["run_id"]
        _wait_for(gated_client, run_id, "awaiting_approval", "failed")
        response = gated_client.post(f"/runs/{run_id}/approval", json={"decision": "maybe"})
        assert response.status_code == 422
