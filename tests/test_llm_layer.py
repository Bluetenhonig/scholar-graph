"""Cassettes, cost accounting, budget enforcement, schema normalisation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from scholar_graph.config import LLMMode, Settings
from scholar_graph.domain import CostBreakdown
from scholar_graph.llm.budget import BudgetExceeded, BudgetTracker
from scholar_graph.llm.cassette import CassetteMiss, CassetteStore, cache_key
from scholar_graph.llm.pricing import estimate_cost, price_for
from scholar_graph.llm.provider import LLMProvider, LLMRefusal, LLMResponse, LLMTruncated
from scholar_graph.llm.schema import to_output_schema
from tests.conftest import FakeAPIProvider


class Sample(BaseModel):
    name: str = Field(description="A name", min_length=2)
    score: int = Field(ge=0, le=10)
    tags: list[str]


class TestSchemaNormalisation:
    def test_objects_forbid_extra_properties(self) -> None:
        schema = to_output_schema(Sample)
        assert schema["additionalProperties"] is False

    def test_every_property_is_required(self) -> None:
        schema = to_output_schema(Sample)
        assert schema["required"] == ["name", "score", "tags"]

    def test_unsupported_constraints_are_stripped(self) -> None:
        schema = to_output_schema(Sample)
        name = schema["properties"]["name"]
        score = schema["properties"]["score"]
        assert "minLength" not in name
        assert "minimum" not in score and "maximum" not in score

    def test_descriptions_survive(self) -> None:
        # Descriptions are how the model learns what a field means; stripping
        # them along with the constraints would be a silent quality regression.
        assert to_output_schema(Sample)["properties"]["name"]["description"] == "A name"

    def test_nested_models_are_normalised_too(self) -> None:
        class Outer(BaseModel):
            inner: Sample

        schema = to_output_schema(Outer)
        defs = schema.get("$defs", {})
        assert defs["Sample"]["additionalProperties"] is False


class TestCacheKey:
    def test_key_is_order_independent(self) -> None:
        assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})

    def test_key_changes_when_prompt_changes(self) -> None:
        assert cache_key({"system": "x"}) != cache_key({"system": "y"})


class TestCassetteStore:
    def test_round_trip(self, tmp_path: Any) -> None:
        store = CassetteStore(tmp_path, "llm")
        key = cache_key({"model": "m"})
        store.save(key, {"model": "m"}, {"text": "hello"})
        loaded = store.load(key)
        assert loaded is not None
        assert loaded["response"]["text"] == "hello"

    def test_miss_returns_none(self, tmp_path: Any) -> None:
        assert CassetteStore(tmp_path, "llm").load("deadbeef") is None

    def test_count_reflects_saved_entries(self, tmp_path: Any) -> None:
        store = CassetteStore(tmp_path, "llm")
        assert store.count() == 0
        store.save(cache_key({"a": 1}), {"a": 1}, {"text": "x"})
        assert store.count() == 1


class TestPricing:
    def test_opus_costs_more_than_haiku(self) -> None:
        assert (
            price_for("claude-opus-5").output_per_mtok
            > price_for("claude-haiku-4-5").output_per_mtok
        )

    def test_unknown_model_is_priced_conservatively(self) -> None:
        # Never make an unpriced model look free.
        assert price_for("claude-does-not-exist").input_per_mtok > 0

    def test_cost_arithmetic(self) -> None:
        cost = estimate_cost("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
        assert cost.usd == pytest.approx(5.00)

    def test_cache_reads_are_cheaper_than_fresh_input(self) -> None:
        fresh = estimate_cost("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
        cached = estimate_cost(
            "claude-opus-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
        )
        assert cached.usd < fresh.usd

    def test_breakdowns_merge(self) -> None:
        a = CostBreakdown(input_tokens=10, usd=0.01, calls=1)
        b = CostBreakdown(input_tokens=5, usd=0.02, calls=1)
        merged = a.merged(b)
        assert merged.input_tokens == 15
        assert merged.calls == 2
        assert merged.usd == pytest.approx(0.03)


class TestBudget:
    def test_check_passes_with_headroom(self) -> None:
        BudgetTracker(1.0).check("anything")

    def test_check_raises_once_exhausted(self) -> None:
        tracker = BudgetTracker(0.01)
        tracker.record(CostBreakdown(usd=0.02, calls=1))
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check("synthesize")
        assert "synthesize" in str(exc.value)

    def test_remaining_never_goes_negative(self) -> None:
        tracker = BudgetTracker(0.01)
        tracker.record(CostBreakdown(usd=5.0, calls=1))
        assert tracker.remaining_usd == 0.0

    def test_can_afford_looks_ahead(self) -> None:
        tracker = BudgetTracker(0.10)
        tracker.record(CostBreakdown(usd=0.09, calls=1))
        assert tracker.can_afford(0.005)
        assert not tracker.can_afford(0.05)


class TestProviderRecordReplay:
    """The end-to-end promise: record once, replay forever, offline."""

    async def test_record_then_replay_returns_identical_text(self, tmp_path: Any) -> None:
        recording = Settings(
            llm_mode=LLMMode.record,
            anthropic_api_key="test",
            cassette_dir=tmp_path,
            max_usd_per_run=1.0,
        )
        budget = BudgetTracker(1.0)
        recorder = FakeAPIProvider(
            recording,
            budget,
            {
                "plan": [
                    '{"interpretation":"x","sub_questions":[],'
                    '"search_queries":[],"success_criteria":[]}'
                ]
            },
        )
        text, cost = await recorder.text(
            purpose="plan", system="S", user="U", model="claude-opus-5", effort="low"
        )
        assert cost.usd > 0

        # A *real* provider, no API key path taken, reading what was recorded.
        replaying = recording.model_copy(update={"llm_mode": LLMMode.replay})
        replayer = LLMProvider(replaying, BudgetTracker(1.0))
        replayed, replayed_cost = await replayer.text(
            purpose="plan", system="S", user="U", model="claude-opus-5", effort="low"
        )
        assert replayed == text
        assert replayed_cost.usd == pytest.approx(cost.usd)

    async def test_replay_cost_still_counts_against_budget(self, tmp_path: Any) -> None:
        # Replayed runs must report what they *would* have cost, or the eval
        # suite's cost assertions would be meaningless.
        recording = Settings(llm_mode=LLMMode.record, anthropic_api_key="t", cassette_dir=tmp_path)
        await FakeAPIProvider(recording, BudgetTracker(1.0), {"p": ["ok"]}).text(
            purpose="p", system="S", user="U", model="claude-opus-5", effort="low"
        )

        replaying = recording.model_copy(update={"llm_mode": LLMMode.replay})
        tracker = BudgetTracker(1.0)
        await LLMProvider(replaying, tracker).text(
            purpose="p", system="S", user="U", model="claude-opus-5", effort="low"
        )
        assert tracker.spent_usd > 0

    async def test_changed_prompt_misses_rather_than_silently_reusing(self, tmp_path: Any) -> None:
        recording = Settings(llm_mode=LLMMode.record, anthropic_api_key="t", cassette_dir=tmp_path)
        await FakeAPIProvider(recording, BudgetTracker(1.0), {"p": ["ok"]}).text(
            purpose="p", system="ORIGINAL", user="U", model="claude-opus-5", effort="low"
        )

        replaying = recording.model_copy(update={"llm_mode": LLMMode.replay})
        with pytest.raises(CassetteMiss) as exc:
            await LLMProvider(replaying, BudgetTracker(1.0)).text(
                purpose="p", system="EDITED", user="U", model="claude-opus-5", effort="low"
            )
        # The error must tell you how to fix it, not just that it happened.
        assert "SCHOLAR_GRAPH_LLM_MODE=record" in str(exc.value)


class TestProviderErrorHandling:
    async def test_refusal_raises(self, settings: Settings, budget: BudgetTracker) -> None:
        class Refusing(LLMProvider):
            async def _call_api(self, request: dict[str, Any], purpose: str) -> LLMResponse:
                raise LLMRefusal(purpose, "cyber")

        with pytest.raises(LLMRefusal):
            await Refusing(settings, budget).text(purpose="p", system="s", user="u")

    async def test_truncation_raises_with_actionable_message(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        class Truncating(LLMProvider):
            async def _call_api(self, request: dict[str, Any], purpose: str) -> LLMResponse:
                raise LLMTruncated(f"{purpose}: hit max_tokens (10).")

        with pytest.raises(LLMTruncated, match="max_tokens"):
            await Truncating(settings, budget).text(purpose="p", system="s", user="u")

    async def test_malformed_json_names_the_model_that_produced_it(
        self, settings: Settings, budget: BudgetTracker
    ) -> None:
        provider = FakeAPIProvider(settings, budget, {"p": ["{not json at all"]})
        with pytest.raises(ValueError, match="Sample"):
            await provider.structured(purpose="p", system="s", user="u", response_model=Sample)

    async def test_budget_is_checked_before_the_call_not_after(self, settings: Settings) -> None:
        tracker = BudgetTracker(0.0001)
        provider = FakeAPIProvider(settings, tracker, {"p": ["ok"]})
        await provider.text(purpose="p", system="s", user="u")  # first call allowed
        with pytest.raises(BudgetExceeded):
            await provider.text(purpose="p", system="s", user="u")
        assert provider.calls == ["p"]  # the second call never reached the API
