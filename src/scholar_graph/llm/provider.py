"""The single place this codebase talks to a language model.

Every model call in the system — graph nodes and the AutoGen review panel
alike — goes through :class:`LLMProvider`. That chokepoint is what makes
record/replay, cost accounting and budget enforcement properties of the
system rather than things each call site has to remember.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from scholar_graph.config import LLMMode, Settings, get_settings
from scholar_graph.domain import CostBreakdown
from scholar_graph.llm.budget import BudgetTracker
from scholar_graph.llm.cassette import CassetteMiss, CassetteStore, cache_key
from scholar_graph.llm.pricing import estimate_cost
from scholar_graph.llm.schema import to_output_schema
from scholar_graph.observability import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class LLMRefusal(RuntimeError):
    """Safety classifiers declined the request."""

    def __init__(self, purpose: str, category: str | None) -> None:
        self.purpose = purpose
        self.category = category
        super().__init__(f"Model declined {purpose!r} (category={category or 'unspecified'}).")


class LLMTruncated(RuntimeError):
    """The response hit max_tokens mid-structure and cannot be parsed."""


class LLMResponse(BaseModel):
    text: str
    cost: CostBreakdown
    stop_reason: str | None = None
    replayed: bool = False


class LLMProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        budget: BudgetTracker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget = budget or BudgetTracker(self.settings.max_usd_per_run)
        self.cassettes = CassetteStore(self.settings.cassette_dir, "llm")
        self._client: Any = None

    # -- client -----------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self.settings.require_api_key(),
                max_retries=3,
                timeout=120.0,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- public API -------------------------------------------------------

    async def structured(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        response_model: type[T],
        model: str | None = None,
        effort: Effort = "medium",
        max_tokens: int | None = None,
    ) -> tuple[T, CostBreakdown]:
        """Call the model and parse the reply into ``response_model``.

        The schema is enforced server-side by structured outputs *and*
        validated client-side by pydantic, because a schema-shaped response
        can still be a semantically empty one.
        """
        schema = to_output_schema(response_model)
        response = await self._complete(
            purpose=purpose,
            system=system,
            user=user,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            output_format={"type": "json_schema", "schema": schema},
        )
        try:
            parsed = response_model.model_validate_json(response.text)
        except ValidationError as exc:
            raise ValueError(
                f"{purpose}: model returned JSON that does not satisfy "
                f"{response_model.__name__}: {exc}"
            ) from exc
        return parsed, response.cost

    async def text(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        model: str | None = None,
        effort: Effort = "medium",
        max_tokens: int | None = None,
    ) -> tuple[str, CostBreakdown]:
        response = await self._complete(
            purpose=purpose,
            system=system,
            user=user,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            output_format=None,
        )
        return response.text, response.cost

    # -- core -------------------------------------------------------------

    async def _complete(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        model: str | None,
        effort: Effort,
        max_tokens: int | None,
        output_format: dict[str, Any] | None,
    ) -> LLMResponse:
        model = model or self.settings.reasoning_model
        max_tokens = max_tokens or self.settings.max_tokens

        output_config: dict[str, Any] = {"effort": effort}
        if output_format is not None:
            output_config["format"] = output_format

        request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        key = cache_key(request)

        if self.settings.llm_mode is LLMMode.replay:
            return self._replay(key, purpose, request)

        self.budget.check(purpose)
        response = await self._call_api(request, purpose)

        if self.settings.llm_mode is LLMMode.record:
            path = self.cassettes.save(
                key,
                request,
                {
                    "text": response.text,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.cost.input_tokens,
                        "output_tokens": response.cost.output_tokens,
                        "cache_read_tokens": response.cost.cache_read_tokens,
                        "cache_write_tokens": response.cost.cache_write_tokens,
                    },
                    "purpose": purpose,
                },
            )
            log.debug("cassette.recorded", purpose=purpose, path=str(path))

        total = self.budget.record(response.cost)
        log.info(
            "llm.call",
            purpose=purpose,
            model=model,
            effort=effort,
            input_tokens=response.cost.input_tokens,
            output_tokens=response.cost.output_tokens,
            usd=round(response.cost.usd, 6),
            run_usd=round(total.usd, 6),
        )
        return response

    def _replay(self, key: str, purpose: str, request: dict[str, Any]) -> LLMResponse:
        stored = self.cassettes.load(key)
        if stored is None:
            raise CassetteMiss(key, purpose, self.cassettes.path_for(key))

        payload = stored["response"]
        usage = payload.get("usage", {})
        cost = estimate_cost(
            request["model"],
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            cache_write_tokens=usage.get("cache_write_tokens", 0),
        )
        # Replayed runs still account for cost. The number is what the run
        # *would* have cost, which is what makes the eval suite's cost
        # assertions meaningful without spending anything.
        self.budget.record(cost)
        log.debug("llm.replay", purpose=purpose, key=key[:12], usd=round(cost.usd, 6))
        return LLMResponse(
            text=payload["text"],
            cost=cost,
            stop_reason=payload.get("stop_reason"),
            replayed=True,
        )

    async def _call_api(self, request: dict[str, Any], purpose: str) -> LLMResponse:
        client = self._get_client()
        response = await client.messages.create(
            model=request["model"],
            max_tokens=request["max_tokens"],
            system=[
                {
                    "type": "text",
                    "text": request["system"],
                    # No-ops below the model's minimum cacheable prefix, and a
                    # meaningful saving above it — the system prompt is the
                    # stable part of every call we make.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=request["messages"],
            output_config=request["output_config"],
        )

        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMRefusal(purpose, category)
        if response.stop_reason == "max_tokens":
            raise LLMTruncated(
                f"{purpose}: hit max_tokens ({request['max_tokens']}). "
                "Raise SCHOLAR_GRAPH_MAX_TOKENS or narrow the request."
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        cost = estimate_cost(
            request["model"],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        return LLMResponse(text=text, cost=cost, stop_reason=response.stop_reason)
