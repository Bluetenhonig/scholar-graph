"""An AutoGen ``ChatCompletionClient`` backed by our own provider.

AutoGen ships an Anthropic client, and using it would have been two lines.
It is not used here on purpose: it would open a second path to the model that
bypasses cassettes, cost accounting and the run budget. Every one of those is
a system-level property, and a property that holds for "most" call sites is
not a property.

Implementing ~80 lines of adapter buys: the review panel replays offline like
everything else, its spend lands in the same budget, and its calls appear in
the same structured logs.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)

from scholar_graph.domain import CostBreakdown
from scholar_graph.llm.provider import Effort, LLMProvider

CONTEXT_WINDOW = 1_000_000
CHARS_PER_TOKEN = 4
"""Deliberately crude. AutoGen uses token counts only for context-window
bookkeeping; the authoritative numbers come back from the API in usage."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part for part in content if isinstance(part, str))
    return str(content)


def _split_messages(messages: Sequence[LLMMessage]) -> tuple[str, str]:
    """Flatten an AutoGen conversation into (system prompt, user turn)."""
    system_parts: list[str] = []
    dialogue: list[str] = []

    for message in messages:
        text = _content_to_text(getattr(message, "content", ""))
        if not text.strip():
            continue
        if isinstance(message, SystemMessage):
            system_parts.append(text)
        elif isinstance(message, UserMessage):
            dialogue.append(f"{getattr(message, 'source', 'user')}: {text}")
        elif isinstance(message, AssistantMessage):
            dialogue.append(f"{getattr(message, 'source', 'assistant')}: {text}")
        else:
            dialogue.append(text)

    return "\n\n".join(system_parts), "\n\n".join(dialogue)


class ProviderChatCompletionClient(ChatCompletionClient):
    """Adapts :class:`LLMProvider` to the interface AutoGen agents expect."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        purpose: str,
        effort: Effort = "low",
        max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self._model = model
        self._purpose = purpose
        self._effort = effort
        self._max_tokens = max_tokens
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # -- required surface -------------------------------------------------

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        if tools:
            # The panel is a critique loop, not a tool-using agent. Failing
            # loudly beats silently ignoring a tool the caller expected to work.
            raise NotImplementedError(
                "ProviderChatCompletionClient does not expose tools to panel agents."
            )

        system, user = _split_messages(messages)
        text, cost = await self._provider.text(
            purpose=self._purpose,
            system=system or "You are a helpful reviewer.",
            user=user or "(no content)",
            model=self._model,
            effort=self._effort,
            max_tokens=self._max_tokens,
        )
        self._record(cost)

        return CreateResult(
            finish_reason="stop",
            content=text,
            usage=self._last,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        # Cassette replay has nothing to stream, so the "stream" is one chunk.
        # Panel output is never rendered token by token, so nothing is lost.
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        system, user = _split_messages(messages)
        return (len(system) + len(user)) // CHARS_PER_TOKEN

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return CONTEXT_WINDOW - self.count_tokens(messages, tools=tools)

    async def close(self) -> None:
        # The provider is owned by the graph run, not by this adapter.
        return None

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            structured_output=False,
            family=ModelFamily.UNKNOWN,
            multiple_system_messages=True,
        )

    # -- internals --------------------------------------------------------

    def _record(self, cost: CostBreakdown) -> None:
        self._last = RequestUsage(
            prompt_tokens=cost.input_tokens,
            completion_tokens=cost.output_tokens,
        )
        self._total = RequestUsage(
            prompt_tokens=self._total.prompt_tokens + cost.input_tokens,
            completion_tokens=self._total.completion_tokens + cost.output_tokens,
        )
