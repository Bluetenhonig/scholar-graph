# ADR 0003 — Write an AutoGen adapter instead of using its Anthropic client

**Status:** accepted

## Context

The review board is built with AutoGen (`RoundRobinGroupChat` over three
`AssistantAgent`s). AutoGen ships `AnthropicChatCompletionClient`, so wiring it
up is roughly:

```python
model_client = AnthropicChatCompletionClient(model="claude-haiku-4-5", api_key=...)
```

Two lines. The alternative is implementing AutoGen's `ChatCompletionClient`
interface over this project's own `LLMProvider` — about 80 lines, plus the risk
of tracking an interface we do not control.

## Decision

Write the adapter (`src/scholar_graph/panel/model_client.py`).

## Rationale

`LLMProvider` is a deliberate chokepoint. Three system-level properties depend
on *every* model call passing through it:

1. **Record/replay determinism** — a call that bypasses cassettes makes the
   whole run non-reproducible.
2. **Cost accounting** — panel spend is real spend and belongs in the same
   ledger as everything else.
3. **Budget enforcement** — a ceiling that a subsystem can spend past is not a
   ceiling.

A property that holds for *most* call sites is not a property; it is a
coincidence waiting to be broken. Using AutoGen's client would have punched a
hole through all three at once, and the hole would have been invisible — the
panel would appear to work perfectly while quietly spending unbudgeted money
and making runs unreproducible.

The concrete payoff: the multi-agent panel replays offline like everything
else, its calls appear in the same structured logs with the same `run_id`, and
`test_panel.py` asserts that panel spend lands in the run's budget.

## Implementation notes

- Only the subset AutoGen actually uses is implemented: `create`,
  `create_stream`, usage accounting, token counting, `model_info`.
- `create_stream` yields one chunk. A replayed cassette has nothing to stream,
  and panel output is never rendered token-by-token, so nothing is lost.
- Tool use raises `NotImplementedError` rather than silently ignoring tools —
  the panel is a critique loop, and a silently-dropped tool would be a
  confusing bug rather than an obvious one.
- `count_tokens` is a crude `len // 4`. AutoGen uses it only for
  context-window bookkeeping; the authoritative numbers come back in usage.

## Consequences

**Good:** the guarantees hold everywhere, without exception.

**Bad:** the adapter is coupled to an interface owned by another project and
will need maintenance across AutoGen major versions. Accepted: the surface is
small, and `test_panel.py` covers it directly, so a breaking upstream change
fails a test rather than silently degrading production behaviour.
