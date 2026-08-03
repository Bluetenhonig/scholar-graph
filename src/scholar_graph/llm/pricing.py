"""Token pricing.

Rates are USD per million tokens, from the Anthropic pricing page as of
2026-08. They are checked into the repo deliberately: a cost regression in
CI should be a failing test, not a surprise on next month's invoice. When
rates move, update this table and the eval baselines move with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholar_graph.domain import CostBreakdown

CACHE_READ_MULTIPLIER = 0.1
"""Cache reads bill at ~10% of the input rate."""

CACHE_WRITE_MULTIPLIER = 1.25
"""5-minute-TTL cache writes bill at ~125% of the input rate."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

UNKNOWN_MODEL_PRICE = ModelPrice(5.00, 25.00)
"""Unknown models are priced as Opus so an unpriced model never looks free."""


def price_for(model: str) -> ModelPrice:
    return PRICES.get(model, UNKNOWN_MODEL_PRICE)


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> CostBreakdown:
    price = price_for(model)
    usd = (
        input_tokens * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * price.input_per_mtok * CACHE_READ_MULTIPLIER
        + cache_write_tokens * price.input_per_mtok * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000
    return CostBreakdown(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        usd=round(usd, 8),
        calls=1,
    )
