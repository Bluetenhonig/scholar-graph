"""Per-run spend accounting and enforcement.

An agent that can loop is an agent that can loop expensively. The budget is
checked *before* each call, not tallied after it, so the ceiling is a ceiling
rather than a post-mortem.
"""

from __future__ import annotations

import threading

from scholar_graph.domain import CostBreakdown


class BudgetExceeded(RuntimeError):
    def __init__(self, spent: float, limit: float, purpose: str) -> None:
        self.spent = spent
        self.limit = limit
        self.purpose = purpose
        super().__init__(
            f"Budget exhausted before {purpose!r}: ${spent:.4f} spent of ${limit:.4f} limit."
        )


class BudgetTracker:
    """Thread-safe running total for one research run."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self._total = CostBreakdown()
        self._lock = threading.Lock()

    @property
    def total(self) -> CostBreakdown:
        with self._lock:
            return self._total

    @property
    def spent_usd(self) -> float:
        return self.total.usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def check(self, purpose: str) -> None:
        """Raise if there is no headroom left for another call."""
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(self.spent_usd, self.limit_usd, purpose)

    def can_afford(self, estimated_usd: float) -> bool:
        return self.spent_usd + estimated_usd <= self.limit_usd

    def record(self, cost: CostBreakdown) -> CostBreakdown:
        with self._lock:
            self._total = self._total.merged(cost)
            return self._total
