"""Runtime configuration.

Everything the service needs to boot lives here, sourced from environment
variables (or a local ``.env``). There is no other configuration surface —
if it is not in this file, it is not configurable.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class LLMMode(StrEnum):
    """How the LLM layer talks to the outside world.

    ``replay`` is the default on purpose: a freshly cloned repo runs its full
    test suite and a real end-to-end research run with no API key and no spend.
    """

    replay = "replay"
    record = "record"
    live = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCHOLAR_GRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -------------------------------------------------------------
    llm_mode: LLMMode = LLMMode.replay
    anthropic_api_key: str | None = None

    reasoning_model: str = "claude-opus-5"
    """Used for planning, synthesis and verification — the judgement calls."""

    worker_model: str = "claude-haiku-4-5"
    """Used for screening and note extraction — high volume, low difficulty."""

    max_tokens: int = 16_000

    # --- Budgets ---------------------------------------------------------
    max_usd_per_run: float = Field(default=0.50, gt=0)
    """Hard ceiling. The graph degrades gracefully rather than blowing past it."""

    max_search_rounds: int = Field(default=3, ge=1, le=10)
    max_documents: int = Field(default=24, ge=1, le=200)

    # --- Human in the loop ----------------------------------------------
    require_approval_over_usd: float = Field(default=0.25, ge=0)
    """Runs projected to cost more than this pause for a human decision."""

    enable_review_panel: bool = True
    """AutoGen multi-agent critique pass before the report is finalised."""

    # --- Storage ---------------------------------------------------------
    cassette_dir: Path = REPO_ROOT / "cassettes"

    checkpoint_db: Path | None = REPO_ROOT / ".scholar-graph" / "checkpoints.sqlite"
    """SQLite path for durable checkpoints, or ``None`` for in-memory.

    In-memory is fine for one-shot runs; durability is what lets a run pause
    for approval and resume in a different process, hours later.
    """

    # --- Tools -----------------------------------------------------------
    openalex_mailto: str = "scholar-graph@example.com"
    """OpenAlex asks for a contact address to put callers in the polite pool."""

    http_timeout_seconds: float = 20.0
    http_max_attempts: int = 3

    # --- Observability ---------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("cassette_dir", "checkpoint_db")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                f"llm_mode={self.llm_mode.value} needs SCHOLAR_GRAPH_ANTHROPIC_API_KEY. "
                "Use llm_mode=replay to run against recorded cassettes instead."
            )
        return self.anthropic_api_key


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Override the singleton. Tests use this; application code should not."""
    global _settings
    _settings = settings
