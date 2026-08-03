"""Domain types.

These are the nouns the whole system agrees on. Nodes read and write these;
the API serialises them; the evals score them. Keeping them in one module
means a schema change shows up as one diff rather than five.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CITATION_RE = re.compile(r"\[S(\d+)\]")


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


class Document(Frozen):
    """A candidate source returned by a search tool.

    ``source_id`` is stable across runs (derived from the external id), which
    is what makes replayed runs comparable and citations checkable.
    """

    source_id: str
    provider: Literal["openalex", "arxiv"]
    external_id: str
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    published: date | None = None
    venue: str | None = None
    url: str | None = None
    citation_count: int = 0

    @staticmethod
    def make_source_id(provider: str, external_id: str) -> str:
        digest = hashlib.sha256(f"{provider}:{external_id}".encode()).hexdigest()
        return f"{provider[:2]}-{digest[:10]}"

    @property
    def citation_label(self) -> str:
        first_author = self.authors[0].split()[-1] if self.authors else "Anon"
        year = self.published.year if self.published else "n.d."
        return f"{first_author} {year}"

    def as_context(self, index: int) -> str:
        """Render for an LLM prompt with the marker the model must cite by."""
        parts = [
            f"[S{index}] {self.title}",
            f"    authors: {', '.join(self.authors[:4]) or 'unknown'}",
            f"    venue: {self.venue or 'unknown'} ({self.published or 'no date'})",
            f"    abstract: {self.abstract.strip() or '(no abstract available)'}",
        ]
        return "\n".join(parts)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretation: str = Field(
        description="What the question is actually asking, in one sentence."
    )
    sub_questions: list[str] = Field(description="Independent facets that must each be answered.")
    search_queries: list[str] = Field(description="Literal keyword queries for a paper database.")
    success_criteria: list[str] = Field(
        description="What a complete answer must contain, used to decide when to stop searching."
    )


class ScreeningVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    relevant: bool
    reason: str


class ScreeningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[ScreeningVerdict]


class Note(BaseModel):
    """One extracted finding, bound to the exact span it came from.

    ``quote`` must appear verbatim in the source document — that is the
    invariant the verifier checks, and it is what makes a citation mean
    something rather than decorate something.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    claim: str
    quote: str


class NoteSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[Note]


class CoverageAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    satisfied_criteria: list[str]
    unmet_criteria: list[str]
    follow_up_queries: list[str] = Field(
        description="Empty when coverage is sufficient; otherwise new queries to run."
    )


class DraftReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="The direct answer, 2-4 sentences, every claim cited [Sn].")
    body: str = Field(description="Markdown. Every factual sentence carries a [Sn] marker.")
    limitations: str = Field(description="What the evidence does not establish.")

    def cited_indices(self) -> set[int]:
        text = f"{self.summary}\n{self.body}"
        return {int(m) for m in CITATION_RE.findall(text)}


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class CitationIssue(Frozen):
    kind: Literal["unknown_marker", "unsupported_quote", "uncited_section"]
    detail: str
    marker: int | None = None


class VerificationReport(Frozen):
    checked_markers: int
    valid_markers: int
    grounded_notes: int
    total_notes: int
    issues: tuple[CitationIssue, ...] = ()

    @property
    def citation_precision(self) -> float:
        if self.checked_markers == 0:
            return 1.0
        return self.valid_markers / self.checked_markers

    @property
    def groundedness(self) -> float:
        if self.total_notes == 0:
            return 1.0
        return self.grounded_notes / self.total_notes

    @property
    def passed(self) -> bool:
        return not self.issues


class PanelCritique(Frozen):
    reviewer: str
    verdict: Literal["accept", "revise"]
    comments: str


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


class CostBreakdown(Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0
    calls: int = 0

    def merged(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            usd=round(self.usd + other.usd, 8),
            calls=self.calls + other.calls,
        )


class ResearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    question: str
    summary: str
    body: str
    limitations: str
    sources: list[Document]
    verification: VerificationReport
    critiques: list[PanelCritique] = Field(default_factory=list)
    cost: CostBreakdown
    search_rounds: int
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.question}",
            "",
            self.summary,
            "",
            "## Findings",
            "",
            self.body,
            "",
            "## Limitations",
            "",
            self.limitations,
            "",
            "## Sources",
            "",
        ]
        for i, doc in enumerate(self.sources, start=1):
            url = f" <{doc.url}>" if doc.url else ""
            lines.append(f"{i}. **[S{i}]** {doc.title} — {doc.citation_label}.{url}")
        if self.warnings:
            lines += ["", "## Warnings", ""]
            lines += [f"- {w}" for w in self.warnings]
        lines += [
            "",
            "---",
            "",
            f"_Citation precision {self.verification.citation_precision:.0%} · "
            f"groundedness {self.verification.groundedness:.0%} · "
            f"{self.cost.calls} model calls · ${self.cost.usd:.4f}_",
        ]
        return "\n".join(lines)
