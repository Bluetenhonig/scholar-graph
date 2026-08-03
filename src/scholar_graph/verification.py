"""Citation verification.

The failure mode that matters for a research agent is not "wrong answer" —
it is "plausible answer with citations that do not support it". A reader
cannot tell the difference without doing the work themselves, which defeats
the point of the agent.

So citations are checked mechanically, before the report is returned:

1. every ``[Sn]`` marker resolves to a real retrieved source;
2. every note the report was written from quotes text that actually appears
   in the source it names;
3. every substantive paragraph carries at least one marker.

None of this needs a model, which is what makes it trustworthy — the checker
cannot be talked out of its verdict by the thing it is checking.
"""

from __future__ import annotations

import re

from scholar_graph.domain import (
    CITATION_RE,
    CitationIssue,
    Document,
    DraftReport,
    Note,
    VerificationReport,
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Below this share of the quote's tokens recoverable in order from the source,
# a quote is treated as fabricated rather than loosely transcribed.
QUOTE_MATCH_THRESHOLD = 0.9

# Paragraphs shorter than this are treated as connective tissue (lead-ins,
# transitions) and are not required to carry a citation.
MIN_CITED_PARAGRAPH_CHARS = 120


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _subsequence_ratio(needle: list[str], haystack: list[str]) -> float:
    """Share of ``needle`` tokens appearing in ``haystack`` in order.

    Tolerates the small transcription drift models produce (a dropped
    article, a normalised hyphen) without tolerating invention: reordered or
    absent content scores low because order is preserved.
    """
    if not needle:
        return 1.0
    matched = 0
    position = 0
    for token in needle:
        try:
            position = haystack.index(token, position) + 1
        except ValueError:
            continue
        matched += 1
    return matched / len(needle)


def quote_is_supported(quote: str, source_text: str) -> bool:
    quote_tokens = _tokens(quote)
    if not quote_tokens:
        return False

    source_tokens = _tokens(source_text)
    if not source_tokens:
        return False

    # Fast path: normalised verbatim containment.
    if " ".join(quote_tokens) in " ".join(source_tokens):
        return True

    return _subsequence_ratio(quote_tokens, source_tokens) >= QUOTE_MATCH_THRESHOLD


def _substantive_paragraphs(markdown: str) -> list[str]:
    paragraphs: list[str] = []
    for block in markdown.split("\n\n"):
        stripped = block.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if len(stripped) < MIN_CITED_PARAGRAPH_CHARS:
            continue
        paragraphs.append(stripped)
    return paragraphs


def verify(
    draft: DraftReport,
    notes: list[Note],
    sources: list[Document],
) -> VerificationReport:
    """Check ``draft`` against the evidence it was supposedly written from."""
    issues: list[CitationIssue] = []
    by_source_id = {doc.source_id: doc for doc in sources}
    valid_indices = set(range(1, len(sources) + 1))

    # 1. Markers resolve to retrieved sources.
    markers = [int(m) for m in CITATION_RE.findall(f"{draft.summary}\n{draft.body}")]
    valid_markers = 0
    for marker in markers:
        if marker in valid_indices:
            valid_markers += 1
        else:
            issues.append(
                CitationIssue(
                    kind="unknown_marker",
                    marker=marker,
                    detail=(
                        f"[S{marker}] does not correspond to any of the "
                        f"{len(sources)} retrieved sources."
                    ),
                )
            )

    # 2. Notes quote text that exists in the source they name.
    grounded = 0
    for note in notes:
        document = by_source_id.get(note.source_id)
        if document is None:
            issues.append(
                CitationIssue(
                    kind="unsupported_quote",
                    detail=f"Note cites unknown source_id {note.source_id!r}.",
                )
            )
            continue
        source_text = f"{document.title}\n{document.abstract}"
        if quote_is_supported(note.quote, source_text):
            grounded += 1
        else:
            issues.append(
                CitationIssue(
                    kind="unsupported_quote",
                    detail=(
                        f"Quote attributed to {document.citation_label} "
                        f"({note.source_id}) does not appear in that source: "
                        f"{note.quote[:120]!r}"
                    ),
                )
            )

    # 3. Substantive prose is cited.
    for paragraph in _substantive_paragraphs(draft.body):
        if not CITATION_RE.search(paragraph):
            issues.append(
                CitationIssue(
                    kind="uncited_section",
                    detail=f"Uncited paragraph: {paragraph[:120]!r}",
                )
            )

    return VerificationReport(
        checked_markers=len(markers),
        valid_markers=valid_markers,
        grounded_notes=grounded,
        total_notes=len(notes),
        issues=tuple(issues),
    )


def format_issues_for_revision(report: VerificationReport, limit: int = 12) -> str:
    """Render issues as instructions the drafting model can act on."""
    lines = []
    for issue in report.issues[:limit]:
        lines.append(f"- ({issue.kind}) {issue.detail}")
    if len(report.issues) > limit:
        lines.append(f"- …and {len(report.issues) - limit} more of the same kinds.")
    return "\n".join(lines)
