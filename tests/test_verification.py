"""Verification is the quality gate, so it gets the most adversarial tests."""

from __future__ import annotations

from scholar_graph.domain import DraftReport, Note
from scholar_graph.verification import quote_is_supported, verify
from tests.conftest import DOC_A, DOC_B


def _draft(
    summary: str = "Summary [S1].", body: str = "", limitations: str = "None."
) -> DraftReport:
    return DraftReport(summary=summary, body=body, limitations=limitations)


class TestQuoteMatching:
    def test_verbatim_quote_is_supported(self) -> None:
        assert quote_is_supported("sparse attention patterns reduce inference cost", DOC_A.abstract)

    def test_case_and_whitespace_differences_are_tolerated(self) -> None:
        assert quote_is_supported(
            "Sparse   Attention  Patterns\nReduce Inference Cost", DOC_A.abstract
        )

    def test_minor_transcription_drift_is_tolerated(self) -> None:
        # A dropped article is transcription noise, not invention.
        assert quote_is_supported(
            "sparse attention patterns reduce inference cost by up to 40 percent on sequences",
            DOC_A.abstract,
        )

    def test_fabricated_quote_is_rejected(self) -> None:
        assert not quote_is_supported(
            "sparse attention eliminates inference cost entirely", DOC_A.abstract
        )

    def test_reordered_content_is_rejected(self) -> None:
        # Order matters: the same words rearranged can invert a claim's meaning.
        assert not quote_is_supported(
            "accuracy reduce cost inference preserving patterns attention sparse downstream",
            DOC_A.abstract,
        )

    def test_empty_quote_is_rejected(self) -> None:
        assert not quote_is_supported("", DOC_A.abstract)
        assert not quote_is_supported("   ", DOC_A.abstract)


class TestMarkerResolution:
    def test_valid_markers_pass(self) -> None:
        draft = _draft(summary="Cost falls [S1] and memory falls [S2].")
        report = verify(draft, [], [DOC_A, DOC_B])
        assert report.checked_markers == 2
        assert report.valid_markers == 2
        assert report.passed

    def test_marker_beyond_source_list_is_flagged(self) -> None:
        draft = _draft(summary="A claim [S7].")
        report = verify(draft, [], [DOC_A])
        assert not report.passed
        assert [i.kind for i in report.issues] == ["unknown_marker"]
        assert report.issues[0].marker == 7
        assert report.citation_precision == 0.0

    def test_precision_is_a_ratio_not_a_boolean(self) -> None:
        draft = _draft(summary="One good [S1], one bad [S9].")
        report = verify(draft, [], [DOC_A])
        assert report.citation_precision == 0.5


class TestNoteGrounding:
    def test_grounded_note_counts(self) -> None:
        note = Note(
            source_id=DOC_A.source_id,
            claim="Cost falls.",
            quote="reduce inference cost by up to 40 percent",
        )
        report = verify(_draft(), [note], [DOC_A])
        assert report.grounded_notes == 1
        assert report.groundedness == 1.0

    def test_fabricated_quote_is_flagged(self) -> None:
        note = Note(
            source_id=DOC_A.source_id,
            claim="Cost vanishes.",
            quote="inference is now completely free of cost",
        )
        report = verify(_draft(), [note], [DOC_A])
        assert report.groundedness == 0.0
        assert any(i.kind == "unsupported_quote" for i in report.issues)

    def test_note_citing_an_unretrieved_source_is_flagged(self) -> None:
        note = Note(source_id="xx-deadbeef", claim="Something.", quote="anything")
        report = verify(_draft(), [note], [DOC_A])
        assert any("unknown source_id" in i.detail for i in report.issues)

    def test_note_attributed_to_the_wrong_source_is_caught(self) -> None:
        # The quote is real, but it belongs to a different paper. This is the
        # failure mode a human reader would almost never catch by eye.
        note = Note(
            source_id=DOC_B.source_id,
            claim="Sparse attention cuts cost.",
            quote="sparse attention patterns reduce inference cost by up to 40 percent",
        )
        report = verify(_draft(), [note], [DOC_A, DOC_B])
        assert report.groundedness == 0.0
        assert any(i.kind == "unsupported_quote" for i in report.issues)


class TestParagraphCitation:
    def test_long_uncited_paragraph_is_flagged(self) -> None:
        body = (
            "This is a substantive paragraph making a factual assertion about inference "
            "cost and quality trade-offs, and it carries no citation marker anywhere in it."
        )
        report = verify(_draft(summary="Fine [S1].", body=body), [], [DOC_A])
        assert any(i.kind == "uncited_section" for i in report.issues)

    def test_short_connective_text_is_not_required_to_cite(self) -> None:
        report = verify(_draft(summary="Fine [S1].", body="In short:"), [], [DOC_A])
        assert not any(i.kind == "uncited_section" for i in report.issues)

    def test_headings_are_not_required_to_cite(self) -> None:
        body = "## A heading that is quite long but is still only a heading, not a claim"
        report = verify(_draft(summary="Fine [S1].", body=body), [], [DOC_A])
        assert not any(i.kind == "uncited_section" for i in report.issues)


class TestEmptyCases:
    def test_no_markers_and_no_notes_is_vacuously_clean(self) -> None:
        report = verify(_draft(summary="No claims here.", body=""), [], [])
        assert report.passed
        assert report.citation_precision == 1.0
        assert report.groundedness == 1.0
