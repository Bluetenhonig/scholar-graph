"""Search tools: parsing, deduplication, retry and failure isolation."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from scholar_graph.config import Settings
from scholar_graph.domain import Document
from scholar_graph.tools import SearchService, dedupe
from scholar_graph.tools.arxiv import API_URL as ARXIV_URL
from scholar_graph.tools.arxiv import ArxivClient
from scholar_graph.tools.http import HttpFetcher
from scholar_graph.tools.openalex import API_URL as OPENALEX_URL
from scholar_graph.tools.openalex import OpenAlexClient, _reconstruct_abstract
from tests.conftest import DOC_A, DOC_B

OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "display_name": "Efficient attention mechanisms",
            "abstract_inverted_index": {"Sparse": [0], "attention": [1], "helps": [2]},
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "publication_date": "2024-05-01",
            "primary_location": {"source": {"display_name": "NeurIPS"}},
            "cited_by_count": 42,
            "doi": "https://doi.org/10.1/abc",
        }
    ]
}

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <title>Quantisation   for
    serving</title>
    <summary>Eight-bit quantisation  works
    well.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Grace Hopper</name></author>
  </entry>
</feed>
"""


class TestAbstractReconstruction:
    def test_inverted_index_becomes_linear_text(self) -> None:
        inverted = {"cost": [2], "Inference": [0], "is": [1], "high": [3]}
        assert _reconstruct_abstract(inverted) == "Inference is cost high"

    def test_repeated_words_land_at_every_position(self) -> None:
        assert _reconstruct_abstract({"the": [0, 2], "big": [1], "dog": [3]}) == "the big the dog"

    def test_missing_index_yields_empty_string(self) -> None:
        assert _reconstruct_abstract(None) == ""
        assert _reconstruct_abstract({}) == ""


class TestOpenAlex:
    @respx.mock
    async def test_parses_a_work(self, fetcher: HttpFetcher) -> None:
        respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json=OPENALEX_PAYLOAD))
        docs = await OpenAlexClient(fetcher, fetcher.settings).search("attention")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.title == "Efficient attention mechanisms"
        assert doc.abstract == "Sparse attention helps"
        assert doc.authors == ("Ada Lovelace",)
        assert doc.venue == "NeurIPS"
        assert doc.citation_count == 42

    @respx.mock
    async def test_records_without_a_title_are_dropped(self, fetcher: HttpFetcher) -> None:
        respx.get(OPENALEX_URL).mock(
            return_value=httpx.Response(200, json={"results": [{"id": "https://x/W1"}]})
        )
        assert await OpenAlexClient(fetcher, fetcher.settings).search("q") == []


class TestArxiv:
    @respx.mock
    async def test_parses_an_entry_and_normalises_whitespace(self, fetcher: HttpFetcher) -> None:
        respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=ARXIV_XML))
        docs = await ArxivClient(fetcher).search("quantisation")

        assert len(docs) == 1
        assert docs[0].title == "Quantisation for serving"
        assert docs[0].abstract == "Eight-bit quantisation works well."
        assert docs[0].external_id == "2401.99999v1"

    @respx.mock
    async def test_html_error_page_returns_empty_not_an_exception(
        self, fetcher: HttpFetcher
    ) -> None:
        # arXiv sometimes serves an HTML error body with a 200 status.
        respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text="<html>down</html>"))
        assert await ArxivClient(fetcher).search("q") == []


class TestRetries:
    @respx.mock
    async def test_transient_500_is_retried(self, fetcher: HttpFetcher) -> None:
        route = respx.get(OPENALEX_URL).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=OPENALEX_PAYLOAD),
            ]
        )
        docs = await OpenAlexClient(fetcher, fetcher.settings).search("attention")
        assert route.call_count == 2
        assert len(docs) == 1

    @respx.mock
    async def test_404_is_not_retried(self, fetcher: HttpFetcher) -> None:
        route = respx.get(OPENALEX_URL).mock(return_value=httpx.Response(404))
        with pytest.raises(httpx.HTTPStatusError):
            await OpenAlexClient(fetcher, fetcher.settings).search("attention")
        assert route.call_count == 1


class TestDedupe:
    def test_identical_ids_collapse(self) -> None:
        assert len(dedupe([DOC_A, DOC_A])) == 1

    def test_same_paper_from_two_providers_collapses(self) -> None:
        preprint = Document(
            source_id=Document.make_source_id("arxiv", "2403.1"),
            provider="arxiv",
            external_id="2403.1",
            title="Sparse attention reduces transformer inference cost!",
            abstract="preprint version",
        )
        merged = dedupe([DOC_A, preprint])
        assert len(merged) == 1
        # The more-cited record wins, so the citation carries the better metadata.
        assert merged[0].provider == "openalex"

    def test_distinct_papers_are_kept(self) -> None:
        assert len(dedupe([DOC_A, DOC_B])) == 2


class TestSearchServiceResilience:
    async def test_one_dead_provider_degrades_rather_than_fails(self, settings: Settings) -> None:
        service = SearchService(settings)

        async def ok(query: str, *, limit: int = 6) -> list[Document]:
            return [DOC_A]

        async def boom(query: str, *, limit: int = 6) -> list[Document]:
            raise httpx.ConnectError("arxiv unreachable")

        service.openalex.search = ok  # type: ignore[assignment]
        service.arxiv.search = boom  # type: ignore[assignment]

        docs = await service.search(["q"])
        assert [d.source_id for d in docs] == [DOC_A.source_id]

    async def test_all_providers_failing_raises(self, settings: Settings) -> None:
        service = SearchService(settings)

        async def boom(query: str, *, limit: int = 6) -> list[Document]:
            raise httpx.ConnectError("down")

        service.openalex.search = boom  # type: ignore[assignment]
        service.arxiv.search = boom  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="All 2 search calls failed"):
            await service.search(["q"])


class TestHttpCassettes:
    async def test_record_then_replay(self, tmp_path: object) -> None:
        from scholar_graph.config import LLMMode

        recording = Settings(llm_mode=LLMMode.record, cassette_dir=tmp_path)  # type: ignore[arg-type]
        with respx.mock:
            respx.get(OPENALEX_URL).mock(return_value=httpx.Response(200, json=OPENALEX_PAYLOAD))
            recorder = HttpFetcher(recording)
            body = await recorder.get(OPENALEX_URL, {"search": "x"}, purpose="t")
            await recorder.aclose()

        # No respx mock here: replay must not touch the network at all.
        replaying = recording.model_copy(update={"llm_mode": LLMMode.replay})
        replayed = await HttpFetcher(replaying).get(OPENALEX_URL, {"search": "x"}, purpose="t")
        assert json.loads(replayed) == json.loads(body)
