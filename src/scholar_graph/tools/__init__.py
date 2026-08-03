"""Search tools and the retrieval facade the graph calls."""

from __future__ import annotations

import asyncio

from scholar_graph.config import Settings, get_settings
from scholar_graph.domain import Document
from scholar_graph.observability import get_logger
from scholar_graph.tools.arxiv import ArxivClient
from scholar_graph.tools.http import HttpFetcher
from scholar_graph.tools.openalex import OpenAlexClient

log = get_logger(__name__)

__all__ = ["ArxivClient", "HttpFetcher", "OpenAlexClient", "SearchService"]


class SearchService:
    """Fans a query out across providers and merges the results.

    One provider failing degrades the result set; it does not fail the run.
    A research agent that dies because arXiv is having a bad afternoon is
    not one you can leave running unattended.
    """

    def __init__(
        self, settings: Settings | None = None, fetcher: HttpFetcher | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.fetcher = fetcher or HttpFetcher(self.settings)
        self.openalex = OpenAlexClient(self.fetcher, self.settings)
        self.arxiv = ArxivClient(self.fetcher)

    async def search(self, queries: list[str], *, per_query: int = 6) -> list[Document]:
        tasks = []
        for query in queries:
            tasks.append(self.openalex.search(query, limit=per_query))
            tasks.append(self.arxiv.search(query, limit=per_query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: list[Document] = []
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                log.warning("search.provider_failed", error=str(result))
                continue
            documents.extend(result)

        if failures and not documents:
            raise RuntimeError(f"All {failures} search calls failed; no documents retrieved.")

        return dedupe(documents)

    async def aclose(self) -> None:
        await self.fetcher.aclose()


def dedupe(documents: list[Document]) -> list[Document]:
    """Drop exact duplicates by source_id, then near-duplicates by title.

    The same paper routinely appears as an arXiv preprint and an OpenAlex
    journal record. Citing both as independent evidence would overstate how
    much support a claim has.
    """
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Document] = []

    for doc in sorted(documents, key=lambda d: (-d.citation_count, d.source_id)):
        title_key = "".join(ch for ch in doc.title.lower() if ch.isalnum())
        if doc.source_id in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(doc.source_id)
        seen_titles.add(title_key)
        unique.append(doc)

    return unique
