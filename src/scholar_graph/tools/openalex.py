"""OpenAlex search.

OpenAlex is free, needs no key, and covers ~250M works — which is why this
repo can ship a demo that actually retrieves real papers rather than mocking
a retriever and calling it a research agent.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from scholar_graph.config import Settings, get_settings
from scholar_graph.domain import Document
from scholar_graph.observability import get_logger
from scholar_graph.tools.http import HttpFetcher

log = get_logger(__name__)

API_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as ``{word: [positions]}`` for licensing reasons.

    Rebuilding the linear text is the caller's job.
    """
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = [
        (pos, word) for word, spots in inverted.items() for pos in spots
    ]
    positions.sort()
    return " ".join(word for _, word in positions)


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _to_document(work: dict[str, Any]) -> Document | None:
    external_id = (work.get("id") or "").rsplit("/", 1)[-1]
    title = work.get("display_name") or work.get("title") or ""
    if not external_id or not title:
        return None

    authors = tuple(
        a["author"]["display_name"]
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    )
    location = work.get("primary_location") or {}
    source = location.get("source") or {}

    return Document(
        source_id=Document.make_source_id("openalex", external_id),
        provider="openalex",
        external_id=external_id,
        title=title.strip(),
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=authors[:12],
        published=_parse_date(work.get("publication_date")),
        venue=source.get("display_name"),
        url=work.get("doi") or location.get("landing_page_url") or work.get("id"),
        citation_count=int(work.get("cited_by_count") or 0),
    )


class OpenAlexClient:
    def __init__(self, fetcher: HttpFetcher, settings: Settings | None = None) -> None:
        self.fetcher = fetcher
        self.settings = settings or get_settings()

    async def search(self, query: str, *, limit: int = 8) -> list[Document]:
        params = {
            "search": query,
            "per-page": limit,
            "mailto": self.settings.openalex_mailto,
            # Abstract-less records cannot be cited against, so never retrieve them.
            "filter": "has_abstract:true",
            "sort": "relevance_score:desc",
            "select": ",".join(
                [
                    "id",
                    "display_name",
                    "abstract_inverted_index",
                    "authorships",
                    "publication_date",
                    "primary_location",
                    "cited_by_count",
                    "doi",
                ]
            ),
        }
        body = await self.fetcher.get(API_URL, params, purpose=f"openalex:{query}")
        payload = json.loads(body)

        documents: list[Document] = []
        for work in payload.get("results", []):
            doc = _to_document(work)
            if doc is not None:
                documents.append(doc)

        log.info("openalex.search", query=query, returned=len(documents))
        return documents
