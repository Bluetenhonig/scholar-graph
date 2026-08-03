"""arXiv search.

A second provider with different coverage and a different failure mode.
Two providers is not redundancy for its own sake: OpenAlex lags on preprints
by weeks, which is exactly the window that matters for ML questions.
"""

from __future__ import annotations

from datetime import date, datetime
from xml.etree import ElementTree

from scholar_graph.domain import Document
from scholar_graph.observability import get_logger
from scholar_graph.tools.http import HttpFetcher

log = get_logger(__name__)

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


def _parse_published(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _text(node: ElementTree.Element, tag: str) -> str:
    child = node.find(f"{ATOM}{tag}")
    return (child.text or "").strip() if child is not None else ""


class ArxivClient:
    def __init__(self, fetcher: HttpFetcher) -> None:
        self.fetcher = fetcher

    async def search(self, query: str, *, limit: int = 8) -> list[Document]:
        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        body = await self.fetcher.get(API_URL, params, purpose=f"arxiv:{query}")

        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            # arXiv occasionally returns an HTML error page with a 200.
            log.warning("arxiv.parse_failed", query=query, error=str(exc))
            return []

        documents: list[Document] = []
        for entry in root.findall(f"{ATOM}entry"):
            raw_id = _text(entry, "id")
            external_id = raw_id.rsplit("/", 1)[-1]
            title = " ".join(_text(entry, "title").split())
            if not external_id or not title:
                continue
            authors = tuple(
                name for name in (_text(a, "name") for a in entry.findall(f"{ATOM}author")) if name
            )
            documents.append(
                Document(
                    source_id=Document.make_source_id("arxiv", external_id),
                    provider="arxiv",
                    external_id=external_id,
                    title=title,
                    abstract=" ".join(_text(entry, "summary").split()),
                    authors=authors[:12],
                    published=_parse_published(_text(entry, "published")),
                    venue="arXiv",
                    url=raw_id or None,
                )
            )

        log.info("arxiv.search", query=query, returned=len(documents))
        return documents
