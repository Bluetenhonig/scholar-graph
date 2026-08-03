"""HTTP client shared by the search tools.

Same record/replay contract as the LLM layer, for the same reason: a run
that replays its model calls but still hits live search APIs is not
reproducible, and OpenAlex results change week to week.

Retries are bounded and only cover the errors worth retrying — a 404 is an
answer, not a hiccup.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from scholar_graph.config import LLMMode, Settings, get_settings
from scholar_graph.llm.cassette import CassetteMiss, CassetteStore, cache_key
from scholar_graph.observability import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return False


class HttpFetcher:
    """Cassette-backed GET. Returns raw response text; callers parse."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cassettes = CassetteStore(self.settings.cassette_dir, "http")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.http_timeout_seconds,
                headers={"User-Agent": f"scholar-graph (+{self.settings.openalex_mailto})"},
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, url: str, params: dict[str, Any], *, purpose: str) -> str:
        request = {"method": "GET", "url": url, "params": {k: str(v) for k, v in params.items()}}
        key = cache_key(request)

        if self.settings.llm_mode is LLMMode.replay:
            stored = self.cassettes.load(key)
            if stored is None:
                raise CassetteMiss(key, purpose, self.cassettes.path_for(key))
            log.debug("http.replay", purpose=purpose, key=key[:12])
            body: str = stored["response"]["body"]
            return body

        body = await self._fetch(url, params)

        if self.settings.llm_mode is LLMMode.record:
            self.cassettes.save(key, request, {"body": body, "purpose": purpose})
        log.info("http.get", purpose=purpose, url=url, bytes=len(body))
        return body

    async def _fetch(self, url: str, params: dict[str, Any]) -> str:
        client = await self._get_client()
        attempts = self.settings.http_max_attempts

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        async def _do() -> str:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.text

        return await _do()
