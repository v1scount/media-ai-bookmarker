from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.config import Settings
from app.models import Confidence, Entity

logger = logging.getLogger(__name__)

KAGI_SEARCH_URL = "https://kagi.com/api/v1/search"
KAGI_RESULT_LIMIT = 5


def authorization_header(api_key: str) -> str:
    """Kagi expects `Bot <token>`. Leave an existing Bot/Bearer prefix alone."""
    key = api_key.strip()
    lowered = key.lower()
    if lowered.startswith("bot ") or lowered.startswith("bearer "):
        return key
    return f"Bot {key}"


def _item_url(item: object) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned.lower().startswith("http"):
        return None
    return cleaned


def top_result_url(payload: object) -> Optional[str]:
    """First search result URL from a v1 or legacy Kagi Search API body."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, dict):
        search = data.get("search")
        if isinstance(search, list):
            for item in search:
                url = _item_url(item)
                if url:
                    return url
        return None
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("t") == 0:
                url = _item_url(item)
                if url:
                    return url
    return None


def select_kagi_candidates(entities: list[Entity], limit: int) -> list[Entity]:
    """Items worth an API lookup: no link yet, not low-confidence, main topics first."""
    if limit <= 0:
        return []
    eligible = [
        entity
        for entity in entities
        if not entity.suggested_link and entity.confidence != Confidence.low
    ]
    main = [entity for entity in eligible if entity.is_main_topic]
    rest = [entity for entity in eligible if not entity.is_main_topic]
    return (main + rest)[:limit]


class KagiClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, Optional[str]] = {}
        self._client = httpx.AsyncClient(timeout=settings.kagi_timeout_seconds)

    @property
    def enabled(self) -> bool:
        return bool(self._settings.kagi_api_key) and self._settings.kagi_search_per_job > 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def top_url(self, query: str) -> Optional[str]:
        """Best search result for query, or None. Never raises."""
        query = (query or "").strip()
        if not query or not self.enabled:
            return None
        if query in self._cache:
            return self._cache[query]
        try:
            url = await self._search(query)
        except Exception:
            logger.warning("kagi search failed query=%r", query, exc_info=True)
            return None
        self._cache[query] = url
        logger.info("kagi search query=%r url=%s", query, url or "(none)")
        return url

    async def _search(self, query: str) -> Optional[str]:
        response = await self._client.post(
            KAGI_SEARCH_URL,
            headers={
                "Authorization": authorization_header(self._settings.kagi_api_key),
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "limit": KAGI_RESULT_LIMIT,
                "workflow": "search",
            },
        )
        if response.status_code >= 400:
            logger.warning(
                "kagi search HTTP %s query=%r body=%s",
                response.status_code,
                query,
                response.text[:300],
            )
            return None
        payload: Any = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            logger.warning("kagi search error query=%r error=%s", query, payload["error"])
            return None
        return top_result_url(payload)
