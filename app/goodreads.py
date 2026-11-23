"""Optional Goodreads Want-to-Read sync via the logged-in web surface.

Mirrors app.hardcover: runs after extraction, fail-soft, books only. Goodreads
closed its public API, so this uses the public autocomplete search plus a
browser session cookie for shelf writes. The Obsidian note is a separate
optional step.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import quote

import httpx

from app.config import Settings
from app.hardcover import (
    TitleTranslator,
    authors_overlap,
    entity_title_candidates,
    hardcover_search_queries,
    title_needs_catalog_translation,
    titles_match,
)
from app.models import Confidence, Entity, EntityType

logger = logging.getLogger(__name__)

GOODREADS_BASE_URL = "https://www.goodreads.com"
GOODREADS_AUTOCOMPLETE_URL = f"{GOODREADS_BASE_URL}/book/auto_complete"
GOODREADS_ADD_TO_SHELF_URL = f"{GOODREADS_BASE_URL}/shelf/add_to_shelf"
GOODREADS_REVIEW_LIST_URL = f"{GOODREADS_BASE_URL}/review/list"
GOODREADS_USER_AGENT = (
    "Mozilla/5.0 (compatible; media-ai-bookmarker; personal goodreads sync)"
)
TO_READ_SHELF = "to-read"

STATUS_LABELS: dict[str, str] = {
    "to-read": "Want to Read",
    "currently-reading": "Currently Reading",
    "read": "Read",
}
_EXCLUSIVE_SHELVES = frozenset(STATUS_LABELS)

_LOOKUP_FAILED = object()

_CSRF_RE = re.compile(
    r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']'
    r'|name=["\']authenticity_token["\'][^>]*value=["\']([^"\']+)["\']'
    r'|value=["\']([^"\']+)["\'][^>]*name=["\']authenticity_token["\']',
    re.IGNORECASE,
)
_SELECTED_SHELF_RE = re.compile(
    r'<option[^>]+selected[^>]*value=["\']('
    + "|".join(re.escape(name) for name in STATUS_LABELS)
    + r')["\']'
    r'|value=["\']('
    + "|".join(re.escape(name) for name in STATUS_LABELS)
    + r')["\'][^>]*selected',
    re.IGNORECASE,
)
_CHECKED_SHELF_RE = re.compile(
    r'<input[^>]+(?:'
    r'type=["\'](?:radio|checkbox|hidden)["\'][^>]*'
    r'name=["\'](?:shelf|name|book_shelf_id)["\'][^>]*'
    r'value=["\']('
    + "|".join(re.escape(name) for name in STATUS_LABELS)
    + r')["\']'
    r'|name=["\'](?:shelf|name|book_shelf_id)["\'][^>]*'
    r'value=["\']('
    + "|".join(re.escape(name) for name in STATUS_LABELS)
    + r')["\'][^>]*'
    r'(?:checked|selected)'
    r')',
    re.IGNORECASE,
)
_SHELF_HINT_RE = re.compile(
    r'data-(?:shelf|exclusive-shelf)=["\']('
    + "|".join(re.escape(name) for name in STATUS_LABELS)
    + r')["\']',
    re.IGNORECASE,
)


class GoodreadsOutcome(str, Enum):
    added = "added"
    already_on_shelf = "already_on_shelf"
    no_match = "no_match"
    error = "error"


@dataclass(frozen=True)
class GoodreadsHit:
    book_id: str
    title: str
    author_name: str
    book_url: str
    book_title_bare: str = ""

    @property
    def url(self) -> str:
        path = (self.book_url or "").strip()
        if path.startswith("http"):
            return path
        if path.startswith("/"):
            return f"{GOODREADS_BASE_URL}{path}"
        if self.book_id:
            return f"{GOODREADS_BASE_URL}/book/show/{self.book_id}"
        return ""

    @property
    def titles(self) -> list[str]:
        return [
            title
            for title in (self.title, self.book_title_bare)
            if title and title.strip()
        ]


@dataclass(frozen=True)
class GoodreadsAction:
    entity_name: str
    outcome: GoodreadsOutcome
    status_label: str = ""
    goodreads_url: str = ""

    def summary_line(self) -> str:
        name = self.entity_name
        if self.outcome == GoodreadsOutcome.added:
            return f"Goodreads: added {name} as Want to Read"
        if self.outcome == GoodreadsOutcome.already_on_shelf:
            status = f" ({self.status_label})" if self.status_label else ""
            return f"Goodreads: {name} already on your shelf{status} — left as-is"
        if self.outcome == GoodreadsOutcome.no_match:
            return f'Goodreads: skipped "{name}" (no confident match)'
        return f"Goodreads: error looking up {name}"


def cookie_header(raw: str) -> str:
    """Accept a full Cookie header or a bare `_session_id2` value."""
    cleaned = (raw or "").strip()
    if not cleaned:
        return ""
    if cleaned.lower().startswith("cookie:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if "=" not in cleaned:
        return f"_session_id2={cleaned}"
    return cleaned


def parse_csrf_token(html: str) -> str:
    match = _CSRF_RE.search(html or "")
    if not match:
        return ""
    return next((group for group in match.groups() if group), "")


def parse_existing_shelf(html: str) -> Optional[str]:
    """Exclusive shelf name from an authenticated review/edit page, if any."""
    for pattern in (_SELECTED_SHELF_RE, _CHECKED_SHELF_RE, _SHELF_HINT_RE):
        match = pattern.search(html or "")
        if not match:
            continue
        shelf = next((group for group in match.groups() if group), "")
        shelf = shelf.strip().lower()
        if shelf in _EXCLUSIVE_SHELVES:
            return shelf
    return None


def parse_autocomplete_hits(payload: object) -> list[GoodreadsHit]:
    if not isinstance(payload, list):
        return []
    hits: list[GoodreadsHit] = []
    for item in payload:
        hit = _hit_from_item(item)
        if hit is not None:
            hits.append(hit)
    return hits


def _hit_from_item(item: object) -> Optional[GoodreadsHit]:
    if not isinstance(item, dict):
        return None
    book_id = str(item.get("bookId") or item.get("book_id") or "").strip()
    if not book_id:
        return None
    title = str(item.get("title") or "").strip()
    bare = str(item.get("bookTitleBare") or item.get("book_title_bare") or "").strip()
    if not title and not bare:
        return None
    author = item.get("author")
    author_name = ""
    if isinstance(author, dict):
        author_name = str(author.get("name") or "").strip()
    elif isinstance(author, str):
        author_name = author.strip()
    book_url = str(item.get("bookUrl") or item.get("book_url") or "").strip()
    return GoodreadsHit(
        book_id=book_id,
        title=title or bare,
        author_name=author_name,
        book_url=book_url,
        book_title_bare=bare,
    )


def pick_match(
    entity: Entity,
    hits: list[GoodreadsHit],
    extra_titles: list[str] | None = None,
) -> Optional[GoodreadsHit]:
    """First hit whose title is close and whose author overlaps when known."""
    extracted = entity_title_candidates(entity)
    if extra_titles:
        extracted = [*extracted, *extra_titles]
    for hit in hits:
        if not any(
            titles_match(candidate, hit_title, entity.creator_or_author)
            for candidate in extracted
            for hit_title in hit.titles
        ):
            continue
        author_names = [hit.author_name] if hit.author_name else []
        if not authors_overlap(entity.creator_or_author, author_names):
            continue
        return hit
    return None


def select_goodreads_candidates(entities: list[Entity], limit: int) -> list[Entity]:
    """Books worth a Goodreads write: not low-confidence, main topics first."""
    if limit <= 0:
        return []
    eligible = [
        entity
        for entity in entities
        if entity.type == EntityType.book and entity.confidence != Confidence.low
    ]
    main = [entity for entity in eligible if entity.is_main_topic]
    rest = [entity for entity in eligible if not entity.is_main_topic]
    return (main + rest)[:limit]


def apply_goodreads_actions(
    entities: list[Entity],
    actions: list[GoodreadsAction],
) -> None:
    """Stamp goodreads_url onto matched entities, first action per title wins."""
    by_name: dict[str, GoodreadsAction] = {}
    for action in actions:
        if action.goodreads_url and action.entity_name not in by_name:
            by_name[action.entity_name] = action
    for entity in entities:
        action = by_name.get(entity.name)
        if action:
            entity.goodreads_url = action.goodreads_url


def format_goodreads_report(actions: list[GoodreadsAction]) -> str:
    return "\n".join(action.summary_line() for action in actions)


def autocomplete_url(query: str) -> str:
    return f"{GOODREADS_AUTOCOMPLETE_URL}?format=json&q={quote(query)}"


def review_edit_url(book_id: str) -> str:
    return f"{GOODREADS_BASE_URL}/review/edit/{book_id}"


class GoodreadsClient:
    def __init__(
        self,
        settings: Settings,
        title_translator: TitleTranslator | None = None,
    ) -> None:
        self._settings = settings
        self._title_translator = title_translator
        self._csrf_token = ""
        self._client = httpx.AsyncClient(
            timeout=settings.goodreads_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": GOODREADS_USER_AGENT},
        )

    @property
    def enabled(self) -> bool:
        return (
            bool(self._settings.goodreads_cookie)
            and self._settings.goodreads_books_per_job > 0
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def sync_books(self, entities: list[Entity]) -> list[GoodreadsAction]:
        """Search and Want-to-Read each book. Never raises."""
        if not self.enabled:
            return []
        candidates = select_goodreads_candidates(
            entities, self._settings.goodreads_books_per_job
        )
        actions: list[GoodreadsAction] = []
        for entity in candidates:
            try:
                actions.append(await self._sync_one(entity))
            except Exception:
                logger.warning(
                    "goodreads sync failed name=%r",
                    entity.name,
                    exc_info=True,
                )
                actions.append(
                    GoodreadsAction(
                        entity_name=entity.name,
                        outcome=GoodreadsOutcome.error,
                    )
                )
        return actions

    async def _sync_one(self, entity: Entity) -> GoodreadsAction:
        queries = hardcover_search_queries(entity)
        if not queries:
            return GoodreadsAction(
                entity_name=entity.name,
                outcome=GoodreadsOutcome.no_match,
            )
        match, errored = await self._search_queries(entity, queries)
        if errored:
            return GoodreadsAction(
                entity_name=entity.name,
                outcome=GoodreadsOutcome.error,
            )
        if match is None:
            fallback, extra_titles = await self._catalog_title_queries(entity)
            if fallback:
                match, errored = await self._search_queries(
                    entity, fallback, extra_titles
                )
                if errored:
                    return GoodreadsAction(
                        entity_name=entity.name,
                        outcome=GoodreadsOutcome.error,
                    )
        if match is None:
            logger.info(
                "goodreads no_match name=%r author=%r",
                entity.name,
                entity.creator_or_author,
            )
            return GoodreadsAction(
                entity_name=entity.name,
                outcome=GoodreadsOutcome.no_match,
            )
        existing = await self._existing_shelf(match.book_id)
        if existing is _LOOKUP_FAILED:
            return GoodreadsAction(
                entity_name=entity.name,
                outcome=GoodreadsOutcome.error,
                goodreads_url=match.url,
            )
        if isinstance(existing, str):
            return GoodreadsAction(
                entity_name=entity.name,
                outcome=GoodreadsOutcome.already_on_shelf,
                status_label=STATUS_LABELS.get(existing, existing),
                goodreads_url=match.url,
            )
        if not await self._insert(match.book_id):
            return GoodreadsAction(
                entity_name=entity.name,
                outcome=GoodreadsOutcome.error,
                goodreads_url=match.url,
            )
        logger.info(
            "goodreads added book_id=%s name=%r url=%s",
            match.book_id,
            entity.name,
            match.url or "(none)",
        )
        return GoodreadsAction(
            entity_name=entity.name,
            outcome=GoodreadsOutcome.added,
            status_label=STATUS_LABELS[TO_READ_SHELF],
            goodreads_url=match.url,
        )

    async def _search_queries(
        self,
        entity: Entity,
        queries: list[str],
        extra_titles: list[str] | None = None,
    ) -> tuple[Optional[GoodreadsHit], bool]:
        for query in queries:
            payload = await self._autocomplete(query)
            if payload is None:
                return None, True
            hits = parse_autocomplete_hits(payload)
            match = pick_match(entity, hits, extra_titles=extra_titles)
            if match is not None:
                return match, False
            logger.info(
                "goodreads miss query=%r name=%r candidates=%s hit_titles=%s",
                query,
                entity.name,
                [*entity_title_candidates(entity), *(extra_titles or [])],
                [hit.title for hit in hits],
            )
        return None, False

    async def _catalog_title_queries(
        self, entity: Entity
    ) -> tuple[list[str], list[str]]:
        if self._title_translator is None or not title_needs_catalog_translation(
            entity.name
        ):
            return [], []
        try:
            translated = (await self._title_translator(entity)).strip()
        except Exception:
            logger.warning(
                "goodreads catalog title failed name=%r",
                entity.name,
                exc_info=True,
            )
            return [], []
        if not translated:
            logger.info("goodreads catalog title empty name=%r", entity.name)
            return [], []
        known = {
            title.casefold() for title in entity_title_candidates(entity)
        }
        if translated.casefold() in known:
            logger.info(
                "goodreads catalog title already searched name=%r title=%r",
                entity.name,
                translated,
            )
            return [], []
        author = entity.creator_or_author
        query = " ".join(part for part in (translated, author) if part).strip()
        if not query:
            return [], []
        logger.info(
            "goodreads catalog title name=%r title=%r",
            entity.name,
            translated,
        )
        return [query], [translated]

    async def _existing_shelf(self, book_id: str) -> object:
        response = await self._authed_get(review_edit_url(book_id))
        if response is None:
            return _LOOKUP_FAILED
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            logger.warning(
                "goodreads review/edit HTTP %s book_id=%s body=%s",
                response.status_code,
                book_id,
                response.text[:300],
            )
            return _LOOKUP_FAILED
        self._remember_csrf(response.text)
        return parse_existing_shelf(response.text)

    async def _insert(self, book_id: str) -> bool:
        token = await self._ensure_csrf()
        if not token:
            logger.warning("goodreads CSRF missing book_id=%s", book_id)
            return False
        try:
            response = await self._client.post(
                GOODREADS_ADD_TO_SHELF_URL,
                headers={
                    "Cookie": cookie_header(self._settings.goodreads_cookie),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Referer": review_edit_url(book_id),
                    "X-Requested-With": "XMLHttpRequest",
                },
                data={
                    "authenticity_token": token,
                    "book_id": book_id,
                    "name": TO_READ_SHELF,
                },
            )
        except Exception:
            logger.warning(
                "goodreads add_to_shelf failed book_id=%s",
                book_id,
                exc_info=True,
            )
            return False
        if response.status_code >= 400:
            logger.warning(
                "goodreads add_to_shelf HTTP %s book_id=%s body=%s",
                response.status_code,
                book_id,
                response.text[:300],
            )
            return False
        return True

    async def _ensure_csrf(self) -> str:
        if self._csrf_token:
            return self._csrf_token
        response = await self._authed_get(GOODREADS_REVIEW_LIST_URL)
        if response is None:
            return ""
        if response.status_code >= 400:
            logger.warning(
                "goodreads review/list HTTP %s body=%s",
                response.status_code,
                response.text[:300],
            )
            return ""
        self._remember_csrf(response.text)
        return self._csrf_token

    def _remember_csrf(self, html: str) -> None:
        token = parse_csrf_token(html)
        if token:
            self._csrf_token = token

    async def _autocomplete(self, query: str) -> Optional[Any]:
        try:
            response = await self._client.get(
                autocomplete_url(query),
                headers={"Accept": "application/json"},
            )
        except Exception:
            logger.warning("goodreads autocomplete failed query=%r", query, exc_info=True)
            return None
        if response.status_code >= 400:
            logger.warning(
                "goodreads autocomplete HTTP %s query=%r body=%s",
                response.status_code,
                query,
                response.text[:300],
            )
            return None
        try:
            return response.json()
        except Exception:
            logger.warning("goodreads autocomplete non-JSON query=%r", query)
            return None

    async def _authed_get(self, url: str) -> Optional[httpx.Response]:
        try:
            return await self._client.get(
                url,
                headers={
                    "Cookie": cookie_header(self._settings.goodreads_cookie),
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
        except Exception:
            logger.warning("goodreads GET failed url=%s", url, exc_info=True)
            return None
