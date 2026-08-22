from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import Settings
from app.models import Confidence, Entity, EntityType, ExtractionResult
from app.obsidian import save_to_obsidian

logger = logging.getLogger(__name__)

HARDCOVER_GRAPHQL_URL = "https://api.hardcover.app/v1/graphql"
HARDCOVER_USER_AGENT = "media-ai-bookmarker (personal hardcover sync)"
HARDCOVER_BOOK_URL = "https://hardcover.app/books/{slug}"
WANT_TO_READ_STATUS_ID = 1

STATUS_LABELS: dict[int, str] = {
    1: "Want to Read",
    2: "Currently Reading",
    3: "Read",
    4: "Paused",
    5: "Did Not Finish",
    6: "Ignored",
}

_TITLE_STOP_WORDS = frozenset({"a", "an", "the", "book", "novel"})
_LOOKUP_FAILED = object()

SEARCH_QUERY = """
query SearchBooks($query: String!) {
  search(query: $query, query_type: "Book", per_page: 5, page: 1) {
    results
  }
}
"""

EXISTING_QUERY = """
query ExistingUserBook($bookId: Int!) {
  me {
    id
    user_books(where: { book_id: { _eq: $bookId } }, limit: 1) {
      id
      status_id
    }
  }
}
"""

INSERT_MUTATION = """
mutation AddWantToRead($bookId: Int!) {
  insert_user_book(object: { book_id: $bookId, status_id: 1 }) {
    id
    error
    user_book {
      id
      status_id
      book_id
    }
  }
}
"""


class HardcoverOutcome(str, Enum):
    added = "added"
    already_on_shelf = "already_on_shelf"
    no_match = "no_match"
    error = "error"


@dataclass(frozen=True)
class HardcoverHit:
    book_id: int
    title: str
    author_names: list[str]
    slug: str

    @property
    def url(self) -> str:
        if not self.slug:
            return ""
        return HARDCOVER_BOOK_URL.format(slug=self.slug)


@dataclass(frozen=True)
class HardcoverAction:
    entity_name: str
    outcome: HardcoverOutcome
    status_label: str = ""
    hardcover_url: str = ""

    def summary_line(self) -> str:
        name = self.entity_name
        if self.outcome == HardcoverOutcome.added:
            return f"Hardcover: added {name} as Want to Read"
        if self.outcome == HardcoverOutcome.already_on_shelf:
            status = f" ({self.status_label})" if self.status_label else ""
            return f"Hardcover: {name} already on your shelf{status} — left as-is"
        if self.outcome == HardcoverOutcome.no_match:
            return f'Hardcover: skipped "{name}" (no confident match)'
        return f"Hardcover: error looking up {name}"


def authorization_header(api_key: str) -> str:
    """Hardcover expects `Bearer <token>`. Leave an existing Bearer prefix alone."""
    key = api_key.strip()
    if key.lower().startswith("bearer "):
        return key
    return f"Bearer {key}"


def hardcover_search_query(entity: Entity) -> str:
    """Title plus author only — no generic 'book' search hint."""
    return " ".join(
        part for part in (entity.name, entity.creator_or_author) if part
    ).strip()


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _strip_parentheticals(text: str) -> str:
    return re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", text)


def _core_title_tokens(title: str) -> list[str]:
    text = _strip_parentheticals(title or "")
    text = re.split(r"[:\u2014\u2013]", text, maxsplit=1)[0]
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [
        token
        for token in text.split()
        if token and token not in _TITLE_STOP_WORDS
    ]


def titles_match(extracted: str, hit_title: str) -> bool:
    left = _core_title_tokens(extracted)
    right = _core_title_tokens(hit_title)
    return bool(left) and left == right


def _name_tokens(name: str) -> set[str]:
    text = (name or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return {token for token in text.split() if len(token) > 1}


def authors_overlap(extracted: str, author_names: list[str]) -> bool:
    wanted = _name_tokens(extracted)
    if not wanted:
        return True
    for author in author_names:
        if wanted & _name_tokens(author):
            return True
    return False


def parse_search_hits(results: object) -> list[HardcoverHit]:
    """Typesense `results`: a dict or a JSON string with `hits[].document`."""
    payload: object = results
    if isinstance(results, str):
        try:
            payload = json.loads(results)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    raw_hits = payload.get("hits")
    if not isinstance(raw_hits, list):
        return []
    hits: list[HardcoverHit] = []
    for item in raw_hits:
        hit = _hit_from_item(item)
        if hit is not None:
            hits.append(hit)
    return hits


def _hit_from_item(item: object) -> Optional[HardcoverHit]:
    if not isinstance(item, dict):
        return None
    document = item.get("document")
    if not isinstance(document, dict):
        document = item
    book_id = _as_int(document.get("id"))
    if book_id is None:
        book_id = _as_int(item.get("id"))
    if book_id is None:
        return None
    title = str(document.get("title") or "").strip()
    if not title:
        return None
    raw_authors = document.get("author_names") or []
    if isinstance(raw_authors, str):
        raw_authors = [raw_authors]
    author_names = [
        str(name).strip()
        for name in raw_authors
        if isinstance(name, str) and name.strip()
    ]
    slug = str(document.get("slug") or "").strip()
    return HardcoverHit(
        book_id=book_id,
        title=title,
        author_names=author_names,
        slug=slug,
    )


def pick_match(entity: Entity, hits: list[HardcoverHit]) -> Optional[HardcoverHit]:
    """First hit whose title is close and whose authors overlap when known."""
    for hit in hits:
        if not titles_match(entity.name, hit.title):
            continue
        if not authors_overlap(entity.creator_or_author, hit.author_names):
            continue
        return hit
    return None


def select_hardcover_candidates(entities: list[Entity], limit: int) -> list[Entity]:
    """Books worth a Hardcover write: not low-confidence, main topics first."""
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


def apply_hardcover_actions(
    entities: list[Entity],
    actions: list[HardcoverAction],
) -> None:
    """Stamp hardcover_url onto matched entities, first action per title wins."""
    by_name: dict[str, HardcoverAction] = {}
    for action in actions:
        if action.hardcover_url and action.entity_name not in by_name:
            by_name[action.entity_name] = action
    for entity in entities:
        action = by_name.get(entity.name)
        if action:
            entity.hardcover_url = action.hardcover_url


def format_hardcover_report(actions: list[HardcoverAction]) -> str:
    return "\n".join(action.summary_line() for action in actions)


def _search_results(payload: dict[str, Any]) -> object:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    search = data.get("search")
    if not isinstance(search, dict):
        return None
    return search.get("results")


def _me_record(me: object) -> Optional[dict[str, Any]]:
    if isinstance(me, list) and me:
        first = me[0]
        return first if isinstance(first, dict) else None
    if isinstance(me, dict):
        return me
    return None


def parse_existing_status(payload: object) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    me = _me_record(data.get("me"))
    if me is None:
        return None
    books = me.get("user_books")
    if not isinstance(books, list) or not books:
        return None
    first = books[0]
    if not isinstance(first, dict):
        return None
    return _as_int(first.get("status_id"))


def insert_succeeded(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    inserted = data.get("insert_user_book")
    if not isinstance(inserted, dict):
        return False
    if inserted.get("error"):
        return False
    return inserted.get("id") is not None or inserted.get("user_book") is not None


class HardcoverClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.hardcover_timeout_seconds)

    @property
    def enabled(self) -> bool:
        return (
            bool(self._settings.hardcover_api_key)
            and self._settings.hardcover_books_per_job > 0
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def sync_books(self, entities: list[Entity]) -> list[HardcoverAction]:
        """Search and Want-to-Read each book. Never raises."""
        if not self.enabled:
            return []
        candidates = select_hardcover_candidates(
            entities, self._settings.hardcover_books_per_job
        )
        actions: list[HardcoverAction] = []
        for entity in candidates:
            try:
                actions.append(await self._sync_one(entity))
            except Exception:
                logger.warning(
                    "hardcover sync failed name=%r",
                    entity.name,
                    exc_info=True,
                )
                actions.append(
                    HardcoverAction(
                        entity_name=entity.name,
                        outcome=HardcoverOutcome.error,
                    )
                )
        return actions

    async def _sync_one(self, entity: Entity) -> HardcoverAction:
        query = hardcover_search_query(entity)
        if not query:
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.no_match,
            )
        search_payload = await self._graphql(SEARCH_QUERY, {"query": query})
        if search_payload is None:
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.error,
            )
        match = pick_match(entity, parse_search_hits(_search_results(search_payload)))
        if match is None:
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.no_match,
            )
        existing = await self._existing_status(match.book_id)
        if existing is _LOOKUP_FAILED:
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.error,
                hardcover_url=match.url,
            )
        if isinstance(existing, int):
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.already_on_shelf,
                status_label=STATUS_LABELS.get(existing, str(existing)),
                hardcover_url=match.url,
            )
        if not await self._insert(match.book_id):
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.error,
                hardcover_url=match.url,
            )
        logger.info(
            "hardcover added book_id=%s name=%r url=%s",
            match.book_id,
            entity.name,
            match.url or "(none)",
        )
        return HardcoverAction(
            entity_name=entity.name,
            outcome=HardcoverOutcome.added,
            status_label=STATUS_LABELS[WANT_TO_READ_STATUS_ID],
            hardcover_url=match.url,
        )

    async def _existing_status(self, book_id: int) -> object:
        payload = await self._graphql(EXISTING_QUERY, {"bookId": book_id})
        if payload is None:
            return _LOOKUP_FAILED
        return parse_existing_status(payload)

    async def _insert(self, book_id: int) -> bool:
        payload = await self._graphql(INSERT_MUTATION, {"bookId": book_id})
        return insert_succeeded(payload)

    async def _graphql(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        response = await self._client.post(
            HARDCOVER_GRAPHQL_URL,
            headers={
                "Authorization": authorization_header(self._settings.hardcover_api_key),
                "Content-Type": "application/json",
                "User-Agent": HARDCOVER_USER_AGENT,
            },
            json={"query": query, "variables": variables},
        )
        if response.status_code >= 400:
            logger.warning(
                "hardcover HTTP %s body=%s",
                response.status_code,
                response.text[:300],
            )
            return None
        payload: Any = response.json()
        if not isinstance(payload, dict):
            return None
        if payload.get("errors"):
            logger.warning("hardcover graphql errors=%s", payload["errors"])
            return None
        if payload.get("error"):
            logger.warning("hardcover error=%s", payload["error"])
            return None
        return payload


async def sync_hardcover_then_save(
    settings: Settings,
    hardcover: HardcoverClient | None,
    result: ExtractionResult,
) -> tuple[Path, list[HardcoverAction]]:
    """Want-to-Read matched books, then write the Obsidian note. Fail-soft."""
    actions: list[HardcoverAction] = []
    if hardcover is not None and hardcover.enabled:
        actions = await hardcover.sync_books(result.entities)
        apply_hardcover_actions(result.entities, actions)
    path = await asyncio.to_thread(save_to_obsidian, settings, result)
    return path, actions
