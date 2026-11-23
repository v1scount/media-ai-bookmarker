from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import httpx

from app.config import Settings
from app.letterboxd import (
    LetterboxdAction,
    LetterboxdClient,
    apply_letterboxd_actions,
    select_letterboxd_candidates,
)
from app.models import Confidence, Entity, EntityType, ExtractionResult

if TYPE_CHECKING:
    from app.goodreads import GoodreadsAction, GoodreadsClient

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

_TITLE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "book",
        "novel",
        "el",
        "la",
        "los",
        "las",
        "lo",
        "un",
        "una",
        "unos",
        "unas",
        "del",
        "al",
    }
)
# Function words that show a title is not already an English catalog title.
# Checked on a miss only, before spending a translation call.
_FOREIGN_TITLE_TOKENS = frozenset(
    {
        "el",
        "los",
        "las",
        "del",
        "al",
        "un",
        "una",
        "unos",
        "unas",
        "lo",
        "como",
        "para",
        "por",
        "con",
        "de",
        "en",
        "y",
        "les",
        "des",
        "une",
        "du",
        "dans",
        "und",
        "der",
        "die",
        "das",
        "den",
        "ein",
        "eine",
        "gli",
        "dei",
        "delle",
        "nel",
        "nella",
        "van",
        "het",
        "een",
    }
)
_LOOKUP_FAILED = object()
MAX_TITLE_CANDIDATES = 3
_SUBTITLE_SPLIT_RE = re.compile(r"[:\u2014\u2013]|[ \t]+-[ \t]+|,\s+")

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
    alternative_titles: list[str] = field(default_factory=list)

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


def _title_author_query(title: str, author: str) -> str:
    return " ".join(part for part in (title, author) if part).strip()


def _has_subtitle(title: str) -> bool:
    return bool(_SUBTITLE_SPLIT_RE.search(title or ""))


def _comma_to_colon_title(title: str) -> str:
    """Hardcover titles use a colon; extracts often use a comma instead."""
    return re.sub(r",\s+", ": ", (title or "").strip(), count=1)


def _raw_title_tokens(title: str) -> set[str]:
    text = _fold_accents(_strip_parentheticals(title or "").lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return {token for token in text.split() if token}


def title_needs_catalog_translation(title: str) -> bool:
    """True when a missed title still looks like a translation or local edition."""
    if any(ch.isalpha() and ord(ch) > 127 for ch in title or ""):
        return True
    return bool(_raw_title_tokens(title) & _FOREIGN_TITLE_TOKENS)


def hardcover_search_query(entity: Entity) -> str:
    """First title plus author only — no generic 'book' search hint."""
    queries = hardcover_search_queries(entity)
    return queries[0] if queries else ""


def hardcover_search_queries(entity: Entity) -> list[str]:
    """One title+author query per distinct extracted title, name first."""
    author = entity.creator_or_author
    queries: list[str] = []
    seen: set[str] = set()
    for title in entity_title_candidates(entity):
        query = _title_author_query(_comma_to_colon_title(title), author)
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


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


def _fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _core_title_tokens(title: str, *, leading_only: bool = False) -> list[str]:
    text = _strip_parentheticals(title or "")
    if leading_only:
        text = _SUBTITLE_SPLIT_RE.split(text, maxsplit=1)[0]
    else:
        text = _SUBTITLE_SPLIT_RE.sub(" ", text)
    text = _fold_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [
        token
        for token in text.split()
        if token and token not in _TITLE_STOP_WORDS
    ]


def titles_match(extracted: str, hit_title: str, author: str = "") -> bool:
    left = _core_title_tokens(extracted)
    right = _core_title_tokens(hit_title)
    if bool(left) and left == right:
        return True
    if _has_subtitle(extracted) or _has_subtitle(hit_title):
        lead_left = _core_title_tokens(extracted, leading_only=True)
        lead_right = _core_title_tokens(hit_title, leading_only=True)
        if bool(lead_left) and lead_left == lead_right:
            return True
    author_tokens = _name_tokens(author)
    if not author_tokens:
        return False
    left = [token for token in left if token not in author_tokens]
    right = [token for token in right if token not in author_tokens]
    return bool(left) and left == right


def _title_dedupe_key(title: str) -> tuple[str, ...]:
    tokens = tuple(_core_title_tokens(title))
    if tokens:
        return tokens
    return (_fold_accents(title.lower().strip()),)


def _title_segments(title: str) -> list[str]:
    cleaned = (title or "").strip()
    if not cleaned:
        return []
    segments = [cleaned]
    for part in _SUBTITLE_SPLIT_RE.split(cleaned):
        part = part.strip()
        if part and part not in segments:
            segments.append(part)
    return segments


def _segment_is_author_only(segment: str, author: str) -> bool:
    tokens = set(_core_title_tokens(segment))
    if not tokens:
        return True
    return not (tokens - _name_tokens(author))


def entity_title_candidates(entity: Entity) -> list[str]:
    """Unique extracted titles, name first, capped so search cannot fan out."""
    seen: set[tuple[str, ...]] = set()
    candidates: list[str] = []
    segments: list[str] = []
    for title in (entity.name, *entity.alternate_names):
        for segment in _title_segments(title):
            if segment not in segments:
                segments.append(segment)
    for index, cleaned in enumerate(segments):
        if index > 0 and _segment_is_author_only(cleaned, entity.creator_or_author):
            continue
        key = _title_dedupe_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cleaned)
        if len(candidates) >= MAX_TITLE_CANDIDATES:
            break
    return candidates


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, list):
        return []
    titles: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                titles.append(cleaned)
        elif isinstance(item, dict):
            text = str(item.get("title") or item.get("name") or "").strip()
            if text:
                titles.append(text)
    return titles


def _hit_titles(hit: HardcoverHit) -> list[str]:
    return [title for title in (hit.title, *hit.alternative_titles) if title.strip()]


def _name_tokens(name: str) -> set[str]:
    text = _fold_accents((name or "").lower())
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
        alternative_titles=_string_list(document.get("alternative_titles")),
    )


def pick_match(
    entity: Entity,
    hits: list[HardcoverHit],
    extra_titles: list[str] | None = None,
) -> Optional[HardcoverHit]:
    """First hit whose title is close and whose authors overlap when known."""
    extracted = entity_title_candidates(entity)
    if extra_titles:
        extracted = [*extracted, *extra_titles]
    for hit in hits:
        if not any(
            titles_match(candidate, hit_title, entity.creator_or_author)
            for candidate in extracted
            for hit_title in _hit_titles(hit)
        ):
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


TitleTranslator = Callable[[Entity], Awaitable[str]]


class HardcoverClient:
    def __init__(
        self,
        settings: Settings,
        title_translator: TitleTranslator | None = None,
    ) -> None:
        self._settings = settings
        self._title_translator = title_translator
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
        queries = hardcover_search_queries(entity)
        if not queries:
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.no_match,
            )
        match, errored = await self._search_queries(entity, queries)
        if errored:
            return HardcoverAction(
                entity_name=entity.name,
                outcome=HardcoverOutcome.error,
            )
        if match is None:
            fallback, extra_titles = await self._catalog_title_queries(entity)
            if fallback:
                match, errored = await self._search_queries(
                    entity, fallback, extra_titles
                )
                if errored:
                    return HardcoverAction(
                        entity_name=entity.name,
                        outcome=HardcoverOutcome.error,
                    )
        if match is None:
            logger.info(
                "hardcover no_match name=%r author=%r",
                entity.name,
                entity.creator_or_author,
            )
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

    async def _search_queries(
        self,
        entity: Entity,
        queries: list[str],
        extra_titles: list[str] | None = None,
    ) -> tuple[Optional[HardcoverHit], bool]:
        """Search each query. Returns (match, errored)."""
        for query in queries:
            search_payload = await self._graphql(SEARCH_QUERY, {"query": query})
            if search_payload is None:
                return None, True
            hits = parse_search_hits(_search_results(search_payload))
            match = pick_match(entity, hits, extra_titles=extra_titles)
            if match is not None:
                return match, False
            logger.info(
                "hardcover miss query=%r name=%r candidates=%s hit_titles=%s",
                query,
                entity.name,
                [*entity_title_candidates(entity), *(extra_titles or [])],
                [hit.title for hit in hits],
            )
        return None, False

    async def _catalog_title_queries(
        self, entity: Entity
    ) -> tuple[list[str], list[str]]:
        """English or original catalog title after the extracted titles missed.

        Author-only search is a worse query here: Hardcover returns five hits,
        and annotated editions and summaries outrank the book itself. A catalog
        title plus the author reuses the same strict matcher.
        """
        if self._title_translator is None or not title_needs_catalog_translation(
            entity.name
        ):
            return [], []
        try:
            translated = (await self._title_translator(entity)).strip()
        except Exception:
            logger.warning(
                "hardcover catalog title failed name=%r",
                entity.name,
                exc_info=True,
            )
            return [], []
        if not translated:
            logger.info("hardcover catalog title empty name=%r", entity.name)
            return [], []
        known = {
            _title_dedupe_key(title) for title in entity_title_candidates(entity)
        }
        if _title_dedupe_key(translated) in known:
            logger.info(
                "hardcover catalog title already searched name=%r title=%r",
                entity.name,
                translated,
            )
            return [], []
        query = _title_author_query(
            _comma_to_colon_title(translated), entity.creator_or_author
        )
        if not query:
            return [], []
        logger.info(
            "hardcover catalog title name=%r title=%r",
            entity.name,
            translated,
        )
        return [query], [translated]

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


async def sync_catalogs(
    settings: Settings,
    hardcover: HardcoverClient | None,
    result: ExtractionResult,
    letterboxd: LetterboxdClient | None = None,
    goodreads: GoodreadsClient | None = None,
) -> tuple[list[HardcoverAction], list[LetterboxdAction], list[GoodreadsAction]]:
    """Dispatch by entity type. Fail-soft. Does not write an Obsidian note.

    Books go to Hardcover and/or Goodreads when those are enabled; movies go
    to Letterboxd. A service is not called when that type is absent or the
    client is disabled. Mixed extracts hit each configured sink. Saving the
    note is a separate, optional step.
    """
    from app.goodreads import (
        GoodreadsAction,
        apply_goodreads_actions,
        select_goodreads_candidates,
    )

    hardcover_actions: list[HardcoverAction] = []
    letterboxd_actions: list[LetterboxdAction] = []
    goodreads_actions: list[GoodreadsAction] = []
    if hardcover is not None and hardcover.enabled:
        if select_hardcover_candidates(
            result.entities, settings.hardcover_books_per_job
        ):
            hardcover_actions = await hardcover.sync_books(result.entities)
            apply_hardcover_actions(result.entities, hardcover_actions)
    if goodreads is not None and goodreads.enabled:
        if select_goodreads_candidates(
            result.entities, settings.goodreads_books_per_job
        ):
            goodreads_actions = await goodreads.sync_books(result.entities)
            apply_goodreads_actions(result.entities, goodreads_actions)
    if letterboxd is not None and letterboxd.enabled:
        if select_letterboxd_candidates(
            result.entities, settings.letterboxd_movies_per_job
        ):
            letterboxd_actions = await letterboxd.sync_movies(result.entities)
            apply_letterboxd_actions(result.entities, letterboxd_actions)
    return hardcover_actions, letterboxd_actions, goodreads_actions
