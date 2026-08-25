"""Optional Letterboxd watchlist sync via letterboxd-middleman.

Mirrors app.hardcover: Save-only, fail-soft, movies only. Matching lives in
the middleman; this client posts title/year/director and records the outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from app.config import Settings
from app.models import Confidence, Entity, EntityType, resolved_year

logger = logging.getLogger(__name__)

LETTERBOXD_USER_AGENT = "media-ai-bookmarker (personal letterboxd sync)"


class LetterboxdOutcome(str, Enum):
    added = "added"
    already_on_watchlist = "already_on_watchlist"
    no_match = "no_match"
    ambiguous = "ambiguous"
    auth_required = "auth_required"
    error = "error"


@dataclass(frozen=True)
class LetterboxdAction:
    entity_name: str
    outcome: LetterboxdOutcome
    letterboxd_url: str = ""
    message: str = ""

    def summary_line(self) -> str:
        name = self.entity_name
        if self.outcome == LetterboxdOutcome.added:
            return f"Letterboxd: added {name} to watchlist"
        if self.outcome == LetterboxdOutcome.already_on_watchlist:
            return f"Letterboxd: {name} already on your watchlist — left as-is"
        if self.outcome == LetterboxdOutcome.no_match:
            return f'Letterboxd: skipped "{name}" (no confident match)'
        if self.outcome == LetterboxdOutcome.ambiguous:
            return f'Letterboxd: skipped "{name}" (ambiguous match)'
        if self.outcome == LetterboxdOutcome.auth_required:
            return f"Letterboxd: {name} needs the phone signed in"
        extra = f" ({self.message})" if self.message else ""
        return f"Letterboxd: error looking up {name}{extra}"


def authorization_header(api_key: str) -> str:
    """Middleman expects `Bearer <token>`. Leave an existing Bearer prefix alone."""
    key = api_key.strip()
    if key.lower().startswith("bearer "):
        return key
    return f"Bearer {key}"


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def resolve_middleman_url(url: str, *, in_docker: bool | None = None) -> str:
    """Inside a container, 127.0.0.1 is the bot itself, not the host middleman."""
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if in_docker is None:
        in_docker = running_in_docker()
    if not in_docker:
        return cleaned
    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return cleaned
    netloc = parsed.netloc
    if parsed.hostname:
        netloc = netloc.replace(parsed.hostname, "host.docker.internal", 1)
    rewritten = urlunparse(parsed._replace(netloc=netloc)).rstrip("/")
    logger.warning(
        "LETTERBOXD_MIDDLEMAN_URL %s is this container; using %s instead",
        cleaned,
        rewritten,
    )
    return rewritten


def watchlist_url(base_url: str) -> str:
    return f"{resolve_middleman_url(base_url).rstrip('/')}/watchlist"


def watchlist_payload(entity: Entity) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": entity.name,
        "director": entity.creator_or_author or "",
    }
    year = resolved_year(entity)
    if year is not None:
        payload["year"] = year
    return payload


def select_letterboxd_candidates(entities: list[Entity], limit: int) -> list[Entity]:
    """Movies worth a watchlist write: not low-confidence, main topics first."""
    if limit <= 0:
        return []
    eligible = [
        entity
        for entity in entities
        if entity.type == EntityType.movie and entity.confidence != Confidence.low
    ]
    main = [entity for entity in eligible if entity.is_main_topic]
    rest = [entity for entity in eligible if not entity.is_main_topic]
    return (main + rest)[:limit]


def apply_letterboxd_actions(
    entities: list[Entity],
    actions: list[LetterboxdAction],
) -> None:
    """Stamp letterboxd_url onto matched entities, first action per title wins."""
    by_name: dict[str, LetterboxdAction] = {}
    for action in actions:
        if action.letterboxd_url and action.entity_name not in by_name:
            by_name[action.entity_name] = action
    for entity in entities:
        action = by_name.get(entity.name)
        if action:
            entity.letterboxd_url = action.letterboxd_url


def format_letterboxd_report(actions: list[LetterboxdAction]) -> str:
    return "\n".join(action.summary_line() for action in actions)


def parse_watchlist_response(payload: object, entity_name: str) -> LetterboxdAction:
    if not isinstance(payload, dict):
        return LetterboxdAction(
            entity_name=entity_name,
            outcome=LetterboxdOutcome.error,
        )
    raw_outcome = str(payload.get("outcome") or "").strip().lower()
    try:
        outcome = LetterboxdOutcome(raw_outcome)
    except ValueError:
        outcome = LetterboxdOutcome.error
    url = str(payload.get("url") or "").strip()
    if url and not url.lower().startswith("http"):
        url = ""
    message = str(payload.get("message") or "").strip()
    stamp_url = url if outcome in {
        LetterboxdOutcome.added,
        LetterboxdOutcome.already_on_watchlist,
    } else ""
    return LetterboxdAction(
        entity_name=entity_name,
        outcome=outcome,
        letterboxd_url=stamp_url,
        message=message,
    )


class LetterboxdClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.letterboxd_timeout_seconds,
                connect=10.0,
            ),
            headers={"User-Agent": LETTERBOXD_USER_AGENT},
        )

    @property
    def enabled(self) -> bool:
        return (
            bool(self._settings.letterboxd_middleman_url)
            and bool(self._settings.letterboxd_middleman_api_key)
            and self._settings.letterboxd_movies_per_job > 0
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def sync_movies(self, entities: list[Entity]) -> list[LetterboxdAction]:
        """POST each movie to the middleman. Never raises."""
        if not self.enabled:
            return []
        candidates = select_letterboxd_candidates(
            entities, self._settings.letterboxd_movies_per_job
        )
        actions: list[LetterboxdAction] = []
        for entity in candidates:
            try:
                actions.append(await self._sync_one(entity))
            except Exception:
                logger.warning(
                    "letterboxd sync failed name=%r",
                    entity.name,
                    exc_info=True,
                )
                actions.append(
                    LetterboxdAction(
                        entity_name=entity.name,
                        outcome=LetterboxdOutcome.error,
                    )
                )
        return actions

    async def _sync_one(self, entity: Entity) -> LetterboxdAction:
        if not entity.name.strip():
            return LetterboxdAction(
                entity_name=entity.name,
                outcome=LetterboxdOutcome.no_match,
            )
        url = watchlist_url(self._settings.letterboxd_middleman_url)
        try:
            response = await self._client.post(
                url,
                headers={
                    "Authorization": authorization_header(
                        self._settings.letterboxd_middleman_api_key
                    ),
                    "Content-Type": "application/json",
                },
                json=watchlist_payload(entity),
            )
        except Exception as exc:
            target = watchlist_url(self._settings.letterboxd_middleman_url)
            logger.warning(
                "letterboxd request failed name=%r url=%s: %s",
                entity.name,
                target,
                exc,
            )
            return LetterboxdAction(
                entity_name=entity.name,
                outcome=LetterboxdOutcome.error,
                message=type(exc).__name__,
            )
        if response.status_code == 401:
            logger.warning("letterboxd HTTP 401 name=%r", entity.name)
            return LetterboxdAction(
                entity_name=entity.name,
                outcome=LetterboxdOutcome.error,
                message="unauthorized",
            )
        if response.status_code >= 400:
            logger.warning(
                "letterboxd HTTP %s body=%s",
                response.status_code,
                response.text[:300],
            )
            return LetterboxdAction(
                entity_name=entity.name,
                outcome=LetterboxdOutcome.error,
            )
        try:
            payload: Any = response.json()
        except Exception:
            return LetterboxdAction(
                entity_name=entity.name,
                outcome=LetterboxdOutcome.error,
            )
        action = parse_watchlist_response(payload, entity.name)
        logger.info(
            "letterboxd %s name=%r url=%s",
            action.outcome.value,
            entity.name,
            action.letterboxd_url or "(none)",
        )
        return action
