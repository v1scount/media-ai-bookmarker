from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.models import ExtractionResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You identify things worth looking up later from TikTok videos.

You receive:
- Video metadata (title/caption, creator, description with hashtags)
- An ephemeral spoken transcript (do NOT quote it back or include a transcript field)
- Sampled video frames (use them for on-screen text, titles, covers, logos)

Always answer in English, even when the video is in another language. Keep proper
names, titles, and handles in their original spelling.

What counts as an item: tools and apps, physical products, books, movies, series,
music albums, YouTube videos or channels, podcasts, courses, articles, places,
recipes. Skip filler, jokes, generic advice, and unrelated chatter.

How thorough to be:
- If the video is a list or roundup ("5 apps I use", "my favourite books"), set
  video_kind to "list" and capture every item mentioned, each with short notes.
- If the video is about a single thing, set video_kind to "single", return that one
  item with is_main_topic true, and put the useful detail in its notes.
- Set is_main_topic true only for what the video is actually about, not passing mentions.

Naming and links:
- Give the exact name. Put the author, artist, band, studio, or channel in creator_or_author.
- suggested_link only for a URL you are confident is correct (official site, store page,
  YouTube link stated in the video). Otherwise null - do not guess, a plain web search
  is added automatically.
- confidence reflects how sure you are the name is right: "high" when it is stated
  clearly or shown on screen, "low" when you are inferring it.

Other rules:
- Never include a full transcript or raw OCR dump.
- Use "" for unknown text fields, never null.
- Keep it compact: at most 12 items, notes under 200 characters.
- If nothing is worth looking up, return an empty entities array and say so in the summary.
"""

REPAIR_SYSTEM_PROMPT = """You fix malformed JSON.
Return ONLY a JSON object, no markdown fences and no commentary."""

# title, creator and source_url come from yt-dlp metadata, and the search query is
# built locally, so the model is never asked for any of them.
ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "type",
        "name",
        "creator_or_author",
        "notes",
        "is_main_topic",
        "confidence",
        "suggested_link",
    ],
    "properties": {
        "type": {
            "type": "string",
            "description": (
                "Category of the item. Use 'tool' for software, apps, websites and "
                "services; 'product' for physical things; 'album' for music "
                "(artist or band goes in creator_or_author); 'video' for YouTube "
                "videos and channels; 'other' only when nothing else fits."
            ),
            "enum": [
                "tool",
                "product",
                "book",
                "movie",
                "series",
                "album",
                "video",
                "podcast",
                "course",
                "article",
                "place",
                "recipe",
                "other",
            ],
        },
        "name": {
            "type": "string",
            "description": (
                "Exact name of the item as stated or shown, without extra words."
            ),
        },
        "creator_or_author": {
            "type": "string",
            "description": (
                "Author, artist, band, director, studio, or channel behind the item. "
                "Empty string if unknown."
            ),
        },
        "notes": {
            "type": "string",
            "description": (
                "Why it was mentioned and what it is for, in English, under 200 "
                "characters. No quotes from the transcript."
            ),
        },
        "is_main_topic": {
            "type": "boolean",
            "description": (
                "True only if the video is primarily about this item."
            ),
        },
        "confidence": {
            "type": "string",
            "description": (
                "How certain the name is correct: 'high' if stated or shown clearly, "
                "'low' if inferred."
            ),
            "enum": ["high", "medium", "low"],
        },
        "suggested_link": {
            "type": ["string", "null"],
            "description": (
                "Official URL only when confident it is correct, otherwise null. "
                "Never guess a URL."
            ),
        },
    },
}

RESULT_JSON_SCHEMA: dict[str, Any] = {
    "name": "tiktok_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "video_kind", "entities"],
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "One to three sentences in English describing what the video "
                    "recommends and who it is useful for."
                ),
            },
            "video_kind": {
                "type": "string",
                "description": (
                    "'list' for roundups covering several items, 'single' when the "
                    "video is about one thing, 'other' otherwise."
                ),
                "enum": ["single", "list", "other"],
            },
            "entities": {
                "type": "array",
                "description": "Items worth looking up later, most important first.",
                "items": ENTITY_SCHEMA,
            },
        },
    },
}


@dataclass
class ChatReply:
    content: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None


class TruncatedResponseError(RuntimeError):
    """Model hit the output token cap, so retrying the same way would truncate again."""


def truncate_text(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[{label} truncated]"


def _image_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    return f"data:{mime};base64,{b64}"


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _parse_result(raw: str, source_url: str) -> ExtractionResult:
    cleaned = _strip_code_fence(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    data["source_url"] = source_url
    return ExtractionResult.model_validate(data)


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/local/tiktok-parser",
                "X-Title": "TikTok Telegram Extractor",
            },
            timeout=settings.openrouter_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def verify_model(self) -> bool:
        """Warn at startup if the configured model cannot accept images."""
        model = self._settings.openrouter_model
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            entries = response.json().get("data") or []
        except Exception as exc:
            logger.warning("Could not verify model capabilities: %s", exc)
            return True

        for entry in entries:
            if entry.get("id") != model:
                continue
            modalities = (entry.get("architecture") or {}).get(
                "input_modalities"
            ) or []
            if "image" in modalities:
                logger.info("Model %s supports image input", model)
                return True
            logger.warning(
                "Model %s does not accept images (input_modalities=%s). "
                "Frames will be skipped to avoid failed requests.",
                model,
                modalities,
            )
            return False

        logger.warning("Model %s not found in OpenRouter catalog", model)
        return True

    async def extract(
        self,
        *,
        source_url: str,
        title: str,
        creator: str,
        description: str,
        transcript: str,
        frame_paths: list[Path],
    ) -> ExtractionResult:
        settings = self._settings
        description = truncate_text(
            description, settings.max_description_chars, "description"
        )
        transcript = truncate_text(
            transcript, settings.max_transcript_chars, "transcript"
        )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"title: {title}\n"
                    f"creator: {creator}\n"
                    f"description: {description}\n\n"
                    f"spoken_transcript (ephemeral, do not echo):\n"
                    f"{transcript or '(none)'}"
                ),
            }
        ]
        for path in frame_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(path)},
                }
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        reply = await self._chat(messages)
        if reply.finish_reason == "length":
            raise TruncatedResponseError(
                "Model output hit the token cap before finishing the JSON. "
                "Raise MAX_OUTPUT_TOKENS or try a shorter video."
            )

        try:
            return _parse_result(reply.content, source_url)
        except Exception as first_err:
            logger.warning(
                "Schema validation failed, repairing without images: %s", first_err
            )
            return await self._repair(reply.content, first_err, source_url)

    async def _repair(
        self,
        bad_content: str,
        error: Exception,
        source_url: str,
    ) -> ExtractionResult:
        """Second pass that resends only the broken text, never the frames."""
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Fix this response so it matches the required schema "
                    "(keys: summary, video_kind, entities; "
                    "use \"\" instead of null for text fields).\n\n"
                    f"Validation error:\n{str(error)[:800]}\n\n"
                    f"Invalid response:\n{bad_content[:4000]}"
                ),
            },
        ]
        reply = await self._chat(repair_messages)
        return _parse_result(reply.content, source_url)

    async def _chat(self, messages: list[dict[str, Any]]) -> ChatReply:
        payload = {
            "model": self._settings.openrouter_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self._settings.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": RESULT_JSON_SCHEMA,
            },
            "usage": {"include": True},
        }
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"OpenRouter error {response.status_code}: {detail}")

        data = response.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {data!r}") from exc

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        logger.info(
            "OpenRouter usage: prompt=%s completion=%s total=%s cost=%s images=%s",
            prompt_tokens,
            completion_tokens,
            usage.get("total_tokens"),
            usage.get("cost"),
            sum(
                1
                for message in messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            ),
        )

        return ChatReply(
            content=content or "",
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
