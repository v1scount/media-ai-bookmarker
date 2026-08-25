from __future__ import annotations

import re
from enum import Enum
from typing import Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, field_validator


TIKTOK_URL_RE = re.compile(
    r"https?://(?:(?:www|vm|vt)\.)?tiktok\.com/[^\s<>\"']+",
    re.IGNORECASE,
)

# x.com plus the mirror domains people paste when they want an inline preview
X_HOSTS = (
    "x.com",
    "twitter.com",
    "vxtwitter.com",
    "fxtwitter.com",
    "fixupx.com",
    "fixvx.com",
    "twittpr.com",
)

# Only post URLs are actionable, so a bare profile link is deliberately not matched
X_URL_RE = re.compile(
    r"https?://(?:(?:www|mobile|m)\.)?(?:"
    + "|".join(host.replace(".", r"\.") for host in X_HOSTS)
    + r")/(?:[^\s<>\"'/]+/)*status(?:es)?/\d+[^\s<>\"']*",
    re.IGNORECASE,
)

# Matches /status/123, /statuses/123 and /i/status/123, with or without a suffix
X_STATUS_ID_RE = re.compile(r"/status(?:es)?/(\d+)", re.IGNORECASE)

TRAILING_PUNCTUATION = ").,];>"

SEARCH_URL_TEMPLATE = "https://kagi.com/search?q={query}"
DEFAULT_AMAZON_SEARCH_HOST = "www.amazon.com"
MERCADO_LIBRE_SEARCH_TEMPLATE = "https://listado.mercadolibre.com.ar/{slug}"
AMAZON_SEARCH_TEMPLATE = "https://{host}/s?k={query}"
EBAY_SEARCH_TEMPLATE = "https://www.ebay.com/sch/i.html?_nkw={query}"
TRAILING_HASHTAGS_RE = re.compile(r"(?:(?:^|\s)#\w[\w.-]*)+$", re.UNICODE)
TRAILING_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
YEAR_MIN = 1870
YEAR_MAX = 2100


class SourceKind(str, Enum):
    tiktok = "tiktok"
    x = "x"


class EntityType(str, Enum):
    tool = "tool"
    product = "product"
    book = "book"
    movie = "movie"
    series = "series"
    album = "album"
    video = "video"
    podcast = "podcast"
    course = "course"
    article = "article"
    place = "place"
    recipe = "recipe"
    other = "other"


# Words the model is likely to reach for that map onto our categories
TYPE_ALIASES: dict[str, EntityType] = {
    "app": EntityType.tool,
    "application": EntityType.tool,
    "software": EntityType.tool,
    "website": EntityType.tool,
    "site": EntityType.tool,
    "service": EntityType.tool,
    "extension": EntityType.tool,
    "plugin": EntityType.tool,
    "gadget": EntityType.product,
    "gear": EntityType.product,
    "device": EntityType.product,
    "track": EntityType.album,
    "song": EntityType.album,
    "music": EntityType.album,
    "artist": EntityType.album,
    "band": EntityType.album,
    "film": EntityType.movie,
    "documentary": EntityType.movie,
    "tv": EntityType.series,
    "tv show": EntityType.series,
    "show": EntityType.series,
    "anime": EntityType.series,
    "youtube": EntityType.video,
    "youtube video": EntityType.video,
    "youtube channel": EntityType.video,
    "channel": EntityType.video,
    "paper": EntityType.article,
    "blog": EntityType.article,
    "blog post": EntityType.article,
    "newsletter": EntityType.article,
    "essay": EntityType.article,
    "restaurant": EntityType.place,
    "location": EntityType.place,
    "city": EntityType.place,
    "class": EntityType.course,
    "tutorial": EntityType.course,
}

# Extra keyword appended to a plain web search to disambiguate the result
SEARCH_HINTS: dict[EntityType, str] = {
    EntityType.book: "book",
    EntityType.movie: "film",
    EntityType.series: "series",
    EntityType.album: "album",
    EntityType.video: "youtube",
    EntityType.podcast: "podcast",
    EntityType.course: "course",
    EntityType.article: "article",
    EntityType.recipe: "recipe",
}


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class VideoKind(str, Enum):
    single = "single"
    list = "list"
    other = "other"


def _none_to_empty(value: object) -> object:
    return "" if value is None else value


def parse_year(value: object) -> Optional[int]:
    """Coerce a model year to int, or None when missing or implausible."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    year: Optional[int] = None
    if isinstance(value, int):
        year = value
    elif isinstance(value, float) and value.is_integer():
        year = int(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            year = int(cleaned)
    if year is None or year < YEAR_MIN or year > YEAR_MAX:
        return None
    return year


def year_from_text(text: str) -> Optional[int]:
    """Trailing '(2010)' in a name or note, otherwise None. Never guesses."""
    match = TRAILING_YEAR_RE.search((text or "").strip())
    if not match:
        return None
    return parse_year(match.group(1))


def split_caption(text: str) -> tuple[str, str]:
    """Split a trailing hashtag dump from the prose of a caption."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "", ""
    match = TRAILING_HASHTAGS_RE.search(cleaned)
    if not match:
        return cleaned, ""
    tags = match.group(0).strip()
    prose = cleaned[: match.start()].strip()
    if not prose:
        return "", tags
    if len(re.findall(r"#\w", tags)) < 2:
        return cleaned, ""
    return prose, tags


def normalize_amazon_host(value: object) -> str:
    """Accept a bare host or a pasted Amazon URL and return just the host."""
    text = str(value or DEFAULT_AMAZON_SEARCH_HOST).strip()
    text = re.sub(r"^https?://", "", text, flags=re.IGNORECASE)
    host = text.split("/")[0].strip().lower()
    return host or DEFAULT_AMAZON_SEARCH_HOST


def mercadolibre_slug(query: str) -> str:
    """Hyphenated path segment used by listado.mercadolibre.com.ar."""
    text = (query or "").lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text or "producto"


class Entity(BaseModel):
    type: EntityType = EntityType.other
    name: str
    creator_or_author: str = ""
    notes: str = ""
    is_main_topic: bool = False
    confidence: Confidence = Confidence.medium
    suggested_link: Optional[str] = None
    hardcover_url: Optional[str] = None
    letterboxd_url: Optional[str] = None
    # Release year when the model (or a trailing "(2010)") knows it
    year: Optional[int] = None

    @field_validator("creator_or_author", "notes", "name", mode="before")
    @classmethod
    def coerce_null_strings(cls, value: object) -> object:
        return _none_to_empty(value)

    @field_validator("suggested_link", "hardcover_url", "letterboxd_url", mode="before")
    @classmethod
    def empty_link_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or not cleaned.lower().startswith("http"):
                return None
            return cleaned
        return value

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, value: object) -> object:
        return parse_year(value)

    @field_validator("is_main_topic", mode="before")
    @classmethod
    def coerce_main_topic(cls, value: object) -> object:
        return False if value is None else value

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return Confidence(normalized)
            except ValueError:
                return Confidence.medium
        return Confidence.medium if value is None else value

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, value: object) -> object:
        if value is None or value == "":
            return EntityType.other
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return EntityType(normalized)
            except ValueError:
                return TYPE_ALIASES.get(normalized, EntityType.other)
        return value

    @property
    def search_query(self) -> str:
        """Local search string: name, author, and a type hint. No LLM involved."""
        parts = [self.name, self.creator_or_author, SEARCH_HINTS.get(self.type, "")]
        return " ".join(part for part in parts if part).strip()

    @property
    def search_url(self) -> str:
        """Plain web search for this item, built locally (no LLM research)."""
        return SEARCH_URL_TEMPLATE.format(query=quote_plus(self.search_query))

    def shop_query(self) -> str:
        """Name and brand only — no type hint. Used for store searches."""
        return " ".join(
            part for part in (self.name, self.creator_or_author) if part
        ).strip()

    def shop_links(
        self, amazon_host: str = DEFAULT_AMAZON_SEARCH_HOST
    ) -> list[tuple[str, str]]:
        """Mercado Libre / Amazon / eBay search URLs. Empty unless this is a product."""
        if self.type != EntityType.product:
            return []
        query = self.shop_query()
        if not query:
            return []
        host = normalize_amazon_host(amazon_host)
        quoted = quote_plus(query)
        return [
            (
                "Mercado Libre",
                MERCADO_LIBRE_SEARCH_TEMPLATE.format(slug=mercadolibre_slug(query)),
            ),
            ("Amazon", AMAZON_SEARCH_TEMPLATE.format(host=host, query=quoted)),
            ("eBay", EBAY_SEARCH_TEMPLATE.format(query=quoted)),
        ]


def resolved_year(entity: Entity) -> Optional[int]:
    """Prefer the structured year; fall back to a trailing '(YYYY)' in name/notes."""
    if entity.year is not None:
        return entity.year
    for text in (entity.name, entity.notes):
        found = year_from_text(text)
        if found is not None:
            return found
    return None


class MediaKind(str, Enum):
    image = "image"
    video = "video"


class LinkRef(BaseModel):
    """A link found in a post, with whatever the page told us about itself."""

    url: str
    title: str = ""
    description: str = ""

    @field_validator("title", "description", "url", mode="before")
    @classmethod
    def coerce_null_strings(cls, value: object) -> object:
        return _none_to_empty(value)


class MediaRef(BaseModel):
    kind: MediaKind = MediaKind.image
    url: str = ""
    # Vault-relative path, filled in only once the file is saved into the vault
    vault_path: str = ""

    @field_validator("url", "vault_path", mode="before")
    @classmethod
    def coerce_null_strings(cls, value: object) -> object:
        return _none_to_empty(value)


class PostContent(BaseModel):
    """A post captured verbatim, for notes built without the model."""

    text: str = ""
    links: list[LinkRef] = Field(default_factory=list)
    media: list[MediaRef] = Field(default_factory=list)

    @field_validator("text", mode="before")
    @classmethod
    def coerce_null_strings(cls, value: object) -> object:
        return _none_to_empty(value)

    @property
    def images(self) -> list[MediaRef]:
        return [item for item in self.media if item.kind == MediaKind.image]

    @property
    def videos(self) -> list[MediaRef]:
        return [item for item in self.media if item.kind == MediaKind.video]


class ExtractionResult(BaseModel):
    source_url: str
    # creator, source_kind and source_description come from the pipeline.
    # title is written by the model on the LLM path, with a metadata fallback.
    source_kind: SourceKind = SourceKind.tiktok
    title: str = ""
    creator: str = ""
    source_description: str = ""
    summary: str = ""
    video_kind: VideoKind = VideoKind.other
    entities: list[Entity] = Field(default_factory=list)
    # Set only for notes captured verbatim instead of sent to the model
    post: Optional[PostContent] = None

    @property
    def is_raw_capture(self) -> bool:
        return self.post is not None

    @field_validator("source_kind", mode="before")
    @classmethod
    def coerce_source_kind(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return SourceKind(normalized)
            except ValueError:
                return SourceKind.tiktok
        return SourceKind.tiktok if value is None else value

    @field_validator(
        "title", "creator", "summary", "source_url", "source_description", mode="before"
    )
    @classmethod
    def coerce_null_strings(cls, value: object) -> object:
        return _none_to_empty(value)

    @field_validator("video_kind", mode="before")
    @classmethod
    def coerce_video_kind(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return VideoKind(normalized)
            except ValueError:
                return VideoKind.other
        return VideoKind.other if value is None else value

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_null_entities(cls, value: object) -> object:
        return [] if value is None else value

    def ordered_entities(self) -> list[Entity]:
        """Main topics first, original order preserved within each group."""
        main = [entity for entity in self.entities if entity.is_main_topic]
        rest = [entity for entity in self.entities if not entity.is_main_topic]
        return main + rest

    def entity_type_tags(self) -> list[str]:
        """Distinct entity types, in note order, for frontmatter tags."""
        seen: list[str] = []
        for entity in self.ordered_entities():
            value = entity.type.value
            if value not in seen:
                seen.append(value)
        return seen


def extract_tiktok_url(text: str) -> Optional[str]:
    """Return the first TikTok URL found in text, or None."""
    return _first_url(TIKTOK_URL_RE, text)


def extract_x_url(text: str) -> Optional[str]:
    """Return the first X/Twitter status URL found in text, or None.

    Profile links (x.com/alice) are ignored; only /status/{id} counts.
    """
    if not text:
        return None
    for match in X_URL_RE.finditer(text):
        url = match.group(0).rstrip(TRAILING_PUNCTUATION)
        if extract_status_id(url):
            return url
    return None


def extract_supported_url(text: str) -> Optional[tuple[SourceKind, str]]:
    """Return (kind, url) for whichever supported link appears first in text."""
    if not text:
        return None
    candidates: list[tuple[int, SourceKind, str]] = []
    tiktok = TIKTOK_URL_RE.search(text)
    if tiktok:
        candidates.append(
            (
                tiktok.start(),
                SourceKind.tiktok,
                tiktok.group(0).rstrip(TRAILING_PUNCTUATION),
            )
        )
    for match in X_URL_RE.finditer(text):
        url = match.group(0).rstrip(TRAILING_PUNCTUATION)
        if extract_status_id(url):
            candidates.append((match.start(), SourceKind.x, url))
            break
    if not candidates:
        return None
    _start, kind, url = min(candidates, key=lambda item: item[0])
    return kind, url


def extract_status_id(url: str) -> Optional[str]:
    """Return the numeric tweet id from an X status URL, or None."""
    if not url:
        return None
    match = X_STATUS_ID_RE.search(url)
    return match.group(1) if match else None


def _first_url(pattern: re.Pattern[str], text: str) -> Optional[str]:
    if not text:
        return None
    match = pattern.search(text)
    if not match:
        return None
    return match.group(0).rstrip(TRAILING_PUNCTUATION)
