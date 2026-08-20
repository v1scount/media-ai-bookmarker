"""Fetch a single X/Twitter post and the pages it links to.

The X-specific counterpart to app.download. No API key: metadata comes from the
public FxTwitter/VxTwitter mirrors, with one used as a fallback for the other.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

X_HEADERS = {
    "User-Agent": CHROME_UA,
    "Referer": "https://x.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

PAGE_HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

FXTWITTER_API = "https://api.fxtwitter.com/status/{status_id}"
VXTWITTER_API = "https://api.vxtwitter.com/Twitter/status/{status_id}"

URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# Link previews are for the pages a tweet points at, not X itself or raw media
SKIP_PREVIEW_HOSTS = {
    "x.com",
    "twitter.com",
    "mobile.twitter.com",
    "t.co",
    "vxtwitter.com",
    "fxtwitter.com",
    "fixupx.com",
    "fixvx.com",
    "twittpr.com",
    "pbs.twimg.com",
    "video.twimg.com",
    "abs.twimg.com",
}

# Shorteners must be followed before we can tell what they point at
REDIRECTOR_HOSTS = {"t.co"}

MAX_LINK_PREVIEWS = 5
LINK_PREVIEW_TIMEOUT = 8.0
MAX_PAGE_BYTES = 64 * 1024


class TweetFetchError(RuntimeError):
    """Neither mirror could return this tweet (private, deleted, or mirror down)."""


@dataclass
class LinkPreview:
    url: str
    title: str = ""
    description: str = ""


@dataclass
class TweetData:
    tweet_id: str
    text: str = ""
    author_name: str = ""
    author_handle: str = ""
    urls: list[str] = field(default_factory=list)
    photo_urls: list[str] = field(default_factory=list)
    video_urls: list[str] = field(default_factory=list)
    has_video: bool = False
    quoted_text: str = ""
    quoted_urls: list[str] = field(default_factory=list)

    @property
    def all_urls(self) -> list[str]:
        return _dedupe(self.urls + self.quoted_urls)


def resolve_redirect(url: str, *, timeout: float = 15.0) -> str:
    """Follow redirects (t.co and friends) to the destination URL."""
    try:
        import httpx

        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=X_HEADERS,
        ) as client:
            response = client.head(url)
            if response.status_code >= 400:
                response = client.get(url)
            final = str(response.url)
            if final and final != url:
                logger.info("Resolved %s -> %s", url, final)
            return final or url
    except Exception as exc:
        logger.warning("URL resolve failed for %s: %s", url, exc)
    return url


def fetch_tweet(status_id: str, *, timeout: float = 20.0) -> TweetData:
    """Fetch tweet metadata, trying FxTwitter then VxTwitter."""
    import httpx

    attempts = (
        ("fxtwitter", FXTWITTER_API.format(status_id=status_id), _map_fxtwitter),
        ("vxtwitter", VXTWITTER_API.format(status_id=status_id), _map_vxtwitter),
    )

    errors: list[str] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers=X_HEADERS,
    ) as client:
        for label, api_url, mapper in attempts:
            try:
                response = client.get(api_url)
                if response.status_code >= 400:
                    errors.append(f"{label}: HTTP {response.status_code}")
                    continue
                tweet = mapper(response.json(), status_id=status_id)
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                logger.warning("Tweet fetch via %s failed: %s", label, exc)
                continue
            if tweet is not None:
                logger.info(
                    "Fetched tweet %s via %s (photos=%s video=%s urls=%s)",
                    status_id,
                    label,
                    len(tweet.photo_urls),
                    tweet.has_video,
                    len(tweet.urls),
                )
                return tweet
            errors.append(f"{label}: no tweet in response")

    raise TweetFetchError(
        "Could not read this post (it may be private, deleted, or the mirrors are "
        f"down). Tried — {'; '.join(errors) or 'no details'}"
    )


def _map_fxtwitter(payload: Any, status_id: str) -> Optional[TweetData]:
    if not isinstance(payload, dict):
        return None
    tweet = payload.get("tweet")
    if not isinstance(tweet, dict):
        return None

    author = tweet.get("author") or {}
    media = tweet.get("media") or {}
    photos = [
        str(item.get("url"))
        for item in (media.get("photos") or [])
        if isinstance(item, dict) and item.get("url")
    ]
    videos = [
        str(item.get("url"))
        for item in (media.get("videos") or [])
        if isinstance(item, dict) and item.get("url")
    ]
    has_video = bool(media.get("videos"))

    quote = tweet.get("quote") or {}
    quoted_text = str(quote.get("text") or "") if isinstance(quote, dict) else ""

    return TweetData(
        tweet_id=str(tweet.get("id") or status_id),
        text=str(tweet.get("text") or ""),
        author_name=str(author.get("name") or ""),
        author_handle=_normalize_handle(author.get("screen_name")),
        urls=_urls_from_fx(tweet),
        photo_urls=photos,
        video_urls=videos,
        has_video=has_video,
        quoted_text=quoted_text,
        quoted_urls=_urls_from_fx(quote) if isinstance(quote, dict) else [],
    )


def tweet_data_from_json(
    payload: dict[str, Any],
    *,
    status_id: str = "",
) -> TweetData:
    """Map an FxTwitter or VxTwitter JSON payload onto TweetData. No network."""
    tweet = _map_fxtwitter(payload, status_id) or _map_vxtwitter(payload, status_id)
    if tweet is None:
        raise ValueError("Unrecognized tweet JSON")
    return tweet


def _urls_from_fx(tweet: dict[str, Any]) -> list[str]:
    """Collect expanded URLs from tweet text, entities, and FxTwitter facets.

    X's own t.co expansions are substituted into the text links, so a shortener
    never has to be resolved over the network later.
    """
    expansions: dict[str, str] = {}
    entities = tweet.get("entities") or {}
    if isinstance(entities, dict):
        for ent in entities.get("urls") or []:
            if not isinstance(ent, dict):
                continue
            short = ent.get("url")
            expanded = ent.get("expanded_url")
            if short and expanded:
                expansions[str(short)] = str(expanded)

    urls: list[str] = []
    for found in URL_IN_TEXT_RE.findall(str(tweet.get("text") or "")):
        found = found.rstrip(").,];>")
        urls.append(expansions.get(found, found))

    raw = tweet.get("raw_text")
    if isinstance(raw, dict):
        for facet in raw.get("facets") or []:
            if not isinstance(facet, dict) or facet.get("type") != "url":
                continue
            candidate = facet.get("replacement") or facet.get("original")
            if candidate:
                urls.append(str(candidate).rstrip(").,];>"))

    urls.extend(expansions.values())
    return _dedupe(urls)


def _map_vxtwitter(payload: Any, status_id: str) -> Optional[TweetData]:
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if text is None and not payload.get("tweetID"):
        return None

    photos: list[str] = []
    videos: list[str] = []
    extended = payload.get("media_extended")
    if isinstance(extended, list) and extended:
        for item in extended:
            if not isinstance(item, dict):
                continue
            media_type = str(item.get("type") or "").lower()
            url = item.get("url")
            if media_type == "image" and url:
                photos.append(str(url))
            elif media_type in {"video", "gif"} and url:
                videos.append(str(url))
    else:
        for url in payload.get("mediaURLs") or []:
            if _looks_like_image(str(url)):
                photos.append(str(url))
            else:
                videos.append(str(url))
    has_video = bool(videos)

    urls = URL_IN_TEXT_RE.findall(str(text or ""))
    quoted_text = ""
    quoted_urls: list[str] = []
    qrt = payload.get("qrt")
    if isinstance(qrt, dict):
        quoted_text = str(qrt.get("text") or "")
        quoted_urls.extend(URL_IN_TEXT_RE.findall(quoted_text))
    if payload.get("qrtURL"):
        quoted_urls.append(str(payload["qrtURL"]))

    return TweetData(
        tweet_id=str(payload.get("tweetID") or status_id),
        text=str(text or ""),
        author_name=str(payload.get("user_name") or ""),
        author_handle=_normalize_handle(payload.get("user_screen_name")),
        urls=_dedupe(url.rstrip(").,];>") for url in urls),
        photo_urls=photos,
        video_urls=videos,
        has_video=has_video,
        quoted_text=quoted_text,
        quoted_urls=_dedupe(url.rstrip(").,];>") for url in quoted_urls),
    )


def _normalize_handle(value: object) -> str:
    handle = str(value or "").strip().lstrip("@")
    return f"@{handle}" if handle else ""


def _looks_like_image(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp"))


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".gif"}
DEFAULT_MAX_MEDIA_BYTES = 100 * 1024 * 1024


def media_suffix(url: str, *, video: bool = False) -> str:
    """File extension for a media URL, with a sane default for X's odd paths.

    Animated GIFs on X are served as .mp4, so they are treated as video here and
    keep whatever extension the URL actually carries.
    """
    suffix = Path(urlparse(url).path).suffix.lower()
    if video:
        return suffix if suffix in VIDEO_EXTS else ".mp4"
    return suffix if suffix in IMAGE_EXTS else ".jpg"


def download_media(
    url: str,
    dest: Path,
    *,
    max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
    timeout: float = 60.0,
) -> bool:
    """Stream one file to an exact path. Returns False if it could not be saved.

    Written to a .part file first so an aborted or oversized download never
    leaves a truncated file sitting in the vault.
    """
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    size = 0
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=X_HEADERS,
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with part.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"exceeds {max_bytes} bytes")
                        handle.write(chunk)
    except Exception as exc:
        logger.warning("Media download failed for %s: %s", url, exc)
        part.unlink(missing_ok=True)
        return False

    if size == 0:
        part.unlink(missing_ok=True)
        return False

    part.replace(dest)
    return True


def download_photos(
    urls: list[str],
    dest_dir: Path,
    *,
    limit: int,
    timeout: float = 30.0,
) -> list[Path]:
    """Download tweet photos into dest_dir; failures are skipped, not fatal."""
    if not urls or limit <= 0:
        return []

    import httpx

    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers=X_HEADERS,
    ) as client:
        for index, url in enumerate(urls[:limit]):
            try:
                response = client.get(url)
                response.raise_for_status()
                data = response.content
            except Exception as exc:
                logger.warning("Photo download failed for %s: %s", url, exc)
                continue
            if not data:
                continue
            path = dest_dir / f"photo_{index:02d}{media_suffix(url)}"
            path.write_bytes(data)
            saved.append(path)
    return saved


class _OpenGraphParser(HTMLParser):
    """Pull og:title / og:description / <title> out of a page head."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.og_title = ""
        self.og_description = ""
        self.meta_description = ""
        self.doc_title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag != "meta":
            return
        values = {key.lower(): (value or "") for key, value in attrs}
        key = (values.get("property") or values.get("name") or "").strip().lower()
        content = values.get("content", "").strip()
        if not content:
            return
        if key in {"og:title", "twitter:title"} and not self.og_title:
            self.og_title = content
        elif key in {"og:description", "twitter:description"} and not self.og_description:
            self.og_description = content
        elif key == "description" and not self.meta_description:
            self.meta_description = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.doc_title:
            self.doc_title = data.strip()

    @property
    def title(self) -> str:
        return self.og_title or self.doc_title

    @property
    def description(self) -> str:
        return self.og_description or self.meta_description


def parse_open_graph(html: str) -> tuple[str, str]:
    """Return (title, description) for a page, empty strings when absent."""
    parser = _OpenGraphParser()
    try:
        parser.feed(html)
    except Exception as exc:
        logger.warning("HTML parse failed: %s", exc)
    return _collapse(parser.title), _collapse(parser.description)


def _collapse(text: str, limit: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:limit]


def _host_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def should_preview(url: str) -> bool:
    """True when a fully resolved URL points at a page worth summarising."""
    host = _host_of(url)
    return bool(host) and host not in SKIP_PREVIEW_HOSTS


def worth_resolving(url: str) -> bool:
    """Like should_preview, but shorteners pass because they hide the real host."""
    host = _host_of(url)
    if not host:
        return False
    return host in REDIRECTOR_HOSTS or host not in SKIP_PREVIEW_HOSTS


def fetch_link_previews(
    urls: list[str],
    *,
    limit: int = MAX_LINK_PREVIEWS,
    timeout: float = LINK_PREVIEW_TIMEOUT,
) -> list[LinkPreview]:
    """Resolve each link and read only its title/OG description.

    A page that blocks us still yields a LinkPreview with just the URL, so the
    model always sees where the tweet points.
    """
    import httpx

    previews: list[LinkPreview] = []
    seen: set[str] = set()
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers=PAGE_HEADERS,
    ) as client:
        for url in urls:
            if len(previews) >= limit:
                break
            if not worth_resolving(url):
                continue
            final_url, title, description = _read_head_of_page(client, url)
            if not should_preview(final_url) or final_url in seen:
                continue
            seen.add(final_url)
            previews.append(
                LinkPreview(url=final_url, title=title, description=description)
            )
    return previews


def _read_head_of_page(client, url: str) -> tuple[str, str, str]:
    """Return (final_url, title, description), downloading at most MAX_PAGE_BYTES."""
    try:
        with client.stream("GET", url) as response:
            final_url = str(response.url) or url
            if response.status_code >= 400:
                return final_url, "", ""
            content_type = (response.headers.get("content-type") or "").lower()
            if content_type and "html" not in content_type:
                return final_url, "", ""
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_PAGE_BYTES:
                    break
            html = b"".join(chunks).decode(
                response.charset_encoding or "utf-8", errors="replace"
            )
    except Exception as exc:
        logger.warning("Open Graph fetch failed for %s: %s", url, exc)
        return url, "", ""

    title, description = parse_open_graph(html)
    return final_url, title, description
