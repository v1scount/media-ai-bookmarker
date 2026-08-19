from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

TIKTOK_HEADERS = {
    "User-Agent": CHROME_UA,
    "Referer": "https://www.tiktok.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

X_HEADERS = {
    "User-Agent": CHROME_UA,
    "Referer": "https://x.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


def resolve_tiktok_url(url: str) -> str:
    """Follow short-link redirects (vm/vt) to a canonical tiktok.com URL when possible."""
    try:
        import httpx

        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers=TIKTOK_HEADERS,
        ) as client:
            response = client.head(url)
            if response.status_code >= 400:
                response = client.get(url)
            final = str(response.url)
            if "tiktok.com" in final.lower():
                logger.info("Resolved TikTok URL %s -> %s", url, final)
                return final
    except Exception as exc:
        logger.warning("URL resolve failed for %s: %s", url, exc)
    return url


def _usable_cookies(cookies_file: Optional[Path]) -> Optional[Path]:
    if not cookies_file:
        return None
    try:
        if not cookies_file.is_file():
            return None
        # Ignore placeholder/comment-only files
        text = cookies_file.read_text(encoding="utf-8", errors="ignore")
        has_cookie_line = any(
            line.strip() and not line.lstrip().startswith("#")
            for line in text.splitlines()
        )
        if not has_cookie_line:
            return None
        return cookies_file
    except OSError:
        return None


VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
AUDIO_EXTS = {".m4a", ".aac", ".mp3", ".opus", ".ogg", ".flac"}


def pick_media_files(
    info: dict[str, Any] | None,
    work_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Return (video, standalone_audio) for what yt-dlp actually wrote.

    yt-dlp reports its final (post-merge) file in requested_downloads, which is
    the only reliable source: globbing can pick up per-format intermediates like
    media.f123.mp4, and those are video-only.
    """
    video: Optional[Path] = None
    for entry in (info or {}).get("requested_downloads") or []:
        raw = entry.get("filepath") or entry.get("_filename")
        if raw and Path(raw).is_file():
            video = Path(raw)
            break

    files = sorted(p for p in work_dir.iterdir() if p.is_file()) if work_dir.is_dir() else []

    if video is None:
        # Prefer the merged "media.mp4" over intermediates such as "media.f5.mp4"
        merged = [
            path
            for path in files
            if path.stem == "media" and path.suffix.lower() in VIDEO_EXTS
        ]
        any_video = [path for path in files if path.suffix.lower() in VIDEO_EXTS]
        for group in (merged, any_video, files):
            if group:
                video = group[0]
                break

    audio = next(
        (path for path in files if path.suffix.lower() in AUDIO_EXTS),
        None,
    )
    return video, audio


def _base_ydl_opts(
    work_dir: Path,
    cookies_file: Optional[Path],
    headers: dict[str, str] = TIKTOK_HEADERS,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "outtmpl": str(work_dir / "media.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Prefer a single stream that already has audio, so no merge step can
        # leave us with a silent video-only file
        "format": "b/bv*+ba/bv*",
        "merge_output_format": "mp4",
        "paths": {"home": str(work_dir)},
        "http_headers": dict(headers),
    }
    usable = _usable_cookies(cookies_file)
    if usable:
        opts["cookiefile"] = str(usable)
        logger.info("Using yt-dlp cookies from %s", usable)
    return opts


def download_tiktok(
    url: str,
    work_dir: Path,
    *,
    cookies_file: Optional[Path] = None,
) -> dict[str, Any]:
    """Download a TikTok URL with several extractor strategies.

    Raises the last yt_dlp DownloadError if all attempts fail.
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError

    resolved = resolve_tiktok_url(url)
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("web+headers", {}),
        (
            "mobile-api-host",
            {
                "extractor_args": {
                    "tiktok": {
                        "api_hostname": ["api16-normal-c-useast1a.tiktokv.com"],
                    }
                }
            },
        ),
    ]

    last_error: Exception | None = None
    for label, extra in attempts:
        opts = _base_ydl_opts(work_dir, cookies_file)
        opts.update(extra)
        for leftover in work_dir.glob("media.*"):
            leftover.unlink(missing_ok=True)
        try:
            logger.info("yt-dlp attempt=%s url=%s", label, resolved)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(resolved, download=True)
            if info:
                return info
        except DownloadError as exc:
            last_error = exc
            logger.warning("yt-dlp attempt %s failed: %s", label, exc)
        except Exception as exc:
            last_error = exc
            logger.warning("yt-dlp attempt %s errored: %s", label, exc)

    assert last_error is not None
    raise last_error


def download_x_video(
    url: str,
    work_dir: Path,
    *,
    cookies_file: Optional[Path] = None,
) -> dict[str, Any]:
    """Download the video attached to an X post.

    Single attempt: unlike TikTok there are no extractor host workarounds worth
    trying, and a failure here is non-fatal because the tweet text still carries
    the useful content.
    """
    import yt_dlp

    for leftover in work_dir.glob("media.*"):
        leftover.unlink(missing_ok=True)

    opts = _base_ydl_opts(work_dir, cookies_file, X_HEADERS)
    logger.info("yt-dlp X video url=%s", url)
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True) or {}
