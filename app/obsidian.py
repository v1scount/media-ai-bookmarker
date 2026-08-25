from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.models import (
    DEFAULT_AMAZON_SEARCH_HOST,
    Confidence,
    Entity,
    EntityType,
    ExtractionResult,
    MediaKind,
    SourceKind,
    VideoKind,
    split_caption,
)

logger = logging.getLogger(__name__)

SOURCE_LABELS: dict[SourceKind, str] = {
    SourceKind.tiktok: "TikTok extract",
    SourceKind.x: "X extract",
}

WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*]')
FILENAME_MAX_LEN = 80


# A plain YAML scalar may not start with an indicator character. An X handle
# begins with "@", which is reserved, and would break the whole frontmatter block.
YAML_UNSAFE_FIRST = set("@`-?:,[]{}#&*!|>'\"%")


def _yaml_value(value: str) -> str:
    """Quote a frontmatter value only when a plain scalar would be invalid."""
    text = str(value or "").strip()
    if not text:
        return '""'
    needs_quotes = (
        text[0] in YAML_UNSAFE_FIRST
        or ": " in text
        or text.endswith(":")
        or " #" in text
    )
    if not needs_quotes:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _safe_filename(text: str, fallback: str, max_len: int = FILENAME_MAX_LEN) -> str:
    cleaned = WINDOWS_FORBIDDEN.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned or not re.search(r"[\w]", cleaned, flags=re.UNICODE):
        cleaned = fallback
    return cleaned[:max_len].rstrip(" .")


def _render_entity_links(entity: Entity, amazon_host: str) -> list[str]:
    lines: list[str] = []
    if entity.suggested_link:
        lines.append(f"- [link]({entity.suggested_link})")
    if entity.type == EntityType.product:
        for label, url in entity.shop_links(amazon_host):
            lines.append(f"- [{label}]({url})")
    elif not entity.suggested_link:
        lines.append(f"- [search]({entity.search_url})")
    if entity.hardcover_url:
        lines.append(f"- [hardcover]({entity.hardcover_url})")
    return lines


def _render_entity(entity: Entity, amazon_host: str) -> list[str]:
    type_label = entity.type.value.capitalize()
    uncertain = " _(uncertain)_" if entity.confidence == Confidence.low else ""
    lines = [f"### {type_label} — {entity.name}{uncertain}", ""]
    if entity.creator_or_author:
        lines.extend([f"*{entity.creator_or_author}*", ""])
    if entity.notes:
        lines.extend([entity.notes, ""])
    link_lines = _render_entity_links(entity, amazon_host)
    if link_lines:
        lines.extend(link_lines)
        lines.append("")
    return lines


def _frontmatter_tags(result: ExtractionResult) -> str:
    tags = [result.source_kind.value, "extract", *result.entity_type_tags()]
    return f"tags: [{', '.join(tags)}]"


def _frontmatter(result: ExtractionResult, extra: list[str]) -> list[str]:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kind = result.source_kind
    return [
        "---",
        f"source: {_yaml_value(result.source_url)}",
        f"source_kind: {kind.value}",
        f"creator: {_yaml_value(result.creator)}",
        f"created: {created}",
        *extra,
        _frontmatter_tags(result),
        "---",
    ]


def _render_caption_callout(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    prose, tags = split_caption(text)
    body = prose or text.strip()
    lines = ["> [!quote] Original caption"]
    for line in body.splitlines() or [body]:
        lines.append(f"> {line}" if line else ">")
    if tags and prose:
        lines.append(">")
        lines.append(f"> {tags}")
    lines.append("")
    return lines


def render_post_markdown(result: ExtractionResult) -> str:
    """A note that is just the post: its text, its links, and its media."""
    post = result.post
    assert post is not None

    title = result.title or SOURCE_LABELS.get(result.source_kind, "Extract")
    lines = _frontmatter(result, [])
    lines.extend(["", f"# {title}", ""])

    if post.text:
        lines.extend([post.text, ""])

    if post.links:
        lines.extend(["## Links", ""])
        for link in post.links:
            lines.append(f"- [{link.title or link.url}]({link.url})")
            if link.description:
                lines.append(f"  - {link.description}")
        lines.append("")

    if post.media:
        lines.extend(["## Media", ""])
        unsaved_videos = 0
        for item in post.media:
            # Wikilinks handle spaces in vault paths, and Obsidian renders an
            # inline player for embedded mp4/webm just as it does for images
            if item.vault_path:
                lines.extend([f"![[{item.vault_path}]]", ""])
            elif item.kind == MediaKind.image and item.url:
                lines.extend([f"![]({item.url})", ""])
            else:
                # Only reached when a download failed; keep the post reachable
                unsaved_videos += 1
                label = "Video" if len(post.videos) == 1 else f"Video {unsaved_videos}"
                lines.extend([f"- [{label}]({item.url or result.source_url})", ""])

    lines.extend([f"[Source]({result.source_url})", ""])
    return "\n".join(lines)


def render_markdown(
    result: ExtractionResult,
    amazon_host: str = DEFAULT_AMAZON_SEARCH_HOST,
) -> str:
    if result.post is not None:
        return render_post_markdown(result)

    source_kind = result.source_kind
    title = result.title or SOURCE_LABELS.get(source_kind, "Extract")
    lines = _frontmatter(result, [f"kind: {result.video_kind.value}"])
    lines.extend(["", f"# {title}", ""])
    lines.extend(_render_caption_callout(result.source_description))
    if result.summary:
        lines.extend([result.summary, ""])

    entities = result.ordered_entities()
    if not entities:
        lines.extend(["## Items", "", "_Nothing worth looking up was found._", ""])
    elif result.video_kind == VideoKind.single:
        main, *rest = entities
        lines.extend(["## Recommendation", ""])
        lines.extend(_render_entity(main, amazon_host))
        if rest:
            lines.extend(["## Also mentioned", ""])
            for entity in rest:
                lines.extend(_render_entity(entity, amazon_host))
    else:
        lines.extend(["## Items", ""])
        for entity in entities:
            lines.extend(_render_entity(entity, amazon_host))

    source_label = "TikTok" if source_kind == SourceKind.tiktok else "X"
    lines.extend(["## Source", "", f"[{source_label}]({result.source_url})", ""])
    return "\n".join(lines)


def build_note_filename(result: ExtractionResult) -> str:
    base = result.title or (result.entities[0].name if result.entities else "")
    fallback = SOURCE_LABELS.get(result.source_kind, "Extract")
    return f"{_safe_filename(base, fallback)}.md"


def _apply_host_ownership(settings: Settings, path: Path) -> None:
    """Make Docker-created notes owned/readable by the Obsidian host user."""
    uid = settings.puid
    gid = settings.pgid
    if uid < 0 or gid < 0:
        return
    try:
        os.chown(path, uid, gid)
        if path.is_dir():
            os.chmod(path, 0o775)
        else:
            os.chmod(path, 0o664)
    except OSError as exc:
        logger.warning("Could not chown/chmod %s: %s", path, exc)


def _nudge_filesystem_watchers(paths: list[Path]) -> None:
    """Bump mtime so Obsidian / sync tools notice externally created files."""
    for path in paths:
        try:
            if path.exists():
                os.utime(path, None)
        except OSError:
            pass


def _save_attachments(
    settings: Settings,
    result: ExtractionResult,
    note_stem: str,
) -> list[Path]:
    """Pull post photos into the vault so the note keeps working offline.

    Downloading here rather than during the pipeline means a dismissed preview
    never leaves files behind, and nothing depends on the temp job directory.
    """
    post = result.post
    if post is None:
        return []
    pending = [item for item in post.media if item.url and not item.vault_path]
    if not pending:
        return []

    from app.xfetch import download_media, media_suffix

    attachments_dir = settings.attachments_dir
    attachments_dir.mkdir(parents=True, exist_ok=True)
    _apply_host_ownership(settings, attachments_dir)

    saved: list[Path] = []
    for index, item in enumerate(pending, start=1):
        is_video = item.kind == MediaKind.video
        name = f"{note_stem}-{index}{media_suffix(item.url, video=is_video)}"
        dest = attachments_dir / name
        if not download_media(
            item.url,
            dest,
            max_bytes=settings.max_attachment_bytes,
        ):
            continue
        _apply_host_ownership(settings, dest)
        item.vault_path = f"{settings.attachments_relative_dir}/{name}"
        saved.append(dest)

    logger.info(
        "Saved %s/%s attachments to %s", len(saved), len(pending), attachments_dir
    )
    return saved


def save_to_obsidian(settings: Settings, result: ExtractionResult) -> Path:
    """Atomically write a markdown note into the configured vault folder."""
    notes_dir = settings.notes_dir
    notes_dir.mkdir(parents=True, exist_ok=True)
    _apply_host_ownership(settings, notes_dir)

    filename = build_note_filename(result)
    target = notes_dir / filename

    # Avoid clobbering existing notes
    if target.exists():
        stem = target.stem
        idx = 2
        while True:
            candidate = notes_dir / f"{stem}-{idx}.md"
            if not candidate.exists():
                target = candidate
                break
            idx += 1

    saved_images = _save_attachments(settings, result, target.stem)
    content = render_markdown(result, amazon_host=settings.amazon_search_host)
    tmp = target.with_suffix(".md.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)

    _apply_host_ownership(settings, target)
    _nudge_filesystem_watchers(
        [target, notes_dir, settings.obsidian_vault_path, *saved_images]
    )

    logger.info(
        "Wrote Obsidian note: %s (uid=%s gid=%s)",
        target,
        settings.puid,
        settings.pgid,
    )
    return target


def relative_vault_path(settings: Settings, path: Path) -> str:
    try:
        return str(path.relative_to(settings.obsidian_vault_path))
    except ValueError:
        return str(path)
