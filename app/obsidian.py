from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.models import Confidence, Entity, ExtractionResult, VideoKind

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return (text or "tiktok-extract")[:max_len]


def _render_entity_bullet(entity: Entity) -> list[str]:
    type_label = entity.type.value.capitalize()
    author = f" ({entity.creator_or_author})" if entity.creator_or_author else ""
    if entity.suggested_link:
        link = f" — [link]({entity.suggested_link})"
    else:
        link = f" — [search]({entity.search_url})"
    uncertain = " _(uncertain)_" if entity.confidence == Confidence.low else ""

    lines = [f"- **{type_label}** — *{entity.name}*{author}{link}{uncertain}"]
    if entity.notes:
        lines.append(f"  - {entity.notes}")
    return lines


def render_markdown(result: ExtractionResult) -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    title = result.title or "TikTok extract"
    lines = [
        "---",
        f"source: {result.source_url}",
        f"creator: {result.creator}",
        f"created: {created}",
        f"kind: {result.video_kind.value}",
        "tags: [tiktok, extract]",
        "---",
        "",
        f"# {title}",
        "",
        result.summary or "",
        "",
    ]

    entities = result.ordered_entities()
    if not entities:
        lines.extend(["## Items", "", "_Nothing worth looking up was found._", ""])
    elif result.video_kind == VideoKind.single:
        # Single-topic video: lead with the one thing, keep the rest brief
        main, *rest = entities
        lines.extend(["## Recommendation", ""])
        lines.extend(_render_entity_bullet(main))
        lines.append("")
        if rest:
            lines.extend(["## Also mentioned", ""])
            for entity in rest:
                lines.extend(_render_entity_bullet(entity))
            lines.append("")
    else:
        lines.extend(["## Items", ""])
        for entity in entities:
            lines.extend(_render_entity_bullet(entity))
        lines.append("")

    lines.append(f"[Source]({result.source_url})")
    lines.append("")
    return "\n".join(lines)


def build_note_filename(result: ExtractionResult) -> str:
    base = result.title or (result.entities[0].name if result.entities else "extract")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{day}-{_slugify(base)}.md"


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

    content = render_markdown(result)
    tmp = target.with_suffix(".md.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)

    _apply_host_ownership(settings, target)
    _nudge_filesystem_watchers(
        [target, notes_dir, settings.obsidian_vault_path]
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
