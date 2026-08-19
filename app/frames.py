"""Frame selection helpers.

Candidate frames are cheap (local ffmpeg), but every frame we send to
OpenRouter costs image tokens. These helpers drop near-duplicate frames so a
static talking-head video does not pay for eight copies of the same picture.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HASH_SIDE = 16
HASH_BITS = HASH_SIDE * HASH_SIDE


def average_hash(path: Path) -> int | None:
    """Perceptual average hash, or None when Pillow is unavailable."""
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as image:
            small = image.convert("L").resize(
                (HASH_SIDE, HASH_SIDE), Image.Resampling.BILINEAR
            )
            pixels = list(small.getdata())
    except Exception as exc:
        logger.warning("Could not hash frame %s: %s", path, exc)
        return None

    if not pixels:
        return None

    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value > mean:
            bits |= 1 << index
    return bits


def hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def _even_subsample(paths: list[Path], max_frames: int) -> list[Path]:
    if len(paths) <= max_frames:
        return list(paths)
    step = len(paths) / max_frames
    return [paths[min(len(paths) - 1, int(i * step))] for i in range(max_frames)]


def dedupe_frames(
    paths: list[Path],
    *,
    max_frames: int,
    max_distance: int,
) -> list[Path]:
    """Keep at most max_frames visually distinct frames, preserving order."""
    if max_frames <= 0 or not paths:
        return []

    kept: list[Path] = []
    kept_hashes: list[int] = []
    hashing_available = True

    for path in paths:
        if len(kept) >= max_frames:
            break

        digest = average_hash(path)
        if digest is None:
            hashing_available = False
            break

        if any(
            hamming_distance(digest, existing) <= max_distance
            for existing in kept_hashes
        ):
            continue

        kept.append(path)
        kept_hashes.append(digest)

    if not hashing_available:
        logger.info("Perceptual hashing unavailable, falling back to even sampling")
        return _even_subsample(paths, max_frames)

    dropped = len(paths) - len(kept)
    if dropped > 0:
        logger.info(
            "Frame dedupe kept %s of %s candidates (%s near-duplicates dropped)",
            len(kept),
            len(paths),
            dropped,
        )
    return kept
