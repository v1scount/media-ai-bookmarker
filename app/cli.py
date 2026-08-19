"""CLI harness to test the extraction pipeline without Telegram.

Usage:
  python -m app.cli "https://www.tiktok.com/@user/video/123"
  python -m app.cli "https://vm.tiktok.com/xxx" --save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from app.config import get_settings
from app.models import extract_tiktok_url
from app.obsidian import relative_vault_path, save_to_obsidian
from app.openrouter import OpenRouterClient
from app.pipeline import Pipeline, format_preview

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)


async def _run(url: str, save: bool) -> int:
    settings = get_settings()
    openrouter = OpenRouterClient(settings)
    pipeline = Pipeline(settings, openrouter)
    try:
        pipeline.model_supports_images = await openrouter.verify_model()

        async def progress(msg: str) -> None:
            print(f"[progress] {msg}", flush=True)

        result = await pipeline.run(url, progress_cb=progress)
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print("\n--- Telegram-style preview ---\n")
        # Strip HTML tags roughly for console
        preview = format_preview(result)
        print(preview.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "`").replace("</code>", "`"))
        if save:
            path = save_to_obsidian(settings, result)
            print(f"\nSaved: {relative_vault_path(settings, path)}")
        return 0
    finally:
        await openrouter.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TikTok extraction pipeline")
    parser.add_argument("url", help="TikTok URL")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write Markdown into the Obsidian vault folder",
    )
    args = parser.parse_args(argv)

    url = extract_tiktok_url(args.url) or args.url
    if "tiktok.com" not in url.lower():
        print("Not a TikTok URL", file=sys.stderr)
        return 2

    return asyncio.run(_run(url, args.save))


if __name__ == "__main__":
    raise SystemExit(main())
