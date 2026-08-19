from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Optional
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import get_settings
from app.models import ExtractionResult, extract_tiktok_url
from app.obsidian import relative_vault_path, save_to_obsidian
from app.openrouter import OpenRouterClient
from app.pipeline import Pipeline, format_preview

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# In-memory pending results keyed by short id (for inline button callbacks)
PENDING_RESULTS: dict[str, ExtractionResult] = {}
PENDING_LOCK = asyncio.Lock()
MAX_PENDING = 50


def _is_allowed(user_id: Optional[int], allowed: list[int]) -> bool:
    if not allowed:
        # Empty allowlist = deny all (safer for a home bot)
        return False
    return user_id is not None and user_id in allowed


async def _store_result(result: ExtractionResult) -> str:
    result_id = uuid4().hex[:12]
    async with PENDING_LOCK:
        PENDING_RESULTS[result_id] = result
        # Evict oldest-ish if too many (dict preserves insertion order on 3.7+)
        while len(PENDING_RESULTS) > MAX_PENDING:
            oldest = next(iter(PENDING_RESULTS))
            PENDING_RESULTS.pop(oldest, None)
    return result_id


async def _pop_result(result_id: str) -> Optional[ExtractionResult]:
    async with PENDING_LOCK:
        return PENDING_RESULTS.pop(result_id, None)


async def _get_result(result_id: str) -> Optional[ExtractionResult]:
    async with PENDING_LOCK:
        return PENDING_RESULTS.get(result_id)


def _save_keyboard(result_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Save to Obsidian",
                    callback_data=f"save:{result_id}",
                ),
                InlineKeyboardButton(
                    "Dismiss",
                    callback_data=f"dismiss:{result_id}",
                ),
            ]
        ]
    )


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    user = update.effective_user
    if not _is_allowed(user.id if user else None, settings.allowed_telegram_user_ids):
        if update.message:
            await update.message.reply_text("You are not authorized to use this bot.")
        return
    await update.message.reply_text(
        "Send me a TikTok link. I'll extract tools, books, movies, and music "
        "recommendations, then let you save a Markdown note to Obsidian."
    )


async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Helper to discover your Telegram user id for the allowlist."""
    user = update.effective_user
    if update.message and user:
        await update.message.reply_text(f"Your Telegram user id: `{user.id}`", parse_mode=ParseMode.MARKDOWN)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    pipeline: Pipeline = context.application.bot_data["pipeline"]
    user = update.effective_user
    message = update.message
    if not message or not message.text:
        return

    if not _is_allowed(user.id if user else None, settings.allowed_telegram_user_ids):
        await message.reply_text("You are not authorized to use this bot.")
        return

    url = extract_tiktok_url(message.text)
    if not url:
        await message.reply_text(
            "Send a TikTok URL (tiktok.com or vm.tiktok.com)."
        )
        return

    job_lock: asyncio.Lock = context.application.bot_data["job_lock"]
    if job_lock.locked():
        await message.reply_text("Busy with another video — try again in a moment.")
        return

    status = await message.reply_text("Queued…")
    last_edit = ""

    async def progress_cb(msg: str) -> None:
        nonlocal last_edit
        if msg == last_edit:
            return
        last_edit = msg
        try:
            await status.edit_text(msg)
            await context.bot.send_chat_action(
                chat_id=message.chat_id,
                action=ChatAction.TYPING,
            )
        except Exception:
            pass

    async with job_lock:
        try:
            result = await pipeline.run(url, progress_cb=progress_cb)
        except asyncio.TimeoutError:
            await status.edit_text(
                "Timed out while processing this video. Try again or pick a shorter clip."
            )
            return
        except Exception as exc:
            logger.exception("Pipeline failed for %s", url)
            err = str(exc)[:300] or type(exc).__name__
            await status.edit_text(f"Failed to process video:\n{err}")
            return

    result_id = await _store_result(result)
    preview = format_preview(result)
    # Telegram message limit ~4096; trim if needed
    if len(preview) > 3900:
        preview = preview[:3900] + "\n…"

    try:
        await status.edit_text(
            preview,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=_save_keyboard(result_id),
        )
    except Exception:
        # Fallback without HTML if Telegram rejects entities
        await status.edit_text(
            f"{result.title or 'Extract'}\n\n{result.summary}\n\n"
            + "\n".join(
                f"- [{e.type.value}] {e.name}" for e in result.entities
            )
            + f"\n\n{result.source_url}",
            reply_markup=_save_keyboard(result_id),
            disable_web_page_preview=True,
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    query = update.callback_query
    if not query:
        return

    user = update.effective_user
    if not _is_allowed(user.id if user else None, settings.allowed_telegram_user_ids):
        await query.answer("Not authorized.", show_alert=True)
        return

    data = query.data or ""

    if data.startswith("dismiss:"):
        await query.answer()
        result_id = data.split(":", 1)[1]
        await _pop_result(result_id)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        if query.message:
            await query.message.reply_text("Dismissed. Nothing saved.")
        return

    if data.startswith("save:"):
        result_id = data.split(":", 1)[1]
        result = await _get_result(result_id)
        if result is None:
            await query.answer("Result expired. Send the link again.", show_alert=True)
            return
        await query.answer()
        try:
            path = await asyncio.to_thread(save_to_obsidian, settings, result)
            rel = relative_vault_path(settings, path)
            await _pop_result(result_id)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(
                    f"Saved to Obsidian:\n`{rel}`",
                    parse_mode=ParseMode.MARKDOWN,
                )
        except Exception as exc:
            logger.exception("Failed to save Obsidian note")
            if query.message:
                await query.message.reply_text(f"Could not save note: {exc}")
        return

    await query.answer()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error: %s\n%s", context.error, traceback.format_exc())


async def post_init(application: Application) -> None:
    settings = get_settings()
    openrouter = OpenRouterClient(settings)
    pipeline = Pipeline(settings, openrouter)
    application.bot_data["settings"] = settings
    application.bot_data["openrouter"] = openrouter
    application.bot_data["pipeline"] = pipeline
    application.bot_data["job_lock"] = asyncio.Lock()
    # Checked once so we never upload frames to a text-only model
    pipeline.model_supports_images = await openrouter.verify_model()
    try:
        import yt_dlp

        ytdlp_ver = getattr(yt_dlp, "version", None)
        ytdlp_ver = getattr(ytdlp_ver, "__version__", None) or "unknown"
    except Exception:
        ytdlp_ver = "unknown"
    logger.info(
        "Bot ready. Allowlist=%s model=%s notes_dir=%s yt-dlp=%s",
        settings.allowed_telegram_user_ids,
        settings.openrouter_model,
        settings.notes_dir,
        ytdlp_ver,
    )


async def post_shutdown(application: Application) -> None:
    openrouter: OpenRouterClient | None = application.bot_data.get("openrouter")
    if openrouter:
        await openrouter.aclose()


def main() -> None:
    settings = get_settings()
    if not settings.allowed_telegram_user_ids:
        logger.warning(
            "ALLOWED_TELEGRAM_USER_IDS is empty — all users will be denied. "
            "Message the bot with /whoami to get your id, then set the env var and restart."
        )

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_error_handler(error_handler)

    logger.info("Starting Telegram polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
