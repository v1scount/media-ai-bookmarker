from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_user_ids(value: str) -> list[int]:
    text = (value or "").strip()
    if not text:
        return []
    # Allow JSON list form too: [123,456]
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="google/gemini-2.5-flash",
        alias="OPENROUTER_MODEL",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )

    # Inside the container this is /vault; on host use OBSIDIAN_VAULT_PATH in compose
    obsidian_vault_path: Path = Field(
        default=Path("/vault"),
        alias="OBSIDIAN_VAULT_PATH",
    )
    obsidian_relative_dir: str = Field(
        default="Extracts",
        alias="OBSIDIAN_RELATIVE_DIR",
    )
    # Where post photos are stored. Empty means "attachments" under the notes dir.
    obsidian_attachments_dir: str = Field(
        default="",
        alias="OBSIDIAN_ATTACHMENTS_DIR",
    )

    # X posts are mostly text and links, so the model earns its cost far less
    # often than on a TikTok video. Off by default: notes are built locally.
    x_use_llm: bool = Field(default=False, alias="X_USE_LLM")
    # Per-file ceiling for photos and videos copied into the vault
    max_attachment_mb: int = Field(
        default=100,
        alias="MAX_ATTACHMENT_MB",
        ge=1,
        le=2000,
    )

    # Stored as a plain string so empty / comma-separated env values work.
    # pydantic-settings JSON-decodes list[int] before validators can run.
    allowed_telegram_user_ids_raw: str = Field(
        default="",
        alias="ALLOWED_TELEGRAM_USER_IDS",
    )

    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    whisper_download_root: Path = Field(
        default=Path("/models"),
        alias="WHISPER_DOWNLOAD_ROOT",
    )
    job_tmp_dir: Path = Field(
        default=Path("/tmp/tiktok-jobs"),
        alias="JOB_TMP_DIR",
    )
    frame_count: int = Field(default=8, alias="FRAME_COUNT", ge=3, le=16)
    frame_max_width: int = Field(
        default=512,
        alias="FRAME_MAX_WIDTH",
        ge=256,
        le=1080,
    )
    # Hamming distance (out of 256 bits) below which two frames count as duplicates
    frame_dedupe_distance: int = Field(
        default=12,
        alias="FRAME_DEDUPE_DISTANCE",
        ge=0,
        le=64,
    )
    max_transcript_chars: int = Field(
        default=6000,
        alias="MAX_TRANSCRIPT_CHARS",
        ge=500,
    )
    max_description_chars: int = Field(
        default=1000,
        alias="MAX_DESCRIPTION_CHARS",
        ge=100,
    )
    max_audio_seconds: int = Field(
        default=420,
        alias="MAX_AUDIO_SECONDS",
        ge=30,
    )
    max_output_tokens: int = Field(
        default=1200,
        alias="MAX_OUTPUT_TOKENS",
        ge=256,
        le=8192,
    )
    result_cache_size: int = Field(
        default=32,
        alias="RESULT_CACHE_SIZE",
        ge=0,
        le=500,
    )
    job_timeout_seconds: int = Field(
        default=300,
        alias="JOB_TIMEOUT_SECONDS",
        ge=60,
        le=1800,
    )
    openrouter_timeout_seconds: float = Field(
        default=120.0,
        alias="OPENROUTER_TIMEOUT_SECONDS",
    )
    # Optional. Empty = Kagi search links only (unlimited browser searches).
    # Set to spend API credit resolving a few items to a direct page URL.
    kagi_api_key: str = Field(default="", alias="KAGI_API_KEY")
    kagi_search_per_job: int = Field(
        default=3,
        alias="KAGI_SEARCH_PER_JOB",
        ge=0,
        le=12,
    )
    kagi_timeout_seconds: float = Field(
        default=15.0,
        alias="KAGI_TIMEOUT_SECONDS",
        ge=1.0,
        le=60.0,
    )
    # Optional. Empty = skip Hardcover. Set to mark extracted books Want to Read on save.
    hardcover_api_key: str = Field(default="", alias="HARDCOVER_API_KEY")
    hardcover_books_per_job: int = Field(
        default=8,
        alias="HARDCOVER_BOOKS_PER_JOB",
        ge=0,
        le=20,
    )
    hardcover_timeout_seconds: float = Field(
        default=15.0,
        alias="HARDCOVER_TIMEOUT_SECONDS",
        ge=1.0,
        le=60.0,
    )
    ytdlp_cookies_file: Path | None = Field(
        default=None,
        alias="YTDLP_COOKIES_FILE",
    )
    # Host user/group so vault notes are visible to Obsidian (not root-owned)
    puid: int = Field(default=1000, alias="PUID")
    pgid: int = Field(default=1000, alias="PGID")

    @field_validator("kagi_api_key", "hardcover_api_key", mode="before")
    @classmethod
    def strip_optional_api_key(cls, value: object) -> object:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("ytdlp_cookies_file", mode="before")
    @classmethod
    def empty_cookies_path(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    @field_validator("obsidian_relative_dir", "obsidian_attachments_dir", mode="before")
    @classmethod
    def strip_relative_dir(cls, value: object) -> object:
        if isinstance(value, str):
            # Strip whitespace and accidental surrounding quotes from .env
            return value.strip().strip("\"'")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_telegram_user_ids(self) -> list[int]:
        return _parse_user_ids(self.allowed_telegram_user_ids_raw)

    @property
    def notes_dir(self) -> Path:
        return self.obsidian_vault_path / self.obsidian_relative_dir

    @property
    def attachments_relative_dir(self) -> str:
        """Vault-relative folder for saved photos, used to build embed links."""
        if self.obsidian_attachments_dir:
            return self.obsidian_attachments_dir.replace("\\", "/").strip("/")
        base = self.obsidian_relative_dir.replace("\\", "/").strip("/")
        return f"{base}/attachments" if base else "attachments"

    @property
    def attachments_dir(self) -> Path:
        return self.obsidian_vault_path / self.attachments_relative_dir

    @property
    def max_attachment_bytes(self) -> int:
        return self.max_attachment_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
