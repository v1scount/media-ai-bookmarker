from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.models import ExtractionResult
from app.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class DownloadArtifacts:
    work_dir: Path
    audio_path: Path | None
    video_path: Path | None
    title: str
    creator: str
    description: str
    duration: float | None
    video_id: str


class Pipeline:
    def __init__(self, settings: Settings, openrouter: OpenRouterClient) -> None:
        self.settings = settings
        self.openrouter = openrouter
        self._whisper_model = None
        self._cache: OrderedDict[str, ExtractionResult] = OrderedDict()
        self.model_supports_images = True

    def _cache_get(self, key: str) -> ExtractionResult | None:
        if self.settings.result_cache_size <= 0:
            return None
        result = self._cache.get(key)
        if result is not None:
            self._cache.move_to_end(key)
            return result.model_copy(deep=True)
        return None

    def _cache_put(self, key: str, result: ExtractionResult) -> None:
        if self.settings.result_cache_size <= 0:
            return
        self._cache[key] = result.model_copy(deep=True)
        self._cache.move_to_end(key)
        while len(self._cache) > self.settings.result_cache_size:
            self._cache.popitem(last=False)

    def _ensure_tmp_root(self) -> Path:
        root = self.settings.job_tmp_dir
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _load_whisper(self):
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper model=%s", self.settings.whisper_model)
            self.settings.whisper_download_root.mkdir(parents=True, exist_ok=True)
            self._whisper_model = WhisperModel(
                self.settings.whisper_model,
                device="cpu",
                compute_type="int8",
                download_root=str(self.settings.whisper_download_root),
            )
        return self._whisper_model

    async def run(
        self,
        url: str,
        *,
        progress_cb=None,
    ) -> ExtractionResult:
        """Full extract pipeline. Temp media is always deleted."""
        cached = self._cache_get(url)
        if cached is not None:
            logger.info("Cache hit for %s, skipping download and LLM call", url)
            if progress_cb:
                await progress_cb("Using cached extraction…")
            return cached

        work_dir = self._ensure_tmp_root() / f"job-{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            return await asyncio.wait_for(
                self._run_inner(url, work_dir, progress_cb=progress_cb),
                timeout=self.settings.job_timeout_seconds,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _run_inner(
        self,
        url: str,
        work_dir: Path,
        *,
        progress_cb=None,
    ) -> ExtractionResult:
        async def progress(msg: str) -> None:
            if progress_cb:
                await progress_cb(msg)

        await progress("Downloading video…")
        artifacts = await asyncio.to_thread(self._download, url, work_dir)

        if artifacts.video_id:
            cached = self._cache_get(artifacts.video_id)
            if cached is not None:
                logger.info(
                    "Cache hit for video id %s, skipping LLM call", artifacts.video_id
                )
                await progress("Using cached extraction…")
                self._cache_put(url, cached)
                return cached

        await progress("Transcribing audio…")
        transcript = ""
        if artifacts.audio_path and artifacts.audio_path.exists():
            transcript = await asyncio.to_thread(
                self._transcribe, artifacts.audio_path
            )

        frames: list[Path] = []
        if not self.model_supports_images:
            logger.info("Skipping frames: model has no image input")
        elif artifacts.video_path and artifacts.video_path.exists():
            await progress("Sampling frames…")
            frames = await asyncio.to_thread(
                self._extract_frames,
                artifacts.video_path,
                work_dir / "frames",
                artifacts.duration,
            )

        if not transcript and not frames:
            logger.warning(
                "No transcript and no frames for %s, skipping OpenRouter call", url
            )
            return ExtractionResult(
                source_url=url,
                title=artifacts.title,
                creator=artifacts.creator,
                summary=(
                    "No spoken audio or readable frames could be extracted from this "
                    "video, so nothing was sent for analysis."
                ),
                entities=[],
            )

        await progress("Extracting with OpenRouter…")
        result = await self.openrouter.extract(
            source_url=url,
            title=artifacts.title,
            creator=artifacts.creator,
            description=artifacts.description,
            transcript=transcript,
            frame_paths=frames,
        )
        # Title and creator always come from the video metadata, not the model
        result.source_url = url
        result.title = artifacts.title
        result.creator = artifacts.creator

        self._cache_put(url, result)
        if artifacts.video_id:
            self._cache_put(artifacts.video_id, result)
        return result

    def _download(self, url: str, work_dir: Path) -> DownloadArtifacts:
        from app.download import download_tiktok, pick_media_files

        cookies = self.settings.ytdlp_cookies_file
        info = download_tiktok(url, work_dir, cookies_file=cookies)

        title = (info or {}).get("title") or ""
        creator = (
            (info or {}).get("uploader")
            or (info or {}).get("creator")
            or (info or {}).get("channel")
            or ""
        )
        description = (info or {}).get("description") or ""
        duration = (info or {}).get("duration")
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None

        video_path, sidecar_audio = pick_media_files(info, work_dir)
        audio_path: Path | None = None

        logger.info(
            "Downloaded format=%s vcodec=%s acodec=%s file=%s sidecar_audio=%s",
            (info or {}).get("format_id"),
            (info or {}).get("vcodec"),
            (info or {}).get("acodec"),
            video_path.name if video_path else None,
            sidecar_audio.name if sidecar_audio else None,
        )

        # Whisper needs an audio source. Normally that is the video itself, but a
        # video-only download (or a merge that did not happen) leaves the audio in
        # a separate file, and photo-mode slideshows have no audio at all.
        audio_source: Path | None = None
        if video_path is not None and self._has_audio_stream(video_path):
            audio_source = video_path
        elif sidecar_audio is not None:
            logger.info(
                "No audio stream in %s, falling back to %s",
                video_path.name if video_path else "video",
                sidecar_audio.name,
            )
            audio_source = sidecar_audio
        else:
            logger.warning(
                "No audio track found for %s, continuing with frames only",
                video_path.name if video_path else url,
            )

        # Bounded so a long video cannot blow up transcription time or prompt size
        if audio_source is not None:
            audio_path = work_dir / "audio.wav"
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_source),
                    "-t",
                    str(self.settings.max_audio_seconds),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(audio_path),
                ],
                check=False,
            )
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                logger.warning("Audio extraction produced nothing for %s", audio_source)
                audio_path = None

        return DownloadArtifacts(
            work_dir=work_dir,
            audio_path=audio_path,
            video_path=video_path,
            title=title,
            creator=creator,
            description=description,
            duration=duration,
            video_id=str((info or {}).get("id") or ""),
        )

    def _transcribe(self, audio_path: Path) -> str:
        model = self._load_whisper()
        segments, _info = model.transcribe(
            str(audio_path),
            beam_size=1,
            vad_filter=True,
        )
        parts: list[str] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                parts.append(text)
        # Keep only in memory for OpenRouter; never write to disk/Telegram/Obsidian
        return " ".join(parts).strip()

    def _extract_frames(
        self,
        video_path: Path,
        frames_dir: Path,
        duration: float | None,
    ) -> list[Path]:
        from app.frames import dedupe_frames

        frames_dir.mkdir(parents=True, exist_ok=True)
        max_frames = self.settings.frame_count
        width = self.settings.frame_max_width

        # Short clips need fewer frames; sample roughly one candidate per 1.5s and
        # let dedupe pick the distinct ones. Candidates are local-only, so extra
        # candidates cost ffmpeg time but never image tokens.
        if duration and duration > 0:
            wanted = max(3, min(max_frames, round(duration / 3) or 3))
            candidates = min(max_frames * 2, max(wanted, round(duration / 1.5) or 3))
            fps_filter = f"fps={candidates / duration:.6f}"
        else:
            wanted = max_frames
            candidates = max_frames * 2
            fps_filter = "fps=1/2"

        pattern = str(frames_dir / "cand_%03d.jpg")
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"{fps_filter},scale='min({width},iw)':-2",
                "-frames:v",
                str(candidates),
                "-q:v",
                "6",
                pattern,
            ],
            check=False,
        )

        found = sorted(frames_dir.glob("cand_*.jpg"))
        found = [path for path in found if path.stat().st_size > 0]
        if not found:
            return []

        kept = dedupe_frames(
            found,
            max_frames=wanted,
            max_distance=self.settings.frame_dedupe_distance,
        )

        # Remove candidates we are not sending so nothing lingers in the job dir
        for path in found:
            if path not in kept:
                path.unlink(missing_ok=True)

        logger.info(
            "Frames: %s candidates -> %s sent (width<=%s)",
            len(found),
            len(kept),
            width,
        )
        return kept

    @staticmethod
    def _has_audio_stream(video_path: Path) -> bool:
        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            logger.warning("ffprobe unavailable (%s), assuming audio present", exc)
            return True
        return bool(result.stdout.strip())

    @staticmethod
    def _ffmpeg_error_summary(stderr: str) -> str:
        """Pull the useful lines out of ffmpeg's banner-heavy output."""
        interesting = [
            line.strip()
            for line in stderr.splitlines()
            if any(
                marker in line
                for marker in (
                    "Error",
                    "error",
                    "Invalid",
                    "does not contain",
                    "No such file",
                    "Unable to",
                )
            )
        ]
        if interesting:
            return " | ".join(interesting[-3:])[:300]
        return stderr.strip().splitlines()[-1][:300] if stderr.strip() else "no output"

    @classmethod
    def _run_ffmpeg(cls, cmd: list[str], *, check: bool = True) -> None:
        import subprocess

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return

        summary = cls._ffmpeg_error_summary(result.stderr)
        if check:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}): {summary}")
        logger.warning("ffmpeg soft-fail (%s): %s", result.returncode, summary)


def format_preview(result: ExtractionResult) -> str:
    from app.models import Confidence

    lines = [
        f"<b>{_html(result.title or 'TikTok extract')}</b>",
        f"by {_html(result.creator or 'unknown')}",
        "",
        _html(result.summary or "(no summary)"),
        "",
        "<b>Items</b>",
    ]
    entities = result.ordered_entities()
    if not entities:
        lines.append("<i>Nothing worth looking up was found.</i>")
    else:
        for ent in entities:
            author = (
                f" ({_html(ent.creator_or_author)})" if ent.creator_or_author else ""
            )
            star = "* " if ent.is_main_topic else ""
            if ent.suggested_link:
                link = f' — <a href="{_html(ent.suggested_link)}">link</a>'
            else:
                link = f' — <a href="{_html(ent.search_url)}">search</a>'
            uncertain = " <i>(uncertain)</i>" if ent.confidence == Confidence.low else ""
            notes = f"\n  <i>{_html(ent.notes)}</i>" if ent.notes else ""
            lines.append(
                f"• {star}<b>{_html(ent.type.value)}</b> — {_html(ent.name)}"
                f"{author}{link}{uncertain}{notes}"
            )
    lines.extend(["", f'<a href="{_html(result.source_url)}">Source</a>'])
    return "\n".join(lines)


def _html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
