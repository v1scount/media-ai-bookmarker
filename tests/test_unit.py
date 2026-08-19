"""Lightweight unit tests (no network / Whisper / ffmpeg)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings, _parse_user_ids
from app.download import _usable_cookies, pick_media_files
from app.frames import dedupe_frames, hamming_distance
from app.models import (
    Confidence,
    Entity,
    EntityType,
    ExtractionResult,
    VideoKind,
    extract_tiktok_url,
)
from app.obsidian import render_markdown, save_to_obsidian
from app.openrouter import RESULT_JSON_SCHEMA, truncate_text
from app.pipeline import Pipeline


class CookiesHelperTests(unittest.TestCase):
    def test_placeholder_cookies_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cookies.txt"
            path.write_text("# comment only\n", encoding="utf-8")
            self.assertIsNone(_usable_cookies(path))

    def test_real_cookie_line_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cookies.txt"
            path.write_text(
                ".tiktok.com\tTRUE\t/\tFALSE\t0\tsessionid\tabc\n",
                encoding="utf-8",
            )
            self.assertEqual(_usable_cookies(path), path)


class AllowedIdsTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(_parse_user_ids(""), [])
        self.assertEqual(_parse_user_ids("   "), [])

    def test_comma_separated(self) -> None:
        self.assertEqual(_parse_user_ids("123,456"), [123, 456])

    def test_settings_empty_env(self) -> None:
        settings = Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
            ALLOWED_TELEGRAM_USER_IDS="",
        )
        self.assertEqual(settings.allowed_telegram_user_ids, [])

    def test_settings_single_id(self) -> None:
        settings = Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
            ALLOWED_TELEGRAM_USER_IDS="42",
        )
        self.assertEqual(settings.allowed_telegram_user_ids, [42])


class UrlParseTests(unittest.TestCase):
    def test_standard_url(self) -> None:
        text = "check https://www.tiktok.com/@user/video/1234567890 please"
        self.assertEqual(
            extract_tiktok_url(text),
            "https://www.tiktok.com/@user/video/1234567890",
        )

    def test_short_vm(self) -> None:
        self.assertEqual(
            extract_tiktok_url("https://vm.tiktok.com/ZMabcdef/"),
            "https://vm.tiktok.com/ZMabcdef/",
        )

    def test_strips_trailing_punct(self) -> None:
        self.assertEqual(
            extract_tiktok_url("(https://www.tiktok.com/@u/video/1)."),
            "https://www.tiktok.com/@u/video/1",
        )

    def test_none(self) -> None:
        self.assertIsNone(extract_tiktok_url("no link here"))


class ExtractionSchemaTests(unittest.TestCase):
    def test_null_optional_strings_coerced(self) -> None:
        result = ExtractionResult.model_validate(
            {
                "source_url": "https://www.tiktok.com/@u/video/1",
                "title": None,
                "creator": None,
                "summary": None,
                "entities": [
                    {
                        "type": "tool",
                        "name": "Cursor",
                        "creator_or_author": None,
                        "notes": None,
                        "is_main_topic": None,
                        "confidence": None,
                        "suggested_link": None,
                    },
                    {
                        "type": "weird-type",
                        "name": "Something",
                        "suggested_link": "",
                    },
                ],
            }
        )
        self.assertEqual(result.title, "")
        self.assertEqual(result.entities[0].creator_or_author, "")
        self.assertEqual(result.entities[0].notes, "")
        self.assertFalse(result.entities[0].is_main_topic)
        self.assertEqual(result.entities[0].confidence, Confidence.medium)
        self.assertEqual(result.entities[1].type, EntityType.other)
        self.assertIsNone(result.entities[1].suggested_link)
        self.assertEqual(result.video_kind, VideoKind.other)

    def test_type_aliases_and_removed_track(self) -> None:
        cases = {
            "app": EntityType.tool,
            "song": EntityType.album,
            "track": EntityType.album,
            "youtube channel": EntityType.video,
            "paper": EntityType.article,
            "TV Show": EntityType.series,
        }
        for raw, expected in cases.items():
            entity = Entity.model_validate({"type": raw, "name": "x"})
            self.assertEqual(entity.type, expected, raw)

    def test_guessed_links_rejected(self) -> None:
        entity = Entity.model_validate(
            {"name": "Thing", "suggested_link": "probably example.com"}
        )
        self.assertIsNone(entity.suggested_link)

    def test_search_url_built_locally(self) -> None:
        entity = Entity.model_validate(
            {"type": "book", "name": "Dune", "creator_or_author": "Frank Herbert"}
        )
        self.assertEqual(
            entity.search_url,
            "https://www.google.com/search?q=Dune+Frank+Herbert+book",
        )

    def test_main_topic_sorted_first(self) -> None:
        result = ExtractionResult.model_validate(
            {
                "source_url": "u",
                "entities": [
                    {"name": "side", "is_main_topic": False},
                    {"name": "main", "is_main_topic": True},
                ],
            }
        )
        self.assertEqual(
            [entity.name for entity in result.ordered_entities()], ["main", "side"]
        )


class TokenGuardTests(unittest.TestCase):
    def test_truncate_leaves_short_text(self) -> None:
        self.assertEqual(truncate_text("hello", 100, "transcript"), "hello")

    def test_truncate_marks_long_text(self) -> None:
        result = truncate_text("x" * 50, 10, "transcript")
        self.assertTrue(result.startswith("x" * 10))
        self.assertIn("[transcript truncated]", result)

    def test_schema_forbids_extra_keys(self) -> None:
        schema = RESULT_JSON_SCHEMA["schema"]
        self.assertTrue(RESULT_JSON_SCHEMA["strict"])
        self.assertFalse(schema["additionalProperties"])
        # Metadata we already know locally is never requested from the model
        for field in ("source_url", "title", "creator"):
            self.assertNotIn(field, schema["properties"])

    def test_every_schema_field_is_documented(self) -> None:
        entity_props = RESULT_JSON_SCHEMA["schema"]["properties"]["entities"]["items"][
            "properties"
        ]
        for name, spec in entity_props.items():
            self.assertIn("description", spec, name)
        self.assertNotIn("search_query", entity_props)

    def test_hamming_distance(self) -> None:
        self.assertEqual(hamming_distance(0b1011, 0b1001), 1)
        self.assertEqual(hamming_distance(0b1111, 0b1111), 0)

    def test_dedupe_falls_back_without_hashing(self) -> None:
        # Non-image files cannot be hashed, so this exercises the fallback path
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for idx in range(6):
                path = Path(tmp) / f"cand_{idx}.jpg"
                path.write_bytes(b"not-an-image")
                paths.append(path)
            kept = dedupe_frames(paths, max_frames=3, max_distance=12)
            self.assertEqual(len(kept), 3)
            self.assertEqual(kept[0], paths[0])

    def test_dedupe_respects_zero_max(self) -> None:
        self.assertEqual(dedupe_frames([Path("a.jpg")], max_frames=0, max_distance=5), [])


class MediaPickerTests(unittest.TestCase):
    def test_prefers_path_reported_by_ytdlp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "media.f5.mp4").write_bytes(b"video-only")
            merged = work / "media.mp4"
            merged.write_bytes(b"merged")
            info = {"requested_downloads": [{"filepath": str(merged)}]}
            video, _audio = pick_media_files(info, work)
            self.assertEqual(video, merged)

    def test_glob_fallback_skips_per_format_intermediates(self) -> None:
        # "media.f5.mp4" sorts before "media.mp4" alphabetically, and is video-only
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "media.f5.mp4").write_bytes(b"video-only")
            merged = work / "media.mp4"
            merged.write_bytes(b"merged")
            video, _audio = pick_media_files(None, work)
            self.assertEqual(video, merged)

    def test_finds_sidecar_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "media.mp4").write_bytes(b"video")
            audio_file = work / "media.f2.m4a"
            audio_file.write_bytes(b"audio")
            video, audio = pick_media_files(None, work)
            self.assertEqual(video, work / "media.mp4")
            self.assertEqual(audio, audio_file)

    def test_ignores_our_own_wav_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "media.mp4").write_bytes(b"video")
            (work / "audio.wav").write_bytes(b"our own extraction")
            _video, audio = pick_media_files(None, work)
            self.assertIsNone(audio)


class FfmpegErrorTests(unittest.TestCase):
    def test_summary_picks_error_lines_over_banner(self) -> None:
        stderr = (
            "ffmpeg version 6.1\n"
            "  Stream #0:0[0x2](und): Video: hevc (Main), 720x1280, 30 fps\n"
            "Output file #0 does not contain any stream\n"
        )
        summary = Pipeline._ffmpeg_error_summary(stderr)
        self.assertIn("does not contain any stream", summary)
        self.assertNotIn("ffmpeg version", summary)

    def test_summary_handles_empty_output(self) -> None:
        self.assertEqual(Pipeline._ffmpeg_error_summary(""), "no output")


class ObsidianTests(unittest.TestCase):
    def test_single_topic_note_leads_with_recommendation(self) -> None:
        result = ExtractionResult(
            source_url="https://www.tiktok.com/@u/video/1",
            title="Original TikTok caption",
            creator="creator1",
            summary="Recommends a novel.",
            video_kind=VideoKind.single,
            entities=[
                Entity(
                    type=EntityType.book,
                    name="Dune",
                    creator_or_author="Frank Herbert",
                    notes="Main pick",
                    is_main_topic=True,
                    confidence=Confidence.high,
                ),
                Entity(type=EntityType.tool, name="Kindle"),
            ],
        )
        md = render_markdown(result)
        # The TikTok caption is preserved verbatim as the note title
        self.assertIn("# Original TikTok caption", md)
        self.assertIn("kind: single", md)
        self.assertIn("## Recommendation", md)
        self.assertIn("## Also mentioned", md)
        self.assertIn("[search](https://www.google.com/search?q=Dune+", md)

    def test_low_confidence_marked(self) -> None:
        result = ExtractionResult(
            source_url="u",
            video_kind=VideoKind.list,
            entities=[
                Entity(name="Maybe This", confidence=Confidence.low),
                Entity(name="Solid", confidence=Confidence.high),
            ],
        )
        md = render_markdown(result)
        self.assertIn("## Items", md)
        self.assertIn("_(uncertain)_", md)
        self.assertEqual(md.count("_(uncertain)_"), 1)

    def test_render_and_save(self) -> None:
        result = ExtractionResult(
            source_url="https://www.tiktok.com/@u/video/1",
            title="Cool Book Rec",
            creator="creator1",
            summary="Recommends a novel.",
            entities=[
                Entity(
                    type=EntityType.book,
                    name="Dune",
                    creator_or_author="Frank Herbert",
                    notes="Main pick",
                    suggested_link=None,
                )
            ],
        )
        md = render_markdown(result)
        self.assertIn("Dune", md)
        self.assertIn("tags: [tiktok, extract]", md)
        self.assertNotIn("transcript", md.lower())

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                TELEGRAM_BOT_TOKEN="x",
                OPENROUTER_API_KEY="x",
                OBSIDIAN_VAULT_PATH=tmp,
                OBSIDIAN_RELATIVE_DIR="TikTok Extracts",
                ALLOWED_TELEGRAM_USER_IDS="1",
                PUID=-1,
                PGID=-1,
            )
            path = save_to_obsidian(settings, result)
            self.assertTrue(path.exists())
            self.assertTrue(path.read_text(encoding="utf-8").startswith("---"))
            # Second save should not clobber
            path2 = save_to_obsidian(settings, result)
            self.assertNotEqual(path, path2)


if __name__ == "__main__":
    unittest.main()
