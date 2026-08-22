"""Lightweight unit tests (no network / Whisper / ffmpeg)."""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import httpx

from app.config import Settings, _parse_user_ids
from app.download import _usable_cookies, pick_media_files
from app.frames import dedupe_frames, hamming_distance
from app.kagi import (
    KagiClient,
    authorization_header,
    select_kagi_candidates,
    top_result_url,
)
from app.models import (
    Confidence,
    Entity,
    EntityType,
    ExtractionResult,
    LinkRef,
    MediaKind,
    MediaRef,
    PostContent,
    SourceKind,
    VideoKind,
    extract_status_id,
    extract_supported_url,
    extract_tiktok_url,
    extract_x_url,
)
from app.obsidian import build_note_filename, render_markdown, save_to_obsidian
from app.openrouter import RESULT_JSON_SCHEMA, truncate_text
from app.pipeline import (
    Pipeline,
    SourceArtifacts,
    build_post_content,
    format_preview,
    has_analysable_content,
)
from app.xfetch import (
    LinkPreview,
    TweetData,
    parse_open_graph,
    should_preview,
    tweet_data_from_json,
)


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

    def test_kagi_defaults_are_links_only(self) -> None:
        settings = Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
        )
        self.assertEqual(settings.kagi_api_key, "")
        self.assertEqual(settings.kagi_search_per_job, 3)
        self.assertEqual(settings.kagi_timeout_seconds, 15.0)


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


class XUrlParseTests(unittest.TestCase):
    def test_x_com_status(self) -> None:
        url = "https://x.com/alice/status/1234567890123456789"
        self.assertEqual(extract_x_url(f"see {url} now"), url)
        self.assertEqual(extract_status_id(url), "1234567890123456789")

    def test_twitter_com(self) -> None:
        url = "https://twitter.com/alice/status/99"
        self.assertEqual(extract_x_url(url), url)
        self.assertEqual(extract_status_id(url), "99")

    def test_www_and_mobile(self) -> None:
        self.assertEqual(
            extract_status_id("https://www.twitter.com/u/status/1"),
            "1",
        )
        self.assertEqual(
            extract_status_id("https://mobile.twitter.com/u/status/2"),
            "2",
        )

    def test_i_status(self) -> None:
        self.assertEqual(extract_status_id("https://x.com/i/status/42"), "42")
        self.assertEqual(extract_x_url("https://x.com/i/status/42"), "https://x.com/i/status/42")

    def test_wrappers(self) -> None:
        for host in ("vxtwitter.com", "fxtwitter.com", "fixupx.com"):
            url = f"https://{host}/alice/status/99"
            self.assertEqual(extract_x_url(url), url, host)
            self.assertEqual(extract_status_id(url), "99", host)

    def test_query_string_still_yields_id(self) -> None:
        self.assertEqual(
            extract_status_id("https://x.com/u/status/123?s=20"),
            "123",
        )

    def test_strips_trailing_punct(self) -> None:
        self.assertEqual(
            extract_x_url("(https://x.com/u/status/1)."),
            "https://x.com/u/status/1",
        )

    def test_profile_is_not_a_status(self) -> None:
        self.assertIsNone(extract_x_url("https://x.com/alice"))
        self.assertIsNone(extract_status_id("https://x.com/alice"))
        self.assertIsNone(extract_x_url("no link here"))

    def test_supported_prefers_first_tiktok(self) -> None:
        text = (
            "https://www.tiktok.com/@u/video/1 then "
            "https://x.com/u/status/2"
        )
        self.assertEqual(
            extract_supported_url(text),
            (SourceKind.tiktok, "https://www.tiktok.com/@u/video/1"),
        )

    def test_supported_prefers_first_x(self) -> None:
        text = (
            "https://x.com/u/status/2 then "
            "https://www.tiktok.com/@u/video/1"
        )
        self.assertEqual(
            extract_supported_url(text),
            (SourceKind.x, "https://x.com/u/status/2"),
        )

    def test_supported_none(self) -> None:
        self.assertIsNone(extract_supported_url("hello"))

    def test_supported_skips_profile_for_later_tiktok(self) -> None:
        text = "https://x.com/alice https://www.tiktok.com/@u/video/1"
        self.assertEqual(
            extract_supported_url(text),
            (SourceKind.tiktok, "https://www.tiktok.com/@u/video/1"),
        )

    def test_extraction_defaults_source_kind_tiktok(self) -> None:
        result = ExtractionResult(source_url="u")
        self.assertEqual(result.source_kind, SourceKind.tiktok)


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
            "https://kagi.com/search?q=Dune+Frank+Herbert+book",
        )
        self.assertEqual(entity.search_query, "Dune Frank Herbert book")

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
        for field in ("source_url", "title", "creator", "source_kind"):
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
        self.assertIn("[search](https://kagi.com/search?q=Dune+", md)

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
                OBSIDIAN_RELATIVE_DIR="Extracts",
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

    def test_x_tags_and_source_kind(self) -> None:
        result = ExtractionResult(
            source_url="https://x.com/u/status/1",
            source_kind=SourceKind.x,
            title="A tweet about a book",
        )
        md = render_markdown(result)
        self.assertIn("tags: [x, extract]", md)
        self.assertIn("source_kind: x", md)
        self.assertNotIn("tags: [tiktok, extract]", md)

    def test_x_handle_is_quoted_in_frontmatter(self) -> None:
        # A bare "@handle" is invalid YAML and breaks every Obsidian property
        result = ExtractionResult(
            source_url="https://x.com/u/status/1",
            source_kind=SourceKind.x,
            creator="@alice",
        )
        md = render_markdown(result)
        self.assertIn('creator: "@alice"', md)

    def test_plain_creator_is_left_unquoted(self) -> None:
        result = ExtractionResult(source_url="u", creator="someuser")
        md = render_markdown(result)
        self.assertIn("creator: someuser", md)
        # created stays a bare scalar so Obsidian still types it as a date
        self.assertNotIn('created: "', md)

    def test_creator_with_colon_is_quoted(self) -> None:
        result = ExtractionResult(source_url="u", creator="News: Daily")
        md = render_markdown(result)
        self.assertIn('creator: "News: Daily"', md)

    def test_frontmatter_parses_as_yaml(self) -> None:
        yaml = __import__("importlib").util.find_spec("yaml")
        if yaml is None:
            self.skipTest("PyYAML not installed")
        import yaml as yaml_mod

        result = ExtractionResult(
            source_url="https://x.com/u/status/1",
            source_kind=SourceKind.x,
            creator="@alice",
            title='A "quoted" post: part 2',
        )
        block = render_markdown(result).split("---")[1]
        data = yaml_mod.safe_load(block)
        self.assertEqual(data["creator"], "@alice")
        self.assertEqual(data["source_kind"], "x")
        self.assertEqual(data["tags"], ["x", "extract"])

    def test_tiktok_frontmatter_includes_source_kind(self) -> None:
        result = ExtractionResult(source_url="u")
        md = render_markdown(result)
        self.assertIn("tags: [tiktok, extract]", md)
        self.assertIn("source_kind: tiktok", md)

    def test_x_filename_slug_fallback(self) -> None:
        result = ExtractionResult(
            source_url="https://x.com/u/status/1",
            source_kind=SourceKind.x,
            title="!!!",
        )
        self.assertIn("x-extract", build_note_filename(result))

    def test_tiktok_filename_slug_fallback(self) -> None:
        result = ExtractionResult(source_url="u", title="!!!")
        self.assertIn("tiktok-extract", build_note_filename(result))


FX_TWEET_FIXTURE = {
    "code": 200,
    "message": "OK",
    "tweet": {
        "id": "1234567890123456789",
        "text": "Great article on note-taking https://t.co/abcd",
        "author": {"name": "Alice Example", "screen_name": "alice"},
        "media": {
            "photos": [
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/photo1.jpg",
                    "width": 100,
                    "height": 100,
                }
            ]
        },
        "quote": {
            "id": "111",
            "text": "Original take https://quoted.example/post",
            "author": {"name": "Bob", "screen_name": "bob"},
        },
        "entities": {
            "urls": [
                {
                    "url": "https://t.co/abcd",
                    "expanded_url": "https://example.com/article",
                }
            ]
        },
    },
}

VX_TWEET_FIXTURE = {
    "tweetID": "1234567890123456789",
    "text": "Check this https://example.com/article",
    "user_name": "Alice Example",
    "user_screen_name": "alice",
    "media_extended": [
        {"type": "image", "url": "https://pbs.twimg.com/media/photo1.jpg"},
        {"type": "video", "url": "https://video.twimg.com/ext_tw_video/foo.mp4"},
    ],
    "qrt": {
        "text": "Quoted thought https://other.example/page",
        "user_screen_name": "bob",
    },
}


class TweetMappingTests(unittest.TestCase):
    def test_fxtwitter_fixture(self) -> None:
        tweet = tweet_data_from_json(FX_TWEET_FIXTURE, status_id="1234567890123456789")
        self.assertEqual(tweet.tweet_id, "1234567890123456789")
        self.assertIn("Great article", tweet.text)
        self.assertEqual(tweet.author_name, "Alice Example")
        self.assertEqual(tweet.author_handle, "@alice")
        self.assertIn("https://example.com/article", tweet.urls)
        self.assertEqual(
            tweet.photo_urls, ["https://pbs.twimg.com/media/photo1.jpg"]
        )
        self.assertFalse(tweet.has_video)
        self.assertIn("Original take", tweet.quoted_text)
        self.assertIn("https://quoted.example/post", tweet.quoted_urls)
        self.assertIn("https://example.com/article", tweet.all_urls)

    def test_fxtwitter_replaces_tco_with_expansion(self) -> None:
        tweet = tweet_data_from_json(FX_TWEET_FIXTURE, status_id="1")
        # The shortener is substituted, so it never costs a resolve request later
        self.assertNotIn("https://t.co/abcd", tweet.urls)

    def test_vxtwitter_fixture(self) -> None:
        tweet = tweet_data_from_json(VX_TWEET_FIXTURE)
        self.assertEqual(tweet.tweet_id, "1234567890123456789")
        self.assertEqual(tweet.author_handle, "@alice")
        self.assertIn("https://example.com/article", tweet.urls)
        self.assertEqual(
            tweet.photo_urls, ["https://pbs.twimg.com/media/photo1.jpg"]
        )
        self.assertTrue(tweet.has_video)
        self.assertIn("Quoted thought", tweet.quoted_text)
        self.assertIn("https://other.example/page", tweet.quoted_urls)


class OpenGraphTests(unittest.TestCase):
    def test_prefers_og_tags(self) -> None:
        html = """
        <html><head>
          <title>Fallback Title</title>
          <meta property="og:title" content="OG Title">
          <meta property="og:description" content="OG Description">
        </head><body></body></html>
        """
        title, description = parse_open_graph(html)
        self.assertEqual(title, "OG Title")
        self.assertEqual(description, "OG Description")

    def test_falls_back_to_title_tag(self) -> None:
        html = "<html><head><title>  Page Title  </title></head></html>"
        title, description = parse_open_graph(html)
        self.assertEqual(title, "Page Title")
        self.assertEqual(description, "")

    def test_skips_x_and_media_hosts(self) -> None:
        self.assertFalse(should_preview("https://x.com/u/status/1"))
        self.assertFalse(should_preview("https://twitter.com/u/status/1"))
        self.assertFalse(should_preview("https://pbs.twimg.com/media/x.jpg"))
        self.assertFalse(should_preview("https://t.co/abcd"))
        self.assertTrue(should_preview("https://example.com/article"))


class PipelineSkipTests(unittest.TestCase):
    def test_x_text_only_is_analysable_without_media_files(self) -> None:
        artifacts = SourceArtifacts(
            work_dir=Path("."),
            source_kind=SourceKind.x,
            title="Hello",
            description="Hello https://example.com/article",
            link_previews=[
                LinkPreview(
                    url="https://example.com/article",
                    title="Article",
                    description="Desc",
                )
            ],
        )
        self.assertTrue(has_analysable_content(artifacts, "", []))

    def test_tiktok_without_transcript_or_frames_is_skipped(self) -> None:
        artifacts = SourceArtifacts(
            work_dir=Path("."),
            source_kind=SourceKind.tiktok,
            title="caption",
            description="caption",
        )
        self.assertFalse(has_analysable_content(artifacts, "", []))


class PipelineXRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_x_text_only_skips_whisper_and_frames(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                TELEGRAM_BOT_TOKEN="x",
                OPENROUTER_API_KEY="x",
                JOB_TMP_DIR=tmp,
                RESULT_CACHE_SIZE=0,
            )
            openrouter = MagicMock()
            extracted = ExtractionResult(
                source_url="https://x.com/u/status/1",
                source_kind=SourceKind.x,
                title="Hello",
                creator="@u",
                summary="A tweet.",
            )
            openrouter.extract = AsyncMock(return_value=extracted)

            pipeline = Pipeline(settings, openrouter)
            artifacts = SourceArtifacts(
                work_dir=Path(tmp),
                source_kind=SourceKind.x,
                title="Hello",
                creator="@u",
                description="Hello https://example.com/article",
                source_id="1",
                expanded_urls=["https://example.com/article"],
                link_previews=[
                    LinkPreview(
                        url="https://example.com/article",
                        title="Article",
                        description="Desc",
                    )
                ],
            )
            pipeline._download_x = MagicMock(return_value=artifacts)
            pipeline._download = MagicMock(
                side_effect=AssertionError("TikTok download should not run")
            )
            pipeline._transcribe = MagicMock(
                side_effect=AssertionError("Whisper should not run")
            )
            pipeline._extract_frames = MagicMock(
                side_effect=AssertionError("ffmpeg frames should not run")
            )

            result = await pipeline.run("https://x.com/u/status/1")

            openrouter.extract.assert_awaited()
            kwargs = openrouter.extract.await_args.kwargs
            self.assertEqual(kwargs["source_kind"], SourceKind.x)
            self.assertEqual(kwargs["frame_paths"], [])
            self.assertTrue(kwargs["link_previews"])
            pipeline._transcribe.assert_not_called()
            pipeline._extract_frames.assert_not_called()
            pipeline._download.assert_not_called()
            self.assertEqual(result.source_kind, SourceKind.x)
            self.assertEqual(result.title, "Hello")


class PostContentTests(unittest.TestCase):
    def _tweet(self) -> TweetData:
        return TweetData(
            tweet_id="1",
            text="Worth reading https://example.com/article",
            author_handle="@alice",
            urls=["https://example.com/article"],
            photo_urls=["https://pbs.twimg.com/media/a.jpg"],
            video_urls=["https://video.twimg.com/b.mp4"],
            has_video=True,
            quoted_text="Original point",
        )

    def test_build_keeps_text_links_and_media(self) -> None:
        previews = [
            LinkPreview(
                url="https://example.com/article",
                title="Article",
                description="Desc",
            )
        ]
        post = build_post_content(self._tweet(), previews)
        self.assertIn("Worth reading", post.text)
        self.assertIn("> Original point", post.text)
        self.assertEqual(post.links[0].title, "Article")
        self.assertEqual(len(post.images), 1)
        self.assertEqual(len(post.videos), 1)

    def test_links_without_previews_are_still_kept(self) -> None:
        post = build_post_content(self._tweet(), [])
        self.assertEqual(
            [link.url for link in post.links], ["https://example.com/article"]
        )
        self.assertEqual(post.links[0].title, "")


class RawNoteTests(unittest.TestCase):
    def _result(self, **overrides) -> ExtractionResult:
        post = overrides.pop(
            "post",
            PostContent(
                text="Worth reading this.",
                links=[
                    LinkRef(
                        url="https://example.com/article",
                        title="Article",
                        description="Desc",
                    )
                ],
                media=[
                    MediaRef(
                        kind=MediaKind.image,
                        url="https://pbs.twimg.com/media/a.jpg",
                    )
                ],
            ),
        )
        return ExtractionResult(
            source_url="https://x.com/alice/status/1",
            source_kind=SourceKind.x,
            title="Worth reading this.",
            creator="@alice",
            post=post,
            **overrides,
        )

    def test_note_has_text_links_and_media_sections(self) -> None:
        md = render_markdown(self._result())
        self.assertIn("Worth reading this.", md)
        self.assertIn("## Links", md)
        self.assertIn("[Article](https://example.com/article)", md)
        self.assertIn("  - Desc", md)
        self.assertIn("## Media", md)
        # No AI sections when nothing was extracted
        self.assertNotIn("## Items", md)
        self.assertNotIn("## Recommendation", md)

    def test_note_omits_meaningless_kind_property(self) -> None:
        md = render_markdown(self._result())
        self.assertIn("source_kind: x", md)
        self.assertIn('creator: "@alice"', md)
        self.assertNotIn("\nkind:", md)

    def test_image_embeds_use_vault_path_when_saved(self) -> None:
        result = self._result()
        result.post.media[0].vault_path = "Extracts/attachments/note-1.jpg"
        md = render_markdown(result)
        self.assertIn("![[Extracts/attachments/note-1.jpg]]", md)
        self.assertNotIn("![](https://pbs.twimg.com", md)

    def test_image_falls_back_to_url_before_saving(self) -> None:
        md = render_markdown(self._result())
        self.assertIn("![](https://pbs.twimg.com/media/a.jpg)", md)

    def test_saved_video_is_embedded_like_an_image(self) -> None:
        post = PostContent(
            media=[
                MediaRef(
                    kind=MediaKind.video,
                    url="https://video.twimg.com/b.mp4",
                    vault_path="Extracts/attachments/note-1.mp4",
                )
            ]
        )
        md = render_markdown(self._result(post=post))
        self.assertIn("![[Extracts/attachments/note-1.mp4]]", md)
        self.assertNotIn("- [Video]", md)

    def test_unsaved_video_falls_back_to_a_link(self) -> None:
        post = PostContent(
            media=[
                MediaRef(kind=MediaKind.video, url="https://video.twimg.com/b.mp4")
            ]
        )
        md = render_markdown(self._result(post=post))
        self.assertIn("- [Video](https://video.twimg.com/b.mp4)", md)
        self.assertNotIn("![[", md)

    def test_preview_lists_links_and_media_counts(self) -> None:
        preview = format_preview(self._result())
        self.assertIn("Worth reading this.", preview)
        self.assertIn("<b>Links</b>", preview)
        self.assertIn("1 photo attached", preview)
        self.assertNotIn("Nothing worth looking up", preview)

    @staticmethod
    def _vault_settings(tmp: str) -> Settings:
        return Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
            OBSIDIAN_VAULT_PATH=tmp,
            OBSIDIAN_RELATIVE_DIR="Extracts",
            ALLOWED_TELEGRAM_USER_IDS="1",
            PUID=-1,
            PGID=-1,
        )

    @staticmethod
    def _fake_download(url: str, dest: Path, **_kwargs) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"media-bytes")
        return True

    def test_save_downloads_photos_into_vault(self) -> None:
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._vault_settings(tmp)
            with unittest.mock.patch(
                "app.xfetch.download_media", self._fake_download
            ):
                path = save_to_obsidian(settings, result)

            attachment = Path(tmp) / "Extracts" / "attachments" / f"{path.stem}-1.jpg"
            self.assertTrue(attachment.exists())
            md = path.read_text(encoding="utf-8")
            self.assertIn(f"![[Extracts/attachments/{path.stem}-1.jpg]]", md)

    def test_save_downloads_videos_into_vault(self) -> None:
        post = PostContent(
            media=[
                MediaRef(
                    kind=MediaKind.image, url="https://pbs.twimg.com/media/a.png"
                ),
                MediaRef(kind=MediaKind.video, url="https://video.twimg.com/b.mp4"),
                # X serves animated GIFs as mp4, so they save as video too
                MediaRef(kind=MediaKind.video, url="https://video.twimg.com/c/gif"),
            ]
        )
        result = self._result(post=post)
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._vault_settings(tmp)
            with unittest.mock.patch(
                "app.xfetch.download_media", self._fake_download
            ):
                path = save_to_obsidian(settings, result)

            attachments = Path(tmp) / "Extracts" / "attachments"
            self.assertTrue((attachments / f"{path.stem}-1.png").exists())
            self.assertTrue((attachments / f"{path.stem}-2.mp4").exists())
            # No usable extension on the URL, so it still lands as .mp4
            self.assertTrue((attachments / f"{path.stem}-3.mp4").exists())
            md = path.read_text(encoding="utf-8")
            self.assertIn(f"![[Extracts/attachments/{path.stem}-2.mp4]]", md)
            self.assertNotIn("- [Video]", md)

    def test_oversized_attachment_is_skipped(self) -> None:
        post = PostContent(
            media=[MediaRef(kind=MediaKind.video, url="https://video.twimg.com/b.mp4")]
        )
        result = self._result(post=post)
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._vault_settings(tmp)
            seen: dict[str, int] = {}

            def refuse(url: str, dest: Path, *, max_bytes: int, **_kw) -> bool:
                seen["max_bytes"] = max_bytes
                return False

            with unittest.mock.patch("app.xfetch.download_media", refuse):
                path = save_to_obsidian(settings, result)

            self.assertEqual(seen["max_bytes"], 100 * 1024 * 1024)
            md = path.read_text(encoding="utf-8")
            self.assertIn("- [Video](https://video.twimg.com/b.mp4)", md)

    def test_failed_photo_download_leaves_url_embed(self) -> None:
        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._vault_settings(tmp)
            with unittest.mock.patch(
                "app.xfetch.download_media", lambda *a, **k: False
            ):
                path = save_to_obsidian(settings, result)
            md = path.read_text(encoding="utf-8")
            self.assertIn("![](https://pbs.twimg.com/media/a.jpg)", md)


class RawPipelineTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, tmp: str, use_llm: bool) -> Settings:
        return Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
            JOB_TMP_DIR=tmp,
            RESULT_CACHE_SIZE=0,
            X_USE_LLM=use_llm,
        )

    async def test_x_skips_the_model_by_default(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp, use_llm=False)
            self.assertFalse(settings.x_use_llm)

            openrouter = MagicMock()
            openrouter.extract = AsyncMock(
                side_effect=AssertionError("model must not be called")
            )
            pipeline = Pipeline(settings, openrouter)
            post = PostContent(text="Hello", links=[LinkRef(url="https://e.com/a")])
            pipeline._download_x = MagicMock(
                return_value=SourceArtifacts(
                    work_dir=Path(tmp),
                    source_kind=SourceKind.x,
                    title="Hello",
                    creator="@u",
                    source_id="1",
                    post=post,
                )
            )

            result = await pipeline.run("https://x.com/u/status/1")

            openrouter.extract.assert_not_awaited()
            self.assertTrue(result.is_raw_capture)
            self.assertEqual(result.post.text, "Hello")
            self.assertEqual(result.source_kind, SourceKind.x)

    async def test_x_uses_the_model_when_enabled(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp, use_llm=True)
            openrouter = MagicMock()
            openrouter.extract = AsyncMock(
                return_value=ExtractionResult(
                    source_url="https://x.com/u/status/1",
                    summary="A tweet.",
                )
            )
            pipeline = Pipeline(settings, openrouter)
            pipeline._download_x = MagicMock(
                return_value=SourceArtifacts(
                    work_dir=Path(tmp),
                    source_kind=SourceKind.x,
                    title="Hello",
                    creator="@u",
                    description="Hello there",
                    source_id="1",
                    post=PostContent(text="Hello there"),
                )
            )

            result = await pipeline.run("https://x.com/u/status/1")

            openrouter.extract.assert_awaited()
            self.assertFalse(result.is_raw_capture)
            self.assertEqual(result.summary, "A tweet.")


class KagiClientTests(unittest.TestCase):
    def test_authorization_adds_bot_prefix(self) -> None:
        self.assertEqual(authorization_header("abc"), "Bot abc")

    def test_authorization_keeps_existing_prefix(self) -> None:
        self.assertEqual(authorization_header("Bot abc"), "Bot abc")
        self.assertEqual(authorization_header("Bearer abc"), "Bearer abc")

    def test_v1_payload_uses_first_search_url(self) -> None:
        payload = {
            "data": {
                "search": [
                    {"url": "https://en.wikipedia.org/wiki/Dune_(novel)", "title": "Dune"},
                    {"url": "https://example.com/other", "title": "Other"},
                ]
            }
        }
        self.assertEqual(
            top_result_url(payload),
            "https://en.wikipedia.org/wiki/Dune_(novel)",
        )

    def test_legacy_payload_uses_first_t0_url(self) -> None:
        payload = {
            "data": [
                {"t": 1, "list": ["related"]},
                {
                    "t": 0,
                    "url": "https://en.wikipedia.org/wiki/Steve_Jobs",
                    "title": "Steve Jobs",
                },
            ]
        }
        self.assertEqual(
            top_result_url(payload),
            "https://en.wikipedia.org/wiki/Steve_Jobs",
        )

    def test_empty_or_invalid_payload_is_none(self) -> None:
        self.assertIsNone(top_result_url(None))
        self.assertIsNone(top_result_url({}))
        self.assertIsNone(top_result_url({"data": {"search": []}}))
        self.assertIsNone(top_result_url({"data": [{"t": 1, "list": ["x"]}]}))
        self.assertIsNone(top_result_url({"data": {"search": [{"url": "not-a-url"}]}}))

    def test_candidates_skip_linked_and_low_confidence(self) -> None:
        already = Entity(
            name="Linked",
            suggested_link="https://example.com",
            confidence=Confidence.high,
        )
        low = Entity(name="Guess", confidence=Confidence.low)
        main = Entity(
            name="Main",
            is_main_topic=True,
            confidence=Confidence.medium,
        )
        side = Entity(name="Side", confidence=Confidence.high)
        picked = select_kagi_candidates([already, low, side, main], limit=3)
        self.assertEqual([entity.name for entity in picked], ["Main", "Side"])

    def test_candidates_respect_cap_and_zero(self) -> None:
        items = [
            Entity(name="A", is_main_topic=True, confidence=Confidence.high),
            Entity(name="B", confidence=Confidence.high),
            Entity(name="C", confidence=Confidence.medium),
        ]
        self.assertEqual(
            [entity.name for entity in select_kagi_candidates(items, 2)],
            ["A", "B"],
        )
        self.assertEqual(select_kagi_candidates(items, 0), [])


class KagiHttpTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, **overrides: object) -> Settings:
        values: dict[str, object] = {
            "TELEGRAM_BOT_TOKEN": "x",
            "OPENROUTER_API_KEY": "x",
            "KAGI_API_KEY": "secret",
        }
        values.update(overrides)
        return Settings(**values)

    async def test_top_url_parses_v1_and_caches(self) -> None:
        client = KagiClient(self._settings())
        calls = {"n": 0}

        async def fake_post(*_args, **_kwargs):
            calls["n"] += 1
            request = httpx.Request("POST", "https://kagi.com/api/v1/search")
            return httpx.Response(
                200,
                json={
                    "data": {
                        "search": [
                            {"url": "https://en.wikipedia.org/wiki/Dune_(novel)"}
                        ]
                    }
                },
                request=request,
            )

        client._client.post = fake_post  # type: ignore[method-assign]
        try:
            url = await client.top_url("Dune Frank Herbert book")
            again = await client.top_url("Dune Frank Herbert book")
        finally:
            await client.aclose()
        self.assertEqual(url, "https://en.wikipedia.org/wiki/Dune_(novel)")
        self.assertEqual(again, url)
        self.assertEqual(calls["n"], 1)

    async def test_http_error_returns_none(self) -> None:
        client = KagiClient(self._settings())

        async def fake_post(*_args, **_kwargs):
            request = httpx.Request("POST", "https://kagi.com/api/v1/search")
            return httpx.Response(402, text="no credit", request=request)

        client._client.post = fake_post  # type: ignore[method-assign]
        try:
            url = await client.top_url("Dune")
        finally:
            await client.aclose()
        self.assertIsNone(url)

    async def test_disabled_without_key(self) -> None:
        client = KagiClient(self._settings(KAGI_API_KEY=""))
        self.assertFalse(client.enabled)
        try:
            self.assertIsNone(await client.top_url("Dune"))
        finally:
            await client.aclose()

    async def test_disabled_when_per_job_is_zero(self) -> None:
        client = KagiClient(self._settings(KAGI_SEARCH_PER_JOB=0))
        self.assertFalse(client.enabled)
        try:
            self.assertIsNone(await client.top_url("Dune"))
        finally:
            await client.aclose()


class KagiEnrichmentTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, tmp: str) -> Settings:
        return Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
            KAGI_API_KEY="secret",
            JOB_TMP_DIR=tmp,
            RESULT_CACHE_SIZE=0,
        )

    def _artifacts(self, tmp: str) -> SourceArtifacts:
        return SourceArtifacts(
            work_dir=Path(tmp),
            source_kind=SourceKind.x,
            title="Hello",
            creator="@u",
            description="Hello https://example.com/article",
            source_id="1",
            link_previews=[
                LinkPreview(
                    url="https://example.com/article",
                    title="Article",
                    description="Desc",
                )
            ],
        )

    async def test_fills_suggested_link_from_kagi(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            openrouter = MagicMock()
            openrouter.extract = AsyncMock(
                return_value=ExtractionResult(
                    source_url="https://x.com/u/status/1",
                    entities=[
                        Entity(
                            type=EntityType.book,
                            name="Dune",
                            creator_or_author="Frank Herbert",
                            is_main_topic=True,
                            confidence=Confidence.high,
                        ),
                        Entity(
                            name="Kindle",
                            type=EntityType.tool,
                            confidence=Confidence.low,
                        ),
                        Entity(
                            name="Goodreads",
                            suggested_link="https://www.goodreads.com/book/show/1",
                            confidence=Confidence.high,
                        ),
                    ],
                )
            )
            kagi = MagicMock()
            kagi.enabled = True
            kagi.top_url = AsyncMock(
                return_value="https://en.wikipedia.org/wiki/Dune_(novel)"
            )
            pipeline = Pipeline(settings, openrouter, kagi)
            pipeline._download_x = MagicMock(return_value=self._artifacts(tmp))

            result = await pipeline.run("https://x.com/u/status/1")

            kagi.top_url.assert_awaited_once_with("Dune Frank Herbert book")
            dune = next(entity for entity in result.entities if entity.name == "Dune")
            kindle = next(entity for entity in result.entities if entity.name == "Kindle")
            goodreads = next(
                entity for entity in result.entities if entity.name == "Goodreads"
            )
            self.assertEqual(
                dune.suggested_link, "https://en.wikipedia.org/wiki/Dune_(novel)"
            )
            self.assertIsNone(kindle.suggested_link)
            self.assertEqual(
                goodreads.suggested_link, "https://www.goodreads.com/book/show/1"
            )
            preview = format_preview(result)
            self.assertIn("en.wikipedia.org/wiki/Dune", preview)
            md = render_markdown(result)
            self.assertIn("[link](https://en.wikipedia.org/wiki/Dune_(novel))", md)
            self.assertIn("[search](https://kagi.com/search?q=Kindle", md)

    async def test_api_failure_leaves_kagi_search_url(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            openrouter = MagicMock()
            openrouter.extract = AsyncMock(
                return_value=ExtractionResult(
                    source_url="https://x.com/u/status/1",
                    entities=[
                        Entity(
                            name="Dune",
                            type=EntityType.book,
                            creator_or_author="Frank Herbert",
                            confidence=Confidence.high,
                        )
                    ],
                )
            )
            kagi = MagicMock()
            kagi.enabled = True
            kagi.top_url = AsyncMock(return_value=None)
            pipeline = Pipeline(settings, openrouter, kagi)
            pipeline._download_x = MagicMock(return_value=self._artifacts(tmp))

            result = await pipeline.run("https://x.com/u/status/1")

            self.assertIsNone(result.entities[0].suggested_link)
            self.assertTrue(
                result.entities[0].search_url.startswith("https://kagi.com/search?q=")
            )
            md = render_markdown(result)
            self.assertIn("[search](https://kagi.com/search?q=", md)


class HardcoverHelperTests(unittest.TestCase):
    def test_authorization_adds_bearer_prefix(self) -> None:
        from app.hardcover import authorization_header

        self.assertEqual(authorization_header("abc"), "Bearer abc")

    def test_authorization_keeps_existing_prefix(self) -> None:
        from app.hardcover import authorization_header

        self.assertEqual(authorization_header("Bearer abc"), "Bearer abc")
        self.assertEqual(authorization_header("bearer abc"), "bearer abc")

    def test_search_query_omits_book_hint(self) -> None:
        from app.hardcover import hardcover_search_query

        entity = Entity(
            type=EntityType.book,
            name="Dune",
            creator_or_author="Frank Herbert",
        )
        self.assertEqual(hardcover_search_query(entity), "Dune Frank Herbert")

    def test_parse_search_hits_from_dict(self) -> None:
        from app.hardcover import parse_search_hits

        hits = parse_search_hits(
            {
                "hits": [
                    {
                        "document": {
                            "id": "401",
                            "title": "Dune",
                            "author_names": ["Frank Herbert"],
                            "slug": "dune",
                        }
                    }
                ]
            }
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].book_id, 401)
        self.assertEqual(hits[0].title, "Dune")
        self.assertEqual(hits[0].author_names, ["Frank Herbert"])
        self.assertEqual(hits[0].slug, "dune")
        self.assertEqual(hits[0].url, "https://hardcover.app/books/dune")

    def test_parse_search_hits_from_json_string(self) -> None:
        from app.hardcover import parse_search_hits

        hits = parse_search_hits(
            '{"hits":[{"document":{"id":7,"title":"Neuromancer",'
            '"author_names":["William Gibson"],"slug":"neuromancer"}}]}'
        )
        self.assertEqual(hits[0].book_id, 7)
        self.assertEqual(hits[0].slug, "neuromancer")

    def test_parse_search_hits_ignores_invalid(self) -> None:
        from app.hardcover import parse_search_hits

        self.assertEqual(parse_search_hits(None), [])
        self.assertEqual(parse_search_hits({}), [])
        self.assertEqual(parse_search_hits("not-json"), [])
        self.assertEqual(parse_search_hits({"hits": [{"document": {"title": "x"}}]}), [])

    def test_pick_match_accepts_title_and_author(self) -> None:
        from app.hardcover import HardcoverHit, pick_match

        entity = Entity(
            type=EntityType.book,
            name="Dune",
            creator_or_author="Frank Herbert",
        )
        hits = [
            HardcoverHit(
                book_id=2,
                title="Dune Messiah",
                author_names=["Frank Herbert"],
                slug="dune-messiah",
            ),
            HardcoverHit(
                book_id=1,
                title="Dune",
                author_names=["Frank Herbert"],
                slug="dune",
            ),
        ]
        match = pick_match(entity, hits)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.book_id, 1)

    def test_pick_match_rejects_sequel_and_wrong_author(self) -> None:
        from app.hardcover import HardcoverHit, pick_match

        dune = Entity(
            type=EntityType.book,
            name="Dune",
            creator_or_author="Frank Herbert",
        )
        sequel_only = [
            HardcoverHit(
                book_id=2,
                title="Dune Messiah",
                author_names=["Frank Herbert"],
                slug="dune-messiah",
            )
        ]
        self.assertIsNone(pick_match(dune, sequel_only))

        wrong_author = [
            HardcoverHit(
                book_id=9,
                title="Dune",
                author_names=["Someone Else"],
                slug="dune-else",
            )
        ]
        self.assertIsNone(pick_match(dune, wrong_author))

    def test_pick_match_skips_author_check_when_extract_has_none(self) -> None:
        from app.hardcover import HardcoverHit, pick_match

        entity = Entity(type=EntityType.book, name="Dune")
        hits = [
            HardcoverHit(
                book_id=1,
                title="Dune",
                author_names=["Frank Herbert"],
                slug="dune",
            )
        ]
        match = pick_match(entity, hits)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.book_id, 1)

    def test_candidates_books_only_skip_low_confidence(self) -> None:
        from app.hardcover import select_hardcover_candidates

        already_low = Entity(
            type=EntityType.book,
            name="Guess",
            confidence=Confidence.low,
        )
        tool = Entity(type=EntityType.tool, name="Kindle", confidence=Confidence.high)
        side = Entity(
            type=EntityType.book,
            name="Side",
            confidence=Confidence.high,
        )
        main = Entity(
            type=EntityType.book,
            name="Main",
            is_main_topic=True,
            confidence=Confidence.medium,
        )
        picked = select_hardcover_candidates(
            [already_low, tool, side, main],
            limit=3,
        )
        self.assertEqual([entity.name for entity in picked], ["Main", "Side"])

    def test_candidates_respect_cap_and_zero(self) -> None:
        from app.hardcover import select_hardcover_candidates

        items = [
            Entity(
                type=EntityType.book,
                name="A",
                is_main_topic=True,
                confidence=Confidence.high,
            ),
            Entity(type=EntityType.book, name="B", confidence=Confidence.high),
            Entity(type=EntityType.book, name="C", confidence=Confidence.medium),
        ]
        self.assertEqual(
            [entity.name for entity in select_hardcover_candidates(items, 2)],
            ["A", "B"],
        )
        self.assertEqual(select_hardcover_candidates(items, 0), [])

    def test_hardcover_defaults_are_disabled(self) -> None:
        settings = Settings(
            TELEGRAM_BOT_TOKEN="x",
            OPENROUTER_API_KEY="x",
        )
        self.assertEqual(settings.hardcover_api_key, "")
        self.assertEqual(settings.hardcover_books_per_job, 8)
        self.assertEqual(settings.hardcover_timeout_seconds, 15.0)


class HardcoverHttpTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, **overrides: object) -> Settings:
        values: dict[str, object] = {
            "TELEGRAM_BOT_TOKEN": "x",
            "OPENROUTER_API_KEY": "x",
            "HARDCOVER_API_KEY": "secret",
        }
        values.update(overrides)
        return Settings(**values)

    def _dune(self) -> Entity:
        return Entity(
            type=EntityType.book,
            name="Dune",
            creator_or_author="Frank Herbert",
            is_main_topic=True,
            confidence=Confidence.high,
        )

    def _search_payload(self) -> dict:
        return {
            "data": {
                "search": {
                    "results": {
                        "hits": [
                            {
                                "document": {
                                    "id": 401,
                                    "title": "Dune",
                                    "author_names": ["Frank Herbert"],
                                    "slug": "dune",
                                }
                            }
                        ]
                    }
                }
            }
        }

    async def test_disabled_without_key(self) -> None:
        from app.hardcover import HardcoverClient

        client = HardcoverClient(self._settings(HARDCOVER_API_KEY=""))
        self.assertFalse(client.enabled)
        try:
            self.assertEqual(await client.sync_books([self._dune()]), [])
        finally:
            await client.aclose()

    async def test_disabled_when_per_job_is_zero(self) -> None:
        from app.hardcover import HardcoverClient

        client = HardcoverClient(self._settings(HARDCOVER_BOOKS_PER_JOB=0))
        self.assertFalse(client.enabled)
        try:
            self.assertEqual(await client.sync_books([self._dune()]), [])
        finally:
            await client.aclose()

    async def test_inserts_when_not_on_shelf(self) -> None:
        from app.hardcover import HardcoverClient, HardcoverOutcome

        client = HardcoverClient(self._settings())
        calls: list[str] = []

        async def fake_post(*_args, **kwargs):
            body = kwargs.get("json") or {}
            query = body.get("query", "")
            request = httpx.Request("POST", "https://api.hardcover.app/v1/graphql")
            if "search(" in query:
                calls.append("search")
                return httpx.Response(200, json=self._search_payload(), request=request)
            if "user_books" in query:
                calls.append("user_books")
                return httpx.Response(
                    200,
                    json={"data": {"me": {"id": 1, "user_books": []}}},
                    request=request,
                )
            if "insert_user_book" in query:
                calls.append("insert")
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "insert_user_book": {
                                "id": 99,
                                "user_book": {"id": 99, "status_id": 1, "book_id": 401},
                            }
                        }
                    },
                    request=request,
                )
            calls.append("other")
            return httpx.Response(400, json={"error": "unexpected"}, request=request)

        client._client.post = fake_post  # type: ignore[method-assign]
        try:
            actions = await client.sync_books([self._dune()])
        finally:
            await client.aclose()

        self.assertEqual(calls, ["search", "user_books", "insert"])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].outcome, HardcoverOutcome.added)
        self.assertEqual(actions[0].entity_name, "Dune")
        self.assertEqual(actions[0].hardcover_url, "https://hardcover.app/books/dune")
        self.assertEqual(actions[0].status_label, "Want to Read")

    async def test_skips_when_already_on_shelf(self) -> None:
        from app.hardcover import HardcoverClient, HardcoverOutcome

        client = HardcoverClient(self._settings())
        calls: list[str] = []

        async def fake_post(*_args, **kwargs):
            body = kwargs.get("json") or {}
            query = body.get("query", "")
            request = httpx.Request("POST", "https://api.hardcover.app/v1/graphql")
            if "search(" in query:
                calls.append("search")
                return httpx.Response(200, json=self._search_payload(), request=request)
            if "user_books" in query:
                calls.append("user_books")
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "me": [
                                {
                                    "id": 1,
                                    "user_books": [{"id": 12, "status_id": 3}],
                                }
                            ]
                        }
                    },
                    request=request,
                )
            calls.append("insert")
            return httpx.Response(500, text="should not insert", request=request)

        client._client.post = fake_post  # type: ignore[method-assign]
        try:
            actions = await client.sync_books([self._dune()])
        finally:
            await client.aclose()

        self.assertEqual(calls, ["search", "user_books"])
        self.assertEqual(actions[0].outcome, HardcoverOutcome.already_on_shelf)
        self.assertEqual(actions[0].status_label, "Read")
        self.assertEqual(actions[0].hardcover_url, "https://hardcover.app/books/dune")

    async def test_no_match_does_not_mutate(self) -> None:
        from app.hardcover import HardcoverClient, HardcoverOutcome

        client = HardcoverClient(self._settings())
        calls: list[str] = []

        async def fake_post(*_args, **kwargs):
            body = kwargs.get("json") or {}
            query = body.get("query", "")
            request = httpx.Request("POST", "https://api.hardcover.app/v1/graphql")
            if "search(" in query:
                calls.append("search")
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "search": {
                                "results": {
                                    "hits": [
                                        {
                                            "document": {
                                                "id": 2,
                                                "title": "Dune Messiah",
                                                "author_names": ["Frank Herbert"],
                                                "slug": "dune-messiah",
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    request=request,
                )
            calls.append("other")
            return httpx.Response(500, text="should not continue", request=request)

        client._client.post = fake_post  # type: ignore[method-assign]
        try:
            actions = await client.sync_books([self._dune()])
        finally:
            await client.aclose()

        self.assertEqual(calls, ["search"])
        self.assertEqual(actions[0].outcome, HardcoverOutcome.no_match)
        self.assertEqual(actions[0].hardcover_url, "")

    async def test_http_error_does_not_raise(self) -> None:
        from app.hardcover import HardcoverClient, HardcoverOutcome

        client = HardcoverClient(self._settings())

        async def fake_post(*_args, **_kwargs):
            request = httpx.Request("POST", "https://api.hardcover.app/v1/graphql")
            return httpx.Response(429, text="Too Many Requests", request=request)

        client._client.post = fake_post  # type: ignore[method-assign]
        try:
            actions = await client.sync_books([self._dune()])
        finally:
            await client.aclose()
        self.assertEqual(actions[0].outcome, HardcoverOutcome.error)


class HardcoverSaveTests(unittest.IsolatedAsyncioTestCase):
    def _result(self) -> ExtractionResult:
        return ExtractionResult(
            source_url="https://www.tiktok.com/@u/video/1",
            title="Book Rec",
            creator="@u",
            summary="A novel.",
            video_kind=VideoKind.single,
            entities=[
                Entity(
                    type=EntityType.book,
                    name="Dune",
                    creator_or_author="Frank Herbert",
                    is_main_topic=True,
                    confidence=Confidence.high,
                )
            ],
        )

    async def test_save_without_key_skips_sync(self) -> None:
        from unittest.mock import AsyncMock

        from app.hardcover import HardcoverClient, sync_hardcover_then_save

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                TELEGRAM_BOT_TOKEN="x",
                OPENROUTER_API_KEY="x",
                OBSIDIAN_VAULT_PATH=tmp,
                OBSIDIAN_RELATIVE_DIR="Extracts",
                PUID=-1,
                PGID=-1,
            )
            client = HardcoverClient(settings)
            client.sync_books = AsyncMock()  # type: ignore[method-assign]
            result = self._result()
            try:
                path, actions = await sync_hardcover_then_save(
                    settings, client, result
                )
            finally:
                await client.aclose()
            client.sync_books.assert_not_awaited()
            self.assertEqual(actions, [])
            self.assertTrue(path.exists())
            self.assertIsNone(result.entities[0].hardcover_url)

    async def test_save_with_key_syncs_before_vault_write(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.hardcover import (
            HardcoverAction,
            HardcoverClient,
            HardcoverOutcome,
            sync_hardcover_then_save,
        )

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                TELEGRAM_BOT_TOKEN="x",
                OPENROUTER_API_KEY="x",
                OBSIDIAN_VAULT_PATH=tmp,
                OBSIDIAN_RELATIVE_DIR="Extracts",
                HARDCOVER_API_KEY="secret",
                PUID=-1,
                PGID=-1,
            )
            client = MagicMock(spec=HardcoverClient)
            client.enabled = True
            client.sync_books = AsyncMock(
                return_value=[
                    HardcoverAction(
                        entity_name="Dune",
                        outcome=HardcoverOutcome.added,
                        status_label="Want to Read",
                        hardcover_url="https://hardcover.app/books/dune",
                    )
                ]
            )
            result = self._result()
            path, actions = await sync_hardcover_then_save(settings, client, result)
            client.sync_books.assert_awaited_once()
            self.assertEqual(actions[0].outcome, HardcoverOutcome.added)
            self.assertEqual(
                result.entities[0].hardcover_url,
                "https://hardcover.app/books/dune",
            )
            note = path.read_text(encoding="utf-8")
            self.assertIn("[hardcover](https://hardcover.app/books/dune)", note)


class HardcoverObsidianTests(unittest.TestCase):
    def test_bullet_includes_hardcover_link(self) -> None:
        result = ExtractionResult(
            source_url="https://www.tiktok.com/@u/video/1",
            title="Book Rec",
            summary="A novel.",
            video_kind=VideoKind.single,
            entities=[
                Entity(
                    type=EntityType.book,
                    name="Dune",
                    creator_or_author="Frank Herbert",
                    is_main_topic=True,
                    hardcover_url="https://hardcover.app/books/dune",
                )
            ],
        )
        md = render_markdown(result)
        self.assertIn("[hardcover](https://hardcover.app/books/dune)", md)
        self.assertIn("[search](https://kagi.com/search?q=Dune+", md)


if __name__ == "__main__":
    unittest.main()
