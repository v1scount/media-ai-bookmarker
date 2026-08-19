# TikTok / X → Telegram → Obsidian Extractor

Self-hosted bot: send a TikTok or X (Twitter) link on Telegram → local extraction → OpenRouter multimodal extraction → optional save as Markdown into your Obsidian vault.

- **TikTok:** `yt-dlp` download → `faster-whisper` transcript + sampled frames → model extraction.
- **X:** saved verbatim by default, with **no model call** — the post text, its links (each with the linked page's title and description), and its photos downloaded into your vault. Set `X_USE_LLM=true` to run X posts through the model like TikTok videos instead.

No transcripts, audio, or video are persisted. Temp media is deleted after each job.

## Requirements

- Docker + Docker Compose
- Telegram bot token ([@BotFather](https://t.me/BotFather))
- [OpenRouter](https://openrouter.ai/) API key
- Path to your Obsidian vault on the host

## Setup

1. Copy env file and fill values:

```bash
cp .env.example .env
```

2. Set at least:

| Variable | Notes |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From BotFather |
| `OPENROUTER_API_KEY` | From OpenRouter |
| `OPENROUTER_MODEL` | Multimodal model, e.g. `google/gemini-2.5-flash` |
| `OBSIDIAN_VAULT_PATH` | Absolute host path to the vault |
| `OBSIDIAN_RELATIVE_DIR` | Folder inside vault, shared by both sources (recommended: `Extracts`) |
| `OBSIDIAN_ATTACHMENTS_DIR` | Optional folder for saved photos and videos (default: `attachments` under the notes folder) |
| `MAX_ATTACHMENT_MB` | Per-file ceiling for saved media (default `100`) |
| `X_USE_LLM` | `false` (default) saves X posts verbatim; `true` sends them to the model |
| `ALLOWED_TELEGRAM_USER_IDS` | Your numeric Telegram user id |
| `PUID` / `PGID` | Host uid/gid that owns the vault (`id -u` / `id -g`). Required so notes are not root-owned |

3. Discover your Telegram user id: message the bot with `/whoami` (no allowlist needed), or use [@userinfobot](https://t.me/userinfobot). Put that number in `ALLOWED_TELEGRAM_USER_IDS`.

On the server, set ownership to match your login:

```bash
id -u   # -> PUID
id -g   # -> PGID
# One-time fix for notes already created as root:
sudo chown -R "$(id -u):$(id -g)" "$HOME/Obsidian Vault/Extracts"
```

4. Start:

```bash
docker compose up -d --build
```

Notes are written to:

`$OBSIDIAN_VAULT_PATH/$OBSIDIAN_RELATIVE_DIR/YYYY-MM-DD-slug.md`

## TikTok download troubleshooting

If the bot fails with `Unexpected response from webpage request`:

1. Rebuild so you get the latest yt-dlp master extractor: `docker compose up -d --build`
2. Export TikTok cookies (while logged in on tiktok.com) to `cookies.txt` in the project root using a browser extension such as **Get cookies.txt LOCALLY**, then restart the container.
3. Retry the same link.

The downloader already sends a Chrome User-Agent + TikTok Referer and follows `vm`/`vt` short links.

## X posts

X metadata comes from the public FxTwitter mirror, with VxTwitter as a fallback, so
no X API key or login is needed for text and photos. Only the single post you send
is read — threads are not unrolled, though a quoted post's text and links are included.

By default an X post is **captured, not summarised**: the note is the post's own text,
a `## Links` section, and a `## Media` section embedding its photos. Nothing is sent to
OpenRouter, so these notes are free and near-instant. Set `X_USE_LLM=true` if you would
rather have summaries and extracted items for X too.

Photos, videos, and GIFs are all downloaded into `OBSIDIAN_ATTACHMENTS_DIR` (by
default an `attachments` folder beside your notes) and embedded as `![[...]]`
wikilinks, which Obsidian renders as an inline player for video. X serves animated
GIFs as `.mp4`, so they are saved that way too. The download happens when you tap
**Save**, never on preview, so a dismissed post leaves nothing behind.

Files above `MAX_ATTACHMENT_MB` (100 by default) are skipped and the note falls back
to a plain link, so one huge video cannot fill your vault.

Each non-X link in the post is expanded (including `t.co`) and its title plus Open
Graph description are fetched — the first 64 KB of the page only, never the full
article. At most 5 links, ~8 s each; a site that blocks us still contributes its URL.

Accepted URL forms: `x.com` and `twitter.com` (with or without `www`/`mobile`),
`/status/123` and `/i/status/123`, plus the `vxtwitter.com` / `fxtwitter.com` /
`fixupx.com` wrappers.

Private or NSFW posts may fail, in which case the bot reports the error. Video in an
X post uses `yt-dlp`, which sometimes needs cookies — set `YTDLP_COOKIES_FILE` with
x.com cookies if you hit that. Text, links, and photos never need cookies.

## Usage

1. Share a TikTok URL (`tiktok.com` / `vm.tiktok.com` / `vt.tiktok.com`) or an X post URL to the bot.
2. Wait for the preview (summary + entities).
3. Tap **Save to Obsidian** or **Dismiss**.

Both sources write into the same vault folder. Notes are tagged `[tiktok, extract]`
or `[x, extract]` and carry a `source_kind` frontmatter field, so they stay easy to
filter apart.

## Local CLI (optional)

For debugging without Telegram, run inside the container:

```bash
docker compose exec bot python -m app.cli "https://www.tiktok.com/@user/video/123"
docker compose exec bot python -m app.cli "https://vm.tiktok.com/xxx" --save
docker compose exec bot python -m app.cli "https://x.com/user/status/123"
```

## What gets extracted

The model is asked for things worth looking up later, always answered in English
even when the video is not:

- `summary` — one to three sentences on what the video recommends
- `video_kind` — `list` for roundups, `single` when the video is about one thing
- `entities` — one per item, each with a category, exact `name`,
  `creator_or_author`, short `notes`, `is_main_topic`, `confidence`, and an
  optional `suggested_link`

Categories: `tool` (software, apps, websites, services), `product`, `book`,
`movie`, `series`, `album` (artist or band in `creator_or_author`), `video`
(YouTube videos and channels), `podcast`, `course`, `article`, `place`,
`recipe`, `other`.

Notes on behaviour:

- The note title is always the original TikTok caption, or the first line of the X
  post with links stripped; the model never rewrites it.
- `suggested_link` is only filled when the model is confident. For X posts the
  expanded links are handed to the model, so they can be used verbatim. Otherwise the note
  gets a plain web search link built locally from the name and author, so nothing
  is ever guessed and no research tokens are spent.
- Items marked low confidence render as `_(uncertain)_`.
- List videos render every item under `## Items`; single-topic videos lead with
  `## Recommendation` and push passing mentions to `## Also mentioned`.

## Pipeline

1. Detect a TikTok or X URL (first supported link in the message wins)
2. Gather the source, temp only:
   - TikTok: `yt-dlp` download
   - X: post JSON from FxTwitter/VxTwitter plus link previews. Media is downloaded
     here only when `X_USE_LLM=true`
3. X with `X_USE_LLM=false` stops here and builds the note locally — steps 3-5 are skipped
4. When there is video: `ffmpeg` → mono 16 kHz WAV for Whisper; sample ~8 JPEG frames
5. `faster-whisper` `base` (CPU int8) → transcript kept **in memory only**
6. OpenRouter multimodal → structured JSON (tools / books / movies / music)
7. Always delete temp job directory
8. Telegram preview + save keyboard; photos are fetched into the vault on save

## Token cost controls

Frame images dominate the OpenRouter bill, so the pipeline works to send as few
as possible and to avoid requests that would fail:

- Frames are sampled generously with ffmpeg, then near-duplicates are dropped via
  perceptual hashing, so a static talking-head clip sends 2-3 frames instead of 8.
- Frames are downscaled to `FRAME_MAX_WIDTH` (512 by default).
- Responses are constrained by a strict JSON schema, so malformed output is rare.
- If the response still needs repair, the retry sends only the broken text —
  never the images again.
- Transcript, description, and model output are all capped
  (`MAX_TRANSCRIPT_CHARS`, `MAX_DESCRIPTION_CHARS`, `MAX_OUTPUT_TOKENS`).
- Videos with no speech and no usable frames skip the API call entirely, as do X
  posts with no text, links, or photos.
- Repeat links are served from an in-memory cache (`RESULT_CACHE_SIZE`).
- The configured model is checked for image support at startup, so frames are
  never uploaded to a text-only model.

Per-request token usage is logged, so `docker compose logs bot | grep usage`
shows what each video actually cost.

## Security

- Empty `ALLOWED_TELEGRAM_USER_IDS` denies everyone (safe default).
- Keep `.env` off shared/public remotes.

## Project layout

```
app/
  bot.py          Telegram bot
  pipeline.py     Source dispatch + whisper / ffmpeg orchestration
  download.py     yt-dlp: TikTok strategies (headers, cookies, retries) + X video
  xfetch.py       X post metadata, photo downloads, Open Graph link previews
  frames.py       Near-duplicate frame dropping (saves image tokens)
  openrouter.py   Multimodal extraction client
  obsidian.py     Markdown renderer + atomic vault write
  config.py       Settings from env
  models.py       Schema + URL parsing
  cli.py          CLI test harness
docker-compose.yml
Dockerfile
.env.example
```
