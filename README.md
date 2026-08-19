# TikTok → Telegram → Obsidian Extractor

Self-hosted bot: send a TikTok link on Telegram → local `yt-dlp` + `faster-whisper` + frame sampling → OpenRouter multimodal extraction → optional save as Markdown into your Obsidian vault.

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
| `OBSIDIAN_RELATIVE_DIR` | Folder inside vault (default `TikTok Extracts`) |
| `ALLOWED_TELEGRAM_USER_IDS` | Your numeric Telegram user id |
| `PUID` / `PGID` | Host uid/gid that owns the vault (`id -u` / `id -g`). Required so notes are not root-owned |

3. Discover your Telegram user id: message the bot with `/whoami` (no allowlist needed), or use [@userinfobot](https://t.me/userinfobot). Put that number in `ALLOWED_TELEGRAM_USER_IDS`.

On the server, set ownership to match your login:

```bash
id -u   # -> PUID
id -g   # -> PGID
# One-time fix for notes already created as root:
sudo chown -R "$(id -u):$(id -g)" "$HOME/Obsidian Vault/TikTok Extracts"
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

## Usage

1. Share a TikTok URL to the bot (or paste `tiktok.com` / `vm.tiktok.com` / `vt.tiktok.com` links).
2. Wait for the preview (summary + entities).
3. Tap **Save to Obsidian** or **Dismiss**.

## Local CLI (optional)

For debugging without Telegram, run inside the container:

```bash
docker compose exec bot python -m app.cli "https://www.tiktok.com/@user/video/123"
docker compose exec bot python -m app.cli "https://vm.tiktok.com/xxx" --save
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

- The note title is always the original TikTok caption; the model never rewrites it.
- `suggested_link` is only filled when the model is confident. Otherwise the note
  gets a plain web search link built locally from the name and author, so nothing
  is ever guessed and no research tokens are spent.
- Items marked low confidence render as `_(uncertain)_`.
- List videos render every item under `## Items`; single-topic videos lead with
  `## Recommendation` and push passing mentions to `## Also mentioned`.

## Pipeline

1. Validate TikTok URL
2. `yt-dlp` download (temp only)
3. `ffmpeg` → mono 16 kHz WAV for Whisper; sample ~8 JPEG frames
4. `faster-whisper` `base` (CPU int8) → transcript kept **in memory only**
5. OpenRouter multimodal → structured JSON (tools / books / movies / music)
6. Always delete temp job directory
7. Telegram preview + save keyboard

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
- Videos with no speech and no usable frames skip the API call entirely.
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
  pipeline.py     yt-dlp / whisper / ffmpeg orchestration
  download.py     TikTok download strategies (headers, cookies, retries)
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
