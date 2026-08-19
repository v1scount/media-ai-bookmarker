FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install deps first; then force newest yt-dlp from master for TikTok extractor fixes.
# Do NOT install curl_cffi — browser impersonation currently breaks TikTok webpage extraction.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -U \
        "yt-dlp @ https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz"

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_DOWNLOAD_ROOT=/models

RUN mkdir -p /models /tmp/tiktok-jobs /vault /config

CMD ["python", "-m", "app.bot"]
