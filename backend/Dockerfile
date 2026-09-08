FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_QPA_PLATFORM=offscreen \
    STORAGE_ROOT=/data \
    MUSESCORE_BIN=/usr/bin/musescore3 \
    FFMPEG_BIN=ffmpeg \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg musescore3 libsndfile1 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY requirements.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml main.py ./
COPY app ./app
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 scribe \
    && mkdir /data && chown scribe:scribe /data

USER scribe
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
