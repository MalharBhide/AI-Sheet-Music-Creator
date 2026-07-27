FROM python:3.11-slim

WORKDIR /app/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg musescore3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STORAGE_ROOT=/app/storage
ENV FFMPEG_BIN=ffmpeg
ENV MUSESCORE_BIN=musescore3

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
