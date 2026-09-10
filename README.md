# AI Sheet Music Creator

Upload an audio recording and download a piano score as PDF, MusicXML, MIDI, or SVG. The React interface shows processing status and a paginated sheet preview. FastAPI runs the real Basic Pitch → music21 → MuseScore pipeline.

## Quick start with Docker

Install and open [Docker Desktop](https://www.docker.com/products/docker-desktop/), then run from this repository:

```bash
docker compose up --build
```

Open **[http://localhost:5173](http://localhost:5173)**. API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

The first build downloads the model dependencies and MuseScore. Later starts of the same version only need `docker compose up`. After pulling changes or switching branches, rebuild both services with `docker compose up -d --build`, then reload the browser page. Restarting containers alone keeps their previously built code. Stop with Ctrl+C or `docker compose down`. Job records and outputs remain in the Docker volume between starts. Removing that volume deletes them.

Docker includes Python 3.11, FFmpeg, and Debian Bookworm's [MuseScore 3 package](https://packages.debian.org/bookworm/musescore3), and serves a production frontend build through Nginx. The backend uses an amd64 image for the transcription dependencies; Apple Silicon runs it through Docker's emulation. Allow several GB of memory and disk space.

## Using the app

If a previously opened page says **“Choose a non-empty file up to 0 MB”**, it is running the old upload validator. After rebuilding, hard-refresh that page (Cmd+Shift+R on macOS, Ctrl+Shift+R on Windows/Linux) or open a fresh tab. Zero means unlimited uploads. Updated pages are served without browser caching.

1. Choose or drop a WAV, MP3, FLAC, OGG, M4A, AAC, or AIFF file.
2. Wait for audio preparation, transcription, notation cleanup, and rendering.
3. Browse the score pages and download the PDF. **More file formats** includes MIDI, MusicXML, the first SVG page, and a ZIP of all SVG pages. MIDI and MusicXML become downloadable as their stages finish, so a later rendering failure does not hide the files already created.

There is no default file size, recording duration, or processing time limit. MP3 files are decoded by FFmpeg, and transcription processes the entire recording in overlapping chunks so model memory does not grow with audio length. Longer recordings need more processing time and disk space; the upload and transcription progress are shown in the app. The page URL contains the job ID, so reloading or returning to that URL restores the job, and status polling reconnects automatically after a temporary network interruption. Save the downloads within 24 hours after processing finishes.

This is a first draft of a score, intended for clear solo piano or single-instrument recordings. Basic Pitch detects pitches and timing; it does not separate instruments or arrange a full song for piano. The MVP uses 120 BPM, 4/4, a sixteenth-note grid, and a fixed middle-C split between hands. Sustained notes, chords, rests, voices, and barline ties are preserved during cleanup, but phrasing, spelling, rhythm, and hand assignment still need musical review. Tempo, meter, and grid can be supplied through the API.

## Manual development setup

Use **Python 3.11** and **Node.js 24**. The full transcription dependency set is pinned for Python 3.11; Python 3.12 is not supported by this setup.

Install FFmpeg and MuseScore CLI:

```bash
# Debian / Ubuntu with the musescore3 package
sudo apt-get update
sudo apt-get install ffmpeg musescore3

# macOS
brew install ffmpeg
brew install --cask musescore
```

MuseScore 3 is the target server renderer. Some MuseScore 4 releases export only the first SVG page from the CLI. If SVG export is incomplete or fails, the app keeps the valid PDF and previews it directly. Use the Docker setup if your native installation cannot run with Qt's offscreen platform.

From the repository root:

```bash
cp .env.example .env
python3.11 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m pip install --no-deps -e backend
cd backend
uvicorn main:app --reload --port 8000
```

On Windows, use `py -3.11 -m venv backend/.venv` and `backend\.venv\Scripts\Activate.ps1`; Docker is the easiest way to provide the native tools.

Alternatively, with [uv](https://docs.astral.sh/uv/), run `uv sync --frozen --extra transcription --extra dev` inside `backend/`, then `uv run uvicorn main:app --reload --port 8000`.

In a second terminal, from the repository root:

```bash
npm install -g pnpm@11.19.0
pnpm install --frozen-lockfile
pnpm --dir frontend dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite development proxy forwards `/api` to port 8000. Build the frontend with `pnpm --dir frontend build`.

## Configuration

The backend reads root `.env` and then `backend/.env`; environment variables take precedence. See `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `STORAGE_ROOT` | `storage` | Database and job directories; relative paths resolve from the repository root |
| `CORS_ORIGINS` | localhost and 127.0.0.1 on port 5173 | Comma-separated origins or a JSON array |
| `MAX_UPLOAD_MB` | 0 | Optional file size limit; 0 disables it |
| `MAX_AUDIO_SECONDS` | 0 | Optional decoded recording duration limit; 0 disables it |
| `MAX_QUEUED_JOBS` | 10 | Maximum jobs waiting to be processed |
| `JOB_TIMEOUT_SECONDS` | 0 | Optional overall processing deadline; 0 disables it |
| `RENDER_TIMEOUT_SECONDS` | 0 | Optional deadline for each MuseScore export; 0 disables it |
| `RETENTION_HOURS` | 24 | Retention after a job finishes |
| `DEFAULT_TEMPO_BPM` | 120 | Tempo used when the upload omits it |
| `FFMPEG_BIN` | `ffmpeg` | FFmpeg command or executable path |
| `MUSESCORE_BIN` | Auto-detected | MuseScore executable path |

For macOS, a typical explicit path is `MUSESCORE_BIN="/Applications/MuseScore 4.app/Contents/MacOS/mscore"`. The readiness endpoint checks that packages and executables are present; a successful transcription also requires those tools to run.

Docker reads these size, duration, timeout, and retention settings from the root environment file through `docker-compose.yml`. Nginx streams uploads without a default body size cap, and the browser imposes no total upload deadline. Positive `MAX_UPLOAD_MB` values are enforced by the API and shown in the UI. Proxy timeouts apply to stalled connections, not the total recording length.

The frontend defaults to same-origin `/api`. To use a separate backend during development, put `VITE_API_BASE_URL=http://localhost:8000/api` in `frontend/.env`. Include the `/api` suffix and add the frontend origin to the backend's CORS settings.

## API

```bash
curl -i http://localhost:8000/api/upload \
  -F "file=@recording.wav" \
  -F "tempo_bpm=120" \
  -F "time_signature=4/4" \
  -F "grid=sixteenth"
```

Successful upload returns **202 Accepted**, a JSON job, and a `Location: /api/jobs/{job_id}` header. Tempo accepts 30–240 BPM, meter accepts `4/4`, `3/4`, or `6/8`, and grid accepts `eighth` or `sixteenth`.

| Endpoint | Result |
| --- | --- |
| `GET /api/health` | Dependency readiness and upload/retention limits |
| `POST /api/upload` | Accept multipart `file` and optional score settings |
| `GET /api/jobs/{job_id}` | Status, progress, errors, download URLs, and SVG page URLs |
| `GET /api/downloads/{job_id}/{kind}` | Download `pdf`, `musicxml`, `midi`, `svg`, or `svg_zip` |
| `GET /api/jobs/{job_id}/files/{name}` | Download an exact file from the job's artifact manifest |

The state sequence is `queued → preprocessing → transcribing → scoring → rendering → done`, or `failed`. Poll the status URL until a terminal state. `svg_pages` lists every available SVG preview page; the PDF is used when SVG preview is unavailable. `artifacts` lists every downloadable file, including completed stages of a processing or failed job. Add `?preview=true` for inline display. Downloads return 409 when the requested file is not yet ready, and expired or unavailable files return 404 after the job finishes. A full queue returns 429 with `Retry-After`.

## Storage and process model

SQLite persists the queue and job records. One coordinator processes one job at a time in a separate process, with an optional timeout. Run **one API worker and one backend replica per storage directory**. Uvicorn's development reload is supported, but restarting it interrupts the active job.

Queued jobs resume after a server restart. A job interrupted during processing becomes failed with a retry message; completed jobs stay downloadable until expiration. Original and normalized audio are deleted after processing, and a periodic cleanup removes expired job records and output files. Server-side `worker.log` files help diagnose native-tool failures and are never exposed by download routes.

Manual runs use `storage/jobs.sqlite3` and `storage/jobs/{job_id}/outputs/`. Docker uses its named `sheet-music-data` volume. Files from the previous in-memory implementation are left alone; old job statuses cannot be reconstructed automatically. The new Docker volume does not import the earlier `./storage` bind mount.

This MVP has no login or shared-user authorization. Job URLs act as access links. Docker binds to localhost by default; public hosting needs authentication, quotas, TLS, and a shared queue/storage design before scaling.

## Tests

Run the frontend upload regressions with `pnpm --dir frontend test` (Node.js 24). They cover unlimited uploads, empty recordings, positive limits and their boundaries, and MP3 filename/MIME variants. CI runs these tests before the frontend production build.

Run API, queue recovery, audio validation, notation, renderer-contract, and worker failure tests:

```bash
cd backend
uv sync --frozen --extra dev
uv run pytest -q
```

The native integration cases are skipped by default because they require the real Basic Pitch and MuseScore stack. With all dependencies installed:

```bash
cd backend
uv sync --frozen --extra transcription --extra dev
RUN_PIPELINE_INTEGRATION=1 uv run pytest -q tests/test_integration.py
```

The integration gate uploads a piano-like WAV, a 181-second stereo VBR MP3, a 0.125-second MP3, and a silent MP3 through FastAPI, waits for actual transcription/rendering, and downloads every format. It also checks that a long score produces all SVG pages. The default suite covers MP3 sample rates, channels, CBR/VBR, metadata, missing duration headers, chunk boundaries, dense polyphony, and recordings exceeding ten minutes. GitHub Actions runs unit tests, the frontend production build, and this native gate in the backend Docker image for pull requests and main-branch pushes.

To exercise a running app from the repository root:

```bash
python scripts/make_sample.py sample.wav
python scripts/test_transcription.py sample.wav
# Or:
python scripts/test_transcription.py /path/to/recording.mp3 --api http://localhost:8000
```

The Python runtime lock is `backend/uv.lock`. `backend/requirements-runtime.txt` is its transcription-enabled export used by Docker and pip. Regenerate it after dependency changes with `uv export --frozen --extra transcription --no-dev --no-emit-project --no-hashes --output-file requirements-runtime.txt` from `backend/`.

## File structure

| Path | Responsibility |
| --- | --- |
| `backend/main.py` | FastAPI app, readiness, coordinator lifecycle |
| `backend/app/routes/` | Upload, status, and manifest-only downloads |
| `backend/app/models/` | Job responses and score settings |
| `backend/app/services/` | SQLite, audio normalization, transcription, notation, rendering |
| `backend/app/workers/` | Queue coordinator and isolated job entry point |
| `backend/tests/` | Unit and opt-in native integration tests |
| `frontend/src/` | React upload, status, preview, and download UI |
| `frontend/nginx.conf` | Production SPA and API proxy |
| `scripts/` | Native-tool helper, sample generator, live API smoke test |
| `.github/workflows/ci.yml` | Backend, frontend, and native pipeline checks |

Core projects: [Basic Pitch](https://github.com/spotify/basic-pitch), [music21](https://www.music21.org/music21docs/), and [MuseScore CLI](https://musescore.org/en/handbook/3/command-line-options).
