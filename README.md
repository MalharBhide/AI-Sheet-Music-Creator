# AI Sheet Music Creator

A local MVP that accepts an audio upload and returns piano-focused sheet music outputs:

- MIDI from Basic Pitch
- MusicXML from music21 cleanup
- PDF and SVG from MuseScore CLI

The first version is intentionally scoped to clear piano or single-instrument audio. Full songs with vocals, drums, and dense mixes will run, but the notation quality will vary.

## Prerequisites

- Python 3.11
- Node.js 20+
- `ffmpeg`
- MuseScore with CLI access

Install system tools:

```bash
# macOS
brew install ffmpeg
brew install --cask musescore

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg musescore3
```

If MuseScore is not available as `mscore`, set `MUSESCORE_BIN` in `.env`.

## Backend

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

The first Basic Pitch run can take longer while model dependencies initialize.

## Frontend

In another terminal:

```bash
cd frontend
pnpm install
pnpm run dev
```

Open [http://localhost:5173](http://localhost:5173).

`npm install && npm run dev` also works if you prefer npm.

If pnpm asks about ignored build scripts for `esbuild`, run:

```bash
pnpm approve-builds esbuild
```

## Optional Pipeline Test

After installing backend dependencies:

```bash
source .venv/bin/activate
python scripts/test_transcription.py /path/to/audio.mp3
```

Outputs are written under `storage/jobs/<job_id>/outputs/`.

## API

- `POST /api/upload` with multipart field `file`
- `GET /api/jobs/{job_id}`
- `GET /api/downloads/{job_id}/midi`
- `GET /api/downloads/{job_id}/musicxml`
- `GET /api/downloads/{job_id}/pdf`
- `GET /api/downloads/{job_id}/svg`

## Notes

The backend uses an in-memory job store for the MVP. Restarting the API clears job status, although generated files remain under `storage/jobs`. For production, replace `app/services/job_store.py` with Redis, Postgres, or another persistent queue-backed store.

For better future output, add manual correction tools, key/time-signature controls, instrument separation, and a Celery/RQ worker so transcription jobs survive server restarts.
