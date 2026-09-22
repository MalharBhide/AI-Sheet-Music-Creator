# Piano Scribe

Piano Scribe turns recordings into editable piano sheet music, with a browser studio for choosing the right transcription mode, reviewing the score, and listening to the notes before downloading. It exports PDF, MusicXML, MIDI, and SVG. The React/Vite frontend talks to a FastAPI backend with a durable job queue, local transcription models, and MuseScore rendering.

The transcription design is informed by an [extensive review of Songscription, Klangio, AnthemScore, ScoreCloud, and open-source models](docs/transcription-research.md). The report distinguishes documented product features from measured accuracy. A completed PDF is a working output, not evidence that every note is correct.

Balanced full-song accompaniment now also uses a trained note verifier to reject unsupported candidate notes. See its [model card and measured limits](docs/accompaniment-verifier.md). Re-upload a recording to apply it to an older score.

The latest [V4 residual classifier](docs/accompaniment-residual-v4.md) learns from
false notes that survive the existing filter. It removes a further 23 false notes
on 126 regression excerpts while preserving all previously matched notes; the
newly scored dataset sections were unchanged. This is a modest measured gain.

Balanced arrangements give the detected melody priority: backing stays below it
and uses at most two simultaneous voices, alongside the separate bass. The
reduction selects complete notes instead of cutting a held note short whenever
a louder note appears. When no vocal melody is detected, three backing voices
remain available. This simplifies the arrangement; it does not certify that the
detected melody or harmony is correct.

Full-song melody recognition now includes a locally trained neural decoder. See the [transcription framework](docs/transcription-framework.md), [training workflow](training/README.md), and [model card with held-out results](docs/melody-model-card.md).

## Quick start with Docker

Install and open [Docker Desktop](https://www.docker.com/products/docker-desktop/), then run from this repository:

```bash
docker compose up --build
```

Open **[http://localhost:5173](http://localhost:5173)**. Interactive API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

The first build installs the Python dependencies, FFmpeg, and MuseScore and downloads the piano and source-separation model checkpoints. Later starts of the same version only need `docker compose up`. After pulling changes or switching branches, rebuild with `docker compose up -d --build`, then reload the browser. Restarting containers alone keeps their previously built code. Stop with Ctrl+C or `docker compose down`; job records and outputs remain in the named Docker volume until expiration. Removing that volume deletes them.

Docker includes Python 3.11 and Debian Bookworm's [MuseScore 3 package](https://packages.debian.org/bookworm/musescore3), and serves a production frontend build through Nginx. The backend builds for the host architecture: native ARM64 on Apple Silicon and AMD64 on Intel/AMD hosts. The Linux ARM runtime uses TensorFlow's AWS CPU package with the same pinned TensorFlow and PyTorch versions. Allow several GB of memory and disk space. Full-song separation is substantially more work than a short solo-piano transcription.

## Using the studio

1. Drop or choose a WAV, MP3, FLAC, OGG, M4A, AAC, or AIFF recording. **Try an example instead** submits an original 96 BPM piano study through the real transcription pipeline. The bundled recording is rendered by MuseScore from the reproducible quality fixture in `scripts/benchmark_transcription.py`.
2. Select **Full song**, **Solo piano**, or **Melody**. Expand **More options** to adjust notation detail, tempo, time signature, or the note grid.
3. Click **Create piano score**. The studio shows actual upload and processing progress and reconnects after temporary network interruptions.
4. Press **Play** to hear the score. A green cursor follows the music and turns pages automatically. Click a note to jump to that moment; playback continues if playing and stays paused if paused. The score also supports left/right arrow keys. Page controls let you browse independently; **Follow playback** returns to the current music. Zoom and playback speed remain available beside the score. For a recording uploaded in the current session, expand the original-audio comparison to listen to the source.
5. **Download sheet music** saves the PDF. Expand **Other formats** for editable MusicXML, MIDI, the first SVG page, or a ZIP of every SVG page.

The URL contains the job ID, so reloading or returning to that URL restores the job. Recent transcription links are stored on this device in the browser's local storage. Generated files are available for **24 hours after processing finishes** by default; an older history entry may remain after its files expire. Original-audio comparison is available only while the uploaded file remains in the current page session. The server deletes source audio after processing.

There is **no default upload-size, recording-duration, or processing-time cap**. Zero in the limit settings means unlimited, not a 0 MB allowance. FFmpeg decodes supported audio, including MP3, and transcription uses overlapping windows across the recording. Audio buffers are bounded rather than loading an entire long recording into the model at once. Long jobs still require sufficient time, disk, and memory for models, accumulated notes, and score rendering. Corrupt or undecodable files receive an error; no system can guarantee valid output for every possible file or unlimited duration.

If an old open tab still displays **“Choose a non-empty file up to 0 MB”**, rebuild the app and hard-refresh that tab with Cmd+Shift+R on macOS or Ctrl+Shift+R on Windows/Linux. The updated frontend is served without page caching.

## Transcription modes and musical limits

| Mode | Pipeline | Best use |
| --- | --- | --- |
| **Solo piano** (`piano`) | High-resolution piano CRNN → note cleanup → grand-staff notation | Clear recordings of piano notes; uses a dedicated piano onset/offset model |
| **Full song** (`full_mix`) | Demucs → trained vocal melody decoder + Basic Pitch support → piano reduction | A playable draft from vocals, bass and accompaniment; drums are excluded from pitched notation |
| **Melody** (`melody`) | Basic Pitch → single-line selection and cleanup | A clear solo instrument or isolated melody; this mode does not perform source separation |

The website initially selects **Full song**. The API defaults to **Solo piano** when `transcription_mode` is omitted, preserving compatibility with existing callers.

Full-song mode creates a piano arrangement of detected material. Separated accompaniment can contain several instruments and artifacts; it is not an isolated original piano part. Balanced detail reduces clutter, while more detail keeps additional detections. Both need musical review, especially for dense mixes, heavy reverb, pedal, repeated notes, and overlapping vocals.

The vocal decoder learns pitch, silence and note attacks from real annotated singing. It combines continuous Basic Pitch, pYIN, spectral and energy evidence, allowing it to recover melody that the previous event-filtering approach missed. Its small checked checkpoint ships with the backend and uses bounded audio windows. On six held-out Vocadito recordings, correct melody pitch improved from 63.1% to 73.4%, and missed melody fell from 23.6% to 15.0%. A controlled source-separation stress test also improved. These measure the vocal recognition stage, not complete commercial-song or piano-arrangement accuracy. Re-upload an older recording to use the new model.

Tempo is estimated automatically from bounded excerpts unless you supply a 30–240 BPM override. The estimator compares several beat-speed hypotheses against onset strength and coverage, fits timing robustly, and reconciles clear half/double-time differences between excerpts. If a steady tempo cannot be found, the system uses 120 BPM and reports a warning. The score currently uses one tempo throughout, so rubato, swing, tempo changes, and beat-level interpretations may need manual correction. Time signature is selected by the user, with `4/4` as the default; it is not automatically detected.

The historical [vocal regression benchmark](docs/vocal-quality-benchmark.json) for the previous event-filtering engine compares identical neural predictions with and without refinement on seven original synthesized phrases. Across 70 reference notes, pitch/onset F1 increased from **0.7558 to 0.9420** and pitch/onset/offset F1 from **0.4767 to 0.8696**. These controlled fixtures cover harmonics, vibrato, breath-like noise and repeated attacks; they do not measure real singing or full-song arrangement accuracy. Reproduce them with `PYTHONPATH=backend python scripts/benchmark_vocals.py /tmp/vocal-quality` in the complete backend runtime.

The system estimates a key from detected pitch durations when the evidence is sufficient. This guides notation and does not change detected pitches. Rhythm is quantized to the chosen grid, with cleanup for duplicate detections and overlapping notes. Grand-staff notation preserves chords, rests, voices, and barline ties, but hand assignment, spelling, beat alignment, and phrasing may still need editing.

MusicXML is the final notation source. Its exported MIDI and playback JSON drive the website's synthesized piano, so completed-score playback agrees with the score rather than an unrelated raw model output. MIDI is also made available earlier in processing as a recoverable intermediate; if a job fails before score export finishes, that partial MIDI may not yet include final notation changes. Available artifacts remain downloadable when a later stage fails. Silence or a source with no detected pitches can produce a rest-only score with a warning.

Score export follows ties for each pitch, including chords whose membership changes, and preserves individual note velocities. Full-song arrangements place the vocal melody above a quieter bass and accompaniment while retaining local dynamic variation. Browser piano tones decay while a key is held and damp on release; seeking resumes the existing decay. These changes improve playback and arrangement balance, but do not correct every pitch or rhythm detected by the model. Previously completed files need a new transcription to include the export and balance changes.

## Manual development setup

Use **Python 3.11** and **Node.js 24**. The transcription dependency set is pinned for Python 3.11; Python 3.12 is not supported by this setup.

Install FFmpeg and the MuseScore CLI:

```bash
# Debian / Ubuntu with the musescore3 package
sudo apt-get update
sudo apt-get install ffmpeg musescore3

# macOS
brew install ffmpeg
brew install --cask musescore
```

MuseScore 3 is the target server renderer. Some MuseScore 4 releases export only the first SVG page from the CLI. If SVG export is incomplete or fails, the app keeps the valid PDF and previews it directly. Use Docker if your native installation cannot run with Qt's offscreen platform.

From the repository root:

```bash
cp .env.example .env
python3.11 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m pip install --no-deps -e backend
cd backend
python -m app.services.model_assets
uvicorn main:app --reload --port 8000
```

The model setup downloads the piano checkpoint to `~/.cache/piano-scribe/piano.pth` and caches Demucs `htdemucs` weights before processing jobs. To use a different piano checkpoint location, export `PIANO_MODEL_PATH=/absolute/path/to/piano.pth` before running both model setup and the server. `TORCH_HOME` can relocate PyTorch/Demucs's cache. Docker preloads both models during its build and sets the corresponding paths.

On Windows, use `py -3.11 -m venv backend/.venv` and `backend\.venv\Scripts\Activate.ps1`; Docker is the easiest way to provide the native tools.

Alternatively, using [uv](https://docs.astral.sh/uv/) from `backend/`:

```bash
uv sync --frozen --extra transcription --extra dev
uv run python -m app.services.model_assets
uv run uvicorn main:app --reload --port 8000
```

In a second terminal, from the repository root:

```bash
npm install -g pnpm@11.19.0
pnpm install --frozen-lockfile
pnpm --dir frontend dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite development proxy forwards `/api` to port 8000. Build the frontend with `pnpm --dir frontend build`.

## Configuration

The backend reads root `.env` and then `backend/.env`; environment variables take precedence. See `.env.example`. The standalone model setup command reads exported environment variables, so export a custom `PIANO_MODEL_PATH` when downloading to a nondefault location.

| Variable | Default | Purpose |
| --- | --- | --- |
| `STORAGE_ROOT` | `storage` | Database and job directories; relative paths resolve from the repository root |
| `CORS_ORIGINS` | localhost and 127.0.0.1 on port 5173 | Comma-separated origins or a JSON array |
| `MAX_UPLOAD_MB` | 0 | Optional file-size limit; 0 disables it |
| `MAX_AUDIO_SECONDS` | 0 | Optional decoded-duration limit; 0 disables it |
| `MAX_QUEUED_JOBS` | 10 | Maximum jobs waiting to be processed |
| `JOB_TIMEOUT_SECONDS` | 0 | Optional overall processing deadline; 0 disables it |
| `RENDER_TIMEOUT_SECONDS` | 0 | Optional deadline for each MuseScore export; 0 disables it |
| `RETENTION_HOURS` | 24 | File and job retention after processing finishes |
| `PIANO_MODEL_PATH` | `~/.cache/piano-scribe/piano.pth` | Piano checkpoint; Docker uses `/opt/models/piano.pth` |
| `TORCH_HOME` | PyTorch's cache default | Optional model-cache location; Docker uses `/opt/models/torch` |
| `FFMPEG_BIN` | `ffmpeg` | FFmpeg command or executable path |
| `MUSESCORE_BIN` | Auto-detected | MuseScore executable path |

For macOS, a typical explicit renderer path is `MUSESCORE_BIN="/Applications/MuseScore 4.app/Contents/MacOS/mscore"`. The readiness endpoint checks required dependencies and the piano checkpoint; completing a transcription also requires sufficient resources and successfully running the native tools.

Docker passes size, duration, timeout, and retention settings from the root environment file through `docker-compose.yml`. Nginx streams uploads without a default body-size cap, and the browser imposes no total upload deadline. Positive `MAX_UPLOAD_MB` values are enforced by the API and shown in the studio. Proxy timeouts apply to stalled connections, not the total recording length.

The frontend defaults to same-origin `/api`. To use a separate backend during development, put `VITE_API_BASE_URL=http://localhost:8000/api` in `frontend/.env`. Include the `/api` suffix and add the frontend origin to the backend's CORS settings.

## API

Create a full-song arrangement with automatic tempo:

```bash
curl -i http://localhost:8000/api/upload \
  -F "file=@recording.mp3" \
  -F "transcription_mode=full_mix" \
  -F "detail=balanced" \
  -F "time_signature=4/4" \
  -F "grid=sixteenth"
```

To override tempo, add `-F "tempo_bpm=120"`. Omit the field for automatic estimation.

| Upload field | Accepted values | Default |
| --- | --- | --- |
| `file` | Supported nonempty audio file | Required |
| `transcription_mode` | `piano`, `full_mix`, `melody` | `piano` |
| `detail` | `balanced`, `detailed` | `balanced` |
| `tempo_bpm` | 30–240 | Automatic estimation |
| `time_signature` | `4/4`, `3/4`, `6/8` | `4/4` |
| `grid` | `eighth`, `sixteenth` | `sixteenth` |

Successful upload returns **202 Accepted**, a JSON job, and a `Location: /api/jobs/{job_id}` header.

| Endpoint | Result |
| --- | --- |
| `GET /api/health` | Dependency readiness and upload/retention limits |
| `GET /api/example` | Generated short piano-study WAV for a real example transcription |
| `POST /api/upload` | Accept multipart `file` and optional score settings |
| `GET /api/jobs/{job_id}` | Status, progress, settings, analysis, errors, download URLs, and SVG page URLs |
| `GET /api/downloads/{job_id}/{kind}` | Download `pdf`, `musicxml`, `midi`, `svg`, `svg_zip`, or `playback` |
| `GET /api/jobs/{job_id}/files/{name}` | Download an exact file from the job's artifact manifest |

The public state sequence is `queued → preprocessing → transcribing → scoring → rendering → done`, or `failed`. Poll until a terminal state. `analysis` includes the selected engine, tempo, optional estimated key, note counts, source roles, duration, and musical warnings when available. `svg_pages` lists available preview pages; PDF is used when SVG preview is unavailable. `artifacts` lists downloadable files, including completed stages of processing or failed jobs. Add `?preview=true` for inline display.

`download_urls.playback` provides JSON in this shape, with timing in seconds and MIDI pitches/velocities:

```json
{
  "duration": 8.0,
  "tempo_bpm": 120.0,
  "notes": [{ "pitch": 60, "start": 0.0, "end": 0.5, "velocity": 80 }],
  "positions": [{ "time": 0.0, "page": 0, "x": 0.2, "y": 0.15, "height": 0.1 }]
}
```

`positions` is optional: MuseScore 3's segment-position export supplies onset times in seconds, zero-based page numbers, and coordinates normalized to each SVG page. The cursor spans the staff system. Invalid or unavailable maps leave audio and downloads usable, with manual page browsing. The Docker renderer exports positions at an explicit 300 DPI to match its SVG coordinate scale. To add following to unexpired scores created before this feature, without rerunning transcription:

```bash
docker compose exec backend python -m app.services.score_positions
```

Downloads return 409 when a requested file is not yet ready. Expired or unavailable files return 404 after the job finishes. A full queue returns 429 with `Retry-After`.

## Storage and process model

SQLite persists the queue and job records. One coordinator processes one job at a time in a separate process, with an optional timeout. Run **one API worker and one backend replica per storage directory**. Uvicorn's development reload is supported, but restarting it interrupts the active job.

Queued jobs resume after a server restart. A job interrupted during processing becomes failed with a retry message; completed jobs stay downloadable until expiration. Original and normalized audio are deleted after processing, and a periodic cleanup removes expired job records and output files. Server-side `worker.log` files help diagnose native-tool failures and are never exposed by download routes.

Manual runs use `storage/jobs.sqlite3` and `storage/jobs/{job_id}/outputs/`. Docker uses its named `sheet-music-data` volume and does not automatically import files from an older local bind mount. Browser history contains links and filenames, not copies of generated audio or scores.

The app currently has no login or shared-user authorization. Job URLs act as access links. Docker binds to localhost by default. A public multi-user deployment needs authentication, quotas, TLS, and an appropriate shared queue/storage design.

## Tests

Run frontend tests and the production build:

```bash
pnpm --dir frontend test
pnpm --dir frontend build
```

Frontend regressions cover unlimited uploads, empty recordings, positive size limits, and MP3 filename/MIME variants. Playback tests cover note normalization, long-duration time labels, speed and position math, seeking into held notes, and bounded voice scheduling with pause cleanup.

Run backend API, queue recovery, audio validation, transcription, musical-analysis, notation, score-playback, renderer-contract, and worker-failure tests:

```bash
cd backend
uv sync --frozen --extra dev
uv run pytest -q
```

Native integration cases are skipped by default because they require the real models, FFmpeg, and MuseScore. With the complete dependencies and model assets installed:

```bash
cd backend
uv sync --frozen --extra transcription --extra dev
uv run python -m app.services.model_assets
RUN_PIPELINE_INTEGRATION=1 uv run pytest -q tests/test_integration.py
```

The native gate uploads a piano-like WAV, a stereo VBR MP3 longer than three minutes, a very short MP3, and a silent MP3 through FastAPI, waits for actual transcription/rendering, and downloads the generated formats, including playback JSON. It also checks multipage SVG output. The default suite covers decoding variants, chunk boundaries, musical cleanup and analysis, dense polyphony, artifact preservation, and agreement between final notation and playback. These are reliability and regression checks, not a controlled accuracy comparison with commercial products. GitHub Actions runs backend tests, frontend tests/build, and the native gate in the backend Docker image for pull requests and main-branch pushes.

To exercise a running app from the repository root:

```bash
python scripts/make_sample.py sample.wav
python scripts/test_transcription.py sample.wav
# Or:
python scripts/test_transcription.py /path/to/recording.mp3 --api http://localhost:8000
```

The Python runtime lock is `backend/uv.lock`. `backend/requirements-runtime.txt` is its transcription-enabled export used by Docker and pip. Regenerate it from `backend/` after dependency changes:

```bash
uv export --frozen --extra transcription --no-dev --no-emit-project \
  --no-hashes --emit-index-url --output-file requirements-runtime.txt
```

Keep `--emit-index-url`: the export needs the pinned PyTorch CPU package index on Linux.

## File structure

| Path | Responsibility |
| --- | --- |
| `backend/main.py` | FastAPI app, readiness, coordinator lifecycle |
| `backend/app/routes/` | Upload, status, example audio, and manifest-only downloads |
| `backend/app/models/` | Job responses and score settings |
| `backend/app/services/` | Storage, normalization, model assets, separation, transcription, musical analysis, notation, playback export, rendering |
| `backend/app/workers/` | Queue coordinator and isolated job entry point |
| `backend/tests/` | Unit and opt-in native integration tests |
| `frontend/src/` | React studio, recording setup, history, preview, playback, and downloads |
| `frontend/src/playback.ts` | Web Audio synthesis, bounded scheduler, and timing helpers |
| `frontend/nginx.conf` | Production SPA and API proxy |
| `scripts/` | Native-tool helper, sample generator, live API smoke test |
| `docs/transcription-research.md` | Product/model research, sources, design recommendations, and evaluation approach |
| `.github/workflows/ci.yml` | Backend, frontend, and native pipeline checks |

Core projects: [high-resolution piano transcription](https://github.com/qiuqiangkong/piano_transcription_inference), [Demucs](https://github.com/facebookresearch/demucs), [Basic Pitch](https://github.com/spotify/basic-pitch), [music21](https://www.music21.org/music21docs/), and [MuseScore CLI](https://musescore.org/en/handbook/3/command-line-options). The piano checkpoint by Qiuqiang Kong is distributed under [CC BY 4.0 in its official model record](https://zenodo.org/records/4034264); the research report records the distinction between model and code licenses.
