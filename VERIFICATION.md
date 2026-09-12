# Verification

## Transcription studio overhaul — 2026-09-11

The application now uses a dedicated high-resolution piano model for solo piano, Demucs four-source separation before full-song arrangement, and a melody mode. Automatic tempo estimation, common onset-grid alignment, source-specific cleanup, and estimated key spelling replace the single fixed-tempo pipeline. The UI includes a complete studio, score playback with seeking/speed/volume, original-recording comparison, and a bundled original piano study.

- Final native ARM64 backend suite: **109 passed in 163 seconds**, including all six native integration cases. The cases exercise WAV, a 181-second stereo VBR MP3, a 0.125-second MP3, a silent MP3, full-song mode on MP3, and multipage rendering through real models and native tools. The regression suite also covers stereo preservation, mode validation, schema upgrades, tied-note playback, grid-phase correction, and more than 20,000 notes. The large-score fixture uses PrettyMIDI so the test measures the application rather than music21's slow fixture export. The earlier AMD64 regression run passed all 103 non-integration tests.
- Frontend: **21 tests passed**, including unlimited-upload regressions and a two-hour playback fixture. Production TypeScript/Vite builds passed.
- Backend lint and whitespace checks passed.
- Native ARM64 Docker image built successfully; all model imports and `pip check` passed. Compose now builds for the host architecture instead of forcing Intel emulation on Apple Silicon. The AMD64 image also built successfully.
- A real four-minute MP3 completed the full-song pipeline and produced a five-page PDF, five SVG pages, MusicXML, MIDI, SVG ZIP and playback JSON. Every file downloaded through Nginx; all 1,264 playback notes matched the canonical MIDI pitches, onsets and releases. Browser checks verified play/pause, 90-second seeking, speed selection, pagination and the responsive 390px layout.
- The benchmark uses real model inference on an original 96 BPM, 20-note soundfont-rendered study. The final piano engine matched 20/20 expected notes with 1 extra; the prior Basic Pitch path matched 20/20 with 6 extras. Pitch/onset F1 was 0.9756 vs 0.8696, using a 120 ms matching tolerance. Automatic tempo was 96.28 BPM and estimated key was correctly C major.
- A separate piano-plus-synthetic-percussion input completed real Demucs separation and transcription: 19/20 expected notes matched, 3 extra, F1=0.9048. This is a limited arrangement smoke test, not a representative commercial-song benchmark.
- An initial benchmark exposed missed notes at the start of a recording. Adding silent model context recovered both opening notes; the context is removed from output times.

These measurements do not establish accuracy on arbitrary music, exact note offsets, readable rhythm or hand assignment, or parity with Songscription. The piano engine took 248 seconds on the emulated Intel Docker runtime; that wall time included local verification overhead. Use the saved [measurements](docs/quality-benchmark.json), [reproduction script](scripts/benchmark_transcription.py), and [primary-source research](docs/transcription-research.md) for the scope and method.

## Website upload regression — 2026-09-10

The reported “Choose a non-empty file up to 0 MB” message was traced to the old frontend in commit `c725d9c`. The current backend uses zero to mean unlimited. Updated the local checkout from the already-merged GitHub main, extracted upload validation into a tested helper, and rebuilt both Docker services.

- All 14 frontend validation tests passed; they now run in CI. Production TypeScript/Vite build passed.
- Live Nginx checks confirmed `Cache-Control: no-store` for `/`, `/index.html`, SPA routes, and API responses. Hashed assets are immutable, and missing old assets return 404 instead of HTML.
- The browser showed `Feel No Ways.mp3` in `done` state with seven SVG pages. Every artifact downloaded successfully through the actual Nginx endpoint.

## Pipeline verification — 2026-09-09

Verified locally on 2026-09-09 using the app's Linux amd64 Docker runtime, including real FFmpeg, Basic Pitch 0.4.0, music21 9.7.1, and MuseScore 3.

- Backend regression suite: **72 passed**. Native integration cases are opt-in and skipped by the default command.
- Native integration suite: **5 passed**. Full FastAPI upload, worker process, transcription, notation, rendering, and every download were verified for a generated WAV, a 181.125-second stereo VBR MP3, a 0.125-second MP3, and a silent MP3. The fifth case verifies multi-page PDF/SVG parity.
- Additional real Basic Pitch check: a 65.125-second stereo VBR MP3 retained notes beyond 60 seconds across both chunk boundaries.
- FFmpeg coverage includes eight combinations of 8–48 kHz sample rates, mono/stereo, CBR/VBR, ID3 metadata, headerless MP3, RF64, sub-quarter-second audio, and complete decoding of a 601.125-second recording.
- Notation regressions preserve dense overlapping notes, repeated attacks, per-pitch/barline ties, selected meter, silent time, and scores containing more than 20,000 notes.
- Worker/API regressions verify intermediate downloads survive a later rendering failure, optional positive limits still work, and zero disables size/duration/processing deadlines.
- Renderer regressions preserve the complete PDF when optional SVG export fails or omits pages.
- Frontend TypeScript and Vite production build passed. Backend Ruff and `git diff --check` passed.
- Headless browser checks passed for a 26 MB MP3 upload, continued polling after six failed status requests, visible MIDI downloads after rendering failure, and PDF preview when SVG is unavailable.
- The rebuilt Docker app passed a fresh stereo VBR MP3 upload through its actual Nginx endpoint on port 5173; the job completed and all five output formats downloaded successfully.

The saved production job that previously failed with “The recording is too dense to notate clearly” was rerun from its retained MIDI through the corrected notation and real MuseScore renderer, producing PDF, MusicXML, MIDI, SVG pages, and a ZIP. The other saved failure was the former 180-second audio limit; its original audio had already been deleted and needs a new upload.

There are no default recording length or file size caps. Audio/model memory is bounded by chunk size; accumulated notes, score layout, disk usage, and total runtime still grow with the recording. Invalid or undecodable audio reports an error. These checks verify output generation and timing preservation, not musical transcription accuracy for arbitrary songs or instrument mixtures.

To reproduce:

```bash
cd backend
uv sync --frozen --extra transcription --extra dev
uv run pytest -q
RUN_PIPELINE_INTEGRATION=1 uv run pytest -q tests/test_integration.py
uv run ruff check app tests main.py
```

The GitHub Actions native-pipeline job runs the same integration suite in the backend Docker image.
