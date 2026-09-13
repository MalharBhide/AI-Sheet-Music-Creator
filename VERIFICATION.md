# Verification

## Trained vocal melody model — 2026-09-13

Built the [audio-to-piano framework](docs/transcription-framework.md), downloaded
checksum-verified CC BY 4.0 Vocadito recordings, and actually trained a
2,187-parameter temporal neural decoder for 30 epochs/1,800 updates. Training
uses 26 recordings; eight validation and six test recordings have separate
singer IDs. The shipped checkpoint is used for **every Full song upload's vocal
melody**, independent of song name or file format. No user recording was used
for fitting or choosing model settings.

- Held-out real singing, macro averages: correct pitch **63.1% → 73.4%**, false
  silence **23.6% → 15.0%**, note-onset F1 **.525 → .622**, onset/offset F1
  **.388 → .504**. Every test recording improved in pitch recall and onset F1.
- The frozen model also improved on held-out real voices mixed with procedural
  accompaniment and passed through Demucs: onset F1 **.518 → .618**, false
  silence **22.8% → 15.1%**. This controlled stress test is not a benchmark of
  commercial-song arrangements. The test singers are the same as the clean test.
- Training data, exact splits, optimizer history, checkpoint hash, individual
  failures, two annotators' results and reproduction commands are retained in
  the [model card](docs/melody-model-card.md) and [training directory](training/README.md).
- **145 regression checks passed**, including eight decoder contracts; **all
  six native integration checks passed** through real models, FFmpeg and
  MuseScore. Native coverage includes WAV, 181-second stereo VBR MP3, 0.125-second
  MP3, silence, full-song MP3 and multipage rendering. Backend/training lint and
  diff whitespace checks passed. Targeted readiness/routing tests also passed
  after adding the packaged-model check.
- The model is packaged in the backend wheel, verified by SHA-256 at runtime,
  and loaded with `weights_only=True`. Existing overlapping audio windows keep
  model input bounded; there is no new recording-length or upload-size cap.
- Rebuilt the local backend and regenerated the user's complete 240.7-second
  MP3 at the same 97 BPM/eighth-note settings as the prior score. Job
  `6c14a15d-72d7-4d2f-9524-0c8f94c0a1b8` completed with the new model, four score
  pages and 1,090 playback notes. All six download formats were nonempty.
  Browser checks confirmed playback, click-to-seek and keyboard seeking.
  This unannotated recording was an operational check, not training data or an
  accuracy benchmark. The previous result remains available for comparison.

These tests establish a measured improvement in vocal recognition and working
exports. They do not establish that a full commercial mix has become an accurate
piano arrangement. Six held-out recordings are a small test set; further tuning
requires new independent test data and labeled real mixtures.

## Vocal refinement and beat interpretation — 2026-09-13

The previous full-song melody selector could replace a sustained vocal fundamental with a later overtone attack, or re-attack the same piano key on each vibrato cycle. Isolated vocals now receive an independent pYIN pitch check before quantization. Conflicting harmonics are rejected when tracking is reliable; supported note spans trim outer tails. Repeated same-pitch detections join only with continuous voicing and no renewed amplitude attack. Uncertain tracking retains the neural event. The weights and polyphonic piano/accompaniment engines are unchanged.

- Seven original synthesized vocal-like fixtures contain **70 reference notes**. Refinement reduced detected events from **102 to 68**, preserving all **65 pitch/onset matches** from the baseline. Pooled pitch/onset F1 improved from **0.7558 to 0.9420**; adding offset constraints improved F1 from **0.4767 to 0.8696** (41 to 60 complete-note matches). The same neural predictions and quantization are used on both sides. The breath-like-noise fixture still has missed or mistimed onsets. These are narrow regression fixtures, not evidence of general singing or commercial-song accuracy. [Results](docs/vocal-quality-benchmark.json), [reproduction script](scripts/benchmark_vocals.py).
- The old tempo tracker chose approximately 129 BPM on a known 96 BPM synthetic pulse with a competing rhythmic layer. Comparing acoustic support across beat hypotheses returns approximately 96 BPM. Robust slope fitting and alignment of clear half/double-time estimates reduce timing errors between excerpts. The user's recording now yields **97 BPM**, compared with 129.4 BPM on the earlier score; this is an audio-derived beat interpretation, not an annotated ground-truth measurement.
- **140 backend tests passed**, including all six native pipeline cases through real models, FFmpeg and MuseScore. New regressions cover false overtone attacks, genuine octave jumps, repeated articulation, breath gaps, held-note tails, uncertain tracking, vibrato and competing beat hypotheses. Backend lint passed. No frontend code changed in this update.
- A fresh full-length upload of the user's recording completed through the rebuilt Nginx/API deployment at **97 BPM**, with **1,303 cleaned detections and seven score pages**. All six output formats were nonempty and downloadable; playback contained position maps for every page. Browser checks confirmed play/pause and synchronized keyboard and pointer seeking. The original piano study still estimates at its known 96 BPM. These checks verify the delivered result's operation; the full-song notes do not have an annotated reference for an accuracy score.

## Score following, click-to-seek and simpler studio — 2026-09-13

The player and sheet preview now share the audio clock and MuseScore's engraved segment positions. A green cursor follows musical onsets, scrolls between staff systems, and turns pages. Clicking the score seeks to the nearest onset on that page and staff system while preserving play/pause. The layout puts playback beside the score, emphasizes PDF download, and collapses advanced settings, secondary formats, transcription notes and history.

- Backend regression suite: **124 passed**, with six native cases deselected. Frontend: **26 tests passed**, with TypeScript/Vite production build and backend lint passing.
- The updated native multipage renderer test passed through real music21 and MuseScore. It checks automatic position export across every PDF/SVG page and verifies that half-beat note timestamps stay on the same 120 BPM audio clock. Audio-model inference was not rerun for this presentation change.
- Added timing, nearest-onset, invalid-map and optional-renderer-failure regressions. A missing position map preserves audio and downloads instead of displaying a guessed cursor.
- Added position maps to all three existing completed local scores without rerunning transcription or changing their notes. Browser checks on the eight-page result verified paused and playing click-to-seek, keyboard note navigation, seeking across pages, automatic following, manual page browsing and resuming following.
- Checked the score and upload layouts at desktop and 390px phone width. Zoomed score clicking works with no horizontal overflow of the overall mobile page. A navigation check exposed duplicate React component keys that could leave the previous player visible after **New score**; player and preview now have separate keys.

This change synchronizes the existing synthesized score with its notation. It does not change transcription accuracy or the piano sound engine. Scores rendered without MuseScore 3 segment positions retain manual page navigation.

## Piano sustain, dynamics and export fixes — 2026-09-12

The reported held-note sound had several concrete causes. Browser synthesis decayed to a constant sustain floor, score export flattened all velocities to 90, and music21's generic MIDI export re-attacked individual pitches when a tied chord changed membership. The revised exporter follows each pitch's ties within its staff/voice, preserves per-note MusicXML dynamics, and retains meter/key metadata. Full-song arrangements now balance the vocal melody above bass and accompaniment. The synthesized piano decays toward silence while held, damps on release, and resumes its existing decay when seeking.

- Re-exporting the user's existing MusicXML reduced playback from **1,951 to 1,359 attacks**, removing **592 false attacks** without rerunning the model or deleting its written pitches. A controlled dense passage now retains eight original attacks instead of manufacturing 64.
- A locally tested 45-second excerpt produced 208 source notes and 207 score notes (one coincident duplicate merged), with 53 distinct velocities preserved through MusicXML and MIDI. No user recording or excerpt is checked into the repository.
- A fresh full-length upload through the rebuilt website completed with **1,359 notes, eight pages and 66 distinct velocities (40–106)**. All six download formats returned valid, nonempty files; PDF/SVG/ZIP page counts agreed. Canonical MIDI and browser playback matched every pitch and velocity, with onset/release differences no greater than one microsecond from JSON rounding.
- Browser checks confirmed play, pause and seeking to 90 seconds on the regenerated score. The website was left on the new result, paused at the beginning.
- Playback tests cover sparse and dense chords, repeated keys, independent unison releases, per-note velocities, meter/key, fractional tempo encoding, and a sparse note beyond six hours. Canonical export no longer reparses MIDI into a dense tick-to-time array proportional to recording length. This is an export regression, not a six-hour model-inference benchmark.
- Final backend regression suite: **117 passed**. All six native pipeline cases completed, covering WAV, long/short/silent MP3, full-song mode, and multipage rendering. Frontend: **23 tests passed**, with TypeScript/Vite production build and backend lint passing.
- Native testing also exposed a shutdown race: the coordinator could overwrite a completed result while its worker was exiting. Failure updates now atomically preserve terminal results and the original worker error. Dedicated shutdown regressions pass, including a fresh native MP3/API check asserting the saved result remains completed after shutdown.
- Updated the development lockfile and native CI test install to use `httpx2`, required by the pinned Starlette runtime's test client.

These checks establish timing/export correctness and a more useful piano balance. They do not establish that all detected pitches and rhythms match the original song. Full-mix transcription remains an approximate arrangement, and the browser instrument remains synthesized rather than a sampled acoustic piano.

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
