# Verification

## Relationship-aware model experiment — not released

Added 56 licensed piano training passages and 15 validation passages, retaining
the existing performer splits and audited clocks. Trained a model with 20 new
relative note-relationship features. Corrected a cropped-cache context mismatch
by using a two-second neighborhood and protecting incomplete-context edge notes.

The frozen final candidate removed **20 false notes** on 142 regression excerpts
but also lost **one correct note**. It failed the per-recording preservation gate
and was not deployed. An earlier candidate and the superseded unbounded-feature
attempt are archived as well. V4 weights and website behavior remain unchanged.
No user recordings, score regeneration or website transcription jobs were used.
See the [experiment report](docs/relational-verifier-experiment.md) for metrics,
data scope, rejected checkpoints and reproducibility.
**13 focused model-workflow tests passed**, including crop/full-context parity,
edge-note preservation, transposition and event-order invariance, and comparison
against the actual V4 baseline. Training-code lint and diff checks passed.

## Trained residual false-note filter — 2026-09-22

Trained a 300-tree classifier on 50,442 labeled, V3-retained candidate events
from 431 licensed/original training clips. V3 decisions are preserved unless an
exact shared, uncertain candidate is rejected by the new model. No user uploads
were processed and no website transcription jobs or scores were created.

- Validation: **3,247 → 3,214 false notes**, retaining all 10,316 matched notes.
  The threshold was halved as a predeclared safety margin, then frozen.
- Consumed regression: **3,032 → 3,009 false notes (0.76%)**, retaining all
  12,301 matched notes in every one of 126 excerpts. The newly scored 16 later
  GuitarSet sections were unchanged: 578 matches and 299 false notes. They share
  test performers/compositions with earlier sections, so they are not an
  independent generalization benchmark. See the [model card](docs/accompaniment-residual-v4.md).
- Runtime parity passed on **31,156 candidate events across 250 clips**:
  actual deployed filter decisions match offline evaluation, with all retained
  note objects, timing, pitches, dynamics and source assignments preserved.
  No audio inference or score generation was needed for this check.
- **168 tests passed**; six audio-to-score integration cases were deliberately
  skipped. Backend/model-training lint passed. Checks cover damaged weights,
  invalid confidence values, protected notes, exact decoder matching, runtime
  filtering, and rejecting candidates that lose correct held notes.
- Rebuilt and deployed the backend after confirming the queue was idle. The
  deployed process loaded V4 at threshold .0375 with a verified checkpoint hash,
  and the frontend-proxied health endpoint reported ready. No upload or score
  regeneration was performed for deployment verification.

This is a new trained accompaniment model, not another reduction in arrangement
density. It does not change melody inference or claim accurate full-song scores.

## Melody-first reduction and held-note preservation — 2026-09-21

The previous accompaniment reducer interrupted an active note whenever a louder
candidate arrived. This could convert a detected one-second hold into a short
fragment. It now selects complete supporting lines using duration and velocity,
preserving the detected onset and release of every selected note. Balanced
full-song arrangements keep at most two supporting voices below the melody;
bass remains separate. With no detected vocal melody, three backing voices
remain available. Detailed mode retains its five-voice allowance. Passages
already within their voice allowance are not thinned further.

The chunk merger also discarded a detected right-context tail when the next
window missed the continuation. It now retains that already-detected tail,
bounded by the next core and recording end, while preserving new attacks on the
same key. This does not extrapolate an undetected sustain through silence.

- On two cached diagnostic excerpts, accompaniment counts changed **143 → 83**
  and **80 → 48**. All 69 and 53 melody events respectively retained exactly the
  same pitches, onsets, releases and dynamics. Reduction-induced shortened notes
  changed **18 → 0** and **2 → 0**. Six retained notes in the first excerpt regained
  their detected holds, including one changed from approximately 0.28 to 1.10
  seconds. [Comparison record](docs/arrangement-sustain-check.json).
- These are arrangement comparisons on identical raw detections, not ground-truth
  pitch-accuracy measurements. Fewer backing notes do not prove that all removed
  pitches were incorrect, and the detected melody can still contain mistakes.
- Focused tests cover complete held-note selection, active-melody priority,
  preserving already-sparse passages, missed seam continuations, genuine repeated
  attacks, final-file clipping and the revised full-song source routing.
- **162 backend tests passed**, including all six native pipeline cases through
  real models, FFmpeg and MuseScore. Backend and diagnostic-helper lint passed.
- Rebuilt the local backend and regenerated the full 145-second user recording
  at the same 163 BPM and sixteenth-note grid. Job
  `33ec59af-04c0-4d95-99a9-fe902ad5540b` completed with **335 backing notes**
  (previously 459), **286 melody notes** (previously 285), and the same 102 bass
  notes. Its 722 playback notes have 980 score positions across six pages.
  PDF, MusicXML, MIDI, SVG, ZIP and playback JSON passed parsing/integrity checks;
  MIDI and playback note counts matched. These are operational and arrangement
  checks, not evidence that the remaining pitches match the original song.
- Browser verification on 2026-09-22 confirmed advancing playback, page following,
  clicking the sheet to seek, keyboard seeking while paused, and page navigation.
  The regenerated score was left paused at the beginning. This release changes
  arrangement and chunk merging; it does not introduce newly trained weights.

## Conservative accompaniment context correction — 2026-09-21

Released two newly trained boosted-tree context classifiers alongside the existing
V2 neural verifier. They remove a note only when both assign a score below .01,
V2's retained score is at most .20, and the event exactly matches the training
decoder. The correction applies to every balanced full-song upload. Vocal melody,
bass and dedicated piano inference are unchanged. The [model card](docs/accompaniment-context-v3.md)
records training data, commercial-compatible attribution, rejected candidates,
post-failure selection decisions, and all evaluation limitations.

- On 126 evaluated excerpts, false notes decreased **3,047 → 3,032 (0.49%)**.
  All **12,301** previously matched notes remained matched, with no per-recording
  increase in false positives. This is a modest improvement. The last eight
  Oxford sections were newly scored after freeze and showed unchanged accuracy;
  the other 118 were consumed regressions. These are candidate-note measurements,
  not a claim of accurate commercial-song piano arrangements.
- A broader correction was rejected after losing two correct notes on later
  piano excerpts, despite passing the earlier regression set. Rejected artifacts
  and decisions remain archived. Full-replacement release gates were not relaxed.
- **163 backend/training checks passed**, plus **all six native integration cases**
  using actual models, FFmpeg and MuseScore: WAV, 181-second stereo VBR MP3,
  0.125-second MP3, silent MP3, full-song MP3, and multipage score rendering.
  Backend/training lint and diff checks passed. Tests cover exact candidate
  matching, both-head agreement, confident-note preservation, feature parity,
  malformed model files, and retained note/instrument identity.
- The debugging subagent independently verified 14,163 shared validation events
  have bit-identical legacy features; two unmatched events preserve V2 decisions.
  Earlier frontend fixes for Recent-score re-selection and pending playback seeks
  are already pushed and deployed, with 26 frontend tests and production build
  passing. This is a bounded audit, not proof that every code path is error-free.
- Rebuilt the local backend and uploaded an actual stereo VBR MP3 through the
  deployed frontend proxy. Job `99131bd6-b6bd-4c7c-91bc-a32bd064aa60` completed with
  the V3 engine, 22 playback notes and 31 score positions. PDF, MusicXML, MIDI,
  SVG, ZIP and playback JSON were nonempty and parsed successfully; MIDI and
  playback event counts matched. Browser checks verified advancing playback,
  click-to-seek, keyboard seeking while playing, and restart to paused time zero.
  The sample is an operational check, not additional accuracy evidence.

Numerical model arrays are packaged in the backend wheel and checked by SHA-256
before inference. Readiness includes both model files. No new upload or overall
audio-length cap was introduced. Existing scores need a new transcription to
receive the model update.

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

## Historical accompaniment verifier V2 — 2026-09-20

**Correction:** the Vienna clock assessment below was invalid for trimmed Chopin audio. See the correction audit; the original promotion decision is preserved as history, not current validation evidence.

- Trained two 961-parameter classifiers on commercially compatible labeled data;
  rejected the initial thresholds because the digital-piano recall regression
  exceeded the predeclared gate. Expanded training contains 327 clips and 45,825
  unambiguous candidate events. The final release uses the expanded weights with
  a validation-selected conservative threshold of 0.10.
- Preserved all rejected and final experiment records in
  `training/results/accompaniment-verifier-v2/`. Prior tests are labeled consumed
  regressions. The final previously unscored Oxford sections reduced false
  detections 105→74, retained 615/623 previously correct detections, and improved
  F1 0.8750→0.8881. Same recordings/performers as the prior Oxford check: no new
  performer-independence claim. Full measured limits are in the model card.
- Backend regression suite: **146 passed, six opt-in cases skipped**. The six
  real native integration cases then **all passed** (142 seconds), covering
  normal audio, long VBR MP3, subsecond MP3, silent MP3, full-mix MP3, and multipage
  rendering. All download formats and playback positions remained valid.
- Supervision/feature regression tests: **four passed**. Python lint passed.
- Production scope: balanced full-song accompaniment on every uploaded file;
  bounded windows, checked checkpoint hash, unchanged retained note events.
  Other transcription/detail modes retain their previous models.


## Accompaniment clock correction and retraining — 2026-09-20

- Identified stale silence offsets in trimmed Chopin recordings. Corrected the
  preparation and rebuilt 495 feature caches in an isolated experiment directory.
  Fresh preparation reproduced all 88 repaired reference lists exactly.
  Training/validation clock checks now operate per composition, so unrelated
  well-aligned works cannot hide broken labels.
- Trained a 961-parameter neural candidate on 327 clips / 41,744 unambiguous
  events for 50 epochs, and a 128-tree ensemble on the same corrected examples.
  Their validation-selected thresholds were 0.02 and 0.12. Each failed one
  individual regression recording; neither was promoted.
- A fixed consensus retained all 9,825 originally correct detections across
  102 regression excerpts, but removed only 13 false detections and underperformed
  the current V2 release on false-note removal. It was also rejected. Original
  frozen measurements and separate stricter promotion decisions are retained in
  `training/results/accompaniment-clock-correction/`.
- Added model-promotion checks for individual-recording degradation, comparison
  with the deployed model, and an actual reduction in false notes. These are
  regression controls, not proof of accuracy on unseen commercial mixes.
- Backend and training regression suite: **158 passed, six opt-in native cases
  skipped**. Candidate helper matches the original Basic Pitch bounded decoder
  without mutating acoustic evidence. Forest export predictions matched sklearn
  within 1e-12. Lint passed. No production model, runtime routing, or website
  behavior changed, so native end-to-end cases from V2 were not rerun.
- Reviewed SheetSage2: pretrained model, not a labeled training corpus; its
  non-commercial weights and parent model do not fit the project's current
  commercial plans. Nothing was downloaded or integrated.


## Playback functional audit — 2026-09-21

- Fixed reselecting the current Recent score clearing the loaded job without
  triggering another fetch.
- Fixed seeking while audio resume is pending: invalidate the stale start and
  update the paused audio clock before resuming at the requested position.
- Subagent audit: 26 frontend tests, TypeScript/Vite production build, and 77
  backend API/job/audio/render/playback/position tests passed. An isolated browser
  preview verified current-score re-selection, play/pause, and paused sheet seek.
- Scope is these reproduced failures; this is not a claim that all functional
  errors are eliminated. Model experiments are separate from these UI fixes.
