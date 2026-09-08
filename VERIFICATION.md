# Verification

The default backend suite exercises upload size/type validation, queue capacity, durable job status, artifact access, FFmpeg decoding limits, piano notation, rendering output validation, and worker failure/recovery. The frontend gate compiles TypeScript and makes a production Vite build.

Native tests are explicit: run with RUN_PIPELINE_INTEGRATION=1. They use real FFmpeg, Basic Pitch, music21, and MuseScore without replacing any pipeline stage. One test uploads generated audio and downloads all output formats; the other verifies multi-page PDF/SVG parity.

The local development environment has passed the unit suite and frontend build. Basic Pitch has also transcribed a generated short melody into MIDI and two-staff MusicXML locally. The local environment cannot launch MuseScore's native display dependencies and does not provide Docker. The GitHub Actions native-pipeline job is configured to verify full native rendering separately; check the current commit's Actions results for that gate.

No subjective transcription-quality claim follows from these automated checks. Review real recordings musically before using the generated scores.
