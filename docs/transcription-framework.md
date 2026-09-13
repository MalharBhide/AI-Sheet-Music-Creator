# Audio-to-piano transcription and training framework

## Objective

For a mixed song, recover a recognizable main melody and supporting piano part.
Preserve real rests and articulation while reducing missed vocal passages, false
pitches, octave errors and accompaniment that masks the melody. A complete PDF
and a short synthetic example are not accuracy tests.

## Stages and contracts

| Stage | Input → output | Principal failure | Measurement |
|---|---|---|---|
| Decode | MP3/audio → time-aligned PCM windows | Truncation or channel loss | Duration, seams, sample rate |
| Separate | Stereo mixture → vocals, bass, accompaniment | Lead leaks into another stem | Annotated mixture/stem evaluation |
| Recognize melody | Vocal audio → pitch, voicing and attacks over time | Missing phrases, octave errors, vibrato mistaken for notes | Frame pitch recall, false silence, note onset/offset F1 |
| Recognize support | Bass/accompaniment → candidate notes | Harmonics and pads dominate | Per-role precision and polyphony |
| Arrange | Candidate roles → foreground melody and playable support | Main tune obscured or buried in bass staff | Melody retention, hand span, listening review |
| Interpret rhythm | Seconds → beats, durations and rests | Wrong tempo, long invented rests or tied clouds | Timing error, gap audit, readable rhythm |
| Export/play | One score → MusicXML, MIDI, PDF, browser audio | Playback differs from notation | Exact note/timing agreement and page positions |

Each stage must expose what it changes. A melody decoder needs access to the
continuous audio evidence: filtering an existing MIDI list can remove errors but
cannot recover a phrase for which the first detector emitted no notes.

## First training experiment

Train a small supervised melody decoder using frozen pretrained acoustic
features and human note annotations. Learn pitch/voicing and note boundaries,
instead of training a large source-separation or piano model from scratch on this
computer. Keep bass and piano polyphony separate from the monophonic objective.

Use real singing examples for optimization and validation. Synthetic examples
remain regression tests, not evidence of real-song accuracy. Record the dataset
version, license, checksum, exact split, random seed, feature version, optimizer,
epoch selection and checkpoint hash. All windows and augmentations from one
recording belong to one split. Group by singer/composition where reliable IDs
exist; disclose when the corpus cannot support a singer-independent split.

Choose the checkpoint and decoder settings on validation recordings only, then
evaluate the frozen choice once on held-out recordings. Compare against the
current deployed pipeline on the same audio and annotations. Report frame pitch
accuracy/recall, fraction of annotated melody dropped as silence, and note F1
both with and without an offset constraint. Report individual recordings as well
as aggregates, including failures. Preserve a second annotator's labels for a
sensitivity check when available.

## Promotion criteria

The trained candidate must improve held-out melody recognition without increasing
false silence or damaging note segmentation. It must pass silence, repeated-note,
octave-jump, sustained-note and chunk-seam tests. Before using it for mixed songs,
also evaluate separated stems: clean solo singing alone does not demonstrate
robustness to separation artifacts, backing vocals or rap.

Keep an unsuccessful or insufficiently validated checkpoint as a research
artifact; do not quietly replace a working engine with it. Broad accuracy claims
require a larger, representative real-recording corpus and musician listening
review. Training a small decoder is a first measured step, not a guarantee that
the complete arrangement now matches every song.
