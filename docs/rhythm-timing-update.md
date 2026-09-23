# Rhythm timing update

The precise rhythm decoder now retains clear eighth-note triplets instead of
forcing every attack onto straight sixteenths. At 120 BPM, the old decoder moved
an equal-spaced group at 0, 1/6 and 1/3 seconds to 0, 1/8 and 3/8 seconds. That
changed two equal intervals into a short–long pattern even with correct model
detections. The corrected path preserves the equal spacing through MusicXML,
MIDI and browser playback.

This is a rhythm-decoding correction, **not a new trained neural checkpoint**.
It applies to all new precise-mode transcriptions, without song-specific rules.
The Simple eighth-note setting retains its intentionally coarse behavior.

## Conservative pattern recognition

Within each beat, the decoder requires three distinct onset positions close to
the three eighth-triplet slots. Duplicate pitches in a chord do not add evidence.
The triplet grid must also have less than half the mean squared onset error of
the straight-sixteenth grid. This prevents ordinary sixteenth-note jitter from
being interpreted as a triplet. Quantization then uses a common twelfth-quarter
lattice to represent both patterns exactly; it does not turn every note into a
twelfth-quarter subdivision. The notation stage uses that same lattice, avoiding
a second quantization that would erase the corrected spacing.

Tests cover complete triplets, unchanged straight patterns, incomplete groups,
chord duplicates, jittered sixteenths, mixed straight/triplet beats, rests and
sustained notes. A symbolic fixture checks MusicXML time-modification elements
and exact resulting playback spacing. No user upload, audio transcription job,
or user-score regeneration was used for testing.

This fixes one demonstrated class of rhythm error. It does not solve swing,
incomplete tuplets, tempo changes, rubato, beat-detection errors or arbitrary
expressive timing. The score still uses a single tempo. Remaining errors in raw
pitch and boundary detection also remain possible.

## Timing-model experiment: not released

Two 150-tree boundary regressors were actually fitted on cached, commercially
compatible Vocadito and VocalSet features. The 314 training clips provided 3,815
matched onset examples and 2,940 matched release examples. Features cover a
pitch-relative 260 ms neighborhood around each detected boundary, plus duration.
Pitches and note counts were frozen, and proposed time changes were limited to
80 ms, preserving monophony, positive duration and recording bounds.

Validation used 56 other-singer clips and fixed correction strengths of .25,
.5 and 1. At full strength, mean onset error decreased from **31.8 to 25.1 ms**
on Vocadito and **66.6 to 40.2 ms** on VocalSet, measured on fixed same-pitch
baseline matches within 200 ms. These means exclude unmatched detections and
are not complete transcription-accuracy scores. Release errors decreased from
54.4 to 52.0 ms and 189.6 to 178.2 ms respectively.

However, every correction strength lost previously correct onset or onset/offset
matches in at least one validation recording: 4, 7 and 13 recordings respectively.
The candidate therefore failed the preservation gate and remains offline. No
test split was evaluated or used to tune it. The deployed melody and V4 note
filter weights are unchanged. [Archived experiment results](../training/results/melody-timing-v1)
include fitting settings, validation results, hashes and the rejection decision.
Existing [dataset attribution and limitations](context-melody-experiment.md) apply.

Reproduce with the existing corrected caches in the Python 3.11 training
environment, scikit-learn 1.9.0 and `PYTHONPATH=backend:training`:

```bash
python training/train_melody_timing.py .training --run melody-timing-v1
python -m pytest -q training/test_melody_timing.py
```

Completed runs cannot be overwritten. Raw features and fitted research pickles
remain local under ignored `.training/`; the website never loads those pickles.
