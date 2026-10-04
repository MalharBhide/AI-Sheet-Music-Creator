# Melody clarity and repeated-note rhythm

This update changes rhythm decoding and balanced piano arrangement after note
recognition. The deployed V7 classifiers and melody recognition weights are
unchanged. It applies to new uploads without rules for any particular recording.
No user recording was transcribed and no existing score was regenerated.

## Arrangement

Balanced full-song backing omits a lower-octave copy of the detected lead only
when at least four consecutive notes have identical attack and release times,
include three distinct lead pitches, and span at least one second. Copies one
or two octaves below are eligible. This leaves the lead unchanged and avoids
removing isolated octave chord tones, support with different holds, and repeated
single-pitch accompaniment. Dedicated bass, solo-piano inference, and detailed
arrangement mode retain their previous selection rules.

This is simplification, not proof that removed notes were wrong: intentional
octave doublings can also be omitted. It does not remove arbitrary false pitches
or repair an incorrectly detected melody. Existing complete-note selection and
held-bass priority remain in place.

## Rhythm

- Full-song timing uses detected lead attacks when at least eight distinct
  onsets are available among its first 64 notes. Otherwise it uses all parts.
  Chord tones at the same onset no longer get multiple votes for grid phase.
- Precise rhythm can align a shifted triplet run when the straight phase is
  incoherent, ternary phase is strong, and at least two complete triplet beats
  cover 75% of the sampled attacks. Missing triplet slots do not establish this
  evidence. The simple eighth-note option retains straight quantization.
- Automatic tempo can be refined within 2% of the acoustic estimate using at
  least 12 near-equal repeated-key intervals spanning six seconds. A fitted run
  must have low drift, and conflicting proposals cancel the correction. Work is
  bounded to three windows of at most 2,048 notes. Manual tempo overrides are
  unchanged; this does not choose half/double-time or estimate a tempo map.

Short, swung, or gradually changing repeated-key runs do not trigger tempo
refinement. This guard does not add swing notation or fix every expressive
rhythm: the score still uses one tempo and a finite note grid. Analysis records
the refinement in `tempo_refinement_bpm` and the common shift in
`timing_offset_seconds`.

## Verification and limits

The archived [symbolic benchmark](melody-repeat-benchmark.json) compares exact
original note fixtures against main revision `b9ee29e`, with a baseline source
SHA-256. It does not measure model accuracy on audio.

| Original fixture | Before | After |
| --- | --- | --- |
| 48 shifted triplets at 80/120/160 BPM | Maximum spacing error 125/83.33/62.5 ms | Under 0.000001 ms; all 48 attacks retained |
| 48 quarter repeats with a 1% tempo error at 80/120/160 BPM | Maximum spacing error 178.22/118.81/89.11 ms | Under 0.000001 ms; all 48 attacks retained |
| Six lead notes, six exact octave copies, one held backing note | Seven backing notes | One backing note; all six lead notes unchanged |

**256 tests passed; six opt-in real-audio integration cases were skipped.** New
checks cover contradictory tempo evidence, swing and changing tempo, manual
overrides, sparse triplet evidence, held support, detailed-mode preservation,
lead timing under dense backing, and all 18 repeated triplet attacks through
real MIDI, MusicXML, and browser playback export. Inference in the new routing
tests uses controlled stubs. Lint and diff checks passed.

Reproduce the fixture comparison in an environment with backend dependencies:

```bash
PYTHONPATH=backend python scripts/benchmark_arrangement_rhythm.py /tmp/melody-repeat-benchmark.json
```

The helper refuses to overwrite existing results. When Git is unavailable in the
runtime, export the baseline source with `git show` and supply
`--baseline-source`. No recording-specific score generation is required.
