# Chord context and piano note accuracy — 19 October 2026

**A new chord-context network was trained, but it did not pass the development
quality gate. It is not deployed.** The current website weights, source routing,
engraving size and saved scores remain unchanged.

This experiment moves beyond the recent bass-only deletion models. It examines
the full-register accompaniment detector and jointly evaluates note accuracy and
recovery of simultaneous pitches. Its result is evidence against shipping this
particular candidate, rather than a claim of improved song transcription.

## What was trained

[ChordNoteVerifier](../training/chord_note_model.py) reuses the released note
verifier's frozen acoustic encoder and normalization. A small residual network
sees the note's embedding, mean/max embeddings of other pitches detected within
40 ms, their relative pitch classes, peer count and original note logit. Its
final layer starts at zero, exactly preserving the pretrained logits before
training. Only the residual network is fitted.

The inference inputs contain no reference notes, chord dictionary or prescribed
triads. Duplicate detections of one pitch are not independent chord evidence.
Peer distances are pairwise, so neighboring attacks do not transitively connect
an entire arpeggio. This architecture can select existing candidates; it cannot
invent or recover a pitch that the detector never proposed, or correct onset
and release times.

We ran one seed for 30 epochs, with AdamW at 0.0003. Epoch 30 had the lowest
weighted validation loss. Validation loss fell from the initial training epoch
to 0.36693, but lower loss alone was not accepted as evidence for a release.

## Data and evaluation

- 63 training excerpts from 21 musical works; 15 development excerpts from five
  different works. Each excerpt is 20 seconds: 26 minutes total.
- Existing MusicNet solo-piano recordings, previously verified under CC BY 4.0,
  with MP3 variants at 48, 128 and 192 kbps. No guitar dataset, new download,
  user recording, saved job or reserved work was used.
- Original publisher label clocks, waveform hashes and work partitions were
  checked before preparation. Previously retained caches contained bass-only
  events, so new **full-register** accompaniment candidates were extracted from
  the existing dataset WAVs. No new audio copies were needed.
- These development works have been used before. They are not a fresh test set.
  MusicNet's known annotation limitations and solo-piano versus separated-stem
  domain difference limit the conclusions.
- Correct note matches require the right pitch and an onset within 50 ms, with
  one-to-one matching. Thus an unmatched detection can be a timing error, not
  necessarily a wrong pitch. Release accuracy is not measured by this metric.
- A reference chord is an attack group with at least two distinct pitches within
  40 ms of its first attack. Complete-chord recall requires every note in that
  group to be matched. This measures simultaneous-note recovery, not harmonic
  chord naming. Extra tones are penalized separately as unmatched detections.
- Late correctly pitched detections, ambiguous overlaps and excerpt-edge holds
  are excluded from negative training labels. Evaluation still counts errors.

## Development results

Across 2,730 reference notes and 824 simultaneous-note groups:

| System | Correct pitch/onset matches | Unmatched detections | Note F1 | Complete chords |
| --- | ---: | ---: | ---: | ---: |
| Raw accompaniment detector | 1,728 | 1,124 | 0.61913 | 363 / 824 |
| Released accompaniment verifier | 1,718 | 1,055 | 0.62439 | 359 / 824 |
| Candidate at threshold 0.4 | 1,716 | 1,037 | 0.62593 | 355 / 824 |
| Candidate at threshold 0.5 | 1,696 | 938 | 0.63236 | 345 / 824 |
| Candidate at threshold 0.6 | 1,666 | 838 | 0.63661 | 327 / 824 |

Threshold 0.5 removes 117 unmatched detections relative to the released verifier,
but also loses 22 correct matches and 14 complete chords. Even the highest F1
among the seven planned thresholds damages chord recovery. None passed the
predetermined gate: at least 5% fewer unmatched detections in aggregate, with
per-work note F1 and complete-chord recall nondecreasing and onset recall loss
no more than one percentage point. No threshold was chosen for deployment.

The raw detector recovers only 44.05% of complete reference attack groups under
this strict timing test. These totals include piano pitches outside the
accompaniment decoder's 36–95 range; they are not a full-song pipeline accuracy
measurement. Increasing filtering strength cannot solve missing detections. This result motivates evaluating the upstream detector instead of adding
another layer of rejection to the website.

### Timing diagnostic

For the raw accompaniment candidates, correct-pitch matches increase from 1,728
at a 50 ms onset tolerance to 2,152 at 100 ms and 2,273 at 200 ms. Among matches
within 200 ms, signed onset error has a median of +0.4 ms, with 10th/90th
percentiles of -62.7/+62.3 ms. A global latency shift is therefore not supported
by this diagnostic. Per-note timing and missing-pitch errors need separate
solutions; pruning all timing mismatches would destroy genuine musical content.

### Detector comparison: a stronger upstream target

We also evaluated the already-installed high-resolution piano detector on the
same 15 development excerpts. It used its existing balanced threshold of 0.3;
there was no threshold search, fine-tuning or change to production routing.

For a like-for-like comparison, both predictions and references were restricted
to the accompaniment detector's MIDI 36–95 range: 2,675 reference notes and
807 reference attack groups.

| System, common pitch range | Note precision | Note recall | Note F1 | Unmatched detections | Complete chords |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw Basic Pitch accompaniment | 60.59% | 64.60% | 0.62529 | 1,124 | 369 / 807 (45.72%) |
| Released accompaniment verifier | 61.95% | 64.22% | 0.63069 | 1,055 | 365 / 807 (45.23%) |
| Existing dedicated piano detector | 72.80% | 77.53% | 0.75091 | 775 | 544 / 807 (67.41%) |

The dedicated detector recovers 356 more correct note attacks and 179 more
complete groups, with 280 fewer unmatched detections than the accompaniment
verifier. The [comparison report](diagnostics/chord-notes-v1/piano-detector-comparison.json)
also includes full-register and per-work results. Its scores still vary greatly
by work; it is not an accurate transcription guarantee.

**Solo-piano mode already uses this detector.** This is a diagnostic comparison,
not a newly trained improvement in that mode. Mixed-song accompaniment contains
other instruments and source-separation artifacts that these solo-piano excerpts
do not cover. Therefore the website's full-song routing was not switched.

The next justified architecture experiment is better upstream polyphonic
detection with per-note onset evidence, evaluated on labeled accompaniment stems
before final arrangement. Another stricter deletion-only filter is not supported
by these results. Any detector replacement must also preserve repeated notes,
true release durations and non-piano accompaniment under the commercial data
constraints.

## Evidence and reproducibility

The [frozen plan](diagnostics/chord-notes-v1/plan.json),
[complete training report](diagnostics/chord-notes-v1/report.json),
and [timing diagnostic](diagnostics/chord-notes-v1/timing-diagnostic.json)
contain the measured results. Exact training source witnesses are retained beside
the report. Subsequent source cleanup adds import ordering, explicit gate failure
messages and a source-drift check; it does not change the trained architecture or
the numerical gate used for this run.

Nine chord-training tests and 25 related production verifier/decoder tests pass.
The final gate reporter reproduces all seven original threshold decisions.
Ruff and the Git whitespace check pass. The running backend reports healthy.
Local training storage remains approximately 18.4 GB, below the 25 GB limit.

The local candidate and approximately 2 MB of cached features are under
`.training/chord-notes-v1/`. They are excluded from Git and are never loaded by
the backend. To reproduce with the final scripts in the existing training
container, choose a **new** output directory:

```sh
PYTHONPATH=/src/backend:/src/training python /src/training/train_chord_notes.py prepare --output /training/chord-notes-reproduction
PYTHONPATH=/src/backend:/src/training python /src/training/train_chord_notes.py train --output /training/chord-notes-reproduction
```

There is deliberately no automatic export or production-promotion command.
Even a development-gate pass would require separated-stem and final-arrangement
evaluation, repeated-note/hold preservation and an independent release test.
The current arrangement reducer also still needs attention: its duration-times-
velocity ranking can favor a loud wrong tone over a quieter genuine chord tone,
as demonstrated in the [pipeline audit](pipeline-diagnostic-2026-10-19.md).
