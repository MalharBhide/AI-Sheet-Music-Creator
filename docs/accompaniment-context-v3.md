# Conservative accompaniment correction, September 2026

V3 keeps the existing V2 neural verifier and adds two newly trained context
classifiers. It is a **small reduction in false accompaniment notes**, not a
claim that full-song piano arrangements are now accurate. Vocal melody, bass,
the dedicated piano model, and the detailed transcription mode are unchanged.
The correction applies to every balanced full-song upload, regardless of filename
or MP3 encoding. Inference still uses bounded overlapping audio windows.

## What the model changes

Each context classifier has 300 boosted trees with at most 31 leaves per tree.
Its 52 inputs include the original 26 pitch-relative features plus attack/release
contrast, harmonic power ratios, lower-harmonic envelope correlation, and attack
independence. There is no song identity, absolute pitch, key, or filename input.
These are fitted ML classifiers; Basic Pitch and Demucs themselves are frozen.

An event changes only when all of these conditions hold:

1. V2 originally retained it, with a score between 0.10 and 0.20 inclusive.
2. Both context classifiers assign it a score below **0.01**.
3. The original bounded training decoder reproduces exactly the same pitch,
   onset, endpoint and velocity as the deployed decoder.

All other V2 decisions remain unchanged. No rejected V2 event is restored.
Retained notes keep their pitch, timing, velocity and instrument assignment.
Classifier scores are not claimed to be calibrated probabilities of correctness.
The two heads share training examples; their agreement is not independent evidence.

## Training and rights

The first head uses 327 clips: 239 GuitarSet training-player clips, 56 corrected
Vienna piano clips, and 32 original procedural piano phrases. The second uses
431: those same clips plus 64 new original dense/quiet piano phrases and 40
separated training-player GuitarSet mixtures with procedural unpitched percussion.
No validation or test performer enters these additional training mixtures.

Both heads use seed 260921, learning rate 0.05, 300 iterations, minimum leaf size
20, L2 regularization 1, positive-event weight 2, and equal weighting across
corpora and recordings. Training considered 15- and 31-leaf trees. The expanded
head fits 52,694 unambiguous events. Validation has 108 clips, including 16 new
procedural phrases and ten separated examples. Ambiguous timing negatives remain
excluded from fitting but included in evaluation. Corrected Vienna clocks and
the per-composition label audit are described in the [earlier model card](accompaniment-verifier.md).

GuitarSet and Vienna carry CC BY 4.0; original procedural material and the
MuseScore General soundfont have the provenance recorded in the
[asset notice](../backend/app/assets/accompaniment-context.LICENSE.txt).
Oxford MIDItest is CC BY 4.0 and used only for evaluation. No user uploads or
non-commercial-only datasets enter fitting. SheetSage2 was excluded for the
[previously documented reasons](sheetsage2-assessment.md).

## Selection history and measured limits

Full replacements using the new features still regressed and were rejected.
A single-head correction removed 105 false notes on 102 regression excerpts
but lost two correct notes in a guitar excerpt. A two-head correction at 0.05
passed those 102 excerpts but lost two correct notes in the subsequent 16 piano
sections. Neither candidate was deployed.

After those failures, the fixed follow-up reduced the rejection-score threshold
fivefold to 0.01 without another threshold search. This is a **post-failure
development decision**. The 118 previously evaluated excerpts are therefore
consumed regression data, not an independent generalization estimate. A final
check used eight previously unscored Oxford sections from seconds 60–90 (or the
remaining audio), with the unchanged audio-only clocks. Those sections contain
new notes but share recordings and performers with earlier evaluations.

| Evaluation | Clips | Correct matches before → after | False notes before → after |
| --- | ---: | ---: | ---: |
| GuitarSet regression | 60 | 6,635 → 6,635 | 1,947 → 1,942 |
| Vienna regression | 16 | 1,216 → 1,216 | 173 → 169 |
| Oxford first scoring window | 8 | 310 → 310 | 44 → 44 |
| Oxford 30–60 seconds | 8 | 615 → 615 | 73 → 72 |
| Separated guitar stress | 10 | 996 → 996 | 375 → 375 |
| Later Vienna regression | 16 | 1,998 → 1,998 | 371 → 366 |
| Final Oxford 60–90 seconds | 8 | 531 → 531 | 64 → 64 |

Across these 126 excerpts, false notes decrease **3,047 → 3,032 (0.49%)**,
while all **12,301** previously matched notes remain matched. Every individual
recording preserves its previously matched reference-note set and has no
increase in false positives. Validation removes 11 false notes with the same
preservation requirement. The final eight excerpts show unchanged accuracy,
not an additional measured improvement.

Matching uses one-to-one 50 ms onset/50 cent pitch tolerance without offset
constraints. These are raw candidate-note measurements, not complete sheet-music
or commercial-mixture arrangement ratings. The residual false notes remain
substantial. Repeated use of these corpora, limited instrument diversity,
synthetic additions and shared compositions restrict what the results establish.
No finite benchmark guarantees preservation on every future recording.

The stricter full-replacement gate remains unchanged. This correction has its
own gate against actual deployed V2 events: no loss of any matched reference
and no increase in false notes in any recording. It does not claim to pass the
full-replacement gate against the unfiltered original detector.

## Artifacts and reproducibility

[Archived records](../training/results/accompaniment-context-v3) preserve rejected
attempts, frozen selections, every recording's metrics, compact splits, source
model hashes and provenance. Raw recordings and full labels remain local.
Inference uses checksum-checked numerical NPZ arrays, with validated tree
topology and feature versions; it never loads sklearn pickles. Export matches
sklearn probabilities to 1e-12 on all validation recordings.

- Expanded head SHA-256: `d06bbe84ee9ad251960969e0097b217e340d52562ecf178295d7c4ae57c6ee8d`
- Original head SHA-256: `381c150b19bdb223fc9e85f88b78dd69892b0de5c7591189967f30b59a26a8e7`
- The V2 checkpoint hash and first 26 features are unchanged.

In the Python 3.11 transcription environment, with `PYTHONPATH=backend:training`,
`mir_eval`, and scikit-learn 1.9.0, start from the corrected data preparation in
the [training guide](../training/README.md). Use a new directory for reproduction;
commands refuse to overwrite saved runs or reports.

```bash
python training/prepare_context_piano.py .training
python training/prepare_training_stems.py .training
python training/prepare_context_dataset.py .training
python training/cache_note_verifier.py .training/note-verifier-context-data --workers 2
python training/cache_note_verifier.py .training/note-verifier-context-expanded --workers 2
python training/train_context_verifier.py .training/note-verifier-context-data --run boosted-development
python training/train_context_verifier.py .training/note-verifier-context-expanded --run expanded-context-v1
python training/select_agreement_correction.py .training/note-verifier-context-expanded .training/note-verifier-context-expanded/expanded-context-v1/model-31.pickle .training/note-verifier-context-data/boosted-development/model-31.pickle
python training/freeze_correction_margin.py .training/note-verifier-context-expanded
python training/evaluate_context_correction.py .training/note-verifier-context-expanded --run agreement-margin
python training/prepare_later_piano.py .training
python training/evaluate_context_correction.py .training/note-verifier-context-expanded --run agreement-margin --fresh .training/later-vienna-context/manifest.json
python training/prepare_oxford_later.py .training
python training/evaluate_context_correction.py .training/note-verifier-context-expanded --run agreement-margin --fresh .training/oxford-later-context/manifest.json --report oxford-final.json
```

This reproduces consumed evaluations; rerunning does not make them fresh tests.
The later-Vienna report under the margin run is a regression rerun even though
the generic evaluation script calls its input option `--fresh`.
