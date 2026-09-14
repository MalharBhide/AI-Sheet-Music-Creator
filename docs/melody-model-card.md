# Vocadito melody decoder v1

V1 remains the deployed recognition model. A larger context-model experiment
was trained and tested but failed its false-voicing release gate; see the
[experiment and released arrangement fixes](context-melody-experiment.md).

## Purpose and release

The full-song pipeline now uses a trained monophonic decoder for the separated
vocal melody. It can recover notes from continuous acoustic evidence even when
the previous Basic Pitch event detector emitted nothing. Bass/accompaniment
still use Basic Pitch; solo piano still uses the dedicated piano CRNN.

This is a first measured melody improvement, not a trained end-to-end piano
arrangement system or a claim of parity with Songscription. It remains a draft
transcription, particularly for rap, overlapping singers, effects and dense mixes.

## Architecture and training

- **2,187 trainable parameters:** pitch-shared temporal convolutions, an explicit
  silence class and a note-attack head; 300 ms receptive field at 50 frames/second.
- **Frozen features:** Basic Pitch note/onset evidence, neighboring octaves,
  pYIN pitch/voicing, CQT energies and relative RMS. Basic Pitch and Demucs weights
  were not fine-tuned.
- **Actual fitting:** 30 epochs × 60 optimizer updates, AdamW, seed 1729, CPU.
  Training audio totals 555 seconds across 26 real recordings. Validation uses
  eight recordings/144 seconds; testing uses six recordings/119 seconds.
- **Split:** 19/5/5 distinct supplied singer IDs. Composition independence and
  upstream-model pretraining overlap have not been established.
- **Selection:** epoch 30 by validation frame loss; decoder chosen on validation
  note F1: five-frame smoothing, 60 ms minimum duration, .8 onset threshold.
  No test recordings were used to fit parameters or select these settings.
- **Checkpoint:** [vocal-melody-v1.pt](../backend/app/assets/vocal-melody-v1.pt),
  SHA-256 `60209b56b6618335f22d1e4e9fc4604d7a2e501df3242d4a18e3df01d81034f9`.
  Runtime verifies the hash and loads tensor weights with `weights_only=True`.

## Held-out results

Macro averages over six recordings from five unseen singer IDs, using annotator
A1. Pitch recall is the fraction of annotated voiced frames with the correct
piano pitch. False silence is annotated melody incorrectly omitted. Note F1 uses
50 ms onset/50 cent pitch tolerance; offsets use max(50 ms, 20% of note duration).
These are **recognition results before score quantization**, not whole-arrangement
accuracy percentages.

| Metric | Previous detector | Trained decoder |
| --- | ---: | ---: |
| Correct melody pitch | 63.1% | **73.4%** |
| Melody incorrectly silent ↓ | 23.6% | **15.0%** |
| Notes during annotated silence ↓ | 8.9% | **4.9%** |
| Note onset F1 | .525 | **.622** |
| Note onset + offset F1 | .388 | **.504** |

All six recordings improved in pitch recall and note-onset F1. Difficult tracks
remain: track 26 reaches only 57.7% pitch recall and still omits 30.0% of annotated
voiced frames; track 28's note F1 improves only from .482 to .493. A second
annotator gives note F1 .547 → .647 and offset F1 .421 → .528. Human annotations
also disagree about expressive singing; neither is an infallible musical score.

A separate controlled stress test adds generated accompaniment to those same
held-out voices and runs the app's Demucs separator. The frozen decoder improves
pitch recall 63.4% → 73.6%, false silence 22.8% → 15.1%, onset F1 .518 → .618 and
offset F1 .376 → .513. These are **real vocals with procedural backing**, not
commercial-song recordings or a second independent singer test set.

## Data, attribution and reproducibility

[Vocadito, Zenodo record 5578807](https://zenodo.org/records/5578807), by Rachel
Bittner, Katherine Pasalo, Juan José Bosch, Gabriel Meseguer Brocal and David
Rubinstein; dataset license **CC BY 4.0**. See the
[dataset loader/license documentation](https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/vocadito.html).
Training uses musician note annotations, not guessed MIDI or user uploads.
No audio corpus is committed to this repository.

The newly trained checkpoint is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution to
Piano Scribe contributors and the Vocadito dataset authors. Frozen upstream
models retain their own licenses. This release modifies learned parameters
through the supervised training described above; it does not redistribute the
training audio or upstream weights in this small checkpoint.

[Commands and dataset choices](../training/README.md),
[exact split](../training/results/vocadito-melody-v1/split.json),
[training history](../training/results/vocadito-melody-v1/history.json),
[validation](../training/results/vocadito-melody-v1/validation.json),
[held-out per-track results](../training/results/vocadito-melody-v1/test.json),
[separation stress results](../training/results/vocadito-melody-v1/separated-test.json)
and [run metadata](../training/results/vocadito-melody-v1/run.json) are retained.

## Limits and next evidence needed

Thirteen and a half minutes of singing is a small development corpus, with only
about nine minutes used for fitting. The test cannot support reliable performance
estimates for every language, singer, genre or recording condition. This model
does not learn chord arrangement, hand assignment, musical phrasing or tempo.
The downstream score grid can still remove short notes or shift their timing.

Further model changes require new independent test data, especially labeled
lead melodies in real mixed songs and musician-approved piano arrangements.
The consumed test split must not become a tuning set while retaining its
"held-out" label. Compare actual phrase-level listening and exported-score timing
alongside metrics before claiming broad improvement.
