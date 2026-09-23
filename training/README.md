# Supervised melody training

Model development uses labeled corpus features and explicit regression gates.
Do not retranscribe user uploads or regenerate sheet music to demonstrate a model
update; the user can upload audio again when they want to hear the new result.
The [residual false-note classifier](../docs/accompaniment-residual-v4.md)
was trained and checked entirely with licensed datasets and cached features.

The latest [low-register specialist](../docs/left-hand-transcription.md) targets
left-hand accompaniment and adds a per-recording onset-offset preservation gate.
Its frozen candidate passed regression and runtime parity. It does not apply the
accompaniment model to unvalidated bass-stem predictions.

A subsequent [relationship-aware experiment](../docs/relational-verifier-experiment.md)
added later labeled piano passages and explicit neighboring-note features. It
failed the correct-note preservation gate and remains offline; do not deploy its
checkpoints. Its bounded-feature and dataset preparation code support further
model development without processing user uploads.

This experiment trains a small temporal neural decoder on **real human-annotated
singing**. The resulting checkpoint is used for the vocal melody of every
Full song upload, independent of filename, file extension or song identity. Basic Pitch, pYIN and CQT supply frozen acoustic features; the new
network learns piano pitch, silence and note attacks. It does not retrain Demucs,
the piano CRNN or a complete audio-to-arrangement model.

## Data and scope

| Corpus | Labels and intended use | Decision |
| --- | --- | --- |
| [Vocadito](https://zenodo.org/records/5578807) | 40 real singing recordings, frame pitch and two independent note annotations | Downloaded, checksum verified; first melody training/evaluation corpus |
| [VocalSet & Annotated VocalSet](https://zenodo.org/records/10200775) | Reviewed pitch/boundary annotations, 20 singers, varied techniques | CC BY 4.0; pinned revision downloaded and checked for the context-model experiment |
| [Children's Song Dataset](https://zenodo.org/records/4916302) | Labeled singing phrases | Dataset is CC BY-NC-SA 4.0, despite the paper's separate license; excluded from this potentially commercial service |
| [DALI](https://github.com/gabolsgabs/DALI) | Aligned lyrics and melody notes | Non-commercial dataset terms; excluded from new training |
| [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro) | 198.7 hours of aligned piano audio/MIDI | CC BY-NC-SA 4.0; excluded from new training for this potentially commercial service |
| [MedleyVox](https://github.com/jeonchangbin49/MedleyVox) | Multiple-singer separation evaluation | Relevant to separation failures, not a substitute for paired piano-arrangement labels |

Vocadito is CC BY 4.0; credit Rachel Bittner, Katherine Pasalo, Juan José Bosch,
Gabriel Meseguer Brocal and David Rubinstein. See the
[dataset documentation and license](https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/vocadito.html)
and [paper](https://arxiv.org/abs/2110.05580). The archive's metadata contains 29
distinct singer IDs, although the paper describes 28 volunteers. Splits follow
the supplied IDs. No user uploads or commercial recordings enter training.

Seed 1729 fixes 19 training, five validation and five test singer IDs, producing
26/8/6 recordings. No singer ID appears in two partitions. Reliable composition
IDs are absent; this is not a claimed composition-disjoint benchmark. Upstream
Basic Pitch/Demucs pretraining overlap has not been audited.

## Reproduction

Use the repository's pinned Python 3.11 transcription environment. Run from the
repository root (or use the backend Docker image with this repository mounted):

```bash
export PYTHONPATH="$PWD/backend:$PWD/training"
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2
python training/prepare_vocadito.py .training
python training/cache_vocadito.py .training
python training/train_melody.py .training --epochs 30
python training/train_melody.py .training --test
python training/stress_melody.py .training
python -m pytest -q training/test_melody_decoder.py
```

Data, feature caches and temporary runs stay in ignored `.training/`. The
download is pinned to Zenodo record 5578807 and publisher MD5
`dea40fd18f14d899643c4ba221b33a46`. Features use a common 50 Hz clock, including
Basic Pitch's actual output frame times. The 2,187-parameter network shares a
temporal classifier across all 88 piano pitches; its receptive field is 300 ms.
Training uses AdamW (learning rate .001, weight decay .0001), batches of four
256-frame windows, 60 updates per epoch, 30 epochs and gradient clipping at 3.
Loss combines frame cross entropy and .3 times weighted onset binary cross
entropy. Seeds and deterministic PyTorch operations are fixed.

Choose the epoch by validation frame cross entropy. Choose minimum duration,
onset threshold and smoothing by validation note F1. The final test command
checks the selected checkpoint hash, refuses to overwrite an existing test,
and never fits or selects settings on test recordings. Retraining a consumed
experiment is also refused. A new model revision needs new independent test
data; deleting a report does not make those recordings unseen again.

## Evaluation contracts

The baseline is the deployed balanced Basic Pitch vocal detector plus pYIN
refinement, before rhythm quantization or arrangement. Both systems receive the
same waveform. This comparison isolates melody recognition; it does not measure
the complete exported sheet music. Reference note CSVs store onset, **Hz** and
**duration**, not MIDI pitch and end time. The decoder rounds pitch to piano
semitones; evaluation compares against the original annotated frequencies.

[mir_eval](https://mir-eval.readthedocs.io/latest/api/transcription.html) measures
note matching with 50 ms onset and 50 cent pitch tolerance, and optional offsets
within max(50 ms, 20% of reference duration). Frame metrics use a 20 ms grid:
correct piano pitch over annotated voiced frames, missed melody as silence, and
false voicing over annotated silent frames. Results are macro averages across
recordings, with per-recording results retained. A1 is used for training; A2 is
an independent-annotator sensitivity check, not extra training data.

The separation stress test mixes held-out real singing with independently
generated bass/chords/percussion at −3 dB relative RMS, then uses the same Demucs
separator as the app. It uses the frozen checkpoint and settings. This tests
separation artifacts under controlled conditions; it does not establish
accuracy on commercial mixes, overlapping singers or rap. Synthetic contract
tests cover real rests, repeated notes, octave changes and short/long windows;
their pass count is not an accuracy score.

See [the framework](../docs/transcription-framework.md) for promotion gates and
the distinction between melody recognition and a complete piano arrangement.

The released checkpoint, measured improvements and known failures are documented
in the [model card](../docs/melody-model-card.md).

## Larger commercial-compatible experiment

**Status:** training and frozen testing are complete. The candidate failed its
false-voicing release gate and remains a research artifact. See the
[results and release decision](../docs/context-melody-experiment.md).

The context experiment adds 288 VocalSet excerpts (3,260.58 seconds) to the
original 26 Vocadito training clips (555.16 seconds). VocalSet validation uses
48 clips from four other singers, and the new test reserves 48 clips from four
unseen singers. The earlier six-clip Vocadito test is not reused for fitting or
selection. All clips from each singer stay together. Shared scales and excerpts
mean this is **singer-disjoint, not composition-disjoint**.

The revised archive is pinned to Zenodo record 10200775, publisher MD5
`8d39344bbc775aa040840783ae73cfa4`. Credit original audio authors Julia Wilkins,
Prem Seetharaman, Alison Wahl and Bryan Pardo; annotation authors Behnam Faghih
and Joseph Timoney; revised archive author Santiago Donaher. The original audio,
annotations and revised combined record carry CC BY 4.0. Data are kept locally
under `.training/`; user MP3s are diagnostic inputs only and never enter fitting.

The [annotation paper](https://www.mdpi.com/2076-3417/12/18/9257) describes
semi-automatic pitch/boundary labels with manual review. Its estimated pitches
partly originate from pYIN, also an input to this model; that dependence limits
the benchmark's independence. Labels are not a musician-approved piano
arrangement. The loader chooses the `extended 2` variant before evaluation,
requires coherent nominal/performed pitch agreement, excludes speech and
ambiguous audio, and uses one constant semitone transposition per recording.
It never sets reference pitches from app predictions. This quality-filtered
sample does not represent every file in VocalSet.

**Clock audit:** the revised audio and annotations have a 2:1 duration mismatch
for all 12 selected `m9` validation recordings. Correct note times by exactly
0.5, inferred from file duration versus the annotation's `Total Duration`, before
cropping. A 2% tolerance admits only 1x, 0.5x or 2x clock ratios; other mismatches
stop preparation for review. Singer splits and acoustic features remain fixed.
No selected training or test recording needed this correction. The initial
experiment with unreconciled validation clocks is invalid for model selection.

```bash
python training/prepare_vocalset.py .training
python training/cache_vocalset.py .training --workers 2
python training/train_context_melody.py .training --epochs 30 --rehearsal
python training/train_context_melody.py .training --test
python training/stress_context_melody.py .training
python -m pytest -q training/test_context_melody.py
```

Only an older pre-audit cache needs `repair_vocalset_clocks.py .training`; new
caches reconcile clocks on creation. Preserve rejected runs before starting a
new experiment; do not delete or reuse consumed test reports.

The candidate adds a 1.26-second pitch-shared context branch to V1, initially
outputting zero correction. It has 5,853 total parameters. An unconstrained
fine-tune improved exercise recognition but regressed on natural-song
validation; it was rejected before testing. The rehearsal variant freezes V1's
2,187 parameters, trains the remaining 3,666, and adds distillation on **training
phrases only** to preserve existing pitch distributions and attack logits.
Balanced sampling draws equally from the two corpora. Short examples are
padded with ignored targets, never mislabeled as silence.

Model selection uses mean validation frame loss across both corpora, then mean
note/offset F1, preferring decoder settings that satisfy the natural-song
validation regression gate. Evaluation compares directly against deployed V1,
not the weaker pre-V1 detector. The frozen separation stress check uses the first
two test clips per singer in the fixed manifest, adds independently generated
backing at −3 dB RMS and runs Demucs. It is a controlled artifact test, not a
commercial-song accuracy measurement.

`audit_arrangement.py` separately measures detected versus retained notes for
each source. It can reuse saved raw MIDI with `--reuse` and compare `--grid
eighth` against `--grid sixteenth`. Note retention is not note correctness.

## Accompaniment note verification

See [the verifier model card](../docs/accompaniment-verifier.md) for the two rejected
candidates, corrected clock audit, data rights, and consumed-test caveats.
The historical V2 Vienna assessment is invalid. Corrected full replacements were
rejected; the later [V3 context correction](../docs/accompaniment-context-v3.md)
preserves V2 and adds a narrowly applied learned filter. Its reproduction commands,
failed attempts and modest measured improvement are documented separately.
The 961-parameter event classifier uses pitch-relative acoustic evidence; Basic
Pitch/Demucs are frozen. User recordings are not training examples.

In the same Python 3.11 transcription environment with `mir_eval` installed:

```bash
python training/prepare_note_data.py .training
python training/make_verifier_piano.py .training
python training/cache_note_verifier.py .training --workers 2
python training/train_note_verifier.py .training --recall-margin .02
python training/prepare_verifier_tests.py .training
python training/prepare_verifier_tests.py .training --stems validation
python training/train_note_verifier.py .training --test
python training/test_note_verifier.py .training .training/note-verifier-oxford/manifest.json oxford-test
python training/test_note_verifier.py .training .training/note-verifier-stems-validation/manifest.json separated-validation

python training/prepare_vienna.py .training
python training/prepare_vienna.py .training --align
python training/cache_note_verifier.py .training --workers 2
python training/train_note_verifier.py .training --run note-verifier-v2
python training/prepare_verifier_tests.py .training --stems test
python training/train_note_verifier.py .training --run note-verifier-v2 --test
python training/test_note_verifier.py .training .training/note-verifier-oxford/manifest.json oxford-regression --run note-verifier-v2
python training/test_note_verifier.py .training .training/note-verifier-stems-test/manifest.json separated-test --run note-verifier-v2

python training/calibrate_note_verifier.py .training
python training/prepare_extended_piano_test.py .training
python training/train_note_verifier.py .training --run note-verifier-v2-conservative --test
python training/test_note_verifier.py .training .training/note-verifier-oxford/manifest.json oxford-regression --run note-verifier-v2-conservative
python training/test_note_verifier.py .training .training/note-verifier-stems-test/manifest.json separated-regression --run note-verifier-v2-conservative
python training/test_note_verifier.py .training .training/note-verifier-oxford-extended/manifest.json extended-piano-test --run note-verifier-v2-conservative
python -m pytest -q training/test_note_evidence.py backend/tests/test_note_verifier.py
```

Commands refuse to overwrite experiment directories or consumed evaluation
reports. The sequence reproduces an already completed experiment; its test sets
are **not fresh benchmarks for future tuning**. Downloaded recordings, feature
caches and full reference labels remain in ignored `.training/`. Committed split
manifests contain reference hashes and counts, not recording media.


### Corrected accompaniment experiments

The original V2 Chopin training labels used stale silence offsets. `prepare_vienna.py --align` now handles trimmed audio, calibrates from seconds 0–15, and excludes those seconds from supervision. `audit_verifier_labels.py` checks training and validation compositions separately before fitting; a good corpus average cannot hide a nearly unaligned work.

For the saved V2 experiment, preserve the earlier artifacts and rebuild in an isolated directory:

```bash
python training/repair_verifier_clocks.py .training
python training/cache_note_verifier.py .training/note-verifier-v3-data --workers 2
python training/train_note_verifier.py .training/note-verifier-v3-data --run note-verifier-v3 --positive-weight 4 --protect-recordings --recall-margin .0025
python training/evaluate_verifier_regression.py .training/note-verifier-v3-data
python training/train_verifier_forest.py .training/note-verifier-v3-data
python training/evaluate_verifier_regression.py .training/note-verifier-v3-data --run note-verifier-v4
python training/evaluate_verifier_regression.py .training/note-verifier-v3-data --run note-verifier-consensus
python training/verifier_release_gate.py .training/note-verifier-v3-data/note-verifier-consensus/regression.json
```

Run with `PYTHONPATH=backend:training` and the transcription dependencies plus `mir_eval` and `scikit-learn`. The corrected candidate decoder lives in a shared helper but is used only by these offline experiments, not the deployed V2 path. Both trained candidates and their consensus were rejected. Do not copy their checkpoints into the app. The stricter gate checks individual recordings and compares precision/recall/F1 against the current release, not just the original detector. All evaluation recordings have been consumed; new future accuracy claims need additional independent recordings.
