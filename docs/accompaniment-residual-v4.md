# Residual false-note classifier — September 2026

V4 adds a newly trained classifier that targets false accompaniment notes still
accepted by V3. It is a small measured improvement, not a solution to general
audio-to-piano transcription. It applies to every balanced full-song upload,
regardless of filename or encoding. Vocal melody, bass, dedicated piano inference,
arrangement density, and existing held-note handling are unchanged.

No user recordings were processed, no scores were regenerated, and no website
transcription jobs were created for this experiment or its release checks.

## Training

The fixed 300-tree classifier uses 52 pitch-relative acoustic features already
available to the deployed model. It is trained specifically on V3-retained events
whose decoder identity matches exactly. Rejected candidates are excluded from
fitting, and ambiguous timing negatives remain excluded. Targets are recomputed
with one-to-one note matching against the deployed candidate events.

The existing commercial-compatible expanded split contains 431 training clips:
239 GuitarSet, 56 corrected Vienna piano, 96 original procedural piano, and 40
separated mixtures from training-only GuitarSet performers. The fit uses **50,442
events: 44,599 positive and 5,843 unambiguous negative**. No validation/test player
enters fitting. The source license, attribution, clocks and split limitations are
documented in the [V3 card](accompaniment-context-v3.md) and
[asset notice](../backend/app/assets/accompaniment-residual-v4.LICENSE.txt).

Parameters were fixed before training: seed 260922, 31 maximum leaves, 40 minimum
events per leaf, learning rate .05, L2 regularization 3, 300 iterations, no early
stopping. Recordings and corpora receive equal base weight; correct-note examples
receive six times the weight of negative examples. This is a fitted classifier,
not a change to an arrangement limit or to Basic Pitch/Demucs weights.

The 108-clip validation set selects a rejection threshold from a fixed grid,
requiring every previously matched reference note to survive in every recording.
The best threshold was .075. A predeclared factor of .5 makes the final threshold
**.0375**, frozen before regression or new-tail evaluation. No post-test threshold
adjustment was made. Scores are not calibrated probabilities of musical correctness.

## Runtime policy

V3 first makes its unchanged decision. The residual model may reject a retained
note only if its V2 score is at most .5, its event exactly matches the shared
training decoder, and its residual score is below .0375. A note rejected by V3
cannot be restored. Stronger V2 events and unmatched decoder events are protected.
Retained note objects, pitches, onset/release times, velocities and source roles
are preserved. The model operates within the existing bounded audio windows.

Numerical NPZ arrays are SHA-256 checked and loaded without pickle. Missing or
damaged weights produce an explicit error. Readiness checks include the new
asset. The SHA-256 is
`ad958557c44312c8d082b9339801ae794a7673956f098106f7fb6381fa3cd24d`.

## Results and limits

| Evaluation | Clips | Correct matches V3 → V4 | False notes V3 → V4 |
| --- | ---: | ---: | ---: |
| Validation, used for selection | 108 | 10,316 → 10,316 | 3,247 → 3,214 |
| Consumed regression excerpts | 126 | 12,301 → 12,301 | 3,032 → 3,009 |
| Newly scored GuitarSet tails | 16 | 578 → 578 | 299 → 299 |

Regression false notes decrease **0.76%**, preserving every matched reference
in every recording. The fresh-tail check shows unchanged accuracy, not a measured
gain. All 16 eligible test-player recordings at least 35 seconds long were used;
evaluation covers seconds 30–60 (or the remaining audio), excluding .25 seconds
at each edge. Previous experiments used only the first 30 seconds.

Those tails share the existing test performer and pieces. They are new note
sections, **not independent new performers/compositions**. The other 126 excerpts
were previously consumed, and repeated development on these corpora limits the
generalization claim. No test labels were used for fitting or threshold selection
in this experiment. All these evaluations are now consumed.

Matching uses 50 ms onset and 50 cent pitch tolerance without an offset criterion.
These are candidate-note results, not judgments of complete arrangements or
proof of accurate commercial-song transcription. Many wrong notes remain. A
finite evaluation cannot guarantee preservation of every correct note on a new
recording. This filter cannot recover missing melody notes or repair wrong pitches.

## Reproduction

Use the existing corrected V3 feature caches and the Python 3.11 transcription
environment, with scikit-learn 1.9.0 and `PYTHONPATH=backend:training`. Commands
refuse to overwrite a completed run/evaluation. The baseline reconstructs V3
from its original three checkpoints even after V4 has been installed.

```bash
python training/train_residual_verifier.py .training/note-verifier-context-expanded
python training/train_residual_verifier.py .training/note-verifier-context-expanded --test
python training/prepare_residual_holdout.py .training
python training/train_residual_verifier.py .training/note-verifier-context-expanded --fresh .training/residual-guitar-tails/manifest.json
python -m pytest -q training/test_residual_verifier.py backend/tests/test_note_verifier.py
```

Use a new `--run` name for a new experiment. Rerunning these evaluations does not
make them fresh. Compact provenance and per-recording results are archived in
[the results directory](../training/results/accompaniment-residual-v4).
