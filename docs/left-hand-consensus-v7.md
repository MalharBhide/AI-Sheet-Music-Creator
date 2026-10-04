# V7: trained consensus for uncertain left-hand notes

Two newly trained classifiers must agree before removing a surviving low-register
accompaniment note. This produces a small measured improvement over V6: 18 fewer
false notes across 154 regression excerpts and one fewer across eight previously
unscored piano passages. Every previously matched attack and hold survives in
every evaluated recording. These are candidate-note metrics, not whole-song
accuracy. No user uploads were processed or existing scores regenerated.

## Training

The fit used 551 clips and **25,680 eligible events: 22,243 positive and 3,437
negative**, from the existing licensed GuitarSet/Vienna, original procedural
piano and training-only separated-mixture caches. The 139 validation clips remain
separate by performer/seed. Composition independence and upstream pretraining
overlap are not established. See the [asset notice](../backend/app/assets/left-consensus-v7.LICENSE.txt).

Labels are recomputed by one-to-one matching after V6 filtering. Earlier rejected
notes cannot return; ambiguous negatives are excluded. Equal corpus/recording
weights precede an eightfold positive-event weight.

The relation head uses 52 acoustic features plus 20 pitch-relative measurements
of nearby attacks, overlap, harmonic intervals and frozen V2 confidence. Only
neighbors within two seconds of the target onset contribute. The acoustic
guardian uses the original 52 features. Both fits use 300 trees, minimum leaf
size 40, learning rate .05, L2 4, no early stopping; relation/acoustic heads use
31/15 leaves and seeds 261004/261005.

Validation-only calibration selected raw thresholds .075/.6. The predeclared
half-margin policy froze **.0375/.3** before regression. Both policies preserve
matched attacks and holds, with no per-recording false-note increase. No test
threshold retuning was performed.

An earlier single-head candidate removed 24 regression false notes but lost one
matched piano note and was rejected. The two-head candidate with equal thresholds
had no validation gain at the safety margin and was not tested. Its separate
threshold calibration used validation only. The earlier failure informed the
new experiment, so regression is consumed evidence. All attempts are archived.

## Results

| Evaluation | Clips | Matched notes, V6 → V7 | False notes, V6 → V7 | Precision, V6 → V7 |
| --- | ---: | ---: | ---: | ---: |
| Selection validation | 139 | 12,936 → 12,936 | 4,149 → 4,131 | 75.716% → 75.795% |
| Consumed regression | 154 | 14,270 → 14,270 | 3,593 → 3,575 | 79.886% → 79.966% |
| Previously unscored piano | 8 | 723 → 723 | 287 → 286 | 71.584% → 71.655% |

Matching uses 50 ms onset and 50 cent pitch tolerance. The hold gate adds 20%
duration or 50 ms offset tolerance. Matched reference sets survive in each
recording, rather than allowing aggregate gains to hide individual losses.
Finite benchmarks cannot guarantee preservation on every upload.

The eight new sections cover seconds 90–120 (or the remaining audio) from all
eligible reserved Vienna test recordings at least 95 seconds long. Preparation
uses source duration and frozen clock corrections, never predictions. Scored
onsets exclude .25 seconds at excerpt edges. These sections share performers
and compositions with earlier tests: new notes, not independent new pieces.
They are now consumed and must enter regression in subsequent experiments.

## Runtime and verification

Only V6-retained notes in MIDI 36–59 with exact shared decoder identity and V2
confidence at most .5 are eligible. The first/last 2.5 seconds of each physical
window retain V6 decisions, protecting incomplete neighborhoods. Training and
production import the same relationship features; large annotation crops load
full original neighboring candidates rather than silently omitting context.

The release applies to balanced full-song accompaniment across uploads,
independent of filename/encoding. Independent bass-stem inference, vocal melody,
solo-piano inference, detailed mode and notation rules are outside its scope.
It cannot recover missing notes or correct pitches. Retained note objects,
attacks, releases, velocities and instrument assignments are preserved.

Both NPZ assets load without pickle, use pinned hashes and explicit feature
contracts, and fail explicitly when missing/damaged. Export probabilities match
sklearn on all 18,009 validation candidates within 1e-12. Production filter
parity passed on **37,957 scored candidate events across 301 clips**, including
full neighboring geometry/confidence and retained identity/attributes/source
assignment. Outside-crop neighbors supply context only; they are not newly
labeled evaluation.

**235 backend/training tests passed; six opt-in audio cases were skipped.** Tests
cover consensus agreement, threshold ties, invalid scores, damaged assets,
protected strong/treble/unshared/edge notes, earlier rejections, holds, crop
context and portable feature contracts. Changed Python files and backend pass
lint; broader historical training lint findings remain outside this change.

## Reproduction

Settings, searches, frozen hashes, per-recording results, export and runtime
parity are [archived](../training/results/left-relations-v7). Audio, feature
caches and local research pickle files remain outside Git. Production uses NPZ.
Use Python 3.11 / scikit-learn 1.9.0 with `PYTHONPATH=backend:training`.
Completed runs cannot be overwritten; choose a new name for a new experiment.

```bash
python training/train_left_relations_v7.py .training/note-verifier-context-expanded --run left-relations-v7-consensus --consensus --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json
python training/calibrate_left_consensus.py .training/note-verifier-context-expanded .training/note-verifier-context-expanded/left-relations-v7-consensus .training/note-verifier-context-expanded/left-relations-v7-calibrated --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json
python training/train_left_relations_v7.py .training/note-verifier-context-expanded --run left-relations-v7-calibrated --test
python training/prepare_relational_piano.py .training --fresh --test-offset 90
python training/train_left_relations_v7.py .training/note-verifier-context-expanded --run left-relations-v7-calibrated --fresh .training/relational-piano-fresh-90/manifest.json
python training/export_left_consensus.py .training/note-verifier-context-expanded .training/note-verifier-context-expanded/left-relations-v7-calibrated
python training/verify_left_consensus_runtime.py .training/note-verifier-context-expanded .training/note-verifier-context-expanded/left-relations-v7-calibrated .training/left-consensus-v7-parity.json
```
