# V6: trained correction for remaining left-hand errors

The new model increases measured accompaniment precision over the deployed V5
baseline, with matched attacks and holds preserved in every evaluated recording.
The gain is **small**: five fewer false notes on 142 consumed regression excerpts
and one fewer on 12 previously unscored piano passages. This is not evidence of
accurate full-song piano arrangements or a large perceptual improvement.

No user recordings were transcribed and no existing scores were regenerated.
The update applies to future balanced full-song uploads, independent of their
filename or audio encoding.

## What changed in training

The earlier replacement candidates could restore mistakes already removed by
V5. This experiment instead freezes V5 decisions and fits a residual classifier
to remaining low-register candidates. A V5 rejection cannot be reversed. This
adds a small correction model rather than lowering existing thresholds.

Training labels are recomputed with one-to-one reference matching **after** V5
filtering. If a duplicate candidate was removed, its surviving counterpart can
become the matched positive; stale pre-filter labels must not teach the model to
remove that surviving correct note. Ambiguous timing negatives remain excluded.

The fit used **32,067 events: 27,757 positive and 4,310 negative**, from 551
training clips. These are the expanded licensed GuitarSet/Vienna/original-piano
and training-only separated-mixture caches, including the previously prepared
64 later GuitarSet training passages. The 139 validation clips remain separate
by performer/seed; their labels never enter fitting. Source credits and licenses
are in the [asset notice](../backend/app/assets/accompaniment-left-refinement-v3.LICENSE.txt).

Fixed settings: 300 boosted trees, at most 31 leaves, minimum leaf size 40,
learning rate .05, L2 4, no early stopping, seed 260925. Recordings/corpora receive
equal base weights before an eightfold positive-event multiplier. Features are
the same 52 pitch-relative acoustic measurements used in production.

Validation selected raw threshold .075. The predeclared half-margin froze
**.0375**, with all per-recording gates passing at both thresholds. The checkpoint
and policy were frozen before regression. No post-test retuning was performed.

## Measured results

| Evaluation | Clips | Matched notes, V5 → V6 | False notes, V5 → V6 | Precision, V5 → V6 |
| --- | ---: | ---: | ---: | ---: |
| Validation used for selection | 139 | 12,936 → 12,936 | 4,157 → 4,149 | 75.680% → 75.716% |
| Consumed regression | 142 | 12,879 → 12,879 | 3,298 → 3,293 | 79.613% → 79.638% |
| Previously unscored piano sections | 12 | 1,391 → 1,391 | 301 → 300 | 82.210% → 82.259% |

Both onset matches and onset-offset reference matches survive in every recording.
Matching uses 50 ms onset/50 cent pitch tolerance; the hold check adds 20% duration
or 50 ms offset tolerance. Recall is unchanged and precision/F1 improve. These
are candidate-note metrics, not percentage accuracy of whole songs or notation.

The new piano evaluation covers seconds 60–90 (or the remaining recording) from
every eligible reserved Vienna test recording at least 65 seconds long, excluding
.25 seconds at each edge from scored onsets. Selection uses duration and the
existing frozen clock corrections, not predictions. These passages were prepared
only after regression passed. They share test performers and compositions with
earlier excerpts: new note sections, **not independent new performers/pieces**.
All evaluations are now consumed; they must not be relabeled as fresh in future
experiments. A finite benchmark cannot guarantee preservation on every upload.

## Runtime and scope

Only V5-retained accompaniment notes in MIDI 36–59 are eligible, and only when
they exactly match the shared candidate decoder and their original V2 score is
at most .5. Treble accompaniment, vocal melody, independent bass-stem inference,
dedicated solo-piano inference and arrangement/quantization rules are unchanged.
Retained note objects, pitches, onsets, releases, velocities and source assignments
are preserved. The classifier cannot recover missing notes or fix wrong pitches.

The NPZ arrays load without pickle and have SHA-256
`6b0f9bb5260f9645831dd409abd6fcb17cd8acbdd09e7a13df9c356cd76414e2`.
Missing/damaged weights fail explicitly; readiness includes the new asset.
187 tests passed, with six opt-in audio integration cases deliberately skipped.
Checks include baseline rejection preservation, relabeling duplicates, protected
treble/strong events, held notes, threshold boundaries and damaged weights.
Production/offline parity passed on **36,833 candidate events across 293 clips**,
including preservation of retained note identity, attributes and source assignment.
Lint and the production backend build passed.
After an idle-queue check, the deployed backend loaded the verified checkpoint at
.0375 and reported ready through the frontend health proxy. No transcription job
was created to verify deployment.

## Reproduction

Settings, frozen baseline/checkpoint hashes, validation search and per-recording
evaluations are [archived](../training/results/left-hand-refinement-v3). Use the
existing Python 3.11 environment, scikit-learn 1.9.0 and `PYTHONPATH=backend:training`.

```bash
python training/train_left_hand_expanded.py .training/note-verifier-context-expanded --run left-hand-refinement-v3 --leaves 31 --policy refinement --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json
python training/train_left_hand_expanded.py .training/note-verifier-context-expanded --run left-hand-refinement-v3 --test
python training/prepare_relational_piano.py .training --fresh
python training/train_left_hand_expanded.py .training/note-verifier-context-expanded --run left-hand-refinement-v3 --fresh .training/relational-piano-fresh/manifest.json
python training/verify_left_refinement_runtime.py .training/note-verifier-context-expanded .training/left-refinement-parity.json
```

Completed training/evaluation artifacts cannot be overwritten. Use a new run
name for a new experiment; fresh evaluation is blocked until regression passes.
