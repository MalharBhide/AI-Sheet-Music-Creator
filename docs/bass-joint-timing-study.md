# Learned bass attack timing: V17 and V18 development batches

Four temporal CNN candidates were trained specifically to correct attack
spacing and reduce unsupported bass notes. None passed the complete validation
release gate. **The website still uses the released V16 model and the previously
verified fast-repeat notation decoder.** No new weights or timing corrections
from these experiments are deployed. Original 7 mm notation and pagination are
unchanged. No user recording or saved score was transcribed or regenerated.

## Current-model fitting baseline

The new baseline applies the released seventh bass stage to all 1,029 existing
labeled recordings. Exact asset, source, cache, audio-clock and completed V16
release witnesses are checked; older guards are preserved. V16 rejects 71 of
55,163 V15 survivors, leaving **55,092 actual current-model predictions**.
Per-note acoustic/CQT evidence remains valid because V16 only deletes notes.
Neighbor features are recomputed after deletion.

Partitions remain 573 training, 160 validation and 296 consumed regression
recordings, with disjoint source groups. Only training/validation caches are
read by the fitting runner. There are 30,492 eligible supervised training events
across 570 nonempty training recordings. Sources are the previously audited
GuitarSet, Slakh, Vienna piano recordings and project-authored procedural audio;
commercial compatibility and attribution follow the existing
[V16 model notice](../backend/app/assets/bass-temporal-refinement-v1.LICENSE.txt).
Shared compositions and upstream model pretraining prevent claims of
independent human-song generalization.

Baseline manifest SHA256:
`72e8de4841f9b401bd07fdd9949cbf3f9c3c31ab2af90ef18665376a0ba9be82`.
The baseline contains predicted notes and references, not audio or user files.
The archived manifest uses gzip compression of the exact original JSON bytes;
its uncompressed SHA256 remains the value above, and the local plain manifest
remains available to the fitting loader.

## Model, transformations and release measurements

Each batch predeclares two width-48 CNN profiles, 40 epochs, 40 updates per
epoch, batch size 256, AdamW and checkpoint stages 10/20/40. The deployed V16
encoder/hidden layers initialize fitting; a new four-class output learns
20 ms early, unchanged, 20 ms late, or remove. Inputs remain observed 40-step,
nine-channel pitch-relative CQT sequences and 84 acoustic/neighbor features.
Normalization, corpus/recording weighting and all gradient updates use training
examples only. Fitted confidences, reference annotations, filenames and song
identities never enter inference features. The frozen V14 acoustic guardian
protects supported notes from deletion.

Retained pitches and velocities cannot change. Corrections protect short notes,
same-key attack order, genuine rests and clip edges. A touching same-key release
moves with its repeated attack; this is measured against release annotations,
rather than assumed correct. Pitch correction and missing-note recovery remain
outside this model's scope.

Release gates compare each recording separately:

- Every previously matched 50 ms attack and offset-aware hold stays matched.
- Every supplied reference-pitch interval retains its prior coverage.
- False positives cannot increase.
- Onset error, release error and repeated-key interval error cannot increase.
- Both normal and stricter-confidence/half-strength corrections must have
  positive measured onset improvement.

Timing uses fixed baseline same-pitch pairs within 200 ms. Deleting a prediction
retains its old error and denominator, so removing difficult notes earns no
timing credit. This prevents a cleaner-looking score from concealing lost rhythm.

V17's confidence margin added .05; at .99 this disables all actions. Its
predeclared gate was honored and the rejected results retained. V18 declares a
relative margin `t + (1-t)/2`, with disabled 1.01 remaining disabled. No attack,
hold, coverage, false-positive or timing preservation gate was relaxed.

V18 also aligns fitting labels with release and spacing checks. It changes 476
otherwise proposed training corrections into KEEP counterexamples. Early/late
labels fall from 2,542/1,104 to 2,178/992; 5,667 unsupported-note REMOVE labels
stay unchanged. This reduces, but does not eliminate, unsafe corrections.

## Results and release decision

Validation-only selection rejects all four candidates. No consumed-regression
model scoring, fresh MP3 inference, portable export or website deployment is
performed after the failed validation gates. Reserved first-pass seeds remain
unused: V17 263401–263432, V18 263501–263532. They are not fresh-result evidence.

The diagnostic below uses **final epoch 40**, raw timing confidence .95 and
removal confidence .99, purely to explain rejection. These settings are not
selected release candidates. Error reductions are sums over fixed validation
note pairs, not percentages or general transcription accuracy.

| Batch/profile | Retimed notes | Onset error reduction (seconds) | Fewer false notes | Failing recordings |
| --- | ---: | ---: | ---: | ---: |
| V17 protected | 39 | 0.69718 | 9 | 5 |
| V17 attacks | 187 | 3.12462 | 22 | 15 |
| V18 protected | 3 | 0 | 0 | 1 |
| V18 attacks | 42 | 0.51847 | 13 | 4 |

V18 attacks substantially reduces the number of failing recordings in this
diagnostic, but still worsens attack timing on three recordings and a touching
release on another; one of those attack cases also worsens repeated spacing.
None loses a matched attack/hold or supplied pitch coverage at these diagnostic
settings. The timing regressions alone are sufficient to reject deployment.
Aggregate improvements do not override individual regressions.

Committed evidence is under
[V16 baseline](../training/results/bass-v16-baseline-v1),
[V17 batch](../training/results/bass-joint-timing-v17-v1) and
[V18 batch](../training/results/bass-joint-timing-v18-v1).
All twelve fitted checkpoints remain local, with hashes archived; none is a
website asset. Source/plan hashes are checked again at fitting completion.

Verification passes 631 application/training tests and six native audio-to-score
integration tests. A read-only live check confirms the website is ready and
still uses the approved V16 weights and fast-repeat decoder. The 25 focused
tests cover shared releases, rest/order preservation, deletion-free timing
credit, rhythm regression detection, four-class inference, weighting, label
counterexamples and conservative confidence margins.

The next experiment needs more precise attack-local acoustic evidence or
phrase-aware timing predictions. Relaxing the gates or regenerating a user's
score would not establish an improvement in the model.
