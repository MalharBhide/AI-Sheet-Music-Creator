# Joint timing experiments

These models test corrections of systematic attack delays while retaining every
release and every repeated-key interval. They do **not** correct the interval
pattern itself. Uniform shifting is deliberately narrower than arbitrary timing
changes: a key moves only when all its occurrences agree on the same direction.
Ineligible, conflicting or deleted occurrences prevent a partial shift. Existing
holds, pitches and velocities are unchanged; no new same-key overlap is allowed.

## V20: trained, frozen, rejected for insufficient first-pass improvement

Two dual-scale CNN profiles train for 60 epochs on 573 training recordings and
select solely on 160 validation recordings. The original commercial-compatible
[V16 dataset notice](../backend/app/assets/bass-temporal-refinement-v1.LICENSE.txt)
applies. Inputs reuse the sealed attack-local acoustic evidence from V19; labels
are recomputed from training/validation references only. No user audio or saved
score is used or regenerated.

The winner is `consensus-attacks` epoch 20, timing confidence .8, removal disabled
at 1.01. Checkpoint SHA256:
`d55427d213a9a8ec735569c76ead3f2482841eb665257a2fd3b6a6acb158ad9f`.

| Evaluation | Raw retimed notes / summed onset gain | Stricter, half-strength retimed notes / gain | Per-recording preservation and timing gates |
| --- | --- | --- | --- |
| 160 validation recordings | 9 / .18 seconds | 1 / .01 seconds | Pass |
| 296 consumed regressions | 14 / .141 seconds | 10 / .05 seconds | Pass |
| 32 first-pass procedural MP3 clips | 1 / .02 seconds | 0 / 0 seconds | Pass |

Every recording preserves previously matched attacks, holds and pitch coverage;
false-positive counts and onset/release/spacing error never increase. Validation
has one fewer unmatched note and consumed regression has two fewer because of
corrected timing. No notes are physically removed by this winner.

The frozen first-pass policy requires positive onset gain for both configurations.
The stricter configuration has zero first-pass gain, so the candidate is rejected.
Passing preservation gates does not override that predeclared requirement.
No portable export, ML deployment, threshold retuning, replacement checkpoint
or replacement first-pass seeds follows this result. V16 remains deployed.

The 32 original/texture clips use project-authored signals encoded at 48/128/192
kbps, with seeds 265101–265132 declared before fitting. They are now consumed
regressions. All 328 recordings must remain in subsequent regression gates.
These results are development evidence, not arbitrary-song accuracy estimates.

V20 [fitting evidence](../training/results/pitch-consensus-v20-v1) includes exact
plans, selection, learning curves, per-recording evaluations (losslessly gzipped),
local checkpoint hashes and checks. The [first-pass manifest](../training/results/pitch-consensus-v20-fresh-v1)
contains source/cache hashes; audio, feature caches and weights stay local.

## V21: rejected for regression failures

Training labels are heavily imbalanced: 464 EARLY, 24,322 KEEP, 39 LATE and 5,667
REMOVE events. V21 increases the loss weight of rare timing labels in two fixed
profiles and trains for 80 epochs. Its fitting split, acoustic inputs, decoder
and preservation gates are unchanged. All 328 consumed recordings are excluded
from fitting and checkpoint selection.

V21 also rounds summed onset gains to nine decimal seconds during checkpoint
ranking. Without this, floating-point differences smaller than a picosecond can
choose a worse validation loss when the actual timing improvements are equal.
This rule is frozen before V21 fitting; V20's original selection stays untouched.

The single validation winner must pass all 328 consumed regressions and 32 new
predeclared procedural MP3 clips (266101–266132), under both normal and stricter,
half-strength corrections. Failing either prevents export or deployment. No
previously tested clip is reused as a fresh first-pass result.

The validation winner, `consensus-selective` epoch 80, passed validation but
failed five of the 328 consumed recordings under normal corrections and one
under stricter corrections. Fixed releases do not guarantee preserved pitch
coverage: moving an attack later can shorten a supported interval. No new
first-pass audio was generated, and no weights were exported or deployed.
Seeds 266101–266132 remain unused. The interrupted fitting run was replayed
with the same frozen plan; all seven checkpoints completed before interruption
were byte-identical to the replay. Exact evaluations and inventories are in
[the V21 archive](../training/results/pitch-consensus-v21-v1).

## V22: early-only corrections rejected for zero first-pass improvement

This experiment disables later attacks and teaches the CNN only group-wide
early corrections with more than .018 seconds of supported gain per event.
Retained intervals can expand but cannot shrink. Two moderate class-weight
profiles train for 80 epochs on the same 573 training recordings, with selection
on the same 160 validation recordings. All 328 consumed regressions remain
excluded from training and selection.

The winner is `early-supported` epoch 20, confidence .8, removal disabled at
1.01. Its checkpoint SHA256 is
`0ace1a70c5252c6f2af73bb893c254bbaf3b166271d5df3ae0319134b4b40655`.

| Evaluation | Raw retimed notes / summed onset gain | Stricter, half-strength retimed notes / gain | Preservation and timing gates |
| --- | --- | --- | --- |
| 160 validation recordings | 29 / .554744 seconds | 14 / .14 seconds | Pass |
| 328 consumed regressions | 10 / .098951 seconds | 7 / .04 seconds | Pass |
| 32 first-pass procedural MP3 clips | 1 / 0 seconds | 0 / 0 seconds | Pass |

The final MP3 test has no improvement under either configuration. The candidate
therefore fails its frozen release policy even though preservation gates pass.
No export or deployment follows, and no alternative checkpoint is selected.
The previously approved V16 weights remain on the website. These results do
not establish accuracy on arbitrary commercial songs.

Seeds 267101–267132 are now consumed; subsequent candidates must evaluate all
360 consumed recordings. [V22 fitting and evaluation evidence](../training/results/early-consensus-v22-v1)
and [the first-pass manifest](../training/results/early-consensus-v22-fresh-v1)
preserve exact reports and hashes without distributing audio, feature caches
or experimental checkpoints.
