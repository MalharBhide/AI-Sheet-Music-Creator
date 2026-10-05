# Training against the current V12 website

The original note-size correction is independent of model fitting: 7 mm default
staff scaling, default page spacing and automatic system breaks are restored.
Secondary rests remain hidden and voices retain consistent stems. No stored
user score was regenerated.

The new baseline seals the released V12 route, all six V10/V11/V12 bass arrays,
source contracts, duration-correct V11 features and the completed V12 release
witnesses. It applies V12 deletion decisions to those actual post-hold intervals.
The 52 per-note acoustic channels are independent of candidate neighbors and
may be reused only because pitch/attack/end/velocity are unchanged. Candidate
neighborhoods are recomputed after deletion. No audio inference was needed for
this baseline. All 705 recordings match the prior runtime's 642 rejections.
Fitting/validation/consumed counts are 459/122/124 with disjoint source groups.
Historical V11 guards remain unchanged and still reject the newer route.

The first experiment fits four heads on 25,746 supervised V12-retained events,
with a 76-channel relation head and 30-channel guardian including the frozen
V12 confidences. No declared checkpoint passes all validation gates with a
positive gain. Weights are withheld, with no consumed evaluation, fresh check,
export or website change.

The second experiment removes fitted-model confidences from learner inputs.
It uses 72 context channels, with observed salience defined as the fixed mean
of note strength, attack strength and relative note strength, and a 52-channel
acoustic guardian including attack/release evidence. Two fixed profiles fit
four heads; positive weights are 12/24 and guardian weight 30. Validation selects
the guarded 400-tree pair at .2/.2 strict thresholds, removing 24 false notes
while preserving all 4,648 matched attacks/holds and annotated pitch coverage.
The frozen pair removes 14 false notes from all 124 consumed regressions while
preserving all 2,344 matched attacks/holds and pitch coverage per recording.

A first-pass check uses 32 unseen original seeds 262301–262332, encoded as
48/128/192 kbps MP3 with zero codec lag. Baseline features come from the actual
native V12 route, not a simulation of earlier models. All 647 matched notes and
holds are preserved, but **no false notes are removed**. The required positive
fresh gain is absent, so this winner is also withheld. No runner-up, threshold
retuning, replacement first-pass test, export or production routing change is
allowed. Those 32 recordings are now consumed regression. These seeds are from
the existing generator and are not evidence of human-song generalization.

The third bounded batch adds 96 original timbre/percussion-leakage variants,
seeds 262401–262496, keeping every deliberately played key label unchanged.
Varying decay, stronger harmonic partials, soft clipping and unpitched leakage
can produce spurious detector candidates without introducing extra played bass
keys. True played octave keys remain in the labels. Ground-truth note times
remain unchanged. These controlled mixtures are only synthetic stress, not
proof of stem separation quality on real songs.

The new acquisition keeps 72 train and 24 validation seeds apart, uses the
unchanged native V12 baseline and the same MP3 clock checks, and reads no user
files. Combined counts are 531 train, 146 validation and 156 consumed regression.
The newly consumed 32 cannot enter fitting or selection. Both original generator
families have a predeclared future first-pass subset (16+16 unseen seeds); none
of those seeds is generated or inspected during fitting. Model release still
requires all per-recording preservation gates, positive first-pass gain,
portable/runtime parity, native routing and appropriate integration checks.

Plans, curves and withheld outcomes are preserved under training/results with
matching local evidence under .training. Checkpoints are local training artifacts;
only independently gated portable arrays can be copied into production assets.
The third batch passed all runtime, native and deployment gates and is now
released as V13; the earlier two batches remain withheld.
No thresholds, feature labels or safety requirements are tuned against consumed
or first-pass test results. Validation is reused for development and is not an
independent commercial-song accuracy claim. Shared compositions, synthetic
limitations and upstream training overlap remain.

The third frozen winner is the guarded 400-tree pair at strict .025/.3
thresholds. Validation removes 69 false notes, preserving all 5,068 matched
attacks/holds; consumed regression removes two, preserving all 2,991. Every
recording also preserves supplied pitch intervals. The predeclared first pass
uses 32 new seeds 262601–262632, 16 from each original generator: 36 false notes
removed, all 604 matched notes preserved. F1 rises from .77238 to .79058 with
recall unchanged. These 32 are now consumed regression; do not tune on them.
See docs/bass-texture-v13.md for the final release witnesses and limitations.
