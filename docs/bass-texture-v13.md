# V13: acoustic rejection of extra bass notes

This release trains two new gradient-boosted classifiers on the notes that
actually survive the V12 website bass route. It targets unsupported left-hand
pitches in balanced Full-song piano arrangements. Both classifiers must reject
before a note is removed. Basic Pitch, Demucs, solo-piano, vocal and existing
accompaniment networks retain their weights. No user uploads were transcribed,
used for fitting, or regenerated.

## Baseline and data

The baseline seals all six V10/V11/V12 arrays, their source contracts and the
completed V12 release witnesses. It reconstructs actual post-hold V12 intervals
from immutable caches, with 642 V12 rejections across 705 recordings. Features
from longer merged notes remain duration-correct. The 52 independent acoustic
channels can be reused after deletion only because each survivor keeps its
original pitch, attack, end and velocity; neighborhoods are recomputed.

Two earlier experiments remain withheld. The confidence-feature experiment had
no safe positive validation winner. The first acoustic-only winner passed
validation and consumed regression but removed no false notes on its 32 new
original MP3 fixtures. It failed the positive first-pass gain requirement. There
was no test-driven threshold adjustment, runner-up or replacement fresh test.
Those fixtures became consumed regression, never fitting or selection data.

The released batch adds 96 project-authored texture variants: 72 training and
24 validation. Seeds 262401–262496 vary decay, attack, harmonic partials, soft
clipping and unpitched percussive leakage while keeping all deliberately played
key labels unchanged. True played octaves remain positive supervision. MP3
bitrate variants use the same decoded audio clock checks. Existing labeled
BabySlakh and GuitarSet data are CC BY 4.0; attribution accompanies the arrays.
The combined partitions are 531 training, 146 validation and 156 consumed
recordings, with disjoint source groups.

## Fitting and gates

The context head uses 72 features: per-note acoustic evidence and pitch-relative
candidate relations. Its observed salience is the fixed mean of note strength,
attack strength and relative note strength. The guardian uses all 52 per-note
acoustic channels. Neither head receives fitted-model probabilities, recording
names, annotations or absolute pitch as learner features.

Two predeclared profiles fit four heads with a 600-tree limit, learning rate
.04, positive weights 12/24 and guardian weight 30. Checkpoints and threshold
grids are fixed before fitting. Validation alone selects the guarded 400-tree
pair, with strict .025/.3 thresholds after the declared safety margin. Every
matched attack, offset-aware hold and supplied pitch interval must be preserved
in every recording. No recording may gain false positives.

| Cohort | Matched attacks unchanged | False notes before → after |
| --- | ---: | ---: |
| 146 validation recordings | 5,068 | 4,155 → 4,086 |
| 156 consumed regression recordings | 2,991 | 1,676 → 1,674 |
| 32 first-pass original MP3 fixtures | 604 | 273 → 237 |

The first-pass seeds 262601–262632 were declared before fitting: 16 ordinary
original signals and 16 texture variants, encoded at 48/128/192 kbps. Every
previously matched attack/hold and annotated pitch-time interval is retained
per recording. First-pass F1 increases .77238 → .79058 with recall unchanged
at .87918. All 32 fixtures are now consumed regression; future experiments
cannot use them as fresh evidence or fit them into a later model.

This is a limited-corpus improvement. The fresh seeds belong to the two existing
synthetic generator families, not independent human performances. Validation
is repeatedly used during development. Shared compositions and upstream
pretraining overlap remain possible. Licensed validation recall is only .53140;
this filter cannot recover missed pitches or establish accurate transcription
of arbitrary commercial songs. An independent commercially usable human-song
benchmark is still needed.

## Runtime and engraving

The stage runs after V10 verification, V11 hold repair and V12 rejection on
bass pitches 21–59, at least 2.5 seconds inside each bounded inference window.
Either-head support or threshold equality vetoes deletion. All retained note
objects keep their source part, pitch, onset, endpoint and velocity. The stage
cannot add notes, transpose pitches, change rhythm or revive earlier deletions.
Metadata records texture rejections separately from prior stages. Checksums,
feature names and thresholds are enforced when the arrays load; readiness
requires both new arrays.

Portable/runtime parity covers all 865 cached recordings and 48,113 retained
notes. Feature values and probabilities match sklearn exactly (maximum error
zero at 1e-12 tolerance). The 315 rejected notes include fitting examples and
are not independent accuracy evidence. Pre-routing witnesses are sealed before
changing production routing. Native checks compare the full four-stage bass
route to frozen sklearn, using actual audio and detector features, with held
notes, repeats, octaves, licensed Slakh, observed new rejections, 32-second
windows and subframe audio.

The original 7 mm staff scale, default page spacing and automatic pagination
remain unchanged from abb46fe. A 96-bar original layout fixture shrank from six
pages to four while preserving all 960 audible notes and 672 seek positions.
Stored user scores are not regenerated: a new upload uses the released model
and original note size.

Evidence is archived under training/results/bass-texture-learner-v3 and the
associated preparation and first-pass directories. Fitted pickle checkpoints
remain local; only independently gated portable arrays ship. Historical V12
helpers deliberately reject V13 routing. Future fitting needs a new baseline
with all four stages, actual surviving intervals and all 188 consumed tests.

Release verification: 466 backend/training tests and all six native upload/render
integrations passed, along with all eight native bass parity cases. Changed
Python files pass lint. The backend build and idle-queue deployment passed; live
source/asset hashes match the checked release through the frontend proxy. All
eight completed user jobs remain unchanged. Frontend sources are unchanged.
Detailed evidence is in release-checks.json and live-check.json.
