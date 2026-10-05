# V10: trained bass note consensus

V10 keeps V9 accompaniment/hold processing and adds a separate learned filter
for the **balanced Full-song bass stem**. It uses the actual frequency-bounded
Basic Pitch events; all retained pitches, starts, ends, velocities and source
parts remain unchanged. No user upload was processed or saved score regenerated.
The local website is healthy and loads the exact tested arrays.

## Training

The paired 52-feature context / 26-feature acoustic classifiers fit **22,684**
eligible labeled events from **359 training clips**, with **94 validation clips**.
Data include CC BY 4.0 [BabySlakh v2](https://zenodo.org/records/4603870),
[licensed GuitarSet](https://zenodo.org/records/3371780), and 80 original timbre
seeds. Oracle bass and full-mixture Demucs variants keep their whole-source
partition. Demucs labels protect real low notes from every rendered pitched
source, including piano/guitar. Guitar performers and original source seeds
stay disjoint across training and validation; four Slakh groups remain reserved
until validation and consumed stress qualify.

Original MIDI key releases remain separate from CC64/crop-crossing pitch support.
Ambiguous timing detections are excluded from negative supervision; supported
held-note fragments are positive KEEP labels. Features contain no song name,
key signature or absolute pitch. Inference protects incomplete start/end
neighborhoods and pitches outside 21–59. Both heads must reject an interior
candidate; confidence equality is retained. No retained note is shortened or
shifted, and no missing note is synthesized.

Four predeclared paired profiles and staged checkpoints are selected only on
validation. Every recording must preserve matched attacks, onset-offset matches
and each supplied pitch-support interval, with no false-note increase, at both
raw and half-margin thresholds. The regularized **400-tree** pair wins at
**.1/.1**. Tests do not choose a runner-up or retune thresholds.

## Frozen release evidence

| Cohort | Clips | Matched notes retained | False detections |
| --- | ---: | ---: | ---: |
| Validation | 94 | 4,150 | 3,030 → 2,888 |
| Consumed original bass stress | 8 | 210 | 10 → 10 |
| Previously unscored Slakh groups | 20 | 357 | 1,091 → 1,050 |

Every individual attack, offset and pitch-coverage gate passes. The reserved
check improves bass F1 **.2846→.2893**, with recall unchanged at **.3365**. This
is a small false-note reduction; missing notes remain a substantial limitation.
Slakh is synthesized and small. Composition/near-duplicate/pretraining overlap
is not ruled out; these results do not establish accurate unseen commercial-song
arrangements. Other model routes retain their existing training and behavior.

The earlier bass-only fit removed many validation false notes but lost 112
genuine attacks in stress; it was withheld. Broader timbre/guitar-positive
supervision preserves all of those regression matches. Staged selection also
retains an earlier checkpoint when further training lowers loss but reduces
measured note improvement. Interrupted empty-window fitting is archived;
no-note windows now bypass sklearn safely. All experiments remain available in
[the training framework](bass-consensus-training.md).

## Verification and deployment

Portable/runtime comparison covers **481 clips / 36,989 candidates**, including
all additional validation and both release cohorts: probabilities match sklearn
exactly, final masks match, and retained identity/source/timing attributes are
unchanged. Native checks cover actual Basic Pitch routing/features on original
holds, repeats, octaves, a licensed bass excerpt, 32-second context and subframe
audio. The native bass sample removes three detections among 109 candidates
with exact expected event tuples.

**355 backend/training tests pass**, plus all **six native pipeline cases**
(129 seconds): WAV, long VBR MP3, short/silent MP3, Full-song MP3, multipage
rendering, every download format and playback positions. Model loading, damaged
assets, threshold equality, edge/register protection, independent source-part
identity, balanced routing and detailed-mode protection are covered.

Backend build passes. Deployment checked an idle queue, then the frontend health
proxy reported ready with bass models present. The live service loads two .1
threshold heads with 52/26 features; source and routing hashes match native
evidence. Its three existing completed jobs remain unchanged. New uploads use
this release; existing saved scores retain their original model output.

Frozen outcomes, manifests, learning curves, selected checkpoints and parity
reports are under `training/results/bass-positive-consensus-v2b` and
`training/results/slakh-bass-positive-reserved-v2`. Model arrays contain no
executable pickle and ship with source attribution.
