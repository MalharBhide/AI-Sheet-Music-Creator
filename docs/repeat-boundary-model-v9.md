# Learned repeat-attack model V9

V9 joins model-verified touching fragments of a held accompaniment note. Unlike
removing a fragment, joining preserves its audible pitch span. The two new
classifiers learn genuine repeated attacks versus decoder splits from existing
licensed annotations. They do not force a tempo or merge across even a short rest.
No user upload was processed and no existing score was regenerated.

## Training and conservative selection

The 660-clip training cohort includes GuitarSet, Vienna, original procedural
piano and the completed training-only MP3/percussion/separation augmentations.
There are **19,292 eligible labeled boundaries**: 16,735 genuine repeated attacks
and 2,557 split holds. Validation uses 168 clips from its established partition.
Only V8-retained, exact shared interior accompaniment events enter a boundary.
Nearest same-key fragments must be spaced 0.15–2 seconds, with near-contiguous
observed edges. Genuine annotated attacks within 50 ms are keep labels; ambiguous
nearby attacks and unsupported holds are excluded from supervision.

Two profiles were declared before fitting, with positive repeat weights 30/50,
350/500 trees, 15 leaves, learning rate .04 and no early stopping. Paired
relation/acoustic heads have 152/112 features from both fragments and four timing
values. These features contain no labels, song name or absolute pitch.
The recall profile wins solely on gated validation gain, using fixed order for
ties. Thresholds .05/.025 are frozen after a 50% safety margin. Regression does
not choose a runner-up or adjust those thresholds.

Both heads must classify a boundary as a split. Merging preserves the first
fragment's pitch, onset, velocity and source part and extends its end through the
joined fragment. Chain ownership is explicit. Nonshared events, prior rejections,
window edges and positive gaps remain protected. Every evaluated recording must
preserve matched attacks, onset-offset reference matches, annotated pitch
coverage and the **exact complete detected pitch-time union**, with no increase
in unmatched detections. Real repeated notes that were already recognized cannot
be merged away by a passing candidate.

## Frozen outcomes

| Cohort | Clips | Extra attacks removed | Preservation |
| --- | ---: | ---: | --- |
| Validation | 168 | 7 | Every per-recording gate passes |
| Consumed regression | 162 | 2 (3,860 → 3,858) | All 14,993 matches retained |
| Latest consumed piano sections | 3 | 0 | Every gate passes |
| New MP3/separation stress | 10 | 0 | Every gate passes |

The stress cases use the second sorted comp/solo recording in each GuitarSet
genre, reserved performer 05. Original percussion, MP3 bitrates 48/128/192 kbps
and backing levels −6/0/+3 dB are fixed before inference. All codec checks report
zero timing drift. These are previously unscored conditions on familiar test
performers/compositions, not independent recordings. This small improvement does
not establish accurate full-song piano arrangements or guarantee unseen-audio
accuracy. Dedicated solo piano, vocals and bass-stem models are unchanged.

Arrays-only export agrees with every validation boundary prediction at 1e-12.
Production filtering matches the frozen model across **343 recordings / 43,607
scored events**, including the exact expected held-note extensions and retained
source identities. Outside-crop neighbors provide context; their decisions are
not certified by this parity audit. The two frozen profiles, curves, gate
outcomes, codec/source hashes and original gate version are in
[the result archive](../training/results/repeat-boundaries-v1).

## Verification and deployment

318 backend/training tests passed; six opt-in real-audio integration cases skipped.
Checks cover real repeats, held-note chains, source ownership, exact pitch spans,
head vetoes, threshold ties, invalid confidence, damaged weights and frozen
release evidence. Lint and diff checks passed. Backend build and idle queue
deployment passed. The running website loads the tested V9 weights at .05/.025
and frontend-proxied health reports ready. No transcription job was created.

Older `current_accompaniment_baseline.py` and V7 preparation helpers pin their
historical V8/V7 studies. New experiments must include the V9 boundary decisions
and compare against V9 rather than claiming a gain over a superseded model.

Dataset and model attribution: [license](../backend/app/assets/repeat-boundary-v1.LICENSE.txt).
