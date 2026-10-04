# Pitch-preserving accompaniment V8

The release removes three additional unmatched detections on validation and one
on consumed regression, preserving every previously matched attack, onset-offset
reference and annotated pitch interval in every evaluated recording. The later
reserved piano passages are unchanged. This is a **very small** candidate-note
improvement; it is not evidence of accurate whole-song piano arrangements.
No user upload was processed and no saved score was regenerated.

## Data and fitting

Completed 50 fixed GuitarSet MP3/percussion/Demucs augmentations: 40 training,
10 validation, with no test performer. Bitrates 48/128/192 kbps and percussion
levels −6/0/+3 dB cycle by source order. All codec clock checks report zero lag.
Original licensed annotations are retained; generated media and feature caches
remain local. The training cohort totals 660 clips and validation 168 clips.

A pitch-support label policy marks otherwise ambiguous retained events as keep
examples when at least 90% of the detection is covered by annotated same-pitch
intervals. It adds 4,886 eligible training examples (64,233 total) and protects
continuations that have no separate annotated attack. Evaluation attack labels
are unchanged. Four predeclared paired boosted-tree profiles were fitted;
validation selects the shallow profile, 800 trees per head, at .005/.05 after a
50% threshold safety margin. Both heads must reject a candidate. Features are
74 relation/confidence and 54 acoustic/confidence values, without song identity.

## Preservation gates and measured outcomes

| Cohort | Clips | Unmatched detections removed | Preservation |
| --- | ---: | ---: | --- |
| Validation | 168 | 3 | All matched attacks, offsets and annotated pitch coverage |
| Consumed regression | 162 | 1 (3,861 → 3,860) | All 14,993 matched detections retained; same gates |
| Later reserved piano | 3 | 0 | Same gates; all 49 matches retained |

The three later piano sections share reserved performers/compositions with
previous tests and are not independent generalization evidence. Upstream model
pretraining overlap is not audited. Annotation crops may omit holds whose attack
precedes the crop; coverage guarantees apply to the supplied reference intervals.

Arrays-only export matches every validation prediction within 1e-12. Production
filtering matches frozen decisions across 333 clips and 42,446 scored events.
Retained notes preserve object identity, source part, pitch, timing and velocity.
Outside-crop neighbors supply geometry; parity does not certify their decisions.
The release changes accompaniment filtering only: previous V7 rejections persist,
physical window edges and nonshared candidates remain protected. Dedicated solo
piano, vocal melody and independent bass-stem inference have not been retrained.

## Rejected experiments and next training

An initial hold-repeat label experiment was interrupted because it could delete
a continuation after an incomplete first detection. The corrected coverage-based
label rule produced zero new eligible supervision; it reproduces earlier model
hashes and is not an improvement. A new no-op guard prevents redundant refitting.

The first MP3-augmented winner removed two unmatched regression detections and
passed attack/offset checks, but shortened reference pitch coverage in four
validation/regression recordings. It was withheld without fresh evaluation.
The new interval-coverage gate caught that failure, and subsequent selection
requires the gate at raw and half-margin thresholds as well as frozen release.

A training/validation-only boundary audit found 2,557 split-hold and 16,735 genuine
repeat examples. A separate learned boundary model can merge spurious attacks
while preserving continuous pitch coverage, instead of deleting audible holds.
This audit is supervision evidence, not a trained accuracy improvement.

`current_accompaniment_baseline.py` pins the actual V8 release for future work.
Older V7 preparation helpers remain historical reproducibility tools. New
experiments must compare against the current baseline and preserve completed
results; consumed tests must never be used to tune a frozen winner.

All candidate selections, rejected outcomes, gate code versions, source hashes,
learning curves, MP3 clock metadata and release parity are in
[the archived results](../training/results/pitch-preserving-v8).
Dataset and model-array credit is in [the license](../backend/app/assets/pitch-preserving-consensus-v8.LICENSE.txt).

## Verification

293 backend/training tests passed; six opt-in audio integration cases skipped.
The additional current-baseline preservation test passed. Backend build and idle
queue deployment passed. The running website loads the tested hashes at .005/.05,
and frontend-proxied health reports ready. No transcription job was created.
