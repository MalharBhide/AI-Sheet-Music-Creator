# V16 bass temporal refinement: expanded real-piano training

This study trains a new temporal bass KEEP classifier on actual V15 survivors
and additional real acoustic-piano examples. It follows all six existing bass
stages, including V15, and cannot restore their rejected notes. Original 7 mm
notation, spacing and automatic pagination remain unchanged. No user recordings or saved scores are used or regenerated.

## Sealed baseline and added data

The six-stage V15 baseline exactly reproduces its 74 runtime deletions across
929 recordings, leaving 49,140 candidates. It retains 531 training, 146
validation and 252 consumed regression recordings. Cached note-centered CQT
frames and interval-independent acoustic channels can be reused because the
released stage changes no retained pitch, onset, endpoint or velocity. Candidate
relationships are recomputed after deletion. Exact historical supervision
parity passes all 929 recordings and 49,140 surviving notes. Older V14 guards
remain unchanged and reject the newer route.

The added data are Werner Goebl's Vienna 4x22 acoustic-piano performances,
CC BY 4.0, DOI 10.21939/4X22, already acquired and checked locally. Corrected V3
audio-only clock calibration is reused. The historical root clock file has
incorrect trimmed Chopin anchors and is explicitly excluded. Training uses
performers 01–14; validation uses 15–18. Reserved performers 19–22 never fit
or select this candidate. Synthesized average performer 23 and special
1st–3rd Ballade variants remain excluded.

A predeclared duration rule selects complete 30–60-second crops: 42 training
and 12 validation performances. Audio is encoded/decoded at 48/128/192 kbps
with verified codec clocks. Instrument key releases are separate from CC64
pedal-supported pitch intervals; pre-crop holds remain protected. No annotation
clock is adjusted from model predictions.

The preparation also declares Demucs variants for performers 01, 14, 15 and
18. Production separation emits two nonempty bass sources, both validation
examples. Ten no-bass variants are audited and excluded; no silent fallbacks
are fitted as musical negatives. Thus the **added fitting recordings are piano
register counterexamples**, not newly acquired real separated bass training.
This limitation must not be hidden. Future acquisition may add more train-side
separated recordings under a new declared experiment.

The resulting batch has 573 training and 160 validation recordings, with
30,538 eligible supervised training events. The sources share compositions and
recording instrument, and these development performers were used previously
for accompaniment work. They are not independent evidence of general song
accuracy.

## Frozen fitting and measured results

Two predeclared width-32/48 CNN profiles fit from scratch for 60 epochs with
40 updates per epoch, positive weights 12/24, AdamW and fixed checkpoint and
threshold grids. Inputs use the unchanged 40-step, nine-channel pitch-relative
CQT transform plus 84 observed acoustic/neighbor features. Normalization uses
eligible supervised training rows only. The V14 acoustic guardian is frozen;
no fitted confidence enters the CNN input.

Validation alone selects conservative epoch 60, strict thresholds .2/.3 after
the declared safety margin. Its checkpoint SHA256 is
`993095e4ecaae7d9f9cda01042860442f81aaddc9b5c9be94ffae9b5307c302e`.
The runner-up is not evaluated as an alternative.

The selected candidate removes five false notes on validation and preserves
every recording's matched attacks, offset-aware holds and supplied pitch
coverage. The single frozen candidate also passes all 252 consumed regressions:
false notes fall 2,329 → 2,311, all 4,712 previously matched notes remain, F1
rises .72392 → .72492 and recall stays .78836. No test-driven retuning occurs.

These are modest improvements on limited development data. They do not prove
that arbitrary songs transcribe accurately, and a KEEP filter does not recover
missed pitches or repair detected rhythms.

## Reserved piano regression

After freezing the candidate, the complete reserved-performer check evaluated
12 original MP3 crops. All 12 declared Demucs variants emitted no bass and are
audited exclusions. The candidate preserves all 645 matched attacks, holds and
pitch coverage, with unchanged F1 .76649 and recall .79531. It removes zero false
notes on this set. Thus all 264 consumed regression recordings pass preservation,
but this piano set adds no evidence of clutter reduction.

## First-pass fixtures and runtime verification

The single frozen winner passed the 32 predeclared first-pass MP3 fixtures
(seeds 263301–263332, unchanged original/texture generators). False notes fell
183 → 181, with all 675 matched notes, offset-aware holds and supplied pitch
coverage preserved per recording. F1 rose .85606 → .85714; recall remained
.93880. These unseen procedural seeds provide a small positive result within
two existing generator distributions, not independent human-song accuracy.
All 32 are now consumed regressions; future experiments must preserve all 296
consumed recordings rather than reuse this set for a new first-pass claim.

Portable weights match the frozen checkpoint exactly across all 1,029 fitting,
validation and regression recordings (55,163 V15 survivor events). Runtime
probabilities also match with two versus four CPU threads, with identical
rejection decisions and no change to global thread settings. Single-part and
multiple-part checks preserve retained note objects, pitch, start, end, velocity
and staff assignment. The total 71 runtime rejections includes training
recordings and must not be presented as held-out accuracy.

A source/weight/evidence seal was created while the old V15 route guards still
passed. The new route adds one V16 filter after V15; all original assets and
frozen source contracts remain unchanged except the explicitly permitted
routing source. Eight complete native cases, including held notes, repeated
notes, a licensed bass recording, actual learned rejections, a 32-second window
and a short clip, match the frozen decisions. The backend/training suite passes
591 tests, and all six opt-in native pipeline/renderer checks also pass.
Readiness now includes the V16 model asset.

The guarded original route will intentionally reject new fitting against a
changed website. Future training must declare a new baseline of actual V16
survivors and retain the full consumed regression history. Never change the
frozen V16 checkpoint or threshold from these test outcomes.

Frozen plans, manifests, curves and release evidence are under
`training/results/bass-temporal-v16-v1`, `bass-temporal-v16-fresh-v1`,
`v16-piano-regression-v1`, `bass-v15-baseline-v1` and `vienna-bass-v15-v1`.
Audio and fitting checkpoints remain local. Only the verified portable weights
are included with the backend. The first-pass and consumed results are modest;
this KEEP filter does not recover missed pitches or repair rhythms.

## Website release

The backend was rebuilt and deployed after all gates passed and an immediate
read-only queue check confirmed no queued or processing jobs. The running
website is healthy through the frontend proxy. Its V16 weight checksum, runtime
and routing source match the verified release; the V15 array and original
notation source remain unchanged. No user job was submitted or regenerated.
See `release-status.json`, `website-before-v16.json` and `website-v16.json` for
the frozen evidence and live checks.
