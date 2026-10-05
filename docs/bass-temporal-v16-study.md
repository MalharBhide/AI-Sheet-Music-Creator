# V16 study: expanded real-piano training, release pending

This study trains a new temporal bass KEEP classifier on actual V15 survivors
and additional real acoustic-piano examples. It is **not deployed**. The website
continues using the verified V15 model and original 7 mm notation, spacing and
pagination. No user recordings or saved scores are used or regenerated.

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

## Outstanding release gates

Reserved real-piano regression is acquired only after the frozen candidate
passes the earlier consumed set. Its rule includes all eligible reserved
performers with complete 30–60-second crops, both original MP3 and actual
production Demucs bass variants; absent bass sources are audited. These
performers remain consumed development regressions, never a fresh-human claim.
No failed result may be replaced or tuned around.

Further required checks are the one predeclared first-pass fixture set (seeds
263301–263332, unchanged original/texture generators), positive preserved
first-pass gain, portable export and full runtime parity, native complete-route
checks and application tests, followed by verified idle-queue deployment.
No portable weights or website routing are changed while those gates remain
unproven. Only this single frozen winner may proceed.

Frozen plans, manifests, learning curves and current evidence are under
`training/results/bass-temporal-v16-v1`, `bass-v15-baseline-v1` and
`vienna-bass-v15-v1`. Audio and checkpoints remain local. The new framework
passes the existing backend/training suite and additional source/pedal/identity
tests. See the checkpoint status report for exact test counts and pending work.
