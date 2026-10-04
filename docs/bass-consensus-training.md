# Bass consensus training

The website's independent bass stem uses a frequency-bounded Basic Pitch decoder.
V9's accompaniment verifier does not cover it. This experiment trains separate
KEEP classifiers with that exact decoder. **V9 remains deployed; these bass
weights are offline until all release gates and production parity pass.**
No user upload was processed, job created or saved score regenerated.

## Licensed, source-disjoint preparation

[Official BabySlakh v2](https://zenodo.org/records/4603870) is CC BY 4.0. Credit
Ethan Manilow, Gordon Wichern, Prem Seetharaman and Jonathan Le Roux, WASPAA 2019.
The verified archive and pinned publisher duplicate metadata are documented in
[the polyphonic framework](polyphonic-training-framework.md). The same fixed
12/4/4 source-group partition is reused, keeping duplicate identities together.
This is a small synthesized prototype; composition, near-duplicate and upstream
pretraining overlap are not ruled out. It is not proof of real-song accuracy.

There are **56 training / 20 validation clips**, 5,209 note-release annotations
and 5,319 pitch-support intervals. Oracle examples use actual rendered bass MIDI
programs 32–39 at offsets 0/30/60/90 seconds. Full-mixture Demucs bass examples
use offset zero. One unrendered bass source has no oracle examples. Actual source
presence, sample clocks, MIDI bounds and frozen source hashes are audited.

Demucs labels include low notes from every rendered pitched source, protecting
real piano/guitar notes that can cross into its bass output. Both variants use
21–59 MIDI pitches, with separate key releases and CC64/crop-crossing hold
support. Fixed MP3 bitrates cycle 48/128/192 kbps; every codec check reports zero
sample lag. All four reserved source groups remain uninferred.

The candidate decoder uses production bass bounds 21–60 exclusive, onset .5,
frame .3, 90 ms minimum length, no melodia trick or pitch bends. Its returned
acoustic bands are constrained, matching production. Accompaniment decoder
caches and classifiers are not reused. Fifty-two pitch-relative features and a
26-feature acoustic view contain no source name, key signature or absolute pitch.
Incomplete start/end neighborhoods stay protected.

## Training and checkpoint protection

Four paired boosted-tree profiles fit **3,361 eligible labeled events**: 1,317
KEEP and 2,044 negatives. Ambiguous timing detections are excluded, and supported
held-note fragments gain positive KEEP supervision. Corpus/clip weighting prevents
long dense pieces from dominating. Profiles, seeds, threshold grids and checkpoint
stages are fixed before fitting. These are trained event verifiers; the upstream
Basic Pitch neural network is frozen.

Every proposed deletion requires both heads to reject it. Raw and half-margin
thresholds must preserve each recording's matched attack set, onset-offset set
and every supplied pitch-support interval, with no extra-note increase. Validation
chooses the maximum safe false-note reduction; ties use loss, fewer trees, then
fixed profile order. Tests never select a runner-up or retune thresholds.

The validation winner is `capacity`, **500 trees**, thresholds **.05/.3**. It
retains all 505 baseline matches and reduces unmatched detections **1,308→618**;
F1 **.2897→.3612**, recall unchanged at .3019. This retains already detected notes;
it does not recover missing notes or correct their timing. Every individual
recording passes attack, offset and pitch-coverage gates.

Its 600-tree stage has lower loss but removes only 673 false detections versus
690 at 500 trees, so the earlier checkpoint is retained. Independent portable
tree traversal matches selected checkpoint predictions within 1e-12 in tests.
Actual trained-array export still requires successful release evaluation.

## Release requirements

First, eight original hold/repeat/triplet/octave/low/legato/sparse/syncopation
stress cases must preserve every measured correct note and hold. Only then may
all four reserved Slakh source groups be prepared and scored once. That stage
must add positive false-note reduction while passing every preservation gate.
Checkpoint, plan, preparation, code and manifest hashes are frozen. Changed or
failed evidence prevents test preparation and portable export. Production
integration and full event parity are additional prerequisites for deployment.

**339 backend/training tests passed, six opt-in audio cases skipped**, plus
focused checkpoint/portable/release-guard tests. The site health endpoint remains
ready and the job queue contains only its three existing completed jobs.
Preparation and all profile curves/searches are archived under
`training/results/slakh-bass-v1` and `training/results/bass-consensus-v1`.

## Withheld outcome

The frozen winner fails seven of eight original stress cases: matched attacks
**210→98**, false notes **10→10**, F1 **.9722→.6125**. The held-only case passes,
but genuine repeats, simultaneous octaves, low, legato, sparse and syncopated
notes lose matches and pitch coverage. This exposes severe timbre/domain shift
despite positive validation results. The candidate is **withheld**. No runner-up
is selected and thresholds are not adjusted using these test labels.

Reserved Slakh groups were not consumed, arrays were not exported and production
was not changed. Future experiments must label these original stress cases as
consumed regression. Broader source/timbre-positive training is required before
a bass classifier can qualify. This is evidence that release gates are needed,
not an accuracy improvement claim for the website.
