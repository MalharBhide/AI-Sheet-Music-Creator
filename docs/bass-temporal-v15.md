# V15: temporal spectral evidence for unsupported bass notes

This release adds a trained temporal CNN KEEP classifier after all five V14
bass stages. It observes audio before, during and after each surviving note,
alongside the existing acoustic candidate relationships. A frozen acoustic
guardian must independently reject the note before deletion. It targets extra
low notes in balanced Full-song arrangements. The underlying Basic Pitch,
Demucs, solo-piano, vocal and accompaniment weights are unchanged. No user
uploads or stored scores are used for fitting or evaluation.

## Baseline, data and supervision

The sealed V14 baseline contains 48,447 surviving notes across 897 recordings:
531 training, 146 validation and 220 consumed regressions. Its 174 deletions
exactly match the V14 runtime witness. Per-note acoustic features can be reused
because surviving pitch, onset, endpoint and velocity remain unchanged;
neighbor relationships are recomputed on the actual survivors.

Fitting uses CC BY 4.0 GuitarSet and BabySlakh crops plus project-authored
signals, including previously partitioned MP3 and Demucs-separated variants.
No new recordings are added to the fitting partition. Spectral sequences are
prepared only for the 677 training/validation recordings before fitting;
consumed-test audio is read only after selecting and freezing one candidate.
Source groups, annotations, audio clocks, hashes and partitions are checked.
No user filenames or audio enter this process.

Labels retain the existing matched-attack, offset-aware hold and partial-pitch
coverage protections. Historical label parity passes all 897 recordings and
48,447 notes. The learner fits 27,799 eligible supervised training events.

## Architecture and selection

The input is a 40-step, nine-channel CQT sequence relative to the candidate
pitch: the candidate band, neighboring semitones, and harmonic/subharmonic
intervals 12/19/24. Eight steps precede the attack, 24 span the detected note
and eight follow its endpoint. The CQT uses 22,050 Hz audio, a 256-sample hop,
88 piano bands and an 80 dB relative range; silence is explicitly zero.
The same frozen transform produces fitting and runtime evidence.

Two width-32/48 convolutional layers feed eight pooled temporal positions,
concatenated with 84 observed acoustic/neighbor features. A small dense
classifier emits KEEP probability. Context normalization is fitted only on
eligible supervised training rows. Inputs contain no absolute candidate pitch,
reference labels, recording identifiers or fitted-model confidences.

Two predeclared profiles use positive weights 12/24, 60 epochs, 40 updates per
epoch, AdamW and fixed checkpoint/threshold grids. Validation alone selects
the width-48 conservative epoch-60 checkpoint. Strict rejection thresholds
are .005 for the CNN and .3 for the unchanged V14 acoustic guardian, after the
declared half-margin check. The runner-up is not tested or released.

| Cohort | Matched attacks preserved | False notes before → after |
| --- | ---: | ---: |
| 146 validation recordings | 5,068 | 4,055 → 4,043 |
| 220 consumed regression recordings | 4,139 | 2,163 → 2,144 |
| 32 first-pass original MP3 fixtures | 573 | 194 → 185 |

Every recording preserves its matched attacks, offset-aware holds and supplied
pitch intervals; no recording gains false positives. First-pass F1 increases
.80989 → .81508, with recall unchanged at .88426. Seeds 263101–263132 were
declared before fitting: 16 ordinary signals and 16 texture variants, encoded
at 48/128/192 kbps with verified clocks. They now become consumed regression,
bringing the next release's regression requirement to 252 recordings. No
test-driven threshold change, runner-up substitution or replacement fresh
check occurs.

These fresh fixtures are new seeds from existing procedural families, not
independent human songs. Reused validation, synthetic timbre limits and
upstream pretraining overlap restrict the evidence. This is a modest measured
improvement, not an assurance of accuracy for arbitrary songs. The classifier
can remove unsupported candidates; it cannot recover missed notes or fix
incorrect detected rhythm. Conservative vetoes intentionally leave many
unresolved false notes.

## Runtime and release checks

The CNN runs after V10 verification, V11 hold repair, V12 residual rejection,
V13 texture rejection and V14 harmonic rejection. Evidence is recomputed on
their actual surviving intervals. Only bass pitches 21–59 at least 2.5 seconds
inside each window are eligible. Either-head support, threshold equality,
window edges and other registers preserve notes. It cannot restore prior
deletions or change retained note properties. Short/empty candidate lists skip
unnecessary spectral computation.

Portable NPZ loading excludes pickled objects and verifies checksums, exact
architecture, array keys/shapes, finite weights, positive normalization scales
and thresholds. It reuses the frozen V14 guardian. The filter leaves global
Torch thread settings unchanged. Metadata reports temporal rejections
separately, and health readiness requires the new asset.

Runtime checks cover all 929 recordings and 49,214 surviving candidates:
frozen/exported probabilities are identical, including two versus four Torch
threads, and all retained object identities, parts and note properties match.
The 74 runtime deletions include fitting data and are not independent accuracy
evidence. Eight native cases match the full six-stage route on real audio:
holds, repeated keys, played octaves, licensed Slakh, two learned-rejection
cases, a 32-second window and a subframe clip.

All 526 backend/training tests and six native upload/render integrations pass.
Runtime/routing/export code and tests pass lint, and the backend build passes.
The deployed backend is ready through the frontend proxy; live model, routing
and original-notation hashes match the checked release. The queue remains at
eight completed user jobs, with no user transcription or score regeneration.

Frozen plans, learning curves, provenance and release reports are under
`training/results/bass-temporal-cnn-v1` and the associated baseline/sequence/
consumed/fresh directories. Checkpoints and audio stay local; only gated
portable weights ship. See `release-checks.json` and `live-check.json` for
application checks and deployment verification. Historical V14 helpers
intentionally reject newer routing; its pre-routing source is in commit
88b976e. Future training needs a new six-stage survivor baseline and all 252
consumed regressions.

The original 7 mm note scaling, default spacing and automatic pagination are
unchanged. Stored user scores remain untouched; uploading again applies the
released model.
