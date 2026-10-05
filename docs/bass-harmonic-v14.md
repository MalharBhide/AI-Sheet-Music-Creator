# V14: harmonic attack and release evidence for extra bass notes

This release trains a paired classifier on the actual bass notes surviving the
V13 website route. It adds observed harmonic attack/release relationships to
separate detector overtones from independently played keys. Both heads must
reject before deleting a note. It targets extra low notes in balanced Full-song
piano arrangements. The Basic Pitch, Demucs, solo-piano, vocal and existing
accompaniment weights remain unchanged. No user uploads are transcribed, fitted
or regenerated, and the original note size remains unchanged.

## Current baseline and unchanged supervision

The new baseline seals the complete V13 release, all eight bass arrays and all
four stages. Every one of its 315 decisions matches the prior runtime witness.
It contains 47,798 V13 survivors across 865 recordings: 531 training, 146
validation and 188 consumed regression. Deletions cannot alter another note's
pitch, attack, end or velocity, so independent per-note acoustic channels can
be reused; candidate relationships are always recomputed after deletion.
Historical V12 guards stay unchanged and reject the newer route.

The labeled data are CC BY 4.0 BabySlakh/GuitarSet and original project-authored
signals. This bounded batch acquires no additional fitting recordings. The
loader protects matched attacks, offset-aware held notes and every supplied
pitch overlap beyond the existing 1e-6-second coverage tolerance. Exact label
parity against the historical policy passes all 865 recordings and 47,798
notes. Reference annotations, clocks and partitions remain unchanged.

## Learned evidence and selection

The context head has 84 observed features: the existing 72 acoustic/salience
channels plus 12 harmonic relationships. For lower candidates at intervals
12/19/24 semitones, it measures lower attack strength, attack-strength difference,
attack alignment and release alignment, within the same bounded overlapping
neighborhood. These features contain no absolute pitch, recording names,
annotations or fitted-model confidences. True played octaves remain positive
supervision. The guardian independently uses all 52 per-note acoustic channels.

Four heads fit on 27,915 supervised training events. Two predeclared profiles use
positive weights 24/40, guardian weight 40, learning rate .04 and a 600-tree
limit. Checkpoints and threshold grids are fixed before fitting. Validation
alone selects the harmonic 600-tree pair at strict .025/.3 thresholds after the
declared safety margin. The runner-up is never tested as an alternative.

| Cohort | Matched attacks unchanged | False notes before → after |
| --- | ---: | ---: |
| 146 validation recordings | 5,068 | 4,086 → 4,055 |
| 188 consumed regression recordings | 3,595 | 1,911 → 1,897 |
| 32 first-pass original MP3 fixtures | 544 | 279 → 266 |

Every recording preserves all its matched attacks, offset-aware holds and
supplied pitch-time coverage. No recording gains false positives. The first-pass
seeds 262801–262832 are declared before fitting, with 16 ordinary original
signals and 16 texture variants, encoded at 48/128/192 kbps using the verified
MP3 clock checks. Its F1 rises .76566 → .77273, recall unchanged at .90970.
All 32 fixtures now become consumed regression, bringing future required
regressions to 220. No test-driven retuning or replacement fresh check occurs.

The fresh recordings are new seeds from existing synthetic generator families,
not independent human songs. Validation is reused for development; compositions,
synthetic timbre limits and upstream pretraining overlap remain. This is a
modest, limited-corpus improvement. The filter cannot recover missed pitches,
correct wrong timing or establish accurate transcription of arbitrary songs.

## Runtime and release evidence

The stage runs after V10 verification, V11 hold repair, V12 rejection and V13
texture rejection. Only bass pitches 21–59 at least 2.5 seconds inside each
inference window are eligible. Either-head support or threshold equality vetoes
deletion. Surviving objects keep their part, pitch, onset, endpoint and velocity;
prior rejected notes cannot return. Metadata distinguishes harmonic rejections
from all previous stages. Model loading enforces array checksums, features and
thresholds; health readiness requires both new arrays.

The original 7 mm staff scaling, default spacing and automatic pagination from
abb46fe remain unchanged. Stored user scores stay untouched; a new upload applies
the new model. Training/results/bass-harmonic-v13-v1 contains the frozen plans,
curves, source hashes and release evidence. Checkpoints stay local; only gated
portable arrays ship. Future fitting needs a new baseline with all five stages,
actual survivors and all 220 consumed regressions. Historical V13 helpers
intentionally reject newer routing; preserve their pre-routing snapshot.

Portable/runtime features and probabilities match sklearn exactly across all
897 recordings and 48,621 notes (zero error at 1e-12 tolerance). The 174 rejected
events include fitting data and are not independent accuracy evidence. All eight
native bass cases match the five-stage route, including real features on holds,
repeats, octaves, licensed Slakh, observed new rejections, 32-second windows and
subframe audio. 489 backend/training tests and all six native upload/render
integrations passed. Changed Python files pass lint; the backend build passed.
The idle-queue deployment is ready through the frontend proxy, and live model,
route and original-notation hashes match the checked release. Eight completed
user jobs remain unchanged. See release-checks.json and live-check.json.
