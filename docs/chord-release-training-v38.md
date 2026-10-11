# Independent release specialist V38

V38 is a conservative, newly trained correction model for balanced full-song
bass/accompaniment (MIDI pitches 21–59). It follows every existing filtering
stage. It can move an independently detected note ending by at most 80 ms;
note starts, pitches, velocities, note counts and source parts remain fixed.
Notes near the first/last 2.5 seconds remain protected. Same-key rests and
touching repeats cannot become overlaps. Simultaneous chord tones keep separate
endings rather than being forced to sustain together.

This is deliberately a smaller objective than the rejected V36/V37 joint
models. The release-only objective and all preservation requirements were
declared **before** V38 training. V36/V37 weights were not reused. There is no
claim that V38 improves wrong-note counts, quantization, missing melody notes,
or seconds-long onset errors. It does not replace the solo-piano model.

## A new dataset

The project had not previously used the Mutopia Joplin corpus. Six specific
editions were acquired directly from their publisher, each explicitly marked
Public Domain. The edition pages, source URLs and original MIDI SHA256 hashes
are recorded in the source manifest.

| Publisher work | Split | Typesetter |
| --- | --- | --- |
| [Magnetic Rag, 441](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=441) | Training | Kurt Holtrop |
| [Original Rags, 345](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=345) | Training | Tom Harke |
| [Solace, 482](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=482) | Training | Magnus Lewis-Smith |
| [Elite Syncopations, 1540](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=1540) | Training | Benjamin Bloomfield |
| [The Entertainer, 263](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=263) | Validation | Chris Sawer |
| [Pineapple Rag, 1899](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=1899) | Reserved first pass | Coyau |

All audio is newly synthesized by project-authored harmonic rendering code;
these are **not recorded human performances**. Two twenty-second score crops
per work are placed inside a 26-second padded window, with two independently
seeded timbres encoded at 48/128 kbps. The final reserved-work variants use the
predeclared 48/128/192 kbps first-pass cycle. All variants of a work stay in one
partition. The reserved work is rendered only after candidate freezing and
regression success. No user recording or saved score is processed.

The generated MIDI/audio clocks supply exact key boundaries. A declared 60 ms
acoustic decay tail and 50/100 ms attack/release support padding protect
ambiguous sounding intervals. Held-at-crop notes protect pitch support without
being mislabeled as new attacks. PrettyMIDI warns about some source tempo
events outside track zero: labels and audio both use the same parsed score
clock; this subset does not claim original performance timing.

The new corpus contributes 16 training clips and four validation clips. The
existing licensed corpora contribute 765/221, making **781 training and 225
validation clips**. Their hashes and original source partitions are checked by
the existing guarded readers. MusicNet, Vienna, GuitarSet, BabySlakh, NSynth,
Groove mixtures and original procedural examples remain in their existing
partitions. No consumed regression examples enter training or normalization.

## Training and selection

Two fresh seeds (283001/283002) fit 16/24-channel attack/release CNN encoders,
100 observed context features and a single trainable release head for 40 epochs.
The seven release actions are −80/−40/−20/0/+20/+40/+80 ms. Support and attack
outputs are immutable. References supply training labels only; neither labels,
publisher IDs nor source groups enter inference features.

Training-only targets use the V37 protection rules. Unmatched or unresolved
eligible notes explicitly teach STAY. Corpus/recording-balanced losses, training
normalization, class balancing and extra STAY weights reduce risky corrections.
AdamW uses 25 updates/epoch, batch 256, initial learning rate 0.0005, weight decay
0.003 and cosine decay. Validation checks epochs 10/20/40 and the confidence
settings declared in the plan.

The frozen winner is **release-small, epoch 40**, threshold 0.8; its stricter
check is 0.83. Checkpoint SHA256:
`19bda4c45c587868e262c33de468a2d6ce0691bd96f90e9674c29660b2c8978a`.
No runner-up was tried after regression results became available.

| Evaluation | Clips | Normal: corrected notes / summed release-error reduction | Stricter |
| --- | ---: | ---: | ---: |
| Validation | 225 | 9 / 0.36 s | 3 / 0.12 s |
| Consumed regressions | 519 | 139 / 5.56 s | 43 / 1.72 s |
| Fresh first pass | 44 | 5 / 0.20 s | 1 / 0.04 s |

Both settings pass **every recording's** existing pitch coverage, attack/hold
matching, fixed-pair onset/release error, repeated-key spacing, chord onset
spacing, unsupported sustain and false-note preservation checks. Deleted notes
cannot earn timing credit; this model cannot delete notes at all. The numbers
above are sums over corrected boundaries, not percentage accuracy or a claim
that every song improves. Wrong-note counts and onset errors are unchanged.

The 44 fresh clips comprise seeds 283101–283140 (alternating the existing
original/texture renderers at cycling MP3 bitrates) and four crops/timbres from
the reserved Mutopia work. All results are retained, with no clip replacement,
threshold retuning or test-label fitting.

## Website verification

The portable model and runtime code reproduce the offline probabilities and
decisions exactly across **1,569 clips / 69,521 candidate notes**, at both
settings. Recomputed attack/release windows and neighbor features on all 44
fresh decoded MP3s have maximum error zero. Native staged Basic Pitch routing
reproduces all five actual fresh corrections, preserving the rest of each
event array exactly.

The model asset is `backend/app/assets/chord-release-v1.npz`, SHA256
`eeef2fae2d0c4ce180af47dbb9670c5c6b82095df5f01722fb06ea2a6534b6d0`.
The runtime validates its checksum, metadata, finite arrays, shapes, probability
normalization and positive scaling. Missing/corrupt arrays fail clearly.
Release evidence, complete functional/native test results and deployment
identity are stored under `training/results/release-only-v38-v1`.

The staged and installed backend/training suites each pass 947 tests (six
opt-in integrations skipped in those runs). All six native integration tests
then pass separately, including long/short/silent MP3 upload, full-mixture
separation, downloadable outputs, playback timing and multiple-page rendering.
The running local website and its frontend proxy report ready, and the loaded
model checksum matches the exported asset. Original 7 mm engraving is unchanged.

The V32 guard intentionally rejects a changed website route: old survivor
caches must not masquerade as observations from the new model. Historical
reproduction requires the recorded source snapshot; a future training study
must declare a new baseline or explicitly replay that snapshot. No historical
guard or prior release evidence is weakened to accept the new route.
