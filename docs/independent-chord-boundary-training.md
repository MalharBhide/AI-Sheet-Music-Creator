# Independent chord attacks and releases

This study targets wrong notes and timing in the approved V32 balanced full-song
bass/accompaniment branch. It does not use a particular uploaded song, regenerate
saved scores, change engraving, or replace the solo-piano or melody models.

## Supervision and inputs

The guarded V32 caches and MusicNet observations supply 765 training recordings
and 221 validation recordings. Source groups stay in their existing partitions;
no consumed regressions or reserved examples enter fitting or normalization.
There are 52,920 existing candidate notes and 34,415 eligible supervised training
events. The corpora include the project's original MP3 mixtures, GuitarSet,
BabySlakh, NSynth, Vienna piano, Groove mixtures and real-piano MusicNet excerpts.
Existing source licenses and provenance remain recorded with those earlier
training studies; this experiment acquires no new recordings.

Separate CNNs observe attack and release windows, each 61 frames by 18 spectral
channels. They share 100 observed context features: the existing 84 features,
eight same-key release relationships, and eight chord/overlap relationships.
Three heads predict KEEP/REMOVE, attack correction, and release correction.
The seven boundary choices are -80/-40/-20/0/+20/+40/+80 milliseconds.
Reference matches supply labels only; references, source IDs and model scores
are absent from inference inputs. Out-of-window and ambiguous boundaries abstain.
Normalization and class balancing use training data only.

Chord notes retain independent releases. The decoder never forces a chord to
end together, creates a pitch, changes velocity, or moves an ineligible register
or edge note. Repeated-key collisions and incompatible touching releases abstain;
a second chain check prevents pair rollback from introducing another collision.
This bounded correction cannot repair a missed note or a seconds-long timing
error. Those need a different acoustic segmentation or rhythmic model.

## V36 completed study

Two profiles completed 40 epochs each with seeds 281001/281002. No checkpoint
passed the full validation gate. Some candidates reduced aggregate boundary
errors and wrong-note counts, but harmed supplied pitch support or individual
recordings. Such aggregate gains do not authorize deployment.

The most active final acoustic setting removed 14 false notes and reduced summed
fixed-pair onset/release errors by approximately 1.00/20.17 seconds across the
validation set, but failed 75 recording-level checks. These are diagnostic sums,
not a claimed improvement in overall accuracy. All checkpoints were rejected.
The executed fitter/model/evaluator are archived with reports and checkpoint
hashes under `training/results/chord-boundary-v36-v1`.

## V37 protected supervision

The second study uses the same network, inference rules and evaluation gates.
Its training-only targets instead label a correction STAY if it would lose
supplied pitch support, break a protected attack/hold, worsen a matched boundary
or repeat/chord relation, or add unsupported sustain. This distinguishes a
removable unsupported tail from a genuine continued pitch. It also exposes an
important limitation: MIDI key release and supplied acoustic decay/support
intervals may describe different endpoints; the training targets must respect
the currently declared preservation contract rather than silently weaken it.
Two fresh seeds, 282001/282002, completed 40 epochs each independently of V36 weights.

The protected-holds profile at epoch 20 passed all 221 validation preservation
checks at both settings. It reduced summed fixed-pair release error by 6.16
seconds across 87 retimed notes (stricter setting: 4.40 seconds / 60 notes).
Onsets and false-note counts were unchanged, so it does not meet the predeclared
requirement to improve all three objectives. Later checkpoints again lost some
pitch support; the last checkpoint is therefore not automatically the best one.
No winner was selected and no regression or fresh-test audio was processed.
Reports, executed sources and checkpoint hashes are preserved under
`training/results/protected-boundaries-v37-v1`.

## Release requirements

Both normal and stricter confidence settings must preserve every validation
recording's attacks, holds, supplied pitch coverage, fixed-pair onset/release
errors, repeated-key spacing, chord onset spacing and unsupported hold duration.
Deleted predictions keep their old timing error, so removing a difficult note
cannot manufacture a timing improvement. The declared selection also requires
positive onset, release and false-note gains.

Only a frozen passing validation winner can proceed to all 519 consumed
regressions, then a predeclared 40-clip project-authored MP3 first pass. Portable
inference parity, physical feature parity and native pipeline checks are still
required before any website model changes. Failed candidates stay offline.
