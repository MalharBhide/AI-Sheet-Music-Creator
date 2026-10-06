# Partial support and recorded percussion studies

The website retains its approved V25 model while experimental candidates train
separately. No user uploads, saved scores, note size, or pagination are changed.

V28 separates three training targets: partial pitch support, other protected
support, and unsupported events. A protected interval with more than one
microsecond of reference support covering less than 25% of its duration receives
the partial subtype. The other protected subtype also includes previously
matched attacks and holds. All references are training loss targets; no reference
endpoint, coverage fraction, source identity, or subtype is an inference input.
Inference sums the two protected probability masses and uses their complement
for a bounded binary removal decision. It cannot rewrite retained note clocks.

The acoustic encoders start from the shipped V25 checkpoint. Its original KEEP
logit is duplicated with a log(2) bias adjustment so that the initial protected
mass is unchanged. Nonzero random-weight parity tests verify that property at
both recurrence widths. Context normalization is inherited; recurrence
normalization uses supervised training events only.

Both predeclared profiles train for 80 epochs on 639 training recordings, with
178 validation recordings selecting one checkpoint and two confidence settings.
There are 650 partial, 24,633 other protected, and 5,458 unsupported supervised
training events. The acoustic features and licensed instrument sources are the
same sealed inputs described in [the NSynth study](periodicity-nsynth-training.md).

The first run stopped at its first validation scoring step: the strict binary
probability guard rejected the new protected probability sum. Summing float32
softmax columns can round slightly above one. Computing the protected mass as
the complement of unsupported mass fixes that numerical issue; an explicit
rounding regression test passes. The original failed run's source, plan and log
are preserved in [the numerical failure archive](../training/results/fragment-v28-aborted-numerics-v1).
The restarted profiles keep the same training seeds and policy; no completed
validation selection or test result existed in the aborted run.

The completed restarted fit selected `subtype-conservative`, epoch 60, recurrence
width 32, removal confidence .7 and guardian .5. It removes 18 additional wrong
notes at normal confidence and nine at stricter confidence while preserving all
validation attacks, holds, pitch coverage and fixed-pair timing. The frozen
392-record regression removes 25/14 extra false notes, but both confidence
settings lose 7.3977 ms of supported pitch coverage in the same earlier
continuation regression. All matched attacks, holds and fixed-pair timing remain
unchanged. **V28 is rejected.** No runner-up or confidence switch is tried. The
actual first-pass tool refuses it before creating an output directory; all four
reserved NSynth instruments and seeds 273101–273132 remain unconsumed. No weights
are exported or deployed.
All 776 backend/training tests pass; six native integration checks are skipped
because production code is unchanged. Five focused subtype/numeric tests pass.

## A new recorded-drum nuisance source

The [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) provides
human-played electronic drum recordings and aligned MIDI, released by Google
under CC BY 4.0. Drum MIDI numbers identify drum instruments and must **never**
be treated as piano pitch labels. This source is intended for drum-only negative
examples and mixtures with separately labeled pitched recordings, to challenge
kick/tom false positives while protecting real accompaniment.

The larger [Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd)
uses 43 drum kits and the same license, but its audio archive is 90 GB. Its kit
identities span the original train/validation/test splits; source sequences and
performers would also need to remain separated in a project-specific study.
It is researched but not acquired or fitted in this round.

The bounded GMD acquisition streams the entire original 5,111,599,714-byte
archive through SHA256 before extracting any selected files. Its checksum must
equal the publisher's
`21559feb2f1c96ca53988fd4d7060b1f2afe1d854fb2a8dcea5ff95cf3cce7e9`.
Generation-specific HTTPS range reads then retrieve only selected files from
that same verified object; status, content ranges, lengths, ETag, generation,
ZIP CRC, path traversal, symlinks and member sizes are checked. The full archive
is never stored. The verified 3.11 MB original MIDI-only archive was inspected
first to determine bounded audio availability.

Metadata selects at most two clips per eligible performer, preferring different
primary styles, with durations 8–40 seconds. All original test sequences are
excluded. Nine eligible performers are sorted: five train, two validation, and
two reserved. The resulting 16 clips divide 9/4/3. Some performers have only one
qualifying clip. This is a declared project-specific performer split, **not** an
official GMD benchmark. All recordings use the same TD-11 sound source, so this
does not establish unseen-drum-kit generalization.

[Acquisition records and attribution](../training/results/groove-subset-v1) pin
the full original object, metadata, selected audio/MIDI and source code. Seven
acquisition safety tests pass. GMD is not part of the
already frozen V28 training or evaluation policy. Its reserved performers remain
unanalyzed. Only the 13 train/validation source clips enter the next preparation.


## V29: long-hold context and recorded percussion

[The new fitting fixtures](../training/results/groove-bass-v29-v1) run all seven
declared variants for each of those 13 source clips: drums only and the six
NSynth pitched sequence types. There are 91 physical MP3 observations, split
63 training/28 validation, with 48/128/192 kbps and 19-second decoded clocks.
References describe only the separately played pitched notes, their three-second
key holds and complete one-second decay support. Drum MIDI never supplies piano
pitch labels. Reserved drum performers and NSynth instruments do not enter these
observations.

Eight new features describe same-pitch preceding releases, overlap, and following
attacks. A previous note can start more than two seconds earlier yet release near
the current attack. The older start-centered context excludes that long-held
neighbor. New features use predicted clocks and observed acoustic strengths;
reference coverage, fitted probabilities, filenames and absolute pitch are not
inputs. Tests check long preceding holds, equal-release duplicates, order,
transposition, translation, empty notes, and malformed clocks.

V29 preserves the three support subtypes but learns its recurrence branch from
71 features: the existing 63 waveform measurements and eight release-neighbor
relationships. The approved V25 encoders and normalization are inherited with
zero initial influence from the new branch, verified at both widths. The expanded
fitting set has 702 training and 206 validation recordings. Two profiles and all
confidence grids are declared before fitting.

A preflight check caught a checksum-key spelling error in the first-pass producer
after four initial training epochs. That run was stopped before any validation
selection or test evaluation. The failed source, plan and log are preserved in
[the preflight archive](../training/results/release-neighbor-v29-preflight-abort-v1).
The corrected run restarts the same seeds, data and policies. A new explicit test
checks exact SHA256 keys and the complete 392-record regression identity list.

One frozen validation winner must pass all 392 consumed recordings at both
confidence settings before any new first-pass observations can run. The declared
77-record first pass comprises 24 sequences from four reserved NSynth instruments,
21 drum mixtures from three clips belonging to two reserved GMD performers using
only reserved pitched instruments, and 32 new authored MP3s with seeds
274101–274132. Both settings must reduce false notes while preserving attacks,
holds, pitch coverage and fixed-pair timing. No runner-up, confidence switch,
replacement seeds or new fixtures may follow a failed test. Runtime probability/
filter parity and native physical pipeline checks remain mandatory before release.

Training is running. No V29 weights are exported or deployed.
