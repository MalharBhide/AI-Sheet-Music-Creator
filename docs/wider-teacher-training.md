# Wider learned corrections with a frozen acoustic teacher

The [V30 experiment](teacher-anchored-training.md) preserved every validation
recording, but its .25/.5 logit corrections did not reduce false notes at both
required confidence settings. It selected no checkpoint and read no regression
or reserved first-pass recordings. Its source, searches and results remain
unchanged.

V31 independently declares correction caps 2.0/4.0 before fitting. The approved
V25 acoustic parameters and 84-feature normalization remain frozen, with teacher
dropout disabled. The correction MLP starts at zero influence and learns from
71 observed recurrence/release-neighbor features. It receives no reference
labels, endpoints, filenames, absolute pitch, or fitted teacher probabilities
as inputs. A larger allowed correction can remove more clutter, but cannot
itself prove higher accuracy; all preservation and release gates stay unchanged.

The two profiles use widths 64/96, caps 2.0/4.0, KEEP weights 16/32, partial
fragment boosts 8/16, and seeds 276001/276002. Both train for 80 epochs with
20/40/60/80 checkpoints on the same sealed 702 training and 206 validation
recordings. The confidence grids and source splits are identical to V30.
Validation alone chooses one frozen checkpoint and confidence pair; every
recording must preserve matched attacks, holds, reference-pitch coverage and
fixed-pair timing. Retained events cannot change clocks, pitches or velocities.

The winner must pass all 392 earlier regression recordings at both confidence
settings before the declared 77-record first pass. Four reserved NSynth
instruments and two reserved GMD performers remain untouched until that gate.
Authored first-pass seeds are 276101–276132. Test-based retuning, runner-ups,
confidence switches, replacement seeds or fixtures are prohibited. Positive
noise reduction and complete preservation at both settings, portable/runtime
parity, and native physical pipeline checks are mandatory before release.

Six focused tests verify exact initial teacher parity, teacher parameters frozen
through optimizer steps, disabled dropout, saturated positive/negative correction
bounds, and first-pass refusal. Both 80-epoch fits completed. All 206 validation recordings were preserved at
every searched checkpoint/setting, but none reduced false notes at the stricter
confidence setting. No winner was selected, no regression was read, and the
actual first-pass tool refused before creating an output directory. All reserved
sources and 32 declared seeds remain unconsumed. Results and eight checkpoint
hashes are archived in `training/results/teacher-wide-v31-v1`. The full suite
passed 797 tests, with six native tests skipped for unchanged production code. No experimental model is
deployed, and no user uploads or saved scores are processed.
