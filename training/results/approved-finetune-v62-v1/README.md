# Fine-tune the existing approved verifier

Two independently initialized profiles start from the exact approved V43 state
and original normalizer, both width 12. No new network architecture or phase
channels are introduced. Learning rates are 1e-5 and 3e-6, below the original
training rate, with inherited objective weights or a conservative preservation
objective. Normal/stricter thresholds are fixed at .9/.2 and .905/.16 rather than
searched; only learned weights change. This is initialization and protocol
evidence, not an accuracy improvement or release.

The complete 2,302 fitting and 618 validation recordings are required. Both
policies must preserve every validation recording and separately improve overall
extra notes, real piano, sampled piano and held-out isolated pitch. All 603
consumed regressions and one untouched 40-clip fresh MP3 test must follow, with
no runner-up, retuning, replacement or weakened gate after a failed test.
Portable/native/runtime checks remain mandatory before website weights change.

All 112 previously used authored instrument fixtures are restored and verified
through the native data loader (88 fitting, 24 validation, no instrument leakage).
The remaining original feature sources and regressions still need restoration.
Piano transfer and dynamics feature restoration are live; no fitting run has
started. Local training must remain below 25 GB. No user uploads or saved scores
are processed. Nine focused and 1,545 full-suite tests pass (seven full-suite
skips, five warnings). The protocol is frozen before any candidate predictions.
