# V54 training study

Train two fresh 20-channel boundary-conditioned note-rejection and masked
frame-presence models on 2,230 fitting recordings. The new 108 CC BY 4.0-derived
piano recordings retain all 36 generated parents in fitting only. Generated key
commands supply conservative pitch-support upper bounds: all positive support
frames are masked and only known absences contribute to their frame loss.
Candidate KEEP targets protect declared key identity; human sustain-duration
accuracy is not claimed. No inherited weights, teacher targets or failed V52 pair.

Use fitting-only real-piano weighting (4x/8x) and supervised TinySOL wrong-pitch
negative weighting (8x/16x), separately renormalized over fitting candidates.
These labels/corpus identities affect training loss only and never become model
inputs. Retain the same 381 timed and 237 pitch-only validation recordings,
thresholds, stricter policies and per-recording preservation gates.

Require positive overall, real-piano, NRG-CP and TinySOL gains at both policies.
Freeze one validation winner before all 603 consumed regressions, then one
untouched 40-clip first pass with seeds 299101–299140 and separate family gains.
No runner-up, retuning or replacement clips after test failure. Portable/native
runtime parity is required before deployment. Training may start only after the
strict completed-cache audit succeeds. No user audio or saved scores are used.
