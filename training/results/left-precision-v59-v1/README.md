# Known-absence maximum-frame fitting objective V59

This study keeps the V58 phase architecture and all inference policies unchanged.
An additional fitting penalty uses the strongest of the 24 candidate-interior
frame logits for supervised REMOVE examples only when every interior target is
known zero. Genuine KEEP, UNKNOWN interiors and even one supplied positive
fragment are excluded. The objective may better match the existing maximum-frame
veto; a higher accuracy is a hypothesis to test, not an established result.

Two fresh width-20 profiles use the unchanged 2,302 licensed training recordings
and 381 timed plus 237 pitch-only validation recordings. The balanced/protected
maximum-absence coefficients are 1/.5. All prior class/presence/corpus protection
weights and hard-KEEP coefficients remain those of the corresponding V58 profile,
with new initialization seeds. No inherited checkpoints or failed test data.
The 49 later real-piano proposals remain unacquired and outside this fitting run.

Both policies must preserve every recording's required identities, pitch, holds,
repeated/chord timing, unsupported duration and retained clocks, with separate
positive overall, real-piano, NRG-CP and isolated wrong-pitch gains. One validation
winner may enter all 603 consumed regressions, one untouched 40-clip MP3 first
pass, then runtime checks. No runner-up, retuning or replacement clips after test
failure. Production remains V43 unless all release gates pass.

Status: preregistered after 1,433 passing tests, six skips and five warnings.
Native model/release imports and the exact phase contract pass. Fitting is ready;
no new candidate predictions or consumed/fresh test reads have occurred.
