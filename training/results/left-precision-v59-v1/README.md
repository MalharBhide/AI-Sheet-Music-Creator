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

Final status: both profiles completed 100 epochs and all 144 settings.
57 settings preserve all required validation recordings, but no single setting
meets every separate positive domain gain under both policies. The study is
withheld, with no consumed or fresh tests read and no website asset changed.
The maximum-absence term applied to 11,025 fitting candidates; all 53,070 genuine
KEEP candidates and 23 partial/unknown REMOVE candidates were excluded. Both
profiles proved all-KEEP initial permissions on all 64,118 fitting candidates.

Final code checks: 1,453 tests passed, six skipped and five warnings. Later
real-piano source acquisition is independent and cannot change this closed
study's fitting population, selected weights or validation outcome.

Native release smoke check passes: this withheld study cannot reach consumed
or fresh inference. Frozen fitting sources and approved production assets remain
unchanged. Training has finished; later-source acquisition is independently live.
