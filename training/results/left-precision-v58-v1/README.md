# Candidate-relative phase model V58

The new frame-support architecture receives four fixed phase channels alongside
its nine observed coarse CQT channels: frame-index rank, before-note mask,
interior mask and after-note mask. These match the unchanged 8/24/8 observation
clock; they are not uniformly spaced seconds, tempo labels or reference truth.
Raw attack/release boundary conditioning and periodic note rejection remain.
The nontrainable phase buffer must survive checkpoints exactly.

Two fresh width-20 profiles use the same licensed fitting population as V57:
2,302 training recordings, with 381 timed and 237 pitch-only validation recordings.
No new audio acquisition or inference, no failed member checkpoints or teacher
outputs. The balanced profile has no hard-KEEP term; the protected profile adds
the fixed worst-10-percent genuine-note loss with weight six. The complete
training population must have all-KEEP additional permissions before fitting.

Both policies must preserve every genuine identity, pitch, hold, repeated/chord
timing and retained clock, with separate positive overall, real-piano, NRG-CP
and isolated wrong-pitch gains. One validation winner may enter all 603 consumed
regressions, then one untouched 40-clip MP3 pass and runtime parity. No runner-up
or retuning after test failure. Production stays V43 until every gate passes.

Status: preregistered after 1,363 passing tests, six skips and five warnings;
native model/release imports and phase contract pass. Fitting is ready, with
no candidate predictions or consumed/fresh test reads yet.
