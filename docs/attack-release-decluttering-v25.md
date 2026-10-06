# Acoustic note endings for accompaniment decluttering

V23 and V24 removed incorrect notes but failed coverage preservation on a brief
supported fragment near a detected note's ending. Their frozen results remain
rejected. V25 adds observed ending evidence rather than changing their gates or
trying alternative checkpoints after regression results.

Each candidate supplies three acoustic views: 40 positions of coarse CQT context,
61 positions around its predicted attack, and 61 positions around its predicted
ending. Both local views contain the same 18 pitch-relative CQT/STFT/flux channels
on the same physical clocks. Separate trainable local encoders feed a binary
KEEP/REMOVE classifier with 84 context features and the coarse encoder. The
ending window is an input hypothesis to test; its usefulness is not assumed.
Neither local view supplies reference timing, labels or fitted model scores.

Ending evidence was prepared for exactly 573 training and 160 validation
recordings, with zero regression audio and no transcription inference. Waveform
bytes, source groups, candidate intervals and caches are tied to the frozen V16
baseline. The reader independently validates and joins the existing attack cache
by identity. Duplicate, overlapping, incorrectly sized or regression partitions
are rejected before preparation reads audio or creates output files.

Two profiles retain V24's KEEP weights 16/32 and partial-fragment loss multipliers
4/8; they train for 80 epochs with new seeds 270001/270002. The 160 validation
recordings alone select one checkpoint and confidence/guardian thresholds.
Normal and stricter confidence settings must reduce false notes while preserving
every matched attack, hold and covered reference-pitch interval per recording.
Retained onsets, endings, pitches and velocities remain exact. Fixed baseline-pair
timing and repeated-key spacing cannot worsen; deletion earns no timing credit.

After selection is frozen, ending windows for the 360 consumed regressions are
derived from their pinned waveforms without transcription inference. These
observations cannot enter fitting or selection. Any regression failure prevents
first-pass audio generation. A passing candidate receives one 32-clip MP3 batch,
with seeds 270101–270132 and original/texture generators at 48/128/192 kbps fixed
before training. Both configurations again require positive false-note reduction
and all preservation checks. No threshold tuning, replacement checkpoints or
replacement first-pass seeds are allowed. A portable export uses format version
2 for the additional encoder; runtime parity remains a separate release gate.

The [existing commercial-compatible dataset notice](../backend/app/assets/bass-temporal-refinement-v1.LICENSE.txt)
applies. No user uploads or saved scores are used or regenerated. V25 is currently
training; no candidate weights are installed on the website. These development
datasets cannot establish accuracy on arbitrary songs.
