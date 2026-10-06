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
applies. No user uploads or saved scores are used or regenerated.

Both training profiles completed 80 epochs. Validation selected
`support-conservative` epoch 60, removal confidence .8 and guardian threshold .5.
Checkpoint SHA256:
`c1d8d11c559c6cd47f32a1cdfe89ca19c1e691c942119a166c90cc780fb5a074`.
The normal/stricter settings remove 105/93 incorrect validation notes and
196/182 incorrect notes across the 360 consumed regressions. Every recording
passes attack, hold, pitch-coverage and timing preservation checks, including
the fragment that rejected V23 and V24. The frozen first-pass set of 32 new
project-authored MP3s also passes both settings, removing 26/22 incorrect notes
without losing tested attacks, holds, pitch coverage or repeated-note spacing.
Those 32 recordings are now consumed; future candidates need 392 regressions.

The portable format-2 asset has SHA256
`f4c5d1fe9c411eca32b28b3eb2c6c7e38e292e1a0d4191d2d6fa81542b6e424c`.
Runtime parity covers all 1,125 cached recordings / 57,627 candidate notes at one
and two Torch threads, with zero probability error and identical normal/stricter
decisions. Actual filter calls preserve note objects, source parts and retained
fields. Four consumed project-authored MP3s additionally reproduce native V16
candidates and physical context/coarse/attack/ending features with zero error.
Executed prototype sources and the pre-V25 route are archived under
`source-witnesses`; installed runtime changes only import ordering/whitespace,
verified by canonical Python syntax trees. Frozen old-baseline hash guards remain
unchanged and intentionally reject the new production route; replay the scientific
study against the pre-activation source or commit `e56554c`.

Production adds the verifier after all existing V16 bass stages, only for the
balanced Full-song arrangement. It may delete unsupported bass notes but never
rewrites retained pitches, onsets, endings or velocities. First/last 2.5 seconds,
short clips, other registers and slightly overshooting legacy decoder intervals
are preserved. Metadata records its model name and rejection count. Solo-piano
and melody routes, score note size (7 mm staff), and pagination are unchanged.
The final backend image contains this sealed asset and its
[commercial-compatible dataset notice](../backend/app/assets/bass-attack-declutter-v1.LICENSE.txt).

[Frozen fitting/evaluation evidence](../training/results/attack-release-v25-v1)
and [ending-feature preparation metadata](../training/results/release-local-bass-v1)
preserve exact reports, plans, source/cache hashes and checkpoint inventories.
All 742 backend/training tests passed (six opt-in integration tests checked
separately: all six native integration tests passed). Runtime tests include sealed asset loading/corruption, short/edge
abstention, source-part retention, guardian vetoes and routing after V16.
These development datasets cannot establish accuracy on arbitrary songs.

The local website deployment was verified through `http://localhost:5173/api/health`
and by loading the sealed model in its running backend. All dependencies report
ready. The original score renderer checksum is unchanged; five completed saved
jobs were preserved. Existing scores are not regenerated. Upload again in
balanced Full-song mode to test the new accompaniment verifier.
