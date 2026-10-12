# V45 licensed weak-pitch fine-tuning study

This framework starts from the approved V43 weights and training-only
normalization. Two predeclared 100-epoch profiles rehearse earlier fitting
predictions while learning from 712 new TinySOL recordings. The 1,221 earlier
training recordings remain included. Teacher predictions are a stability loss,
not reference truth, and never become inference inputs or validation labels.

TinySOL supplies isolated fundamental pitch and dynamics, without precise human
onset, release or pedal labels. Source-fundamental frames inside each recording
are UNKNOWN and contribute no frame loss or gradient. Wrong pitches and authored
silence supply absence labels. Conservative note KEEP labels protect overlapping
fundamentals. These clips cannot establish human timing accuracy.

Selection requires positive false-note reduction and separate real-piano gain
while preserving every one of 345 timed validation recordings. An additional
237 TinySOL validation recordings, split by whole instruments, must preserve all
baseline fundamental candidate identities and show positive wrong-pitch gain.
Lost genuine notes and restored wrong notes fail the gate. All retained note
clocks remain identical to the full-context V43/V38 baseline.

After one validation winner is frozen, all 603 consumed regressions and one new
40-clip MP3 first pass must pass. No runner-up, retuning or replacement test clips
after a failure. Runtime parity and native functional checks precede deployment.

Status: both 100-epoch profiles completed. None of the 108 validation settings
qualified; all V45 weights are withheld. Every setting restored extra notes
on timed validation, and TinySOL wrong-pitch gain remained zero. No consumed or
fresh tests were read. No V45 weights have been deployed. The website asset is unchanged. The test proof is in
preflight.json: 22 targeted checks and 1,014 full-suite checks passed, with six
full-suite skips. The initial combined-suite collection attempt encountered
duplicate test-module basenames; isolated imports resolved collection without
changing application code.

User uploads and saved scores are excluded from every stage.

The completed audit verified all 2,515 fitting/validation recordings and both
initial policies on all 582 validation recordings, with zero changed decisions.
The partitions contain no shared source groups. All 121 TinySOL clips without
predictions remain included. The original audit script is preserved as a text
witness.
