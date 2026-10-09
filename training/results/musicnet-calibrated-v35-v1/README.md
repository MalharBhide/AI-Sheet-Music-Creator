# Completed calibrated-protection MusicNet V35 study — withheld

Two predeclared profiles completed 80 epochs each on the same verified
765 training and 221 validation recordings (including 63/15 real-piano MusicNet
excerpts). Complete V32 weights and normalization stayed frozen. Independent
955-input correction and protection branches learned from physical evidence;
neither uses labels, source IDs or fitted probabilities as feature inputs.

A preceding fitting-only diagnostic showed that the old guardian blocked every
high-confidence labeled false piano note. This motivated a new protection head,
initially .99, with independent parameters. It does not justify dropping safety
checks or raising cutoffs on consumed recordings.

Neither profile yielded a validation winner: no checkpoint preserved every
recording while achieving separately positive real-piano wrong-note reduction at
both declared confidence settings. Training loss decreased, but this does not
establish better transcription accuracy. No consumed regression, Oxford first
pass, portable export or deployment followed. The website retains V32.

All 519 consumed regressions remain required for a future frozen winner. All
eight Oxford original piano/MIDI pairs and seeds 280101–280132 remain untouched.
The separately declared real-piano first-pass gate must remain independently
positive. No runner-up, confidence retuning, replacement clips or weakened
attack/hold/coverage/timing checks are permitted after a first pass.

Plans, per-stage searches, curves and actual checkpoint hashes are archived here;
audio, feature caches and weights stay local. The full suite passed 886 tests
(6 skipped), and the separate new source-metadata reader passed 7 tests.
No user uploads or saved scores were processed.

A possible next study needs stronger physically distinct evidence or more
verified source diversity; relaxing the guardian alone harmed preservation.
The separate CocoChorales metadata assessment does not fit new weights or alter
this already frozen study, and its synthetic non-piano sources cannot substitute
for an independent piano benefit test.
