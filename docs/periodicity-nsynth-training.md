# New waveform recurrence and NSynth training studies

This study adds a previously unused data source and a previously unused acoustic
feature family. It does not process user uploads or regenerate saved scores.
The website remains on the approved V25 model until every subsequent release
check passes.

Google's [original NSynth dataset](https://magenta.withgoogle.com/datasets/nsynth)
is distributed under CC BY 4.0. Credit Jesse Engel, Cinjon Resnick, Adam Roberts,
Sander Dieleman, Douglas Eck, Karen Simonyan and Mohammad Norouzi, *Neural Audio
Synthesis of Musical Notes with WaveNet Autoencoders* (2017). It supplies
four-second, 16 kHz notes from acoustic, electronic and synthesized instruments;
the key is held for three seconds and the last second is its decay. These are
instrument samples, not a full-song transcription benchmark.

The original `nsynth-valid.jsonwav.tar.gz` archive was retrieved through its
Google Storage bucket, with verified HTTPS, after the older download hostname
failed certificate validation. Its 1,068,767,009 bytes match the publisher's MD5
`87e94a00a19b6dbc99cf6d4c0c0cae87`. SHA256 and the exact license, source, extraction
recipe, selected file hashes and metadata are archived under
[NSynth provenance](../training/results/nsynth-bass-subset-v2). No mirror-derived
or pseudo-transcription labels were substituted.

The bounded selection retains 497 bass/keyboard notes at pitches 21–59 and
velocity 75. Entire instrument identities are separated: 11 train, three
validation, four reserved. This is our declared partition of the publisher's
validation archive, not the official NSynth benchmark. Six declared sequences
per fitting instrument cover single notes, repeated keys, true octaves, dyads,
quiet notes and changing layers, giving 66 train and 18 validation MP3 clips at
48/128/192 kbps. Actual key holds and the complete decay tails remain separate
reference targets. Four reserved instruments are acquired but not transcribed,
scored or fitted until a frozen candidate passes all prior regressions. Upstream
pretraining overlap and sample-library duplicates are not ruled out.

An explicit [V25 fitting baseline](../training/results/bass-v25-baseline-v1)
applies the shipped verifier to the sealed prior interval observations, retaining
573 train and 160 validation recordings and removing 501 earlier candidates.
No decoder inference is needed. Legacy cache readers run against the unchanged
`e56554c` source snapshot, with the archived parity-executed V25 runtime. Their
old V16 route guards remain unchanged. The new reader independently pins the
currently deployed V25 route, its arrays and runtime/source checksums.

Waveform recurrence uses linear FFT autocorrelation at 8 kHz, 2048-sample frames
and 64-sample hops. Overlapping segment energies normalize each lag. Nine
positions around the predicted attack, interval and ending sample recurrence at
half, one, two and three candidate periods, adjacent-semitone periods, and
relative energy: 63 features. All frequencies are pitch-relative. No labels,
reference endpoints, filenames, corpus identifiers or fitted classifier scores
are inputs. Authored sine tests distinguish a fundamental from an octave ghost;
gain scaling, short silence and malformed clocks are checked separately.

V26 fits a two-layer binary support MLP on those features plus 84 acoustic
context features, using 639 training and 178 validation recordings. Two frozen
profiles train for 80 epochs; validation alone selects one checkpoint, confidence
and acoustic guardian threshold. Every recording must preserve matched attacks,
holds, reference-pitch coverage and fixed-pair onset/release/repeated-key timing.
Deletion earns no timing credit and retained clocks are never rewritten.

Validation selected `recurrence-conservative` epoch 80, removal .7 and guardian
.5. It removed eight additional incorrect notes at normal confidence and five
at stricter confidence without violating any validation gate. The frozen
392-record regression removed nine/five incorrect notes but both settings lost
7.3977 ms of supported pitch coverage in `bass-temporal-fresh-v1-263121`.
All matched attacks/holds and fixed-pair timing survived; coverage still failed.
**V26 is rejected.** No runner-up or threshold switch was tried after observing
that failure. The actual first-pass tool refused the failed candidate before
creating an output directory. Reserved NSynth instruments and seeds
271101–271132 remain unconsumed; no weights were exported or deployed.

[Frozen V26 reports](../training/results/periodicity-v26-v1) include learning
curves, all eight checkpoint hashes, exact validation/regression reports and
selection metadata. The first 760 backend/training tests passed, with the six
native integration tests skipped because production code did not change.

The next V27 study combines detailed attack/ending encoders with the recurrence
branch. It starts from the approved V25 checkpoint and gives the new branch zero
initial influence on predictions, verified with nonzero random-weight parity
tests. Separate branch widths 32/64 and KEEP/fragment weights 16/8 and 32/16 train
on the same disjoint sources. It reuses the 392 consumed acoustic observations;
they never fit or select. A winner must pass every normal/stricter regression
before the declared four-instrument NSynth batch and 32 new procedural MP3s
(seeds 272101–272132) can run. Positive first-pass noise reduction, full
preservation, portable runtime parity and native pipeline checks are mandatory
before deployment. A smaller loss or a smaller note count alone cannot release
an experimental model.

Both V27 profiles completed 80 epochs. Validation selected `fusion-protected`
epoch 40, recurrence width 64, removal confidence .9 and guardian .5, with
eight/five additional wrong notes removed at normal/stricter confidence. All 178
validation recordings pass every preservation gate. The selected checkpoint
SHA256 is `82bc1d25e4e93c52b091b8fa764f2091aef476a1759fc66ba6ffce66569a3843`.
The frozen 392-record regression removes 21/15 extra incorrect notes. Normal
confidence loses 7.3977 ms of supported pitch coverage in the same earlier
regression; stricter confidence preserves every recording. Both settings must
pass, so **V27 is rejected**. No runner-up or threshold switch was tried. The
actual first-pass tool refuses the failed candidate before creating its output
directory. All four reserved NSynth instruments and seeds 272101–272132 remain
unconsumed; no experimental weights are exported or deployed. All 763 backend/training tests passed (six native tests skipped),
and four focused warm-branch/first-pass guard tests passed.
