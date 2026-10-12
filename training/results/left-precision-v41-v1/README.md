# Dual physical left-hand support V41 — refused release

Added 88 training and 24 validation authored NSynth MP3 sequences, with the
original instrument split and protected decay tails. Two independent 100-epoch
profiles train spectral/periodic rejection and physical KEEP heads on 853
excerpts. A removal needs both heads' permission. Selection uses 245 separate
excerpts, including 15 real MusicNet piano clips.

The frozen winner is dual-small, epoch 100, threshold .99 (stricter .995).
Validation removes 75/54 false notes and 2/1 on real piano, preserving every
recording. The complete 563-clip consumed suite then removes 202/162 false notes
but loses 3/2 genuine attacks. Both settings fail. No runner-up, retuning,
fresh inference, export or website deployment follows. Seeds 286101–286140
remain untouched. Retained clocks stay identical to the deployed V38 baseline.

`outcome.json` records all six checkpoint hashes. Plans, source witnesses,
learning curves, full per-recording diagnostics, data manifests and the
962-pass test log are preserved. The six optional native checks are skipped;
there are no production changes requiring a new native deployment test.
No user audio or saved scores were processed. Raw audio, caches and checkpoints
remain local. The website retains the approved model and original note size.
