# Periodic left-hand support V40 — refused release

Two fresh 100-epoch profiles trained on 765 licensed excerpts and selected on
221 separate excerpts. The model combines coarse/attack/release CNNs, 100
physical context features and 63 waveform periodicity features. V39 is a
completed, withheld predecessor; its scores and weights are not inference inputs.

The frozen winner is support-protected, epoch 50, threshold .99 (stricter .995).
It removes 72/58 false validation notes, including 1/1 on the separate real piano
subset, while preserving all 221 validation excerpts. Later checkpoints do not
automatically supersede the passing epoch.

The frozen winner then fails five of the 563 required consumed recordings at
both settings. It removes 247/193 false predictions, but also loses three genuine
bass attacks and two further short pitch continuations. Aggregate improvement
does not permit these regressions. Retained clocks remain identical to V38.

**No new first pass, runner-up, threshold retuning, export or deployment.**
Fresh seeds 285101–285140 remain untouched. The website retains its approved V38
release model and prior note filtering. No user audio or saved score was used.

See `outcome.json` for the decision and actual checkpoint hashes, and
`consumed-regression.json` for every recording at both confidence settings.
Plans, searches, curves, diagnostics and the exact executed source witnesses are
preserved. Audio, feature caches and weights remain local. The method and limits
are described in `docs/left-hand-precision-training.md`.
