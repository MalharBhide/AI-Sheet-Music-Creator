Approved-model fine-tuning with regenerated native observations
=============================================================

This replaces the deleted-cache dependency with an explicit new data recipe. It keeps all 908 original core recordings and their exact archived labels, source groups and waveform checksums. Features are regenerated through the unchanged pinned V32 decoder; they are not claimed byte-identical to historical caches. Historical readers and guards stay unchanged. Full waveform hashing is mandatory before preparation, and the fitting reader checks every physical cache and source binding again.

The complete fitting scope stays 2302 training and 618 validation recordings, including 381 timed and 237 isolated-pitch validation recordings. All 2012 supplemental records are retained through their native checks. MusicNet component checks are copied unchanged without the deleted legacy parent dependency. No original recording or difficult evaluation case is excluded.

V63 fine-tunes the exact approved V43 model and immutable normalizer with the same two independent profiles, fixed thresholds and release gates as V62. Both policies must pass full validation, then all 603 consumed regressions, then a single untouched 40-MP3 first pass, then physical/runtime checks. No runner-up or retuning after test failure. No new architecture, no user uploads or saved scores, no weaker release gate.

19 focused tests pass. Full code suite: 1599 passed, 7 skipped, 5 warnings. Native registry preflight passes all 908 original records and unchanged decoder guard; full audio checks await the final separated-source recovery. Preparation and fitting have not started; no candidate predictions, accuracy gains or new deployment are claimed. Training stays below 25 GB.
