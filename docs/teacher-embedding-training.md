# Frozen spectral representation with physical recurrence

V30 and V31 could preserve all validation recordings, but their recurrence-only
correction branches did not reduce wrong notes at both required confidence
settings. Their frozen plans, results and source contracts remain unchanged.

V32 adds the approved acoustic teacher's 32 pre-logit ReLU features to the 71
physical recurrence/release-neighbor features. The new branch receives 103
inputs, using `log1p` on the nonnegative learned representation. It receives no
teacher probabilities, reference labels, source identities or absolute pitch.
The teacher's parameters and original 84-feature normalization remain frozen;
dropout stays disabled. The explicit teacher computation preserves its exact
logits, and a zero-initialized bounded correction starts at exact teacher parity.

This controlled feature experiment uses the already verified 908 fitting
recordings, including new NSynth and human-played Groove mixtures, with 702
training and 206 validation recordings. The two profiles use widths 64/96,
correction caps 2/4, KEEP weights 16/32, partial-support boosts 8/16, seeds
277001/277002, and 80 epochs each. Checkpoints and confidence grids are declared
before fitting. Every validation recording must preserve attacks, holds, pitch
coverage and fixed-pair timing at both settings, while reducing wrong notes.

A single frozen winner must then pass all 392 consumed regressions, followed by
77 first-pass fixtures with reserved NSynth instruments/Groove performers and
authored seeds 277101–277132. No runner-ups, threshold changes or replacement
fixtures after testing. Portable/runtime parity and native physical pipeline
checks remain mandatory before deployment.

MusicNet acquisition is a separate new-corpus effort. No unverified MusicNet
transfer enters V32 fitting. The new real-piano corpus will only enter a later
independently frozen experiment after its complete publisher checksum, clocks,
source partition and label validation pass. No user uploads or saved scores are
processed. Every V25 stage remains in the released pipeline; V32 adds a gated final verifier.

Both profiles completed all 80 epochs. The frozen winner is conservative width
64, cap 2, epoch 60, removal confidence .8 and guardian .5. It preserved every
validation recording while removing four false notes normally and two at the
stricter setting. It then passed all 392 consumed regressions (15/11 false notes
removed) and all 77 independent fixtures (2/1 removed), preserving tested
attacks, holds, pitch coverage and fixed-pair timing throughout.

The portable asset passed every release gate and is deployed on the local website. Runtime verification covers 1,377
recordings/59,849 events with exact probability parity and identical decisions
at both settings. Physical feature checks on all 77 new fixtures match to within
2.98e-8. A memory-bounded recurrence observer avoids a duration-sized correlation
table. All six native upload/rendering checks passed against an isolated route
that applies V32 only after V25. The staged route and installed source both passed 832 tests (six native
tests skipped in those ordinary suites and run separately). The rebuilt website is healthy and its live asset and source fingerprints match
the gated stage. The original engraving source remains unchanged. The V25 asset and all eight preceding stages are retained. Results, source hashes and fixture provenance are archived in
`training/results/teacher-embedding-v32-v1`.

The 77 independent fixtures are now consumed regression evidence. They must not
be reused as untouched first-pass data for future experiments. MusicNet reserved
works remain uninferred while its full publisher archive is being verified.

Deployment evidence is `training/results/teacher-embedding-v32-v1/release-checks.json`.
The ordinary image build encountered a Docker registry metadata timeout. The
successful fallback copied exactly the gated files onto the already approved
V25 image, retaining its dependencies. The local overlay recipe and base image
identity are archived. Backend host port 18000 avoids ports used by another app;
the browser URL remains http://localhost:5173/ through the unchanged frontend.
Saved scores and user uploads were not regenerated. Future candidates must use
an explicit V32 baseline or an archived V25 replay, preserving historical guards.
