# Expanded V8 experiments — no model promoted

See [methods and results](../../../docs/accompaniment-v8-experiments.md).

- `guarded`: 27,078 events; no validation gain at required raw/half-margin gates.
- `expanded`: same cohort, V7-style settings; failed regression by losing one matched piano note.
- `evidence`: adds frozen V7 confidence features; regression preserved notes but removed no additional false notes.
- `full-accompaniment`: 53,302 events across MIDI 36–95; no eligible validation gain.

`training-expansion.json` records all 54 added passages and the source manifest
hash. `coverage-audit.json` contains validation-only eligible/protected counts.
All threshold searches and evaluation outcomes are preserved. Regression is
consumed; no new test passages were prepared. Raw audio/features/research pickle
files are ignored local artifacts. The live website continues using V7.
