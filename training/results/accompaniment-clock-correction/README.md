# Corrected accompaniment experiments

No candidate in this directory is promoted. Production remains V2 while a better replacement is developed.

- `clock-audit.json`, `split.json`, and `preparation-reproduction.json` document the repaired Vienna references and exact reproduction across all 88 recordings.
- `label-audit.json` checks training/validation compositions separately; no regression references enter fitting.
- `note-verifier-v3` is the retrained neural model; `note-verifier-v4` is the tree ensemble. Each fails an individual recording. The `note-verifier-consensus` uses their unchanged validation thresholds and preserves original correct detections, but removes too few false notes to beat the deployed release.
- Each `regression.json` preserves the initial measured result. Its historical `passes` field refers to the original-detector check only. The separate `promotion.json` applies the stricter final current-release comparison. **All promotion decisions are false.** The current evaluator emits both `passes_original_detector_guard` and the final release decision to avoid conflating them.
- `decision.json` is the overall release decision. Source recordings and full reference arrays are excluded; split entries retain reference counts and hashes. Checkpoint SHA-256 values are in each selection file. Full trained checkpoints remain in ignored local experiment storage.
- Dataset authors, licenses and archive checksums are retained in the preceding `../accompaniment-verifier-v2/` source manifests and notices. The old Vienna measurements there are invalid because of stale Chopin clock offsets; preserve them only as experiment history.

All 102 evaluation excerpts are consumed regression data. These runs do not establish independent mixed-song or perceptual piano-arrangement accuracy.
