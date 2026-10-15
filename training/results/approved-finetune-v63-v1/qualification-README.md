V63 release qualification
========================

Qualification code is registered before candidate fitting or regression access. It requires the one full preserving validation winner, unchanged fitting manifests/protocol/source hashes, and exact checkpoint identity. The checkpoint must retain the approved V43 architecture, initialization metadata and float32 normalization byte for byte; all learned parameters must be finite float32 tensors.

Only after these checks may all 603 consumed regressions be loaded. Both normal and strict policies must preserve genuine identities, clocks, timing and coverage and provide positive false-note gain. A completed regression cannot be replaced or retuned. Evaluation results bind the qualification protocol and evaluator hashes. The remaining fresh first-pass and runtime gates are still required before deployment.

The original 11 focused tests passed; a further float32-weight rejection case is covered by the full suite: 1611 passed, 7 skipped, 5 warnings. No actual regression or fresh evaluation data has been read, no candidate is trained, and no new weights are deployed.
