Independent V63 first-pass qualification
=======================================

The fixed 40-MP3 first pass uses seeds 305101–305140, alternating original and texture families with 48/128/192 kbps encoding. Its source hashes, parent qualification protocol, fixed seeds, storage reservation and non-replacement rule are registered before any candidate fitting. No fresh audio has been generated or evaluated.

The runner first requires the one complete preserving validation winner and all 603 consumed regressions at both policies, including genuine identity, timing and coverage preservation. It rejects incomplete, duplicate, changed-checkpoint or changed-protocol evidence before generating audio. A partial attempt or failed completed first pass cannot be replaced or retuned. Each family must provide positive false-note gain at both policies. A 600 MB reservation is required under the 25 GB local storage limit. Original decoder and physical feature source hashes are bound to the evaluation and checked again before recording its result. Runtime/export checks remain mandatory before deployment.

All 21 access-control tests pass using fabricated metadata only. The native source-protocol preflight passes. Full backend/training suite: 1632 passed, 7 skipped, 5 warnings, using `python -m pytest -q -p no:cacheprovider --import-mode=importlib -c backend/pyproject.toml backend/tests training`. An initial root collection attempt failed because existing backend and training/script test files share names; isolated imports resolve collection without changing application code or excluding backend/training tests.

The one-shot handoff was updated to import ML libraries in a short-lived preflight child. The waiting parent uses approximately 19 MB rather than 714 MB, leaving more memory for the fitting process. Native preflight passes; no training source or release gate changed.
