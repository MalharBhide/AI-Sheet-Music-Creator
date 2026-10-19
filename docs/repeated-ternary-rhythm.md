# Preserve repeated swing and partial-triplet timing

The [pipeline audit](pipeline-diagnostic-2026-10-19.md) showed that correct repeated 2:1 swing and two-note triplet fragments were forced onto straight sixteenths. This patch changes rhythm cleanup rather than training another note-removal model.

In Precise rhythm, three consecutive beats containing the same tightly aligned on-beat/off-beat pair now establish a ternary pattern. The evidence must fit better than straight sixteenths and thirty-seconds. One or two isolated pairs, chord duplicates, patterns separated by rests, loose timing, and ordinary straight rhythms do not establish this pattern. Complete-triplet recognition remains supported.

Phase alignment also compares both straight grids before selecting a ternary phase. This prevents jittered swing from locking onto a weaker straight interpretation and prevents sparse straight thirty-second pairs from being shifted into a plausible-looking triplet pattern.

For newly recognized partial-triplet beats, release duration is chosen independently of the eighth-triplet attack spacing. Straight and ternary sixteenth durations are represented on the existing notation lattice, so staccato notes need not become full eighth-triplet holds. This is quantization, not learned acoustic release detection.

## Measurements

The saved pre-change revision is `e9ac0a0`. The same authored notes were run through phase estimation, cleanup, MusicXML and actual score playback before and after the patch.

| Authored fixture at 120 BPM | Previous repeated-spacing error | Updated error |
| --- | --- | --- |
| Repeated 2:1 swing | 41.67 ms | Less than 0.001 ms |
| Repeated two-note triplet fragments | 41.67 ms | Less than 0.001 ms |

Both fixtures retain all eight notes with no additional pitches or missing attacks. Their release errors are also below 0.001 ms after playback export. The other eight original diagnostic cases produce exactly the same playback events as before. In particular, this patch does **not** fix the previously documented short/quiet bass deletions, the general minimum-duration issue outside the recognized patterns, variable-tempo handling, or loud incorrect accompaniment selection.

New tests cover 80/120/160 BPM, shifted phases, small timing jitter, vocal/bass/piano roles, neighboring straight passages, held chords, staccato releases, sparse thirty-second counterexamples, and actual score playback. One existing test intentionally changes its expected result for a long, consistent incomplete-triplet run; it still rejects an insufficient two-beat fragment.

Backend verification: **432 passed**, six real-transcription integration cases deselected, two dependency deprecation warnings. No user audio was processed, no training ran, and no model weights or detection thresholds changed. These symbolic results establish the specific correction, not a real-song accuracy percentage or proof of fewer acoustic false notes.

## Scope and release state

The source change is ready for review and a subsequent backend rebuild. This work does not restart local website containers or queued workers. Existing saved scores are not regenerated. Frontend layout, note size, and page scaling are unchanged. The remaining acoustic and accompaniment issues still require the stage-attribution work described in the audit; further bass-only fine-tuning is not justified by this result.

Evidence: [comparison plan](diagnostics/ternary-rhythm-v1/comparison-plan.json), [before/after metrics](diagnostics/ternary-rhythm-v1/comparison.json), [complete updated measurements](diagnostics/ternary-rhythm-v1/updated-report.json), [test output](diagnostics/ternary-rhythm-v1/backend-tests.log).

Reproduce the diagnostic with existing backend dependencies:

```sh
PYTHONPATH=backend python scripts/diagnose_transcription_pipeline.py \
  --root . --output .training/rhythm-fragments-v1/updated-report.json
PYTHONPATH=backend:scripts python -m pytest backend/tests \
  scripts/test_diagnose_transcription_pipeline.py -q -m 'not integration' -p no:cacheprovider
```
