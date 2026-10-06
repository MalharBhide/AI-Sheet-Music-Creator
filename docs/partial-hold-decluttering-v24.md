# Partial-hold protection during decluttering training

V23 reduced incorrect notes but failed its coverage gate on a short supported
fragment. Its release remains rejected. V24 is a new, separately frozen fitting
experiment; no V23 checkpoints or thresholds are retuned after regression.

The existing labels already protect every positive reference-pitch overlap.
V24 changes training loss, without changing those labels or acoustic inputs:
protected KEEP examples with positive reference overlap covering less than 25%
of their predicted duration receive additional weight. The two fixed profiles
use KEEP weights 16/32 and partial-fragment multipliers 4/8. Training references
determine these loss weights only; no overlap or reference labels reach the CNN
inputs or runtime decisions. Regression examples remain excluded from fitting
and selection. This policy applies to all training recordings and all pitches
in the existing eligible bass range, with no song or recording-specific rule.

The same coarse/local acoustic architecture trains for 80 epochs on 573
recordings. The 160 validation recordings alone select one checkpoint and
confidence/guardian thresholds. Normal and stricter confidence settings must
remove incorrect notes while retaining every matched attack, hold and supplied
pitch interval in each recording. Retained timing, pitch and velocity are exact.
All 360 consumed regressions must pass before any new MP3 first-pass inference.
The first-pass seeds are fixed to 269101–269132, with the same original/texture
generators and 48/128/192 kbps codec policy. Positive false-note reduction and
every preservation gate are required under both configurations; a failure
prevents export. Runtime parity is also required before deployment.

The [existing commercially compatible dataset notice](../backend/app/assets/bass-temporal-refinement-v1.LICENSE.txt)
applies. No user upload or saved score is used or regenerated.

Training completed. Validation selected `support-conservative` epoch 60,
confidence .7 and guardian .5; checkpoint SHA256:
`85c2ceb5c1e5c37525b22ff2c7a22adabf327134c2d9a2a4945401df9cb58791`.
Normal/stricter settings remove 120/101 false validation notes and pass all
160 validation recordings. They remove 219/208 false notes in the 360 consumed
regressions. The normal setting still loses .0073977324 seconds of reference
pitch coverage in `bass-temporal-fresh-v1-263121`; the stricter setting passes.
Matched attacks, holds and retained event clocks remain intact in both settings.

The frozen policy requires both settings to pass. V24 is rejected, without
switching to the stricter setting or trying the other checkpoint. Seeds
269101–269132 remain unused; no first-pass audio is inferred, and no weights are
exported or deployed. The website continues to use the approved V16 model.

[Archived evidence](../training/results/declutter-local-v24-v1) includes exact
plans, learning curves, selection, lossless evaluations and checkpoint hashes.
All 709 backend/training tests passed (six opt-in integration tests skipped).
Eleven additional inactive runtime tests passed with random initializer
fixtures. These results do not establish accuracy on arbitrary songs.
