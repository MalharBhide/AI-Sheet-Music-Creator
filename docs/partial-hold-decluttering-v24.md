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
applies. No user upload or saved score is used or regenerated. V24 is currently
fitting; the website continues to use the previously approved V16 model. There
is no claim yet of improved accuracy or a successful release.
