# V46 additional left-hand removal study

This study trains two fresh spectral-periodic note-rejection and frame-presence
models on 1,933 training recordings, including the newly prepared TinySOL pitch
corpus. Unlike V44/V45 replacements, the new verifier intersects its output with
the corresponding frozen V43 normal/stricter permissions. A V43 rejection can
never be restored. V43 weights remain unchanged and are not learner inputs.

The full V38 release context is computed before this intersection. Retained
pitch, onset, release and velocity must remain identical. The training data,
whole-source partitions and unknown TinySOL timing masks match the verified V45
data audit. New normalization uses only eligible labeled training candidates;
no teacher targets, validation normalization or inherited weights are used.

Selection still requires positive real-piano and overall false-note gains while
preserving each of 345 timed recordings, plus positive wrong-pitch gain and all
baseline fundamental identities on 237 held-out TinySOL recordings. Every
recording must have zero restored predictions. Additional removal can still
delete genuine notes, so the composition guard does not replace those tests.

One frozen winner must subsequently pass all 603 consumed regressions, one new
40-clip original/texture MP3 first pass, runtime parity and native functional
checks. No runner-up, retuning or replacement clips after failure.

Status: both predeclared 100-epoch profiles are running. Fifteen targeted checks
and the full 1,024-test suite passed, with six full-suite skips. No website weights
have changed, and no user uploads or saved scores are processed.
