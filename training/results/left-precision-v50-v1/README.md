# V50 real-piano training emphasis

This study retains the V49 warm-start boundary architecture, exact released V43
initialization, frozen training-only normalizer and immutable V43 rejection
permissions. It changes loss weights only: verified real-piano training candidates
receive 4x or 8x emphasis, followed by training-candidate mean normalization.
Each profile retains the corresponding V49 KEEP, presence and rehearsal weights.
Corpus identities are never model inputs. No new observations, validation-derived
targets or user recordings enter fitting. Both profiles run 100 epochs.

The fixed partitions contain 2,077 training recordings, 381 timed validation
recordings and 237 isolated-pitch validation recordings. Every earlier per-recording
note, pitch, hold, chord and timing preservation gate remains. Positive extra-note
gain is required separately overall, on real piano, on NRG-CP and on held-out
TinySOL pitches at both policies. One frozen winner must then pass all 603 consumed
regressions and one untouched 40-clip first pass (seeds 295101–295140), physical
runtime parity and native functional checks. No runner-up, retuning or replacement
fresh clips after failure. No gate has been relaxed.

Status: both predeclared 100-epoch profiles completed; none of 144 settings met
all requirements. Every candidate is withheld. Some settings reduced false notes
while preserving measured recordings, but stricter real-piano and isolated-pitch
gains remained zero. Per-domain maxima can come from different settings and are
not release claims. No rejected notes were restored. No consumed/fresh tests or
user recordings were read, and website weights remain unchanged. Curves,
validation reports, checkpoint hashes, complete source witnesses and the exact
62,124-candidate initial parity proof are archived. Audio and checkpoints stay
local. PDMX remains source research and was excluded from this study.
