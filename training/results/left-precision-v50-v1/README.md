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

Status: preflight passes 1,084 tests with six skips, and final runners import with
the pinned historical reader. Fitting has started through the module entry point with the verified historical
reader. The initial direct-script attempt was stopped by the unchanged V32 guard
before creating output; no guard was weakened. Consumed/fresh tests have not been
read, and website weights are unchanged.
PDMX remains source research and is excluded from this study.
