# CocoChorales piano preparation V2

V1 stopped on its first native render before model inference; its exact plan,
source witnesses and failure are preserved. V2 changes only the new preparation
and adds a separate native renderer. Historical V47/V53 renderer files are unchanged.

All 108 fixed derivatives rendered successfully in the native preflight, retaining
all original authored key clocks. FluidSynth documents that note-off failure can
mean no matching voice remains. An unmatched release is accepted only for the
isolated pitch renderer with a synchronous zero active voice count; each case is
recorded. Other failures remain fatal. No audio driver runs concurrently.

The command clocks are conservative pitch-support bounds, not acoustic-presence
labels. V54 training masks every positive support frame of this new source; only
known absences receive frame loss. Candidate KEEP supervision conservatively
protects declared pitches. This study makes no human sustain-timing claim.

All 36 parents and three fixed dynamics/MP3 variants are fitting-only. No user
recording, consumed test or fresh evaluation is processed. New training requires
strict completed-cache audits; deployment gates remain unchanged.

Primary API: https://www.fluidsynth.org/api/group__midi__messages.html
and https://www.fluidsynth.org/api/group__synthesis__params.html
