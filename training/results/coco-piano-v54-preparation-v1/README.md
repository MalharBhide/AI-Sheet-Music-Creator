# CocoChorales authored piano fitting preparation

Exactly the 36 training-only parent sources from the completed symbolic audit
are selected before any new rendering or predictions. Their complete clocks
have 3,248 keys and 62 holds over four seconds. All same-key-overlap sources are
excluded; no voice merge, shortening or guessed note-off is used.

The [publisher](https://magenta.tensorflow.org/datasets/cocochorales) grants
CC BY 4.0. Dataset/source attribution and raw symbolic provenance are retained
locally. This derivative explicitly maps expression volume to integer piano
velocity, then renders the full 4ms command sequence at the pinned sample-piano
block clock, with authored 300ms damping and conservative pitch support.
Three predeclared dynamics/MP3 variants produce 108 fitting-only recordings.
No crops/retriggering, original ensemble recordings, pitch-corrected F0, human
performance labels or new sampled-instrument independence are claimed.

Parent MIDI identities group all derivatives together. No new source is added
to validation or tests. The full suite passes 1,194 checks with six opt-in skips
and five dependency warnings. Acquisition/preparation is pending; no model
training uses this source until strict completed-cache audits pass. All existing
model release gates remain unchanged, and website weights stay V43.
