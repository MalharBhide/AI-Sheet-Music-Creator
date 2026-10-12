# TinySOL version 6 — new source assessment

The publisher licenses TinySOL under CC BY 4.0. This assessment concerns only
TinySOL; it does not extend that license to larger SOL or OrchideaSOL releases.
The metadata bytes match the publisher checksum. Counts and source links are
recorded in metadata-assessment.json. Audio acquisition passed the publisher checksum; 949 original recordings were
extracted into five training and two validation instrument groups. All other
instruments remain unextracted. The measured subset spans 1.315–16.235 seconds;
preparation uses measured clocks rather than assuming a nominal duration.

The proposed role is a separate future pitch-presence fitting study. The metadata
contains isolated-note pitch and dynamics, without exact human onset, key release
or pedal times. Protect the entire clip at the supplied fundamental and do not
invent timing labels or use these clips as timing validation. Official folds are
not whole-instrument partitions; freeze project-specific instrument partitions
before feature inference. Retuning flags must remain visible in provenance.

Preparation and strict cache loading completed for all 949 recordings. The V45
fine-tuning study completed without an eligible validation winner; no TinySOL
weights have been shipped. The already
frozen V44 study remains unchanged. No user uploads or saved scores are read.

Source: https://zenodo.org/records/3685367
License: https://creativecommons.org/licenses/by/4.0/
Authors: Carmine-Emanuele Cella, Daniele Ghisi, Vincent Lostanlen, Fabien Lévy,
Joshua Fineberg, and Yan Maresz. Recordings originate from Ircam SOL.

The pitch-only preparation adds three seconds of authored silence at each end,
cycles MP3 bitrates 48/128/192 kbps, and verifies codec alignment. Same-fundamental
candidates overlapping the source clip receive conservative KEEP supervision.
Fundamental frames inside the source have UNKNOWN timing and zero frame loss
weight; only other pitches and authored silence supply frame absence. A future
trainer must use these masks and cannot treat the whole clip as a held note.
No human onset/release or constant-sustain annotations are manufactured.
Seventeen targeted acquisition, supervision and codec-padding tests passed.

The initial preparation stopped after 39 clips on a 30-sample MP3 end-padding
difference. Its source, plan and log hash are preserved. Alignment was exactly
zero and decoded padding was exactly silent. The new v2 preparation canonicalizes
only proven silent far-end padding and preserves the raw codec waveform. It
rejects nonzero alignment or any non-silent padding adjustment. This repair
changes no musical samples, training partitions or independent evaluation data.

The completed preparation retained 121 zero-prediction clips and 2,149 candidate
notes. All measured codec lags were zero. Seventy-nine clips needed only verified
silent end-padding repairs. The fitting audit found no source-group overlap;
60,750 fundamental frames have UNKNOWN timing and remain masked. See
preparation-completion.json and the V45 data-audit.json witness.
