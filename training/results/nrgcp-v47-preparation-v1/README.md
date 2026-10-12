# NRG-CP V47 sampled-piano preparation

Sixty licensed NRG-CP MIDI files were selected from the verified publisher archive
before any rendering or inference: 48 training progression families and 12
validation families. Entire ambiguous same-key-overlap clips are excluded. All
three variations of a source stay together. Exact transposition patterns are
also checked across partitions. No reserved MIDI is extracted or rendered.

The fixed variations use 90/120/150 BPM, velocity multipliers 0.45/0.7/1, and
whole-phrase octave shifts constrained to the piano range. No notes are dropped.
Direct offline FluidSynth rendering uses hash-pinned MuseScore General samples;
the complete MIT/PD/CC0 notice and source attribution are retained here. The
rendering library and samples are training dependencies, not website changes.

Key commands are explicitly rounded up to the 64-frame synthesis grid (under
2.91ms at 22,050Hz). Chords share command boundaries; repeated keys retain their
individual attacks and offsets. Per-pitch external 300ms damping gives known
synthetic support, protected conservatively through another 100ms. These are
**authored synthetic key clocks, not human timing or pedal labels**. MIDI is
never routed through notation import. The same sampled instrument is shared
across partitions: this is progression-held-out data, not instrument-held-out
evidence. Independent real-piano validation remains mandatory.

Every MP3 must have zero measured waveform lag. Only proven silent final codec
padding can be adjusted under the pinned existing codec policy. Rendered,
raw-decoded and canonical waveforms, MP3s, labels and candidate features have
separate hashes. The strict loader recomputes source transformations, references,
partitions and physical context. No user upload or saved score is processed.

Status: all 180 clips are prepared and the strict physical/label audit passed.
Training contains 144 clips from 48 families: 11,691 candidates (9,679 conservative
KEEP, 2,012 REMOVE). Validation contains 36 clips from 12 families: 3,257
candidates (2,744 KEEP, 513 REMOVE). Nine training and three validation clips
produced no candidates and remain in the corpus. All codec lags are zero; no
padding repairs were needed. The 6,939 authored chord attack groups and 25,824
same-pitch recurrences provide new rhythmic supervision. There are 166 key holds
over one second, none over four seconds; earlier corpora still provide longer
holds and human performance evidence. This audit is not an accuracy result.
Twenty-seven source/render/preparation checks and the full 1,051-test suite passed
(six full-suite skips). Both 100-epoch V47 profiles completed without a validation winner; their weights
are withheld. See the separate V47 study archive for measured results. The
original framework commit checkpoint preceded those runs.
The planned V47 run uses all earlier fitting partitions plus these new clips,
with immutable V43 rejection permissions and unchanged V38 timing context. A
winner must preserve every measured recording, improve real piano and these new
progressions at both policies, preserve held-out TinySOL fundamentals, then pass
603 consumed recordings, one untouched fresh first pass, and native/runtime checks.
