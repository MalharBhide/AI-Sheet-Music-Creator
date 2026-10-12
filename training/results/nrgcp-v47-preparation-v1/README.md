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

Status at this framework checkpoint: preparation is running toward 144 training
and 36 validation clips. It is not yet a completed data audit or accuracy result.
Twenty-seven source/render/preparation checks and the full 1,051-test suite passed
(six full-suite skips). No new weights are trained or deployed at this checkpoint.
The planned V47 run uses all earlier fitting partitions plus these new clips,
with immutable V43 rejection permissions and unchanged V38 timing context. A
winner must preserve every measured recording, improve real piano and these new
progressions at both policies, preserve held-out TinySOL fundamentals, then pass
603 consumed recordings, one untouched fresh first pass, and native/runtime checks.
