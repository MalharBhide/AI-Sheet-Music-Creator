# New CC0 Keynata piano fitting preparation

The five pre-audited generated piano compositions supply 72 fixed 24-second
windows and dynamics derivatives, entirely fitting-only. Fourteen sources with
same-key overlap remain excluded in full. All native waveform crops must render
before transcriber inference; actual preparation must reproduce their float32
sample values exactly. Complete continuous work rendering preserves notes across
crops without manufacturing attacks or releases. MP3 lag must remain zero.

Publisher grants CC0-1.0 for generated content; the source license and pinned
metadata were archived in V55. These compositions are not human-performance or
independent piano-timbre accuracy evidence. Positive command-support frames are
masked as UNKNOWN during fitting because natural sample decay is not a held
acoustic note. Existing real-piano, held-out instrument and timed gates remain.

Status: preparation preregistered after 1,305 passing tests, six skips and five
warnings. Native preflight, observation preparation and strict audit remain
pending. No user audio, saved score, consumed or fresh test is transcribed.

Native preflight completed: all 72 fixed waveform crops rendered before any
transcriber inference. Native synthesis and complete crop-clock witnesses are
archived in `native-preflight.json`; strict preparation is next.

Preparation and strict audit completed: all 72 clips from five fitting-only
parents reproduce native float32 samples exactly, with zero measured MP3 lag.
The caches contain 4,968 candidate notes, including 3,032 supervised KEEP and
735 supervised REMOVE examples. All 143,812 positive command-support frames
are masked as UNKNOWN; 54,908 absent frames remain supervised. No held-out or
user audio was inferred. This licenses fitting, not a production release.
