# Fixed later MusicNet fitting passages: symbolic assessment

The current real-piano fitting corpus uses three 20-second excerpts at 15, 35
and 55 seconds from each of 21 fitting performances. This audit inspects only
those training works' verified metadata and publisher note-label CSV files.
It proposes fixed later passages at 95, 155 and 215 seconds where complete
20-second waveforms fit the original duration, with no existing excerpt overlap.

Result: 49 proposed excerpts from 20 of the 21 audited fitting works, with
6,196 complete publisher note intervals and 302 boundary-crossing intervals.
Crossing intervals are pitch-support-only; no new attack or release is implied.
No proposal is chosen using transcription outcomes or held-out labels.

The [original publisher record](https://zenodo.org/records/5120004) grants
CC BY 4.0; cached publisher metadata and the verified original 11,097,394,998-byte
archive witness are checked against their recorded hashes. No source audio was
read or acquired, no model inference occurred, and validation/reserved/consumed
labels were not read. Original archive reacquisition, waveform clocks, MP3
alignment and strict preparation audits remain necessary before fitting.
The publisher labels do not establish sustain-pedal or acoustic release truth.

All 13 focused source/clock tests pass; the full current suite passes 1,415
tests with six skips and five warnings. This assessment does not modify V58
fitting, selection or evaluation and is not a training clearance.
