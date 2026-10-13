# Corrected PDMX piano preparation

The isolated v2 preparation fixes only the dynamics-list serialization check.
The original 45-clip v1 preparation and exact source witnesses remain preserved;
no fitting used v1. No old source/hash guard was relaxed, and v2 uses separate
native rendering and dataset-only MP3 inference. The same two CC0 solo-piano
sources, actual tempo maps, fixed windows, dynamics and codec policy apply.
The source, licensing, crop-clock and composer partition rationale is recorded
in ../pdmx-piano-v53-preparation-v1/README.md .

The full suite passes 1,163 tests, six opt-in skips and five dependency warnings.
V2 is preparing 45 training-only observations. No user audio, consumed or fresh
test inference is performed; website model weights remain unchanged.
