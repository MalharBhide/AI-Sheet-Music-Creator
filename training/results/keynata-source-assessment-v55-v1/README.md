# Keynata Commons symbolic source assessment V1

The pinned public-domain library contains 19 solo-piano MIDI/metadata pairs.
Only those commands and metadata were acquired, with official Git blob and
per-track SHA-256 authentication. No audio was downloaded or inferred.

The unchanged strict historical PDMX parser rejects all 19 because they contain
pitchwheel messages. The inspection finds neutral pitchwheel resets at tick zero
in the first file; this does not yet establish that every source is usable. This
failed audit remains frozen. A separate parser/audit may accept demonstrably
neutral initial events, but must reject real pitch changes or ambiguous clocks.
No source is admitted to training and no weights are deployed.

Publisher: https://carf-coder.github.io/keynata-commons/
Pinned repository: https://github.com/carf-coder/keynata-commons/tree/f4b3d0582c7a1ecb725634383ccd41379d2587a2
