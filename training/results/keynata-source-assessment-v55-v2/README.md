# Keynata Commons symbolic assessment V2

A separate tested reader removes only neutral pitchwheel-0 and program-0
resets at absolute tick zero. The historical strict PDMX parser remains
unchanged and validates every remaining musical event and tempo. Tests require
bit-identical key clocks after neutral resets, and reject nonzero/late pitch
changes, other programs, pedal and overlapping same-key events.

Five of the 19 fixed solo-piano sources pass. Fourteen are excluded in full for
ambiguous same-key overlaps; notes are not merged, shortened or silently lost.
The 3,047 eligible key commands are generated compositions, not model-transcribed
labels or human piano timing evidence. Displayed form lengths never rescale
source note clocks. Source metadata and MIDI hashes are independently checked
against the pinned publisher tree and per-track SHA-256 declarations.

No audio was acquired or inferred. Native preparation and partition audits are
still required before fitting; no new source enters validation or consumed/fresh
tests. V54 uses only its already frozen data and is unaffected by this assessment.

Publisher: https://carf-coder.github.io/keynata-commons/
Pinned source: https://github.com/carf-coder/keynata-commons/tree/f4b3d0582c7a1ecb725634383ccd41379d2587a2
