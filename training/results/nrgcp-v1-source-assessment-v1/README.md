# NRG-CP v1 — new chord/rhythm source assessment

[Patchbanks / WaivOps](https://zenodo.org/records/15304989) publishes NRG-CP under
CC BY 4.0. Its [repository](https://github.com/patchbanks/WaivOps-NRG-CP) confirms
the dataset license and contributor-clearance statement. Retain attribution to
WaivOps, Patchbanks and DOI 10.5281/zenodo.15304989, identify changes and link the
license when using derivatives.

The 9,774,387-byte archive passed its publisher MD5. Inspection found all 30,943
MIDI files, with 96 ticks/quarter, 120 BPM and 4/4 meter. These are symbolic chord
progressions; they do not include human piano audio or human note-duration labels.
No audio has been rendered, inferred or used for training.

The full note-pair audit found 1,426 clips with ambiguous overlapping same-key
attacks. Exclude these entire clips before fitting; guessing note-off pairings
could teach exactly the repeated-note/hold mistakes this project needs to avoid.
The observed pitch range is 19–78, including notes below the piano range. Preserve
or explicitly transpose whole phrases; do not silently drop low notes.

The archive has 30,943 unique byte sequences but only 30,477 transposition-invariant
note patterns and 4,239 coarse harmonic-progression families. Related progressions
need whole-family partitions before audio generation. Byte hashes or filenames
alone do not establish independent train/validation splits. The bar pitch-class
signature in this assessment is a coarse candidate grouping, not a finalized
split contract or a guarantee against every form of musical similarity.

Actual MIDI endings range from 2,689 to 3,073 ticks; rendering must use observed
clocks rather than forcing every clip to exactly eight bars. Any future piano
rendering supplies authored synthetic key clocks, not measured human performance.
The remaining acquisition, partition and validation requirements are recorded in
assessment.json. No website weights or user recordings were changed.
