# CocoChorales bounded acquisition — stopped at wrap-up request

Eighteen sources were frozen from metadata before acquisition: 12 train and six
validation, with distinct original MIDI identities. Four original objects,
10,227,590,480 compressed bytes in total, are pinned by publisher MD5, Google
Storage generation, ETag, byte count and the immutable local producer hash.
The reader streams complete checksums and decompression footers and retains only
selected WAV/MIDI/YAML/CSV members. Full archives are not stored.

Both complete note-expression packages verified successfully, retaining 72 CSV
files. The two main audio packages were still streaming when the user requested
wrapping up with limited remaining usage. Both workers were explicitly stopped
and returned terminal exit code 143. Partial audio remains local and **unverified**.
No main-object completion witnesses exist. The actual source reader refuses this
state before observation, so these examples cannot enter model training.

No CocoChorales audio was inferred, no weights were trained on it, and no user
uploads or saved scores were processed. The website retains the approved V32
model. Existing score engraving and playback remain unchanged.

Seventeen tests cover complete-stream checks, bad footers/checksums, unsafe and
duplicate paths, immutable grouping, synthesis-frame timing, repeated attacks,
clipped holds and decay protection. The actual interrupted-source refusal is
also recorded. Final complete-suite results are stored separately here.

Original synthesis timing uses 250 Hz CSV frames. Upstream pitch correction is
an intonation coefficient, not a semitone shift; stem and CSV voice numbering
also differ. These semantics must be respected in any future observation study.
The corpus is synthesized chamber audio, not a real-piano benchmark. Future
release still requires every consumed preservation gate and independent piano
benefit. No training or download work is left running after this wrap-up.

Retained audio, labels and partial objects remain outside Git. Resuming later
must complete original audio-object verification; incomplete main streams must
be downloaded again, while completed note-expression witnesses can be reused.
The publisher declares CC BY 4.0; attribution and selection are retained in the
plan and preceding source-assessment archive.
