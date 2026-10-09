# Completed real-piano V33 study — withheld

The first fit using newly acquired MusicNet solo-piano excerpts completed two
predeclared profiles, 80 epochs each, on 765 training and 221 validation fixtures.
This adds 63 real-piano training excerpts and 15 real-piano validation excerpts
to the 908 existing licensed fitting fixtures. All released V32 weights and
normalization stayed frozen; only a neutral additional correction branch learned.

Neither profile produced a selected checkpoint. Some early settings removed one
validation false note at normal confidence, but none improved both normal and
stricter confidence while preserving every recording. Decreasing training loss
is not accepted as evidence of a better transcription model.

No consumed regression or first-pass evaluation ran. All 18 reserved MusicNet
excerpts and the 32 declared authored first-pass seeds remain untouched. All eight
Oxford piano/MIDI pairs also remain reserved. No trained weights are deployed.
The website continues using the approved V32 verifier and original engraving.
The next study should expose the full frozen local/coarse/release encodings to
the new branch, rather than only the teacher's compressed 32-dimensional output.
This is an untested next direction, not a claim of improvement.

Training checkpoints, physical caches and licensed audio remain local and
excluded from Git. Complete JSON plans, learning curves, search results and
checkpoint hashes are archived here. No user uploads or saved scores were used.
