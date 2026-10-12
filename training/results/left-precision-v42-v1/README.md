# Frame-presence support V42 — withheld at validation

Two independent 100-epoch profiles train a spectral/periodic rejection head and
a 40-frame pitch-presence head. A supplied sub-frame continuation cannot be
silently labeled absent. Both heads start protected; input features contain no
reference labels. The minimum of note rejection and complementary maximum
central presence determines removal confidence.

The fitting set has 869 training and 249 validation excerpts. It includes the
88/24 authored NSynth sequences and newly measured recurrence on the existing
16/4 public-domain Mutopia excerpts. Original decoder events, labels and work
partitions are preserved; reserved recordings remain outside fitting.

Neither profile qualifies. Preserving settings remove at most one extra note,
with zero real-piano benefit. The small profile's more active final setting
removes six extras but harms a supplied note. No consumed tests, fresh pass,
export or website deployment follows. Seeds 287101–287140 remain untouched.

Plans, six checkpoint hashes, learning curves, searches, per-recording
diagnostics, exact executed source witnesses and data provenance are preserved.
The suite passes 968 tests, with six optional native checks skipped. Audio,
features and weights stay local. No user uploads or saved scores were processed;
the website retains its approved V38 model and original note size.
