# V48 boundary-conditioned frame support

Two fresh 100-epoch profiles use the same audited V47 fitting partitions: 2,077
training recordings, 381 timed validation recordings and 237 pitch-only held-out
TinySOL recordings. The frame-support head adds separate raw attack/release CNNs,
with average/max pooling conditioning the 40 CQT frames alongside physical
context. Input observations and supervised targets are unchanged. There are no
absolute-pitch, source-ID, teacher or reference-label inputs at inference.

V43 rejection permissions remain immutable. V38 computes release on complete
context before deletion; retained events must remain identical. Every recording
must preserve genuine attacks, holds, pitch coverage, matched identities,
fixed-pair/repeated timing, chord spacing and unsupported duration. Both policies
must improve overall false notes, real piano, NRG-CP and TinySOL wrong pitches,
while retaining all baseline fundamental candidates. These gates are unchanged.

All 144 settings failed the combined requirements; both profiles and all their
weights are withheld. The immutable guard prevented every note restoration.
Some settings preserved measured notes and reduced general false detections;
none met every domain-specific gain requirement at both policies. Per-domain
maxima in validation-summary.json may come from different settings and are not
accuracy or release claims. No consumed/fresh tests were read, no user uploads
or saved scores were processed, and no website weights changed.

The full suite passed 1,055 tests with six skips, including conservative initial
permissions, exact checkpoint roundtrip, and learning paths from both boundary
windows. This does not substitute for native/runtime release checks. Checkpoints
and audio remain local; plans, curves, diagnostics and source witnesses are
archived for comparison. The new-data audit describes authored synthetic key
clocks, not measured human timing; independent real-piano gain remains required.
