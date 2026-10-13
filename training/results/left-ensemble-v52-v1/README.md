# V52 joint trained permissions

This study is preregistered while V51 completes. It tests whether independently
trained high-precision members can jointly remove the different extra notes each
recognizes. It does not fit validation labels or add a corpus-dependent rule.

Select one preserving V48 member prioritizing its minimum isolated-pitch gain,
and one preserving V51 member prioritizing its minimum real-piano gain. The exact
lexicographic rules and source-plan hashes are frozen in `preregistration.json`.
Every considered member must preserve all timed and pitch-only recordings under
both policies. No alternate source or runner-up is substituted.

The fixed pair requests removal when either learned member gives its predeclared
high-confidence rejection with absent frame support. Its output still intersects
immutable V43 permissions; V38 retains complete input context. Genuine-note,
pitch, hold, chord, repeat and timing checks must be evaluated on the combined
result. Two individually preserving members do not prove a preserving union.

The one pair must supply all separate positive domain gains at both policies,
then pass all 603 consumed regressions, one untouched 40-clip first pass with
seeds 297101–297140, portable runtime parity and native functional checks. No
runner-up, retuning or replacement clips after failure. Website weights remain
unchanged; user recordings are never read or transcribed.

The original preregistration proposed a .5 threshold for binary combined output.
The existing evaluator accepts .8–1. The preserved protocol amendment uses .9;
binary decisions are identical throughout the supported interval. A regression
check proves this with eligible and ineligible notes. No member threshold, choice
rule or release gate changed; no joint predictions preceded the correction.

Status: composition/selection helpers implemented; model selection, loading and
joint evaluation remain pending. No joint predictions, consumed tests or fresh
tests have been read.

Composition preflight: nine focused checks and the full 1,102-test suite pass
with six skips. Joint model loading and validation remain pending.

The complete validation runner and frozen-source contract pass 1,113 tests with
six skips. Both exact preregistered choices load with their own verified
checkpoints and normalizers. Joint validation is ready to run once; source
calibration, complete population, per-recording preservation and aggregate gain
receipts are checked before any consumed or fresh test access.

The one fixed pair qualifies on complete validation: 38/24 extra timed notes
removed (normal/stricter), including one real-piano note under each policy and
four/two NRG-CP notes. One extra isolated-pitch note is removed under each policy.
All 381 timed and 237 pitch-only recordings preserve every required identity,
pitch, hold, chord and timing measure; no notes are restored. The pair is frozen
before any consumed or fresh test access. This is validation-only evidence;
website weights have not changed and all later release gates remain pending.

Consumed/fresh phase tooling passes the full 1,122-test suite with six skips.
The qualifying pair was reverified against its frozen validation receipt before
any test access. All 603 consumed regressions are next; fresh inference remains
blocked until they pass at both policies with positive benefit.

Final status: withheld after consumed regression. The fixed pair removed 71/60
extra notes overall under normal/stricter policies, but one of 603 recordings
loses a genuine reference note and fails preservation at both policies. Aggregate
gain cannot override that failure. No runner-up or retuning is attempted. The
untouched 40-clip first pass was not generated or read. Website weights remain
on V43. This study cannot be presented as a qualified release.
