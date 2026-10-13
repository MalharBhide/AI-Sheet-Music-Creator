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
