# Project unsafe support updates before nonlinear validation

V70 keeps V69's approved initialization, 3265 existing support parameters, immutable rejection classifier, inference thresholds, 61 fitting-only veto-dependent protection candidates, and protected-example sampling. Earlier full-support candidates made little progress because backtracking shrank whole parameter updates when a few protected scores crossed their margins.

For an unsafe update, V70 computes the protected scores' Jacobian at the previous accepted weights in independent batches of at most sixteen. A small nonnegative dual optimization projects the proposed update onto the resulting local halfspaces. Solver success alone is never trusted: linear residuals are explicitly checked and repaired, with retreat toward the feasible zero step if necessary. The actual nonlinear network is then checked on every protected fitting candidate. Existing backtracking and exact rollback remain the final safeguard. No validation input or label participates in this optimizer operation.

Both fixed profiles use the original dataset and partitions: 2302 training and 618 validation recordings. Every original preservation and positive-family-gain requirement remains mandatory at both inference policies. Only one full validation winner across V66–V70 may access consumed regressions and fresh tests; no runner-up after failure, and full native/runtime parity before deployment.

All four protocols were registered before fitting. Full tests: 2258 passed, 7 skipped, 5 warnings. No new audio was downloaded, and no user upload or saved score was processed. Training result pending; no accuracy improvement or deployment claimed.
