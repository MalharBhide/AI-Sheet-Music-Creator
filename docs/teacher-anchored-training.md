# Bounded correction around the approved acoustic model

Earlier fully fine-tuned candidates removed additional wrong notes but repeatedly
failed a strict continuation-coverage regression. They remain rejected. The
website still uses the approved V25 weights; no saved scores are regenerated.

V30 keeps all approved acoustic encoder and classifier parameters frozen. Its
teacher dropout stays disabled even during training, and its original 84-feature
context normalization is unchanged. A separate 71-feature MLP learns from
waveform recurrence and observed same-pitch release neighbors. It receives no
reference intervals, filenames, absolute pitch, fitted classifier probabilities,
or corpus identities.

The branch adds `cap * tanh(output)` to the teacher's REMOVE logit. The two
predeclared profiles use widths 64/96 and caps .25/.5. The correction therefore
has a strict bound regardless of the learned branch weights. Zero-initialized
last-layer weights/biases preserve the teacher's initial probabilities exactly.
Tests use nonzero random teacher weights, optimizer steps, training/evaluation
mode transitions, and saturated positive/negative corrections to verify initial
parity, frozen teacher state, disabled dropout and the declared bounds.

The [same sealed fitting data](partial-support-drum-training.md) supplies 702
training and 206 validation recordings. Binary KEEP/REMOVE supervision retains
all existing attack, hold and pitch-continuation protections, including weighted
partial fragments. Only training observations normalize the 71 new features.
Both profiles train for 80 epochs, with checkpoints at 20/40/60/80. Confidence
grids `.8/.85/.9/.95/.99` and unchanged guardian grids `.15/.3/.5` are declared
before fitting. Validation alone selects one checkpoint and confidence pair.
Every recorded checkpoint must still contain the exact approved teacher weights.

Both confidence settings must remove additional wrong notes while preserving
every validation recording's attacks, holds, pitch coverage and fixed-pair
timing. Retained event clocks, pitches and velocities are unchanged. All 392
earlier regression recordings must then pass before the predeclared 77-record
first pass can use four reserved NSynth instruments, two reserved GMD performers,
and new authored seeds 275101–275132. No runner-up, threshold switch, replacement
fixtures or test-based retuning is allowed after a failed frozen evaluation.

The bounded logit change is a training safeguard, not proof of improved accuracy.
Portable/runtime parity and native physical pipeline checks remain mandatory
before any release. Both profiles completed 80 epochs. All searched settings preserve the 206
validation recordings, but none reduces wrong notes at both declared confidence
settings. **No validation winner is selected.** This study does not reach
regression evaluation. The actual first-pass tool refuses the study before
creating an output directory. All reserved sources and seeds 275101–275132
remain unconsumed. Six focused architecture/guard tests and all 791
backend/training tests pass (six unchanged-production native checks skipped).
[Learning curves, searches and checkpoint hashes](../training/results/teacher-anchor-v30-v1)
are archived. No V30 arrays are exported or deployed, and no user audio is
processed.
