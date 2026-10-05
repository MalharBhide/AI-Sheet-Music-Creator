# V19 timing model: finer acoustic attack evidence

Two dual-scale temporal CNNs were trained on the actual V16 bass survivors.
The validation winner improves measured attack timing and removes unsupported
notes, but fails two consumed regression recordings. **No V19 weights are
deployed.** The website retains V16's released bass and vocal models; chord
engraving is an independently verified notation change.

## Features, data and fitting

The fitting baseline remains 573 training and 160 validation recordings from
the previously audited commercial-compatible data. All 296 regression recordings
are excluded from feature preparation, label fitting, normalization and selection.
The dataset's rights and attribution follow the
[V16 notice](../backend/app/assets/bass-temporal-refinement-v1.LICENSE.txt).
No user recording or saved score is used or regenerated.

The earlier note-wide 40-step CQT transform spans each note's entire duration,
so a long hold leaves relatively sparse samples around its attack. The new
branch adds 61 positions centered on the predicted attack, independent of its
endpoint. Its clock is 128 samples at 22,050 Hz (about 5.8 ms between positions).
This sampling interval is not a claim of 5.8 ms transcription accuracy.

The 18 observed channels contain nine pitch-relative CQT bands, four harmonic
bands from each of short/long STFT windows (1,024/4,096 samples), and short-window
positive spectral flux. The existing CQT clock remains 256 samples; interpolation
does not create additional physical CQT resolution. Each recording's observed
amplitude supplies normalization; reference notes and learned scores never enter
inference features. Silence stays zero, and short clips retain their true duration.

Both profiles reuse the released width-48 coarse encoder and hidden weights,
with a new width-24/32 attack branch and four-class output. Initialization
preserves the coarse hidden representation within floating-point tolerance.
Training uses 60 epochs, 40 updates per epoch, batch size 256 and fixed checkpoint
stages 20/40/60. Context normalization and all gradients use the 30,492 eligible
supervised training events. Corpus/recording weighting, labels and quality gates
are unchanged from V18.

Validation alone freezes `local-attacks` epoch 60 at timing confidence .99,
removal confidence .95, and the inherited acoustic guardian .3. Checkpoint SHA256:
`7949d4e280c7b54025c203b46e5d0171b16b457eacb92d6dedcfc8d56209d4f2`.
The runner-up is not tested as a replacement.

## Measured results and rejection

Timing uses fixed baseline same-pitch reference pairs within 200 ms; deletion
earns no timing credit. Every recording must retain matched attacks, offset-aware
holds and supplied pitch coverage, add no false positives, and avoid increasing
onset, release or repeated-key spacing error. Both normal and stricter-confidence,
half-strength corrections must pass with positive onset improvement.

| Evaluation | Correction | Retimed notes | Onset error reduction (seconds, summed) | Fewer false notes | Preservation/timing gates |
| --- | --- | ---: | ---: | ---: | --- |
| 160 validation recordings, 6,008 fixed pairs | Normal | 32 | .46 | 14 | Pass |
| Same validation | Stricter/half strength | 19 | .15 | 5 | Pass |
| 296 consumed regressions, 6,243 fixed pairs | Normal | 50 | .92 | 19 | Fail |
| Same regressions | Stricter/half strength | 39 | .37 | 8 | Pass |

All regression recordings retain matched attacks, matched holds and reference
pitch coverage under both corrections. However, normal corrections increase
repeated-key spacing error by .0092 seconds on `bass-held-v1-low` and release
error by .02 seconds on `bass-articulation-stress-v2-261314`. These individual
timing regressions reject the candidate despite its aggregate improvement.

The safer correction's result cannot be used to replace the frozen settings
after seeing regression results. No thresholds, strengths or checkpoints are
retuned, and no replacement candidate is evaluated. First-pass seeds
264101–264132 remain unused. No fresh MP3 inference, portable model export or
ML deployment follows this rejection.

Committed [feature evidence](../training/results/attack-local-bass-v1) and
[V19 results](../training/results/attack-local-bass-v19-v1) preserve plans,
source hashes, validation searches, per-recording regression results and local
checkpoint hashes. Feature caches and six fitted checkpoints remain local.
The source-group/composition limitations of the older data still apply; these
are development results rather than arbitrary-song accuracy claims.

The next training experiment needs joint phrase timing and release handling,
so improving a single attack does not distort a repeated pattern or neighboring
hold. The failed cases remain regression-only examples, never fitting data.
