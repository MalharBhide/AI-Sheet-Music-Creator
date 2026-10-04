# V7 left-hand consensus experiments

See [model card](../../../docs/left-hand-consensus-v7.md) for methods, licenses,
measurements, limitations and reproduction.

- `single`: rejected single-head fit; one correct matched piano note lost in consumed regression.
- `consensus-shared-threshold`: two trained heads, equal thresholds; no validation gain at safety margin, never tested.
- `release`: same two heads, separate validation-only calibration; frozen before regression. All release gates passed.

Regression is consumed. The eight later piano passages are now consumed too and
must enter future regression. Raw media/features/research pickle files stay in
ignored `.training`; production ships checked data-only NPZ arrays.
