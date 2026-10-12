# Independent frame-presence calibration V43 — runtime validation pending

Two fresh 100-epoch profiles fit 869 excerpts and validate on 249. This study
requires V42 to have stopped before consumed regression; it does not retune a
consumed-test failure. Rejection and presence thresholds are independently
predeclared and selected on validation only.

The frozen winner is calibrated-small, epoch 50, rejection .9 and presence veto
.2. The stricter policy is .905 and .16. Its checkpoint SHA256 is
22f1f9bcf5332e470d289579a3243f7ae215da9e1752f72060fdc3bcad8faac9.

Normal / stricter false-note reductions:
- Validation: 42 / 32, including 1 / 1 real-piano extra notes.
- All 563 consumed regressions: 109 / 72.
- Forty independent first-pass MP3 examples: 18 / 14, including 1 / 1 in the
  original family and 17 / 13 in the texture family.

Both policies preserve every tested excerpt's required attacks, supplied pitch
support, holds and retained clocks. The independent seeds 288101–288140 are now
consumed and must join any future regression gate. These limited dataset results
do not establish accuracy on arbitrary songs or user uploads.

No weights are deployed. Export, native inference parity, complete runtime
composition and website integration remain required. The website keeps V38.
The suite passes 971 tests with six optional native checks skipped. All six
checkpoint hashes, plans, curves, source witnesses, diagnostics and independent
manifest are archived. Weights/audio remain local. No user upload or saved score
was processed, and the original engraving size is unchanged.
