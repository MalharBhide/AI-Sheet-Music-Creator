# Independent frame-presence calibration V43 — released

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

The frozen winner is now deployed after parity across 1,721 excerpts / 72,256
notes, physical-feature checks on 40 MP3 fixtures, native routing of all 12
fresh deletion cases plus two unchanged controls, 986 staged tests, and all six
native integration checks. Runtime probabilities match exactly; maximum
physical feature error is 5.960464477539063e-08. The live image, model hash,
confidence thresholds, frontend proxy and dependencies are verified.

The release asset is backend/app/assets/left-presence-v1.npz, SHA256
38899dd8dbbefafb3a2d5d2f210dc2e6db66087ffc6ee95fcae2635f5d9bbfda.
All prior assets and original engraving size are unchanged. The approved V38
image and source tree are retained locally for rollback. No user upload or saved
score was processed. Training checkpoints and audio stay local; only the
validated CC BY 4.0 release arrays are shipped.
