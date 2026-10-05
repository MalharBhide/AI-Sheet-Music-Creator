# Bass hold and repeat training against V10

This historical experiment used the [V10 bass verifier](bass-verifier-v10.md).
The website now includes the subsequently trained [V11 articulation release](bass-articulation-v11.md).
This earlier offline experiment addressed a different error: the detector can split
one held bass note into multiple audible attacks. Deleting those fragments would
shorten the hold, so the experiment learns when touching fragments can be joined.
It does not change tempo, quantization, the melody detector or user scores.

`current_bass_baseline.py` checks the two published V10 array hashes, feature
contract and `.1/.1` thresholds. It reproduces the deployed keep mask on existing
cached bass candidates. Previously rejected notes never return. Both events in a
candidate pair must be retained, in the tested interior window and adjacent among
retained events of the same pitch. Their overlap is at most 30 ms, their start
spacing is 150 ms–2 seconds, and there is **no positive gap**. Real rests remain.

Labels come from supplied key-attack and pitch-support annotations. A key attack
within 50 ms is a genuine repeat, including repeats under a pedal hold. Attacks
50–150 ms away remain ambiguous. A negative label requires a continuous annotated
pitch hold covering both fragments; unsupported tails remain ambiguous. Labels,
absolute pitch, filenames and recording identities never enter model features.

The context head receives two 52-feature note evidence vectors, their pinned V10
head probabilities and four observed timing measurements: 112 features. The
acoustic guardian receives two 26-feature base vectors with the same probabilities
and timing: 60 features. Either head can veto merging. Merging keeps the earliest
attack/velocity and joins only touching intervals of the same pitch. No new pitch
or time coverage is created, and no existing pitch coverage can disappear.

The dataset is unchanged from the V10 training release: 359 training clips and
94 validation clips, using commercially compatible BabySlakh/GuitarSet sources
and original signals. Source groups stay separated. The supervision audit found
7,042 labeled training pairs (5,813 genuine repeats, 1,229 split holds), and
1,385 validation pairs (1,053 repeats, 332 split holds). Another 1,775 training
and 633 validation pairs are ambiguous and excluded from supervised fitting.
The small synthesized corpus and shared compositions/upstream pretraining limit
any generalization claim; these are not independent commercial-song benchmarks.

Two predeclared model pairs use 350/500 boosting trees, genuine-repeat weights
30/60, learning rate `.04`, L2 regularization `8`, fixed random seeds, and no
automatic early stopping. Context heads have at most 15 leaves; guardians have
at most 7. The earlier 100/200/350/500 checkpoints compete on validation only.
The threshold grid and half-margin rule are frozen before training.

Every candidate must preserve **each previously matched attack and offset in
each recording**, supplied pitch support, and the exact entire detected
pitch-time union. False-note counts cannot increase in any recording. Validation
must improve before selecting a checkpoint. Later stages cannot replace an
earlier checkpoint merely because they trained longer. Ties use lower validation
loss, fewer trees and fixed profile order.

The original eight-case stress corpus and previously tested twenty BabySlakh
clips are now **consumed regression**, never fitting/selection or fresh evidence.
The winner, checkpoint, code, manifests and validation report are hash-bound
before a one-shot regression check. A failed winner cannot fall back to a
runner-up or be retuned against those tests. Deployment additionally needs
independent evidence, portable parity and native runtime checks.

The first run was interrupted by a mechanical checkpoint-helper incompatibility:
the reused detector helper rejected the boundary plan's 350-tree stage. No batch
winner or test was produced. Its plan, learning curves and completed searches are
archived under `training/results/bass-boundaries-v1-incomplete`. The repaired
boundary-specific helper preserves the same profiles/stages and is covered by a
regression test. The completed rerun is `.training/bass-boundaries-v1b`. Its
500-tree recall winner removed ten validation false attacks (2,888 → 2,878), with
4,150 matched attacks unchanged and all individual gates passing. The 28 consumed
regression clips also passed, with no additional reduction (567 matched attacks,
1,060 false notes unchanged).

A one-shot check used 32 new original note/timbre seeds, 261201–261232, distinct
from fitting and earlier stress. They were rendered with the same procedural
generator as training, encoded as 48/128/192 kbps MP3, and decoded on the verified
zero-lag clock. This is unseen-seed stress within a synthesized distribution,
not an independent human-song benchmark. The candidate reduced false attacks
199 → 193 with 615 matched attacks unchanged, but lost a previously correct key
offset in one recording. Exact pitch-time coverage and aggregate attack recall
alone would have missed this regression. **The winner is withheld.** No runner-up,
threshold adjustment, portable export or deployment followed. All 32 recordings
are now consumed regression and cannot enter fitting or be called fresh again.

Results are archived under `training/results/bass-boundaries-v1b` and
`training/results/bass-boundary-stress-v1`. All 376 backend/training tests pass
(six opt-in native integration cases skipped here; already passed for unchanged
V10). This includes eighteen framework and three first-pass safeguard tests.
Ruff and diff checks pass. V10 remains deployed.

The next training hypothesis is to protect key releases explicitly in boundary
supervision. Pitch support can continue under pedal after a key release, so
continuity alone is insufficient to label a boundary safe to remove. Any next
candidate must learn this from fitting annotations, without fitting the failed
test fixture, and preserve the failed case in consumed regression.

The next framework is now saved in `bass_articulation_labels.py`,
`train_bass_articulation.py` and `bass_articulation_release.py`. It labels a
boundary protected whenever joining the observed pair would lose a matched key
attack or offset from fitting annotations. Inference features, model profiles,
seeds, thresholds and validation gates stay the same. It keeps the historical
experiment code intact, binds its own label policy and code, and includes the
32 newly consumed original fixtures only in regression. This new framework is
**not trained or deployed**: the user requested a pause to test the released V10
model first. Do not call its unfitted code a new model release.

On a later authorized training run, use a new output directory:

```bash
python training/train_bass_articulation.py .training/bass-articulation-v2 \
  .training/slakh-bass-v1 .training/bass-positive-data-v1 \
  --regression .training/bass-held-regression-v1 \
  --regression .training/slakh-bass-positive-reserved-v2 \
  --regression .training/bass-boundary-stress-v1
python training/bass_articulation_release.py .training/bass-articulation-v2
```

Passing consumed regression is not sufficient for release; a next winner still
needs independently planned first-pass evidence, portable export/parity and
native production checks. No fitting started before the pause.

```bash
python training/train_bass_boundaries.py .training/bass-boundaries-v1b \
  .training/slakh-bass-v1 .training/bass-positive-data-v1 \
  --regression .training/bass-held-regression-v1 \
  --regression .training/slakh-bass-positive-reserved-v2
python training/bass_boundary_release.py .training/bass-boundaries-v1b
python training/prepare_bass_boundary_stress.py .training/bass-boundaries-v1b \
  .training/bass-boundary-stress-v1
python training/prepare_bass_boundary_stress.py .training/bass-boundaries-v1b \
  .training/bass-boundary-stress-v1 --evaluate
```

The frozen release command requires a positive validation winner. All audio
inference here is from dataset preparation, never user uploads. This experiment
uses cached features and generates no transcription jobs or sheet music.
