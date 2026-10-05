# V11 bass articulation release

V11 retains the V10 bass note verifier and V9 accompaniment processing, and adds
a trained bass hold/repeat classifier. It targets false repeated attacks within
a held bass note. It does not transpose notes, invent missing pitches, alter
tempo or retrain the primary Basic Pitch/piano/vocal neural detectors.

The earlier boundary winner was withheld because it erased a correctly detected
key release despite improving aggregate accuracy. The new fitting labels protect
both actual key attacks and previously matched key offsets. Continuous pedal
pitch support cannot override an observed key-release boundary. All features
remain label-free. The failed fixtures stay outside fitting/selection as consumed
regression, and the historical code/weights were not retuned or overwritten.

Training used 359 clips and validation used 94, with the existing licensed
BabySlakh/GuitarSet split and original procedural signals. Of 8,817 eligible
training boundaries, 7,220 were supervised: 5,991 protected attacks/releases and
1,229 split holds. Another 1,597 remain ambiguous. Validation includes 1,433
supervised boundaries and 585 ambiguous ones. This synthesised/limited-source
corpus is not an independent commercial-song benchmark; shared compositions and
upstream pretraining overlap are not ruled out.

Two fixed paired profiles completed fitting, with declared 100/200/350/500-tree
checkpoints. Selection used only gated validation: safe false-attack reduction,
then lower log loss, fewer trees and fixed profile order. The guarded 350-tree
pair won, with two strict `.05` veto thresholds. Context/guardian widths are
112/60, containing paired pitch-relative note evidence, frozen V10 probabilities
and observed boundary geometry. Either head can preserve a boundary.

| Cohort | Matched attacks | False attacks before → after | Preservation |
| --- | ---: | ---: | --- |
| 94 validation recordings | 4,150 unchanged | 2,888 → 2,879 | All recordings pass |
| 60 consumed regression recordings | 1,182 unchanged | 1,259 → 1,255 | All recordings pass |
| 32 first-pass original MP3 seeds | 583 unchanged | 169 → 155 | All recordings pass |

Every gate preserves each previously matched attack and offset in each recording,
supplied pitch support, and the exact entire detected pitch-time union. No
recording may gain false notes. The new 32 original seeds, 261301–261332, came
from the same generator distribution as fitting, encoded as 48/128/192 kbps MP3
with verified zero codec lag. They are unseen-seed stress, not human-song
generalization evidence, and are now consumed regression. First-pass F1 increased
`.84188 → .85047`, with recall `.92101` unchanged. The improvement is small.

Production only joins retained adjacent notes of the same pitch and source part,
within the tested interior, with no positive gap, at most 30 ms overlap and
150 ms–2 seconds start spacing. Real rests and threshold equality are protected.
The earliest attack, pitch, velocity, object identity and source part remain;
only the selected hold's end extends. Balanced full-song bass uses this stage
after V10 filtering. Detailed, vocal, solo-piano and other accompaniment routing
keep their existing detectors. Metadata records removed notes and merged
boundaries separately.

Portable/runtime parity passed across all 545 cached training, validation,
consumed and first-pass recordings: 38,090 V10-retained events and 11,584
boundaries. Probability error was exactly zero against frozen sklearn at a
`1e-12` tolerance. The 55 merged boundaries across all cohorts include fitting
examples and are not independent accuracy. Retained note identity, source part,
attack, pitch, velocity and selected end extensions all match. Cross-part merging
is forbidden and tested separately.

Eight native routing cases passed, including genuine repeats, true octaves,
licensed Slakh, two actual learned-merge fixtures, 32-second context and subframe
audio. Native Basic Pitch/audio evidence and the complete production bass route
matched frozen sklearn exactly. The six native pipeline integration cases and
397 backend/training tests passed. The backend image built successfully; the
idle-queue deployment is ready, and running arrays and routing hashes match the
verified release. User uploads were
never retranscribed or used for fitting; integration uses isolated original
fixtures only.

The separate wrong-pitch residual batch trained eight heads on 22,403 labeled
V10-retained events, adding pitch-relative neighbor/confidence features. No
checkpoint passed all validation preservation gates with positive improvement.
Those weights are withheld; no tests, exports or production routing used them.
This release addresses false bass attacks rather than claiming to fix all wrong
melody/accompaniment pitches.

The frozen training contract records the pre-V11 `piano_transcription.py` hash.
Portable parity was completed before adding routing; native routing separately
binds the updated source. Reproduction must use the recorded historical baseline
sources. **Future training must include V11's filtered and merged decisions in
its baseline. `current_bass_baseline.py` reproduces historical V10 only and must
not be used to claim improvement against the current website.** New feature
caches must use the actual merged note durations, not reuse pre-merge features
under changed intervals. All first-pass sources used here are now consumed.

Checkpoint SHA256:
`6a8fbfb8e700d621d6b028c67ac777e3e1f648817674ed1b4d819f84797f5f24`.
Portable arrays:
`dec1bc16a2a41165e9c809fc8b11712f0392464f240ef6477b9c2d7195a6cd59`
and `101556c153e6b34688e6003dd4178152522290f3543fdceeeeb8aef1d8d3dc51`.
Attribution and license accompany the assets in
`backend/app/assets/bass-articulation-v1.LICENSE.txt`. Training/release witnesses
are archived under `training/results/bass-articulation-v2` and the first-pass
corpus under `training/results/bass-articulation-stress-v2`.
