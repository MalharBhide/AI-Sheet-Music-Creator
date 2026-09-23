# Low-register accompaniment and bass-key ownership

This update targets two sources of left-hand clutter in full-song arrangements:
false low accompaniment detections, and repeated strikes caused by overlapping
bass and accompaniment predictions for the same piano key. It applies to future
uploads independent of filename or audio encoding. No user recordings were
transcribed and no existing scores were regenerated during development.

## Newly trained low-register filter

V5 retains the existing V4 classifier and adds a specialist for accompaniment
notes from MIDI 36 through 59, matching the website's bass-staff range within the
accompaniment decoder. It may only remove V4-retained, exactly shared decoder
events whose original V2 confidence is at most .5. Treble accompaniment, vocal
melody, independent bass-stem inference and dedicated solo-piano inference are
unchanged. Retained note objects and their timing/velocity are unchanged.

The model was actually fitted on **30,145 low-register events: 26,129 positives
and 4,016 unambiguous negatives**, drawn from 487 existing training clips. It uses
52 cached pitch-relative acoustic features. Data are GuitarSet, corrected Vienna
piano, original procedural piano and separated training-only GuitarSet mixtures,
including the additional later Vienna passages prepared previously. Existing
performer partitions and clock corrections are retained. License and source
credits are in the [asset notice](../backend/app/assets/accompaniment-left-hand-v1.LICENSE.txt).
No restricted singing data or user recordings entered fitting.

Fixed parameters: 300 boosted trees, at most 15 leaves, minimum leaf size 40,
learning rate .05, L2 4, no early stopping, seed 260925. Corpus/recording base
weights are balanced before an eightfold positive-event weight. The 123-clip
validation set selected threshold .1 from the declared grid; the predeclared
half-margin froze **.05** before regression. The score is not a calibrated
probability of musical correctness.

| Evaluation | Clips | Matched notes, V4 → V5 | False notes, V4 → V5 |
| --- | ---: | ---: | ---: |
| Validation used for selection | 123 | 12,196 → 12,196 | 3,766 → 3,758 |
| Consumed regression | 142 | 12,879 → 12,879 | 3,308 → 3,298 |

Every recording must preserve both its previously matched attacks and its
previously matched onset-offset references. Matching uses 50 ms onset/50 cent
pitch tolerance; the hold check adds 20% duration or 50 ms offset tolerance.
This rejects a candidate that leaves a short duplicate onset but removes the
correct sustained note. Both gates passed in every validation/regression clip.

The regression reduction is **10 false notes (0.30% of all remaining false notes)**.
The regression improvements occurred in GuitarSet and a separated GuitarSet
mixture; piano regression counts were unchanged. These recordings were used in
earlier evaluations: this is a small regression improvement, **not an independent
generalization benchmark or proof of accurate bass transcription**. The filter
cannot restore missing notes or correct wrong pitches. An isolated-bass model
still needs suitable bass-specific labeled data and evaluation; applying an
accompaniment classifier to that different decoder without testing is avoided.

The numerical NPZ checkpoint is loaded without pickle and checked against SHA-256
`7b3343637fa0d19ba1cd468845fcc309f30ac55b7fddc00f68bbc607abc2555b`.
Missing/damaged weights fail explicitly, and readiness includes the new asset.
Runtime/offline parity passed on **33,815 events across 265 clips**, including
preservation of retained object identity, source assignment and all note attributes.

## Resolve bass/accompaniment collisions

Previously, a bass C from seconds 0–2 plus accompaniment C detections at .5 and
1 seconds could become repeated strikes in the score. The arrangement now gives
the bass line ownership of that key: backing attacks inside the bass hold are
omitted, and backing notes beginning earlier end at the bass attack. A backing
note that starts in a bass rest remains; distinct chord pitches and actual
repeated bass attacks remain. No replacement attack is invented at release.

This reconciliation runs before backing-voice selection so duplicate bass keys
do not consume voice slots. It is an explicit piano-arrangement priority, not
learned evidence that the bass prediction is correct. A wrong bass prediction
can still be wrong; this does not remove unrelated false chord tones. The existing
melody priority remains in effect, and solo-piano transcription is unaffected.

## Verification and reproduction

178 tests passed; six opt-in audio integration cases were deliberately skipped.
Checks cover held bass notes, repeated attacks, rests, unrelated pitches,
MusicXML/MIDI/browser playback, full-song routing, damaged weights, protected
treble events and the strengthened hold-preservation gate. No user audio was
processed. Archived [run and per-recording results](../training/results/left-hand-verifier-v1)
include the validation search, frozen checkpoint identity and runtime parity.
Lint and the backend production build passed. Deployment followed an idle-queue
check; the running process loaded the expected checkpoint and threshold, passed
an in-memory bass-hold check, and reported ready through the frontend health proxy.

Using the existing Python 3.11 training environment, scikit-learn 1.9.0 and
`PYTHONPATH=backend:training`:

```bash
python training/train_left_hand_verifier.py .training/note-verifier-context-expanded --extra .training/relational-piano-training/manifest.json
python training/train_left_hand_verifier.py .training/note-verifier-context-expanded --test
python training/verify_left_hand_runtime.py .training/note-verifier-context-expanded .training/left-hand-runtime-parity.json
python -m pytest -q backend/tests training/test_left_hand_verifier.py training/test_residual_verifier.py training/test_relational_verifier.py
```

Training/evaluation refuse to overwrite completed artifacts. Use a new `--run`
name for a new experiment; repeating consumed evaluations does not make them fresh.
