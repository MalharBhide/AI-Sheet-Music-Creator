# Expanded accompaniment training: four candidates remain offline

Four paired classifiers were fitted to reduce unsupported accompaniment notes.
None demonstrated a safer regression improvement over the deployed V7 model, so
**the website keeps V7**. No new weights were exported or deployed. No user
uploads were transcribed or existing scores regenerated.

## Additional labeled data

Prepared seconds 60–90, or the available remainder, from all eligible existing
Vienna recordings at least 65 seconds long. This adds **42 training passages and
12 validation passages**, containing 5,692/1,668 reference notes respectively.
The original performer groups and frozen audio/annotation clock corrections are
preserved. Selection uses source duration, never predictions. These are later
sections of existing compositions, not independent new performers or pieces.
Existing Vienna/GuitarSet CC BY 4.0 and original procedural-piano/source-separated
training data remain the only fitting sources; no new datasets were downloaded.
See the existing [source credits](../backend/app/assets/left-consensus-v7.LICENSE.txt).

The expanded cohort is **593 training clips and 151 validation clips**. Tests do
not enter fitting or threshold selection. All eight V7 later piano excerpts now
enter the 162-clip consumed regression suite. The next later test sections were
not prepared or evaluated because no candidate qualified for release.

## Training and safeguards

Each fit trains a relation head and an acoustic guardian, requiring both to
reject a candidate. Only V7-retained notes can change. Labels are recomputed after
V7 filtering so surviving correct duplicates receive their new positive match.
Ambiguous timing negatives are excluded. Equal corpus/recording weights precede
an eightfold positive-event multiplier, or tenfold for the guarded profile.

All heads use 300 trees, learning rate .05 and no early stopping. The guarded
profile uses minimum leaf size 60, L2 8 and 31/15 leaves, seeds 261006/261007.
The other profiles use minimum leaf size 40, L2 4, 31/15 leaves, positive weight
8; seeds are 261008/261009, 261010/261011 and 261012/261013 respectively.

Joint threshold selection uses validation only and requires every recording to
preserve both matched onset and onset-offset reference sets, with no false-note
increase. The predeclared half-margin policy remains mandatory. A candidate must
also improve false-note removal over pinned V7 on regression before export.
Evaluation records include the frozen selection digest, model digest and both
thresholds; an export rejects changed or failed release evidence.

The first two profiles use existing 72 relation/52 acoustic features. The next
two add frozen V7 relation/guardian confidence as label-free inputs, producing
74/54 features. Both training and any future inference use the same checked
feature construction. Default runtime model contracts still reject these newer
formats unless their trusted contract is explicitly supplied.

The first three fits target MIDI 36–59. The fourth expands to MIDI 36–95 shared
accompaniment candidates. All preserve original V2 confidence above .5, unshared
candidates and the first/last 2.5 seconds of physical inference windows. The
separate melody, bass-stem and solo-piano models are outside these experiments.

## Results

| Candidate | Training events | Released-margin validation false notes removed | Regression | Decision |
| --- | ---: | ---: | --- | --- |
| Guarded expanded data | 27,078 | 0 | Not consumed | No eligible gain |
| V7-style settings | 27,078 | 1 | Two false notes removed, one correct matched piano note lost | Rejected |
| Frozen V7 confidence inputs | 27,078 | 1 | All matched notes preserved, zero additional false notes removed | Rejected |
| Both accompaniment registers | 53,302 | 0 | Not consumed | No eligible gain |

The three low-register fits use 23,573 positive/3,505 negative eligible labels.
The broad fit uses 47,452 positive/5,850 negative labels. Training-event counts
include confidently accepted examples for supervision; production rejection
eligibility remains narrower. These are actual fitted classifiers, not hand-set
pitch or song-specific removal rules.

V7's 162 regression clips contain 14,993 matched detections and 3,861 unmatched
detections. The second fit changed those to 14,992/3,859 and failed the individual
recording gate. The third remained 14,993/3,861. These regression results informed
subsequent experiments; they are consumed evidence, not independent test accuracy.
No thresholds were retuned on a failed regression result.

## What limits further decluttering

A validation-only coverage audit classifies every V7-retained detection under the
broad residual policy. It finds:

| Group | Matched reference detections | Unmatched detections |
| --- | ---: | ---: |
| Eligible for residual correction | 1,295 | 1,234 |
| Protected by confidence, decoder identity or window edges | 12,941 | 3,276 |

**About 73% of remaining unmatched detections are outside this policy's reach.**
The audit does not distinguish how much each protection contributes and cannot
conclude that all unmatched notes are audibly wrong; onset/pitch matching can
also count timing errors as unmatched. It explains why repeating this restricted
residual fit is unlikely to remove most perceived clutter.

Further work should build full-context labeled examples for the currently
protected candidate populations, then train and validate a broader operating
policy before changing those protections. Arbitrarily deleting protected notes
or raising thresholds would risk erasing the melody and holds the user wants.
These experiments do not establish higher whole-song accuracy.

## Verification and reproduction

**245 backend/training tests passed**, with six opt-in audio integration cases
skipped. Backend and changed training Python lint pass. The live website reports
ready and still identifies the pinned V7 verifier. No rejected weights were
packaged and no production transcription routing was changed.


Frozen settings, model hashes, validation searches, rejection decisions,
per-recording regression and the coverage audit are
[archived](../training/results/left-consensus-v8). Audio, feature caches and local
research pickle files stay outside Git. The website never loads research pickle.

Use Python 3.11 / scikit-learn 1.9.0 with `PYTHONPATH=backend:training`. Preserve
completed runs and select a new run name for a new fit. Example profile commands:

```bash
python training/prepare_relational_piano.py .training --training-offset 60
python training/train_left_consensus_v8.py .training/note-verifier-context-expanded --run new-expanded --profile expanded --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json --extra .training/relational-piano-training-60/manifest.json
python training/train_left_consensus_v8.py .training/note-verifier-context-expanded --run new-expanded --test
python training/audit_accompaniment_coverage.py .training/note-verifier-context-expanded .training/note-verifier-context-expanded/new-expanded .training/new-coverage-audit.json
```

Other profiles are `guarded`, `baseline-evidence` and `full-accompaniment`.
Regression rejects ineligible validation selections. Fresh evaluation is blocked
until regression passes; export additionally requires positive regression gain,
fresh preservation evidence and exact frozen selection/model/threshold agreement.
