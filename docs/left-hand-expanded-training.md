# Expanded left-hand dataset training: candidates withheld

Added 64 labeled GuitarSet training passages and 16 validation passages, then
actually fitted two replacement low-register classifiers. Neither qualified for
release. The website retains the verified V5 model and bass-hold reconciliation;
no user uploads were processed and no sheet music was generated.

## Additional data

Every eligible GuitarSet train/validation recording at least 35 seconds long
contributes seconds 30–60 (or its remaining tail), excluding .25 seconds at each
edge from scored onsets. Long reference releases are clipped only at the actual
audio boundary. The existing documented annotation-error exclusions are retained.
No model predictions or accuracy scores decide which recordings enter this set.

The training additions contain **3,659 labeled notes, including 2,040 in MIDI
36–59**, over 661.19 scored seconds. The validation additions contain 926 notes,
including 476 low notes, over 165.30 seconds. Players 00–03 remain training-only;
player 04 remains validation-only. Player 05 and other test corpora are untouched.
These are new passages from existing performers/compositions, not an independent
new corpus. GuitarSet is the same CC BY 4.0 source credited in the
[deployed model notice](../backend/app/assets/accompaniment-left-hand-v1.LICENSE.txt).

Feature preparation reads only these licensed dataset recordings. It runs the
frozen acoustic detector and caches the same 52 features used by the production
filter. It does not run the website transcription pipeline or render scores.
Cache identities cover the manifest entries; preparation can resume, but rejects
a changed source split. The compact [dataset manifest](../training/results/left-hand-expanded-v2/dataset.json)
records provenance and per-passage counts without committing audio or annotations.

## Fitting and selection

The expanded experiment has **551 training clips and 139 validation clips**,
including the previously added later Vienna passages. Each model fitted **32,122
unambiguous low-register events: 27,757 positive and 4,365 negative**. Test data
did not enter fitting, threshold selection, or capacity selection.

The first model retained the deployed head's 300 trees, 15 maximum leaves,
minimum leaf size 40, learning rate .05, L2 4, eightfold positive weight and seed
260925. After its validation failure, a separately recorded run raised only the
maximum leaves to 31. Corpora and recordings receive equal base weights before
the positive multiplier.

These candidates would **replace**, not append to, the existing left-hand head.
The evaluation therefore reconstructs the actual V5 baseline, including the
original left-hand head. A replacement can restore previously rejected notes;
the gate explicitly detects resulting false-note increases. Prior V5 gains are
not counted again as gains from this experiment.

Every validation recording must preserve its previously matched attacks and
onset-offset references, with no increase in false notes. A candidate must pass
at both its raw grid threshold and the predeclared half-threshold safety margin;
the margin result must show a net improvement. No candidate passed this rule.

| Candidate | Illustrative validation result | Why withheld |
| --- | --- | --- |
| 15 leaves | At .075, 12 fewer false notes and all per-recording gates pass | At the .0375 margin, two recordings fail despite a net reduction of one false note |
| 31 leaves | At .075, 34 fewer false notes and all per-recording gates pass | At the .0375 margin, four recordings fail, with no net reduction |

These are validation diagnostics, not held-out performance or website gains.
No alternative threshold was selected after failure, no preservation rule was
weakened, and regression/fresh evaluation was not run. `selection.json` records
`selected: false`; its zero-threshold report is a fallback diagnostic, not a
release recommendation. Rejected weights stay in ignored `.training/` and are
not copied into the website's assets.

## Checks and reproduction

Eleven focused tests passed for performer separation, exact cropped label clocks,
held-note preservation, detecting restored false notes, and comparison against
the deployed baseline. Lint and diff checks passed. This change affects training
tools and reports only, so the running website does not require a rebuild.

Use the existing Python 3.11 training environment, scikit-learn 1.9.0 and
`PYTHONPATH=backend:training`:

```bash
python training/prepare_left_hand_expansion.py .training
python training/train_left_hand_expanded.py .training/note-verifier-context-expanded --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json
python training/train_left_hand_expanded.py .training/note-verifier-context-expanded --run left-hand-expanded-v2-capacity31 --leaves 31 --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json
python -m pytest -q training/test_left_hand_expanded.py training/test_left_hand_verifier.py training/test_residual_verifier.py
```

Completed training/evaluations cannot be overwritten. Use a new run name for a
new experiment. Both attempts' settings, frozen baseline/checkpoint hashes,
threshold searches and per-recording validation metrics are
[archived](../training/results/left-hand-expanded-v2).
