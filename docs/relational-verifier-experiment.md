# Relationship-aware note filter: rejected experiment

The website remains on V4. This experiment trained models using neighboring-note
evidence and added **56 piano training passages and 15 validation passages**. The
final candidate reduced false notes but lost one previously correct note, so it
was not released. No user audio was processed and no sheet music was generated.

## Hypothesis and training

The existing filter examines a candidate's acoustic evidence. The proposed model
adds 20 pitch-relative relationships: shared attacks, overlapping notes, velocity
and duration comparisons, lower-octave/harmonic candidates, repeated-pitch and
near-pitch continuity, and nearby attacks. These inputs use predictions and V2
confidence only, not reference labels, absolute pitch, key, filename or song ID.

It operates after the actual V4 decisions, preserving stronger V2 events,
unmatched decoder events and all already-rejected notes. The two-second neighbor
radius is explicit. Events near crop boundaries remain protected because the
evaluation caches omit neighboring candidates outside those crops. Tests verify
feature equality for eligible events before and after cropping, transposition
and time-shift invariance, candidate-order independence, and edge protection.

Additional data are seconds 30–60 of existing Vienna 4x22 training/validation
recordings. Every eligible recording at least 35 seconds long is included,
independent of model accuracy. Performer partitions and the previously audited
audio-only clock shifts are unchanged. No test player enters training or
validation; compositions are shared across performers. The corpus is CC BY 4.0;
Werner Goebl and the other existing source attributions remain documented in the
[V4 model card](accompaniment-residual-v4.md). No new downloads were needed.

Each attempt used 300 boosted trees, maximum 31 leaves, minimum leaf size 40,
learning rate .05, L2 3, sixfold positive-event weight and seed 260923. Corpora and
recordings receive equal base weight before the positive multiplier. The final
bounded model fitted **48,942 unambiguous retained events from 487 training clips**.
Its 123 validation clips selected threshold .05, then the predeclared half-margin
froze .025 before regression evaluation. Neither evaluated candidate was retuned
after its regression failure.

## Decisions

| Attempt | Validation false notes removed | Regression outcome | Decision |
| --- | ---: | --- | --- |
| Original relationship features, existing data | 6 | 5 false notes removed, 1 correct piano note lost | Rejected |
| Expanded data, unbounded neighbor features | 65 | Not evaluated | Superseded after identifying crop-context mismatch |
| Expanded data, bounded features and edge protection | 29 | 20 false notes removed, 1 correct guitar note lost | Rejected |

The final 142-excerpt regression comparison is **3,308 → 3,288 false notes**,
but **12,879 → 12,878 matched correct notes**. The failed recording is
`05_Jazz2-110-Bb_solo`. Aggregate F1 improvement cannot override this per-recording
preservation failure. V4's clearer melody and current behavior are unchanged.

These are consumed regression recordings, not independent generalization
evidence. Matching uses 50 ms onset/50 cent pitch tolerance without offset
constraints. The additional data and corrected feature workflow are useful
development work; they do not establish improved deployed accuracy. Twelve
eligible later test-piano sections were identified by duration only, but their
features and accuracy were not evaluated because the regression gate failed.

## Reproduction and artifacts

Raw audio, labels and rejected fitted pickles remain in ignored `.training/`.
They are never loaded by the website. The [archived results](../training/results/relational-verifier)
include every attempt's selection, metrics, checkpoint hash, release decision and
a compact additional-data manifest. The old feature source is archived to explain
the superseded runs; current `note_relations.py` uses the corrected bounded version.

With the existing Python 3.11 training container, scikit-learn 1.9.0,
`PYTHONPATH=backend:training`, and the pinned V4 caches:

```bash
python training/prepare_relational_piano.py .training
python training/train_relational_verifier.py .training/note-verifier-context-expanded --run relational-verifier-bounded --extra .training/relational-piano-training/manifest.json
python training/train_relational_verifier.py .training/note-verifier-context-expanded --run relational-verifier-bounded --test
python -m pytest -q training/test_note_relations.py training/test_relational_verifier.py training/test_residual_verifier.py
```

Commands refuse to overwrite completed experiments. Use a new run name for new
development. Repeating consumed evaluations does not make them new tests. Fresh
evaluation is blocked until the frozen candidate passes regression preservation.
