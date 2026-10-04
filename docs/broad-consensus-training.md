# Broad accompaniment training — eight candidates withheld

Eight paired classifiers were fitted against the deployed V7 models. Both
validation-selected batch winners preserved every previously matched attack and
hold on the 162-recording consumed regression set, but removed no additional
unmatched detections. **Neither is eligible for release.** Website weights and
transcription routing remain unchanged. No user upload or score regeneration
was used for training or evaluation.

## Broader supervised population

Earlier residual classifiers could reject only candidates with V2 confidence
at most .5. A validation-only diagnosis now separates that confidence protection
from missing decoder identity or incomplete context:

| V7-retained validation population | Matched detections | Unmatched detections |
| --- | ---: | ---: |
| Shared interior, confidence at most .5 | 1,295 | 1,234 |
| Shared interior, confidence above .5 | 10,309 | 2,371 |
| Unshared decoder events or window edges | 2,632 | 905 |

These are onset/pitch matching counts, not a judgment that every unmatched note
is audibly wrong. The new offline policy allows the paired models to reject
shared interior candidates across MIDI 36–95 at all V2 confidence levels. It
still freezes prior V7 rejections, uses the identical candidate events and
features, protects incomplete neighborhoods, and excludes ambiguous timing
negatives from fitting. Melody, bass-stem and dedicated solo-piano inference
remain outside this experiment. This broader policy is **not enabled in the
website**.

The relation head has 74 pitch-relative acoustic/context/confidence features;
the guardian has 54 acoustic/confidence features. Both must reject a note. Every
validation recording must retain its matched attack and onset-offset reference
sets, with no false-note increase, at both raw and half-margin thresholds.

## Fixed batches and outcomes

Four profiles are declared before fitting: regularized (500 trees), capacity
(600 trees), recall (450 trees), and shallow (800 trees). They use positive-label
weights 10/8/16/10, learning rate .04, and no early stopping. Exact settings and
deterministic seeds are archived. Per-corpus/recording weighting precedes the
positive-label weight. Validation log loss is recorded every 50 trees; it is
diagnostic, not a replacement for note-preservation checks.

Each batch chooses one winner solely by maximum gated validation gain, with
profile order breaking ties. Its checkpoint and selection are frozen before a
single regression evaluation. Other profiles are not searched on regression.
Changed models, fitting manifests, policy settings, or selections block the
evaluation. A later dataset batch may use consumed results as development
evidence; neither batch represents independent generalization testing.

The first batch contains **593 training / 151 validation clips**, with **53,302
supervised events**. The second adds seconds 90–120, or the remaining audio, from
eligible existing Vienna recordings: **27 training / 7 validation passages**,
with 2,941/838 reference notes. Original performer splits and audited annotation
clock corrections remain fixed. Sources are selected by duration, never model
predictions. The expanded batch contains **620 / 158 clips** and **55,405 events**.
It was launched with the same four profiles before inspecting first-batch
regression results.

| Profile | First-batch validation unmatched notes removed | Expanded-batch validation unmatched notes removed |
| --- | ---: | ---: |
| Regularized | 2 — batch winner | 1 |
| Capacity | 1 | 1 |
| Recall | 1 | 2 — batch winner |
| Shallow | 1 | 1 |

Both winners leave regression counts unchanged at **14,993 matched / 3,861
unmatched detections**, preserving matched attacks and holds in every recording.
Their validation gains did not carry over. Fresh reserved passages were not
consumed, and neither checkpoint was exported or deployed. Continued training
does not guarantee monotonically improving unseen-song accuracy; release gates
keep these unproven candidates off the website.

## Next dataset preparation

A resumable augmentation workflow is now preparing **40 training / 10
validation GuitarSet excerpts**. It selects the second sorted comp/solo example
per genre for training performers 00–03 and validation performer 04, never test
performer 05. It adds original unpitched percussion at −6/0/+3 dB relative RMS,
cycles MP3 bitrates 48/128/192 kbps, decodes and verifies gapless waveform
alignment, then applies the production Demucs separator and caches note evidence.
These variations retain original GuitarSet annotations, with clipped crop edges;
no label times come from model predictions.

This adds compression/separation conditions to previously seen training pieces;
it is not an independent accuracy benchmark or a trained release yet. Audio and
features stay in ignored local storage. Completed plan, waveform and cache hashes
are checked before resuming. Preparation and codec clock checks are covered by
three additional tests. Existing CC BY 4.0 dataset attribution is recorded in
the [V7 source license](../backend/app/assets/left-consensus-v7.LICENSE.txt).

## Verification and reproduction

**263 backend/training tests passed; six opt-in real-audio integration cases
skipped.** Three additional preparation tests passed. Changed-code lint and diff
checks passed. Tests cover confidence-policy isolation, preservation of correct
holds and prior rejections, test-label exclusion, ambiguous labels, frozen
selection/model checks, performer splits, codec delay, and safe preparation
resumption.

The [archived results](../training/results/broad-consensus-v9) contain every
profile's validation/search/learning curves, both frozen regression outcomes,
dataset expansion metadata and the next augmentation plan. Research pickle
checkpoints and recordings are not shipped with the website.

Use the existing Python 3.11 training environment with
`PYTHONPATH=backend:training` and pinned scikit-learn 1.9.0. Preserve completed
experiment directories and choose a new run name for new fitting:

```bash
python training/train_broad_consensus.py .training/note-verifier-context-expanded --run new-batch --extra .training/relational-piano-training/manifest.json --extra .training/left-hand-expanded-data/manifest.json --extra .training/relational-piano-training-60/manifest.json --extra .training/relational-piano-training-90/manifest.json
python training/train_broad_consensus.py .training/note-verifier-context-expanded --run new-batch --test
python training/prepare_robust_training_stems.py .training
```

The cache identities pin resolved audio paths, so reproduce within the original
runtime mount layout or prepare a new isolated cache. Reserved fresh evaluation
is blocked unless regression preserves notes **and** removes additional false
notes. Future release also requires fresh preservation evidence, portable model
export, production/offline parity and normal functional checks.
