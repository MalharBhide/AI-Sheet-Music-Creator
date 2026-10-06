# Attack-local accompaniment decluttering study

V23 trains a binary KEEP/REMOVE model on the notes that survive the currently
deployed V16 accompaniment pipeline. It combines the sealed V19 short attack
features (61 positions, 18 CQT/STFT channels), longer CQT context (40 positions,
9 channels), and 84 acoustic/context features. It starts from the approved V16
coarse encoder. References provide supervision; they never appear in acoustic
inputs. Timing actions and creation of additional notes are unavailable.

Two profiles use KEEP-class protection weights of 12 and 24, train for 80 epochs,
and save checkpoints at epochs 20, 40, 60 and 80. The fitting split contains 573
recordings; the 160 validation recordings alone select the checkpoint, removal
confidence, and frozen acoustic guardian threshold. Per-corpus and per-recording
loss balancing prevents one large source from dominating fitting. Normalization
uses eligible, supervised training examples only.

The existing [V16 dataset notice](../backend/app/assets/bass-temporal-refinement-v1.LICENSE.txt)
applies: Vienna, GuitarSet and BabySlakh use CC BY 4.0, alongside project-authored
signals. User uploads and saved scores are excluded; no score is regenerated.

The winner must reduce false notes under both its normal confidence threshold and
the stricter threshold `t + (1 - t) / 2`. Every validation recording must retain
each previously matched attack and hold and every covered reference-pitch
interval, without additional false positives. Retained events have identical
onsets, ends, pitches and velocities. Fixed baseline-pair timing and repeated-key
spacing are also checked; deleting an event earns no timing improvement.

Only the frozen validation winner can proceed to all 360 consumed regressions:
the earlier 296 recordings plus the V20 and V22 first-pass MP3 batches. These
records never enter fitting or checkpoint selection. Failure prevents any new
first-pass inference. A passing regression permits one 32-clip project-authored
MP3 first-pass batch, with seeds 268101–268132 and 48/128/192 kbps codecs fixed
before fitting. Both configurations must again reduce false notes and pass every
preservation gate. No test-driven checkpoint replacement, threshold adjustment
or replacement first-pass seeds are allowed.

The two profiles completed all 80 epochs. Validation selected
`support-conservative` epoch 80, removal confidence .9 and guardian threshold .5.
Checkpoint SHA256:
`a076b1c7f81aa326447cc37163bf7e747e91848a78079635d5258f4c7a73a2b5`.

| Evaluation | Normal false notes removed | Stricter false notes removed | Every recording passes |
| --- | --- | --- | --- |
| 160 validation recordings | 108 | 91 | Yes |
| 360 consumed regressions | 212 | 206 | No |

Both regression configurations lose .0073977324 seconds of supplied reference
pitch coverage in `bass-temporal-fresh-v1-263121`. All matched attacks and holds
remain, and retained clocks are unchanged, but the coverage requirement still
rejects the candidate. No replacement checkpoint or threshold is tried.
First-pass seeds 268101–268132 remain unused; no new audio is inferred, and no
weights are exported, routed into the website or deployed. The approved V16
model remains deployed.

[Exact fitting/evaluation evidence](../training/results/declutter-local-v23-v1)
includes lossless per-recording reports, source contracts, learning curves,
eight checkpoint hashes and completed code checks. All 702 backend/training
tests passed, with six opt-in integration tests skipped; two additional export
guard tests passed. Audio, feature caches and experimental checkpoints stay
local. These results do not establish accuracy on arbitrary songs or guarantee
every real note is retained.
