# Left-hand note precision studies

These studies address extra low notes in balanced full-song piano arrangements.
**These candidates are not released. The website continues using V32 note filtering and
the approved V38 release specialist.** No user uploads, saved jobs or scores were
processed, and the original 7 mm engraving is unchanged.

## Preserve the improved releases

V38 corrects some note endings but does not remove extra notes. Its historical
input caches contain V32 survivors, so they cannot simply be renamed as V38
observations. Both new studies explicitly replay the original V32 readers in an
isolated tree, retaining their checksum and source-partition guards. The current
website's V38 sources and model checksum are checked separately.

The new classifier proposes a removal mask from the complete V32 observations.
V38 then corrects releases using that same complete input, and the mask is
applied afterward. This composition preserves every retained event's deployed
V38 onset, release, pitch and velocity exactly. A removed candidate cannot alter
a neighboring retained note's release by changing its model context. Neither
classifier changes note clocks, adds notes, rewrites pitches or removes notes
outside MIDI 21–59 or within the protected first/last 2.5 seconds.

## What was trained

V39 trains independent coarse sustain, attack and release CNNs from scratch,
followed by a two-class KEEP/REMOVE head with 100 physical context features.
It does not inherit the old rejection logits or use the old guardian's output.
The two profiles have 12/20 convolution channels and KEEP loss weights 16/64.
Each completes 40 epochs, 25 updates per epoch, batch size 256.

The verified fitting partition contains 781 training and 225 validation excerpts:
the existing MusicNet, Vienna, GuitarSet, BabySlakh, NSynth, Groove and procedural
examples, plus the 16/4 public-domain Mutopia Joplin excerpts used for V38.
Composition/source groups remain separate. Real piano validation contains 15
MusicNet excerpts; the independent piano benefit requirement cannot be replaced
by procedural accuracy.

V39's preserving settings remove only 3/1 extra validation notes and no real
piano false notes. Its more active settings damage quiet notes and pitch
continuations. Both profiles are therefore withheld before any consumed test.

V40 adds 63 waveform periodicity features at pitch-relative lags across nine
positions, supplementing the local spectral evidence. These physical features
can help distinguish weak fundamentals from harmonics without labels or source
IDs as inference inputs. The two fresh profiles use 12/20 channels, a 128/64
hidden classifier, KEEP loss weights 16/32, and 100 epochs of 40 updates.
Training-only normalization and corpus/recording-balanced losses remain in use.
AdamW starts at .0005, weight decay .003, with cosine decay to .00005.

V40 uses 765 training and 221 validation excerpts. The Mutopia caches lack
observed waveform recurrence, so those items are excluded rather than filled
with invented feature values. All labels still protect genuine attacks, held
matches and any supplied pitch continuation, including short fragments.

## Selection and the failed release check

Thresholds and evaluation stages are declared before fitting. Selection requires
every validation recording to preserve attacks, holds, supplied pitch intervals,
fixed-pair timing error, repeated-key spacing, chord timing and unsupported
duration. Both the normal threshold and a stricter threshold must reduce false
notes, with separately positive MusicNet real-piano gain. Lower training loss
alone does not qualify a model; later V40 checkpoints fail preservation checks.

The frozen V40 winner is **support-protected, epoch 50**, threshold .99
(stricter .995), checkpoint SHA256:
`b2dcbacf3f789a3349ffea092578af5899680b9a2c42dc0c258cc30fd070db89`.

| Evaluation | Clips | Normal / stricter false-note reduction | Preservation |
| --- | ---: | ---: | --- |
| Validation | 221 | 72 / 58 | Every excerpt passes |
| MusicNet real-piano validation subset | 15 | 1 / 1 | Every excerpt passes |
| All consumed regressions | 563 | 247 / 193 | Five excerpts fail at both settings |

On the wider regression suite, three genuine bass attacks are lost. Two further
recordings lose short supplied pitch continuations despite retaining their
attack/hold matches. All retained clocks remain identical, but deletion harms
the musical content. Aggregate false-note reduction cannot override these
failures. The model is **refused release**.

The regression suite includes all 519 prior recordings and the 44 now-consumed
V38 first-pass clips. Existing physical cache hashes are verified. The latter
44 lack recurrence in their original caches, so recurrence alone is computed
from their checked decoded waveforms, without decoder inference or new labels.

No runner-up, confidence retuning, new first pass, export or deployment follows
the failure. The 40 predeclared fresh MP3 seeds 285101–285140 remain untouched.
Future studies still need every consumed regression plus an independent fresh
pass, physical/runtime parity and native pipeline checks before deployment.
Quiet bass under overlapping instruments and legitimate short continuations
need more independent fitting evidence; the failed test recordings must remain
outside fitting and selection.

## Reproduction and evidence

The recipes are `training/train_left_precision_v39.py` and
`training/train_left_precision_v40.py`. `prepare_left_precision_replay.py`
creates an isolated copy using the original V32 runtime preserved during V38
installation. Its strict readers continue checking original cache provenance,
waveforms, model arrays, source hashes and composition-disjoint groups. Run the
recipes with that replay's backend/training directories on PYTHONPATH and
`/src` as the separately checked production root. The existing training Docker
container mounts ignored local corpora at `/training`.

Complete plans, searches, diagnostic rows, learning curves, frozen source
witnesses, checkpoint hashes and the V40 regression report are archived under
`training/results/left-precision-v39-v1` and `left-precision-v40-v1`.
Checkpoints and licensed audio/caches remain local; model arrays are not shipped.
The new functional checks cover scope protection, invalid observations,
label-independent inference, genuine held-note fragments, repeated spacing and
the V38 composition. These checks establish implementation behavior, not song
accuracy.

## V41: new quiet-note evidence and a separate physical guardian

V41 adds 112 authored NSynth MP3 examples, using 88 training and 24 validation
sequences from the original 11/3 instrument partition. All four reserved
instruments remain outside fitting. New deterministic pitch choices cover quiet
single notes, quiet overlapping notes and chords, long overlaps, short and
variable repeated attacks, short chords and broken chords. Original long samples
retain their 3 s key labels and complete 1 s acoustic tails. Short articulations
are explicitly authored sample gates with an 80 ms damping tail protected in
full; they are not presented as recorded human key releases. MP3 rates cycle
48/128/192 kbps, with checked decoded clocks. No user upload is involved.

Two fresh 100-epoch profiles train on 853 excerpts and validate on 245. The
rejection CNN receives the V40 physical inputs. A separate MLP receives the 163
context/recurrence features and per-channel mean, maximum and standard deviation
of all three windows. Removal requires permission from both heads. The training
set supplies 29,564 KEEP and 6,229 REMOVE targets; source IDs and labels never
enter inference. Both heads start with a strong KEEP preference, with separate
weighted losses and training-only normalization.

The frozen winner is `dual-small`, epoch 100, threshold .99 (stricter .995),
checkpoint SHA256:
`b721246b9415d6389781b3349ca88f5454e1e9c6a56311ebf28da54972f43cd0`.

| Evaluation | Clips | Normal / stricter false-note reduction | Genuine attacks lost |
| --- | ---: | ---: | ---: |
| Validation | 245 | 75 / 54 | 0 / 0 |
| Real-piano validation subset | 15 | 2 / 1 | 0 / 0 |
| All consumed regressions | 563 | 202 / 162 | 3 / 2 |

Every validation excerpt preserves attacks, holds, pitch support and timing, but
the complete regression rejects the winner at both settings. A separate
physical guardian alone still misidentifies some genuine low notes. Aggregate
false-note reduction cannot override that loss. No runner-up, threshold change,
fresh pass, export or deployment follows; seeds 286101–286140 remain untouched.
The website's approved V38 model and retained note clocks remain unchanged.

The full suite passes 962 tests, with six optional native integrations skipped.
The plan, supervision, six checkpoint hashes, curves, searches, per-recording
diagnostics and exact executed source witnesses are archived under
`training/results/left-precision-v41-v1`. Audio and model weights remain local.

## V42: learning pitch presence throughout an event

V42 trains a separate convolutional frame-presence head alongside a fresh
rejection head. Presence targets follow the exact 40-sample coarse-window
clock. Every supplied pitch fragment is protected, including fragments shorter
than the sampling grid, and genuine jittered attack/hold matches. Inference
receives acoustic windows and normalized physical context only. A note can be
removed only when rejection is confident and every one of its 24 central
presence samples is below the complementary confidence threshold.

Measured recurrence is added to the existing verified 16/4 public-domain
Mutopia fitting excerpts from their checksum-checked waveforms. Decoder output,
labels, work partitions and reserved recordings are unchanged; there is no
new decoder inference. These are existing project sources receiving a new
physical observation, not a newly acquired performance corpus.

Two fresh 100-epoch profiles fit 869 excerpts and validate on 249. Neither
qualifies: preserving settings remove at most one extra validation note with
no real-piano gain; the small model's more active final checkpoint removes six
extra notes but harms a supplied note. No consumed regression, independent first
pass, export or deployment occurs, and seeds 287101–287140 remain untouched.
The suite passes 968 tests, with six optional native integrations skipped.
All six checkpoints' hashes, executed sources and measured-data provenance are
archived in `training/results/left-precision-v42-v1`.

## V43: independent rejection and presence calibration

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
