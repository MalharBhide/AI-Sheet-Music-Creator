# Left-hand note precision studies V39 and V40

These studies address extra low notes in balanced full-song piano arrangements.
**Neither model is released. The website continues using V32 note filtering and
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
