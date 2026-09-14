# Context melody training: evaluated, not released

The new model recovered more melody on the independent test but also emitted
more notes during annotated silence. It failed the release gate. The app keeps
the V1 recognition weights; this experiment is retained for reproducibility,
not silently substituted into production.

## Data and fitting

- 314 training clips: 26 Vocadito song phrases plus 288 VocalSet excerpts,
  totaling 3,815.74 seconds (63.6 minutes). No user uploads enter training.
- VocalSet adds 12 training singers, four validation singers (48 clips), and
  four held-out singers (48 clips). Existing Vocadito validation remains a
  regression check; its previously consumed test is not reused for selection.
- All new data are CC BY 4.0. Source revisions, checksums, author attribution,
  annotation limitations and commands are in the [training guide](../training/README.md).
  Non-commercial CSD and MAESTRO data were excluded from this new training.
- A 5,853-parameter network extends V1 with a 1.26-second temporal context
  branch. Its 3,666 new parameters were actually fitted for 30 epochs × 60
  updates; V1's 2,187 parameters remain frozen. Each update uses four supervised
  crops plus one training-only rehearsal crop. Frame-distribution and attack
  distillation constrain forgetting on natural song phrases.
- Epoch 30 was selected on validation loss. Validation selected five-frame
  smoothing, a 60 ms minimum note and an .8 attack threshold. Parameters and
  decoder settings were frozen before inspecting the new test.
- Checkpoint SHA-256:
  `85993789a7d62faa4a32825b8201790e8223b814c472a48f4d5f19776f8f01fc`.

An initial run used incorrect validation clocks. The dataset's `m9` annotation
durations were twice their audio durations; a metadata-based 0.5 clock correction
fixed all 12 selected clips. Splits and audio features were unchanged. Training
was repeated with corrected validation. An unconstrained fine-tune then failed
the natural-song validation gate and was rejected before testing. Those attempts
are retained under `prior-attempts`; their results are not release evidence.

## Frozen test against the deployed V1 model

Macro averages across 48 recordings from four held-out singers. These measure
recognition before arrangement and quantization, not commercial-song accuracy.

| Metric | Deployed V1 | Candidate |
| --- | ---: | ---: |
| Correct pitch on annotated voiced frames | 58.8% | 68.6% |
| Annotated melody incorrectly silent ↓ | 22.1% | 8.9% |
| Notes during annotated silence ↓ | 13.0% | **15.3%** |
| Note onset F1 | .202 | .291 |
| Note onset + offset F1 | .145 | .212 |

Pitch recall improved on 45 of 48 clips; onset F1 improved on 35. False voicing
increased by **2.25 percentage points**, exceeding the pre-test maximum increase
of 2 points. This matters for the reported unwanted held notes, so improvement
in the other metrics does not override the failure.

On the eight natural-song validation clips, onset F1 changed .651 → .644 and
offset F1 .521 → .510, within the 0.02 regression limits. False voicing changed
6.18% → 6.48%. These are validation figures, not new held-out song evidence.

A fixed eight-clip subset of the new test (the first two clips per singer in the
seed-frozen manifest) was also mixed with procedural backing and separated with
Demucs. Pitch recall changed 61.2% → 68.6%, false silence 15.7% → 9.1%, onset F1
.187 → .315 and offset F1 .140 → .269. False voicing rose 15.7% → 16.9%. This
subset passed its gate, but does not erase the failed 48-clip clean test. It is
not an independent corpus or a benchmark of professionally mixed pop songs.

## Changes released to the app

The shared full-song arranger now:

1. Retains short notes and low pitches already accepted by the trained vocal
   decoder, instead of applying another generic 90 ms/low-pitch cutoff.
2. Moves a low sung melody by a consistent whole octave into a clearer piano
   register, preserving its intervals, timing and rests. Solo-piano recognition
   does not pass through this arrangement step.
3. Gives melody articulation priority when accompaniment plays the same key.
   Earlier support ends at the melody attack; support attacks inside that melody
   note are omitted, preventing a backing note from holding the key past release.
4. Reports source-level retention and octave transposition. Explicit eighth-note
   settings remain respected, with a warning when Precise rhythm retains
   materially more melody notes.

Two 30-second diagnostics used the same shared code, with no filename logic:
“Feel No Ways” retained 53 of 58 recognized melody notes at sixteenth-note
resolution (previous cleanup retained 47), and “I'm Spent” retained 69 of 70.
Both received a one-octave melody arrangement shift. Supporting notes previously
extended melody releases in two and five instances respectively; the shared-key
priority removes that class of conflict. These are retention/arrangement checks,
not evidence that every retained pitch is correct. Neither song was training
data, and neither has a labeled accuracy score here.

## Reproducibility and remaining work

The [research checkpoint](../training/results/context-melody-v2/candidate.pt),
[split and label hashes](../training/results/context-melody-v2/split.json),
[training history](../training/results/context-melody-v2/history.json),
[per-recording test results](../training/results/context-melody-v2/test.json),
[separation results](../training/results/context-melody-v2/separated-test.json)
and [release decision](../training/results/context-melody-v2/promotion.json) are
committed. The checkpoint has its own attribution/license file and is not
packaged as a runtime asset.

VocalSet labels are semi-automatic with manual review and partly depend on pYIN;
the model also receives pYIN evidence. Shared exercises prevent a claim of
composition independence. The dataset emphasizes sung vowels and techniques,
not rap, overlapping singers, dense backing or finished piano arrangements.

The next recognition experiment needs stronger negative examples for breaths,
reverb tails and unpitched voice, plus fresh independent evaluation. Changing a
threshold against this consumed test and calling the result “held out” would
not establish improvement. The remaining recognition errors are unresolved;
the released score-conversion changes do not make a wrong detected pitch right.
