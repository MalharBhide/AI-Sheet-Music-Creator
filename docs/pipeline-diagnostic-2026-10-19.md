# Why the recent training did not improve the product

**Decision: stop repeating V43 bass-support fine-tuning. The next work should address post-detection rhythm handling and measure errors by source branch before choosing another model-training target.** The audit demonstrates downstream errors even with correct input notes. It also confirms that the recent training cannot change several paths responsible for the user's score.

This is a bounded diagnostic, not a new model release. No audio was read, downloaded, separated or transcribed. No fitting, user jobs, consumed-regression evaluation or fresh release tests ran. Production files and weights are unchanged. Findings come from current source, existing V70 validation diagnostics, and ten authored symbolic fixtures plus an accompaniment-selection probe. These fixtures establish specific behavior, not its prevalence in real songs.

## Confirmed findings

### 1. The optimized component does not cover the requested behavior

Full-song mode runs Demucs, then distinct vocal, bass and accompaniment paths. The recent V66–V70 experiments change a final bass pitch-support filter, after earlier detectors and rejection filters. Their evaluated composition can only delete additional candidate notes; it cannot add missing pitches, correct a pitch, or move an onset or release. The timing-preservation checks therefore cannot establish improved timing.

That filter runs only for `bass` in `balanced` detail. It does not run on the `other` accompaniment branch, on the vocal melody, in solo-piano mode, or in detailed bass mode. The left staff is assigned by pitch below MIDI 60, **not by source role**. Consequently, wrong accompaniment notes below middle C appear in the left hand but are untouched by this recent training. The other branch has its own older accompaniment verifier; it is not unfiltered, but it was not the target of these experiments.

Evidence: [engine routing](../backend/app/services/piano_transcription.py), [staff assignment](../backend/app/services/midi_to_score.py), [V70 composition and fitting scope](../training/train_approved_finetune_v70.py), [presence eligibility](../backend/app/services/left_presence_network.py). The frontend defaults to full-song/balanced, so this is relevant to the default product path, not a claim about which options any individual user selected.

### 2. Cleanup can damage already-correct rhythms and durations

The diagnostic supplies exact note pitches and times at 120 BPM, bypassing acoustic inference. It then executes the production phase estimation, balanced cleanup, MusicXML creation, and actual score-playback exporter. It also passes the uncleaned reference directly through notation to distinguish rounding there from cleanup.

| Authored case | Measured cleanup effect |
| --- | --- |
| Straight sixteenths | All 16 attacks and durations preserved |
| Complete triplets | All 12 attacks and durations preserved |
| 2:1 swing | All 8 notes retained, but repeated spacing differs by 41.67 ms on average |
| Two-note triplet fragments | All 8 notes retained, but repeated spacing differs by 41.67 ms on average |
| Fast 62.5 ms bass notes | All 16 deleted |
| Separate 70 ms staccato bass notes | All 8 deleted despite 250 ms attack spacing |
| Quiet bass notes at velocity 28 | All 8 deleted |
| 95 ms staccato bass releases | All 8 extended to 125 ms |
| 120-to-150 BPM change at fixed score tempo | Maximum attack displacement 50 ms |
| Independent chord holds and repeated key | All 5 notes and individual holds preserved |

The causes are explicit in [audio cleanup](../backend/app/services/audio_analysis.py): a 90 ms minimum and velocity floor 32 for balanced bass/accompaniment; a mostly straight grid; complete-three-slot evidence required for triplets; a minimum quantized duration; one tempo for the score. Basic Pitch's balanced event decoder also uses a 90 ms minimum, so changing only the later cleanup cutoff cannot recover notes already omitted upstream.

The quiet-note result is conditional on the incoming velocity. Model velocity is treated as a loudness proxy, not calibrated correctness. Removing the cutoff without acoustic evidence could also retain false detections; this audit does not justify blindly disabling it.

### 3. The arrangement heuristic can prefer an existing wrong note

With two correct backing tones (MIDI 48 and 55, velocities 80 and 70) and one supplied false tone (MIDI 52, velocity 110), all held for two seconds under a higher melody, the balanced reducer selects MIDI 48 and 52. It drops a genuine chord tone and retains the false one. With just the two correct tones, both survive.

[Accompaniment reduction](../backend/app/services/audio_analysis.py) selects up to two supporting lines using duration multiplied by velocity. It has no ground-truth harmonic correctness in that objective. A long, loud error can outrank a true supporting line. This is a controlled counterexample, not a measured false-note rate on recordings. Reducing the voice count further is not an evidence-based fix.

### 4. Notation/playback was not the source of these demonstrated errors

Across the nonempty authored fixtures, MusicXML and playback preserved the cleaned notes without additional missing or extra attacks. Additional timing differences were at most 0.000334 ms, attributable to exported numeric precision. Straight rhythm, complete triplets and independent chord holds were intact end to end.

This narrows the investigation for these cases; it does not certify every score, visual layout or browser audio behavior. The audit used the actual `export_score_playback` path, which joins individual pitch ties, rather than a generic MIDI round trip.

### 5. The existing evaluation does not measure the complete product

V70's 618 validation examples contain 381 timed bass examples and 237 isolated-pitch clips. The timed set includes 15 `musicnet-real-piano`, 12 `vienna-piano-mp3-bass`, two `vienna-demucs-bass-bass`, 36 authored sampled-piano progressions, 58 guitar-bass-register examples, plus other synthetic/bass/mixture families. These are clips, not 618 independent songs. The training set includes 712 isolated-pitch clips out of 2,302. Full corpus counts are in the attached report.

These corpora can test particular bass errors. They do not establish whether a finished full-song piano arrangement has a recognizable lead, appropriate left-hand harmony, correct local rhythm, or pleasing playback. The current cache evaluation ends before global cleanup, arrangement, notation and playback, so it cannot detect the errors demonstrated above.

Existing normal-policy baseline diagnostics on the 15 MusicNet piano clips show approximately **44.83 ms onset error and 138.10 ms release error** over 1,055 fixed matched pairs. These are conditional errors on matched notes, not overall accuracy; missing notes are excluded, and reference release semantics still matter. V70 leaves retained clocks identical. More removal-only fine-tuning cannot reduce those timing errors.

Repeated development against the same validation material also makes it development data, not a fresh final test. Do not relax the preservation gates or rebrand it as untouched evaluation to obtain a release.

## What remains unknown

The dominant acoustic failure on actual full songs has **not** been established. Existing diagnostics do not preserve a complete, aligned trace from separated sources through every detector and filter to the final arrangement. We cannot yet rank separation leakage, raw pitch mistakes, early over-filtering and vocal-decoder failures by contribution. No user recordings were inspected, and this audit cannot identify which source produced a specific wrong note the user heard.

The confirmed explanation for the stalled experiments is narrower and defensible: the training target excludes melody and timing, does not cover all left-staff notes, and is evaluated before transformations that demonstrably damage correct notes.

## Concrete next work, in order

1. **Fix and validate rhythm handling before more model training.** Develop an offline alternative to the single hard grid that preserves repeated-note intervals, incomplete triplets and local tempo changes, with independent treatment of attacks and releases. Keep original event times as the reference. Do not globally loosen detector thresholds; include both genuine short notes and short false fragments in comparisons. Retain the current production path until the alternative passes controls and representative audio evaluation.
2. **Add opt-in stage traces to an offline benchmark runner.** Record source role, stable candidate identity, incoming/outgoing pitch and times, and deletion/modification reason after decoding, each filter, cleanup, arrangement and score playback. Tag low notes from both `bass` and `other`. Do not run this on queued uploads or enable it on the website as part of this audit. Measure missing notes as well as extra notes; otherwise deleting difficult notes can look like progress.
3. **Assemble a small product-relevant benchmark before selecting another model.** Proposed initial scope: 12 legally usable, verified 15–30 second excerpts across six families—repeated bass, swing/triplet fragments, sustained chords, sparse accompaniment, dense separated mixes, and changing tempo. Keep all excerpts and variants of a composition in one partition. Use development material for diagnosis and reserve separate compositions for a single frozen final comparison. This benchmark is proposed, not yet assembled or licensed by this audit.
4. **Separate the two reference tasks.** Solo-piano audio needs verified notes, onsets and releases. Full mixes additionally need a musician-approved melody and playable piano reduction; an original multitrack MIDI file alone does not define the desired arrangement. Report timing tolerances, missing and extra notes, hold errors, and listening judgments independently.
5. **Select the target from stage attribution.** Wrong raw pitches call for decoder/acoustic-model work; notes lost by filters call for filter changes; correct notes damaged downstream call for cleanup/arrangement fixes. If low `other` notes dominate clutter, benchmark its verifier and selection objective instead of another bass-only support model. The correct target is not settled by increasing epochs.

Stop rule: no next training batch until this process shows which stage contributes most to the chosen failure category and supplies measurable baseline metrics. No new checkpoint is promoted merely because training loss falls or unit tests pass. The commercial-data and 25 GB storage constraints remain in force.

## Reproduction and verification

[Diagnostic runner](../scripts/diagnose_transcription_pipeline.py), [accounting tests](../scripts/test_diagnose_transcription_pipeline.py), [full measurements and source hashes](diagnostics/pipeline-audit-v1/report.json).

With existing backend dependencies:

```sh
PYTHONPATH=backend python scripts/diagnose_transcription_pipeline.py \
  --root . --output .training/pipeline-diagnostic-v1/report.json
```

No model assets are loaded by the runner. Temporary MIDI/MusicXML/playback fixtures are deleted at completion. The archived V70 diagnostic files are read, not rerun. The ten oracle fixtures and accompaniment probe completed; five diagnostic-accounting tests plus the relevant existing cleanup, rhythm, chord, notation and playback tests passed: **99 passed**, with two environment/deprecation warnings. Test success verifies accounting and supported behavior, not an accuracy improvement.
