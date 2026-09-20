# Learning to reject extra accompaniment notes

**2026-09-20 correction:** V2 used incorrect clocks for the trimmed Chopin recordings. Its old Vienna scores and the conclusion that all release checks passed are withdrawn. V2 remains the current runtime while corrected candidates are evaluated; none of this audit’s candidates is a demonstrated improvement over it. See the correction below.

This experiment targets false notes from Basic Pitch on the separated accompaniment. A small neural verifier estimates whether each detected note is supported by the audio. It learns from labeled events; it does not use a song title, filename, key signature, or a list of allowed notes. Vocal melody recognition and the dedicated solo-piano model are separate systems.

## Evidence and training

The 961-parameter network takes 26 pitch-relative features: note/onset activation strength, spectral energy, onset contrast, duration, velocity, nearby semitones and harmonics, and simultaneous activity. Basic Pitch and Demucs remain frozen. Training updates all verifier weights with AdamW, positive-weighted binary cross entropy, and balanced sampling across corpora.

Reference matching uses 50 ms onset tolerance and 50 cents pitch tolerance, without an offset requirement. An unmatched detection that overlaps a correct pitch or starts within 150 ms is ambiguous: it remains an error in evaluation but is excluded from training negatives. Otherwise, timing errors could teach the verifier to delete correct pitches. Evaluation rematches retained events one-to-one; duplicate predictions cannot both count as correct.

Selection uses validation loss for the checkpoint, followed by validation F1 for the probability threshold with separate recall/F1 guards for each corpus. Test references never enter gradient updates. Fewer notes alone do not count as improved accuracy.

## Data and rights

- **GuitarSet 1.1.0**, Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye and Juan P. Bello. [Publisher record](https://zenodo.org/records/3371780), CC BY 4.0. Real microphone audio and hexaphonic-derived note annotations. Players 00–03 train; 04 validates; 05 tests. Exclude publisher-reported annotation problems in `04_BN3-154-E_comp`, `04_Jazz1-200-B_comp`, and `02_Funk2-119-G_comp` before prediction. This is performer-disjoint for the verifier; shared progressions and possible Basic Pitch pretraining overlap limit independence.
- **40 original piano phrases**, procedural compositions rendered by MuseScore. Seeds 00–31 train, 32–39 validate. Reference events come from the same MusicXML used for rendering. These are synthetic training examples, not evidence of real-performance accuracy. The installed MuseScore General soundfont has MIT/PD/CC0 provenance; preserve the bundled soundfont attribution.
- **Oxford MIDItest**, A. Sophia Koepke, Olivia Wiles, Yael Moses and Andrew Zisserman, *Sight to sound* (ICASSP 2020). [Publisher page and license](https://www.robots.ox.ac.uk/~vgg/research/sighttosound/), CC BY 4.0. Eight actual digital-piano MIDI/audio pairs, not PianoYT pseudo-labels. Initially a separate instrument test; after the first failure it becomes a consumed regression benchmark.
- **Vienna 4x22**, Werner Goebl. [Publisher record](https://datasets.mdw.ac.at/datasets/dataset/98ea25fa-2468-43ff-929b-3c926e163583), [DOI 10.21939/4X22](https://doi.org/10.21939/4X22), CC BY 4.0. Real acoustic-piano performances with instrument-recorded labels. Players 01–14 train, 15–18 validate, 19–22 were originally a test and are now consumed regression recordings. Exclude synthesized average performer 23. Scores and recording instrument are shared across performers; this is not composition- or instrument-disjoint.

No user uploads, commercial song recordings, MAESTRO, CSD, or MAPS enter this new training run. Local user recordings may be inspected as unlabeled diagnostics; note-count changes on them are not accuracy measurements.

## Clocks and separation

The Oxford audio and MIDI have roughly 0.19–0.39 s clock offsets. A fixed CQT spectral-onset alignment uses only seconds 0–15 and searches -0.6 to +0.2 s in 5 ms increments. Only onsets in seconds 15.25–29.75 are evaluated; calibration data is excluded. The clock fit uses no transcription model predictions. Clock audit files retain every correction. Vienna timing is separately audited before training.

The source-separation stress set adds seeded unpitched percussion at -3 dB relative RMS, then runs the same Demucs `other` stem used by the website. Select the first comp/solo recording in each of five GuitarSet genres, separately for validation player 04 and test player 05. This controls the true pitched notes; it does not simulate every commercial mix.

## First attempt: rejected

Training: 271 clips, 36,688 unambiguous candidate events, 50 epochs × 80 updates. Minimum validation loss selected epoch 40; the initial 2-percentage-point validation recall guard selected threshold 0.58.

| Initial test | Precision, before → after | Recall, before → after | F1, before → after |
| --- | --- | --- | --- |
| GuitarSet player 05 | 74.26% → 83.19% | 84.13% → 82.55% | 78.89% → 82.87% |
| Oxford digital piano | 82.32% → 95.55% | 87.39% → 78.15% | 84.78% → 85.98% |

Guitar false positives fell 42.8%, but Oxford lost 9.24 percentage points of recall, exceeding the predeclared maximum of 2. **This checkpoint was not deployed.** Its tests are consumed. The second attempt adds real piano training and tightens validation recall loss to at most 1 percentage point. Repeated measurements on the first tests must be labeled regression results, not fresh held-out success.

## Historical V2 release (Vienna assessment invalid)

The expanded network was trained from scratch for 50 epochs × 80 updates on 327 clips and 45,825 unambiguous candidate events. There are 92 validation clips, including ten separated accompaniment examples. Epoch 48 had the lowest validation loss. Threshold 0.22 passed the fresh Vienna test and separated-guitar test, but its Oxford regression recall loss was 3.08 percentage points, so that candidate was also rejected.

The released weights are the expanded network, recalibrated **using validation only** to require recall loss no greater than 0.25 percentage points in every validation corpus. This selected threshold **0.10**. This policy change followed observed test failures: all previously scored tests are now explicitly regression benchmarks. No test audio enters training. A final, previously unscored 30–60 s section of each Oxford recording was checked after this threshold was frozen. Those sections contain new events but share performers and recordings with the earlier test.

| Final check | False detections, before → after | Correct detections, before → after | F1, before → after |
| --- | --- | --- | --- |
| Guitar regression, 60 recordings | 2,303 → 1,949 | 6,644 → 6,635 | 78.89% → 80.52% |
| Vienna piano regression, 16 recordings | 1,577 → 1,448 | 1,231 → 1,227 | 44.28% → 45.22% |
| Oxford piano regression, eight excerpts | 67 → 44 | 312 → 310 | 84.78% → 87.20% |
| Separated accompaniment regression, ten excerpts | 413 → 375 | 996 → 996 | 77.36% → 78.52% |
| Previously unscored Oxford sections, eight excerpts | 105 → 74 | 623 → 615 | 87.50% → 88.81% |

**Historical conclusion, withdrawn after the clock audit:** All original release guards appeared to pass at this conservative setting: guitar false positives decrease at least 10% and precision improves; every corpus has nondecreasing F1 and recall loss at most 2 percentage points. The final section test reduced false detections 29.5% with 1.15 percentage points less recall. The conservative verifier improves precision modestly and does not solve missing notes or all wrong notes. The low Vienna baseline was subsequently traced to invalid clocks; it cannot support a detector-accuracy conclusion. These numbers are not full-song arrangement accuracy.

The historical V2 run used the publisher's per-recording `FirstOnsets.txt` anchors to align the first MIDI onset to audio, then scored 0.25–29.75 s. **Do not reuse that clock method for the trimmed Chopin audio.** Exclude `1st-3rd` special Ballade variants as well as performer 23, avoiding duplicate performances. An initial exploratory CQT alignment failed before any Vienna feature extraction/training; it was replaced by the supplied anchors, whose offsets are retained in the audit. No test-prediction-derived clock adjustment is used.

### Runtime scope

The checked checkpoint runs on the `other` accompaniment stem for **every balanced Full song upload**, independent of filename, song or codec. Basic Pitch produces unconstrained acoustic evidence, then the existing accompaniment pitch bounds are applied to candidate events. This preserves harmonic evidence that Basic Pitch's frequency constraint would otherwise zero out. The verifier only removes candidates: retained notes keep their pitch, onset, release and velocity. It runs on the existing bounded windows, so recording length does not increase its audio-buffer size.

The Detailed option, solo-piano CRNN, vocal melody model and bass detector retain their existing recognition paths. No evidence here establishes that the verifier is safe on those other candidate distributions. `analysis.accompaniment_verification` reports its model and window-level candidate/rejection counts; overlap-context notes can appear in multiple windows, so these counters are not unique final-score note totals. Existing completed jobs keep their original artifacts and need retranscription to use the model.

The production checkpoint SHA-256 is `2b75ac2d3c80a2644c7df5c0ec5fab9c10086ff2fa072177868e7e6c3091fe60`. [Run records](../training/results/accompaniment-verifier-v2) retain the rejected attempts, selection history, licenses, source checksums, reference hashes, per-recording results and promotion decision.


## Corrected labels and retraining audit

The Chopin WAV files start at the music, while the publisher's older FirstOnsets anchors include removed silence. This shifted correct piano notes away from their labels and taught V2 to reject some valid notes. Mozart and Schubert retain the initial silence. The corrected preparation treats those cases separately, refines clocks from CQT onset evidence in seconds 0–15, and supervises/evaluates only seconds 15.25–29.75. Calibration does not use transcription predictions. All 88 clock corrections are recorded; residual spectral adjustments are between −0.005 and +0.020 seconds.

A second issue was candidate decoding. V2 applies pitch bounds after unconstrained MIDI decoding, which is not identical to Basic Pitch's original bounded decoder. Corrected experiments decode copies with the original bounds, preserving the unmodified acoustic arrays for features. This helper is tested against the original decoder. It is **not wired into the current runtime**, because the replacement candidates were not promoted.

Retraining uses 327 clips and 41,744 unambiguous candidates, with 92 validation clips. No test recording enters fitting. The corrected Vienna composition-level detector F1 ranges from 0.788 to 0.863 across training and validation groups; a new label audit rejects grossly misaligned groups before fitting.

- **Neural candidate:** 961 parameters, 50 epochs × 80 updates, positive loss weight 4; epoch 48, validation-selected threshold 0.02. One guitar recording regressed: one correct and one false note were removed, slightly reducing its F1. Rejected.
- **Tree candidate:** 128 ExtraTrees, maximum depth 12, minimum leaf size 8; equal corpus/recording weights and positive weight 4. Validation selected threshold 0.12. One Oxford excerpt lost a correct note without removing a false note. Rejected.
- **Consensus:** reject only when both frozen candidates agree at their unchanged thresholds. This post-failure development experiment retains all 9,825 originally correct detections across 102 regression excerpts, but removes only 13 false notes. It leaves substantially more false detections than deployed V2, so it is also rejected.

| Regression corpus | Original correct / false notes | Neural correct / false | Tree correct / false | Deployed V2 correct / false |
| --- | --- | --- | --- | --- |
| Guitar, 60 excerpts | 6,646 / 2,309 | 6,645 / 2,253 | 6,646 / 2,254 | 6,635 / 1,947 |
| Vienna piano, 16 excerpts | 1,248 / 222 | 1,248 / 217 | 1,248 / 213 | 1,216 / 173 |
| Oxford piano, eight excerpts | 312 / 67 | 312 / 66 | 312 / 63 | 310 / 44 |
| Later Oxford sections, eight excerpts | 623 / 104 | 623 / 96 | 622 / 100 | 615 / 73 |
| Separated guitar, ten excerpts | 996 / 413 | 996 / 403 | 996 / 408 | 996 / 375 |

All systems above use the same corrected references and scoring windows. V2 has its actual post-decoding pitch constraint; the new candidates use the original bounded decoder. These are consumed regression sets, not a fresh generalization benchmark. The table exposes the tradeoff: current V2 removes more false notes but also loses correct notes, especially on acoustic piano. The corrected candidates have not solved both problems simultaneously. Durations, perceptual arrangement quality, and commercial mixed-song accuracy are not established by these onset/pitch metrics.

The new release gate checks each recording against the original detector, then requires nondecreasing precision, recall and F1 against both the original detector and current release in every corpus, plus an actual false-note reduction versus the current release. This stricter gate was added after observing the failures; it is an engineering regression requirement, not a pre-registered scientific result. No runtime model or deployed note-decoding behavior changed in this audit. [Reproducible results and rejection decisions](../training/results/accompaniment-clock-correction/).
