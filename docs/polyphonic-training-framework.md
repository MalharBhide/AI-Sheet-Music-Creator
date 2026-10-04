# Polyphonic training against website V9

Status: the four-profile batch completed (eight newly fitted heads). The frozen
validation winner removed one false note in validation and zero in the 175-clip
regression. It preserved every measured attack, hold and supplied pitch interval,
but adds no regression improvement, so all candidates are withheld.
Website V9 remains deployed;
no user upload, saved score or transcription job was processed for this work.

## Additional supervised data

The official [BabySlakh v2 record](https://zenodo.org/records/4603870) provides
20 synthesized instrumental songs with aligned source MIDI and audio, licensed
CC BY 4.0. The 882,818,115-byte archive matched publisher MD5
`311096dc2bde7d61c97e930edbfc7f78`. Credit Ethan Manilow, Gordon Wichern,
Prem Seetharaman and Jonathan Le Roux, *Cutting Music Source Separation Some
Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity*,
WASPAA 2019. Source media remain local and are not shipped with the website.

Publisher [split/duplicate metadata](https://github.com/ethman/slakh-utils/tree/3f62d6a4b0e5952237dd5178ac3513e42161ec0a/splits)
is pinned to that revision. A project-specific, hash-seeded partition assigns
12 source groups to training, four to validation and four to reserved testing.
This is not the publisher's official Slakh benchmark split. Duplicate groups
stay together; near-duplicate/composition overlap and upstream model pretraining
are not audited. The small sample does not establish real-song generalization.

Offsets 0/30/60/90 seconds are fixed before inference: **48 training / 16
validation clips**, 22,893 note-release annotations and 23,532 pitch-support
intervals. No test-group features are prepared. Actual non-bass/non-drum source
stems are summed into an oracle accompaniment submix, then MP3-encoded at fixed
48/128/192 kbps cycles. This broadens instrument/polyphony conditions but is not a
Demucs separation benchmark. Existing separated GuitarSet examples stay in the
training/validation cohort. Every new MP3 clock check reports zero sample lag.

The prototype's rendered/saved flags are false despite actual paired files.
Preparation checks real file presence, matching sample clocks and MIDI bounds,
and records the discrepancy. It uses the exact per-source MIDI specified by the
publisher, avoiding the original unmodified `all_src.mid` labels. Source hashes,
rights, partition rules and codec evidence are [archived](../training/results/slakh-training-v1).

## Holds and accuracy protections

Source key releases remain separate from pitch support derived from CC64 pedal
events. Pedal support stops at release or the next same-key attack; onset/offset
references retain genuine repeated key presses. Holds that began before the
clip/crop remain in pitch support without gaining an invented attack label.
Waveform correlation verifies codec timing on a fixed energetic segment, even
when the intro is silent; no model prediction moves annotations.

New classifiers run conceptually after V8 note filtering and before V9 boundary
merging. Evaluation reapplies the actual frozen V9 boundary model to each proposed
keep mask, and compares final events against the **complete V9 baseline**. It
requires matched attacks, offsets, supplied pitch coverage and no per-recording
extra-note increase at both raw and half-margin thresholds. Only a frozen
validation winner can consume regression. The regression includes all latest V9
piano and MP3-stress clips. Reserved Slakh test groups remain untouched unless
that frozen winner passes positive regression improvement.

Pair-confidence caching reuses identical frozen evidence in threshold sweeps;
changed acoustic inputs invalidate it, and newly formed neighbor pairs get real
model inference. Tests reject deletion of a V9 hold fragment even when onset
counts stay equal, including crop-crossing holds absent from onset references.

**329 backend/training tests passed; six opt-in audio cases skipped.** Dataset,
pedal, crop, codec and complete-V9 baseline tests passed. These checks establish
contracts, not an accuracy improvement. Four paired profiles fitted on 708
training clips / 75,279 eligible labeled events, with 184 validation clips.
Weighted validation loss decreased within each profile, but this did not translate
into release-worthy note accuracy. Only `recall` qualified for regression at the
frozen 0.015/0.1 thresholds. No test-group inference was performed. All profile
curves, searches, selection and regression results are
[archived](../training/results/polyphonic-consensus-v10).
