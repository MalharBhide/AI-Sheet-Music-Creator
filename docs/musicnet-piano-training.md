# Bounded original MusicNet piano training

The original [MusicNet publisher record](https://zenodo.org/records/5120004)
identifies CC BY 4.0 and supplies real classical recordings with aligned
note/instrument labels. Credit John Thickstun, Zaid Harchaoui and Sham M. Kakade,
University of Washington. Per-recording performance/transcription provenance is
retained from the publisher metadata. This is the original release, not
MusicNetEM or a wrapper repository with a different license.

`acquire_musicnet_subset.py` retains only short solo-piano excerpts while
streaming the complete 11,097,394,998-byte archive through the publisher MD5
`844764911fa0d5b97c97da944a057590` and a separate SHA256. It also checks the live
publisher's record, license and file identity. No complete archive is stored.
Only after complete gzip/footer, byte-count and checksum verification can a
source manifest be written or consumed by the observer. Partial transfers are
unusable for fitting.

Metadata alone selects 32 works: evenly spaced catalog entries, up to 12 per
composer, one lowest-ID recording per work. Works remain disjoint: first
two-thirds train, next one-sixth validation, remaining works reserved. Only
original `train_data`/`train_labels` members are retained. If a selected recording
is original test-only it is excluded without replacement. Three 20-second crops
begin at 15, 35 and 55 seconds. No predictions or label outcomes select clips.
This project split does not establish performer or upstream pretraining
independence and is not the official benchmark.

The streaming RIFF parser preserves original 44100-Hz samples and crop clocks,
accepts bounded PCM/float formats, rejects malformed sizes, archive links and
unsafe paths, and never loads an entire performance into memory. Source crops,
original CSV labels, complete source hash, metadata, attribution and split plans
remain locally available. Eight acquisition tests cover exact float sample
preservation, corrupt/truncated formats, unsafe members, checksum refusal and
work split invariance.

`prepare_musicnet_v32.py` will observe verified training/validation excerpts
through the released V32 route after MP3 encoding at 48/128/192 kbps. It records
the same coarse, attack, release, recurrence, neighbor and context evidence used
by current studies. Reserved works receive no decoder inference. The fitting
reader rejects reserved access before reading files and checks cache/source
hashes, window shapes and partition identities.

Publisher labels use sample endpoints; these must match the original waveform
clock and solo-piano instrument identity. Notes already held when a crop begins
are protected pitch support, not fabricated new attacks. Exact attacks inside
the crop remain separate. Because the publisher estimates approximately 4%
label error and supplies no exact pedal/key-release truth, pitch-support edges
receive .05-second preceding/.1-second following ambiguity protection. This is
conservative supervision, not a claim of perfect ground truth. Eight label and
reader tests verify these boundaries and reject invalid clock/instrument/pitch
values and unverified transfers.

Acquisition is currently in progress. No MusicNet recording has entered fitting
or validation selection yet. V32 completed a spectral-feature fit on the already verified licensed corpus
and passed all release gates while this transfer runs.
Future MusicNet candidates require separately frozen training/validation plans,
all consumed regressions, untouched first-pass source groups and runtime/native
release checks. No user audio or saved scores are processed, and all eight preceding V25 stages and the original engraving scale remain
unchanged; the gated V32 verifier is deployed as a final additional stage.

The next fit freezes the **complete released V32 network**, including its learned
spectral/recurrence correction and all 155 normalization values. A new neutral
branch begins with identical V32 probabilities and learns only from the combined
licensed fitting corpus and the newly acquired real piano excerpts. Model weights
and source IDs are never inputs to the new feature branch. Two profiles and four
evaluation epochs are declared before fitting; validation selects one winner.

The baseline fitting caches contain 908 recordings (702 train / 206 validation).
The released V32 verifier removed 21 notes during this baseline preparation,
without changing retained event clocks or acoustic windows. A separate consumed
regression cache contains all 469 recordings from previous studies and first-pass
tests; its V32 baseline removed 17 notes. These labels cannot enter fitting.

The V33 plan must declare the identities and hashes of **every acquired reserved
MusicNet work and all three excerpts**, plus 32 new authored MP3 seeds, before
training starts. Declaring identities reads only metadata; reserved audio and
labels remain unused until a frozen winner passes the full consumed regression
set. Prior NSynth/GMD first-pass sources are now consumed regressions and cannot
be reported as fresh evidence. Both confidence settings must reduce wrong notes
and preserve attacks, holds, pitch coverage, and fixed-pair timing per recording.
A failed first pass is retained, with no replacement clips or confidence retuning.

Optional V33 runtime modules are now prepared but **not wired into the website**.
They load an explicit hashed format-4 asset, retain the exact released physical
feature extraction, reject malformed weights/normalization/confidence values, and
keep the same event eligibility and guardian protections. Random nonzero model
fixtures verify exact prediction parity with the training implementation.

`export_musicnet_anchor_v33.py` refuses failed consumed or first-pass evidence and
requires every predeclared reserved/test identity. `verify_musicnet_v33_runtime.py`
will compare all fitting, consumed, and first-pass recordings, both confidence
masks, and independently recomputed physical features. Runtime and export source
hashes are sealed; a successful export does not authorize deployment by itself.
The actual candidate still requires these checks and native pipeline verification.

Backend checkpoint unit tests require Torch even without the full transcription
extras. CI now installs the project's pinned CPU Torch 2.5.1 runtime using the
[official CPU wheel index](https://pytorch.org/get-started/previous-versions/).
Training-to-runtime parity tests remain under `training/`, so backend-only tests
do not need to import training scripts. The local preparation suite passed 862
tests (6 skipped); the backend-only scope passed 326 (6 skipped). This validates
release preparation, not MusicNet accuracy or a new model release.
