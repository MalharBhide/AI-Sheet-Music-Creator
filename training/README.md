# Supervised melody training

This experiment trains a small temporal neural decoder on **real human-annotated
singing**. The resulting checkpoint is used for the vocal melody of every
Full song upload, independent of filename, file extension or song identity. Basic Pitch, pYIN and CQT supply frozen acoustic features; the new
network learns piano pitch, silence and note attacks. It does not retrain Demucs,
the piano CRNN or a complete audio-to-arrangement model.

## Data and scope

| Corpus | Labels and intended use | Decision |
| --- | --- | --- |
| [Vocadito](https://zenodo.org/records/5578807) | 40 real singing recordings, frame pitch and two independent note annotations | Downloaded, checksum verified; first melody training/evaluation corpus |
| [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro) | 198.7 hours of aligned piano audio/MIDI | Future solo-piano fine-tuning; 101 GB download and CC BY-NC-SA 4.0 terms; not downloaded for this vocal experiment |
| [MedleyVox](https://github.com/jeonchangbin49/MedleyVox) | Multiple-singer separation evaluation | Relevant to separation failures, not a substitute for paired piano-arrangement labels |

Vocadito is CC BY 4.0; credit Rachel Bittner, Katherine Pasalo, Juan José Bosch,
Gabriel Meseguer Brocal and David Rubinstein. See the
[dataset documentation and license](https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/vocadito.html)
and [paper](https://arxiv.org/abs/2110.05580). The archive's metadata contains 29
distinct singer IDs, although the paper describes 28 volunteers. Splits follow
the supplied IDs. No user uploads or commercial recordings enter training.

Seed 1729 fixes 19 training, five validation and five test singer IDs, producing
26/8/6 recordings. No singer ID appears in two partitions. Reliable composition
IDs are absent; this is not a claimed composition-disjoint benchmark. Upstream
Basic Pitch/Demucs pretraining overlap has not been audited.

## Reproduction

Use the repository's pinned Python 3.11 transcription environment. Run from the
repository root (or use the backend Docker image with this repository mounted):

```bash
export PYTHONPATH="$PWD/backend:$PWD/training"
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2
python training/prepare_vocadito.py .training
python training/cache_vocadito.py .training
python training/train_melody.py .training --epochs 30
python training/train_melody.py .training --test
python training/stress_melody.py .training
python -m pytest -q training/test_melody_decoder.py
```

Data, feature caches and temporary runs stay in ignored `.training/`. The
download is pinned to Zenodo record 5578807 and publisher MD5
`dea40fd18f14d899643c4ba221b33a46`. Features use a common 50 Hz clock, including
Basic Pitch's actual output frame times. The 2,187-parameter network shares a
temporal classifier across all 88 piano pitches; its receptive field is 300 ms.
Training uses AdamW (learning rate .001, weight decay .0001), batches of four
256-frame windows, 60 updates per epoch, 30 epochs and gradient clipping at 3.
Loss combines frame cross entropy and .3 times weighted onset binary cross
entropy. Seeds and deterministic PyTorch operations are fixed.

Choose the epoch by validation frame cross entropy. Choose minimum duration,
onset threshold and smoothing by validation note F1. The final test command
checks the selected checkpoint hash, refuses to overwrite an existing test,
and never fits or selects settings on test recordings. Retraining a consumed
experiment is also refused. A new model revision needs new independent test
data; deleting a report does not make those recordings unseen again.

## Evaluation contracts

The baseline is the deployed balanced Basic Pitch vocal detector plus pYIN
refinement, before rhythm quantization or arrangement. Both systems receive the
same waveform. This comparison isolates melody recognition; it does not measure
the complete exported sheet music. Reference note CSVs store onset, **Hz** and
**duration**, not MIDI pitch and end time. The decoder rounds pitch to piano
semitones; evaluation compares against the original annotated frequencies.

[mir_eval](https://mir-eval.readthedocs.io/latest/api/transcription.html) measures
note matching with 50 ms onset and 50 cent pitch tolerance, and optional offsets
within max(50 ms, 20% of reference duration). Frame metrics use a 20 ms grid:
correct piano pitch over annotated voiced frames, missed melody as silence, and
false voicing over annotated silent frames. Results are macro averages across
recordings, with per-recording results retained. A1 is used for training; A2 is
an independent-annotator sensitivity check, not extra training data.

The separation stress test mixes held-out real singing with independently
generated bass/chords/percussion at −3 dB relative RMS, then uses the same Demucs
separator as the app. It uses the frozen checkpoint and settings. This tests
separation artifacts under controlled conditions; it does not establish
accuracy on commercial mixes, overlapping singers or rap. Synthetic contract
tests cover real rests, repeated notes, octave changes and short/long windows;
their pass count is not an accuracy score.

See [the framework](../docs/transcription-framework.md) for promotion gates and
the distinction between melody recognition and a complete piano arrangement.

The released checkpoint, measured improvements and known failures are documented
in the [model card](../docs/melody-model-card.md).
