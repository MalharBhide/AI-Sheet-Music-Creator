Remaining baseline reconstruction evidence
==========================================

54 real-piano and 60 labeled oracle-bass waveforms reproduce original archived SHA256 values. Together with the prior 297 guitar and 256 authored waveforms, 667 of 733 legacy baseline audio files are restored. One separated-bass pilot also reproduces its original SHA256 after correcting only the libsndfile PEAK creation timestamp; musical samples are unchanged. Recovery of all 66 separated files is running with exact hash acceptance and a bounded original timestamp search.

The V25 feature-recovery pilot uses the unchanged hash-pinned native decoder on a training fixture, never user audio or evaluation clips. Full-window decoding emits 95 notes versus the archived cropped cache's 94; removing the single event before the old 0.25-second window restores the count. Full cache bytes still differ. These results do not prove exact cache restoration, and no original cache guard or checksum is bypassed. The complete core baseline requires further validated reconstruction before model fitting.

No new candidate is trained, no accuracy improvement is claimed and no weights are deployed. Website code/assets remain unchanged. Training storage remains below 25 GB; the separated-wave recovery stage reserves 2 GB before starting.
