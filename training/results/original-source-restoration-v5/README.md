Baseline and piano feature restoration
======================================

All 45 PDMX piano observations pass the unchanged native loader. All 108 chorale-derived piano observations pass the original strict audit, with zero codec lag and bit-identical samples against the original native rendering preflight.

BabySlakh: original full archive MD5 verified, all 389 original fitting files match archived SHA256 values; reserved source files were not extracted. PDMX original inputs pass archived MD5 and SHA256 checks; only the two already approved CC0 piano sources are retained. Large temporary source archives were removed after validation.

Baseline audio: 256 authored and 297 GuitarSet waveforms match archived original file SHA256 values exactly. Guitar WAV recovery changes only the four libsndfile PEAK creation-timestamp bytes; no musical sample changes. A bounded initial search failed on one file, whose later original timestamp was subsequently recovered with an exact full-file hash; all 297 are now verified. This restores an existing baseline, without adding guitar data or changing its loss weight. Native restoration of 54 Vienna piano waveforms is running.

Docker was confirmed stopped (exit137) and restarted for training only. Website containers remain stopped. All supplemental feature groups are now complete, but the 908-record core feature baseline still requires reconstruction. Model fitting, full accuracy qualification and website deployment have not happened. No improved accuracy is claimed. User uploads, saved scores, reserved sources and evaluation clips were not processed.
