# Completed full-encoding MusicNet V34 study — withheld

Two predeclared 80-epoch profiles learned from 765 training and 221 validation
recordings, including 63/15 original MusicNet piano excerpts. The complete
released V32 network and its normalizer stayed frozen. The new branch received
all frozen coarse, attack and release encodings, recurrence, context and hidden
features (955 values). No source IDs, labels or fitted probabilities were inputs.

Validation selected the conservative epoch-80 checkpoint at removal confidence
.85, guardian .3, and stricter confidence .925. It removed 5/1 validation false
notes while preserving every recording. The frozen winner preserved all 469
consumed recordings and removed 16/11 false notes. The one original first pass
preserved all 50 predeclared recordings and removed 1/1 false notes overall.

However, it removed **zero** false notes on the 18 reserved real-piano excerpts
at either setting. The separately predeclared real-piano release rule failed.
The aggregate authored gain cannot waive this rule. No confidence retuning,
runner-up, replacement first pass, portable export or deployment followed.
The website retains V32 and the original engraving size.

All 50 first-pass recordings are now consumed regression evidence. Future
candidates must cover all **519** consumed recordings and use genuinely untouched
real-piano sources for independent benefit testing. All eight Oxford piano/MIDI
pairs remain untouched. MusicNet reserved works cannot be described as fresh again.

Plans, curves, selected checkpoint hashes, complete per-recording evaluations and
test logs are archived here. Checkpoints, audio and physical feature caches remain
local, outside Git. The complete suite passed 878 tests (6 skipped); additional
real-piano gate tests are recorded separately. No user audio or saved scores were
processed. Neither training loss nor passing authored tests establishes better
transcription accuracy on real recordings.
