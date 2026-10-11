# Released V38 training evidence

Two fresh 40-epoch release-only models, 781 training / 225 validation clips,
including a newly acquired public-domain Mutopia Joplin subset. One validation
winner was frozen before all 519 consumed regressions and the 44-clip first pass.
Every preservation gate passes at normal and stricter settings.

The portable NPZ is shipped under backend/app/assets with its license.
Raw audio, feature caches and PyTorch checkpoint tensors are not committed.
Checkpoint hashes, original MIDI hashes/license pages, source witnesses, fitting
curves, classification, every recording-level result, runtime parity, native
routing and functional test logs are retained here.

This model makes bounded independent note-ending corrections. It does not
improve wrong-note counts or onset rhythm; those stay unchanged. See
[the report](../../../docs/chord-release-training-v38.md) for measured limits.
Deployment status and final identities are recorded in release-checks.json.
