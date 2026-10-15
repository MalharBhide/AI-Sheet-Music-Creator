# Recover original inputs for approved-model fine-tuning

Restored all 28 historical reader sources using exact immutable hashes from
archived witnesses or the recorded Git revision. The unchanged historical guard
verified all 28 sources and 14 inherited model arrays. No production source was
replaced and no guard was relaxed.

The original NSynth archive passed its pinned size, MD5 and SHA-256. The restored
497-note subset manifest equals the previously archived manifest, including
identities, waveform hashes and instrument partitions. Also restored the existing
1,515-note dynamics subset (1,188 train, 327 validation); reserved instruments
remain excluded. Every extracted waveform hash was verified before deleting the
full archive. Attribution and CC BY 4.0 source witnesses are preserved.

An initial storage preflight rejected the replay's shared repository-evidence
symlink before any download. Storage audit now accepts only explicitly declared
links with exactly matching existing targets; all unknown links still fail. The
original rejection log is preserved. Eighteen focused and 1,536 full-suite tests
passed, with seven full-suite skips and five warnings.

The original piano archive stream and original 112-clip authored feature cache
restoration remain in progress. The latter covers quiet notes, short repeats,
long overlaps, chords and broken chords, using the same guarded source partitions.
These are original dataset/generated fixtures, never user uploads or saved scores.
Combined peak stage reservations remain below the 25 GB local training limit.
There is no dataset fitting result, accuracy claim, or new model deployment yet.
