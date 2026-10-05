# Chords from supported simultaneous notes

New scores stack simultaneous pitches as a chord even when their endings differ.
Previously, notes needed identical starts **and** identical endings to share a
chord; otherwise ordinary passages used separate voices. The new engraving
retains each pitch's release using individual ties.

Only notes already detected on the same notation tick and staff are grouped.
No harmonic pitches are generated, no attacks are moved to form a chord, and
arpeggios remain separate. This improves notation; it does not establish that
every detected pitch is correct. Original 7 mm notation and default page/system
spacing are preserved.

Each complete tied chord remains in one voice until its final release. A melody
attack cannot steal the voice needed by a held chord tone. Independent overlapping
unisons retain separate voices and velocities. More than four voices still uses
the existing dense-polyphony fallback.

The generated four-bar comparison retains all 28 original pitches, attacks,
releases and velocities. The left hand changes from three voices with no chords
to one voice with six chord events. Both native PDFs remain one page. SVG score
positions retain the playback clock, including rests and tied-note boundaries.
Both PDFs were visually inspected.

The full suite passes 656 application/training tests plus six native pipeline
checks; 55 focused notation/playback tests pass. Tests cover unequal chord holds,
repeated chords, arpeggios, overlapping chords with shared pitches, long chord
tones under a moving melody, dynamics, fast repeats and unchanged note size.

The local backend is rebuilt and updated after an idle queue check. Live source,
health and exact V16/vocal weight hashes are verified. No user upload or saved
score is regenerated. Upload again to see the new chord engraving.

Evidence: [chord comparison](../training/results/chord-notation-v1/benchmark.json)
and [release checks](../training/results/chord-notation-v1/checks.json).
