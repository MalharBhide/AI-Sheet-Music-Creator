# Repeated-note timing and the attack-timing experiment

The precise score decoder now preserves supported 32nd-note runs. Previously,
even correct fast detections were rounded to a straight-sixteenth grid: a run
of 48 attacks at eight notes per beat became 25 score attacks, roughly halving
the repeated-note rate. This was a score-decoding error downstream of the ML
model, not evidence that its pitches were correct.

A finer subdivision requires at least four adjacent distinct 32nd positions
within a beat and less than half the straight-grid squared timing error. Chord
copies, sparse off-grid notes and loose jitter cannot supply that support.
Recognized triplets keep their existing priority. Finer phase alignment also
requires at least two supported beats and 75% coverage; an unsupported rhythm
keeps the existing grid. Simple eighth-note mode remains intentionally coarse.

Cleanup and MusicXML now share a 24-tick quarter-note lattice. It represents
straight sixteenths, supported 32nds and eighth triplets exactly; it does not
turn every note into a tiny subdivision. MusicXML, exported MIDI and playback
preserve the fast attacks without another rounding pass. Staff size, note size,
page spacing and automatic pagination retain their original settings.

Six original symbolic fixtures at 80, 120 and 160 BPM, with aligned and shifted
attacks, retain all 48 notes with essentially zero spacing error. Twelve
quarter/eighth/sixteenth/triplet controls have exactly unchanged event values.
Three actual MusicXML/MIDI round trips retain all 48 attacks and their spacing.
These are controlled decoder/notation results, not arbitrary-song accuracy.
The fix cannot recover attacks the audio model never detected or correct
wrong pitches, swing, incomplete fast groups, rubato or varying tempo.

## Actual supervised timing fitting: not deployed

A new three-class attack classifier was fitted on 314 existing commercial-
compatible Vocadito/VocalSet training clips with corrected annotation clocks.
Its 7,953 event examples learn a 20 ms advance, no change or a 20 ms delay from
131 observed pitch-relative acoustic features. Equal corpus/recording weighting
prevents long scale exercises from dominating. Unmatched detections default to
no shift; positive training targets must preserve matched attacks, offset-aware
holds and every reference's supplied pitch coverage. A touching previous release
moves with the attack; pitches and event counts remain fixed, with no overlaps
or invalid durations. The research pickle remains local and the website never
loads it.

Validation alone chose confidence .9 and full strength, adjusting seven attacks
across 56 other-singer clips without losing attacks, holds or pitch coverage.
Mean onset error on fixed baseline matches changed 31.763 → 31.696 ms for
Vocadito and 66.631 → 66.411 ms for VocalSet. These tiny gains exclude unmatched
notes and are not overall transcription-accuracy scores.

The single frozen winner also preserved all 54 previously consumed singer
regressions, but adjusted zero attacks there. No threshold was changed or
alternative candidate evaluated from that outcome. The trained classifier stays
offline: meaningful independent and final-score improvement is not established.
The released change is the repeated-note score decoder; V16 bass and V1 vocal
weights remain unchanged. Existing dataset attribution is in
`docs/context-melody-experiment.md` and `docs/melody-model-card.md`.

Plans, fitting settings, selected checkpoint hash, per-recording validation,
consumed results and release decision are archived under
`training/results/melody-attack-timing-v2`. The controlled decoder benchmark is
under `training/results/fast-rhythm-v1`. No user upload was transcribed and no
saved user score was regenerated.

The release passes 606 application/training tests and all six native pipeline
and renderer tests. Live checks confirm the new timing clock and unchanged
V16/V1 production weights on the healthy website, with an idle queue and the
original note-size settings. The backend was rebuilt after the checks passed.
