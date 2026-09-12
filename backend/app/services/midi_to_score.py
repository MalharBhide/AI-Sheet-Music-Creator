import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from music21 import (
    bar,
    chord,
    clef,
    instrument,
    key,
    layout,
    metadata,
    meter,
    midi,
    note,
    stream,
    tempo,
    tie,
)

from app.models import ScoreOptions


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    pitches: tuple[int, ...]
    tied_from: frozenset[int] = frozenset()
    tied_to: frozenset[int] = frozenset()


def _read_notes(midi_path: Path, grid: int) -> tuple[list[dict], int]:
    """Read note events directly, without building and then dismantling MIDI measures."""
    source = midi.MidiFile()
    source.open(str(midi_path))
    try:
        source.read()
    finally:
        source.close()
    grouped: list[dict] = [defaultdict(set), defaultdict(set)]
    last_tick = 0
    for track in source.tracks:
        elapsed = 0
        active = defaultdict(deque)
        for event in track.events:
            if isinstance(event, midi.DeltaTime):
                elapsed += event.time
            elif event.isNoteOn():
                active[(event.channel, event.pitch)].append(elapsed)
            elif event.isNoteOff():
                starts = active[(event.channel, event.pitch)]
                if not starts:
                    continue
                onset = starts.popleft()
                pitch = event.pitch
                if not 21 <= pitch <= 108:
                    continue
                start = max(0, int(math.floor(onset * grid / source.ticksPerQuarterNote + 0.5)))
                end = max(start + 1, int(math.floor(
                    elapsed * grid / source.ticksPerQuarterNote + 0.5)))
                grouped[0 if pitch >= 60 else 1][(start, end)].add(pitch)
                last_tick = max(last_tick, end)
    return grouped, last_tick


def _chord_segments(groups: dict) -> list[_Span]:
    """Fold dense polyphony into chords, tying only pitches that keep sounding.

    A note boundary splits the chord, so every attack and release is retained.
    A repeated piano pitch starts a new attack even if an earlier detection of the
    same key overlaps it. This avoids requiring unsupported fifth/sixth voices.
    """
    events = defaultdict(lambda: (Counter(), Counter()))
    for (start, end), pitches in groups.items():
        events[start][0].update(pitches)
        events[end][1].update(pitches)
    active: Counter = Counter()
    previous = 0
    tied_from: frozenset[int] = frozenset()
    spans = []
    for position, (starts, stops) in sorted(events.items()):
        sounding = frozenset(active)
        active.subtract(stops)
        active.update(starts)
        active = +active  # Remove zero counters after note-offs.
        continuing = sounding.intersection(active).difference(starts)
        if sounding and position > previous:
            spans.append(_Span(previous, position, tuple(sorted(sounding)), tied_from,
                               frozenset(continuing)))
        tied_from = frozenset(continuing)
        previous = position
    return spans


def _voices(groups: dict) -> list[list[_Span]]:
    voices: list[list[_Span]] = []
    ends: list[int] = []
    for (start, end), pitches in sorted(groups.items()):
        voice_index = next((i for i, stop in enumerate(ends) if stop <= start), None)
        if voice_index is None:
            if len(voices) == 4:
                return [_chord_segments(groups)]
            voice_index = len(voices)
            voices.append([])
            ends.append(0)
        voices[voice_index].append(_Span(start, end, tuple(sorted(pitches))))
        ends[voice_index] = end
    return voices or [[]]


def _add_span(voices: list[stream.Voice], span: _Span, bar_ticks: int, grid: int,
              tonal_key: key.Key | None = None) -> None:
    """Split a sounding event at barlines and attach ties to individual chord notes."""
    position = span.start
    while position < span.end:
        measure_index, offset = divmod(position, bar_ticks)
        stop = min(span.end, (measure_index + 1) * bar_ticks)
        notes = []
        for pitch in span.pitches:
            item = note.Note(pitch)
            if tonal_key is not None:
                # MIDI stores no spelling. Prefer the estimated key's diatonic
                # spelling, then its accidental direction for chromatic notes.
                diatonic = next((p for p in tonal_key.pitches if p.pitchClass == pitch % 12), None)
                if diatonic is not None:
                    item.pitch.name = diatonic.name
                    # Changing B to C-flat (or C to B-sharp) can cross an octave.
                    item.pitch.octave += (pitch - item.pitch.midi) // 12
                elif tonal_key.sharps < 0 and item.pitch.accidental and item.pitch.accidental.alter > 0:
                    item.pitch = item.pitch.getEnharmonic()
            before = position > span.start or pitch in span.tied_from
            after = stop < span.end or pitch in span.tied_to
            if before or after:
                item.tie = tie.Tie("continue" if before and after else "stop" if before else "start")
            notes.append(item)
        item = notes[0] if len(notes) == 1 else chord.Chord(notes)
        item.quarterLength = (stop - position) / grid
        voices[measure_index].insert(offset / grid, item)
        position = stop


def midi_to_musicxml(midi_path: Path, xml_path: Path, options: ScoreOptions, title: str,
                     *, duration_seconds: float | None = None,
                     key_signature: str | None = None) -> None:
    """Quantize a whole recording into two piano staves, including its silent time.

    Ordinary polyphony uses independent voices. Dense passages use chord segments
    with individual pitch ties, so MuseScore's four-voice limit never drops notes
    or rejects a recording. Notation is prepared one measure at a time to avoid
    repeatedly scanning a full recording while filling rests and splitting ties.
    """
    grid = 4 if options.grid == "sixteenth" else 2
    effective_tempo = options.tempo_bpm or 120.0
    tonal_key = None
    if key_signature:
        tonic, mode = key_signature.rsplit(" ", 1)
        tonal_key = key.Key(tonic, mode)
    grouped, last_tick = _read_notes(midi_path, grid)
    if duration_seconds is not None and math.isfinite(duration_seconds) and duration_seconds > 0:
        last_tick = max(last_tick, math.ceil(duration_seconds * effective_tempo / 60 * grid))

    score = stream.Score(id="piano-score")
    score.metadata = metadata.Metadata(title=title[:120], composer="")
    signature = meter.TimeSignature(options.time_signature)
    bar_ticks = int(signature.barDuration.quarterLength * grid)
    measure_count = max(1, math.ceil(last_tick / bar_ticks))
    staves = []
    for index, groups in enumerate(grouped):
        staff = stream.PartStaff(id="right-hand" if index == 0 else "left-hand")
        staff.partName = "Piano" if index == 0 else ""
        staff.insert(0, instrument.Piano())
        measures = [stream.Measure(number=i + 1) for i in range(measure_count)]
        measures[0].insert(0, clef.TrebleClef() if index == 0 else clef.BassClef())
        measures[0].insert(0, meter.TimeSignature(options.time_signature))
        if tonal_key is not None:
            measures[0].insert(0, key.Key(tonal_key.tonic.name, tonal_key.mode))
        if index == 0:
            measures[0].insert(0, tempo.MetronomeMark(number=effective_tempo))
        for voice_index, spans in enumerate(_voices(groups)):
            voices = [stream.Voice(id=voice_index + 1) for _ in measures]
            for span in spans:
                _add_span(voices, span, bar_ticks, grid, tonal_key)
            for measure, voice in zip(measures, voices, strict=True):
                voice.makeRests(fillGaps=True, timeRangeFromBarDuration=False,
                                refStreamOrTimeRange=[0, bar_ticks / grid], inPlace=True)
                measure.insert(0, voice)
        for measure_index, measure in enumerate(measures):
            staff.insert(measure_index * bar_ticks / grid, measure)
            measure.makeNotation(inPlace=True)
            # Measure.makeNotation materializes the contextual meter for beaming.
            # Keep it implicit after the first bar so export does not print a new
            # time signature (and courtesy signature) at every single barline.
            if measure_index:
                measure.timeSignature = None
        measures[-1].rightBarline = bar.Barline("final")
        staves.append(staff)
        score.insert(0, staff)
    score.insert(0, layout.StaffGroup(staves, name="Piano", symbol="brace", barTogether=True))
    score.write("musicxml", fp=str(xml_path))
