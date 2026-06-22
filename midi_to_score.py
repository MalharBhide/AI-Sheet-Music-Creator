from __future__ import annotations

from copy import deepcopy
from pathlib import Path


class ScoreConversionError(RuntimeError):
    pass


def midi_to_piano_musicxml(
    midi_path: Path,
    output_path: Path,
    title: str = "Generated Piano Score",
    split_pitch_midi: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from music21 import (
            chord,
            clef,
            converter,
            instrument,
            key,
            layout,
            metadata,
            meter,
            note,
            stream,
            tempo,
        )
    except ImportError as exc:
        raise ScoreConversionError(
            "music21 is not installed. Install backend requirements first."
        ) from exc

    try:
        parsed_score = converter.parse(str(midi_path))
        chordified = parsed_score.chordify()
        events = list(chordified.flatten().notes)
        if not events:
            raise ScoreConversionError("No notes were detected in the transcribed MIDI.")

        time_signature = _first_or_default(
            parsed_score.recurse().getElementsByClass(meter.TimeSignature),
            meter.TimeSignature("4/4"),
        )
        key_signature = _detect_key_signature(parsed_score, key)
        tempo_mark = _first_or_default(
            parsed_score.recurse().getElementsByClass(tempo.MetronomeMark),
            tempo.MetronomeMark(number=120),
        )

        right = stream.Part(id="PianoRight")
        right.partName = "Piano"
        right.partAbbreviation = "Pno."
        right.insert(0, instrument.Piano())
        right.insert(0, clef.TrebleClef())
        right.insert(0, deepcopy(time_signature))
        right.insert(0, deepcopy(key_signature))
        right.insert(0, deepcopy(tempo_mark))

        left = stream.Part(id="PianoLeft")
        left.partName = "Piano"
        left.partAbbreviation = "Pno."
        left.insert(0, instrument.Piano())
        left.insert(0, clef.BassClef())
        left.insert(0, deepcopy(time_signature))
        left.insert(0, deepcopy(key_signature))
        left.insert(0, deepcopy(tempo_mark))

        for event in events:
            right_pitches, left_pitches = _split_event_pitches(event, split_pitch_midi)
            if right_pitches:
                right.insert(float(event.offset), _make_notation_event(event, right_pitches, note, chord))
            if left_pitches:
                left.insert(float(event.offset), _make_notation_event(event, left_pitches, note, chord))

        for part in (right, left):
            _quantize_part(part)
            part.makeRests(fillGaps=True, inPlace=True)
            part.makeMeasures(inPlace=True)
            part.makeNotation(inPlace=True)

        score = stream.Score(id="GeneratedPianoScore")
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = "AI Sheet Music Creator"
        score.insert(0, right)
        score.insert(0, left)
        score.insert(0, layout.StaffGroup([right, left], symbol="brace", barTogether="yes"))
        score.write("musicxml", fp=str(output_path))
    except ScoreConversionError:
        raise
    except Exception as exc:
        raise ScoreConversionError(f"Could not convert MIDI to MusicXML. {exc}") from exc

    return output_path


def _first_or_default(elements, default):
    items = list(elements)
    return deepcopy(items[0]) if items else default


def _detect_key_signature(parsed_score, key_module):
    try:
        detected_key = parsed_score.analyze("key")
        return detected_key.asKeySignature()
    except Exception:
        return key_module.KeySignature(0)


def _split_event_pitches(event, split_pitch_midi: int):
    pitches = list(getattr(event, "pitches", []))
    if not pitches and hasattr(event, "pitch"):
        pitches = [event.pitch]

    right_pitches = [pitch for pitch in pitches if pitch.midi >= split_pitch_midi]
    left_pitches = [pitch for pitch in pitches if pitch.midi < split_pitch_midi]

    if not right_pitches and left_pitches:
        return [], left_pitches
    if not left_pitches and right_pitches:
        return right_pitches, []
    return right_pitches, left_pitches


def _make_notation_event(source_event, pitches, note_module, chord_module):
    duration = deepcopy(source_event.duration)
    if len(pitches) == 1:
        notation_event = note_module.Note(deepcopy(pitches[0]))
    else:
        notation_event = chord_module.Chord([deepcopy(pitch) for pitch in pitches])
    notation_event.duration = duration
    return notation_event


def _quantize_part(part) -> None:
    try:
        part.quantize(
            quarterLengthDivisors=(4, 3),
            processOffsets=True,
            processDurations=True,
            inPlace=True,
            recurse=True,
        )
    except Exception:
        # Quantization is best-effort; music21 can still write unquantized durations.
        return
