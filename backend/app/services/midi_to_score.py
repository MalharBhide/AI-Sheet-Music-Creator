import math
from collections import defaultdict
from pathlib import Path

from music21 import bar, chord, clef, converter, instrument, layout, metadata, meter, note, stream, tempo

from app.models import PipelineError, ScoreOptions


def midi_to_musicxml(midi_path: Path, xml_path: Path, options: ScoreOptions, title: str) -> None:
    """Quantize MIDI and construct two piano staves without losing overlapping durations.

    A fixed middle-C split is an MVP hand-assignment heuristic. Notes with equal
    quantized onset and length become chords. Overlapping groups occupy separate
    voices; music21 fills rests, makes measures, ties bar crossings and adds beams.
    """
    # The MIDI importer already inserts barline ties. Rejoin those fragments before
    # quantization so sustained notes are not turned into repeated attacks.
    source = converter.parse(str(midi_path), quantizePost=False).stripTies()
    ticks_per_quarter = 4 if options.grid == "sixteenth" else 2
    grouped: list[dict] = [defaultdict(set), defaultdict(set)]
    last_tick = 0
    note_count = 0
    for element in source.flatten().notes:
        start = max(0, int(math.floor(float(element.offset) * ticks_per_quarter + 0.5)))
        end = max(start + 1, int(math.floor(
            (float(element.offset) + float(element.quarterLength)) * ticks_per_quarter + 0.5)))
        for pitch in element.pitches:
            midi = pitch.midi
            if 21 <= midi <= 108:
                grouped[0 if midi >= 60 else 1][(start, end)].add(midi)
                last_tick = max(last_tick, end)
                note_count += 1
    if not note_count:
        raise PipelineError("No playable piano notes remained after cleanup.")
    if note_count > 20000:
        raise PipelineError("Too many notes were detected for this MVP. Try a shorter, clearer clip.")

    score = stream.Score(id="piano-score")
    score.metadata = metadata.Metadata(title=title[:120], composer="")
    signature = meter.TimeSignature(options.time_signature)
    bar_ticks = int(signature.barDuration.quarterLength * ticks_per_quarter)
    end_tick = math.ceil(last_tick / bar_ticks) * bar_ticks
    staves = []
    for index, groups in enumerate(grouped):
        staff = stream.PartStaff(id="right-hand" if index == 0 else "left-hand")
        staff.partName = "Piano" if index == 0 else ""
        staff.insert(0, instrument.Piano())
        staff.insert(0, clef.TrebleClef() if index == 0 else clef.BassClef())
        staff.insert(0, meter.TimeSignature(options.time_signature))
        if index == 0:
            staff.insert(0, tempo.MetronomeMark(number=options.tempo_bpm))
        voices: list[stream.Voice] = []
        voice_ends: list[int] = []
        for (start, end), pitches in sorted(groups.items()):
            voice_index = next((i for i, stop in enumerate(voice_ends) if stop <= start), None)
            if voice_index is None:
                # MuseScore supports four independent voices per staff.
                if len(voices) >= 4:
                    raise PipelineError("The recording is too dense to notate clearly. Try a simpler passage.")
                voice_index = len(voices)
                voices.append(stream.Voice(id=voice_index + 1))
                voice_ends.append(0)
            item = (note.Note(next(iter(pitches))) if len(pitches) == 1
                    else chord.Chord(sorted(pitches)))
            item.quarterLength = (end - start) / ticks_per_quarter
            voices[voice_index].insert(start / ticks_per_quarter, item)
            voice_ends[voice_index] = end
        if not voices:
            voices.append(stream.Voice(id=1))
        for voice in voices:
            voice.makeRests(fillGaps=True, timeRangeFromBarDuration=False,
                            refStreamOrTimeRange=[0, end_tick / ticks_per_quarter], inPlace=True)
            staff.insert(0, voice)
        staff.makeNotation(inPlace=True)
        if measures := list(staff.getElementsByClass(stream.Measure)):
            measures[-1].rightBarline = bar.Barline("final")
        staves.append(staff)
        score.insert(0, staff)
    score.insert(0, layout.StaffGroup(staves, name="Piano", symbol="brace", barTogether=True))
    score.write("musicxml", fp=str(xml_path))
