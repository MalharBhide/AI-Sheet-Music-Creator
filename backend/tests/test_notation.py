import xml.etree.ElementTree as ET

import pytest
from music21 import chord, converter, note, stream, tempo

from app.models import ScoreOptions
from app.services.midi_to_score import midi_to_musicxml


def make_midi(path, *, empty_left=False):
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=120))
    part.insert(0, chord.Chord([60, 64, 67], quarterLength=1))
    part.insert(1, note.Note(72, quarterLength=5))  # overlaps melody and a bar line
    part.insert(2, note.Note(76, quarterLength=1))
    part.insert(3, note.Note(77, quarterLength=1))
    if not empty_left:
        part.insert(0, note.Note(48, quarterLength=6))
    score.insert(0, part)
    score.write("midi", fp=str(path))


@pytest.mark.parametrize("signature", ["4/4", "3/4", "6/8"])
@pytest.mark.parametrize("grid", ["eighth", "sixteenth"])
def test_real_musicxml_has_grand_staff_ties_chords_and_voices(tmp_path, signature, grid):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    make_midi(midi)
    midi_to_musicxml(midi, xml, ScoreOptions(time_signature=signature, grid=grid), 'A <test> & score')
    tree = ET.parse(xml)
    assert tree.findtext(".//staves") == "2"
    assert {x.text for x in tree.findall(".//clef/sign")} >= {"G", "F"}
    assert tree.findall(".//chord")
    assert tree.findall(".//tie")
    assert len({x.text for x in tree.findall(".//voice")}) >= 2
    assert tree.findtext(".//work-title") == 'A <test> & score'
    assert len(tree.findall('.//time')) == 1
    assert tree.findtext('.//time/beats') == signature.split('/')[0]
    parsed = converter.parse(str(xml))
    assert {p.midi for n in parsed.flatten().notes for p in n.pitches} == {48, 60, 64, 67, 72, 76, 77}
    # A sustained C5 survives overlapping melody events (and measure splits).
    held_duration = sum(float(n.quarterLength) for n in parsed.flatten().notes
                        if any(p.midi == 72 for p in n.pitches))
    assert held_duration == 5


def test_silent_left_hand_is_not_omitted(tmp_path):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    make_midi(midi, empty_left=True)
    midi_to_musicxml(midi, xml, ScoreOptions(), "Solo right hand")
    parsed = converter.parse(str(xml))
    assert len(parsed.parts) == 2
    assert not list(parsed.parts[1].flatten().notes)
    assert list(parsed.parts[1].flatten().getElementsByClass(note.Rest))


@pytest.mark.parametrize("duration, expected_quarters", [(None, 4), (0.01, 4), (7, 16)])
def test_silence_produces_a_score_of_rests(tmp_path, duration, expected_quarters):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream([note.Rest(quarterLength=1)])
    score.write("midi", fp=str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(), "Silence", duration_seconds=duration)
    parsed = converter.parse(str(xml))
    assert len(parsed.parts) == 2
    for part in parsed.parts:
        assert not list(part.flatten().notes)
        assert part.highestTime == expected_quarters
        assert list(part.flatten().getElementsByClass(note.Rest))


@pytest.mark.parametrize("signature", ["4/4", "3/4", "6/8"])
def test_dense_polyphony_preserves_attacks_and_durations_with_pitch_ties(tmp_path, signature):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream()
    for index in range(8):
        score.insert(index / 4, note.Note(60 + index, quarterLength=2))
    score.insert(0, note.Note(80, quarterLength=9))
    score.write("midi", fp=str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(time_signature=signature), "Dense passage")
    parsed = converter.parse(str(xml))
    attacks = []
    durations = {}
    for element in parsed.parts[0].flatten().notes:
        for pitched_note in element.notes if isinstance(element, chord.Chord) else [element]:
            pitch = pitched_note.pitch.midi
            durations[pitch] = durations.get(pitch, 0) + float(element.quarterLength)
            if pitched_note.tie is None or pitched_note.tie.type == "start":
                attacks.append((float(element.offset), pitch))
    assert sorted(attacks) == sorted([(index / 4, 60 + index) for index in range(8)] + [(0, 80)])
    assert durations == {**{60 + index: 2 for index in range(8)}, 80: 9}
    assert ET.parse(xml).findall(".//tie")
    # Each staff remains inside MuseScore's four-voice limit.
    for part in parsed.parts:
        assert all(len(measure.voices) <= 4 for measure in part.getElementsByClass(stream.Measure))


def test_dense_repeated_pitch_gets_a_new_attack(tmp_path):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream()
    for index in range(6):
        score.insert(index / 4, note.Note(60 + index, quarterLength=4))
    score.insert(2, note.Note(60, quarterLength=4))
    score.write("midi", fp=str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(), "Repeated pitch")
    parsed = converter.parse(str(xml))
    attacks = []
    for element in parsed.parts[0].flatten().notes:
        for item in element.notes if isinstance(element, chord.Chord) else [element]:
            if item.pitch.midi == 60 and (item.tie is None or item.tie.type == "start"):
                attacks.append(float(element.offset))
    assert attacks == [0, 2]


def test_leading_and_trailing_silence_preserve_recording_duration(tmp_path):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream()
    score.insert(5, note.Note(72, quarterLength=1))
    score.write("midi", fp=str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(tempo_bpm=60), "Silence around a note",
                     duration_seconds=10)
    parsed = converter.parse(str(xml))
    assert parsed.parts[0].flatten().notes.first().offset == 5
    assert [part.highestTime for part in parsed.parts] == [12, 12]


def test_more_than_twenty_thousand_notes_produce_musicxml(tmp_path):
    # Repeated full-keyboard chords exercise the former 20k-note rejection with
    # only 60 measures, keeping this regression practical to run in the suite.
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream()
    pitches = list(range(21, 109))
    for beat in range(228):
        score.insert(beat, chord.Chord(pitches, quarterLength=1))
    score.write("midi", fp=str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(), "Long recording")
    tree = ET.parse(xml)
    assert len(tree.findall(".//note/pitch")) == 228 * 88
    assert len(tree.findall(".//measure")) == 57
