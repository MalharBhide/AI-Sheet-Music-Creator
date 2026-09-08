import xml.etree.ElementTree as ET

import pytest
from music21 import chord, converter, note, stream, tempo

from app.models import PipelineError, ScoreOptions
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


def test_no_notes_fails_clearly(tmp_path):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream([note.Rest(quarterLength=1)])
    score.write("midi", fp=str(midi))
    with pytest.raises(PipelineError, match="No playable"):
        midi_to_musicxml(midi, xml, ScoreOptions(), "Silence")


def test_more_than_four_overlapping_voices_fails_before_renderer_can_drop_notes(tmp_path):
    midi, xml = tmp_path / "input.mid", tmp_path / "score.musicxml"
    score = stream.Stream()
    for index in range(5):
        score.insert(index / 4, note.Note(60 + index, quarterLength=2))
    score.write("midi", fp=str(midi))
    with pytest.raises(PipelineError, match="too dense"):
        midi_to_musicxml(midi, xml, ScoreOptions(), "Dense passage")
