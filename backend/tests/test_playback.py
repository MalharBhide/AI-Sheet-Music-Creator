import json

import pretty_midi
import pytest
from music21 import chord, note, stream, tempo, tie

from app.services.playback import export_score_playback


def test_playback_preserves_score_ties_chords_rests_and_tempo(tmp_path):
    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=90))
    first = chord.Chord([60, 64], quarterLength=4)
    first.tie = tie.Tie('start')
    continuation = chord.Chord([60, 64], quarterLength=1)
    continuation.tie = tie.Tie('stop')
    part.append(first)
    part.append(continuation)
    part.append(note.Rest(quarterLength=1))
    part.append(note.Note(67, quarterLength=2))
    part.append(note.Rest(quarterLength=4))
    score.append(part)
    xml = tmp_path / 'score.musicxml'
    score.write('musicxml', fp=str(xml))
    midi = tmp_path / 'transcription.mid'
    midi.write_bytes(b'old raw detections')
    playback = tmp_path / 'playback.json'
    result = export_score_playback(xml, midi, playback, 90)
    assert result == json.loads(playback.read_text())
    assert result['duration'] == pytest.approx(8)
    assert [item['pitch'] for item in result['notes']] == [60, 64, 67]
    assert [item['start'] for item in result['notes']] == pytest.approx([0, 0, 4], abs=1e-5)
    assert result['notes'][0]['end'] == pytest.approx(10 / 3, abs=1e-5)
    canonical = pretty_midi.PrettyMIDI(str(midi))
    pitches = sorted(n.pitch for part in canonical.instruments for n in part.notes)
    assert pitches == [60, 64, 67]


def test_silent_score_has_playable_duration_and_valid_midi(tmp_path):
    score = stream.Stream([tempo.MetronomeMark(number=120), note.Rest(quarterLength=4)])
    xml = tmp_path / 'score.musicxml'
    score.write('musicxml', fp=str(xml))
    midi = tmp_path / 'transcription.mid'
    result = export_score_playback(xml, midi, tmp_path / 'playback.json', 120)
    assert result == {'duration': 2, 'tempo_bpm': 120, 'notes': []}
    assert not pretty_midi.PrettyMIDI(str(midi)).instruments
