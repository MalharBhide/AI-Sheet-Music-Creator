import json

import pretty_midi
import pytest
from music21 import chord, note, stream, tempo, tie

from app.models import ScoreOptions
from app.services.audio_analysis import preserve_bass_releases, preserve_melody_releases
from app.services.midi_to_score import midi_to_musicxml
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
    # MIDI stores 90 BPM as an integer 666666 microseconds per quarter.
    assert result['duration'] == 7.999992
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


def roundtrip_notes(tmp_path, notes):
    source = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(0)
    piano.notes = [pretty_midi.Note(velocity, pitch, start, end)
                   for pitch, start, end, velocity in notes]
    source.instruments.append(piano)
    midi, xml = tmp_path / 'source.mid', tmp_path / 'score.musicxml'
    source.write(str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(tempo_bpm=120, grid='sixteenth'), 'Regression')
    result = export_score_playback(xml, midi, tmp_path / 'playback.json', 120)
    playback = [(item['pitch'], item['start'], item['end'], item['velocity'])
                for item in result['notes']]
    canonical = pretty_midi.PrettyMIDI(str(midi))
    midi_notes = [(item.pitch, item.start, item.end, item.velocity)
                  for part in canonical.instruments for item in part.notes]
    assert sorted(playback) == sorted(midi_notes)
    assert all(part.program == 0 and not part.is_drum for part in canonical.instruments)
    return playback


def test_triplet_rhythm_survives_notation_and_browser_playback(tmp_path):
    import xml.etree.ElementTree as ET

    from app.services.audio_analysis import clean_notes

    source = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    melody = pretty_midi.Instrument(0)
    melody.notes = clean_notes([pretty_midi.Note(80, 60 + i, i / 6, (i + 1) / 6)
                               for i in range(6)], role='vocals', detail='balanced',
                              tempo_bpm=120, grid='sixteenth')
    source.instruments.append(melody)
    midi, xml = tmp_path / 'triplet.mid', tmp_path / 'triplet.musicxml'
    source.write(str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(tempo_bpm=120), 'Controlled rhythm fixture')
    tree = ET.parse(xml)
    assert tree.findall('.//time-modification')
    playback = export_score_playback(xml, midi, tmp_path / 'playback.json', 120)
    assert [n['pitch'] for n in playback['notes']] == list(range(60, 66))
    assert [n['start'] for n in playback['notes']] == pytest.approx([i / 6 for i in range(6)], abs=1e-6)
    assert [n['end'] for n in playback['notes']] == pytest.approx([(i + 1) / 6 for i in range(6)], abs=1e-6)
    canonical = pretty_midi.PrettyMIDI(str(midi))
    assert [n.start for p in canonical.instruments for n in p.notes] == pytest.approx([i / 6 for i in range(6)])


def test_sparse_voices_preserve_sustain_releases_and_individual_velocities(tmp_path):
    notes = [(60, 0, 4, 70), (64, .5, 1, 50), (65, 1.5, 2, 100), (67, 2.5, 3, 80)]
    assert sorted(roundtrip_notes(tmp_path, notes)) == sorted(notes)


def test_bass_ownership_preserves_holds_repeats_and_rests_in_score_playback(tmp_path):
    bass, backing = pretty_midi.Instrument(0), pretty_midi.Instrument(0)
    bass.notes = [pretty_midi.Note(68, 48, 0, 2), pretty_midi.Note(72, 48, 3, 4)]
    backing.notes = [pretty_midi.Note(54, 48, .5, 1), pretty_midi.Note(54, 48, 1, 3),
                     pretty_midi.Note(54, 48, 2.5, 3.5), pretty_midi.Note(54, 55, 0, 4)]
    assert preserve_bass_releases({'bass': bass, 'other': backing}) == 3
    events = [(n.pitch, n.start, n.end, n.velocity) for part in (bass, backing) for n in part.notes]
    assert sorted(roundtrip_notes(tmp_path, events)) == sorted([
        (48, 0, 2, 68), (48, 2.5, 3, 54), (48, 3, 4, 72), (55, 0, 4, 54)])


def test_melody_priority_survives_notation_and_browser_playback_export(tmp_path):
    lead, support = pretty_midi.Instrument(0), pretty_midi.Instrument(0)
    lead.notes = [pretty_midi.Note(92, 64, 1, 1.5), pretty_midi.Note(92, 64, 2, 2.5)]
    support.notes = [pretty_midi.Note(54, 64, 0, 4), pretty_midi.Note(54, 67, 0, 4)]
    preserve_melody_releases({'vocals': lead, 'other': support})
    arranged = [(n.pitch, n.start, n.end, n.velocity) for part in (lead, support) for n in part.notes]
    result = roundtrip_notes(tmp_path, arranged)
    assert sorted(item for item in result if item[0] == 64) == [
        (64, 0, 1, 54), (64, 1, 1.5, 92), (64, 2, 2.5, 92)]
    assert (67, 0, 4, 54) in result


def test_changing_dense_chords_do_not_manufacture_reattacks(tmp_path):
    # Eight overlapping pitches force the dense chord fallback. The generic
    # music21 MIDI exporter used to turn these eight attacks into 64 attacks.
    notes = [(60 + index, index * .125, 2 + index * .125, 40 + index * 10)
             for index in range(8)]
    assert sorted(roundtrip_notes(tmp_path, notes)) == sorted(notes)


def test_overlapping_unisons_in_independent_voices_keep_both_releases(tmp_path):
    # A shared MIDI channel would let the first C's note-off cut the second C
    # short. Separate piano voice channels preserve both written durations.
    notes = [(60, 0, 2, 60), (60, 1, 3, 100), (64, .5, 1, 75), (65, 1.5, 2, 80)]
    assert sorted(roundtrip_notes(tmp_path, notes)) == sorted(notes)


def test_repeated_pitch_inside_dense_tied_chords_has_exactly_two_attacks(tmp_path):
    notes = [(60 + index, index * .125, 4 + index * .125, 40 + index * 10)
             for index in range(8)]
    notes.append((60, 2, 5, 100))
    # Dense notation restarts this piano key at the second onset, then follows
    # its written ties until the final release. Other changing pitches must not
    # restart either C, or change the second attack's velocity.
    expected = [(60, 0, 2, 40), *notes[1:]]
    assert sorted(roundtrip_notes(tmp_path, notes)) == sorted(expected)


@pytest.mark.parametrize('bpm', [87.3, 112.7, 129.9993])
def test_playback_seconds_match_midi_integer_tempo_encoding(tmp_path, bpm):
    score = stream.Stream([tempo.MetronomeMark(number=bpm),
                           note.Rest(quarterLength=3), note.Note(60, quarterLength=5)])
    xml, midi = tmp_path / 'score.musicxml', tmp_path / 'score.mid'
    score.write('musicxml', fp=str(xml))
    result = export_score_playback(xml, midi, tmp_path / 'playback.json', bpm)
    canonical = pretty_midi.PrettyMIDI(str(midi))
    actual = canonical.instruments[0].notes[0]
    assert result['notes'][0]['start'] == round(actual.start, 6)
    assert result['notes'][0]['end'] == round(actual.end, 6)


def test_sparse_score_after_six_hours_exports_without_a_dense_tick_array(tmp_path, monkeypatch):
    import mido

    xml, midi = tmp_path / 'long.musicxml', tmp_path / 'long.mid'
    # One sparse measure isolates playback's duration handling from engraving.
    # Its C begins six hours into the recording at 120 BPM (>20 million ticks).
    xml.write_text('''<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1"><measure number="1" implicit="yes">
    <attributes><divisions>1</divisions></attributes>
    <forward><duration>43200</duration></forward>
    <note dynamics="80"><pitch><step>C</step><octave>4</octave></pitch>
      <duration>2</duration><type>half</type></note>
  </measure></part>
</score-partwise>''')

    def reject_dense_timing(self, max_tick):
        raise AssertionError('Playback must not allocate timing arrays by recording length')

    monkeypatch.setattr(pretty_midi.PrettyMIDI, '_update_tick_to_time', reject_dense_timing)
    result = export_score_playback(xml, midi, tmp_path / 'playback.json', 120)
    assert result['notes'] == [{'pitch': 60, 'start': 21600, 'end': 21601, 'velocity': 72}]
    assert result['duration'] == 21601
    # mido walks sparse events and can validate the written file at any length.
    assert mido.MidiFile(midi).length == pytest.approx(21601, abs=.01)


@pytest.mark.parametrize('signature, tonal_key, key_number', [
    ('3/4', 'D minor', 14), ('6/8', 'B- major', 10),
])
def test_canonical_midi_preserves_score_meter_and_key(tmp_path, signature, tonal_key, key_number):
    midi, xml = tmp_path / 'score.mid', tmp_path / 'score.musicxml'
    source = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(0)
    piano.notes.append(pretty_midi.Note(90, 62, 0, 1))
    source.instruments.append(piano)
    source.write(str(midi))
    midi_to_musicxml(midi, xml, ScoreOptions(time_signature=signature), 'Meter and key',
                     key_signature=tonal_key)
    export_score_playback(xml, midi, tmp_path / 'playback.json', 120)
    canonical = pretty_midi.PrettyMIDI(str(midi))
    written_meter = canonical.time_signature_changes[0]
    assert (written_meter.numerator, written_meter.denominator) == tuple(
        int(value) for value in signature.split('/'))
    assert written_meter.time == 0
    assert canonical.key_signature_changes[0].key_number == key_number
    assert canonical.key_signature_changes[0].time == 0
