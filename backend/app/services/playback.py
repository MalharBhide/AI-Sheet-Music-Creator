"""Create audible playback from the final score, not the raw model detections."""
import json
from pathlib import Path

import pretty_midi
from music21 import converter


def export_score_playback(xml_path: Path, midi_path: Path, playback_path: Path,
                          tempo_bpm: float) -> dict:
    # music21's MIDI exporter joins tied fragments, preserving their sustain.
    score = converter.parse(str(xml_path))
    midi_temporary = midi_path.with_name('score-playback.tmp.mid')
    score.write('midi', fp=str(midi_temporary))
    midi = pretty_midi.PrettyMIDI(str(midi_temporary))
    notes = [
        {'pitch': note.pitch, 'start': round(note.start, 6), 'end': round(note.end, 6),
         'velocity': note.velocity}
        for instrument in midi.instruments if not instrument.is_drum
        for note in instrument.notes
    ]
    notes.sort(key=lambda note: (note['start'], note['pitch'], note['end']))
    duration = max(float(score.highestTime) * 60 / tempo_bpm,
                   max((note['end'] for note in notes), default=0))
    payload = {'duration': round(duration, 6), 'tempo_bpm': tempo_bpm, 'notes': notes}
    temporary = playback_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    midi_temporary.replace(midi_path)
    temporary.replace(playback_path)
    return payload
