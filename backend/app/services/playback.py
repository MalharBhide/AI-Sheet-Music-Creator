"""Create audible playback from the final score, not the raw model detections."""
import json
import math
from pathlib import Path

import pretty_midi
from music21 import chord, converter, key, meter, note, stream


def _score_midi(score: stream.Score, tempo_bpm: float) -> pretty_midi.PrettyMIDI:
    """Join individual pitch ties, including ties between changing chords.

    music21's generic MIDI exporter joins whole-chord ties only when the chord's
    membership stays the same. Our dense notation changes chord membership at
    every attack/release, so that exporter manufactures repeated attacks. Follow
    the individual notes' ties instead, within each staff and voice.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm, resolution=480)
    signature = score.recurse().getElementsByClass(meter.TimeSignature).first()
    if signature is not None:
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(signature.numerator, signature.denominator, 0))
    tonal_signature = score.recurse().getElementsByClass(key.KeySignature).first()
    if tonal_signature is not None:
        tonal_key = (tonal_signature if isinstance(tonal_signature, key.Key)
                     else tonal_signature.asKey('major'))
        number = tonal_key.tonic.pitchClass + (12 if tonal_key.mode == 'minor' else 0)
        midi.key_signature_changes.append(pretty_midi.KeySignature(number, 0))
    seconds_per_quarter = 60 / tempo_bpm
    for part_index, part in enumerate(score.parts or [score]):
        instruments: dict[str, pretty_midi.Instrument] = {}
        continuing: dict[tuple[str, int], pretty_midi.Note] = {}
        elements = sorted(part.recurse().notes,
                          key=lambda item: float(item.getOffsetInHierarchy(part)))
        for element in elements:
            if not isinstance(element, (note.Note, chord.Chord)) or element.quarterLength <= 0:
                continue
            voice = element.getContextByClass(stream.Voice)
            voice_id = str(voice.id) if voice is not None else '1'
            if voice_id not in instruments:
                # Independent voices use separate MIDI channels. Otherwise an
                # earlier note-off can cut short an overlapping unison voice.
                instruments[voice_id] = pretty_midi.Instrument(
                    0, name=f'Piano staff {part_index + 1}, voice {voice_id}')
            instrument = instruments[voice_id]
            start = float(element.getOffsetInHierarchy(part)) * seconds_per_quarter
            end = start + float(element.quarterLength) * seconds_per_quarter
            for pitched_note in element.notes if isinstance(element, chord.Chord) else [element]:
                pitch = int(round(pitched_note.pitch.midi))
                identity = (voice_id, pitch)
                tie_type = pitched_note.tie.type if pitched_note.tie is not None else None
                previous = continuing.get(identity)
                if (tie_type in {'continue', 'stop'} and previous is not None
                        and math.isclose(previous.end, start, abs_tol=1e-7)):
                    previous.end = end
                    current = previous
                else:
                    velocity = pitched_note.volume.velocity
                    if velocity is None:
                        velocity = round(pitched_note.volume.realized * 127)
                    current = pretty_midi.Note(
                        velocity=max(1, min(127, velocity)),
                        pitch=pitch, start=start, end=end)
                    instrument.notes.append(current)
                if tie_type in {'start', 'continue'}:
                    continuing[identity] = current
                else:
                    continuing.pop(identity, None)
        midi.instruments.extend(instruments.values())
    return midi


def export_score_playback(xml_path: Path, midi_path: Path, playback_path: Path,
                          tempo_bpm: float) -> dict:
    score = converter.parse(str(xml_path))
    midi_temporary = midi_path.with_name('score-playback.tmp.mid')
    midi = _score_midi(score, tempo_bpm)
    midi.write(str(midi_temporary))
    # Reading the MIDI back with PrettyMIDI creates a dense tick-to-time array
    # proportional to recording length. Convert the sparse note ticks directly
    # using the same integer microsecond tempo that PrettyMIDI.write encodes.
    tick_scale = 60.0 / (tempo_bpm * midi.resolution)
    encoded_tempo = int(6e7 / (60.0 / (tick_scale * midi.resolution)))
    seconds_per_tick = encoded_tempo / 1e6 / midi.resolution
    notes = [
        {'pitch': note.pitch,
         'start': round(midi.time_to_tick(note.start) * seconds_per_tick, 6),
         'end': round(midi.time_to_tick(note.end) * seconds_per_tick, 6),
         'velocity': note.velocity}
        for instrument in midi.instruments if not instrument.is_drum
        for note in instrument.notes
    ]
    notes.sort(key=lambda note: (note['start'], note['pitch'], note['end']))
    duration = max(float(score.highestTime) * encoded_tempo / 1e6,
                   max((note['end'] for note in notes), default=0))
    payload = {'duration': round(duration, 6), 'tempo_bpm': tempo_bpm, 'notes': notes}
    temporary = playback_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    midi_temporary.replace(midi_path)
    temporary.replace(playback_path)
    return payload
