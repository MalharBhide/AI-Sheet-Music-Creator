"""Conservative symbolic piano clock; never infer ambiguous key or pedal releases."""

import io

import mido
import numpy as np

META = {"track_name", "time_signature", "key_signature", "end_of_track", "text",
        "copyright", "marker", "cue_marker", "instrument_name"}
DEFAULT_CONTROLLERS = {7: 100, 10: 64, 91: 0, 93: 0, 121: 0}


def notes(data):
    midi = mido.MidiFile(file=io.BytesIO(data))
    if midi.type not in (0, 1) or not 1 <= midi.ticks_per_beat <= 32767:
        raise ValueError("Unsupported simultaneous-track PPQ clock")
    tempo, tick, seconds = 500000, 0, 0.
    active, events, tempos = {}, [], {}
    channel, started = None, False
    for message in mido.merge_tracks(midi.tracks):
        if not isinstance(message.time, int) or message.time < 0:
            raise ValueError("Invalid symbolic delta clock")
        seconds += message.time * tempo / (1_000_000 * midi.ticks_per_beat)
        tick += message.time
        if message.type == "set_tempo":
            if not 100000 <= message.tempo <= 3000000:
                raise ValueError("Unsupported source tempo")
            if tick in tempos and tempos[tick] != message.tempo:
                raise ValueError("Conflicting simultaneous tempo events")
            tempo = tempos[tick] = message.tempo
        elif message.type in ("note_on", "note_off"):
            if channel is None:
                channel = message.channel
            if message.channel != channel or channel == 9:
                raise ValueError("Multiple key channels or percussion")
            if not 21 <= message.note <= 108:
                raise ValueError("Source outside full piano range")
            if message.type == "note_on" and message.velocity > 0:
                started = True
                if message.note in active:
                    raise ValueError("Ambiguous same-key overlap")
                active[message.note] = (seconds, message.velocity)
            else:
                if message.note not in active:
                    raise ValueError("Unmatched key release")
                start, velocity = active.pop(message.note)
                if seconds <= start:
                    raise ValueError("Nonpositive key duration")
                events.append((start, seconds, message.note, velocity))
        elif message.type == "control_change":
            if (started or tick != 0 or message.control not in DEFAULT_CONTROLLERS
                    or message.value != DEFAULT_CONTROLLERS[message.control]):
                raise ValueError("Pedal, expressive or nondefault controller")
        elif message.type == "program_change":
            if started or tick != 0 or message.program != 0 or message.channel == 9:
                raise ValueError("Unsupported piano program")
        elif message.type == "midi_port":
            if message.port != 0:
                raise ValueError("Multiple MIDI ports")
        elif message.type not in META:
            raise ValueError("Unsupported source message: " + message.type)
    if active or not events or not 0 < seconds <= 1200:
        raise ValueError("Incomplete symbolic piano reference")
    result = np.asarray(sorted(events), dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite piano clock")
    return result


def excerpt(events, start, length=24.):
    """Use only complete natural key boundaries, including their damping tails.

    Fixed windows touching a held key are rejected instead of truncating or
    retriggering it. Neighbouring notes whose 300ms support crosses the window
    also make the window unusable. Nothing is silently dropped inside a window.
    """
    events = np.asarray(events, dtype=np.float64)
    if (events.ndim != 2 or events.shape[1] != 4 or not len(events)
            or not np.isfinite(events).all() or not np.isfinite(start) or start < 0
            or not np.isfinite(length) or not 8 <= length <= 24):
        raise ValueError("Invalid complete-key excerpt")
    end = start + length
    touching = events[(events[:, 0] < end) & (events[:, 1] + .3 > start)]
    if not len(touching) or np.any(touching[:, 0] < start) or np.any(touching[:, 1] > end):
        return None
    result = touching.copy()
    result[:, :2] += 3. - start
    return result


def synthesis_clock(events, velocity_scale, *, whole_work=False):
    """Preserve tempo map and relative timing; round to actual synthesis blocks."""
    from offline_piano_v47 import BLOCK, RATE

    events = np.asarray(events, dtype=np.float64)
    if (events.ndim != 2 or events.shape[1] != 4 or not len(events)
            or not np.isfinite(events).all() or not .1 <= velocity_scale <= 1
            or np.any(events[:, 0] < 3) or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 1] > (1200 if whole_work else 27))
            or np.any(events[:, 2:] != np.floor(events[:, 2:]))
            or np.any((events[:, 2] < 21) | (events[:, 2] > 108))
            or np.any((events[:, 3] < 1) | (events[:, 3] > 127))):
        raise ValueError("Invalid synthetic piano keys")
    result = events.copy()
    result[:, :2] = np.ceil(events[:, :2] * RATE / BLOCK) * BLOCK / RATE
    result[:, 3] = np.clip(np.rint(result[:, 3] * velocity_scale), 1, 127)
    for pitch in np.unique(result[:, 2]):
        keys = result[result[:, 2] == pitch]
        keys = keys[np.argsort(keys[:, 0], kind="stable")]
        if (np.any(keys[:, 1] <= keys[:, 0])
                or np.any(keys[:-1, 1] > keys[1:, 0])):
            raise ValueError("Invalid rounded same-key clock")
    return result


def crop_labels(reference, support, start, length):
    """No manufactured attacks/releases: crossing holds provide pitch support only."""
    reference, support = (np.asarray(a, dtype=np.float64) for a in (reference, support))
    if (any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all()
            or np.any(a[:, 1] <= a[:, 0]) for a in (reference, support))
            or not np.isfinite(start) or start < 0 or length != 24):
        raise ValueError("Invalid source labels or fixed crop")
    end = start + length
    attacks = reference[(reference[:, 0] >= start) & (reference[:, 1] <= end)].copy()
    attacks[:, :2] -= start
    present = support[(support[:, 0] < end) & (support[:, 1] > start)].copy()
    present[:, :2] = np.clip(present[:, :2] - start, 0, length)
    return attacks, present
