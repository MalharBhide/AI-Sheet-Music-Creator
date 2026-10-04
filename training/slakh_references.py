"""Deterministic key releases and sustain-pitch support from source MIDI labels."""

import bisect
import hashlib

import numpy as np


def partition(identities, duplicate_map, seed=261033):
    if len(identities) != 20 or len(set(identities)) != 20:
        raise ValueError('Expected the complete 20-track official prototype subset')
    groups = {}
    for identity in identities:
        record = duplicate_map[identity]
        key = min([identity, *record['midi_duplicates']])
        groups.setdefault(key, []).append(identity)
    if len(groups) != 20:
        raise ValueError('Prototype duplicates require a revised group-level split plan')
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    result = {}
    for index, key in enumerate(ordered):
        group = 'train' if index < 12 else 'validation' if index < 16 else 'test'
        for identity in groups[key]:
            result[identity] = group
    return result


def intervals(midi, duration):
    """Keep note-off references separate from MIDI-annotated pedal support."""
    keys, support = [], []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        controls = sorted((c.time, c.value) for c in instrument.control_changes if c.number == 64)
        times = [c[0] for c in controls]
        by_pitch = {}
        for note in instrument.notes:
            by_pitch.setdefault(note.pitch, []).append(note.start)
        for starts in by_pitch.values():
            starts.sort()
        for note in instrument.notes:
            if not 0 <= note.start < note.end <= duration + .1:
                raise ValueError('Source MIDI note interval exceeds its audio clock')
            end = min(duration, note.end)
            keys.append((note.start, end, note.pitch))
            position = bisect.bisect_right(times, note.end)
            if position and controls[position - 1][1] >= 64:
                end = next((t for t, value in controls[position:] if value < 64), duration)
                # The next same-key attack replaces the earlier sustained voice,
                # retaining the distinct attack label in the key references.
                starts = by_pitch[note.pitch]
                later = bisect.bisect_right(starts, note.start)
                if later < len(starts):
                    end = min(end, max(note.end, starts[later]))
            support.append((note.start, min(duration, max(note.end, end)), note.pitch))
    return [np.asarray(sorted(set(rows)), dtype=float).reshape(-1, 3) for rows in (keys, support)]


def crop(keys, support, offset, duration):
    stop = duration - .25
    reference = keys[(keys[:, 0] >= offset + .25) & (keys[:, 0] < offset + stop)].copy()
    reference[:, :2] -= offset
    reference[:, 1] = np.minimum(reference[:, 1], duration)
    # Retain holds that began before the clip/crop. Their attack is deliberately
    # absent from onset evaluation, but their audible pitch must stay protected.
    pitch = support[(support[:, 0] < offset + duration) & (support[:, 1] > offset)].copy()
    pitch[:, :2] -= offset
    pitch[:, 0] = np.maximum(0., pitch[:, 0])
    pitch[:, 1] = np.minimum(duration, pitch[:, 1])
    return reference, pitch
