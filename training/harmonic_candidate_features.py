"""Observed harmonic attack/release relationships, without fitted confidences."""

import numpy as np
from app.services.bass_texture import BASE_NAMES
from app.services.bass_texture import NAMES as SALIENCE_NAMES
from app.services.bass_texture import features as salience_features

INTERVALS = (12, 19, 24)
HARMONIC_NAMES = tuple(name + '_' + str(interval) for interval in INTERVALS
                       for name in ('lower_attack_strength', 'harmonic_attack_difference',
                                    'harmonic_attack_alignment', 'harmonic_release_alignment'))
NAMES = SALIENCE_NAMES + HARMONIC_NAMES
ACOUSTIC_NAMES = BASE_NAMES


def features(events, base_x):
    events = np.asarray(events, float)
    base_x = np.asarray(base_x, np.float32)
    x = salience_features(events, base_x)
    if not len(events):
        return np.empty((0, len(NAMES)), np.float32)
    starts, ends, pitches, _ = events.T
    attacks = base_x[:, BASE_NAMES.index('onset_at_attack')]
    strength = base_x[:, BASE_NAMES.index('note_mean')]
    rows = []
    for index, (start, end, pitch, _) in enumerate(events):
        neighborhood = (np.abs(starts - start) <= 2.) & (np.arange(len(events)) != index)
        overlaps = (np.minimum(end, ends) - np.maximum(start, starts)) > 0
        row = []
        for interval in INTERVALS:
            lower = neighborhood & overlaps & (pitches == pitch - interval)
            maximum = float(np.max(attacks[lower], initial=0.))
            attack_alignment = attacks[lower] * np.exp(-np.abs(starts[lower] - start) / .12)
            release_alignment = strength[lower] * np.exp(-np.abs(ends[lower] - end) / .12)
            row.extend((maximum, float(np.clip(attacks[index] - maximum, -1., 1.)),
                        float(np.max(attack_alignment, initial=0.)),
                        float(np.max(release_alignment, initial=0.))))
        rows.append(row)
    result = np.column_stack((x, np.asarray(rows, np.float32))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Invalid harmonic candidate evidence')
    return result


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid V13 harmonic feature contract')
    return x[:, :len(BASE_NAMES)]
