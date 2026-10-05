"""Label-free paired bass evidence; annotations are used only for training labels."""

import numpy as np
from repeat_boundary_features import merge, pitch_union

VERSION = 'bass-touching-boundary-v1'
MIN_SPACING = .15
MAX_SPACING = 2.
MAX_OVERLAP = .03


def boundaries(item):
    previous, pairs = {}, []
    events = item['events']
    for index in np.argsort(events[:, 0], kind='stable'):
        if not item['keep'][index]:
            continue
        start, _, pitch, _ = events[index]
        earlier = previous.get(int(pitch))
        previous[int(pitch)] = int(index)
        if earlier is None or not (item['shared'][earlier] and item['shared'][index]):
            continue
        gap = start - events[earlier, 1]
        if -MAX_OVERLAP <= gap <= 0 and MIN_SPACING <= start - events[earlier, 0] <= MAX_SPACING:
            pairs.append((earlier, int(index)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def labels(reference, pitch_reference, events, pairs):
    """Real key attacks veto merging; pedal support can label a split continuation."""
    y, mask = np.zeros(len(pairs), dtype=np.float32), np.zeros(len(pairs), dtype=bool)
    for index, (before, after) in enumerate(pairs):
        first, current = events[before], events[after]
        keys = reference[np.abs(reference[:, 2] - current[2]) <= .5]
        if np.any(np.abs(keys[:, 0] - current[0]) <= .05):
            y[index], mask[index] = 1., True
            continue
        if np.any(np.abs(keys[:, 0] - current[0]) <= .15):
            continue
        support = pitch_reference[np.abs(pitch_reference[:, 2] - current[2]) <= .5]
        held = (support[:, 0] <= first[0] + .05) & (support[:, 1] >= max(first[1], current[1]) - .05)
        held &= (support[:, 0] + .15 < current[0]) & (current[0] < support[:, 1] - .05)
        mask[index] = bool(np.any(held))
    return y, mask


def features(item, pairs, acoustic=False):
    x = item['x'][:, :26] if acoustic else item['x']
    x = np.column_stack((x, item['baseline_p'], item['baseline_g']))
    before, after = pairs[:, 0], pairs[:, 1]
    events = item['events']
    geometry = np.column_stack((events[after, 0] - events[before, 0],
                               events[after, 0] - events[before, 1],
                               events[before, 1] - events[before, 0],
                               events[after, 1] - events[after, 0]))
    return np.column_stack((x[before], x[after], geometry)).astype(np.float32)


def acoustic_view(x):
    # Two 54-wide context vectors plus four observed timing measurements.
    return np.column_stack((x[:, :26], x[:, 52:54], x[:, 54:80], x[:, 106:108], x[:, -4:]))


def annotate(items):
    for item in items:
        item['pairs'] = boundaries(item)
        item['boundary_y'], item['boundary_mask'] = labels(
            item['reference'], item['pitch_reference'], item['events'], item['pairs'])
    return items


__all__ = ['VERSION', 'acoustic_view', 'annotate', 'boundaries', 'features', 'labels', 'merge', 'pitch_union']
