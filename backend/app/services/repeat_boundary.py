"""Paired label-free evidence and coverage-preserving split-hold merging."""

import numpy as np

from app.services.left_baseline_evidence import (
    ACOUSTIC_EVIDENCE_NAMES,
    RELATION_EVIDENCE_NAMES,
    acoustic_evidence,
)

VERSION = 'paired-repeat-boundary-v1'
GEOMETRY_NAMES = ('attack_spacing', 'boundary_gap', 'before_duration', 'after_duration')
RELATION_NAMES = tuple('before_' + n for n in RELATION_EVIDENCE_NAMES) + tuple(
    'after_' + n for n in RELATION_EVIDENCE_NAMES) + GEOMETRY_NAMES
ACOUSTIC_NAMES = tuple('before_' + n for n in ACOUSTIC_EVIDENCE_NAMES) + tuple(
    'after_' + n for n in ACOUSTIC_EVIDENCE_NAMES) + GEOMETRY_NAMES


def features(item, pairs, acoustic=False):
    x = acoustic_evidence(item['x']) if acoustic else item['x']
    before, after = pairs[:, 0], pairs[:, 1]
    events = item['events']
    geometry = np.column_stack((events[after, 0] - events[before, 0],
                                events[after, 0] - events[before, 1],
                                events[before, 1] - events[before, 0],
                                events[after, 1] - events[after, 0]))
    return np.column_stack((x[before], x[after], geometry)).astype(np.float32)


def merge(item, pairs, repeat, guardian, threshold, guardian_threshold):
    """Join only touching same-key fragments; never fill even a short real rest."""
    repeat, guardian = np.asarray(repeat), np.asarray(guardian)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or any(p.shape != (len(pairs),) for p in (repeat, guardian))
            or any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in (repeat, guardian))
            or not 0 <= threshold <= 1 or not 0 <= guardian_threshold <= 1):
        raise ValueError('Invalid frozen boundary confidence')
    events, keep = item['events'].copy(), item['keep'].copy()
    owner = np.arange(len(events))
    merged = []
    for index, (previous, current) in enumerate(pairs):
        first, second = owner[previous], owner[current]
        if (first == second or not keep[first] or not keep[second]
                or repeat[index] >= threshold or guardian[index] >= guardian_threshold):
            continue
        a, b = events[first], events[second]
        # Near-contiguous positive gaps remain untouched. Preserve the original
        # pitch-time union exactly, including negative/out-of-annotation spans.
        if (a[2] != b[2] or b[0] < a[0] or b[0] > a[1]
                or not (item['shared'][previous] and item['shared'][current])):
            continue
        events[first, 1] = max(a[1], b[1])
        keep[second] = False
        owner[owner == second] = first
        merged.append((int(previous), int(current)))
    return events[keep], merged


def pitch_union(events):
    """Canonical intervals allow a release gate to check all pitch-time coverage."""
    result = {}
    for pitch in sorted(set(events[:, 2])):
        spans = sorted((float(s), float(e)) for s, e, p, *_ in events if p == pitch)
        union = []
        for start, end in spans:
            if not union or start > union[-1][1]:
                union.append([start, end])
            else:
                union[-1][1] = max(union[-1][1], end)
        result[int(pitch)] = union
    return result


def boundaries(item):
    """Return adjacent same-key pairs only when observed timing is near-contiguous."""
    events = item['events']
    previous, pairs = {}, []
    for index in np.argsort(events[:, 0], kind='stable'):
        if not item['keep'][index]:
            continue
        start, _, pitch, _ = events[index]
        earlier = previous.get(int(pitch))
        previous[int(pitch)] = int(index)
        if earlier is None or not (item['shared'][earlier] and item['shared'][index]):
            continue
        gap = start - events[earlier, 1]
        if not (-1e-6 <= gap <= .03 and .15 <= start - events[earlier, 0] <= 2.):
            continue
        pairs.append((earlier, int(index)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)

