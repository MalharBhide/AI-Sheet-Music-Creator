"""Reference-pitch interval coverage, shared by supervision and release gates."""

import numpy as np


def reference_coverage(reference, events):
    values = []
    for start, end, pitch in reference:
        same = events[np.abs(events[:, 2] - pitch) <= .5]
        spans = sorted((max(start, first), min(end, last)) for first, last, *_ in same
                       if min(end, last) > max(start, first))
        total, stop = 0., start
        for first, last in spans:
            total += max(0., last - max(stop, first))
            stop = max(stop, last)
        values.append(total)
    return np.asarray(values)


def compare(reference, before, after):
    old, new = [reference_coverage(reference, events) for events in (before, after)]
    lost = np.maximum(0., old - new)
    return {'passes': not np.any(lost > 1e-6), 'lost_reference_seconds': float(lost.sum()),
            'references_with_lost_coverage': int(np.sum(lost > 1e-6))}
