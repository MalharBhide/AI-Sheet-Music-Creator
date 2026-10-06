"""Learn bounded attack shifts together across occurrences of each piano key.

This experiment corrects a consistent onset bias, not the rhythm itself. All
same-key attacks move together or stay unchanged, and releases never move.
"""

import numpy as np
from bass_joint_timing import EARLY, KEEP, LATE, REMOVE, SHIFTS, protected
from bass_temporal_data import annotate
from score_bass_joint_timing import timing

VERSION = 'bass-v16-pitch-consensus-attack-v20-v1'


def transform(events, action, duration, strength=1.):
    events, action = np.asarray(events, float), np.asarray(action)
    n = len(events)
    if (events.shape != (n, 4) or action.shape != (n,) or not np.isfinite(events).all()
            or np.any(~np.isin(action, [EARLY, KEEP, LATE, REMOVE])) or strength not in (.5, 1.)
            or not np.isfinite(duration) or duration <= 0
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 1] > duration)):
        raise ValueError('Invalid consensus boundary actions')
    eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
                & (events[:, 2] >= 21) & (events[:, 2] < 60))
    action = np.where(eligible, action, KEEP)
    keep = action != REMOVE
    result = events.copy()
    for pitch in np.unique(events[:, 2]):
        indices = np.flatnonzero(events[:, 2] == pitch)
        group = action[indices]
        # Include removed/ineligible occurrences too. A deletion cannot hide a
        # different shift inside the fixed-pair repeated-note timing metric.
        if not len(group) or group[0] not in (EARLY, LATE) or np.any(group != group[0]):
            continue
        indices = indices[np.argsort(events[indices, 0], kind='stable')]
        start = events[indices, 0]+SHIFTS[group[0]]*strength
        if (np.any(start < 2.5) or np.any(events[indices, 1]-start < .045)
                or np.any(start[1:]-start[:-1] < .045)
                or np.any(start[1:] < np.minimum(events[indices[:-1], 1], events[indices[1:], 0])-1e-9)):
            continue
        result[indices, 0] = start
    changed = result[keep]
    if (np.any(changed[:, 1] <= changed[:, 0])
            or np.any(np.abs(result[:, 0]-events[:, 0]) > .020001)
            or np.any(result[:, 1:] != events[:, 1:])):
        raise ValueError('Consensus correction changed a release, pitch or velocity')
    return changed, np.flatnonzero(keep)


def targets(item):
    labeled = annotate(item)
    y = np.full(len(item['events']), KEEP, dtype=np.int64)
    y[(labeled['y'] == 0) & labeled['mask']] = REMOVE
    for pitch in np.unique(item['events'][:, 2]):
        indices = np.flatnonzero(item['events'][:, 2] == pitch)
        if not np.all(item['eligible'][indices] & labeled['mask'][indices] & (labeled['y'][indices] == 1)):
            continue
        best = None
        for direction in (EARLY, LATE):
            action = np.full(len(y), KEEP, dtype=np.int64)
            action[indices] = direction
            new, kept = transform(item['events'], action, item['duration'])
            if np.array_equal(new, item['events']):
                continue
            clock = timing(item, new, kept)
            gain = clock['before_onset_release_error_sum'][0]-clock['after_onset_release_error_sum'][0]
            if gain > .01*len(indices) and clock['passes'] and protected(item, new)[0]:
                if best is None or gain > best[0]:
                    best = (gain, direction)
        if best:
            y[indices] = best[1]
    return {**labeled, 'action_y': y}
