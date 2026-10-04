"""Teach a pitch filter to retain supported held-note fragments.

Original one-to-one attack labels remain the evaluation truth. Ambiguous events
whose duration is at least 90% covered by a reference of the same pitch become
positive KEEP supervision only. A separate articulation model must repair extra
attacks; deleting their audible pitch continuation is not a valid repair.
"""

import numpy as np
from cache_note_verifier import targets
from pitch_interval_coverage import reference_coverage

VERSION = 'pitch-continuation-support-v1'


def keep_targets(reference, events):
    y, mask = targets(reference, events)
    promoted = np.zeros(len(events), dtype=bool)
    if not len(events):
        return y, mask, promoted
    # Reverse the interval query: union annotation support under each event.
    annotated = np.column_stack((reference, np.ones(len(reference))))
    coverage = reference_coverage(events[:, :3], annotated)
    duration = events[:, 1] - events[:, 0]
    promoted = (y == 0) & ~mask & (duration > 0) & (coverage >= .9 * duration)
    y[promoted], mask[promoted] = 1., True
    return y, mask, promoted
