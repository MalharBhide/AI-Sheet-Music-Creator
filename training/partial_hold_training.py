"""Emphasize short supported fragments in training loss; never an inference feature."""

import numpy as np
from pitch_interval_coverage import reference_coverage


def support_boost(item, multiplier):
    events = np.asarray(item['events'], float)
    if (not np.isfinite(multiplier) or multiplier < 1 or not np.isfinite(events).all()
            or events.shape != (len(events), 4) or np.any(events[:, 1] <= events[:, 0])):
        raise ValueError('Invalid partial-hold supervision')
    annotated = np.column_stack((item['pitch_reference'], np.ones(len(item['pitch_reference']))))
    supported = reference_coverage(events[:, :3], annotated)
    fraction = supported / (events[:, 1] - events[:, 0])
    # Even a small true continuation is protected by the existing full-coverage
    # gates. Give its positive label more influence, without supplying reference
    # overlap to either the CNN or production decoder.
    partial = ((item['y'] == 1) & item['mask'] & (supported > 1e-6) & (fraction < .25))
    return np.where(partial, multiplier, 1.)
