"""Protect even short annotated pitch continuations in residual KEEP labels.

The release gate already forbids loss of any supplied pitch interval. Align
fitting with that requirement instead of excluding partially supported fragments
as ambiguous. Inference features, clocks and gate annotations are unchanged.
"""

import numpy as np
from bass_residual_v11_data import ACOUSTIC_NAMES, NAMES, acoustic_view, features
from bass_residual_v11_data import load_cached as load_v11
from pitch_interval_coverage import reference_coverage

VERSION = 'bass-v11-pitch-coverage-residual-v2'
__all__ = ['ACOUSTIC_NAMES', 'NAMES', 'VERSION', 'acoustic_view', 'features', 'load_cached', 'collect', 'protect']


def protect(item):
    annotated = np.column_stack((item['pitch_reference'], np.ones(len(item['pitch_reference']))))
    support = reference_coverage(item['events'][:, :3], annotated)
    required = support > 1e-6
    y, mask = item['y'].copy(), item['mask'].copy()
    newly_protected = required & ((y != 1) | ~mask)
    y[required], mask[required] = 1., True
    return {**item, 'y': y, 'mask': mask,
            'partial_pitch_support_promoted': int(newly_protected.sum())}


def load_cached(directory, groups):
    return [protect(item) for item in load_v11(directory, groups)]


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if (not training or not validation
            or {i['source_group'] for i in training} & {i['source_group'] for i in validation}):
        raise ValueError('Pitch-preserving V11 residual fitting split overlaps or is empty')
    return training, validation
