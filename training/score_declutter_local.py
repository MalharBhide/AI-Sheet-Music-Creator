"""Score removal-only actions against unchanged attacks, holds, coverage and clocks."""

import numpy as np
from bass_joint_timing_v18 import margin
from score_early_consensus import evaluate as joint_evaluate


def evaluate(items, probabilities, guardians, threshold, guardian_threshold, memo=None):
    if (not np.isfinite([threshold, guardian_threshold]).all()
            or not 0 <= threshold <= 1 or not 0 < guardian_threshold <= 1):
        raise ValueError('Invalid removal-only thresholds')
    ps, gates = [], []
    for item, p, g in zip(items, probabilities, guardians, strict=True):
        p, g = np.asarray(p), np.asarray(g)
        if (p.shape != (len(item['events']), 2) or g.shape != (len(p),)
                or not np.isfinite(p).all() or not np.isfinite(g).all()
                or np.any((p < 0) | (p > 1)) or np.any((g < 0) | (g > 1))
                or not np.allclose(p.sum(axis=1), 1., atol=1e-5)):
            raise ValueError('Invalid binary support probabilities')
        joint = np.zeros((len(p), 4), np.float32)
        joint[:, 1], joint[:, 3] = p[:, 0], p[:, 1]
        ps.append(joint)
        # The unchanged joint action guard uses < .3. Boolean acoustic
        # permission implements the declared guardian threshold exactly.
        gates.append(np.where(g < guardian_threshold, 0., 1.))
    result = joint_evaluate(items, ps, gates, 1.01, threshold, memo=memo)
    if result['retimed_notes'] or abs(result['onset_error_reduction_seconds']) > 1e-9:
        raise ValueError('Removal-only candidate changed a retained clock')
    result['guardian_threshold'] = guardian_threshold
    return result


def choose(items, ps, guardians, thresholds, guardian_thresholds, memo):
    eligible, search = [], []
    for threshold in thresholds:
        for guardian_threshold in guardian_thresholds:
            raw = evaluate(items, ps, guardians, threshold, guardian_threshold, memo)
            guarded = evaluate(items, ps, guardians, margin(threshold), guardian_threshold, memo)
            passes = (raw['passes'] and guarded['passes']
                      and raw['false_notes_removed'] > 0 and guarded['false_notes_removed'] > 0)
            search.append({'threshold': threshold, 'guardian_threshold': guardian_threshold,
                           'eligible': bool(passes), 'raw_passes': raw['passes'],
                           'raw_false_notes_removed': raw['false_notes_removed'],
                           'raw_failed_recordings': raw['failed_recordings'],
                           'margin_passes': guarded['passes'],
                           'margin_false_notes_removed': guarded['false_notes_removed'],
                           'margin_failed_recordings': guarded['failed_recordings']})
            if passes:
                key = (raw['false_notes_removed'], guarded['false_notes_removed'], threshold, -guardian_threshold)
                eligible.append((key, {'raw': raw, 'margin': guarded}))
    return (max(eligible, key=lambda pair: pair[0])[1] if eligible else None), search
