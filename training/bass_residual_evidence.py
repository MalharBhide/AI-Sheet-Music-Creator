"""Pitch-relative evidence for wrong bass notes retained by shipped V10.

Only retained candidates are exposed. Labels are recomputed after prior deletion;
held pitch support and actual key attacks stay positive training examples.
"""

import numpy as np
from app.services.candidate_relations import RELATION_NAMES, relation_features
from bass_training_data import NAMES as BASE_NAMES
from cache_note_verifier import targets
from pitch_support_targets import keep_targets
from train_bass_boundaries import collect as collect_baseline

VERSION = 'bass-v10-residual-relations-v1'
NAMES = BASE_NAMES + ('v10_consensus_mean',) + RELATION_NAMES[1:] + ('v10_context_confidence', 'v10_guardian_confidence')
ACOUSTIC_NAMES = BASE_NAMES + ('v10_context_confidence', 'v10_guardian_confidence')


def features(events, x, context, guardian):
    context, guardian = np.asarray(context), np.asarray(guardian)
    if any(p.shape != (len(events),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
           for p in (context, guardian)):
        raise ValueError('Invalid shipped bass confidence')
    relation = relation_features(events, x, (context + guardian) * .5)
    return np.column_stack((relation, context, guardian)).astype(np.float32)


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid residual bass evidence')
    return np.column_stack((x[:, :52], x[:, -2:])).astype(np.float32)


def advance(item):
    keep = item['baseline_keep']
    events, x = item['events'][keep], item['x'][keep]
    context, guardian = item['baseline_p'][keep], item['baseline_g'][keep]
    if (x.shape != (len(events), 52) or context.shape != (len(events),) or guardian.shape != (len(events),)):
        raise ValueError('Invalid shipped bass inputs')
    y, mask, promoted = keep_targets(item['pitch_reference'], events)
    actual, _ = targets(item['reference'], events)
    y[actual == 1], mask[actual == 1] = 1., True
    return {**item, 'events': events, 'x': features(events, x, context, guardian),
            'eligible': item['eligible'][keep], 'y': y, 'mask': mask, 'promoted': promoted,
            'baseline_keep': np.ones(len(events), dtype=bool), 'keep': np.ones(len(events), dtype=bool),
            'raw_indices': np.flatnonzero(keep), 'raw_count': len(keep)}


def collect(roots):
    return [[advance(item) for item in items] for items in collect_baseline(roots)]
