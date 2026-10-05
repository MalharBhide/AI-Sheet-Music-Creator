"""Bind offline experiments to the shipped V10 bass keep decisions.

These arrays have independent sklearn/runtime parity evidence in the V10 card.
Experiments may operate only on retained events; prior deletions cannot return.
"""

from pathlib import Path

import numpy as np
from app.services import bass_verifier as service
from bass_training_data import NAMES
from prepare_robust_training_stems import digest

VERSION = 'shipped-bass-v10-baseline-v1'
EXPECTED = (
    '9864e9c040629f5412fab624dbf565ebc04677d0d046ebf22d188da80dec00c7',
    '59e3b1cf6c2b27198c9aa046127f98a1aff15ee287a34e981113382c01f371dd',
)


def hashes():
    if tuple(expected for _, expected, _ in service.CHECKPOINTS) != EXPECTED:
        raise ValueError('The shipped bass baseline changed; declare a new experiment')
    arrays = {path.name: digest(path) for path, _, _ in service.CHECKPOINTS}
    if tuple(arrays.values()) != EXPECTED:
        raise ValueError('The shipped bass arrays changed')
    root = Path(__file__).resolve().parents[1]
    return {'version': VERSION, 'arrays': arrays, 'thresholds': [.1, .1],
            'sources': {name: digest(root / name) for name in (
                'training/current_bass_baseline.py', 'training/bass_training_data.py',
                'backend/app/services/bass_verifier.py',
                'backend/app/services/note_context_model.py',
                'backend/app/services/note_evidence.py',
                'backend/app/services/piano_transcription.py')}}


def prepare(items):
    hashes()
    verifier = service.BassVerifier()
    result = []
    for item in items:
        x, events, eligible = item['x'], item['events'], item['eligible']
        if (x.shape != (len(events), len(NAMES)) or events.shape != (len(x), 4)
                or eligible.shape != (len(x),) or eligible.dtype != bool
                or not np.isfinite(x).all() or not np.isfinite(events).all()):
            raise ValueError('Invalid cached bass baseline evidence')
        p, g = [model.probability(x[:, :model.feature_count]) for model in verifier.models]
        if [model.threshold for model in verifier.models] != [.1, .1]:
            raise ValueError('Changed shipped bass thresholds')
        keep = ~(eligible & (p < .1) & (g < .1))
        result.append({**item, 'baseline_keep': keep.copy(), 'keep': keep.copy(),
                       'shared': eligible.copy(), 'baseline_p': p, 'baseline_g': g})
    return result
