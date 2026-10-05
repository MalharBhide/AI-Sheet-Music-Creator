"""Bind new training to the shipped three-stage bass route and actual intervals."""

from pathlib import Path

import numpy as np
from app.services import bass_articulation, bass_residual, bass_verifier
from bass_training_data import eligible
from prepare_robust_training_stems import digest

VERSION = 'shipped-bass-v12-baseline-v1'
ROUTE_SHA = '741b5befe2b5f5ded1ae62ebb2811b37f2587b0e62c32a5a9fc160d1b3aa4a8a'


def hashes():
    root = Path(__file__).resolve().parents[1]
    route = root / 'backend/app/services/piano_transcription.py'
    if digest(route) != ROUTE_SHA:
        raise ValueError('Changed V12 production route; declare a new experiment')
    arrays = {}
    for checkpoints in (bass_verifier.CHECKPOINTS, bass_articulation.CHECKPOINTS, bass_residual.CHECKPOINTS):
        for path, expected, *_ in checkpoints:
            actual = digest(path)
            if actual != expected:
                raise ValueError('Changed V12 baseline array')
            arrays[path.name] = actual
    names = ('backend/app/services/piano_transcription.py', 'backend/app/services/bass_verifier.py',
             'backend/app/services/bass_articulation.py', 'backend/app/services/bass_residual.py',
             'backend/app/services/note_evidence.py', 'backend/app/services/candidate_relations.py',
             'backend/app/services/note_context_model.py', 'training/current_bass_v12_baseline.py')
    return {'version': VERSION, 'arrays': arrays, 'thresholds': [[.1, .1], [.05, .05], [.05, .1]],
            'sources': {name: digest(root / name) for name in names}}


def decisions(item, duration, baseline=None, residual=None):
    """Filter sealed post-V11 evidence; deletion never changes another interval."""
    hashes()
    events, x = item['events'], item['base_x']
    if (not np.isfinite(duration) or duration <= 0 or events.shape != (len(x), 4)
            or x.shape != (len(events), 52) or not np.isfinite(x).all()
            or not np.isfinite(events).all()):
        raise ValueError('Invalid duration-correct V11 evidence')
    np.testing.assert_array_equal(item['eligible'], eligible(events, duration))
    baseline = baseline or bass_verifier.BassVerifier()
    residual = residual or bass_residual.BassResidual()
    if [m.threshold for m in residual.models] != [.05, .1]:
        raise ValueError('Changed V12 residual thresholds')
    context, guardian = [m.probability(x[:, :m.feature_count]) for m in baseline.models]
    residual_x = bass_residual.features(events, x, context, guardian)
    p, g = [m.probability(view) for m, view in zip(residual.models,
            (residual_x, bass_residual.acoustic_view(residual_x)), strict=True)]
    keep = ~(item['eligible'] & (p < .05) & (g < .1))
    return {'events': events[keep], 'base_x': x[keep], 'eligible': item['eligible'][keep],
            'context': context[keep], 'guardian': guardian[keep],
            'residual_context': p[keep], 'residual_guardian': g[keep],
            'v11_indices': np.flatnonzero(keep), 'residual_rejections': int((~keep).sum())}
