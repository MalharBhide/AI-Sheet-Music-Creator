"""Bind fitting to actual survivors of the five-stage shipped V14 bass route."""

from pathlib import Path

import numpy as np
from app.services import (
    bass_articulation,
    bass_harmonic,
    bass_residual,
    bass_texture,
    bass_verifier,
)
from bass_training_data import eligible
from prepare_robust_training_stems import digest

VERSION = 'shipped-bass-v14-baseline-v1'
ROUTE_SHA = 'fd583fc64f0bdb4a882a0d0bac708cda713dcb3bbf43f1ea20c628bf3f020b73'


def hashes():
    root = Path(__file__).resolve().parents[1]
    if digest(root / 'backend/app/services/piano_transcription.py') != ROUTE_SHA:
        raise ValueError('Changed V14 production route; declare a new experiment')
    arrays = {}
    for checkpoints in (bass_verifier.CHECKPOINTS, bass_articulation.CHECKPOINTS,
                        bass_residual.CHECKPOINTS, bass_texture.CHECKPOINTS, bass_harmonic.CHECKPOINTS):
        for path, expected, *_ in checkpoints:
            if digest(path) != expected:
                raise ValueError('Changed V14 baseline array')
            arrays[path.name] = expected
    names = ('backend/app/services/piano_transcription.py', 'backend/app/services/bass_verifier.py',
             'backend/app/services/bass_articulation.py', 'backend/app/services/bass_residual.py',
             'backend/app/services/bass_texture.py', 'backend/app/services/bass_harmonic.py',
             'backend/app/services/note_evidence.py',
             'backend/app/services/candidate_relations.py', 'backend/app/services/note_context_model.py',
             'training/current_bass_v14_baseline.py')
    return {'version': VERSION, 'arrays': arrays,
            'thresholds': [[.1, .1], [.05, .05], [.05, .1], [.025, .3], [.025, .3]],
            'sources': {name: digest(root / name) for name in names}}


def decisions(item, duration, harmonic=None):
    """Deletion reuses only interval-independent channels; recompute relations."""
    hashes()
    events, x = item['events'], item['base_x']
    if (not np.isfinite(duration) or duration <= 0 or events.shape != (len(x), 4)
            or x.shape != (len(events), 52) or not np.isfinite(x).all()
            or not np.isfinite(events).all()):
        raise ValueError('Invalid duration-correct V13 evidence')
    np.testing.assert_array_equal(item['eligible'], eligible(events, duration))
    harmonic = harmonic or bass_harmonic.BassHarmonic()
    if [m.threshold for m in harmonic.models] != [.025, .3]:
        raise ValueError('Changed V14 harmonic thresholds')
    harmonic_x = bass_harmonic.features(events, x)
    p, g = [model.probability(view) for model, view in zip(harmonic.models,
            (harmonic_x, bass_harmonic.acoustic_view(harmonic_x)), strict=True)]
    keep = ~(item['eligible'] & (p < .025) & (g < .3))
    return {'events': events[keep], 'base_x': x[keep], 'eligible': item['eligible'][keep],
            'v13_indices': np.flatnonzero(keep), 'harmonic_rejections': int((~keep).sum())}
