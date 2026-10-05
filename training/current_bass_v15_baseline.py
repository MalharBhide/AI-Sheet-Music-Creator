"""Bind fitting to actual survivors of the six-stage shipped V15 bass route."""

from pathlib import Path

import numpy as np
from app.services import (
    bass_articulation,
    bass_harmonic,
    bass_residual,
    bass_temporal,
    bass_texture,
    bass_verifier,
)
from bass_training_data import eligible
from prepare_robust_training_stems import digest

VERSION = 'shipped-bass-v15-baseline-v1'
ROUTE_SHA = '04f782935d4e8a4c51d9ba920418c14dc5dbd94fdce50345e6f43b56037c6295'


def hashes():
    root = Path(__file__).resolve().parents[1]
    if digest(root / 'backend/app/services/piano_transcription.py') != ROUTE_SHA:
        raise ValueError('Changed V15 production route; declare a new experiment')
    arrays = {}
    for checkpoints in (bass_verifier.CHECKPOINTS, bass_articulation.CHECKPOINTS,
                        bass_residual.CHECKPOINTS, bass_texture.CHECKPOINTS, bass_harmonic.CHECKPOINTS):
        for path, expected, *_ in checkpoints:
            if digest(path) != expected:
                raise ValueError('Changed V15 baseline array')
            arrays[path.name] = expected
    if digest(bass_temporal.ASSET) != bass_temporal.EXPECTED_SHA:
        raise ValueError('Changed V15 temporal array')
    arrays[bass_temporal.ASSET.name] = bass_temporal.EXPECTED_SHA
    names = ('backend/app/services/piano_transcription.py', 'backend/app/services/bass_verifier.py',
             'backend/app/services/bass_articulation.py', 'backend/app/services/bass_residual.py',
             'backend/app/services/bass_texture.py', 'backend/app/services/bass_harmonic.py',
             'backend/app/services/bass_temporal.py', 'backend/app/services/temporal_note_model.py',
             'backend/app/services/note_evidence.py',
             'backend/app/services/candidate_relations.py', 'backend/app/services/note_context_model.py',
             'training/current_bass_v15_baseline.py')
    return {'version': VERSION, 'arrays': arrays,
            'thresholds': [[.1, .1], [.05, .05], [.05, .1], [.025, .3], [.025, .3], [.005, .3]],
            'sources': {name: digest(root / name) for name in names}}


def decisions(item, duration, temporal=None):
    """Only per-note frames/channels survive deletion; recompute neighbors later."""
    hashes()
    events, x, frames = item['events'], item['base_x'], item['frames']
    if (not np.isfinite(duration) or duration <= 0 or events.shape != (len(x), 4)
            or x.shape != (len(events), 52) or frames.shape != (len(events), 40, 9)
            or not np.isfinite(x).all() or not np.isfinite(events).all()
            or not np.isfinite(frames).all() or np.any((frames < 0) | (frames > 1))):
        raise ValueError('Invalid duration-correct V14 temporal evidence')
    np.testing.assert_array_equal(item['eligible'], eligible(events, duration))
    temporal = temporal or bass_temporal.BassTemporal()
    if (temporal.model.threshold, temporal.guardian.threshold) != (.005, .3):
        raise ValueError('Changed V15 temporal thresholds')
    context = bass_harmonic.features(events, x)
    p, g = temporal.model.probability(frames, context), temporal.guardian.probability(x)
    keep = ~(item['eligible'] & (p < .005) & (g < .3))
    return {'events': events[keep], 'base_x': x[keep], 'frames': frames[keep],
            'eligible': item['eligible'][keep], 'v14_indices': np.flatnonzero(keep),
            'temporal_rejections': int((~keep).sum())}
