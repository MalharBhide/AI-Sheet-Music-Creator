"""Bind new fitting to actual seven-stage V16 bass survivors, never user jobs."""

from pathlib import Path

import numpy as np
from app.services import bass_temporal_refinement
from app.services.bass_harmonic import features
from bass_training_data import eligible
from prepare_robust_training_stems import digest

VERSION = 'shipped-bass-v16-baseline-v1'
ROUTE_SHA = 'd46c98ae93a62abc6b3ca8bcf6a71ba5416d304ddbc63f5b9443ba9f5d835182'


def hashes():
    # The old guard remains unchanged and deliberately rejects V16. Read its
    # declared asset/source inventory without pretending its route is current.
    from app.services import (
        bass_articulation,
        bass_harmonic,
        bass_residual,
        bass_temporal,
        bass_texture,
        bass_verifier,
    )

    root = Path(__file__).resolve().parents[1]
    if digest(root / 'backend/app/services/piano_transcription.py') != ROUTE_SHA:
        raise ValueError('Changed V16 raw production route; declare a new baseline')
    arrays = {}
    for checkpoints in (bass_verifier.CHECKPOINTS, bass_articulation.CHECKPOINTS,
                        bass_residual.CHECKPOINTS, bass_texture.CHECKPOINTS, bass_harmonic.CHECKPOINTS):
        for path, expected, *_ in checkpoints:
            if digest(path) != expected:
                raise ValueError('Changed V16 upstream array')
            arrays[path.name] = expected
    for service in (bass_temporal, bass_temporal_refinement):
        if digest(service.ASSET) != service.EXPECTED_SHA:
            raise ValueError('Changed V16 temporal array')
        arrays[service.ASSET.name] = service.EXPECTED_SHA
    names = ('backend/app/services/piano_transcription.py', 'backend/app/services/bass_verifier.py',
             'backend/app/services/bass_articulation.py', 'backend/app/services/bass_residual.py',
             'backend/app/services/bass_texture.py', 'backend/app/services/bass_harmonic.py',
             'backend/app/services/bass_temporal.py', 'backend/app/services/bass_temporal_refinement.py',
             'backend/app/services/temporal_note_model.py', 'backend/app/services/note_evidence.py',
             'backend/app/services/candidate_relations.py', 'backend/app/services/note_context_model.py',
             'training/current_bass_v16_baseline.py')
    return {'version': VERSION, 'arrays': arrays,
            'thresholds': [[.1,.1],[.05,.05],[.05,.1],[.025,.3],[.025,.3],[.005,.3],[.2,.3]],
            'sources': {name:digest(root/name) for name in names}}


def decisions(item, verifier=None):
    events, base_x, frames = [item[key] for key in ('events','base_x','frames')]
    if (events.shape != (len(events),4) or base_x.shape != (len(events),52)
            or frames.shape != (len(events),40,9)
            or any(not np.isfinite(a).all() for a in (events,base_x,frames))
            or np.any((frames<0)|(frames>1))):
        raise ValueError('Invalid actual V15 temporal/acoustic evidence')
    np.testing.assert_array_equal(item['eligible'],eligible(events,item['duration']))
    verifier = verifier or bass_temporal_refinement.BassTemporalRefinement()
    if (verifier.model.threshold,verifier.guardian.threshold) != (.2,.3):
        raise ValueError('Changed V16 thresholds')
    context = features(events,base_x)
    p, g = verifier.model.probability(frames,context), verifier.guardian.probability(base_x)
    if (p.shape != (len(events),) or g.shape != (len(events),)
            or not np.isfinite(p).all() or not np.isfinite(g).all()):
        raise ValueError('Invalid V16 probabilities')
    keep = ~(item['eligible'] & (p<.2) & (g<.3))
    return {'events':events[keep], 'base_x':base_x[keep], 'frames':frames[keep],
            'eligible':item['eligible'][keep], 'v15_indices':np.flatnonzero(keep),
            'v16_rejections':int((~keep).sum())}
