"""Reproduce shipped V11 decisions, including merged durations, without labels."""

from pathlib import Path

import numpy as np
from app.services import bass_articulation as articulation
from bass_training_data import eligible
from current_bass_baseline import hashes as v10_hashes
from current_bass_baseline import prepare as v10_prepare
from prepare_robust_training_stems import digest

VERSION = 'shipped-bass-v11-baseline-v1'
EXPECTED = (
    'dec1bc16a2a41165e9c809fc8b11712f0392464f240ef6477b9c2d7195a6cd59',
    '101556c153e6b34688e6003dd4178152522290f3543fdceeeeb8aef1d8d3dc51',
)
SOURCES = {
    'backend/app/services/bass_articulation.py': 'e8cbcc8effe925461c12fbe735ab39144d06925b8ad2b05f89c53dbe284ab8a7',
    'backend/app/services/piano_transcription.py': '0b9a6b1a442c98bfe58a2110d912ad84b042f14f391195c048c5766a733f2e23',
}


def hashes():
    root = Path(__file__).resolve().parents[1]
    if tuple(expected for _, expected, _ in articulation.CHECKPOINTS) != EXPECTED:
        raise ValueError('Changed shipped V11 articulation declarations')
    arrays = {path.name: digest(path) for path, _, _ in articulation.CHECKPOINTS}
    if tuple(arrays.values()) != EXPECTED or any(digest(root / name) != value for name, value in SOURCES.items()):
        raise ValueError('Changed shipped V11 baseline; declare a new experiment')
    return {'version': VERSION, 'v10': v10_hashes(), 'articulation_arrays': arrays,
            'articulation_thresholds': [.05, .05], 'sources': SOURCES,
            'helper_sha256': digest(Path(__file__))}


def decisions(item, duration, verifier=None):
    """Return original-index identities and exact post-merge events, never old X."""
    hashes()
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid V11 waveform duration')
    np.testing.assert_array_equal(item['eligible'], eligible(item['events'], duration))
    prepared = v10_prepare([item])[0]
    verifier = verifier or articulation.BassArticulation()
    pairs = articulation.boundaries(prepared)
    repeat, guardian = [model.probability(articulation.features(prepared, pairs, acoustic=view))
                        for model, view in zip(verifier.models, (False, True), strict=True)]
    events, merged = articulation.merge(prepared, pairs, repeat, guardian, .05, .05)
    owner = np.arange(len(item['events']))
    keep = prepared['keep'].copy()
    for before, after in merged:
        first, second = int(owner[before]), int(owner[after])
        keep[second] = False
        owner[owner == second] = first
    indices = np.flatnonzero(keep)
    changed = events[:, 1] != item['events'][indices, 1]
    np.testing.assert_array_equal(events[:, [0, 2, 3]], item['events'][indices][:, [0, 2, 3]])
    if len(events) != len(indices) or np.any(events[:, 1] < item['events'][indices, 1]):
        raise ValueError('V11 changed identities or shortened holds')
    return {'events': events, 'raw_indices': indices, 'changed_intervals': changed,
            'merged_boundaries': len(merged), 'raw_count': len(item['events'])}
