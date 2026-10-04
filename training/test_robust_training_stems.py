"""Codec augmentation keeps original labels and performer partitions."""

import numpy as np
import pytest
from prepare_robust_training_stems import codec_clock, preserve, select


def test_selection_stays_in_fixed_performer_roles_and_excludes_first_stem_source():
    manifest = {'tracks': {'train': [], 'validation': [], 'test': [{'id': 'must-not-read'}]}}
    for player in range(5):
        group = 'train' if player < 4 else 'validation'
        for genre in ('BN', 'Funk', 'Jazz', 'Rock', 'SS'):
            for role in ('comp', 'solo'):
                for index in range(3):
                    manifest['tracks'][group].append({'id': f'{player:02}_{genre}{index}-{role}',
                                                      'corpus': 'guitarset', 'player': f'{player:02}'})
    selected = select(manifest)
    assert len(selected) == 50
    assert all('1-' in item['id'] for item in selected)
    assert sum(item['group'] == 'train' for item in selected) == 40
    assert sum(item['group'] == 'validation' for item in selected) == 10
    manifest['tracks']['validation'][0]['player'] = '05'
    manifest['tracks']['validation'][1]['player'] = '05'
    with pytest.raises(ValueError, match='performer'):
        select(manifest)


def test_codec_check_accepts_gapless_and_rejects_shifted_or_truncated_audio():
    rate = 2000
    original = np.random.default_rng(182).normal(size=rate * 3).astype(np.float32)
    assert codec_clock(original, original * .9, rate)['codec_lag_samples'] == 0
    shifted = np.roll(original, 50)
    with pytest.raises(ValueError, match='clock drift'):
        codec_clock(original, shifted, rate)
    with pytest.raises(ValueError, match='duration'):
        codec_clock(original, original[:-100], rate)
    with pytest.raises(ValueError, match='Insufficient'):
        codec_clock(np.zeros(rate), np.zeros(rate), rate)


def test_preparation_preserves_completed_evidence_and_does_not_clobber_it(tmp_path):
    path = tmp_path / 'plan.json'
    preserve(path, {'seed': 17})
    before = path.read_bytes()
    preserve(path, {'seed': 17})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='overwrite'):
        preserve(path, {'seed': 18})
    assert path.read_bytes() == before
