"""Supervised CNN rows must remain bound to source/interval/frame identity."""

import numpy as np
import pytest
from bass_temporal_v15_data import piano_items, read_items
from bass_training_data import eligible, identity
from prepare_robust_training_stems import digest


def cache(tmp_path, change=None):
    audio = tmp_path / 'authored.wav'
    audio.write_bytes(b'original test evidence')
    row = {'id': 'original-example', 'source_group': 'original-source', 'group': 'train',
           'corpus': 'original-test', 'audio': str(audio), 'audio_sha256': digest(audio),
           'reference': [[3., 6., 40]], 'pitch_reference': [[3., 6., 40]],
           'duration': 12., 'seconds': 11.5, 'retained_v15': 2}
    events = np.array([[3., 6., 40, 80], [4., 5., 52, 70]])
    frames = np.full((2, 40, 9), .5, np.float32)
    mask = eligible(events, 12.)
    if change == 'frames':
        frames[1, 0, 0] = 1.1
    elif change == 'eligible':
        mask[0] = False
    folder = tmp_path / 'features'
    folder.mkdir()
    path = folder / (row['id'] + '.npz')
    np.savez(path, events=events, frames=frames, base_x=np.zeros((2, 52), np.float32),
             eligible=mask, identity=identity(row), plan_sha256='sealed-plan')
    row['cache_sha256'] = digest(path)
    return row


def test_frames_and_supervision_stay_on_the_same_actual_intervals(tmp_path):
    row = cache(tmp_path)
    items = read_items(tmp_path, [row], 'sealed-plan', ('train',), True)
    assert len(items) == 1
    assert items[0]['y'].tolist() == [1., 0.]
    assert items[0]['mask'].tolist() == [True, True]
    assert items[0]['frames'].shape == (2, 40, 9)
    assert read_items(tmp_path, [row], 'sealed-plan', ('validation',), True) == []


@pytest.mark.parametrize('change', ['source', 'frames', 'eligible'])
def test_resigned_invalid_evidence_or_swapped_source_cannot_fit(tmp_path, change):
    row = cache(tmp_path, change)
    if change == 'source':
        row['source_group'] = 'different-performance'
    with pytest.raises((ValueError, AssertionError)):
        read_items(tmp_path, [row], 'sealed-plan', ('train',), True)


def test_piano_test_access_is_rejected_before_any_files_are_opened(tmp_path):
    with pytest.raises(ValueError, match='excludes test'):
        piano_items(tmp_path / 'not-present', ('test',))
