"""A new baseline must retain V14 decisions and protect all older truth."""

from types import SimpleNamespace

import current_bass_v15_baseline as baseline
import numpy as np
import pytest
from bass_temporal_data import annotate
from bass_training_data import eligible
from harmonic_candidate_features import BASE_NAMES, acoustic_view, features
from train_bass_consensus import score


def models(first, second, thresholds=(.005, .3)):
    return SimpleNamespace(model=SimpleNamespace(threshold=thresholds[0],
           probability=lambda frames, x: np.asarray(first, float)),
           guardian=SimpleNamespace(threshold=thresholds[1], probability=lambda x: np.asarray(second, float)))


def test_actual_survivors_preserve_long_holds_and_threshold_equality(monkeypatch):
    monkeypatch.setattr(baseline, 'hashes', lambda: {})
    events = np.array([[3., 9., 40., 90.], [4., 5., 52., 60.], [2., 4., 44., 80.]])
    item = {'events': events, 'base_x': np.zeros((3, 52), np.float32), 'eligible': eligible(events, 12.), 'frames': np.full((3, 40, 9), .4, np.float32)}
    result = baseline.decisions(item, 12., models([.005, 0., 0.], [0., 0., 0.]))
    assert result['v14_indices'].tolist() == [0, 2]
    np.testing.assert_array_equal(result['events'], events[[0, 2]])
    assert result['temporal_rejections'] == 1
    np.testing.assert_array_equal(result['frames'], item['frames'][[0, 2]])
    np.testing.assert_array_equal(events[:, 1], [9., 5., 4.])


def test_stale_mask_or_different_threshold_is_not_a_current_baseline(monkeypatch):
    monkeypatch.setattr(baseline, 'hashes', lambda: {})
    item = {'events': np.array([[3., 4., 40., 90.]]), 'base_x': np.zeros((1, 52)), 'eligible': np.zeros(1, bool), 'frames': np.zeros((1, 40, 9), np.float32)}
    with pytest.raises(AssertionError):
        baseline.decisions(item, 12.)
    item['eligible'][:] = True
    with pytest.raises(ValueError, match='thresholds'):
        baseline.decisions(item, 12., models([0.], [0.], (.1, .3)))
    with pytest.raises(ValueError, match='duration-correct'):
        baseline.decisions(item, np.nan)


def test_sealed_array_contract_and_older_guards_stay_stale(monkeypatch):
    from current_bass_v14_baseline import hashes as old_hashes

    original = baseline.digest
    monkeypatch.setattr(baseline, 'digest', lambda path: baseline.ROUTE_SHA if path.name == 'piano_transcription.py' else original(path))
    assert len(baseline.hashes()['arrays']) == 11
    with pytest.raises(ValueError, match='V14 production route'):
        old_hashes()


def test_changed_route_is_rejected(monkeypatch):
    original = baseline.digest
    monkeypatch.setattr(baseline, 'digest', lambda path: 'stale' if path.name == 'piano_transcription.py' else original(path))
    with pytest.raises(ValueError, match='V15 production route'):
        baseline.hashes()


def fixture():
    events = np.array([[3., 5., 40., 90.], [3., 5., 52., 60.], [4., 5., 44., 50.]])
    x = np.zeros((3, 52), np.float32)
    x[:, BASE_NAMES.index('onset_at_attack')] = [.9, .2, .4]
    x[:, BASE_NAMES.index('note_mean')] = [.8, .6, .7]
    return {'id': 'original-octave', 'corpus': 'original-fixture', 'events': events,
            'reference': events[:2, :3].copy(), 'pitch_reference': events[:2, :3].copy(),
            'base_x': x, 'eligible': np.ones(3, bool), 'seconds': 12.}


def test_new_features_use_observed_harmonic_attacks_and_releases():
    item = fixture()
    x = features(item['events'], item['base_x'])
    assert x.shape == (3, 84) and acoustic_view(x).shape == (3, 52)
    np.testing.assert_allclose(x[1, -12:-8], [.9, -.7, .9, .8])
    np.testing.assert_allclose(x[0, -12:], [0., .9, 0., 0.] * 3)
    alone = features(item['events'][1:2], item['base_x'][1:2])
    assert alone[0, -12] == 0.  # a deleted lower candidate cannot remain in context


def test_features_are_pitch_relative_and_labels_protect_real_octaves_and_holds():
    item = fixture()
    x = features(item['events'], item['base_x'])
    item['events'][:, 2] += 5
    item['reference'][:, 2] += 5
    item['pitch_reference'][:, 2] += 5
    np.testing.assert_array_equal(x, features(item['events'], item['base_x']))
    labeled = annotate(item)
    assert labeled['y'].tolist() == [1., 1., 0.]
    result = score([labeled], [np.array([1., 0., 0.])], [np.array([1., 0., 0.])], .1, .1)
    assert not result['passes'] and not result['per_recording'][0]['held_references_preserved']
    item['reference'] = item['pitch_reference'] = np.empty((0, 3))
    np.testing.assert_array_equal(x, features(item['events'], item['base_x']))


@pytest.mark.parametrize('x', [np.full((1, 52), np.nan), np.zeros((2, 52)), np.zeros((1, 51))])
def test_invalid_acoustics_do_not_enter_features(x):
    with pytest.raises(ValueError):
        features(np.array([[3., 4., 40., 90.]]), x)
