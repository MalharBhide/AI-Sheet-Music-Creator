"""New fitting must match current decisions and cannot consume stale intervals."""

from types import SimpleNamespace

import current_bass_v12_baseline as baseline
import numpy as np
import pytest
from bass_residual_v12_data import acoustic_view, annotate, features
from bass_training_data import eligible
from train_bass_consensus import score


def models(probabilities, thresholds):
    return SimpleNamespace(models=[SimpleNamespace(feature_count=n, threshold=t,
           probability=lambda x, p=p: np.asarray(p, float))
           for n, p, t in zip((52, 26), probabilities, thresholds, strict=True)])


def test_current_decisions_do_not_change_kept_pitches_or_extended_holds(monkeypatch):
    monkeypatch.setattr(baseline, 'hashes', lambda: {})
    events = np.array([[3., 6., 40., 90.], [4., 5., 52., 60.], [2., 4., 44., 80.]])
    item = {'events': events, 'base_x': np.zeros((3, 52), np.float32), 'eligible': eligible(events, 12.)}
    first = models(([.9] * 3, [.8] * 3), (.1, .1))
    residual = models(([.05, 0., 0.], [0., 0., 0.]), (.05, .1))
    result = baseline.decisions(item, 12., first, residual)
    assert result['v11_indices'].tolist() == [0, 2]
    np.testing.assert_array_equal(result['events'], events[[0, 2]])
    assert result['residual_rejections'] == 1
    np.testing.assert_array_equal(events[:, 1], [6., 5., 4.])


def test_stale_window_mask_or_changed_threshold_cannot_enter_new_baseline(monkeypatch):
    monkeypatch.setattr(baseline, 'hashes', lambda: {})
    item = {'events': np.array([[3., 4., 40., 90.]]), 'base_x': np.zeros((1, 52)),
            'eligible': np.zeros(1, bool)}
    with pytest.raises(AssertionError):
        baseline.decisions(item, 12.)
    item['eligible'][:] = True
    with pytest.raises(ValueError, match='thresholds'):
        baseline.decisions(item, 12., residual=models(([0.], [0.]), (.1, .1)))
    with pytest.raises(ValueError, match='duration-correct'):
        baseline.decisions(item, np.nan)


def test_changed_production_route_is_rejected(monkeypatch):
    original = baseline.digest
    monkeypatch.setattr(baseline, 'digest', lambda path: 'stale' if path.name == 'piano_transcription.py' else original(path))
    with pytest.raises(ValueError, match='production route'):
        baseline.hashes()


def item():
    events = np.array([[3., 5., 40., 90.], [3., 5., 52., 60.], [4., 5., 44., 50.]])
    return {'id': 'original-octave', 'corpus': 'original-fixture', 'events': events,
            'reference': events[:2, :3].copy(), 'pitch_reference': events[:2, :3].copy(),
            'base_x': np.zeros((3, 52), np.float32), 'context': np.array([.9, .8, .2]),
            'guardian': np.array([.8, .7, .1]), 'residual_context': np.array([.8, .7, .2]),
            'residual_guardian': np.array([.9, .8, .1]), 'eligible': np.ones(3, bool), 'seconds': 12.}


def test_context_labels_protect_real_octaves_and_hold_offsets():
    actual = annotate(item())
    assert actual['x'].shape == (3, 76) and acoustic_view(actual['x']).shape == (3, 30)
    assert actual['y'].tolist() == [1., 1., 0.]
    result = score([actual], [np.array([1., 0., 0.])], [np.array([1., 0., 0.])], .1, .1)
    assert not result['passes'] and not result['per_recording'][0]['held_references_preserved']


def test_new_features_are_label_free_pitch_relative_and_recompute_neighbors():
    source = item()
    first = annotate(source)['x']
    source['events'][:, 2] += 5
    source['reference'][:, 2] += 5
    source['pitch_reference'][:, 2] += 5
    np.testing.assert_array_equal(first, annotate(source)['x'])
    source['reference'] = np.empty((0, 3))
    source['pitch_reference'] = np.empty((0, 3))
    np.testing.assert_array_equal(first, annotate(source)['x'])
    kept = features(source['events'][:1], source['base_x'][:1], source['context'][:1],
                    source['guardian'][:1], source['residual_context'][:1], source['residual_guardian'][:1])
    assert not np.array_equal(kept, first[:1])  # deleted neighbors must not remain as context


@pytest.mark.parametrize('confidence', [np.array([np.nan]), np.array([1.2]), np.array([])])
def test_invalid_release_confidence_cannot_be_used_for_fitting(confidence):
    with pytest.raises(ValueError, match='confidence'):
        features(np.array([[3., 4., 40., 90.]]), np.zeros((1, 52)), [.9], [.9], confidence, [.9])
