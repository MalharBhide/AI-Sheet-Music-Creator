"""New training must include V11 mergers and protect correct pitch evidence."""

from types import SimpleNamespace

import current_bass_v11_baseline as baseline
import numpy as np
import pytest
from bass_residual_v11_data import acoustic_view, annotate, features
from bass_training_data import eligible
from train_bass_consensus import score


def test_v11_decisions_preserve_identity_and_extend_the_actual_interval(monkeypatch):
    events = np.array([[3., 4., 40., 90.], [4., 5., 40., 60.], [5., 6., 40., 70.], [7., 8., 44., 80.]])
    item = {'events': events, 'x': np.zeros((4, 52), np.float32), 'eligible': eligible(events, 12.)}
    monkeypatch.setattr(baseline, 'v10_prepare', lambda rows: [{**rows[0], 'keep': np.ones(4, bool),
                        'shared': item['eligible'], 'baseline_p': np.ones(4), 'baseline_g': np.ones(4)}])
    verifier = SimpleNamespace(models=[SimpleNamespace(probability=lambda x: np.zeros(len(x))) for _ in range(2)])
    result = baseline.decisions(item, 12., verifier)
    assert result['raw_indices'].tolist() == [0, 3]
    assert result['changed_intervals'].tolist() == [True, False]
    assert result['merged_boundaries'] == 2
    np.testing.assert_array_equal(result['events'], [[3., 6., 40., 90.], [7., 8., 44., 80.]])
    np.testing.assert_array_equal(events[:, 1], [4., 5., 6., 8.])
    assert 'x' not in result  # duration-dependent evidence must be recomputed


def test_v11_decisions_reject_stale_eligibility_or_invalid_clock():
    events = np.array([[3., 4., 40., 90.]])
    item = {'events': events, 'x': np.zeros((1, 52)), 'eligible': np.zeros(1, bool)}
    with pytest.raises(AssertionError):
        baseline.decisions(item, 12.)
    with pytest.raises(ValueError, match='duration'):
        baseline.decisions(item, np.nan)


def fixture():
    events = np.array([[3., 5., 40., 90.], [3., 5., 52., 60.], [4., 5., 44., 50.]])
    return {'id': 'original-octave', 'corpus': 'original-fixture', 'events': events,
            'reference': events[:2, :3].copy(), 'pitch_reference': events[:2, :3].copy(),
            'x': np.zeros((3, 52), np.float32), 'context': np.array([.9, .8, .2]),
            'guardian': np.array([.8, .7, .1]), 'eligible': np.ones(3, bool), 'seconds': 12.}


def test_v11_labels_protect_real_octaves_holds_and_unmodified_feature_clocks():
    item = fixture()
    actual = annotate(item)
    assert actual['y'].tolist() == [1., 1., 0.]
    assert actual['protected_offset_assignments'] == 2
    assert actual['x'].shape == (3, 74) and acoustic_view(actual['x']).shape == (3, 28)
    result = score([actual], [np.array([1., 0., 0.])], [np.array([1., 0., 0.])], .1, .1)
    assert not result['passes']
    assert not result['per_recording'][0]['held_references_preserved']


def test_v11_features_are_label_free_and_pitch_relative():
    item = fixture()
    first = annotate(item)['x']
    item['events'][:, 2] += 5
    item['reference'][:, 2] += 5
    item['pitch_reference'][:, 2] += 5
    np.testing.assert_array_equal(first, annotate(item)['x'])
    item['reference'] = np.empty((0, 3))
    item['pitch_reference'] = np.empty((0, 3))
    np.testing.assert_array_equal(first, annotate(item)['x'])


def test_v11_empty_rows_and_per_head_nonfinite_confidence():
    empty = features(np.empty((0, 4)), np.empty((0, 52)), np.array([]), np.array([]))
    assert empty.shape == (0, 74) and acoustic_view(empty).shape == (0, 28)
    item = fixture()
    with pytest.raises(ValueError, match='confidence'):
        features(item['events'], item['x'], np.full(3, 1.2), np.zeros(3))
