"""Wrong-pitch training must preserve true octaves and prior bass decisions."""

import bass_residual_evidence as evidence
import numpy as np
import pytest
from train_bass_consensus import score


def fixture():
    events = np.array([[3., 5., 40., 90.], [3., 5., 52., 60.], [4., 5., 44., 50.]])
    return {'id': 'original-bass-octave', 'corpus': 'original-fixture', 'group': 'train',
            'events': events, 'x': np.zeros((3, 52), np.float32),
            'reference': np.array([[3., 5., 40.]]), 'pitch_reference': np.array([[3., 5., 40.]]),
            'baseline_keep': np.array([True, True, False]), 'eligible': np.ones(3, dtype=bool),
            'baseline_p': np.array([.9, .2, .01]), 'baseline_g': np.array([.8, .1, .01]), 'seconds': 10.}


def test_baseline_rejections_are_absent_from_relationships_and_recomputed_targets():
    item = fixture()
    actual = evidence.advance(item)
    assert actual['raw_indices'].tolist() == [0, 1] and actual['raw_count'] == 3
    assert actual['events'][:, 2].tolist() == [40., 52.]
    assert actual['x'].shape == (2, 74)
    assert evidence.acoustic_view(actual['x']).shape == (2, 54)
    assert actual['y'].tolist() == [1., 0.]
    assert actual['mask'].all()
    assert item['events'].shape == (3, 4) and item['x'].shape == (3, 52)
    assert actual['x'][1, 60] > 0  # lower octave's confidence, never a pitch ID


def test_features_are_pitch_relative_and_do_not_read_annotations():
    item = fixture()
    first = evidence.advance(item)['x']
    item['events'][:, 2] += 7
    item['reference'][:, 2] += 7
    item['pitch_reference'][:, 2] += 7
    np.testing.assert_array_equal(first, evidence.advance(item)['x'])
    item['reference'] = np.empty((0, 3))
    item['pitch_reference'] = np.empty((0, 3))
    np.testing.assert_array_equal(first, evidence.advance(item)['x'])


def test_rejecting_true_octave_fails_attack_hold_and_pitch_support_gates():
    item = fixture()
    item['reference'] = item['events'][:2, :3].copy()
    item['pitch_reference'] = item['reference'].copy()
    actual = evidence.advance(item)
    assert actual['y'].tolist() == [1., 1.]
    result = score([actual], [np.array([1., 0.])], [np.array([1., 0.])], .1, .1)
    assert not result['passes']
    assert not result['per_recording'][0]['held_references_preserved']
    assert not result['per_recording'][0]['reference_pitch_coverage_preserved']


def test_no_retained_notes_produce_empty_guardian_features_without_sklearn():
    item = fixture()
    item['baseline_keep'][:] = False
    actual = evidence.advance(item)
    assert actual['x'].shape == (0, 74) and actual['events'].shape == (0, 4)
    assert evidence.acoustic_view(actual['x']).shape == (0, 54)
    with pytest.raises(ValueError, match='residual bass evidence'):
        evidence.acoustic_view(np.zeros((0, 52)))


def test_invalid_individual_confidence_cannot_hide_inside_a_valid_mean():
    item = fixture()
    with pytest.raises(ValueError, match='shipped bass confidence'):
        evidence.features(item['events'], item['x'], np.full(3, 1.2), np.zeros(3))
    with pytest.raises(ValueError, match='shipped bass confidence'):
        evidence.features(item['events'], item['x'], np.full(3, np.nan), np.zeros(3))
