"""Meaningful acoustic rejection, hold preservation and probability contract checks."""

import numpy as np
import pytest
from declutter_local_model import model, probabilities
from score_declutter_local import choose, evaluate
from train_declutter_local_v23 import supervised


def item(events, reference):
    return {'id': 'fixture', 'corpus': 'original', 'duration': 30.,
            'events': np.asarray(events, float).reshape(-1, 4),
            'reference': np.asarray(reference, float).reshape(-1, 3),
            'pitch_reference': np.asarray(reference, float).reshape(-1, 3),
            'eligible': np.ones(len(events), bool)}


def test_unsupported_pitch_removed_without_retained_clock_or_hold_changes():
    i = item([[4, 6, 48, 90], [5, 5.3, 37, 50], [7, 7.4, 48, 90]],
             [[4, 6, 48], [7, 7.4, 48]])
    result = evaluate([i], [np.array([[.99, .01], [.01, .99], [.99, .01]])],
                      [np.array([.9, .1, .9])], .9, .3)
    assert result['passes'] and result['false_notes_removed'] == 1
    assert result['retimed_notes'] == result['onset_error_reduction_seconds'] == 0
    row = result['per_recording'][0]
    assert row['matches']['attack']['lost'] == row['matches']['hold']['lost'] == 0
    assert row['coverage']['lost_reference_seconds'] == 0
    assert row['timing']['before_repeated_spacing_error_sum'] == row['timing']['after_repeated_spacing_error_sum']


def test_remove_real_hold_fails_even_when_removal_prediction_confident():
    i = item([[4, 8, 48, 90]], [[4, 8, 48]])
    result = evaluate([i], [np.array([[.01, .99]])], [np.array([.1])], .9, .3)
    assert not result['passes']
    assert result['per_recording'][0]['matches']['hold']['lost'] == 1
    assert result['per_recording'][0]['coverage']['lost_reference_seconds'] == 4


def test_guardian_equality_abstains():
    i = item([[4, 4.3, 37, 50]], [])
    result = evaluate([i], [np.array([[.01, .99]])], [np.array([.3])], .9, .3)
    assert result['per_recording'][0]['removed_notes'] == 0


def test_margin_requires_real_false_note_gain():
    i = item([[4, 4.3, 37, 50]], [])
    winner, _ = choose([i], [np.array([[.08, .92]])], [np.array([.1])], [.9], [.3], {})
    assert winner is None  # raw removes; stricter .95 preserves: no margin gain.


@pytest.mark.parametrize('bad', [np.array([[.8, .8]]), np.array([[np.nan, .8]]), np.array([[1.2, -.2]])])
def test_invalid_probabilities_rejected(bad):
    i = item([[4, 4.3, 37, 50]], [])
    with pytest.raises(ValueError, match='probabilities'):
        evaluate([i], [bad], [np.array([.1])], .9, .3)


def test_binary_network_probability_shapes_and_sum():
    import torch
    torch.set_num_threads(1)
    net = model(24)
    context = np.zeros((3, 84), np.float32)
    p = probabilities(net, np.zeros((3, 40, 9), np.float32), np.zeros((3, 61, 18), np.float32),
                      context, (np.zeros(84), np.ones(84)))
    assert p.shape == (3, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.)
    assert probabilities(net, np.zeros((0, 40, 9)), np.zeros((0, 61, 18)), context[:0],
                         (np.zeros(84), np.ones(84))).shape == (0, 2)


def test_supervision_excludes_ambiguous_and_ineligible_examples():
    i = {'corpus': 'original', 'x': np.arange(3*84).reshape(3,84),
         'frames': np.zeros((3,40,9)), 'attack_frames': np.zeros((3,61,18)),
         'y': np.array([1.,0.,0.]), 'mask': np.array([True,False,True]),
         'eligible': np.array([True,True,False])}
    x, _, _, y, weights, counts = supervised([i], 12.)
    assert len(x) == 1 and y.tolist() == [0] and weights.tolist() == [1.]
    assert counts['original']['class_counts'] == [1,0]
