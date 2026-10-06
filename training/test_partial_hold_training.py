"""Train partial holds as protected positives without changing model evidence."""

import numpy as np
import pytest
from partial_hold_training import support_boost
from train_declutter_local_v24 import supervised


def fixture():
    return {'events': np.array([[4., 5., 48, 90], [6., 7., 48, 90], [8., 9., 37, 90]]),
            'pitch_reference': np.array([[4.99, 5.5, 48], [6., 7., 48]]),
            'y': np.array([1., 1., 0.]), 'mask': np.ones(3, bool),
            'eligible': np.ones(3, bool), 'corpus': 'original',
            'x': np.arange(3*84).reshape(3, 84), 'frames': np.zeros((3, 40, 9)),
            'attack_frames': np.zeros((3, 61, 18))}


def test_small_positive_overlap_receives_boost_but_full_hold_and_noise_do_not():
    i = fixture()
    np.testing.assert_array_equal(support_boost(i, 4.), [4., 1., 1.])
    i['mask'][0] = False
    np.testing.assert_array_equal(support_boost(i, 4.), np.ones(3))


def test_training_reference_overlap_changes_only_loss_not_acoustic_inputs_or_labels():
    i = fixture()
    x, f, af, y, w, counts = supervised([i], 16., 4.)
    for actual, original in ((x, i['x']), (f, i['frames']), (af, i['attack_frames'])):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(y, [0, 0, 1])
    assert w[0]/w[1] == 4. and w[1]/w[2] == 16.
    assert counts['original']['boosted_partial_hold_events'] == 1


@pytest.mark.parametrize('multiplier', [0., float('nan'), float('inf')])
def test_invalid_loss_boost_rejected(multiplier):
    with pytest.raises(ValueError):
        support_boost(fixture(), multiplier)
