import numpy as np
import pytest
from left_precision_v42 import CONTEXT_WIDTH, combine, model, presence_targets, probabilities
from periodicity_features import NAMES
from test_chord_boundary_v36 import item


def fixture():
    data = item([[3., 5., 48, 80], [3., 6., 52, 90]])
    data["frames"] = np.zeros((2, 40, 9), np.float32)
    data["periodicity"] = np.zeros((2, len(NAMES)), np.float32)
    return data


def test_single_supported_frame_vetoes_deletion_and_context_is_separate():
    reject = np.array([[.001, .999], [.001, .999]], np.float32)
    presence = np.zeros((2, 40), np.float32)
    presence[0, 20], presence[1, 2] = .95, .95
    result = combine(reject, presence)
    np.testing.assert_allclose(result[:, 1], [.05, .999], atol=1e-7)
    presence[0, 1] = np.nan
    with pytest.raises(ValueError, match="frame support"):
        combine(reject, presence)


def test_sub_frame_pitch_fragment_is_not_silently_labeled_absent():
    data = fixture()
    data.update(pitch_reference=np.array([[4.041, 4.042, 48.]]), y=np.array([1., 0.]), mask=np.ones(2, bool))
    targets = presence_targets(data)
    assert targets.shape == (2, 40)
    assert targets[0, 8:32].sum() == 1
    assert targets[1].sum() == 0


def test_full_hold_and_jittered_attack_keep_targets_remain_protected():
    data = fixture()
    data.update(pitch_reference=np.array([[3., 5., 48.], [2.8, 2.99, 52.]]),
                y=np.ones(2), mask=np.ones(2, bool))
    targets = presence_targets(data)
    assert targets[0, 8:32].sum() == 24
    assert targets[1, 8:32].sum() >= 1


@pytest.mark.parametrize("width", [12, 20])
def test_reference_frames_never_enter_inference(width):
    data = fixture()
    network = model(width)
    normalizer = np.zeros(CONTEXT_WIDTH), np.ones(CONTEXT_WIDTH)
    before = probabilities(network, data, normalizer)
    data.update(reference=np.empty((0, 3)), pitch_reference=np.empty((0, 3)),
                y=np.zeros(2), mask=np.zeros(2, bool), id="different-song")
    np.testing.assert_array_equal(before, probabilities(network, data, normalizer))
    assert np.all(before[:, 0] > .98)


def test_empty_observations_have_valid_shapes():
    data = item([])
    data.update(events=np.empty((0, 4)), frames=np.empty((0, 40, 9), np.float32),
        periodicity=np.empty((0, len(NAMES)), np.float32), pitch_reference=np.empty((0, 3)),
        y=np.empty(0), mask=np.empty(0, bool))
    assert presence_targets(data).shape == (0, 40)
    assert probabilities(model(12), data, (np.zeros(CONTEXT_WIDTH), np.ones(CONTEXT_WIDTH))).shape == (0, 2)
