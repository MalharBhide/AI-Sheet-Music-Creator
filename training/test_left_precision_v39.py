import numpy as np
import pytest
from left_precision_v39 import keep, model, probabilities
from score_left_precision_v39 import baseline, evaluate
from test_chord_boundary_v36 import item


def test_removal_is_scoped_and_keeps_original_note_values():
    events = np.array([[0., 1., 48, 80], [3., 4., 48, 90], [5., 6., 60, 80],
                       [9., 10., 40, 80]])
    accepted = keep(events, np.tile([0., 1.], (4, 1)), 10., .99)
    assert accepted.tolist() == [True, False, True, True]
    np.testing.assert_array_equal(events[accepted], events[[0, 2, 3]])


@pytest.mark.parametrize("width", [12, 20])
def test_independent_model_starts_conservative_and_validates_windows(width):
    data = item([[3., 5., 48, 80], [3., 6., 52, 90]])
    data["frames"] = np.zeros((2, 40, 9), np.float32)
    network = model(width)
    normalizer = (np.zeros(100), np.ones(100))
    ps = probabilities(network, data, normalizer)
    assert ps.shape == (2, 2)
    assert np.all(ps[:, 0] > .98)
    np.testing.assert_array_equal(keep(data["events"], ps, data["duration"], .8), [True, True])
    data["attack_frames"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="physical"):
        probabilities(network, data, normalizer)


def test_empty_candidates_and_bad_probabilities():
    data = item([])
    data["frames"] = np.empty((0, 40, 9), np.float32)
    assert probabilities(model(12), data, (np.zeros(100), np.ones(100))).shape == (0, 2)
    assert keep(np.empty((0, 4)), np.empty((0, 2)), 10., .9).shape == (0,)
    with pytest.raises(ValueError, match="proposal"):
        keep(np.array([[3., 4., 48, 80]]), np.array([[.9, .9]]), 10., .9)


def test_removing_false_pitch_keeps_deployed_holds_and_repeat_spacing():
    data = item([[3., 4.08, 48, 80], [4.5, 5., 48, 80], [3., 4., 49, 70]],
                [[3., 4., 48], [4.5, 5., 48]])
    data.update(id="real-piano", corpus="musicnet-real-piano")

    class Release:
        threshold = .8

        def probability(self, value):
            result = np.zeros((len(value["events"]), 7))
            result[:, 3] = 1
            result[0] = [1, 0, 0, 0, 0, 0, 0]
            return result

    deployed = baseline([data], Release())
    assert deployed[0]["events"][0, 1] == 4.
    ps = [np.array([[1., 0.], [1., 0.], [0., 1.]])]
    result = evaluate([data], deployed, ps, .99)
    assert result["passes"]
    assert result["false_notes_removed"] == result["real_piano_gain"] == 1
    row = result["per_recording"][0]
    assert row["retained_clocks_identical"]
    assert row["timing"]["before_onset_release_error_sum"] == row["timing"]["after_onset_release_error_sum"]
    assert row["timing"]["before_repeated_spacing_error_sum"] == row["timing"]["after_repeated_spacing_error_sum"]
    bad = evaluate([data], deployed, [np.tile([0., 1.], (3, 1))], .99)
    assert not bad["passes"]
    assert bad["per_recording"][0]["matches"]["hold"]["lost"] == 2


def test_deleting_a_supported_fragment_is_a_regression():
    data = item([[3., 3.5, 48, 80], [3.5, 4., 48, 80]], [[3., 4., 48]])
    data.update(id="held-fragments", corpus="test")
    result = evaluate([data], [data], [np.array([[1., 0.], [0., 1.]])], .99)
    assert not result["passes"]
    assert not result["per_recording"][0]["coverage"]["passes"]
