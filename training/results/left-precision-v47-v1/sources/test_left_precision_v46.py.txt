import numpy as np
import pytest
from left_precision_v39 import keep
from left_precision_v46 import locked, permissions, pitch_evaluate
from test_left_precision_v45 import baseline, weak_item


def test_additional_permissions_cannot_restore_rejected_notes():
    original = np.array([[1., 0.], [1., 0.], [.02, .98]])
    accepted = np.array([False, True, True])
    result = permissions(original, accepted)
    np.testing.assert_array_equal(result, [[0., 1.], [1., 0.], [.02, .98]])
    np.testing.assert_array_equal(original, [[1., 0.], [1., 0.], [.02, .98]])
    events = np.array([[3., 4., 40., 80.], [3.2, 4., 42., 80.], [3.5, 4., 44., 80.]])
    for threshold in (.9, .95, .98, .99, .995, .9999):
        assert not np.any(keep(events, result, 8., threshold) & ~accepted)


def test_restored_wrong_octave_is_intercepted_without_losing_fundamental():
    item = weak_item()
    report = pitch_evaluate([item], [baseline(item, (True, False))], [np.ones((2, 2))*.5], .9)
    assert report["passes"] and report["wrong_pitch_gain"] == 0
    assert report["per_recording"][0]["recovered_notes"] == 0


def test_new_removal_still_fails_if_it_deletes_a_genuine_fundamental():
    item = weak_item()
    report = pitch_evaluate([item], [baseline(item)], [np.array([[0., 1.], [0., 1.]])], .9)
    assert not report["passes"] and report["removed_fundamental_candidates"] == 1


def test_baseline_decisions_must_match_valid_identity_and_eligibility():
    item = weak_item()
    changed = baseline(item)
    changed["id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        locked([item], [changed], [np.ones((2, 2))*.5])
    item["events"][0, 0] = 0.
    with pytest.raises(ValueError, match="ineligible"):
        locked([item], [baseline(item, (False, True))], [np.ones((2, 2))*.5])


@pytest.mark.parametrize("probability,accepted", [
    (np.array([[np.nan, 0.]]), np.array([True])),
    (np.array([[1., 1.]]), np.array([True])),
    (np.array([[.5, .5]]), np.array([1])),
    (np.array([[.5, .5]]), np.array([[True]])),
    (np.array([[.5, .5]]), np.array(True)),
])
def test_malformed_permissions_fail_closed(probability, accepted):
    with pytest.raises(ValueError, match="permissions"):
        permissions(probability, accepted)


def test_empty_permissions_and_pitch_clips_remain_valid():
    item = weak_item()
    item["events"], item["y"] = np.empty((0, 4)), np.empty(0)
    report = pitch_evaluate([item], [baseline(item, np.empty(0, bool))], [np.empty((0, 2))], .9)
    assert report["passes"] and report["wrong_pitch_gain"] == 0
