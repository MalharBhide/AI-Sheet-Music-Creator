"""Only fully represented holds establish clear phantom-repeat negatives."""

import numpy as np
from represented_hold_labels import clarify


def test_repeated_attack_inside_a_represented_hold_is_a_supervised_negative():
    reference = np.array([[0., 4., 48.]])
    events = np.array([[0., 4., 48., 80.], [1., 1.5, 48., 70.], [2., 3.5, 48., 90.]])
    before = [a.copy() for a in (reference, events)]
    y, mask, clarified = clarify(reference, events)
    np.testing.assert_array_equal(y, [1., 0., 0.])
    np.testing.assert_array_equal(mask, [True, True, True])
    np.testing.assert_array_equal(clarified, [False, True, True])
    for actual, expected in zip((reference, events), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_isolated_late_detection_does_not_establish_a_phantom_repeat():
    y, mask, clarified = clarify(np.array([[0., 4., 48.]]), np.array([[1., 2., 48., 80.]]))
    assert y.tolist() == [0.] and mask.tolist() == [False] and not clarified.any()


def test_later_fragment_remains_ambiguous_when_initial_detection_missed_the_hold():
    reference = np.array([[0., 4., 48.]])
    for release in (1., 2.8):
        events = np.array([[0., release, 48., 80.], [2., 3., 48., 70.]])
        y, mask, clarified = clarify(reference, events)
        assert y.tolist() == [1., 0.]
        assert mask.tolist() == [True, False]
        assert not clarified.any()


def test_actual_repeated_attacks_and_their_nearby_ambiguous_candidates_stay_protected():
    reference = np.array([[0., 4., 48.], [2., 3., 48.]])
    events = np.array([[0., 4., 48., 80.], [2., 3., 48., 80.], [2.1, 3., 48., 70.]])
    y, mask, clarified = clarify(reference, events)
    assert y.tolist() == [1., 1., 0.] and mask.tolist() == [True, True, False]
    assert not clarified.any()


def test_near_attack_extending_release_and_other_pitch_are_not_reclassified():
    reference = np.array([[0., 4., 48.]])
    events = np.array([[0., 4., 48., 80.], [.12, 1., 48., 70.],
                       [2.5, 5., 48., 80.], [1., 2., 50., 80.]])
    y, mask, clarified = clarify(reference, events)
    assert y.tolist() == [1., 0., 0., 0.]
    assert mask.tolist() == [True, False, False, True]
    assert not clarified.any()


def test_empty_candidate_population_is_safe():
    y, mask, clarified = clarify(np.array([[0., 4., 48.]]), np.empty((0, 4)))
    assert len(y) == len(mask) == len(clarified) == 0
