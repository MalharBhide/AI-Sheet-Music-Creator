"""Pitch filtering must retain supported continuations before articulation repair."""

import numpy as np
from pitch_support_targets import keep_targets


def test_correct_pitch_fragments_are_positive_keep_targets_without_relabeling_evaluation():
    reference = np.array([[0., 4., 48.]])
    events = np.array([[0., 1., 48., 80.], [2., 4., 48., 80.], [2., 4., 60., 80.]])
    y, mask, promoted = keep_targets(reference, events)
    assert y.tolist() == [1., 1., 0.]
    assert mask.tolist() == [True, True, True]
    assert promoted.tolist() == [False, True, False]


def test_low_reference_coverage_does_not_promote_ambiguous_tails():
    reference = np.array([[0., 1., 48.]])
    events = np.array([[.12, 2., 48., 80.]])
    y, mask, promoted = keep_targets(reference, events)
    assert y.tolist() == [0.] and mask.tolist() == [False] and not promoted.any()


def test_overlapping_annotations_do_not_double_count_pitch_support():
    reference = np.array([[0., .6, 48.], [.2, .7, 48.]])
    events = np.array([[.12, 1.12, 48., 80.]])
    _, _, promoted = keep_targets(reference, events)
    assert not promoted.any()


def test_no_candidates_or_annotations_are_safe():
    y, mask, promoted = keep_targets(np.empty((0, 3)), np.array([[0., 1., 48., 80.]]))
    assert y.tolist() == [0.] and mask.tolist() == [True] and not promoted.any()
    assert len(keep_targets(np.array([[0., 1., 48.]]), np.empty((0, 4)))[0]) == 0
