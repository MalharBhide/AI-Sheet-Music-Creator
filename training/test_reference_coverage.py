"""Preserve long-note continuations, not just their initial matched attack."""

import numpy as np
from pitch_interval_coverage import compare, reference_coverage


def test_removing_a_later_hold_fragment_fails_even_with_its_attack_unchanged():
    reference = np.array([[0., 4., 48.]])
    before = np.array([[0., 1., 48., 80.], [2., 4., 48., 70.]])
    result = compare(reference, before, before[:1])
    assert not result['passes']
    assert result['lost_reference_seconds'] == 2
    assert result['references_with_lost_coverage'] == 1


def test_removing_a_duplicate_inside_a_complete_hold_preserves_coverage():
    reference = np.array([[0., 4., 48.]])
    before = np.array([[0., 4., 48., 80.], [1., 2., 48., 70.]])
    assert compare(reference, before, before[:1])['passes']
    assert reference_coverage(reference, before).tolist() == [4.]


def test_wrong_pitches_and_notes_outside_reference_intervals_do_not_add_coverage():
    reference = np.array([[1., 3., 48.], [4., 5., 52.]])
    events = np.array([[0., 4., 50., 80.], [2., 4., 48., 80.], [5., 6., 52., 80.]])
    assert reference_coverage(reference, events).tolist() == [1., 0.]
    assert compare(reference, events, events[1:])['passes']


def test_overlapping_reference_holds_are_checked_individually_and_empty_is_safe():
    reference = np.array([[0., 3., 48.], [1., 4., 48.]])
    before = np.array([[0., 4., 48., 80.]])
    after = np.array([[0., 2., 48., 80.]])
    result = compare(reference, before, after)
    assert not result['passes'] and result['references_with_lost_coverage'] == 2
    assert compare(np.empty((0, 3)), before, np.empty((0, 4)))['passes']
