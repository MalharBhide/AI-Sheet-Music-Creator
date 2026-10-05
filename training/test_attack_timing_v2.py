"""Attack corrections must never change pitches or remove attacks/held support."""

import numpy as np
from train_attack_timing_v2 import correct, protected


def test_shared_boundary_moves_within_limit_without_overlaps_or_pitch_changes():
    events = np.array([[0., .2, 60.], [.2,.8,62.], [1.,2.,64.]])
    before = events.copy()
    changed = correct(events, np.array([0.,-.02,.02]))
    np.testing.assert_array_equal(changed[:,2], events[:,2])
    assert changed.shape == events.shape
    np.testing.assert_allclose(changed, [[0.,.18,60.],[.18,.8,62.],[1.02,2.,64.]])
    np.testing.assert_array_equal(events,before)


def test_silence_and_short_notes_reject_invalid_earlier_attacks():
    events = np.array([[0.,.03,60.],[.03,.2,62.]])
    np.testing.assert_array_equal(correct(events,np.array([-.02,-.02])),events)
    isolated = np.array([[0.,.1,60.],[.11,.2,62.]])
    np.testing.assert_array_equal(correct(isolated,np.array([0.,-.02])),isolated)


def test_gate_rejects_lost_coverage_even_when_note_matches_remain():
    events = np.array([[.2,.8,60.]])
    changed = correct(events,np.array([.02]))
    passes, counts, coverage = protected(events,events,changed)
    assert counts['onset']['lost'] == counts['hold']['lost'] == 0
    assert not passes and coverage['references_with_lost_coverage'] == 1


def test_advancing_a_late_attack_can_improve_coverage_without_losing_a_hold():
    reference = np.array([[.18,.8,60.]])
    events = np.array([[.2,.8,60.]])
    assert protected(reference,events,correct(events,np.array([-.02])))[0]
