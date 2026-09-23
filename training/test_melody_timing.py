"""Timing-only models must preserve the detected melody's pitch and note count."""

import numpy as np
from train_melody_timing import boundary_features, evaluate, retime


def test_timing_changes_are_bounded_and_cannot_create_overlaps_or_new_pitches():
    events = np.array([[0., .2, 60.], [.2, .5, 62.], [.8, 1.2, 64.]])
    before = events.copy()
    new = retime(events, np.array([[-1., 1.], [-1., -1.], [1., 1.]]), 1., 1.2)
    np.testing.assert_array_equal(new[:, 2], events[:, 2])
    assert np.all(new[:, 1] > new[:, 0])
    assert np.all(new[1:, 0] >= new[:-1, 1])
    assert np.all(np.abs(new[:, :2] - events[:, :2]) <= .080001)
    assert new[0, 0] >= 0 and new[-1, 1] <= 1.2
    np.testing.assert_array_equal(events, before)


def test_boundary_evidence_is_pitch_relative_and_handles_window_edges():
    x = np.zeros((50, 88, 10), dtype=np.float32)
    x[:, 39] = 1.
    events = np.array([[0., 1., 60.]])
    start, end = [boundary_features(x, events, side) for side in (0, 1)]
    assert start.shape == end.shape == (1, 131)
    np.testing.assert_array_equal(start[:, :130], 1.)
    np.testing.assert_array_equal(end[:, :130], 1.)


def test_timing_gate_rejects_moving_a_correct_attack_outside_tolerance():
    from types import SimpleNamespace

    item = {'id': 'correct-attack', 'corpus': 'controlled', 'events': np.array([[.2, .8, 60.]]),
            'reference': np.array([[.2, .8, 60.]]), 'duration': 1.,
            'onset_x': np.zeros((1, 131)), 'release_x': np.zeros((1, 131))}
    models = [SimpleNamespace(predict=lambda _: np.array([.08])),
              SimpleNamespace(predict=lambda _: np.array([0.]))]
    result = evaluate([item], models, 1.)
    assert not result['passes'] and result['failed_recordings'] == ['correct-attack']
