"""Relationship evidence must not depend on song identity, order or register."""

import numpy as np
import pytest
from note_relations import ALL_NAMES, RELATION_NAMES, complete_context, relation_features


def sample():
    return np.array([[0., 2., 48., 90.], [.01, 1.8, 60., 70.],
                     [.7, 1.1, 62., 80.], [2.1, 2.5, 60., 70.]])


def test_relationships_are_transposition_time_and_permutation_invariant():
    events = sample()
    x = np.arange(4 * 52, dtype=np.float32).reshape(4, 52) / 100
    p = np.array([.9, .2, .8, .7])
    original = events.copy()
    result = relation_features(events, x, p)
    shifted = events.copy()
    shifted[:, :2] += 10
    shifted[:, 2] += 7
    np.testing.assert_allclose(relation_features(shifted, x, p), result, atol=1e-6)
    order = np.array([3, 1, 0, 2])
    np.testing.assert_allclose(relation_features(events[order], x[order], p[order]), result[order])
    np.testing.assert_array_equal(result[:, :52], x)
    np.testing.assert_array_equal(events, original)


def test_octave_support_is_separate_from_genuine_later_repeated_attack():
    events = sample()
    result = relation_features(events, np.zeros((4, 52)), np.array([.9, .2, .8, .7]))
    def column(name):
        return 52 + RELATION_NAMES.index(name)

    assert result[1, column('octave_coattack')] == pytest.approx(.9)
    assert result[3, column('octave_coattack')] == 0
    assert result[3, column('octave_overlap')] == 0
    assert result[3, column('same_pitch_previous')] == 0  # Outside the bounded neighborhood.
    assert result[0, column('same_pitch_previous')] == 0


def test_empty_and_isolated_notes_have_finite_evidence():
    assert relation_features(np.empty((0, 4)), np.empty((0, 52)), np.empty(0)).shape == (0, len(ALL_NAMES))
    result = relation_features(sample()[:1], np.zeros((1, 52)), np.array([.5]))
    assert np.isfinite(result).all()
    assert result[0, 52 + RELATION_NAMES.index('coattack_count')] == 0
    assert result[0, 52 + RELATION_NAMES.index('overlap_count')] == 0


def test_cropping_preserves_every_eligible_targets_features():
    events = np.array([[0., 8., 48., 90.], [1., 7., 60., 70.], [3.5, 4., 62., 80.],
                       [4., 7., 60., 70.], [5., 7., 65., 60.], [6.5, 8., 67., 70.]])
    x, p = np.zeros((6, 52)), np.array([.9, .2, .8, .7, .8, .6])
    full = relation_features(events, x, p)
    crop = (events[:, 0] >= 1.5) & (events[:, 0] < 6.5)
    cropped = relation_features(events[crop], x[crop], p[crop])
    eligible = complete_context(events[crop], 1.5, 6.5)
    assert eligible.any()
    np.testing.assert_array_equal(cropped[eligible], full[crop][eligible])


@pytest.mark.parametrize('events,p', [(np.array([[1, 0, 60, 80]]), np.array([.5])),
                                    (np.array([[0, 1, 60, 80]]), np.array([np.nan]))])
def test_bad_relationship_inputs_are_rejected(events, p):
    with pytest.raises(ValueError, match='relationships'):
        relation_features(events, np.zeros((len(events), 52)), p)
