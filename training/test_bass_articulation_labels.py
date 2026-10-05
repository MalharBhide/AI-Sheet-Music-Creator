"""Audible pitch continuation must not erase a correct key-release boundary."""

import bass_articulation_labels as labels
import bass_boundary_evidence as previous
import numpy as np
from test_bass_boundaries import fixture


def test_key_release_under_continuous_pedal_support_overrides_negative_split_label():
    item = fixture()
    item['reference'] = np.array([[3., 4., 40.]])
    previous.annotate([item])
    assert item['boundary_y'][0] == 0 and item['boundary_mask'][0]
    before_features = previous.features(item, item['pairs']).copy()
    labels.annotate([item])
    assert item['boundary_y'][0] == 1 and item['boundary_mask'][0]
    assert item['protected_key_offsets'][0]
    np.testing.assert_array_equal(previous.features(item, item['pairs']), before_features)


def test_true_continuous_hold_remains_negative_without_losing_a_matched_key_end():
    item = labels.annotate([fixture()])[0]
    assert item['boundary_y'].tolist() == [0., 0.] and item['boundary_mask'].all()
    assert not item['protected_key_offsets'].any()


def test_genuine_repeated_attacks_and_offsets_remain_positive():
    item = fixture()
    item['reference'] = item['events'][:, :3].copy()
    labels.annotate([item])
    assert item['boundary_y'].tolist() == [1., 1.]
    assert item['protected_key_offsets'].all() and item['protected_key_attacks'].all()
