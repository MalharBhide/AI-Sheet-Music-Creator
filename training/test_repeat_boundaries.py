"""A spurious attack may be merged only without changing audible pitch spans."""

import numpy as np
import pytest
from repeat_boundary_features import features, merge, pitch_union
from train_repeat_boundaries import acoustic_view, annotate, matrix, score


def fixture():
    events = np.array([[0., 1., 48., 80.], [1., 2., 48., 70.], [2., 4., 48., 60.]])
    return {'id': 'original-hold', 'corpus': 'original-fixture', 'events': events,
            'keep': np.ones(3, dtype=bool), 'baseline_keep': np.ones(3, dtype=bool),
            'shared': np.ones(3, dtype=bool), 'p': np.full(3, .9), 'seconds': 4.,
            'reference': np.array([[0., 4., 48.]]),
            'x': np.arange(3 * 74, dtype=np.float32).reshape(3, 74) / 1000}


def test_chain_merges_preserve_exact_pitch_spans_initial_velocity_and_source_events():
    item = fixture()
    pairs = np.array([[0, 1], [1, 2]])
    original = item['events'].copy()
    after, merged = merge(item, pairs, np.zeros(2), np.zeros(2), .01, .1)
    assert len(merged) == 2
    np.testing.assert_array_equal(after, [[0., 4., 48., 80.]])
    assert pitch_union(after) == pitch_union(original)
    np.testing.assert_array_equal(item['events'], original)
    assert item['keep'].all()


def test_genuine_repeated_attack_is_rejected_by_reference_gate_even_with_same_pitch_union():
    item = fixture()
    item['reference'] = np.array([[0., 1., 48.], [1., 2., 48.], [2., 4., 48.]])
    annotate([item])
    result = score([item], [(np.zeros(2), np.zeros(2))], .01, .1)
    assert not result['passes']
    assert not result['per_recording'][0]['matched_references_preserved']
    assert result['per_recording'][0]['pitch_time_union_preserved']


def test_merging_split_hold_improves_offsets_without_erasing_any_pitch_coverage():
    item = annotate([fixture()])[0]
    result = score([item], [(np.zeros(2), np.zeros(2))], .01, .1)
    assert result['passes'] and result['false_notes_removed'] == 2
    row = result['per_recording'][0]
    assert row['held_references_preserved'] and row['lost_reference_pitch_seconds'] == 0
    assert row['pitch_time_union_preserved']


@pytest.mark.parametrize('mutation', ['gap', 'unshared', 'rejected', 'different-pitch'])
def test_merge_protects_real_rests_unshared_prior_rejections_and_other_pitches(mutation):
    item = fixture()
    if mutation == 'gap':
        item['events'][1, 0] += .001
    elif mutation == 'unshared':
        item['shared'][1] = False
    elif mutation == 'rejected':
        item['keep'][1] = False
    else:
        item['events'][1, 2] = 49
    before = item['events'][item['keep']].copy()
    after, merged = merge(item, np.array([[0, 1]]), np.zeros(1), np.zeros(1), .01, .1)
    assert not merged
    np.testing.assert_array_equal(after, before)


def test_each_head_veto_and_threshold_equality_preserve_attacks():
    item, pairs = fixture(), np.array([[0, 1]])
    for repeat, guardian in [(.01, 0.), (0., .1), (1., 0.), (0., 1.)]:
        after, merged = merge(item, pairs, np.array([repeat]), np.array([guardian]), .01, .1)
        assert not merged
        np.testing.assert_array_equal(after, item['events'])
    with pytest.raises(ValueError, match='confidence'):
        merge(item, pairs, np.array([np.nan]), np.zeros(1), .01, .1)


def test_paired_features_have_no_labels_and_both_acoustic_views_match():
    item, pairs = fixture(), np.array([[0, 1], [1, 2]])
    x = features(item, pairs)
    assert x.shape == (2, 152)
    np.testing.assert_array_equal(acoustic_view(x), features(item, pairs, acoustic=True))
    assert features(item, np.empty((0, 2), dtype=int)).shape == (0, 152)
    item['reference'][:] = 0
    np.testing.assert_array_equal(features(item, pairs), x)


def test_training_excludes_ambiguous_boundaries():
    item = annotate([fixture()])[0]
    item['boundary_mask'][1] = False
    x, y, weight, counts = matrix([item], 30.)
    assert x.shape == (1, 152) and y.tolist() == [0.] and weight.tolist() == [1.]
    assert counts[0]['split_holds'] == 1
    item['boundary_mask'][:] = False
    with pytest.raises(ValueError, match='No supervised'):
        matrix([item], 30.)
