"""Guard dataset partitions and compare replacement heads with actual website V5."""

import numpy as np
import pytest
import train_left_hand_expanded as expanded
from prepare_left_hand_expansion import crop_labels, eligible_entries
from train_left_hand_expanded import score


def test_expansion_retains_performer_splits_and_never_reads_test_items():
    split = {'tracks': {'train': [{'id': 'train', 'corpus': 'guitarset', 'player': '02'}],
                        'validation': [{'id': 'val', 'corpus': 'guitarset', 'player': '04'}],
                        'test': [None]}}
    assert [(group, item['id']) for group, item in eligible_entries(split)] == [('train', 'train'), ('validation', 'val')]
    split['tracks']['train'][0]['player'] = '05'
    with pytest.raises(ValueError, match='crossed'):
        list(eligible_entries(split))


def test_later_labels_keep_clock_holds_and_exclude_incomplete_edge_attacks():
    notes = [{'time': t, 'duration': d, 'value': p} for t, d, p in
             [(29.9, 2., 48), (30.25, 3., 50), (34., 20., 52), (34.75, 1., 54)]]
    reference = crop_labels({'annotations': [{'namespace': 'note_midi', 'data': notes}]}, 5.)
    assert reference == [[.25, 3.25, 50.], [4., 5., 52.]]


def fixture():
    return {'id': 'known', 'corpus': 'controlled', 'seconds': 4.,
            'reference': np.array([[0., 3., 48.]]),
            'events': np.array([[0., 3., 48., 80.], [1., 2., 53., 80.], [2., 3., 55., 80.]]),
            'keep': np.ones(3, dtype=bool), 'baseline_keep': np.array([True, False, True]),
            'shared': np.ones(3, dtype=bool), 'p': np.full(3, .2)}


def test_previous_left_hand_gain_cannot_be_claimed_again():
    result = score([fixture()], [np.array([1., 0., 1.])], .05)
    assert result['passes'] and result['false_notes_removed'] == 0
    assert result['per_recording'][0]['deployed_v5']['false_positives'] == 1
    improved = score([fixture()], [np.array([1., 0., 0.])], .05)
    assert improved['passes'] and improved['false_notes_removed'] == 1


def test_restoring_previously_removed_false_notes_fails_replacement_gate():
    result = score([fixture()], [np.ones(3)], .05)
    assert result['false_notes_removed'] == -1
    assert not result['passes']


def test_aggregate_false_note_gain_cannot_hide_a_lost_correct_hold():
    result = score([fixture()], [np.zeros(3)], .05)
    assert result['false_notes_removed'] == 1
    assert not result['passes']
    assert not result['per_recording'][0]['held_references_preserved']


def test_refinement_cannot_restore_previously_removed_false_notes():
    item = fixture()
    item['keep'] = item['baseline_keep'].copy()
    result = score([item], [np.ones(3)], .05)
    assert result['passes'] and result['false_notes_removed'] == 0


def test_refinement_relabels_surviving_duplicate_after_baseline_filter(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    item = {'reference': np.array([[0., 2., 48.]]),
            'events': np.array([[0., 2., 48., 80.], [0., 2., 48., 70.]]),
            'x': np.zeros((2, 52)), 'keep': np.ones(2, dtype=bool),
            'shared': np.ones(2, dtype=bool), 'p': np.full(2, .2),
            'y': np.array([1., 0.]), 'mask': np.ones(2, dtype=bool)}
    monkeypatch.setattr(expanded, 'hashes', dict)
    monkeypatch.setattr(expanded, 'AccompanimentVerifier', lambda: None)
    monkeypatch.setattr(expanded, 'prepare_v4', lambda *args: [item])
    monkeypatch.setattr(expanded.np, 'load', lambda *args, **kwargs: nullcontext({}))
    monkeypatch.setattr(expanded, 'ContextNoteModel', lambda _: SimpleNamespace(
        threshold=.05, probability=lambda x: np.array([0., 1.])))
    result = expanded.prepare(None, [], 'refinement')[0]
    np.testing.assert_array_equal(result['keep'], [False, True])
    np.testing.assert_array_equal(result['baseline_keep'], [False, True])
    np.testing.assert_array_equal(result['y'], [0., 1.])
    np.testing.assert_array_equal(result['mask'], [False, True])
