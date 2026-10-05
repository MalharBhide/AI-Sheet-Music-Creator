"""Protect repeated attacks, real rests, and held notes during learned retiming."""

import numpy as np
import pytest
from bass_joint_timing import (
    EARLY,
    KEEP,
    LATE,
    REMOVE,
    actions,
    probabilities,
    protected,
    transform,
)
from score_bass_joint_timing import row, timing
from train_bass_joint_timing_v17 import model, supervised


def notes(rows):
    return np.array([(*row, 70.) for row in rows], float).reshape(-1, 4)


def test_touching_release_tracks_repeated_attack():
    old = notes([(3., 4., 48), (4., 5., 48)])
    new, indices = transform(old, [KEEP, LATE], 10.)
    np.testing.assert_allclose(new[:, :2], [[3., 4.02], [4.02, 5.]])
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(new[:, 2:], old[:, 2:])


def test_early_attack_cannot_fill_a_real_same_key_rest():
    old = notes([(3., 3.99, 48), (4., 5., 48)])
    new, _ = transform(old, [KEEP, EARLY], 10.)
    np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize('strength', [.5, 1.])
def test_combined_same_key_corrections_preserve_order_and_minimum_spacing(strength):
    old = notes([(3., 3.06, 48), (3.06, 3.12, 48), (3.12, 3.2, 48)])
    new, _ = transform(old, [LATE, EARLY, EARLY], 10., strength)
    assert np.all(np.diff(new[:, 0]) >= .045-1e-9)
    assert np.all(new[:-1, 1] <= new[1:, 0]+1e-9)
    assert np.max(np.abs(new[:, :2]-old[:, :2])) <= .020001


def test_deletion_does_not_retime_remaining_notes_or_touch_edges():
    old = notes([(0., 1., 48), (3., 4., 48), (6., 7., 50), (9., 10., 48)])
    new, indices = transform(old, [REMOVE, REMOVE, KEEP, REMOVE], 10.)
    np.testing.assert_array_equal(indices, [0, 2, 3])
    np.testing.assert_array_equal(new, old[indices])


def test_guardian_protects_supported_note_from_delete_but_not_timing():
    item = {'events': notes([(3., 4., 48), (5., 6., 50)]), 'eligible': np.ones(2, bool)}
    p = np.array([[0., .01, 0., .99], [.99, .01, 0., 0.]])
    np.testing.assert_array_equal(actions(item, p, [.8, .8], .9, .9), [KEEP, EARLY])
    np.testing.assert_array_equal(actions(item, p, [.1, .8], 1.01, .9), [REMOVE, KEEP])
    with pytest.raises(ValueError):
        actions(item, p*2, [.1, .8], .9, .9)


def test_pitch_interval_loss_is_detected_even_if_attack_still_matches():
    old = notes([(3., 4., 48)])
    item = {'events': old, 'reference': old[:, :3], 'pitch_reference': old[:, :3]}
    new, _ = transform(old, [LATE], 10.)
    passes, counts, coverage = protected(item, new)
    assert counts['attack']['lost'] == 0
    assert not passes and not coverage['passes']


def test_probability_rejects_invalid_envelopes_and_batch_size():
    args = (None, np.zeros((1, 40, 9)), np.zeros((1, 84)), (np.zeros(84), np.ones(84)))
    with pytest.raises(ValueError):
        probabilities(*args, batch_size=0)
    with pytest.raises(ValueError):
        probabilities(None, np.full((1, 40, 9), 2.), args[2], args[3])


def sample_item(events, reference=None):
    return {'id':'example','corpus':'piano','events':events,'duration':10.,
            'reference':events[:, :3].copy() if reference is None else reference,
            'pitch_reference':events[:, :3].copy() if reference is None else reference}


def test_deleting_a_badly_timed_prediction_earns_no_timing_gain():
    old = notes([(3.1, 4., 48), (5., 6., 50)])
    item = sample_item(old,np.array([[3.,4.,48.],[5.,6.,50.]]))
    new, indices = transform(old,[REMOVE,KEEP],10.)
    result = timing(item,new,indices)
    np.testing.assert_array_equal(result['before_onset_release_error_sum'],result['after_onset_release_error_sum'])
    assert result['pairs']==2


def test_attack_improvement_can_still_worsen_repeated_note_spacing():
    old = notes([(3.1,3.8,48),(4.1,4.8,48)])
    ref = np.array([[3.,3.8,48.],[4.,4.8,48.]])
    result = row(sample_item(old,ref),np.array([EARLY,KEEP]),1.,{})
    assert result['timing']['after_onset_release_error_sum'][0] < result['timing']['before_onset_release_error_sum'][0]
    assert result['matches']['attack']['lost']==0
    assert not result['timing']['passes']


def test_release_timing_regression_cannot_hide_behind_better_onset():
    old = notes([(3.,4.,48),(4.,5.,48)])
    ref = np.array([[3.,4.,48.],[3.96,5.,48.]])
    result = row(sample_item(old,ref),np.array([KEEP,EARLY]),1.,{})
    assert result['timing']['after_onset_release_error_sum'][0] < result['timing']['before_onset_release_error_sum'][0]
    assert result['timing']['after_onset_release_error_sum'][1] > result['timing']['before_onset_release_error_sum'][1]
    assert not result['passes']


def test_corpus_recording_weighting_and_four_class_output():
    import torch

    def item(corpus,n):
        return {'corpus':corpus,'eligible':np.ones(n,bool),'mask':np.ones(n,bool),
                'x':np.zeros((n,84),np.float32),'frames':np.zeros((n,40,9),np.float32),
                'action_y':np.full(n,KEEP,np.int64)}
    _,_,_,w,counts = supervised([item('a',2),item('a',4),item('b',3)],[1.,1.,1.,1.])
    np.testing.assert_allclose(w[:2].sum(),w[2:6].sum())
    np.testing.assert_allclose(w[:6].sum(),w[6:].sum())
    assert counts['a']['class_counts']==[0,6,0,0]
    threads = torch.get_num_threads()
    network = model()
    p = probabilities(network,np.zeros((2,40,9)),np.zeros((2,84)),(np.zeros(84),np.ones(84)))
    assert p.shape==(2,4) and np.all(p.argmax(axis=1)==KEEP)
    np.testing.assert_allclose(p.sum(axis=1),1.,atol=2e-7)
    assert torch.get_num_threads()==threads
