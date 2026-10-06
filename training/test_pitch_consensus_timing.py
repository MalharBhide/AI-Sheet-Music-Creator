"""Joint corrections must preserve fixed releases and every repeated interval."""

import numpy as np
import pytest
from bass_joint_timing import EARLY, KEEP, LATE, REMOVE
from pitch_consensus_timing import targets, transform
from score_bass_joint_timing import timing
from score_pitch_consensus_timing import row


def notes(rows):
    return np.asarray([(*r,70.) for r in rows],float).reshape(-1,4)


def item(events,reference=None):
    reference = events[:,:3].copy() if reference is None else np.asarray(reference,float)
    return {'id':'example','corpus':'piano','events':events,'reference':reference,
            'pitch_reference':reference,'duration':12.,'base_x':np.zeros((len(events),52),np.float32),
            'eligible':np.ones(len(events),bool)}


@pytest.mark.parametrize('strength',[.5,1.])
@pytest.mark.parametrize('direction',[EARLY,LATE])
def test_consensus_moves_every_attack_but_no_release(strength,direction):
    old=notes([(3.1,3.8,48),(4.1,4.8,48),(6.1,7.,48)])
    new,indices=transform(old,[direction]*3,12.,strength)
    delta=(-.02 if direction==EARLY else .02)*strength
    np.testing.assert_allclose(new[:,0],old[:,0]+delta)
    np.testing.assert_array_equal(new[:,1:],old[:,1:])
    np.testing.assert_array_equal(indices,[0,1,2])
    reference=old[:,:3].copy()
    reference[:,0]+=-.1 if direction==EARLY else .1
    clock=timing(item(old,reference),new,indices)
    assert clock['after_repeated_spacing_error_sum']==pytest.approx(clock['before_repeated_spacing_error_sum'],abs=1e-9)
    assert clock['after_onset_release_error_sum'][1]==clock['before_onset_release_error_sum'][1]


@pytest.mark.parametrize('actions',[[EARLY,KEEP],[EARLY,LATE],[EARLY,REMOVE]])
def test_partial_or_disagreeing_consensus_cannot_retime(actions):
    old=notes([(3.1,3.8,48),(4.1,4.8,48)])
    new,indices=transform(old,actions,12.)
    np.testing.assert_array_equal(new,old[indices])


def test_touching_releases_never_follow_a_moved_attack():
    old=notes([(3.,4.,48),(4.,5.,48)])
    early,_=transform(old,[EARLY,EARLY],12.)
    np.testing.assert_array_equal(early,old)  # No new same-key overlap.
    late,_=transform(old,[LATE,LATE],12.)
    np.testing.assert_allclose(late[:,0],[3.02,4.02])
    np.testing.assert_array_equal(late[:,1:],old[:,1:])


def test_edge_note_in_pitch_group_prevents_partial_clock_shift():
    old=notes([(1.,2.,48),(3.1,4.,48),(5.1,6.,50)])
    new,_=transform(old,[EARLY,EARLY,EARLY],12.)
    np.testing.assert_array_equal(new[:2],old[:2])
    assert new[2,0]==pytest.approx(5.08)


def test_minimum_duration_and_spacing_protect_whole_group():
    old=notes([(3.,3.06,48),(3.1,4.,48)])
    new,_=transform(old,[LATE,LATE],12.)
    np.testing.assert_array_equal(new,old)


def test_uniform_training_target_can_correct_systematic_repeat_delay():
    old=notes([(3.1,3.8,48),(4.1,4.8,48)])
    example=item(old,[[3.,3.8,48],[4.,4.8,48]])
    labeled=targets(example)
    np.testing.assert_array_equal(labeled['action_y'],[EARLY,EARLY])
    result=row(example,labeled['action_y'],1.,{})
    assert result['passes'] and result['retimed_notes']==2


def test_conflicting_group_truth_is_not_a_timing_label():
    old=notes([(3.1,3.8,48),(4.1,4.8,48)])
    example=item(old,[[3.,3.8,48],[4.2,4.8,48]])
    np.testing.assert_array_equal(targets(example)['action_y'],[KEEP,KEEP])


def test_deletion_has_no_timing_credit():
    old=notes([(3.1,3.8,48),(4.1,4.8,48)])
    result=row(item(old,[[3.,3.8,48],[4.,4.8,48]]),np.array([REMOVE,KEEP]),1.,{})
    assert result['timing']['after_onset_release_error_sum']==result['timing']['before_onset_release_error_sum']


def test_clock_invariant_even_with_interleaved_keys_and_unsorted_input():
    old=notes([(6.1,6.8,48),(4.1,4.8,50),(3.1,3.8,48),(5.1,5.8,50)])
    new,_=transform(old,[EARLY,LATE,EARLY,LATE],12.)
    np.testing.assert_allclose(new[:,0],old[:,0]+[-.02,.02,-.02,.02])
    np.testing.assert_array_equal(new[:,1:],old[:,1:])


@pytest.mark.parametrize('actions,duration,strength', [([4],12.,1.),([KEEP],0.,1.),([KEEP],12.,.7)])
def test_invalid_inputs_are_rejected(actions,duration,strength):
    with pytest.raises(ValueError):
        transform(notes([(3.,4.,48)]),actions,duration,strength)


def test_empty_recording_and_short_recording_remain_valid():
    new,indices=transform(np.empty((0,4)),[],.02)
    assert new.shape==(0,4) and indices.shape==(0,)
