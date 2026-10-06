"""The early-only model cannot shorten intervals or label tiny timing gains."""

import numpy as np
import pytest
from bass_joint_timing import EARLY, KEEP, LATE
from pitch_consensus_early import targets, transform
from pitch_interval_coverage import compare


def item(events,reference):
    return {'events':np.asarray(events,float),'reference':np.asarray(reference,float),
            'pitch_reference':np.asarray(reference,float),'duration':12.,
            'base_x':np.zeros((len(events),52),np.float32),'eligible':np.ones(len(events),bool)}


@pytest.mark.parametrize('strength',[.5,1.])
def test_late_is_unavailable_and_keeps_every_original_note(strength):
    events=np.array([[3.,4.,48.,70.],[5.,6.,48.,80.]])
    new,indices=transform(events,[LATE,LATE],12.,strength)
    np.testing.assert_array_equal(new,events)
    np.testing.assert_array_equal(indices,[0,1])


@pytest.mark.parametrize('strength',[.5,1.])
def test_early_interval_superset_preserves_any_reference_coverage(strength):
    events=np.array([[3.1,3.8,48.,70.],[4.1,4.8,48.,80.]])
    new,_=transform(events,[EARLY,EARLY],12.,strength)
    assert np.all(new[:,0]<=events[:,0])
    np.testing.assert_array_equal(new[:,1:],events[:,1:])
    rng=np.random.default_rng(20261005)
    reference=np.array([[s,s+length,48.] for s,length in zip(rng.uniform(2.9,4.9,100),rng.uniform(.01,1.,100),strict=True)])
    assert compare(reference,events,new)['passes']


@pytest.mark.parametrize('delta,expected',[(.1,EARLY),(.016,KEEP),(-.1,KEEP)])
def test_labels_require_substantial_correct_direction_gain(delta,expected):
    events=[[3.1,3.8,48.,70.],[4.1,4.8,48.,80.]]
    example=item(events,[[3.1-delta,3.8,48.],[4.1-delta,4.8,48.]])
    np.testing.assert_array_equal(targets(example)['action_y'],[expected,expected])
