"""Timing supervision must agree with release and repeated-key spacing gates."""

import numpy as np
import pytest
from bass_joint_timing import EARLY, KEEP
from bass_joint_timing import targets as previous_targets
from bass_joint_timing_v18 import margin, targets


def item(events, reference):
    return {'events':np.asarray(events,float),'reference':np.asarray(reference,float),
            'pitch_reference':np.asarray(reference,float),'duration':10.,
            'base_x':np.zeros((len(events),52),np.float32),'eligible':np.ones(len(events),bool)}


def test_single_attack_label_cannot_worsen_a_touching_release():
    example=item([[3.,4.,48.,70.],[4.,5.,48.,70.]],[[3.,4.,48.],[3.96,5.,48.]])
    assert previous_targets(example)['action_y'][1]==EARLY
    assert targets(example)['action_y'][1]==KEEP


def test_isolated_correctable_early_attack_still_has_a_timing_label():
    example=item([[3.1,4.,48.,70.]],[[3.,4.,48.]])
    assert targets(example)['action_y'][0]==EARLY


def test_repeated_spacing_regression_is_a_keep_counterexample():
    example=item([[3.1,3.8,48.,70.],[4.1,4.8,48.,70.]],[[3.,3.8,48.],[4.,4.8,48.]])
    np.testing.assert_array_equal(previous_targets(example)['action_y'],[EARLY,EARLY])
    np.testing.assert_array_equal(targets(example)['action_y'],[KEEP,KEEP])


@pytest.mark.parametrize('threshold,expected',[(.8,.9),(.9,.95),(.95,.975),(.99,.995),(1.,1.),(1.01,1.01)])
def test_margin_remains_stricter_without_disabling_high_confidence_actions(threshold,expected):
    assert margin(threshold)==pytest.approx(expected)


@pytest.mark.parametrize('threshold',[-.1,1.1,np.nan,np.inf])
def test_margin_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        margin(threshold)
