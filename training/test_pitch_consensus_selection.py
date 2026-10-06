"""Numerically equivalent timing gains must not override meaningful tie-breaks."""

import numpy as np
import pytest
from train_pitch_consensus_v21 import selection_key


def test_roundoff_does_not_hide_better_validation_loss():
    first=selection_key(.1799999999999966,1,.609,20)
    better=selection_key(.17999999999999616,1,.591,40)
    assert better>first


def test_real_timing_gain_still_precedes_secondary_metrics():
    assert selection_key(.19,0,2.,80)>selection_key(.18,100,.1,20)


def test_false_note_improvement_and_earlier_epoch_break_real_ties():
    assert selection_key(.18,2,.7,40)>selection_key(.18,1,.5,20)
    assert selection_key(.18,2,.7,20)>selection_key(.18,2,.7,40)


def test_nonfinite_metric_cannot_select_a_checkpoint():
    with pytest.raises(ValueError):
        selection_key(np.nan,1,.7,20)
