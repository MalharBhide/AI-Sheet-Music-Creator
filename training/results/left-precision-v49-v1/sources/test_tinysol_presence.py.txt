import numpy as np
import pytest
from prepare_tinysol_presence_v45 import RATE, canonical_padding, weak_labels


def test_fundamental_frames_have_no_fabricated_hold_labels():
    events = np.array([[3.1, 4.8, 40, 80], [3.1, 4.8, 52, 80], [0.1, 1.1, 40, 80]])
    keep, targets, weights = weak_labels(events, 40, 3., 5.)
    np.testing.assert_array_equal(keep, [1, 0, 0])
    assert not targets.any()
    assert not weights[0, 8:32].any()  # Unknown true-note clock, not constant sustain.
    assert weights[1].all()  # Wrong octave is absent in an isolated fundamental.
    assert weights[2].all()  # Authored silent padding is known absence.


def test_partial_fundamental_overlap_remains_conservatively_protected():
    events = np.array([[2.9, 3.01, 40, 80], [4.99, 5.1, 40, 80]])
    keep, _, weights = weak_labels(events, 40, 3., 5.)
    assert keep.all()
    assert (weights == 0).any() and (weights == 1).any()


def test_empty_predictions_are_retained_as_valid_examples():
    keep, targets, weights = weak_labels(np.empty((0, 4)), 40, 3., 5.)
    assert keep.shape == (0,) and targets.shape == weights.shape == (0, 40)


def test_invalid_clock_cannot_supply_pitch_supervision():
    with pytest.raises(ValueError, match="supervision"):
        weak_labels(np.array([[4., 3., 40., 80.]]), 40, 3., 5.)


@pytest.mark.parametrize("delta", [-30, 30])
def test_codec_rounding_changes_only_proven_silent_end_padding(delta):
    expected = 8*RATE
    raw = np.zeros(expected-delta, np.float32)
    raw[3*RATE:4*RATE] = .1
    clock = {"codec_lag_samples": 0, "input_frames": expected, "decoded_frames": len(raw)}
    canonical, result = canonical_padding(raw, expected, clock)
    assert len(canonical) == expected and result["silent_padding_adjustment_samples"] == delta
    np.testing.assert_array_equal(canonical[3*RATE:4*RATE], raw[3*RATE:4*RATE])
    raw[-1] = .01
    with pytest.raises(ValueError, match="silent padding"):
        canonical_padding(raw, expected, clock)


def test_codec_alignment_or_large_duration_changes_are_not_repaired():
    expected = 8*RATE
    raw = np.zeros(expected+1000, np.float32)
    clock = {"codec_lag_samples": 0, "input_frames": expected, "decoded_frames": len(raw)}
    with pytest.raises(ValueError, match="alignment"):
        canonical_padding(raw, expected, clock)
    raw = raw[:expected]
    clock.update(codec_lag_samples=1, decoded_frames=expected)
    with pytest.raises(ValueError, match="alignment"):
        canonical_padding(raw, expected, clock)
