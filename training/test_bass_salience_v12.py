"""Acoustic context cannot learn from fitted confidence or label metadata."""

import numpy as np
import pytest
from bass_salience_v12_data import ACOUSTIC_NAMES, NAMES, acoustic_view, features


def evidence():
    events = np.array([[3., 5., 40., 90.], [3., 5., 52., 60.]])
    x = np.zeros((2, 52), np.float32)
    x[:, [2, 7, 16]] = [[.8, .7, .9], [.3, .2, .2]]
    x[:, 50:52] = [[.7, .1], [.3, .2]]
    return events, x


def test_context_has_observed_salience_and_guardian_keeps_release_evidence():
    events, x = evidence()
    actual = features(events, x)
    assert actual.shape == (2, len(NAMES)) == (2, 72)
    assert acoustic_view(actual).shape == (2, len(ACOUSTIC_NAMES)) == (2, 52)
    np.testing.assert_array_equal(acoustic_view(actual), x)
    np.testing.assert_allclose(actual[:, 52], [.8, .7 / 3], atol=1e-7)


def test_acoustic_context_is_pitch_relative_and_empty_windows_are_supported():
    events, x = evidence()
    expected = features(events, x)
    events[:, 2] += 7
    np.testing.assert_array_equal(features(events, x), expected)
    assert features(np.empty((0, 4)), np.empty((0, 52))).shape == (0, 72)


@pytest.mark.parametrize('corrupt', ['shape', 'nonfinite', 'interval'])
def test_invalid_acoustic_inputs_are_rejected(corrupt):
    events, x = evidence()
    if corrupt == 'shape':
        x = x[:, :26]
    elif corrupt == 'nonfinite':
        x[0, 0] = np.nan
    else:
        events[0, 1] = events[0, 0]
    with pytest.raises(ValueError):
        features(events, x)
