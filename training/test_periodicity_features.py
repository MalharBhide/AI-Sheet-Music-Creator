import numpy as np
import pytest
from periodicity_features import NAMES, features, observe


def test_fundamental_recurrence_distinguishes_octave_ghost_and_is_gain_invariant():
    times = np.arange(22050 * 3) / 22050
    audio = np.sin(2 * np.pi * 55 * times).astype(np.float32)
    notes = np.array([[0.5, 2.5, 33, 80], [0.5, 2.5, 45, 80]], float)
    first = features(observe(audio, 22050), notes).reshape(2, 9, 7)
    second = features(observe(audio * 0.03, 22050), notes).reshape(2, 9, 7)
    np.testing.assert_allclose(first, second, atol=1e-5)
    assert np.min(first[0, 1:7, 1]) > 0.98
    assert np.max(first[1, 1:7, 1]) < -0.98


def test_silence_short_audio_and_empty_notes_are_finite():
    seen = observe(np.zeros(40, np.float32), 22050)
    assert features(seen, np.empty((0, 4))).shape == (0, len(NAMES))
    result = features(seen, np.array([[0, 0.001, 21, 80]], float))
    assert not np.any(result)


@pytest.mark.parametrize("change", ["rate", "stereo", "nan", "empty", "clock", "pitch"])
def test_invalid_physical_clock_and_waveform_are_rejected(change):
    waveform = np.zeros(22050, np.float32)
    if change in ("clock", "pitch"):
        event = np.array(
            [[0, 2 if change == "clock" else 0.5, 21.5 if change == "pitch" else 21, 80]], float
        )
        with pytest.raises(ValueError):
            features(observe(waveform, 22050), event)
    else:
        if change == "stereo":
            waveform = np.column_stack([waveform, waveform])
        if change == "nan":
            waveform[0] = np.nan
        if change == "empty":
            waveform = np.empty(0)
        with pytest.raises(ValueError):
            observe(waveform, 16000 if change == "rate" else 22050)
