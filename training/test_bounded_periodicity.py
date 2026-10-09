import numpy as np
import pytest
from bounded_periodicity import features
from periodicity_features import features as reference_features
from periodicity_features import observe


@pytest.mark.parametrize("kind", ["silence", "tone", "noise"])
def test_bounded_features_match_original_at_edges_and_many_batch_sizes(kind):
    rng = np.random.default_rng(119)
    t = np.arange(5 * 22050) / 22050
    wave = (
        np.zeros(len(t))
        if kind == "silence"
        else np.sin(2 * np.pi * 55 * t)
        if kind == "tone"
        else rng.normal(0, 0.2, len(t))
    )
    wave = wave.astype(np.float32)
    starts = np.array([0, 0.08, 0.2, 0.75, 1.137, 3.123, 4.9])
    events = np.column_stack(
        (
            starts,
            np.minimum(starts + 0.09, 5),
            np.array([21, 33, 45, 59, 72, 95, 108]),
            np.full(len(starts), 80),
        )
    )
    expected = reference_features(observe(wave, 22050), events)
    for batch in (1, 3, 64):
        actual = features(wave, 22050, events, batch)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_empty_invalid_and_one_sample_waveforms():
    assert features(np.zeros(1), 22050, np.empty((0, 4))).shape == (0, 63)
    with pytest.raises(ValueError, match="clock"):
        features(np.zeros(5), 22050, np.array([[0, 1, 48, 80]]))
    with pytest.raises(ValueError, match="clock"):
        features(np.zeros(5), 22050, np.empty((0, 4)), 0)
