"""Broader positive bass signals keep deterministic sample/reference clocks."""

import numpy as np
from prepare_bass_positive_data import RATE, SEED, original


def test_original_bass_sources_are_deterministic_and_differ_from_regression_phrases():
    for seed in (SEED, SEED + 63, SEED + 64, SEED + 79):
        samples, notes = original(seed)
        same, labels = original(seed)
        np.testing.assert_array_equal(samples, same)
        np.testing.assert_array_equal(notes, labels)
        assert len(samples) == 30 * RATE and np.isfinite(samples).all()
        assert np.all((notes[:, 0] >= 2.75) & (notes[:, 1] <= 27.) & (notes[:, 1] > notes[:, 0]))
        assert np.all((notes[:, 2] >= 21) & (notes[:, 2] < 60))
        assert np.count_nonzero(samples[:round(2.75 * RATE)]) == 0
    assert not np.array_equal(original(SEED)[1], original(SEED + 1)[1])
