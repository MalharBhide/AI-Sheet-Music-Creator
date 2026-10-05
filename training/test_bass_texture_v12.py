"""Original augmentation changes timbre without inventing played key labels."""

import numpy as np
import pytest
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import SEEDS, TRAIN_COUNT, texture


def test_texture_is_reproducible_finite_and_keeps_every_played_key_label():
    clean, labels = original(SEEDS[0])
    audio, actual = texture(SEEDS[0])
    repeated, repeated_labels = texture(SEEDS[0])
    np.testing.assert_array_equal(actual, labels)
    np.testing.assert_array_equal(repeated_labels, labels)
    np.testing.assert_array_equal(repeated, audio)
    assert audio.shape == (30 * RATE,) and np.isfinite(audio).all()
    assert np.max(np.abs(audio)) < 1. and not np.array_equal(audio, clean)
    assert np.all(actual[:, 1] > actual[:, 0])


def test_texture_partition_does_not_include_consumed_first_pass_seeds():
    assert TRAIN_COUNT == 72 and len(SEEDS) == 96
    consumed = set(range(262301, 262333)) | set(range(261901, 261933))
    assert not set(SEEDS) & consumed


def test_partial_texture_preparation_cannot_enter_fitting(tmp_path):
    from bass_texture_v12_data import catalog

    # A producer that stops before its final manifest must not become a smaller
    # accidental validation set. The trainer requires its complete fixed plan.
    (tmp_path / 'plan.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        catalog(tmp_path)
