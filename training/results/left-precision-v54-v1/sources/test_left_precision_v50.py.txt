import numpy as np
import pytest
from left_precision_v50 import FRESH_SEEDS, real_piano_weights


def items(group="train"):
    return [{"group": group, "corpus": "musicnet-real-piano"},
            {"group": "train", "corpus": "nrgcp-authored-sampled-piano"}]


def masks():
    return [np.array([True, False, True]), np.array([False, True])]


def test_fresh_bank_has_exactly_forty_predeclared_new_seeds():
    assert FRESH_SEEDS == tuple(range(295101, 295141))
    assert len(set(FRESH_SEEDS)) == 40


def test_only_training_real_piano_loss_changes_with_preserved_relative_recording_weights():
    result = real_piano_weights(items(), masks(), np.array([2., 1., 3.]), 4)
    np.testing.assert_allclose(result.mean(), 1)
    np.testing.assert_allclose(result[0]/result[1], 2)
    np.testing.assert_allclose(result[1]/result[2], 4/3)
    assert result.dtype == np.float32


def test_validation_cannot_set_training_loss_weights():
    with pytest.raises(ValueError, match="training"):
        real_piano_weights(items("validation"), masks(), np.ones(3), 4)


@pytest.mark.parametrize("weights,factor", [
    (np.ones(2), 4), (np.array([1., 0., 1.]), 4),
    (np.array([1., np.nan, 1.]), 4), (np.ones(3), 0), (np.ones(3), np.inf),
    (np.ones(3), 1e100),
])
def test_misalignment_or_nonfinite_loss_weights_fail_closed(weights, factor):
    with pytest.raises(ValueError):
        real_piano_weights(items(), masks(), weights, factor)
