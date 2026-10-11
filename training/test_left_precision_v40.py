import numpy as np
import pytest
from left_precision_v40 import CONTEXT_WIDTH, model, observations, probabilities
from periodicity_features import NAMES
from test_chord_boundary_v36 import item


@pytest.mark.parametrize("width", [12, 20])
def test_periodic_features_are_physical_and_do_not_use_targets(width):
    data = item([[3., 5., 48, 80], [3., 6., 52, 90]])
    data["frames"] = np.zeros((2, 40, 9), np.float32)
    data["periodicity"] = np.linspace(-1., 1., 2*len(NAMES), dtype=np.float32).reshape(2, -1)
    observed = observations(data)[-1]
    assert observed.shape == (2, CONTEXT_WIDTH)
    np.testing.assert_array_equal(observed[:, 100:], data["periodicity"])
    network = model(width)
    normalizer = (np.zeros(CONTEXT_WIDTH), np.ones(CONTEXT_WIDTH))
    before = probabilities(network, data, normalizer)
    data.update(reference=np.empty((0, 3)), pitch_reference=np.empty((0, 3)),
                id="different-song", corpus="different-corpus", y=np.zeros(2))
    np.testing.assert_array_equal(probabilities(network, data, normalizer), before)
    assert np.all(before[:, 0] > .98)
    data["periodicity"][0, 0] = 1.1
    with pytest.raises(ValueError, match="periodicity"):
        observations(data)
