import numpy as np
import pytest
from left_precision_v39 import keep
from left_precision_v43 import CONTEXT_WIDTH, gate, model, probabilities
from test_left_precision_v42 import fixture


def test_stricter_presence_policy_can_only_retain_more_notes():
    rejection = np.tile([.001, .999], (4, 1))
    presence = np.zeros((4, 40))
    presence[:, 20] = [.01, .08, .09, .2]
    raw = gate((rejection, presence), .1)
    strict = gate((rejection, presence), .08)
    np.testing.assert_array_equal(raw[:, 1] > .99, [True, True, True, False])
    np.testing.assert_array_equal(strict[:, 1] > .99, [True, False, False, False])
    assert np.all(strict[:, 1] <= raw[:, 1])
    with pytest.raises(ValueError, match="presence veto"):
        gate((rejection, presence), 0.)


def test_independent_veto_keeps_scope_protections():
    events = np.array([[1., 2., 40., 80], [3., 4., 61., 80], [3., 4., 48., 80]])
    heads = (np.tile([0., 1.], (3, 1)), np.zeros((3, 40)))
    np.testing.assert_array_equal(keep(events, gate(heads, .1), 20., .99), [True, True, False])


def test_separate_head_confidences_do_not_read_labels_and_start_protected():
    data = fixture()
    network = model(12)
    normalizer = np.zeros(CONTEXT_WIDTH), np.ones(CONTEXT_WIDTH)
    before = probabilities(network, data, normalizer)
    data.update(reference=np.empty((0, 3)), pitch_reference=np.empty((0, 3)),
                mask=np.zeros(2, bool), y=np.zeros(2), id="another-recording")
    after = probabilities(network, data, normalizer)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    assert before[0].shape == (2, 2) and before[1].shape == (2, 40)
    np.testing.assert_array_equal(gate(before, .3), np.tile([1., 0.], (2, 1)))
