"""Endpoint features have their own trainable path and validated probability contract."""

import numpy as np
import pytest
from attack_release_model import model, probabilities


def test_endpoint_encoder_is_independent_and_receives_gradients():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(270000)
    net=model(24)
    assert net.ending[0].weight.data_ptr()!=net.local[0].weight.data_ptr()
    with torch.no_grad():
        net.classifier[-1].weight[0].fill_(.1)
        net.classifier[-1].weight[1].fill_(-.1)
    logits=net(torch.zeros((2,40,9)),torch.zeros((2,61,18)),torch.ones((2,61,18)),torch.zeros((2,84)))
    torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])).backward()
    grad=net.ending[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)>0


def test_three_scale_probabilities_sum_to_one_and_empty_shape_is_binary():
    import torch
    torch.set_num_threads(1)
    net=model(24)
    p=probabilities(net,np.zeros((2,40,9)),np.zeros((2,61,18)),np.zeros((2,61,18)),
                    np.zeros((2,84)),(np.zeros(84),np.ones(84)))
    assert p.shape==(2,2)
    np.testing.assert_allclose(p.sum(axis=1),1.)
    p=probabilities(net,np.empty((0,40,9)),np.empty((0,61,18)),np.empty((0,61,18)),
                    np.empty((0,84)),(np.zeros(84),np.ones(84)))
    assert p.shape==(0,2)


@pytest.mark.parametrize('endings',[np.zeros((1,60,18)),np.full((1,61,18),np.nan),
    np.full((1,61,18),1.01)])
def test_bad_endpoint_features_rejected(endings):
    with pytest.raises(ValueError,match='inputs'):
        probabilities(model(24),np.zeros((1,40,9)),np.zeros((1,61,18)),endings,
                      np.zeros((1,84)),(np.zeros(84),np.ones(84)))
