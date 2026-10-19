import pytest
import torch

from approved_readout_constraints_v66 import KeepConstraints, readout, training_constraints
from left_precision_v43 import model


def test_projection_preserves_all_halfspaces_and_frozen_features():
    network = model(12)
    original = readout(network)
    frozen = {k: v.clone() for k, v in network.state_dict().items() if not k.startswith('frame.')}
    rows = torch.eye(13, dtype=torch.float64)
    constraints = KeepConstraints(rows, torch.full((13,), -.01, dtype=torch.float64), original)
    with torch.no_grad():
        network.frame.weight.sub_(1)
        network.frame.bias.sub_(1)
    constraints.project(network)
    constraints.require(readout(network))
    assert torch.all(readout(network) >= original-.010002)
    for key, value in frozen.items():
        assert torch.equal(network.state_dict()[key], value)


def test_opposing_constraints_cannot_be_bypassed_by_projection():
    network = model(12)
    original = readout(network)
    rows = torch.cat((torch.eye(13), -torch.eye(13))).double()
    constraints = KeepConstraints(rows, torch.zeros(26, dtype=torch.float64), original)
    with torch.no_grad():
        network.frame.weight.add_(.2)
        network.frame.bias.sub_(.2)
    constraints.project(network)
    torch.testing.assert_close(readout(network), original, atol=2e-6, rtol=0)


def test_only_supported_labeled_keep_notes_create_constraints():
    network = model(12)
    inputs = [torch.zeros(3, 40, 9), torch.zeros(3, 1), torch.zeros(3, 1),
              torch.zeros(3, network.condition.in_features)]
    constraints = training_constraints(network, inputs, torch.tensor([0, 1, 0]), .25)
    assert len(constraints.rows) == 2
    constraints.require(readout(network))
    # Changing REMOVE-only input cannot change the KEEP constraints.
    inputs[0][1] = 100
    other = training_constraints(network, inputs, torch.tensor([0, 1, 0]), .25)
    torch.testing.assert_close(other.rows, constraints.rows)
    torch.testing.assert_close(other.lower_delta, constraints.lower_delta)


def test_nonfinite_or_empty_constraints_fail_closed():
    network = model(12)
    inputs = [torch.zeros(2, 40, 9), torch.zeros(2, 1), torch.zeros(2, 1),
              torch.zeros(2, network.condition.in_features)]
    with pytest.raises(ValueError, match='Missing'):
        training_constraints(network, inputs, torch.ones(2, dtype=torch.int64), .25)
    constraints = KeepConstraints(torch.eye(13, dtype=torch.float64),
                                  torch.zeros(13, dtype=torch.float64), readout(network))
    with torch.no_grad():
        network.frame.bias.fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        constraints.project(network)


def test_constraints_allow_false_support_reduction_when_features_separate_it():
    network = model(12)
    original = readout(network)
    keep = torch.zeros(1, 13, dtype=torch.float64)
    keep[0, 0] = keep[0, -1] = 1
    absent = torch.zeros(1, 13, dtype=torch.float64)
    absent[0, 1] = absent[0, -1] = 1
    constraints = KeepConstraints(keep, torch.tensor([-1.], dtype=torch.float64), original)
    with torch.no_grad():
        network.frame.weight[0, 1, 0] = -10
    constraints.project(network)
    assert torch.sigmoid(keep @ readout(network)) > .2
    assert torch.sigmoid(absent @ readout(network)) < .16


@pytest.mark.parametrize('retention', [0, -1, 1.1, float('nan')])
def test_invalid_margin_retention_rejected(retention):
    with pytest.raises(ValueError):
        training_constraints(model(12), [], torch.zeros(1, dtype=torch.int64), retention)
