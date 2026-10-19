import pytest
import torch

from approved_support_guard_v68 import (
    SupportGuard, freeze_rejection, require_frozen_rejection, support_logits, support_state,
)
from left_precision_v43 import model


def fixture():
    network = model(12)
    inputs = [torch.zeros(3, 40, 9), torch.zeros(3, 1), torch.zeros(3, 1),
              torch.zeros(3, network.condition.in_features)]
    frames = torch.zeros(3, 40)
    frames[1, 15] = 1
    teacher = support_logits(network, inputs[0], inputs[3]).detach()
    guard = SupportGuard(network, inputs, torch.tensor([0, 1, 1]), frames,
                         torch.ones_like(frames), teacher, .75)
    return network, inputs, guard


def test_existing_support_features_train_while_rejection_remains_immutable():
    network, _, _ = fixture()
    original = freeze_rejection(network)
    assert all(p.requires_grad == (not name.startswith('reject.')) for name, p in network.named_parameters())
    assert sum(p.numel() for p in network.parameters() if p.requires_grad) == 3265
    with torch.no_grad():
        network.condition.bias.add_(.1)
    require_frozen_rejection(network, original)
    with torch.no_grad():
        next(network.reject.parameters()).reshape(-1)[0] += .1
    with pytest.raises(ValueError, match='classifier changed'):
        require_frozen_rejection(network, original)


def test_unsafe_step_is_backtracked_and_checked_on_all_protected_candidates():
    network, _, guard = fixture()
    assert guard.selected.tolist() == [True, True, False]
    with torch.no_grad():
        network.frame.bias.sub_(20)
    with pytest.raises(ValueError, match='decision violated'):
        guard.require(network)
    receipt = guard.project(network)
    assert 0 < receipt['accepted_step_fraction'] < 1
    guard.require(network)


def test_exhausted_backtracking_restores_exact_previous_state():
    network, _, guard = fixture()
    original = support_state(network)
    with torch.no_grad():
        network.frame.bias.sub_(1e9)
    assert guard.project(network)['accepted_step_fraction'] == 0
    for key, value in original.items():
        assert torch.equal(network.state_dict()[key], value)


def test_support_margin_loss_excludes_unprotected_absence_candidates():
    _, _, guard = fixture()
    logits = torch.full((3, 40), -10., requires_grad=True)
    guard.loss(logits, torch.arange(3), torch.ones(3)).backward()
    assert logits.grad[:2].abs().sum() > 0
    assert logits.grad[2].abs().sum() == 0


def test_nonfinite_update_is_not_silently_accepted_or_repaired():
    network, _, guard = fixture()
    with torch.no_grad():
        network.frame.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        guard.project(network)


def test_guarded_optimizer_can_learn_existing_support_features():
    network, inputs, guard = fixture()
    original_rejection = freeze_rejection(network)
    with torch.no_grad():
        network.frame.weight.fill_(.01)
    before = network.condition.bias.detach().clone()
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=1e-4)
    support_logits(network, inputs[0], inputs[3]).mean().backward()
    optimizer.step()
    guard.project(network)
    assert not torch.equal(network.condition.bias, before)
    require_frozen_rejection(network, original_rejection)
    guard.require(network)
