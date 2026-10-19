import numpy as np
import pytest
import torch

from approved_decision_guard_v69 import DecisionGuard, fitting_indices, veto_dependencies
from approved_support_guard_v68 import support_logits
from left_precision_v43 import model


def logit(values):
    return torch.logit(torch.tensor(values))


def test_guard_dependencies_follow_both_unchanged_inference_policies():
    removal = logit([.8, .95, .95, .902, .902, .99])
    support = logit([.8, .3, .18, .18, .3, .1])
    dependent, cutoff = veto_dependencies(removal, support)
    assert dependent.tolist() == [False, True, True, False, True, False]
    torch.testing.assert_close(torch.sigmoid(cutoff[torch.tensor([1, 2, 4])]), torch.tensor([.2, .16, .2]))
    # Outside dependencies, lowering support cannot newly reject a parent KEEP.
    for rejection, presence, need in zip(torch.sigmoid(removal), torch.sigmoid(support), dependent, strict=True):
        for threshold, veto in ((.9, .2), (.905, .16)):
            original_keep = rejection < threshold or presence >= veto
            if original_keep and not need:
                assert rejection < threshold


def test_no_dependent_decisions_is_valid_and_does_not_create_fake_constraints():
    network = model(12)
    inputs = [torch.zeros(2, 40, 9), torch.zeros(2, 1), torch.zeros(2, 1),
              torch.zeros(2, network.condition.in_features)]
    frames = torch.zeros(2, 40)
    teacher = support_logits(network, inputs[0], inputs[3]).detach()
    guard = DecisionGuard(network, inputs, torch.zeros(2, dtype=torch.int64), frames,
                          torch.ones_like(frames), teacher, .25, logit([.1, .2]))
    assert not guard.selected.any()
    with torch.no_grad():
        network.frame.bias.sub_(1)
    assert guard.project(network)['accepted_step_fraction'] == 1


def test_sampler_rehearses_only_registered_training_protection_indices():
    protected = torch.zeros(100, dtype=torch.bool)
    protected[[2, 17, 99]] = True
    first = fitting_indices(np.random.default_rng(3), 100, protected)
    second = fitting_indices(np.random.default_rng(3), 100, protected)
    assert torch.equal(first, second) and len(first) == 256
    assert protected[first[-26:]].all()
    assert first.min() >= 0 and first.max() < 100
    assert len(fitting_indices(np.random.default_rng(3), 100, torch.zeros_like(protected))) == 256


def test_invalid_teacher_or_sampler_inputs_fail_closed():
    with pytest.raises(ValueError):
        veto_dependencies(torch.tensor([float('nan')]), torch.zeros(1))
    with pytest.raises(ValueError):
        fitting_indices(np.random.default_rng(3), 2, torch.ones(3, dtype=torch.bool))
