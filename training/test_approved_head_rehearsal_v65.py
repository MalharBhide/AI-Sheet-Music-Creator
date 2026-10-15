import pytest
import torch

from approved_head_rehearsal_v65 import freeze_backbone, keep_rehearsal, require_frozen_backbone
from left_precision_v43 import model


def test_only_thirteen_readout_parameters_can_change():
    network = model(12)
    original = freeze_backbone(network)
    parameters = [p for p in network.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 13
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    sum(p.sum() for p in parameters).backward()
    optimizer.step()
    require_frozen_backbone(network, original)
    with torch.no_grad():
        next(network.reject.parameters()).reshape(-1)[0] += .01
    with pytest.raises(ValueError, match="classifier changed"):
        require_frozen_backbone(network, original)


def test_original_prediction_has_zero_rehearsal_gradient():
    teacher = torch.linspace(-3., 3., 40).repeat(2, 1)
    logits = teacher.clone().requires_grad_()
    loss = keep_rehearsal(logits, teacher, torch.ones(2, 40), torch.zeros(2, dtype=torch.int64), torch.ones(2))
    loss.backward()
    assert abs(loss.item()) < 1e-6 and logits.grad.abs().max() < 1e-7


def test_unknown_frames_and_remove_candidates_never_get_teacher_gradients():
    logits = torch.full((3, 40), -4., requires_grad=True)
    mask = torch.ones(3, 40)
    mask[0, 10] = 0
    mask[1] = 0
    loss = keep_rehearsal(logits, torch.zeros(3, 40), mask, torch.tensor([0, 0, 1]), torch.ones(3))
    loss.backward()
    assert logits.grad[0].abs().sum() > 0 and logits.grad[0, 10] == 0
    assert logits.grad[1:].abs().sum() == 0


def test_empty_known_keep_population_is_a_safe_zero_gradient():
    logits = torch.zeros(2, 40, requires_grad=True)
    loss = keep_rehearsal(logits, torch.zeros(2, 40), torch.zeros(2, 40),
                          torch.zeros(2, dtype=torch.int64), torch.ones(2))
    loss.backward()
    assert loss.item() == 0 and logits.grad.abs().sum() == 0
