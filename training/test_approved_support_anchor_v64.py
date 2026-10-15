import pytest
import torch

from approved_support_anchor_v64 import (
    freeze_support, note_anchor, require_frozen_support, support_anchor,
    teacher_outputs,
)
from left_precision_v43 import model


def test_note_anchor_protects_keep_without_teaching_remove_from_teacher():
    logits = torch.tensor([[0., 2.], [0., 100.]], requires_grad=True)
    target = torch.tensor([0, 1])
    loss = note_anchor(logits, target, torch.zeros(2), torch.ones(2))
    loss.backward()
    assert loss.item() == 4.
    assert logits.grad[0, 1] > 0
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_note_anchor_has_zero_loss_below_original_keep_confidence():
    logits = torch.tensor([[0., -2.]], requires_grad=True)
    loss = note_anchor(logits, torch.tensor([0]), torch.tensor([-1.]), torch.ones(1))
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))


def test_support_anchor_never_trains_unknown_or_remove_frames():
    logits = torch.full((3, 40), -4., requires_grad=True)
    target, mask = torch.ones(3, 40), torch.ones(3, 40)
    mask[1] = 0
    mask[0, 10] = 0
    loss = support_anchor(logits, target, mask, torch.tensor([0, 0, 1]), torch.zeros(3, 40), torch.ones(3))
    loss.backward()
    assert loss.item() > 0
    assert logits.grad[0, 8:32].abs().sum() > 0
    assert logits.grad[0, 10] == 0
    assert logits.grad[0, :8].abs().sum() == 0 and logits.grad[0, 32:].abs().sum() == 0
    assert logits.grad[1:].abs().sum() == 0


def test_unknown_positive_targets_and_known_negative_frames_have_zero_anchor_gradient():
    logits = torch.zeros(2, 40, requires_grad=True)
    target, mask = torch.zeros(2, 40), torch.ones(2, 40)
    target[0, 10] = 1
    mask[0, 10] = 0
    loss = support_anchor(logits, target, mask, torch.zeros(2, dtype=torch.int64),
                          torch.full((2, 40), 10.), torch.ones(2))
    loss.backward()
    assert loss.item() == 0 and logits.grad.abs().sum() == 0


def test_one_known_positive_fragment_is_protected_without_other_frame_gradients():
    logits = torch.full((1, 40), -4., requires_grad=True)
    target = torch.zeros(1, 40)
    target[0, 17] = 1
    loss = support_anchor(logits, target, torch.ones(1, 40), torch.zeros(1, dtype=torch.int64),
                          torch.zeros(1, 40), torch.ones(1))
    loss.backward()
    assert logits.grad[0, 17] < 0 and torch.count_nonzero(logits.grad) == 1


def test_freezing_support_preserves_exact_approved_branch_during_optimizer_step():
    network = model(12)
    original = freeze_support(network)
    trainable = [p for p in network.parameters() if p.requires_grad]
    assert trainable and len(trainable) < len(list(network.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=.01)
    sum(p.sum() for p in trainable).backward()
    optimizer.step()
    require_frozen_support(network, original)
    with torch.no_grad():
        next(network.presence.parameters()).reshape(-1)[0] += .01
    with pytest.raises(ValueError, match="branch changed"):
        require_frozen_support(network, original)


def test_teacher_outputs_are_detached_and_can_be_used_safely_in_gradient_losses():
    network = model(12)
    inputs = [torch.zeros(2, *shape) for shape in ((40, 9), (61, 18), (61, 18), (163,))]
    notes, frames = teacher_outputs(network, inputs)
    assert notes.shape == (2,) and frames.shape == (2, 40)
    assert not notes.requires_grad and not frames.requires_grad
    rejection, presence = network(*inputs)
    loss = note_anchor(rejection, torch.zeros(2, dtype=torch.int64), notes, torch.ones(2))
    loss += support_anchor(presence, torch.ones(2, 40), torch.ones(2, 40),
                           torch.zeros(2, dtype=torch.int64), frames, torch.ones(2))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
