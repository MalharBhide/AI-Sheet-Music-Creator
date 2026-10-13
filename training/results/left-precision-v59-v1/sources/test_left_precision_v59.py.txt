import pytest
import torch
from left_precision_v59 import absence_candidates, max_absence_loss


def supervision(count=4):
    return torch.zeros(count, 40), torch.ones(count, 40), torch.ones(count, dtype=torch.int64)


def test_only_fully_known_absent_remove_interiors_receive_the_new_penalty():
    target, mask, labels = supervision()
    labels[0] = 0
    target[1, 8] = 1
    mask[2, 31] = 0
    torch.testing.assert_close(absence_candidates(target, mask, labels), torch.tensor([False, False, False, True]))


def test_subframe_fragment_and_last_interior_frame_still_protect_the_candidate():
    target, mask, labels = supervision(2)
    target[0, 31], target[1, 8] = 1, .01
    assert not absence_candidates(target, mask, labels).any()


def test_exterior_support_or_unknown_does_not_invent_an_interior_presence_label():
    target, mask, labels = supervision(2)
    target[0, 7], target[0, 32] = 1, 1
    mask[1, 7], mask[1, 32] = 0, 0
    assert absence_candidates(target, mask, labels).all()


def test_loss_and_gradient_target_the_strongest_wrong_interior_support_only():
    target, mask, labels = supervision()
    labels[0] = 0
    mask[1, 8] = 0
    target[2, 31] = 1
    logits = torch.zeros(4, 40, requires_grad=True)
    with torch.no_grad():
        logits[:3] = 100
        logits[3, 12] = 4
        logits[3, 7] = 100
    weights = torch.tensor([1., 2., 3., 4.])
    loss = max_absence_loss(logits, target, mask, labels, weights)
    assert loss.item() == pytest.approx(4*torch.nn.functional.softplus(torch.tensor(4.)).item())
    loss.backward()
    assert logits.grad[3, 12] > 0
    gradient = logits.grad.clone()
    gradient[3, 12] = 0
    assert torch.count_nonzero(gradient) == 0


def test_all_excluded_candidates_have_differentiable_zero_loss_without_overflow():
    target, mask, labels = supervision(2)
    labels.zero_()
    logits = torch.full((2, 40), 1e38, requires_grad=True)
    loss = max_absence_loss(logits, target, mask, labels, torch.ones(2))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_extreme_wrong_frame_logit_has_finite_loss_and_gradient():
    target, mask, labels = supervision(1)
    logits = torch.full((1, 40), -1e6, requires_grad=True)
    with torch.no_grad():
        logits[0, 15] = 1e6
    loss = max_absence_loss(logits, target, mask, labels, torch.ones(1))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 15] > 0


@pytest.mark.parametrize("mutation", ["shape", "unknown-note", "note-type", "target-nan", "mask-half", "target-gradient"])
def test_invalid_supervision_cannot_supply_absence_labels(mutation):
    target, mask, labels = supervision()
    if mutation == "shape":
        target = target[:, :-1]
    elif mutation == "unknown-note":
        labels[0] = -100
    elif mutation == "note-type":
        labels = labels.float()
    elif mutation == "target-nan":
        target[0, 0] = float("nan")
    elif mutation == "mask-half":
        mask[0, 0] = .5
    else:
        target.requires_grad_()
    with pytest.raises(ValueError, match="supervision"):
        absence_candidates(target, mask, labels)


@pytest.mark.parametrize("mutation", ["shape", "logit-nan", "weights-shape", "weight-zero", "weight-nan", "weight-gradient"])
def test_invalid_loss_batch_rejected(mutation):
    target, mask, labels = supervision()
    logits, weights = torch.zeros(4, 40), torch.ones(4)
    if mutation == "shape":
        logits = logits[:, :-1]
    elif mutation == "logit-nan":
        logits[0, 0] = float("nan")
    elif mutation == "weights-shape":
        weights = weights[:1]
    elif mutation == "weight-zero":
        weights[0] = 0
    elif mutation == "weight-nan":
        weights[0] = float("nan")
    else:
        weights.requires_grad_()
    with pytest.raises(ValueError, match="batch"):
        max_absence_loss(logits, target, mask, labels, weights)
