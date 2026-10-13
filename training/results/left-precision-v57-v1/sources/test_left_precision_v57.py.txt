import numpy as np
import pytest
import torch
from left_precision_v57 import hard_keep_loss


def test_hard_genuine_note_protection_targets_only_the_worst_keep_examples():
    logits = torch.tensor([[5., 0.], [0., 4.], [0., 10.], [2., 1.]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 0])
    weights = torch.tensor([1., 2., 3., 4.])
    loss = hard_keep_loss(logits, labels, weights, .1)
    assert loss.item() == pytest.approx(2*torch.nn.functional.softplus(torch.tensor(4.)).item())
    loss.backward()
    assert logits.grad[1, 0] < 0 and logits.grad[1, 1] > 0
    assert torch.count_nonzero(logits.grad[[0, 2, 3]]) == 0


def test_no_keep_batch_has_differentiable_zero_loss():
    logits = torch.tensor([[0., 10.], [1., 2.]], requires_grad=True)
    loss = hard_keep_loss(logits, torch.ones(2, dtype=torch.int64), torch.ones(2))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_extreme_false_rejection_loss_and_gradient_remain_finite():
    logits = torch.tensor([[-1e6, 1e6], [1e6, -1e6]], requires_grad=True)
    loss = hard_keep_loss(logits, torch.zeros(2, dtype=torch.int64), torch.ones(2))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_full_tail_fraction_preserves_every_keep_loss_and_original_weights():
    logits = torch.tensor([[0., 2.], [0., 3.], [0., 99.]])
    weights = torch.tensor([2., 3., 1.])
    expected = (torch.nn.functional.softplus(torch.tensor([2., 3.]))*weights[:2]).mean()
    torch.testing.assert_close(hard_keep_loss(logits, torch.tensor([0, 0, 1]), weights, 1), expected)


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_hard_keep_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        hard_keep_loss(torch.zeros(2, 2), torch.zeros(2, dtype=torch.int64), torch.ones(2), fraction)


@pytest.mark.parametrize("mutation", ["shape", "target-type", "unknown", "weight", "nan"])
def test_invalid_hard_keep_batch_rejected(mutation):
    logits, labels, weights = torch.zeros(2, 2), torch.zeros(2, dtype=torch.int64), torch.ones(2)
    if mutation == "shape":
        weights = torch.ones(3)
    elif mutation == "target-type":
        labels = labels.float()
    elif mutation == "unknown":
        labels[0] = -1
    elif mutation == "weight":
        weights[0] = 0
    else:
        logits[0, 0] = float("nan")
    with pytest.raises(ValueError):
        hard_keep_loss(logits, labels, weights)


def test_generated_key_support_is_not_an_audibility_target(monkeypatch):
    import left_precision_v57 as study

    target = np.array([[0, 1, 0, 1]], np.float32)
    weight = np.ones_like(target)
    monkeypatch.setattr(study, "previous_frame_supervision", lambda item: (target, weight))
    result = study.frame_supervision({"corpus": "keynata-authored-sampled-piano", "group": "train"})
    np.testing.assert_array_equal(result[0], target)
    np.testing.assert_array_equal(result[1], [[1, 0, 1, 0]])
    np.testing.assert_array_equal(weight, np.ones_like(target))
    with pytest.raises(ValueError, match="fitting-only"):
        study.frame_supervision({"corpus": "keynata-authored-sampled-piano", "group": "validation"})


@pytest.mark.parametrize("mutation", [None, "missing", "group", "size", "duplicate", "overlap"])
def test_all_new_generated_parents_are_complete_and_fitting_only(monkeypatch, mutation):
    import left_precision_v57 as study

    earlier = [{"id": f"old-{i}", "corpus": "earlier", "source_group": f"old/{i}"}
               for i in range(2848)]
    new = [{"id": f"{name}/{i}", "corpus": "keynata-authored-sampled-piano", "group": "train",
            "source_group": "keynata-generated/"+name}
           for name, count in study.WINDOWS.items() for i in range(3*count)]
    checked = []
    monkeypatch.setattr(study, "previous_partitions", lambda items: checked.append(items))
    if mutation == "missing":
        new.pop()
    elif mutation == "group":
        new[0]["group"] = "validation"
    elif mutation == "size":
        new[0]["source_group"] = new[-1]["source_group"]
    elif mutation == "duplicate":
        new[0]["id"] = earlier[0]["id"]
    elif mutation == "overlap":
        earlier[0]["source_group"] = new[0]["source_group"]
    if mutation:
        with pytest.raises(ValueError, match="partition"):
            study.validate_partitions(earlier+new)
    else:
        study.validate_partitions(earlier+new)
    assert checked == [earlier]
