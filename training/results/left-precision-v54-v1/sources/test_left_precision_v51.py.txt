import numpy as np
import pytest
import torch
from left_precision_v46 import permissions
from left_precision_v51 import FRESH_SEEDS, gate, initialize, model, rejection_rehearsal_loss
from test_left_precision_v48 import inputs
from test_left_precision_v49 import teacher


def test_warm_rejection_is_exact_but_fresh_frame_veto_keeps_all_baseline_survivors():
    original = teacher()
    before = {n: p.clone() for n, p in original.state_dict().items()}
    candidate = initialize(original).eval()
    values = inputs()
    with torch.no_grad():
        expected = original(*values)[0]
        rejection, frame = candidate(*values)
        torch.testing.assert_close(rejection, expected, rtol=0, atol=0)
        torch.testing.assert_close(frame, torch.full_like(frame, 4.), rtol=0, atol=0)
        heads = torch.softmax(rejection, dim=1).numpy(), torch.sigmoid(frame).numpy()
    baseline = np.arange(len(frame)) % 2 == 0
    for veto in (.05, .1, .2, .3):
        for margin in (1., .8):
            allowed = gate(heads, veto*margin)
            np.testing.assert_array_equal(allowed, np.tile([1., 0.], (len(frame), 1)))
            locked = permissions(allowed, baseline)
            np.testing.assert_array_equal(locked[:, 0] == 1, baseline)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_fresh_frame_learns_from_both_boundaries_and_checkpoint_roundtrips(tmp_path):
    network = initialize(teacher()).eval()
    optimizer = torch.optim.SGD(network.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, frame = network(*inputs())
        torch.nn.functional.binary_cross_entropy_with_logits(frame, torch.zeros_like(frame)).backward()
        optimizer.step()
    for encoder in (network.attack_condition, network.release_condition):
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())
    path = tmp_path / "checkpoint.pt"
    torch.save(network.state_dict(), path)
    restored = model(12).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for a, b in zip(network(*inputs()), restored(*inputs()), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rehearsal_uses_rejection_probabilities_without_teacher_gradient():
    logits = torch.tensor([[.3, .7]], requires_grad=True)
    reference = torch.softmax(logits.detach(), dim=1).requires_grad_()
    loss = rejection_rehearsal_loss(logits, reference)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    loss.sum().backward()
    assert reference.grad is None
    assert rejection_rehearsal_loss(torch.tensor([[3., -3.]]), reference).item() > 0


@pytest.mark.parametrize("reference", [torch.ones(2, 2), torch.tensor([[float("nan"), 0.]]),
                                        torch.tensor([[-1., 2.]]), torch.ones(1, 1)])
def test_invalid_rehearsal_targets_fail_closed(reference):
    with pytest.raises(ValueError):
        rejection_rehearsal_loss(torch.zeros(1, 2), reference)


def test_initialization_and_fresh_bank_are_fixed():
    assert FRESH_SEEDS == tuple(range(296101, 296141)) and len(set(FRESH_SEEDS)) == 40
    with pytest.raises(ValueError):
        model(20)
    with pytest.raises(ValueError):
        initialize(torch.nn.Linear(2, 2))


def test_nonfinite_released_parameters_cannot_initialize_candidate():
    original = teacher()
    with torch.no_grad():
        next(original.reject.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError):
        initialize(original)
