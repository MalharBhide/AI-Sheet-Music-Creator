import pytest
import torch
from left_precision_v43 import model as released_model
from left_precision_v49 import initialize, model
from test_left_precision_v48 import inputs


def teacher():
    torch.manual_seed(46291)
    result = released_model(12).eval()
    # Exercise nonconstant, already-learned heads; default heads are constant.
    for parameter in result.parameters():
        torch.nn.init.uniform_(parameter, -.12, .12)
    return result


def test_warm_start_preserves_both_learned_paths_exactly_without_mutating_teacher():
    original = teacher()
    before = {n: v.detach().clone() for n, v in original.state_dict().items()}
    candidate = initialize(original).eval()
    values = inputs()
    with torch.no_grad():
        for expected, actual in zip(original(*values), candidate(*values), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not torch.any(candidate.boundary_condition.weight)


def test_zero_residual_can_start_learning_and_reach_both_boundary_encoders():
    candidate = initialize(teacher()).eval()
    optimizer = torch.optim.SGD(candidate.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, frame = candidate(*inputs())
        torch.nn.functional.binary_cross_entropy_with_logits(frame, torch.zeros_like(frame)).backward()
        optimizer.step()
    assert torch.any(candidate.boundary_condition.weight != 0)
    for encoder in (candidate.attack_condition, candidate.release_condition):
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_width", "unrelated"])
def test_incompatible_or_nonfinite_teacher_is_rejected(kind):
    original = teacher()
    if kind == "nonfinite":
        with torch.no_grad():
            original.frame.weight.fill_(float("nan"))
    elif kind == "wrong_width":
        original = released_model(20)
    else:
        original = torch.nn.Linear(10, 2)
    with pytest.raises(ValueError, match="architecture"):
        initialize(original)


def test_width_cannot_change_in_a_released_weight_initialization():
    with pytest.raises(ValueError):
        model(20)
