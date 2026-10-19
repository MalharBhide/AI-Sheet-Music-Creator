import pytest
import torch

from approved_tangent_guard_v70 import TangentGuard, project_step
from approved_support_guard_v68 import freeze_rejection, require_frozen_rejection, support_logits
from left_precision_v43 import model


def test_projection_preserves_useful_motion_while_repairing_unsafe_component():
    step, receipt = project_step(torch.tensor([-2., 3.]), torch.tensor([[1., 0.]]), torch.tensor([-.1]))
    torch.testing.assert_close(step, torch.tensor([-.1, 3.], dtype=torch.float64), atol=1e-7, rtol=0)
    assert receipt['linear_constraints'] == 1


def test_multiple_nearly_parallel_constraints_remain_feasible():
    rows = torch.tensor([[1., 0.], [1., .001], [-1., 1.]], dtype=torch.float64)
    lower = torch.tensor([0., 0., -.01], dtype=torch.float64)
    step, _ = project_step(torch.tensor([-4., -2.]), rows, lower)
    assert torch.all(rows @ step >= lower-1e-8)


def test_nonlinear_guard_checks_actual_model_after_tangent_repair():
    network = model(12)
    frozen = freeze_rejection(network)
    inputs = [torch.zeros(2, 40, 9), torch.zeros(2, 1), torch.zeros(2, 1),
              torch.zeros(2, network.condition.in_features)]
    frames = torch.zeros(2, 40)
    teacher = support_logits(network, inputs[0], inputs[3]).detach()
    guard = TangentGuard(network, inputs, torch.zeros(2, dtype=torch.int64), frames,
                         torch.ones_like(frames), teacher, .75, torch.logit(torch.tensor([.95, .95])))
    with torch.no_grad():
        network.frame.bias.sub_(10)
    receipt = guard.project(network)
    assert receipt['tangent_repair'] and receipt['linear_constraints'] == 2
    guard.require(network)
    require_frozen_rejection(network, frozen)


@pytest.mark.parametrize('fault', ['nonfinite', 'infeasible_origin', 'zero_gradient'])
def test_invalid_projection_problems_fail_closed(fault):
    delta, rows, lower = torch.ones(2), torch.eye(2), torch.zeros(2)
    if fault == 'nonfinite': delta[0] = float('nan')
    if fault == 'infeasible_origin': lower[0] = .1
    if fault == 'zero_gradient': rows[0] = 0
    with pytest.raises(ValueError):
        project_step(delta, rows, lower)
