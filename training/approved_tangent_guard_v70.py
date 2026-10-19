"""Repair unsafe support updates along fitting-only constraint gradients."""

import numpy as np
import torch
from scipy.optimize import minimize

from approved_decision_guard_v69 import DecisionGuard
from approved_support_guard_v68 import support_logits, support_state


def project_step(delta, jacobian, lower):
    """Euclidean projection onto local support halfspaces, checked explicitly.

    Solve the small nonnegative dual; never trust solver success as a safety
    proof. Repair residual violations and retreat toward the feasible zero step.
    The caller must still verify the actual nonlinear network afterwards.
    """
    if (delta.ndim != 1 or jacobian.shape != (len(lower), len(delta)) or lower.ndim != 1
            or any(v.requires_grad or not torch.isfinite(v).all() for v in (delta, jacobian, lower))
            or torch.any(lower > 0)):
        raise ValueError('Invalid locally feasible training constraints')
    delta, jacobian, lower = (v.double() for v in (delta, jacobian, lower))
    if not len(lower):
        return delta, {'solver_success': True, 'linear_constraints': 0, 'linear_retreat': 1.}
    if torch.any(jacobian.square().sum(dim=1) <= 0):
        raise ValueError('Zero support-constraint gradient')
    gram = (jacobian @ jacobian.T).numpy()
    slack = (jacobian @ delta-lower).numpy()
    result = minimize(lambda x: (.5*x @ gram @ x+slack @ x, gram @ x+slack),
                      np.zeros(len(lower)), method='L-BFGS-B', jac=True,
                      bounds=[(0., None)]*len(lower),
                      options={'maxiter': 512, 'maxls': 40, 'ftol': 1e-14, 'gtol': 1e-10})
    if not np.isfinite(result.x).all():
        raise ValueError('Nonfinite support-constraint solution')
    repaired = delta+jacobian.T @ torch.from_numpy(result.x)
    for _ in range(128):
        amount, index = (lower-jacobian @ repaired).max(dim=0)
        if amount <= 1e-10:
            break
        row = jacobian[index]
        repaired += (amount+1e-10)*row/row.square().sum()
    movement = jacobian @ repaired
    crossed = movement < lower-1e-9
    retreat = 1.
    if crossed.any():
        retreat = max(0., min(1., float((lower[crossed]/movement[crossed]).min())*.999))
        repaired *= retreat
    if not torch.isfinite(repaired).all() or torch.any(jacobian @ repaired < lower-1e-8):
        raise ValueError('Local support projection failed its explicit check')
    return repaired, {'solver_success': bool(result.success), 'linear_constraints': len(lower),
                      'linear_retreat': retreat}


class TangentGuard(DecisionGuard):
    def project(self, network):
        proposed = support_state(network)
        if any(not torch.isfinite(v).all() for v in proposed.values()):
            raise ValueError('Nonfinite proposed support weights')
        try:
            self.require(network)
        except ValueError:
            pass
        else:
            return {**super().project(network), 'tangent_repair': False}
        parameters = [(k, p) for k, p in network.named_parameters() if k in self.previous]
        with torch.no_grad():
            for key, parameter in parameters:
                parameter.copy_(self.previous[key])
        self.require(network)
        indices = torch.where(self.selected)[0]
        gradients, values = [], []
        # Small independent batches bound memory used by batched autograd.
        with torch.enable_grad():
            for first in range(0, len(indices), 16):
                idx = indices[first:first+16]
                logits = support_logits(network, self.coarse[idx], self.context[idx])
                selected = logits.gather(1, self.where[idx, None]).squeeze(1)
                derivative = torch.autograd.grad(selected, [p for _, p in parameters],
                    grad_outputs=torch.eye(len(idx), dtype=selected.dtype), is_grads_batched=True)
                gradients.append(torch.cat([g.flatten(start_dim=1) for g in derivative], dim=1).detach().double())
                values.append(selected.detach().double())
        jacobian, current = torch.cat(gradients), torch.cat(values)
        delta = torch.cat([(proposed[k]-self.previous[k]).reshape(-1) for k, _ in parameters]).double()
        # The previously accepted network may be below a floor only within the
        # fixed 2e-6 numerical tolerance; do not demand infeasible zero movement.
        lower = (self.floor[indices].double()-current).clamp_max(0)
        corrected, receipt = project_step(delta, jacobian, lower)
        offset = 0
        with torch.no_grad():
            for key, parameter in parameters:
                count = parameter.numel()
                update = corrected[offset:offset+count].reshape_as(parameter).float()
                parameter.copy_(self.previous[key]+update)
                offset += count
        # The tangent approximation is NOT a release/training safety proof.
        # Reuse the full nonlinear checks, backtracking and exact rollback.
        return {**super().project(network), **receipt, 'tangent_repair': True}
