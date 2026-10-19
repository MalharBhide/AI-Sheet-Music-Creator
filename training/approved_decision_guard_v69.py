"""Protect only teacher decisions whose immutable rejection head needs a veto."""

import math

import torch

from approved_support_guard_v68 import SupportGuard, support_state


def veto_dependencies(remove_odds, peak):
    if (remove_odds.shape != peak.shape or peak.ndim != 1
            or remove_odds.requires_grad or peak.requires_grad
            or not torch.isfinite(remove_odds).all() or not torch.isfinite(peak).all()):
        raise ValueError('Invalid immutable teacher gate scores')
    rejection, presence = torch.sigmoid(remove_odds), torch.sigmoid(peak)
    # Include a small conservative band for softmax-versus-sigmoid rounding.
    normal = (rejection >= .9-1e-6) & (presence >= .2)
    strict = (rejection >= .905-1e-6) & (presence >= .16)
    cutoff = torch.where(normal, math.log(.2/.8), math.log(.16/.84))
    return normal | strict, cutoff


class DecisionGuard(SupportGuard):
    def __init__(self, network, inputs, target, frame_target, frame_weight, teacher, retention, remove_odds):
        super().__init__(network, inputs, target, frame_target, frame_weight, teacher, retention)
        peak = teacher[:, 8:32].amax(dim=1)
        dependent, cutoff = veto_dependencies(remove_odds, peak)
        self.selected &= dependent
        self.floor = peak-(1-retention)*(peak-cutoff).clamp_min(0)
        self.require(network)

    def require(self, network):
        if self.selected.any():
            return super().require(network)
        if any(not torch.isfinite(v).all() for v in support_state(network).values()):
            raise ValueError('Nonfinite support weights')
        return 0.


def fitting_indices(rng, count, protected):
    """Draw the fixed 10% protection rehearsal only from training indices."""
    if (type(count) is not int or count <= 0 or protected.ndim != 1
            or protected.dtype != torch.bool or len(protected) != count or protected.requires_grad):
        raise ValueError('Invalid training protection sampler')
    members = torch.where(protected)[0].numpy()
    extra = 26 if len(members) else 0  # round(256 * .1)
    ordinary = torch.from_numpy(rng.integers(0, count, 256-extra))
    if not extra:
        return ordinary
    selected = torch.from_numpy(rng.choice(members, extra, replace=True))
    return torch.cat((ordinary, selected))
