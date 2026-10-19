"""Fit existing support features with fitting-only decision rollback guards."""

import math

import torch

from approved_coverage_constraints_v67 import coverage_targets


def freeze_rejection(network):
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(not name.startswith('reject.'))
    return {k: v.detach().clone() for k, v in network.state_dict().items() if k.startswith('reject.')}


def require_frozen_rejection(network, original):
    current = {k: v for k, v in network.state_dict().items() if k.startswith('reject.')}
    if current.keys() != original.keys() or any(not torch.equal(current[k], v) for k, v in original.items()):
        raise ValueError('Approved note-rejection classifier changed')


def support_state(network):
    return {k: v.detach().clone() for k, v in network.state_dict().items() if not k.startswith('reject.')}


def support_logits(network, coarse, context):
    hidden = torch.relu(network.presence(coarse.transpose(1, 2))+network.condition(context).unsqueeze(2))
    return network.frame(hidden).squeeze(1)


class SupportGuard:
    def __init__(self, network, inputs, target, frame_target, frame_weight, teacher, retention):
        protection = coverage_targets(target, frame_target, frame_weight) == 0
        if (not 0 < retention <= 1 or teacher.shape != frame_target.shape or teacher.requires_grad
                or not torch.isfinite(teacher).all() or len(inputs) != 4
                or any(len(v) != len(target) for v in inputs)):
            raise ValueError('Invalid fitting-only support guard')
        peak, where = teacher[:, 8:32].max(dim=1)
        self.selected = protection & (torch.sigmoid(peak) >= .16)
        self.where = where+8
        cutoff = torch.where(torch.sigmoid(peak) >= .2, math.log(.2/.8), math.log(.16/.84))
        self.floor = peak-(1-retention)*(peak-cutoff).clamp_min(0)
        self.coarse, self.context = inputs[0], inputs[3]
        self.previous = support_state(network)
        if not self.selected.any():
            raise ValueError('Missing protected training support')
        self.require(network)

    def loss(self, logits, indices, weights):
        selected = self.selected[indices]
        if not selected.any():
            return logits.reshape(-1)[:0].sum()
        current = logits.gather(1, self.where[indices, None]).squeeze(1)
        gap = (self.floor[indices][selected]-current[selected]).clamp_min(0)
        return (gap.square()*weights[selected]).mean()

    def require(self, network):
        if any(not torch.isfinite(v).all() for v in support_state(network).values()):
            raise ValueError('Nonfinite support weights')
        minimum = float('inf')
        with torch.inference_mode():
            indices = torch.where(self.selected)[0]
            for first in range(0, len(indices), 256):
                idx = indices[first:first+256]
                logits = support_logits(network, self.coarse[idx], self.context[idx])
                current = logits.gather(1, self.where[idx, None]).squeeze(1)
                margin = float((current-self.floor[idx]).min())
                minimum = min(minimum, margin)
                if not torch.isfinite(current).all() or margin < -2e-6:
                    raise ValueError('Training pitch-support decision violated')
        return minimum

    def project(self, network):
        """Accept only a full-training-preserving step; otherwise retreat.

        The nonlinear network is checked at each proposed fraction. We do not
        assume monotonic feasibility or infer safety from interpolation alone.
        """
        proposed = support_state(network)
        if any(not torch.isfinite(v).all() for v in proposed.values()):
            raise ValueError('Nonfinite proposed support weights')
        for attempt in range(13):
            fraction = 2.**-attempt
            if attempt:
                with torch.no_grad():
                    for key, parameter in network.named_parameters():
                        if key in proposed:
                            parameter.copy_(self.previous[key]+fraction*(proposed[key]-self.previous[key]))
            try:
                margin = self.require(network)
                break
            except ValueError:
                continue
        else:
            with torch.no_grad():
                for key, parameter in network.named_parameters():
                    if key in self.previous:
                        parameter.copy_(self.previous[key])
            fraction, margin = 0., self.require(network)
        self.previous = support_state(network)
        return {'accepted_step_fraction': fraction, 'minimum_margin': margin}
