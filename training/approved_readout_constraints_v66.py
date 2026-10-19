"""Constrain readout updates using labeled training KEEP decisions only.

These are teacher decision constraints, not labels for unknown acoustic frames.
The supervised frame loss continues to give unknown frames zero gradient.
"""

import math
from dataclasses import dataclass

import torch


def readout(network):
    return torch.cat((network.frame.weight.detach().reshape(-1), network.frame.bias.detach())).double()


@dataclass
class KeepConstraints:
    rows: torch.Tensor
    lower_delta: torch.Tensor
    original: torch.Tensor

    def require(self, parameters):
        if (parameters.shape != self.original.shape or not torch.isfinite(parameters).all()
                or torch.any(self.rows @ (parameters.double()-self.original) < self.lower_delta-2e-6)):
            raise ValueError("Training KEEP support constraint violated")

    def project(self, network):
        """Repair violated halfspaces, then fall back toward the feasible teacher.

        Projection uses every registered training constraint, never validation.
        A final float32 round-trip is checked before accepting the update.
        """
        proposed = readout(network)
        if not torch.isfinite(proposed).all():
            raise ValueError("Nonfinite proposed readout")
        delta = proposed-self.original
        repairs = 0
        for _ in range(64):
            violation = self.lower_delta-self.rows @ delta
            amount, index = violation.max(dim=0)
            if amount <= 0:
                break
            row = self.rows[index]
            delta += (amount+1e-8)*row/row.square().sum()
            repairs += 1
        movement = self.rows @ delta
        crossed = movement < self.lower_delta
        fraction = 1.
        if crossed.any():
            fraction = max(0., min(1., float((self.lower_delta[crossed]/movement[crossed]).min())*.999))
            delta *= fraction
        # Rounding must not defeat a hard constraint. Retreat toward the exact
        # approved vector if needed; that vector is always feasible.
        for _ in range(12):
            accepted = (self.original+delta).float().double()
            if torch.all(self.rows @ (accepted-self.original) >= self.lower_delta-1e-6):
                break
            delta *= .5
        else:
            accepted = self.original.clone()
        self.require(accepted)
        with torch.no_grad():
            network.frame.weight.copy_(accepted[:-1].reshape_as(network.frame.weight).float())
            network.frame.bias.copy_(accepted[-1:].float())
        self.require(readout(network))
        return {"halfspace_repairs": repairs, "fallback_fraction": fraction,
                "minimum_margin": float((self.rows @ (readout(network)-self.original)-self.lower_delta).min())}


def training_constraints(network, inputs, target, retention):
    if (not 0 < retention <= 1 or target.ndim != 1 or target.dtype != torch.int64
            or target.requires_grad or torch.any((target != 0) & (target != 1))
            or len(inputs) != 4 or any(len(v) != len(target) for v in inputs)):
        raise ValueError("Invalid training-only KEEP constraint inputs")
    original = readout(network)
    if original.shape != (13,) or not torch.isfinite(original).all():
        raise ValueError("Expected approved twelve-channel readout")
    rows, bounds = [], []
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(target), 256):
            coarse, _, _, context = (v[first:first+256] for v in inputs)
            hidden = torch.relu(network.presence(coarse.transpose(1, 2))+
                                network.condition(context).unsqueeze(2))[:, :, 8:32]
            logits = network.frame(hidden).squeeze(1)
            peaks, indices = logits.max(dim=1)
            keep = target[first:first+256] == 0
            # Preserve the stronger normal veto when available, otherwise the
            # strict veto. Rejection logits stay immutable throughout fitting.
            cutoff = torch.where(torch.sigmoid(peaks) >= .2,
                                 math.log(.2/.8), math.log(.16/.84)).double()
            selected = keep & (torch.sigmoid(peaks) >= .16)
            features = hidden.transpose(1, 2)[torch.arange(len(hidden)), indices].double()
            features = torch.cat((features, torch.ones(len(features), 1, dtype=torch.float64)), dim=1)
            features = features[selected]
            headroom = (features @ original-cutoff[selected]).clamp_min(0)
            rows.append(features)
            bounds.append(-(1-retention)*headroom)
    rows, bounds = torch.cat(rows).clone(), torch.cat(bounds).clone()
    if not len(rows) or not torch.isfinite(rows).all() or not torch.isfinite(bounds).all():
        raise ValueError("Missing or nonfinite labeled KEEP constraints")
    result = KeepConstraints(rows, bounds, original.clone())
    result.require(original)
    return result
