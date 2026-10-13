"""Independent rejection and physical KEEP heads; both must permit deletion."""

import numpy as np
from left_precision_v40 import CONTEXT_WIDTH, observations
from left_precision_v40 import model as periodic_model

VERSION = "dual-physical-left-support-v41-v1"
GUARD_WIDTH = CONTEXT_WIDTH+3*(9+18+18)


def model(width):
    import torch
    from torch import nn

    class DualSupport(nn.Module):
        def __init__(self):
            super().__init__()
            self.reject = periodic_model(width)
            self.guard = nn.Sequential(nn.Linear(GUARD_WIDTH, 64), nn.ReLU(), nn.Dropout(.1),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.guard[-1].weight)
            nn.init.zeros_(self.guard[-1].bias)
            with torch.no_grad():
                self.guard[-1].bias[0] = 4.

        def forward(self, coarse, attack, release, context):
            physical = torch.cat((context, *[stat for window in (coarse, attack, release)
                for stat in (window.mean(dim=1), window.amax(dim=1), window.std(dim=1, unbiased=False))]), dim=1)
            return self.reject(coarse, attack, release, context), self.guard(physical)

    return DualSupport()


def combine(rejection, guardian):
    rejection, guardian = np.asarray(rejection), np.asarray(guardian)
    if (rejection.shape != guardian.shape or rejection.ndim != 2 or rejection.shape[1] != 2
            or any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
                   or not np.allclose(p.sum(axis=1), 1., atol=1e-5) for p in (rejection, guardian))):
        raise ValueError("Invalid dual support probabilities")
    remove = np.minimum(rejection[:, 1], guardian[:, 1])
    return np.column_stack((1.-remove, remove)).astype(np.float32)


def probabilities(network, item, normalizer):
    import torch

    coarse, attack, release, x = observations(item)
    mean, scale = map(np.asarray, normalizer)
    if (mean.shape != (CONTEXT_WIDTH,) or scale.shape != mean.shape or np.any(scale <= 0)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()):
        raise ValueError("Invalid dual support normalization")
    result = []
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(x), 256):
            span = slice(first, first+256)
            inputs = (coarse[span], attack[span], release[span], ((x[span]-mean)/scale).astype(np.float32))
            heads = network(*map(torch.from_numpy, inputs))
            result.append(combine(*[torch.softmax(p, dim=1).numpy() for p in heads]))
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)
