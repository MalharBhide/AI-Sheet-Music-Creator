"""Removal specialist adding waveform periodicity to local spectral evidence."""

import numpy as np
from left_precision_v39 import model as spectral_model
from left_precision_v39 import observations as spectral_observations
from periodicity_features import NAMES

VERSION = "periodic-left-support-v40-v1"
CONTEXT_WIDTH = 100+len(NAMES)


def model(width):
    import torch
    from torch import nn

    result = spectral_model(width)
    result.classifier = nn.Sequential(nn.Linear(3*width*8+CONTEXT_WIDTH, 128), nn.ReLU(),
        nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    nn.init.zeros_(result.classifier[-1].weight)
    nn.init.zeros_(result.classifier[-1].bias)
    with torch.no_grad():
        result.classifier[-1].bias[0] = 4.
    return result


def observations(item):
    coarse, attack, release, context = spectral_observations(item)
    periodicity = np.asarray(item["periodicity"], np.float32)
    if (periodicity.shape != (len(context), len(NAMES)) or not np.isfinite(periodicity).all()
            or np.any(np.abs(periodicity) > 1)):
        raise ValueError("Invalid physical periodicity evidence")
    return coarse, attack, release, np.column_stack((context, periodicity)).astype(np.float32)


def probabilities(network, item, normalizer):
    import torch

    coarse, attack, release, x = observations(item)
    mean, scale = map(np.asarray, normalizer)
    if (mean.shape != (CONTEXT_WIDTH,) or scale.shape != mean.shape or np.any(scale <= 0)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()):
        raise ValueError("Invalid periodic support normalization")
    result = []
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(x), 256):
            span = slice(first, first+256)
            inputs = (coarse[span], attack[span], release[span],
                      ((x[span]-mean)/scale).astype(np.float32))
            result.append(torch.softmax(network(*map(torch.from_numpy, inputs)), dim=1).numpy())
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)
