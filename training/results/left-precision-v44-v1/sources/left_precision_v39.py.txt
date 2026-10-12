"""Independent, removal-only left-hand learner over physical V32 survivors.

Decisions are made before V38, then applied after the unchanged V38 release
pass. Keeping that pass's complete input prevents a deletion from changing the
release of a neighboring, retained note.
"""

import numpy as np
from chord_boundary_v36 import context

VERSION = "independent-left-support-v39-v1"


def model(width):
    import torch
    from torch import nn

    if width not in (12, 20):
        raise ValueError("Invalid left-hand support encoder")

    class Support(nn.Module):
        def __init__(self):
            super().__init__()

            def encoder(channels):
                return nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.ReLU(),
                                     nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(8), nn.Flatten())

            self.coarse, self.attack, self.release = encoder(9), encoder(18), encoder(18)
            self.classifier = nn.Sequential(nn.Linear(3*width*8+100, 96), nn.ReLU(),
                                            nn.Dropout(.15), nn.Linear(96, 2))
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
            with torch.no_grad():
                self.classifier[-1].bias[0] = 4.

        def forward(self, coarse, attack, release, observed):
            return self.classifier(torch.cat((self.coarse(coarse.transpose(1, 2)),
                self.attack(attack.transpose(1, 2)), self.release(release.transpose(1, 2)),
                observed), dim=1))

    return Support()


def observations(item):
    x = context(item)
    windows = []
    for key, shape in (("frames", (40, 9)), ("attack_frames", (61, 18)),
                       ("release_frames", (61, 18))):
        array = np.asarray(item[key], np.float32)
        if (array.shape != (len(x), *shape) or not np.isfinite(array).all()
                or np.any((array < 0) | (array > 1))):
            raise ValueError("Invalid physical left-hand windows")
        windows.append(array)
    return (*windows, x)


def probabilities(network, item, normalizer):
    import torch

    coarse, attack, release, x = observations(item)
    mean, scale = map(np.asarray, normalizer)
    if (mean.shape != (100,) or scale.shape != (100,) or np.any(scale <= 0)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()):
        raise ValueError("Invalid support normalization")
    result = []
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(x), 256):
            span = slice(first, first+256)
            inputs = (coarse[span], attack[span], release[span],
                      ((x[span]-mean)/scale).astype(np.float32))
            result.append(torch.softmax(network(*map(torch.from_numpy, inputs)), dim=1).numpy())
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)


def keep(events, probability, duration, threshold):
    events, probability = np.asarray(events), np.asarray(probability)
    if (events.shape != (len(events), 4) or probability.shape != (len(events), 2)
            or not np.isfinite(events).all() or not np.isfinite(probability).all()
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or not np.isfinite(duration) or duration <= 0 or np.any(events[:, 1] > duration)
            or not np.isfinite(threshold) or not .8 <= threshold <= 1.
            or np.any((probability < 0) | (probability > 1))
            or not np.allclose(probability.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("Invalid left-hand support proposal")
    eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
                & (events[:, 2] >= 21) & (events[:, 2] < 60))
    return ~(eligible & (probability[:, 1] >= threshold))
