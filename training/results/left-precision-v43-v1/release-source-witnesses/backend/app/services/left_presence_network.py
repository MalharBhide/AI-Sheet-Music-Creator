"""Physical note rejection with an independently calibrated pitch-presence veto."""

import numpy as np

from app.services.chord_release_network import context as chord_context
from app.services.waveform_recurrence import NAMES

VERSION = "calibrated-frame-support-v43-v1"
CONTEXT_WIDTH = 100 + len(NAMES)


def model(width):
    import torch
    from torch import nn

    if width not in (12, 20):
        raise ValueError("Invalid pitch-presence encoder width")

    class Rejection(nn.Module):
        def __init__(self):
            super().__init__()

            def encoder(channels):
                return nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8), nn.Flatten())

            self.coarse, self.attack, self.release = encoder(9), encoder(18), encoder(18)
            self.classifier = nn.Sequential(nn.Linear(3*width*8+CONTEXT_WIDTH, 128), nn.ReLU(),
                nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

        def forward(self, coarse, attack, release, context):
            return self.classifier(torch.cat((self.coarse(coarse.transpose(1, 2)),
                self.attack(attack.transpose(1, 2)), self.release(release.transpose(1, 2)),
                context), dim=1))

    class PresenceSupport(nn.Module):
        def __init__(self):
            super().__init__()
            self.reject = Rejection()
            self.presence = nn.Sequential(nn.Conv1d(9, width, 5, padding=2), nn.ReLU(),
                nn.Conv1d(width, width, 5, padding=2), nn.ReLU())
            self.condition = nn.Linear(CONTEXT_WIDTH, width)
            self.frame = nn.Conv1d(width, 1, 1)

        def forward(self, coarse, attack, release, context):
            hidden = self.presence(coarse.transpose(1, 2))+self.condition(context).unsqueeze(2)
            return self.reject(coarse, attack, release, context), self.frame(torch.relu(hidden)).squeeze(1)

    return PresenceSupport()


def observations(item):
    context = chord_context(item)
    phase = np.asarray(item["periodicity"], np.float32)
    if (phase.shape != (len(context), len(NAMES)) or not np.isfinite(phase).all()
            or np.any(np.abs(phase) > 1)):
        raise ValueError("Invalid pitch-presence recurrence")
    windows = []
    for key, shape in (("frames", (40, 9)), ("attack_frames", (61, 18)),
                       ("release_frames", (61, 18))):
        array = np.asarray(item[key], np.float32)
        if (array.shape != (len(context), *shape) or not np.isfinite(array).all()
                or np.any((array < 0) | (array > 1))):
            raise ValueError("Invalid pitch-presence spectral windows")
        windows.append(array)
    return (*windows, np.column_stack((context, phase)).astype(np.float32))


def heads(network, item, normalizer):
    import torch

    coarse, attack, release, context = observations(item)
    mean, scale = map(np.asarray, normalizer)
    if (mean.shape != (CONTEXT_WIDTH,) or scale.shape != mean.shape or np.any(scale <= 0)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()):
        raise ValueError("Invalid pitch-presence normalization")
    result = [[], []]
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(context), 256):
            span = slice(first, first+256)
            inputs = (coarse[span], attack[span], release[span],
                      ((context[span]-mean)/scale).astype(np.float32))
            rejection, presence = network(*map(torch.from_numpy, inputs))
            result[0].append(torch.softmax(rejection, dim=1).numpy())
            result[1].append(torch.sigmoid(presence).numpy())
    return tuple(np.concatenate(rows) if rows else np.empty((0, size), np.float32)
                 for rows, size in zip(result, (2, 40), strict=True))


def gated(heads, veto):
    rejection, presence = map(np.asarray, heads)
    if (rejection.shape != (len(rejection), 2) or presence.shape != (len(rejection), 40)
            or any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in (rejection, presence))
            or not np.allclose(rejection.sum(axis=1), 1., atol=1e-5)
            or not np.isfinite(veto) or not 0 < veto <= .3):
        raise ValueError("Invalid pitch-presence probabilities")
    result = rejection.copy()
    result[presence[:, 8:32].max(axis=1) >= veto] = [1., 0.]
    return result


def keep(events, probability, duration, threshold):
    events, probability = np.asarray(events), np.asarray(probability)
    if (events.shape != (len(events), 4) or probability.shape != (len(events), 2)
            or not np.isfinite(events).all() or not np.isfinite(probability).all()
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or not np.isfinite(duration) or duration <= 0 or np.any(events[:, 1] > duration)
            or not np.isfinite(threshold) or not .8 <= threshold <= 1
            or np.any((probability < 0) | (probability > 1))
            or not np.allclose(probability.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("Invalid pitch-presence removal proposal")
    allowed = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
               & (events[:, 2] >= 21) & (events[:, 2] < 60))
    return ~(allowed & (probability[:, 1] >= threshold))
