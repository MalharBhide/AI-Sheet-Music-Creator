"""A removal head guarded by supervised pitch presence throughout each event."""

import numpy as np
from left_precision_v40 import CONTEXT_WIDTH, observations
from left_precision_v40 import model as periodic_model

VERSION = "frame-presence-left-support-v42-v1"


def presence_targets(item):
    """Truth is used only for fitting; protect sub-frame fragments explicitly."""
    events = np.asarray(item["events"], float)
    refs = np.asarray(item["pitch_reference"], float)
    if (events.shape != (len(events), 4) or refs.shape != (len(refs), 3)
            or not np.isfinite(events).all() or not np.isfinite(refs).all()
            or np.any(events[:, 1] <= events[:, 0]) or np.any(refs[:, 1] <= refs[:, 0])):
        raise ValueError("Invalid frame support supervision")
    result = np.zeros((len(events), 40), np.float32)
    for index, (start, end, pitch, _) in enumerate(events):
        times = np.concatenate((start-np.linspace(.24, .03, 8), np.linspace(start, end, 24), end+np.linspace(.03, .24, 8)))
        selected = refs[refs[:, 2] == pitch]
        for left, right, _ in selected:
            # Protect a CQT half-hop around true boundaries, plus any even
            # shorter supplied fragment that falls between the 24 samples.
            result[index] = np.maximum(result[index], ((times >= left-128/22050) & (times <= right+128/22050)))
            if min(end, right) > max(start, left):
                center = (min(end, right)+max(start, left))/2
                result[index, 8+np.argmin(np.abs(times[8:32]-center))] = 1.
        if item["mask"][index] and item["y"][index] == 1 and not result[index, 8:32].any():
            # Genuine jittered attacks/held matches also remain protected even
            # if the supplied support interval is just outside the candidate.
            if not len(selected):
                result[index, 8:32] = 1.
            else:
                distance = np.minimum(np.abs(times[8:32, None]-selected[:, 0]),
                                      np.abs(times[8:32, None]-selected[:, 1]))
                result[index, 8+np.unravel_index(np.argmin(distance), distance.shape)[0]] = 1.
    return result


def model(width):
    import torch
    from torch import nn

    class PresenceSupport(nn.Module):
        def __init__(self):
            super().__init__()
            self.reject = periodic_model(width)
            self.presence = nn.Sequential(nn.Conv1d(9, width, 5, padding=2), nn.ReLU(),
                nn.Conv1d(width, width, 5, padding=2), nn.ReLU())
            self.condition = nn.Linear(CONTEXT_WIDTH, width)
            self.frame = nn.Conv1d(width, 1, 1)
            nn.init.zeros_(self.frame.weight)
            nn.init.constant_(self.frame.bias, 4.)

        def forward(self, coarse, attack, release, context):
            hidden = self.presence(coarse.transpose(1, 2))+self.condition(context).unsqueeze(2)
            return self.reject(coarse, attack, release, context), self.frame(torch.relu(hidden)).squeeze(1)

    return PresenceSupport()


def combine(rejection, presence):
    rejection, presence = np.asarray(rejection), np.asarray(presence)
    if (rejection.shape != (len(rejection), 2) or presence.shape != (len(rejection), 40)
            or any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in (rejection, presence))
            or not np.allclose(rejection.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("Invalid frame support proposal")
    remove = np.minimum(rejection[:, 1], 1.-presence[:, 8:32].max(axis=1))
    return np.column_stack((1.-remove, remove)).astype(np.float32)


def probabilities(network, item, normalizer):
    import torch

    coarse, attack, release, x = observations(item)
    mean, scale = map(np.asarray, normalizer)
    if (mean.shape != (CONTEXT_WIDTH,) or scale.shape != mean.shape or np.any(scale <= 0)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()):
        raise ValueError("Invalid frame support normalization")
    result = []
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(x), 256):
            span = slice(first, first+256)
            inputs = (coarse[span], attack[span], release[span], ((x[span]-mean)/scale).astype(np.float32))
            rejection, presence = network(*map(torch.from_numpy, inputs))
            result.append(combine(torch.softmax(rejection, dim=1).numpy(), torch.sigmoid(presence).numpy()))
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)
