"""Observed-context release specialist; independent endings and immutable attacks."""

import numpy as np

VERSION = "protected-release-only-v38-v1"
SHIFTS = np.array([-.08, -.04, -.02, 0., .02, .04, .08])
STAY = 3
CONTEXT_WIDTH = 100


def chord_context(events):
    events = np.asarray(events, float)
    if (events.shape != (len(events), 4) or not np.isfinite(events).all()
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])):
        raise ValueError("Invalid chord observation intervals")
    result = np.zeros((len(events), 8), np.float32)
    starts, ends, pitches, _ = events.T
    for i, (start, end, pitch, _) in enumerate(events):
        others = np.arange(len(events)) != i
        attacks = others & (np.abs(starts-start) <= .03)
        releases = others & (np.abs(ends-end) <= .03)
        same = others & (pitches == pitch)
        previous = ends[same & (starts < start)]
        following = starts[same & (starts > start)]
        result[i] = [min(attacks.sum(), 8)/8, min(releases.sum(), 8)/8,
                     min(np.ptp(pitches[attacks]), 48)/48 if attacks.any() else 0,
                     min(np.sum(others & (starts < start) & (ends > start)), 8)/8,
                     min(np.sum(others & (starts < end) & (ends > end)), 8)/8,
                     np.clip(following.min()-end, -1, 1) if len(following) else 1,
                     np.clip(start-previous.max(), -1, 1) if len(previous) else 1,
                     min(end-start, 4)/4]
    return result

def context(item):
    result = np.column_stack((item["x"], item["release_neighbors"], chord_context(item["events"])))
    if result.shape != (len(item["events"]), CONTEXT_WIDTH) or not np.isfinite(result).all():
        raise ValueError("Invalid chord boundary context")
    return result.astype(np.float32)

def model(width):
    import torch
    from torch import nn

    if width not in (16, 24):
        raise ValueError("Invalid release-only encoder")

    class ReleaseOnly(nn.Module):
        def __init__(self):
            super().__init__()

            def encoder():
                return nn.Sequential(nn.Conv1d(18, width, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(16), nn.Flatten())

            self.attack, self.release = encoder(), encoder()
            self.ending = nn.Sequential(nn.Linear(2*width*16+100, 96), nn.ReLU(),
                                        nn.Dropout(.1), nn.Linear(96, 7))
            nn.init.zeros_(self.ending[-1].weight)
            nn.init.zeros_(self.ending[-1].bias)
            with torch.no_grad():
                self.ending[-1].bias[STAY] = 2

        def forward(self, attacks, releases, observed):
            hidden = torch.cat((self.attack(attacks.transpose(1, 2)),
                                self.release(releases.transpose(1, 2)), observed), dim=1)
            # Deterministic inert heads let the unchanged musical evaluator check
            # that neither note support nor attacks can be changed by this model.
            keep = hidden.new_full((len(hidden), 2), -100.)
            attack = hidden.new_full((len(hidden), 7), -100.)
            keep[:, 0], attack[:, STAY] = 0., 0.
            return keep, attack, self.ending(hidden)

    return ReleaseOnly()

def probabilities(network, item, normalizer):
    import torch

    x = context(item)
    mean, scale = normalizer
    if (np.shape(mean) != (CONTEXT_WIDTH,) or np.shape(scale) != (CONTEXT_WIDTH,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid training-only normalization")
    for key in ("attack_frames", "release_frames"):
        array = item[key]
        if (array.shape != (len(x), 61, 18) or not np.isfinite(array).all()
                or np.any((array < 0) | (array > 1))):
            raise ValueError("Invalid boundary acoustic windows")
    output = [[], [], []]
    network.eval()
    with torch.inference_mode():
        for first in range(0, len(x), 256):
            stop = first+256
            heads = network(torch.from_numpy(item["attack_frames"][first:stop].astype(np.float32)),
                            torch.from_numpy(item["release_frames"][first:stop].astype(np.float32)),
                            torch.from_numpy(((x[first:stop]-mean)/scale).astype(np.float32)))
            for rows, head in zip(output, heads, strict=True):
                rows.append(torch.softmax(head, dim=1).numpy())
    return tuple(np.concatenate(rows) if rows else np.empty((0, size), np.float32)
                 for rows, size in zip(output, (2, 7, 7), strict=True))

