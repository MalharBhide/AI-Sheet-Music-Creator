"""Offline V32-survivor support, attack and independent key-release learner."""

import numpy as np
from bass_training_data import eligible
from train_melody_timing import matches

VERSION = "independent-chord-boundaries-v36-v1"
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


def labels(item):
    """References supply targets only; ambiguous or out-of-window clocks abstain."""
    events = item["events"]
    result = np.full((len(events), 3), -100, np.int64)
    supervised = item["mask"] & item["eligible"]
    result[supervised, 0] = 1-item["y"][supervised].astype(np.int64)
    for ref, index in matches(item["reference"], events, .2):
        if not item["eligible"][index]:
            continue
        for side in (0, 1):
            delta = item["reference"][ref, side]-events[index, side]
            if abs(delta) <= .09:
                result[index, side+1] = np.argmin(np.abs(SHIFTS-delta))
    return result


def model(width):
    import torch
    from torch import nn

    if width not in (16, 24):
        raise ValueError("Unsupported boundary encoder width")

    class Boundaries(nn.Module):
        def __init__(self):
            super().__init__()

            def encoder():
                return nn.Sequential(nn.Conv1d(18, width, 5, padding=2), nn.ReLU(),
                                     nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(16), nn.Flatten())

            self.attack = encoder()
            self.release = encoder()
            self.shared = nn.Sequential(nn.Linear(2*width*16+CONTEXT_WIDTH, 96),
                                        nn.ReLU(), nn.Dropout(.1))
            self.support = nn.Linear(96, 2)
            self.onset = nn.Linear(96, len(SHIFTS))
            self.ending = nn.Linear(96, len(SHIFTS))
            for head in (self.support, self.onset, self.ending):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            with torch.no_grad():
                self.support.bias[0] = 2
                self.onset.bias[STAY] = self.ending.bias[STAY] = 2

        def forward(self, attacks, releases, observed):
            hidden = self.shared(torch.cat((self.attack(attacks.transpose(1, 2)),
                                           self.release(releases.transpose(1, 2)), observed), dim=1))
            return self.support(hidden), self.onset(hidden), self.ending(hidden)

    return Boundaries()


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


def transform(events, onset, release, remove, duration):
    """Independent releases; never merge a chord or smear a repeated-key attack."""
    events = np.asarray(events, float)
    onset, release, remove = [np.asarray(v) for v in (onset, release, remove)]
    if (events.shape != (len(events), 4) or not np.isfinite(events).all()
            or any(v.shape != (len(events),) for v in (onset, release, remove))
            or any(not np.isfinite(v).all() for v in (onset, release))
            or not np.isin(onset, SHIFTS).all() or not np.isin(release, SHIFTS).all()
            or remove.dtype != bool or not np.isfinite(duration) or duration <= 0
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])):
        raise ValueError("Invalid boundary proposals")
    allowed = eligible(events, duration)
    keep = ~(remove & allowed)
    result = events.copy()
    result[:, 0] += np.where(allowed, onset, 0.)
    result[:, 1] += np.where(allowed, release, 0.)
    invalid = ((result[:, 1]-result[:, 0] < .045) | (result[:, 0] < 2.5)
               | (result[:, 1] > duration-2.5)) & allowed
    result[invalid, :2] = events[invalid, :2]
    for pitch in np.unique(events[:, 2]):
        indices = np.flatnonzero(keep & (events[:, 2] == pitch))
        indices = indices[np.argsort(events[indices, 0], kind="stable")]
        for i, j in zip(indices, indices[1:], strict=False):
            if events[j, 0] == events[i, 0]:
                result[[i, j], :2] = events[[i, j], :2]
                continue
            touching = abs(events[i, 1]-events[j, 0]) < 1e-6
            collision = events[i, 1] <= events[j, 0] and result[i, 1] > result[j, 0]
            reordered = result[j, 0] <= result[i, 0]
            if collision or reordered or (touching and abs(result[i, 1]-result[j, 0]) > 1e-6):
                result[[i, j], :2] = events[[i, j], :2]
        # Reverting one pair can invalidate a previous pair in the same chain.
        # Abstain for the pitch rather than silently introduce a new overlap.
        for i, j in zip(indices, indices[1:], strict=False):
            if ((events[i, 1] <= events[j, 0] and result[i, 1] > result[j, 0])
                    or (events[i, 0] < events[j, 0] and result[i, 0] >= result[j, 0])
                    or (abs(events[i, 1]-events[j, 0]) < 1e-6
                        and abs(result[i, 1]-result[j, 0]) > 1e-6)):
                result[indices, :2] = events[indices, :2]
                break
    return result[keep], np.flatnonzero(keep)


def decode(item, probability, guardian, timing_threshold, remove_threshold):
    if (not np.isfinite([timing_threshold, remove_threshold]).all()
            or not 0 <= timing_threshold <= 1 or not 0 <= remove_threshold <= 1):
        raise ValueError("Invalid confidence thresholds")
    if len(probability) != 3:
        raise ValueError("Missing boundary model heads")
    for head, size in zip(probability, (2, 7, 7), strict=True):
        if (head.shape != (len(item["events"]), size) or not np.isfinite(head).all()
                or np.any((head < 0) | (head > 1)) or not np.allclose(head.sum(axis=1), 1., atol=1e-5)):
            raise ValueError("Invalid boundary probabilities")
    guardian = np.asarray(guardian)
    if (guardian.shape != (len(item["events"]),) or not np.isfinite(guardian).all()
            or np.any((guardian < 0) | (guardian > 1))):
        raise ValueError("Invalid protected-note guardian")
    support, attack, ending = probability
    shifts = [np.where(p.max(axis=1) >= timing_threshold, SHIFTS[p.argmax(axis=1)], 0.)
              for p in (attack, ending)]
    remove = (support[:, 1] >= remove_threshold) & (guardian < .15)
    return transform(item["events"], *shifts, remove, item["duration"])
