"""Warm V25 acoustic encoders with an initially neutral waveform-recurrence branch."""

import numpy as np
from attack_release_model import model as previous_model
from periodicity_features import NAMES as PERIODICITY_NAMES
from release_neighbor_features import NAMES as NEIGHBOR_NAMES

NAMES = PERIODICITY_NAMES + NEIGHBOR_NAMES

VERSION = "wide-bounded-teacher-support-v31-v1"
WARM_SHA = "c1d8d11c559c6cd47f32a1cdfe89ca19c1e691c942119a166c90cc780fb5a074"


def model(width, cap):
    import torch
    from torch import nn

    if width not in (64, 96) or cap not in (2.0, 4.0):
        raise ValueError("Changed bounded teacher correction architecture")

    class AnchoredNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.teacher = previous_model(24)
            self.correction = nn.Sequential(
                nn.Linear(len(NAMES), width),
                nn.ReLU(),
                nn.Linear(width, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.cap = float(cap)
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                self.correction[-1].weight.zero_()
                self.correction[-1].bias.zero_()

        def forward(self, frames, attacks, endings, context, recurrence):
            # The approved teacher's dropout must remain off during fitting.
            self.teacher.eval()
            logits = self.teacher(frames, attacks, endings, context)
            delta = self.cap * torch.tanh(self.correction(recurrence))
            return torch.cat((logits[:, :1], logits[:, 1:] + delta), dim=1)

    return AnchoredNet()


def initialize(network, saved):
    if saved["version"] != "attack-release-bass-support-v25-v1" or saved["width"] != 24:
        raise ValueError("Changed approved teacher checkpoint")
    if saved["mean"].shape != (84,) or saved["scale"].shape != (84,):
        raise ValueError("Changed approved context normalizer")
    network.teacher.load_state_dict(saved["state_dict"], strict=True)
    network.teacher.eval()
    return network


def probabilities(network, item, normalizer):
    import torch

    arrays = [
        np.asarray(item[k], np.float32) for k in ("frames", "attack_frames", "release_frames", "x")
    ]
    phase = np.asarray(item["periodicity"], np.float32)
    neighbors = np.asarray(item["release_neighbors"], np.float32)
    if phase.shape != (len(item["events"]), len(PERIODICITY_NAMES)) or neighbors.shape != (
        len(item["events"]),
        len(NEIGHBOR_NAMES),
    ):
        raise ValueError("Changed recurrence/release-neighbor input alignment")
    arrays.append(np.column_stack((phase, neighbors)))
    mean, scale = [np.asarray(v, np.float32) for v in normalizer]
    n = len(arrays[3])
    expected = ((n, 40, 9), (n, 61, 18), (n, 61, 18), (n, 84), (n, len(NAMES)))
    if (
        any(
            a.shape != shape or not np.isfinite(a).all()
            for a, shape in zip(arrays, expected, strict=True)
        )
        or mean.shape != (84 + len(NAMES),)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or any(np.any((a < 0) | (a > 1)) for a in arrays[:3])
        or np.any(np.abs(arrays[4]) > 1)
    ):
        raise ValueError("Invalid acoustic fusion shape or normalization")
    network.eval()
    result = []
    with torch.inference_mode():
        for first in range(0, n, 256):
            f, a, e, x, p = [v[first : first + 256] for v in arrays]
            args = [
                torch.from_numpy(v)
                for v in (
                    f,
                    a,
                    e,
                    ((x - mean[:84]) / scale[:84]).astype(np.float32),
                    ((p - mean[84:]) / scale[84:]).astype(np.float32),
                )
            ]
            result.append(torch.softmax(network(*args), dim=-1).numpy())
    if not result:
        return np.empty((0, 2), np.float32)
    raw = np.concatenate(result)
    removed = np.clip(raw[:, 1], 0, 1)
    return np.column_stack((1 - removed, removed))
