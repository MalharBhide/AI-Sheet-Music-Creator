"""Warm V25 acoustic encoders with an initially neutral waveform-recurrence branch."""

import numpy as np
from attack_release_model import model as previous_model
from periodicity_features import NAMES

VERSION = "attack-ending-periodicity-support-v27-v1"
WARM_SHA = "c1d8d11c559c6cd47f32a1cdfe89ca19c1e691c942119a166c90cc780fb5a074"


def model(recurrence_width):
    import torch
    from torch import nn

    if recurrence_width not in (32, 64):
        raise ValueError("Invalid waveform fusion width")

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            previous = previous_model(24)
            self.coarse, self.local, self.ending = previous.coarse, previous.local, previous.ending
            self.recurrence = nn.Sequential(
                nn.Linear(len(NAMES), recurrence_width),
                nn.ReLU(),
                nn.Linear(recurrence_width, recurrence_width),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(852 + recurrence_width, 96),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(96, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, frames, attacks, endings, context, periodicity):
            return self.classifier(
                torch.cat(
                    (
                        self.coarse(frames.transpose(1, 2)),
                        context,
                        self.local(attacks.transpose(1, 2)),
                        self.ending(endings.transpose(1, 2)),
                        self.recurrence(periodicity),
                    ),
                    dim=1,
                )
            )

    return FusionNet()


def initialize(network, saved):
    import torch

    if saved["version"] != "attack-release-bass-support-v25-v1" or saved["width"] != 24:
        raise ValueError("Changed shipped V25 warm-start network")
    source = saved["state_dict"]
    state = network.state_dict()
    for key in state:
        if key.startswith("recurrence."):
            continue
        if key == "classifier.0.weight":
            if source[key].shape != (96, 852):
                raise ValueError("Changed V25 classifier input contract")
            state[key].zero_()
            state[key][:, :852] = source[key]
        else:
            if key not in source or state[key].shape != source[key].shape:
                raise ValueError("Changed inherited acoustic encoder")
            state[key] = source[key].clone()
    network.load_state_dict(state, strict=True)
    if saved["mean"].shape != (84,) or saved["scale"].shape != (84,):
        raise ValueError("Changed V25 context normalization")
    # The new branch starts with zero influence on logits. Its parameters are
    # learned subsequently; labels and fitted probabilities never enter inputs.
    with torch.no_grad():
        assert not torch.count_nonzero(network.classifier[0].weight[:, 852:])
    return network


def probabilities(network, item, normalizer):
    import torch

    arrays = [
        np.asarray(item[k], np.float32)
        for k in ("frames", "attack_frames", "release_frames", "x", "periodicity")
    ]
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
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)
