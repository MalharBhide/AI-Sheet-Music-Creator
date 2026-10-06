"""Residual binary note support using observed waveform recurrence and acoustic context."""

import numpy as np
from periodicity_features import NAMES

VERSION = "periodicity-bass-support-v26-v1"
INPUTS = 84 + len(NAMES)


def model(width):
    from torch import nn

    if width not in (64, 96):
        raise ValueError("Invalid periodicity classifier width")
    return nn.Sequential(
        nn.Linear(INPUTS, width),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(width, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def inputs(item):
    result = np.column_stack((item["x"], item["periodicity"])).astype(np.float32)
    if result.shape != (len(item["events"]), INPUTS) or not np.isfinite(result).all():
        raise ValueError("Invalid periodicity inputs")
    return result


def probabilities(network, values, normalizer):
    import torch

    values = np.asarray(values, np.float32)
    mean, scale = normalizer
    if (
        values.shape != (len(values), INPUTS)
        or mean.shape != (INPUTS,)
        or scale.shape != (INPUTS,)
        or not np.isfinite(values).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid periodicity inference contract")
    network.eval()
    result = []
    with torch.inference_mode():
        for first in range(0, len(values), 256):
            result.append(
                torch.softmax(
                    network(
                        torch.from_numpy(
                            ((values[first : first + 256] - mean) / scale).astype(np.float32)
                        )
                    ),
                    dim=1,
                ).numpy()
            )
    return np.concatenate(result) if result else np.empty((0, 2), np.float32)
