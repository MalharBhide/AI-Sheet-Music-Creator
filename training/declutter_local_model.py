"""Dual-scale binary pitch-support CNN; no timing actions or new note generation."""

import numpy as np
from attack_local_model import model as joint_model
from attack_local_model import warm_start as joint_warm_start

VERSION = 'bass-v16-attack-local-declutter-v23-v1'


def model(width=32):
    from torch import nn
    network = joint_model(width)
    network.classifier[-1] = nn.Linear(32, 2)
    nn.init.zeros_(network.classifier[-1].weight)
    nn.init.zeros_(network.classifier[-1].bias)
    return network


def warm_start(saved, width):
    return joint_warm_start(model(width), saved)


def probabilities(network, frames, attack_frames, context, normalizer, batch_size=256):
    from attack_local_model import probabilities as joint_probability
    # The original helper validates all shapes/ranges and applies softmax to
    # arbitrary output width. Its empty output needs this binary-specific case.
    if not len(context):
        result = joint_probability(network, frames, attack_frames, context, normalizer, batch_size)
        return np.empty((len(result), 2), np.float32)
    result = joint_probability(network, frames, attack_frames, context, normalizer, batch_size)
    if result.shape != (len(context), 2):
        raise ValueError('Wrong binary pitch-support architecture')
    return result
