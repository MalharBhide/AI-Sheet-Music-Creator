"""Separate partial support from complete support instead of conflating both KEEP labels."""

import math

import numpy as np
from fusion_model import WARM_SHA
from fusion_model import initialize as initialize_binary
from fusion_model import model as binary_model
from fusion_model import probabilities as binary_probabilities
from pitch_interval_coverage import reference_coverage

VERSION = "partial-complete-unsupported-bass-v28-v1"
CLASSES = ("partial_support", "other_protected_support", "unsupported")
__all__ = ["VERSION", "WARM_SHA", "CLASSES", "model", "initialize", "targets", "probabilities"]


def model(width):
    from torch import nn

    network = binary_model(width)
    network.classifier[-1] = nn.Linear(32, 3)
    return network


def initialize(network, saved):
    import torch

    width = network.recurrence[0].out_features
    binary = initialize_binary(binary_model(width), saved)
    state = binary.state_dict()
    weight, bias = state["classifier.5.weight"], state["classifier.5.bias"]
    # Split inherited KEEP mass equally between the two protected subtypes.
    # Their summed probability remains exactly the approved V25 KEEP score.
    state["classifier.5.weight"] = torch.stack((weight[0], weight[0], weight[1]))
    state["classifier.5.bias"] = torch.stack(
        (bias[0] - math.log(2), bias[0] - math.log(2), bias[1])
    )
    network.load_state_dict(state, strict=True)
    return network


def targets(item):
    """Training-only taxonomy; neither reference intervals nor classes are inputs."""
    if item["group"] != "train":
        raise ValueError("Support subtype targets are training-only")
    events = np.asarray(item["events"], float)
    truth = np.asarray(item["pitch_reference"], float)
    labels, mask = np.asarray(item["y"]), np.asarray(item["mask"])
    if (
        events.shape != (len(events), 4)
        or truth.shape != (len(truth), 3)
        or labels.shape != (len(events),)
        or mask.shape != labels.shape
        or not np.isfinite(events).all()
        or not np.isfinite(truth).all()
        or np.any(events[:, 1] <= events[:, 0])
        or np.any(truth[:, 1] <= truth[:, 0])
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("Invalid support subtype intervals")
    covered = reference_coverage(events[:, :3], truth)
    fraction = covered / (events[:, 1] - events[:, 0])
    partial = (labels == 1) & mask & (covered > 1e-6) & (fraction < 0.25)
    return np.where(partial, 0, np.where(labels == 1, 1, 2)).astype(np.int64)


def probabilities(network, item, normalizer):
    values = binary_probabilities(network, item, normalizer)
    if values.shape != (len(item["events"]), 3):
        # The binary observation reader's empty-input result has two columns.
        if len(item["events"]) == 0 and values.shape == (0, 2):
            return values
        raise ValueError("Changed support subtype output contract")
    # Complement avoids a float32 two-class sum rounding slightly above one
    # when unsupported mass is tiny; the strict scorer must remain unchanged.
    removed = np.clip(values[:, 2], 0, 1)
    return np.column_stack((1 - removed, removed))
