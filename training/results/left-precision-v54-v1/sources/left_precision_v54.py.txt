"""Fresh learned permissions with fitting-only isolated wrong-pitch emphasis."""

from collections import Counter

import numpy as np
from left_precision_v45 import frame_supervision as previous_frame_supervision
from left_precision_v53 import gate, model, probabilities, real_piano_weights
from left_precision_v53 import validate_partitions as previous_partitions

VERSION = "coco-hold-protected-isolated-negative-v54-v1"
FRESH_SEEDS = tuple(range(299101, 299141))
__all__ = ["VERSION", "FRESH_SEEDS", "gate", "model", "probabilities", "real_piano_weights",
           "validate_partitions", "isolated_negative_weights", "frame_supervision"]


def frame_supervision(item):
    target, weight = previous_frame_supervision(item)
    if item["corpus"] == "coco-authored-sampled-piano":
        if item["group"] != "train":
            raise ValueError("Generated piano derivatives must remain fitting-only")
        # Command duration bounds pitch support, but cannot prove acoustic
        # presence after a naturally decayed sample. Teach only known absences.
        weight = weight.copy()
        weight[target > 0] = 0
    return target, weight


def validate_partitions(items):
    new = [i for i in items if i["corpus"] == "coco-authored-sampled-piano"]
    earlier = [i for i in items if i["corpus"] != "coco-authored-sampled-piano"]
    previous_partitions(earlier)
    groups = Counter(i["source_group"] for i in new)
    if (len(items) != 2848 or len(new) != 108 or len(groups) != 36 or set(groups.values()) != {3}
            or len({i["id"] for i in items}) != len(items)
            or {i["source_group"] for i in new} & {i["source_group"] for i in earlier}
            or any(i["group"] != "train" or not i["source_group"].startswith("coco-generated/") for i in new)):
        raise ValueError("Incomplete or leaking generated-hold fitting partition")


def isolated_negative_weights(items, selected, weights, factor):
    """Use labels only during fitting; preserve genuine-note weights relative to peers."""
    if not np.isfinite(factor) or not 1 <= factor <= 64:
        raise ValueError("Invalid fitting-only wrong-pitch emphasis")
    multipliers = []
    for item, mask in zip(items, selected, strict=True):
        mask, labels = np.asarray(mask), np.asarray(item["y"])
        if (item["group"] != "train" or mask.dtype != np.bool_ or mask.ndim != 1
                or labels.shape != mask.shape or not np.array_equal(mask, item["eligible"] & item["mask"])
                or not np.isin(labels[mask], [0, 1]).all()):
            raise ValueError("Only complete supervised fitting candidates may set weights")
        value = np.ones(int(mask.sum()), np.float32)
        if item["corpus"] == "tinysol-isolated-pitch":
            value[labels[mask] == 0] = factor
        multipliers.append(value)
    multiplier = np.concatenate(multipliers)
    weights = np.asarray(weights)
    if (weights.shape != multiplier.shape or not len(weights)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Misaligned or invalid fitting weights")
    result = weights * multiplier
    return (result / result.mean()).astype(np.float32)
