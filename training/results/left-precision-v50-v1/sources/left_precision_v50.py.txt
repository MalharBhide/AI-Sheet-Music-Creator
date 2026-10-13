"""Warm boundary architecture with training-only real-piano loss emphasis."""

import numpy as np
from left_precision_v49 import gate, initialize, model, probabilities

VERSION = "real-piano-emphasis-warm-boundary-v50-v1"
FRESH_SEEDS = tuple(range(295101, 295141))
__all__ = ["VERSION", "FRESH_SEEDS", "gate", "initialize", "model", "probabilities", "real_piano_weights"]


def real_piano_weights(items, selected, weights, factor):
    if not np.isfinite(factor) or not 1 <= factor <= 16:
        raise ValueError("Invalid real-piano fitting emphasis")
    multipliers = []
    for item, mask in zip(items, selected, strict=True):
        mask = np.asarray(mask)
        if item["group"] != "train" or mask.dtype != np.bool_ or mask.ndim != 1:
            raise ValueError("Only selected training candidates may set loss weights")
        multipliers.append(np.full(int(mask.sum()),
            factor if item["corpus"] == "musicnet-real-piano" else 1, np.float32))
    multiplier = np.concatenate(multipliers)
    weights = np.asarray(weights)
    if (weights.shape != multiplier.shape or not len(weights)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Misaligned or invalid real-piano loss weights")
    result = weights * multiplier
    return (result / result.mean()).astype(np.float32)
