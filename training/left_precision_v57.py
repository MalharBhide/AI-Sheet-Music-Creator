"""New CC0 piano fitting and explicit protection of hard genuine-note examples."""

import math
from collections import Counter

from keynata_piano_source_v57 import WINDOWS
from left_precision_v54 import frame_supervision as previous_frame_supervision
from left_precision_v54 import (
    gate,
    isolated_negative_weights,
    model,
    probabilities,
    real_piano_weights,
)
from left_precision_v54 import validate_partitions as previous_partitions

VERSION = "new-keynata-hard-keep-tail-v57-v1"
FRESH_SEEDS = tuple(range(301101, 301141))
__all__ = ["VERSION", "FRESH_SEEDS", "model", "gate", "probabilities", "real_piano_weights",
           "isolated_negative_weights", "frame_supervision", "validate_partitions", "hard_keep_loss"]


def frame_supervision(item):
    target, weight = previous_frame_supervision(item)
    if item["corpus"] == "keynata-authored-sampled-piano":
        if item["group"] != "train":
            raise ValueError("Generated Keynata parents must remain fitting-only")
        weight = weight.copy()
        weight[target > 0] = 0
    return target, weight


def validate_partitions(items):
    new = [item for item in items if item["corpus"] == "keynata-authored-sampled-piano"]
    earlier = [item for item in items if item["corpus"] != "keynata-authored-sampled-piano"]
    previous_partitions(earlier)
    expected = {"keynata-generated/"+name: 3*count for name, count in WINDOWS.items()}
    if (len(items) != 2920 or len(new) != 72
            or dict(Counter(item["source_group"] for item in new)) != expected
            or any(item["group"] != "train" for item in new)
            or len({item["id"] for item in items}) != len(items)
            or {item["source_group"] for item in new} & {item["source_group"] for item in earlier}):
        raise ValueError("Incomplete or leaking Keynata fitting partitions")


def hard_keep_loss(logits, target, weights, fraction=.1):
    """Fit the worst genuine-note losses, using supervised minibatch labels only.

    Class zero is KEEP. Selection is detached from autograd; the chosen losses
    retain gradients. REMOVE examples never enter this auxiliary protection.
    No held-out examples, source identities or historical model outputs enter it.
    """
    import torch

    if (logits.ndim != 2 or logits.shape[1] != 2 or target.shape != logits.shape[:1]
            or weights.shape != target.shape or target.dtype != torch.int64
            or not math.isfinite(fraction) or not 0 < fraction <= 1
            or not torch.isfinite(logits).all() or not torch.isfinite(weights).all()
            or torch.any(weights <= 0) or torch.any((target != 0) & (target != 1))):
        raise ValueError("Invalid supervised hard-KEEP fitting batch")
    kept = target == 0
    if not kept.any():
        return logits.sum()*0
    losses = torch.nn.functional.softplus(logits[kept, 1]-logits[kept, 0])
    count = max(1, math.ceil(len(losses)*fraction))
    indices = torch.topk(losses.detach(), count, sorted=False).indices
    return (losses[indices]*weights[kept][indices]).mean()
